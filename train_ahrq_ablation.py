"""
AH-RQ消融实验脚本（含Stage 2 推荐训练）

验证创新点的有效性：
1. 自适应码本大小（不同语义层次使用不同码本大小）
2. 层次化语义一致性学习(HSCL)
3. EMA更新+死码重置

实验配置：
- Baseline-RQ: 原始RQ-VAE（4层等码本，无EMA，无HSCL）
- AHRQ-HierCodebook: 自适应层次化码本
- AHRQ-EMA: +EMA更新
- AHRQ-HSCL: +层次一致性学习
- AHRQ-Full: 完整配置

评估指标：
- Stage 1: 重构损失、码本利用率
- Stage 2: HR@5/10/20, NDCG@5/10/20, MRR
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd

from config import new_config
from data_utils import get_all_item_pretrain_dataloader, get_pmat_dataloader
from log import Logger
from our_models.ah_rq import (
    AdaptiveHierarchicalQuantizer,
    VectorQuantizer,
    ResidualVectorQuantizer,
    ResidualVectorQuantizerEMA,
    HierarchicalSemanticConsistency
)
from our_models.sasrec_ahrq import SASRecAHRQ
from utils.loss import compute_rqvae_recon_loss
from utils.utils import seed_everything, calculate_mrr_full
from metrics import codebook_usage_rate
from utils.utils import calculate_metrics


NUM_WORKS = 0


@dataclass
class AblationConfig:
    """消融实验配置"""
    experiment_name: str
    use_ema: bool = False
    use_hscl: bool = False
    use_emotion: bool = False
    use_hierarchy_weight: bool = True
    # 码本配置 - 最佳配置 [256, 512, 512, 512]（4层）
    # 分层设计：topic(2层), style(2层), emotion(2层)
    topic_codebook: int = 256      # topic层使用较小码本
    style_codebook: int = 512      # style层使用标准码本
    emotion_codebook: int = 512   # emotion层使用标准码本
    baseline_codebook: int = 512
    # 损失权重 - 降低HSCL权重减少对重构的干扰
    hscl_weight: float = 0.03      # 原0.1→0.03，降低对重建的干扰
    quant_weight: float = 1.0


def get_semantic_hierarchy(config: AblationConfig) -> dict:
    """根据消融配置生成语义层次

    最佳配置: [256, 512, 512, 512] (4层)
    分层设计:
    - topic: 1层 (256)
    - style: 2层 (512, 512)
    - emotion: 1层 (512, 仅在use_emotion=True时启用)

    注意：无论use_emotion是否为True，都保持4层结构
    """
    # 确定损失权重
    if config.use_hierarchy_weight:
        topic_weight = 1.0
        style_weight = 0.8
        emotion_weight = 0.6
    else:
        topic_weight = 1.0
        style_weight = 1.0
        emotion_weight = 1.0

    if config.experiment_name == "Baseline-RQ":
        # 基线：4层等码本，无EMA
        num_layers = 4
        return {
            "topic": {
                "layers": list(range(num_layers)),
                "codebook_size": config.baseline_codebook,
                "loss_weight": topic_weight,
                "ema_decay": 0.99
            }
        }
    elif config.experiment_name in ["AHRQ-HierCodebook", "AHRQ-EMA", "AHRQ-HSCL", "AHRQ-Full"]:
        # 自适应码本：固定4层结构 [256, 512, 512, 512]
        # 无论是否有emotion，都保持4层结构：
        # - topic: 1层(layers=[0])
        # - style: 2层(layers=[1,2])
        # - emotion: 1层(layers=[3])
        hierarchy = {
            "topic": {
                "layers": [0],
                "codebook_size": config.topic_codebook,  # 256
                "loss_weight": topic_weight,
                "ema_decay": 0.99
            },
            "style": {
                "layers": [1, 2],
                "codebook_size": config.style_codebook,  # [512, 512]
                "loss_weight": style_weight,
                "ema_decay": 0.99
            }
        }
        # emotion层：始终包含，保持4层结构
        # 消融emotion时，使用较小的权重
        hierarchy["emotion"] = {
            "layers": [3],
            "codebook_size": config.emotion_codebook,  # 512
            "loss_weight": emotion_weight if config.use_emotion else 0.0,  # 消融时权重为0
            "ema_decay": 0.99
        }
        return hierarchy
    else:
        raise ValueError(f"Unknown experiment: {config.experiment_name}")


def build_ahrq_model(config: AblationConfig, device: torch.device):
    """构建AHRQ模型"""
    semantic_hierarchy = get_semantic_hierarchy(config)

    # 构建n_e_list
    n_e_list = []
    layer_to_codebook = {}
    for semantic_type, cfg in semantic_hierarchy.items():
        for layer_idx in cfg['layers']:
            layer_to_codebook[layer_idx] = cfg['codebook_size']
    for layer_idx in sorted(layer_to_codebook.keys()):
        n_e_list.append(layer_to_codebook[layer_idx])

    model = AdaptiveHierarchicalQuantizer(
        hidden_dim=new_config.ahrq_hidden_dim,
        semantic_hierarchy=semantic_hierarchy,
        use_multimodal=True,
        text_dim=new_config.text_dim,
        visual_dim=new_config.visual_dim,
        beta=new_config.ahrq_beta,
        use_ema=config.use_ema,
        ema_decay=0.99,
        reset_unused_codes=config.use_ema,
        reset_threshold=50,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.0 if config.use_ema else 0.0,
        sk_iters=100,
        dropout=0.1,
        bn=True
    ).to(device)

    return model, n_e_list, semantic_hierarchy


def train_stage1_quantization(
    model,
    pretrain_loader,
    config: AblationConfig,
    device: torch.device,
    logger
):
    """Stage 1: 量化预训练"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # 如果启用HSCL，创建层次一致性模块
    hscl_module = None
    if config.use_hscl:
        hscl_module = HierarchicalSemanticConsistency(
            hidden_dim=new_config.ahrq_hidden_dim,
            semantic_hierarchy=model.semantic_hierarchy,
            predictor_type="mlp"
        ).to(device)
        hscl_optimizer = torch.optim.AdamW(
            hscl_module.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )

    best_recon_loss = float('inf')
    train_losses = []
    codebook_usage_history = []

    for epoch in range(new_config.stage1_epochs):
        model.train()
        if hscl_module:
            hscl_module.train()
        epoch_losses = []
        epoch_id_metrics = []

        train_bar = tqdm(pretrain_loader, desc=f"Stage1 {config.experiment_name} Epoch {epoch + 1}")

        for batch in train_bar:
            text_feat = batch['text_feat'].float().to(device)
            vision_feat = batch['vision_feat'].float().to(device)

            quantized, indices, raw, quant_loss = model(text_feat, vision_feat)

            loss, loss_dict = compute_rqvae_recon_loss(
                quantized, raw, None, None, new_config, [quant_loss]
            )

            # 如果启用HSCL，计算一致性损失
            if config.use_hscl and hscl_module:
                # 提取各层量化后的特征
                quantized_layers = []
                layer_dim = new_config.ahrq_hidden_dim // model.num_layers
                for layer_idx in range(model.num_layers):
                    layer_feat = quantized[:, layer_idx * layer_dim:(layer_idx + 1) * layer_dim]
                    quantized_layers.append(layer_feat)

                # 计算一致性损失
                consistency_losses = hscl_module.compute_consistency_loss(quantized_layers, indices)
                total_consistency_loss = consistency_losses['total_consistency_loss']

                # 将一致性损失加入总损失
                loss = loss + config.hscl_weight * total_consistency_loss
                loss_dict['consistency_loss'] = total_consistency_loss.item()

            optimizer.zero_grad()
            if hscl_module and config.use_hscl:
                hscl_optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), new_config.grad_clip)
            if hscl_module and config.use_hscl:
                torch.nn.utils.clip_grad_norm_(hscl_module.parameters(), new_config.grad_clip)

            optimizer.step()
            if hscl_module and config.use_hscl:
                hscl_optimizer.step()

            epoch_losses.append(loss.item())

            if config.use_hscl:
                train_bar.set_postfix({
                    "loss": f"{np.mean(epoch_losses):.4f}",
                    "recon": f"{loss_dict.get('rqvae_recon_loss', 0):.6f}",
                    "consistency": f"{loss_dict.get('consistency_loss', 0):.6f}",
                })
            else:
                train_bar.set_postfix({
                    "loss": f"{np.mean(epoch_losses):.4f}",
                    "recon": f"{loss_dict.get('rqvae_recon_loss', 0):.6f}",
                })

        scheduler.step()
        avg_loss = np.mean(epoch_losses)

        # 验证并计算码本利用率
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in pretrain_loader:
                text_feat = batch['text_feat'].float().to(device)
                vision_feat = batch['vision_feat'].float().to(device)
                quantized, indices, raw, quant_loss = model(text_feat, vision_feat)
                loss, loss_dict = compute_rqvae_recon_loss(
                    quantized, raw, None, None, new_config, [quant_loss]
                )
                val_losses.append(loss_dict['rqvae_recon_loss'])

        avg_val_loss = np.mean(val_losses)

        # 计算码本利用率
        all_indices = []
        with torch.no_grad():
            for batch in pretrain_loader:
                text_feat = batch['text_feat'].float().to(device)
                vision_feat = batch['vision_feat'].float().to(device)
                _, indices, _, _ = model(text_feat, vision_feat)
                all_indices.append(indices)
        all_indices = torch.cat(all_indices, dim=0)

        # 获取各层码本大小（按层索引排序，与模型输出顺序一致）
        layer_to_codebook = {}
        for semantic_type, cfg in model.semantic_hierarchy.items():
            for layer_idx in cfg['layers']:
                layer_to_codebook[layer_idx] = cfg['codebook_size']
        n_e_list = [layer_to_codebook[i] for i in sorted(layer_to_codebook.keys())]

        usage_rates = codebook_usage_rate(all_indices, n_e_list)
        avg_usage = np.mean(usage_rates)
        codebook_usage_history.append(avg_usage)

        print(f"{config.experiment_name} Epoch {epoch + 1}: Loss={avg_loss:.4f}, "
              f"Val Recon={avg_val_loss:.6f}, Codebook Usage={avg_usage:.4f}")

        if avg_val_loss < best_recon_loss:
            best_recon_loss = avg_val_loss

        train_losses.append(avg_loss)

    return {
        "final_train_loss": np.mean(train_losses[-5:]),
        "best_val_recon_loss": best_recon_loss,
        "final_codebook_usage": codebook_usage_history[-1] if codebook_usage_history else 0.0,
        "codebook_usage_history": codebook_usage_history
    }


def run_ablation_experiment(
    config: AblationConfig,
    device: torch.device,
    logger
) -> Dict:
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {config.experiment_name}")
    print(f"Config: use_ema={config.use_ema}, use_hscl={config.use_hscl}, use_emotion={config.use_emotion}")
    print(f"{'='*60}")

    seed_everything(new_config.seed)

    # 构建模型
    model, n_e_list, semantic_hierarchy = build_ahrq_model(config, device)

    # 获取数据
    pretrain_loader, all_item_meta = get_all_item_pretrain_dataloader(
        cache_dir="./data",
        category='Video_Games',
        batch_size=new_config.batch_size,
        shuffle=True,
        quick_mode=True,
        num_workers=NUM_WORKS,
        logger=logger
    )

    # Stage 1: 量化预训练
    stage1_results = train_stage1_quantization(
        model, pretrain_loader, config, device, logger
    )

    # 计算各层码本利用率
    model.eval()
    with torch.no_grad():
        all_item_text = all_item_meta['text_features'].float().to(device)
        all_item_vision = all_item_meta['image_features'].float().to(device)
        _, indices_list, _, _ = model(all_item_text, all_item_vision)

    usage_rates = codebook_usage_rate(indices_list, n_e_list)

    # 分析各层码本的聚类质量
    print("  Analyzing codebook clusters...")
    cluster_results = analyze_codebook_clusters(model, device, pretrain_loader, num_clusters=10)
    if cluster_results:
        print("  Silhouette Scores:")
        for layer, info in cluster_results.items():
            if info.get("silhouette_score") is not None:
                print(f"    {layer}: {info['silhouette_score']:.4f}")
            else:
                print(f"    {layer}: {info.get('note', 'N/A')}")

    # 按语义层次分组统计
    layer_usage_by_group = {}
    for semantic_type, cfg in semantic_hierarchy.items():
        layer_indices = cfg['layers']
        group_rates = [usage_rates[i] for i in layer_indices]
        layer_usage_by_group[semantic_type] = {
            'avg': np.mean(group_rates),
            'layers': group_rates
        }

    results = {
        "experiment_name": config.experiment_name,
        "config": {
            "use_ema": config.use_ema,
            "use_hscl": config.use_hscl,
            "use_emotion": config.use_emotion,
            "codebook_sizes": n_e_list,
            "semantic_hierarchy": {k: v['codebook_size'] for k, v in semantic_hierarchy.items()}
        },
        "metrics": {
            "train_loss": stage1_results["final_train_loss"],
            "val_recon_loss": stage1_results["best_val_recon_loss"],
            "codebook_usage": {
                "overall": np.mean(usage_rates),
                "by_layer": usage_rates,
                "by_group": layer_usage_by_group
            },
            "cluster_analysis": cluster_results
        }
    }

    print(f"\n{config.experiment_name} Results:")
    print(f"  Val Recon Loss: {results['metrics']['val_recon_loss']:.6f}")
    print(f"  Codebook Usage (Overall): {results['metrics']['codebook_usage']['overall']:.4f}")
    for group, info in layer_usage_by_group.items():
        print(f"  {group} Usage: {info['avg']:.4f}")

    # 保存模型到 results/ahrq_ablation/models 目录
    model_save_dir = "./results/ahrq_ablation/models"
    os.makedirs(model_save_dir, exist_ok=True)

    model_filename = f"{config.experiment_name.lower().replace('-', '_')}_model.pth"
    model_path = os.path.join(model_save_dir, model_filename)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "experiment_name": config.experiment_name,
            "use_ema": config.use_ema,
            "use_hscl": config.use_hscl,
            "use_emotion": config.use_emotion,
            "semantic_hierarchy": semantic_hierarchy,
            "n_e_list": n_e_list
        },
        "metrics": results['metrics']
    }, model_path)
    print(f"  AHRQ Model saved to: {model_path}")

    # ===== Stage 2: 训练SASRec推荐模型 =====
    stage2_results = train_stage2_recommendation(
        ahrq=model,
        indices_list=indices_list,
        config=config,
        device=device,
        logger=logger
    )

    # 将Stage 2结果添加到最终结果中
    results['stage2_best_val'] = {
        "best_ndcg": stage2_results['best_val_ndcg'],
    }
    results['test_metrics'] = stage2_results['test_metrics']
    results['stage2_train_history'] = stage2_results['train_history']
    results['stage2_val_history'] = stage2_results['val_history']

    return results


def evaluate_test_full(model, test_loader, indices_list, topk_list=[5, 10, 20]):
    """测试集全量排序评估"""
    model.eval()
    metrics_dict = {f'HR@{k}': 0.0 for k in topk_list}
    metrics_dict.update({f'NDCG@{k}': 0.0 for k in topk_list})
    metrics_dict['MRR'] = 0.0
    total_users = 0

    with torch.no_grad():
        all_item_feat = model.get_all_item_sem_feat(indices_list)
        for batch in tqdm(test_loader, desc=f'Full Ranking Evaluation on Test Set'):
            user_emb, _ = model(batch)
            target_idx = batch["target_item"].to(new_config.device) - 1

            all_scores = torch.matmul(user_emb, all_item_feat.T)
            rec_metrics = calculate_metrics(all_scores, target_idx, k_list=topk_list)
            mrr = calculate_mrr_full(all_scores, target_idx, new_config.device)

            for k in topk_list:
                metrics_dict[f'HR@{k}'] += rec_metrics[f'HR@{k}']
                metrics_dict[f'NDCG@{k}'] += rec_metrics[f'NDCG@{k}']
            metrics_dict['MRR'] += mrr
            total_users += 1

    if total_users == 0:
        return {k: 0.0 for k in metrics_dict.keys()}

    for k in topk_list:
        metrics_dict[f'HR@{k}'] /= total_users
        metrics_dict[f'NDCG@{k}'] /= total_users
    metrics_dict['MRR'] /= total_users

    return metrics_dict


def train_stage2_recommendation(
    ahrq,
    indices_list,
    config: AblationConfig,
    device: torch.device,
    logger
) -> Dict:
    """Stage 2: SASRec推荐训练和评估"""
    print(f"\n{'='*60}")
    print(f"Stage 2: Training SASRec with Semantic IDs")
    print(f"{'='*60}")

    # 获取数据加载器
    train_loader, val_loader, test_loader, all_item_features = get_pmat_dataloader(
        cache_dir="./data",
        category='Video_Games',
        batch_size=new_config.batch_size,
        max_history_len=new_config.sasrec_max_len,
        num_negative_samples=new_config.num_negative_samples,
        shuffle=True,
        quick_mode=True,
        num_workers=NUM_WORKS,
        indices_list=indices_list,
        logger=logger
    )

    # 构建SASRecAHRQ模型
    # 使用最佳参数: hidden_dim=64, dropout=0.35
    dynamic_sasrec_params = {
        # 核心基线参数（对齐Pure SASRec）
        "sasrec_num_layers": 2,  # 基线2层
        "dim_feedforward": ahrq.hidden_dim * 4,  # 64*4=256（基线FFN）
        "num_heads": 1,  # 基线1头（修改为1以对齐基线）
        "dropout": 0.35,  # 从配置读取
        "lr_scale": 1.0,  # 学习率不缩放
        "scale": 1.0,  # 基线scale=1.0
    }
    num_items = all_item_features['num_items']
    sasrec_model = SASRecAHRQ(
        ahrq_model=ahrq,
        num_items=num_items,
        fusion_type='add',
        fixed_alpha=0.5,
        dynamic_params=dynamic_sasrec_params
    ).to(device)

    # 训练配置
    rec_params = [p for p in sasrec_model.parameters() if p.requires_grad]
    optimizer_rec = torch.optim.AdamW(
        rec_params,
        lr=new_config.lr,
        weight_decay=new_config.weight_decay
    )
    scheduler_rec = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_rec,
        T_max=new_config.stage2_epochs
    )

    # 训练循环
    best_ndcg = 0.0
    topk_list = [5, 10, 20]
    train_history = []
    val_history = []

    for epoch in range(new_config.stage2_epochs):
        # ===== 训练阶段 =====
        sasrec_model.train()
        train_bar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch + 1}/{new_config.stage2_epochs}")
        train_losses = []
        train_metrics = []

        with torch.no_grad():
            all_item_feat = sasrec_model.get_all_item_sem_feat(indices_list)
            all_item_feat.requires_grad = False

        for batch in train_bar:
            user_emb, pos_sem_feat = sasrec_model(batch)
            logits = torch.matmul(user_emb, all_item_feat.T)
            target_idx = batch["target_item"].to(device) - 1

            loss = F.cross_entropy(logits, target_idx, ignore_index=-1)

            optimizer_rec.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rec_params, new_config.grad_clip)
            optimizer_rec.step()

            train_losses.append(loss.item())
            with torch.no_grad():
                all_scores = torch.matmul(user_emb, all_item_feat.T)
                rec_metrics = calculate_metrics(all_scores, target_idx, k_list=topk_list)
                train_metrics.append(rec_metrics)

            avg_loss = np.mean(train_losses)
            hr10 = rec_metrics.get("HR@10", 0.0)
            ndcg10 = rec_metrics.get("NDCG@10", 0.0)
            train_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "HR@10": f"{hr10:.4f}",
                "NDCG@10": f"{ndcg10:.4f}"
            })

        scheduler_rec.step()

        # 计算训练集平均指标
        avg_train_metrics = {}
        for key in train_metrics[0].keys():
            avg_train_metrics[key] = np.mean([m[key] for m in train_metrics])
        avg_train_loss = np.mean(train_losses)

        # ===== 验证阶段 =====
        sasrec_model.eval()
        val_metrics = []
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                user_emb, pos_sem_feat = sasrec_model(batch)
                logits = torch.matmul(user_emb, all_item_feat.T)
                target_idx = batch["target_item"].to(device) - 1

                loss = F.cross_entropy(logits, target_idx, ignore_index=-1)
                val_losses.append(loss.item())

                all_scores = torch.matmul(user_emb, all_item_feat.T)
                rec_metrics = calculate_metrics(all_scores, target_idx, k_list=topk_list)
                val_metrics.append(rec_metrics)

        avg_val_metrics = {}
        for key in val_metrics[0].keys():
            avg_val_metrics[key] = np.mean([m[key] for m in val_metrics])
        avg_val_loss = np.mean(val_losses)

        # 保存训练历史
        train_history.append({
            "epoch": epoch + 1,
            "loss": avg_train_loss,
            **{f"train_{k}": v for k, v in avg_train_metrics.items()}
        })
        val_history.append({
            "epoch": epoch + 1,
            "loss": avg_val_loss,
            **{f"val_{k}": v for k, v in avg_val_metrics.items()}
        })

        print(f"\n{config.experiment_name} Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train HR@10: {avg_train_metrics['HR@10']:.4f} | Val HR@10: {avg_val_metrics['HR@10']:.4f}")
        print(f"Train NDCG@10: {avg_train_metrics['NDCG@10']:.4f} | Val NDCG@10: {avg_val_metrics['NDCG@10']:.4f}")

        # 保存最佳模型
        if avg_val_metrics["NDCG@10"] > best_ndcg:
            best_ndcg = avg_val_metrics["NDCG@10"]
            output_dir = f"./results/ahrq_ablation/{config.experiment_name}"
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                "stage": 2,
                "epoch": epoch + 1,
                "model_state_dict": sasrec_model.state_dict(),
                "optimizer_rec_state_dict": optimizer_rec.state_dict(),
                "best_ndcg": best_ndcg,
            }, f"{output_dir}/best_sasrec_model.pth")
            print(f"Best SASRec model saved! NDCG@10: {best_ndcg:.4f}")

    # 测试集评估
    print("\n========== Testing on Full Ranking ==========")
    test_metrics = evaluate_test_full(
        model=sasrec_model,
        test_loader=test_loader,
        indices_list=indices_list,
        topk_list=topk_list
    )

    print(f"\n{config.experiment_name} Test Results (Full Ranking):")
    for k in topk_list:
        print(f"HR@{k}: {test_metrics[f'HR@{k}']:.4f} | NDCG@{k}: {test_metrics[f'NDCG@{k}']:.4f}")
    print(f"MRR: {test_metrics['MRR']:.4f}")

    return {
        "best_val_ndcg": best_ndcg,
        "test_metrics": test_metrics,
        "train_history": train_history,
        "val_history": val_history
    }


def save_results(all_results: List[Dict], output_dir: str = "./results/ahrq_ablation"):
    """保存实验结果（含Stage 1和Stage 2）"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存每个实验的详细结果
    for result in all_results:
        exp_name = result['experiment_name']
        filename = f"{exp_name.lower().replace('-', '_')}_results.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved: {filepath}")

    # 保存汇总表格
    summary_data = []
    for result in all_results:
        # 获取测试指标
        test_metrics = result.get('test_metrics', {})

        row = {
            "Experiment": result['experiment_name'],
            "Use EMA": result['config']['use_ema'],
            "Use HSCL": result['config']['use_hscl'],
            "Use Emotion": result['config']['use_emotion'],
            # Stage 1 指标
            "Val Recon Loss": result['metrics']['val_recon_loss'],
            "Codebook Usage": result['metrics']['codebook_usage']['overall'],
            # Stage 2 指标
            "Val NDCG@10": result.get('stage2_best_val', {}).get('best_ndcg', 0.0),
            "Test HR@5": f"{test_metrics.get('HR@5', 0.0):.4f}",
            "Test HR@10": f"{test_metrics.get('HR@10', 0.0):.4f}",
            "Test HR@20": f"{test_metrics.get('HR@20', 0.0):.4f}",
            "Test NDCG@5": f"{test_metrics.get('NDCG@5', 0.0):.4f}",
            "Test NDCG@10": f"{test_metrics.get('NDCG@10', 0.0):.4f}",
            "Test NDCG@20": f"{test_metrics.get('NDCG@20', 0.0):.4f}",
            "Test MRR": f"{test_metrics.get('MRR', 0.0):.4f}",
        }
        # 添加各层使用率
        for i, rate in enumerate(result['metrics']['codebook_usage']['by_layer']):
            row[f"L{i}_Usage"] = f"{rate:.4f}"

        # 添加各层Silhouette Score
        cluster_analysis = result['metrics'].get('cluster_analysis', {})
        for layer in ['L0', 'L1', 'L2', 'L3']:
            if layer in cluster_analysis:
                score = cluster_analysis[layer].get('silhouette_score')
                row[f"{layer}_Silhouette"] = f"{score:.4f}" if score is not None else "N/A"

        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "ablation_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # 同时保存为更易读的格式
    readable_path = os.path.join(output_dir, "ablation_summary_readable.txt")
    with open(readable_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("AHRQ 消融实验结果汇总\n")
        f.write("=" * 80 + "\n\n")

        # 打印Stage 2主要测试指标表格
        f.write("推荐任务测试指标:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Experiment':<20} {'HR@10':>10} {'NDCG@10':>10} {'MRR':>10}\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            test = result.get('test_metrics', {})
            f.write(f"{result['experiment_name']:<20} {test.get('HR@10', 0):>10.4f} {test.get('NDCG@10', 0):>10.4f} {test.get('MRR', 0):>10.4f}\n")
        f.write("-" * 80 + "\n")

        # 打印Stage 1指标
        f.write("\n量化训练指标:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Experiment':<20} {'Recon Loss':>15} {'Codebook Usage':>18}\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            metrics = result['metrics']
            f.write(f"{result['experiment_name']:<20} {metrics['val_recon_loss']:>15.6f} {metrics['codebook_usage']['overall']:>18.4f}\n")
        f.write("-" * 80 + "\n")

        # 打印完整指标
        f.write("\n完整指标:\n")
        for result in all_results:
            test = result.get('test_metrics', {})
            metrics = result['metrics']
            f.write(f"\n{result['experiment_name']}:\n")
            f.write(f"  Stage 1 - Recon Loss: {metrics['val_recon_loss']:.6f}, Codebook Usage: {metrics['codebook_usage']['overall']:.4f}\n")
            f.write(f"  Stage 2 - Test HR@10: {test.get('HR@10', 0):.4f}, NDCG@10: {test.get('NDCG@10', 0):.4f}, MRR: {test.get('MRR', 0):.4f}\n")

    print(f"Saved readable summary to: {readable_path}")

    return df


def analyze_codebook_clusters(model, device, test_loader, num_clusters=10):
    """
    分析各层码本的聚类质量

    使用K-means对各层码本向量进行聚类，计算Silhouette Score来评估聚类质量。
    Silhouette Score范围[-1,1]，越接近1表示聚类质量越好。

    Args:
        model: 训练好的模型（AdaptiveHierarchicalQuantizer）
        device: 设备
        test_loader: 测试数据加载器
        num_clusters: 聚类数量

    Returns:
        dict: 各层的Silhouette Score
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("Warning: sklearn not installed, skipping cluster analysis")
        return {}

    model.eval()
    layer_results = {}

    # 通过model.rq获取码本（ResidualVectorQuantizer或ResidualVectorQuantizerEMA）
    if not hasattr(model, 'rq'):
        print("Warning: model does not have rq attribute")
        return {}

    # 直接访问每个量化层的码本（避免 torch.stack 失败）
    if not hasattr(model.rq, 'vq_layers'):
        print("Warning: model.rq does not have vq_layers attribute")
        return {}

    num_layers = len(model.rq.vq_layers)

    with torch.no_grad():
        for layer_idx in range(num_layers):
            # 直接获取该层的码本向量
            layer_codebook = model.rq.vq_layers[layer_idx].embedding.weight.data.cpu().numpy()  # [codebook_size, dim]
            codebook_size = layer_codebook.shape[0]

            # 如果码本大小小于聚类数，跳过
            if codebook_size < num_clusters:
                layer_results[f"L{layer_idx}"] = {
                    "silhouette_score": None,
                    "note": f"codebook_size ({codebook_size}) < num_clusters ({num_clusters})"
                }
                continue

            # 使用K-means聚类
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(layer_codebook)

            # 计算Silhouette Score
            score = silhouette_score(layer_codebook, labels)
            layer_results[f"L{layer_idx}"] = {
                "silhouette_score": float(score),
                "codebook_size": codebook_size,
                "num_clusters": num_clusters
            }

    return layer_results


def main():
    """运行所有消融实验"""
    logger = Logger("./logs/train_ahrq_ablation.log")
    device = new_config.device

    # 定义实验配置（严格消融：每个实验只改变一个变量）
    # 控制变量：默认使用ema=True, hierarchy_weight=True作为基础配置
    experiments = [
        # 1. Baseline-RQ: 固定码本，无EMA，无HSCL（完全基线）
        AblationConfig(
            experiment_name="Baseline-RQ",
            use_ema=False,
            use_hscl=False,
            use_emotion=False,
            use_hierarchy_weight=False,
            baseline_codebook=512
        ),
        # 2. AHRQ-HierCodebook: 仅测试层次化码本效果（无EMA，无HSCL）
        AblationConfig(
            experiment_name="AHRQ-HierCodebook",
            use_ema=False,  # 与基线相同
            use_hscl=False,  # 与基线相同
            use_emotion=False,
            use_hierarchy_weight=True,  # 仅开启层次化码本配置
        ),
        # 3. AHRQ-EMA: 仅测试EMA效果（保持其他与基线相同）
        AblationConfig(
            experiment_name="AHRQ-EMA",
            use_ema=True,  # 唯一变化
            use_hscl=False,
            use_emotion=False,
            use_hierarchy_weight=False,  # 与基线相同
        ),
        # 4. AHRQ-HSCL: 仅测试HSCL效果（需要EMA作为基础）
        AblationConfig(
            experiment_name="AHRQ-HSCL",
            use_ema=True,  # 需要EMA作为基础
            use_hscl=True,  # 唯一变化
            use_emotion=False,
            use_hierarchy_weight=True,
        ),
        # 5. AHRQ-Full: 完整配置（所有创新点）
        AblationConfig(
            experiment_name="AHRQ-Full",
            use_ema=True,
            use_hscl=True,
            use_emotion=True,
            use_hierarchy_weight=True,
        )
    ]

    all_results = []

    for exp_config in experiments:
        try:
            result = run_ablation_experiment(exp_config, device, logger)
            all_results.append(result)
        except Exception as e:
            print(f"Error running {exp_config.experiment_name}: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    if all_results:
        save_results(all_results)
        print("\n" + "="*60)
        print("All ablation experiments completed!")
        print("="*60)


if __name__ == "__main__":
    main()
"""
SASRec-AHRQ 层级参数搜索脚本

搜索最佳层级配置以提升创新点效果：
- 不同层配置不同的码本大小（金字塔结构）

目标：找到对下游推荐效果最好的层级设置
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
from data_utils import get_pmat_dataloader, get_all_item_pretrain_dataloader
from log import Logger
from our_models.ah_rq import AdaptiveHierarchicalQuantizer, HierarchicalSemanticConsistency
from our_models.sasrec_ahrq import SASRecAHRQ
from utils.utils import (
    calculate_metrics, calculate_id_metrics, seed_everything,
    EarlyStopping, fast_codebook_reset, calculate_mrr_full
)
from utils.loss import compute_rqvae_recon_loss
from metrics import codebook_usage_rate

NUM_WORKS = 0


def analyze_codebook_clusters(model, device, num_clusters=10):
    """
    分析各层码本的聚类质量

    使用K-means对各层码本向量进行聚类，计算Silhouette Score来评估聚类质量。

    Args:
        model: 训练好的模型（AdaptiveHierarchicalQuantizer）
        device: 设备
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

    if not hasattr(model, 'rq'):
        print("Warning: model does not have rq attribute")
        return {}

    if not hasattr(model.rq, 'vq_layers'):
        print("Warning: model.rq does not have vq_layers attribute")
        return {}

    num_layers = len(model.rq.vq_layers)

    with torch.no_grad():
        for layer_idx in range(num_layers):
            layer_codebook = model.rq.vq_layers[layer_idx].embedding.weight.data.cpu().numpy()
            codebook_size = layer_codebook.shape[0]

            if codebook_size < num_clusters:
                layer_results[f"L{layer_idx}"] = {
                    "silhouette_score": None,
                    "note": f"codebook_size ({codebook_size}) < num_clusters ({num_clusters})"
                }
                continue

            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(layer_codebook)

            score = silhouette_score(layer_codebook, labels)
            layer_results[f"L{layer_idx}"] = {
                "silhouette_score": float(score),
                "codebook_size": codebook_size,
                "num_clusters": num_clusters
            }

    return layer_results


def calculate_dynamic_sasrec_params(codebook_sizes: List[int], base_hidden_dim: int = 512) -> dict:
    """
    根据码本大小动态调整SASRec参数

    核心思想：语义ID的信息密度不同，需要不同复杂度的SASRec模型
    - 总bits越多（信息密度高）→ 需要更复杂的模型避免欠拟合
    - 总bits越少（信息密度低）→ 需要更简单的模型避免过拟合

    Args:
        codebook_sizes: 每层的码本大小列表
        base_hidden_dim: 基础隐藏维度

    Returns:
        dict: 动态调整的SASRec参数
    """
    total_bits = sum([np.log2(cb) for cb in codebook_sizes])
    num_layers = len(codebook_sizes)

    # 基准：假设36bits是baseline（对应[256,512,512,512]配置）
    baseline_bits = 36.0
    # 使用更敏感的scale计算
    scale = total_bits / baseline_bits

    # 动态调整参数 - 放大差异
    # 1. Transformer层数：基于总bits绝对值 + scale
    # bits >= 35: 2-3层; bits 30-35: 2层; bits < 30: 1-2层
    if total_bits >= 35:
        sasrec_num_layers = 2
    elif total_bits >= 32:
        sasrec_num_layers = 2
    elif total_bits >= 28:
        sasrec_num_layers = 1
    else:
        sasrec_num_layers = 1

    # 2. FFN维度：基于bits数量线性缩放，范围更大
    # 36bits -> 2048, 29bits -> 1024
    dim_feedforward = int(512 + (total_bits - 28) * 128)
    dim_feedforward = max(512, min(2048, dim_feedforward))

    # 3. Dropout：信息少需要更大的dropout防止过拟合
    # 36bits -> 0.1, 29bits -> 0.15
    dropout = 0.1 + (35 - total_bits) * 0.005
    dropout = min(0.2, max(0.05, dropout))

    # 4. 注意力头数：基于总bits绝对值，必须是hidden_dim的约数
    # 信息密度越高，使用越多的head
    if total_bits >= 34:
        num_heads = 4
    elif total_bits >= 30:
        num_heads = 2
    else:
        num_heads = 1

    # 5. 学习率微调：信息少时用更小的学习率
    lr_scale = max(0.5, min(1.2, scale))

    return {
        "sasrec_num_layers": sasrec_num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "num_heads": num_heads,
        "lr_scale": lr_scale,
        "total_bits": total_bits,
        "scale": scale
    }


@dataclass
class HPSearchConfig:
    """层级参数搜索配置"""
    experiment_name: str
    # 层级配置 - 核心搜索参数
    codebook_sizes: List[int]  # 每层的码本大小，如 [256, 512, 512, 512]
    hidden_dim: int = 512  # 固定隐藏维度
    # AHRQ创新点配置
    use_ema: bool = False      # 是否使用EMA更新和死码重置
    use_hscl: bool = False     # 是否使用层次化语义一致性学习
    use_emotion: bool = False  # 是否使用情感编码
    hscl_weight: float = 0.03  # HSCL损失权重
    # 模型融合参数（固定）
    fusion_type: str = "add"
    alpha: float = 0.5
    # 训练参数
    stage1_epochs: int = 20  # AHRQ预训练轮数
    stage2_epochs: int = 50  # SASRec训练轮数
    lr: float = 1e-4
    # 动态SASRec参数（由calculate_dynamic_sasrec_params计算）
    dynamic_sasrec_params: dict = None


def build_semantic_hierarchy(codebook_sizes: List[int], hidden_dim: int, use_emotion: bool = False) -> dict:
    """
    根据码本大小配置构建语义层次结构

    Args:
        codebook_sizes: 每层的码本大小列表
        hidden_dim: 隐藏维度
        use_emotion: 是否使用情感编码层

    Returns:
        semantic_hierarchy: 语义层次配置字典
    """
    num_layers = len(codebook_sizes)

    # 计算每层的维度
    layer_dim = hidden_dim // num_layers
    if hidden_dim % num_layers != 0:
        print(f"Warning: hidden_dim {hidden_dim} not divisible by {num_layers} layers, using {layer_dim}")

    semantic_hierarchy = {}

    # 确定使用的层数（排除emotion层）
    if use_emotion and num_layers >= 3:
        # 最后几层用于emotion
        num_main_layers = num_layers - 1
        emotion_codebook = codebook_sizes[-1]
    else:
        num_main_layers = num_layers
        emotion_codebook = None

    # 第一层：topic（基础语义）
    semantic_hierarchy["topic"] = {
        "layers": [0],
        "codebook_size": codebook_sizes[0],
        "loss_weight": 1.0,
        "ema_decay": 0.99
    }

    # 后续层：style（风格变体）
    if num_main_layers > 1:
        semantic_hierarchy["style"] = {
            "layers": list(range(1, num_main_layers)),
            "codebook_size": codebook_sizes[1] if len(codebook_sizes) > 1 else codebook_sizes[0],
            "loss_weight": 0.8,
            "ema_decay": 0.99
        }

    # 情感层（如果启用）
    if use_emotion and emotion_codebook is not None:
        semantic_hierarchy["emotion"] = {
            "layers": [num_layers - 1],  # 最后一层
            "codebook_size": emotion_codebook,
            "loss_weight": 0.6,
            "ema_decay": 0.99
        }

    return semantic_hierarchy


def train_ahrq_stage1(
    model,
    pretrain_loader,
    device: torch.device,
    epochs: int = 20,
    use_hscl: bool = False,
    hscl_weight: float = 0.03,
    logger=None
) -> Dict:
    """Stage 1: 训练AHRQ量化器"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # 如果启用HSCL，创建层次一致性模块
    hscl_module = None
    hscl_optimizer = None
    if use_hscl:
        hscl_module = HierarchicalSemanticConsistency(
            hidden_dim=model.hidden_dim,
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

    for epoch in range(epochs):
        model.train()
        if hscl_module:
            hscl_module.train()
        epoch_losses = []

        train_bar = tqdm(pretrain_loader, desc=f"Stage1 Epoch {epoch + 1}/{epochs}")

        for batch in train_bar:
            text_feat = batch['text_feat'].float().to(device)
            vision_feat = batch['vision_feat'].float().to(device)

            quantized, indices, raw, quant_loss = model(text_feat, vision_feat)

            loss, loss_dict = compute_rqvae_recon_loss(
                quantized, raw, None, None, new_config, [quant_loss]
            )

            # 如果启用HSCL，计算一致性损失
            if use_hscl and hscl_module:
                # 提取各层量化后的特征
                quantized_layers = []
                layer_dim = model.hidden_dim // model.num_layers
                for layer_idx in range(model.num_layers):
                    layer_feat = quantized[:, layer_idx * layer_dim:(layer_idx + 1) * layer_dim]
                    quantized_layers.append(layer_feat)

                # 计算一致性损失
                consistency_losses = hscl_module.compute_consistency_loss(quantized_layers, indices)
                total_consistency_loss = consistency_losses['total_consistency_loss']

                # 将一致性损失加入总损失
                loss = loss + hscl_weight * total_consistency_loss
                loss_dict['consistency_loss'] = total_consistency_loss.item()

            optimizer.zero_grad()
            if hscl_optimizer:
                hscl_optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), new_config.grad_clip)
            if hscl_module:
                torch.nn.utils.clip_grad_norm_(hscl_module.parameters(), new_config.grad_clip)

            optimizer.step()
            if hscl_optimizer:
                hscl_optimizer.step()

            epoch_losses.append(loss.item())
            if use_hscl:
                train_bar.set_postfix({
                    "loss": f"{np.mean(epoch_losses):.4f}",
                    "recon": f"{loss_dict.get('rqvae_recon_loss', 0):.6f}",
                    "consistency": f"{loss_dict.get('consistency_loss', 0):.6f}"
                })
            else:
                train_bar.set_postfix({
                    "loss": f"{np.mean(epoch_losses):.4f}",
                    "recon": f"{loss_dict.get('rqvae_recon_loss', 0):.6f}"
                })

        scheduler.step()
        avg_loss = np.mean(epoch_losses)

        # 验证
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
        if avg_val_loss < best_recon_loss:
            best_recon_loss = avg_val_loss

        train_losses.append(avg_loss)
        print(f"Stage1 Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Recon={avg_val_loss:.6f}")

    return {
        "final_train_loss": np.mean(train_losses[-5:]),
        "best_val_recon_loss": best_recon_loss,
        "hscl_module": hscl_module
    }


def evaluate_test_full(model, test_loader, indices_list, topk_list=[5, 10, 20]):
    """测试集全量排序评估"""
    model.eval()
    metrics_dict = {f'HR@{k}': 0.0 for k in topk_list}
    metrics_dict.update({f'NDCG@{k}': 0.0 for k in topk_list})
    metrics_dict['MRR'] = 0.0
    total_users = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Full Ranking Evaluation'):
            # 使用新的batch格式
            user_emb = model.get_user_embedding(batch)
            target_idx = batch["target_item"].to(new_config.device) - 1

            # 全量物品打分
            all_scores = model.predict_all(batch, indices_list)

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


def train_single_config(
    hp_config: HPSearchConfig,
    device: torch.device,
    logger: Logger
) -> Dict:
    """运行单个层级配置实验"""
    print(f"\n{'='*60}")
    print(f"Running: {hp_config.experiment_name}")
    print(f"  Codebook sizes: {hp_config.codebook_sizes}")
    print(f"  Hidden dim: {hp_config.hidden_dim}")
    print(f"  AHRQ创新点配置: use_ema={hp_config.use_ema}, use_hscl={hp_config.use_hscl}, use_emotion={hp_config.use_emotion}")
    if hp_config.use_hscl:
        print(f"  HSCL weight: {hp_config.hscl_weight}")
    print(f"  Stage1 epochs: {hp_config.stage1_epochs}, Stage2 epochs: {hp_config.stage2_epochs}")
    print(f"{'='*60}")

    seed_everything(new_config.seed)

    # 输出目录
    OUTPUT_DIR = f"./results/sasrec_ahrq_hp_search/{hp_config.experiment_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # ===== Stage 1: 训练AHRQ量化器 =====
    print(f"\n{'='*60}")
    print("Stage 1: Training AHRQ Quantizer")
    print(f"{'='*60}")

    # 构建语义层次结构
    semantic_hierarchy = build_semantic_hierarchy(
        hp_config.codebook_sizes,
        hp_config.hidden_dim,
        use_emotion=hp_config.use_emotion
    )

    # 创建AHRQ模型
    ahrq = AdaptiveHierarchicalQuantizer(
        hidden_dim=hp_config.hidden_dim,
        semantic_hierarchy=semantic_hierarchy,
        use_multimodal=True,
        text_dim=new_config.text_dim,
        visual_dim=new_config.visual_dim,
        beta=new_config.ahrq_beta,
        use_ema=hp_config.use_ema,
        ema_decay=0.99,
        reset_unused_codes=hp_config.use_ema,
        reset_threshold=50,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.0 if hp_config.use_ema else 0.0,
        sk_iters=100,
        dropout=0.1,
        bn=True
    ).to(device)

    # 训练Stage 1
    stage1_results = train_ahrq_stage1(
        ahrq, pretrain_loader, device,
        epochs=hp_config.stage1_epochs,
        use_hscl=hp_config.use_hscl,
        hscl_weight=hp_config.hscl_weight,
        logger=logger
    )

    # 提取所有物品的语义ID
    print("\nExtracting all item semantics...")
    ahrq.eval()
    all_item_text = all_item_meta['text_features'].float().to(device)
    all_item_vision = all_item_meta['image_features'].float().to(device)
    _, indices_list, _, _ = ahrq(all_item_text, all_item_vision)

    # 计算码本使用率
    n_e_list = hp_config.codebook_sizes
    usage_rates = codebook_usage_rate(indices_list, n_e_list)
    print(f"Codebook usage rates: {[f'{r:.4f}' for r in usage_rates]}")

    # 分析码本聚类质量 (Silhouette Score)
    print("\nAnalyzing codebook clusters...")
    cluster_results = analyze_codebook_clusters(ahrq, device, num_clusters=10)
    if cluster_results:
        for layer, info in cluster_results.items():
            if info.get("silhouette_score") is not None:
                print(f"  {layer} Silhouette Score: {info['silhouette_score']:.4f}")
            else:
                print(f"  {layer}: {info.get('note', 'N/A')}")

    # 构建完整的语义ID质量评估结果
    semantic_id_quality = {
        "codebook_usage_rates": usage_rates,
        "avg_usage_rate": float(np.mean(usage_rates)),
        "min_usage_rate": float(np.min(usage_rates)),
        "max_usage_rate": float(np.max(usage_rates)),
        "usage_std": float(np.std(usage_rates)),
        "cluster_analysis": cluster_results,
        "total_bits": float(sum([np.log2(cb) for cb in hp_config.codebook_sizes])),
        "effective_bits": float(sum([u * np.log2(cb) for u, cb in zip(usage_rates, hp_config.codebook_sizes)])),
        "val_recon_loss": stage1_results.get('best_val_recon_loss', 0.0),
        "final_train_loss": stage1_results.get('final_train_loss', 0.0)
    }

    # 保存Stage 1结果
    stage1_save_path = f"{OUTPUT_DIR}/stage1_model.pth"
    torch.save({
        "model_state_dict": ahrq.state_dict(),
        "semantic_hierarchy": semantic_hierarchy,
        "codebook_sizes": hp_config.codebook_sizes,
        "stage1_results": stage1_results,
        "codebook_usage": usage_rates,
        "semantic_id_quality": semantic_id_quality,
        "cluster_analysis": cluster_results
    }, stage1_save_path)
    print(f"Stage 1 model saved to: {stage1_save_path}")

    # ===== Stage 2: 训练SASRec =====
    print(f"\n{'='*60}")
    print("Stage 2: Training SASRec with Semantic IDs")
    print(f"{'='*60}")

    # 创建SASRecAHRQ模型
    num_items = all_item_meta['text_features'].shape[0]

    # 计算动态SASRec参数（基于码本配置）
    dynamic_sasrec_params = calculate_dynamic_sasrec_params(
        codebook_sizes=hp_config.codebook_sizes,
        base_hidden_dim=hp_config.hidden_dim
    )

    # 打印动态参数配置
    print(f"\n>>> Dynamic SASRec Parameters (based on codebook {hp_config.codebook_sizes}):")
    print(f"    Total bits: {dynamic_sasrec_params['total_bits']:.2f}, Scale: {dynamic_sasrec_params['scale']:.2f}")
    print(f"    Transformer layers: {dynamic_sasrec_params['sasrec_num_layers']}, FFN dim: {dynamic_sasrec_params['dim_feedforward']}")
    print(f"    Dropout: {dynamic_sasrec_params['dropout']:.3f}, Num heads: {dynamic_sasrec_params['num_heads']}")
    print(f"    Learning rate scale: {dynamic_sasrec_params['lr_scale']:.2f}")

    model = SASRecAHRQ(
        ahrq_model=ahrq,
        num_items=num_items,
        fusion_type=hp_config.fusion_type,
        fixed_alpha=hp_config.alpha,
        dynamic_params=dynamic_sasrec_params
    ).to(device)

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

    # 训练配置 - 使用动态学习率缩放
    rec_params = [p for p in model.parameters() if p.requires_grad]
    dynamic_lr = hp_config.lr * dynamic_sasrec_params.get('lr_scale', 1.0)
    optimizer_rec = torch.optim.AdamW(
        rec_params,
        lr=dynamic_lr,
        weight_decay=new_config.weight_decay
    )
    scheduler_rec = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_rec,
        T_max=hp_config.stage2_epochs
    )

    # 训练循环
    best_ndcg = 0.0
    topk_list = [5, 10, 20]
    train_history = []
    val_history = []

    for epoch in range(hp_config.stage2_epochs):
        # ===== 训练阶段 =====
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{hp_config.stage2_epochs}")
        train_losses = []
        train_metrics = []

        with torch.no_grad():
            all_item_feat = model.get_all_item_sem_feat(indices_list)
            all_item_feat.requires_grad = False

        for batch in train_bar:
            user_emb, pos_sem_feat = model(batch)
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
        model.eval()
        val_metrics = []
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                user_emb, pos_sem_feat = model(batch)
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

        print(f"\n{hp_config.experiment_name} Epoch {epoch + 1}:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train HR@10: {avg_train_metrics['HR@10']:.4f} | Val HR@10: {avg_val_metrics['HR@10']:.4f}")
        print(f"Train NDCG@10: {avg_train_metrics['NDCG@10']:.4f} | Val NDCG@10: {avg_val_metrics['NDCG@10']:.4f}")

        # 保存最佳模型
        if avg_val_metrics["NDCG@10"] > best_ndcg:
            best_ndcg = avg_val_metrics["NDCG@10"]
            torch.save({
                "stage": 2,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_rec_state_dict": optimizer_rec.state_dict(),
                "best_ndcg": best_ndcg,
                "config": hp_config.__dict__
            }, f"{OUTPUT_DIR}/best_model.pth")
            print(f"Best model saved! NDCG@10: {best_ndcg:.4f}")

    # ===== 测试集评估 =====
    print("\n========== Testing on Full Ranking ==========")
    test_metrics = evaluate_test_full(
        model=model,
        test_loader=test_loader,
        indices_list=indices_list,
        topk_list=topk_list
    )

    print(f"\n{hp_config.experiment_name} Test Results:")
    for k in topk_list:
        print(f"HR@{k}: {test_metrics[f'HR@{k}']:.4f} | NDCG@{k}: {test_metrics[f'NDCG@{k}']:.4f}")
    print(f"MRR: {test_metrics['MRR']:.4f}")

    # ===== 汇总结果 =====
    results = {
        "experiment_name": hp_config.experiment_name,
        "config": {
            "codebook_sizes": hp_config.codebook_sizes,
            "hidden_dim": hp_config.hidden_dim,
            # AHRQ创新点配置
            "use_ema": hp_config.use_ema,
            "use_hscl": hp_config.use_hscl,
            "use_emotion": hp_config.use_emotion,
            "hscl_weight": hp_config.hscl_weight,
            "fusion_type": hp_config.fusion_type,
            "alpha": hp_config.alpha,
            "stage1_epochs": hp_config.stage1_epochs,
            "stage2_epochs": hp_config.stage2_epochs,
            "lr": hp_config.lr
        },
        # 动态SASRec参数
        "dynamic_sasrec_params": {
            "total_bits": dynamic_sasrec_params["total_bits"],
            "scale": dynamic_sasrec_params["scale"],
            "sasrec_num_layers": dynamic_sasrec_params["sasrec_num_layers"],
            "dim_feedforward": dynamic_sasrec_params["dim_feedforward"],
            "dropout": dynamic_sasrec_params["dropout"],
            "num_heads": dynamic_sasrec_params["num_heads"],
            "lr_scale": dynamic_sasrec_params["lr_scale"]
        },
        "stage1": {
            "best_val_recon_loss": stage1_results["best_val_recon_loss"],
            "codebook_usage": usage_rates,
            "semantic_id_quality": semantic_id_quality
        },
        "stage2_best_val": {
            "best_ndcg": best_ndcg,
            "best_epoch": next((h["epoch"] for h in val_history if h["val_NDCG@10"] == best_ndcg), -1)
        },
        "test_metrics": test_metrics,
        "train_history": train_history,
        "val_history": val_history
    }

    # 保存详细结果到JSON
    result_path = f"./results/sasrec_ahrq_hp_search/{hp_config.experiment_name}_results.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {result_path}")

    return results


def save_hp_search_summary(all_results: List[Dict], output_dir: str = "./results/sasrec_ahrq_hp_search"):
    """保存层级搜索汇总表格"""
    os.makedirs(output_dir, exist_ok=True)

    summary_data = []
    for result in all_results:
        test_metrics = result['test_metrics']
        stage2_best = result['stage2_best_val']
        config = result['config']
        semantic_quality = result.get('stage1', {}).get('semantic_id_quality', {})

        # 获取动态SASRec参数
        dynamic_params = result.get('dynamic_sasrec_params', {})

        row = {
            "Experiment": result['experiment_name'],
            "Codebook_Sizes": str(config['codebook_sizes']),
            "Layers": len(config['codebook_sizes']),
            # AHRQ创新点配置
            "Use_EMA": config.get('use_ema', False),
            "Use_HSCL": config.get('use_hscl', False),
            "Use_Emotion": config.get('use_emotion', False),
            # 动态SASRec参数
            "SASRec_Layers": dynamic_params.get('sasrec_num_layers', 2),
            "FFN_Dim": dynamic_params.get('dim_feedforward', 2048),
            "Dropout": f"{dynamic_params.get('dropout', 0.1):.3f}",
            "Num_Heads": dynamic_params.get('num_heads', 4),
            # Stage 1指标 - 语义ID质量
            "Stage1_Recon_Loss": f"{result['stage1']['best_val_recon_loss']:.6f}",
            "Stage1_Codebook_Usage": f"{np.mean(result['stage1']['codebook_usage']):.4f}",
            "Total_Bits": f"{semantic_quality.get('total_bits', 0):.2f}",
            "Effective_Bits": f"{semantic_quality.get('effective_bits', 0):.2f}",
            # 测试集指标
            "Test HR@5": f"{test_metrics['HR@5']:.4f}",
            "Test HR@10": f"{test_metrics['HR@10']:.4f}",
            "Test HR@20": f"{test_metrics['HR@20']:.4f}",
            "Test NDCG@5": f"{test_metrics['NDCG@5']:.4f}",
            "Test NDCG@10": f"{test_metrics['NDCG@10']:.4f}",
            "Test NDCG@20": f"{test_metrics['NDCG@20']:.4f}",
            "Test MRR": f"{test_metrics['MRR']:.4f}",
            "Val NDCG@10": f"{stage2_best['best_ndcg']:.4f}",
        }
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "hp_search_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSaved HP search summary to: {summary_path}")

    # 打印主要指标表格
    print("\n" + "=" * 120)
    print("Hierarchy Search Results Summary (语义ID质量 vs 推荐效果)")
    print("=" * 120)
    print(f"{'Experiment':<25} {'Codebook':<20} {'EMA':>4} {'HSCL':>5} {'Usage':>8} {'EffBits':>8} {'HR@10':>8} {'NDCG@10':>8} {'MRR':>8}")
    print("-" * 120)
    for result in all_results:
        test = result['test_metrics']
        config = result['config']
        codebook_str = str(config['codebook_sizes'])[:18]
        semantic_quality = result.get('stage1', {}).get('semantic_id_quality', {})
        usage = np.mean(result['stage1']['codebook_usage'])
        eff_bits = semantic_quality.get('effective_bits', 0)
        ema = "Yes" if config.get('use_ema', False) else "No"
        hscl = "Yes" if config.get('use_hscl', False) else "No"
        print(f"{result['experiment_name']:<25} {codebook_str:<20} {ema:>4} {hscl:>5} {usage:>8.4f} {eff_bits:>8.2f} {test['HR@10']:>8.4f} {test['NDCG@10']:>8.4f} {test['MRR']:>8.4f}")
    print("-" * 120)
    print(f"Pure SASRec Baseline:       HR@10=0.1704, NDCG@10=0.0814")
    print("=" * 120)

    return df


def main():
    """运行层级参数搜索"""
    logger = Logger("./logs/train_sasrec_ahrq_hp_search.log")
    device = new_config.device

    # 定义搜索配置 - 基于金字塔设计理念
    # 第一层小（捕获基础类别），后续层大（捕获变体）
    search_configs = [
        # 1. Baseline: 当前配置（对比基准）- 使用AHRQ-Full（所有创新点）
        HPSearchConfig(
            experiment_name="ahrq_full_baseline",
            codebook_sizes=[256, 512, 512, 512],
            hidden_dim=512,
            use_ema=True,       # EMA更新+死码重置
            use_hscl=True,      # 层次化语义一致性学习
            use_emotion=True,   # 情感编码
            hscl_weight=0.03,
            stage1_epochs=20,
            stage2_epochs=30,
            lr=1e-4
        ),
        # 2. 第一层减小（AHRQ-Full模式）
        HPSearchConfig(
            experiment_name="ahrq_full_small_first",
            codebook_sizes=[64, 512, 512, 512],
            hidden_dim=512,
            use_ema=True,
            use_hscl=True,
            use_emotion=True,
            hscl_weight=0.03,
            stage1_epochs=20,
            stage2_epochs=30,
            lr=1e-4
        ),
        # 3. 第一层小，后续层大（AHRQ-Full模式）
        HPSearchConfig(
            experiment_name="ahrq_full_small_first_large_rest",
            codebook_sizes=[64, 1024, 1024, 1024],
            hidden_dim=512,
            use_ema=True,
            use_hscl=True,
            use_emotion=True,
            hscl_weight=0.03,
            stage1_epochs=20,
            stage2_epochs=30,
            lr=1e-4
        ),
        # 4. 金字塔结构（AHRQ-Full模式）
        HPSearchConfig(
            experiment_name="ahrq_full_pyramid",
            codebook_sizes=[64, 256, 512, 1024],
            hidden_dim=512,
            use_ema=True,
            use_hscl=True,
            use_emotion=True,
            hscl_weight=0.03,
            stage1_epochs=20,
            stage2_epochs=30,
            lr=1e-4
        ),
        # 5. 小型金字塔（AHRQ-Full模式）
        HPSearchConfig(
            experiment_name="ahrq_full_small_pyramid",
            codebook_sizes=[32, 128, 256, 512],
            hidden_dim=512,
            use_ema=True,
            use_hscl=True,
            use_emotion=True,
            hscl_weight=0.03,
            stage1_epochs=20,
            stage2_epochs=30,
            lr=1e-4
        ),
    ]

    all_results = []

    for hp_config in search_configs:
        try:
            result = train_single_config(hp_config, device, logger)
            all_results.append(result)
        except Exception as e:
            print(f"Error running {hp_config.experiment_name}: {e}")
            import traceback
            traceback.print_exc()

    # 保存汇总结果
    if all_results:
        save_hp_search_summary(all_results)
        print("\n" + "=" * 60)
        print("All Hierarchy search experiments completed!")
        print("=" * 60)


if __name__ == "__main__":
    main()
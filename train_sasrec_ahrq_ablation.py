"""
SASRec-AHRQ 消融实验脚本

在Stage 2 推荐训练阶段对不同AHRQ模型配置进行消融实验

实验配置（与Stage 1对应）：
- Baseline-RQ: 原始RQ-VAE（8层等码本，无EMA，无HSCL）
- AHRQ-HierCodebook: 自适应层次化码本
- AHRQ-EMA: +EMA更新+死码重置
- AHRQ-HSCL: +层次化语义一致性学习(HSCL)
- AHRQ-Full: 完整配置（所有创新点）

评估指标：
- 训练/验证/测试集：HR@5/10/20, NDCG@5/10/20, MRR
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd

from config import new_config
from data_utils import get_pmat_dataloader, get_all_item_pretrain_dataloader
from log import Logger
from metrics import codebook_usage_rate
from our_models.ah_rq import AdaptiveHierarchicalQuantizer, HierarchicalSemanticConsistency
from our_models.sasrec_ahrq import SASRecAHRQ
from utils.loss import compute_rqvae_recon_loss
from utils.utils import (
    calculate_metrics, calculate_id_metrics, seed_everything,
    EarlyStopping, fast_codebook_reset, calculate_mrr_full
)

NUM_WORKS = 0


@dataclass
class AblationExperimentConfig:
    """消融实验配置"""
    experiment_name: str
    ahrq_model_name: str  # 对应results/ahrq_ablation/models/中的模型文件名
    use_ema: bool
    use_hscl: bool
    use_emotion: bool
    dropout: float
    hidden_dim: int
    semantic_hierarchy: dict


def evaluate_test_full(model, test_loader, indices_list, topk_list=[5, 10, 20]):
    """测试集全量排序评估 - 支持多k值"""
    model.eval()
    metrics_dict = {f'HR@{k}': 0.0 for k in topk_list}
    metrics_dict.update({f'NDCG@{k}': 0.0 for k in topk_list})
    metrics_dict['MRR'] = 0.0
    total_users = 0
    num_items = indices_list.shape[0]

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


def train_single_experiment(
    exp_config: AblationExperimentConfig,
    device: torch.device,
    logger: Logger
) -> Dict:
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {exp_config.experiment_name}")
    print(f"Loading AHRQ model: {exp_config.ahrq_model_name}")
    print(f"{'='*60}")

    seed_everything(new_config.seed)

    # 输出目录
    OUTPUT_DIR = f"./results/sasrec_ahrq_ablation/{exp_config.experiment_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)



    ahrq = AdaptiveHierarchicalQuantizer(
        hidden_dim=new_config.ahrq_hidden_dim,
        semantic_hierarchy=exp_config.semantic_hierarchy,
        use_multimodal=True,
        text_dim=new_config.text_dim,
        visual_dim=new_config.visual_dim,
        beta=new_config.ahrq_beta,
        use_ema=exp_config.use_ema,
        ema_decay=0.99,
        reset_unused_codes=exp_config.use_ema,
        reset_threshold=new_config.ahrq_reset_threshold
    ).to(device)
    pretrain_loader, all_item_meta = get_all_item_pretrain_dataloader(
        cache_dir="./data",
        category='Video_Games',
        batch_size=new_config.batch_size,
        shuffle=True,
        quick_mode=True,
        num_workers=NUM_WORKS,
        logger=logger
    )
    stage1_results = train_stage1_quantization(
        ahrq, pretrain_loader, exp_config, device, logger
    )

    # 计算各层码本利用率
    ahrq.eval()
    with torch.no_grad():
        all_item_text = all_item_meta['text_features'].float().to(device)
        all_item_vision = all_item_meta['image_features'].float().to(device)
        _, indices_list, _, _ = ahrq(all_item_text, all_item_vision)

    usage_rates = codebook_usage_rate(indices_list, exp_config.semantic_hierarchy)

    # 2. 获取数据
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

    num_items = all_item_features['num_items']

    # 3. 创建SASRecAHRQ模型和数据加载器
    # 使用最佳参数配置：hidden_dim=64, dropout=0.35
    model = SASRecAHRQ(
        ahrq_model=ahrq,
        hidden_dim=exp_config.hidden_dim,
        dynamic_params={"dropout": exp_config.dropout},
        num_items=num_items
    ).to(device)


    # 4. 训练配置
    rec_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_rec = torch.optim.AdamW(
        rec_params,
        lr=new_config.lr,
        weight_decay=new_config.weight_decay
    )
    scheduler_rec = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_rec,
        T_max=new_config.stage2_epochs
    )

    # 5. 训练循环
    best_ndcg = 0.0
    topk_list = [5, 10, 20]
    train_history = []
    val_history = []

    for epoch in range(new_config.stage2_epochs):
        # ===== 训练阶段 =====
        model.train()
        train_bar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch + 1}/{new_config.stage2_epochs}")
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

        print(f"\n{exp_config.experiment_name} Epoch {epoch + 1} Summary:")
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
                "ahrq_model_name": exp_config.ahrq_model_name
            }, f"{OUTPUT_DIR}/best_model.pth")
            print(f"Best model saved! NDCG@10: {best_ndcg:.4f}")

    # 保存最终模型
    torch.save({
        "model_state_dict": model.state_dict(),
        "ahrq_model_name": exp_config.ahrq_model_name
    }, f"{OUTPUT_DIR}/final_model.pth")

    # 6. 测试集评估
    print("\n========== Testing on Full Ranking ==========")
    test_metrics = evaluate_test_full(
        model=model,
        test_loader=test_loader,
        indices_list=indices_list,
        topk_list=topk_list
    )

    print(f"\n{exp_config.experiment_name} Test Results (Full Ranking):")
    for k in topk_list:
        print(f"HR@{k}: {test_metrics[f'HR@{k}']:.4f} | NDCG@{k}: {test_metrics[f'NDCG@{k}']:.4f}")
    print(f"MRR: {test_metrics['MRR']:.4f}")

    # 7. 汇总结果
    results = {
        "experiment_name": exp_config.experiment_name,
        "ahrq_model_name": exp_config.ahrq_model_name,
        "config": {
            "use_ema": exp_config.use_ema,
            "use_hscl": exp_config.use_hscl,
            "use_emotion": exp_config.use_emotion,
            "dropout": exp_config.dropout,
            "hidden_dim": exp_config.hidden_dim,
            "codebook_sizes": exp_config.semantic_hierarchy,
        },
        "stage1_metrics": {
            "best_val_recon_loss": stage1_results['best_val_recon_loss']
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
    result_path = f"./results/sasrec_ahrq_ablation/{exp_config.experiment_name}_results.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {result_path}")

    return results

def train_stage1_quantization(
    model,
    pretrain_loader,
    config: AblationExperimentConfig,
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


def save_ablation_summary(all_results: List[Dict], output_dir: str = "./results/sasrec_ahrq_ablation"):
    """保存消融实验汇总表格"""
    os.makedirs(output_dir, exist_ok=True)

    summary_data = []
    for result in all_results:
        test_metrics = result['test_metrics']
        stage2_best = result['stage2_best_val']

        row = {
            "Experiment": result['experiment_name'],
            "AHRQ Model": result['ahrq_model_name'],
            "Use EMA": result['config']['use_ema'],
            "Use HSCL": result['config']['use_hscl'],
            "Use Emotion": result['config']['use_emotion'],
            "Dropout": result['config']['dropout'],
            "Hidden Dim": result['config']['hidden_dim'],
            "Codebook Sizes": str(result['config']['codebook_sizes']),
            # Stage 1 指标
            "Stage1 Recon Loss": result['stage1_metrics']['best_val_recon_loss'],
            # Stage 2 最佳验证指标
            "Val NDCG@10": stage2_best['best_ndcg'],
            "Best Epoch": stage2_best['best_epoch'],
            # 测试集指标
            "Test HR@5": f"{test_metrics['HR@5']:.4f}",
            "Test HR@10": f"{test_metrics['HR@10']:.4f}",
            "Test HR@20": f"{test_metrics['HR@20']:.4f}",
            "Test NDCG@5": f"{test_metrics['NDCG@5']:.4f}",
            "Test NDCG@10": f"{test_metrics['NDCG@10']:.4f}",
            "Test NDCG@20": f"{test_metrics['NDCG@20']:.4f}",
            "Test MRR": f"{test_metrics['MRR']:.4f}",
        }
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "ablation_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSaved ablation summary to: {summary_path}")

    # 同时保存为更易读的格式
    readable_path = os.path.join(output_dir, "ablation_summary_readable.txt")
    with open(readable_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SASRec-AHRQ 消融实验结果汇总\n")
        f.write("=" * 80 + "\n\n")

        # 打印主要指标表格
        f.write("主要测试指标:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Experiment':<20} {'HR@10':>10} {'NDCG@10':>10} {'MRR':>10}\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            test = result['test_metrics']
            f.write(f"{result['experiment_name']:<20} {test['HR@10']:>10.4f} {test['NDCG@10']:>10.4f} {test['MRR']:>10.4f}\n")
        f.write("-" * 80 + "\n")

        # 打印完整指标
        f.write("\n完整指标:\n")
        for result in all_results:
            f.write(f"\n{result['experiment_name']}:\n")
            test = result['test_metrics']
            f.write(f"  HR@5: {test['HR@5']:.4f}, HR@10: {test['HR@10']:.4f}, HR@20: {test['HR@20']:.4f}\n")
            f.write(f"  NDCG@5: {test['NDCG@5']:.4f}, NDCG@10: {test['NDCG@10']:.4f}, NDCG@20: {test['NDCG@20']:.4f}\n")
            f.write(f"  MRR: {test['MRR']:.4f}\n")
            f.write(f"  Stage1 Recon Loss: {result['stage1_metrics']['best_val_recon_loss']:.6f}\n")
            f.write(f"  Val NDCG@10: {result['stage2_best_val']['best_ndcg']:.4f}\n")

    print(f"Saved readable summary to: {readable_path}")

    return df


def main():
    """运行所有消融实验"""
    logger = Logger("./logs/train_sasrec_ahrq_ablation.log")
    device = new_config.device

    # 定义实验配置（与Stage 1对应）- 使用最佳参数: dropout=0.35, hidden_dim=64
    experiments = [

        AblationExperimentConfig(
            experiment_name="AHRQ-HierCodebook",
            ahrq_model_name="ahrq_hiercodebook",
            use_ema=False,
            use_hscl=False,
            use_emotion=False,
            dropout=0.5,
            hidden_dim=128,
            semantic_hierarchy={
                "topic": {
                    "layers": [0],
                    "codebook_size": 512,
                    "loss_weight": 1.0,
                    "ema_decay": 0.99
                },
                "style": {
                    "layers": [1, 2, 3],
                    "codebook_size": 512,
                    "loss_weight": 0.8,
                    "ema_decay": 0.99
                }
            }
        ),
        AblationExperimentConfig(
            experiment_name="AHRQ-EMA",
            ahrq_model_name="ahrq_ema",
            use_ema=True,
            use_hscl=False,
            use_emotion=False,
            dropout=0.35,
            hidden_dim=64,
            semantic_hierarchy={
                "topic": {
                    "layers": [0],
                    "codebook_size": 512,
                    "loss_weight": 1.0,
                    "ema_decay": 0.99
                },
                "style": {
                    "layers": [1, 2, 3],
                    "codebook_size": 512,
                    "loss_weight": 0.8,
                    "ema_decay": 0.99
                }
            }
        ),
        AblationExperimentConfig(
            experiment_name="AHRQ-HSCL",
            ahrq_model_name="ahrq_hscl",
            use_ema=True,
            use_hscl=True,
            use_emotion=False,
            dropout=0.3,
            hidden_dim=128,
            semantic_hierarchy={
                "topic": {
                    "layers": [0],
                    "codebook_size": 512,
                    "loss_weight": 1.0,
                    "ema_decay": 0.99
                },
                "style": {
                    "layers": [1, 2, 3],
                    "codebook_size": 512,
                    "loss_weight": 0.8,
                    "ema_decay": 0.99
                }
            }
        ),
        AblationExperimentConfig(
            experiment_name="AHRQ-Full",
            ahrq_model_name="ahrq_full",
            use_ema=True,
            use_hscl=True,
            use_emotion=True,
            dropout=0.3,
            hidden_dim=128,
            semantic_hierarchy={
                "topic": {
                    "layers": [0],
                    "codebook_size": 512,
                    "loss_weight": 1.0,
                    "ema_decay": 0.99
                },
                "style": {
                    "layers": [1, 2, 3],
                    "codebook_size": 512,
                    "loss_weight": 0.8,
                    "ema_decay": 0.99
                }
            }
        ),
    ]

    all_results = []

    for exp_config in experiments:
        try:
            result = train_single_experiment(exp_config, device, logger)
            all_results.append(result)
        except Exception as e:
            print(f"Error running {exp_config.experiment_name}: {e}")
            import traceback
            traceback.print_exc()

    # 保存汇总结果
    if all_results:
        save_ablation_summary(all_results)
        print("\n" + "=" * 60)
        print("All ablation experiments completed!")
        print("=" * 60)


if __name__ == "__main__":
    main()
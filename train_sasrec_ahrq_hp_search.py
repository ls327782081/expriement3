"""
SASRec-AHRQ 参数搜索脚本

搜索最佳参数配置以提升创新点效果：
1. 融合权重 alpha: [0.0, 0.3, 0.5, 0.7, 1.0]
2. 融合方式: ["add", "concat", "none"]
3. 隐藏维度: [256, 512]
4. 训练轮数: [20, 30, 50]

目标：接近或超过 Pure SASRec (HR@10=0.1704)
"""

import os
import json
import itertools
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
from our_models.ah_rq import AdaptiveHierarchicalQuantizer
from our_models.sasrec_ahrq import SASRecAHRQ
from utils.utils import (
    calculate_metrics, calculate_id_metrics, seed_everything,
    EarlyStopping, fast_codebook_reset, calculate_mrr_full
)

NUM_WORKS = 0


@dataclass
class HPSearchConfig:
    """超参数搜索配置"""
    experiment_name: str
    # 模型参数
    fusion_type: str = "add"  # "add", "concat", "none"
    alpha: float = 0.5  # 融合权重
    hidden_dim: Optional[int] = None  # 隐藏维度（None则使用config默认）
    # 训练参数
    stage2_epochs: int = 30
    lr: float = 1e-4
    # AHRQ模型
    ahrq_model_path: str = "./results/ahrq_ablation/models/ahrq_full_model.pth"


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
    """运行单个参数配置实验"""
    print(f"\n{'='*60}")
    print(f"Running: {hp_config.experiment_name}")
    print(f"  fusion_type: {hp_config.fusion_type}, alpha: {hp_config.alpha}")
    print(f"  stage2_epochs: {hp_config.stage2_epochs}, lr: {hp_config.lr}")
    print(f"{'='*60}")

    seed_everything(new_config.seed)

    # 输出目录
    OUTPUT_DIR = f"./results/sasrec_ahrq_hp_search/{hp_config.experiment_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载预训练的AHRQ模型
    if not os.path.exists(hp_config.ahrq_model_path):
        raise FileNotFoundError(f"AHRQ model not found at {hp_config.ahrq_model_path}")

    print(f"Loading AHRQ model from {hp_config.ahrq_model_path}...")
    checkpoint = torch.load(hp_config.ahrq_model_path, map_location=device, weights_only=False)
    saved_config = checkpoint['config']
    semantic_hierarchy = saved_config['semantic_hierarchy']

    # 如果指定了hidden_dim，使用新的配置
    if hp_config.hidden_dim is not None:
        new_hidden_dim = hp_config.hidden_dim
    else:
        new_hidden_dim = new_config.ahrq_hidden_dim

    # 重新计算semantic_hierarchy的层数配置
    total_layers = sum(len(config["layers"]) for config in semantic_hierarchy.values())
    layer_dim = new_hidden_dim // total_layers
    if new_hidden_dim % total_layers != 0:
        print(f"Warning: hidden_dim {new_hidden_dim} not divisible by {total_layers} layers")

    ahrq = AdaptiveHierarchicalQuantizer(
        hidden_dim=new_hidden_dim,
        semantic_hierarchy=semantic_hierarchy,
        use_multimodal=True,
        text_dim=new_config.text_dim,
        visual_dim=new_config.visual_dim,
        beta=new_config.ahrq_beta,
        use_ema=saved_config.get('use_ema', False),
        ema_decay=0.99,
        reset_unused_codes=saved_config.get('use_ema', False),
        reset_threshold=new_config.ahrq_reset_threshold
    ).to(device)
    ahrq.load_state_dict(checkpoint['model_state_dict'])
    print(f"AHRQ model loaded! Best recon loss: {checkpoint.get('metrics', {}).get('val_recon_loss', 0):.6f}")

    # 2. 获取数据
    pretrain_loader, all_item_meta = get_all_item_pretrain_dataloader(
        cache_dir="./data",
        category='Video_Games',
        batch_size=new_config.batch_size,
        shuffle=True,
        quick_mode=True,
        num_workers=NUM_WORKS,
        logger=logger
    )

    # 重新计算indices_list
    print("Recomputing all item semantics...")
    all_item_text = all_item_meta['text_features'].float().to(device)
    all_item_vision = all_item_meta['image_features'].float().to(device)
    _, indices_list, _, _ = ahrq(all_item_text, all_item_vision)

    # 3. 创建SASRecAHRQ模型（带融合）
    num_items = all_item_meta['text_features'].shape[0]
    model = SASRecAHRQ(
        ahrq_model=ahrq,
        num_items=num_items,
        fusion_type=hp_config.fusion_type,
        fixed_alpha=hp_config.alpha
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

    # 4. 训练配置
    rec_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_rec = torch.optim.AdamW(
        rec_params,
        lr=hp_config.lr,
        weight_decay=new_config.weight_decay
    )
    scheduler_rec = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_rec,
        T_max=hp_config.stage2_epochs
    )

    # 5. 训练循环
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

    # 6. 测试集评估
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

    # 7. 汇总结果
    results = {
        "experiment_name": hp_config.experiment_name,
        "config": hp_config.__dict__,
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
    """保存超参数搜索汇总表格"""
    os.makedirs(output_dir, exist_ok=True)

    summary_data = []
    for result in all_results:
        test_metrics = result['test_metrics']
        stage2_best = result['stage2_best_val']
        config = result['config']

        row = {
            "Experiment": result['experiment_name'],
            "Fusion": config.get('fusion_type', 'none'),
            "Alpha": config.get('alpha', 0.5),
            "Epochs": config.get('stage2_epochs', 30),
            "LR": config.get('lr', 1e-4),
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
    print("\n" + "=" * 80)
    print("HP Search Results Summary")
    print("=" * 80)
    print(f"{'Experiment':<30} {'HR@10':>10} {'NDCG@10':>10} {'MRR':>10}")
    print("-" * 80)
    for result in all_results:
        test = result['test_metrics']
        print(f"{result['experiment_name']:<30} {test['HR@10']:>10.4f} {test['NDCG@10']:>10.4f} {test['MRR']:>10.4f}")
    print("-" * 80)
    print(f"Pure SASRec Baseline:       HR@10=0.1704, NDCG@10=0.0814")
    print("=" * 80)

    return df


def main():
    """运行超参数搜索"""
    logger = Logger("./logs/train_sasrec_ahrq_hp_search.log")
    device = new_config.device

    # 定义搜索配置
    # 第一轮：搜索最佳alpha值（固定其他参数）
    search_configs = []

    # Alpha搜索
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        config = HPSearchConfig(
            experiment_name=f"alpha_{alpha}",
            fusion_type="add",
            alpha=alpha,
            stage2_epochs=30,
            lr=1e-4
        )
        search_configs.append(config)

    # 融合方式对比
    for fusion_type in ["concat", "none"]:
        config = HPSearchConfig(
            experiment_name=f"fusion_{fusion_type}",
            fusion_type=fusion_type,
            alpha=0.5,
            stage2_epochs=30,
            lr=1e-4
        )
        search_configs.append(config)

    # 训练轮数对比
    for epochs in [20, 50]:
        config = HPSearchConfig(
            experiment_name=f"epochs_{epochs}",
            fusion_type="add",
            alpha=0.5,
            stage2_epochs=epochs,
            lr=1e-4
        )
        search_configs.append(config)

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
        print("All HP search experiments completed!")
        print("=" * 60)


if __name__ == "__main__":
    main()
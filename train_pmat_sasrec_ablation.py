"""
PMAT-SASRec 消融实验脚本

消融实验配置（基于最优超参数 lr=0.001, lambda_dynamic=0.0005, lambda_modal=0.7）：
- PMAT-Full: 完整PMAT-SASRec (基线)
- PMAT-w/o-Dynamic: 去掉动态一致性损失 (lambda_dynamic=0)
- PMAT-w/o-Modal: 去掉模态熵损失 (lambda_modal=0)
- PMAT-w/o-Fusion-Dynamic: 纯动态特征 (fusion_alpha=1.0)
- PMAT-w/o-Fusion-Modal: 纯模态特征 (fusion_alpha=0.0)
- PMAT-w/o-Both: 去掉动态损失和模态损失 (lambda_dynamic=0, lambda_modal=0)

评估指标：HR@5/10/20, NDCG@5/10/20, MRR
结果保存：CSV格式 (不保存模型权重)
"""

import os
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import new_config
from data_utils import get_pmat_dataloader, get_all_item_pretrain_dataloader
from log import Logger
from our_models.ah_rq import AdaptiveHierarchicalQuantizer
from our_models.pmat_sasrec import PMATSASRec
from utils.utils import calculate_metrics, seed_everything

NUM_WORKS = 0

# 消融实验配置
ABLATION_CONFIGS = {
    'full': {
        'name': 'PMAT-Full',
        'description': '完整PMAT-SASRec (基线)',
        'lr': 0.001,
        'lambda_dynamic': 0.0005,
        'lambda_modal': 0.7,
        'fusion_alpha': 0.7,
        'use_dynamic': True
    },
    'w/o_dynamic': {
        'name': 'PMAT-w/o-Dynamic',
        'description': '去掉动态嵌入（使用静态嵌入）',
        'lr': 0.001,
        'lambda_dynamic': 0,
        'lambda_modal': 0.7,
        'fusion_alpha': 0.7,
        'use_dynamic': False
    },
    'w/o_modal': {
        'name': 'PMAT-w/o-Modal',
        'description': '去掉模态熵损失',
        'lr': 0.001,
        'lambda_dynamic': 0.0005,
        'lambda_modal': 0,
        'fusion_alpha': 0.7,
        'use_dynamic': True
    },
    'w/o_fusion_dynamic': {
        'name': 'PMAT-w/o-Fusion-Dynamic',
        'description': '纯动态特征',
        'lr': 0.001,
        'lambda_dynamic': 0.0005,
        'lambda_modal': 0.7,
        'fusion_alpha': 1.0,
        'use_dynamic': True
    },
    'w/o_fusion_modal': {
        'name': 'PMAT-w/o-Fusion-Modal',
        'description': '纯模态特征',
        'lr': 0.001,
        'lambda_dynamic': 0.0005,
        'lambda_modal': 0.7,
        'fusion_alpha': 0.0,
        'use_dynamic': True
    },
    'w/o_both': {
        'name': 'PMAT-w/o-Both',
        'description': '去掉动态损失和模态损失',
        'lr': 0.001,
        'lambda_dynamic': 0,
        'lambda_modal': 0,
        'fusion_alpha': 0.7,
        'use_dynamic': True
    }
}


def evaluate_test_full(model, test_loader, indices_list, all_item_meta, topk_list=[5, 10, 20]):
    """测试集全量排序评估"""
    model.eval()
    metrics_dict = {f'HR@{k}': 0.0 for k in topk_list}
    metrics_dict.update({f'NDCG@{k}': 0.0 for k in topk_list})
    metrics_dict['MRR'] = 0.0
    metrics_dict['Mean_Rank'] = 0.0
    total_users = 0

    with torch.no_grad():
        all_item_text = all_item_meta['text_features'].float().to(new_config.device)
        all_item_vision = all_item_meta['image_features'].float().to(new_config.device)

        for batch in tqdm(test_loader, desc='Evaluating on Test Set'):
            # 为每个batch动态计算物品特征（支持动态嵌入）
            all_item_feat = model.get_all_item_sem_feat(
                indices_list=indices_list,
                batch=batch,
                all_item_text=all_item_text,
                all_item_vision=all_item_vision
            )

            user_emb, _, _ = model(batch)
            target_idx = batch["target_item"].to(new_config.device) - 1

            all_scores = torch.matmul(user_emb, all_item_feat.T)
            rec_metrics = calculate_metrics(all_scores, target_idx, k_list=topk_list)

            # 计算MRR和Mean Rank
            batch_size = all_scores.shape[0]
            _, sorted_indices = torch.sort(all_scores, dim=1, descending=True)
            pos_ranks = []
            for i in range(batch_size):
                rank = torch.where(sorted_indices[i] == target_idx[i])[0].item() + 1
                pos_ranks.append(rank)
            pos_ranks = torch.tensor(pos_ranks, device=new_config.device, dtype=torch.float32)
            mrr = (1 / pos_ranks).mean().item()
            mean_rank = pos_ranks.mean().item()

            for k in topk_list:
                metrics_dict[f'HR@{k}'] += rec_metrics[f'HR@{k}']
                metrics_dict[f'NDCG@{k}'] += rec_metrics[f'NDCG@{k}']
            metrics_dict['MRR'] += mrr
            metrics_dict['Mean_Rank'] += mean_rank
            total_users += 1

    if total_users == 0:
        return {k: 0.0 for k in metrics_dict.keys()}

    for k in topk_list:
        metrics_dict[f'HR@{k}'] /= total_users
        metrics_dict[f'NDCG@{k}'] /= total_users
    metrics_dict['MRR'] /= total_users
    metrics_dict['Mean_Rank'] /= total_users

    return metrics_dict


def run_ablation_experiment(args, config_name, config):
    """运行单次消融实验"""
    exp_name = config['name']
    description = config['description']

    print(f"\n{'='*70}")
    print(f"消融实验: {exp_name}")
    print(f"描述: {description}")
    print(f"参数: lr={config['lr']}, lambda_dynamic={config['lambda_dynamic']}, "
          f"lambda_modal={config['lambda_modal']}, fusion_alpha={config['fusion_alpha']}")
    print(f"{'='*70}")

    lr = config['lr']
    lambda_dynamic = config['lambda_dynamic']
    lambda_modal = config['lambda_modal']
    fusion_alpha = config['fusion_alpha']
    use_dynamic = config.get('use_dynamic', True)

    start_time = time.time()
    logger = Logger(f"./logs/pmat_sasrec_ablation_{config_name}.log")
    seed_everything(new_config.seed)

    # ===== 加载预训练的 AHRQ 模型 =====
    ahrq_model_path = f"./results/ahrq_ablation/models/{args.ahrq_model}_model.pth"
    ahrq_model_path = f"./results/sasrec_ahrq_hp_search/expD_combined/stage1_model.pth"
    if not os.path.exists(ahrq_model_path):
        raise FileNotFoundError(f"AHRQ model not found at {ahrq_model_path}")

    print(f"\nLoading AHRQ model from {ahrq_model_path}")
    checkpoint = torch.load(ahrq_model_path, map_location=new_config.device, weights_only=False)

    # saved_config = checkpoint['config']
    semantic_hierarchy = checkpoint['semantic_hierarchy']

    ahrq = AdaptiveHierarchicalQuantizer(
        hidden_dim=256,
        semantic_hierarchy=semantic_hierarchy,
        use_multimodal=True,
        text_dim=new_config.text_dim,
        visual_dim=new_config.visual_dim,
        beta=new_config.ahrq_beta,
        use_ema=True,
        ema_decay=0.99,
        reset_unused_codes=True,
        reset_threshold=new_config.ahrq_reset_threshold
    ).to(new_config.device)

    ahrq.load_state_dict(checkpoint['model_state_dict'])
    print(f"AHRQ model loaded! Using model: {args.ahrq_model}")

    # ===== 获取数据 =====
    pretrain_loader, all_item_meta = get_all_item_pretrain_dataloader(
        cache_dir="./data",
        category='Video_Games',
        batch_size=new_config.batch_size,
        shuffle=True,
        quick_mode=True,
        num_workers=NUM_WORKS,
        logger=logger
    )

    # 重新计算 indices_list
    print("Recomputing all item semantics...")
    all_item_text = all_item_meta['text_features'].float().to(new_config.device)
    all_item_vision = all_item_meta['image_features'].float().to(new_config.device)
    _, indices_list, raw_feat, _ = ahrq(all_item_text, all_item_vision)

    # 计算 num_layers 和 layer_dim
    num_layers = sum(len(config["layers"]) for config in semantic_hierarchy.values())
    layer_dim = new_config.ahrq_hidden_dim // num_layers

    # ===== 创建 PMAT-SASRec 模型 =====
    model = PMATSASRec(
        num_items=all_item_meta['num_items'],
        semantic_hierarchy=semantic_hierarchy,
        num_layers=num_layers,
        layer_dim=layer_dim,
        fusion_type="add",
        fixed_alpha=fusion_alpha,  # ID嵌入融合权重
        fusion_alpha=fusion_alpha,  # 动态/模态融合权重
        use_dynamic=use_dynamic  # 是否使用动态嵌入
    ).to(new_config.device)

    # ===== 获取数据加载器 =====
    train_loader, val_loader, test_loader, _ = get_pmat_dataloader(
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

    # ===== 训练循环 =====
    print(f"\nTraining PMAT-SASRec (Stage 2)...")
    print(f"Hyperparameters: lr={lr}, lambda_dynamic={lambda_dynamic}, "
          f"lambda_modal={lambda_modal}, fusion_alpha={fusion_alpha}")

    rec_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_rec = torch.optim.AdamW(
        rec_params,
        lr=lr,
        weight_decay=new_config.weight_decay
    )
    scheduler_rec = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_rec,
        T_max=new_config.stage2_epochs
    )

    best_ndcg = 0.0
    for epoch in range(new_config.stage2_epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{new_config.stage2_epochs}")
        train_losses = []

        with torch.no_grad():
            all_item_text = all_item_meta['text_features'].float().to(new_config.device)
            all_item_vision = all_item_meta['image_features'].float().to(new_config.device)

        for batch in train_bar:
            # 为每个batch动态计算物品特征（支持动态嵌入）
            all_item_feat = model.get_all_item_sem_feat(
                indices_list=indices_list,
                batch=batch,
                all_item_text=all_item_text,
                all_item_vision=all_item_vision
            ).detach()  # Detach from computation graph first
            all_item_feat.requires_grad = False

            user_emb, pos_sem_feat, pmat_out = model(batch)
            batch_final_feat = all_item_feat

            # 计算损失
            logits = torch.matmul(user_emb, batch_final_feat.T)
            target_idx = batch["target_item"].to(new_config.device) - 1
            ce_loss = F.cross_entropy(logits, target_idx, ignore_index=-1)

            # 动态一致性损失 - 仅在启用动态嵌入时计算
            if use_dynamic:
                target_dynamic = pmat_out["target_emb"]
                target_static = pmat_out["target_static_emb"]
                pmat_mse_loss = F.mse_loss(target_dynamic, target_static)
            else:
                pmat_mse_loss = 0

            # 模态熵损失
            modal_weights = pmat_out["modal_weights"]
            modal_entropy = -torch.sum(modal_weights * torch.log(modal_weights + 1e-8), dim=-1).mean()
            modal_loss = 1 - modal_entropy

            # 总损失
            loss = ce_loss + lambda_dynamic * pmat_mse_loss + lambda_modal * modal_loss

            optimizer_rec.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rec_params, new_config.grad_clip)
            optimizer_rec.step()

            train_losses.append(loss.item())
            avg_loss = np.mean(train_losses)
            train_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        scheduler_rec.step()

        # ===== 验证阶段 =====
        model.eval()
        val_losses = []
        val_metrics = []
        with torch.no_grad():
            all_item_text = all_item_meta['text_features'].float().to(new_config.device)
            all_item_vision = all_item_meta['image_features'].float().to(new_config.device)

            for batch in val_loader:
                # 为每个batch动态计算物品特征（支持动态嵌入）
                all_item_feat = model.get_all_item_sem_feat(
                    indices_list=indices_list,
                    batch=batch,
                    all_item_text=all_item_text,
                    all_item_vision=all_item_vision
                )

                user_emb, pos_sem_feat, pmat_out = model(batch)
                batch_final_feat = all_item_feat

                logits = torch.matmul(user_emb, batch_final_feat.T)
                target_idx = batch["target_item"].to(new_config.device) - 1
                ce_loss = F.cross_entropy(logits, target_idx, ignore_index=-1)

                # 动态一致性损失 - 仅在启用动态嵌入时计算
                if use_dynamic:
                    target_dynamic = pmat_out["target_emb"]
                    target_static = pmat_out["target_static_emb"]
                    pmat_mse_loss = F.mse_loss(target_dynamic, target_static)
                else:
                    pmat_mse_loss = 0

                modal_weights = pmat_out["modal_weights"]
                modal_entropy = -torch.sum(modal_weights * torch.log(modal_weights + 1e-8), dim=-1).mean()
                modal_loss = 1 - modal_entropy

                loss = ce_loss + lambda_dynamic * pmat_mse_loss + lambda_modal * modal_loss
                val_losses.append(loss.item())

                all_scores = torch.matmul(user_emb, batch_final_feat.T)
                rec_metrics = calculate_metrics(all_scores, target_idx)
                val_metrics.append({**rec_metrics})

        avg_val_loss = np.mean(val_losses)
        avg_val_metrics = {}
        for key in val_metrics[0].keys():
            avg_val_metrics[key] = np.mean([m[key] for m in val_metrics])

        print(f"{exp_name} Epoch {epoch + 1}: Val Loss: {avg_val_loss:.4f} | "
              f"Val HR@10: {avg_val_metrics['HR@10']:.4f} | Val NDCG@10: {avg_val_metrics['NDCG@10']:.4f}")

        if avg_val_metrics["NDCG@10"] > best_ndcg:
            best_ndcg = avg_val_metrics["NDCG@10"]

    # ===== 测试评估 =====
    print("\nEvaluating on test set...")
    test_metrics = evaluate_test_full(
        model=model,
        test_loader=test_loader,
        indices_list=indices_list,
        all_item_meta=all_item_meta,
        topk_list=[5, 10, 20]
    )

    elapsed = time.time() - start_time

    # 打印结果
    print(f"\n{'='*70}")
    print(f"消融实验完成: {exp_name}")
    print(f"  Test HR@5: {test_metrics['HR@5']:.4f} | HR@10: {test_metrics['HR@10']:.4f} | HR@20: {test_metrics['HR@20']:.4f}")
    print(f"  Test NDCG@5: {test_metrics['NDCG@5']:.4f} | NDCG@10: {test_metrics['NDCG@10']:.4f} | NDCG@20: {test_metrics['NDCG@20']:.4f}")
    print(f"  Test MRR: {test_metrics['MRR']:.4f} | Mean Rank: {test_metrics['Mean_Rank']:.2f}")
    print(f"  耗时: {elapsed:.1f}秒")
    print(f"{'='*70}")

    # 返回结果字典
    result = {
        'experiment': exp_name,
        'description': description,
        'lr': lr,
        'lambda_dynamic': lambda_dynamic,
        'lambda_modal': lambda_modal,
        'fusion_alpha': fusion_alpha,
        'HR@5': test_metrics['HR@5'],
        'HR@10': test_metrics['HR@10'],
        'HR@20': test_metrics['HR@20'],
        'NDCG@5': test_metrics['NDCG@5'],
        'NDCG@10': test_metrics['NDCG@10'],
        'NDCG@20': test_metrics['NDCG@20'],
        'MRR': test_metrics['MRR'],
        'Mean_Rank': test_metrics['Mean_Rank'],
        'elapsed_time': elapsed
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='PMAT-SASRec Ablation Experiment')
    parser.add_argument('--ahrq_model', type=str, default='ahrq_full',
                        choices=['baseline_rq', 'ahrq_hiercodebook', 'ahrq_ema', 'ahrq_hscl', 'ahrq_full', 'ahrq_inverted'],
                        help='AHRQ model to load from results/ahrq_ablation')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=list(ABLATION_CONFIGS.keys()),
                        help='Run specific ablation experiment. If not specified, runs all.')
    args = parser.parse_args()

    # 输出目录
    OUTPUT_DIR = "./results/pmat_sasrec_ablation"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    CSV_PATH = os.path.join(OUTPUT_DIR, "pmat_sasrec_ablation_results.csv")

    # 确定要运行的实验
    if args.ablation:
        experiments_to_run = {args.ablation: ABLATION_CONFIGS[args.ablation]}
    else:
        experiments_to_run = ABLATION_CONFIGS

    # 运行实验
    all_results = []
    for config_name, config in experiments_to_run.items():
        result = run_ablation_experiment(args, config_name, config)
        all_results.append(result)

    # 保存结果到CSV
    df = pd.DataFrame(all_results)
    # 检查文件是否存在，决定是否写入header
    file_exists = os.path.exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode='a', header=not file_exists, index=False)
    print(f"\n结果已保存到: {CSV_PATH}")

    # 打印汇总表格
    print("\n" + "="*70)
    print("消融实验结果汇总:")
    print("="*70)
    summary_df = df[['experiment', 'HR@10', 'NDCG@10', 'MRR']].copy()
    summary_df = summary_df.sort_values('NDCG@10', ascending=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
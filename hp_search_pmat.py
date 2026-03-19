"""
PMAT-SASRec 超参数搜索脚本

直接整合训练逻辑，不依赖外部 train_pmat_sasrec.py。

用法：
    # Stage 1: 核心参数搜索 (27组)
    python hp_search_pmat.py --stage 1

    # Stage 2: 次要参数搜索 (基于Stage1最优结果)
    python hp_search_pmat.py --stage 2 --lr 0.001 --lambda_dynamic 0.001 --lambda_modal 0.5

    # 单次运行
    python hp_search_pmat.py --stage 0 --lr 0.001 --lambda_dynamic 0.001 --lambda_modal 0.5 --hidden_dim 64 --dropout 0.35 --fusion_alpha 0.7
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

from config import new_config
from data_utils import get_pmat_dataloader, get_all_item_pretrain_dataloader
from log import Logger
from our_models.ah_rq import AdaptiveHierarchicalQuantizer
from our_models.pmat_sasrec import PMATSASRec
from utils.utils import calculate_metrics, seed_everything

NUM_WORKS = 0


def parse_args():
    parser = argparse.ArgumentParser(description='PMAT-SASRec Hyperparameter Search')

    # 实验阶段
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2, 3],
                        help='实验阶段: 0=单次运行, 1=Stage1, 2=Stage2, 3=Stage3验证')

    # 核心参数（Stage 1）
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--lambda_dynamic', type=float, default=None, help='动态一致性损失权重')
    parser.add_argument('--lambda_modal', type=float, default=None, help='模态熵损失权重')

    # 次要参数（Stage 2）
    parser.add_argument('--hidden_dim', type=int, default=None, help='隐藏维度')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout率')
    parser.add_argument('--fusion_alpha', type=float, default=None, help='动态/模态融合权重')

    # 通用参数
    parser.add_argument('--ahrq_model', type=str, default='ahrq_full', help='AHRQ模型名称')
    parser.add_argument('--output', type=str, default='./results/hp_search', help='结果输出目录')
    parser.add_argument('--dry_run', action='store_true', help='仅打印命令不执行')

    return parser.parse_args()


def evaluate_test_full(model, test_loader, indices_list, all_item_meta, k_list=[5, 10, 20]):
    """测试集全量排序评估（用动态+模态融合特征）"""
    model.eval()
    total_hr = {k: 0.0 for k in k_list}
    total_ndcg = {k: 0.0 for k in k_list}
    total_mrr = 0.0
    total_users = 0

    with torch.no_grad():
        # 预加载全量物品的文本/视觉特征
        all_item_text = all_item_meta['text_features'].float().to(new_config.device)
        all_item_vision = all_item_meta['image_features'].float().to(new_config.device)

        # 获取全量物品的动态+模态融合特征
        all_item_feat = model.get_all_item_sem_feat(
            indices_list=indices_list,
            all_item_text=all_item_text,
            all_item_vision=all_item_vision
        )

        for batch in tqdm(test_loader, desc=f'Full Ranking Evaluation on Test Set'):
            user_emb, _, pmat_out = model(batch)

            # 修正target_idx偏移（1~num_items+1 → 0~num_items）
            target_idx = batch["target_item"].to(new_config.device) - 1

            # 使用预计算好的动态+模态融合特征
            batch_final_feat = all_item_feat

            # 用个性化动态特征计算得分
            all_scores = torch.matmul(user_emb, batch_final_feat.T)
            rec_metrics = calculate_metrics(all_scores, target_idx, k_list=k_list)

            # 累加所有指标
            for k in k_list:
                total_hr[k] += rec_metrics[f'HR@{k}']
                total_ndcg[k] += rec_metrics[f'NDCG@{k}']
            total_mrr += rec_metrics.get('MRR', 0.0)
            total_users += 1

    if total_users == 0:
        return {k: 0.0 for k in k_list}, {k: 0.0 for k in k_list}, 0.0

    avg_hr = {k: total_hr[k] / total_users for k in k_list}
    avg_ndcg = {k: total_ndcg[k] / total_users for k in k_list}
    avg_mrr = total_mrr / total_users
    return avg_hr, avg_ndcg, avg_mrr


def run_single_experiment(args, experiment_name, **params):
    """运行单次实验 - 直接整合训练逻辑"""
    print(f"\n{'='*60}")
    print(f"运行实验: {experiment_name}")
    print(f"参数: {params}")
    print(f"{'='*60}")

    # 从 params 获取超参数，使用默认值
    lr = params.get('lr', new_config.lr)
    lambda_dynamic = params.get('lambda_dynamic', new_config.lambda_dynamic)
    lambda_modal = params.get('lambda_modal', new_config.lambda_modal)
    hidden_dim = params.get('hidden_dim', new_config.sasrec_hidden_dim)
    dropout = params.get('dropout', new_config.sasrec_dropout)
    fusion_alpha = params.get('fusion_alpha', new_config.fusion_alpha)

    if args.dry_run:
        print("[DRY RUN] 仅打印参数，不实际执行")
        return None

    start_time = time.time()
    logger = Logger(f"./logs/hp_search_{experiment_name}.log")
    seed_everything(new_config.seed)

    # 输出目录
    OUTPUT_DIR = f"./results/pmat_sasrec/{experiment_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===== 加载预训练的 AHRQ 模型 =====
    ahrq_model_path = f"./results/ahrq_ablation/models/{args.ahrq_model}_model.pth"
    if not os.path.exists(ahrq_model_path):
        raise FileNotFoundError(f"AHRQ model not found at {ahrq_model_path}")

    print(f"\n========== Loading AHRQ model from {ahrq_model_path} ==========")
    checkpoint = torch.load(ahrq_model_path, map_location=new_config.device, weights_only=False)

    saved_config = checkpoint['config']
    semantic_hierarchy = saved_config['semantic_hierarchy']

    ahrq = AdaptiveHierarchicalQuantizer(
        hidden_dim=new_config.ahrq_hidden_dim,
        semantic_hierarchy=semantic_hierarchy,
        use_multimodal=True,
        text_dim=new_config.text_dim,
        visual_dim=new_config.visual_dim,
        beta=new_config.ahrq_beta,
        use_ema=saved_config.get('use_ema', False),
        ema_decay=0.99,
        reset_unused_codes=saved_config.get('use_ema', False),
        reset_threshold=new_config.ahrq_reset_threshold
    ).to(new_config.device)

    ahrq.load_state_dict(checkpoint['model_state_dict'])
    print(f"AHRQ model loaded! Using model: {args.ahrq_model}")
    best_recon_loss = checkpoint.get('metrics', {}).get('val_recon_loss', float('inf'))

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
    print("\nRecomputing all item semantics...")
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
        fixed_alpha=fusion_alpha,
    ).to(new_config.device)

    # ===== 获取数据加载器 =====
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

    # ===== 训练循环 =====
    print("\n========== Stage 2: Recommendation Training (PMAT-SASRec) ==========")
    print(f"Hyperparameters: lr={lr}, lambda_dynamic={lambda_dynamic}, lambda_modal={lambda_modal}")
    print(f"                 hidden_dim={hidden_dim}, dropout={dropout}, fusion_alpha={fusion_alpha}")

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
        train_metrics = []

        with torch.no_grad():
            # 预计算全量物品的动态+模态融合特征
            all_item_text = all_item_meta['text_features'].float().to(new_config.device)
            all_item_vision = all_item_meta['image_features'].float().to(new_config.device)
            all_item_feat = model.get_all_item_sem_feat(
                indices_list=indices_list,
                all_item_text=all_item_text,
                all_item_vision=all_item_vision
            )
            all_item_feat.requires_grad = False

        for batch in train_bar:
            user_emb, pos_sem_feat, pmat_out = model(batch)

            # 使用预计算的特征
            batch_final_feat = all_item_feat

            # 计算损失
            logits = torch.matmul(user_emb, batch_final_feat.T)
            target_idx = batch["target_item"].to(new_config.device) - 1
            ce_loss = F.cross_entropy(logits, target_idx, ignore_index=-1)

            # 动态一致性损失
            target_dynamic = pmat_out["target_emb"]
            target_static = pmat_out["target_static_emb"]
            pmat_mse_loss = F.mse_loss(target_dynamic, target_static)

            # 模态熵损失
            modal_weights = pmat_out["modal_weights"]
            modal_entropy = -torch.sum(modal_weights * torch.log(modal_weights + 1e-8), dim=-1).mean()
            modal_loss = 1 - modal_entropy

            # 使用传入的超参数
            loss = ce_loss + lambda_dynamic * pmat_mse_loss + lambda_modal * modal_loss

            optimizer_rec.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rec_params, new_config.grad_clip)
            optimizer_rec.step()

            train_losses.append(loss.item())
            with torch.no_grad():
                all_scores = torch.matmul(user_emb, batch_final_feat.T)
                rec_metrics = calculate_metrics(all_scores, target_idx)
                train_metrics.append(rec_metrics)

            hr10 = rec_metrics.get("HR@10", 0.0)
            ndcg10 = rec_metrics.get("NDCG@10", 0.0)
            avg_loss = np.mean(train_losses)
            train_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "HR@10": f"{hr10:.4f}",
                "NDCG@10": f"{ndcg10:.4f}"
            })

        scheduler_rec.step()

        # ===== 验证阶段 =====
        model.eval()
        val_losses = []
        val_metrics = []
        with torch.no_grad():
            all_item_text = all_item_meta['text_features'].float().to(new_config.device)
            all_item_vision = all_item_meta['image_features'].float().to(new_config.device)
            all_item_feat = model.get_all_item_sem_feat(
                indices_list=indices_list,
                all_item_text=all_item_text,
                all_item_vision=all_item_vision
            )

            for batch in val_loader:
                user_emb, pos_sem_feat, pmat_out = model(batch)

                batch_final_feat = all_item_feat

                logits = torch.matmul(user_emb, batch_final_feat.T)
                target_idx = batch["target_item"].to(new_config.device) - 1
                ce_loss = F.cross_entropy(logits, target_idx, ignore_index=-1)

                target_dynamic = pmat_out["target_emb"]
                target_static = pmat_out["target_static_emb"]
                pmat_mse_loss = F.mse_loss(target_dynamic, target_static)

                modal_weights = pmat_out["modal_weights"]
                modal_entropy = -torch.sum(modal_weights * torch.log(modal_weights + 1e-8), dim=-1).mean()
                modal_loss = 1 - modal_entropy

                loss = ce_loss + lambda_dynamic * pmat_mse_loss + lambda_modal * modal_loss
                val_losses.append(loss.item())

                all_scores = torch.matmul(user_emb, batch_final_feat.T)
                rec_metrics = calculate_metrics(all_scores, target_idx)
                val_metrics.append({**rec_metrics})

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        avg_train_metrics = {}
        for key in train_metrics[0].keys():
            avg_train_metrics[key] = np.mean([m[key] for m in train_metrics])

        avg_val_metrics = {}
        for key in val_metrics[0].keys():
            avg_val_metrics[key] = np.mean([m[key] for m in val_metrics])

        print(f"\n{experiment_name} Epoch {epoch + 1}:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train HR@10: {avg_train_metrics['HR@10']:.4f} | Val HR@10: {avg_val_metrics['HR@10']:.4f}")
        print(f"Train NDCG@10: {avg_train_metrics['NDCG@10']:.4f} | Val NDCG@10: {avg_val_metrics['NDCG@10']:.4f}")

        if avg_val_metrics["NDCG@10"] > best_ndcg:
            best_ndcg = avg_val_metrics["NDCG@10"]
            torch.save({
                "stage": 2,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_rec_state_dict": optimizer_rec.state_dict(),
                "best_ndcg": best_ndcg,
                "params": params
            }, f"{OUTPUT_DIR}/best_pmat_sasrec.pth")
            print(f"Best model saved! NDCG@10: {best_ndcg:.4f}")

    # ===== 测试评估 =====
    k_list = [5, 10, 20]
    test_hr_full, test_ndcg_full, test_mrr = evaluate_test_full(
        model=model,
        test_loader=test_loader,
        indices_list=indices_list,
        all_item_meta=all_item_meta,
        k_list=k_list,
    )

    elapsed = time.time() - start_time

    # 保存完整的评估结果
    result_info = {
        'experiment_name': experiment_name,
        'params': params,
        # HR 指标
        'test_hr5': test_hr_full[5],
        'test_hr10': test_hr_full[10],
        'test_hr20': test_hr_full[20],
        # NDCG 指标
        'test_ndcg5': test_ndcg_full[5],
        'test_ndcg10': test_ndcg_full[10],
        'test_ndcg20': test_ndcg_full[20],
        # MRR 指标
        'test_mrr': test_mrr,
        # 验证集指标
        'val_ndcg10': best_ndcg,
        # 训练信息
        'epochs': new_config.stage2_epochs,
        'elapsed_time': elapsed
    }

    print(f"\n实验完成: {experiment_name}")
    print(f"  Test HR@5: {test_hr_full[5]:.4f} | HR@10: {test_hr_full[10]:.4f} | HR@20: {test_hr_full[20]:.4f}")
    print(f"  Test NDCG@5: {test_ndcg_full[5]:.4f} | NDCG@10: {test_ndcg_full[10]:.4f} | NDCG@20: {test_ndcg_full[20]:.4f}")
    print(f"  Test MRR: {test_mrr:.4f}")
    print(f"  耗时: {elapsed:.1f}秒")

    return result_info


def save_results(results, output_path):
    """保存实验结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")


def save_results_csv(results, output_path):
    """保存为CSV格式表格（用于Excel或数据分析）"""
    import pandas as pd

    rows = []
    for r in results:
        row = {
            'experiment_name': r.get('experiment_name', ''),
            # 超参数
            'lr': r['params'].get('lr', ''),
            'lambda_dynamic': r['params'].get('lambda_dynamic', ''),
            'lambda_modal': r['params'].get('lambda_modal', ''),
            'hidden_dim': r['params'].get('hidden_dim', ''),
            'dropout': r['params'].get('dropout', ''),
            'fusion_alpha': r['params'].get('fusion_alpha', ''),
            # HR 指标
            'test_hr5': r.get('test_hr5', ''),
            'test_hr10': r.get('test_hr10', ''),
            'test_hr20': r.get('test_hr20', ''),
            # NDCG 指标
            'test_ndcg5': r.get('test_ndcg5', ''),
            'test_ndcg10': r.get('test_ndcg10', ''),
            'test_ndcg20': r.get('test_ndcg20', ''),
            # MRR 指标
            'test_mrr': r.get('test_mrr', ''),
            # 验证集
            'val_ndcg10': r.get('val_ndcg10', ''),
            # 训练信息
            'epochs': r.get('epochs', ''),
            'elapsed_time': r.get('elapsed_time', '')
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # 按 test_hr10 降序排序
    df = df.sort_values('test_hr10', ascending=False)
    df.to_csv(output_path, index=False)
    print(f"CSV表格已保存到: {output_path}")


def save_results_latex(results, output_path):
    """保存为LaTeX表格格式（用于论文）"""
    # 按 test_hr10 降序排序
    sorted_results = sorted(results, key=lambda x: x.get('test_hr10', 0), reverse=True)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{PMAT-SASRec 超参数搜索结果}")
    lines.append("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|c|}")
    lines.append("\\hline")
    lines.append("实验名 & lr & $\\lambda_{dyn}$ & $\\lambda_{mod}$ & HR@5 & HR@10 & HR@20 & NDCG@5 & NDCG@10 & NDCG@20 & MRR \\\\")
    lines.append("\\hline")

    for r in sorted_results:
        name = r.get('experiment_name', '')
        lr = r['params'].get('lr', '-')
        lambda_dyn = r['params'].get('lambda_dynamic', '-')
        lambda_mod = r['params'].get('lambda_modal', '-')

        # 评估指标
        hr5 = r.get('test_hr5', 0)
        hr10 = r.get('test_hr10', 0)
        hr20 = r.get('test_hr20', 0)
        ndcg5 = r.get('test_ndcg5', 0)
        ndcg10 = r.get('test_ndcg10', 0)
        ndcg20 = r.get('test_ndcg20', 0)
        mrr = r.get('test_mrr', 0)

        # 格式化数值
        lr_str = f"{lr:.0e}" if isinstance(lr, float) and lr < 0.01 else str(lr)
        lines.append(f"{name} & {lr_str} & {lambda_dyn} & {lambda_mod} & {hr5:.4f} & {hr10:.4f} & {hr20:.4f} & {ndcg5:.4f} & {ndcg10:.4f} & {ndcg20:.4f} & {mrr:.4f} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\label{tab:pmat_sasrec_hp_search}")
    lines.append("\\end{table}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX表格已保存到: {output_path}")


def stage1_experiments(args):
    """Stage 1: 核心参数搜索 (27组)"""
    lr_values = [5e-4, 1e-3, 2e-3]
    lambda_dynamic_values = [0.0005, 0.001, 0.005]
    lambda_modal_values = [0.3, 0.5, 0.7]

    results = []
    experiment_id = 1

    for lr in lr_values:
        for lambda_dyn in lambda_dynamic_values:
            for lambda_mod in lambda_modal_values:
                experiment_name = f"S1-{experiment_id:02d}"
                params = {
                    'lr': lr,
                    'lambda_dynamic': lambda_dyn,
                    'lambda_modal': lambda_mod
                }

                result = run_single_experiment(args, experiment_name, **params)
                if result:
                    results.append(result)

                experiment_id += 1

    return results


def stage2_experiments(args):
    """Stage 2: 次要参数搜索"""
    best_lr = args.lr or 0.001
    best_lambda_dynamic = args.lambda_dynamic or 0.001
    best_lambda_modal = args.lambda_modal or 0.5

    hidden_dim_values = [32, 64, 128]
    dropout_values = [0.2, 0.35, 0.5]
    fusion_alpha_values = [0.3, 0.5, 0.7]

    results = []
    experiment_id = 1

    for hidden_dim in hidden_dim_values:
        for dropout in dropout_values:
            for fusion_alpha in fusion_alpha_values:
                if experiment_id > 10:  # 限制在10组以内
                    break

                experiment_name = f"S2-{experiment_id:02d}"
                params = {
                    'lr': best_lr,
                    'lambda_dynamic': best_lambda_dynamic,
                    'lambda_modal': best_lambda_modal,
                    'hidden_dim': hidden_dim,
                    'dropout': dropout,
                    'fusion_alpha': fusion_alpha
                }

                result = run_single_experiment(args, experiment_name, **params)
                if result:
                    results.append(result)

                experiment_id += 1

    return results


def main():
    args = parse_args()

    print(f"PMAT-SASRec 超参数搜索")
    print(f"实验阶段: {args.stage}")
    print(f"参数: lr={args.lr}, lambda_dynamic={args.lambda_dynamic}, lambda_modal={args.lambda_modal}")
    print(f"       hidden_dim={args.hidden_dim}, dropout={args.dropout}, fusion_alpha={args.fusion_alpha}")

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(args.output, f'results_stage{args.stage}_{timestamp}.json')

    results = []

    if args.stage == 0:
        # 单次运行
        params = {}
        if args.lr is not None:
            params['lr'] = args.lr
        if args.lambda_dynamic is not None:
            params['lambda_dynamic'] = args.lambda_dynamic
        if args.lambda_modal is not None:
            params['lambda_modal'] = args.lambda_modal
        if args.hidden_dim is not None:
            params['hidden_dim'] = args.hidden_dim
        if args.dropout is not None:
            params['dropout'] = args.dropout
        if args.fusion_alpha is not None:
            params['fusion_alpha'] = args.fusion_alpha

        if not params:
            # 使用默认参数
            params = {
                'lr': new_config.lr,
                'lambda_dynamic': new_config.lambda_dynamic,
                'lambda_modal': new_config.lambda_modal,
                'hidden_dim': new_config.sasrec_hidden_dim,
                'dropout': new_config.sasrec_dropout,
                'fusion_alpha': new_config.fusion_alpha
            }

        result = run_single_experiment(args, "single_run", **params)
        if result:
            results.append(result)

    elif args.stage == 1:
        # Stage 1: 核心参数搜索
        results = stage1_experiments(args)

    elif args.stage == 2:
        # Stage 2: 次要参数搜索
        if args.lr is None or args.lambda_dynamic is None or args.lambda_modal is None:
            print("错误: Stage 2 需要指定 --lr, --lambda_dynamic, --lambda_modal")
            return
        results = stage2_experiments(args)

    elif args.stage == 3:
        # Stage 3: 最终验证
        params = {}
        if args.lr is not None:
            params['lr'] = args.lr
        if args.lambda_dynamic is not None:
            params['lambda_dynamic'] = args.lambda_dynamic
        if args.lambda_modal is not None:
            params['lambda_modal'] = args.lambda_modal
        if args.hidden_dim is not None:
            params['hidden_dim'] = args.hidden_dim
        if args.dropout is not None:
            params['dropout'] = args.dropout
        if args.fusion_alpha is not None:
            params['fusion_alpha'] = args.fusion_alpha

        result = run_single_experiment(args, "final_validation", **params)
        if result:
            results.append(result)

    # 保存结果
    if results:
        # 保存JSON格式
        save_results(results, results_file)

        # 生成CSV表格
        csv_path = results_file.replace('.json', '_table.csv')
        save_results_csv(results, csv_path)

        # 生成LaTeX表格
        latex_path = results_file.replace('.json', '_table.tex')
        save_results_latex(results, latex_path)

        # 打印最佳结果
        if len(results) > 0:
            best = max(results, key=lambda x: x.get('test_hr10', 0))
            print(f"\n{'='*60}")
            print(f"最佳结果: {best['experiment_name']}")
            print(f"Test HR@10: {best.get('test_hr10')}")
            print(f"Test NDCG@10: {best.get('test_ndcg10')}")
            print(f"参数: {best['params']}")
            print(f"{'='*60}")

    print("\n超参数搜索完成!")


if __name__ == '__main__':
    main()
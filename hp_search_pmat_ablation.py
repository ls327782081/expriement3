"""
PMAT 参数消融实验脚本

搜索以下参数对模型性能的影响：
1. 融合系数 (fusion_alpha): 动态/模态融合权重
2. 兴趣漂移阈值 (pmat_drift_threshold): 用户兴趣变化超过该阈值则判定为兴趣漂移
3. 短期兴趣窗口长度 (sasrec_max_len): 最近L个行为作为短期兴趣

用法：
    # 搜索融合系数
    python hp_search_pmat_ablation.py --param fusion_alpha

    # 搜索兴趣漂移阈值
    python hp_search_pmat_ablation.py --param drift_threshold

    # 搜索短期兴趣窗口长度
    python hp_search_pmat_ablation.py --param short_term_window

    # 搜索所有参数 (全网格搜索)
    python hp_search_pmat_ablation.py --param all

    # 指定基准参数运行单个实验
    python hp_search_pmat_ablation.py --param single --fusion_alpha 0.6 --drift_threshold 0.3 --pmat_short_term_window 10
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
    parser = argparse.ArgumentParser(description='PMAT 参数消融实验')

    # 参数类型选择
    parser.add_argument('--param', type=str, default='all',
                        choices=['fusion_alpha', 'drift_threshold', 'short_term_window', 'all', 'single'],
                        help='搜索参数: fusion_alpha, drift_threshold, short_term_window, all, single')
                        help='要搜索的参数')

    # 融合系数
    parser.add_argument('--fusion_alpha', type=float, default=0.7,
                        help='动态/模态融合权重')

    # 兴趣漂移阈值
    parser.add_argument('--drift_threshold', type=float, default=0.7,
                        help='兴趣漂移阈值')

    # 短期兴趣窗口长度
    parser.add_argument('--pmat_short_term_window', type=int, default=10,
                        help='PMAT内部短期兴趣窗口长度（用于漂移检测）')
    parser.add_argument('--max_history_len', type=int, default=50,
                        help='数据加载的历史序列最大长度')

    # 其他超参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--lambda_dynamic', type=float, default=0.001, help='动态一致性损失权重')
    parser.add_argument('--lambda_modal', type=float, default=0.5, help='模态熵损失权重')

    # AHRQ模型
    parser.add_argument('--ahrq_model', type=str, default='ahrq_full', help='AHRQ模型名称')

    # 输出目录
    parser.add_argument('--output', type=str, default='./results/hp_search_ablation',
                        help='结果输出目录')

    # 基准参数（用于单次运行）
    parser.add_argument('--base_fusion_alpha', type=float, default=0.7, help='基准融合系数')
    parser.add_argument('--base_drift_threshold', type=float, default=0.3, help='基准兴趣漂移阈值')
    parser.add_argument('--base_pmat_short_term_window', type=int, default=10, help='基准短期兴趣窗口长度(PMAT内部)')
    parser.add_argument('--base_max_history_len', type=int, default=50, help='基准历史序列最大长度(数据加载)')

    parser.add_argument('--dry_run', action='store_true', help='仅打印命令不执行')

    return parser.parse_args()


def evaluate_test_full(model, test_loader, indices_list, all_item_meta, k_list=[5, 10, 20]):
    """测试集全量排序评估"""
    model.eval()
    total_hr = {k: 0.0 for k in k_list}
    total_ndcg = {k: 0.0 for k in k_list}
    total_mrr = 0.0
    total_users = 0

    with torch.no_grad():
        all_item_text = all_item_meta['text_features'].float().to(new_config.device)
        all_item_vision = all_item_meta['image_features'].float().to(new_config.device)

        all_item_feat = model.get_all_item_sem_feat(
            indices_list=indices_list,
            all_item_text=all_item_text,
            all_item_vision=all_item_vision
        )

        for batch in tqdm(test_loader, desc=f'Full Ranking Evaluation on Test Set'):
            user_emb, _, pmat_out = model(batch)

            target_idx = batch["target_item"].to(new_config.device) - 1

            batch_final_feat = all_item_feat

            all_scores = torch.matmul(user_emb, batch_final_feat.T)
            rec_metrics = calculate_metrics(all_scores, target_idx, k_list=k_list)

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


def run_single_experiment(args, experiment_name, **override_params):
    """运行单次实验"""
    print(f"\n{'='*60}")
    print(f"运行实验: {experiment_name}")
    print(f"参数: {override_params}")
    print(f"{'='*60}")

    # 合并参数：默认参数 + 覆盖参数
    fusion_alpha = override_params.get('fusion_alpha', args.base_fusion_alpha)
    drift_threshold = override_params.get('drift_threshold', args.base_drift_threshold)
    pmat_short_term_window = override_params.get('pmat_short_term_window', args.base_pmat_short_term_window)
    max_history_len = override_params.get('max_history_len', args.base_max_history_len)

    lr = override_params.get('lr', args.lr)
    lambda_dynamic = override_params.get('lambda_dynamic', args.lambda_dynamic)
    lambda_modal = override_params.get('lambda_modal', args.lambda_modal)

    if args.dry_run:
        print("[DRY RUN] 仅打印参数，不实际执行")
        return None

    start_time = time.time()

    # 临时修改config中的参数
    original_fusion_alpha = new_config.fusion_alpha
    original_drift_threshold = new_config.pmat_drift_threshold
    original_pmat_short_term_window = new_config.pmat_short_term_window

    new_config.fusion_alpha = fusion_alpha
    new_config.pmat_drift_threshold = drift_threshold
    new_config.pmat_short_term_window = pmat_short_term_window

    logger = Logger(f"./logs/hp_search_ablation_{experiment_name}.log")
    seed_everything(new_config.seed)

    OUTPUT_DIR = f"./results/hp_search_ablation/{experiment_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载预训练的 AHRQ 模型
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

    print("\nRecomputing all item semantics...")
    all_item_text = all_item_meta['text_features'].float().to(new_config.device)
    all_item_vision = all_item_meta['image_features'].float().to(new_config.device)
    _, indices_list, raw_feat, _ = ahrq(all_item_text, all_item_vision)

    num_layers = sum(len(config["layers"]) for config in semantic_hierarchy.values())
    layer_dim = new_config.ahrq_hidden_dim // num_layers

    # 创建 PMAT-SASRec 模型
    model = PMATSASRec(
        num_items=all_item_meta['num_items'],
        semantic_hierarchy=semantic_hierarchy,
        num_layers=num_layers,
        layer_dim=layer_dim,
        fusion_type="add",
        fixed_alpha=fusion_alpha,
    ).to(new_config.device)

    # 获取数据加载器
    train_loader, val_loader, test_loader, all_item_features = get_pmat_dataloader(
        cache_dir="./data",
        category='Video_Games',
        batch_size=new_config.batch_size,
        max_history_len=max_history_len,  # 历史序列最大长度
        num_negative_samples=new_config.num_negative_samples,
        shuffle=True,
        quick_mode=True,
        num_workers=NUM_WORKS,
        indices_list=indices_list,
        logger=logger
    )

    # 训练循环
    print("\n========== PMAT-SASRec Training ==========")
    print(f"Hyperparameters: fusion_alpha={fusion_alpha}, drift_threshold={drift_threshold}, pmat_short_term_window={pmat_short_term_window}, max_history_len={max_history_len}")
    print(f"                 lr={lr}, lambda_dynamic={lambda_dynamic}, lambda_modal={lambda_modal}")

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

        # 验证阶段
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
                "params": override_params
            }, f"{OUTPUT_DIR}/best_pmat_sasrec.pth")
            print(f"Best model saved! NDCG@10: {best_ndcg:.4f}")

    # 测试评估
    k_list = [5, 10, 20]
    test_hr_full, test_ndcg_full, test_mrr = evaluate_test_full(
        model=model,
        test_loader=test_loader,
        indices_list=indices_list,
        all_item_meta=all_item_meta,
        k_list=k_list,
    )

    elapsed = time.time() - start_time

    # 恢复原始config
    new_config.fusion_alpha = original_fusion_alpha
    new_config.pmat_drift_threshold = original_drift_threshold
    new_config.pmat_short_term_window = original_pmat_short_term_window

    # 保存完整的评估结果
    result_info = {
        'experiment_name': experiment_name,
        'params': override_params,
        # 搜索的关键参数
        'fusion_alpha': fusion_alpha,
        'drift_threshold': drift_threshold,
        'pmat_short_term_window': pmat_short_term_window,
        'max_history_len': max_history_len,
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
    """保存实验结果为JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")


def save_results_csv(results, output_path):
    """保存为CSV格式表格"""
    import pandas as pd

    rows = []
    for r in results:
        row = {
            'experiment_name': r.get('experiment_name', ''),
            # 搜索的关键参数
            'fusion_alpha': r.get('fusion_alpha', ''),
            'drift_threshold': r.get('drift_threshold', ''),
            'pmat_short_term_window': r.get('pmat_short_term_window', ''),
            'max_history_len': r.get('max_history_len', ''),
            # 其他参数
            'lr': r['params'].get('lr', ''),
            'lambda_dynamic': r['params'].get('lambda_dynamic', ''),
            'lambda_modal': r['params'].get('lambda_modal', ''),
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
    df = df.sort_values('test_hr10', ascending=False)
    df.to_csv(output_path, index=False)
    print(f"CSV表格已保存到: {output_path}")


def save_results_latex(results, output_path, param_name):
    """保存为LaTeX表格格式（用于论文）"""
    sorted_results = sorted(results, key=lambda x: x.get('test_hr10', 0), reverse=True)

    if param_name == 'fusion_alpha':
        caption = "融合系数 $\\alpha$ 对模型性能的影响"
        col_format = "|l|c|c|c|c|c|c|"
        header = "$\\alpha$ & HR@5 & HR@10 & HR@20 & NDCG@5 & NDCG@10 & NDCG@20 \\\\"
    elif param_name == 'drift_threshold':
        caption = "兴趣漂移阈值 $\\gamma$ 对模型性能的影响"
        col_format = "|l|c|c|c|c|c|c|"
        header = "$\\gamma$ & HR@5 & HR@10 & HR@20 & NDCG@5 & NDCG@10 & NDCG@20 \\\\"
    elif param_name == 'short_term_window':
        caption = "短期兴趣窗口长度 $L$ 对模型性能的影响"
        col_format = "|l|c|c|c|c|c|c|"
        header = "$L$ & HR@5 & HR@10 & HR@20 & NDCG@5 & NDCG@10 & NDCG@20 \\\\"
    else:
        caption = "参数搜索结果"
        col_format = "|l|c|c|c|c|c|c|c|c|c|"
        header = "Exp & $\\alpha$ & $\\gamma$ & $L$ & HR@5 & HR@10 & NDCG@5 & NDCG@10 & MRR \\\\"

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\begin{{{col_format}}}")
    lines.append("\\hline")
    lines.append(header)
    lines.append("\\hline")

    for r in sorted_results:
        if param_name == 'all':
            name = r.get('experiment_name', '')
            alpha = r.get('fusion_alpha', '-')
            gamma = r.get('drift_threshold', '-')
            L = r.get('pmat_short_term_window', '-')
            hr5 = r.get('test_hr5', 0)
            hr10 = r.get('test_hr10', 0)
            ndcg5 = r.get('test_ndcg5', 0)
            ndcg10 = r.get('test_ndcg10', 0)
            mrr = r.get('test_mrr', 0)
            lines.append(f"{name} & {alpha} & {gamma} & {L} & {hr5:.4f} & {hr10:.4f} & {ndcg5:.4f} & {ndcg10:.4f} & {mrr:.4f} \\\\")
        else:
            if param_name == 'fusion_alpha':
                value = r.get('fusion_alpha', 0)
            elif param_name == 'drift_threshold':
                value = r.get('drift_threshold', 0)
            else:
                value = r.get('pmat_short_term_window', 0)

            hr5 = r.get('test_hr5', 0)
            hr10 = r.get('test_hr10', 0)
            hr20 = r.get('test_hr20', 0)
            ndcg5 = r.get('test_ndcg5', 0)
            ndcg10 = r.get('test_ndcg10', 0)
            ndcg20 = r.get('test_ndcg20', 0)
            lines.append(f"{value} & {hr5:.4f} & {hr10:.4f} & {hr20:.4f} & {ndcg5:.4f} & {ndcg10:.4f} & {ndcg20:.4f} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\label{{tab:pmat_{param_name}_ablation}}")
    lines.append("\\end{table}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX表格已保存到: {output_path}")


def search_fusion_alpha(args):
    """搜索融合系数 fusion_alpha"""
    print("\n" + "="*60)
    print("搜索融合系数 fusion_alpha")
    print("="*60)

    fusion_alpha_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    for i, fusion_alpha in enumerate(fusion_alpha_values):
        experiment_name = f"fusion_alpha_{fusion_alpha}"
        params = {
            'fusion_alpha': fusion_alpha,
            'drift_threshold': args.base_drift_threshold,
            'pmat_short_term_window': args.base_pmat_short_term_window,
            'max_history_len': args.base_max_history_len,
            'lr': args.lr,
            'lambda_dynamic': args.lambda_dynamic,
            'lambda_modal': args.lambda_modal
        }

        result = run_single_experiment(args, experiment_name, **params)
        if result:
            results.append(result)

    return results, 'fusion_alpha'


def search_drift_threshold(args):
    """搜索兴趣漂移阈值 drift_threshold"""
    print("\n" + "="*60)
    print("搜索兴趣漂移阈值 drift_threshold")
    print("="*60)

    drift_threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for i, drift_threshold in enumerate(drift_threshold_values):
        experiment_name = f"drift_threshold_{drift_threshold}"
        params = {
            'fusion_alpha': args.base_fusion_alpha,
            'drift_threshold': drift_threshold,
            'pmat_short_term_window': args.base_pmat_short_term_window,
            'max_history_len': args.base_max_history_len,
            'lr': args.lr,
            'lambda_dynamic': args.lambda_dynamic,
            'lambda_modal': args.lambda_modal
        }

        result = run_single_experiment(args, experiment_name, **params)
        if result:
            results.append(result)

    return results, 'drift_threshold'


def search_short_term_window(args):
    """搜索短期兴趣窗口长度 pmat_short_term_window"""
    print("\n" + "="*60)
    print("搜索短期兴趣窗口长度 pmat_short_term_window")
    print("="*60)

    pmat_short_term_window_values = [5, 10, 20, 30, 50]
    results = []

    for i, pmat_short_term_window in enumerate(pmat_short_term_window_values):
        experiment_name = f"pmat_short_term_window_{pmat_short_term_window}"
        params = {
            'fusion_alpha': args.base_fusion_alpha,
            'drift_threshold': args.base_drift_threshold,
            'pmat_short_term_window': pmat_short_term_window,
            'max_history_len': args.base_max_history_len,
            'lr': args.lr,
            'lambda_dynamic': args.lambda_dynamic,
            'lambda_modal': args.lambda_modal
        }

        result = run_single_experiment(args, experiment_name, **params)
        if result:
            results.append(result)

    return results, 'short_term_window'


def search_all(args):
    """全网格搜索所有参数"""
    print("\n" + "="*60)
    print("全网格搜索 (所有参数组合)")
    print("="*60)

    fusion_alpha_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    drift_threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    pmat_short_term_window_values = [5, 10, 20, 30, 50]

    results = []
    experiment_id = 1

    total_experiments = len(fusion_alpha_values) * len(drift_threshold_values) * len(pmat_short_term_window_values)
    print(f"总共需要运行 {total_experiments} 组实验")

    for fusion_alpha in fusion_alpha_values:
        for drift_threshold in drift_threshold_values:
            for pmat_short_term_window in pmat_short_term_window_values:
                experiment_name = f"exp_{experiment_id:03d}"
                params = {
                    'fusion_alpha': fusion_alpha,
                    'drift_threshold': drift_threshold,
                    'pmat_short_term_window': pmat_short_term_window,
                    'max_history_len': args.base_max_history_len,
                    'lr': args.lr,
                    'lambda_dynamic': args.lambda_dynamic,
                    'lambda_modal': args.lambda_modal
                }

                print(f"\n[{experiment_id}/{total_experiments}] 运行实验: {experiment_name}")
                result = run_single_experiment(args, experiment_name, **params)
                if result:
                    results.append(result)

                experiment_id += 1

    return results, 'all'


def main():
    args = parse_args()

    print(f"PMAT 参数消融实验")
    print(f"搜索参数: {args.param}")
    print(f"基准参数: fusion_alpha={args.base_fusion_alpha}, drift_threshold={args.base_drift_threshold}, pmat_short_term_window={args.base_pmat_short_term_window}, max_history_len={args.base_max_history_len}")

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results = []
    param_name = 'single'

    if args.param == 'fusion_alpha':
        results, param_name = search_fusion_alpha(args)
    elif args.param == 'drift_threshold':
        results, param_name = search_drift_threshold(args)
    elif args.param == 'short_term_window':
        results, param_name = search_short_term_window(args)
    elif args.param == 'all':
        results, param_name = search_all(args)
    elif args.param == 'single':
        # 单次运行
        experiment_name = f"single_run"
        params = {
            'fusion_alpha': args.fusion_alpha,
            'drift_threshold': args.drift_threshold,
            'pmat_short_term_window': args.pmat_short_term_window,
            'max_history_len': args.max_history_len,
            'lr': args.lr,
            'lambda_dynamic': args.lambda_dynamic,
            'lambda_modal': args.lambda_modal
        }
        result = run_single_experiment(args, experiment_name, **params)
        if result:
            results.append(result)

    # 保存结果
    if results:
        results_file = os.path.join(args.output, f'results_{param_name}_{timestamp}.json')
        save_results(results, results_file)

        csv_path = results_file.replace('.json', '_table.csv')
        save_results_csv(results, csv_path)

        latex_path = results_file.replace('.json', '_table.tex')
        save_results_latex(results, latex_path, param_name)

        # 打印最佳结果
        best = max(results, key=lambda x: x.get('test_hr10', 0))
        print(f"\n{'='*60}")
        print(f"最佳结果: {best['experiment_name']}")
        print(f"Test HR@10: {best.get('test_hr10')}")
        print(f"Test NDCG@10: {best.get('test_ndcg10')}")
        print(f"参数: fusion_alpha={best.get('fusion_alpha')}, drift_threshold={best.get('drift_threshold')}, pmat_short_term_window={best.get('pmat_short_term_window')}")
        print(f"{'='*60}")

    print("\n参数消融实验完成!")


if __name__ == '__main__':
    main()
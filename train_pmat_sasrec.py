import os
import argparse

import numpy as np
import torch
from tqdm import tqdm

from config import new_config


# 命令行参数解析
parser = argparse.ArgumentParser(description='Train PMAT-SASRec with AHRQ')
parser.add_argument('--ahrq_model', type=str, default='ahrq_full',
                    choices=['baseline_rq', 'ahrq_hiercodebook', 'ahrq_ema', 'ahrq_hscl', 'ahrq_full', 'ahrq_inverted'],
                    help='AHRQ model to load from results/ahrq_ablation')
args = parser.parse_args()

from data_utils import get_pmat_dataloader, get_all_item_pretrain_dataloader
from log import Logger
from our_models.ah_rq import AdaptiveHierarchicalQuantizer
from our_models.pmat_sasrec import PMATSASRec
from utils.utils import calculate_metrics, seed_everything
import torch.nn.functional as F

NUM_WORKS = 0


def evaluate_test_full(model, test_loader, indices_list, topk=10):
    """测试集全量排序评估"""
    model.eval()
    total_hr = 0.0
    total_ndcg = 0.0
    total_users = 0

    with torch.no_grad():
        all_item_feat = model.get_all_item_sem_feat(indices_list)
        for batch in tqdm(test_loader, desc=f'Full Ranking Evaluation on Test Set'):
            user_emb, _, _ = model(batch)

            # 修正target_idx偏移（1~num_items+1 → 0~num_items）
            target_idx = batch["target_item"].to(new_config.device) - 1  # 关键：减1偏移

            all_scores = torch.matmul(user_emb, all_item_feat.T)
            rec_metrics = calculate_metrics(all_scores, target_idx)

            total_hr += rec_metrics[f'HR@{topk}']
            total_ndcg += rec_metrics[f'NDCG@{topk}']
            total_users += 1

    if total_users == 0:
        return 0.0, 0.0
    avg_hr = total_hr / total_users
    avg_ndcg = total_ndcg / total_users
    return avg_hr, avg_ndcg


def train_pmat_sasrec():
    logger = Logger("./logs/train_pmat_sasrec.log")
    seed_everything(new_config.seed)

    # 输出目录
    OUTPUT_DIR = "./results/pmat_sasrec"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 从 results/ahrq_ablation/models/ 加载预训练的AHRQ模型
    ahrq_model_path = f"./results/ahrq_ablation/models/{args.ahrq_model}_model.pth"
    if os.path.exists(ahrq_model_path):
        print(f"\n========== Loading AHRQ model from {ahrq_model_path} ==========")
        checkpoint = torch.load(ahrq_model_path, map_location=new_config.device, weights_only=False)

        # 从保存的配置重建模型结构
        saved_config = checkpoint['config']

        # 重建语义层次配置
        semantic_hierarchy = saved_config['semantic_hierarchy']

        # 重新创建AHRQ模型，使用保存的配置
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

        # 加载模型权重
        ahrq.load_state_dict(checkpoint['model_state_dict'])
        print(f"AHRQ model loaded! Using model: {args.ahrq_model}")
        best_recon_loss = checkpoint.get('metrics', {}).get('val_recon_loss', float('inf'))
    else:
        raise FileNotFoundError(f"AHRQ model not found at {ahrq_model_path}. Please run train_ahrq_ablation.py first.")

    # 获取数据加载器和物品元数据
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
    print("\nRecomputing all item semantics after Stage 1...")
    all_item_text = all_item_meta['text_features'].float().to(new_config.device)
    all_item_vision = all_item_meta['image_features'].float().to(new_config.device)
    _, indices_list, raw_feat, _ = ahrq(all_item_text, all_item_vision)

    # 构建dynamic_params以匹配hp_search最佳实验
    dynamic_params = {
        "num_heads": new_config.sasrec_num_heads,  # 1
        "sasrec_num_layers": new_config.sasrec_num_layers,  # 2
        "dropout": new_config.sasrec_dropout,  # 0.4
        "dim_feedforward": new_config.sasrec_hidden_dim * 4,  # 256
    }
    # 使用raw_fusion (最佳实验配置)
    raw_feat_tensor = raw_feat.detach().clone()

    # 从semantic_hierarchy计算num_layers和layer_dim
    num_layers = sum(len(config["layers"]) for config in semantic_hierarchy.values())
    layer_dim = new_config.ahrq_hidden_dim // num_layers

    # 创建PMAT-SASRec模型
    model = PMATSASRec(
        num_items=all_item_meta['num_items'],
        semantic_hierarchy=semantic_hierarchy,
        num_layers=num_layers,
        layer_dim=layer_dim,
        fusion_type="add",
        fixed_alpha=0.5,
    ).to(new_config.device)

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

    pmat_sasrec_model_path = f"{OUTPUT_DIR}/final_pmat_sasrec.pth"
    if os.path.exists(pmat_sasrec_model_path):
        print(f"\n========== Loading existing PMAT-SASRec model from {pmat_sasrec_model_path} ==========")
        checkpoint = torch.load(pmat_sasrec_model_path, map_location=new_config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"PMAT-SASRec model loaded! Skipping training.")
    else:
        print("\n========== Stage 2: Recommendation Training (PMAT-SASRec) ==========")

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

        best_ndcg = 0.0
        for epoch in range(new_config.stage2_epochs):
            model.train()
            train_bar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch + 1}/{new_config.stage2_epochs}")
            train_losses = []
            train_metrics = []

            with torch.no_grad():
                all_item_feat = model.get_all_item_sem_feat(indices_list)
                all_item_feat.requires_grad = False  # 标记无梯度，节省显存

            for batch in train_bar:
                user_emb, pos_sem_feat, pmat_out = model(batch)

                # 2) 推荐 CE 损失
                logits = torch.matmul(user_emb, all_item_feat.T)
                target_idx = batch["target_item"].to(new_config.device) - 1
                ce_loss = F.cross_entropy(logits, target_idx, ignore_index=-1)

                # 3) PMAT 动态嵌入一致性损失
                target_dynamic = pmat_out["target_emb"]
                target_static = pmat_out["target_static_emb"]
                pmat_mse_loss = F.mse_loss(target_dynamic, target_static)

                # 4) 模态注意力约束
                modal_weights = pmat_out["modal_weights"]
                modal_loss = (modal_weights ** 2).mean()

                loss = ce_loss + 0.01 * pmat_mse_loss + 0.001 * modal_loss

                optimizer_rec.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rec_params, new_config.grad_clip)
                optimizer_rec.step()

                train_losses.append(loss.item())
                with torch.no_grad():
                    # 计算全量物品打分（复用预计算的all_item_feat）
                    all_scores = torch.matmul(user_emb, all_item_feat.T)
                    rec_metrics = calculate_metrics(all_scores, target_idx)
                    batch_metrics = {**rec_metrics}
                    train_metrics.append(batch_metrics)

                hr10 = batch_metrics.get("HR@10", 0.0)
                ndgc10 = batch_metrics.get("NDCG@10", 0.0)
                avg_loss = np.mean(train_losses)
                train_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "HR@10": f"{hr10:.4f}",
                    "NDCG@10": f"{ndgc10:.4f}"
                })

            scheduler_rec.step()

            model.eval()
            val_losses = []
            val_metrics = []
            with torch.no_grad():
                for batch in val_loader:
                    user_emb, pos_sem_feat, pmat_out = model(batch)

                    # 2) 推荐 CE 损失
                    logits = torch.matmul(user_emb, all_item_feat.T)
                    target_idx = batch["target_item"].to(new_config.device) - 1
                    ce_loss = F.cross_entropy(logits, target_idx, ignore_index=-1)

                    # 3) PMAT 动态嵌入一致性损失
                    target_dynamic = pmat_out["target_emb"]
                    target_static = pmat_out["target_static_emb"]
                    pmat_mse_loss = F.mse_loss(target_dynamic, target_static)

                    # 4) 模态注意力约束
                    modal_weights = pmat_out["modal_weights"]
                    modal_loss = (modal_weights ** 2).mean()

                    # ========================
                    # 最终总损失（完整版）
                    # ========================
                    loss = ce_loss + 0.01 * pmat_mse_loss + 0.001 * modal_loss
                    val_losses.append(loss.item())
                    all_scores = torch.matmul(user_emb, all_item_feat.T)
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

            print(f"\nStage2 Epoch {epoch + 1} Summary:")
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
                    "best_ndcg": best_ndcg
                }, f"{OUTPUT_DIR}/best_pmat_sasrec.pth")
                print(f"Stage2 Best model saved! NDCG@10: {best_ndcg:.4f}")

        torch.save({"model_state_dict": model.state_dict()}, f"{OUTPUT_DIR}/final_pmat_sasrec.pth")

    test_hr_full, test_ndcg_full = evaluate_test_full(
        model=model,
        test_loader=test_loader,
        indices_list=indices_list,
        topk=10,
    )

    print("\n========== Final Test Result (Full Ranking) ==========")
    print(f"Test HR@10 (Full): {test_hr_full:.4f}")
    print(f"Test NDCG@10 (Full): {test_ndcg_full:.4f}")

    print("\n========== Two-Stage Training Completed! ==========")
    print(f"Best Quant Recon Loss: {best_recon_loss:.6f} | Best Recommendation NDCG (Val): {best_ndcg:.4f}")
    print(f"Final Test HR@10 (Full): {test_hr_full:.4f} | Final Test NDCG@10 (Full): {test_ndcg_full:.4f}")


if __name__ == "__main__":
    train_pmat_sasrec()
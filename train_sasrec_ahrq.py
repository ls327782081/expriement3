import os

import numpy as np
import torch
from tqdm import tqdm

from config import new_config
from data_utils import get_pmat_dataloader, get_all_item_pretrain_dataloader
from log import Logger
from our_models.sasrec_ahrq import SASRecAHRQ
from utils.loss import compute_quantization_loss, compute_ranking_loss
from utils.utils import calculate_metrics, calculate_id_metrics, seed_everything, EarlyStopping, fast_codebook_reset
import torch.nn.functional as F


NUM_WORKS = 0 if os.name == 'nt' else 4


def evaluate_test_full(model, test_loader, all_item_features, topk=10, device="cuda"):
    """
    测试集全量排序评估（适配批量batch_size>1，特征索引=物品ID）
    Args:
        model: 训练好的SASRecAHRQ模型
        test_loader: 测试集dataloader（任意batch_size）
        all_item_features: dict，包含所有物品的文本/视觉特征
            - text: (num_items, text_dim) 文本特征，索引=物品ID
            - visual: (num_items, visual_dim) 视觉特征，索引=物品ID
            - num_items: 物品总数
        topk: 评估HR@10/NDCG@10
        device: 设备
    Returns:
        avg_hr: 全量评估HR@10
        avg_ndcg: 全量评估NDCG@10
    """
    model.eval()
    total_hr = 0.0
    total_ndcg = 0.0
    total_users = 0

    # 提取全量物品特征并移到设备（特征索引=物品ID）
    all_item_text = all_item_features["text"].float().to(device)  # (num_items, text_dim)
    all_item_vision = all_item_features["visual"].float().to(device)  # (num_items, visual_dim)
    num_items = all_item_features["num_items"]  # 物品总数

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Full Ranking Evaluation on Test Set"):
            # 1. 提取批量用户的历史特征
            batch_size = batch["history_text_feat"].shape[0]
            history_text = batch["history_text_feat"].float().to(device)  # (batch_size, max_len, text_dim)
            history_vision = batch["history_vision_feat"].float().to(device)  # (batch_size, max_len, visual_dim)

            # 2. 提取批量用户的正样本ID和已交互物品ID
            target_item_ids = batch["target_item"].cpu().numpy()  # (batch_size,) 批量正样本ID
            history_item_ids = batch["history_items"].cpu().numpy()  # (batch_size, max_len) 批量历史ID

            # 3. 批量全量打分：每个用户对所有物品的推荐分数
            history_items = batch["history_items"].to(device)  # (batch_size, max_len) 用于确定有效序列长度
            all_scores = model.predict_all(
                history_text=history_text,
                history_vision=history_vision,
                all_item_text=all_item_text,
                all_item_vision=all_item_vision,
                history_items=history_items
            )  # (batch_size, num_items)

            # 4. 逐用户处理：排除已交互物品 + 计算指标
            for i in range(batch_size):
                # 4.1 单个用户的已交互物品ID（排除padding/无效ID）
                interacted_ids = [id for id in history_item_ids[i] if id != 0 and id < num_items]
                # 4.2 排除已交互物品
                if len(interacted_ids) > 0:
                    all_scores[i, interacted_ids] = -float('inf')

                # 4.3 单个用户的全量排序，取Top-K
                user_scores = all_scores[i:i + 1, :]  # (1, num_items)
                _, topk_indices = torch.topk(user_scores, k=topk, dim=1)  # (1, topk)
                topk_ids = topk_indices.squeeze(0).cpu().numpy()  # (topk,)

                # 4.4 单个用户的正样本ID
                target_id = target_item_ids[i]
                # 跳过无效正样本ID
                if target_id >= num_items or target_id == 0:
                    continue

                # 4.5 计算单个用户的HR@10/NDCG@10
                if target_id in topk_ids:
                    hr = 1.0
                    rank = np.where(topk_ids == target_id)[0][0] + 1  # 排名从1开始
                    ndcg = 1.0 / np.log2(rank + 1)
                else:
                    hr = 0.0
                    ndcg = 0.0

                total_hr += hr
                total_ndcg += ndcg
                total_users += 1

    # 计算平均指标（避免除以0）
    if total_users == 0:
        return 0.0, 0.0
    avg_hr = total_hr / total_users
    avg_ndcg = total_ndcg / total_users
    return avg_hr, avg_ndcg

# ===================== 训练逻辑 =====================
def train_sasrec_ahrq():
    logger = Logger("./logs/train_sasrec_ahrq.log")
    # 固定种子
    seed_everything(new_config.seed)

    # 2. 初始化模型
    model = SASRecAHRQ().to(new_config.device)

    # ===================== Stage 1：量化预训练（仅训AHRQ，无排序损失） =====================
    print("\n========== Stage 1: Quantization Pre-training (AHRQ Only) ==========")
    # 应用Stage1冻结逻辑（模型内置方法）
    model.freeze_for_stage1()
    # Stage1优化器：仅AH-RQ可训练参数
    quant_params = [p for p in model.ahrq.parameters() if p.requires_grad]
    optimizer_quant = torch.optim.AdamW(
        quant_params,
        lr=1e-4,  # 比之前的5e-5略大，全量数据收敛更快
        weight_decay=1e-5,  # 降低权重衰减，避免过度正则
        betas=(0.9, 0.999)
    )
    # 学习率调度：更平缓的衰减，适配全量数据
    scheduler_quant = torch.optim.lr_scheduler.StepLR(
        optimizer_quant,
        step_size=10,
        gamma=0.9
    )

    # 初始化Stage1早停（Gini系数越小越好，mode='min'）
    early_stopping_quant = EarlyStopping(
        patience=getattr(new_config, "stage1_patience", 5),
        verbose=True,
        delta=1e-6,  # 重构损失的最小提升阈值
        path="./best_quant_ahrq.pth",
        mode='min'  # 关键：适配Gini系数
    )

    pretrain_loader, all_item_meta = get_all_item_pretrain_dataloader(
        cache_dir="./data",
        category='Video_Games',
        batch_size=new_config.batch_size,
        shuffle=False,
        quick_mode=True,
        num_workers=NUM_WORKS,
        logger=logger
    )

    best_recon_loss = float('inf')  # 重构损失

    for epoch in range(new_config.stage1_epochs):
        model.train()
        train_bar = tqdm(pretrain_loader, desc=f"Stage1 Epoch {epoch + 1}/{new_config.stage1_epochs}")
        train_losses = []
        train_id_metrics = []

        for batch in train_bar:
            text_feat = batch['text_feat'].float()
            vision_feat = batch['vision_feat'].float()
            # 前向传播（仅获取量化相关输出）
            quantized, indices, quant_layers,code_probs, raw= model.ahrq(text_feat, vision_feat)

            # Stage1总损失
            loss, loss_dict = compute_quantization_loss(quantized, raw,code_probs, indices, new_config)


            # 反向传播（仅更新AH-RQ参数）
            optimizer_quant.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(quant_params, new_config.grad_clip)
            optimizer_quant.step()

            # 记录指标
            train_losses.append(loss.item())

            with torch.no_grad():
                # 仅记录ID质量指标（Stage1无排序指标）
                id_metrics = calculate_id_metrics(indices)
            train_id_metrics.append(id_metrics)

            # 更新进度条
            avg_loss = np.mean(train_losses)
            train_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "recon_loss": f"{loss_dict['recon_loss']:.6f}",
                "usage_loss": f"{loss_dict['usage_loss']:.6f}",
                "total_loss": f"{loss_dict['total_loss']:.6f}"
            })

        # 学习率调度
        scheduler_quant.step()

        # Stage1验证（仅评估量化质量）
        model.eval()
        val_recon_losses = []  # 存储验证集重构损失
        val_id_metrics = []
        with torch.no_grad():
            val_bar = tqdm(pretrain_loader, desc=f"Stage1 Epoch {epoch + 1}/{new_config.stage1_epochs}")

            for batch in val_bar:
                text_feat = batch['text_feat'].float()
                vision_feat = batch['vision_feat'].float()
                quantized, indices_list, quant_layers, code_probs, raw = model.ahrq(text_feat,vision_feat)

                recon_loss = F.mse_loss(quantized, raw).item()
                val_recon_losses.append(recon_loss)

                v_id_metrics = calculate_id_metrics(indices_list)
                val_id_metrics.append(v_id_metrics)
                val_bar.set_postfix({
                    "val_recon_loss": f"{np.mean(val_recon_losses):.6f}",
                    "id_repeat_rate": f"{v_id_metrics['id_repeat_rate']:.4f}"
                })
            val_bar.close()

        # 计算Stage1平均指标
        avg_train_loss = np.mean(train_losses)
        avg_val_recon_loss = np.mean(val_recon_losses)
        avg_train_id_repeat = np.mean([m["id_repeat_rate"] for m in train_id_metrics]) if train_id_metrics else 0.0
        avg_val_id_repeat = np.mean([m["id_repeat_rate"] for m in val_id_metrics]) if val_id_metrics else 0.0

        # ========== 核心修复3：日志只打印重构损失+ID重复率（移除Gini） ==========
        print(f"\nStage1 Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Recon Loss: {avg_val_recon_loss:.6f}")
        print(f"Train ID Repeat Rate: {avg_train_id_repeat:.4f} | Val ID Repeat Rate: {avg_val_id_repeat:.4f}")

        # ========== 核心修复4：早停只传重构损失（移除Gini） ==========
        early_stopping_quant(avg_val_recon_loss, model, optimizer_quant)

        # 更新最优重构损失（日志用）
        if avg_val_recon_loss < best_recon_loss:
            best_recon_loss = avg_val_recon_loss

        # 触发早停则终止训练
        if early_stopping_quant.early_stop:
            print(f"Stage1 Early stop at epoch {epoch + 1}! Best Val Recon Loss: {early_stopping_quant.best_score:.6f}")
            break

        # 每5轮执行稳定的全局死码重置（避免扰动）
        if (epoch + 1) % 5 == 0 and epoch > 0:
            # 修复：特征加device
            fast_codebook_reset(
                model.ahrq,
                all_item_meta['text_features'].float().to(new_config.device),
                all_item_meta['image_features'].float().to(new_config.device),
                new_config
            )# 用训练集特征重置

    print("\n========== Stage 2: Recommendation Training (SASRec Only) ==========")
    # 应用Stage2冻结逻辑（模型内置方法）
    model.freeze_for_stage2()

    train_loader, val_loader, test_loader, all_item_features = get_pmat_dataloader(
        cache_dir="./data",
        category='Video_Games',
        batch_size=new_config.batch_size,
        max_history_len=new_config.sasrec_max_len,
        num_negative_samples=new_config.num_negative_samples,
        shuffle=True,
        quick_mode=True,
        num_workers=NUM_WORKS,
        logger=logger
    )

    # Stage2优化器：仅SASRec可训练参数
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
        train_id_metrics = []

        for batch in train_bar:
            # 前向传播（AH-RQ已冻结，仅更新SASRec）
            pos_scores, neg_scores, _, user_emb, indices_list, _, _, _ = model(batch)

            # Stage2损失：仅排序损失（BPR+分数正则，无量化损失）

            loss, loss_dict = compute_ranking_loss(pos_scores, neg_scores, new_config)


            # 反向传播（仅更新SASRec参数）
            optimizer_rec.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rec_params, new_config.grad_clip)
            optimizer_rec.step()

            # 记录指标
            train_losses.append(loss.item())
            # 推荐指标
            rec_metrics = calculate_metrics(pos_scores.detach(), neg_scores.detach())
            # ID质量指标（监控是否稳定）
            id_metrics = calculate_id_metrics([idx.detach() for idx in indices_list])
            batch_metrics = {**loss_dict, **rec_metrics, **id_metrics}
            train_metrics.append(batch_metrics)

            # 更新进度条
            hr10 = batch_metrics.get("HR@10", 0.0)
            avg_loss = np.mean(train_losses)
            train_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "HR@10": f"{hr10:.4f}",
                **loss_dict
            })

        # 学习率调度
        scheduler_rec.step()

        # Stage2验证
        model.eval()
        val_losses = []
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                pos_scores, neg_scores, _, _, indices_list, _, _, _ = model(batch)

                # 验证阶段排序损失
                loss, loss_dict = compute_ranking_loss(pos_scores, neg_scores, new_config)

                val_losses.append(loss.item())
                rec_metrics = calculate_metrics(pos_scores, neg_scores)
                id_metrics = calculate_id_metrics(indices_list)
                val_metrics.append({**loss_dict, **rec_metrics, **id_metrics})

        # 计算Stage2平均指标
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        avg_train_metrics = {}
        for key in train_metrics[0].keys():
            avg_train_metrics[key] = np.mean([m[key] for m in train_metrics])

        avg_val_metrics = {}
        for key in val_metrics[0].keys():
            avg_val_metrics[key] = np.mean([m[key] for m in val_metrics])

        # 打印Stage2日志（保持和原脚本一致的输出格式）
        print(f"\nStage2 Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train HR@10: {avg_train_metrics['HR@10']:.4f} | Val HR@10: {avg_val_metrics['HR@10']:.4f}")
        print(f"Train NDCG@10: {avg_train_metrics['NDCG@10']:.4f} | Val NDCG@10: {avg_val_metrics['NDCG@10']:.4f}")
        print(
            f"Train ID Repeat Rate: {avg_train_metrics['id_repeat_rate']:.4f} | Val ID Repeat Rate: {avg_val_metrics['id_repeat_rate']:.4f}")
        print(
            f"Train Gini Layer0: {avg_train_metrics['gini_layer0']:.4f} | Val Gini Layer0: {avg_val_metrics['gini_layer0']:.4f}")

        # 保存Stage2最优排序模型
        if avg_val_metrics["NDCG@10"] > best_ndcg:
            best_ndcg = avg_val_metrics["NDCG@10"]
            torch.save({
                "stage": 2,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_rec_state_dict": optimizer_rec.state_dict(),
                "best_ndcg": best_ndcg
            }, "./best_sasrec_ahrq.pth")
            print(f"Stage2 Best model saved! NDCG@10: {best_ndcg:.4f}")

    # 执行全量评估
    test_hr_full, test_ndcg_full = evaluate_test_full(
        model=model,
        test_loader=test_loader,
        all_item_features=all_item_features,
        topk=10,
        device=new_config.device
    )

    # 打印最终全量评估结果
    print("\n========== Final Test Result (Full Ranking) ==========")
    print(f"Test HR@10 (Full): {test_hr_full:.4f}")
    print(f"Test NDCG@10 (Full): {test_ndcg_full:.4f}")

    # 最终保存完整模型
    torch.save(model.state_dict(), "./final_sasrec_ahrq.pth")
    print("\n========== Two-Stage Training Completed! ==========")
    print(f"Best Quant Recon Loss: {best_recon_loss:.6f} | Best Recommendation NDCG (Val): {best_ndcg:.4f}")
    print(f"Final Test HR@10 (Full): {test_hr_full:.4f} | Final Test NDCG@10 (Full): {test_ndcg_full:.4f}")


if __name__ == "__main__":
    train_sasrec_ahrq()
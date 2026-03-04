import torch
import os
import numpy as np
from tqdm import tqdm
from config import new_config
from data_utils import get_pmat_dataloader
from our_models.sasrec_ahrq import SASRecAHRQ
from utils.loss import total_loss
from utils.utils import calculate_metrics, calculate_id_metrics, seed_everything
from log import Logger


NUM_WORKS = 0 if os.name == 'nt' else 4

# ===================== 训练逻辑 =====================
def train_sasrec_ahrq():
    logger = Logger("./logs/train_sasrec_ahrq.log")
    # 固定种子
    seed_everything(new_config.seed)

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

    # 2. 初始化模型
    model = SASRecAHRQ().to(new_config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=new_config.lr, weight_decay=new_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=new_config.epochs)

    # 3. 训练
    best_ndcg = 0.0
    for epoch in range(new_config.epochs):
        # 训练阶段
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{new_config.epochs}")
        train_losses = []
        train_metrics = []
        train_id_metrics = []

        for batch in train_bar:

            # 前向传播
            pos_scores, neg_scores, quantized, user_emb, indices_list, quantized_layers = model(batch)

            diff = pos_scores.unsqueeze(1) - neg_scores
            sigmoid_diff = torch.sigmoid(diff)
            loss = -torch.log(sigmoid_diff + 1e-8).mean()

            # 此处加断点，抓取以下数据
            print("=== 环节5：损失计算 ===")
            print(f"diff mean: {diff.mean().item()}, min: {diff.min().item()}, max: {diff.max().item()}")
            print(f"sigmoid_diff mean: {sigmoid_diff.mean().item()}")
            print(f"loss: {loss.item()}")
            print(f"是否有nan: {torch.isnan(loss).item()}")

            # 计算损失
            loss, loss_dict = total_loss(
                pos_scores, neg_scores, quantized, user_emb,
                quantized_layers, indices_list, new_config.semantic_hierarchy
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), new_config.grad_clip)
            optimizer.step()

            # 记录指标
            train_losses.append(loss.item())
            # 推荐指标
            rec_metrics = calculate_metrics(pos_scores.detach(), neg_scores.detach())
            # ID质量指标
            id_metrics = calculate_id_metrics([idx.detach() for idx in indices_list])
            # 合并指标
            batch_metrics = {**loss_dict, **rec_metrics, **id_metrics}
            train_metrics.append(batch_metrics)

            # 更新进度条
            # 1. 合法的参数名 + 格式化为4位小数
            # 2. 空值保护：避免key不存在导致报错
            hr10 = batch_metrics.get("HR@10", 0.0)  # 无该键则默认0.0
            postfix_dict = {
                "loss": f"{np.mean(train_losses):.4f}",  # 平均损失保留4位
                "HR@10": f"{hr10:.4f}"  # HR@10保留4位，参数名无空格
            }
            # 3. 安全更新进度条
            train_bar.set_postfix(**postfix_dict)

            # 学习率调度
            scheduler.step()

            # 4. 验证阶段
            model.eval()
            val_losses = []
            val_metrics = []
            with torch.no_grad():
                for batch in val_loader:

                    pos_scores, neg_scores, quantized, user_emb, indices_list, quantized_layers = model(batch)
                    loss, loss_dict = total_loss(
                        pos_scores, neg_scores, quantized, user_emb,
                        quantized_layers, indices_list, new_config.semantic_hierarchy
                    )

                    val_losses.append(loss.item())
                    rec_metrics = calculate_metrics(pos_scores, neg_scores)
                    id_metrics = calculate_id_metrics(indices_list)
                    val_metrics.append({**loss_dict, **rec_metrics, **id_metrics})

            # 5. 计算平均指标
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            # 训练指标平均
            avg_train_metrics = {}
            for key in train_metrics[0].keys():
                avg_train_metrics[key] = np.mean([m[key] for m in train_metrics])

            # 验证指标平均
            avg_val_metrics = {}
            for key in val_metrics[0].keys():
                avg_val_metrics[key] = np.mean([m[key] for m in val_metrics])

            # 6. 打印日志
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Train HR@10: {avg_train_metrics['HR@10']:.4f} | Val HR@10: {avg_val_metrics['HR@10']:.4f}")
            print(f"Train NDCG@10: {avg_train_metrics['NDCG@10']:.4f} | Val NDCG@10: {avg_val_metrics['NDCG@10']:.4f}")
            print(
                f"Train ID Repeat Rate: {avg_train_metrics['id_repeat_rate']:.4f} | Val ID Repeat Rate: {avg_val_metrics['id_repeat_rate']:.4f}")
            print(
                f"Train Gini Layer0: {avg_train_metrics['gini_layer0']:.4f} | Val Gini Layer0: {avg_val_metrics['gini_layer0']:.4f}")

            # 7. 保存最优模型
            if avg_val_metrics["NDCG@10"] > best_ndcg:
                best_ndcg = avg_val_metrics["NDCG@10"]
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_ndcg": best_ndcg
                }, "./best_sasrec_ahrq.pth")
                print(f"Best model saved! NDCG@10: {best_ndcg:.4f}")

if __name__ == "__main__":
    train_sasrec_ahrq()
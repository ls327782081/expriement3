import os

import numpy as np
import torch
from tqdm import tqdm

from config import new_config
from data_utils import get_pmat_dataloader, get_all_item_pretrain_dataloader
from log import Logger
from our_models.ah_rq import AdaptiveHierarchicalQuantizer
from our_models.sasrec_ahrq import SASRecAHRQ
from utils.loss import compute_ranking_loss, compute_rqvae_recon_loss
from utils.utils import calculate_metrics, calculate_id_metrics, seed_everything, EarlyStopping, fast_codebook_reset


NUM_WORKS = 0 if os.name == 'nt' else 4

# 固定使用RQ-VAE风格残差量化


def evaluate_test_full(model, test_loader, indices_list, topk=10, device="cuda"):
    """测试集全量排序评估"""
    model.eval()
    total_hr = 0.0
    total_ndcg = 0.0
    total_users = 0
    num_items = indices_list.shape[0]

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Full Ranking Evaluation on Test Set"):
            batch_size = batch["history_indices"].shape[0]
            hist_indices = batch["history_indices"]
            target_item_ids = batch["target_item"].cpu().numpy()
            history_item_ids = batch["history_items"].cpu().numpy()

            history_items = batch["history_items"].to(device)
            history_len = batch["history_len"]
            all_scores = model.predict_all(
                hist_indices=hist_indices,
                indices_list=indices_list,
                history_items=history_items,
                history_len=history_len,
            )

            # 在 evaluate_test_full 里加
            print(f"all_scores min: {all_scores.min():.4f}, max: {all_scores.max():.4f}")
            print(f"score std: {all_scores.std():.4f}")

            for i in range(batch_size):
                interacted_ids = [id for id in history_item_ids[i] if id != 0 and id < num_items + 1]
                if len(interacted_ids) > 0:
                    all_scores[i, interacted_ids] = -float('inf')

                user_scores = all_scores[i:i + 1, :]
                _, topk_indices = torch.topk(user_scores, k=topk, dim=1)
                topk_ids = topk_indices.squeeze(0).cpu().numpy()

                target_id = target_item_ids[i]
                if target_id - 1 in topk_ids:
                    hr = 1.0
                    rank = np.where(topk_ids == target_id - 1)[0][0] + 1
                    ndcg = 1.0 / np.log2(rank + 1)
                else:
                    hr = 0.0
                    ndcg = 0.0

                total_hr += hr
                total_ndcg += ndcg
                total_users += 1

    if total_users == 0:
        return 0.0, 0.0
    avg_hr = total_hr / total_users
    avg_ndcg = total_ndcg / total_users
    return avg_hr, avg_ndcg


def train_sasrec_ahrq():
    logger = Logger("./logs/train_sasrec_ahrq.log")
    seed_everything(new_config.seed)

    ahrq = AdaptiveHierarchicalQuantizer(
        hidden_dim=new_config.ahrq_hidden_dim,
        semantic_hierarchy=new_config.semantic_hierarchy,
        use_multimodal=True,
        text_dim=new_config.text_dim,
        visual_dim=new_config.visual_dim,
        beta=new_config.ahrq_beta,
        use_ema=new_config.ahrq_use_ema,
        ema_decay=0.99,
        reset_unused_codes=new_config.ahrq_reset_unused_codes,
        reset_threshold=new_config.ahrq_reset_threshold
    ).to(new_config.device)


    mode_name = "RQ-VAE Residual"

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


    # 检查是否已存在训练好的AHRQ模型
    ahrq_model_path = "./ahrq_final.pth"
    if os.path.exists(ahrq_model_path):
        print(f"\n========== Loading existing AHRQ model from {ahrq_model_path} ==========")
        checkpoint = torch.load(ahrq_model_path, map_location=new_config.device, weights_only=False)
        ahrq.load_state_dict(checkpoint['model_state_dict'])
        print(f"AHRQ model loaded! Skipping Stage 1 training.")
        best_recon_loss = checkpoint.get('best_recon_loss', float('inf'))
    else:
        print(f"\n========== Stage 1: Quantization Pre-training ({mode_name}) ==========")

        optimizer_quant = torch.optim.AdamW(
            ahrq.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        scheduler_quant = torch.optim.lr_scheduler.StepLR(
            optimizer_quant,
            step_size=10,
            gamma=0.9
        )

        early_stopping_quant = EarlyStopping(
            patience=getattr(new_config, "stage1_patience", 5),
            verbose=True,
            delta=1e-6,
            path="./best_ahrq.pth",
            mode='min'
        )

        best_recon_loss = float('inf')

        for epoch in range(new_config.stage1_epochs):
            ahrq.train()
            train_bar = tqdm(pretrain_loader, desc=f"Stage1 Epoch {epoch + 1}/{new_config.stage1_epochs}")
            train_losses = []
            train_id_metrics = []
            train_rqvae_losses = []

            for batch in train_bar:
                text_feat = batch['text_feat'].float()
                vision_feat = batch['vision_feat'].float()

                # 前向传播（8个返回值）
                quantized, indices, raw, quant_loss = \
                    ahrq(text_feat, vision_feat)
                # 新的返回值：quantized, indices, raw, quant_loss

                # 使用RQ-VAE风格的重构损失
                loss, loss_dict = compute_rqvae_recon_loss(
                    quantized, raw, None, None, new_config, [quant_loss]
                )
                train_rqvae_losses.append(loss_dict)

                optimizer_quant.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ahrq.parameters(), new_config.grad_clip)
                optimizer_quant.step()

                train_losses.append(loss.item())

                with torch.no_grad():
                    id_metrics = calculate_id_metrics(indices)
                train_id_metrics.append(id_metrics)

                avg_loss = np.mean(train_losses)
                train_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "recon": f"{loss_dict.get('rqvae_recon_loss', 0):.6f}",
                    "vq": f"{loss_dict.get('rqvae_vq_loss', 0):.6f}",
                })

            scheduler_quant.step()

            # 验证
            ahrq.eval()
            val_recon_losses = []
            val_id_metrics = []
            with torch.no_grad():
                val_bar = tqdm(pretrain_loader, desc=f"Stage1 Val {epoch + 1}/{new_config.stage1_epochs}")

                for batch in val_bar:
                    text_feat = batch['text_feat'].float()
                    vision_feat = batch['vision_feat'].float()
                    quantized, indices, raw, quant_loss = \
                        ahrq(text_feat, vision_feat)
                # 新的返回值：quantized, indices, raw, quant_loss

                    loss, loss_dict = compute_rqvae_recon_loss(
                        quantized, raw, None, None, new_config, [quant_loss]
                    )
                    val_recon_losses.append(loss_dict['rqvae_recon_loss'])

                    v_id_metrics = calculate_id_metrics(indices)
                    val_id_metrics.append(v_id_metrics)
                    val_bar.set_postfix({
                        "val_recon_loss": f"{np.mean(val_recon_losses):.6f}",
                        "id_repeat_rate": f"{v_id_metrics['id_repeat_rate']:.4f}"
                    })
                val_bar.close()

            avg_train_loss = np.mean(train_losses)
            avg_val_recon_loss = np.mean(val_recon_losses)
            avg_train_id_repeat = np.mean([m["id_repeat_rate"] for m in train_id_metrics]) if train_id_metrics else 0.0
            avg_val_id_repeat = np.mean([m["id_repeat_rate"] for m in val_id_metrics]) if val_id_metrics else 0.0

            avg_rqvae = {}
            if train_rqvae_losses:
                for key in train_rqvae_losses[0].keys():
                    avg_rqvae[key] = np.mean([d[key] for d in train_rqvae_losses])

            print(f"\nStage1 Epoch {epoch + 1} Summary ({mode_name}):")
            print(f"Train Loss: {avg_train_loss:.4f} | Val Recon Loss: {avg_val_recon_loss:.6f}")
            print(f"Train ID Repeat Rate: {avg_train_id_repeat:.4f} | Val ID Repeat Rate: {avg_val_id_repeat:.4f}")
            print(f"  RQ-VAE: recon={avg_rqvae.get('rqvae_recon_loss', 0):.6f}, vq={avg_rqvae.get('rqvae_vq_loss', 0):.6f}")

            early_stopping_quant(avg_val_recon_loss, ahrq, optimizer_quant)

            if avg_val_recon_loss < best_recon_loss:
                best_recon_loss = avg_val_recon_loss

            if early_stopping_quant.early_stop:
                print(f"Stage1 Early stop at epoch {epoch + 1}! Best Val Recon Loss: {early_stopping_quant.best_score:.6f}")
                break

            if (epoch + 1) % 5 == 0 and epoch > 0:
                fast_codebook_reset(
                    ahrq,
                    all_item_meta['text_features'].float().to(new_config.device),
                    all_item_meta['image_features'].float().to(new_config.device),
                    new_config
                )
        # 保存AHRQ模型
        torch.save({
            "stage": 1,
            "epoch": epoch + 1,
            "model_state_dict": ahrq.state_dict(),
            "best_recon_loss": best_recon_loss
        }, "./ahrq_final.pth")
        print(f"AHRQ model saved to ahrq_final.pth")


    # Stage 1结束后重新计算indices_list
    print("\nRecomputing all item semantics after Stage 1...")
    all_item_text = all_item_meta['text_features'].float().to(new_config.device)
    all_item_vision = all_item_meta['image_features'].float().to(new_config.device)
    _, indices_list, _, _ = ahrq(all_item_text, all_item_vision)

    # 检查是否已存在训练好的SASRec_AHRQ模型
    model = SASRecAHRQ().to(new_config.device)
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
    
    sasrec_ahrq_model_path = "./final_sasrec_ahrq.pth"
    if os.path.exists(sasrec_ahrq_model_path):
        print(f"\n========== Loading existing SASRec_AHRQ model from {sasrec_ahrq_model_path} ==========")
        checkpoint = torch.load(sasrec_ahrq_model_path, map_location=new_config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"SASRec_AHRQ model loaded! Skipping Stage 2 training.")
    else:
        print("\n========== Stage 2: Recommendation Training (SASRec Only) ==========")

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

            for batch in train_bar:
                pos_scores, neg_scores, user_emb = model(batch)

                score_diff = (pos_scores.detach().unsqueeze(1) - neg_scores.detach()).mean().item()
                logger.info(f"Epoch {epoch}: BPR分数差 = {score_diff:.4f}")

                loss, loss_dict = compute_ranking_loss(pos_scores, neg_scores, new_config)

                optimizer_rec.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rec_params, new_config.grad_clip)
                optimizer_rec.step()

                train_losses.append(loss.item())
                rec_metrics = calculate_metrics(pos_scores.detach(), neg_scores.detach())
                batch_metrics = {**loss_dict, **rec_metrics}
                train_metrics.append(batch_metrics)

                hr10 = batch_metrics.get("HR@10", 0.0)
                ndgc10 = batch_metrics.get("NDCG@10", 0.0)
                avg_loss = np.mean(train_losses)
                train_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "HR@10": f"{hr10:.4f}",
                    "NDCG@10": f"{ndgc10:.4f}",
                    **loss_dict
                })

            scheduler_rec.step()

            model.eval()
            val_losses = []
            val_metrics = []
            with torch.no_grad():
                for batch in val_loader:
                    pos_scores, neg_scores, _ = model(batch)
                    loss, loss_dict = compute_ranking_loss(pos_scores, neg_scores, new_config)
                    val_losses.append(loss.item())
                    rec_metrics = calculate_metrics(pos_scores, neg_scores)
                    val_metrics.append({**loss_dict, **rec_metrics})

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
                }, "./best_sasrec_ahrq.pth")
                print(f"Stage2 Best model saved! NDCG@10: {best_ndcg:.4f}")

        torch.save({"model_state_dict": model.state_dict()}, "./final_sasrec_ahrq.pth")
    test_hr_full, test_ndcg_full = evaluate_test_full(
        model=model,
        test_loader=test_loader,
        indices_list=indices_list,
        topk=10,
        device=new_config.device
    )

    print("\n========== Final Test Result (Full Ranking) ==========")
    print(f"Test HR@10 (Full): {test_hr_full:.4f}")
    print(f"Test NDCG@10 (Full): {test_ndcg_full:.4f}")


    print("\n========== Two-Stage Training Completed! ==========")
    print(f"Best Quant Recon Loss: {best_recon_loss:.6f} | Best Recommendation NDCG (Val): {best_ndcg:.4f}")
    print(f"Final Test HR@10 (Full): {test_hr_full:.4f} | Final Test NDCG@10 (Full): {test_ndcg_full:.4f}")


if __name__ == "__main__":
    train_sasrec_ahrq()
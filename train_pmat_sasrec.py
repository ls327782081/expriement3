import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

# 导入配置和核心模块
from config import config
from our_models.pmat_sasrec import PMATSASRec  # 完整推荐模型
from utils.loss import total_loss
from log import Logger
from utils.utils import seed_everything, EarlyStopping, calculate_metrics, calculate_id_metrics


# ===================== 多模态数据集定义（适配PMAT输入） =====================
class PMATSASRecDataset(Dataset):
    """
    多模态序列推荐数据集
    输入：用户-物品交互序列 + 物品多模态特征（文本/视觉）
    输出：PMAT-SASRec所需的多模态特征batch
    """

    def __init__(self, data_path, feat_root, max_len=50, num_neg=4, phase="train"):
        self.max_len = max_len
        self.num_neg = num_neg
        self.phase = phase

        # 1. 加载多模态特征（文本/视觉）
        self.text_feat = np.load(os.path.join(feat_root, "item_text_feat.npy"))  # (num_items, text_dim)
        self.visual_feat = np.load(os.path.join(feat_root, "item_visual_feat.npy"))  # (num_items, visual_dim)
        self.num_items = self.text_feat.shape[0]

        # 2. 加载用户-物品交互序列
        self.user_seq = self._load_user_sequence(data_path)

        # 3. 生成训练/验证/测试样本
        self.samples = self._generate_samples()

    def _load_user_sequence(self, data_path):
        """加载用户交互序列，格式：{user_id: [item1, item2, ...]}"""
        user_seq = {}
        with open(data_path, "r", encoding="utf-8") as f:
            header = f.readline()  # 跳过表头
            for line in f:
                user_id, item_id, timestamp = line.strip().split(",")
                user_id, item_id = int(user_id), int(item_id)
                if user_id not in user_seq:
                    user_seq[user_id] = []
                user_seq[user_id].append(item_id)
        return user_seq

    def _generate_samples(self):
        """生成样本：(历史序列ID, 正样本ID, 负样本ID, 历史长度)"""
        samples = []
        for user_id, seq in self.user_seq.items():
            if len(seq) < 2:
                continue

            # 训练/验证：用前n-1个物品预测第n个；测试：用全部历史预测最后一个
            if self.phase == "test":
                seq = seq[:-1]  # 测试集用最后一个物品作为标签
                if len(seq) < 1:
                    continue

            for i in range(1, len(seq)):
                # 历史序列（截断/补零到max_len）
                hist_ids = seq[:i][-self.max_len:]
                hist_len = len(hist_ids)
                hist_ids = [0] * (self.max_len - hist_len) + hist_ids  # 补零padding

                # 正样本
                pos_id = seq[i]

                # 负样本（保证负样本≠正样本）
                neg_ids = np.random.choice(
                    [x for x in range(self.num_items) if x != pos_id],
                    size=self.num_neg,
                    replace=False
                )

                samples.append((hist_ids, pos_id, neg_ids, hist_len))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """将ID转换为多模态特征，返回模型输入"""
        hist_ids, pos_id, neg_ids, hist_len = self.samples[idx]

        # 1. 历史序列多模态特征
        history_text = self.text_feat[hist_ids]  # (max_len, text_dim)
        history_visual = self.visual_feat[hist_ids]  # (max_len, visual_dim)

        # 2. 正样本多模态特征
        target_text = self.text_feat[pos_id]  # (text_dim,)
        target_visual = self.visual_feat[pos_id]  # (visual_dim,)

        # 3. 负样本多模态特征
        neg_text = self.text_feat[neg_ids]  # (num_neg, text_dim)
        neg_visual = self.visual_feat[neg_ids]  # (num_neg, visual_dim)

        # 转换为tensor
        return {
            "history_text": torch.tensor(history_text, dtype=torch.float),
            "history_visual": torch.tensor(history_visual, dtype=torch.float),
            "history_len": torch.tensor(hist_len, dtype=torch.long),
            "target_text": torch.tensor(target_text, dtype=torch.float),
            "target_visual": torch.tensor(target_visual, dtype=torch.float),
            "neg_text": torch.tensor(neg_text, dtype=torch.float),
            "neg_visual": torch.tensor(neg_visual, dtype=torch.float)
        }


# ===================== 核心实验流程 =====================
def run_experiment():
    # 1. 初始化配置
    seed_everything(config.seed)  # 固定随机种子
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    # 初始化日志和早停
    logger = Logger(os.path.join(config.log_dir, "pmat_sasrec_experiment.log"))
    early_stopping = EarlyStopping(patience=config.patience, verbose=True,
                                   path=os.path.join(config.model_dir, "best_pmat_sasrec.pth"))

    # 2. 加载数据集
    logger.info("Loading datasets...")
    train_dataset = PMATSASRecDataset(
        data_path=config.train_data_path,
        feat_root=config.feat_root,
        max_len=config.sasrec_max_len,
        num_neg=config.num_neg,
        phase="train"
    )
    val_dataset = PMATSASRecDataset(
        data_path=config.val_data_path,
        feat_root=config.feat_root,
        max_len=config.sasrec_max_len,
        num_neg=config.num_neg,
        phase="val"
    )
    test_dataset = PMATSASRecDataset(
        data_path=config.test_data_path,
        feat_root=config.feat_root,
        max_len=config.sasrec_max_len,
        num_neg=config.num_neg,
        phase="test"
    )

    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # 3. 初始化模型和优化器
    logger.info("Initializing model...")
    model = PMATSASRec().to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6
    )

    # 4. 训练循环
    logger.info("Starting training...")
    for epoch in range(config.epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_losses = []
        train_hr10_list = []
        train_ndcg10_list = []
        train_id_repeat_list = []

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{config.epochs}] Train")
        for batch in train_bar:
            # 数据移到设备
            batch = {k: v.to(config.device) for k, v in batch.items()}

            # 前向传播
            pos_scores, neg_scores, quantized, user_emb, indices_list, quantized_layers = model(batch)

            # 计算损失（复用total_loss，与sasrec_ahrq一致）
            loss, loss_dict = total_loss(
                pos_scores=pos_scores,
                neg_scores=neg_scores,
                quantized=quantized,
                user_emb=user_emb,
                quantized_layers=quantized_layers,
                indices_list=indices_list,
                semantic_hierarchy=config.semantic_hierarchy
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)  # 梯度裁剪
            optimizer.step()

            # 记录指标
            train_losses.append(loss.item())

            # 推荐指标（HR@10, NDCG@10）
            rec_metrics = calculate_metrics(pos_scores.detach(), neg_scores.detach(), k=10)
            train_hr10_list.append(rec_metrics["HR@10"])
            train_ndcg10_list.append(rec_metrics["NDCG@10"])

            # ID质量指标（重复率）
            id_metrics = calculate_id_metrics(indices_list)
            train_id_repeat_list.append(id_metrics["id_repeat_rate"])

            # 更新进度条
            train_bar.set_postfix(
                loss=f"{np.mean(test_losses):.4f}",
                HR10=f"{np.mean(test_hr10_list):.4f}",
                NDCG10=f"{np.mean(test_ndcg10_list):.4f}",
                ID_Repeat=f"{np.mean(test_id_repeat_list):.4f}"
            )

            # 学习率调度
            scheduler.step()

            # ========== 验证阶段 ==========
            model.eval()
            val_losses = []
            val_hr10_list = []
            val_ndcg10_list = []
            val_id_repeat_list = []

            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{config.epochs}] Val")
                for batch in val_bar:
                    batch = {k: v.to(config.device) for k, v in batch.items()}

                    # 前向传播
                    pos_scores, neg_scores, quantized, user_emb, indices_list, quantized_layers = model(batch)

                    # 计算损失
                    loss, _ = total_loss(
                        pos_scores=pos_scores,
                        neg_scores=neg_scores,
                        quantized=quantized,
                        user_emb=user_emb,
                        quantized_layers=quantized_layers,
                        indices_list=indices_list,
                        semantic_hierarchy=config.semantic_hierarchy
                    )
                    val_losses.append(loss.item())

                    # 计算指标
                    rec_metrics = calculate_metrics(pos_scores, neg_scores, k=10)
                    id_metrics = calculate_id_metrics(indices_list)

                    val_hr10_list.append(rec_metrics["HR@10"])
                    val_ndcg10_list.append(rec_metrics["NDCG@10"])
                    val_id_repeat_list.append(id_metrics["id_repeat_rate"])

                    # 更新进度条
                    val_bar.set_postfix(
                        loss=f"{np.mean(test_losses):.4f}",
                        HR10=f"{np.mean(test_hr10_list):.4f}",
                        NDCG10=f"{np.mean(test_ndcg10_list):.4f}",
                        ID_Repeat=f"{np.mean(test_id_repeat_list):.4f}"
                    )

                    # ========== 日志记录 ==========
                    avg_train_loss = np.mean(train_losses)
                    avg_train_hr10 = np.mean(train_hr10_list)
                    avg_train_ndcg10 = np.mean(train_ndcg10_list)
                    avg_train_id_repeat = np.mean(train_id_repeat_list)

                    avg_val_loss = np.mean(val_losses)
                    avg_val_hr10 = np.mean(val_hr10_list)
                    avg_val_ndcg10 = np.mean(val_ndcg10_list)
                    avg_val_id_repeat = np.mean(val_id_repeat_list)

                    logger.info(f"Epoch [{epoch + 1}/{config.epochs}]")
                    logger.info(
                        f"Train - Loss: {avg_train_loss:.4f}, HR@10: {avg_train_hr10:.4f}, NDCG@10: {avg_train_ndcg10:.4f}, ID_Repeat: {avg_train_id_repeat:.4f}")
                    logger.info(
                        f"Val   - Loss: {avg_val_loss:.4f}, HR@10: {avg_val_hr10:.4f}, NDCG@10: {avg_val_ndcg10:.4f}, ID_Repeat: {avg_val_id_repeat:.4f}")

                    # ========== 早停和模型保存 ==========
                    early_stopping(avg_val_ndcg10, model, optimizer)
                    if early_stopping.early_stop:
                        logger.info("Early stopping triggered!")
                        break

                # 5. 测试阶段（加载最优模型）
                logger.info("Starting testing...")
                model.load_state_dict(
                    torch.load(os.path.join(config.model_dir, "best_pmat_sasrec.pth"), weights_only=False)["model_state_dict"])
                model.eval()

                test_losses = []
                test_hr10_list = []
                test_ndcg10_list = []
                test_id_repeat_list = []

                with torch.no_grad():
                    test_bar = tqdm(test_loader, desc="Testing")
                    for batch in test_bar:
                        batch = {k: v.to(config.device) for k, v in batch.items()}

                        pos_scores, neg_scores, quantized, user_emb, indices_list, quantized_layers = model(batch)

                        # 计算损失
                        loss, _ = total_loss(
                            pos_scores=pos_scores,
                            neg_scores=neg_scores,
                            quantized=quantized,
                            user_emb=user_emb,
                            quantized_layers=quantized_layers,
                            indices_list=indices_list,
                            semantic_hierarchy=config.semantic_hierarchy
                        )
                        test_losses.append(loss.item())

                        # 计算指标
                        rec_metrics = calculate_metrics(pos_scores, neg_scores, k=10)
                        id_metrics = calculate_id_metrics(indices_list)

                        test_hr10_list.append(rec_metrics["HR@10"])
                        test_ndcg10_list.append(rec_metrics["NDCG@10"])
                        test_id_repeat_list.append(id_metrics["id_repeat_rate"])

                        test_bar.set_postfix(
                            loss=f"{np.mean(test_losses):.4f}",
                            HR10=f"{np.mean(test_hr10_list):.4f}",
                            NDCG10=f"{np.mean(test_ndcg10_list):.4f}",
                            ID_Repeat=f"{np.mean(test_id_repeat_list):.4f}"
                        )

                        # 记录测试结果
                        avg_test_loss = np.mean(test_losses)
                        avg_test_hr10 = np.mean(test_hr10_list)
                        avg_test_ndcg10 = np.mean(test_ndcg10_list)
                        avg_test_id_repeat = np.mean(test_id_repeat_list)

                        logger.info("=" * 50)
                        logger.info("Final Test Results:")
                        logger.info(f"Test Loss: {avg_test_loss:.4f}")
                        logger.info(f"Test HR@10: {avg_test_hr10:.4f}")
                        logger.info(f"Test NDCG@10: {avg_test_ndcg10:.4f}")
                        logger.info(f"Test ID Repeat Rate: {avg_test_id_repeat:.4f}")
                        logger.info("=" * 50)



                    # ===================== 运行实验 =====================
                    if __name__ == "__main__":
                        run_experiment()
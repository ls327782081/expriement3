import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, MobileNetV2, MobileNet_V2_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import config


# 固定随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(config.seed)


# 离线提取多模态特征（适配L4，避免实时编码）
class AmazonBooksProcessor:
    def __init__(self):
        # 轻量化文本/视觉编码器
        self.text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.vision_model = MobileNetV2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(config.device)
        self.vision_model.eval()  # 仅特征提取

    def load_data(self):
        """加载并预处理Amazon_Books数据集"""
        # 加载数据集（轻量化：仅5万样本）
        dataset = load_dataset("amazon_reviews_multi", "en", split="train")
        dataset = dataset.select(range(config.max_samples))

        # 数据过滤与整理
        df = pd.DataFrame(dataset)
        df = df[["reviewer_id", "product_id", "review_body", "product_image", "star_rating"]]
        df = df.dropna(subset=["reviewer_id", "product_id", "review_body"])

        # 映射用户/物品ID为连续索引
        user2id = {u: i for i, u in enumerate(df["reviewer_id"].unique())}
        item2id = {i: idx for idx, i in enumerate(df["product_id"].unique()[:config.item_vocab_size])}
        df["user_idx"] = df["reviewer_id"].map(user2id)
        df["item_idx"] = df["product_id"].map(item2id)
        df = df.dropna(subset=["user_idx", "item_idx"])

        # 提取文本特征（离线）
        print("提取文本特征...")
        text_features = []
        for text in tqdm(df["review_body"].tolist()):
            inputs = self.text_tokenizer(
                text, truncation=True, max_length=128, padding="max_length", return_tensors="pt"
            ).to(config.device)
            with torch.no_grad():
                text_feat = self.text_tokenizer.encode_plus(text, return_tensors="pt")["input_ids"].squeeze()
                text_features.append(text_feat.cpu().numpy())
        df["text_feat"] = text_features

        # 提取视觉特征（简化：用随机特征模拟，避免图片加载耗时）
        # 注：真实实验可替换为实际图片特征提取
        df["vision_feat"] = [np.random.rand(1280) for _ in range(len(df))]

        # 划分训练/验证/测试集
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=config.seed)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=config.seed)

        # 保存预处理数据
        os.makedirs("./data", exist_ok=True)
        train_df.to_pickle("./data/train.pkl")
        val_df.to_pickle("./data/val.pkl")
        test_df.to_pickle("./data/test.pkl")
        return train_df, val_df, test_df


class BooksDataset(Dataset):
    """自定义数据集类"""

    def __init__(self, df_path):
        self.df = pd.read_pickle(df_path)
        self.user_ids = self.df["user_idx"].values
        self.item_ids = self.df["item_idx"].values
        self.text_feats = np.stack(self.df["text_feat"].values)
        self.vision_feats = np.stack(self.df["vision_feat"].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "item_id": torch.tensor(self.item_ids[idx], dtype=torch.long),
            "text_feat": torch.tensor(self.text_feats[idx], dtype=torch.float),
            "vision_feat": torch.tensor(self.vision_feats[idx], dtype=torch.float)
        }


def get_dataloader(df_path, shuffle=True):
    """获取DataLoader"""
    dataset = BooksDataset(df_path)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        pin_memory=True,  # 加速GPU传输
        num_workers=2
    )


# 初始化数据处理器（首次运行执行）
if not os.path.exists("./data/train.pkl"):
    processor = AmazonBooksProcessor()
    processor.load_data()
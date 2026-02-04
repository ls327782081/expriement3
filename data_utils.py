import logging
import os
import pickle
import re
import time
import gc
import json
import numpy as np
import pandas as pd
import torch
import requests
from PIL import Image
from io import BytesIO
import datasets
from datasets import load_dataset
HAS_DATASETS = True

from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

HAS_TRANSFORMERS = True

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple, Optional
from config import config
import tqdm
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# 固定随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(config.seed)


# 离线提取多模态特征（适配L4，避免实时编码）
class AmazonBooksProcessor:
    def __init__(self,
                 category: str,
                 quick_mode: bool = False,
                 min_interactions: int = 3,
                 min_items: int = 5,

                 bert_model: str = "bert-base-uncased",
                 clip_model: str = "openai/clip-vit-base-patch32",
                 device: str = "auto",
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        """
        初始化Amazon Books数据集处理器

        Args:
            quick_mode: 是否使用快速模式（减少数据量）
            min_interactions: 用户最小交互次数
            min_items: 商品最小交互次数
            max_users: 最大用户数
            max_items: 最大商品数
            bert_model: BERT模型名称
            clip_model: CLIP模型名称
            device: 计算设备
            logger: 日志记录器
            **kwargs: 其他参数
        """
        # 设置日志记录器
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # 基本配置

        self.category=category

        self.quick_mode = quick_mode
        self.min_interactions = min_interactions
        self.min_items = min_items
        self.bert_model_name = bert_model
        self.clip_model_name = clip_model

        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.logger.info(f"Using device: {self.device}")



        # 初始化预训练模型
        self._init_pretrained_models()

        # 其他参数
        self.kwargs = kwargs

    def _init_pretrained_models(self):
        """初始化预训练模型"""
        self.logger.info("Initializing pre-trained models...")

        # 初始化BERT模型和分词器
        self.logger.info(f"Loading BERT model: {self.bert_model_name}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name, cache_dir="./pre-trained_models/")
        self.bert_model = AutoModel.from_pretrained(self.bert_model_name, cache_dir="./pre-trained_models/").to(self.device)
        self.bert_model.eval()

        # 初始化CLIP模型和处理器
        self.logger.info(f"Loading CLIP model: {self.clip_model_name}")
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name, cache_dir="./pre-trained_models/")
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name, cache_dir="./pre-trained_models/").to(self.device)
        self.clip_model.eval()

        self.logger.info("Pre-trained models initialized successfully")

    def _log_memory_usage(self, context: str = ""):
        """记录内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            cpu_mem_mb = process.memory_info().rss / 1024 / 1024

            if torch.cuda.is_available():
                gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_mem_max_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.logger.info(f"[{context}] CPU内存: {cpu_mem_mb:.1f}MB, GPU内存: {gpu_mem_mb:.1f}MB (峰值: {gpu_mem_max_mb:.1f}MB)")
            else:
                self.logger.info(f"[{context}] CPU内存: {cpu_mem_mb:.1f}MB")
        except ImportError:
            pass  # psutil未安装，跳过内存监控

    def load_reviews(self) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
        """加载评论数据"""
        self.logger.info("Loading reviews data...")

        # 如果是mock数据，从pkl文件加载
        if self.category == "mock":
            self.logger.info("Loading mock reviews from pkl file...")
            mock_path = './data/mock/interactions.pkl'
            if os.path.exists(mock_path):
                with open(mock_path, 'rb') as f:
                    mock_data = pickle.load(f)
                # mock_data应该包含reviews_df, user_mapping, item_mapping
                return mock_data['reviews_df'], mock_data['user_mapping'], mock_data['item_mapping']

        # 直接从JSONL文件加载数据（不依赖datasets库）
        self.logger.info("Loading reviews from JSONL file...")
        reviews = []

        jsonl_path = f'./data/{self.category}.jsonl'
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Review data file not found: {jsonl_path}")

        # Quick模式下只加载前N条数据
        # 增加到50000条以确保过滤后有足够的数据用于序列推荐
        max_lines = 50000 if self.quick_mode else None
        line_count = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_lines and line_count >= max_lines:
                    self.logger.info(f"Quick mode: stopped at {max_lines} lines")
                    break

                try:
                    review = json.loads(line.strip())
                    reviews.append({
                        'user_id': review.get('user_id'),
                        'item_id': review.get('parent_asin'),
                        'rating': float(review.get('rating', 0)),
                        'timestamp': int(review.get('timestamp', 0)),
                        'title': review.get('title', ''),
                        'text': review.get('text', ''),
                        'verified_purchase': review.get('verified_purchase', False),
                        'helpful_vote': review.get('helpful_vote', 0)
                    })
                    line_count += 1
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse line: {e}")
                    continue

        df = pd.DataFrame(reviews)
        self.logger.info(f"Loaded {len(df)} reviews")

        # 数据清洗（返回df和mappings）
        df, user_mapping, item_mapping = self._clean_reviews_data(df)

        return df, user_mapping, item_mapping

    def _clean_reviews_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
        """
        清洗评论数据

        Returns:
            tuple: (cleaned_df, user_mapping, item_mapping)
        """
        self.logger.info("Cleaning reviews data...")

        original_size = len(df)

        # 移除缺失关键字段的记录
        df = df.dropna(subset=['user_id', 'item_id'])

        # 过滤用户和商品的最小交互次数
        # user_counts = df['user_id'].value_counts()
        # item_counts = df['item_id'].value_counts()
        #
        # valid_users = user_counts[user_counts >= self.min_interactions].index
        # valid_items = item_counts[item_counts >= self.min_items].index
        #
        # df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]

        # 重新编码用户和商品ID
        # 重要：先排序再创建映射，确保映射顺序一致
        unique_users = sorted(df['user_id'].unique())
        unique_items = sorted(df['item_id'].unique())

        user_mapping = {user_id: i for i, user_id in enumerate(unique_users)}  # 从0开始
        item_mapping = {item_id: i+1 for i, item_id in enumerate(unique_items)}  # 从1开始，0保留给padding

        # 应用映射
        df['user_id'] = df['user_id'].map(user_mapping)
        df['item_id'] = df['item_id'].map(item_mapping)

        # 按时间排序
        df = df.sort_values(['user_id', 'timestamp'])

        self.logger.info(f"Data cleaning completed:")
        self.logger.info(f"  Original size: {original_size}")
        self.logger.info(f"  After cleaning: {len(df)}")
        self.logger.info(f"  Users: {len(user_mapping)}")
        self.logger.info(f"  Items: {len(item_mapping)}")

        return df, user_mapping, item_mapping

    def load_meta(self, item_mapping: Dict[str, int]) -> pd.DataFrame:
        """加载商品元数据"""
        self.logger.info("Loading meta data...")

        # 如果是mock数据，从pkl文件加载
        if self.category == "mock":
            self.logger.info("Loading mock meta data from pkl file...")
            mock_path = './data/mock/metadata.pkl'
            if os.path.exists(mock_path):
                with open(mock_path, 'rb') as f:
                    meta_df = pickle.load(f)
                return meta_df

        # 直接从JSONL文件加载数据（不依赖datasets库）
        self.logger.info("Loading meta data from JSONL file...")
        meta_data = []

        jsonl_path = f'./data/meta_{self.category}.jsonl'
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Meta data file not found: {jsonl_path}")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:

                try:
                    item = json.loads(line.strip())

                    item_id = item.get('parent_asin', '')
                    if item_id not in item_mapping:
                        continue

                    images = item.get('images', [])
                    image_urls = []
                    if images:
                        for img in images:
                            if isinstance(img, dict) and 'large' in img:
                                image_urls.append(img['large'])
                    meta_data.append({
                        'item_id': item_id,
                        'title': item.get('title', ''),
                        'main_category': item.get('main_category', ''),
                        'categories': item.get('categories', []),
                        'average_rating': float(item.get('average_rating', 0)),
                        'rating_number': int(item.get('rating_number', 0)),
                        'price': item.get('price'),
                        'features': item.get('features', []),
                        'description': item.get('description', []),
                        'image_urls': image_urls,
                        'store': item.get('store', ''),
                        'details': item.get('details', {})
                    })

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse line: {e}")
                    continue

        df = pd.DataFrame(meta_data)
        self.logger.info(f"Loaded {len(df)} meta items")

        return df

    def _generate_bert_text_features(self, meta_df: pd.DataFrame, item_mapping: Dict[str, int]) -> Dict[int, torch.Tensor]:
        """使用BERT生成文本特征"""
        self.logger.info("Generating BERT text features...")

        batch_size = 100  # 增加批处理大小以提高效率

        # 准备文本数据
        texts_to_process = []
        item_indices = []

        for _, row in meta_df.iterrows():
            item_id = row['item_id']
            if item_id not in item_mapping:
                continue

            item_idx = item_mapping[item_id]

            # 组合文本信息
            text_parts = []

            # 标题
            if row['title']:
                text_parts.append(row['title'])

            # 特征
            if row['features']:
                text_parts.extend(row['features'][:3])  # 取前3个特征

            # 描述
            if row['description']:
                text_parts.extend(row['description'][:2])  # 取前2个描述

            # 类别
            if row['categories']:
                text_parts.append(' '.join(row['categories']))

            # 合并文本并截断
            combined_text = ' '.join(text_parts)
            # 限制文本长度以适应BERT
            if len(combined_text) > 500:
                combined_text = combined_text[:500]

            texts_to_process.append(combined_text)
            item_indices.append(item_idx)

        # 批处理生成BERT文本特征
        text_features = self._extract_bert_features_batch(texts_to_process, item_indices, batch_size)

        self.logger.info(f"Generated BERT text features for {len(text_features)} items")

        # 记录内存使用
        self._log_memory_usage("After BERT feature extraction")

        return text_features

    def _extract_bert_features_batch(self, texts: List[str], item_indices: List[int], batch_size: int) -> Dict[int, torch.Tensor]:
        """批量提取BERT特征"""
        text_features = {}

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_indices = item_indices[i:i + batch_size]

                # 分词和编码
                encoded = self.bert_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )

                # 移动到设备
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                # 获取BERT输出
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)

                # 使用[CLS]标记的隐藏状态作为文本表示
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

                # 存储特征（移动到CPU省份，使用float16减少内存）
                for j, item_idx in enumerate(batch_indices):
                    text_features[item_idx] = cls_embeddings[j].cpu().half()  # float16

                # 输出进度日志
                if (i // batch_size + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} BERT texts")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        return text_features

    def _generate_clip_image_features(self, meta_df: pd.DataFrame, item_mapping: Dict[str, int]) -> Dict[int, torch.Tensor]:
        """使用CLIP生成图像特征"""
        self.logger.info("Generating CLIP image features...")

        image_features = {}
        batch_size = 50  # 图像处理批次较小

        # 准备图像URL数据
        image_urls = []
        item_indices = []

        for _, row in meta_df.iterrows():
            item_id = row['item_id']
            if item_id not in item_mapping:
                continue

            item_idx = item_mapping[item_id]

            # 获取图像URL
            urls = row['image_urls']
            if urls and len(urls) > 0:
                image_urls.append(urls[0])  # 使用第一张图像
                item_indices.append(item_idx)
            else:
                # 如果没有图像URL，创建零特征
                image_features[item_idx] = torch.zeros(512, dtype=torch.float16)

        if len(image_urls) == 0:
            self.logger.warning("No valid image URLs found")
            return image_features

        # 批量提取CLIP特征
        image_features.update(self._extract_clip_features_batch(image_urls, item_indices, batch_size))

        # 检查缺失项
        valid_items = set(item_indices)
        all_items = set(item_mapping.values())
        missing_items = all_items - valid_items


        if missing_items:
            self.logger.warning(f"Created zero features for {len(missing_items)} items with missing images")

        self.logger.info(f"Generated CLIP image features for {len(image_features)} items")

        # 记录内存使用
        self._log_memory_usage("After CLIP image feature extraction")

        return image_features

    def _extract_clip_features_batch(self, image_urls: List[str], item_indices: List[int], batch_size: int) -> Dict[int, torch.Tensor]:
        """批量提取CLIP图像特征"""
        clip_features = {}

        with torch.no_grad():
            for i in range(0, len(image_urls), batch_size):
                batch_urls = image_urls[i:i + batch_size]
                batch_indices = item_indices[i:i + batch_size]

                # 下载和预处理图像
                batch_images = []
                valid_indices = []

                for j, url in enumerate(batch_urls):
                    try:
                        image = self._download_and_preprocess_image(url)
                        if image is not None:
                            batch_images.append(image)
                            valid_indices.append(batch_indices[j])
                        else:
                            self.logger.warning(f"Failed to download image from {url}, using zero features")
                            # 为失败的图像创建零特征（使用float16减少内存）
                            zero_features = torch.zeros(512, dtype=torch.float16)  # CLIP特征维度
                            clip_features[batch_indices[j]] = zero_features
                    except Exception as e:
                        self.logger.error(f"Error processing image from {url}: {e}")
                        zero_features = torch.zeros(512, dtype=torch.float16)
                        clip_features[batch_indices[j]] = zero_features

                if not batch_images:
                    continue

                # 使用CLIP处理器预处理图像
                inputs = self.clip_processor(images=batch_images, return_tensors="pt").to(self.device)

                # 获取CLIP图像特征
                outputs = self.clip_model.get_image_features(**inputs)

                # 存储特征（移动到CPU省份，使用float16减少内存）
                for j, item_idx in enumerate(valid_indices):
                    clip_features[item_idx] = outputs[j].cpu().half()  # float16

                # 输出进度日志
                if (i // batch_size + 1) % 5 == 0:
                    self.logger.info(f"Processed {i + len(batch_urls)}/{len(image_urls)} CLIP images")

        return clip_features

    def _download_and_preprocess_image(self, url: str, max_retries: int = 3) -> Optional[Image.Image]:
        """下载并预处理图像"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                # 打开图像
                image = Image.open(BytesIO(response.content)).convert("RGB")

                # 简单的预处理（CLIP处理器会处理调整大小等）

                return image

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 指数退避：2, 4, 6秒
                    self.logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Failed to download {url} after {max_retries} attempts: {e}")
                    raise

        return None

    def load_dataset(self) -> Dict[str, Any]:
        """加载完整数据集"""
        self.logger.info(f"Loading {self.category} dataset...")

        # 缓存未命中，正常加载数据
        start_time = time.time()

        # 加载评论和元数据
        reviews_df, user_mapping, item_mapping = self.load_reviews()
        meta_df = self.load_meta(item_mapping)

        # 提取文本特征
        text_features = self._generate_bert_text_features(meta_df, item_mapping)

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 提取图像特征
        image_features = self._generate_clip_image_features(meta_df, item_mapping)

        # 准备数据集字典
        num_users = len(user_mapping)
        num_items = len(item_mapping)


        dataset = {
            'reviews_df': reviews_df,
            'meta_df': meta_df,
            'user_mapping': user_mapping,
            'item_mapping': item_mapping,
            'num_users': num_users,
            'num_items': num_items,
            'text_features': text_features,
            'image_features': image_features
        }


        elapsed_time = time.time() - start_time
        self.logger.info(f"Dataset loaded successfully in {elapsed_time:.2f} seconds")

        return dataset

    def load_dataset_for_experiment(
        self,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        add_padding_item: bool = True
    ) -> Dict[str, Any]:
        """
        加载数据集用于实验（包括序列构建、数据分割等）

        Args:
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            add_padding_item: 是否为padding item 0预留位置

        Returns:
            包含所有实验所需数据的字典
        """
        self.logger.info("="*80)
        self.logger.info("Loading dataset for experiment")
        self.logger.info("="*80)

        # 加载基础数据集
        dataset = self.load_dataset()

        # 构建用户序列
        self.logger.info("Building user sequences...")
        from util import build_user_sequences, split_user_sequences

        user_sequences = build_user_sequences(
            dataset['reviews_df'], logger=self.logger
        )

        # 分割序列
        train_sequences, val_sequences, test_sequences = split_user_sequences(
            user_sequences,
            test_ratio=test_ratio,
            val_ratio=val_ratio
        )

        # 添加到数据集
        data = {
            **dataset,
            'train_sequences': train_sequences,
            'val_sequences': val_sequences,
            'test_sequences': test_sequences,
            'user_sequences': user_sequences
        }

        # 验证数据范围
        # self._validate_data_ranges(data)

        # 转换特征格式（添加padding item）
        if add_padding_item:
            data = self._convert_features_to_tensors(data)

        self.logger.info("="*80)
        self.logger.info("Dataset loaded successfully for experiment")
        self.logger.info("="*80)

        return data

    def _validate_data_ranges(self, data: Dict[str, Any]):
        """验证数据范围，确保item_id在有效范围内"""
        num_items = data['num_items']
        num_users = data['num_users']

        # 检查所有序列中的item_id
        all_sequences = {**data['train_sequences'], **data['val_sequences'], **data['test_sequences']}

        max_item_id = 0
        max_user_id = 0
        invalid_items = []

        for user_id, seq in all_sequences.items():
            max_user_id = max(max_user_id, user_id)
            for item_id in seq['item_indices']:
                max_item_id = max(max_item_id, item_id)
                if item_id > num_items:
                    invalid_items.append((user_id, item_id))

        self.logger.info(f"Data validation:")
        self.logger.info(f"  num_users: {num_users}, max_user_id: {max_user_id}")
        self.logger.info(f"  num_items: {num_items}, max_item_id: {max_item_id}")

        if invalid_items:
            self.logger.warning(f"Found {len(invalid_items)} invalid item_ids (> num_items={num_items})")
            self.logger.warning(f"First 5 invalid items: {invalid_items[:5]}")

            # 修复：将超出范围的item_id截断到有效范围
            self.logger.info(f"Fixing invalid item_ids by clamping to [1, {num_items}]...")

            for user_id, seq in all_sequences.items():
                for i, item_id in enumerate(seq['item_indices']):
                    if item_id > num_items:
                        seq['item_indices'][i] = min(item_id, num_items)

    def _convert_features_to_tensors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """将特征转换为张量格式，并添加padding item"""
        num_items = data['num_items']

        self.logger.info("Converting features to tensors...")

        # 处理文本特征
        if isinstance(data['text_features'], dict):
            self.logger.info(f"Converting text features to tensor ({num_items} items)...")

            # 创建张量（+1为padding item 0预留位置）
            text_tensor = torch.zeros(num_items + 1, 768, dtype=torch.float16)  # BERT特征维度

            # 填充特征
            for item_idx, feat in data['text_features'].items():
                # 确保索引在有效范围内
                if 0 <= item_idx <= num_items:
                    text_tensor[item_idx] = feat
                else:
                    self.logger.warning(f"Skipping invalid item_idx {item_idx} (num_items={num_items})")

            data['text_features'] = text_tensor
            self.logger.info(f"✅ Converted text_features to tensor: {text_tensor.shape}")

        elif isinstance(data['text_features'], torch.Tensor):
            # 如果已经是tensor，检查维度是否正确
            current_shape = data['text_features'].shape
            if current_shape[0] == num_items:
                # 需要添加padding item 0
                self.logger.info(f"Adding padding item 0 to text_features: {current_shape} -> [{num_items+1}, {current_shape[1]}]")
                text_dim = current_shape[1]
                text_tensor = torch.zeros(num_items + 1, text_dim, dtype=data['text_features'].dtype)
                text_tensor[1:] = data['text_features']  # items 1..num_items
                data['text_features'] = text_tensor
                self.logger.info(f"✅ Added padding item 0 to text_features: {text_tensor.shape}")
            elif current_shape[0] == num_items + 1:
                # 已经包含padding item 0
                self.logger.info(f"✅ text_features already has correct shape: {current_shape}")
            else:
                self.logger.warning(f"⚠️ Unexpected text_features shape: {current_shape}, expected [{num_items}] or [{num_items+1}]")

        # 处理图像特征
        if isinstance(data['image_features'], dict):
            self.logger.info(f"Converting image features to tensor ({num_items} items)...")

            # 创建张量（+1为padding item 0预留位置）
            image_tensor = torch.zeros(num_items + 1, 512, dtype=torch.float16)  # CLIP特征维度

            # 填充特征
            for item_idx, feat in data['image_features'].items():
                # 确保索引在有效范围内
                if 0 <= item_idx <= num_items:
                    image_tensor[item_idx] = feat
                else:
                    self.logger.warning(f"Skipping invalid item_idx {item_idx} (num_items={num_items})")

            data['image_features'] = image_tensor
            # self.logger.info(f"✅ Converted image_features to tensor: {image_tensor.shape}")

        elif isinstance(data['image_features'], torch.Tensor):
            # 如果已经是tensor，检查维度是否正确
            current_shape = data['image_features'].shape
            if current_shape[0] == num_items:
                # 需要添加padding item 0
                self.logger.info(f"Adding padding item 0 to image_features: {current_shape} -> [{num_items+1}, {current_shape[1]}]")
                image_dim = current_shape[1]
                image_tensor = torch.zeros(num_items + 1, image_dim, dtype=data['image_features'].dtype)
                image_tensor[1:] = data['image_features']  # items 1..num_items
                data['image_features'] = image_tensor
                self.logger.info(f"✅ Added padding item 0 to image_features: {image_tensor.shape}")
            elif current_shape[0] == num_items + 1:
                # 已经包含padding item 0
                self.logger.info(f"✅ image_features already has correct shape: {current_shape}")
            else:
                self.logger.warning(f"⚠️ Unexpected image_features shape: {current_shape}, expected [{num_items}] or [{num_items+1}]")

        self._log_memory_usage("After feature tensor conversion")

        return data

def collate_fn_pad(batch):
    """
    自定义collate函数，处理变长序列
    对于基线模型，只使用最后一个物品的特征（目标物品）
    """
    user_ids = []
    item_id_list = []
    text_feat_list = []
    vision_feat_list = []

    for item in batch:
        user_ids.append(item['user_id'])

        # 取最后一个物品作为目标
        item_id_list.append(item['item_indices'][-1])

        # 取最后一个物品的特征
        text_feat_list.append(item['text_feat'][-1])
        vision_feat_list.append(item['vision_feat'][-1])

    return {
        'user_id': torch.stack(user_ids),
        'item_id': torch.stack(item_id_list),
        'text_feat': torch.stack(text_feat_list),
        'vision_feat': torch.stack(vision_feat_list)
    }


class AmazonDataset(Dataset):
    def __init__(self, data: Dict[str, Any], sequence_key:str, feature_type: str = "text", logger: logging.Logger=None):
        """
        初始化数据集

        Args:
            data: 包含所有数据的字典
            feature_type: 特征类型，"text"或"image"或"multimodal"
        """
        self.feature_type = feature_type

        # 获取序列数据
        self.sequences = data.get(sequence_key, {})

        # 获取特征
        self.text_features = data.get('text_features')
        self.image_features = data.get('image_features')

        # 创建用户序列列表
        self.user_ids = list(self.sequences.keys())
        self.num_users = len(self.user_ids)

        self.logger = logger
        self.logger.info(f"Initialized dataset with {self.num_users} users, feature_type={feature_type}")

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        seq = self.sequences[user_id]

        # 获取物品序列
        item_indices = seq['item_indices']

        # 获取文本和图像特征（分开返回，供模型使用）
        text_feat = self.text_features[item_indices]
        vision_feat = self.image_features[item_indices]

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_indices': torch.tensor(item_indices, dtype=torch.long),
            'text_feat': text_feat,
            'vision_feat': vision_feat,
            'ratings': torch.tensor(seq.get('ratings', [0] * len(item_indices)), dtype=torch.float)
        }


def get_dataloader(cache_dir: str,
                   category: str = "Video_Games",
                  feature_type: str = "multimodal",
                  batch_size: int = 32,
                  shuffle: bool = True,
                  num_workers: int = 0,
                  quick_mode: bool = False,
                  logger: logging.Logger=None):
    """
    创建数据加载器

    Args:
        cache_dir: 缓存文件夹
        category: 亚马逊数据集类别
        feature_type: 特征类型，"text"或"image"或"multimodal"
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        quick_mode: 是否快速模式（抽样数据）
        logger: 日志记录器

    Returns:
        DataLoader: 数据加载器
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    data = None
    # Quick模式使用单独的缓存文件
    cache_suffix = "_quick" if quick_mode else ""
    cache_file_name = f"{cache_dir}/{category}{cache_suffix}.pkl"
    if os.path.exists(cache_file_name):
        logger.info(f"Loading cached data from {cache_file_name}")
        with open(cache_file_name, "rb") as f:
            data = pickle.load(f)

    if data is None:
        processor = AmazonBooksProcessor(category=category, quick_mode=quick_mode, logger=logger)
        data = processor.load_dataset_for_experiment(
            test_ratio=0.2,
            val_ratio=0.1,
            add_padding_item=True
        )
        with open(cache_file_name, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    """
       data的格式
       {
            'reviews_df': reviews_df,
            'meta_df': meta_df,
            'user_mapping': user_mapping,
            'item_mapping': item_mapping,
            'num_users': num_users,
            'num_items': num_items,
            'text_features': text_features,
            'image_features': image_features,
            'train_sequences': train_sequences,
            'val_sequences': val_sequences,
            'test_sequences': test_sequences,
            'user_sequences': user_sequences
       }
    """
    # 读取真实物品库规模和用户库规模
    config.item_vocab_size = data['num_items']
    config.user_vocab_size = data['num_users']

    train_dataset = AmazonDataset(data, "train_sequences", feature_type, logger)
    val_dataset = AmazonDataset(data, "val_sequences", feature_type, logger)
    test_dataset = AmazonDataset(data, "test_sequences", feature_type, logger)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_pad,
        pin_memory=torch.cuda.is_available()
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_pad,
        pin_memory=torch.cuda.is_available()
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_pad,
        pin_memory=torch.cuda.is_available()
    )

    return train_dataloader, val_dataloader, test_dataloader


class MockDataset(Dataset):
    """模拟数据集（用于快速测试）"""

    def __init__(self, interactions, item_features, user_histories, indices):
        """
        Args:
            interactions: 交互数据字典
            item_features: 物品特征字典
            user_histories: 用户历史字典
            indices: 数据索引（train/val/test）
        """
        self.interactions = interactions
        self.item_features = item_features
        self.user_histories = user_histories
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 获取真实索引
        real_idx = self.indices[idx]

        user_id = self.interactions['user_id'][real_idx]
        item_id = self.interactions['item_id'][real_idx]

        # 获取物品的多模态特征
        visual_feat = torch.from_numpy(self.item_features['visual'][item_id])
        text_feat = torch.from_numpy(self.item_features['text'][item_id])

        # 获取用户历史
        history = self.user_histories.get(user_id, [])
        if len(history) == 0:
            history = [0]  # 默认历史

        # 截断或填充历史到固定长度
        max_history_len = 20
        if len(history) > max_history_len:
            history = history[-max_history_len:]
        else:
            history = history + [0] * (max_history_len - len(history))

        history_tensor = torch.LongTensor(history)

        # 构建batch
        batch = {
            'user_id': torch.LongTensor([user_id]),
            'item_id': torch.LongTensor([item_id]),
            'text_feat': text_feat,
            'vision_feat': visual_feat,
            'user_history': history_tensor,
        }

        # 如果有音频特征
        if 'audio' in self.item_features:
            audio_feat = torch.from_numpy(self.item_features['audio'][item_id])
            batch['audio_feat'] = audio_feat

        return batch


def get_mock_dataloader(
    data_dir='data/mock',
    batch_size=32,
    shuffle=True,
    num_workers=0
):
    """加载模拟数据集

    Args:
        data_dir: 数据目录
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数

    Returns:
        train_loader, val_loader, test_loader
    """
    from pathlib import Path

    data_path = Path(data_dir)

    # 加载数据
    with open(data_path / 'interactions.pkl', 'rb') as f:
        interactions = pickle.load(f)

    with open(data_path / 'item_features.pkl', 'rb') as f:
        item_features = pickle.load(f)

    with open(data_path / 'user_histories.pkl', 'rb') as f:
        user_histories = pickle.load(f)

    with open(data_path / 'splits.pkl', 'rb') as f:
        splits = pickle.load(f)

    # 创建数据集
    train_dataset = MockDataset(
        interactions, item_features, user_histories, splits['train']
    )
    val_dataset = MockDataset(
        interactions, item_features, user_histories, splits['val']
    )
    test_dataset = MockDataset(
        interactions, item_features, user_histories, splits['test']
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )



# ============================================================================
# Pctx-specific Data Loading
# ============================================================================

class PctxDataset(Dataset):
    """
    专门为 PctxAligned 模型设计的数据集
    使用 PctxTokenizerOfficial 来处理数据
    """
    def __init__(self, sequences: Dict[int, List[int]], tokenizer, logger: logging.Logger = None):
        """
        Args:
            sequences: {user_id: [item1, item2, ...]}
            tokenizer: PctxTokenizerOfficial 实例
            logger: 日志记录器
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.logger = logger or logging.getLogger(__name__)

        # 创建样本列表：每个用户的每个位置都是一个样本
        self.samples = []
        for user_id, item_seq in sequences.items():
            # 每个位置都可以作为一个训练样本
            # 前面的物品作为历史，当前物品作为目标
            for idx in range(len(item_seq)):
                if idx > 0:  # 至少需要一个历史物品
                    history = item_seq[:idx]
                    target = item_seq[idx]
                    self.samples.append({
                        'user_id': user_id,
                        'history': history,
                        'target': target
                    })

        self.logger.info(f"PctxDataset initialized with {len(self.samples)} samples from {len(sequences)} users")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 使用 tokenizer 处理
        tokenized = self.tokenizer.tokenize(
            user_id=sample['user_id'],
            item_sequence=sample['history'],
            target_item=sample['target']
        )

        # 添加 user_id 和 item_id 用于评估
        tokenized['user_id'] = torch.tensor(sample['user_id'], dtype=torch.long)
        tokenized['item_id'] = torch.tensor(sample['target'], dtype=torch.long)

        return tokenized


def extract_user_sequences_from_data(data: Dict[str, Any], sequence_key: str = 'train_sequences') -> Dict[int, List[int]]:
    """
    从数据字典中提取用户序列

    Args:
        data: 包含序列数据的字典
        sequence_key: 序列数据的键名

    Returns:
        {user_id: [item1, item2, ...]}
    """
    sequences_dict = data.get(sequence_key, {})
    user_sequences = {}

    for user_id, seq_data in sequences_dict.items():
        item_indices = seq_data.get('item_indices', [])
        if isinstance(item_indices, torch.Tensor):
            item_indices = item_indices.tolist()
        user_sequences[user_id] = item_indices

    return user_sequences


def get_pctx_dataloader(
    cache_dir: str,
    category: str = "Video_Games",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    quick_mode: bool = False,
    logger: logging.Logger = None,
    tokenizer_path: str = None,
    device: str = "cuda"
) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
    """
    为 PctxAligned 模型创建专用的数据加载器

    Args:
        cache_dir: 缓存目录
        category: 数据集类别
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        quick_mode: 快速模式（使用小数据集）
        logger: 日志记录器
        tokenizer_path: tokenizer 保存路径
        device: 设备

    Returns:
        (train_loader, val_loader, test_loader, tokenizer)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Loading data for PctxAligned...")
    logger.info("=" * 60)

    # 1. 加载数据（使用与 get_dataloader 相同的逻辑）
    data = None
    cache_suffix = "_quick" if quick_mode else ""
    cache_file_name = f"{cache_dir}/{category}{cache_suffix}.pkl"

    if os.path.exists(cache_file_name):
        logger.info(f"Loading cached data from {cache_file_name}")
        with open(cache_file_name, "rb") as f:
            data = pickle.load(f)

    if data is None:
        processor = AmazonBooksProcessor(category=category, quick_mode=quick_mode, logger=logger)
        data = processor.load_dataset_for_experiment(
            test_ratio=0.2,
            val_ratio=0.1,
            add_padding_item=True
        )
        with open(cache_file_name, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. 提取用户序列
    train_sequences = extract_user_sequences_from_data(data, 'train_sequences')
    val_sequences = extract_user_sequences_from_data(data, 'val_sequences')
    test_sequences = extract_user_sequences_from_data(data, 'test_sequences')

    logger.info(f"Extracted sequences:")
    logger.info(f"  Train: {len(train_sequences)} users")
    logger.info(f"  Val: {len(val_sequences)} users")
    logger.info(f"  Test: {len(test_sequences)} users")

    # 3. 构建或加载 tokenizer
    from baseline_models.pctx_tokenizer_official import PctxTokenizerOfficial

    if tokenizer_path and os.path.exists(tokenizer_path):
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = PctxTokenizerOfficial.load(tokenizer_path, device=device)
    else:
        logger.info("Building new tokenizer...")

        # 收集所有唯一的物品ID
        all_items = set()
        for seq in train_sequences.values():
            all_items.update(seq)

        # 创建物品文本（使用简单的 item_id 格式）
        item_texts = {item_id: f"item_{item_id}" for item_id in all_items}

        logger.info(f"Total unique items: {len(item_texts)}")

        # 初始化 tokenizer
        tokenizer = PctxTokenizerOfficial(
            codebook_size=256,
            n_codebooks=3,
            id_length=4,
            max_seq_len=20,
            device=device
        )

        # 构建 semantic IDs
        tokenizer.build_semantic_ids_from_dataset(
            user_sequences=train_sequences,
            item_texts=item_texts,
            save_path=tokenizer_path
        )

    # 4. 创建数据集
    train_dataset = PctxDataset(train_sequences, tokenizer, logger)
    val_dataset = PctxDataset(val_sequences, tokenizer, logger)
    test_dataset = PctxDataset(test_sequences, tokenizer, logger)

    # 5. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pctx_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pctx_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pctx_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    logger.info("PctxAligned dataloaders created successfully!")
    logger.info("=" * 60)

    return train_loader, val_loader, test_loader, tokenizer


def pctx_collate_fn(batch):
    """
    PctxDataset 的 collate 函数
    将多个样本合并成一个 batch，处理变长序列
    """
    # batch 是一个列表，每个元素是 tokenizer.tokenize() 返回的字典
    # 需要将它们合并成一个 batch，并进行 padding

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    user_ids_list = []
    item_ids_list = []

    for item in batch:
        input_ids_list.append(item['input_ids'].squeeze(0))
        attention_mask_list.append(item['attention_mask'].squeeze(0))
        labels_list.append(item['labels'].squeeze(0))
        # 收集 user_id 和 item_id 用于评估
        if 'user_id' in item:
            user_ids_list.append(item['user_id'])
        if 'item_id' in item:
            item_ids_list.append(item['item_id'])

    # 找到最大长度
    max_input_len = max(x.size(0) for x in input_ids_list)
    max_label_len = max(x.size(0) for x in labels_list)

    # Padding
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for input_ids, attention_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
        # Pad input_ids and attention_mask
        input_pad_len = max_input_len - input_ids.size(0)
        if input_pad_len > 0:
            input_ids = torch.cat([input_ids, torch.zeros(input_pad_len, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(input_pad_len, dtype=attention_mask.dtype)])

        # Pad labels (使用 -100 作为 padding，这样在计算 loss 时会被忽略)
        label_pad_len = max_label_len - labels.size(0)
        if label_pad_len > 0:
            labels = torch.cat([labels, torch.full((label_pad_len,), -100, dtype=labels.dtype)])

        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_labels.append(labels)

    result = {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'labels': torch.stack(padded_labels)
    }

    # 添加 user_id 和 item_id（如果存在）
    if user_ids_list:
        result['user_id'] = torch.stack(user_ids_list)
    if item_ids_list:
        result['item_id'] = torch.stack(item_ids_list)

    return result


# ==================== PMAT 专用数据集和 collate 函数 ====================

class PMATDataset(Dataset):
    """
    PMAT推荐模型专用数据集

    支持：
    1. 用户历史序列（用于用户兴趣建模）
    2. 目标物品（正样本）
    3. 负样本采样（用于BPR损失）
    """

    def __init__(
        self,
        data: Dict[str, Any],
        sequence_key: str,
        max_history_len: int = 50,
        num_negative_samples: int = 4,
        logger: logging.Logger = None
    ):
        """
        Args:
            data: 包含所有数据的字典
            sequence_key: 序列数据的键名（train_sequences/val_sequences/test_sequences）
            max_history_len: 最大历史长度
            num_negative_samples: 每个正样本对应的负样本数量
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.max_history_len = max_history_len
        self.num_negative_samples = num_negative_samples

        # 获取序列数据
        self.sequences = data.get(sequence_key, {})

        # 获取特征
        self.text_features = data.get('text_features')
        self.image_features = data.get('image_features')
        self.num_items = data.get('num_items', self.text_features.shape[0])

        # 构建样本列表：每个用户的每个位置都是一个样本
        # 格式：(user_id, history_items, target_item)
        self.samples = []
        self.user_item_set = {}  # 用于负采样时排除用户已交互的物品

        for user_id, seq in self.sequences.items():
            item_indices = seq['item_indices']
            self.user_item_set[user_id] = set(item_indices)

            # 每个位置都可以作为一个训练样本
            # 前面的物品作为历史，当前物品作为目标
            for idx in range(1, len(item_indices)):  # 从1开始，确保至少有1个历史
                history = item_indices[:idx]
                target = item_indices[idx]
                self.samples.append({
                    'user_id': user_id,
                    'history': history,
                    'target': target
                })

        self.logger.info(f"PMATDataset initialized: {len(self.samples)} samples from {len(self.sequences)} users")

    def __len__(self):
        return len(self.samples)

    def _sample_negatives(self, user_id: int, num_samples: int) -> List[int]:
        """采样负样本（用户未交互过的物品）"""
        user_items = self.user_item_set.get(user_id, set())
        negatives = []

        while len(negatives) < num_samples:
            # 随机采样，排除padding item 0
            neg_item = np.random.randint(1, self.num_items)
            if neg_item not in user_items and neg_item not in negatives:
                negatives.append(neg_item)

        return negatives

    def __getitem__(self, idx):
        sample = self.samples[idx]
        user_id = sample['user_id']
        history = sample['history']
        target = sample['target']

        # 截断或填充历史到固定长度
        if len(history) > self.max_history_len:
            history = history[-self.max_history_len:]  # 取最近的历史

        history_len = len(history)

        # 获取历史物品的特征
        history_text_feat = self.text_features[history]
        history_vision_feat = self.image_features[history]

        # 获取目标物品的特征
        target_text_feat = self.text_features[target]
        target_vision_feat = self.image_features[target]

        # 采样负样本
        negative_items = self._sample_negatives(user_id, self.num_negative_samples)
        neg_text_feat = self.text_features[negative_items]
        neg_vision_feat = self.image_features[negative_items]

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'history_items': torch.tensor(history, dtype=torch.long),
            'history_len': torch.tensor(history_len, dtype=torch.long),
            'history_text_feat': history_text_feat,
            'history_vision_feat': history_vision_feat,
            'target_item': torch.tensor(target, dtype=torch.long),
            'target_text_feat': target_text_feat,
            'target_vision_feat': target_vision_feat,
            'negative_items': torch.tensor(negative_items, dtype=torch.long),
            'neg_text_feat': neg_text_feat,
            'neg_vision_feat': neg_vision_feat,
        }


def pmat_collate_fn(batch):
    """
    PMAT数据集的collate函数
    处理变长的用户历史序列，进行padding
    """
    user_ids = []
    history_items_list = []
    history_lens = []
    history_text_feat_list = []
    history_vision_feat_list = []
    target_items = []
    target_text_feat_list = []
    target_vision_feat_list = []
    negative_items_list = []
    neg_text_feat_list = []
    neg_vision_feat_list = []

    # 找到最大历史长度
    max_history_len = max(item['history_len'].item() for item in batch)

    for item in batch:
        user_ids.append(item['user_id'])
        history_lens.append(item['history_len'])
        target_items.append(item['target_item'])
        target_text_feat_list.append(item['target_text_feat'])
        target_vision_feat_list.append(item['target_vision_feat'])
        negative_items_list.append(item['negative_items'])
        neg_text_feat_list.append(item['neg_text_feat'])
        neg_vision_feat_list.append(item['neg_vision_feat'])

        # Padding历史序列
        history_len = item['history_len'].item()
        pad_len = max_history_len - history_len

        # Padding history items
        history_items = item['history_items']
        if pad_len > 0:
            history_items = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),  # 前面padding
                history_items
            ])
        history_items_list.append(history_items)

        # Padding history features
        history_text = item['history_text_feat']
        history_vision = item['history_vision_feat']

        if pad_len > 0:
            text_pad = torch.zeros(pad_len, history_text.shape[-1])
            vision_pad = torch.zeros(pad_len, history_vision.shape[-1])
            history_text = torch.cat([text_pad, history_text], dim=0)
            history_vision = torch.cat([vision_pad, history_vision], dim=0)

        history_text_feat_list.append(history_text)
        history_vision_feat_list.append(history_vision)

    return {
        'user_id': torch.stack(user_ids),
        'history_items': torch.stack(history_items_list),
        'history_len': torch.stack(history_lens),
        'history_text_feat': torch.stack(history_text_feat_list),
        'history_vision_feat': torch.stack(history_vision_feat_list),
        'target_item': torch.stack(target_items),
        'target_text_feat': torch.stack(target_text_feat_list),
        'target_vision_feat': torch.stack(target_vision_feat_list),
        'negative_items': torch.stack(negative_items_list),
        'neg_text_feat': torch.stack(neg_text_feat_list),
        'neg_vision_feat': torch.stack(neg_vision_feat_list),
    }


def get_pmat_dataloader(
    cache_dir: str,
    category: str = "Video_Games",
    batch_size: int = 32,
    max_history_len: int = 50,
    num_negative_samples: int = 4,
    eval_num_negative_samples: int = 99,
    shuffle: bool = True,
    num_workers: int = 0,
    quick_mode: bool = False,
    logger: logging.Logger = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    为PMAT推荐模型创建专用的数据加载器

    Args:
        cache_dir: 缓存目录
        category: 数据集类别
        batch_size: 批次大小
        max_history_len: 最大历史长度
        num_negative_samples: 训练时负样本数量
        eval_num_negative_samples: 评估时负样本数量（更多负样本确保指标准确）
        shuffle: 是否打乱
        num_workers: 工作进程数
        quick_mode: 快速模式
        logger: 日志记录器

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    logger.info("=" * 60)
    logger.info("Creating PMAT Recommendation DataLoaders")
    logger.info("=" * 60)

    # 加载数据
    data = None
    cache_suffix = "_quick" if quick_mode else ""
    cache_file_name = f"{cache_dir}/{category}{cache_suffix}.pkl"

    if os.path.exists(cache_file_name):
        logger.info(f"Loading cached data from {cache_file_name}")
        with open(cache_file_name, "rb") as f:
            data = pickle.load(f)

    if data is None:
        processor = AmazonBooksProcessor(category=category, quick_mode=quick_mode, logger=logger)
        data = processor.load_dataset_for_experiment(
            test_ratio=0.2,
            val_ratio=0.1,
            add_padding_item=True
        )
        with open(cache_file_name, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 更新config
    config.item_vocab_size = data['num_items']
    config.user_vocab_size = data['num_users']

    # 创建数据集
    # 训练集使用较少负样本（加快训练）
    train_dataset = PMATDataset(
        data, "train_sequences",
        max_history_len=max_history_len,
        num_negative_samples=num_negative_samples,
        logger=logger
    )
    # 验证集和测试集使用更多负样本（确保评估指标准确）
    val_dataset = PMATDataset(
        data, "val_sequences",
        max_history_len=max_history_len,
        num_negative_samples=eval_num_negative_samples,
        logger=logger
    )
    test_dataset = PMATDataset(
        data, "test_sequences",
        max_history_len=max_history_len,
        num_negative_samples=eval_num_negative_samples,
        logger=logger
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pmat_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pmat_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pmat_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    logger.info("PMAT dataloaders created successfully!")
    logger.info("=" * 60)

    return train_loader, val_loader, test_loader


# ==================== DGMRec 数据适配器 ====================

class DGMRecDatasetAdapter:
    """
    DGMRec 数据适配器

    将现有数据格式转换为 DGMRec 需要的格式：
    - inter_matrix(): 返回用户-物品交互的稀疏矩阵
    - visual_feat: 视觉特征张量 (n_items, visual_dim)
    - text_feat: 文本特征张量 (n_items, text_dim)
    """

    def __init__(self, data: Dict, logger: logging.Logger = None):
        """
        Args:
            data: 从 get_dataloader 加载的数据字典，包含:
                - num_users: 用户数量
                - num_items: 物品数量
                - text_features: 文本特征张量
                - image_features: 图像特征张量
                - train_sequences / val_sequences / test_sequences: 用户序列
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger("DGMRecAdapter")

        self.n_users = data['num_users']
        # 注意：num_items 可能不包含 padding item，但特征张量包含
        # 我们使用特征张量的大小作为 n_items
        self.visual_feat = data.get('image_features')
        self.text_feat = data.get('text_features')

        # 确保特征是张量格式
        if self.visual_feat is not None and not isinstance(self.visual_feat, torch.Tensor):
            self.visual_feat = torch.tensor(self.visual_feat, dtype=torch.float32)
        if self.text_feat is not None and not isinstance(self.text_feat, torch.Tensor):
            self.text_feat = torch.tensor(self.text_feat, dtype=torch.float32)

        # 使用特征张量的大小作为 n_items（包含 padding item）
        if self.visual_feat is not None:
            self.n_items = self.visual_feat.shape[0]
        elif self.text_feat is not None:
            self.n_items = self.text_feat.shape[0]
        else:
            self.n_items = data['num_items']

        # 构建交互矩阵
        self._interaction_matrix = self._build_interaction_matrix(data)

        self.logger.info(f"DGMRecDatasetAdapter initialized:")
        self.logger.info(f"  - n_users: {self.n_users}, n_items: {self.n_items}")
        self.logger.info(f"  - visual_feat shape: {self.visual_feat.shape if self.visual_feat is not None else None}")
        self.logger.info(f"  - text_feat shape: {self.text_feat.shape if self.text_feat is not None else None}")
        self.logger.info(f"  - interaction_matrix: {self._interaction_matrix.shape}, nnz={self._interaction_matrix.nnz}")

    def _build_interaction_matrix(self, data: Dict):
        """
        从用户序列构建用户-物品交互矩阵

        Returns:
            scipy.sparse.coo_matrix: 交互矩阵 (n_users, n_items)
        """
        import scipy.sparse as sp

        rows = []
        cols = []
        values = []

        # 合并所有序列数据
        all_sequences = {}
        for key in ['train_sequences', 'val_sequences', 'test_sequences', 'user_sequences']:
            if key in data and data[key]:
                for user_id, seq_data in data[key].items():
                    if user_id not in all_sequences:
                        all_sequences[user_id] = set()

                    # 处理不同格式的序列数据
                    if isinstance(seq_data, dict):
                        items = seq_data.get('item_indices', [])
                    elif isinstance(seq_data, (list, tuple)):
                        items = seq_data
                    else:
                        continue

                    all_sequences[user_id].update(items)

        # 构建稀疏矩阵的数据
        for user_id, items in all_sequences.items():
            for item_id in items:
                if 0 <= user_id < self.n_users and 0 <= item_id < self.n_items:
                    rows.append(user_id)
                    cols.append(item_id)
                    values.append(1.0)

        # 创建稀疏矩阵
        interaction_matrix = sp.coo_matrix(
            (values, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )

        self.logger.info(f"Built interaction matrix: {interaction_matrix.shape}, nnz={interaction_matrix.nnz}")

        return interaction_matrix

    def inter_matrix(self, form='coo'):
        """
        返回交互矩阵（兼容 RecBole 接口）

        Args:
            form: 矩阵格式，'coo' 或 'csr'

        Returns:
            scipy.sparse matrix
        """
        if form == 'coo':
            return self._interaction_matrix.tocoo()
        elif form == 'csr':
            return self._interaction_matrix.tocsr()
        else:
            return self._interaction_matrix


class DGMRecBatchDataset(Dataset):
    """
    DGMRec 批次数据集

    为 DGMRec 提供 (user, pos_item, neg_item) 三元组
    """

    def __init__(self, data: Dict, split: str = 'train',
                 num_negatives: int = 1, logger: logging.Logger = None):
        """
        Args:
            data: 数据字典
            split: 'train', 'val', 或 'test'
            num_negatives: 每个正样本的负样本数量
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger("DGMRecBatchDataset")
        self.num_negatives = num_negatives
        self.n_users = data['num_users']
        self.n_items = data['num_items']

        # 获取对应split的序列
        sequence_key = f'{split}_sequences'
        self.sequences = data.get(sequence_key, {})

        # 构建用户已交互物品集合（用于负采样）
        self.user_interacted = {}
        for key in ['train_sequences', 'val_sequences', 'test_sequences']:
            if key in data and data[key]:
                for user_id, seq_data in data[key].items():
                    if user_id not in self.user_interacted:
                        self.user_interacted[user_id] = set()

                    if isinstance(seq_data, dict):
                        items = seq_data.get('item_indices', [])
                    elif isinstance(seq_data, (list, tuple)):
                        items = seq_data
                    else:
                        continue

                    self.user_interacted[user_id].update(items)

        # 构建样本列表: (user_id, pos_item_id)
        self.samples = []
        for user_id, seq_data in self.sequences.items():
            if isinstance(seq_data, dict):
                items = seq_data.get('item_indices', [])
            elif isinstance(seq_data, (list, tuple)):
                items = seq_data
            else:
                continue

            for item_id in items:
                if 0 < item_id < self.n_items:  # 排除 padding item 0
                    self.samples.append((user_id, item_id))

        self.logger.info(f"DGMRecBatchDataset ({split}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _sample_negative(self, user_id: int) -> int:
        """采样一个负样本"""
        interacted = self.user_interacted.get(user_id, set())
        while True:
            neg_item = np.random.randint(1, self.n_items)  # 排除 padding item 0
            if neg_item not in interacted:
                return neg_item

    def __getitem__(self, idx):
        user_id, pos_item = self.samples[idx]
        neg_item = self._sample_negative(user_id)

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'pos_item': torch.tensor(pos_item, dtype=torch.long),
            'neg_item': torch.tensor(neg_item, dtype=torch.long),
        }


def dgmrec_collate_fn(batch):
    """DGMRec 的 collate 函数"""
    users = torch.stack([item['user_id'] for item in batch])
    pos_items = torch.stack([item['pos_item'] for item in batch])
    neg_items = torch.stack([item['neg_item'] for item in batch])

    return users, pos_items, neg_items


def get_dgmrec_dataloader(cache_dir: str,
                          category: str = "Video_Games",
                          batch_size: int = 32,
                          shuffle: bool = True,
                          num_workers: int = 0,
                          quick_mode: bool = False,
                          num_negatives: int = 1,
                          logger: logging.Logger = None):
    """
    创建 DGMRec 专用的数据加载器

    Args:
        cache_dir: 缓存目录
        category: 数据集类别
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        quick_mode: 快速模式
        num_negatives: 负样本数量
        logger: 日志记录器

    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset_adapter)
            - dataset_adapter 包含 DGMRec 需要的交互矩阵和多模态特征
    """
    if logger is None:
        logger = logging.getLogger("DGMRec_DataLoader")

    logger.info("=" * 60)
    logger.info("Creating DGMRec DataLoaders")
    logger.info("=" * 60)

    # 加载数据
    cache_suffix = "_quick" if quick_mode else ""
    cache_file_name = f"{cache_dir}/{category}{cache_suffix}.pkl"

    if os.path.exists(cache_file_name):
        logger.info(f"Loading cached data from {cache_file_name}")
        with open(cache_file_name, "rb") as f:
            data = pickle.load(f)
    else:
        processor = AmazonBooksProcessor(category=category, quick_mode=quick_mode, logger=logger)
        data = processor.load_dataset_for_experiment(
            test_ratio=0.2,
            val_ratio=0.1,
            add_padding_item=True
        )
        with open(cache_file_name, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 更新 config
    config.item_vocab_size = data['num_items']
    config.user_vocab_size = data['num_users']

    # 创建数据适配器（包含交互矩阵和多模态特征）
    dataset_adapter = DGMRecDatasetAdapter(data, logger)

    # 创建批次数据集
    train_dataset = DGMRecBatchDataset(data, 'train', num_negatives, logger)
    val_dataset = DGMRecBatchDataset(data, 'val', num_negatives, logger)
    test_dataset = DGMRecBatchDataset(data, 'test', num_negatives, logger)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dgmrec_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dgmrec_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dgmrec_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    logger.info("DGMRec dataloaders created successfully!")
    logger.info("=" * 60)

    return train_loader, val_loader, test_loader, dataset_adapter
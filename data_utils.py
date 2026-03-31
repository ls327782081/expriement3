import logging
import os
import pickle
import time
import json
import numpy as np
import pandas
import pandas as pd
import torch
import requests
from PIL import Image
from io import BytesIO

from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel


from torch.utils.data import Dataset, DataLoader

from config import config

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
        item_mapping = {item_id: i for i, item_id in enumerate(unique_items)}  # 从0开始

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
        """使用CLIP生成文本特征"""
        self.logger.info("Generating CLIP text features (replacing BERT)...")

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
            # 限制文本长度以适应CLIP (CLIP支持77 tokens，约200字符)
            if len(combined_text) > 200:
                combined_text = combined_text[:200]

            texts_to_process.append(combined_text)
            item_indices.append(item_idx)

        # 批处理生成CLIP文本特征
        text_features = self._extract_clip_text_features_batch(texts_to_process, item_indices, batch_size)

        self.logger.info(f"Generated CLIP text features for {len(text_features)} items")

        # 记录内存使用
        self._log_memory_usage("After CLIP text feature extraction")

        return text_features

    def _extract_clip_text_features_batch(self, texts: List[str], item_indices: List[int], batch_size: int) -> Dict[int, torch.Tensor]:
        """批量提取CLIP文本特征"""
        text_features = {}

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_indices = item_indices[i:i + batch_size]

                # 使用CLIP处理器编码文本
                inputs = self.clip_processor(
                    text=batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=77,  # CLIP最大77 tokens
                    return_tensors='pt'
                )

                # 移动到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 获取CLIP文本特征
                outputs = self.clip_model.get_text_features(**inputs)
                text_embeddings = outputs[0] if isinstance(outputs, tuple) else outputs

                # 存储特征（移动到CPU，使用float16减少内存）
                for j, item_idx in enumerate(batch_indices):
                    text_features[item_idx] = text_embeddings[j].cpu().half()  # float16

                # 输出进度日志
                if (i // batch_size + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} CLIP texts")
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
            text_tensor = torch.zeros(num_items + 1, 512, dtype=torch.float16)  # BERT特征维度

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

# ==================== PMAT 专用数据集和 collate 函数 ====================

class PMATDataset(Dataset):
    """
    PMAT推荐模型专用数据集

    支持：
    1. 用户历史序列（用于用户兴趣建模）
    2. 目标物品（正样本）
    3. 负样本采样（用于BPR损失训练）
    4. Full Ranking评估模式（评估时对所有物品排序）
    """

    def __init__(
            self,
            data: Dict[str, Any],
            sequence_key: str,
            max_history_len: int = 50,
            num_negative_samples: int = 4,
            full_ranking: bool = False,
            indices_list: torch.Tensor = None,
            logger: logging.Logger = None
    ):
        """
        Args:
            data: 包含所有数据的字典
            sequence_key: 序列数据的键名（train_sequences/val_sequences/test_sequences）
            max_history_len: 最大历史长度
            num_negative_samples: 每个正样本对应的负样本数量（仅训练时使用）
            full_ranking: 是否使用Full Ranking评估模式（评估时不采样负样本）
            indices_list: 语义id
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.max_history_len = max_history_len
        self.num_negative_samples = num_negative_samples
        self.full_ranking = full_ranking

        # 获取序列数据
        self.sequences = data.get(sequence_key, {})

        # 获取特征
        self.text_features = data.get('text_features')
        self.image_features = data.get('image_features')
        self.num_items = data.get('num_items', self.text_features.shape[0])
        self.indices_list = indices_list

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

                # ==================== 确认：取最后max_history_len个行为（RecBole标准） ====================
                if len(history) > self.max_history_len:
                    history = history[-self.max_history_len:]  # 保留最近的行为（右对齐数据处理）

                self.samples.append({
                    'user_id': user_id,
                    'history': history,
                    'target': target
                })

        mode_str = "Full Ranking" if full_ranking else f"Sampled ({num_negative_samples} negatives)"
        self.logger.info(
            f"PMATDataset initialized: {len(self.samples)} samples from {len(self.sequences)} users, mode: {mode_str}")

    def __len__(self):
        return len(self.samples)

    def _sample_negatives(self, user_id: int, num_samples: int) -> List[int]:
        """
        采样负样本（用户未交互过的物品）
        优化点：
        1. 先计算可采样的负样本池，避免死循环
        2. 采样数超过可用数量时，给出警告并返回最大可用样本
        3. 保留原有的校验逻辑，保证采样正确性
        """
        # 1. 获取用户已交互物品集合
        user_items = self.user_item_set.get(user_id, set())
        user_items = set(user_items)  # 确保是集合类型，加速查找

        # 2. 计算可采样的负样本池（所有物品 - 已交互物品）
        all_items = set(range(1, self.num_items + 1))  # 全量物品ID集合
        candidate_negatives = list(all_items - user_items)  # 可用负样本列表
        available_neg_num = len(candidate_negatives)

        # 3. 处理采样数超过可用数量的情况（避免死循环核心）
        if num_samples > available_neg_num:
            print(
                f"⚠️ [负采样警告] 用户{user_id}：请求采样{num_samples}个负样本，但仅{available_neg_num}个可用（已交互{len(user_items)}个）")
            # 返回所有可用负样本（不足部分不再补充，避免死循环）
            negatives = candidate_negatives
        else:
            # 4. 从候选池中随机采样（无重复）
            # 用np.random.choice替代while循环，效率更高
            sampled_indices = np.random.choice(
                len(candidate_negatives),
                size=num_samples,
                replace=False  # 不重复采样
            )
            negatives = [candidate_negatives[idx] for idx in sampled_indices]

        # 5. 保留原有的校验逻辑（双重保障）
        invalid = [n for n in negatives if n in user_items]
        if invalid:
            print(f"[负采样验证] 用户{user_id}：交互物品数={len(user_items)}, 采样负样本={negatives}")
            print(f"❌ 用户{user_id}负采样错误：{invalid}在交互列表中！")

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
        if np.any(history == 0):
            raise Exception("商品id应该从1开始")
        history = np.array(history) - 1
        history_text_feat = self.text_features[history]
        history_vision_feat = self.image_features[history]
        history_indices = None
        if self.indices_list is not None:
            history_indices = self.indices_list[history]

        # 获取目标物品的特征
        if target == 0:
            raise Exception("商品id应该从1开始")
        target_text_feat = self.text_features[target - 1]
        target_vision_feat = self.image_features[target - 1]
        target_indices = None
        if self.indices_list is not None:
            target_indices = self.indices_list[target - 1]

        result = {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'history_items': torch.tensor(history, dtype=torch.long),
            'history_len': torch.tensor(history_len, dtype=torch.long),
            'history_text_feat': history_text_feat,
            'history_vision_feat': history_vision_feat,
            'target_item': torch.tensor(target, dtype=torch.long),
            'target_text_feat': target_text_feat,
            'target_vision_feat': target_vision_feat,
            'history_indices': history_indices,
            'target_indices': target_indices,
        }

        # Full Ranking模式下不采样负样本，只返回目标物品ID
        if not self.full_ranking:
            # 训练模式：采样负样本
            negative_items = self._sample_negatives(user_id, self.num_negative_samples)

            if np.any(negative_items == 0):
                raise Exception("商品id应该从1开始")
            negative_items = np.array(negative_items) - 1
            neg_text_feat = self.text_features[negative_items]
            neg_vision_feat = self.image_features[negative_items]
            neg_indices_list = self.indices_list[negative_items]

            result['negative_items'] = torch.tensor(negative_items, dtype=torch.long)
            result['neg_text_feat'] = neg_text_feat
            result['neg_vision_feat'] = neg_vision_feat
            result['neg_indices_list'] = neg_indices_list

        return result


def pmat_collate_fn(batch):
    """
    PMAT数据集的collate函数
    处理变长的用户历史序列，进行padding
    支持训练模式（有负样本）和Full Ranking评估模式（无负样本）
    【核心修改】：从右对齐（前面补0）改为左对齐（后面补0）
    """
    user_ids = []
    history_items_list = []
    history_lens = []
    history_text_feat_list = []
    history_vision_feat_list = []
    target_items = []
    target_text_feat_list = []
    target_vision_feat_list = []
    target_indices=[]
    history_indices_list = []

    # 检查是否是Full Ranking模式（没有负样本）
    has_negatives = 'negative_items' in batch[0]

    if has_negatives:
        negative_items_list = []
        neg_text_feat_list = []
        neg_vision_feat_list = []
        neg_indices_list = []

    # 找到最大历史长度
    max_history_len = max(item['history_len'].item() for item in batch)

    for item in batch:
        user_ids.append(item['user_id'])
        history_lens.append(item['history_len'])
        target_items.append(item['target_item'])
        target_text_feat_list.append(item['target_text_feat'])
        target_vision_feat_list.append(item['target_vision_feat'])
        if 'target_indices' in item and item['target_indices'] is not None:
            target_indices.append(item['target_indices'])

        if has_negatives:
            negative_items_list.append(item['negative_items'])
            neg_text_feat_list.append(item['neg_text_feat'])
            neg_vision_feat_list.append(item['neg_vision_feat'])
            neg_indices_list.append(item['neg_indices_list'])

        # ==================== 核心修改：左对齐Padding ====================
        history_len = item['history_len'].item()
        pad_len = max_history_len - history_len

        # Padding history items（左对齐：有效内容在前，后面补0）
        history_items = item['history_items']
        history_indices = None
        if 'history_indices' in item:
            history_indices = item['history_indices']
        if pad_len > 0:
            history_items = torch.cat([
                history_items,  # 有效内容在前
                torch.zeros(pad_len, dtype=torch.long, device=history_items.device)  # 后面补0（左对齐核心）
            ])
            if history_indices is not None:
                history_indices = torch.cat([
                    history_indices,  # 有效内容在前
                    torch.zeros(pad_len, history_indices.shape[1], dtype=torch.long, device=history_indices.device)  # 后面补0（左对齐核心）
                ])
        history_items_list.append(history_items)
        if history_indices is not None:
            history_indices_list.append(history_indices)

        # Padding history features（左对齐：有效特征在前，后面补0）
        history_text = item['history_text_feat']
        history_vision = item['history_vision_feat']

        if pad_len > 0:
            text_pad = torch.zeros(pad_len, history_text.shape[-1], dtype=history_text.dtype)
            vision_pad = torch.zeros(pad_len, history_vision.shape[-1], dtype=history_vision.dtype)
            history_text = torch.cat([history_text, text_pad], dim=0)  # 有效特征在前
            history_vision = torch.cat([history_vision, vision_pad], dim=0)  # 有效特征在前

        history_text_feat_list.append(history_text)
        history_vision_feat_list.append(history_vision)

    result = {
        'user_id': torch.stack(user_ids),
        'history_items': torch.stack(history_items_list),
        'history_len': torch.stack(history_lens),
        'history_text_feat': torch.stack(history_text_feat_list),
        'history_vision_feat': torch.stack(history_vision_feat_list),
        'target_item': torch.stack(target_items),
        'target_text_feat': torch.stack(target_text_feat_list),
        'target_vision_feat': torch.stack(target_vision_feat_list),
    }
    if len(target_indices) > 0:
        result["target_indices"] = torch.stack(target_indices)
    if len(history_indices_list) > 0:
        result["history_indices"] = torch.stack(history_indices_list)


    if has_negatives:
        result['negative_items'] = torch.stack(negative_items_list)
        result['neg_text_feat'] = torch.stack(neg_text_feat_list)
        result['neg_vision_feat'] = torch.stack(neg_vision_feat_list)
        result['neg_indices_list'] = torch.stack(neg_indices_list)

    return result


def get_pmat_dataloader(
    cache_dir: str,
    category: str = "Video_Games",
    batch_size: int = 32,
    max_history_len: int = 50,
    num_negative_samples: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    quick_mode: bool = False,
    indices_list: torch.Tensor = None,
    logger: logging.Logger = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, torch.Tensor]]:
    """
    为PMAT推荐模型创建专用的数据加载器

    使用 Full Ranking 评估协议：
    - 训练集：使用负采样（BPR损失）
    - 验证集/测试集：Full Ranking模式，对所有物品排序

    Args:
        cache_dir: 缓存目录
        category: 数据集类别
        batch_size: 批次大小
        max_history_len: 最大历史长度
        num_negative_samples: 训练时负样本数量
        shuffle: 是否打乱
        num_workers: 工作进程数
        quick_mode: 快速模式
        indices_list: 语义id
        logger: 日志记录器

    Returns:
        (train_loader, val_loader, test_loader, all_item_features)
        all_item_features: 包含所有物品特征的字典，用于Full Ranking评估
            - 'text': (num_items, text_dim) 文本特征
            - 'visual': (num_items, visual_dim) 视觉特征
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    logger.info("=" * 60)
    logger.info("Creating PMAT Recommendation DataLoaders (Full Ranking Mode)")
    logger.info("=" * 60)

    # 加载数据
    data = None
    cache_suffix = "_quick" if quick_mode else ""
    cache_file_name = f"{cache_dir}/{category}{cache_suffix}.pkl"

    if os.path.exists(cache_file_name):
        logger.info(f"Loading cached data from {cache_file_name}")
        with open(cache_file_name, "rb") as f:
            data = pandas.read_pickle(f)

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

    # 准备所有物品特征（用于Full Ranking评估）
    text_features = data.get('text_features')
    image_features = data.get('image_features')

    # 确保是张量格式
    if not isinstance(text_features, torch.Tensor):
        text_features = torch.tensor(text_features, dtype=torch.float32)
    if not isinstance(image_features, torch.Tensor):
        image_features = torch.tensor(image_features, dtype=torch.float32)

    all_item_features = {
        'text': text_features,
        'visual': image_features,
        'num_items': data['num_items']
    }

    logger.info(f"All item features prepared: {text_features.shape[0]} items")
    logger.info(f"  - Text features: {text_features.shape}")
    logger.info(f"  - Visual features: {image_features.shape}")

    # 创建数据集
    train_dataset = PMATDataset(
        data, "train_sequences",
        max_history_len=max_history_len,
        num_negative_samples=0,
        full_ranking=True,
        indices_list=indices_list,
        logger=logger
    )
    val_dataset = PMATDataset(
        data, "val_sequences",
        max_history_len=max_history_len,
        num_negative_samples=0,
        full_ranking=True,
        indices_list=indices_list,
        logger=logger
    )
    test_dataset = PMATDataset(
        data, "test_sequences",
        max_history_len=max_history_len,
        num_negative_samples=0,  # Full Ranking模式不需要负样本
        full_ranking=True,
        indices_list=indices_list,
        logger=logger
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pmat_collate_fn,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pmat_collate_fn,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pmat_collate_fn,
        pin_memory=False
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    logger.info("Evaluation mode: Full Ranking (against all items)")
    logger.info("PMAT dataloaders created successfully!")
    logger.info("=" * 60)

    return train_loader, val_loader, test_loader, all_item_features



# ==================== 全商品无监督预训练专用 DataLoader ====================
class AllItemPretrainDataset(Dataset):
    """
    全商品无监督预训练专用数据集（第一阶段）
    仅包含：商品ID + 文本特征 + 图像特征
    无用户序列、无负采样、无目标商品，专注于编码所有商品的多模态特征
    """

    def __init__(
            self,
            data: Dict[str, Any],
            logger: logging.Logger = None
    ):
        """
        Args:
            data: 包含所有数据的字典（来自 load_dataset_for_experiment）
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)

        # 核心：提取所有唯一商品ID（排除padding item 0）
        self.all_item_ids = list(range(1, data['num_items'] + 1))  # 1~num_items（0是padding）
        self.num_items = len(self.all_item_ids)

        # 获取商品多模态特征（已包含padding item 0，直接索引即可）
        self.text_features = data.get('text_features')  # shape: (num_items+1, 768)
        self.image_features = data.get('image_features')  # shape: (num_items+1, 512)

        # 验证特征完整性
        self._validate_features(data)

        self.logger.info(
            f"AllItemPretrainDataset initialized: "
            f"{self.num_items} unique items (exclude padding 0) | "
            f"text_feat shape: {self.text_features.shape} | "
            f"image_feat shape: {self.image_features.shape}"
        )

    def _validate_features(self, data):
        """验证特征完整性，确保每个商品都有对应的特征"""
        # 检查特征张量维度
        assert self.text_features.shape[0] == data['num_items'] + 1, \
            f"text_features维度错误：期望 {data['num_items'] + 1}，实际 {self.text_features.shape[0]}"
        assert self.image_features.shape[0] == data['num_items'] + 1, \
            f"image_features维度错误：期望 {data['num_items'] + 1}，实际 {self.image_features.shape[0]}"

        # 检查特征非零率（避免全零特征过多）
        text_non_zero = (self.text_features[1:] != 0).any(dim=1).sum().item()
        image_non_zero = (self.image_features[1:] != 0).any(dim=1).sum().item()

        self.logger.info(
            f"商品特征非零率：文本={text_non_zero / self.num_items:.2%} | 图像={image_non_zero / self.num_items:.2%}")

        if text_non_zero / self.num_items < 0.5:
            self.logger.warning("⚠️ 超过50%的商品文本特征为全零，可能影响预训练效果")
        if image_non_zero / self.num_items < 0.3:
            self.logger.warning("⚠️ 超过70%的商品图像特征为全零，建议检查图像特征提取逻辑")

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        """返回单个商品的ID+多模态特征"""
        item_id = self.all_item_ids[idx]

        # 获取该商品的文本/图像特征（直接从张量索引）
        text_feat = self.text_features[item_id]
        vision_feat = self.image_features[item_id]

        return {
            'item_id': torch.tensor(item_id, dtype=torch.long),  # 商品唯一ID
            'text_feat': text_feat,  # BERT文本特征 (768,)
            'vision_feat': vision_feat,  # CLIP图像特征 (512,)
        }


def all_item_pretrain_collate_fn(batch):
    """
    全商品预训练专用collate函数
    仅合并商品ID、文本特征、图像特征，无复杂padding（所有样本长度一致）
    """
    item_ids = torch.stack([item['item_id'] for item in batch])
    text_feats = torch.stack([item['text_feat'] for item in batch])
    vision_feats = torch.stack([item['vision_feat'] for item in batch])

    return {
        'item_id': item_ids,  # shape: (batch_size,)
        'text_feat': text_feats,  # shape: (batch_size, 768)
        'vision_feat': vision_feats,  # shape: (batch_size, 512)
    }


def get_all_item_pretrain_dataloader(
        cache_dir: str,
        category: str = "Video_Games",
        batch_size: int = 128,  # 预训练可使用更大batch
        shuffle: bool = True,
        num_workers: int = 0,
        quick_mode: bool = False,
        logger: logging.Logger = None
) -> Tuple[DataLoader, Dict[str, torch.Tensor]]:
    """
    第一阶段全商品无监督预训练专用DataLoader
    仅返回所有唯一商品的特征，无用户序列、无负采样、无目标商品

    Args:
        cache_dir: 缓存目录
        category: 数据集类别
        batch_size: 批次大小（预训练建议128/256）
        shuffle: 是否打乱（预训练建议True）
        num_workers: 工作进程数
        quick_mode: 快速模式
        logger: 日志记录器

    Returns:
        (pretrain_loader, all_item_meta)
        - pretrain_loader: 全商品预训练DataLoader
        - all_item_meta: 商品元信息（数量、特征维度等）
    """
    if logger is None:
        logger = logging.getLogger("AllItemPretrain")

    logger.info("=" * 60)
    logger.info("Creating All-Item Unsupervised Pretrain DataLoader")
    logger.info("=" * 60)

    # 加载基础数据（复用现有缓存逻辑）
    data = None
    cache_suffix = "_quick" if quick_mode else ""
    cache_file_name = f"{cache_dir}/{category}{cache_suffix}.pkl"

    if os.path.exists(cache_file_name):
        logger.info(f"Loading cached data from {cache_file_name}")
        with open(cache_file_name, "rb") as f:
            data = pandas.read_pickle(f)
    else:
        processor = AmazonBooksProcessor(category=category, quick_mode=quick_mode, logger=logger)
        data = processor.load_dataset_for_experiment(
            test_ratio=0.2,
            val_ratio=0.1,
            add_padding_item=True
        )
        with open(cache_file_name, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 创建全商品预训练数据集
    pretrain_dataset = AllItemPretrainDataset(data, logger)

    # 创建DataLoader（无复杂collate，仅合并特征）
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=all_item_pretrain_collate_fn,
        pin_memory=False,
        drop_last=True  # 预训练丢弃最后不完整批次，避免维度错误
    )

    # 整理商品元信息（供模型使用）
    all_item_meta = {
        'num_items': pretrain_dataset.num_items,
        'text_feat_dim': pretrain_dataset.text_features.shape[1],
        'vision_feat_dim': pretrain_dataset.image_features.shape[1],
        'text_features': pretrain_dataset.text_features,
        'image_features': pretrain_dataset.image_features
    }

    logger.info(f"✅ All-Item Pretrain DataLoader created:")
    logger.info(f"  - Total items: {pretrain_dataset.num_items}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Total batches: {len(pretrain_loader)}")
    logger.info(f"  - Text feature dim: {all_item_meta['text_feat_dim']}")
    logger.info(f"  - Vision feature dim: {all_item_meta['vision_feat_dim']}")
    logger.info("=" * 60)

    return pretrain_loader, all_item_meta
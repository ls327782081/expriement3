import logging
import os
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
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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
                 data_dir: str,
                 quick_mode: bool = False,
                 min_interactions: int = 5,
                 min_items: int = 5,
                 max_users: Optional[int] = None,
                 max_items: Optional[int] = None,
                 bert_model: str = "bert-base-uncased",
                 clip_model: str = "openai/clip-vit-base-patch32",
                 device: str = "auto",
                 use_cache: bool = True,
                 cache_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        """
        初始化Amazon Books数据集处理器
        
        Args:
            data_dir: 数据目录路径
            quick_mode: 是否使用快速模式（减少数据量）
            min_interactions: 用户最小交互次数
            min_items: 商品最小交互次数
            max_users: 最大用户数
            max_items: 最大商品数
            bert_model: BERT模型名称
            clip_model: CLIP模型名称
            device: 计算设备
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            logger: 日志记录器
            **kwargs: 其他参数
        """
        # 设置日志记录器
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        
        # 基本配置
        self.data_dir = Path(data_dir)
        self.quick_mode = quick_mode
        self.min_interactions = min_interactions
        self.min_items = min_items
        self.max_users = max_users
        self.max_items = max_items
        self.bert_model_name = bert_model
        self.clip_model_name = clip_model
        self.use_cache = use_cache
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.logger.info(f"Using device: {self.device}")
        
        # 初始化缓存管理器
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.cache_manager = None  # 可以根据需要实现缓存管理器
        
        # 初始化预训练模型
        self._init_pretrained_models()
        
        # 其他参数
        self.kwargs = kwargs
        
    def _init_pretrained_models(self):
        """初始化预训练模型"""
        self.logger.info("Initializing pre-trained models...")
        
        # 初始化BERT模型和分词器
        self.logger.info(f"Loading BERT model: {self.bert_model_name}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = AutoModel.from_pretrained(self.bert_model_name).to(self.device)
        self.bert_model.eval()
        
        # 初始化CLIP模型和处理器
        self.logger.info(f"Loading CLIP model: {self.clip_model_name}")
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self.clip_model.eval()
        
        self.logger.info("Pre-trained models initialized successfully")
        
    def _log_memory_usage(self, context: str = ""):
        """记录内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3   # GB
            self.logger.info(f"{context} - GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    
    def load_reviews(self) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
        """加载评论数据"""
        self.logger.info("Loading reviews data...")
        
        # 从HuggingFace加载数据
        self.logger.info("Loading reviews from HuggingFace...")
        review_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            name="raw_review_Books",
            split="full",
            trust_remote_code=True
        )
        
        # 转换为DataFrame
        reviews = []
        for review in review_dataset:
            reviews.append({
                'user_id': review.get('reviewerID', ''),
                'item_id': review.get('asin', ''),
                'rating': review.get('overall', 0),
                'timestamp': review.get('timestamp', 0),
                'title': review.get('title', ''),
                'text': review.get('text', ''),
                'verified_purchase': review.get('verified_purchase', False),
                'helpful_vote': review.get('helpful_vote', 0)
            })
            
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
        df = df.dropna(subset=['user_id', 'item_id', 'rating'])
        
        # 过滤评分范围
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
        
        # 过滤用户和商品的最小交互次数
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= self.min_interactions].index
        valid_items = item_counts[item_counts >= self.min_items].index
        
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
        
        # 限制用户和商品数量
        if self.max_users:
            top_users = user_counts.head(self.max_users).index
            df = df[df['user_id'].isin(top_users)]
            
        if self.max_items:
            top_items = item_counts.head(self.max_items).index
            df = df[df['item_id'].isin(top_items)]
        
        # 创建用户和商品的连续ID映射
        unique_users = df['user_id'].unique()
        unique_items = df['item_id'].unique()
        
        user_mapping = {user_id: i+1 for i, user_id in enumerate(unique_users)}  # 从1开始，0保留给padding
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
    
    def load_meta(self) -> pd.DataFrame:
        """加载商品元数据"""
        self.logger.info("Loading meta data...")
        
        # 从HuggingFace加载数据
        self.logger.info("Loading meta data from HuggingFace...")
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            name="raw_meta_Books",
            split="full",
            trust_remote_code=True
        )
        
        # 转换为DataFrame
        meta_data = []
        for item in meta_dataset:
            meta_data.append({
                'item_id': item.get('asin', ''),
                'title': item.get('title', ''),
                'description': item.get('description', []),
                'features': item.get('features', []),
                'categories': item.get('categories', []),
                'image_url': item.get('imageURLHighRes', [])
            })
            
        df = pd.DataFrame(meta_data)
        self.logger.info(f"Loaded {len(df)} meta items")
        
        return df
    
    def _generate_bert_text_features(self, meta_df: pd.DataFrame, item_mapping: Dict[str, int]) -> Dict[int, torch.Tensor]:
        """使用BERT生成文本特征"""
        self.logger.info("Generating BERT text features...")
        
        text_features = {}
        batch_size = 32  # 增加批处理大小以提高效率
        
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
                    
        return text_features
    
    def _generate_clip_image_features(self, meta_df: pd.DataFrame, item_mapping: Dict[str, int]) -> Dict[int, torch.Tensor]:
        """使用CLIP生成图像特征"""
        self.logger.info("Generating CLIP image features...")
        
        image_features = {}
        batch_size = 8  # 图像处理批次较小
        
        # 准备图像URL数据
        image_urls = []
        item_indices = []
        
        for _, row in meta_df.iterrows():
            item_id = row['item_id']
            if item_id not in item_mapping:
                continue
                
            item_idx = item_mapping[item_id]
            
            # 获取图像URL
            urls = row['image_url']
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
        
        for item_idx in missing_items:
            image_features[item_idx] = torch.zeros(512, dtype=torch.float16)
            
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
                response = requests.get(url, timeout=10)
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
        """加载完整数据集（支持缓存）"""
        self.logger.info("Loading Amazon Books dataset...")
        
        # 尝试从缓存加载
        if self.use_cache and self.cache_manager is not None:
            self.logger.info("🔍 Checking feature cache...")
            cache_config = {
                'quick_mode': self.quick_mode,
                'min_interactions': self.min_interactions,
                'min_items': self.min_items,
                'max_users': self.max_users,
                'max_items': self.max_items,
                'bert_model': self.bert_model_name,
                'clip_model': self.clip_model_name,
            }
            
            cached_data = self.cache_manager.load(cache_config)
            
            if cached_data is not None:
                self.logger.info("✅ Loaded features from cache! Skipping BERT/CLIP extraction.")
                return cached_data
            else:
                self.logger.info("❌ Cache not found. Will extract features and save to cache.")
        
        # 缓存未命中，正常加载数据
        start_time = time.time()
        
        # 加载评论和元数据
        reviews_df, user_mapping, item_mapping = self.load_reviews()
        meta_df = self.load_meta()
        
        # 提取文本特征
        text_features = self._generate_bert_text_features(meta_df, item_mapping)
        
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
        
        # 保存到缓存
        if self.use_cache and self.cache_manager is not None:
            features_to_cache = {
                'text_features': text_features,
                'image_features': image_features,
                'user_mapping': user_mapping,
                'item_mapping': item_mapping,
                'meta_df': meta_df,
                'reviews_df': reviews_df,
            }
            
            metadata = {
                'num_users': num_users,
                'num_items': num_items,
                'num_interactions': len(reviews_df),
                'created_at': datetime.now().isoformat(),
                **cache_config
            }
            
            try:
                self.cache_manager.save(cache_config, features_to_cache, metadata)
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")
                self.logger.warning("Continuing without cache...")
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Dataset loaded successfully in {elapsed_time:.2f} seconds")
        
        return dataset
    
    def load_dataset_for_experiment(
        self,
        build_sequences: bool = True,
        min_seq_len: int = 3,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        add_padding_item: bool = True
    ) -> Dict[str, Any]:
        """
        加载数据集用于实验（包括序列构建、数据分割等）
        
        Args:
            build_sequences: 是否构建用户序列
            min_seq_len: 最小序列长度
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
        if build_sequences:
            self.logger.info("Building user sequences...")
            from util import build_user_sequences, split_user_sequences
            
            user_sequences = build_user_sequences(
                dataset['reviews_df'],
                min_seq_len=min_seq_len
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
            self._validate_data_ranges(data)
        
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
            self.logger.info(f"✅ Converted image_features to tensor: {image_tensor.shape}")
            
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

class BooksDataset(Dataset):
    def __init__(self, data: Dict[str, Any], feature_type: str = "text"):
        """
        初始化数据集
        
        Args:
            data: 包含所有数据的字典
            feature_type: 特征类型，"text"或"image"或"multimodal"
        """
        self.data = data
        self.feature_type = feature_type
        
        # 获取序列数据
        self.sequences = data.get('train_sequences', {})
        
        # 获取特征
        self.text_features = data.get('text_features')
        self.image_features = data.get('image_features')
        
        # 创建用户序列列表
        self.user_ids = list(self.sequences.keys())
        self.num_users = len(self.user_ids)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized dataset with {self.num_users} users, feature_type={feature_type}")
        
    def __len__(self):
        return self.num_users
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        seq = self.sequences[user_id]
        
        # 获取物品序列
        item_indices = seq['item_indices']
        
        # 根据特征类型获取特征
        if self.feature_type == "text":
            features = self.text_features[item_indices]
        elif self.feature_type == "image":
            features = self.image_features[item_indices]
        elif self.feature_type == "multimodal":
            # 多模态：拼接文本和图像特征
            text_feat = self.text_features[item_indices]
            image_feat = self.image_features[item_indices]
            features = torch.cat([text_feat, image_feat], dim=1)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_indices': torch.tensor(item_indices, dtype=torch.long),
            'features': features,
            'ratings': torch.tensor(seq.get('ratings', [0] * len(item_indices)), dtype=torch.float)
        }


def get_dataloader(data: Dict[str, Any], 
                  feature_type: str = "text",
                  batch_size: int = 32,
                  shuffle: bool = True,
                  num_workers: int = 0,
                  logger=None):
    """
    创建数据加载器
    
    Args:
        data: 包含所有数据的字典
        feature_type: 特征类型，"text"或"image"或"multimodal"
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        logger: 日志记录器
        
    Returns:
        DataLoader: 数据加载器
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")
        
    dataset = BooksDataset(data, feature_type)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


# 初始化数据处理器（首次运行执行）
if __name__ == "__main__" and not os.path.exists("./data/train.pkl"):
    logger = logging.getLogger("PMAT_Experiment")
    processor = AmazonBooksProcessor(category="Video_Games", logger=logger)
    processor.run()
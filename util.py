import os
import json
import logging
import pickle
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from config import config


def item_id_to_semantic_id(item_ids, id_length, codebook_size, seed=42):
    """
    Convert item IDs to semantic ID sequences
    Args:
        item_ids: Item ID tensor
        id_length: ID length
        codebook_size: Codebook size
        seed: Random seed
    Returns:
        semantic_ids: Semantic ID sequence (batch_size, id_length)
    """
    torch.manual_seed(seed)
    batch_size = item_ids.size(0)
    
    # Generate pseudo-random mapping
    semantic_ids = torch.zeros(batch_size, id_length, dtype=torch.long)
    for i in range(batch_size):
        for j in range(id_length):
            # Use item ID and position as part of the seed to ensure the same mapping
            pseudo_seed = (item_ids[i].item() * 1000 + j) % (2**32)
            torch.manual_seed(pseudo_seed)
            semantic_ids[i, j] = torch.randint(0, codebook_size, (1,))
    
    return semantic_ids


def save_checkpoint(model, optimizer, epoch, loss, experiment_name, is_best=False, logger=None):
    """
    Save model checkpoint
    Args:
        model: Model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        experiment_name: Experiment name
        is_best: Whether it's the best model
        logger: Logger
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(config.checkpoint_dir, experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save current checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        if logger:
            logger.info(f"Saved best model: {best_path}")
    
    if logger:
        logger.info(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(model, optimizer, experiment_name, logger=None):
    """
    Load model checkpoint
    Args:
        model: Model
        optimizer: Optimizer
        experiment_name: Experiment name
        logger: Logger
    Returns:
        model, optimizer, start_epoch, best_loss
    """
    checkpoint_dir = os.path.join(config.checkpoint_dir, experiment_name)
    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    if os.path.exists(best_path):
        try:
            checkpoint = torch.load(best_path, map_location=config.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["loss"]
            if logger:
                logger.info(f"Loaded checkpoint: {best_path}, continuing from epoch {start_epoch}")
            return model, optimizer, start_epoch, best_loss
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load checkpoint: {e}, starting from scratch")
            # If loading fails, start from scratch
            return model, optimizer, 0, float('inf')
    else:
        if logger:
            logger.info("No checkpoint found, starting from scratch")
        return model, optimizer, 0, float('inf')


def save_results(results, experiment_type, logger=None):
    """
    Save experimental results
    Args:
        results: Results list
        experiment_type: Type of experiment
        logger: Logger
    """
    # Create results directory
    os.makedirs(config.result_dir, exist_ok=True)
    
    # Generate result filename
    timestamp = torch.tensor(int(torch.rand(1) * 1000000)).item()  # Use random number as timestamp
    result_file = os.path.join(config.result_dir, f"{experiment_type}_results_{timestamp}.json")
    
    # Convert results to JSON serializable format
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if torch.is_tensor(value):
                serializable_result[key] = value.item() if value.numel() == 1 else value.tolist()
            elif isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_result[key] = value.item()
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    # Save results
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    if logger:
        logger.info(f"Results saved to: {result_file}")


def filter_cold_start_data(
        reviews_df: pd.DataFrame,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    复刻RecBole的冷启动过滤逻辑（迭代过滤直到稳定）
    先过滤低交互用户/物品，再构建序列，这是提升序列质量的核心

    Args:
        reviews_df: 原始评论数据
        min_user_interactions: 用户最小交互数（和RecBole保持一致）
        min_item_interactions: 物品最小交互数（和RecBole保持一致）
        logger: 日志对象

    Returns:
        过滤后的干净数据
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("开始过滤冷启动用户/物品（复刻RecBole逻辑）")
    logger.info(
        f"过滤前 - 交互数: {len(reviews_df)}, 用户数: {reviews_df['user_id'].nunique()}, 物品数: {reviews_df['item_id'].nunique()}")
    logger.info(f"过滤阈值 - 用户最小交互数: {min_user_interactions}, 物品最小交互数: {min_item_interactions}")

    prev_len = 0
    iteration = 0
    max_iterations = 5  # 限制最大迭代次数，避免无限循环
    filtered_df = reviews_df.copy()

    # 迭代过滤：直到交互数稳定或达到最大迭代次数
    while len(filtered_df) != prev_len and iteration < max_iterations:
        prev_len = len(filtered_df)
        iteration += 1

        # 统计当前用户/物品的交互数
        user_counts = filtered_df['user_id'].value_counts()
        item_counts = filtered_df['item_id'].value_counts()

        # 筛选有效用户/物品
        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_interactions].index

        # 过滤数据
        filtered_df = filtered_df[
            filtered_df['user_id'].isin(valid_users) &
            filtered_df['item_id'].isin(valid_items)
            ]

        logger.info(
            f"  迭代{iteration} - 交互数: {len(filtered_df)}, 用户数: {len(valid_users)}, 物品数: {len(valid_items)}")

    logger.info(
        f"过滤后 - 交互数: {len(filtered_df)}, 用户数: {filtered_df['user_id'].nunique()}, 物品数: {filtered_df['item_id'].nunique()}")
    logger.info("=" * 80)

    return filtered_df


def remap_ids(
        reviews_df: pd.DataFrame,
        logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    重新映射ID（可选但建议做）：让ID从1开始，和RecBole保持一致
    避免padding=0和真实ID冲突

    Args:
        reviews_df: 过滤后的评论数据
        logger: 日志对象

    Returns:
        重映射后的数据 + 用户映射表 + 物品映射表
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # 用户ID从1开始映射
    unique_users = sorted(reviews_df['user_id'].unique())
    user_mapping = {old_uid: new_uid for new_uid, old_uid in enumerate(unique_users, start=1)}

    # 物品ID从1开始映射
    unique_items = sorted(reviews_df['item_id'].unique())
    item_mapping = {old_iid: new_iid for new_iid, old_iid in enumerate(unique_items, start=1)}

    # 应用映射
    reviews_df['user_id'] = reviews_df['user_id'].map(user_mapping)
    reviews_df['item_id'] = reviews_df['item_id'].map(item_mapping)

    logger.info(f"ID重映射完成 - 新用户ID范围: 1~{len(user_mapping)}, 新物品ID范围: 1~{len(item_mapping)}")

    return reviews_df, user_mapping, item_mapping

# User sequence building and splitting functions
def build_user_sequences(
        reviews_df: pd.DataFrame,
        min_seq_len: int = 3,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        remap_id: bool = True,
        logger: Optional[logging.Logger] = None
) -> Dict[int, Dict[str, Any]]:
    """
    构建高质量用户序列（先过滤冷启动数据，再构建序列）
    核心修改：复刻RecBole的过滤逻辑，从源头提升序列质量

    Args:
        reviews_df: 原始评论数据（user_id, item_id, timestamp等）
        min_seq_len: 最小序列长度（建议≥3，和RecBole对齐）
        min_user_interactions: 用户最小交互数（和RecBole保持一致）
        min_item_interactions: 物品最小交互数（和RecBole保持一致）
        remap_id: 是否重映射ID（建议开启，避免padding冲突）
        logger: 日志对象

    Returns:
        高质量用户序列字典：key=user_id，value=包含序列信息的字典
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # ========== 步骤1：先过滤冷启动数据（核心！提升序列质量） ==========
    filtered_df = filter_cold_start_data(
        reviews_df=reviews_df,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        logger=logger
    )

    # ========== 步骤2：可选 - 重映射ID（和RecBole对齐） ==========
    if remap_id:
        filtered_df, user_mapping, item_mapping = remap_ids(filtered_df, logger)

    # ========== 步骤3：构建用户序列（保留你原有逻辑） ==========
    logger.info(f"\n开始构建用户序列（最小序列长度: {min_seq_len}）")

    # 按用户分组
    grouped = filtered_df.groupby('user_id')
    user_sequences = {}

    # 统计序列质量
    total_valid_items = 0
    cold_items_in_seq = 0
    # 先预计算过滤后物品的交互数（用于统计序列质量）
    item_interaction_counts = filtered_df['item_id'].value_counts()

    for user_id, group in grouped:
        # 严格按时间排序（和RecBole一致）
        group = group.sort_values("timestamp", ascending=True)

        # 提取序列信息
        item_ids = group['item_id'].tolist()
        ratings = group['rating'].tolist() if 'rating' in group.columns else [0] * len(item_ids)
        timestamps = group['timestamp'].tolist()

        # 过滤短序列
        if len(item_ids) < min_seq_len:
            continue

        # 统计当前序列中的冷门物品（可选，用于监控）
        for item_id in item_ids:
            total_valid_items += 1
            if item_interaction_counts[item_id] < min_item_interactions:
                cold_items_in_seq += 1

        # 构建序列字典（保留你原有字段）
        user_sequences[user_id] = {
            'item_indices': item_ids,
            'ratings': ratings,
            'timestamps': timestamps,
            'length': len(item_ids)
        }

    # ========== 步骤4：输出序列质量统计（验证效果） ==========
    logger.info("=" * 80)
    logger.info("序列构建完成 - 质量统计")
    logger.info(f"  有效用户数: {len(user_sequences)}")
    logger.info(f"  总序列物品数: {total_valid_items}")
    logger.info(f"  序列中冷门物品数（<{min_item_interactions}次交互）: {cold_items_in_seq}")
    logger.info(
        f"  序列中冷门物品占比: {cold_items_in_seq / total_valid_items * 100:.2f}%" if total_valid_items > 0 else "  无有效物品")
    # 计算平均序列长度
    avg_seq_len = sum([v['length'] for v in user_sequences.values()]) / len(user_sequences) if user_sequences else 0
    logger.info(f"  平均序列长度: {avg_seq_len:.2f}")
    logger.info("=" * 80)

    return user_sequences


def split_user_sequences(
    user_sequences: Dict[int, Dict[str, Any]],
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Split user sequences into training, validation, and test sets
    
    Args:
        user_sequences: User sequence dictionary
        test_ratio: Test set ratio
        val_ratio: Validation set ratio
        random_seed: Random seed
        
    Returns:
        Tuple[Dict, Dict, Dict]: (train_sequences, val_sequences, test_sequences)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Splitting user sequences - test_ratio={test_ratio}, val_ratio={val_ratio}...")

    user_sequences_length = len(user_sequences)
    np.random.seed(random_seed)

    # 2. Generate shuffled indices
    indices = list(user_sequences.keys())
    np.random.shuffle(indices)

    # 3. Calculate split boundaries
    test_size = int(user_sequences_length * test_ratio)
    val_size = int(user_sequences_length * val_ratio)
    train_size = user_sequences_length - test_size - val_size

    # 4. Split indices
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_sequences = {}
    val_sequences = {}
    test_sequences = {}
    # 5. Split dictionary by indices
    for user_id, seq in user_sequences.items():
        items = seq['item_indices']
        ratings = seq['ratings']
        timestamps = seq['timestamps']
        length = seq['length']
        if user_id in train_idx:
            train_sequences[user_id] = {
                'item_indices': items,
                'ratings': ratings,
                'timestamps': timestamps,
                'length': length
            }
        elif user_id in val_idx:
            val_sequences[user_id] = {
                'item_indices': items,
                'ratings': ratings,
                'timestamps': timestamps,
                'length': length
            }
        elif user_id in test_idx:
            test_sequences[user_id] = {
                'item_indices': items,
                'ratings': ratings,
                'timestamps': timestamps,
                'length': length
            }
    
    logger.info(f"Split sequences - Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
    
    return train_sequences, val_sequences, test_sequences
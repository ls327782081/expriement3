import os
import json
import logging
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from config import config
from typing import Dict, List, Tuple, Any


# 检查点保存/加载
def save_checkpoint(model, optimizer, epoch, loss, experiment_name, is_best=False, logger=None):
    """保存检查点（避免实验中断）"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")
        
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"{experiment_name}_epoch_{epoch}.pth"
    )
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config.__dict__
    }
    torch.save(checkpoint, checkpoint_path)
    # 保存最优模型
    if is_best:
        torch.save(checkpoint, os.path.join(config.checkpoint_dir, f"{experiment_name}_best.pth"))
    logger.info(f"检查点已保存：{checkpoint_path}")


def load_checkpoint(model, optimizer, experiment_name, epoch=None, logger=None):
    """加载检查点"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")
        
    if epoch is None:
        checkpoint_path = os.path.join(config.checkpoint_dir, f"{experiment_name}_best.pth")
    else:
        checkpoint_path = os.path.join(config.checkpoint_dir, f"{experiment_name}_epoch_{epoch}.pth")

    if not os.path.exists(checkpoint_path):
        logger.info("检查点不存在，从头训练")
        return model, optimizer, 0, float("inf")

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["loss"]
    logger.info(f"加载检查点成功：从第{start_epoch}轮继续训练")
    return model, optimizer, start_epoch, best_loss


# 实验结果保存
def save_results(results, experiment_type, logger=None):
    """保存实验结果（csv/json）"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")
        
    os.makedirs(config.result_dir, exist_ok=True)

    # 保存为CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(config.result_dir, f"{experiment_type}_results.csv")
    df.to_csv(csv_path, index=False)

    # 保存为JSON
    json_path = os.path.join(config.result_dir, f"{experiment_type}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # 可视化结果
    plot_results(df, experiment_type, logger=logger)
    logger.info(f"实验结果已保存：{csv_path} | {json_path}")


def plot_results(df, experiment_type, logger=None):
    """可视化实验结果"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")
        
    # 1. Top-10指标对比
    metrics = ["Precision@10", "Recall@10", "NDCG@10", "MRR@10"]
    plt.figure(figsize=(12, 8))
    for metric in metrics:
        sns.barplot(x="model", y=metric, data=df, label=metric)
    plt.title(f"{experiment_type} - Top-10 Metrics")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.result_dir, f"{experiment_type}_top10_metrics.png"))

    # 2. 超参实验热力图（仅超参实验）
    if experiment_type == "hyper_param":
        plt.figure(figsize=(10, 8))
        pivot_df = df.pivot(index="id_length", columns="lr", values="NDCG@10")
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu")
        plt.title("Hyper-Param Search - NDCG@10")
        plt.tight_layout()
        plt.savefig(os.path.join(config.result_dir, "hyper_param_heatmap.png"))


# 用户序列构建和分割函数
def build_user_sequences(
    reviews_df: pd.DataFrame,
    min_seq_len: int = 3,
    sort_by_time: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    构建用户交互序列
    
    Args:
        reviews_df: 评论数据框，包含user_id, item_id, rating, timestamp等字段
        min_seq_len: 最小序列长度
        sort_by_time: 是否按时间排序
        
    Returns:
        Dict[int, Dict[str, Any]]: 用户序列字典，键为user_id，值为包含序列信息的字典
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Building user sequences with min_seq_len={min_seq_len}...")
    
    # 如果需要按时间排序
    if sort_by_time and 'timestamp' in reviews_df.columns:
        reviews_df = reviews_df.sort_values(['user_id', 'timestamp'])
    
    # 按用户分组
    grouped = reviews_df.groupby('user_id')
    
    user_sequences = {}
    
    for user_id, group in grouped:
        # 获取物品ID和评分
        item_ids = group['item_id'].tolist()
        ratings = group['rating'].tolist() if 'rating' in group.columns else [0] * len(item_ids)
        timestamps = group['timestamp'].tolist() if 'timestamp' in group.columns else [0] * len(item_ids)
        
        # 过滤短序列
        if len(item_ids) < min_seq_len:
            continue
            
        # 构建序列字典
        user_sequences[user_id] = {
            'item_indices': item_ids,
            'ratings': ratings,
            'timestamps': timestamps,
            'length': len(item_ids)
        }
    
    logger.info(f"Built sequences for {len(user_sequences)} users")
    
    return user_sequences


def split_user_sequences(
    user_sequences: Dict[int, Dict[str, Any]],
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    min_seq_len: int = 3
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    分割用户序列为训练、验证和测试集
    
    Args:
        user_sequences: 用户序列字典
        test_ratio: 测试集比例
        val_ratio: 验证集比例
        min_seq_len: 最小序列长度
        
    Returns:
        Tuple[Dict, Dict, Dict]: (train_sequences, val_sequences, test_sequences)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Splitting user sequences - test_ratio={test_ratio}, val_ratio={val_ratio}...")
    
    train_sequences = {}
    val_sequences = {}
    test_sequences = {}
    
    for user_id, seq in user_sequences.items():
        items = seq['item_indices']
        ratings = seq['ratings']
        timestamps = seq['timestamps']
        length = seq['length']
        
        # 确保序列足够长
        if length < min_seq_len:
            continue
            
        # 计算分割点
        test_size = max(1, int(length * test_ratio))
        val_size = max(1, int(length * val_ratio))
        train_size = length - test_size - val_size
        
        # 确保训练集不为空
        if train_size < 1:
            train_size = 1
            val_size = max(0, length - test_size - train_size)
        
        # 分割序列
        train_items = items[:train_size]
        val_items = items[train_size:train_size+val_size] if val_size > 0 else []
        test_items = items[train_size+val_size:] if test_size > 0 else []
        
        train_ratings = ratings[:train_size]
        val_ratings = ratings[train_size:train_size+val_size] if val_size > 0 else []
        test_ratings = ratings[train_size+val_size:] if test_size > 0 else []
        
        train_timestamps = timestamps[:train_size]
        val_timestamps = timestamps[train_size:train_size+val_size] if val_size > 0 else []
        test_timestamps = timestamps[train_size+val_size:] if test_size > 0 else []
        
        # 添加到相应的字典
        if len(train_items) > 0:
            train_sequences[user_id] = {
                'item_indices': train_items,
                'ratings': train_ratings,
                'timestamps': train_timestamps,
                'length': len(train_items)
            }
        
        if len(val_items) > 0:
            val_sequences[user_id] = {
                'item_indices': val_items,
                'ratings': val_ratings,
                'timestamps': val_timestamps,
                'length': len(val_items)
            }
        
        if len(test_items) > 0:
            test_sequences[user_id] = {
                'item_indices': test_items,
                'ratings': test_ratings,
                'timestamps': test_timestamps,
                'length': len(test_items)
            }
    
    logger.info(f"Split sequences - Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
    
    return train_sequences, val_sequences, test_sequences
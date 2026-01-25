import os
import json
import logging
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import torch
import os
import logging
from config import config
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json


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


def semantic_id_to_item_id(semantic_ids, num_items, seed=42):
    """
    Convert semantic ID sequences back to item IDs
    Args:
        semantic_ids: Semantic ID sequence (batch_size, id_length)
        num_items: Number of items
        seed: Random seed
    Returns:
        item_ids: Item ID tensor
    """
    torch.manual_seed(seed)
    batch_size = semantic_ids.size(0)
    item_ids = torch.zeros(batch_size, dtype=torch.long)
    
    for i in range(batch_size):
        # Simple mapping: convert semantic ID sequence to a number
        id_sum = 0
        for j in range(semantic_ids.size(1)):
            id_sum += semantic_ids[i, j].item() * (j + 1)  # Weight by position
        item_ids[i] = id_sum % num_items
    
    return item_ids


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


# User sequence building and splitting functions
def build_user_sequences(
    reviews_df: pd.DataFrame,
    min_seq_len: int = 3,
    logger: Optional[logging.Logger] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Build user interaction sequences
    
    Args:
        reviews_df: Review dataframe, containing user_id, item_id, rating, timestamp etc.
        min_seq_len: Minimum sequence length
        sort_by_time: Whether to sort by time
        
    Returns:
        Dict[int, Dict[str, Any]]: User sequence dictionary, keys are user_id, values are dictionaries containing sequence information
    """
    logger.info(f"Building user sequences with min_seq_len={min_seq_len}...")

    
    # Group by user
    grouped = reviews_df.groupby('user_id')
    
    user_sequences = {}
    
    for user_id, group in grouped:
        # Sort by timestamp
        group = group.sort_values("timestamp")
        # Get item IDs and ratings
        item_ids = group['item_id'].tolist()
        ratings = group['rating'].tolist() if 'rating' in group.columns else [0] * len(item_ids)

        # Filter short sequences
        if len(item_ids) < min_seq_len:
            continue
            
        # Build sequence dictionary
        user_sequences[user_id] = {
            'item_indices': item_ids,
            'ratings': ratings,
            'timestamps': group["timestamp"].tolist(),
            'length': len(item_ids)
        }
    
    logger.info(f"Built sequences for {len(user_sequences)} users")
    
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
import logging
import torch
import numpy as np


def calculate_metrics(user_ids, predictions, ground_truth, k_list=[5, 10, 20], logger=None):
    """计算推荐系统指标"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")
        
    # 确保predictions和ground_truth是列表形式，即使输入是单个值
    if not isinstance(predictions[0], (list, np.ndarray)):
        predictions = [[p] for p in predictions]
    if not isinstance(ground_truth[0], (list, np.ndarray)):
        ground_truth = [[gt] for gt in ground_truth]
        
    metrics = {}
    for k in k_list:
        precision = precision_at_k(predictions, ground_truth, k)
        recall = recall_at_k(predictions, ground_truth, k)
        ndcg = ndcg_at_k(predictions, ground_truth, k)
        mrr = mrr_at_k(predictions, ground_truth, k)
        metrics[f"Precision@{k}"] = precision
        metrics[f"Recall@{k}"] = recall
        metrics[f"NDCG@{k}"] = ndcg
        metrics[f"MRR@{k}"] = mrr

    # 计算MAP和Coverage
    map_score = mean_average_precision(predictions, ground_truth)
    coverage = coverage_at_k(predictions, ground_truth, k_list[-1])
    metrics["MAP"] = map_score
    metrics[f"Coverage@{k_list[-1]}"] = coverage

    return metrics


def precision_at_k(predictions, ground_truth, k):
    """计算Precision@K"""
    precisions = []
    for pred, gt in zip(predictions, ground_truth):
        if len(gt) == 0:
            continue
        pred_k = pred[:k]
        hits = len(set(pred_k) & set(gt))
        precisions.append(hits / k)
    return sum(precisions) / len(precisions) if precisions else 0


def recall_at_k(predictions, ground_truth, k):
    """计算Recall@K"""
    recalls = []
    for pred, gt in zip(predictions, ground_truth):
        if len(gt) == 0:
            continue
        pred_k = pred[:k]
        hits = len(set(pred_k) & set(gt))
        recalls.append(hits / len(gt))
    return sum(recalls) / len(recalls) if recalls else 0


def ndcg_at_k(predictions, ground_truth, k):
    """计算NDCG@K"""
    ndcgs = []
    for pred, gt in zip(predictions, ground_truth):
        if len(gt) == 0:
            continue
        pred_k = pred[:k]
        dcg = 0
        idcg = 0
        # 计算DCG
        for i, item in enumerate(pred_k):
            if item in gt:
                dcg += 1 / np.log2(i + 2)
        # 计算IDCG
        for i in range(min(k, len(gt))):
            idcg += 1 / np.log2(i + 2)
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0


def mrr_at_k(predictions, ground_truth, k):
    """计算MRR@K"""
    mrrs = []
    for pred, gt in zip(predictions, ground_truth):
        if len(gt) == 0:
            continue
        pred_k = pred[:k]
        for i, item in enumerate(pred_k):
            if item in gt:
                mrrs.append(1 / (i + 1))
                break
        else:
            mrrs.append(0)
    return sum(mrrs) / len(mrrs) if mrrs else 0


def mean_average_precision(predictions, ground_truth):
    """计算MAP"""
    aps = []
    for pred, gt in zip(predictions, ground_truth):
        if len(gt) == 0:
            continue
        hits = 0
        precision_sum = 0
        for i, item in enumerate(pred):
            if item in gt:
                hits += 1
                precision_sum += hits / (i + 1)
        aps.append(precision_sum / len(gt) if len(gt) > 0 else 0)
    return sum(aps) / len(aps) if aps else 0


def coverage_at_k(predictions, ground_truth, k):
    """计算Coverage@K"""
    all_items = set()
    recommended_items = set()
    for pred, gt in zip(predictions, ground_truth):
        all_items.update(gt)
        recommended_items.update(pred[:k])
    return len(recommended_items) / len(all_items) if all_items else 0
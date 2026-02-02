import logging
import torch
import numpy as np


def calculate_metrics(user_ids, predictions, ground_truth, k_list=[1, 5, 10, 20], logger=None, scores=None):
    """计算推荐系统指标

    Args:
        user_ids: 用户ID列表
        predictions: 排序后的物品ID列表，如 [[item1, item2, item3, ...], ...]
        ground_truth: 真实相关的物品ID列表，如 [[item1, item5], ...]
        k_list: 要计算的K值列表，默认[1, 5, 10, 20]
        logger: 日志记录器
        scores: 可选，预测分数列表，形状与predictions相同，用于计算AUC

    Returns:
        metrics: 包含以下指标的字典
            - Precision@K: 精确率
            - Recall@K: 召回率（与HR@K相同，当每个样本只有一个正样本时）
            - HR@K: 命中率（Hit Rate）
            - NDCG@K: 归一化折损累积增益
            - MRR@K: 截断到K的平均倒数排名
            - MRR: 全局平均倒数排名（不截断）
            - MAP: 平均精度均值
            - Coverage@K: 推荐物品覆盖率
            - AUC: ROC曲线下面积（如果提供了scores）
    """
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
        metrics[f"HR@{k}"] = recall  # HR@K与Recall@K相同（当每个样本只有一个正样本时）
        metrics[f"NDCG@{k}"] = ndcg
        metrics[f"MRR@{k}"] = mrr

    # 计算全局MRR（不截断K）
    mrr_global = mrr_global_score(predictions, ground_truth)
    metrics["MRR"] = mrr_global

    # 计算MAP和Coverage
    map_score = mean_average_precision(predictions, ground_truth)
    coverage = coverage_at_k(predictions, ground_truth, k_list[-1])
    metrics["MAP"] = map_score
    metrics[f"Coverage@{k_list[-1]}"] = coverage

    # 如果提供了分数，计算AUC
    if scores is not None:
        auc = auc_score(scores, ground_truth)
        metrics["AUC"] = auc

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
    """计算MRR@K（截断到前K个预测）"""
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


def mrr_global_score(predictions, ground_truth):
    """计算全局MRR（不截断K，遍历整个预测列表）

    与MCRL/PMAT中的MRR计算保持一致
    """
    mrrs = []
    for pred, gt in zip(predictions, ground_truth):
        if len(gt) == 0:
            continue
        # 遍历整个预测列表，找到第一个命中的位置
        for i, item in enumerate(pred):
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


def coverage_at_k(predictions, ground_truth, k, num_total_items=None):
    """计算Coverage@K

    Args:
        predictions: 预测的物品ID列表
        ground_truth: 真实相关的物品ID列表
        k: 截断位置
        num_total_items: 总物品数量。如果为None，则从predictions中推断

    Returns:
        Coverage@K: 推荐物品覆盖率
    """
    recommended_items = set()
    for pred, gt in zip(predictions, ground_truth):
        recommended_items.update(pred[:k])

    # 如果没有指定总物品数，从predictions中推断
    if num_total_items is None:
        all_items = set()
        for pred in predictions:
            all_items.update(pred)
        num_total_items = len(all_items)

    return len(recommended_items) / num_total_items if num_total_items > 0 else 0


def auc_score(pos_scores, neg_scores):
    """计算AUC - 与MCRL/PMAT中的计算方式一致

    计算正样本分数大于负样本分数的比例

    Args:
        pos_scores: 正样本分数，可以是以下格式之一：
            - list/array of floats: 每个样本的正样本分数
            - list of lists: 每个样本的正样本分数列表（取第一个）
        neg_scores: 负样本分数，可以是以下格式之一：
            - list/array of shape (num_samples, num_neg): 每个样本的负样本分数
            - list of lists: 每个样本的负样本分数列表

    Returns:
        AUC值：正样本分数大于负样本分数的比例
    """
    if pos_scores is None or neg_scores is None:
        return 0.0

    # 转换为numpy数组
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    # 如果pos_scores是1D，扩展为2D以便广播
    if pos_scores.ndim == 1:
        pos_scores = pos_scores.reshape(-1, 1)

    # 如果neg_scores是1D，说明只有一个负样本
    if neg_scores.ndim == 1:
        neg_scores = neg_scores.reshape(-1, 1)

    # 计算正样本分数大于负样本分数的比例
    # pos_scores: (num_samples, 1) or (num_samples,)
    # neg_scores: (num_samples, num_neg)
    auc = (pos_scores > neg_scores).astype(float).mean()

    return float(auc)


def auc_at_k(scores, ground_truth):
    """计算AUC（旧版本，保留向后兼容）

    Args:
        scores: 预测分数列表，形状与predictions相同
        ground_truth: 真实相关的物品ID列表

    Returns:
        AUC值
    """
    if not scores or not ground_truth:
        return 0.0

    auc_scores = []
    for score_list, gt_list in zip(scores, ground_truth):
        if len(gt_list) == 0:
            continue

        # 将分数转换为numpy数组
        score_array = np.array(score_list)

        # 创建标签：1表示相关物品，0表示不相关物品
        labels = np.zeros(len(score_array))
        for i, item_id in enumerate(range(len(score_array))):
            if i == 0:  # 假设第一个物品是正样本
                labels[i] = 1

        # 计算AUC
        if len(np.unique(labels)) < 2:  # 如果只有一个类别，跳过
            continue

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, score_array)
            auc_scores.append(auc)
        except:
            # 如果sklearn不可用，使用简单的AUC计算
            pos_scores = score_array[labels == 1]
            neg_scores = score_array[labels == 0]

            if len(pos_scores) == 0 or len(neg_scores) == 0:
                continue

            # 计算正样本分数大于负样本分数的比例
            auc = 0.0
            for pos_score in pos_scores:
                for neg_score in neg_scores:
                    if pos_score > neg_score:
                        auc += 1.0
                    elif pos_score == neg_score:
                        auc += 0.5

            auc = auc / (len(pos_scores) * len(neg_scores))
            auc_scores.append(auc)

    return sum(auc_scores) / len(auc_scores) if auc_scores else 0.0
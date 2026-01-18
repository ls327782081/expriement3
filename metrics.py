import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, ndcg_score


def calculate_metrics(y_true, y_pred, k_list=[5, 10, 20]):
    """
    计算多维度评价指标
    y_true: 真实标签（batch, num_items）
    y_pred: 预测得分（batch, num_items）
    k_list: 评价Top-K
    """
    metrics = {}
    for k in k_list:
        # 1. Precision@K
        precision = []
        for true, pred in zip(y_true, y_pred):
            top_k_idx = np.argsort(pred)[-k:][::-1]
            hit = sum(true[idx] for idx in top_k_idx) / k
            precision.append(hit)
        metrics[f"Precision@{k}"] = np.mean(precision)

        # 2. Recall@K
        recall = []
        for true, pred in zip(y_true, y_pred):
            top_k_idx = np.argsort(pred)[-k:][::-1]
            total_pos = sum(true) if sum(true) > 0 else 1
            hit = sum(true[idx] for idx in top_k_idx) / total_pos
            recall.append(hit)
        metrics[f"Recall@{k}"] = np.mean(recall)

        # 3. NDCG@K
        ndcg = ndcg_score(y_true, y_pred, k=k)
        metrics[f"NDCG@{k}"] = ndcg

        # 4. MRR@K（Mean Reciprocal Rank）
        mrr = []
        for true, pred in zip(y_true, y_pred):
            top_k_idx = np.argsort(pred)[-k:][::-1]
            rank = 0
            for i, idx in enumerate(top_k_idx):
                if true[idx] == 1:
                    rank = i + 1
                    break
            mrr.append(1 / rank if rank > 0 else 0)
        metrics[f"MRR@{k}"] = np.mean(mrr)

    # 5. MAP（Mean Average Precision）
    map_score = []
    for true, pred in zip(y_true, y_pred):
        sorted_idx = np.argsort(pred)[::-1]
        ap = 0
        hits = 0
        for i, idx in enumerate(sorted_idx):
            if true[idx] == 1:
                hits += 1
                ap += hits / (i + 1)
        ap = ap / sum(true) if sum(true) > 0 else 0
        map_score.append(ap)
    metrics["MAP"] = np.mean(map_score)

    # 6. Coverage@K（覆盖率）
    coverage = []
    for k in k_list:
        all_top_k = set()
        for pred in y_pred:
            top_k_idx = np.argsort(pred)[-k:][::-1]
            all_top_k.update(top_k_idx)
        coverage.append(len(all_top_k) / y_true.shape[1])
        metrics[f"Coverage@{k}"] = coverage[-1]

    return metrics
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


def build_sim(features):
    """
    构建相似度矩阵
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    
    sim_matrix = cosine_similarity(features)
    return sim_matrix


def compute_normalized_laplacian(adjacency):
    """
    计算归一化拉普拉斯矩阵
    """
    adjacency = sp.csr_matrix(adjacency)
    # 度矩阵
    degree = np.array(adjacency.sum(axis=1)).flatten()
    # 避免除零
    degree_inv_sqrt = np.sqrt(1.0 / (degree + 1e-8))
    # 归一化
    d_inv_sqrt = sp.diags(degree_inv_sqrt)
    normalized_adj = d_inv_sqrt.dot(adjacency).dot(d_inv_sqrt)
    
    return normalized_adj


def build_knn_neighbourhood(similarity_matrix, topk=10):
    """
    构建KNN邻域图
    """
    # 对每一行找到topk个最大值的索引
    knn_graph = np.zeros_like(similarity_matrix)
    
    for i in range(similarity_matrix.shape[0]):
        # 找到topk个最相似的项
        topk_indices = np.argpartition(similarity_matrix[i], -topk)[-topk:]
        knn_graph[i, topk_indices] = similarity_matrix[i, topk_indices]
    
    return knn_graph


def dict2str(result_dict):
    """
    将字典转换为字符串输出
    """
    result_str = ""
    for key, value in result_dict.items():
        if isinstance(value, float):
            result_str += f"{key}: {value:.4f} "
        else:
            result_str += f"{key}: {value} "
    return result_str.strip()


def get_local_time():
    """
    获取本地时间
    """
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def early_stopping(current_value, best_value, cur_step, max_step, bigger=True):
    """
    早停策略
    """
    update_flag = False
    if bigger:
        if current_value > best_value:
            best_value = current_value
            cur_step = 0
            update_flag = True
        else:
            cur_step += 1
    else:
        if current_value < best_value:
            best_value = current_value
            cur_step = 0
            update_flag = True
        else:
            cur_step += 1

    stop_flag = cur_step >= max_step
    return best_value, cur_step, stop_flag, update_flag
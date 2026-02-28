from collections import defaultdict

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


# ==================== 新增：通用NaN/Inf检查函数 ====================
def check_tensor(tensor: torch.Tensor, module_name: str, step: str = "") :
    """
    检查张量是否包含NaN/Inf，并输出详细信息
    Args:
        tensor: 待检查的张量
        module_name: 模块名称（如"SimpleItemEncoder/text_feat"）
        step: 步骤描述（可选）
    Returns:
        has_error: 是否包含NaN/Inf
    """
    if tensor is None:
        print(f"[检查点] {module_name} {step}: 张量为None")
        return

    # 转换为float（避免bool/int张量干扰）
    tensor = tensor.float()

    # 统计NaN/Inf数量
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    total_count = tensor.numel()

    # 计算最值（处理全NaN/Inf的情况）
    try:
        tensor_clamped = torch.clamp(tensor, min=-1e10, max=1e10)  # 防止最值计算溢出
        max_val = tensor_clamped.max().item()
        min_val = tensor_clamped.min().item()
    except:
        max_val = "NaN"
        min_val = "NaN"

    # 输出检查结果
    if nan_count > 0 or inf_count > 0:
        print(f"\n❌ [数值异常] {module_name} {step}:")
        print(f"  - NaN数量: {nan_count}/{total_count} ({nan_count / total_count * 100:.2f}%)")
        print(f"  - Inf数量: {inf_count}/{total_count} ({inf_count / total_count * 100:.2f}%)")
        print(f"  - 最值: {min_val} ~ {max_val}")
        print(f"  - 张量形状: {tensor.shape}")
        return


def validate_semantic_id_uniqueness(model, all_item_loader, device):
    """
    验证全量商品编码后的语义ID重复情况
    Args:
        model: 训练好的模型（包含item_encoder）
        all_item_loader: 全量商品的dataloader（只需要item_id、text_feat、vision_feat）
        device: 设备（cuda/cpu）
    Returns:
        统计结果字典
    """
    # 1. 模型切换到评估模式，禁止梯度计算
    model.eval()
    semantic_id_map = defaultdict(list)  # key: 语义ID字符串, value: 对应的商品ID列表
    all_item_ids = []
    all_semantic_ids = []

    with torch.no_grad():  # 关键：禁止梯度，加速编码
        for batch in all_item_loader:
            # 2. 取出批次数据
            item_ids = batch['item_id'].cpu().numpy().tolist()  # 商品ID列表
            text_feat = batch['text_feat'].to(device)
            visual_feat = batch['vision_feat'].to(device)

            # 3. 执行编码，获取语义ID（核心步骤）
            # 假设你的encoder返回：(embedding, 语义ID logits, 量化后的emb, 重建损失)
            _, semantic_logits, _, _ = model.item_encoder(
                text_feat, visual_feat, return_semantic_logits=True
            )

            # 4. 从logits中取出最终的语义ID（8层，每层选概率最大的）
            batch_semantic_ids = torch.argmax(semantic_logits, dim=-1)  # [B, 8]
            batch_semantic_ids = batch_semantic_ids.cpu().numpy()

            # 5. 记录映射关系（把8维ID转成字符串当key，方便去重）
            for idx, item_id in enumerate(item_ids):
                semantic_id = batch_semantic_ids[idx]  # [8,]
                semantic_id_str = '_'.join(map(str, semantic_id))  # "669_755_25_573_188_268_32_358"

                semantic_id_map[semantic_id_str].append(item_id)
                all_item_ids.append(item_id)
                all_semantic_ids.append(semantic_id_str)

    # 6. 核心统计（只保留关键结果，去掉冗余打印）
    total_items = len(all_item_ids)
    unique_semantic_ids = len(semantic_id_map)  # 这就是之前日志里的195！
    duplicate_ratio = 1 - (unique_semantic_ids / total_items)  # 重复率

    # 7. 统计重复最严重的Top5语义ID
    top5_duplicate = sorted(
        semantic_id_map.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:5]

    # 8. 输出核心结果（极简版）
    print("===== 语义ID重复率验证结果 =====")
    print(f"1. 商品总数: {total_items}")
    print(f"2. 唯一语义ID数: {unique_semantic_ids}")
    print(f"3. 语义ID重复率: {duplicate_ratio:.4f} ({duplicate_ratio * 100:.2f}%)")
    print(
        f"4. 有效码本利用率: {unique_semantic_ids / total_items:.4f} ({unique_semantic_ids / total_items * 100:.2f}%)")
    print("\n===== Top5 重复最多的语义ID =====")
    for i, (sem_id, items) in enumerate(top5_duplicate, 1):
        print(f"{i}. 语义ID {sem_id} → 对应{len(items)}个商品")

    # 返回统计结果，方便后续分析
    return {
        "total_items": total_items,
        "unique_semantic_ids": unique_semantic_ids,
        "duplicate_ratio": duplicate_ratio,
        "top5_duplicate": top5_duplicate,
        "semantic_id_map": semantic_id_map  # 完整映射，可保存下来分析
    }
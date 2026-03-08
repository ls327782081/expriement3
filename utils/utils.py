import os
from collections import defaultdict

import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from config import new_config


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

def validate_semantic_id_uniqueness(model, all_item_loader, new_config):
    """
    验证全量商品编码后的语义ID重复情况（修正版）
    关键修复：
    1. 正确计算码本利用率
    2. 精准解析语义ID维度
    3. 新增码本使用统计，定位重复根源
    """
    model.eval()
    semantic_id_map = defaultdict(list)
    all_item_ids = []
    all_semantic_ids = []

    # 新增：统计码本使用情况（每层的码本是否被使用）
    codebook_usage = [set() for _ in range(new_config.id_length)]  # 每层一个集合，存被使用的码本索引

    with torch.no_grad():
        for batch in all_item_loader:
            item_ids = batch['item_id'].cpu().numpy().tolist()
            text_feat = batch['text_feat'].to(new_config.device)
            visual_feat = batch['vision_feat'].to(new_config.device)

            # 执行编码
            _, semantic_logits, _, _, _, _, batch_semantic_ids, _ = model.item_encoder(
                text_feat, visual_feat, return_semantic_logits=True
            )

            # ========== 修正1：精准解析语义ID（适配new_config） ==========
            # semantic_logits shape: [B, id_length, codebook_size]
            assert semantic_logits.ndim == 3, f"semantic_logits维度错误，应为3维[B,id_length,codebook_size]，实际是{semantic_logits.ndim}维"
            assert semantic_logits.shape[
                       1] == new_config.id_length, f"语义ID层数错误，配置是{new_config.id_length}，实际是{semantic_logits.shape[1]}"

            # ========== 修正2：统计码本使用情况 ==========
            for layer_idx in range(new_config.id_length):
                # 取出该层所有batch的ID，加入集合（自动去重）
                layer_ids = batch_semantic_ids[:, layer_idx]
                codebook_usage[layer_idx].update(layer_ids.tolist())

            # 记录ID映射
            for idx, item_id in enumerate(item_ids):
                semantic_id = batch_semantic_ids[idx]  # [id_length,]
                # 修正：确保转成纯数值（处理tensor/numpy数组两种情况）
                semantic_id_vals = []
                for val in semantic_id:
                    if isinstance(val, torch.Tensor):
                        semantic_id_vals.append(str(val.item()))
                    else:
                        semantic_id_vals.append(str(val))
                semantic_id_str = '_'.join(semantic_id_vals)

                semantic_id_map[semantic_id_str].append(item_id)
                all_item_ids.append(item_id)
                all_semantic_ids.append(semantic_id_str)

    # ========== 核心统计（修正+新增） ==========
    total_items = len(all_item_ids)
    unique_semantic_ids = len(semantic_id_map)
    duplicate_ratio = 1 - (unique_semantic_ids / total_items)

    # 修正3：正确计算码本利用率
    total_codebooks = new_config.id_length * new_config.codebook_size  # 总码本数
    used_codebooks = sum([len(s) for s in codebook_usage])  # 被使用的码本数
    codebook_utilization = used_codebooks / total_codebooks if total_codebooks > 0 else 0.0

    # ========== 精准输出 ==========
    print("===== 语义ID重复率验证结果（修正版） =====")
    print(f"1. 商品总数: {total_items}")
    print(f"2. 唯一语义ID数: {unique_semantic_ids}")
    print(f"3. 语义ID重复率: {duplicate_ratio:.4f} ({duplicate_ratio * 100:.2f}%)")
    print(f"4. 有效码本利用率: {codebook_utilization:.4f} ({codebook_utilization * 100:.2f}%)")
    print(f"   - 总码本数: {total_codebooks}（{new_config.id_length}层 × {new_config.codebook_size}码本/层）")
    print(f"   - 被使用码本数: {used_codebooks}")
    print(f"   - 每层使用码本数: {[len(s) for s in codebook_usage]}")

    # Top5重复ID
    print("\n===== Top5 重复最多的语义ID =====")
    top5_duplicate = sorted(semantic_id_map.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    for i, (sem_id, items) in enumerate(top5_duplicate, 1):
        print(f"{i}. 语义ID {sem_id} → 对应{len(items)}个商品")

    return {
        "total_items": total_items,
        "unique_semantic_ids": unique_semantic_ids,
        "duplicate_ratio": duplicate_ratio,
        "codebook_utilization": codebook_utilization,
        "total_codebooks": total_codebooks,
        "used_codebooks": used_codebooks,
        "layer_codebook_usage": [len(s) for s in codebook_usage],
        "top5_duplicate": top5_duplicate,
        "semantic_id_map": semantic_id_map
    }


def calculate_metrics(pos_scores, neg_scores, k_list=[5, 10, 20]):
    """计算推荐指标：HR@k、NDCG@k、MRR"""
    metrics = defaultdict(float)
    pos_scores = pos_scores.unsqueeze(1)  # (batch, 1)

    # HR@k
    for k in k_list:
        # 正样本分数 > 负样本分数的数量 / k
        hr = calculate_hr(pos_scores, neg_scores, k)
        metrics[f"HR@{k}"] = hr

    # NDCG@k
    for k in k_list:
        ndcg = calculate_ndcg(pos_scores, neg_scores, k)
        metrics[f"NDCG@{k}"] = ndcg

    # MRR
    all_scores = torch.cat([pos_scores, neg_scores], dim=1)
    _, indices = torch.sort(all_scores, dim=1, descending=True)
    pos_rank = (indices == 0).nonzero()[:, 1] + 1
    mrr = (1 / pos_rank.float()).mean().item()
    metrics["MRR"] = mrr

    return metrics


def calculate_ndcg(pos_scores, neg_scores, k=10):
    """
    适配pos_scores=(batch,1)、neg_scores=(batch,num_neg)的NDCG计算
    理论依据：SIGIR 2022《RecNDCG》标准计算逻辑
    """
    batch_size = pos_scores.shape[0]
    num_neg = neg_scores.shape[1]
    actual_k = min(k, num_neg + 1)  # 正样本+负样本总数

    # 1. 拼接正/负样本分数（维度完全匹配：(256,1)+(256,99) → (256,100)）
    all_scores = torch.cat([pos_scores, neg_scores], dim=1)  # (256, 100)

    # 2. 按分数降序排序，获取每个样本的排名（从1开始）
    _, sorted_indices = torch.sort(all_scores, dim=1, descending=True)  # (256, 100)

    # 3. 找到正样本在排序后的位置（正样本是第0列）
    # 遍历每个batch，找到正样本的索引并+1（排名从1开始）
    pos_rank = []
    for i in range(batch_size):
        # 找到第i个样本中值为0的索引（正样本列）
        rank = torch.where(sorted_indices[i] == 0)[0].item() + 1
        pos_rank.append(rank)
    pos_rank = torch.tensor(pos_rank, device=pos_scores.device)  # (256,)

    # 4. 计算DCG@k和IDCG@k
    # DCG@k：正样本排名≤k时有效，否则为0
    dcg = torch.where(
        pos_rank <= actual_k,
        1 / torch.log2(pos_rank.float() + 1),  # 排名从1开始，log2(2)=1
        torch.tensor(0.0, device=pos_scores.device)
    )

    # IDCG@k：理想情况正样本排名=1，DCG=1/log2(2)=1
    idcg = torch.ones_like(dcg, device=pos_scores.device) / torch.log2(torch.tensor(2.0))

    # 5. 计算平均NDCG
    ndcg = (dcg / idcg).sum().item() / batch_size
    return ndcg


def calculate_hr(pos_scores, neg_scores, k=10):
    """
    适配(256,1)+(256,99)的HR@k计算（核心推荐指标）
    理论依据：RecSys 2021《HR-NDCG评估规范》
    """
    batch_size = pos_scores.shape[0]
    num_neg = neg_scores.shape[1]
    actual_k = min(k, num_neg + 1)

    # 拼接分数并排序
    all_scores = torch.cat([pos_scores, neg_scores], dim=1)  # (256, 100)
    _, sorted_indices = torch.sort(all_scores, dim=1, descending=True)  # (256, 100)

    # 统计正样本排名≤k的数量
    hr_count = 0
    for i in range(batch_size):
        pos_rank = torch.where(sorted_indices[i] == 0)[0].item() + 1
        if pos_rank <= actual_k:
            hr_count += 1

    hr = hr_count / batch_size
    return hr


def calculate_id_metrics(indices_list):
    """计算语义ID指标：重复率、基尼系数、码本利用率"""
    metrics = defaultdict(float)
    # indices_list: [layer0_indices, layer1_indices, ...] (每个元素是(batch,)

    # 1. ID重复率：先拼接所有层的ID成完整ID
    full_ids = []
    for batch_idx in range(len(indices_list[0])):
        id_str = "_".join([str(indices_list[layer][batch_idx].item()) for layer in range(len(indices_list))])
        full_ids.append(id_str)
    unique_ids = len(set(full_ids))
    total_ids = len(full_ids)
    metrics["id_repeat_rate"] = 1 - (unique_ids / total_ids)

    # 2. 各层码本利用率
    for layer_idx, indices in enumerate(indices_list):
        used_codes = len(torch.unique(indices))
        # 从配置中获取该层的码本大小
        if layer_idx in new_config.semantic_hierarchy["topic"]["layers"]:
            total_codes = new_config.semantic_hierarchy["topic"]["codebook_size"]
        else:
            total_codes = new_config.semantic_hierarchy["style"]["codebook_size"]
        metrics[f"codebook_usage_layer{layer_idx}"] = used_codes / total_codes

    # 3. 各层基尼系数（码本选择均衡性）
    for layer_idx, indices in enumerate(indices_list):
        # 统计每个码本被选择的次数
        code_counts = torch.bincount(indices, minlength=total_codes)
        code_counts = code_counts / code_counts.sum()  # 归一化
        # 计算基尼系数
        sorted_counts = torch.sort(code_counts)[0]
        n = len(sorted_counts)
        cum_counts = torch.cumsum(sorted_counts, dim=0)
        gini = (n + 1 - 2 * torch.sum(cum_counts) / cum_counts[-1]) / n
        metrics[f"gini_layer{layer_idx}"] = gini.item()

    return metrics


def seed_everything(seed):
    """固定随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fast_codebook_reset(ahrq_model, text_feat, vision_feat, config):
    """
    重置未使用的码字（适配新版ah_rq）
    使用 ahrq_model.rq.vq_layers[i].embedding.weight 访问码本
    """
    if hasattr(ahrq_model, 'code_usage_count') and ahrq_model.code_usage_count:
        ahrq_model.eval()

        with torch.no_grad():
            output = ahrq_model(text_feat, vision_feat)
            # 新版本返回8个值，旧版本返回4个值
            quantized = output[0]

        feat_mean = quantized.mean(dim=0).detach()
        total_reset = 0

        # 遍历所有量化层
        for layer_idx, quantizer in enumerate(ahrq_model.rq.vq_layers):
            codebook = quantizer.embedding.weight
            cb_size = codebook.size(0)

            # 获取当前层的使用统计
            cb_type = ahrq_model.layer_types[layer_idx] if hasattr(ahrq_model, 'layer_types') else 'unknown'
            cb_key = f"{cb_type}_{layer_idx}"

            usage = ahrq_model.code_usage_count.get(cb_key, {})
            dead_codes = [k for k in range(cb_size) if usage.get(k, 0) < config.ahrq_reset_threshold]

            if len(dead_codes) > 0:
                noise = torch.randn(len(dead_codes), codebook.size(-1)).to(codebook.device) * 0.01
                reset_feat = feat_mean.unsqueeze(0).repeat(len(dead_codes), 1) + noise

                for i, code_idx in enumerate(dead_codes):
                    codebook.data[code_idx] = codebook.data[code_idx] * 0.9 + reset_feat[i] * 0.1

                    if cb_key not in ahrq_model.code_usage_count:
                        ahrq_model.code_usage_count[cb_key] = {}
                    ahrq_model.code_usage_count[cb_key][code_idx] = config.ahrq_reset_threshold

                total_reset += len(dead_codes)

        print(f"Global codebook reset completed! Reset {total_reset} dead codes (stable update)")
    else:
        print("No code usage count found, skipping reset.")


class EarlyStopping:
    """早停工具类：支持「越大越好」（如NDCG）和「越小越好」（如Gini）的指标"""

    def __init__(self, patience=7, verbose=False, delta=0.0001, path='best_model.pth', mode='max'):
        """
        Args:
            patience: 多少个epoch验证集指标没有提升就停止训练
            verbose: 是否打印日志
            delta: 指标提升的最小阈值（小于该值视为无提升）
            path: 最优模型保存路径
            mode: 'max'（指标越大越好，如NDCG）| 'min'（指标越小越好，如Gini）
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.mode = mode  # 新增：适配不同指标类型

        self.counter = 0  # 无提升的epoch计数
        self.best_score = None  # 最优指标值
        self.early_stop = False  # 是否触发早停
        self.best_epoch = 0  # 最优模型对应的epoch

    def __call__(self, val_score, model, optimizer=None):
        """
        每次验证集评估后调用
        Args:
            val_score: 验证集指标（Gini用min，NDCG用max）
            model: 模型实例
            optimizer: 优化器（可选，保存优化器状态）
        """
        # 根据mode转换分数（统一按max处理）
        score = val_score if self.mode == 'max' else -val_score

        # 第一次调用初始化最优分数
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, optimizer)
        # 指标未提升（考虑delta阈值）
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # 触发早停
            if self.counter >= self.patience:
                self.early_stop = True
        # 指标提升，保存新的最优模型
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_score, model, optimizer):
        """保存最优模型"""

        # 构建保存内容（Stage1仅保存AH-RQ，Stage2保存完整模型）
        save_dict = {
            'best_score': val_score,
            'epoch': self.best_epoch + 1
        }
        # Stage1：仅保存AH-RQ参数（减少存储）
        if 'quant' in self.path:
            save_dict['ahrq_state_dict'] = model.ahrq.state_dict()
            if optimizer is not None:
                save_dict['optimizer_quant_state_dict'] = optimizer.state_dict()
        # Stage2：保存完整模型
        else:
            save_dict['model_state_dict'] = model.state_dict()
            if optimizer is not None:
                save_dict['optimizer_rec_state_dict'] = optimizer.state_dict()

        # 创建保存目录
        save_dir = os.path.dirname(self.path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存模型
        torch.save(save_dict, self.path)
        self.best_epoch += 1
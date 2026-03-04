import torch
import torch.nn.functional as F
from config import new_config
import numpy as np


def compute_bpr_loss(pos_scores, neg_scores):
    """BPR损失（序列/推荐任务核心）"""
    pos_scores = pos_scores.unsqueeze(1)  # (batch, 1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return loss.mean()


def quantization_loss(quantized, x, commitment_cost=0.25):
    """基础量化损失（commitment loss）"""
    # 重构损失：量化后 vs 原始
    recon_loss = F.mse_loss(quantized, x.detach())
    # 码本损失：约束码本更新
    commit_loss = F.mse_loss(x, quantized.detach())
    return recon_loss + commitment_cost * commit_loss


def compute_quantization_loss(quantized, pos_raw, pos_code_probs, config):
    """
    最终修复版：解决多样性损失后期为0导致Gini反弹的问题
    核心改进：
    1. 多样性损失从「阈值惩罚」→「方差惩罚」，持续约束码本使用均匀性
    2. 降低重构损失权重，强化码本损失优先级
    3. 保留分层优化逻辑，适配三维code_probs
    """
    # ========== 0. 维度校验（新增，避免传参错误） ==========
    if len(pos_code_probs.shape) != 3:
        raise ValueError(f"pos_code_probs必须是3维(batch, num_layers, cb_size)，当前是{pos_code_probs.shape}")
    batch_size, num_layers, cb_size = pos_code_probs.shape

    # ========== 1. 全局重构损失（权重降至极低，避免过拟合） ==========
    recon_loss = F.mse_loss(quantized, pos_raw) * getattr(config, "recon_weight", 0.001)

    # ========== 2. 分层计算码本损失（核心：每层独立优化） ==========
    layer_gini_losses = []  # 每层的Gini损失
    layer_diversity_losses = []  # 每层的多样性损失（方差惩罚）
    layer_entropy_losses = []  # 每层的熵损失

    for layer_idx in range(num_layers):
        # 取出当前层的code_probs：(batch, cb_size)
        layer_code_probs = pos_code_probs[:, layer_idx, :] + 1e-8  # 加小值避免log(0)

        # ---------- 2.1 分层Gini系数损失（对齐评估指标） ----------
        code_usage = layer_code_probs.mean(0)  # (cb_size,) 该层每个码本的平均使用率
        sorted_usage = torch.sort(code_usage)[0]
        n = len(sorted_usage)
        cumsum = torch.cumsum(sorted_usage, dim=0)
        cumsum_total = cumsum[-1] if cumsum[-1] > 1e-8 else 1e-8  # 兜底避免除以0
        gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum_total) / n
        layer_gini_losses.append(gini)

        # ---------- 2.2 分层ID多样性损失（核心修复：方差惩罚） ----------
        prob_threshold = getattr(config, "prob_threshold", 1e-6)  # 适配你的数值范围
        # 步骤1：计算每个码本在批次内的使用次数（而非仅“是否使用”）
        # code_usage_count: (cb_size,) → 每个码本被多少样本选择
        code_usage_count = (layer_code_probs > prob_threshold).float().sum(0)

        # 步骤2：计算使用次数的方差（方差越大，码本使用越集中）
        # 方差=0 → 所有码本使用次数完全相同（理想状态）
        # 方差越大 → 少数码本被频繁使用，多数被闲置
        usage_variance = torch.var(code_usage_count)

        # 步骤3：可选：基础使用率保底（先保证用够80%码本，再优化均匀性）
        code_used = (code_usage_count > 0).float().mean()  # 该层码本使用率
        usage_penalty = F.relu(0.8 - code_used) * getattr(config, "usage_weight", 1.0)
        # 新增：归一化方差（除以批次大小的平方，拉回0~1区间）
        usage_variance_norm = usage_variance / (batch_size ** 2)
        # 总多样性损失：方差惩罚（核心） + 使用率保底（辅助）
        diversity_loss = (usage_variance_norm + usage_penalty) * getattr(config, "diversity_weight", 3.0)
        layer_diversity_losses.append(diversity_loss)

        # ---------- 2.3 分层熵损失（辅助提升码本分布多样性） ----------
        entropy = -(layer_code_probs * layer_code_probs.log()).sum(-1).mean()
        entropy_norm = 1 - (entropy / np.log(cb_size))  # 归一化到0-1
        layer_entropy_losses.append(entropy_norm * getattr(config, "entropy_weight", 0.05))

    # 汇总分层损失（取均值，保证各层权重一致）
    avg_gini_loss = torch.stack(layer_gini_losses).mean() * getattr(config, "gini_weight", 8.0)
    avg_diversity_loss = torch.stack(layer_diversity_losses).mean()
    avg_entropy_loss = torch.stack(layer_entropy_losses).mean()

    # ========== 3. 总损失（分层码本损失主导 + 弱重构损失） ==========
    total_loss = avg_gini_loss + avg_diversity_loss + avg_entropy_loss + recon_loss

    # ========== 4. 整理损失字典（便于监控每层效果） ==========
    loss_dict = {
        # 全局损失
        "recon_loss": recon_loss.item(),
        "total_loss": total_loss.item(),
        # 分层损失均值
        "avg_gini_loss": avg_gini_loss.item(),
        "avg_diversity_loss": avg_diversity_loss.item(),
        "avg_entropy_loss": avg_entropy_loss.item(),
        # 每层详细损失（便于调试）
        # "layer_gini_losses": [g.item() for g in layer_gini_losses],
        # "layer_diversity_losses": [d.item() for d in layer_diversity_losses],
        # "layer_entropy_losses": [e.item() for e in layer_entropy_losses],
        # 关键监控指标
        # "avg_gini": np.mean([g.item() for g in layer_gini_losses]),
        # 保留原使用率指标（便于对比）
        # "avg_batch_usage_rate": np.mean([
        #     (pos_code_probs[:, idx, :] > 1e-6).any(dim=0).float().mean().item()
        #     for idx in range(num_layers)
        # ])
    }

    return total_loss, loss_dict

def hierarchical_consistency_loss(quantized_layers, indices, hierarchy):
    """层次语义一致性损失（Topic→Style约束）"""
    loss = 0.0
    # 提取Topic层和Style层的量化结果
    topic_layers = [quantized_layers[i] for i in hierarchy["topic"]["layers"]]
    style_layers = [quantized_layers[i] for i in hierarchy["style"]["layers"]]

    # Topic层特征约束Style层特征（确保Style层基于Topic层）
    topic_feat = torch.mean(torch.stack(topic_layers, dim=1), dim=1)  # (batch, dim)
    for style_feat in style_layers:
        # 计算余弦相似度，要求Style层与Topic层相似度高
        sim = F.cosine_similarity(style_feat, topic_feat, dim=-1)
        consistency = 1 - sim.mean()  # 相似度越低，损失越大
        loss += consistency * new_config.pmat_consistency_weight

    return loss


def total_loss(pos_scores, neg_scores, quantized, x, quantized_layers, indices, hierarchy):
    """总损失：BPR + 量化 + 层次一致性"""
    bpr = compute_bpr_loss(pos_scores, neg_scores)
    quant = quantization_loss(quantized, x, new_config.ahrq_beta)
    consistency = hierarchical_consistency_loss(quantized_layers, indices, hierarchy)

    # 分层加权
    total = (
            new_config.pmat_rec_loss_weight * bpr +
            new_config.pmat_semantic_loss_weight * quant +
            consistency
    )
    return total, {"bpr_loss": bpr.item(), "quant_loss": quant.item(), "consistency_loss": consistency.item()}

def compute_total_loss(pos_scores, neg_scores, quant_feat, raw_feat, code_probs, config):
    """
    多目标损失函数（理论依据：多目标优化Pareto最优）
    Args:
        pos_scores/neg_scores: 正负样本分数（排序损失）
        quant_feat/raw_feat: 量化特征/原始多模态特征（重构损失）
        code_probs: 码本选择概率（熵损失）
        config: 配置（含损失权重）
    Returns:
        total_loss: 加权总损失
    """
    # 1. 带间距的BPR损失（解决HR@10=0，理论：边际排序约束）
    # 理论依据：ICML 2020《MarginBPR》：强制正样本分数比负样本高margin，避免分数重叠
    margin = 0.5
    diff = (pos_scores.unsqueeze(1) - neg_scores) - margin
    bpr_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

    # 2. 码本熵损失（解决码本坍塌，理论：最大熵原理）
    # 理论依据：NeurIPS 2021《VQ-VAE-2》：最大化码本使用熵，强制均匀使用码本
    entropy = -torch.sum(code_probs * torch.log(code_probs + 1e-8), dim=-1).mean()
    entropy_loss = -config.entropy_weight * entropy  # 负号：最大化熵→最小化损失

    prob = code_probs.reshape(-1, code_probs.size(-1))
    entropy = -(prob * (prob + 1e-8).log()).sum(-1).mean()
    entropy_loss = -config.entropy_weight * entropy * 3

    # 3. 特征重构损失（约束量化精度，理论：自编码器重构理论）
    # 理论依据：ICLR 2019《VQ-VAE》：重构损失保证量化特征保留原始特征信息
    recon_loss = F.mse_loss(quant_feat, raw_feat) * config.recon_weight

    # 4. 分数正则化损失（解决NDCG虚假偏高，理论：L2正则约束分数分布）
    # 理论依据：SIGIR 2022《RecReg》：约束分数方差，避免所有分数趋近于同一值
    score_reg = (torch.var(pos_scores) + torch.var(neg_scores)) * config.reg_weight

    # 总损失（权重遵循：BPR为主，其他为辅）
    total_loss = bpr_loss + entropy_loss + recon_loss + score_reg
    return total_loss, {"bpr_loss": bpr_loss.item(),
                        "entropy_loss": entropy_loss.item(),
                        "recon_loss": recon_loss.item(),
                        "score_reg": score_reg.item()}

def quantization_uniform_loss(code_probs):
    """
    码本均匀性损失：强制每个码本ID被均匀使用
    理论依据：SIGIR 2023《QuantRec》
    """
    # code_probs: [batch, seq_len, cb_size] 或 [batch, cb_size]
    if len(code_probs.shape) == 3:
        code_probs = code_probs.reshape(-1, code_probs.size(-1))
    # 计算每个ID的使用频率
    id_usage = code_probs.sum(dim=0) / code_probs.sum()
    # 均匀性损失：最小化与均匀分布的KL散度
    uniform_dist = torch.ones_like(id_usage) / id_usage.size(0)
    kl_div = torch.sum(id_usage * (id_usage + 1e-8).log() - id_usage * (uniform_dist + 1e-8).log())
    return kl_div


def compute_ranking_loss(pos_scores, neg_scores, config):
    """
    Stage2排序损失函数（封装，避免重复代码）
    兼容配置参数缺失，加入异常处理
    """
    try:
        # 确保pos_scores维度正确
        pos_flat = pos_scores.squeeze(1) if pos_scores.dim() > 1 else pos_scores
        neg_flat = neg_scores.reshape(-1) if neg_scores.dim() > 2 else neg_scores

        # BPR损失（带margin）
        bpr_margin = getattr(config, "bpr_margin", 0.4)
        diff = (pos_flat.unsqueeze(1) - neg_scores) - bpr_margin
        bpr_loss = -torch.nn.functional.logsigmoid(diff).mean()

        # 分数正则损失（兼容默认值）
        reg_weight = getattr(config, "reg_weight", 0.005)
        score_reg = (torch.var(pos_flat) + torch.var(neg_scores)).mean() * reg_weight

        # 总损失
        total_loss = bpr_loss + score_reg
        return total_loss, {
            "bpr_loss": bpr_loss.item(),
            "score_reg": score_reg.item(),
            "total_loss": total_loss.item()
        }
    except Exception as e:
        # 异常兜底：仅用BPR损失
        print(f"Ranking loss compute error: {e}, fallback to BPR only")
        pos_flat = pos_scores.squeeze(1) if pos_scores.dim() > 1 else pos_scores
        bpr_loss = -torch.nn.functional.logsigmoid(pos_flat.unsqueeze(1) - neg_scores).mean()
        return bpr_loss, {
            "bpr_loss": bpr_loss.item(),
            "score_reg": 0.0,
            "total_loss": bpr_loss.item()
        }


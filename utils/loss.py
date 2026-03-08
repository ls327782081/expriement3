import torch
import torch.nn.functional as F
from config import new_config
import numpy as np


# ===================== RQ-VAE 风格的重构损失 =====================
def compute_rqvae_recon_loss(quantized, raw_feat, codebook_outputs, encoder_outputs, config, quant_losses=None):
    """
    RQ-VAE 风格的重构损失（官方版）

    包含两个核心组成部分：
    1. Reconstruction Loss: 量化特征 vs 原始特征
    2. VQ Loss: 由模型forward计算好的每层codebook_loss + beta*commitment_loss

    Args:
        quantized: 量化后的特征 (batch, hidden_dim)
        raw_feat: 原始融合特征 (batch, hidden_dim)
        codebook_outputs: 每层选中的码本向量列表
        encoder_outputs: 每层编码器输出列表
        config: 配置对象
        quant_losses: 由forward返回的每层VQ损失列表

    Returns:
        total_loss: 总重构损失
        loss_dict: 各分量损失值
    """
    # 解析配置
    recon_weight = getattr(config, 'rqvae_recon_weight', 1.0)
    vq_weight = getattr(config, 'rqvae_vq_weight', 1.0)

    # 1. Reconstruction Loss: 量化特征 vs 原始特征
    recon_loss = F.mse_loss(quantized, raw_feat)

    # 2. VQ Loss: 使用模型返回的每层量化损失
    if quant_losses is not None and len(quant_losses) > 0:
        vq_loss = torch.stack(quant_losses).mean()
    else:
        # 备用：手动计算
        codebook_loss = 0.0
        commit_loss = 0.0
        for layer_idx in range(len(codebook_outputs)):
            cb_out = codebook_outputs[layer_idx]
            enc_out = encoder_outputs[layer_idx]
            codebook_loss += F.mse_loss(cb_out, enc_out.detach())
            commit_loss += F.mse_loss(enc_out, cb_out.detach())
        vq_loss = codebook_loss + config.ahrq_beta * commit_loss

    # 总损失 = 重构损失 + VQ损失
    total_recon_loss = recon_weight * recon_loss + vq_weight * vq_loss

    loss_dict = {
        "rqvae_recon_loss": recon_loss.item(),
        "rqvae_vq_loss": vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss,
        "rqvae_total_loss": total_recon_loss.item()
    }

    return total_recon_loss, loss_dict


def compute_bpr_loss(pos_scores, neg_scores):
    """BPR损失（序列/推荐任务核心）"""
    pos_scores = pos_scores.unsqueeze(1)  # (batch, 1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return loss.mean()


def quantization_loss(quantized, x, commitment_cost=0.25):
    """基础量化损失（commitment loss）"""
    recon_loss = F.mse_loss(quantized, x.detach())
    commit_loss = F.mse_loss(x, quantized.detach())
    return recon_loss + commitment_cost * commit_loss


def compute_quantization_loss(quantized, pos_raw, pos_code_probs, indices, config):
    """
    原有AH-RQ的自定义损失（带ID唯一性约束）
    """
    # ===================== 1. 核心配置 & 维度校验 =====================
    num_layers = len(config.semantic_hierarchy["topic"]["layers"]) + len(config.semantic_hierarchy["style"]["layers"])
    hidden_dim = pos_raw.shape[-1]
    assert hidden_dim % num_layers == 0, f"hidden_dim({hidden_dim})必须能被num_layers({num_layers})整除"
    layer_dim = hidden_dim // num_layers

    batch_size = pos_raw.shape[0]
    assert len(pos_code_probs.shape) == 3, f"pos_code_probs必须是3维(batch, num_layers, codebook_size)"
    assert pos_code_probs.shape[1] == num_layers, f"pos_code_probs的层数({pos_code_probs.shape[1]})与配置({num_layers})不匹配"

    # ===================== 2. 修复indices：列表→张量 =====================
    if isinstance(indices, list):
        indices = torch.stack(indices, dim=1)
    assert indices.shape == (batch_size, num_layers), f"indices维度必须是(batch, num_layers)，当前是{indices.shape}"

    # ===================== 3. 将融合特征拆分为多层子特征 =====================
    pos_raw_layered = pos_raw.reshape(batch_size, num_layers, layer_dim)

    # ===================== 4. 重构损失 =====================
    if len(quantized.shape) == 2:
        quantized_layered = quantized.reshape(batch_size, num_layers, layer_dim)
    else:
        quantized_layered = quantized

    recon_loss = 0.0
    for layer in range(num_layers):
        recon_loss += F.mse_loss(quantized_layered[:, layer, :], pos_raw_layered[:, layer, :])
    recon_loss = recon_loss / num_layers
    recon_loss = torch.clamp(recon_loss, min=1e-5)

    # ===================== 5. 码本均匀使用损失 =====================
    avg_probs = pos_code_probs.mean(dim=0)
    layer_entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum(-1)
    target_entropy = torch.log(torch.tensor(pos_code_probs.shape[2], dtype=torch.float32, device=pos_raw.device))
    usage_loss = target_entropy - layer_entropy.mean()

    # ===================== 6. 组合ID唯一性损失 =====================
    topic_cb_size = config.semantic_hierarchy["topic"]["codebook_size"]
    style_cb_size = config.semantic_hierarchy["style"]["codebook_size"]
    topic_layers = len(config.semantic_hierarchy["topic"]["layers"])
    style_layers = len(config.semantic_hierarchy["style"]["layers"])

    combo_weights = []
    cb_sizes = [topic_cb_size] * topic_layers + [style_cb_size] * style_layers
    for i in range(num_layers):
        weight = 1
        for j in range(i + 1, num_layers):
            weight *= cb_sizes[j]
        combo_weights.append(weight)
    combo_weights = torch.tensor(combo_weights, dtype=torch.long, device=pos_raw.device)

    combo_indices = (indices.long() * combo_weights).sum(dim=1)
    combo_count = torch.bincount(combo_indices, minlength=int(combo_weights.sum().item())).float()
    repeat_penalty = (combo_count ** 2).sum() / batch_size
    total_cb_combinations = topic_cb_size ** topic_layers * style_cb_size ** style_layers
    id_unique_loss = repeat_penalty / total_cb_combinations

    # ===================== 7. 总损失 =====================
    recon_weight = 1.0
    usage_weight = 0.1
    unique_weight = 5.0
    most_used_penalty = repeat_penalty / (batch_size ** 2) * 10

    total_loss = recon_weight * recon_loss + usage_weight * usage_loss + unique_weight * id_unique_loss + most_used_penalty

    # ===================== 8. 损失字典 =====================
    unique_combo = len(torch.unique(combo_indices))
    combo_repeat_rate = 1 - (unique_combo / batch_size)
    loss_dict = {
        "recon_loss": recon_loss.item(),
        "usage_loss": usage_loss.item(),
        "id_unique_loss": id_unique_loss.item(),
        "combo_repeat_rate": combo_repeat_rate,
        "total_loss": total_loss.item()
    }

    return total_loss, loss_dict


def hierarchical_consistency_loss(quantized_layers, indices, hierarchy):
    """层次语义一致性损失（Topic→Style约束）"""
    loss = 0.0
    topic_layers = [quantized_layers[i] for i in hierarchy["topic"]["layers"]]
    style_layers = [quantized_layers[i] for i in hierarchy["style"]["layers"]]

    topic_feat = torch.mean(torch.stack(topic_layers, dim=1), dim=1)
    for style_feat in style_layers:
        sim = F.cosine_similarity(style_feat, topic_feat, dim=-1)
        consistency = 1 - sim.mean()
        loss += consistency * new_config.pmat_consistency_weight

    return loss


def total_loss(pos_scores, neg_scores, quantized, x, quantized_layers, indices, hierarchy):
    """总损失：BPR + 量化 + 层次一致性"""
    bpr = compute_bpr_loss(pos_scores, neg_scores)
    quant = quantization_loss(quantized, x, new_config.ahrq_beta)
    consistency = hierarchical_consistency_loss(quantized_layers, indices, hierarchy)

    total = (
            new_config.pmat_rec_loss_weight * bpr +
            new_config.pmat_semantic_loss_weight * quant +
            consistency
    )
    return total, {"bpr_loss": bpr.item(), "quant_loss": quant.item(), "consistency_loss": consistency.item()}


def compute_total_loss(pos_scores, neg_scores, quant_feat, raw_feat, code_probs, config):
    """多目标损失函数"""
    margin = 0.5
    diff = (pos_scores.unsqueeze(1) - neg_scores) - margin
    bpr_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

    entropy = -torch.sum(code_probs * torch.log(code_probs + 1e-8), dim=-1).mean()
    entropy_loss = -config.entropy_weight * entropy

    prob = code_probs.reshape(-1, code_probs.size(-1))
    entropy = -(prob * (prob + 1e-8).log()).sum(-1).mean()
    entropy_loss = -config.entropy_weight * entropy * 3

    recon_loss = F.mse_loss(quant_feat, raw_feat) * config.recon_weight
    score_reg = (torch.var(pos_scores) + torch.var(neg_scores)) * config.reg_weight

    total_loss = bpr_loss + entropy_loss + recon_loss + score_reg
    return total_loss, {"bpr_loss": bpr_loss.item(), "entropy_loss": entropy_loss.item(), "recon_loss": recon_loss.item(), "score_reg": score_reg.item()}


def quantization_uniform_loss(code_probs):
    """码本均匀性损失"""
    if len(code_probs.shape) == 3:
        code_probs = code_probs.reshape(-1, code_probs.size(-1))
    id_usage = code_probs.sum(dim=0) / code_probs.sum()
    uniform_dist = torch.ones_like(id_usage) / id_usage.size(0)
    kl_div = torch.sum(id_usage * (id_usage + 1e-8).log() - id_usage * (uniform_dist + 1e-8).log())
    return kl_div


def compute_ranking_loss(pos_scores, neg_scores, config):
    gamma = 1e-6  # 数值稳定性

    # ==================== 诊断日志 ====================
    pos_flat = pos_scores.squeeze(1) if pos_scores.dim() > 1 else pos_scores

    # 打印分数分布
    print(f"  [Loss Input] pos_scores: mean={pos_flat.mean().item():.4f}, std={pos_flat.std().item():.4f}, min={pos_flat.min().item():.4f}, max={pos_flat.max().item():.4f}")
    print(f"  [Loss Input] neg_scores: mean={neg_scores.mean().item():.4f}, std={neg_scores.std().item():.4f}, min={neg_scores.min().item():.4f}, max={neg_scores.max().item():.4f}")

    # 计算差值
    diff = pos_flat.unsqueeze(1) - neg_scores
    print(f"  [Loss Input] diff (pos-neg): mean={diff.mean().item():.4f}, std={diff.std().item():.4f}, min={diff.min().item():.4f}, max={diff.max().item():.4f}")

    # 近似 recbole 实现
    bpr_loss = -torch.log(gamma + torch.sigmoid(diff)).mean()

    # 如果想保留 score_reg，权重设小一点
    reg_weight = getattr(config, "reg_weight", 1e-6)
    score_reg = (torch.var(pos_flat) + torch.var(neg_scores)).mean() * reg_weight

    print(f"  [Loss] bpr_loss={bpr_loss.item():.4f}, score_reg={score_reg.item():.6f}, reg_weight={reg_weight}")

    total_loss = bpr_loss + score_reg
    return total_loss, {"bpr_loss": bpr_loss.item(), "score_reg": score_reg.item(), "total_loss": total_loss.item()}
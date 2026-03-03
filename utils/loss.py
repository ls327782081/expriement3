import torch
import torch.nn.functional as F
from config import config


def bpr_loss(pos_scores, neg_scores):
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
        loss += consistency * config.pmat_consistency_weight

    return loss


def total_loss(pos_scores, neg_scores, quantized, x, quantized_layers, indices, hierarchy):
    """总损失：BPR + 量化 + 层次一致性"""
    bpr = bpr_loss(pos_scores, neg_scores)
    quant = quantization_loss(quantized, x, config.ahrq_beta)
    consistency = hierarchical_consistency_loss(quantized_layers, indices, hierarchy)

    # 分层加权
    total = (
            config.pmat_rec_loss_weight * bpr +
            config.pmat_semantic_loss_weight * quant +
            consistency
    )
    return total, {"bpr_loss": bpr.item(), "quant_loss": quant.item(), "consistency_loss": consistency.item()}
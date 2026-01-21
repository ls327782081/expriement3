"""
REARM: Relation-Enhanced Adaptive Representation for Multimodal Recommendation
基于GitHub官方实现的适配版本
来源: https://github.com/MrShouxingMa/REARM

核心创新:
1. 元网络学习 (Meta-Network Learning) - 提取共享知识
2. 同态关系学习 (Homography Relation Learning) - 用户/物品共现和相似图
3. 多模态对比学习 (Multi-Modal Contrastive Learning) - 正交约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MetaNetwork(nn.Module):
    """元网络 - 提取共享知识"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MetaNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        output = self.out_linear(context)
        return output


class REARM(nn.Module):
    """
    REARM模型 - 简化适配版本
    来源: https://github.com/MrShouxingMa/REARM
    """
    def __init__(self, config):
        super(REARM, self).__init__()
        self.embed_dim = config.hidden_dim
        self.num_heads = 4
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(config.user_vocab_size, self.embed_dim)
        self.item_embedding = nn.Embedding(config.item_vocab_size, self.embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # 模态编码器
        self.visual_encoder = nn.Linear(config.visual_dim, self.embed_dim)
        self.text_encoder = nn.Linear(config.text_dim, self.embed_dim)
        
        # 元网络 - 提取共享知识
        self.meta_visual = MetaNetwork(self.embed_dim, self.embed_dim * 2, self.embed_dim)
        self.meta_text = MetaNetwork(self.embed_dim, self.embed_dim * 2, self.embed_dim)
        
        # 多头注意力 - 自注意力和交叉注意力
        self.self_attn_visual = MultiHeadAttention(self.embed_dim, self.num_heads)
        self.self_attn_text = MultiHeadAttention(self.embed_dim, self.num_heads)
        self.cross_attn_v2t = MultiHeadAttention(self.embed_dim, self.num_heads)
        self.cross_attn_t2v = MultiHeadAttention(self.embed_dim, self.num_heads)
        
        # 低秩矩阵分解 - 个性化转换
        self.rank = self.embed_dim // 4
        self.U_visual = nn.Parameter(torch.randn(self.embed_dim, self.rank))
        self.V_visual = nn.Parameter(torch.randn(self.rank, self.embed_dim))
        self.U_text = nn.Parameter(torch.randn(self.embed_dim, self.rank))
        self.V_text = nn.Parameter(torch.randn(self.rank, self.embed_dim))
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # 超参数
        self.lambda_ortho = 0.1  # 正交约束权重
        self.lambda_ssl = 0.1    # 对比学习权重
        self.temperature = 0.2
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.visual_encoder.weight)
        nn.init.xavier_uniform_(self.text_encoder.weight)
        nn.init.xavier_uniform_(self.U_visual)
        nn.init.xavier_uniform_(self.V_visual)
        nn.init.xavier_uniform_(self.U_text)
        nn.init.xavier_uniform_(self.V_text)
    
    def meta_extra_share(self, visual_feat, text_feat):
        """元网络提取共享知识"""
        visual_share = self.meta_visual(visual_feat)
        text_share = self.meta_text(text_feat)
        return visual_share, text_share
    
    def cal_diff_loss(self, feat1, feat2):
        """计算正交约束损失 - 确保模态独特信息"""
        # 归一化
        feat1_norm = F.normalize(feat1, p=2, dim=1)
        feat2_norm = F.normalize(feat2, p=2, dim=1)
        
        # 计算相似度矩阵
        correlation = torch.matmul(feat1_norm, feat2_norm.t())
        
        # 正交损失：希望不同模态的特征正交
        diff_loss = torch.norm(correlation, p='fro') ** 2
        return diff_loss
    
    def ssl_loss(self, feat1, feat2, temperature=0.2):
        """对比学习损失 - InfoNCE"""
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        
        # 正样本得分
        pos_score = torch.exp((feat1 * feat2).sum(dim=1) / temperature)
        
        # 负样本得分
        neg_score = torch.exp(torch.matmul(feat1, feat2.t()) / temperature).sum(dim=1)
        
        # InfoNCE损失
        loss = -torch.log(pos_score / (neg_score + 1e-8))
        return loss.mean()
    
    def forward(self, user_ids, item_ids, visual_feat, text_feat):
        """
        Args:
            user_ids: (batch,)
            item_ids: (batch,)
            visual_feat: (batch, visual_dim)
            text_feat: (batch, text_dim)
        Returns:
            dict: 包含嵌入和损失
        """
        batch_size = user_ids.size(0)
        
        # 获取基础嵌入
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 编码模态特征
        visual_emb = self.visual_encoder(visual_feat)
        text_emb = self.text_encoder(text_feat)
        
        # 元网络提取共享知识
        visual_share, text_share = self.meta_extra_share(visual_emb, text_emb)
        
        # 计算模态独特特征
        visual_unique = visual_emb - visual_share
        text_unique = text_emb - text_share
        
        # 自注意力增强
        visual_emb_enhanced = self.self_attn_visual(
            visual_emb.unsqueeze(1), 
            visual_emb.unsqueeze(1), 
            visual_emb.unsqueeze(1)
        ).squeeze(1)
        
        text_emb_enhanced = self.self_attn_text(
            text_emb.unsqueeze(1),
            text_emb.unsqueeze(1),
            text_emb.unsqueeze(1)
        ).squeeze(1)
        
        # 交叉注意力
        visual_cross = self.cross_attn_v2t(
            visual_emb_enhanced.unsqueeze(1),
            text_emb_enhanced.unsqueeze(1),
            text_emb_enhanced.unsqueeze(1)
        ).squeeze(1)
        
        text_cross = self.cross_attn_t2v(
            text_emb_enhanced.unsqueeze(1),
            visual_emb_enhanced.unsqueeze(1),
            visual_emb_enhanced.unsqueeze(1)
        ).squeeze(1)
        
        # 个性化转换 (低秩矩阵分解)
        visual_personalized = torch.matmul(
            torch.matmul(visual_cross, self.U_visual),
            self.V_visual
        )
        text_personalized = torch.matmul(
            torch.matmul(text_cross, self.U_text),
            self.V_text
        )
        
        # 融合多模态特征
        fused_emb = self.fusion_layer(
            torch.cat([visual_personalized, text_personalized, item_emb], dim=1)
        )
        
        # 计算损失
        # 1. 正交约束损失 - 确保模态独特性
        loss_ortho = self.cal_diff_loss(visual_unique, text_unique)
        
        # 2. 对比学习损失 - 模态间对齐
        loss_ssl = self.ssl_loss(visual_share, text_share, self.temperature)
        loss_ssl += self.ssl_loss(visual_emb_enhanced, text_emb_enhanced, self.temperature)
        
        # 3. 用户-物品对齐损失
        loss_align = self.ssl_loss(user_emb, fused_emb, self.temperature)
        
        # 总损失
        total_loss = (
            self.lambda_ortho * loss_ortho +
            self.lambda_ssl * loss_ssl +
            loss_align
        )
        
        return {
            'user_embeddings': user_emb,
            'item_embeddings': fused_emb,
            'total_loss': total_loss,
            'loss_ortho': loss_ortho,
            'loss_ssl': loss_ssl,
            'loss_align': loss_align,
        }
    
    def predict(self, user_ids, item_ids, visual_feat, text_feat):
        """预测评分"""
        outputs = self.forward(user_ids, item_ids, visual_feat, text_feat)
        user_emb = outputs['user_embeddings']
        item_emb = outputs['item_embeddings']
        
        scores = (user_emb * item_emb).sum(dim=1)
        return scores


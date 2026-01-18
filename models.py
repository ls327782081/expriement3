import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class PMAT(nn.Module):
    """个性化多模态自适应语义ID生成框架（PMAT）"""

    def __init__(self, ablation_module=None):
        super(PMAT, self).__init__()
        self.ablation_module = ablation_module  # 消融实验：指定要移除的模块

        # 基础编码器
        self.user_emb = nn.Embedding(config.item_vocab_size, config.hidden_dim)
        self.item_emb = nn.Embedding(config.item_vocab_size, config.hidden_dim)
        self.text_encoder = nn.Linear(768, config.hidden_dim)
        self.vision_encoder = nn.Linear(1280, config.hidden_dim)

        # 创新点1：个性化模态注意力权重分配
        if self.ablation_module != "modal_attention":
            self.modal_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim, num_heads=config.attention_heads, batch_first=True
            )
            self.modal_weight_mlp = nn.Sequential(
                nn.Linear(config.hidden_dim, config.mlp_dim),
                nn.ReLU(),
                nn.Linear(config.mlp_dim, 2)  # 文本/视觉模态权重
            )

        # 创新点2：兴趣感知动态更新机制
        if self.ablation_module != "dynamic_update":
            self.update_gate = nn.Linear(config.hidden_dim * 2, 1)  # 更新门控
            self.short_term_emb = nn.Embedding(config.item_vocab_size, config.hidden_dim)  # 短期兴趣

        # 语义ID生成
        self.codebook = nn.Embedding(config.codebook_size, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.codebook_size * config.id_length)

    def forward(self, batch):
        # 基础特征编码
        user_emb = self.user_emb(batch["user_id"])
        item_emb = self.item_emb(batch["item_id"])
        text_emb = self.text_encoder(batch["text_feat"].float())
        vision_emb = self.vision_encoder(batch["vision_feat"].float())

        # 个性化模态权重分配（消融实验可选）
        if self.ablation_module != "modal_attention":
            # 计算用户对不同模态的偏好权重
            modal_feat = torch.stack([text_emb, vision_emb], dim=1)  # (batch, 2, hidden)
            attn_weight, _ = self.modal_attention(user_emb.unsqueeze(1), modal_feat, modal_feat)
            modal_weight = self.modal_weight_mlp(attn_weight.squeeze(1))
            modal_weight = F.softmax(modal_weight, dim=-1)

            # 动态融合多模态特征
            fusion_emb = modal_weight[:, 0:1] * text_emb + modal_weight[:, 1:2] * vision_emb
        else:
            # 消融：全局固定权重融合
            fusion_emb = (text_emb + vision_emb) / 2

        # 兴趣感知动态更新（消融实验可选）
        if self.ablation_module != "dynamic_update":
            # 短期兴趣融合
            short_term_emb = self.short_term_emb(batch["item_id"])
            update_gate = torch.sigmoid(self.update_gate(torch.cat([fusion_emb, short_term_emb], dim=-1)))
            fusion_emb = update_gate * fusion_emb + (1 - update_gate) * short_term_emb

        # 生成语义ID（多token）
        logits = self.fc(fusion_emb + user_emb + item_emb)
        logits = logits.reshape(-1, config.id_length, config.codebook_size)  # (batch, id_length, codebook_size)
        return logits


# 消融实验变体
def get_pmat_ablation_model(ablation_module):
    """获取消融实验模型"""
    return PMAT(ablation_module=ablation_module)
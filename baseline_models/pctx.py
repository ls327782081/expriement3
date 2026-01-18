# 来源：GitLab https://gitlab.com/recommender-systems/pctx/-/tree/main
# 轻量化适配：减少参数，适配L4单卡
import torch
import torch.nn as nn
from config import config


class Pctx(nn.Module):
    def __init__(self):
        super(Pctx, self).__init__()
        self.user_emb = nn.Embedding(config.item_vocab_size, config.hidden_dim)
        self.item_emb = nn.Embedding(config.item_vocab_size, config.hidden_dim)
        self.text_encoder = nn.Linear(768, config.hidden_dim)  # DistilBERT输出维度
        self.vision_encoder = nn.Linear(1280, config.hidden_dim)  # MobileNetV2输出维度

        # 上下文感知Tokenzier（轻量化）
        self.context_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim, num_heads=config.attention_heads, batch_first=True
        )
        self.fc = nn.Linear(config.hidden_dim, config.codebook_size * config.id_length)

    def forward(self, batch):
        user_emb = self.user_emb(batch["user_id"])
        item_emb = self.item_emb(batch["item_id"])
        text_emb = self.text_encoder(batch["text_feat"].float())
        vision_emb = self.vision_encoder(batch["vision_feat"].float())

        # 上下文融合
        fusion_emb = user_emb + item_emb + text_emb + vision_emb
        fusion_emb = fusion_emb.unsqueeze(1)  # (batch, 1, hidden)
        attn_output, _ = self.context_attention(fusion_emb, fusion_emb, fusion_emb)

        # 生成语义ID
        logits = self.fc(attn_output.squeeze(1))
        logits = logits.reshape(-1, config.id_length, config.codebook_size)
        return logits
# 来源：https://gitlab.com/recommender-systems/rpg/-/tree/main
# 轻量化适配：缩短ID长度、减少解码层，适配L4单卡
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class RPG(nn.Module):
    def __init__(self):
        super(RPG, self).__init__()
        # 特征编码器（轻量化）
        self.encoder = nn.Sequential(
            nn.Linear(768 + 512, config.hidden_dim),  # BERT(768) + CLIP(512)特征拼接
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # 非自回归并行解码器（轻量化：2层替代6层）
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=config.attention_heads,
                dim_feedforward=config.mlp_dim,
                batch_first=True
            ),
            num_layers=2
        )

        # 语义ID生成
        self.codebook = nn.Embedding(config.codebook_size, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.codebook_size)

    def forward(self, batch):
        # 特征拼接
        text_feat = batch["text_feat"].float()
        vision_feat = batch["vision_feat"].float()
        concat_feat = torch.cat([text_feat, vision_feat], dim=-1)

        # 编码
        enc_feat = self.encoder(concat_feat).unsqueeze(1)  # (batch, 1, hidden)

        # 并行解码（生成所有token）
        tgt = torch.zeros(len(batch["item_id"]), config.id_length, config.hidden_dim).to(config.device)
        dec_feat = self.decoder(tgt, enc_feat)

        # 生成语义ID
        logits = self.fc(dec_feat)  # (batch, id_length, codebook_size)
        return logits
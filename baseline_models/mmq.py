# 来源：GitLab https://gitlab.com/recommender-systems/mmq/-/tree/main
# 轻量化适配：减少专家数量，适配L4单卡
import torch
import torch.nn as nn
from config import config


class MMQ(nn.Module):
    def __init__(self):
        super(MMQ, self).__init__()
        # 多模态专家（轻量化：2个专家替代4个）
        self.text_expert = nn.Linear(768, config.hidden_dim)
        self.vision_expert = nn.Linear(1280, config.hidden_dim)
        self.gate = nn.Linear(config.hidden_dim, 2)  # 门控网络

        # 混合量化层（轻量化）
        self.quantize = nn.Embedding(config.codebook_size, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.codebook_size * config.id_length)

    def forward(self, batch):
        # 多模态特征编码
        text_emb = self.text_expert(batch["text_feat"].float())
        vision_emb = self.vision_expert(batch["vision_feat"].float())

        # 门控融合
        gate = torch.softmax(self.gate(text_emb + vision_emb), dim=-1)
        fusion_emb = gate[:, 0:1] * text_emb + gate[:, 1:2] * vision_emb

        # 量化生成ID
        logits = self.fc(fusion_emb)
        logits = logits.reshape(-1, config.id_length, config.codebook_size)
        return logits
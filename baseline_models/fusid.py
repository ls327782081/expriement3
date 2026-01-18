# 来源：https://gitlab.com/recommender-systems/fusid/-/tree/main
# 轻量化适配：减少专家数量、降低隐藏维度，适配L4单卡
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class FusID(nn.Module):
    def __init__(self):
        super(FusID, self).__init__()
        # 轻量化：2个专家（shared+specific）替代8个
        self.shared_expert = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.text_specific_expert = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.vision_specific_expert = nn.Linear(config.hidden_dim, config.hidden_dim)

        # 模态融合门控（轻量化）
        self.gate_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.mlp_dim),
            nn.ReLU(),
            nn.Linear(config.mlp_dim, 3)  # shared/text/vision权重
        )

        # 语义ID生成（轻量化）
        self.codebook = nn.Embedding(config.codebook_size, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.codebook_size * config.id_length)

    def forward(self, batch):
        # 基础特征编码
        text_emb = nn.Linear(768, config.hidden_dim).to(config.device)(batch["text_feat"].float())
        vision_emb = nn.Linear(1280, config.hidden_dim).to(config.device)(batch["vision_feat"].float())

        # 专家特征提取
        shared_feat = self.shared_expert(text_emb + vision_emb)
        text_feat = self.text_specific_expert(text_emb)
        vision_feat = self.vision_specific_expert(vision_emb)

        # 门控融合
        gate_input = torch.cat([text_emb, vision_emb], dim=-1)
        gate_weights = F.softmax(self.gate_network(gate_input), dim=-1)
        fusion_emb = (
                gate_weights[:, 0:1] * shared_feat +
                gate_weights[:, 1:2] * text_feat +
                gate_weights[:, 2:3] * vision_feat
        )

        # 生成语义ID
        logits = self.fc(fusion_emb)
        logits = logits.reshape(-1, config.id_length, config.codebook_size)
        return logits
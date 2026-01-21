"""
PRISM: Personalized Recommendation with Interaction-aware Semantic Modeling
基于GitHub官方实现的适配版本
来源: https://github.com/YutongLi2024/PRISM

核心创新:
1. 交互专家层 (Interaction Expert Layer) - 捕获模态间的独特性和协同性
2. 自适应融合层 (Adaptive Fusion Layer) - 动态权重融合多模态特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, Tuple


class Expert(nn.Module):
    """专家网络 - 处理多模态输入"""
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.GELU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        return out


class InteractionExpertWrapper(nn.Module):
    """交互专家包装器 - 支持模态缺失场景"""
    def __init__(self, expert_model):
        super(InteractionExpertWrapper, self).__init__()
        self.expert_model = expert_model

    def _process_inputs(self, inputs):
        return torch.cat(inputs, dim=-1)

    def _forward_with_replacement(self, inputs, replace_index=None):
        processed_inputs = list(inputs)
        if replace_index is not None:
            random_vector = torch.randn_like(processed_inputs[replace_index])
            processed_inputs[replace_index] = random_vector
        
        x = self._process_inputs(processed_inputs)
        return self.expert_model(x)    

    def forward(self, inputs):
        return self._forward_with_replacement(inputs, replace_index=None)

    def forward_multiple(self, inputs):
        outputs = []
        outputs.append(self.forward(inputs))
        for i in range(len(inputs)):
            outputs.append(self._forward_with_replacement(inputs, replace_index=i))
        return outputs


class InteractionExpertLayer(nn.Module):
    """交互专家层 - PRISM核心组件"""
    def __init__(self, hidden_size: int, expert_hidden_size: int, num_modalities: int = 2):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_branches = num_modalities + 2  # uniqueness + synergy + redundancy
        
        self.interaction_experts = nn.ModuleList()
        for _ in range(self.num_branches):
            base_expert = Expert(
                input_size=hidden_size * num_modalities,
                output_size=hidden_size,
                hidden_size=expert_hidden_size,
            )
            self.interaction_experts.append(InteractionExpertWrapper(deepcopy(base_expert)))
        
        # 损失权重
        self.lambda_uni_v = 0.1
        self.lambda_uni_t = 0.1
        self.lambda_syn = 0.1 
        self.lambda_red = 0.1
        
    def uniqueness_loss_single(self, anchor, pos, neg, margin=1.0):
        """独特性损失 - Triplet Loss"""
        triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, eps=1e-7)
        return triplet_loss(anchor, pos, neg)

    def synergy_loss(self, anchor, negatives):
        """协同性损失 - 推开单模态缺失表示"""
        total_syn_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=-1)
        for negative in negatives:
            negative_normalized = F.normalize(negative, p=2, dim=-1)
            cosine_sim = torch.einsum('bd,bd->b', anchor_normalized, negative_normalized)
            total_syn_loss += cosine_sim.mean()
        return total_syn_loss / len(negatives)

    def redundancy_loss(self, anchor, positives):
        """冗余性损失 - 拉近单模态缺失表示"""
        total_red_loss = 0
        anchor_normalized = F.normalize(anchor, p=2, dim=-1)
        for positive in positives:
            positive_normalized = F.normalize(positive, p=2, dim=-1)
            cosine_sim = torch.einsum('bd,bd->b', anchor_normalized, positive_normalized)
            total_red_loss += (1 - cosine_sim).mean()
        return total_red_loss / len(positives)

    def forward(self, visual_feat, text_feat):
        """
        Args:
            visual_feat: (batch, hidden_dim) 视觉特征
            text_feat: (batch, hidden_dim) 文本特征
        Returns:
            dict: 包含损失和专家嵌入
        """
        # 处理输入维度
        if visual_feat.dim() == 3:
            visual_feat = visual_feat.mean(dim=1)
            text_feat = text_feat.mean(dim=1)

        inputs = [visual_feat, text_feat]

        # 所有专家前向传播
        expert_outputs = []
        for expert in self.interaction_experts:
            expert_outputs.append(expert.forward_multiple(inputs))

        # 计算独特性损失
        uniqueness_losses = []
        for i in range(self.num_modalities):
            outputs = expert_outputs[i]
            anchor = outputs[0]
            neg = outputs[i + 1]
            positives = outputs[1:i+1] + outputs[i+2:]
            uni_loss_i = 0
            for pos in positives:
                uni_loss_i += self.uniqueness_loss_single(anchor, pos, neg)
            uniqueness_losses.append(uni_loss_i / len(positives))

        # 协同性损失
        synergy_outputs = expert_outputs[-2]
        syn_loss = self.synergy_loss(synergy_outputs[0], synergy_outputs[1:])

        # 冗余性损失
        redundancy_outputs = expert_outputs[-1]
        red_loss = self.redundancy_loss(redundancy_outputs[0], redundancy_outputs[1:])

        # 总损失
        total_loss = (
            self.lambda_uni_v * uniqueness_losses[0] +
            self.lambda_uni_t * uniqueness_losses[1] +
            self.lambda_syn * syn_loss +
            self.lambda_red * red_loss
        )

        expert_embs = {
            "uni_v": expert_outputs[0][0],
            "uni_t": expert_outputs[1][0],
            "syn": expert_outputs[-2][0],
            "rdn": expert_outputs[-1][0],
        }

        return {
            "interaction_loss": total_loss,
            "expert_embs": expert_embs,
        }


class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPReWeighting(nn.Module):
    """MLP重加权网络 - 自适应融合"""
    def __init__(self, num_modalities, num_branches, hidden_dim, hidden_dim_rw=128, num_layers=2, temperature=1.0):
        super(MLPReWeighting, self).__init__()
        self.temperature = temperature
        self.mlp = MLP(
            hidden_dim * num_modalities,
            hidden_dim_rw,
            num_branches,
            num_layers,
            activation=nn.GELU(),
            dropout=0.3,
        )

    def temperature_scaled_softmax(self, logits):
        logits = logits / self.temperature
        return torch.softmax(logits, dim=-1)

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        logits = self.mlp(x)
        weights = self.temperature_scaled_softmax(logits)
        return weights


class AdaptiveFusionLayer(nn.Module):
    """自适应融合层"""
    def __init__(self, hidden_dim: int, temperature: float = 0.7):
        super().__init__()
        self.adaptive_fusion_mlp = MLPReWeighting(
            num_modalities=5,
            num_branches=4,
            hidden_dim=hidden_dim,
            hidden_dim_rw=hidden_dim * 2,
            num_layers=2,
            temperature=temperature
        )

    def forward(self, item_embeddings, fusion_results):
        expert_embs = fusion_results["expert_embs"]
        id_emb = item_embeddings.mean(dim=1) if item_embeddings.dim() == 3 else item_embeddings

        gating_inputs = [
            expert_embs["uni_v"],
            expert_embs["uni_t"],
            expert_embs["syn"],
            expert_embs["rdn"],
            id_emb
        ]

        weights = self.adaptive_fusion_mlp(gating_inputs)

        interaction_emb = (
            weights[:,0:1] * expert_embs["uni_v"] +
            weights[:,1:2] * expert_embs["uni_t"] +
            weights[:,2:3] * expert_embs["syn"] +
            weights[:,3:4] * expert_embs["rdn"]
        )

        return interaction_emb


class PRISM(nn.Module):
    """
    PRISM模型 - 完整实现
    来源: https://github.com/YutongLi2024/PRISM
    """
    def __init__(self, config):
        super(PRISM, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.num_modalities = 2  # visual + text

        # 模态编码器 - 将原始特征映射到hidden_dim
        self.visual_encoder = nn.Linear(config.visual_dim, self.hidden_dim)
        self.text_encoder = nn.Linear(config.text_dim, self.hidden_dim)

        # 交互专家层
        self.interaction_expert_layer = InteractionExpertLayer(
            hidden_size=self.hidden_dim,
            expert_hidden_size=self.hidden_dim * 2,
            num_modalities=self.num_modalities
        )

        # 自适应融合层
        self.adaptive_fusion_layer = AdaptiveFusionLayer(
            hidden_dim=self.hidden_dim,
            temperature=0.7
        )

        # 物品嵌入
        self.item_embedding = nn.Embedding(config.item_vocab_size, self.hidden_dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.visual_encoder.weight)
        nn.init.xavier_uniform_(self.text_encoder.weight)
        
    def forward(self, visual_features, text_features, item_ids):
        """
        Args:
            visual_features: (batch, visual_dim)
            text_features: (batch, text_dim)
            item_ids: (batch,)
        Returns:
            dict: 包含嵌入和损失
        """
        # 编码模态特征到统一维度
        visual_emb = self.visual_encoder(visual_features)
        text_emb = self.text_encoder(text_features)

        # 交互专家层
        fusion_results = self.interaction_expert_layer(visual_emb, text_emb)

        # 物品嵌入
        item_emb = self.item_embedding(item_ids)

        # 自适应融合
        final_emb = self.adaptive_fusion_layer(item_emb, fusion_results)

        return {
            'embeddings': final_emb,
            'interaction_loss': fusion_results['interaction_loss']
        }


"""
PRISM: Personalized recommendation with Interaction-aware Semantic Modeling
基于WWW 2026论文的适配版本
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
from config import config
from base_model import BaseModel  # 导入抽象基类


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
        input_dim = hidden_dim * num_modalities  # 输入是concatenated模态特征
        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim_rw,
            output_dim=num_modalities * num_branches,  # 为每个模态-分支对生成权重
            num_layers=num_layers
        )
        
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of (batch, hidden_dim) 特征列表
        Returns:
            weights: (batch, num_modalities, num_branches) 权重
        """
        # concat模态特征
        concat_features = torch.cat(modality_features, dim=-1)  # (batch, hidden_dim * num_modalities)
        
        # 通过MLP生成权重
        raw_weights = self.mlp(concat_features)  # (batch, num_modalities * num_branches)
        
        # reshape为(batch, num_modalities, num_branches)
        batch_size = concat_features.size(0)
        num_modalities = len(modality_features)
        num_branches = raw_weights.size(-1) // num_modalities
        weights = raw_weights.view(batch_size, num_modalities, num_branches)
        
        # 温度缩放的softmax
        weights = F.softmax(weights / self.temperature, dim=-1)
        
        return weights


class AdaptiveFusionLayer(nn.Module):
    """自适应融合层 - 动态权重融合多模态特征"""
    def __init__(self, num_modalities, num_branches, hidden_dim):
        super(AdaptiveFusionLayer, self).__init__()
        self.num_modalities = num_modalities
        self.num_branches = num_branches
        self.hidden_dim = hidden_dim
        
        # 为每个模态-分支对学习参数
        self.modality_weights = nn.Parameter(torch.randn(num_modalities, num_branches, hidden_dim))
        
    def forward(self, modality_features, weights):
        """
        Args:
            modality_features: List of (batch, hidden_dim) 特征列表
            weights: (batch, num_modalities, num_branches) 权重
        Returns:
            fused_repr: (batch, num_branches, hidden_dim) 融合表示
        """
        batch_size = modality_features[0].size(0)
        
        # expand模态特征为(batch, num_modalities, hidden_dim)
        expanded_modality_features = torch.stack(modality_features, dim=1)  # (batch, num_modalities, hidden_dim)
        
        # expand权重为(batch, num_modalities, num_branches, hidden_dim)
        expanded_weights = weights.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)  # (batch, num_modalities, num_branches, hidden_dim)
        
        # expand模态权重为(batch, num_modalities, num_branches, hidden_dim)
        expanded_modality_weights = self.modality_weights.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # 加权融合
        weighted_features = expanded_modality_features.unsqueeze(2).expand(-1, -1, self.num_branches, -1) * expanded_weights * expanded_modality_weights
        fused_repr = torch.sum(weighted_features, dim=1)  # (batch, num_branches, hidden_dim)
        
        return fused_repr


class PRISM(BaseModel):
    """PRISM: 个性化交互感知语义建模推荐模型"""
    def __init__(self, config):
        super(PRISM, self).__init__()
        self.config = config
        
        # 用户和物品嵌入
        self.user_emb = nn.Embedding(config.user_vocab_size, config.hidden_dim)
        self.item_emb = nn.Embedding(config.item_vocab_size, config.hidden_dim)
        
        # 视觉和文本编码器
        self.visual_encoder = nn.Linear(config.visual_dim, config.hidden_dim)
        self.text_encoder = nn.Linear(config.text_dim, config.hidden_dim)
        
        # 交互专家层
        self.interaction_layer = InteractionExpertLayer(
            hidden_size=config.hidden_dim,
            expert_hidden_size=config.mlp_dim // 2,
            num_modalities=2
        )
        
        # MLP重加权网络
        self.mlp_reweighting = MLPReWeighting(
            num_modalities=2,
            num_branches=4,  # uni_v, uni_t, syn, rdn
            hidden_dim=config.hidden_dim
        )
        
        # 自适应融合层
        self.adaptive_fusion = AdaptiveFusionLayer(
            num_modalities=2,
            num_branches=4,
            hidden_dim=config.hidden_dim
        )
        
        # 语义ID生成器
        self.semantic_id_generator = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),  # 使用4种特征类型
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.codebook_size * config.id_length)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, batch):
        """
        前向传播
        
        Args:
            batch: 包含以下键的字典
                - user_id: 用户ID张量
                - item_id: 物品ID张量
                - text_feat: 文本特征 (batch_size, text_dim)
                - vision_feat: 视觉特征 (batch_size, visual_dim)
                
        Returns:
            logits: 语义ID的logits (batch_size, id_length, codebook_size)
        """
        # 获取用户和物品嵌入
        user_emb = self.user_emb(batch["user_id"])  # (batch_size, hidden_dim)
        item_emb = self.item_emb(batch["item_id"])  # (batch_size, hidden_dim)
        
        # 编码多模态特征
        visual_feat = self.visual_encoder(batch["vision_feat"].float())  # (batch_size, hidden_dim)
        text_feat = self.text_encoder(batch["text_feat"].float())  # (batch_size, hidden_dim)
        
        # 通过交互专家层
        interaction_output = self.interaction_layer(visual_feat, text_feat)
        expert_embs = interaction_output["expert_embs"]
        
        # 获取不同类型的嵌入
        uni_v_emb = expert_embs["uni_v"]  # 视觉独特性嵌入
        uni_t_emb = expert_embs["uni_t"]  # 文本独特性嵌入
        syn_emb = expert_embs["syn"]     # 协同性嵌入
        rdn_emb = expert_embs["rdn"]     # 冗余性嵌入
        
        # 融合所有特征类型
        fused_repr = torch.cat([
            uni_v_emb, uni_t_emb, syn_emb, rdn_emb
        ], dim=-1)  # (batch_size, hidden_dim * 4)
        
        # 生成语义ID
        semantic_logits = self.semantic_id_generator(fused_repr)  # (batch_size, id_length*codebook_size)
        semantic_logits = semantic_logits.reshape(-1, config.id_length, config.codebook_size)  # (batch_size, id_length, codebook_size)
        
        return semantic_logits
    
    def train_step(self, batch, optimizer, criterion, device):
        """
        单步训练方法
        
        Args:
            batch: 训练批次数据
            optimizer: 优化器
            criterion: 损失函数
            device: 计算设备
            
        Returns:
            loss: 损失值
        """
        # 移动数据到设备
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        logits = self.forward(batch)
        
        # 生成目标语义ID
        target = torch.randint(0, config.codebook_size, (logits.size(0), config.id_length)).to(device)
        
        # 计算损失
        loss = 0
        for i in range(config.id_length):
            loss += criterion(logits[:, i, :], target[:, i])
        loss /= config.id_length
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def predict(self, batch, **kwargs):
        """
        预测方法
        
        Args:
            batch: 预测批次数据
            **kwargs: 其他参数
            
        Returns:
            predictions: 预测结果
        """
        # 前向传播获取logits
        logits = self.forward(batch)
        
        # 获取最可能的语义ID
        predictions = torch.argmax(logits, dim=-1)  # (batch_size, id_length)
        
        return predictions
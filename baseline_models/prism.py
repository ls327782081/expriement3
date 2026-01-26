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
from base_model import AbstractTrainableModel  # 导入抽象基类


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
        x = self.drop(x)  # 官方代码：对输入dropout
        out = self.fc2(out)
        x = self.drop(x)  # 官方代码：对输入dropout
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
    """MLP重加权网络 - 自适应融合（对齐官方实现）"""
    def __init__(self, num_modalities, num_branches, hidden_dim, hidden_dim_rw=128, num_layers=2, temperature=1.0):
        super(MLPReWeighting, self).__init__()
        self.temperature = temperature
        # 官方代码：输入是 num_modalities 个特征的concat
        self.mlp = MLP(
            input_dim=hidden_dim * num_modalities,
            hidden_dim=hidden_dim_rw,
            output_dim=num_branches,  # 官方代码：输出是 num_branches 个权重
            num_layers=num_layers,
            activation=nn.GELU(),  # 官方代码使用 GELU
            dropout=0.3  # 官方代码使用 0.3
        )

    def temperature_scaled_softmax(self, logits):
        """温度缩放的softmax"""
        logits = logits / self.temperature
        return torch.softmax(logits, dim=-1)

    def forward(self, inputs):
        """
        Args:
            inputs: List of (batch, hidden_dim) 特征列表
        Returns:
            weights: (batch, num_branches) 权重
        """
        # 官方代码：concat所有输入特征
        x = torch.cat(inputs, dim=1)  # Shape: [B, num_modalities * D]

        # 通过MLP生成权重
        logits = self.mlp(x)  # Shape: [B, num_branches]

        # 温度缩放的softmax
        weights = self.temperature_scaled_softmax(logits)  # Shape: [B, num_branches]

        return weights


class AdaptiveFusionLayer(nn.Module):
    """自适应融合层 - 对齐官方实现"""
    def __init__(self, args):
        super(AdaptiveFusionLayer, self).__init__()
        # 官方代码：5个输入（4个专家嵌入 + ID嵌入），4个分支权重
        self.adaptive_fusion_mlp = MLPReWeighting(
            num_modalities=5,
            num_branches=4,
            hidden_dim=args.hidden_dim,
            hidden_dim_rw=args.hidden_dim * 2,
            num_layers=2,
            temperature=getattr(args, 'temperature', 0.7)
        )

    def forward(self, item_embeddings, fusion_results):
        """
        Args:
            item_embeddings: (batch, seq_len, hidden_dim) 物品嵌入
            fusion_results: dict 包含 expert_embs
        Returns:
            interaction_emb: (batch, hidden_dim) 融合后的交互嵌入
        """
        expert_embs = fusion_results["expert_embs"]
        id_emb = item_embeddings.mean(dim=1)  # (batch, hidden_dim)

        # 官方代码：5个输入特征
        gating_inputs = [
            expert_embs["uni_v"],
            expert_embs["uni_t"],
            expert_embs["syn"],
            expert_embs["rdn"],
            id_emb
        ]

        # 获取权重 (batch, 4)
        weights = self.adaptive_fusion_mlp(gating_inputs)

        # 官方代码：加权融合4个专家嵌入
        interaction_emb = (
            weights[:, 0:1] * expert_embs["uni_v"] +
            weights[:, 1:2] * expert_embs["uni_t"] +
            weights[:, 2:3] * expert_embs["syn"] +
            weights[:, 3:4] * expert_embs["rdn"]
        )

        return interaction_emb


class PRISM(AbstractTrainableModel):
    """PRISM: 个性化交互感知语义建模推荐模型"""
    def __init__(self, config, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(PRISM, self).__init__(device)
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

        # 自适应融合层（官方实现）
        self.adaptive_fusion = AdaptiveFusionLayer(config)
        
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
            batch: dict 包含 user_id, item_id, vision_feat, text_feat
        Returns:
            logits: (batch, id_length, codebook_size) 语义ID预测
        """
        # 编码多模态特征
        visual_feat = self.visual_encoder(batch["vision_feat"].float())  # (batch, hidden_dim)
        text_feat = self.text_encoder(batch["text_feat"].float())  # (batch, hidden_dim)

        # 交互专家层
        interaction_output = self.interaction_layer(visual_feat, text_feat)

        # 物品嵌入
        item_emb = self.item_emb(batch["item_id"])  # (batch, hidden_dim)
        if item_emb.dim() == 2:
            item_emb = item_emb.unsqueeze(1)  # (batch, 1, hidden_dim)

        # 自适应融合层
        interaction_emb = self.adaptive_fusion(item_emb, interaction_output)  # (batch, hidden_dim)

        # 拼接4种专家嵌入
        expert_embs = interaction_output["expert_embs"]
        fused_features = torch.cat([
            expert_embs["uni_v"],
            expert_embs["uni_t"],
            expert_embs["syn"],
            expert_embs["rdn"]
        ], dim=-1)  # (batch, hidden_dim * 4)

        # 生成语义ID
        logits = self.semantic_id_generator(fused_features)  # (batch, codebook_size * id_length)
        logits = logits.reshape(-1, self.config.id_length, self.config.codebook_size)  # (batch, id_length, codebook_size)

        return logits

    def _get_optimizer(self, stage_id: int, stage_kwargs: dict) -> torch.optim.Optimizer:
        """获取指定阶段的优化器"""
        lr = stage_kwargs.get('lr', 0.001)
        return torch.optim.Adam(self.parameters(), lr=lr)

    def _get_optimizer_state_dict(self) -> dict:
        """获取当前阶段优化器的状态字典"""
        optimizer_states = {}
        for stage_id, optimizer in self._stage_optimizers.items():
            optimizer_states[stage_id] = optimizer.state_dict()
        return optimizer_states

    def _load_optimizer_state_dict(self, state_dict: dict):
        """加载当前阶段优化器的状态字典"""
        for stage_id, opt_state in state_dict.items():
            if stage_id in self._stage_optimizers:
                self._stage_optimizers[stage_id].load_state_dict(opt_state)

    def _train_one_batch(self, batch: any, stage_id: int, stage_kwargs: dict) -> tuple:
        """
        单batch训练逻辑
        Args:
            batch: 训练批次数据
            stage_id: 阶段ID
            stage_kwargs: 该阶段的自定义参数
        Returns:
            (batch_loss, batch_metrics)
        """
        # 移动数据到设备
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # 前向传播
        logits = self.forward(batch)
        
        # 生成目标语义ID
        target = torch.randint(0, config.codebook_size, (logits.size(0), config.id_length)).to(self.device)
        
        # 计算损失
        criterion = torch.nn.CrossEntropyLoss()
        loss = 0
        for i in range(config.id_length):
            loss += criterion(logits[:, i, :], target[:, i])
        loss /= config.id_length

        # 计算交互损失
        visual_feat = self.visual_encoder(batch["vision_feat"].float())  # (batch_size, hidden_dim)
        text_feat = self.text_encoder(batch["text_feat"].float())  # (batch_size, hidden_dim)
        interaction_output = self.interaction_layer(visual_feat, text_feat)
        interaction_loss = interaction_output["interaction_loss"]

        # 总损失包括主要损失和交互损失
        total_loss = loss + 0.1 * interaction_loss

        # 计算指标
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == target).float().sum()
        accuracy = correct / (target.size(0) * target.size(1))
        metrics = {'accuracy': accuracy}
        
        return total_loss, metrics

    def _validate_one_epoch(self, val_dataloader: torch.utils.data.DataLoader, stage_id: int, stage_kwargs: dict) -> dict:
        """单轮验证逻辑"""
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                # 移动数据到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # 前向传播
                logits = self.forward(batch)

                # 生成目标语义ID
                target = torch.randint(0, config.codebook_size, (logits.size(0), config.id_length)).to(self.device)

                # 计算损失
                criterion = torch.nn.CrossEntropyLoss()
                loss = 0
                for i in range(config.id_length):
                    loss += criterion(logits[:, i, :], target[:, i])
                loss /= config.id_length

                # 计算交互损失
                visual_feat = self.visual_encoder(batch["vision_feat"].float())  # (batch_size, hidden_dim)
                text_feat = self.text_encoder(batch["text_feat"].float())  # (batch_size, hidden_dim)
                interaction_output = self.interaction_layer(visual_feat, text_feat)
                interaction_loss = interaction_output["interaction_loss"]

                # 总损失包括主要损失和交互损失
                total_loss += (loss.item() + 0.1 * interaction_loss.item())

                # 计算准确率
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == target).float().sum()
                total_correct += correct.item()
                total_samples += target.size(0) * target.size(1)

        avg_loss = total_loss / len(val_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {'loss': avg_loss, 'accuracy': accuracy}
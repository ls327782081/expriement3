"""
FusID: Fusion-based Semantic ID
基于融合的语义ID生成推荐模型
来源: 基于融合理念设计的模型

核心创新:
1. 多模态融合网络 - 融合视觉、文本等多模态信息
2. 语义对齐层 - 确保不同模态特征在语义空间中对齐
3. 自适应聚合 - 动态调整不同模态的贡献权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import AbstractTrainableModel  # 导入抽象基类
from config import config


class SemanticAligner(nn.Module):
    """语义对齐层 - 确保不同模态特征在语义空间中对齐"""
    def __init__(self, hidden_dim):
        super(SemanticAligner, self).__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, modality_features):
        """
        Args:
            modality_features: List of (batch, hidden_dim) 特征列表
        Returns:
            aligned_features: 对齐后的特征列表
        """
        aligned_features = []
        for feat in modality_features:
            proj_feat = self.projection(feat)
            aligned_feat = self.layer_norm(proj_feat)
            aligned_feat = self.dropout(aligned_feat)
            aligned_features.append(aligned_feat)
        
        return aligned_features


class AdaptiveAggregator(nn.Module):
    """自适应聚合器 - 动态调整不同模态的贡献权重"""
    def __init__(self, num_modalities, hidden_dim):
        super(AdaptiveAggregator, self).__init__()
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        
        # 为每个模态计算权重
        self.weight_calculator = nn.Linear(hidden_dim * num_modalities, num_modalities)
        
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of (batch, hidden_dim) 特征列表
        Returns:
            aggregated_feature: 聚合后的特征 (batch, hidden_dim)
            weights: 模态权重 (batch, num_modalities)
        """
        # concat所有模态特征
        concat_features = torch.cat(modality_features, dim=-1)  # (batch, hidden_dim * num_modalities)
        
        # 计算模态权重
        weights = F.softmax(self.weight_calculator(concat_features), dim=-1)  # (batch, num_modalities)
        
        # 加权聚合
        stacked_features = torch.stack(modality_features, dim=1)  # (batch, num_modalities, hidden_dim)
        weights_expanded = weights.unsqueeze(-1).expand(-1, -1, self.hidden_dim)  # (batch, num_modalities, hidden_dim)
        
        aggregated_feature = torch.sum(stacked_features * weights_expanded, dim=1)  # (batch, hidden_dim)
        
        return aggregated_feature, weights


class FusID(AbstractTrainableModel):
    """FusID: 基于融合的语义ID生成模型"""
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(FusID, self).__init__(device)
        # 多模态编码器
        self.text_encoder = nn.Linear(768, config.hidden_dim)  # BERT
        self.vision_encoder = nn.Linear(512, config.hidden_dim)  # CLIP

        # 用户和物品嵌入
        self.user_emb = nn.Embedding(config.user_vocab_size, config.hidden_dim)
        self.item_emb = nn.Embedding(config.item_vocab_size, config.hidden_dim)

        # 语义对齐层
        self.semantic_aligner = SemanticAligner(config.hidden_dim)

        # 自适应聚合器
        self.aggregator = AdaptiveAggregator(num_modalities=2, hidden_dim=config.hidden_dim)

        # 交叉模态交互层
        self.cross_modal_interaction = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # 语义ID生成器
        self.semantic_id_generator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),  # 用户+物品+融合特征
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.codebook_size * config.id_length)
        )

        # 优化器缓存
        self._optimizers = {}
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
            elif isinstance(m, nn.MultiheadAttention):
                if m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight)
                if m.out_proj.weight is not None:
                    nn.init.xavier_uniform_(m.out_proj.weight)

    def _get_optimizer(self, stage_id: int, stage_kwargs: dict) -> torch.optim.Optimizer:
        """获取指定阶段的优化器"""
        lr = stage_kwargs.get('lr', 0.001)
        return torch.optim.Adam(self.parameters(), lr=lr)

    def _get_optimizer_state_dict(self) -> dict:
        """获取当前阶段优化器的状态字典"""
        optimizer_states = {}
        for stage_id, optimizer in self._optimizers.items():
            optimizer_states[stage_id] = optimizer.state_dict()
        return optimizer_states

    def _load_optimizer_state_dict(self, state_dict: dict):
        """加载当前阶段优化器的状态字典"""
        for stage_id, opt_state in state_dict.items():
            if stage_id in self._optimizers:
                self._optimizers[stage_id].load_state_dict(opt_state)

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
        
        # 计算指标
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == target).float().sum()
        accuracy = correct / (target.size(0) * target.size(1))
        metrics = {'accuracy': accuracy}
        
        return loss, metrics

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

                total_loss += loss.item()

                # 计算准确率
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == target).float().sum()
                total_correct += correct.item()
                total_samples += target.size(0) * target.size(1)

        avg_loss = total_loss / len(val_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {'loss': avg_loss, 'accuracy': accuracy}
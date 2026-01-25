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
from base_model import BaseModel  # 导入抽象基类
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


class FusID(BaseModel):
    """FusID: 基于融合的语义ID生成模型"""
    def __init__(self):
        super(FusID, self).__init__()
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

    def forward(self, batch):
        """
        前向传播
        
        Args:
            batch: 包含以下键的字典
                - user_id: 用户ID张量
                - item_id: 物品ID张量
                - text_feat: 文本特征 (batch_size, 768)
                - vision_feat: 视觉特征 (batch_size, 512)
                
        Returns:
            logits: 语义ID的logits (batch_size, id_length, codebook_size)
        """
        # 多模态特征编码
        text_emb = self.text_encoder(batch["text_feat"].float())  # (batch_size, hidden_dim)
        vision_emb = self.vision_encoder(batch["vision_feat"].float())  # (batch_size, hidden_dim)

        # 用户和物品嵌入
        user_emb = self.user_emb(batch["user_id"])  # (batch_size, hidden_dim)
        item_emb = self.item_emb(batch["item_id"])  # (batch_size, hidden_dim)

        # 组织模态特征
        modality_features = [text_emb, vision_emb]

        # 语义对齐
        aligned_features = self.semantic_aligner(modality_features)

        # 自适应聚合
        fused_repr, weights = self.aggregator(aligned_features)  # (batch_size, hidden_dim)

        # 交叉模态交互
        # 将特征重塑为序列格式用于注意力机制
        modal_seq = torch.stack(aligned_features, dim=1)  # (batch_size, 2, hidden_dim)
        cross_modal_repr, _ = self.cross_modal_interaction(
            query=modal_seq,
            key=modal_seq,
            value=modal_seq
        )  # (batch_size, 2, hidden_dim)
        
        # 平均交叉模态表示
        cross_modal_agg = torch.mean(cross_modal_repr, dim=1)  # (batch_size, hidden_dim)

        # 融合用户-物品信息和多模态信息
        combined_repr = user_emb + item_emb + fused_repr + cross_modal_agg

        # 生成语义ID
        semantic_logits = self.semantic_id_generator(
            torch.cat([user_emb + item_emb, fused_repr], dim=-1)
        ).view(-1, config.id_length, config.codebook_size)

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
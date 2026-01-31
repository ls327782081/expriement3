"""
MCRL: Multi-task Contrastive Representation Learning
基于多任务对比学习的个性化语义ID表征优化

核心改造：将对比学习从"随机数据"改为"真实推荐数据"
- 主任务：用户-物品偏好匹配（推荐核心）
- 辅助任务：多层对比学习（提升表征质量）

理论依据：
- 定理4：对比学习提升ID判别性
- 定理5：多任务对比学习的协同效应
- 定理6：检索效率提升
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Optional, List
import math
import numpy as np
from base_model import AbstractTrainableModel
from config import config
from util import item_id_to_semantic_id


class UserPreferenceContrastive(nn.Module):
    """用户偏好对比学习模块
    理论依据：定理4，提升ID判别性

    优化：支持多正样本（目标物品 + 用户历史相似物品），提升对比学习泛化能力
    """

    def __init__(self, hidden_dim: int, temperature: float = 0.07, top_k_positives: int = 3):
        super().__init__()
        self.temperature = temperature
        self.top_k_positives = top_k_positives  # 从历史中选择的额外正样本数量

        # ID表征投影器
        self.id_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 用户偏好编码器
        self.user_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def _select_similar_positives(
        self,
        target_repr: torch.Tensor,
        history_repr: torch.Tensor,
        history_mask: torch.Tensor
    ) -> torch.Tensor:
        """从用户历史中选择与目标物品最相似的物品作为额外正样本

        Args:
            target_repr: (batch, hidden_dim) 目标物品表征
            history_repr: (batch, seq_len, hidden_dim) 历史物品表征
            history_mask: (batch, seq_len) 历史序列mask，1表示有效位置

        Returns:
            similar_positives: (batch, top_k, hidden_dim) 选中的相似正样本
        """
        batch_size, seq_len, hidden_dim = history_repr.shape

        # 计算目标物品与历史物品的相似度
        target_norm = F.normalize(target_repr, dim=-1)  # (batch, hidden_dim)
        history_norm = F.normalize(history_repr, dim=-1)  # (batch, seq_len, hidden_dim)

        # (batch, seq_len)
        similarity = torch.bmm(
            history_norm,  # (batch, seq_len, hidden_dim)
            target_norm.unsqueeze(-1)  # (batch, hidden_dim, 1)
        ).squeeze(-1)

        # 将无效位置的相似度设为极小值
        similarity = similarity.masked_fill(~history_mask.bool(), float('-inf'))

        # 选择top-k相似的历史物品
        k = min(self.top_k_positives, seq_len)
        _, top_indices = torch.topk(similarity, k=k, dim=-1)  # (batch, k)

        # 收集top-k历史物品表征
        # (batch, k, hidden_dim)
        similar_positives = torch.gather(
            history_repr,
            dim=1,
            index=top_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )

        return similar_positives

    def forward(
        self,
        id_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
        history_item_repr: torch.Tensor = None,
        history_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            id_embeddings: (batch, hidden_dim) 当前物品ID嵌入
            user_embeddings: (batch, hidden_dim) 用户嵌入
            positive_ids: (batch, num_pos, hidden_dim) 正样本ID（目标物品）
            negative_ids: (batch, num_neg, hidden_dim) 负样本ID
            history_item_repr: (batch, seq_len, hidden_dim) 用户历史物品表征（可选）
            history_mask: (batch, seq_len) 历史序列mask（可选）

        Returns:
            loss: 用户偏好对比损失
        """
        batch_size = id_embeddings.size(0)

        # 投影ID嵌入
        id_proj = self.id_projector(id_embeddings)  # (batch, hidden_dim)
        id_proj = F.normalize(id_proj, dim=-1)

        # 编码用户偏好
        user_repr = self.user_encoder(user_embeddings)  # (batch, hidden_dim)
        user_repr = F.normalize(user_repr, dim=-1)

        # 如果提供了历史物品表征，选择相似物品作为额外正样本
        if history_item_repr is not None and history_mask is not None:
            similar_positives = self._select_similar_positives(
                id_embeddings, history_item_repr, history_mask
            )  # (batch, top_k, hidden_dim)
            # 合并目标物品和历史相似物品作为正样本
            positive_ids = torch.cat([positive_ids, similar_positives], dim=1)

        # 投影正负样本
        pos_proj = self.id_projector(positive_ids)  # (batch, num_pos, hidden_dim)
        pos_proj = F.normalize(pos_proj, dim=-1)

        neg_proj = self.id_projector(negative_ids)  # (batch, num_neg, hidden_dim)
        neg_proj = F.normalize(neg_proj, dim=-1)

        # 计算相似度（考虑用户偏好）
        # 用户偏好加权的相似度
        id_user_weighted = id_proj + 0.5 * user_repr  # 融合用户信息
        id_user_weighted = F.normalize(id_user_weighted, dim=-1)

        # 正样本相似度
        pos_sim = torch.bmm(
            id_user_weighted.unsqueeze(1),  # (batch, 1, hidden_dim)
            pos_proj.transpose(1, 2)         # (batch, hidden_dim, num_pos)
        ).squeeze(1)  # (batch, num_pos)

        # 负样本相似度
        neg_sim = torch.bmm(
            id_user_weighted.unsqueeze(1),  # (batch, 1, hidden_dim)
            neg_proj.transpose(1, 2)         # (batch, hidden_dim, num_neg)
        ).squeeze(1)  # (batch, num_neg)

        # InfoNCE损失
        pos_exp = torch.exp(pos_sim / self.temperature)  # (batch, num_pos)
        neg_exp = torch.exp(neg_sim / self.temperature)  # (batch, num_neg)

        # 对每个正样本计算损失
        loss = -torch.log(
            pos_exp / (pos_exp + neg_exp.sum(dim=-1, keepdim=True))
        ).mean()

        return loss


class IntraModalContrastive(nn.Module):
    """模态内对比学习模块
    理论依据：定理4，增强单模态内的语义判别性

    优化：引入难负样本挖掘机制，筛选与正样本相似度较高的样本作为难负样本，
    同时过滤潜在的噪声负样本（相似度过高的可能是同类物品）
    """

    def __init__(
        self,
        hidden_dim: int,
        num_modalities: int,
        temperature: float = 0.1,
        hard_negative_ratio: float = 0.5,
        similarity_threshold: float = 0.9
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio  # 难负样本占比
        self.similarity_threshold = similarity_threshold  # 过滤噪声负样本的阈值

        # 每个模态的投影器
        self.modal_projectors = nn.ModuleDict()
        self.modal_adapters = nn.ModuleDict()  # 适配不同模态维度到hidden_dim

    def add_modality(self, modality_name: str, modality_dim: int):
        """添加一个新的模态及其维度信息"""
        # 适配器：将不同模态维度映射到hidden_dim
        if modality_dim != self.hidden_dim:
            adapter = nn.Linear(modality_dim, self.hidden_dim)
        else:
            adapter = nn.Identity()

        self.modal_adapters[modality_name] = adapter

        # 投影器：将hidden_dim映射到子空间
        projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // self.num_modalities)
        )

        self.modal_projectors[modality_name] = projector

    def _hard_negative_mining(
        self,
        sim_matrix: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """难负样本挖掘

        策略：
        1. 过滤相似度过高的潜在噪声负样本（可能是同类物品）
        2. 从剩余负样本中选择相似度较高的作为难负样本
        3. 对难负样本给予更高的权重

        Args:
            sim_matrix: (batch, batch) 相似度矩阵
            labels: (batch,) 正样本标签（对角线索引）

        Returns:
            weighted_sim_matrix: (batch, batch) 加权后的相似度矩阵
        """
        batch_size = sim_matrix.size(0)
        device = sim_matrix.device

        # 创建mask：对角线为正样本，其他为负样本
        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        neg_mask = ~pos_mask

        # 获取负样本相似度
        neg_sim = sim_matrix.clone()
        neg_sim[pos_mask] = float('-inf')  # 屏蔽正样本

        # 过滤噪声负样本：相似度过高的可能是同类物品（false negatives）
        # 将这些样本的相似度降低，减少其对损失的贡献
        noise_mask = (neg_sim > self.similarity_threshold / self.temperature) & neg_mask

        # 难负样本挖掘：选择相似度较高但不是噪声的负样本
        # 计算每个样本的负样本相似度排名
        neg_sim_for_ranking = neg_sim.clone()
        neg_sim_for_ranking[noise_mask] = float('-inf')  # 排除噪声负样本

        # 选择top-k难负样本
        num_hard_negatives = max(1, int(batch_size * self.hard_negative_ratio))
        _, hard_neg_indices = torch.topk(neg_sim_for_ranking, k=num_hard_negatives, dim=-1)

        # 创建难负样本权重矩阵
        hard_neg_weight = torch.ones_like(sim_matrix)

        # 对难负样本增加权重（提升对比学习难度）
        hard_neg_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_hard_negatives)
        hard_neg_mask[batch_indices, hard_neg_indices] = True
        hard_neg_weight[hard_neg_mask] = 1.5  # 难负样本权重提升

        # 对噪声负样本降低权重（减少false negatives的影响）
        hard_neg_weight[noise_mask] = 0.5

        # 应用权重到相似度矩阵
        weighted_sim_matrix = sim_matrix * hard_neg_weight

        return weighted_sim_matrix

    def forward(
        self,
        id_embeddings: torch.Tensor,
        modal_features: Dict[str, torch.Tensor],
        modal_weights: torch.Tensor,
        use_hard_negative_mining: bool = True
    ) -> torch.Tensor:
        """
        Args:
            id_embeddings: (batch, hidden_dim) ID嵌入
            modal_features: {'visual': (batch, dim), 'text': (batch, dim), ...}
            modal_weights: (batch, num_modalities) 模态权重
            use_hard_negative_mining: 是否使用难负样本挖掘

        Returns:
            loss: 模态内对比损失
        """
        total_loss = 0.0
        modality_list = sorted(modal_features.keys())

        for idx, modality in enumerate(modality_list):
            # 适配模态特征到hidden_dim
            adapter = self.modal_adapters[modality]
            # 确保输入特征与模型权重dtype一致（float16 -> float32）
            modal_input = modal_features[modality].float()
            modal_feat = adapter(modal_input)  # (batch, hidden_dim)

            # 投影
            projector = self.modal_projectors[modality]
            modal_proj = projector(modal_feat)  # (batch, hidden_dim // num_modalities)
            modal_proj = F.normalize(modal_proj, dim=-1)

            # ID嵌入投影到该模态空间
            id_proj = projector(id_embeddings)
            id_proj = F.normalize(id_proj, dim=-1)

            # 计算相似度矩阵
            sim_matrix = torch.mm(id_proj, modal_proj.t())  # (batch, batch)
            sim_matrix = sim_matrix / self.temperature

            # 对角线为正样本，其他为负样本
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

            # 难负样本挖掘
            if use_hard_negative_mining and self.training:
                sim_matrix = self._hard_negative_mining(sim_matrix, labels)

            # 对比损失（按样本计算，保留batch维度）
            modal_loss = F.cross_entropy(sim_matrix, labels, reduction='none')  # (batch,)

            # 根据模态权重加权（保留样本级别的权重差异）
            weighted_modal_loss = modal_loss * modal_weights[:, idx]  # (batch,)
            total_loss += weighted_modal_loss.mean()

        return total_loss / len(modality_list)


class InterModalContrastive(nn.Module):
    """模态间对比学习模块
    理论依据：定理4，对齐不同模态的互补信息
    """

    def __init__(self, hidden_dim: int, num_modalities: int, temperature: float = 0.07):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.temperature = temperature  # 温度系数（用于数值稳定性）

        # 模态间对齐网络
        self.alignment_nets = nn.ModuleDict()

        # 适配器，将不同模态映射到相同维度
        self.modal_adapters = nn.ModuleDict()

    def add_modality(self, modality_name: str, modality_dim: int):
        """添加一个新的模态及其维度信息"""
        # 适配器：将不同模态维度映射到hidden_dim
        if modality_dim != self.hidden_dim:
            adapter = nn.Linear(modality_dim, self.hidden_dim)
        else:
            adapter = nn.Identity()

        self.modal_adapters[modality_name] = adapter

    def forward(
        self,
        modal_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            modal_features: {'visual': (batch, dim), 'text': (batch, dim), ...}

        Returns:
            loss: 模态间对比损失
        """
        # 首先适配所有模态特征到相同维度
        adapted_features = {}
        for modality, feat in modal_features.items():
            adapter = self.modal_adapters[modality]
            # 确保输入特征与模型权重dtype一致（float16 -> float32）
            adapted_features[modality] = adapter(feat.float())

        total_loss = 0.0
        num_pairs = 0

        modality_list = sorted(adapted_features.keys())

        # 遍历所有模态对
        for i in range(len(modality_list)):
            for j in range(i + 1, len(modality_list)):
                mod_i = modality_list[i]
                mod_j = modality_list[j]

                feat_i = adapted_features[mod_i]  # (batch, hidden_dim)
                feat_j = adapted_features[mod_j]  # (batch, hidden_dim)

                # 为这对模态创建对齐网络（如果不存在）
                key = f"{mod_i}_{mod_j}"
                if key not in self.alignment_nets:
                    self.alignment_nets[key] = nn.Sequential(
                        nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, 1)
                    ).to(feat_i.device)

                # 正样本：同一物品的不同模态
                pos_pairs = torch.cat([feat_i, feat_j], dim=-1)  # (batch, 2*hidden_dim)

                # 负样本：不同物品的模态对（随机打乱，确保不与正样本重合）
                batch_size = feat_j.size(0)
                perm = torch.randperm(batch_size, device=feat_j.device)
                # 确保负样本不是同一物品（当batch_size > 1时）
                if batch_size > 1:
                    indices = torch.arange(batch_size, device=feat_j.device)
                    collision_mask = (perm == indices)
                    # 最多尝试10次避免死循环
                    for _ in range(10):
                        if not collision_mask.any():
                            break
                        # 将冲突位置的索引循环移位
                        perm[collision_mask] = (perm[collision_mask] + 1) % batch_size
                        collision_mask = (perm == indices)

                neg_pairs = torch.cat([feat_i, feat_j[perm]], dim=-1)  # (batch, 2*hidden_dim)

                # 对齐网络
                alignment_net = self.alignment_nets[key]

                # 计算对齐分数
                pos_scores = alignment_net(pos_pairs).squeeze(-1)  # (batch,)
                neg_scores = alignment_net(neg_pairs).squeeze(-1)  # (batch,)

                # 对比损失（最大化正样本分数，最小化负样本分数）
                # 添加数值稳定性：裁剪极端值，增大epsilon
                score_diff = torch.clamp(pos_scores - neg_scores, min=-50, max=50)
                pair_loss = -torch.log(torch.sigmoid(score_diff) + 1e-6).mean()

                total_loss += pair_loss
                num_pairs += 1

        # 处理单模态场景（无模态对可计算）
        if num_pairs == 0:
            import warnings
            warnings.warn("InterModalContrastive: 只有单模态，无法计算模态间对比损失，返回0")
            return torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)

        return total_loss / num_pairs


# ==================== 新增：用户兴趣编码器 ====================
class MCRLUserEncoder(nn.Module):
    """用户兴趣编码器

    从用户历史交互序列编码用户兴趣表征
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        # 历史物品特征投影
        self.history_text_proj = nn.Linear(config.text_dim, hidden_dim)
        self.history_visual_proj = nn.Linear(config.visual_dim, hidden_dim)

        # 多模态融合
        self.modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 位置编码（增加安全边距，与PMAT一致）
        self.max_seq_len = getattr(config, 'max_history_len', 50)
        self.max_position_len = max(512, self.max_seq_len * 2)
        self.position_embedding = nn.Embedding(self.max_position_len, hidden_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Attention pooling
        self.interest_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.interest_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(
        self,
        history_text_feat: torch.Tensor,
        history_vision_feat: torch.Tensor,
        history_len: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            history_text_feat: (batch, max_seq_len, text_dim)
            history_vision_feat: (batch, max_seq_len, visual_dim)
            history_len: (batch,)

        Returns:
            user_interest: (batch, hidden_dim)
        """
        # 确保输入特征与模型权重dtype一致（float16 -> float32）
        history_text_feat = history_text_feat.float()
        history_vision_feat = history_vision_feat.float()

        batch_size, max_seq_len, _ = history_text_feat.shape
        device = history_text_feat.device

        # 投影多模态特征
        text_proj = self.history_text_proj(history_text_feat)
        visual_proj = self.history_visual_proj(history_vision_feat)

        # 融合多模态特征
        combined = torch.cat([text_proj, visual_proj], dim=-1)
        fused_history = self.modal_fusion(combined)

        # 添加位置编码（处理序列长度超过位置编码表的情况）
        positions = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        # 截断位置索引，防止越界
        positions = positions.clamp(max=self.max_position_len - 1)
        pos_emb = self.position_embedding(positions)
        fused_history = fused_history + pos_emb

        # 创建attention mask
        # mask = True 表示该位置是padding，需要被mask
        # history_len表示有效长度，超过有效长度的位置应该被mask
        mask = torch.arange(max_seq_len, device=device).unsqueeze(0) >= history_len.unsqueeze(1)

        # Transformer编码
        encoded_history = self.sequence_encoder(fused_history, src_key_padding_mask=mask)

        # Attention pooling
        query = self.interest_query.expand(batch_size, -1, -1)
        user_interest, _ = self.interest_attention(
            query, encoded_history, encoded_history,
            key_padding_mask=mask
        )
        user_interest = user_interest.squeeze(1)

        return user_interest


# ==================== 新增：物品编码器 ====================
class MCRLItemEncoder(nn.Module):
    """物品编码器

    从多模态特征编码物品表征
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        # 模态编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(config.text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.visual_encoder = nn.Sequential(
            nn.Linear(config.visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 模态融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            text_feat: (batch, text_dim) 或 (batch, num_items, text_dim)
            vision_feat: (batch, visual_dim) 或 (batch, num_items, visual_dim)

        Returns:
            item_repr: (batch, hidden_dim) 或 (batch, num_items, hidden_dim)
        """
        # 确保输入特征与模型权重dtype一致（float16 -> float32）
        text_feat = text_feat.float()
        vision_feat = vision_feat.float()

        text_encoded = self.text_encoder(text_feat)
        visual_encoded = self.visual_encoder(vision_feat)

        combined = torch.cat([text_encoded, visual_encoded], dim=-1)
        item_repr = self.fusion(combined)

        return item_repr


# ==================== 新增：用户-物品匹配层 ====================
class MCRLMatcher(nn.Module):
    """用户-物品匹配层

    计算用户对物品的偏好得分
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.user_proj = nn.Linear(hidden_dim, hidden_dim)
        self.item_proj = nn.Linear(hidden_dim, hidden_dim)

        # MLP匹配网络
        self.match_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_repr: (batch, hidden_dim)
            item_repr: (batch, hidden_dim) 或 (batch, num_items, hidden_dim)

        Returns:
            scores: (batch,) 或 (batch, num_items)
        """
        user_proj = self.user_proj(user_repr)

        if item_repr.dim() == 2:
            # 单个物品
            item_proj = self.item_proj(item_repr)
            interaction = user_proj * item_proj
            combined = torch.cat([user_proj, item_proj, interaction], dim=-1)
            scores = self.match_mlp(combined).squeeze(-1)
        else:
            # 多个物品
            batch_size, num_items, _ = item_repr.shape
            item_proj = self.item_proj(item_repr)
            user_proj_expanded = user_proj.unsqueeze(1).expand(-1, num_items, -1)
            interaction = user_proj_expanded * item_proj
            combined = torch.cat([user_proj_expanded, item_proj, interaction], dim=-1)
            scores = self.match_mlp(combined).squeeze(-1)

        return scores


class MCRL(AbstractTrainableModel):
    """完整的多任务对比学习推荐框架

    核心改造：
    - 主任务：用户-物品偏好匹配（推荐核心）
    - 辅助任务：多层对比学习（提升表征质量）

    理论依据：定理5，三层对比学习的协同效应
    继承 AbstractTrainableModel 以使用统一训练框架
    """

    def __init__(
        self,
        config,
        ablation_mode: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """初始化MCRL模型

        Args:
            config: 配置对象
            ablation_mode: 消融模式，可选值：
                - None: 完整模型
                - "no_user_cl": 移除用户偏好对比学习
                - "no_intra_cl": 移除模态内对比学习
                - "no_inter_cl": 移除模态间对比学习
            device: 设备
        """
        super().__init__(device=device)
        self.config = config
        self.ablation_mode = ablation_mode

        # 从配置中获取参数
        self.alpha = getattr(config, 'mcrl_alpha', 0.5)  # 模态内对比权重
        self.beta = getattr(config, 'mcrl_beta', 0.5)    # 模态间对比权重
        self.temperature = getattr(config, 'mcrl_temperature', 0.07)
        self.hidden_dim = config.hidden_dim
        self.num_modalities = config.num_modalities

        # 损失权重
        self.rec_loss_weight = getattr(config, 'rec_loss_weight', 1.0)
        self.contrastive_loss_weight = getattr(config, 'mcrl_loss_weight', 0.5)

        # ==================== 新增：推荐模块 ====================
        # 用户兴趣编码器
        self.user_encoder = MCRLUserEncoder(config)

        # 物品编码器
        self.item_encoder = MCRLItemEncoder(config)

        # 用户-物品匹配层
        self.matcher = MCRLMatcher(config.hidden_dim)

        # ==================== 原有：对比学习模块 ====================
        # 三层对比学习模块
        self.user_preference_cl = UserPreferenceContrastive(
            hidden_dim=config.hidden_dim,
            temperature=self.temperature
        )

        self.intra_modal_cl = IntraModalContrastive(
            hidden_dim=config.hidden_dim,
            num_modalities=config.num_modalities,
            temperature=self.temperature
        )

        self.inter_modal_cl = InterModalContrastive(
            hidden_dim=config.hidden_dim,
            num_modalities=config.num_modalities,
            temperature=self.temperature
        )

        # ID表征优化器（对比学习优化后的表征）
        self.id_optimizer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # 可学习的模态权重模块（基于用户和物品表征动态计算模态权重）
        self.modal_weight_learner = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.num_modalities),
            nn.Softmax(dim=-1)
        )

        # 注册模态
        self.register_modalities({
            'visual': config.visual_dim,
            'text': config.text_dim
        })

    def register_modalities(self, modal_dims: Dict[str, int]):
        """注册模态及其维度信息"""
        for modality, dim in modal_dims.items():
            self.intra_modal_cl.add_modality(modality, dim)
            self.inter_modal_cl.add_modality(modality, dim)

    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """验证batch数据的完整性

        Args:
            batch: 输入的batch数据

        Raises:
            KeyError: 如果缺少必要的键
        """
        required_keys = [
            'history_text_feat', 'history_vision_feat', 'history_len',
            'target_text_feat', 'target_vision_feat',
            'neg_text_feat', 'neg_vision_feat'
        ]
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise KeyError(f"Batch缺少必要的键: {missing_keys}。"
                          f"期望的键: {required_keys}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            batch: 包含以下键的字典（来自PMATDataset，可复用）
                - history_text_feat: (batch, max_seq_len, text_dim)
                - history_vision_feat: (batch, max_seq_len, visual_dim)
                - history_len: (batch,)
                - target_text_feat: (batch, text_dim)
                - target_vision_feat: (batch, visual_dim)
                - neg_text_feat: (batch, num_neg, text_dim)
                - neg_vision_feat: (batch, num_neg, visual_dim)

        Returns:
            outputs: 包含各种输出的字典

        Raises:
            KeyError: 如果batch缺少必要的键
        """
        # 训练模式下自动清理缓存，避免内存泄露
        if self.training:
            self.clear_cache()

        # 验证batch数据完整性
        self._validate_batch(batch)

        # 1. 编码用户兴趣（使用真实历史序列）
        user_repr = self.user_encoder(
            batch['history_text_feat'],
            batch['history_vision_feat'],
            batch['history_len']
        )  # (batch, hidden_dim)

        # 2. 编码正样本（目标物品）
        pos_item_repr = self.item_encoder(
            batch['target_text_feat'],
            batch['target_vision_feat']
        )  # (batch, hidden_dim)

        # 3. 编码负样本
        neg_item_repr = self.item_encoder(
            batch['neg_text_feat'],
            batch['neg_vision_feat']
        )  # (batch, num_neg, hidden_dim)

        # 4. 编码历史物品表征（用于对比学习的多正样本）
        batch_size, seq_len, _ = batch['history_text_feat'].shape
        history_text_flat = batch['history_text_feat'].view(batch_size * seq_len, -1)
        history_vision_flat = batch['history_vision_feat'].view(batch_size * seq_len, -1)
        history_item_repr = self.item_encoder(history_text_flat, history_vision_flat)
        history_item_repr = history_item_repr.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_dim)

        # 创建历史序列mask
        history_mask = torch.arange(seq_len, device=batch['history_len'].device).unsqueeze(0) < batch['history_len'].unsqueeze(1)

        # 5. 计算用户-物品匹配分数
        pos_scores = self.matcher(user_repr, pos_item_repr)  # (batch,)
        neg_scores = self.matcher(user_repr, neg_item_repr)  # (batch, num_neg)

        # 6. 对比学习（使用真实物品表征）
        # 准备模态特征
        modal_features = {
            'visual': batch['target_vision_feat'],
            'text': batch['target_text_feat']
        }

        # 使用可学习的模态权重（基于用户和物品表征动态计算）
        weight_input = torch.cat([user_repr, pos_item_repr], dim=-1)  # (batch, hidden_dim * 2)
        modal_weights = self.modal_weight_learner(weight_input)  # (batch, num_modalities)

        # 优化物品表征
        optimized_pos_repr = self.id_optimizer(pos_item_repr) + pos_item_repr

        # 对比学习损失（根据消融模式选择性计算）
        # Layer 1: 用户偏好对比学习（使用历史相似物品作为额外正样本）
        if self.ablation_mode != 'no_user_cl':
            L_user = self.user_preference_cl(
                id_embeddings=pos_item_repr,
                user_embeddings=user_repr,
                positive_ids=pos_item_repr.unsqueeze(1),  # 目标物品作为主正样本
                negative_ids=neg_item_repr,
                history_item_repr=history_item_repr,  # 历史物品表征
                history_mask=history_mask  # 历史序列mask
            )
        else:
            L_user = torch.tensor(0.0, device=pos_item_repr.device)

        # Layer 2: 模态内对比学习
        if self.ablation_mode != 'no_intra_cl':
            L_intra = self.intra_modal_cl(
                id_embeddings=pos_item_repr,
                modal_features=modal_features,
                modal_weights=modal_weights
            )
        else:
            L_intra = torch.tensor(0.0, device=pos_item_repr.device)

        # Layer 3: 模态间对比学习
        if self.ablation_mode != 'no_inter_cl':
            L_inter = self.inter_modal_cl(modal_features=modal_features)
        else:
            L_inter = torch.tensor(0.0, device=pos_item_repr.device)

        # 缓存表征（用于推理时避免重复计算）
        if not self.training:
            self._cached_user_repr = user_repr.detach()
            self._cached_item_repr = optimized_pos_repr.detach()
            self._cache_timestamp = torch.tensor(1)  # 标记缓存已更新

        return {
            'user_repr': user_repr,
            'pos_item_repr': pos_item_repr,
            'neg_item_repr': neg_item_repr,
            'optimized_pos_repr': optimized_pos_repr,
            'pos_scores': pos_scores,
            'neg_scores': neg_scores,
            'contrastive_losses': {
                'user_preference_loss': L_user,
                'intra_modal_loss': L_intra,
                'inter_modal_loss': L_inter
            }
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失

        Args:
            outputs: forward的输出

        Returns:
            losses: 各项损失
        """
        pos_scores = outputs['pos_scores']  # (batch,)
        neg_scores = outputs['neg_scores']  # (batch, num_neg)

        # 1. BPR推荐损失
        # 处理负样本为空的情况
        if neg_scores.numel() == 0 or neg_scores.size(1) == 0:
            # 没有负样本时，使用margin loss作为替代
            bpr_loss = F.relu(1.0 - pos_scores).mean()
        else:
            # 对每个负样本计算BPR损失
            pos_scores_expanded = pos_scores.unsqueeze(1)  # (batch, 1)
            # 添加数值稳定性：裁剪极端值，增大epsilon
            score_diff = torch.clamp(pos_scores_expanded - neg_scores, min=-50, max=50)
            bpr_loss = -torch.log(torch.sigmoid(score_diff) + 1e-6).mean()

        # 2. 对比学习损失
        cl_losses = outputs['contrastive_losses']
        contrastive_loss = (
            cl_losses['user_preference_loss'] +
            self.alpha * cl_losses['intra_modal_loss'] +
            self.beta * cl_losses['inter_modal_loss']
        )

        # 3. 总损失
        total_loss = self.rec_loss_weight * bpr_loss + self.contrastive_loss_weight * contrastive_loss

        return {
            'bpr_loss': bpr_loss,
            'contrastive_loss': contrastive_loss,
            'user_pref_loss': cl_losses['user_preference_loss'],
            'intra_modal_loss': cl_losses['intra_modal_loss'],
            'inter_modal_loss': cl_losses['inter_modal_loss'],
            'total_loss': total_loss
        }

    def get_user_embedding(self, batch: Dict[str, torch.Tensor], use_cache: bool = True) -> torch.Tensor:
        """获取用户嵌入（用于召回）

        Args:
            batch: 包含用户历史的批次数据
            use_cache: 是否使用缓存（推理时可复用forward计算的表征）

        Returns:
            user_embedding: (batch, hidden_dim)
        """
        # 检查缓存过期
        self.check_cache_expiry()

        # 尝试使用缓存（避免重复计算）
        if use_cache and hasattr(self, '_cached_user_repr') and self._cached_user_repr is not None:
            return self._cached_user_repr

        return self.user_encoder(
            batch['history_text_feat'],
            batch['history_vision_feat'],
            batch['history_len']
        )

    def get_item_embedding(self, text_feat: torch.Tensor, vision_feat: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        """获取物品嵌入（用于召回）

        Args:
            text_feat: (num_items, text_dim)
            vision_feat: (num_items, visual_dim)
            use_cache: 是否使用缓存（推理时可复用forward计算的表征）

        Returns:
            item_embedding: (num_items, hidden_dim)
        """
        # 检查缓存过期
        self.check_cache_expiry()

        # 尝试使用缓存（避免重复计算）
        if use_cache and hasattr(self, '_cached_item_repr') and self._cached_item_repr is not None:
            return self._cached_item_repr

        item_repr = self.item_encoder(text_feat, vision_feat)
        # 使用对比学习优化后的表征
        optimized_repr = self.id_optimizer(item_repr) + item_repr
        return optimized_repr

    def clear_cache(self):
        """清除表征缓存

        优化：添加缓存过期机制，避免内存泄露
        - 训练模式下在forward开始时自动调用
        - 推理模式下可手动调用或通过check_cache_expiry自动清理
        """
        if hasattr(self, '_cached_user_repr'):
            del self._cached_user_repr
        if hasattr(self, '_cached_item_repr'):
            del self._cached_item_repr
        if hasattr(self, '_cache_timestamp'):
            del self._cache_timestamp

        self._cached_user_repr = None
        self._cached_item_repr = None
        self._cache_timestamp = None

    def check_cache_expiry(self, max_cache_calls: int = 100):
        """检查缓存是否过期，过期则自动清理

        Args:
            max_cache_calls: 缓存最大使用次数，超过则清理

        Returns:
            bool: 缓存是否被清理
        """
        if not hasattr(self, '_cache_access_count'):
            self._cache_access_count = 0

        self._cache_access_count += 1

        if self._cache_access_count >= max_cache_calls:
            self.clear_cache()
            self._cache_access_count = 0
            return True
        return False

    # ==================== AbstractTrainableModel 抽象方法实现 ====================

    def _get_optimizer(self, stage_id: int, stage_kwargs: Dict) -> torch.optim.Optimizer:
        """获取指定阶段的优化器"""
        lr = stage_kwargs.get('lr', 0.001)
        weight_decay = stage_kwargs.get('weight_decay', 0.01)
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_optimizer_state_dict(self) -> Dict:
        """获取当前阶段优化器的状态字典"""
        optimizer_states = {}
        for stage_id, optimizer in self._stage_optimizers.items():
            optimizer_states[stage_id] = optimizer.state_dict()
        return optimizer_states

    def _load_optimizer_state_dict(self, state_dict: Dict):
        """加载当前阶段优化器的状态字典"""
        for stage_id, opt_state in state_dict.items():
            if stage_id in self._stage_optimizers:
                self._stage_optimizers[stage_id].load_state_dict(opt_state)

    def _update_params(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, scaler=None):
        """重写参数更新逻辑，添加梯度裁剪

        对比学习 + 推荐损失的组合可能导致梯度爆炸，需要梯度裁剪
        """
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

    def _train_one_batch(self, batch: Any, stage_id: int, stage_kwargs: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        单batch训练逻辑（MCRL 推荐 + 对比学习）

        Args:
            batch: 训练批次数据（来自PMATDataset）
            stage_id: 阶段ID
            stage_kwargs: 该阶段的自定义参数

        Returns:
            (batch_loss, batch_metrics)
        """
        # 将数据移到设备上
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # 前向传播
        outputs = self.forward(batch)

        # 计算损失
        losses = self.compute_loss(outputs)
        loss = losses['total_loss']

        # 记录各项损失作为指标
        metrics = {
            'bpr_loss': losses['bpr_loss'].item(),
            'contrastive_loss': losses['contrastive_loss'].item(),
            'user_pref_loss': losses['user_pref_loss'].item(),
            'intra_modal_loss': losses['intra_modal_loss'].item(),
            'inter_modal_loss': losses['inter_modal_loss'].item(),
        }

        return loss, metrics

    def _validate_one_epoch(self, val_dataloader: torch.utils.data.DataLoader, stage_id: int,
                           stage_kwargs: Dict) -> Dict:
        """单轮验证逻辑 - 计算推荐指标"""
        self.eval()

        all_pos_scores = []
        all_neg_scores = []
        total_metrics = {
            'bpr_loss': 0.0,
            'contrastive_loss': 0.0,
            'user_pref_loss': 0.0,
            'intra_modal_loss': 0.0,
            'inter_modal_loss': 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                # 将数据移到设备上
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # 前向传播
                outputs = self.forward(batch)
                losses = self.compute_loss(outputs)

                # 收集分数用于计算推荐指标
                all_pos_scores.append(outputs['pos_scores'].cpu())
                all_neg_scores.append(outputs['neg_scores'].cpu())

                # 累计损失
                total_metrics['bpr_loss'] += losses['bpr_loss'].item()
                total_metrics['contrastive_loss'] += losses['contrastive_loss'].item()
                total_metrics['user_pref_loss'] += losses['user_pref_loss'].item()
                total_metrics['intra_modal_loss'] += losses['intra_modal_loss'].item()
                total_metrics['inter_modal_loss'] += losses['inter_modal_loss'].item()
                num_batches += 1

        # 计算平均损失
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}

        # 计算推荐指标
        all_pos_scores = torch.cat(all_pos_scores, dim=0)
        all_neg_scores = torch.cat(all_neg_scores, dim=0)
        rec_metrics = self._compute_recommendation_metrics(all_pos_scores, all_neg_scores)
        avg_metrics.update(rec_metrics)

        return avg_metrics

    def _compute_recommendation_metrics(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        k_list: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """计算推荐指标，与metrics.py保持一致

        Args:
            pos_scores: (num_samples,) 正样本分数
            neg_scores: (num_samples, num_neg) 负样本分数
            k_list: 计算指标的K值列表

        Returns:
            metrics: 各项推荐指标
        """
        metrics = {}
        num_samples = pos_scores.size(0)
        num_neg = neg_scores.size(1)

        # 将正样本分数与负样本分数拼接，正样本在第一位
        # all_scores: (num_samples, 1 + num_neg)
        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        # 创建预测排名和真实标签，用于计算指标
        # 将分数转换为物品ID排名
        _, sorted_indices = torch.sort(all_scores, dim=1, descending=True)
        predictions = sorted_indices.tolist()  # 每个样本的物品ID排名列表
        ground_truth = [[0] for _ in range(num_samples)]  # 假设物品ID 0是正样本

        # 计算各项指标
        for k in k_list:
            # Precision@K
            precision = 0.0
            # Recall@K (与HR@K相同，因为每个样本只有一个正样本)
            recall = 0.0
            # NDCG@K
            ndcg = 0.0
            # MRR@K
            mrr_k = 0.0
            
            for i in range(num_samples):
                pred_k = predictions[i][:k]
                gt = ground_truth[i]
                
                # Precision@K = 命中数 / K
                hits = len(set(pred_k) & set(gt))
                precision += hits / k
                
                # Recall@K = 命中数 / 正样本数 (这里正样本数为1)
                recall += hits / len(gt) if len(gt) > 0 else 0
                
                # NDCG@K
                dcg = 0.0
                for j, item in enumerate(pred_k):
                    if item in gt:
                        dcg += 1.0 / np.log2(j + 2)
                
                # IDCG: 理想DCG，正样本排在第一位
                idcg = 1.0 / np.log2(2)  # 1.0
                ndcg += dcg / idcg if idcg > 0 else 0
                
                # MRR@K: 第一个相关物品排名的倒数
                for j, item in enumerate(pred_k):
                    if item in gt:
                        mrr_k += 1.0 / (j + 1)
                        break
                else:
                    mrr_k += 0.0
            
            # 计算平均值
            metrics[f'Precision@{k}'] = precision / num_samples
            metrics[f'Recall@{k}'] = recall / num_samples  # 与HR@K相同
            metrics[f'HR@{k}'] = recall / num_samples  # 保持与原代码一致
            metrics[f'NDCG@{k}'] = ndcg / num_samples
            metrics[f'MRR@{k}'] = mrr_k / num_samples

        # 计算MRR (不限制K)
        # 计算排名（降序排列后正样本的位置）
        _, indices = torch.sort(all_scores, dim=1, descending=True)
        ranks = (indices == 0).nonzero(as_tuple=True)[1] + 1  # 1-based rank
        ranks = ranks.float()
        mrr = (1.0 / ranks).mean().item()
        metrics['MRR'] = mrr

        # 计算AUC
        # 正样本分数大于负样本分数的比例
        pos_expanded = pos_scores.unsqueeze(1)  # (num_samples, 1)
        auc = (pos_expanded > neg_scores).float().mean().item()
        metrics['AUC'] = auc
        
        # 计算MAP (Mean Average Precision)
        map_score = 0.0
        for i in range(num_samples):
            pred = predictions[i]
            gt = ground_truth[i]
            hits = 0
            precision_sum = 0.0
            for j, item in enumerate(pred):
                if item in gt:
                    hits += 1
                    precision_sum += hits / (j + 1)
            map_score += precision_sum / len(gt) if len(gt) > 0 else 0.0
        metrics['MAP'] = map_score / num_samples
        
        # 计算Coverage@K (使用最大的K)
        max_k = max(k_list)
        all_items = set(range(num_neg + 1))  # 所有可能的物品ID
        recommended_items = set()
        for i in range(num_samples):
            recommended_items.update(predictions[i][:max_k])
        metrics[f'Coverage@{max_k}'] = len(recommended_items) / len(all_items) if all_items else 0.0

        return metrics



def get_mcrl_ablation_model(ablation_module: str, config=None):
    """
    获取MCRL消融实验模型

    Args:
        ablation_module: 要移除的模块名称
            - "no_user_cl": 移除用户偏好对比学习
            - "no_intra_cl": 移除模态内对比学习
            - "no_inter_cl": 移除模态间对比学习
        config: 配置对象

    Returns:
        MCRL消融模型实例
    """
    if config is None:
        from config import config as default_config
        config = default_config

    return MCRL(config, ablation_mode=ablation_module)
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
    """
    
    def __init__(self, hidden_dim: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
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
        
    def forward(
        self,
        id_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            id_embeddings: (batch, hidden_dim) 当前物品ID嵌入
            user_embeddings: (batch, hidden_dim) 用户嵌入
            positive_ids: (batch, num_pos, hidden_dim) 正样本ID
            negative_ids: (batch, num_neg, hidden_dim) 负样本ID
            
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
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int, temperature: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.temperature = temperature
        
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

    def forward(
        self,
        id_embeddings: torch.Tensor,
        modal_features: Dict[str, torch.Tensor],
        modal_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            id_embeddings: (batch, hidden_dim) ID嵌入
            modal_features: {'visual': (batch, dim), 'text': (batch, dim), ...}
            modal_weights: (batch, num_modalities) 模态权重
            
        Returns:
            loss: 模态内对比损失
        """
        total_loss = 0.0
        modality_list = sorted(modal_features.keys())
        
        for idx, modality in enumerate(modality_list):
            # 适配模态特征到hidden_dim
            adapter = self.modal_adapters[modality]
            modal_feat = adapter(modal_features[modality])  # (batch, hidden_dim)
            
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
            
            # 对比损失
            modal_loss = F.cross_entropy(sim_matrix, labels)
            
            # 根据模态权重加权
            modal_weight = modal_weights[:, idx].mean()  # 该模态的平均权重
            total_loss += modal_weight * modal_loss
        
        return total_loss / len(modality_list)


class InterModalContrastive(nn.Module):
    """模态间对比学习模块
    理论依据：定理4，对齐不同模态的互补信息
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
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
            adapted_features[modality] = adapter(feat)
        
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
                
                # 负样本：不同物品的模态对（随机打乱）
                perm = torch.randperm(feat_j.size(0))
                neg_pairs = torch.cat([feat_i, feat_j[perm]], dim=-1)  # (batch, 2*hidden_dim)
                
                # 对齐网络
                alignment_net = self.alignment_nets[key]
                
                # 计算对齐分数
                pos_scores = alignment_net(pos_pairs).squeeze(-1)  # (batch,)
                neg_scores = alignment_net(neg_pairs).squeeze(-1)  # (batch,)
                
                # 对比损失（最大化正样本分数，最小化负样本分数）
                pair_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
                
                total_loss += pair_loss
                num_pairs += 1
        
        return total_loss / max(num_pairs, 1)


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

        # 位置编码
        max_seq_len = getattr(config, 'max_history_len', 50)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

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
        batch_size, max_seq_len, _ = history_text_feat.shape
        device = history_text_feat.device

        # 投影多模态特征
        text_proj = self.history_text_proj(history_text_feat)
        visual_proj = self.history_visual_proj(history_vision_feat)

        # 融合多模态特征
        combined = torch.cat([text_proj, visual_proj], dim=-1)
        fused_history = self.modal_fusion(combined)

        # 添加位置编码
        positions = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        fused_history = fused_history + pos_emb

        # 创建attention mask
        mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < (max_seq_len - history_len.unsqueeze(1))

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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(device=device)
        self.config = config

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
            num_modalities=config.num_modalities
        )

        # ID表征优化器（对比学习优化后的表征）
        self.id_optimizer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim)
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
        """
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

        # 4. 计算用户-物品匹配分数
        pos_scores = self.matcher(user_repr, pos_item_repr)  # (batch,)
        neg_scores = self.matcher(user_repr, neg_item_repr)  # (batch, num_neg)

        # 5. 对比学习（使用真实物品表征）
        # 准备模态特征
        modal_features = {
            'visual': batch['target_vision_feat'],
            'text': batch['target_text_feat']
        }
        modal_weights = torch.ones(user_repr.size(0), self.num_modalities, device=user_repr.device) / self.num_modalities

        # 优化物品表征
        optimized_pos_repr = self.id_optimizer(pos_item_repr) + pos_item_repr

        # 对比学习损失
        # Layer 1: 用户偏好对比学习
        L_user = self.user_preference_cl(
            id_embeddings=pos_item_repr,
            user_embeddings=user_repr,
            positive_ids=pos_item_repr.unsqueeze(1),  # 正样本是目标物品本身
            negative_ids=neg_item_repr  # 负样本
        )

        # Layer 2: 模态内对比学习
        L_intra = self.intra_modal_cl(
            id_embeddings=pos_item_repr,
            modal_features=modal_features,
            modal_weights=modal_weights
        )

        # Layer 3: 模态间对比学习
        L_inter = self.inter_modal_cl(modal_features=modal_features)

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
        # 对每个负样本计算BPR损失
        pos_scores_expanded = pos_scores.unsqueeze(1)  # (batch, 1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores_expanded - neg_scores) + 1e-8).mean()

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

    def get_user_embedding(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取用户嵌入（用于召回）"""
        return self.user_encoder(
            batch['history_text_feat'],
            batch['history_vision_feat'],
            batch['history_len']
        )

    def get_item_embedding(self, text_feat: torch.Tensor, vision_feat: torch.Tensor) -> torch.Tensor:
        """获取物品嵌入（用于召回）"""
        item_repr = self.item_encoder(text_feat, vision_feat)
        # 使用对比学习优化后的表征
        optimized_repr = self.id_optimizer(item_repr) + item_repr
        return optimized_repr

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
        """计算推荐指标

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

        # 计算排名（降序排列后正样本的位置）
        _, indices = torch.sort(all_scores, dim=1, descending=True)
        ranks = (indices == 0).nonzero(as_tuple=True)[1] + 1  # 1-based rank
        ranks = ranks.float()

        # HR@K 和 NDCG@K
        for k in k_list:
            if k > num_neg + 1:
                continue
            # HR@K: 正样本是否在top-K中
            hr_k = (ranks <= k).float().mean().item()
            metrics[f'HR@{k}'] = hr_k

            # NDCG@K
            dcg = (ranks <= k).float() / torch.log2(ranks + 1)
            ndcg_k = dcg.mean().item()
            metrics[f'NDCG@{k}'] = ndcg_k

        # MRR (Mean Reciprocal Rank)
        mrr = (1.0 / ranks).mean().item()
        metrics['MRR'] = mrr

        # AUC
        # 正样本分数大于负样本分数的比例
        pos_expanded = pos_scores.unsqueeze(1)  # (num_samples, 1)
        auc = (pos_expanded > neg_scores).float().mean().item()
        metrics['AUC'] = auc

        return metrics
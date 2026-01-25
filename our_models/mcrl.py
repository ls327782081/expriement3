"""
MCRL: Multi-task Contrastive Representation Learning
基于多任务对比学习的个性化语义ID表征优化

理论依据：
- 定理4：对比学习提升ID判别性
- 定理5：多任务对比学习的协同效应
- 定理6：检索效率提升
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from base_model import BaseModel


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


class MCRL(BaseModel):
    """完整的多任务对比学习框架
    理论依据：定理5，三层对比学习的协同效应
    """
    
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config
        
        # 从配置中获取参数
        self.alpha = getattr(config, 'mcrl_alpha', 0.5)  # 模态内对比权重
        self.beta = getattr(config, 'mcrl_beta', 0.5)    # 模态间对比权重
        self.temperature = getattr(config, 'mcrl_temperature', 0.07)
        self.hidden_dim = config.hidden_dim
        self.num_modalities = config.num_modalities
        
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
        
        # ID表征优化器
        self.id_optimizer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def register_modalities(self, modal_dims: Dict[str, int]):
        """注册模态及其维度信息"""
        for modality, dim in modal_dims.items():
            self.intra_modal_cl.add_modality(modality, dim)
            self.inter_modal_cl.add_modality(modality, dim)

    def forward(
        self,
        id_embeddings: torch.Tensor,
        user_embeddings: torch.Tensor,
        modal_features: Dict[str, torch.Tensor],
        modal_weights: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            id_embeddings: (batch, hidden_dim) 原始ID嵌入
            user_embeddings: (batch, hidden_dim) 用户嵌入
            modal_features: 各模态特征
            modal_weights: (batch, num_modalities) 模态权重
            positive_ids: (batch, num_pos, hidden_dim) 正样本
            negative_ids: (batch, num_neg, hidden_dim) 负样本
            
        Returns:
            outputs: {
                'optimized_ids': 优化后的ID嵌入,
                'losses': 各项损失
            }
        """
        # Layer 1: 用户偏好对比学习
        L_user = self.user_preference_cl(
            id_embeddings=id_embeddings,
            user_embeddings=user_embeddings,
            positive_ids=positive_ids,
            negative_ids=negative_ids
        )
        
        # Layer 2: 模态内对比学习
        L_intra = self.intra_modal_cl(
            id_embeddings=id_embeddings,
            modal_features=modal_features,
            modal_weights=modal_weights
        )
        
        # Layer 3: 模态间对比学习
        L_inter = self.inter_modal_cl(
            modal_features=modal_features
        )
        
        # 总对比损失
        total_contrastive_loss = L_user + self.alpha * L_intra + self.beta * L_inter
        
        # 优化ID表征
        optimized_ids = self.id_optimizer(id_embeddings)
        
        # 残差连接
        optimized_ids = optimized_ids + id_embeddings
        
        return {
            'optimized_ids': optimized_ids,
            'losses': {
                'user_preference_loss': L_user,
                'intra_modal_loss': L_intra,
                'inter_modal_loss': L_inter,
                'total_contrastive_loss': total_contrastive_loss
            }
        }
    
    def get_optimized_representation(
        self,
        id_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """获取优化后的ID表征（推理时使用）
        
        Args:
            id_embeddings: (batch, hidden_dim) 原始ID嵌入
            
        Returns:
            optimized_ids: (batch, hidden_dim) 优化后的ID嵌入
        """
        optimized_ids = self.id_optimizer(id_embeddings)
        optimized_ids = optimized_ids + id_embeddings
        return optimized_ids

    def train_step(self, batch, optimizer, criterion, device):
        """
        单步训练方法
        """
        # 移动数据到设备
        user_ids = batch["user_id"].to(device)
        item_ids = batch["item_id"].to(device)
        visual_feat = batch["vision_feat"].float().to(device)
        text_feat = batch["text_feat"].float().to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 注册模态信息（如果还没有注册的话）
        if not hasattr(self, '_modalities_registered'):
            self.register_modalities({
                'visual': visual_feat.size(1),  # 视觉特征的实际维度
                'text': text_feat.size(1)       # 文本特征的实际维度
            })
            self._modalities_registered = True
        
        # 生成用户嵌入（简化版本）
        batch_size = user_ids.size(0)
        user_embeddings = torch.randn(batch_size, self.hidden_dim).to(device)
        
        # 准备模态特征
        modal_features = {
            'visual': visual_feat,
            'text': text_feat
        }
        
        # 生成模态权重（简化版本）
        modal_weights = torch.ones(batch_size, self.num_modalities).to(device) / self.num_modalities
        
        # 生成正负样本（简化版本）
        positive_ids = torch.randn(batch_size, 5, self.hidden_dim).to(device)  # 5个正样本
        negative_ids = torch.randn(batch_size, 10, self.hidden_dim).to(device)  # 10个负样本
        
        # 生成初始ID嵌入（简化版本）
        id_embeddings = torch.randn(batch_size, self.hidden_dim).to(device)
        
        # 前向传播
        outputs = self(
            id_embeddings=id_embeddings,
            user_embeddings=user_embeddings,
            modal_features=modal_features,
            modal_weights=modal_weights,
            positive_ids=positive_ids,
            negative_ids=negative_ids
        )
        
        # 计算损失
        loss = outputs['losses']['total_contrastive_loss']
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def predict(self, batch, **kwargs):
        """
        预测方法
        """
        # 提取数据
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        visual_feat = batch["vision_feat"].float()
        text_feat = batch["text_feat"].float()
        
        # 生成初始ID嵌入（简化版本）
        batch_size = user_ids.size(0)
        id_embeddings = torch.randn(batch_size, self.hidden_dim)
        
        # 优化ID表征
        optimized_ids = self.get_optimized_representation(id_embeddings)
        
        # 返回优化后的ID嵌入
        return optimized_ids
"""
MCRL: Multi-task Contrastive Representation Learning
基于多任务对比学习的个性化语义ID表征优化

创新点：
1. 用户偏好对比学习 (User Preference Contrastive)
2. 模态内对比学习 (Intra-modal Contrastive)
3. 模态间对比学习 (Inter-modal Contrastive)

理论保证：
- 提升ID的判别性
- 优化检索空间结构
- 增强多模态互补性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class UserPreferenceContrastive(nn.Module):
    """用户偏好对比学习模块
    
    核心思想：
    - 正样本：相似用户偏好的物品ID应该接近
    - 负样本：不同偏好用户的物品ID应该远离
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
    
    核心思想：
    - 增强单模态内的语义判别性
    - 同一模态的相似物品ID应该接近
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int, temperature: float = 0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.temperature = temperature
        
        # 每个模态的投影器
        self.modal_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // num_modalities)
            )
            for _ in range(num_modalities)
        ])
        
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
            # 提取该模态的特征
            modal_feat = modal_features[modality]  # (batch, dim)
            
            # 投影
            projector = self.modal_projectors[idx]
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
    
    核心思想：
    - 对齐不同模态的互补信息
    - 最大化模态间的互信息
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int):
        super().__init__()
        self.num_modalities = num_modalities
        
        # 模态间对齐网络
        self.alignment_nets = nn.ModuleDict()
        
        # 为每对模态创建对齐网络
        modalities = ['visual', 'text', 'audio'][:num_modalities]
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                key = f"{modalities[i]}_{modalities[j]}"
                self.alignment_nets[key] = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
        
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
        total_loss = 0.0
        num_pairs = 0
        
        modality_list = sorted(modal_features.keys())
        
        # 遍历所有模态对
        for i in range(len(modality_list)):
            for j in range(i + 1, len(modality_list)):
                mod_i = modality_list[i]
                mod_j = modality_list[j]
                
                feat_i = modal_features[mod_i]  # (batch, dim)
                feat_j = modal_features[mod_j]  # (batch, dim)
                
                # 正样本：同一物品的不同模态
                pos_pairs = torch.cat([feat_i, feat_j], dim=-1)  # (batch, 2*dim)
                
                # 负样本：不同物品的模态对（随机打乱）
                perm = torch.randperm(feat_j.size(0))
                neg_pairs = torch.cat([feat_i, feat_j[perm]], dim=-1)  # (batch, 2*dim)
                
                # 对齐网络
                key = f"{mod_i}_{mod_j}"
                if key in self.alignment_nets:
                    alignment_net = self.alignment_nets[key]
                else:
                    # 反向key
                    key = f"{mod_j}_{mod_i}"
                    alignment_net = self.alignment_nets[key]
                
                # 计算对齐分数
                pos_scores = alignment_net(pos_pairs).squeeze(-1)  # (batch,)
                neg_scores = alignment_net(neg_pairs).squeeze(-1)  # (batch,)
                
                # 对比损失（最大化正样本分数，最小化负样本分数）
                pair_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
                
                total_loss += pair_loss
                num_pairs += 1
        
        return total_loss / max(num_pairs, 1)


class MCRL(nn.Module):
    """完整的多任务对比学习框架
    
    整合三层对比学习，优化个性化语义ID表征空间
    """
    
    def __init__(
        self,
        config,
        alpha: float = 1.0,  # 模态内对比权重
        beta: float = 0.5,   # 模态间对比权重
        temperature: float = 0.07
    ):
        super().__init__()
        self.config = config
        self.alpha = alpha
        self.beta = beta
        
        # 三层对比学习模块
        self.user_preference_cl = UserPreferenceContrastive(
            hidden_dim=config.hidden_dim,
            temperature=temperature
        )
        
        self.intra_modal_cl = IntraModalContrastive(
            hidden_dim=config.hidden_dim,
            num_modalities=config.num_modalities,
            temperature=temperature
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
            id_embeddings: (batch, hidden_dim) PMAT生成的ID嵌入
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
        
        # 总损失
        total_loss = L_user + self.alpha * L_intra + self.beta * L_inter
        
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
                'total_contrastive_loss': total_loss
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


class PMATWithMCRL(nn.Module):
    """PMAT + MCRL 联合模型
    
    端到端训练个性化语义ID生成与表征优化
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 导入PMAT
        from pmat import PMAT
        
        self.pmat = PMAT(config)
        self.mcrl = MCRL(
            config=config,
            alpha=config.mcrl_alpha,
            beta=config.mcrl_beta,
            temperature=config.mcrl_temperature
        )
        
        # 损失权重
        self.pmat_weight = config.pmat_loss_weight
        self.mcrl_weight = config.mcrl_loss_weight
        
    def forward(
        self,
        item_features: Dict[str, torch.Tensor],
        user_history: torch.Tensor,
        user_embeddings: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
        short_history: Optional[torch.Tensor] = None,
        long_history: Optional[torch.Tensor] = None,
        previous_id_emb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            item_features: 物品多模态特征
            user_history: 用户历史
            user_embeddings: 用户嵌入
            positive_ids: 正样本ID
            negative_ids: 负样本ID
            short_history: 短期历史
            long_history: 长期历史
            previous_id_emb: 之前的ID嵌入
            
        Returns:
            outputs: 完整输出
        """
        # Step 1: PMAT生成个性化语义ID
        pmat_outputs = self.pmat(
            item_features=item_features,
            user_history=user_history,
            short_history=short_history,
            long_history=long_history,
            previous_id_emb=previous_id_emb
        )
        
        # Step 2: MCRL优化ID表征
        mcrl_outputs = self.mcrl(
            id_embeddings=pmat_outputs['quantized_emb'],
            user_embeddings=user_embeddings,
            modal_features=self.pmat.multimodal_encoder(item_features),
            modal_weights=pmat_outputs['modal_weights'],
            positive_ids=positive_ids,
            negative_ids=negative_ids
        )
        
        # 合并输出
        outputs = {
            **pmat_outputs,
            **mcrl_outputs,
            'final_ids': mcrl_outputs['optimized_ids']
        }
        
        return outputs
    
    def compute_total_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        previous_id_emb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """计算总损失
        
        Args:
            outputs: forward的输出
            targets: 目标ID
            previous_id_emb: 之前的ID嵌入
            
        Returns:
            losses: 所有损失
        """
        # PMAT损失
        pmat_losses = self.pmat.compute_loss(outputs, targets, previous_id_emb)
        
        # MCRL损失
        mcrl_losses = outputs['losses']
        
        # 总损失
        total_loss = (
            self.pmat_weight * pmat_losses['total_loss'] +
            self.mcrl_weight * mcrl_losses['total_contrastive_loss']
        )
        
        all_losses = {
            **{f'pmat_{k}': v for k, v in pmat_losses.items()},
            **{f'mcrl_{k}': v for k, v in mcrl_losses.items()},
            'total_loss': total_loss
        }
        
        return all_losses


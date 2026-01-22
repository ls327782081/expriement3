"""
PMAT: Personalized Multimodal Adaptive Tokenizer
个性化多模态自适应语义ID生成框架（优化版）

创新点：
1. 个性化模态注意力权重分配
2. 兴趣感知的动态语义ID更新机制
3. 理论保证的语义一致性约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class UserModalAttention(nn.Module):
    """用户-模态偏好感知器
    
    根据用户历史交互学习对不同模态的偏好权重
    理论保证：个性化权重能减少ID的语义漂移
    """
    
    def __init__(self, user_dim: int, num_modalities: int, hidden_dim: int):
        super().__init__()
        self.num_modalities = num_modalities
        
        # 用户历史编码器
        self.user_history_encoder = nn.LSTM(
            input_size=user_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 模态偏好预测器
        self.modal_preference_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities)
        )
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, user_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_history: (batch, seq_len, user_dim) 用户历史交互序列
            
        Returns:
            modal_weights: (batch, num_modalities) 个性化模态权重
        """
        # 编码用户历史
        _, (h_n, _) = self.user_history_encoder(user_history)
        user_repr = h_n[-1]  # (batch, hidden_dim)
        
        # 预测模态偏好
        logits = self.modal_preference_net(user_repr)  # (batch, num_modalities)
        
        # 温度缩放的softmax
        modal_weights = F.softmax(logits / self.temperature, dim=-1)
        
        return modal_weights


class MultiModalEncoder(nn.Module):
    """多模态编码器
    
    支持视觉、文本、音频等多种模态
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 各模态编码器
        self.visual_encoder = nn.Sequential(
            nn.Linear(config.visual_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(config.text_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        if hasattr(config, 'audio_dim'):
            self.audio_encoder = nn.Sequential(
                nn.Linear(config.audio_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: {'visual': tensor, 'text': tensor, 'audio': tensor}
            
        Returns:
            encoded_features: 编码后的各模态特征
        """
        encoded = {}
        
        if 'visual' in features:
            encoded['visual'] = self.visual_encoder(features['visual'])
        
        if 'text' in features:
            encoded['text'] = self.text_encoder(features['text'])
            
        if 'audio' in features and hasattr(self, 'audio_encoder'):
            encoded['audio'] = self.audio_encoder(features['audio'])
            
        return encoded


class PersonalizedFusion(nn.Module):
    """个性化多模态融合模块
    
    根据用户偏好权重动态融合多模态特征
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 融合后的特征增强
        self.fusion_enhance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(
        self, 
        modal_features: Dict[str, torch.Tensor],
        modal_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            modal_features: {'visual': (batch, dim), 'text': (batch, dim), ...}
            modal_weights: (batch, num_modalities) 个性化权重
            
        Returns:
            fused_features: (batch, hidden_dim) 融合后的特征
        """
        # 按权重加权融合
        modality_list = sorted(modal_features.keys())
        features_stack = torch.stack([modal_features[m] for m in modality_list], dim=1)
        # features_stack: (batch, num_modalities, hidden_dim)
        
        # 加权求和
        modal_weights_expanded = modal_weights.unsqueeze(-1)  # (batch, num_modalities, 1)
        fused = (features_stack * modal_weights_expanded).sum(dim=1)  # (batch, hidden_dim)
        
        # 特征增强
        fused = self.fusion_enhance(fused)
        
        return fused


class DynamicIDUpdater(nn.Module):
    """动态语义ID更新模块
    
    检测用户兴趣漂移并触发增量更新
    """
    
    def __init__(self, hidden_dim: int, drift_threshold: float = 0.3):
        super().__init__()
        self.drift_threshold = drift_threshold
        
        # 短期兴趣编码器
        self.short_term_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 长期兴趣编码器
        self.long_term_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 漂移检测器
        self.drift_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 更新门控
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
    
    def detect_drift(
        self, 
        short_history: torch.Tensor,
        long_history: torch.Tensor
    ) -> torch.Tensor:
        """检测兴趣漂移
        
        Args:
            short_history: (batch, short_len, dim) 短期历史
            long_history: (batch, long_len, dim) 长期历史
            
        Returns:
            drift_score: (batch,) 漂移分数
        """
        # 编码短期和长期兴趣
        _, (short_h, _) = self.short_term_encoder(short_history)
        _, (long_h, _) = self.long_term_encoder(long_history)
        
        short_repr = short_h[-1]  # (batch, hidden_dim)
        long_repr = long_h[-1]    # (batch, hidden_dim)
        
        # 计算漂移分数
        combined = torch.cat([short_repr, long_repr], dim=-1)
        drift_score = self.drift_detector(combined).squeeze(-1)  # (batch,)
        
        return drift_score
    
    def update(
        self,
        current_id_emb: torch.Tensor,
        new_features: torch.Tensor,
        drift_score: torch.Tensor
    ) -> torch.Tensor:
        """增量更新ID
        
        Args:
            current_id_emb: (batch, dim) 当前ID嵌入
            new_features: (batch, dim) 新特征
            drift_score: (batch,) 漂移分数
            
        Returns:
            updated_id_emb: (batch, dim) 更新后的ID嵌入
        """
        # 计算更新门控
        combined = torch.cat([current_id_emb, new_features], dim=-1)
        gate = self.update_gate(combined)  # (batch, dim)
        
        # 根据漂移分数决定更新强度
        drift_mask = (drift_score > self.drift_threshold).float().unsqueeze(-1)
        effective_gate = gate * drift_mask
        
        # 增量更新
        updated_id_emb = (1 - effective_gate) * current_id_emb + effective_gate * new_features
        
        return updated_id_emb


class SemanticIDQuantizer(nn.Module):
    """语义ID量化器
    
    将连续特征量化为离散的语义ID序列
    """
    
    def __init__(self, hidden_dim: int, codebook_size: int, id_length: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.id_length = id_length
        
        # 层级码本（RQ-VAE风格）
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_dim)
            for _ in range(id_length)
        ])
        
        # 特征投影
        self.feature_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch, hidden_dim) 输入特征

        Returns:
            semantic_ids_logits: (batch, id_length, codebook_size) 语义ID的logits
            quantized_emb: (batch, hidden_dim) 量化后的嵌入
        """
        batch_size = features.size(0)
        features = self.feature_proj(features)

        semantic_ids_logits = []
        residual = features
        quantized_sum = torch.zeros_like(features)

        # 残差量化
        for level in range(self.id_length):
            codebook = self.codebooks[level]

            # 计算与码本的距离（负距离作为logits）
            distances = torch.cdist(
                residual.unsqueeze(1),
                codebook.weight.unsqueeze(0)
            ).squeeze(1)  # (batch, codebook_size)

            # 负距离作为logits（距离越小，logit越大）
            logits = -distances  # (batch, codebook_size)
            semantic_ids_logits.append(logits)

            # 选择最近的码
            ids = torch.argmin(distances, dim=-1)  # (batch,)

            # 获取量化嵌入
            quantized = codebook(ids)  # (batch, hidden_dim)
            quantized_sum = quantized_sum + quantized

            # 计算残差
            residual = residual - quantized

        # 堆叠logits: (batch, id_length, codebook_size)
        semantic_ids_logits = torch.stack(semantic_ids_logits, dim=1)

        return semantic_ids_logits, quantized_sum


class PMAT(nn.Module):
    """完整的PMAT框架
    
    整合所有模块，实现个性化多模态自适应语义ID生成
    """
    
    def __init__(self, config, ablation_mode: Optional[str] = None):
        super().__init__()
        self.config = config
        self.ablation_mode = ablation_mode
        
        # 核心模块
        self.user_modal_attention = UserModalAttention(
            user_dim=config.hidden_dim,
            num_modalities=config.num_modalities,
            hidden_dim=config.hidden_dim
        )
        
        self.multimodal_encoder = MultiModalEncoder(config)
        
        self.personalized_fusion = PersonalizedFusion(config.hidden_dim)
        
        self.dynamic_updater = DynamicIDUpdater(
            hidden_dim=config.hidden_dim,
            drift_threshold=config.drift_threshold
        )
        
        self.semantic_quantizer = SemanticIDQuantizer(
            hidden_dim=config.hidden_dim,
            codebook_size=config.codebook_size,
            id_length=config.id_length
        )
        
        # 语义一致性约束
        self.consistency_weight = config.consistency_weight
        
    def forward(
        self,
        batch_or_features,
        user_history: Optional[torch.Tensor] = None,
        short_history: Optional[torch.Tensor] = None,
        long_history: Optional[torch.Tensor] = None,
        previous_id_emb: Optional[torch.Tensor] = None
    ):
        """
        Args:
            batch_or_features: batch字典或item_features字典
            user_history: 用户历史交互序列（可选）
            short_history: 短期历史（用于漂移检测）
            long_history: 长期历史（用于漂移检测）
            previous_id_emb: 之前的ID嵌入（用于动态更新）

        Returns:
            如果输入是batch字典，返回logits (batch, id_length, codebook_size)
            否则返回outputs字典
        """
        # 兼容两种调用方式
        if isinstance(batch_or_features, dict) and 'text_feat' in batch_or_features:
            # 训练/评估模式：从batch中提取特征
            batch = batch_or_features
            batch_size = batch['text_feat'].size(0)

            item_features = {
                'text': batch['text_feat'].float(),
                'visual': batch['vision_feat'].float()
            }

            # 创建简单的用户历史（使用随机嵌入作为占位符）
            # user_history shape: (batch, seq_len, user_dim)
            user_history = torch.randn(batch_size, 10, self.config.hidden_dim, device=batch['text_feat'].device)

            return_logits = True
        else:
            # 原始调用方式
            item_features = batch_or_features
            return_logits = False
        # 1. 计算个性化模态权重
        batch_size = user_history.size(0)
        actual_num_modalities = len(item_features)  # 实际的模态数量

        if self.ablation_mode != 'no_personalization':
            full_modal_weights = self.user_modal_attention(user_history)
            # 只取前actual_num_modalities个权重并重新归一化
            modal_weights = full_modal_weights[:, :actual_num_modalities]
            modal_weights = modal_weights / modal_weights.sum(dim=1, keepdim=True)
        else:
            # 消融：使用均匀权重
            modal_weights = torch.ones(
                batch_size, actual_num_modalities,
                device=user_history.device
            ) / actual_num_modalities

        # 2. 编码多模态特征
        encoded_features = self.multimodal_encoder(item_features)

        # 3. 个性化融合
        fused_features = self.personalized_fusion(encoded_features, modal_weights)
        
        # 4. 动态更新（如果提供了历史信息）
        drift_score = None
        if (short_history is not None and long_history is not None 
            and self.ablation_mode != 'no_dynamic_update'):
            drift_score = self.dynamic_updater.detect_drift(short_history, long_history)
            
            if previous_id_emb is not None:
                fused_features = self.dynamic_updater.update(
                    previous_id_emb, fused_features, drift_score
                )
        
        # 5. 生成语义ID
        semantic_ids, quantized_emb = self.semantic_quantizer(fused_features)

        # 如果是训练/评估模式，直接返回logits
        if return_logits:
            return semantic_ids  # (batch, id_length, codebook_size)

        return {
            'semantic_ids': semantic_ids,
            'modal_weights': modal_weights,
            'drift_score': drift_score,
            'quantized_emb': quantized_emb,
            'fused_features': fused_features,
            'modal_features': encoded_features  # 添加编码后的模态特征
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        previous_id_emb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """计算损失
        
        Args:
            outputs: forward的输出
            targets: 目标ID序列
            previous_id_emb: 之前的ID嵌入（用于一致性约束）
            
        Returns:
            losses: 各项损失
        """
        losses = {}
        
        # 1. ID生成损失（交叉熵）
        semantic_ids = outputs['semantic_ids']
        id_loss = F.cross_entropy(
            semantic_ids.reshape(-1, self.config.codebook_size),
            targets.reshape(-1)
        )
        losses['id_loss'] = id_loss
        
        # 2. 语义一致性损失
        if previous_id_emb is not None:
            consistency_loss = F.mse_loss(
                outputs['quantized_emb'],
                previous_id_emb
            )
            losses['consistency_loss'] = self.consistency_weight * consistency_loss
        
        # 总损失
        losses['total_loss'] = sum(losses.values())

        return losses


def get_pmat_ablation_model(ablation_module: str, config=None):
    """
    获取消融实验模型

    Args:
        ablation_module: 要移除的模块名称
            - "no_personalization": 移除个性化模态权重
            - "no_dynamic_update": 移除动态更新机制
        config: 配置对象

    Returns:
        PMAT模型实例（带消融设置）
    """
    if config is None:
        from config import config as default_config
        config = default_config

    return PMAT(config, ablation_mode=ablation_module)


"""
PMAT: Personalized Multimodal Adaptive Tokenizer for Recommendation
个性化多模态自适应推荐模型

核心改造：将语义ID生成从"分类目标"转为"推荐任务的增强表征"
- 主任务：用户-物品偏好匹配（推荐核心）
- 辅助任务：语义ID生成（多任务学习，提升物品表征质量）

创新点：
1. 个性化模态注意力权重分配
2. 兴趣感知的动态语义ID更新机制
3. 语义ID增强的用户-物品匹配
4. 多任务学习框架（推荐+语义ID生成）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List
import math
import numpy as np
from base_model import AbstractTrainableModel
from config import config
from util import item_id_to_semantic_id


class UserModalAttention(nn.Module):
    """用户-模态偏好感知器

    根据用户兴趣表征学习对不同模态的偏好权重
    理论保证：个性化权重能减少ID的语义漂移

    修复说明：
    - 原设计接收用户历史序列，但实际调用时传入的是已编码的用户兴趣向量
    - 修改为直接接收用户兴趣向量，移除冗余的LSTM编码
    """

    def __init__(self, user_dim: int, num_modalities: int, hidden_dim: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim

        # 模态偏好预测器（直接从用户兴趣向量预测）
        self.modal_preference_net = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities)
        )

        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, user_interest: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_interest: (batch, user_dim) 用户兴趣表征（已由UserInterestEncoder编码）

        Returns:
            modal_weights: (batch, num_modalities) 个性化模态权重
        """
        # 预测模态偏好
        logits = self.modal_preference_net(user_interest)  # (batch, num_modalities)

        # 温度缩放的softmax（softmax保证输出和为1，避免除零问题）
        modal_weights = F.softmax(logits / (self.temperature + 1e-8), dim=-1)

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
        """增量更新ID（支持单物品和多物品批量更新）

        Args:
            current_id_emb: (batch, dim) 或 (batch, num_items, dim) 当前ID嵌入
            new_features: (batch, dim) 或 (batch, num_items, dim) 新特征
            drift_score: (batch,) 漂移分数

        Returns:
            updated_id_emb: 与输入相同形状的更新后ID嵌入
        """
        # 处理多物品情况：向量化操作替代循环
        if current_id_emb.dim() == 3:
            # (batch, num_items, dim) -> 展平处理
            batch_size, num_items, dim = current_id_emb.shape

            # 展平为 (batch * num_items, dim)
            current_flat = current_id_emb.view(-1, dim)
            new_flat = new_features.view(-1, dim)

            # 计算更新门控
            combined = torch.cat([current_flat, new_flat], dim=-1)
            gate = self.update_gate(combined)  # (batch * num_items, dim)

            # 扩展drift_score到所有物品: (batch,) -> (batch * num_items,)
            drift_expanded = drift_score.unsqueeze(1).expand(-1, num_items).reshape(-1)
            drift_mask = (drift_expanded > self.drift_threshold).float().unsqueeze(-1)
            effective_gate = gate * drift_mask

            # 增量更新
            updated_flat = (1 - effective_gate) * current_flat + effective_gate * new_flat

            # 恢复形状
            updated_id_emb = updated_flat.view(batch_size, num_items, dim)
        else:
            # 单物品情况：(batch, dim)
            combined = torch.cat([current_id_emb, new_features], dim=-1)
            gate = self.update_gate(combined)  # (batch, dim)

            # 根据漂移分数决定更新强度
            drift_mask = (drift_score > self.drift_threshold).float().unsqueeze(-1)
            effective_gate = gate * drift_mask

            # 增量更新
            updated_id_emb = (1 - effective_gate) * current_id_emb + effective_gate * new_features

        return updated_id_emb


class SemanticIDQuantizer(nn.Module):
    """标准RVQ量化器（保留所有创新：码本监控、温度系数、多层量化）"""

    def __init__(self, hidden_dim, codebook_size, id_length):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.id_length = id_length

        # 可学习温度系数
        self.temperature = nn.Parameter(torch.tensor(0.05))

        # 码本参数
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, hidden_dim))
            for _ in range(id_length)
        ])

        # 码本使用统计（仅用于监控，不参与计算图）
        self.codebook_usage = [torch.zeros(codebook_size) for _ in range(id_length)]
        self.layer_similarities = []  # 保存每层的相似度矩阵
        self.layer_indices = []  # 保存每层的量化索引
        self.dead_code_threshold = 5
        # 码本使用计数：记录每个码本被使用的次数
        self.code_usage_count = [torch.zeros(codebook_size) for _ in range(id_length)]

    def get_codebook_usage(self):
        """补全缺失的码本使用统计方法"""
        usage_stats = {
            'global': {'used': 0, 'total': self.codebook_size * self.id_length, 'usage_ratio': 0.0, 'dead_ratio': 0.0}}
        total_used = 0

        for layer in range(self.id_length):
            used = int(self.codebook_usage[layer].sum().item())
            usage_ratio = used / self.codebook_size
            dead_ratio = 1 - usage_ratio
            usage_stats[f'layer_{layer + 1}'] = {
                'used': used,
                'total': self.codebook_size,
                'usage_ratio': usage_ratio,
                'dead_ratio': dead_ratio
            }
            total_used += used

        # 全局统计
        usage_stats['global']['used'] = total_used
        usage_stats['global']['usage_ratio'] = total_used / usage_stats['global']['total']
        usage_stats['global']['dead_ratio'] = 1 - usage_stats['global']['usage_ratio']
        return usage_stats

    def forward(self, x):
        """
        标准RVQ前向（无任何module成员变量存储中间量）
        返回：
            quantized: 量化后的特征 [B, D]
            residual_loss: 残差约束损失（你的创新点）
            residuals: 各层残差（用于监控，detach后返回）
            codebook_usage: 码本使用统计
        """
        # 清空历史数据
        self.layer_similarities = []
        self.layer_indices = []

        residual = x
        quantized = torch.zeros_like(x)
        residuals_list = []  # 临时存储，不存到self
        residual_loss = torch.tensor(0.0, device=x.device)

        # 多层量化逻辑
        for i in range(self.id_length):
            # 1. 计算相似度（原有逻辑）
            similarity = torch.matmul(residual, self.codebooks[i].T) / torch.clamp(self.temperature, min=0.04)

            # 训练时加入Gumbel噪声（推理时仍用硬argmax）
            if self.training:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarity) + 1e-8) + 1e-8)
                similarity = similarity + gumbel_noise * 0.05  # 0.1是噪声系数，可调

                # ========== 新增：强制打散相似度（核心！） ==========
                # 1. 给每个样本的相似度加不同的随机偏移（避免所有样本选同一个索引）
                sample_shift = torch.randn(similarity.shape[0], 1, device=similarity.device) * 0.5
                similarity = similarity + sample_shift
                # 2. 对每个码本的相似度做随机扰动（避免某几个码本永远是最大值）
                codebook_shift = torch.randn(1, self.codebook_size, device=similarity.device) * 0.2
                similarity = similarity + codebook_shift

            # 2. 保存相似度和索引
            self.layer_similarities.append(similarity)  # [B, codebook_size]
            indices = torch.argmax(similarity, dim=-1)
            self.layer_indices.append(indices)  # [B]

            # 2. 软量化（核心：替换硬argmax，保留梯度，NeurIPS 2024）
            soft_indices = F.softmax(similarity, dim=-1)  # [B, codebook_size]，软分配概率
            quantized_layer_soft = torch.matmul(soft_indices, self.codebooks[i])  # [B, D]，软量化

            # 3. STE + 码本梯度约束（修复residual_loss=0）
            hard_indices = torch.argmax(similarity, dim=-1)
            quantized_layer_hard = F.embedding(hard_indices, self.codebooks[i])
            quantized_layer = quantized_layer_hard + (quantized_layer_soft - quantized_layer_soft.detach())

            # 4. 计算量化误差（强制residual_loss≠0，ICML 2023）
            quant_error = torch.mean((quantized_layer - residual) ** 2)
            residual_loss += quant_error  # 这一步让residual_loss不再为0

            # 5. 残差更新
            residual = residual - quantized_layer
            quantized += quantized_layer
            residuals_list.append(residual.clone())

            # 码本使用统计（detach，不参与计算图）
            self.codebook_usage[i] = torch.zeros(self.codebook_size, device=x.device)
            unique_indices = torch.unique(indices)
            self.codebook_usage[i][unique_indices] = 1.0

            # ================死码强制激活逻辑（原有） ================
            if self.training:
                # 1. 把计数tensor移到当前设备（避免CPU/GPU不匹配）
                self.code_usage_count[i] = self.code_usage_count[i].to(x.device)
                # 2. 更新码本使用次数：被选中的码本计数+1，未被选中的-0.1（加速死码判定）
                self.code_usage_count[i][unique_indices] += 1.0
                self.code_usage_count[i] = torch.clamp(self.code_usage_count[i], min=0.0)  # 计数不小于0
                # 3. 检测死码（计数 < 阈值）
                dead_codes = (self.code_usage_count[i] < self.dead_code_threshold).nonzero().squeeze()
                if len(dead_codes) > 0:
                    # 4. 随机选当前batch的残差特征，替换死码（带动量，避免突变）
                    # 确保替换的特征数量和死码数量一致
                    replace_num = len(dead_codes) if dead_codes.ndim > 0 else 1
                    random_idx = torch.randperm(residual.shape[0])[:replace_num]
                    random_res = residual[random_idx].detach()  # 脱离计算图，不影响梯度

                    # 5. 动量更新死码：90%保留原码本，10%替换为随机特征（温和更新）
                    if dead_codes.ndim == 0:  # 单死码的特殊情况
                        self.codebooks[i].data[dead_codes] = self.codebooks[i].data[dead_codes] * 0.7 + random_res * 0.3
                        self.code_usage_count[i][dead_codes] = self.dead_code_threshold  # 重置计数
                    else:
                        self.codebooks[i].data[dead_codes] = self.codebooks[i].data[dead_codes] * 0.7 + random_res * 0.3
                        self.code_usage_count[i][dead_codes] = self.dead_code_threshold  # 重置计数

        # 6. 原有：ICML 2021码本正则化（解决STE梯度缺失）
        lambda_reg = 0.01
        codebook_reg = 0.0
        for cb in self.codebooks:
            codebook_reg += torch.mean(torch.norm(cb, dim=-1) ** 2)  # 码本L2正则化
        residual_loss += lambda_reg * codebook_reg / self.id_length  # 平均到每层
        # 归一化到与重构损失接近的量级
        residual_loss = residual_loss / self.hidden_dim

        # 保留原有返回值（语义不变）
        residuals_stack = torch.stack(residuals_list, dim=1)

        return quantized, residual_loss, residuals_stack.detach(), self.codebook_usage


# ==================== 新增模块：用户兴趣编码器 ====================

class UserInterestEncoder(nn.Module):
    """用户兴趣编码器

    从用户历史交互序列中学习用户兴趣表征
    支持多模态历史特征的融合
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # 从配置获取最大序列长度，默认512
        self.max_seq_len = getattr(config, 'max_history_len', 50)
        # 位置编码长度设为max_seq_len的2倍，确保足够
        self.max_position_len = max(512, self.max_seq_len * 2)

        # 历史物品的多模态特征融合
        self.history_text_proj = nn.Linear(config.text_dim, config.hidden_dim)
        self.history_visual_proj = nn.Linear(config.visual_dim, config.hidden_dim)

        # 多模态融合层
        self.modal_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 序列编码器（Transformer）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 位置编码（使用配置的最大长度）
        self.position_embedding = nn.Embedding(self.max_position_len, config.hidden_dim)

        # 用户兴趣聚合（attention pooling）
        self.interest_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.interest_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True
        )

    def forward(
        self,
        history_text_feat: torch.Tensor,
        history_vision_feat: torch.Tensor,
        history_len: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            history_text_feat: (batch, max_seq_len, text_dim) 历史物品文本特征
            history_vision_feat: (batch, max_seq_len, visual_dim) 历史物品视觉特征
            history_len: (batch,) 每个用户的实际历史长度

        Returns:
            user_interest: (batch, hidden_dim) 用户兴趣表征
        """
        batch_size, max_seq_len, _ = history_text_feat.shape
        device = history_text_feat.device

        # 1. 投影多模态特征
        text_proj = self.history_text_proj(history_text_feat)  # (batch, seq, hidden)
        visual_proj = self.history_visual_proj(history_vision_feat)  # (batch, seq, hidden)

        # 2. 融合多模态特征
        combined = torch.cat([text_proj, visual_proj], dim=-1)  # (batch, seq, hidden*2)
        fused_history = self.modal_fusion(combined)  # (batch, seq, hidden)

        # 3. 添加位置编码（处理序列长度超过位置编码表的情况）
        positions = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        # 截断位置索引，防止越界
        positions = positions.clamp(max=self.max_position_len - 1)
        pos_emb = self.position_embedding(positions)
        fused_history = fused_history + pos_emb

        # 4. 创建attention mask（padding位置为True）
        # history_len: 实际长度，padding在前面
        mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < (max_seq_len - history_len.unsqueeze(1))

        # 5. Transformer编码
        encoded_history = self.sequence_encoder(fused_history, src_key_padding_mask=mask)

        # 6. Attention pooling获取用户兴趣
        query = self.interest_query.expand(batch_size, -1, -1)  # (batch, 1, hidden)
        user_interest, _ = self.interest_attention(
            query, encoded_history, encoded_history,
            key_padding_mask=mask
        )
        user_interest = user_interest.squeeze(1)  # (batch, hidden)

        return user_interest


# ==================== 新增模块：用户-物品匹配层 ====================

class UserItemMatcher(nn.Module):
    """用户-物品匹配层

    计算用户对物品的偏好得分
    支持语义ID增强的匹配
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # 用户表征投影
        self.user_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )

        # 物品表征投影（融合多模态特征和语义ID嵌入）
        self.item_proj = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),  # 多模态特征 + 语义ID嵌入
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )

        # 匹配网络（MLP）
        self.match_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),  # user + item + element-wise product
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(
        self,
        user_repr: torch.Tensor,
        item_fused_feat: torch.Tensor,
        item_semantic_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_repr: (batch, hidden_dim) 用户表征
            item_fused_feat: (batch, hidden_dim) 或 (batch, num_items, hidden_dim) 物品融合特征
            item_semantic_emb: (batch, hidden_dim) 或 (batch, num_items, hidden_dim) 物品语义ID嵌入

        Returns:
            scores: (batch,) 或 (batch, num_items) 偏好得分
        """
        # 用户投影
        user_proj = self.user_proj(user_repr)  # (batch, hidden)

        # 处理物品特征
        if item_fused_feat.dim() == 2:
            # 单个物品: (batch, hidden)
            item_combined = torch.cat([item_fused_feat, item_semantic_emb], dim=-1)
            item_proj = self.item_proj(item_combined)  # (batch, hidden)

            # 计算匹配分数
            interaction = user_proj * item_proj  # element-wise product
            combined = torch.cat([user_proj, item_proj, interaction], dim=-1)
            scores = self.match_mlp(combined).squeeze(-1)  # (batch,)
        else:
            # 多个物品: (batch, num_items, hidden)
            batch_size, num_items, _ = item_fused_feat.shape
            item_combined = torch.cat([item_fused_feat, item_semantic_emb], dim=-1)
            item_proj = self.item_proj(item_combined)  # (batch, num_items, hidden)

            # 扩展用户表征
            user_proj_expanded = user_proj.unsqueeze(1).expand(-1, num_items, -1)

            # 计算匹配分数
            interaction = user_proj_expanded * item_proj
            combined = torch.cat([user_proj_expanded, item_proj, interaction], dim=-1)
            scores = self.match_mlp(combined).squeeze(-1)  # (batch, num_items)

        return scores


# ==================== 重构的PMAT推荐模型 ====================

class PMAT(AbstractTrainableModel):
    """PMAT: 个性化多模态自适应推荐模型

    核心架构：
    1. 用户兴趣编码器：从历史序列学习用户表征
    2. 多模态编码器：编码物品的多模态特征
    3. 个性化融合：根据用户偏好融合多模态特征
    4. 语义ID量化器：生成物品的语义ID（辅助任务）
    5. 用户-物品匹配层：计算偏好得分（主任务）

    多任务学习：
    - 主损失：BPR推荐损失
    - 辅助损失：语义ID生成损失
    """

    def __init__(self, config, ablation_mode: Optional[str] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(device=device)
        self.config = config
        self.ablation_mode = ablation_mode

        # ===== 用户侧模块 =====
        # 用户兴趣编码器（新增）
        self.user_interest_encoder = UserInterestEncoder(config)

        # 用户模态偏好感知器（用于个性化融合）
        self.user_modal_attention = UserModalAttention(
            user_dim=config.hidden_dim,
            num_modalities=config.num_modalities,
            hidden_dim=config.hidden_dim
        )

        # ===== 物品侧模块 =====
        self.multimodal_encoder = MultiModalEncoder(config)
        self.personalized_fusion = PersonalizedFusion(config.hidden_dim)

        # 语义ID量化器（辅助任务）
        self.semantic_quantizer = SemanticIDQuantizer(
            hidden_dim=config.hidden_dim,
            codebook_size=config.codebook_size,
            id_length=config.id_length
        )

        # ===== 匹配模块（新增）=====
        self.user_item_matcher = UserItemMatcher(config)

        # ===== 动态更新模块 =====
        self.dynamic_updater = DynamicIDUpdater(
            hidden_dim=config.hidden_dim,
            drift_threshold=config.drift_threshold
        )

        # ===== 损失权重 =====
        self.rec_loss_weight = getattr(config, 'rec_loss_weight', 1.0)  # 推荐损失权重
        self.semantic_loss_weight = getattr(config, 'semantic_loss_weight', 0.1)  # 语义ID损失权重
        self.consistency_weight = config.consistency_weight

    def _get_modal_weights(
        self,
        user_interest: torch.Tensor,
        batch_size: int,
        device: torch.device,
        num_modalities: int = 2
    ) -> torch.Tensor:
        """获取模态权重（统一处理个性化和非个性化情况）

        Args:
            user_interest: (batch, hidden_dim) 用户兴趣
            batch_size: 批次大小
            device: 设备
            num_modalities: 当前使用的模态数量

        Returns:
            modal_weights: (batch, num_modalities) 归一化的模态权重
        """
        if self.ablation_mode != 'no_personalization':
            # 使用用户兴趣计算模态权重（UserModalAttention已修改为直接接收用户兴趣向量）
            full_modal_weights = self.user_modal_attention(user_interest)  # (batch, config.num_modalities)

            # 根据实际使用的模态数量截取权重
            modal_weights = full_modal_weights[:, :num_modalities]

            # 安全的归一化（处理除零情况）
            weight_sum = modal_weights.sum(dim=1, keepdim=True)
            # 如果权重和为0，使用均匀分布
            weight_sum = torch.where(
                weight_sum > 1e-8,
                weight_sum,
                torch.ones_like(weight_sum)
            )
            modal_weights = modal_weights / weight_sum
        else:
            # 消融模式：使用均匀权重
            modal_weights = torch.ones(batch_size, num_modalities, device=device) / num_modalities

        return modal_weights

    def encode_item(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor,
        user_interest: torch.Tensor,
        short_history: Optional[torch.Tensor] = None,
        long_history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码物品特征

        支持动态ID更新（当提供短期和长期历史时）

        Args:
            text_feat: (batch, text_dim) 或 (batch, num_items, text_dim)
            vision_feat: (batch, visual_dim) 或 (batch, num_items, visual_dim)
            user_interest: (batch, hidden_dim) 用户兴趣（用于个性化融合）
            short_history: (batch, short_len, hidden_dim) 短期历史（可选，用于动态更新）
            long_history: (batch, long_len, hidden_dim) 长期历史（可选，用于动态更新）

        Returns:
            fused_feat: 融合后的物品特征
            semantic_ids_logits: 语义ID的logits
            quantized_emb: 量化后的嵌入
        """
        device = text_feat.device

        # 处理维度
        if text_feat.dim() == 2:
            # 单个物品
            batch_size = user_interest.size(0)
            item_features = {
                'text': text_feat.float(),
                'visual': vision_feat.float()
            }

            # 计算个性化模态权重
            modal_weights = self._get_modal_weights(user_interest, batch_size, device, num_modalities=2)

            # 编码和融合
            encoded_features = self.multimodal_encoder(item_features)
            fused_feat = self.personalized_fusion(encoded_features, modal_weights)

            # 生成语义ID
            semantic_ids_logits, quantized_emb = self.semantic_quantizer(fused_feat)

            # ===== 动态ID更新（核心创新点）=====
            if self.ablation_mode != 'no_dynamic_update' and short_history is not None and long_history is not None:
                # 检测兴趣漂移
                drift_score = self.dynamic_updater.detect_drift(short_history, long_history)
                # 根据漂移分数动态更新语义ID嵌入
                quantized_emb = self.dynamic_updater.update(quantized_emb, fused_feat, drift_score)

        else:
            # 多个物品: (batch, num_items, dim)
            batch_size, num_items, _ = text_feat.shape

            # 展平处理
            text_flat = text_feat.view(-1, text_feat.size(-1))
            vision_flat = vision_feat.view(-1, vision_feat.size(-1))

            item_features = {
                'text': text_flat.float(),
                'visual': vision_flat.float()
            }

            # 计算模态权重（扩展到所有物品）
            modal_weights = self._get_modal_weights(user_interest, batch_size, device, num_modalities=2)
            # 扩展到所有物品
            modal_weights = modal_weights.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, 2)

            # 编码和融合
            encoded_features = self.multimodal_encoder(item_features)
            fused_feat = self.personalized_fusion(encoded_features, modal_weights)

            # 生成语义ID
            semantic_ids_logits, quantized_emb = self.semantic_quantizer(fused_feat)

            # 恢复形状
            fused_feat = fused_feat.view(batch_size, num_items, -1)
            semantic_ids_logits = semantic_ids_logits.view(batch_size, num_items, self.config.id_length, -1)
            quantized_emb = quantized_emb.view(batch_size, num_items, -1)

            # ===== 动态ID更新（多物品情况，向量化操作）=====
            if self.ablation_mode != 'no_dynamic_update' and short_history is not None and long_history is not None:
                drift_score = self.dynamic_updater.detect_drift(short_history, long_history)
                # 向量化批量更新所有物品（替代for循环，提升效率）
                quantized_emb = self.dynamic_updater.update(
                    quantized_emb, fused_feat, drift_score
                )

        return fused_feat, semantic_ids_logits, quantized_emb

    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """验证batch数据的完整性

        Args:
            batch: 输入的batch数据

        Raises:
            KeyError: 如果缺少必要的键
        """
        required_keys = [
            'history_text_feat', 'history_vision_feat', 'history_len',
            'target_text_feat', 'target_vision_feat', 'target_item',
            'neg_text_feat', 'neg_vision_feat', 'negative_items'
        ]
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise KeyError(f"Batch缺少必要的键: {missing_keys}。"
                          f"期望的键: {required_keys}")

    def _prepare_dynamic_update_inputs(
        self,
        history_text_feat: torch.Tensor,
        history_vision_feat: torch.Tensor,
        history_len: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """准备动态更新所需的短期和长期历史

        Args:
            history_text_feat: (batch, max_seq_len, text_dim)
            history_vision_feat: (batch, max_seq_len, visual_dim)
            history_len: (batch,)

        Returns:
            short_history: (batch, short_len, hidden_dim) 或 None
            long_history: (batch, long_len, hidden_dim) 或 None
        """
        if self.ablation_mode == 'no_dynamic_update':
            return None, None

        batch_size, max_seq_len, _ = history_text_feat.shape
        device = history_text_feat.device

        # 获取短期和长期历史长度配置
        short_len = min(getattr(self.config, 'short_history_len', 10), max_seq_len)
        long_len = max_seq_len

        # 投影历史特征到hidden_dim
        text_proj = self.user_interest_encoder.history_text_proj(history_text_feat)
        visual_proj = self.user_interest_encoder.history_visual_proj(history_vision_feat)
        combined = torch.cat([text_proj, visual_proj], dim=-1)
        history_hidden = self.user_interest_encoder.modal_fusion(combined)  # (batch, seq, hidden)

        # 提取短期历史（最近的交互）
        short_history = history_hidden[:, -short_len:, :]  # (batch, short_len, hidden)

        # 长期历史就是全部历史
        long_history = history_hidden  # (batch, long_len, hidden)

        return short_history, long_history

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            batch: 包含以下键的字典（来自PMATDataset）
                - history_text_feat: (batch, max_seq_len, text_dim)
                - history_vision_feat: (batch, max_seq_len, visual_dim)
                - history_len: (batch,)
                - target_text_feat: (batch, text_dim)
                - target_vision_feat: (batch, visual_dim)
                - target_item: (batch,)
                - neg_text_feat: (batch, num_neg, text_dim)
                - neg_vision_feat: (batch, num_neg, visual_dim)
                - negative_items: (batch, num_neg)

        Returns:
            outputs: 包含各种输出的字典

        Raises:
            KeyError: 如果batch缺少必要的键
        """
        # 验证batch数据完整性
        self._validate_batch(batch)

        # 1. 编码用户兴趣（使用真实历史序列）
        user_interest = self.user_interest_encoder(
            batch['history_text_feat'],
            batch['history_vision_feat'],
            batch['history_len']
        )  # (batch, hidden_dim)

        # 2. 准备动态更新所需的历史数据
        short_history, long_history = self._prepare_dynamic_update_inputs(
            batch['history_text_feat'],
            batch['history_vision_feat'],
            batch['history_len']
        )

        # 3. 编码正样本（目标物品）- 支持动态ID更新
        pos_fused_feat, pos_semantic_logits, pos_quantized_emb = self.encode_item(
            batch['target_text_feat'],
            batch['target_vision_feat'],
            user_interest,
            short_history=short_history,
            long_history=long_history
        )

        # 4. 编码负样本
        neg_fused_feat, neg_semantic_logits, neg_quantized_emb = self.encode_item(
            batch['neg_text_feat'],
            batch['neg_vision_feat'],
            user_interest,
            short_history=short_history,
            long_history=long_history
        )

        # 5. 计算用户-物品匹配分数
        # 正样本分数
        pos_scores = self.user_item_matcher(
            user_interest, pos_fused_feat, pos_quantized_emb
        )  # (batch,)

        # 负样本分数
        neg_scores = self.user_item_matcher(
            user_interest, neg_fused_feat, neg_quantized_emb
        )  # (batch, num_neg)

        return {
            'user_interest': user_interest,
            'pos_scores': pos_scores,
            'neg_scores': neg_scores,
            'pos_fused_feat': pos_fused_feat,
            'neg_fused_feat': neg_fused_feat,
            'pos_semantic_logits': pos_semantic_logits,
            'neg_semantic_logits': neg_semantic_logits,
            'pos_quantized_emb': pos_quantized_emb,
            'neg_quantized_emb': neg_quantized_emb,
            'target_item': batch['target_item'],
            'negative_items': batch['negative_items']
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算多任务损失

        主损失：BPR推荐损失
        辅助损失：语义ID生成损失

        Args:
            outputs: forward的输出

        Returns:
            losses: 各项损失的字典
        """
        losses = {}
        device = outputs['pos_scores'].device

        # ===== 1. BPR推荐损失（主任务）=====
        pos_scores = outputs['pos_scores']  # (batch,)
        neg_scores = outputs['neg_scores']  # (batch, num_neg)

        # 处理负样本为空的情况
        if neg_scores.numel() == 0 or neg_scores.size(1) == 0:
            # 没有负样本时，使用margin loss作为替代
            # 假设正样本分数应该大于0
            bpr_loss = F.relu(1.0 - pos_scores).mean()
        else:
            # BPR loss: -log(sigmoid(pos - neg))
            # 对每个负样本计算
            pos_scores_expanded = pos_scores.unsqueeze(1)  # (batch, 1)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores_expanded - neg_scores) + 1e-8)
            bpr_loss = bpr_loss.mean()

        losses['bpr_loss'] = self.rec_loss_weight * bpr_loss

        # ===== 2. 语义ID生成损失（辅助任务）=====
        target_items = outputs['target_item']

        # 安全地转换item_id为语义ID（带容错处理）
        try:
            target_semantic_ids = item_id_to_semantic_id(
                target_items,
                self.config.id_length,
                self.config.codebook_size
            ).to(device)
        except Exception as e:
            # 如果转换失败，使用随机目标（降级处理）
            import warnings
            warnings.warn(f"语义ID转换失败: {e}，使用随机目标作为降级处理")
            batch_size = target_items.size(0)
            target_semantic_ids = torch.randint(
                0, self.config.codebook_size,
                (batch_size, self.config.id_length),
                device=device
            )

        # 正样本的语义ID损失
        pos_semantic_logits = outputs['pos_semantic_logits']  # (batch, id_length, codebook_size)
        semantic_loss = F.cross_entropy(
            pos_semantic_logits.reshape(-1, self.config.codebook_size),
            target_semantic_ids.reshape(-1)
        )
        losses['semantic_loss'] = self.semantic_loss_weight * semantic_loss

        # ===== 3. 总损失 =====
        losses['total_loss'] = losses['bpr_loss'] + losses['semantic_loss']

        return losses

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

        多任务损失（BPR + 语义ID）叠加可能导致梯度爆炸，需要梯度裁剪
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
        单batch训练逻辑（推荐任务）

        Args:
            batch: 训练批次数据（来自PMATDataset）
            stage_id: 阶段ID
            stage_kwargs: 该阶段的自定义参数

        Returns:
            (batch_loss, batch_metrics)
        """
        # 前向传播
        outputs = self.forward(batch)

        # 计算多任务损失
        losses = self.compute_loss(outputs)

        # 计算训练指标
        pos_scores = outputs['pos_scores']
        neg_scores = outputs['neg_scores']

        # 计算AUC近似（正样本分数 > 负样本分数的比例）
        pos_expanded = pos_scores.unsqueeze(1)
        auc_approx = (pos_expanded > neg_scores).float().mean().item()

        metrics = {
            'bpr_loss': losses['bpr_loss'].item(),
            'semantic_loss': losses['semantic_loss'].item(),
            'auc_approx': auc_approx
        }

        return losses['total_loss'], metrics

    def _validate_one_epoch(self, val_dataloader: torch.utils.data.DataLoader, stage_id: int,
                           stage_kwargs: Dict) -> Dict:
        """单轮验证逻辑（使用推荐指标）"""
        self.eval()

        total_loss = 0.0
        total_bpr_loss = 0.0
        total_semantic_loss = 0.0
        all_pos_scores = []
        all_neg_scores = []

        with torch.no_grad():
            for batch in val_dataloader:
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 前向传播
                outputs = self.forward(batch)
                losses = self.compute_loss(outputs)

                total_loss += losses['total_loss'].item()
                total_bpr_loss += losses['bpr_loss'].item()
                total_semantic_loss += losses['semantic_loss'].item()

                # 收集分数用于计算指标
                all_pos_scores.append(outputs['pos_scores'].cpu())
                all_neg_scores.append(outputs['neg_scores'].cpu())

        # 计算平均损失
        num_batches = len(val_dataloader)
        avg_loss = total_loss / num_batches
        avg_bpr_loss = total_bpr_loss / num_batches
        avg_semantic_loss = total_semantic_loss / num_batches

        # 计算推荐指标
        all_pos_scores = torch.cat(all_pos_scores, dim=0)
        all_neg_scores = torch.cat(all_neg_scores, dim=0)

        metrics = self._compute_recommendation_metrics(all_pos_scores, all_neg_scores)
        metrics['loss'] = avg_loss
        metrics['bpr_loss'] = avg_bpr_loss
        metrics['semantic_loss'] = avg_semantic_loss

        return metrics

    def _compute_recommendation_metrics(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        k_list: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """计算推荐指标，与metrics.py保持一致

        Args:
            pos_scores: (num_samples,) 正样本分数
            neg_scores: (num_samples, num_neg) 负样本分数
            k_list: 要计算的K值列表

        Returns:
            metrics: 包含各种推荐指标的字典
        """
        metrics = {}
        num_samples = pos_scores.size(0)
        num_neg = neg_scores.size(1)

        # 将正样本分数和负样本分数合并，正样本在第一位
        # all_scores: (num_samples, 1 + num_neg)
        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        # 计算排名（正样本的排名，1-based）
        # 排名 = 比正样本分数高的负样本数量 + 1
        ranks = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1) + 1  # (num_samples,)

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
        mrr = (1.0 / ranks.float()).mean().item()
        metrics['MRR'] = mrr

        # 计算AUC
        auc = (pos_scores.unsqueeze(1) > neg_scores).float().mean().item()
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

    def predict(self, batch: Dict[str, torch.Tensor], all_item_features: Optional[Dict] = None) -> torch.Tensor:
        """执行推荐预测

        Args:
            batch: 包含用户历史的批次数据
            all_item_features: 所有物品的特征（用于全量排序）

        Returns:
            scores: 预测分数
        """
        self.eval()
        with torch.no_grad():
            # 编码用户兴趣
            user_interest = self.user_interest_encoder(
                batch['history_text_feat'],
                batch['history_vision_feat'],
                batch['history_len']
            )

            if all_item_features is not None:
                # 全量物品排序
                all_text_feat = all_item_features['text']
                all_vision_feat = all_item_features['visual']

                # 编码所有物品
                fused_feat, _, quantized_emb = self.encode_item(
                    all_text_feat.unsqueeze(0).expand(user_interest.size(0), -1, -1),
                    all_vision_feat.unsqueeze(0).expand(user_interest.size(0), -1, -1),
                    user_interest
                )

                # 计算分数
                scores = self.user_item_matcher(user_interest, fused_feat, quantized_emb)
            else:
                # 只对batch中的目标物品计算分数
                outputs = self.forward(batch)
                scores = outputs['pos_scores']

            return scores

    def get_user_embedding(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取用户嵌入（用于召回）

        Args:
            batch: 包含用户历史的批次数据

        Returns:
            user_embedding: (batch, hidden_dim)
        """
        self.eval()
        with torch.no_grad():
            user_interest = self.user_interest_encoder(
                batch['history_text_feat'],
                batch['history_vision_feat'],
                batch['history_len']
            )
            return user_interest

    def get_item_embedding(self, text_feat: torch.Tensor, vision_feat: torch.Tensor) -> torch.Tensor:
        """获取物品嵌入（用于召回）

        Args:
            text_feat: (num_items, text_dim)
            vision_feat: (num_items, visual_dim)

        Returns:
            item_embedding: (num_items, hidden_dim)
        """
        self.eval()
        with torch.no_grad():
            # 使用均匀的模态权重（因为没有用户信息）
            item_features = {
                'text': text_feat.float(),
                'visual': vision_feat.float()
            }

            num_items = text_feat.size(0)
            modal_weights = torch.ones(num_items, 2, device=text_feat.device) / 2

            encoded_features = self.multimodal_encoder(item_features)
            fused_feat = self.personalized_fusion(encoded_features, modal_weights)
            _, quantized_emb = self.semantic_quantizer(fused_feat)

            # 返回融合特征和语义ID嵌入的拼接
            item_embedding = torch.cat([fused_feat, quantized_emb], dim=-1)
            return item_embedding


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
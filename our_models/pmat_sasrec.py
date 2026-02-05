"""
PMAT-SASRec: PMAT语义增强嵌入 + SASRec强排序骨架

架构:
Item raw features (text_feat, vision_feat)
   ↓
PMAT → semantic-enhanced item embedding
   ↓
SASRec (Transformer with causal mask)
   ↓
Preference score

核心创新:
1. 用PMAT的语义增强嵌入替代SASRec的简单item embedding
2. 保留SASRec的自回归序列建模能力
3. 多任务学习: BPR推荐损失 + 语义ID辅助损失
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

# 复用PMAT的核心模块
from our_models.pmat import (
    MultiModalEncoder,
    PersonalizedFusion,
    SemanticIDQuantizer,
    UserModalAttention,
    DynamicIDUpdater,
    UserItemMatcher
)

# 复用SASRec的Transformer块
from baseline_models.sasrec import TransformerBlock


class PMATItemEncoder(nn.Module):
    """PMAT风格的物品编码器

    将多模态原始特征转换为语义增强的物品嵌入

    流程:
    1. MultiModalEncoder: 投影各模态特征到hidden_dim
    2. UserModalAttention: 根据用户兴趣计算个性化模态权重
    3. PersonalizedFusion: 融合多模态特征
    4. SemanticIDQuantizer: 生成语义ID嵌入作为增强
    5. DynamicIDUpdater: 根据兴趣漂移动态更新语义ID
    6. 残差连接: fused_feat + quantized_emb
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # 多模态编码器
        self.multimodal_encoder = MultiModalEncoder(config)

        # 用户模态偏好感知器（个性化模态权重）
        self.user_modal_attention = UserModalAttention(
            user_dim=config.hidden_dim,
            num_modalities=config.num_modalities,
            hidden_dim=config.hidden_dim
        )

        # 个性化融合
        self.personalized_fusion = PersonalizedFusion(config.hidden_dim)

        # 语义ID量化器
        self.semantic_quantizer = SemanticIDQuantizer(
            hidden_dim=config.hidden_dim,
            codebook_size=config.codebook_size,
            id_length=config.id_length
        )

        # 动态ID更新模块
        self.dynamic_updater = DynamicIDUpdater(
            hidden_dim=config.hidden_dim,
            drift_threshold=getattr(config, 'drift_threshold', 0.3)
        )

        # 融合层: 将fused_feat和quantized_emb组合
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )

        # 可学习的全局模态权重（当没有用户兴趣时使用）
        self.modal_weight = nn.Parameter(torch.ones(2) / 2)

    def forward(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor,
        user_interest: Optional[torch.Tensor] = None,
        short_history: Optional[torch.Tensor] = None,
        long_history: Optional[torch.Tensor] = None,
        return_semantic_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            text_feat: (..., text_dim) 文本特征，支持任意前导维度
            vision_feat: (..., visual_dim) 视觉特征
            user_interest: (batch, hidden_dim) 用户兴趣表征（用于个性化模态权重）
            short_history: (batch, short_len, hidden_dim) 短期历史（用于动态更新）
            long_history: (batch, long_len, hidden_dim) 长期历史（用于动态更新）
            return_semantic_logits: 是否返回语义ID的logits

        Returns:
            item_emb: (..., hidden_dim) 语义增强的物品嵌入
            semantic_logits: (..., id_length, codebook_size) 语义ID logits (可选)
            quantized_emb: (..., hidden_dim) 量化后的嵌入（用于UserItemMatcher）
        """
        # 保存原始形状
        original_shape = text_feat.shape[:-1]
        text_dim = text_feat.shape[-1]
        vision_dim = vision_feat.shape[-1]

        # 展平为2D进行处理
        text_flat = text_feat.reshape(-1, text_dim)
        vision_flat = vision_feat.reshape(-1, vision_dim)
        batch_size = text_flat.size(0)

        # 1. 多模态编码
        item_features = {
            'text': text_flat.float(),
            'visual': vision_flat.float()
        }
        encoded_features = self.multimodal_encoder(item_features)

        # 2. 计算模态权重（个性化或全局）
        if user_interest is not None:
            # 使用用户兴趣计算个性化模态权重
            modal_weights = self.user_modal_attention(user_interest)  # (batch, num_modalities)
            # 如果物品数量与用户数量不同，需要扩展
            if modal_weights.size(0) != batch_size:
                num_items_per_user = batch_size // modal_weights.size(0)
                modal_weights = modal_weights.unsqueeze(1).expand(-1, num_items_per_user, -1)
                modal_weights = modal_weights.reshape(-1, modal_weights.size(-1))
        else:
            # 使用全局模态权重
            modal_weights = F.softmax(self.modal_weight, dim=0)
            modal_weights = modal_weights.unsqueeze(0).expand(batch_size, -1)

        # 3. 融合多模态特征
        fused_feat = self.personalized_fusion(encoded_features, modal_weights)

        # 4. 语义ID量化
        semantic_logits, quantized_emb = self.semantic_quantizer(fused_feat)

        # 5. 动态ID更新（如果提供了历史信息）
        if short_history is not None and long_history is not None:
            # 检测兴趣漂移
            drift_score = self.dynamic_updater.detect_drift(short_history, long_history)
            # 根据漂移分数动态更新语义ID嵌入
            # 需要处理维度匹配
            if quantized_emb.size(0) != drift_score.size(0):
                num_items_per_user = quantized_emb.size(0) // drift_score.size(0)
                quantized_emb_reshaped = quantized_emb.view(drift_score.size(0), num_items_per_user, -1)
                fused_feat_reshaped = fused_feat.view(drift_score.size(0), num_items_per_user, -1)
                quantized_emb_updated = self.dynamic_updater.update(
                    quantized_emb_reshaped, fused_feat_reshaped, drift_score
                )
                quantized_emb = quantized_emb_updated.view(-1, self.hidden_dim)
            else:
                quantized_emb = self.dynamic_updater.update(quantized_emb, fused_feat, drift_score)

        # 6. 组合: concat + projection
        combined = torch.cat([fused_feat, quantized_emb], dim=-1)
        item_emb = self.fusion_layer(combined)

        # 恢复原始形状
        item_emb = item_emb.reshape(*original_shape, self.hidden_dim)
        fused_feat_out = fused_feat.reshape(*original_shape, self.hidden_dim)
        quantized_emb_out = quantized_emb.reshape(*original_shape, self.hidden_dim)

        if return_semantic_logits:
            semantic_logits = semantic_logits.reshape(
                *original_shape, self.config.id_length, self.config.codebook_size
            )
            return item_emb, semantic_logits, quantized_emb_out

        return item_emb, None, quantized_emb_out


class PMAT_SASRec(AbstractTrainableModel):
    """PMAT + SASRec 混合推荐模型

    结构:
    1. PMATItemEncoder: 生成语义增强的物品嵌入（含UserModalAttention和DynamicIDUpdater）
    2. SASRec Transformer: 序列建模
    3. UserItemMatcher: 计算用户-物品偏好分数
    """

    def __init__(
        self,
        config,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(device=device)
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.max_seq_len = getattr(config, 'max_history_len', 50)

        # ===== 物品编码器 (PMAT) =====
        self.item_encoder = PMATItemEncoder(config)

        # ===== 序列编码器 (SASRec) =====
        # 位置嵌入
        self.pos_emb = nn.Embedding(self.max_seq_len, config.hidden_dim)

        # Transformer块
        num_blocks = getattr(config, 'num_transformer_blocks', 2)
        num_heads = config.attention_heads
        dropout_rate = config.dropout

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])

        # 输入层归一化
        self.input_layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # ===== 投影层（分离用户和物品，支持两阶段训练） =====
        # 用户投影层：用于序列编码器输出
        self.user_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )
        # 物品投影层：用于物品编码器输出
        self.item_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )
        # 保留 prediction_layer 作为别名（向后兼容，指向物品投影层）
        self.prediction_layer = self.item_projection

        # ===== 用户-物品匹配层 (来自PMAT) =====
        self.user_item_matcher = UserItemMatcher(config)

        # ===== 损失权重 =====
        self.rec_loss_weight = getattr(config, 'rec_loss_weight', 1.0)
        self.semantic_loss_weight = getattr(config, 'semantic_loss_weight', 0.1)

        # ===== 预计算的物品表征（用于 Cross Entropy 损失） =====
        self._all_item_repr = None  # (num_items, hidden_dim)
        self._all_quantized_emb = None  # (num_items, hidden_dim)

        # 缓存因果掩码
        self._causal_mask_cache = {}

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.input_layer_norm.weight, 1.0)
        nn.init.constant_(self.input_layer_norm.bias, 0.0)

        for module in self.user_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        for module in self.item_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    # ==================== 两阶段训练支持 ====================

    def freeze_item_encoder(self):
        """冻结物品编码器（阶段2使用）"""
        for param in self.item_encoder.parameters():
            param.requires_grad = False
        for param in self.item_projection.parameters():
            param.requires_grad = False
        print("物品编码器已冻结（包括item_projection）")

    def unfreeze_item_encoder(self):
        """解冻物品编码器"""
        for param in self.item_encoder.parameters():
            param.requires_grad = True
        for param in self.item_projection.parameters():
            param.requires_grad = True
        print("物品编码器已解冻（包括item_projection）")

    def freeze_sequence_encoder(self):
        """冻结序列编码器（阶段1使用）"""
        for param in self.pos_emb.parameters():
            param.requires_grad = False
        for param in self.input_layer_norm.parameters():
            param.requires_grad = False
        for block in self.transformer_blocks:
            for param in block.parameters():
                param.requires_grad = False
        for param in self.user_item_matcher.parameters():
            param.requires_grad = False
        print("序列编码器已冻结")

    def unfreeze_sequence_encoder(self):
        """解冻序列编码器"""
        for param in self.pos_emb.parameters():
            param.requires_grad = True
        for param in self.input_layer_norm.parameters():
            param.requires_grad = True
        for block in self.transformer_blocks:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.user_item_matcher.parameters():
            param.requires_grad = True
        print("序列编码器已解冻")

    def compute_pretrain_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算预训练损失（阶段1：对比学习）

        使用模态内和模态间对比学习来预训练物品编码器
        """
        # 获取目标物品的多模态特征
        text_feat = batch['target_text_feat'].to(self.device)  # (batch, text_dim)
        visual_feat = batch['target_vision_feat'].to(self.device)  # (batch, visual_dim)
        batch_size = text_feat.size(0)

        # 编码物品
        item_emb, semantic_logits, quantized_emb = self.item_encoder(
            text_feat, visual_feat, return_semantic_logits=True
        )
        item_repr = self.prediction_layer(item_emb)  # (batch, hidden_dim)

        # 获取各模态的编码
        text_encoded = self.item_encoder.multimodal_encoder.text_encoder(text_feat.float())
        visual_encoded = self.item_encoder.multimodal_encoder.visual_encoder(visual_feat.float())

        temperature = getattr(self.config, 'pretrain_temperature', 0.07)

        # ===== 模态内对比损失 =====
        # 同一batch内，同一物品的不同表征应该相似
        # 使用 item_repr 和 quantized_emb 作为同一物品的两个视图
        item_repr_norm = F.normalize(item_repr, dim=-1)
        quantized_emb_norm = F.normalize(quantized_emb, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(item_repr_norm, quantized_emb_norm.T) / temperature
        labels = torch.arange(batch_size, device=self.device)

        intra_loss = (F.cross_entropy(sim_matrix, labels) +
                      F.cross_entropy(sim_matrix.T, labels)) / 2

        # ===== 模态间对比损失 =====
        # 同一物品的文本和视觉表征应该相似
        text_norm = F.normalize(text_encoded, dim=-1)
        visual_norm = F.normalize(visual_encoded, dim=-1)

        sim_tv = torch.matmul(text_norm, visual_norm.T) / temperature

        inter_loss = (F.cross_entropy(sim_tv, labels) +
                      F.cross_entropy(sim_tv.T, labels)) / 2

        # 加权组合
        intra_weight = getattr(self.config, 'pretrain_intra_weight', 1.0)
        inter_weight = getattr(self.config, 'pretrain_inter_weight', 0.5)

        total_loss = intra_weight * intra_loss + inter_weight * inter_loss

        return {
            'total_loss': total_loss,
            'intra_loss': intra_loss,
            'inter_loss': inter_loss
        }

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """获取因果掩码（带缓存）"""
        cache_key = (seq_len, device)
        if cache_key not in self._causal_mask_cache:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )
            self._causal_mask_cache[cache_key] = causal_mask
        return self._causal_mask_cache[cache_key]

    def encode_sequence(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor,
        seq_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """编码历史序列

        Args:
            text_feat: (batch, seq_len, text_dim)
            vision_feat: (batch, seq_len, visual_dim)
            seq_lens: (batch,) 实际序列长度

        Returns:
            user_repr: (batch, hidden_dim) 用户表示
            seq_output: (batch, seq_len, hidden_dim) 序列输出
            semantic_logits: (batch, seq_len, id_length, codebook_size) 语义ID logits
            short_history: (batch, short_len, hidden_dim) 短期历史（用于动态更新）
            long_history: (batch, long_len, hidden_dim) 长期历史（用于动态更新）
        """
        batch_size, seq_len, _ = text_feat.shape
        device = text_feat.device

        # 0. 创建掩码（提前创建，用于后续处理）
        # Padding mask: 后面的位置是 padding (True 表示被 mask 的位置)
        # 注意：数据是左对齐的，即有效内容在序列开头，padding 在末尾
        # 例如 seq_len=4, seq_lens=1 时，位置 0 是有效的，位置 1-3 是 padding
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        padding_mask = positions >= seq_lens.unsqueeze(1)  # (batch, seq_len)

        # 1. PMAT物品编码（不使用个性化权重，因为还没有用户表示）
        item_emb, semantic_logits, _ = self.item_encoder(
            text_feat, vision_feat, return_semantic_logits=True
        )  # (batch, seq_len, hidden_dim)

        # 2. 添加位置嵌入
        pos_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_indices = pos_indices.clamp(max=self.max_seq_len - 1)
        seq_emb = item_emb + self.pos_emb(pos_indices)

        # 3. 处理 padding 位置：将 padding 位置的嵌入设为小的随机值
        # 这样可以避免 LayerNorm 在处理全零输入时产生 NaN
        # 使用 detach() 确保梯度不会流向这些随机值
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(seq_emb)
        # 生成小的随机噪声（不需要梯度）
        noise = torch.randn_like(seq_emb) * 0.01
        seq_emb = torch.where(padding_mask_expanded, noise.detach(), seq_emb)

        # 4. LayerNorm + Dropout
        seq_emb = self.input_layer_norm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        # Causal mask: 上三角为 -inf，防止看到未来
        causal_mask = self._get_causal_mask(seq_len, device)

        # 5. Transformer编码
        # 使用分离的 padding_mask 和 causal_mask
        # 由于是左对齐，有效位置在前面，causal mask 不会导致有效位置的所有 key 被 mask
        for block in self.transformer_blocks:
            seq_emb = block(seq_emb, padding_mask=padding_mask, causal_mask=causal_mask)

        # 处理可能的 NaN（保护措施）
        seq_emb = torch.nan_to_num(seq_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # 6. 获取用户表示（最后一个有效位置）
        # 注意：序列是左对齐的，有效内容在开头
        # 最后一个有效位置的索引是 seq_lens - 1
        seq_lens_idx = (seq_lens - 1).clamp(min=0, max=seq_len - 1).long()
        user_repr = seq_emb[torch.arange(batch_size, device=device), seq_lens_idx]

        # 7. 用户投影层（注意：使用 user_projection 而不是 prediction_layer）
        user_repr = self.user_projection(user_repr)

        # 8. 准备短期和长期历史（用于动态ID更新）
        short_len = min(getattr(self.config, 'short_history_len', 10), seq_len)
        short_history = seq_emb[:, -short_len:, :]  # (batch, short_len, hidden)
        long_history = seq_emb  # (batch, seq_len, hidden)

        return user_repr, seq_emb, semantic_logits, short_history, long_history

    def encode_items(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor,
        user_interest: Optional[torch.Tensor] = None,
        short_history: Optional[torch.Tensor] = None,
        long_history: Optional[torch.Tensor] = None,
        return_semantic_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """编码候选物品

        Args:
            text_feat: (..., text_dim)
            vision_feat: (..., visual_dim)
            user_interest: (batch, hidden_dim) 用户兴趣（用于个性化模态权重）
            short_history: (batch, short_len, hidden_dim) 短期历史（用于动态更新）
            long_history: (batch, long_len, hidden_dim) 长期历史（用于动态更新）
            return_semantic_logits: 是否返回语义ID logits

        Returns:
            item_repr: (..., hidden_dim) 物品表示
            semantic_logits: 语义ID logits（可选）
            quantized_emb: 量化后的嵌入（用于UserItemMatcher）
        """
        item_emb, semantic_logits, quantized_emb = self.item_encoder(
            text_feat, vision_feat,
            user_interest=user_interest,
            short_history=short_history,
            long_history=long_history,
            return_semantic_logits=return_semantic_logits
        )
        # 对候选物品也应用预测层投影（保持空间一致）
        item_repr = self.prediction_layer(item_emb)
        return item_repr, semantic_logits, quantized_emb

    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """验证batch数据完整性"""
        # Cross Entropy 损失不需要负样本
        required_keys = [
            'history_text_feat', 'history_vision_feat', 'history_len',
            'target_text_feat', 'target_vision_feat', 'target_item'
        ]
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise KeyError(f"Batch缺少必要的键: {missing_keys}")

    def set_all_item_features(self, all_item_features: Dict[str, torch.Tensor]):
        """设置所有物品特征，用于预计算物品表征（Cross Entropy 损失需要）

        Args:
            all_item_features: 包含 'text' 和 'visual' 的字典
        """
        all_text_feat = all_item_features['text'].to(self.device)
        all_visual_feat = all_item_features['visual'].to(self.device)
        num_items = all_text_feat.shape[0]

        print(f"预计算 {num_items} 个物品的表征（用于 Cross Entropy 损失）...")

        item_batch_size = 256
        all_item_repr_list = []
        all_quantized_emb_list = []

        with torch.no_grad():
            for start_idx in range(0, num_items, item_batch_size):
                end_idx = min(start_idx + item_batch_size, num_items)
                item_text = all_text_feat[start_idx:end_idx]
                item_visual = all_visual_feat[start_idx:end_idx]

                # 使用全局模态权重编码
                item_emb, _, quantized_emb = self.item_encoder(
                    item_text, item_visual,
                    user_interest=None,
                    return_semantic_logits=False
                )
                item_repr = self.prediction_layer(item_emb)
                all_item_repr_list.append(item_repr)
                all_quantized_emb_list.append(quantized_emb)

        self._all_item_repr = torch.cat(all_item_repr_list, dim=0)  # (num_items, hidden_dim)
        self._all_quantized_emb = torch.cat(all_quantized_emb_list, dim=0)  # (num_items, hidden_dim)
        # 预先 L2 归一化，避免每次 forward 重复计算
        self._all_item_repr = F.normalize(self._all_item_repr, dim=-1)
        print(f"物品表征预计算完成，形状: {self._all_item_repr.shape}（已L2归一化）")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            batch: 包含以下键的字典
                - history_text_feat: (batch, seq_len, text_dim)
                - history_vision_feat: (batch, seq_len, visual_dim)
                - history_len: (batch,)
                - target_text_feat: (batch, text_dim)
                - target_vision_feat: (batch, visual_dim)
                - target_item: (batch,) 目标物品ID

        Returns:
            outputs: 包含 logits 和中间结果的字典
        """
        self._validate_batch(batch)

        # 1. 编码历史序列 → 用户表示
        user_repr, seq_output, history_semantic_logits, short_history, long_history = self.encode_sequence(
            batch['history_text_feat'],
            batch['history_vision_feat'],
            batch['history_len']
        )  # user_repr: (batch, hidden_dim)

        # 2. 计算对所有物品的 logits（Cross Entropy 损失）
        if self._all_item_repr is not None:
            # 使用预计算的物品表征计算 logits
            # 重要：使用 L2 归一化 + 温度缩放，避免数值不稳定
            # 注意：_all_item_repr 已经在 set_all_item_features 中预先归一化了
            temperature = getattr(self.config, 'logit_temperature', 0.1)
            user_repr_norm = F.normalize(user_repr, dim=-1)
            logits = torch.matmul(user_repr_norm, self._all_item_repr.T) / temperature  # (batch, num_items)
        else:
            # 如果没有预计算，使用 None（会在 compute_loss 中处理）
            logits = None

        # 3. 编码正样本（用于语义ID损失）
        pos_repr, pos_semantic_logits, pos_quantized_emb = self.encode_items(
            batch['target_text_feat'],
            batch['target_vision_feat'],
            user_interest=user_repr,
            short_history=short_history,
            long_history=long_history,
            return_semantic_logits=True
        )

        return {
            'user_repr': user_repr,
            'logits': logits,  # (batch, num_items) 对所有物品的分数
            'target_item': batch['target_item'],  # (batch,) 目标物品ID
            'pos_semantic_logits': pos_semantic_logits,
            'history_semantic_logits': history_semantic_logits,
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算多任务损失

        主损失: Cross Entropy 推荐损失（对所有物品的 softmax）
        辅助损失: 语义ID生成损失
        """
        losses = {}

        # ===== 1. Cross Entropy 推荐损失 =====
        logits = outputs['logits']  # (batch, num_items)
        target_items = outputs['target_item']  # (batch,)

        if logits is not None:
            # Cross Entropy loss: softmax over all items
            ce_loss = F.cross_entropy(logits, target_items)
            losses['ce_loss'] = self.rec_loss_weight * ce_loss
        else:
            # 如果没有预计算物品表征，使用占位损失
            losses['ce_loss'] = torch.tensor(0.0, device=target_items.device)

        # ===== 2. 语义ID损失（辅助任务）=====
        device = target_items.device

        try:
            target_semantic_ids = item_id_to_semantic_id(
                target_items,
                self.config.id_length,
                self.config.codebook_size
            ).to(device)
        except Exception as e:
            import warnings
            warnings.warn(f"语义ID转换失败: {e}")
            batch_size = target_items.size(0)
            target_semantic_ids = torch.randint(
                0, self.config.codebook_size,
                (batch_size, self.config.id_length),
                device=device
            )

        pos_semantic_logits = outputs['pos_semantic_logits']
        semantic_loss = F.cross_entropy(
            pos_semantic_logits.reshape(-1, self.config.codebook_size),
            target_semantic_ids.reshape(-1)
        )
        losses['semantic_loss'] = self.semantic_loss_weight * semantic_loss

        # ===== 3. 总损失 =====
        losses['total_loss'] = losses['ce_loss'] + losses['semantic_loss']

        return losses

    # ==================== AbstractTrainableModel 抽象方法实现 ====================

    def _get_optimizer(self, stage_id: int, stage_kwargs: Dict) -> torch.optim.Optimizer:
        """获取优化器"""
        lr = stage_kwargs.get('lr', 0.001)
        weight_decay = stage_kwargs.get('weight_decay', 0.01)
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_optimizer_state_dict(self) -> Dict:
        """获取优化器状态"""
        optimizer_states = {}
        for stage_id, optimizer in self._stage_optimizers.items():
            optimizer_states[stage_id] = optimizer.state_dict()
        return optimizer_states

    def _load_optimizer_state_dict(self, state_dict: Dict):
        """加载优化器状态"""
        for stage_id, opt_state in state_dict.items():
            if stage_id in self._stage_optimizers:
                self._stage_optimizers[stage_id].load_state_dict(opt_state)

    def _update_params(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, scaler=None):
        """参数更新（带梯度裁剪）"""
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

    def _train_one_batch(self, batch: Any, stage_id: int, stage_kwargs: Dict) -> Tuple[torch.Tensor, Dict]:
        """单batch训练

        stage_id=0: 预训练阶段（对比学习，只训练物品编码器）
        stage_id=1: 推荐阶段（Cross Entropy，只训练序列编码器）
        """
        if stage_id == 0:
            # 阶段1：预训练物品编码器
            losses = self.compute_pretrain_loss(batch)
            metrics = {
                'intra_loss': losses['intra_loss'].item(),
                'inter_loss': losses['inter_loss'].item(),
            }
            return losses['total_loss'], metrics
        else:
            # 阶段2：训练序列模型
            outputs = self.forward(batch)
            losses = self.compute_loss(outputs)
            metrics = {
                'ce_loss': losses['ce_loss'].item(),
                'semantic_loss': losses['semantic_loss'].item(),
            }
            return losses['total_loss'], metrics

    def _validate_one_epoch(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        stage_id: int,
        stage_kwargs: Dict
    ) -> Dict:
        """单轮验证 - Full Ranking评估

        使用Full Ranking评估协议：对所有物品计算分数，然后排序
        """
        self.eval()

        # 获取所有物品特征（从stage_kwargs中获取）
        all_item_features = stage_kwargs.get('all_item_features', None)
        if all_item_features is None:
            raise ValueError("Full Ranking评估需要提供all_item_features")

        all_text_feat = all_item_features['text'].to(self.device)
        all_visual_feat = all_item_features['visual'].to(self.device)
        num_items = all_text_feat.shape[0]

        all_target_items = []
        all_ranks = []

        # 添加进度条
        from tqdm import tqdm

        # ========== 使用预计算的物品表征（与 Cross Entropy 训练一致） ==========
        # 如果已经预计算了物品表征，直接使用；否则重新计算
        if self._all_item_repr is not None:
            all_item_repr = self._all_item_repr
            print(f"使用预计算的物品表征，形状: {all_item_repr.shape}")
        else:
            print(f"预计算 {num_items} 个物品的表征...")
            item_batch_size = 256
            all_item_repr_list = []

            with torch.no_grad():
                for start_idx in tqdm(range(0, num_items, item_batch_size), desc="编码物品", leave=False):
                    end_idx = min(start_idx + item_batch_size, num_items)
                    item_text = all_text_feat[start_idx:end_idx]
                    item_visual = all_visual_feat[start_idx:end_idx]

                    item_emb, _, _ = self.item_encoder(
                        item_text, item_visual,
                        user_interest=None,
                        return_semantic_logits=False
                    )
                    item_repr = self.prediction_layer(item_emb)
                    all_item_repr_list.append(item_repr.cpu())

                all_item_repr = torch.cat(all_item_repr_list, dim=0).to(self.device)
                # 重要：需要归一化以与训练一致
                all_item_repr = F.normalize(all_item_repr, dim=-1)

            print(f"物品表征预计算完成，形状: {all_item_repr.shape}（已L2归一化）")

        # ========== 评估用户-物品匹配（使用点积，与 Cross Entropy 训练一致） ==========
        val_pbar = tqdm(val_dataloader, desc=f"Stage {stage_id} Validate", leave=False)

        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                batch_size = batch['history_text_feat'].shape[0]
                target_items = batch['target_item']  # (batch,)
                all_target_items.append(target_items.cpu())

                # 1. 编码用户序列
                user_repr, _, _, _, _ = self.encode_sequence(
                    batch['history_text_feat'],
                    batch['history_vision_feat'],
                    batch['history_len']
                )

                # 2. 使用点积计算分数（与 Cross Entropy 训练一致！）
                # 重要：必须使用与训练时相同的归一化和温度缩放
                temperature = getattr(self.config, 'logit_temperature', 0.1)
                user_repr_norm = F.normalize(user_repr, dim=-1)
                # 注意：all_item_repr 已经在 set_all_item_features 中预先归一化了
                all_scores = torch.matmul(user_repr_norm, all_item_repr.T) / temperature  # (batch, num_items)

                # 3. 计算目标物品的排名
                target_scores = all_scores[torch.arange(batch_size, device=self.device), target_items]
                ranks = (all_scores >= target_scores.unsqueeze(1)).sum(dim=1)  # (batch,)
                all_ranks.append(ranks.cpu())

        # 合并所有batch的结果
        all_target_items = torch.cat(all_target_items, dim=0)
        all_ranks = torch.cat(all_ranks, dim=0).float()

        # 计算指标
        metrics = self._compute_full_ranking_metrics(all_ranks, num_items)

        return metrics

    def _compute_full_ranking_metrics(
        self,
        ranks: torch.Tensor,
        num_items: int,
        k_list: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """计算Full Ranking评估指标

        Args:
            ranks: (num_samples,) 每个样本的目标物品排名（1-based）
            num_items: 总物品数量
            k_list: 计算指标的K值列表

        Returns:
            metrics: 各项推荐指标
        """
        metrics = {}
        num_samples = ranks.size(0)

        for k in k_list:
            # HR@K: 目标物品排名在前K的比例
            hits = (ranks <= k).float()
            metrics[f'HR@{k}'] = hits.mean().item()

            # NDCG@K: 只有排名在前K的才有贡献
            dcg = (ranks <= k).float() / torch.log2(ranks.float() + 1)
            metrics[f'NDCG@{k}'] = dcg.mean().item()

            # MRR@K: 只考虑排名在前K的
            mrr = (ranks <= k).float() / ranks.float()
            metrics[f'MRR@{k}'] = mrr.mean().item()

        # 全局MRR
        metrics['MRR'] = (1.0 / ranks.float()).mean().item()

        # 平均排名
        metrics['Mean_Rank'] = ranks.mean().item()

        return metrics

    def _compute_recommendation_metrics(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        k_list: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """计算推荐指标（已废弃，保留用于兼容性）

        注意：此方法使用Sampled Evaluation，已被Full Ranking替代
        """
        metrics = {}
        num_samples = pos_scores.size(0)
        num_neg = neg_scores.size(1)

        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        _, sorted_indices = torch.sort(all_scores, dim=1, descending=True)
        predictions = sorted_indices.tolist()
        ground_truth = [[0] for _ in range(num_samples)]

        for k in k_list:
            precision = 0.0
            recall = 0.0
            ndcg = 0.0
            mrr_k = 0.0

            for i in range(num_samples):
                pred_k = predictions[i][:k]
                gt = ground_truth[i]

                hits = len(set(pred_k) & set(gt))
                precision += hits / k
                recall += hits / len(gt) if len(gt) > 0 else 0

                dcg = 0.0
                for j, item in enumerate(pred_k):
                    if item in gt:
                        dcg += 1.0 / np.log2(j + 2)
                idcg = 1.0 / np.log2(2)
                ndcg += dcg / idcg if idcg > 0 else 0

                for j, item in enumerate(pred_k):
                    if item in gt:
                        mrr_k += 1.0 / (j + 1)
                        break

            metrics[f'Precision@{k}'] = precision / num_samples
            metrics[f'Recall@{k}'] = recall / num_samples
            metrics[f'HR@{k}'] = recall / num_samples
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

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        all_item_features: Optional[Dict] = None
    ) -> torch.Tensor:
        """执行推荐预测

        Args:
            batch: 包含用户历史的批次数据
            all_item_features: 所有物品的特征（用于全量排序）

        Returns:
            scores: 预测分数
        """
        self.eval()
        with torch.no_grad():
            # 编码用户
            user_repr, _, _, short_history, long_history = self.encode_sequence(
                batch['history_text_feat'],
                batch['history_vision_feat'],
                batch['history_len']
            )

            if all_item_features is not None:
                # 全量物品排序
                all_text_feat = all_item_features['text']
                all_vision_feat = all_item_features['visual']

                item_repr, _, quantized_emb = self.encode_items(
                    all_text_feat, all_vision_feat,
                    user_interest=user_repr,
                    short_history=short_history,
                    long_history=long_history
                )

                # 使用UserItemMatcher计算分数
                scores = self.user_item_matcher(user_repr, item_repr, quantized_emb)
            else:
                # 只对batch中的目标物品计算分数
                outputs = self.forward(batch)
                scores = outputs['pos_scores']

            return scores

    def get_user_embedding(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取用户嵌入"""
        self.eval()
        with torch.no_grad():
            user_repr, _, _, _, _ = self.encode_sequence(
                batch['history_text_feat'],
                batch['history_vision_feat'],
                batch['history_len']
            )
            return user_repr

    def get_item_embedding(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor
    ) -> torch.Tensor:
        """获取物品嵌入"""
        self.eval()
        with torch.no_grad():
            item_repr, _, quantized_emb = self.encode_items(text_feat, vision_feat)
            # 返回融合特征和语义ID嵌入的拼接
            item_embedding = torch.cat([item_repr, quantized_emb], dim=-1)
            return item_embedding


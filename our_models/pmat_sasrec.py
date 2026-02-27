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
        fused_feat = F.normalize(fused_feat, dim=-1)
        semantic_logits, quantized_emb = self.semantic_quantizer(fused_feat)
        # 新增：重建损失（让quantized_emb尽可能还原fused_feat）
        recon_loss = F.mse_loss(quantized_emb, fused_feat.detach())  # detach避免梯度冲突
        # 新增残差连接：让量化嵌入向多模态特征靠拢
        quantized_emb = 0.7 * quantized_emb + 0.3 * fused_feat  # 0.3为残差权重，可调整
        quantized_emb = F.layer_norm(quantized_emb, normalized_shape=[self.hidden_dim])


        # 5. 动态ID更新（如果提供了历史信息）
        if short_history is not None and long_history is not None:
            # 检测兴趣漂移
            drift_score = self.dynamic_updater.detect_drift(short_history, long_history)
            # 根据漂移分数动态更新语义ID嵌入
            # 需要处理维度匹配
            if quantized_emb.size(0) != drift_score.size(0):
                num_users = drift_score.size(0)
                num_items_total = quantized_emb.size(0)

                # 关键：检查是否能整除，避免view操作崩溃
                if num_items_total % num_users != 0:
                    # 无法整除时，跳过动态更新（不报错，保留训练）
                    import warnings
                    warnings.warn(
                        f"物品数量({num_items_total})不能被用户数量({num_users})整除，跳过本轮动态ID更新",
                        UserWarning
                    )
                    # 直接跳过后续更新逻辑，避免崩溃
                    pass
                else:
                    # 能整除时，执行你的原有逻辑
                    num_items_per_user = num_items_total // num_users
                    quantized_emb_reshaped = quantized_emb.view(num_users, num_items_per_user, -1)
                    fused_feat_reshaped = fused_feat.view(num_users, num_items_per_user, -1)
                    quantized_emb_updated = self.dynamic_updater.update(
                        quantized_emb_reshaped, fused_feat_reshaped, drift_score
                    )
                    quantized_emb = quantized_emb_updated.view(-1, self.hidden_dim)
            else:
                # 维度匹配时，正常更新
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
            return item_emb, semantic_logits, quantized_emb_out, recon_loss

        return item_emb, None, quantized_emb_out, recon_loss


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
        """计算预训练损失（阶段1：对比学习，只训练物品编码器）
        优化点：使用batch内真实负采样（4个）替代随机负样本，强化语义ID区分度
        """
        # 获取目标物品（正样本）的多模态特征
        text_feat = batch['target_text_feat'].to(self.device)  # (batch, text_dim)
        visual_feat = batch['target_vision_feat'].to(self.device)  # (batch, visual_dim)
        batch_size = text_feat.size(0)

        # 编码正样本物品（获取语义ID）
        item_emb, semantic_logits, quantized_emb, recon_loss = self.item_encoder(
            text_feat, visual_feat, return_semantic_logits=True
        )
        pos_repr = self.prediction_layer(item_emb)  # (batch, hidden_dim)

        temperature = getattr(self.config, 'pretrain_temperature', 0.1)  # 降低温度强化区分度

        # ===== 1. 模态内对比损失（正样本自身：item_repr vs quantized_emb） =====
        pos_repr_norm = F.normalize(pos_repr, dim=-1)
        quantized_emb_norm = F.normalize(quantized_emb, dim=-1)

        sim_matrix_intra = torch.matmul(pos_repr_norm, quantized_emb_norm.T) / temperature
        labels_intra = torch.arange(batch_size, device=self.device)
        intra_loss = (F.cross_entropy(sim_matrix_intra, labels_intra) +
                      F.cross_entropy(sim_matrix_intra.T, labels_intra)) / 2

        # ===== 2. 模态间对比损失（正样本：文本 vs 视觉） =====
        text_encoded = self.item_encoder.multimodal_encoder.text_encoder(text_feat.float())
        visual_encoded = self.item_encoder.multimodal_encoder.visual_encoder(visual_feat.float())
        text_norm = F.normalize(text_encoded, dim=-1)
        visual_norm = F.normalize(visual_encoded, dim=-1)

        sim_matrix_inter = torch.matmul(text_norm, visual_norm.T) / temperature
        labels_inter = torch.arange(batch_size, device=self.device)
        inter_loss = (F.cross_entropy(sim_matrix_inter, labels_inter) +
                      F.cross_entropy(sim_matrix_inter.T, labels_inter)) / 2


        # ===== 4. 加权组合所有损失 =====
        intra_weight = getattr(self.config, 'pretrain_intra_weight', 1.0)
        inter_weight = getattr(self.config, 'pretrain_inter_weight', 0.5)
        recon_loss_weight = getattr(self.config, 'recon_loss_weight', 0.8)

        total_loss = (intra_weight * intra_loss +
                      inter_weight * inter_loss +
                      recon_loss_weight * recon_loss)

    
        return {
            'total_loss': total_loss,
            'intra_loss': intra_loss,
            'inter_loss': inter_loss,
            'recon_loss': recon_loss,
        }

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
            生成PyTorch Transformer专用的float型因果掩码（0=可关注，-inf=不可关注）
            确保上三角全为-inf，对角线全为0
            """
        mask = (torch.triu(torch.ones(seq_len, seq_len, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def clear_causal_mask_cache(self):
        """清理因果掩码缓存（释放显存，关键优化）"""
        self._causal_mask_cache.clear()
        # 可选：打印清理日志，方便调试
        # print(f"因果掩码缓存已清理，释放显存约 {len(self._causal_mask_cache)} 个掩码")

    def encode_sequence(
            self,
            text_feat: torch.Tensor,
            vision_feat: torch.Tensor,
            seq_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """编码历史序列（左对齐适配版，与pure_sasrec完全一致）"""
        batch_size, seq_len, _ = text_feat.shape
        device = text_feat.device

        # ========== 1. 左对齐Padding Mask（核心修改） ==========
        # 左对齐：padding在末尾，sum(dim=-1)==0的位置是padding
        padding_mask = (text_feat.sum(dim=-1) == 0)  # (batch, seq_len)，True=padding（末尾）
        # 兼容seq_lens（确保和pure_sasrec一致）
        valid_len_mask = torch.arange(seq_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        padding_mask = padding_mask | (~valid_len_mask)  # 双重保障

        # ========== 2. PMAT多模态物品编码 ==========
        item_emb, semantic_logits, _ = self.item_encoder(
            text_feat, vision_feat, return_semantic_logits=True
        )  # (batch, seq_len, hidden_dim)

        # ========== 3. 左对齐位置编码（核心修改） ==========
        # 左对齐：有效位置从0开始连续编码，padding位置编码为0
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)  # (batch, seq_len)
        position_ids = position_ids * (~padding_mask).long()  # padding位置置0
        # 防止位置编码越界
        position_ids = position_ids.clamp(min=0, max=self.max_seq_len - 1)
        seq_emb = item_emb + self.pos_emb(position_ids)

        # ========== 4. Padding位置设为0（左对齐兼容） ==========
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand(seq_emb.size())
        seq_emb = seq_emb.masked_fill(padding_mask_expanded, 0.0)

        # ========== 5. LayerNorm + Dropout（Pre-LN） ==========
        seq_emb = self.input_layer_norm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        # ========== 6. 2D因果掩码（兼容左对齐） ==========
        causal_mask = self._get_causal_mask(seq_len, device)  # [seq_len, seq_len]

        # ========== 7. Transformer编码（左对齐兼容） ==========
        for block in self.transformer_blocks:
            seq_emb = block(
                seq_emb,
                padding_mask=padding_mask,  # [batch, seq_len] 左对齐Padding Mask
                attn_mask=causal_mask  # [seq_len, seq_len] 因果掩码
            )

        # 防止NaN
        seq_emb = torch.nan_to_num(seq_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # ========== 8. 提取用户表征（左对齐逻辑） ==========
        # 左对齐：最后有效位置 = seq_lens - 1（和pure_sasrec一致）
        last_indices = torch.clamp(seq_lens - 1, min=0)  # (batch,)
        batch_idx = torch.arange(batch_size, device=device)
        user_repr = seq_emb[batch_idx, last_indices, :]  # (batch, hidden_dim)

        # ========== 9. 用户投影层 ==========
        user_repr = self.user_projection(user_repr)

        # ========== 10. 短期/长期历史（左对齐兼容） ==========
        short_len = min(getattr(self.config, 'short_history_len', 10), seq_len)
        short_history = seq_emb[:, -short_len:, :]  # 左对齐下，最后short_len个是最新的
        long_history = seq_emb

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
            temperature = getattr(self.config, 'logit_temperature', 0.8)
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

        # try:
        #     target_semantic_ids = item_id_to_semantic_id(
        #         target_items,
        #         self.config.id_length,
        #         self.config.codebook_size
        #     ).to(device)
        # except Exception as e:
        #     import warnings
        #     warnings.warn(f"语义ID转换失败: {e}")
        #     batch_size = target_items.size(0)
        #     target_semantic_ids = torch.randint(
        #         0, self.config.codebook_size,
        #         (batch_size, self.config.id_length),
        #         device=device
        #     )
        #
        # pos_semantic_logits = outputs['pos_semantic_logits']
        # semantic_loss = F.cross_entropy(
        #     pos_semantic_logits.reshape(-1, self.config.codebook_size),
        #     target_semantic_ids.reshape(-1)
        # )
        # losses['semantic_loss'] = self.semantic_loss_weight * semantic_loss

        # ===== 3. 总损失 =====
        losses['total_loss'] = losses['ce_loss']

        return losses

    # ==================== AbstractTrainableModel 抽象方法实现 ====================

    def _get_optimizer(self, stage_id: int, stage_kwargs: Dict) -> torch.optim.Optimizer:
        """获取优化器"""
        lr = stage_kwargs.get('lr', 0.001)
        weight_decay = stage_kwargs.get('weight_decay', 0.01)
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_scheduler(self, optimizer: torch.optim.Optimizer, stage_id: int, stage_kwargs: Dict) -> torch.optim.lr_scheduler.LRScheduler:
        """RecBole官方：StepLR（不是余弦退火）"""
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)

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
                # 'semantic_loss': losses['semantic_loss'].item(),
            }
            return losses['total_loss'], metrics

    import torch
    import torch.nn.functional as F
    from typing import Dict

    def _validate_one_epoch(
            self,
            val_dataloader: torch.utils.data.DataLoader,
            stage_id: int,
            stage_kwargs: Dict
    ) -> Dict:
        self.eval()
        self.clear_causal_mask_cache()

        # ========== 基础配置 ==========
        all_item_features = stage_kwargs.get('all_item_features', None)
        if all_item_features is None:
            raise ValueError("Full Ranking评估需要提供all_item_features")

        all_text_feat = all_item_features['text'].to(self.device, non_blocking=True)
        all_visual_feat = all_item_features['visual'].to(self.device, non_blocking=True)
        num_items = all_text_feat.shape[0]
        hidden_dim = self.config.hidden_dim
        temperature = 0.8

        # ========== 第一步：遍历所有batch，获取全局最大长度 ==========
        print("检测全局最大序列长度（解决维度不匹配）...")
        max_history_len = 0  # history_items的最大长度
        max_short_len = 0  # short_history的最大长度
        max_long_len = 0  # long_history的最大长度

        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # 更新history_items的最大长度
                if batch['history_items'].shape[1] > max_history_len:
                    max_history_len = batch['history_items'].shape[1]

                # 编码序列，更新short/long history的最大长度
                _, seq_emb, _, short_history, long_history = self.encode_sequence(
                    batch['history_text_feat'],
                    batch['history_vision_feat'],
                    batch['history_len']
                )
                if short_history.shape[1] > max_short_len:
                    max_short_len = short_history.shape[1]
                if long_history.shape[1] > max_long_len:
                    max_long_len = long_history.shape[1]

        # ========== 第二步：重新遍历，统一所有张量到最大长度 ==========
        all_user_repr = []
        all_user_short_history = []
        all_user_long_history = []
        all_target_items = []
        all_history_ids = []
        all_seq_lens = []

        print(f"统一序列长度：history={max_history_len}, short={max_short_len}, long={max_long_len}")
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                batch_size = batch['history_text_feat'].shape[0]

                # ========== 修复1：统一history_items长度 ==========
                curr_history_len = batch['history_items'].shape[1]
                if curr_history_len < max_history_len:
                    # 左对齐：在序列维度右侧padding 0（和data_utils一致）
                    pad_len = max_history_len - curr_history_len
                    batch['history_items'] = F.pad(
                        batch['history_items'],
                        (0, pad_len),  # 仅在序列维度右侧padding
                        mode='constant',
                        value=0
                    )

                # ========== 编码用户序列 ==========
                user_repr, seq_emb, _, short_history, long_history = self.encode_sequence(
                    batch['history_text_feat'],
                    batch['history_vision_feat'],
                    batch['history_len']
                )

                # ========== 修复2：统一short_history长度 ==========
                if short_history.shape[1] < max_short_len:
                    pad_len = max_short_len - short_history.shape[1]
                    short_history = F.pad(
                        short_history,
                        (0, 0, 0, pad_len),  # (emb_dim_left, emb_dim_right, seq_left, seq_right)
                        mode='constant',
                        value=0.0
                    )

                # ========== 修复3：统一long_history长度 ==========
                if long_history.shape[1] < max_long_len:
                    pad_len = max_long_len - long_history.shape[1]
                    long_history = F.pad(
                        long_history,
                        (0, 0, 0, pad_len),
                        mode='constant',
                        value=0.0
                    )

                # ========== 收集所有信息（现在维度完全一致） ==========
                all_user_repr.append(F.normalize(user_repr, dim=-1))
                all_user_short_history.append(short_history)
                all_user_long_history.append(long_history)
                all_target_items.append(batch['target_item'].cpu())
                all_history_ids.append(batch['history_items'])
                all_seq_lens.append(batch['history_len'])

        # ========== 第三步：合并所有张量（维度完全匹配） ==========
        all_user_repr = torch.cat(all_user_repr, dim=0)  # (total_users, hidden_dim)
        all_user_short_history = torch.cat(all_user_short_history, dim=0)  # (total_users, max_short_len, hidden_dim)
        all_user_long_history = torch.cat(all_user_long_history, dim=0)  # (total_users, max_long_len, hidden_dim)
        all_target_items = torch.cat(all_target_items, dim=0)  # (total_users,)
        all_history_ids = torch.cat(all_history_ids, dim=0)  # (total_users, max_history_len)
        all_seq_lens = torch.cat(all_seq_lens, dim=0)  # (total_users,)
        total_users = all_user_repr.shape[0]

        # ========== 第四步：预计算个性化物品表征（核心修改：逐用户编码，保留动态ID） ==========
        all_item_repr = None
        if stage_id == 1:
            print("预计算个性化物品表征（仅1次）...")
            # 减小chunk size，降低单批次内存占用
            item_chunk_size = 512 if torch.cuda.is_available() else 128
            all_item_repr_list = []

            with torch.no_grad():
                for start_idx in range(0, num_items, item_chunk_size):
                    end_idx = min(start_idx + item_chunk_size, num_items)
                    chunk_text = all_text_feat[start_idx:end_idx]
                    chunk_vision = all_visual_feat[start_idx:end_idx]
                    chunk_size = end_idx - start_idx

                    # 逐用户编码当前物品chunk（核心：避免全局扩展导致cdist爆炸）
                    chunk_item_repr_list = []
                    for u in range(total_users):
                        # 取单个用户的表征和历史
                        u_repr = all_user_repr[u:u + 1]  # (1, hidden_dim)
                        u_short = all_user_short_history[u:u + 1]  # (1, max_short_len, hidden_dim)
                        u_long = all_user_long_history[u:u + 1]  # (1, max_long_len, hidden_dim)

                        # 扩展当前chunk物品到单个用户维度
                        u_chunk_text = chunk_text.unsqueeze(0).expand(1, chunk_size, -1).reshape(-1,
                                                                                                 chunk_text.shape[-1])
                        u_chunk_vision = chunk_vision.unsqueeze(0).expand(1, chunk_size, -1).reshape(-1,
                                                                                                     chunk_vision.shape[
                                                                                                         -1])
                        u_user_interest = u_repr.unsqueeze(1).expand(1, chunk_size, hidden_dim).reshape(-1, hidden_dim)

                        # 扩展用户历史到当前chunk长度（匹配物品维度）
                        u_short_expand = u_short.expand(chunk_size, u_short.shape[1], u_short.shape[2])
                        u_long_expand = u_long.expand(chunk_size, u_long.shape[1], u_long.shape[2])

                        # 个性化编码（保留动态ID更新，cdist仅计算单个用户）
                        item_emb, _, _ = self.item_encoder(
                            u_chunk_text,
                            u_chunk_vision,
                            user_interest=u_user_interest,
                            short_history=u_short_expand,
                            long_history=u_long_expand
                        )
                        item_repr = self.prediction_layer(item_emb)
                        item_repr_norm = F.normalize(item_repr, dim=-1)
                        # 恢复维度：(chunk_size, hidden_dim) → (1, chunk_size, hidden_dim)
                        chunk_item_repr_list.append(item_repr_norm.unsqueeze(0))

                    # 合并当前chunk所有用户的物品表征
                    chunk_item_repr = torch.cat(chunk_item_repr_list, dim=0)  # (total_users, chunk_size, hidden_dim)
                    all_item_repr_list.append(chunk_item_repr)

            # 合并所有chunk的物品表征
            all_item_repr = torch.cat(all_item_repr_list, dim=1)  # (total_users, num_items, hidden_dim)

        # ========== 第五步：计算全量分数 ==========
        with torch.no_grad():
            if stage_id == 1:
                # Stage2：个性化分数
                all_scores = torch.bmm(
                    all_user_repr.unsqueeze(1),
                    all_item_repr.transpose(1, 2)
                ).squeeze(1) / temperature
            else:
                # Stage1：全局分数
                if self._all_item_repr is None:
                    self.set_all_item_features(all_item_features)
                all_item_repr = self._all_item_repr
                all_scores = torch.matmul(all_user_repr, all_item_repr.T) / temperature

            # ========== 修复4：向量化历史屏蔽（适配统一长度） ==========
            history_mask = torch.zeros((total_users, num_items), dtype=torch.bool, device=self.device)
            # 左对齐：仅保留seq_lens内的有效历史（过滤padding的0）
            seq_range = torch.arange(max_history_len, device=self.device).unsqueeze(0)
            valid_seq_mask = seq_range < all_seq_lens.unsqueeze(1)  # (total_users, max_history_len)
            valid_id_mask = (all_history_ids != 0) & (all_history_ids < num_items) & valid_seq_mask

            # 提取有效索引（向量化，无循环）
            batch_indices = torch.arange(total_users, device=self.device).unsqueeze(1).expand(-1, max_history_len)[
                valid_id_mask]
            item_indices = all_history_ids[valid_id_mask]

            if len(batch_indices) > 0:
                history_mask[batch_indices, item_indices] = True

            # 释放目标物品
            history_mask[torch.arange(total_users), all_target_items.to(self.device)] = False
            all_scores = all_scores.masked_fill(history_mask, -float('inf'))

            # ========== 计算排名 ==========
            target_scores = all_scores[torch.arange(total_users), all_target_items.to(self.device)].unsqueeze(1)
            all_ranks = (all_scores >= target_scores).sum(dim=1).float().cpu()

        # ========== 计算最终指标 ==========
        metrics = self._compute_metrics(all_ranks, k_list=[5, 10, 20])
        print(f"\n===== Stage {stage_id} 验证结果（最终修复版） =====")
        print(
            f"HR@10: {metrics['HR@10']:.4f} | NDCG@10: {metrics['NDCG@10']:.4f} | Mean_Rank: {metrics['Mean_Rank']:.4f}")

        return metrics

    def _compute_metrics(self, ranks: torch.Tensor, k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
        metrics = {}
        for k in k_list:
            hits = (ranks <= k).float()
            metrics[f'HR@{k}'] = hits.mean().item()
            metrics[f'hit@{k}'] = hits.mean().item()  # 新增：RecBole原生hit@k
            metrics[f'recall@{k}'] = hits.mean().item()  # 新增：RecBole原生recall@k（单目标下=hit@k）

            dcg = 1.0 / torch.log2(ranks.clamp(min=1).float() + 1)
            dcg = torch.where(ranks <= k, dcg, torch.zeros_like(dcg))
            metrics[f'NDCG@{k}'] = dcg.mean().item()
            metrics[f'ndcg@{k}'] = dcg.mean().item()  # 新增：RecBole原生ndcg@k

            rr = 1.0 / ranks.clamp(min=1).float()
            rr = torch.where(ranks <= k, rr, torch.zeros_like(rr))
            metrics[f'MRR@{k}'] = rr.mean().item()
            metrics[f'mrr@{k}'] = rr.mean().item()  # 新增：RecBole原生mrr@k

            # 新增：RecBole原生precision@k
            precision = (ranks <= k).float() / k
            metrics[f'precision@{k}'] = precision.mean().item()

        metrics['MRR'] = (1.0 / ranks.clamp(min=1).float()).mean().item()
        metrics['Mean_Rank'] = ranks.mean().item()
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


class TransformerBlock(nn.Module):
    """适配3D因果掩码的TransformerBlock（兼容多模态）"""

    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True  # 必须设为True，适配3D掩码
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, padding_mask=None, attn_mask=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            padding_mask: [batch_size, seq_len] 2D Padding Mask（True=padding）
            attn_mask: [batch_size, seq_len, seq_len] 3D因果掩码
        """
        # Pre-LN（RecBole原生）
        x_norm = self.layer_norm1(x)

        # 自注意力（支持3D attn_mask）
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=padding_mask,  # 2D Padding Mask
            attn_mask=attn_mask,  # 3D因果掩码
            need_weights=False
        )
        x = x + self.dropout(attn_output)

        # 前馈网络
        x_norm = self.layer_norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)

        return x

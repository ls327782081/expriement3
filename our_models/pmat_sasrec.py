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

        # ===== 预测层 =====
        self.prediction_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )

        # ===== 用户-物品匹配层 (来自PMAT) =====
        self.user_item_matcher = UserItemMatcher(config)

        # ===== 损失权重 =====
        self.rec_loss_weight = getattr(config, 'rec_loss_weight', 1.0)
        self.semantic_loss_weight = getattr(config, 'semantic_loss_weight', 0.1)

        # 缓存因果掩码
        self._causal_mask_cache = {}

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.input_layer_norm.weight, 1.0)
        nn.init.constant_(self.input_layer_norm.bias, 0.0)

        for module in self.prediction_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

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

        # 7. 预测层投影
        user_repr = self.prediction_layer(user_repr)

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
        required_keys = [
            'history_text_feat', 'history_vision_feat', 'history_len',
            'target_text_feat', 'target_vision_feat', 'target_item',
            'neg_text_feat', 'neg_vision_feat', 'negative_items'
        ]
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise KeyError(f"Batch缺少必要的键: {missing_keys}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            batch: 包含以下键的字典
                - history_text_feat: (batch, seq_len, text_dim)
                - history_vision_feat: (batch, seq_len, visual_dim)
                - history_len: (batch,)
                - target_text_feat: (batch, text_dim)
                - target_vision_feat: (batch, visual_dim)
                - neg_text_feat: (batch, num_neg, text_dim)
                - neg_vision_feat: (batch, num_neg, visual_dim)

        Returns:
            outputs: 包含分数和中间结果的字典
        """
        self._validate_batch(batch)

        # 1. 编码历史序列 → 用户表示 + 短期/长期历史
        user_repr, seq_output, history_semantic_logits, short_history, long_history = self.encode_sequence(
            batch['history_text_feat'],
            batch['history_vision_feat'],
            batch['history_len']
        )  # user_repr: (batch, hidden_dim)

        # 2. 编码正样本（目标物品）- 使用个性化模态权重和动态ID更新
        pos_repr, pos_semantic_logits, pos_quantized_emb = self.encode_items(
            batch['target_text_feat'],
            batch['target_vision_feat'],
            user_interest=user_repr,
            short_history=short_history,
            long_history=long_history,
            return_semantic_logits=True
        )  # (batch, hidden_dim)

        # 3. 编码负样本 - 使用个性化模态权重和动态ID更新
        neg_repr, neg_semantic_logits, neg_quantized_emb = self.encode_items(
            batch['neg_text_feat'],
            batch['neg_vision_feat'],
            user_interest=user_repr,
            short_history=short_history,
            long_history=long_history,
            return_semantic_logits=True
        )  # (batch, num_neg, hidden_dim)

        # 4. 使用UserItemMatcher计算分数
        # 正样本分数
        pos_scores = self.user_item_matcher(
            user_repr, pos_repr, pos_quantized_emb
        )  # (batch,)

        # 负样本分数
        neg_scores = self.user_item_matcher(
            user_repr, neg_repr, neg_quantized_emb
        )  # (batch, num_neg)

        return {
            'user_repr': user_repr,
            'pos_scores': pos_scores,
            'neg_scores': neg_scores,
            'pos_repr': pos_repr,
            'neg_repr': neg_repr,
            'pos_quantized_emb': pos_quantized_emb,
            'neg_quantized_emb': neg_quantized_emb,
            'pos_semantic_logits': pos_semantic_logits,
            'neg_semantic_logits': neg_semantic_logits,
            'history_semantic_logits': history_semantic_logits,
            'target_item': batch['target_item'],
            'negative_items': batch['negative_items']
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算多任务损失

        主损失: BPR推荐损失
        辅助损失: 语义ID生成损失
        """
        losses = {}
        device = outputs['pos_scores'].device

        # ===== 1. BPR推荐损失 =====
        pos_scores = outputs['pos_scores']  # (batch,)
        neg_scores = outputs['neg_scores']  # (batch, num_neg)

        if neg_scores.numel() == 0 or neg_scores.size(1) == 0:
            bpr_loss = F.relu(1.0 - pos_scores).mean()
        else:
            pos_scores_expanded = pos_scores.unsqueeze(1)
            # 裁剪分数差值，防止数值溢出
            score_diff = torch.clamp(pos_scores_expanded - neg_scores, min=-50, max=50)
            bpr_loss = -F.logsigmoid(score_diff).mean()

        losses['bpr_loss'] = self.rec_loss_weight * bpr_loss

        # ===== 2. 语义ID损失（辅助任务）=====
        target_items = outputs['target_item']

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
        losses['total_loss'] = losses['bpr_loss'] + losses['semantic_loss']

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
        """单batch训练"""
        outputs = self.forward(batch)
        losses = self.compute_loss(outputs)

        pos_scores = outputs['pos_scores']
        neg_scores = outputs['neg_scores']
        pos_expanded = pos_scores.unsqueeze(1)
        auc_approx = (pos_expanded > neg_scores).float().mean().item()

        metrics = {
            'bpr_loss': losses['bpr_loss'].item(),
            'semantic_loss': losses['semantic_loss'].item(),
            'auc_approx': auc_approx
        }

        return losses['total_loss'], metrics

    def _validate_one_epoch(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        stage_id: int,
        stage_kwargs: Dict
    ) -> Dict:
        """单轮验证"""
        self.eval()

        total_loss = 0.0
        total_bpr_loss = 0.0
        total_semantic_loss = 0.0
        all_pos_scores = []
        all_neg_scores = []

        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = self.forward(batch)
                losses = self.compute_loss(outputs)

                total_loss += losses['total_loss'].item()
                total_bpr_loss += losses['bpr_loss'].item()
                total_semantic_loss += losses['semantic_loss'].item()

                all_pos_scores.append(outputs['pos_scores'].cpu())
                all_neg_scores.append(outputs['neg_scores'].cpu())

        num_batches = len(val_dataloader)
        avg_loss = total_loss / num_batches
        avg_bpr_loss = total_bpr_loss / num_batches
        avg_semantic_loss = total_semantic_loss / num_batches

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
        k_list: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """计算推荐指标，与MCRL和metrics.py保持一致

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


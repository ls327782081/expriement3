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

from typing import Dict, Tuple, Optional, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import AbstractTrainableModel
# 复用PMAT的核心模块（注意：这里要导入修复后的SemanticIDQuantizer）
from our_models.pmat import (
    MultiModalEncoder,
    PersonalizedFusion,
    # 确保导入的是修复后的SemanticIDQuantizer
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

    def __init__(self, config, device: torch.device):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.device = device

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

        # 修改1：使用修复后的SemanticIDQuantizer
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

        self.codebook_size = config.codebook_size  # 码本总数
        self.id_length = config.id_length  # 语义ID长度
        # 累计码本使用计数（用于epoch级统计）
        self.global_code_counts = [
            torch.zeros(self.codebook_size, device=self.device)
            for _ in range(self.id_length)
        ]
        # 死亡码标记（连续N步未使用）
        self.dead_code_mask = torch.zeros(self.codebook_size, dtype=torch.bool, device=self.device)

        # 监控频率（每多少个batch打印一次）
        self.monitor_freq = getattr(config, "monitor_freq", 10)
        self.batch_count = 0  # 批次计数器


    # 修改2：修复monitor_codebook方法（适配新的量化器）
    def monitor_codebook(self, epoch=None, stage=None, verbose=False):
        """
        优化后的码本监控方法：适配修复后的SemanticIDQuantizer
        Args:
            epoch/stage: 训练阶段标识（可选）
            verbose: 是否打印详细信息（False=只打印核心指标）
        Returns:
            码本监控核心指标
        """
        # 从量化器获取码本使用统计
        usage_stats = self.semantic_quantizer.get_codebook_usage()

        # 打印（极简版，只保留核心）
        if verbose:
            print(f"\n===== 码本向量监控 [{stage if stage else 'Stage'} | Epoch {epoch if epoch else 'N/A'}] =====")
            print(
                f"全局物理码本利用率: {usage_stats['global']['usage_ratio']:.4f} ({usage_stats['global']['usage_ratio'] * 100:.2f}%)")
            print(
                f"全局死亡码比例: {usage_stats['global']['dead_ratio']:.4f} ({usage_stats['global']['dead_ratio'] * 100:.2f}%)")
            print(f"总使用物理码本数: {usage_stats['global']['used']} / {usage_stats['global']['total']}")

            # 可选：打印每层利用率（精简版）
            print("\n各层码本利用率:")
            for layer in range(self.id_length):
                layer_key = f"layer_{layer + 1}"
                if layer_key in usage_stats:
                    stats = usage_stats[layer_key]
                    print(f"  {layer_key}: {stats['used']}/{stats['total']} ({stats['usage_ratio'] * 100:.2f}%)")

        # 更新全局码本计数（用于epoch级统计）
        for layer in range(self.id_length):
            layer_key = f"layer_{layer + 1}"
            if layer_key in usage_stats:
                used_mask = self.semantic_quantizer.codebook_usage[layer].to(self.device)
                self.global_code_counts[layer] += used_mask.long()

        return usage_stats

    def forward(
            self,
            text_feat: torch.Tensor,
            vision_feat: torch.Tensor,
            user_interest: Optional[torch.Tensor] = None,
            short_history: Optional[torch.Tensor] = None,
            long_history: Optional[torch.Tensor] = None,
            return_semantic_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        重构后：
        - 无任何no_grad/detach（除了监控用的residuals）
        - 损失从forward返回，不偷self变量
        - 保留所有创新点
        """
        # 1. 多模态融合逻辑（完全不变）
        original_shape = text_feat.shape[:-1]
        text_flat = text_feat.reshape(-1, text_feat.shape[-1])
        vision_flat = vision_feat.reshape(-1, vision_feat.shape[-1])

        item_features = {'text': text_flat.float(), 'visual': vision_flat.float()}
        encoded_features = self.multimodal_encoder(item_features)

        if user_interest is not None:
            modal_weights = self.user_modal_attention(user_interest)
            if modal_weights.size(0) != text_flat.size(0):
                num_items_per_user = text_flat.size(0) // modal_weights.size(0)
                modal_weights = modal_weights.unsqueeze(1).expand(-1, num_items_per_user, -1)
                modal_weights = modal_weights.reshape(-1, modal_weights.size(-1))
        else:
            modal_weights = F.softmax(self.modal_weight, dim=0)
            modal_weights = modal_weights.unsqueeze(0).expand(text_flat.size(0), -1)

        fused_feat = self.personalized_fusion(encoded_features, modal_weights)
        fused_feat = F.normalize(fused_feat, dim=-1)

        # 2. 重构核心：调用标准RVQ（无任何梯度截断）
        quantized_emb, residual_loss, residuals, codebook_usage = self.semantic_quantizer(fused_feat)

        # 新增：从量化器中提取语义ID（训练/验证统一用这个）
        semantic_ids = torch.stack(self.semantic_quantizer.layer_indices, dim=1)

        # 保留你的创新：码本监控（仅训练阶段）
        if self.training:
            self.batch_count += 1
            if self.batch_count % self.monitor_freq == 0:
                self.monitor_codebook(verbose=True)

        # 3. 后续逻辑（完全不变）
        recon_loss = F.mse_loss(quantized_emb, fused_feat)  # 无detach，计算图完整
        quantized_emb = 0.7 * quantized_emb + 0.3 * fused_feat
        quantized_emb = F.layer_norm(quantized_emb, normalized_shape=[self.hidden_dim])

        # 保留你的创新：动态ID更新
        if short_history is not None and long_history is not None:
            drift_score = self.dynamic_updater.detect_drift(short_history, long_history)
            if quantized_emb.size(0) != drift_score.size(0):
                num_users = drift_score.size(0)
                num_items_total = quantized_emb.size(0)
                if num_items_total % num_users == 0:
                    num_items_per_user = num_items_total // num_users
                    quantized_emb_reshaped = quantized_emb.view(num_users, num_items_per_user, -1)
                    fused_feat_reshaped = fused_feat.view(num_users, num_items_per_user, -1)
                    quantized_emb = self.dynamic_updater.update(
                        quantized_emb_reshaped, fused_feat_reshaped, drift_score
                    ).view(-1, self.hidden_dim)

        # 4. 组合特征（保留你的逻辑）
        combined = torch.cat([fused_feat, quantized_emb], dim=-1)
        item_emb = self.fusion_layer(combined)

        # 恢复形状
        item_emb = item_emb.reshape(*original_shape, self.hidden_dim)
        quantized_emb_out = quantized_emb.reshape(*original_shape, self.hidden_dim)

        # 返回：5个值（与调用端匹配）
        semantic_logits = None
        if return_semantic_logits:
            # 从量化器中获取每层的相似度矩阵（核心：真实反映RVQ选择逻辑）
            layer_sims = self.semantic_quantizer.layer_similarities  # 列表：[id_length个层的相似度，每个shape=[B, codebook_size]]
            # 拼接成[B, id_length, codebook_size]
            semantic_logits = torch.stack(layer_sims, dim=1)  # [B, id_length, codebook_size]（id_length层+codebook_size码本）

        balance_loss = self.compute_usage_balance_loss(
            self.semantic_quantizer.layer_indices,
            self.config.codebook_size
        )


        return item_emb, semantic_logits, quantized_emb_out, recon_loss, residual_loss, balance_loss, semantic_ids

    def compute_usage_balance_loss(self, layer_indices, codebook_size):
        """
        ICML 2022 / NeurIPS 2023 标准：码本使用均衡损失
        让每个码本被尽量均匀使用，不会扎堆
        """
        loss = 0.0
        num_layers = len(layer_indices)

        for idx, indices in enumerate(layer_indices):

            # indices: [B]
            freq = torch.bincount(indices, minlength=codebook_size).float()
            freq = freq / (freq.sum() + 1e-8)
            entropy = -torch.sum(freq * torch.log(freq + 1e-8))
            max_entropy = torch.log(torch.tensor(codebook_size, device=indices.device))
            layer_loss = (1.0 - entropy / max_entropy)

            loss += layer_loss
        avg_loss = loss / num_layers
        return avg_loss


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
        self.item_encoder = PMATItemEncoder(config, self.device)

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
        """
        重构后：
        - 无任何try/catch兜底
        - 不从self偷residuals，直接从forward返回
        - 保留所有损失加权逻辑
        """
        text_feat = batch['text_feat'].to(self.device)
        visual_feat = batch['vision_feat'].to(self.device)
        batch_size = text_feat.size(0)

        # 调用forward：完整接收5个返回值
        item_emb, semantic_logits, quantized_emb, recon_loss, residual_loss, balance_loss, semantic_ids = self.item_encoder(
            text_feat, visual_feat, return_semantic_logits=True
        )
        pos_repr = self.prediction_layer(item_emb)

        # 保留你的所有对比损失逻辑（完全不变）
        temperature = getattr(self.config, 'pretrain_temperature', 0.1)
        pos_repr_norm = F.normalize(pos_repr, dim=-1)
        quantized_emb_norm = F.normalize(quantized_emb, dim=-1)

        sim_matrix_intra = torch.matmul(pos_repr_norm, quantized_emb_norm.T) / temperature
        labels_intra = torch.arange(batch_size, device=self.device)
        intra_loss = (F.cross_entropy(sim_matrix_intra, labels_intra) +
                      F.cross_entropy(sim_matrix_intra.T, labels_intra)) / 2

        text_encoded = self.item_encoder.multimodal_encoder.text_encoder(text_feat.float())
        visual_encoded = self.item_encoder.multimodal_encoder.visual_encoder(visual_feat.float())
        text_norm = F.normalize(text_encoded, dim=-1)
        visual_norm = F.normalize(visual_encoded, dim=-1)

        sim_matrix_inter = torch.matmul(text_norm, visual_norm.T) / temperature
        labels_inter = torch.arange(batch_size, device=self.device)
        inter_loss = (F.cross_entropy(sim_matrix_inter, labels_inter) +
                      F.cross_entropy(sim_matrix_inter.T, labels_inter)) / 2

        # 保留你的损失加权逻辑（完全不变）
        intra_weight = getattr(self.config, 'pretrain_intra_weight', 0.1)
        inter_weight = getattr(self.config, 'pretrain_inter_weight', 0.01)
        recon_loss_weight = getattr(self.config, 'recon_loss_weight', 1.0)
        residual_loss_weight = getattr(self.config, 'residual_loss_weight', 1.0)
        balance_loss_weight = getattr(self.config, 'balance_loss_weight', 0.1)

        if self.current_stage_epoch < 5:
            balance_loss_weight = 0
        elif self.current_stage_epoch < 10:
            balance_loss_weight = 0.05

        total_loss = (intra_weight * intra_loss +
                      inter_weight * inter_loss +
                      recon_loss_weight * recon_loss +
                      residual_loss_weight * residual_loss +
                      balance_loss_weight * balance_loss)

        return {
            'total_loss': total_loss,
            'intra_loss': intra_loss,
            'inter_loss': inter_loss,
            'recon_loss': recon_loss,
            'residual_loss': residual_loss,
            'balance_loss': balance_loss,
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
        padding_mask = (text_feat.sum(dim=-1) == 0)  # (batch, seq_len)，True=padding（末尾）
        valid_len_mask = torch.arange(seq_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        padding_mask = padding_mask | (~valid_len_mask)  # 双重保障

        # ========== 2. PMAT多模态物品编码 ==========
        # 修复：完整接收5个返回值，只取需要的前3个+忽略损失
        item_emb, semantic_logits, quantized_emb_out, recon_loss, residual_loss, semantic_ids = self.item_encoder(
            text_feat, vision_feat, return_semantic_logits=True
        )  # (batch, seq_len, hidden_dim)

        # ========== 3. 左对齐位置编码（核心修改） ==========
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)  # (batch, seq_len)
        position_ids = position_ids * (~padding_mask).long()  # padding位置置0
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
        """编码候选物品"""
        # 修复：完整接收5个返回值
        item_emb, semantic_logits, quantized_emb, _, _, _, _ = self.item_encoder(
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
            'target_text_feat', 'target_vision_feat', 'target_item'
        ]
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise KeyError(f"Batch缺少必要的键: {missing_keys}")

    def set_all_item_features(self, all_item_features: Dict[str, torch.Tensor]):
        """设置所有物品特征，用于预计算物品表征"""
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
                # 修复：完整接收5个返回值
                item_emb, _, quantized_emb, _, _, _, _ = self.item_encoder(
                    item_text, item_visual,
                    user_interest=None,
                    return_semantic_logits=False
                )
                item_repr = self.prediction_layer(item_emb)
                all_item_repr_list.append(item_repr)
                all_quantized_emb_list.append(quantized_emb)

        self._all_item_repr = torch.cat(all_item_repr_list, dim=0)  # (num_items, hidden_dim)
        self._all_quantized_emb = torch.cat(all_quantized_emb_list, dim=0)  # (num_items, hidden_dim)
        self._all_item_repr = F.normalize(self._all_item_repr, dim=-1)
        print(f"物品表征预计算完成，形状: {self._all_item_repr.shape}（已L2归一化）")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        self._validate_batch(batch)

        # 1. 编码历史序列 → 用户表示
        user_repr, seq_output, history_semantic_logits, short_history, long_history = self.encode_sequence(
            batch['history_text_feat'],
            batch['history_vision_feat'],
            batch['history_len']
        )  # user_repr: (batch, hidden_dim)

        # 2. 计算对所有物品的 logits（Cross Entropy 损失）
        if self._all_item_repr is not None:
            temperature = getattr(self.config, 'logit_temperature', 0.8)
            user_repr_norm = F.normalize(user_repr, dim=-1)
            logits = torch.matmul(user_repr_norm, self._all_item_repr.T) / temperature  # (batch, num_items)
        else:
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
        """计算多任务损失"""
        losses = {}

        # ===== 1. Cross Entropy 推荐损失 =====
        logits = outputs['logits']  # (batch, num_items)
        target_items = outputs['target_item']  # (batch,)

        if logits is not None:
            ce_loss = F.cross_entropy(logits, target_items)
            losses['ce_loss'] = self.rec_loss_weight * ce_loss
        else:
            losses['ce_loss'] = torch.tensor(0.0, device=target_items.device)

        # ===== 3. 总损失 =====
        losses['total_loss'] = losses['ce_loss']

        return losses

    # ==================== AbstractTrainableModel 抽象方法实现 ====================

    def _get_optimizer(self, stage_id: int, stage_kwargs: Dict) -> torch.optim.Optimizer:
        """获取优化器"""
        lr = stage_kwargs.get('lr', 0.001)
        weight_decay = stage_kwargs.get('weight_decay', 0.01)
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_scheduler(self, optimizer: torch.optim.Optimizer, stage_id: int,
                       stage_kwargs: Dict) -> torch.optim.lr_scheduler.LRScheduler:
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
        """单batch训练"""
        if stage_id == 0:
            # 阶段1：预训练物品编码器
            losses = self.compute_pretrain_loss(batch)
            metrics = {
                'intra_loss': losses['intra_loss'].item(),
                'inter_loss': losses['inter_loss'].item(),
                'recon_loss': losses['recon_loss'].item(),
                'residual_loss': losses['residual_loss'].item(),
                'balance_loss': losses['balance_loss'].item(),
            }
            return losses['total_loss'], metrics
        else:
            # 阶段2：训练序列模型
            outputs = self.forward(batch)
            losses = self.compute_loss(outputs)

            metrics = {
                'ce_loss': losses['ce_loss'].item(),
            }
            return losses['total_loss'], metrics

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
                if 'history_items' in batch and batch['history_items'].shape[1] > max_history_len:
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
                if 'history_items' in batch and batch['history_items'].shape[1] < max_history_len:
                    pad_len = max_history_len - batch['history_items'].shape[1]
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
                if 'history_items' in batch:
                    all_history_ids.append(batch['history_items'])
                all_seq_lens.append(batch['history_len'])

        # ========== 第三步：合并所有张量（维度完全匹配） ==========
        all_user_repr = torch.cat(all_user_repr, dim=0)  # (total_users, hidden_dim)
        all_user_short_history = torch.cat(all_user_short_history, dim=0)  # (total_users, max_short_len, hidden_dim)
        all_user_long_history = torch.cat(all_user_long_history, dim=0)  # (total_users, max_long_len, hidden_dim)
        all_target_items = torch.cat(all_target_items, dim=0)  # (total_users,)
        if all_history_ids:
            all_history_ids = torch.cat(all_history_ids, dim=0)  # (total_users, max_history_len)
        all_seq_lens = torch.cat(all_seq_lens, dim=0)  # (total_users,)
        total_users = all_user_repr.shape[0]

        # ========== 第四步：预计算个性化物品表征 ==========
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

                    # 逐用户编码当前物品chunk
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

                        # 个性化编码（保留动态ID更新）
                        item_emb, _, _, _, _, _, _ = self.item_encoder(
                            u_chunk_text,
                            u_chunk_vision,
                            user_interest=u_user_interest,
                            short_history=u_short_expand,
                            long_history=u_long_expand
                        )
                        item_repr = self.prediction_layer(item_emb)
                        item_repr_norm = F.normalize(item_repr, dim=-1)
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

            # ========== 修复4：向量化历史屏蔽 ==========
            history_mask = torch.zeros((total_users, num_items), dtype=torch.bool, device=self.device)
            if all_history_ids is not None and len(all_history_ids) > 0:
                seq_range = torch.arange(max_history_len, device=self.device).unsqueeze(0)
                valid_seq_mask = seq_range < all_seq_lens.unsqueeze(1)  # (total_users, max_history_len)
                valid_id_mask = (all_history_ids != 0) & (all_history_ids < num_items) & valid_seq_mask

                # 提取有效索引
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

    def on_epoch_start(self, epoch: int, stage_id: int, stage_kwargs: Dict):
        # 重置全局码本计数
        for level in range(self.item_encoder.id_length):
            self.item_encoder.global_code_counts[level].zero_()

    def on_epoch_end(self, epoch: int, stage_id: int, stage_kwargs: Dict,
                     train_metrics: Dict, val_metrics: Dict):
        super().on_epoch_end(epoch, stage_id, stage_kwargs, train_metrics, val_metrics)
        # ===== Epoch级码本监控（核心：全局统计+死亡码判断）=====
        encoder = self.item_encoder
        print(f"\n===== Epoch {epoch} 码本全局监控 ======")

        # ========== 修改10：修复全局码本计数逻辑 ==========
        # 从量化器获取最新的码本使用统计
        usage_stats = encoder.semantic_quantizer.get_codebook_usage()
        total_used = usage_stats['global']['used']
        total_codebooks = usage_stats['global']['total']

        # 最终全局统计（正确版）
        global_utilization = total_used / total_codebooks if total_codebooks > 0 else 0
        dead_ratio = 1 - global_utilization
        print(f"全局码本利用率: {global_utilization:.2%}")
        print(f"全局死亡码比例: {dead_ratio:.2%}")
        print(f"总共用了{total_used}个码本 / 总码本数{total_codebooks}")

    def on_batch_start(self, batch: Any, batch_idx: int, stage_id: int, stage_kwargs: Dict):
        """batch开始钩子"""
        # 移除错误的重置逻辑（量化器中无residuals/need_retain_graph变量）
        pass

    def _compute_metrics(self, ranks: torch.Tensor, k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
        metrics = {}
        for k in k_list:
            hits = (ranks <= k).float()
            metrics[f'HR@{k}'] = hits.mean().item()
            metrics[f'hit@{k}'] = hits.mean().item()
            metrics[f'recall@{k}'] = hits.mean().item()

            dcg = 1.0 / torch.log2(ranks.clamp(min=1).float() + 1)
            dcg = torch.where(ranks <= k, dcg, torch.zeros_like(dcg))
            metrics[f'NDCG@{k}'] = dcg.mean().item()
            metrics[f'ndcg@{k}'] = dcg.mean().item()

            rr = 1.0 / ranks.clamp(min=1).float()
            rr = torch.where(ranks <= k, rr, torch.zeros_like(rr))
            metrics[f'MRR@{k}'] = rr.mean().item()
            metrics[f'mrr@{k}'] = rr.mean().item()

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
        """执行推荐预测"""
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
                scores = outputs['logits'] if 'logits' in outputs else torch.zeros(1)

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
            # 修复：完整接收5个返回值
            item_repr, _, quantized_emb, _, _ = self.encode_items(text_feat, vision_feat)
            # 维度检查，防止拼接失败
            if item_repr.dim() != quantized_emb.dim():
                quantized_emb = quantized_emb.reshape(item_repr.shape)
            # 返回融合特征和语义ID嵌入的拼接
            item_embedding = torch.cat([item_repr, quantized_emb], dim=-1)
            return item_embedding


class TransformerBlock(nn.Module):
    """适配2D因果掩码的TransformerBlock（兼容多模态）"""

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
            attn_mask: [seq_len, seq_len] 2D因果掩码
        """
        # Pre-LN（RecBole原生）
        x_norm = self.layer_norm1(x)

        # 自注意力（支持2D attn_mask）
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=padding_mask,  # 2D Padding Mask
            attn_mask=attn_mask,  # 2D因果掩码
            need_weights=False
        )
        x = x + self.dropout(attn_output)

        # 前馈网络
        x_norm = self.layer_norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)

        return x
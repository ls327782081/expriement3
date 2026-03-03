import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from our_models.ah_rq import AdaptiveHierarchicalQuantizer  # 升级后的AH-RQ（带多模态对齐）


# ===================== 保留PMAT核心模块（移除重复的多模态编码）=====================
class UserModalAttention(nn.Module):
    """用户-模态偏好感知器（保留核心逻辑，仅调整输入适配AH-RQ）"""

    def __init__(self, user_dim: int, num_modalities: int, hidden_dim: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.modal_preference_net = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, user_interest: torch.Tensor) -> torch.Tensor:
        logits = self.modal_preference_net(user_interest)
        modal_weights = F.softmax(logits / (self.temperature + 1e-8), dim=-1)
        return modal_weights


class DynamicIDUpdater(nn.Module):
    """动态ID更新（完全保留原有逻辑）"""

    def __init__(self):
        super().__init__()
        self.drift_threshold = config.pmat_drift_threshold
        # 短期/长期兴趣编码器
        self.short_term_encoder = nn.LSTM(
            input_size=config.pmat_hidden_dim,
            hidden_size=config.pmat_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.long_term_encoder = nn.LSTM(
            input_size=config.pmat_hidden_dim,
            hidden_size=config.pmat_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        # 漂移检测器
        self.drift_detector = nn.Sequential(
            nn.Linear(config.pmat_hidden_dim * 2, config.pmat_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.pmat_hidden_dim, 1),
            nn.Sigmoid()
        )
        # 更新门控
        self.update_gate = nn.Sequential(
            nn.Linear(config.pmat_hidden_dim * 2, config.pmat_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.pmat_hidden_dim, config.pmat_hidden_dim),
            nn.Sigmoid()
        )

    def detect_drift(self, short_history, long_history):
        _, (short_h, _) = self.short_term_encoder(short_history)
        _, (long_h, _) = self.long_term_encoder(long_history)
        short_repr = short_h[-1]
        long_repr = long_h[-1]
        combined = torch.cat([short_repr, long_repr], dim=-1)
        drift_score = self.drift_detector(combined).squeeze(-1)
        return drift_score

    def update(self, current_id_emb, new_features, drift_score):
        if current_id_emb.dim() == 3:
            batch_size, num_items, dim = current_id_emb.shape
            current_flat = current_id_emb.view(-1, dim)
            new_flat = new_features.view(-1, dim)
            combined = torch.cat([current_flat, new_flat], dim=-1)
            gate = self.update_gate(combined)
            drift_expanded = drift_score.unsqueeze(1).expand(-1, num_items).reshape(-1)
            drift_mask = (drift_expanded > self.drift_threshold).float().unsqueeze(-1)
            effective_gate = gate * drift_mask
            updated_flat = (1 - effective_gate) * current_flat + effective_gate * new_flat
            updated_id_emb = updated_flat.view(batch_size, num_items, dim)
        else:
            combined = torch.cat([current_id_emb, new_features], dim=-1)
            gate = self.update_gate(combined)
            drift_mask = (drift_score > self.drift_threshold).float().unsqueeze(-1)
            effective_gate = gate * drift_mask
            updated_id_emb = (1 - effective_gate) * current_id_emb + effective_gate * new_features
        return updated_id_emb


class UserInterestEncoder(nn.Module):
    """用户兴趣编码器（核心修改：直接用AH-RQ的多模态对齐结果）"""

    def __init__(self):
        super().__init__()
        self.max_seq_len = config.sasrec_max_len

        # 核心修改：初始化带多模态对齐的AH-RQ（用户兴趣编码用）
        self.ahrq_for_user = AdaptiveHierarchicalQuantizer(
            hidden_dim=config.pmat_hidden_dim,
            semantic_hierarchy=config.semantic_hierarchy,
            use_multimodal=True,  # 启用多模态对齐
            text_dim=config.pmat_text_dim,
            visual_dim=config.pmat_visual_dim,
            beta=config.ahrq_beta,
            use_ema=config.ahrq_use_ema,
            ema_decay=0.99,
            reset_unused_codes=config.ahrq_reset_unused_codes,
            reset_threshold=config.ahrq_reset_threshold
        )

        # Transformer编码器（保留）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.pmat_hidden_dim,
            nhead=config.sasrec_num_heads,
            dim_feedforward=config.pmat_hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 位置编码（保留）
        self.position_embedding = nn.Embedding(self.max_seq_len, config.pmat_hidden_dim)

        # Attention Pooling（保留）
        self.interest_query = nn.Parameter(torch.randn(1, 1, config.pmat_hidden_dim))
        self.interest_attention = nn.MultiheadAttention(
            embed_dim=config.pmat_hidden_dim,
            num_heads=config.sasrec_num_heads,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, history_text, history_visual, history_len):
        batch_size = history_text.shape[0]

        # 核心修改：用AH-RQ完成多模态对齐（替代原有的text/visual投影+融合）
        fused_history, _, _ = self.ahrq_for_user(history_text, history_visual)

        # 位置编码（保留）
        positions = torch.arange(self.max_seq_len, device=history_text.device).unsqueeze(0).expand(batch_size, -1)
        fused_history = fused_history + self.position_embedding(positions)

        # Attention Mask（保留）
        mask = torch.arange(self.max_seq_len).unsqueeze(0) < (self.max_seq_len - history_len.unsqueeze(1))

        # Transformer编码（保留）
        encoded_history = self.sequence_encoder(fused_history, src_key_padding_mask=mask)

        # Attention Pooling（保留）
        query = self.interest_query.expand(batch_size, -1, -1)
        user_interest, _ = self.interest_attention(query, encoded_history, encoded_history, key_padding_mask=mask)
        user_interest = user_interest.squeeze(1)

        return user_interest


class UserItemMatcher(nn.Module):
    """用户-物品匹配层（保留原有逻辑）"""

    def __init__(self):
        super().__init__()
        self.user_proj = nn.Sequential(
            nn.Linear(config.pmat_hidden_dim, config.pmat_hidden_dim),
            nn.LayerNorm(config.pmat_hidden_dim),
            nn.ReLU()
        )
        self.item_proj = nn.Sequential(
            nn.Linear(config.pmat_hidden_dim * 2, config.pmat_hidden_dim),
            nn.LayerNorm(config.pmat_hidden_dim),
            nn.ReLU()
        )
        self.match_mlp = nn.Sequential(
            nn.Linear(config.pmat_hidden_dim * 3, config.pmat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.pmat_hidden_dim, config.pmat_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.pmat_hidden_dim // 2, 1)
        )

    def forward(self, user_repr, item_fused_feat, item_semantic_emb):
        user_proj = self.user_proj(user_repr)
        if item_fused_feat.dim() == 2:
            item_combined = torch.cat([item_fused_feat, item_semantic_emb], dim=-1)
            item_proj = self.item_proj(item_combined)
            interaction = user_proj * item_proj
            combined = torch.cat([user_proj, item_proj, interaction], dim=-1)
            scores = self.match_mlp(combined).squeeze(-1)
        else:
            batch_size, num_items, _ = item_fused_feat.shape
            item_combined = torch.cat([item_fused_feat, item_semantic_emb], dim=-1)
            item_proj = self.item_proj(item_combined)
            user_proj_expanded = user_proj.unsqueeze(1).expand(-1, num_items, -1)
            interaction = user_proj_expanded * item_proj
            combined = torch.cat([user_proj_expanded, item_proj, interaction], dim=-1)
            scores = self.match_mlp(combined).squeeze(-1)
        return scores


# ===================== PMAT-AH-RQ核心模型（整合AH-RQ多模态对齐）=====================
class PMATAHRQ(nn.Module):
    """
    创新点2：PMAT替换语义编码为AH-RQ（整合多模态对齐）
    核心改动：
    1. 移除MultiModalEncoder/PersonalizedFusion（交给AH-RQ处理）
    2. AH-RQ启用多模态对齐，直接输入原始text/visual特征
    3. 保留用户-模态偏好、动态ID更新、用户兴趣编码核心逻辑
    """

    def __init__(self):
        super().__init__()
        # 用户侧模块（保留）
        self.user_interest_encoder = UserInterestEncoder()
        self.user_modal_attention = UserModalAttention(
            user_dim=config.pmat_hidden_dim,
            num_modalities=config.pmat_num_modalities,
            hidden_dim=config.pmat_hidden_dim
        )

        # 核心改动：初始化带多模态对齐的AH-RQ（物品编码用）
        self.semantic_quantizer = AdaptiveHierarchicalQuantizer(
            hidden_dim=config.pmat_hidden_dim,
            semantic_hierarchy=config.semantic_hierarchy,
            use_multimodal=True,  # 启用多模态对齐
            text_dim=config.pmat_text_dim,
            visual_dim=config.pmat_visual_dim,
            beta=config.ahrq_beta,
            use_ema=config.ahrq_use_ema,
            ema_decay=0.99,
            reset_unused_codes=config.ahrq_reset_unused_codes,
            reset_threshold=config.ahrq_reset_threshold
        )

        # 动态ID更新（保留）
        self.dynamic_updater = DynamicIDUpdater()

        # 匹配层（保留）
        self.user_item_matcher = UserItemMatcher()

    def encode_item(self, text_feat, visual_feat, user_interest, short_history=None, long_history=None):
        """编码物品特征（核心修改：用AH-RQ直接处理原始多模态特征）"""
        device = text_feat.device

        if text_feat.dim() == 2:
            batch_size = user_interest.size(0)

            # 核心修改：AH-RQ直接输入原始text/visual，完成对齐+量化
            quantized_emb, indices_list, quantized_layers = self.semantic_quantizer(text_feat, visual_feat)

            # 动态ID更新（保留）
            if short_history is not None and long_history is not None:
                drift_score = self.dynamic_updater.detect_drift(short_history, long_history)
                quantized_emb = self.dynamic_updater.update(quantized_emb, quantized_emb, drift_score)

            # 兼容原有返回格式（fused_feat复用quantized_emb）
            fused_feat = quantized_emb
            return fused_feat, indices_list, quantized_layers, quantized_emb

        else:
            batch_size, num_items, _ = text_feat.shape
            text_flat = text_feat.view(-1, text_feat.size(-1))
            visual_flat = visual_feat.view(-1, visual_feat.size(-1))

            # 核心修改：AH-RQ处理展平的多模态特征
            quantized_emb, indices_list, quantized_layers = self.semantic_quantizer(text_flat, visual_flat)

            # 恢复维度
            fused_feat = quantized_emb.view(batch_size, num_items, -1)
            quantized_emb = quantized_emb.view(batch_size, num_items, -1)

            # 动态ID更新（保留）
            if short_history is not None and long_history is not None:
                drift_score = self.dynamic_updater.detect_drift(short_history, long_history)
                quantized_emb = self.dynamic_updater.update(quantized_emb, fused_feat, drift_score)

            return fused_feat, indices_list, quantized_layers, quantized_emb

    def forward(self, batch):
        """前向传播（接口完全兼容原有逻辑）"""
        # 1. 编码用户兴趣（AH-RQ已完成多模态对齐）
        user_interest = self.user_interest_encoder(
            batch["history_text"], batch["history_visual"], batch["history_len"]
        )

        # 2. 准备动态更新的历史数据（保留）
        batch_size = user_interest.size(0)
        short_history = batch["history_text"][:, -10:, :]  # 短期历史（最近10个）
        long_history = batch["history_text"]  # 长期历史

        # 3. 编码正样本（AH-RQ处理多模态）
        pos_fused_feat, pos_indices, pos_quant_layers, pos_quant_emb = self.encode_item(
            batch["target_text"], batch["target_visual"], user_interest, short_history, long_history
        )

        # 4. 编码负样本（AH-RQ处理多模态）
        neg_fused_feat, neg_indices, neg_quant_layers, neg_quant_emb = self.encode_item(
            batch["neg_text"], batch["neg_visual"], user_interest, short_history, long_history
        )

        # 5. 计算匹配分数（保留）
        pos_scores = self.user_item_matcher(user_interest, pos_fused_feat, pos_quant_emb)
        neg_scores = self.user_item_matcher(user_interest, neg_fused_feat, neg_quant_emb)

        # 6. 返回结果（接口完全兼容）
        return {
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "user_interest": user_interest,
            "pos_fused_feat": pos_fused_feat,
            "pos_quant_emb": pos_quant_emb,
            "pos_indices": pos_indices,
            "pos_quant_layers": pos_quant_layers
        }


# ===================== 兼容原有PMATAHRQEncoder命名（保证pmat_sasrec.py可直接调用）=====================
class PMATAHRQEncoder(PMATAHRQ):
    """兼容原有类名，避免修改pmat_sasrec.py"""

    def __init__(self):
        super().__init__()

    def forward(self, batch):
        # 补充返回history_emb字段（适配pmat_sasrec.py的输入要求）
        base_output = super().forward(batch)

        # 计算history_emb（用户兴趣编码后的历史融合特征）
        history_emb, _, _ = self.user_interest_encoder.ahrq_for_user(
            batch["history_text"], batch["history_visual"]
        )

        base_output.update({
            "history_emb": history_emb,
            "neg_quant_emb": self.encode_item(
                batch["neg_text"], batch["neg_visual"], base_output["user_interest"]
            )[3],
            "quantized_layers": base_output["pos_quant_layers"]
        })

        return base_output
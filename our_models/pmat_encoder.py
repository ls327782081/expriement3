import torch
import torch.nn as nn
import torch.nn.functional as F
from config import new_config


class UserModalAttention(nn.Module):
    """用户模态偏好感知器"""
    def __init__(self, user_dim: int, num_modalities: int, hidden_dim: int):
        super().__init__()
        self.modal_preference_net = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, num_modalities)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, user_interest: torch.Tensor) -> torch.Tensor:
        logits = self.modal_preference_net(user_interest)
        modal_weights = F.softmax(logits / (self.temperature + 1e-8), dim=-1)
        return modal_weights


class DynamicIDUpdater(nn.Module):
    """动态ID更新模块（适配用户兴趣漂移）"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.drift_threshold = new_config.pmat_drift_threshold
        self.hidden_dim = hidden_dim

        # 短期/长期兴趣编码器
        self.short_term_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
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

    def detect_drift(self, short_history_emb, long_history_emb):
        _, (short_h, _) = self.short_term_encoder(short_history_emb)
        _, (long_h, _) = self.long_term_encoder(long_history_emb)
        short_repr = short_h[-1]
        long_repr = long_h[-1]
        combined = torch.cat([short_repr, long_repr], dim=-1)
        drift_score = self.drift_detector(combined).squeeze(-1)
        return drift_score

    def update(self, current_id_emb, new_features, drift_score):
        if current_id_emb.dim() == 3:
            batch_size, num_items, dim = current_id_emb.shape
            current_flat = current_id_emb.reshape(-1, dim)
            new_flat = new_features.reshape(-1, dim)
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
    """用户兴趣编码器（基于预计算语义ID的嵌入序列）"""
    def __init__(self, hidden_dim: int, max_len: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.35,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 位置编码
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        # Attention Pooling
        self.interest_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, history_emb, history_len):
        batch_size = history_emb.shape[0]

        # 位置编码
        positions = torch.arange(self.max_len, device=history_emb.device).unsqueeze(0).expand(batch_size, -1)
        history_emb = history_emb + self.position_embedding(positions)

        # Padding掩码
        mask = torch.arange(self.max_len, device=history_emb.device).unsqueeze(0) >= history_len.unsqueeze(1)

        # Transformer编码
        encoded_history = self.transformer(history_emb, src_key_padding_mask=mask)

        # Attention Pooling生成用户兴趣
        query = self.interest_query.expand(batch_size, -1, -1)
        attn_output, _ = self.attention(query, encoded_history, encoded_history, key_padding_mask=mask)
        user_interest = attn_output.squeeze(1)

        return user_interest


class PMATAHRQEncoder(nn.Module):
    """
    PMAT核心编码器：
    1. 输入：预计算的语义ID（来自Stage1 AHRQ）
    2. 输出：动态更新的物品嵌入（适配用户兴趣）
    """
    def __init__(self, semantic_hierarchy: dict, num_layers: int, layer_dim: int):
        super().__init__()
        self.semantic_hierarchy = semantic_hierarchy
        self.num_layers = num_layers
        self.layer_dim = layer_dim
        self.hidden_dim = num_layers * layer_dim
        self.fusion_alpha = getattr(new_config, 'fusion_alpha', 0.7)  # 动态/模态融合权重

        # 语义ID → 静态嵌入
        self.semantic_emb = nn.ModuleDict()
        for sem_type, config in semantic_hierarchy.items():
            cb_size = config["codebook_size"]
            for layer in config["layers"]:
                self.semantic_emb[f"{sem_type}_{layer}"] = nn.Embedding(cb_size, layer_dim)

        # 核心模块
        self.max_len = new_config.sasrec_max_len
        self.user_interest_encoder = UserInterestEncoder(
            hidden_dim=self.hidden_dim,
            max_len=new_config.sasrec_max_len,
            num_heads=new_config.sasrec_num_heads,
            num_layers=new_config.sasrec_num_layers
        )
        self.modal_attention = UserModalAttention(
            user_dim=self.hidden_dim,
            num_modalities=2,  # 文本+视觉
            hidden_dim=self.hidden_dim
        )
        self.dynamic_updater = DynamicIDUpdater(hidden_dim=self.hidden_dim)

        self.text_proj = nn.Linear(new_config.text_dim, int(self.hidden_dim * 0.4))
        self.vision_proj = nn.Linear(new_config.visual_dim, self.hidden_dim - int(self.hidden_dim * 0.4))
        self.modal_proj = nn.Linear(int(self.hidden_dim * 0.4) + (self.hidden_dim - int(self.hidden_dim * 0.4)),
                                    self.hidden_dim)
        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def semantic_id_to_emb(self, indices_list):
        """语义ID → 静态嵌入"""
        semantic_blocks = []
        layer_idx = 0
        for sem_type, config in self.semantic_hierarchy.items():
            for _ in config["layers"]:
                cb_key = list(self.semantic_emb.keys())[layer_idx]
                indices = indices_list[layer_idx]
                block_emb = self.semantic_emb[cb_key](indices)
                semantic_blocks.append(block_emb)
                layer_idx += 1
        static_emb = torch.cat(semantic_blocks, dim=-1)
        static_emb = F.normalize(static_emb, p=2, dim=-1)
        return static_emb

    def encode_history(self, batch):
        """编码历史序列 → 动态嵌入"""
        # 1. 语义ID → 静态嵌入
        hist_indices = batch["history_indices"]  # (B, actual_len, num_layers)
        indices_list = [hist_indices[:, :, i] for i in range(self.num_layers)]
        hist_static_emb = self.semantic_id_to_emb(indices_list)  # (B, actual_len, hidden_dim)
        batch_size, actual_len, dim = hist_static_emb.shape

        # 2. 编码用户兴趣（先不Pad，用实际长度）
        history_len = batch["history_len"]
        # 临时构建一个和actual_len匹配的UserInterestEncoder（避免Pad）
        temp_encoder = UserInterestEncoder(
            hidden_dim=self.hidden_dim,
            max_len=actual_len,  # 用实际长度，不是max_len
            num_heads=new_config.sasrec_num_heads,
            num_layers=new_config.sasrec_num_layers
        ).to(hist_static_emb.device)
        user_interest = temp_encoder(hist_static_emb, history_len)

        # 3. 动态更新（只更新有效长度，不包含Pad）
        short_hist_emb = hist_static_emb[:, -10:, :] if actual_len >= 10 else hist_static_emb
        drift_score = self.dynamic_updater.detect_drift(short_hist_emb, hist_static_emb)
        new_features = user_interest.unsqueeze(1).expand(-1, actual_len, -1)
        hist_dynamic_emb = self.dynamic_updater.update(hist_static_emb, new_features, drift_score)

        # 4. 最后Pad到max_len（只在最后Pad，避免污染前面的计算）
        if actual_len < self.max_len:
            pad_len = self.max_len - actual_len
            padding = torch.zeros(batch_size, pad_len, dim,
                                  device=hist_dynamic_emb.device, dtype=hist_dynamic_emb.dtype)
            hist_dynamic_emb = torch.cat([hist_dynamic_emb, padding], dim=1)

        return hist_dynamic_emb

    def encode_target(self, batch):
        """编码目标物品 → 动态嵌入"""
        # 1. 语义ID → 静态嵌入
        target_indices = batch["target_indices"]  # (B, num_layers)
        indices_list = [target_indices[:, i].unsqueeze(1) for i in range(self.num_layers)]
        target_static_emb = self.semantic_id_to_emb(indices_list).squeeze(1)  # (B, hidden_dim)

        # 2. 编码用户兴趣
        hist_dynamic_emb = self.encode_history(batch)
        history_len = batch["history_len"]
        user_interest = self.user_interest_encoder(hist_dynamic_emb, history_len)

        # 3. 动态更新
        short_hist_emb = hist_dynamic_emb[:, -10:, :] if hist_dynamic_emb.shape[1] >= 10 else hist_dynamic_emb
        drift_score = self.dynamic_updater.detect_drift(short_hist_emb, hist_dynamic_emb)
        target_dynamic_emb = self.dynamic_updater.update(target_static_emb, user_interest, drift_score)

        return target_dynamic_emb

    def encode_all_items(self, indices_list, device):
        """编码全量物品 → 静态嵌入（推理用）"""
        indices_list = [indices_list[:, i].unsqueeze(1) for i in range(self.num_layers)]
        all_static_emb = self.semantic_id_to_emb(indices_list).squeeze(1).to(device)
        return all_static_emb

    def forward(self, batch):
        # 1. 基础编码：历史/目标的静态+动态嵌入（保留原有逻辑）
        hist_dynamic_emb = self.encode_history(batch)
        target_dynamic_emb = self.encode_target(batch)

        # 历史静态嵌入
        hist_indices = batch["history_indices"]
        indices_list = [hist_indices[:, :, i] for i in range(self.num_layers)]
        hist_static_emb = self.semantic_id_to_emb(indices_list)

        # 目标静态嵌入
        target_indices = batch["target_indices"]
        tgt_list = [target_indices[:, i].unsqueeze(1) for i in range(self.num_layers)]
        target_static_emb = self.semantic_id_to_emb(tgt_list).squeeze(1)

        # 2. 编码用户兴趣 + 计算模态权重（保留原有逻辑）
        history_len = batch["history_len"]
        user_interest = self.user_interest_encoder(hist_dynamic_emb, history_len)
        modal_weights = self.modal_attention(user_interest)  # (B, 2) 文本/视觉权重

        # ===================== 核心修复：dtype + 维度双对齐 =====================
        # 定义目标 dtype（和模型权重一致）
        target_dtype = self.text_proj.weight.dtype
        # 定义模态维度（文本40%，视觉60%）
        text_dim = int(self.hidden_dim * 0.4)

        # 3. 历史序列的模态特征处理（拼接+投影方案，彻底解决维度不匹配）
        # 取出原始特征 + dtype 对齐
        history_text_feat = batch["history_text_feat"].to(hist_dynamic_emb.device,
                                                          dtype=target_dtype)  # (B, max_len, text_raw_dim)
        history_vision_feat = batch["history_vision_feat"].to(hist_dynamic_emb.device,
                                                              dtype=target_dtype)  # (B, max_len, vision_raw_dim)

        # 第一步：降维到各自的目标维度
        history_text_feat = self.text_proj(history_text_feat)  # (B, max_len, text_dim) → 25
        history_vision_feat = self.vision_proj(history_vision_feat)  # (B, max_len, vision_dim) → 39

        # 第二步：扩展模态权重维度（适配序列维度）
        text_weight = modal_weights[:, 0:1].unsqueeze(1)  # (B, 1, 1)
        vision_weight = modal_weights[:, 1:2].unsqueeze(1)  # (B, 1, 1)

        # 第三步：加权后拼接 → 投影到 hidden_dim（核心修复！）
        # 加权各自特征
        weighted_text = text_weight * history_text_feat  # (B, max_len, 25)
        weighted_vision = vision_weight * history_vision_feat  # (B, max_len, 39)
        # 拼接（25+39=64）
        weighted_concat = torch.cat([weighted_text, weighted_vision], dim=-1)  # (B, max_len, 64)
        # 投影到 hidden_dim（确保维度完全匹配）
        hist_modal_emb = self.modal_proj(weighted_concat)  # (B, actual_len, hidden_dim)

        # 如果模态嵌入序列长度小于max_len，则Pad到max_len
        if hist_modal_emb.shape[1] < self.max_len:
            pad_len = self.max_len - hist_modal_emb.shape[1]
            padding = torch.zeros(hist_modal_emb.shape[0], pad_len, self.hidden_dim,
                                  device=hist_modal_emb.device, dtype=hist_modal_emb.dtype)
            hist_modal_emb = torch.cat([hist_modal_emb, padding], dim=1)

        # 4. 目标物品的模态特征处理（复用同一套逻辑）
        if "target_text_feat" in batch and "target_vision_feat" in batch:
            # 有独立目标特征时 + dtype 对齐
            target_text_feat = batch["target_text_feat"].to(target_dynamic_emb.device,
                                                            dtype=target_dtype)  # (B, text_raw_dim)
            target_vision_feat = batch["target_vision_feat"].to(target_dynamic_emb.device,
                                                                dtype=target_dtype)  # (B, vision_raw_dim)

            # 降维
            target_text_feat = self.text_proj(target_text_feat)  # (B, text_dim) →25
            target_vision_feat = self.vision_proj(target_vision_feat)  # (B, vision_dim) →39
        else:
            # 备用方案：从静态嵌入拆分
            target_text_feat = target_static_emb[:, :text_dim]
            target_vision_feat = target_static_emb[:, text_dim:]

        # 目标物品：加权+拼接+投影
        weighted_target_text = modal_weights[:, 0:1] * target_text_feat  # (B, 25)
        weighted_target_vision = modal_weights[:, 1:2] * target_vision_feat  # (B, 39)
        weighted_target_concat = torch.cat([weighted_target_text, weighted_target_vision], dim=-1)  # (B, 64)
        target_modal_emb = self.modal_proj(weighted_target_concat)  # (B, hidden_dim)

        # 5. 动态嵌入 + 模态加权融合（最终的个性化嵌入）
        # fusion_alpha: 动态嵌入权重, (1-fusion_alpha): 模态嵌入权重
        hist_final_emb = self.fusion_alpha * hist_dynamic_emb + (1 - self.fusion_alpha) * hist_modal_emb
        target_final_emb = self.fusion_alpha * target_dynamic_emb + (1 - self.fusion_alpha) * target_modal_emb

        # ===================== 返回值：包含所有核心输出 =====================
        return {
            # 原有返回值
            "history_emb": hist_dynamic_emb,
            "target_emb": target_dynamic_emb,
            "history_static_emb": hist_static_emb,
            "target_static_emb": target_static_emb,
            "user_interest": user_interest,
            "modal_weights": modal_weights,

            # 新增返回值
            "hist_modal_emb": hist_modal_emb,
            "target_modal_emb": target_modal_emb,
            "hist_final_emb": hist_final_emb,
            "target_final_emb": target_final_emb
        }
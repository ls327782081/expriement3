import torch
import torch.nn as nn
from config import new_config

# 这里引入你刚刚改好的 PMAT
from our_models.pmat_encoder import PMATAHRQEncoder


class PMATSASRec(nn.Module):
    """
    真正的 PMAT + SASRec 联合模型
    1. 输入：预计算好的语义 ID（history_indices, target_indices）
    2. PMAT：语义 ID → 动态物品嵌入 + 模态加权
    3. SASRec：动态嵌入序列 → 用户表征
    4. 输出 logits 用于 CE 损失
    无在线 AHRQ，无负样本
    """

    def __init__(
            self,
            num_items: int,
            semantic_hierarchy: dict,
            num_layers: int,
            layer_dim: int,
            fusion_type: str = "none",
            fixed_alpha: float = None
    ):
        super().__init__()
        self.num_items = num_items
        self.fusion_type = fusion_type
        self.use_fusion = fusion_type != "none"

        # ===================== PMAT 核心模块 =====================
        self.pmat = PMATAHRQEncoder(
            semantic_hierarchy=semantic_hierarchy,
            num_layers=num_layers,
            layer_dim=layer_dim
        )
        self.hidden_dim = self.pmat.hidden_dim

        # ===================== SASRec 序列建模 =====================
        self.max_len = new_config.sasrec_max_len
        self.num_heads = new_config.sasrec_num_heads
        self.sasrec_num_layers = new_config.sasrec_num_layers
        self.dropout = new_config.sasrec_dropout
        self.dim_feedforward = self.hidden_dim * 4

        # 位置编码
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.sasrec_num_layers,
            enable_nested_tensor=False
        )

        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=new_config.layer_norm_eps)
        self.dropout_layer = nn.Dropout(self.dropout)

        # 原始 ID Embedding（可选融合用）
        if self.use_fusion:
            self.item_embedding = nn.Embedding(num_items, self.hidden_dim)
            if fixed_alpha is not None:
                self.alpha = nn.Parameter(torch.tensor(fixed_alpha), requires_grad=False)
            else:
                self.alpha = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def fuse(self, id_emb, dyn_emb):
        if not self.use_fusion:
            return dyn_emb
        alpha = torch.sigmoid(self.alpha)
        if self.fusion_type == "add":
            return alpha * id_emb + (1 - alpha) * dyn_emb
        elif self.fusion_type == "concat":
            fused = torch.cat([id_emb, dyn_emb], dim=-1)
            return fused[..., :self.hidden_dim]
        return dyn_emb

    def get_user_embedding(self, batch, history_dyn_emb=None):
        """
        输入：batch 包含 history_indices（预计算语义ID）
        输出：用户表征
        修正点1：修复 history_dyn_emb 的判断逻辑
        """
        device = next(self.parameters()).device

        # ========== 1) PMAT 编码历史序列（修正判断逻辑） ==========
        if history_dyn_emb is None:  # 用 is None 替代 not，避免张量判断错误
            history_dyn_emb = self.pmat(batch)["hist_final_emb"]
        history_dyn_emb = self.dropout_layer(history_dyn_emb)
        B, max_L, D = history_dyn_emb.shape  # max_L 是Pad后的长度（比如50）

        # ========== 2) 可选融合 ID（保留维度对齐逻辑） ==========
        if self.use_fusion and "history_items" in batch:
            hist_ids = batch["history_items"].to(device)  # (B, actual_L)
            actual_L = hist_ids.shape[1]

            # 给hist_ids Pad到max_L（和history_dyn_emb对齐）
            if actual_L < max_L:
                pad_len = max_L - actual_L
                # Pad用0（padding ID）
                hist_ids = torch.cat([
                    hist_ids,
                    torch.zeros(B, pad_len, device=device, dtype=hist_ids.dtype)
                ], dim=1)

            hist_id_emb = self.item_embedding(hist_ids)  # (B, max_L, D)
            history_dyn_emb = self.fuse(hist_id_emb, history_dyn_emb)  # 维度匹配

        # ========== 3) SASRec 位置编码 + Transformer ==========
        pos = torch.arange(max_L, device=device).unsqueeze(0).expand(B, -1)
        history_dyn_emb = history_dyn_emb + self.position_embedding(pos)

        # 掩码
        history_len = batch["history_len"].to(device)
        padding_mask = torch.arange(max_L, device=device).unsqueeze(0) >= history_len.unsqueeze(1)
        causal_mask = torch.tril(torch.ones(max_L, max_L, device=device)).bool()
        causal_mask = torch.zeros_like(causal_mask, dtype=torch.float32).masked_fill(~causal_mask, -1e9)

        enc = self.transformer(
            history_dyn_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        enc = self.layer_norm(enc)

        # 取最后有效位
        last_idx = torch.clamp(history_len - 1, min=0)
        user_emb = enc[torch.arange(B, device=device), last_idx]
        return user_emb

    def forward(self, batch):
        """核心前向逻辑：返回用户表征、目标物品嵌入、PMAT输出"""
        pmat_out = self.pmat(batch)
        # 传入预计算的 hist_final_emb，避免重复调用 self.pmat(batch)
        user_emb = self.get_user_embedding(batch, pmat_out["hist_final_emb"])
        # 用动态+模态融合的目标嵌入
        pos_sem_feat = pmat_out["target_final_emb"]

        # 可选融合原始ID嵌入
        if self.use_fusion and "target_item" in batch:
            target_id = batch["target_item"].to(next(self.parameters()).device)
            target_id_emb = self.item_embedding(target_id)
            pos_sem_feat = self.fuse(target_id_emb, pos_sem_feat)

        return user_emb, pos_sem_feat, pmat_out

    def predict_all(self, batch, all_item_feat):
        """
        修正点2：实现真正的全量预测（用户emb × 全量物品emb）
        Args:
            batch: 测试集batch
            all_item_feat: 全量物品的动态+模态特征 (num_items, hidden_dim)
        Returns:
            logits: (B, num_items) 每个用户对所有物品的得分
        """
        with torch.no_grad():
            user_emb, _, _ = self.forward(batch)
            # 计算用户与全量物品的相似度（核心预测逻辑）
            logits = torch.matmul(user_emb, all_item_feat.T)
        return logits

    def get_all_item_sem_feat(self, indices_list, all_item_text=None, all_item_vision=None):
        """
        修正维度不匹配：确保文本+视觉维度之和 = hidden_dim，且权重扩展方式正确
        Args:
            indices_list: (num_items, num_layers) 全量物品的多层次语义ID
            all_item_text: (num_items, text_raw_dim) 全量物品文本特征（可选）
            all_item_vision: (num_items, vision_raw_dim) 全量物品视觉特征（可选）
        Returns:
            all_item_feat: (num_items, hidden_dim) 动态+模态融合特征
        """
        device = next(self.parameters()).device
        # 1. 获取动态嵌入
        all_dynamic_emb = self.pmat.encode_all_items(indices_list, device)  # (num_items, hidden_dim)

        # 2. 模态加权（核心修复：维度对齐）
        if all_item_text is not None and all_item_vision is not None:
            # 第一步：降维到正确维度（确保 text_dim + vision_dim = hidden_dim）
            text_dim = self.pmat.text_proj.out_features  # 获取文本降维后的维度（比如25）
            vision_dim = self.pmat.vision_proj.out_features  # 获取视觉降维后的维度（比如39）

            # 降维
            all_item_text = self.pmat.text_proj(all_item_text.to(device))  # (num_items, text_dim)
            all_item_vision = self.pmat.vision_proj(all_item_vision.to(device))  # (num_items, vision_dim)

            # 第二步：修复权重扩展方式（关键！）
            # default_weight 维度：(1, 2) → 扩展为 (1, 2)，分别对应文本/视觉权重
            default_weight = torch.tensor([0.4, 0.6], device=device).unsqueeze(0)  # (1, 2)

            # 正确的加权方式：文本权重 * 文本特征 + 视觉权重 * 视觉特征
            # 扩展权重维度：(1,2) → (1,1)，适配 (num_items, text_dim)
            text_weight = default_weight[:, 0:1]  # (1,1)
            vision_weight = default_weight[:, 1:2]  # (1,1)

            # 加权后相加（此时 text_dim 必须 = vision_dim，或用拼接后投影）
            # 方案1：如果 text_dim != vision_dim → 拼接后投影到 hidden_dim（推荐）
            all_item_concat = torch.cat([all_item_text, all_item_vision], dim=-1)  # (num_items, text_dim+vision_dim)
            # 新增投影层：确保最终维度 = hidden_dim
            if not hasattr(self.pmat, 'modal_proj'):
                self.pmat.modal_proj = nn.Linear(text_dim + vision_dim, self.hidden_dim).to(device)
            all_modal_emb = self.pmat.modal_proj(all_item_concat)  # (num_items, hidden_dim)

            # 方案2：如果 text_dim == vision_dim → 直接相加（备用）
            # all_modal_emb = text_weight * all_item_text + vision_weight * all_item_vision

            # 动态+模态融合
            all_item_feat = 0.7 * all_dynamic_emb + 0.3 * all_modal_emb
        else:
            # 备用：无模态特征时返回动态嵌入
            all_item_feat = all_dynamic_emb

        return all_item_feat
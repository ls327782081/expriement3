import torch
import torch.nn as nn
from config import new_config

# 这里引入你刚刚改好的 PMAT
from our_models.pmat_encoder import PMATAHRQEncoder


class PMATSASRec(nn.Module):
    """
    真正的 PMAT + SASRec 联合模型
    1. 输入：预计算好的语义 ID（history_indices, target_indices）
    2. PMAT：语义 ID → 动态物品嵌入
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
        fusion_type: str = "add",
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

    def get_user_embedding(self, batch):
        """
        输入：batch 包含 history_indices（预计算语义ID）
        输出：用户表征
        """
        device = next(self.parameters()).device

        # ========== 1) PMAT 编码历史序列 ==========
        history_dyn_emb = self.pmat.encode_history(batch)
        history_dyn_emb = self.dropout_layer(history_dyn_emb)

        # ========== 2) 可选融合 ID ==========
        if self.use_fusion and "history_items" in batch:
            hist_ids = batch["history_items"].to(device)
            hist_id_emb = self.item_embedding(hist_ids)
            history_dyn_emb = self.fuse(hist_id_emb, history_dyn_emb)

        # ========== 3) SASRec 位置编码 + Transformer ==========
        B, L, D = history_dyn_emb.shape
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        history_dyn_emb = history_dyn_emb + self.position_embedding(pos)

        # 掩码
        history_len = batch["history_len"].to(device)
        padding_mask = torch.arange(L, device=device).unsqueeze(0) >= history_len.unsqueeze(1)
        causal_mask = torch.tril(torch.ones(L, L, device=device)).bool()
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
        pmat_out = self.pmat(batch)
        user_emb = pmat_out["user_interest"]
        pos_sem_feat = pmat_out["target_emb"]

        if self.use_fusion and "target_item" in batch:
            target_id = batch["target_item"].to(next(self.parameters()).device)
            target_id_emb = self.item_embedding(target_id)
            pos_sem_feat = self.fuse(target_id_emb, pos_sem_feat)

        # 关键：返回 pmat_out 给训练脚本算损失
        return user_emb, pos_sem_feat, pmat_out

    def predict_all(self, batch):
        with torch.no_grad():
            logits, _, _ = self.forward(batch)
        return logits

    def get_all_item_sem_feat(self, indices_list):
        """
        全量物品语义ID转特征
        Args:
            indices_list: (num_items, num_layers) 全量物品的多层次语义ID
        Returns:
            all_item_feat: (num_items, hidden_dim) 全量物品的语义特征
        """
        device = next(self.parameters()).device
        all_item_feat = self.pmat.encode_all_items(indices_list, device)
        return all_item_feat

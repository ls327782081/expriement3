import torch
import torch.nn as nn
import torch.nn.functional as F
from config import new_config
from our_models.ah_rq import AdaptiveHierarchicalQuantizer


class PMATSASRec(nn.Module):
    """
    PMAT-SASRec + AH-RQ 序列推荐模型

    核心设计：
    1. 使用外部传入的 AHRQ 预训练模型进行物品编码
    2. 语义ID映射为特征 → 作为SASRec的输入（替代原始ID Embedding）
    3. SASRec序列建模 → 输出推荐分数
    4. 支持ID Embedding与量化特征融合

    与 SASRecAHRQ 的区别：
    - 使用 PMAT 风格的多模态编码（通过 AHRQ 的 use_multimodal=True）
    - 保留了 PMAT 的用户兴趣编码逻辑
    """

    def __init__(
        self,
        ahrq_model: AdaptiveHierarchicalQuantizer,
        num_items: int = None,
        fusion_type: str = "add",
        fixed_alpha: float = None,
        dynamic_params: dict = None
    ):
        """
        Args:
            ahrq_model: AHRQ量化器模型（预训练的）
            num_items: 物品数量
            fusion_type: 融合类型 ("add", "concat", "none")
            fixed_alpha: 固定融合权重（可选）
            dynamic_params: 动态参数dict
        """
        super().__init__()

        # 从 AHRQ 模型动态读取层次配置
        self.semantic_hierarchy = ahrq_model.semantic_hierarchy
        self.num_layers = ahrq_model.num_layers
        self.layer_dim = ahrq_model.layer_dim

        # 动态计算隐藏维度（总层数 * 每层维度）
        self.hidden_dim = self.num_layers * self.layer_dim

        # 保存 AHRQ 模型（用于编码物品）
        self.ahrq_model = ahrq_model

        # 融合配置
        self.fusion_type = fusion_type
        self.use_fusion = fusion_type != "none"

        # 动态创建语义ID Embedding层
        self.semantic_id_emb = nn.ModuleDict()
        for semantic_type, config in self.semantic_hierarchy.items():
            cb_size = config["codebook_size"]
            for layer in config["layers"]:
                self.semantic_id_emb[f"{semantic_type}_{layer}"] = nn.Embedding(cb_size, self.layer_dim)

        # 原始物品ID Embedding（用于融合）
        if num_items is None:
            num_items = new_config.sasrec_ahrq_num_items
        self.num_items = num_items
        self.item_embedding = nn.Embedding(num_items, self.hidden_dim)

        # 融合方式：可学习alpha或固定alpha
        if fixed_alpha is not None:
            self.alpha = nn.Parameter(torch.tensor(fixed_alpha), requires_grad=False)
            self.learnable_alpha = False
        else:
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.learnable_alpha = True

        # SASRec核心配置
        self.max_len = new_config.sasrec_max_len

        if dynamic_params is not None:
            self.num_heads = dynamic_params.get("num_heads", new_config.sasrec_num_heads)
            self.sasrec_num_layers = dynamic_params.get("sasrec_num_layers", new_config.sasrec_num_layers)
            self.dropout = dynamic_params.get("dropout", new_config.sasrec_dropout)
            self.dim_feedforward = dynamic_params.get("dim_feedforward", self.hidden_dim * 4)
            self.dynamic_params_used = dynamic_params
        else:
            self.num_heads = new_config.sasrec_num_heads
            self.sasrec_num_layers = new_config.sasrec_num_layers
            self.dropout = new_config.sasrec_dropout
            self.dim_feedforward = self.hidden_dim * 4
            self.dynamic_params_used = None

        self.dropout_proj = nn.Dropout(self.dropout)
        self.dropout_id = nn.Dropout(self.dropout)

        # 位置编码
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.sasrec_num_layers,
            enable_nested_tensor=False
        )

        # 推荐打分层
        self.score_layer = nn.Linear(self.hidden_dim, 1)

        # 输出LayerNorm
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=new_config.layer_norm_eps)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        for semantic_type, config in self.semantic_hierarchy.items():
            for layer in config["layers"]:
                cb_key = f"{semantic_type}_{layer}"
                if cb_key in self.semantic_id_emb:
                    nn.init.xavier_uniform_(self.semantic_id_emb[cb_key].weight)

        nn.init.xavier_uniform_(self.position_embedding.weight)

        for module in self.transformer_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.MultiheadAttention):
                for param in module.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.constant_(param, 0.0)

        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.alpha, 0.5)

    def get_item_id_emb(self, item_ids):
        """获取原始物品ID Embedding"""
        return self.item_embedding(item_ids)

    def fuse_features(self, id_emb, sem_emb):
        """融合ID特征和语义特征"""
        if not self.use_fusion:
            return sem_emb

        alpha = torch.sigmoid(self.alpha)

        if self.fusion_type == "add":
            fused = alpha * id_emb + (1 - alpha) * sem_emb
        elif self.fusion_type == "concat":
            fused = torch.cat([id_emb, sem_emb], dim=-1)
            fused = F.linear(fused, torch.eye(self.hidden_dim, self.hidden_dim * 2, device=fused.device)[:, :self.hidden_dim])
        else:
            fused = sem_emb

        return fused

    def _tensor_to_indices_list(self, indices_tensor, seq_len=None):
        """将 tensor 格式的语义ID转换为 list 格式"""
        if indices_tensor.dim() == 3:
            return [indices_tensor[:, :, i] for i in range(indices_tensor.shape[2])]
        elif indices_tensor.dim() == 2:
            if seq_len is None:
                seq_len = 1
            return [indices_tensor[:, i].unsqueeze(1) for i in range(indices_tensor.shape[1])]
        else:
            raise ValueError(f"indices_tensor 维度应为2或3，当前为{indices_tensor.dim()}")

    def get_all_item_sem_feat(self, all_item_indices):
        """全量物品语义ID转特征"""
        indices_list = []
        for layer in range(self.num_layers):
            layer_indices = all_item_indices[:, layer]
            indices_list.append(layer_indices)

        all_item_feat = self.semantic_id_to_feat(indices_list)
        all_item_feat = all_item_feat.squeeze(1)

        assert all_item_feat.shape == (all_item_indices.shape[0], self.hidden_dim), \
            f"全量特征维度错误！预期({all_item_indices.shape[0]},{self.hidden_dim})，实际{all_item_feat.shape}"

        return all_item_feat

    def semantic_id_to_feat(self, indices_list):
        """将多层次语义ID映射为特征"""
        semantic_blocks = []
        layer_idx = 0

        for semantic_type, config in self.semantic_hierarchy.items():
            cb_size = config["codebook_size"]
            for layer in config["layers"]:
                cb_key = f"{semantic_type}_{layer}"
                indices = indices_list[layer_idx]

                if (indices < 0).any() or (indices >= cb_size).any():
                    raise ValueError(f"{semantic_type}层{layer}的ID超出范围[0, {cb_size - 1}]")

                block_feat = self.semantic_id_emb[cb_key](indices)
                semantic_blocks.append(block_feat)
                layer_idx += 1

        semantic_feat = torch.cat(semantic_blocks, dim=-1)

        if semantic_feat.shape[-1] != self.hidden_dim:
            raise ValueError(f"拼接后维度{semantic_feat.shape[-1]}≠配置维度{self.hidden_dim}")

        semantic_feat = F.gelu(semantic_feat)
        semantic_feat = F.normalize(semantic_feat, p=2, dim=-1)
        semantic_feat = semantic_feat * (self.hidden_dim ** 0.5)

        return semantic_feat

    def forward(self, batch):
        """
        前向传播：历史语义ID → 语义特征 → SASRec → 用户表征

        Args:
            batch: dict
                - history_indices: (batch, max_len, num_layers) 历史序列语义ID
                - history_items: (batch, max_len) 历史物品ID（用于ID融合）
                - history_len: (batch,) 历史序列有效长度
                - target_indices: (batch, num_layers) 目标正样本语义ID
                - target_item: (batch,) 目标物品ID（用于ID融合）
        Returns:
            user_emb: (batch, hidden_dim) 用户表征
            pos_sem_feat: (batch, hidden_dim) 正样本语义特征（融合后）
        """
        device = next(self.parameters()).device

        # 历史序列语义ID
        hist_indices = batch["history_indices"].to(device)
        history_len = batch["history_len"].to(device)

        # 序列编码
        user_emb = self.get_user_embedding(batch)

        # 正样本特征
        pos_indices = batch["target_indices"]
        pos_indices_list = self._tensor_to_indices_list(pos_indices, seq_len=1)
        pos_sem_feat = self.semantic_id_to_feat(pos_indices_list).squeeze(1)

        # 融合：目标物品ID特征 + 语义特征
        if self.use_fusion and "target_item" in batch:
            target_item = batch["target_item"].to(device)
            pos_id_feat = self.get_item_id_emb(target_item.unsqueeze(1)).squeeze(1)
            pos_id_feat = self.dropout_id(pos_id_feat)
            pos_sem_feat = self.fuse_features(pos_id_feat, pos_sem_feat)

        return user_emb, pos_sem_feat

    def get_user_embedding(self, batch):
        """抽取用户表征"""
        hist_indices = batch["history_indices"]
        history_len = batch["history_len"]
        device = hist_indices.device

        # 语义ID转特征
        hist_indices_list = self._tensor_to_indices_list(hist_indices)
        history_sem_feat = self.semantic_id_to_feat(hist_indices_list)
        history_sem_feat = self.dropout_proj(history_sem_feat)

        # 融合：添加ID Embedding
        if self.use_fusion and "history_items" in batch:
            history_items = batch["history_items"].to(device)
            history_id_feat = self.get_item_id_emb(history_items)
            history_id_feat = self.dropout_id(history_id_feat)
            history_sem_feat = self.fuse_features(history_id_feat, history_sem_feat)

        # 位置编码
        batch_size = history_sem_feat.shape[0]
        positions = torch.arange(history_sem_feat.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
        history_sem_feat = history_sem_feat + self.position_embedding(positions)

        # 掩码构造
        bool_mask = torch.tril(torch.ones((history_sem_feat.shape[1], history_sem_feat.shape[1]), device=device)).bool()
        mask = torch.zeros_like(bool_mask, dtype=torch.float32).masked_fill(~bool_mask, -1e9)

        max_len = history_sem_feat.shape[1]
        position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        src_key_padding_mask = position_ids >= history_len.unsqueeze(1)

        # Transformer编码
        encoded_seq = self.transformer_encoder(history_sem_feat, mask=mask, src_key_padding_mask=src_key_padding_mask)
        encoded_seq = self.layer_norm(encoded_seq)

        # 取最后有效位生成用户表征
        last_indices = torch.clamp(history_len - 1, min=0)
        user_emb = encoded_seq[torch.arange(batch_size, device=device), last_indices]

        return user_emb

    def predict_all(self, batch, indices_list):
        """全量物品打分"""
        self.eval()

        with torch.no_grad():
            user_emb = self.get_user_embedding(batch)

            indices_list_tensor = indices_list.unsqueeze(1)
            all_indices_list = self._tensor_to_indices_list(indices_list_tensor, seq_len=1)
            all_sem_feat = self.semantic_id_to_feat(all_indices_list)
            all_sem_feat = all_sem_feat.squeeze(1)

            if self.use_fusion:
                all_item_ids = torch.arange(self.num_items, device=indices_list.device)
                all_id_feat = self.get_item_id_emb(all_item_ids.unsqueeze(1)).squeeze(1)
                all_sem_feat = self.fuse_features(all_id_feat, all_sem_feat)

            all_scores = (user_emb.unsqueeze(1) * all_sem_feat.unsqueeze(0)).sum(dim=-1)

        return all_scores


# ===================== 便捷调用函数 =====================
def build_pmat_sasrec_model(ahrq_model=None, num_items=None):
    """构建PMAT-SASRec完整推荐模型"""
    if ahrq_model is None:
        # 如果没有传入 AHRQ 模型，创建一个默认的
        ahrq_model = AdaptiveHierarchicalQuantizer(
            hidden_dim=new_config.ahrq_hidden_dim,
            semantic_hierarchy=new_config.semantic_hierarchy,
            use_multimodal=True,
            text_dim=new_config.text_dim,
            visual_dim=new_config.visual_dim,
            beta=new_config.ahrq_beta,
            use_ema=new_config.ahrq_use_ema,
            ema_decay=0.99,
            reset_unused_codes=new_config.ahrq_reset_unused_codes,
            reset_threshold=new_config.ahrq_reset_threshold
        ).to(new_config.device)

    model = PMATSASRec(ahrq_model=ahrq_model, num_items=num_items).to(new_config.device)
    return model


if __name__ == "__main__":
    torch.manual_seed(new_config.seed)

    # 创建测试用的 AHRQ 模型
    ahrq_model = AdaptiveHierarchicalQuantizer(
        hidden_dim=new_config.ahrq_hidden_dim,
        semantic_hierarchy=new_config.semantic_hierarchy,
        use_multimodal=True,
        text_dim=new_config.text_dim,
        visual_dim=new_config.visual_dim,
        beta=new_config.ahrq_beta,
        use_ema=new_config.ahrq_use_ema,
        ema_decay=0.99,
        reset_unused_codes=new_config.ahrq_reset_unused_codes,
        reset_threshold=new_config.ahrq_reset_threshold
    ).to(new_config.device)

    # 构造测试数据（与 train_sasrec_ahrq.py 一致的数据格式）
    batch = {
        "history_indices": torch.randint(0, 256, (4, new_config.sasrec_max_len, 4)),  # (batch, max_len, num_layers)
        "history_items": torch.randint(1, 1000, (4, new_config.sasrec_max_len)),  # (batch, max_len)
        "history_len": torch.tensor([20, 15, 30, 10]),
        "target_indices": torch.randint(0, 256, (4, 4)),  # (batch, num_layers)
        "target_item": torch.tensor([100, 200, 300, 400])
    }

    # 初始化模型
    model = PMATSASRec(ahrq_model=ahrq_model, num_items=1000)
    model.train()

    # 前向传播
    user_emb, pos_sem_feat = model(batch)

    print("模型输出维度验证：")
    print(f"user_emb: {user_emb.shape}")
    print(f"pos_sem_feat: {pos_sem_feat.shape}")
    print("模型前向传播测试通过！")
import torch
import torch.nn as nn
from config import new_config
from our_models.ah_rq import AdaptiveHierarchicalQuantizer
import torch.nn.functional as F


class SASRecAHRQ(nn.Module):
    """
    SASRec + AH-RQ 序列推荐模型

    核心设计：动态从 AHRQ 模型读取层次配置，支持任意语义层级
    1. AH-RQ量化特征 → 生成多层次语义ID
    2. 语义ID映射为特征 → 作为SASRec的输入（替代原始ID Embedding）
    3. SASRec序列建模 → 输出推荐分数
    4. 支持ID Embedding与量化特征融合
    """

    def __init__(
        self,
        ahrq_model: AdaptiveHierarchicalQuantizer,
        num_items: int = None,
        hidden_dim: int = None,
        fusion_type: str = "add",
        fixed_alpha: float = None,
        dynamic_params: dict = None,
        use_quantized_fusion: bool = False,
        use_raw_fusion: bool = False,
        use_semantic_id: bool = True,
        quantized_features = None,
        raw_features = None
    ):
        """
        Args:
            ahrq_model: AHRQ量化器模型
            num_items: 物品数量
            fusion_type: 融合类型 ("add", "concat", "none")
            fixed_alpha: 固定融合权重（可选）
            dynamic_params: 动态参数dict，包含：
                - num_heads: 注意力头数
                - sasrec_num_layers: Transformer层数
                - dropout: Dropout比率
                - dim_feedforward: FFN隐藏层维度
        """
        super().__init__()

        # 从 AHRQ 模型动态读取层次配置
        self.semantic_hierarchy = ahrq_model.semantic_hierarchy
        self.num_layers = ahrq_model.num_layers
        self.layer_dim = ahrq_model.layer_dim

        # 隐藏维度：优先使用传入值，否则使用 AHRQ 计算值
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.num_layers * self.layer_dim

        # 连续特征融合配置（新增）
        self.use_quantized_fusion = use_quantized_fusion
        self.use_raw_fusion = use_raw_fusion
        self.use_semantic_id = use_semantic_id
        self.quantized_features = quantized_features
        self.raw_features = raw_features

        # 融合配置
        self.fusion_type = fusion_type
        self.use_fusion = fusion_type != "none"

        # concat 融合投影层
        if fusion_type == "concat":
            self.concat_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # 动态创建语义ID Embedding层
        self.semantic_id_emb = nn.ModuleDict()
        for semantic_type, config in self.semantic_hierarchy.items():
            cb_size = config["codebook_size"]
            for layer in config["layers"]:
                self.semantic_id_emb[f"{semantic_type}_{layer}"] = nn.Embedding(cb_size, self.layer_dim)

        # ===== 新增：原始物品ID Embedding（用于融合）=====
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

        # SASRec核心配置 - 使用动态参数或默认配置
        self.max_len = new_config.sasrec_max_len

        if dynamic_params is not None:
            # 使用动态参数（根据码本配置调整）
            self.num_heads = dynamic_params.get("num_heads", new_config.sasrec_num_heads)
            self.sasrec_num_layers = dynamic_params.get("sasrec_num_layers", new_config.sasrec_num_layers)
            self.dropout = dynamic_params.get("dropout", new_config.sasrec_dropout)
            self.dim_feedforward = dynamic_params.get("dim_feedforward", self.hidden_dim * 4)
            self.dynamic_params_used = dynamic_params
        else:
            # 使用默认配置
            self.num_heads = new_config.sasrec_num_heads
            self.sasrec_num_layers = new_config.sasrec_num_layers
            self.dropout = new_config.sasrec_dropout
            self.dim_feedforward = self.hidden_dim * 4
            self.dynamic_params_used = None

        self.dropout_proj = nn.Dropout(self.dropout)
        self.dropout_id = nn.Dropout(self.dropout)

        # 位置编码
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_dim)

        # Transformer编码器 - 使用动态dim_feedforward
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

        # 动态遍历所有语义层级进行初始化
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

        # 初始化 item_embedding
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)

        # 初始化 concat_proj
        if hasattr(self, 'concat_proj'):
            nn.init.xavier_uniform_(self.concat_proj.weight)
            nn.init.zeros_(self.concat_proj.bias)

        nn.init.constant_(self.alpha, 0.5)

    def get_item_id_emb(self, item_ids):
        """
        获取原始物品ID Embedding

        Args:
            item_ids: (batch, seq_len) 物品ID
        Returns:
            id_emb: (batch, seq_len, hidden_dim) 物品ID特征
        """
        return self.item_embedding(item_ids)

    def fuse_features(self, id_emb, sem_emb):
        """
        融合ID特征和语义特征

        Args:
            id_emb: (batch, seq_len, hidden_dim) 原始ID特征
            sem_emb: (batch, seq_len, hidden_dim) 量化语义特征
        Returns:
            fused: (batch, seq_len, hidden_dim) 融合后特征
        """
        if not self.use_fusion:
            return sem_emb

        alpha = torch.sigmoid(self.alpha)  # 使用sigmoid约束到(0,1)

        if self.fusion_type == "add":
            # 加权融合: alpha * id + (1-alpha) * sem
            fused = alpha * id_emb + (1 - alpha) * sem_emb
        elif self.fusion_type == "concat":
            # 拼接后投影
            fused = torch.cat([id_emb, sem_emb], dim=-1)
            fused = self.concat_proj(fused)
        else:
            fused = sem_emb

        return fused

    def _tensor_to_indices_list(self, indices_tensor, seq_len=None):
        """
        将 tensor 格式的语义ID转换为 list 格式，供 semantic_id_to_feat 使用
        Args:
            indices_tensor: 语义ID tensor，形状为 (batch, seq_len, num_layers) 或 (batch, num_layers)
            seq_len: 如果是单物品输入(seq_len=1)，指定序列长度
        Returns:
            indices_list: list of tensor，每层语义ID (batch, seq_len)
        """
        if indices_tensor.dim() == 3:
            # (batch, seq_len, num_layers) -> list of (batch, seq_len)
            return [indices_tensor[:, :, i] for i in range(indices_tensor.shape[2])]
        elif indices_tensor.dim() == 2:
            # (batch, num_layers) -> list of (batch, 1)
            if seq_len is None:
                seq_len = 1
            return [indices_tensor[:, i].unsqueeze(1) for i in range(indices_tensor.shape[1])]
        else:
            raise ValueError(f"indices_tensor 维度应为2或3，当前为{indices_tensor.dim()}")

    def get_all_item_sem_feat(self, all_item_indices):
        """
        全量物品语义ID转特征（不含连续特征融合）
        Args:
            all_item_indices: (num_items, num_layers) 全量物品的多层次语义ID
        Returns:
            all_item_feat: (num_items, hidden_dim) 全量物品的语义特征
        """
        # 将(num_items, num_layers)的语义ID拆分为indices_list
        indices_list = []
        for layer in range(self.num_layers):
            layer_indices = all_item_indices[:, layer]
            indices_list.append(layer_indices)

        # 调用 semantic_id_to_feat 函数
        all_item_feat = self.semantic_id_to_feat(indices_list)  # (num_items, 1, hidden_dim)

        # 挤压多余的seq_len维度
        all_item_feat = all_item_feat.squeeze(1)

        assert all_item_feat.shape == (all_item_indices.shape[0], self.hidden_dim), \
            f"全量特征维度错误！预期({all_item_indices.shape[0]},{self.hidden_dim})，实际{all_item_feat.shape}"

        return all_item_feat

    def get_all_item_full_feat(self, all_item_indices):
        """
        全量物品完整特征（含连续特征融合）
        用于训练阶段，包含语义ID特征+连续特征融合

        Args:
            all_item_indices: (num_items, num_layers) 全量物品的多层次语义ID
        Returns:
            all_item_feat: (num_items, hidden_dim) 全量物品的完整特征
        """
        # 语义ID特征
        all_item_feat = self.get_all_item_sem_feat(all_item_indices)

        # 连续特征融合
        if self.use_quantized_fusion or self.use_raw_fusion:
            # 构造indices_list用于get_continuous_feat
            indices_list_tensor = all_item_indices.unsqueeze(1)  # (num_items, 1, num_layers)
            all_indices_list = self._tensor_to_indices_list(indices_list_tensor, seq_len=1)

            # 使用物品ID索引连续特征（而非语义ID）
            all_item_ids = torch.arange(all_item_indices.shape[0], device=all_item_indices.device).unsqueeze(1)
            all_cont_feat = self.get_continuous_feat(all_indices_list, item_ids=all_item_ids)  # (num_items, 1, hidden_dim)
            if all_cont_feat is not None:
                all_cont_feat = all_cont_feat.squeeze(1)  # (num_items, hidden_dim)

                # 融合语义ID特征和连续特征
                alpha = torch.sigmoid(self.alpha)
                if self.use_semantic_id:
                    all_item_feat = alpha * all_item_feat + (1 - alpha) * all_cont_feat
                else:
                    all_item_feat = all_cont_feat

        return all_item_feat

    def semantic_id_to_feat(self, indices_list):
        """
        核心：将AH-RQ生成的多层次语义ID映射为特征
        动态遍历所有语义层级进行特征映射

        Args:
            indices_list: list of tensor，每层语义ID (batch, seq_len)
        Returns:
            semantic_feat: (batch, seq_len, hidden_dim) 语义特征
        """

        semantic_blocks = []
        layer_idx = 0

        # 动态遍历所有语义层级
        for semantic_type, config in self.semantic_hierarchy.items():
            cb_size = config["codebook_size"]
            for layer in config["layers"]:
                cb_key = f"{semantic_type}_{layer}"
                indices = indices_list[layer_idx]  # (batch, seq_len)

                # 越界检查
                if (indices < 0).any() or (indices >= cb_size).any():
                    raise ValueError(f"{semantic_type}层{layer}的ID超出范围[0, {cb_size - 1}]")

                block_feat = self.semantic_id_emb[cb_key](indices)  # (batch, seq_len, layer_dim)
                semantic_blocks.append(block_feat)
                layer_idx += 1

        # 拼接所有层特征
        semantic_feat = torch.cat(semantic_blocks, dim=-1)  # (batch, seq_len, hidden_dim)

        if semantic_feat.shape[-1] != self.hidden_dim:
            raise ValueError(f"拼接后维度{semantic_feat.shape[-1]}≠配置维度{self.hidden_dim}")

        # 激活 + 归一化 + 尺度恢复
        semantic_feat = F.gelu(semantic_feat)
        semantic_feat = F.normalize(semantic_feat, p=2, dim=-1)
        semantic_feat = semantic_feat * (self.hidden_dim ** 0.5)

        return semantic_feat

    def get_continuous_feat(self, indices_list, item_ids=None):
        """
        从物品ID查询对应的连续特征（量化特征或原始特征）

        Args:
            indices_list: list of tensor，每层语义ID (batch, seq_len) - 仅用于占位
            item_ids: (batch, seq_len) 物品ID，用于索引连续特征
        Returns:
            continuous_feat: (batch, seq_len, hidden_dim) 连续特征
        """
        batch_size = indices_list[0].shape[0]
        seq_len = indices_list[0].shape[1]
        device = indices_list[0].device

        # 确定使用哪种连续特征
        if self.use_quantized_fusion and self.quantized_features is not None:
            continuous_feat = self.quantized_features  # (num_items, hidden_dim)
        elif self.use_raw_fusion and self.raw_features is not None:
            continuous_feat = self.raw_features  # (num_items, hidden_dim)
        else:
            return None

        # 使用传入的物品ID索引连续特征，而不是语义ID
        if item_ids is None:
            raise ValueError("get_continuous_feat 需要传入 item_ids 参数来索引连续特征")
        item_indices = item_ids.to(device)
        continuous_feat = continuous_feat.to(device)

        # 提取对应物品的连续特征
        feat = continuous_feat[item_indices]  # (batch, seq_len, hidden_dim)

        # 归一化
        feat = F.normalize(feat, p=2, dim=-1)
        feat = feat * (self.hidden_dim ** 0.5)

        return feat

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
        
        # 语义ID特征
        if self.use_semantic_id:
            pos_sem_feat = self.semantic_id_to_feat(pos_indices_list).squeeze(1)  # (B, D)
        else:
            pos_sem_feat = None
        
        # 连续特征（量化特征或原始特征）
        if self.use_quantized_fusion or self.use_raw_fusion:
            pos_item = batch["target_item"].unsqueeze(1) if "target_item" in batch else None
            pos_cont_feat = self.get_continuous_feat(pos_indices_list, item_ids=pos_item).squeeze(1) if pos_item is not None else None
        else:
            pos_cont_feat = None
        
        # 融合语义ID特征和连续特征
        if pos_sem_feat is not None and pos_cont_feat is not None:
            alpha = torch.sigmoid(self.alpha)
            pos_sem_feat = alpha * pos_sem_feat + (1 - alpha) * pos_cont_feat
        elif pos_cont_feat is not None:
            pos_sem_feat = pos_cont_feat
        
        # 融合：目标物品ID特征 + 语义特征（原有逻辑）
        if self.use_fusion and "target_item" in batch and pos_sem_feat is not None:
            target_item = batch["target_item"].to(device)
            pos_id_feat = self.get_item_id_emb(target_item.unsqueeze(1)).squeeze(1)  # (B, D)
            pos_id_feat = self.dropout_id(pos_id_feat)
            pos_sem_feat = self.fuse_features(pos_id_feat, pos_sem_feat)

        return user_emb, pos_sem_feat

    def get_user_embedding(self, batch):
        """
        抽取用户表征

        Args:
            batch: dict，包含:
                - history_indices: (batch, max_len, num_layers) 历史语义ID
                - history_items: (batch, max_len) 历史物品ID（用于ID融合）
                - history_len: (batch,) 历史序列有效长度
        Returns:
            user_emb: (batch, hidden_dim) 用户表征
        """
        hist_indices = batch["history_indices"]
        history_len = batch["history_len"]
        device = hist_indices.device

        # 语义ID转特征
        hist_indices_list = self._tensor_to_indices_list(hist_indices)
        
        # 语义ID特征
        if self.use_semantic_id:
            history_sem_feat = self.semantic_id_to_feat(hist_indices_list)  # (B, L, D)
        else:
            history_sem_feat = torch.zeros(hist_indices.shape[0], hist_indices.shape[1], self.hidden_dim, device=device)
        
        # 连续特征（量化特征或原始特征）
        if self.use_quantized_fusion or self.use_raw_fusion:
            history_items = batch.get("history_items", None)
            if history_items is not None:
                history_cont_feat = self.get_continuous_feat(hist_indices_list, item_ids=history_items)  # (B, L, D)
            else:
                history_cont_feat = None
            if history_cont_feat is not None:
                # 融合语义ID特征和连续特征
                alpha = torch.sigmoid(self.alpha)
                if self.use_semantic_id:
                    history_sem_feat = alpha * history_sem_feat + (1 - alpha) * history_cont_feat
                else:
                    history_sem_feat = history_cont_feat
        
        history_sem_feat = self.dropout_proj(history_sem_feat)

        # 融合：添加ID Embedding（原有逻辑）
        if self.use_fusion and "history_items" in batch:
            history_items = batch["history_items"].to(device)
            history_id_feat = self.get_item_id_emb(history_items)  # (B, L, D)
            history_id_feat = self.dropout_id(history_id_feat)
            history_sem_feat = self.fuse_features(history_id_feat, history_sem_feat)

        # 位置编码
        batch_size = history_sem_feat.shape[0]
        positions = torch.arange(history_sem_feat.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
        history_sem_feat = history_sem_feat + self.position_embedding(positions)

        # 掩码构造
        # 未来掩码
        bool_mask = torch.tril(torch.ones((history_sem_feat.shape[1], history_sem_feat.shape[1]), device=device)).bool()
        mask = torch.zeros_like(bool_mask, dtype=torch.float32).masked_fill(~bool_mask, -1e9)
        # Padding掩码
        max_len = history_sem_feat.shape[1]
        position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        src_key_padding_mask = position_ids >= history_len.unsqueeze(1)

        # Transformer编码
        encoded_seq = self.transformer_encoder(history_sem_feat, mask=mask, src_key_padding_mask=src_key_padding_mask)
        encoded_seq = self.layer_norm(encoded_seq)

        # 取最后有效位生成用户表征
        last_indices = torch.clamp(history_len - 1, min=0)
        user_emb = encoded_seq[torch.arange(batch_size, device=device), last_indices]  # (B, D)

        return user_emb

    def predict_all(self, batch, indices_list):
        self.eval()
        with torch.no_grad():
            # 抽取用户表征
            user_emb = self.get_user_embedding(batch)  # (batch, hidden_dim)
            # 复用get_all_item_full_feat，避免重复计算
            all_sem_feat = self.get_all_item_full_feat(indices_list)  # (item_num, hidden_dim)
            # 融合：添加物品ID特征
            if self.use_fusion:
                all_item_ids = torch.arange(self.num_items, device=indices_list.device)
                all_id_feat = self.get_item_id_emb(all_item_ids.unsqueeze(1)).squeeze(1)  # (item_num, D)
                all_sem_feat = self.fuse_features(all_id_feat, all_sem_feat)
            # 计算打分
            all_scores = (user_emb.unsqueeze(1) * all_sem_feat.unsqueeze(0)).sum(dim=-1)  # (batch, item_num)
        return all_scores
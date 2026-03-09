import torch
import torch.nn as nn
from config import new_config
from our_models.ah_rq import AdaptiveHierarchicalQuantizer  # 升级后的AH-RQ
import torch.nn.functional as F


class SASRecAHRQ(nn.Module):
    """
    正确逻辑的SASRec+AH-RQ：
    1. 原始物品ID → 加载单模态特征（如物品描述/属性特征）
    2. AH-RQ量化特征 → 生成多层次语义ID
    3. 语义ID映射为特征 → 作为SASRec的输入（替代原始ID Embedding）
    4. SASRec序列建模 → 输出推荐分数
    """

    def __init__(self):
        super().__init__()

        # 3. 语义ID映射层（将多层次语义ID转为特征，替代原始Embedding）
        self.num_layers = len(new_config.semantic_hierarchy["topic"]["layers"]) + len(
            new_config.semantic_hierarchy["style"]["layers"])

        self.semantic_id_emb = nn.ModuleDict()
        # Topic层语义ID Embedding
        for layer in new_config.semantic_hierarchy["topic"]["layers"]:
            cb_size = new_config.semantic_hierarchy["topic"]["codebook_size"]
            self.semantic_id_emb[f"topic_{layer}"] = nn.Embedding(cb_size, new_config.pmat_hidden_dim // self.num_layers)
        # Style层语义ID Embedding
        for layer in new_config.semantic_hierarchy["style"]["layers"]:
            cb_size = new_config.semantic_hierarchy["style"]["codebook_size"]
            self.semantic_id_emb[f"style_{layer}"] = nn.Embedding(cb_size, new_config.pmat_hidden_dim // self.num_layers)

        # 4. SASRec核心（输入为语义ID映射的特征）
        self.hidden_dim = new_config.sasrec_hidden_dim
        self.max_len = new_config.sasrec_max_len
        self.num_heads = new_config.sasrec_num_heads
        self.sasrec_num_layers = new_config.sasrec_num_layers
        self.dropout = new_config.sasrec_dropout

        self.dropout_proj = nn.Dropout(new_config.sasrec_dropout)

        # 位置编码（保留）
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_dim)

        # Transformer编码器（保留）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=self.sasrec_num_layers,
                                                         enable_nested_tensor=False)

        # 推荐打分层：复用语义ID embedding权重（与原生SASRec一致）
        self.score_layer = nn.Linear(self.hidden_dim, 1)

        # 新增：输出LayerNorm（与原生SASRec一致）
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=new_config.layer_norm_eps)

        self.alpha = nn.Parameter(torch.tensor(0.5))

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for layer in new_config.semantic_hierarchy["topic"]["layers"]:
            # 论文推荐的初始化方式（Xavier均匀初始化）
            nn.init.xavier_uniform_(self.semantic_id_emb[f"topic_{layer}"].weight)
            # Style层语义ID Embedding
        for layer in new_config.semantic_hierarchy["style"]["layers"]:
            nn.init.xavier_uniform_(self.semantic_id_emb[f"style_{layer}"].weight)

        nn.init.xavier_uniform_(self.position_embedding.weight)

        for module in self.transformer_encoder.modules():
            if isinstance(module, nn.Linear):
                # 线性层（注意力/QKV/FFN）：Xavier初始化（论文推荐）
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm：权重=1，偏置=0（论文要求）
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.MultiheadAttention):
                # 多头注意力层：单独初始化QKV权重
                for param in module.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.constant_(param, 0.0)

        nn.init.constant_(self.alpha, 0.5)

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
        全量物品语义ID转特征（适配你原始的semantic_id_to_feat函数）
        Args:
            all_item_indices: (num_items, num_layers) 全量物品的多层次语义ID，如(18425,8)
        Returns:
            all_item_feat: (num_items, hidden_dim) 全量物品的语义特征，如(18425,256)
        """
        # 关键：将(num_items, num_layers)的语义ID拆分为indices_list
        # indices_list是长度为num_layers的列表，每个元素shape=(num_items,)
        indices_list = []
        for layer in range(self.num_layers):
            # 取第layer列的语义ID，shape=(num_items,)
            layer_indices = all_item_indices[:, layer]
            indices_list.append(layer_indices)

        # 调用你原始的semantic_id_to_feat函数
        # 你的函数会自动处理为(num_items, 1, hidden_dim)，之后挤压维度
        all_item_feat = self.semantic_id_to_feat(indices_list)  # (num_items, 1, 256)

        # 挤压多余的seq_len维度：(num_items, 1, 256) → (num_items, 256)
        all_item_feat = all_item_feat.squeeze(1)

        # 最终校验（可选，确保维度正确）
        assert all_item_feat.shape == (all_item_indices.shape[0], self.hidden_dim), \
            f"全量特征维度错误！预期({all_item_indices.shape[0]},{self.hidden_dim})，实际{all_item_feat.shape}"

        return all_item_feat

    def semantic_id_to_feat(self, indices_list):
        """
        核心：将AH-RQ生成的多层次语义ID映射为特征
        Args:
            indices_list: list of tensor，每层语义ID (batch, seq_len)
        Returns:
            semantic_feat: (batch, seq_len, hidden_dim) 语义特征
        """

        semantic_blocks = []
        layer_idx = 0
        # Topic层语义ID映射
        for layer in new_config.semantic_hierarchy["topic"]["layers"]:
            cb_key = f"topic_{layer}"
            indices = indices_list[layer_idx]  # (batch, seq_len)

            cb_size = new_config.semantic_hierarchy["topic"]["codebook_size"]
            if (indices < 0).any() or (indices >= cb_size).any():
                raise ValueError(f"Topic层{layer}的ID超出范围[0, {cb_size - 1}]")

            block_feat = self.semantic_id_emb[cb_key](indices)  # (batch, seq_len, layer_dim)
            semantic_blocks.append(block_feat)
            layer_idx += 1
        # Style层语义ID映射
        for layer in new_config.semantic_hierarchy["style"]["layers"]:
            cb_key = f"style_{layer}"
            indices = indices_list[layer_idx]  # (batch, seq_len)
            block_feat = self.semantic_id_emb[cb_key](indices)  # (batch, seq_len, layer_dim)

            cb_size = new_config.semantic_hierarchy["style"]["codebook_size"]
            if (indices < 0).any() or (indices >= cb_size).any():
                raise ValueError(f"Style层{layer}的ID超出范围[0, {cb_size - 1}]")

            semantic_blocks.append(block_feat)
            layer_idx += 1
        # 拼接所有层特征
        semantic_feat = torch.cat(semantic_blocks, dim=-1)  # (batch, seq_len, hidden_dim)

        if semantic_feat.shape[-1] != self.hidden_dim:
            raise ValueError(f"拼接后维度{semantic_feat.shape[-1]}≠配置维度{self.hidden_dim}")

        # 1. 先激活，再归一化（顺序调换，避免先压缩）
        semantic_feat = F.gelu(semantic_feat)
        # 2. 归一化（保留方向信息）
        semantic_feat = F.normalize(semantic_feat, p=2, dim=-1)
        # 3. 尺度恢复（《Transformer》论文标准操作，提升方差）
        # hidden_dim=256 → √256=16，直接提升方差16倍
        semantic_feat = semantic_feat * (self.hidden_dim ** 0.5)
        # 注意：这里不再添加 dropout，避免训练/评估行为不一致
        # 如果需要 dropout，应该在更上层（如 transformer 输入前）添加
        return semantic_feat


    def forward(self, batch):
        """
        前向传播：多模态特征 → AH-RQ语义ID → 语义特征 → SASRec
        Args:
            batch: dict
                - history_items: (batch, max_len) 原始物品ID序列（仅记录，无核心作用）
                - history_text_feat: (batch, max_len, text_dim) 历史序列文本特征
                - history_vision_feat: (batch, max_len, visual_dim) 历史序列视觉特征
                - target_item: (batch, ) 目标物品ID（仅记录，无核心作用）
                - target_text_feat: (batch, text_dim) 目标正样本文本特征
                - target_vision_feat: (batch, visual_dim) 目标正样本视觉特征
                - negative_items: (batch, num_neg) 负样本物品ID（仅记录，无核心作用）
                - neg_text_feat: (batch, num_neg, text_dim) 负样本文本特征
                - neg_vision_feat: (batch, num_neg, visual_dim) 负样本视觉特征
        Returns:
            user_emb: (batch, hidden_dim) 用户表征
            pos_sem_feat: (batch, hidden_dim) 正样本语义特征
        """
        device = next(self.parameters()).device
        # 历史序列语义ID: (batch, max_len, num_layers) -> list of (batch, max_len)
        hist_indices = batch["history_indices"].to(device)
        history_len = batch["history_len"].to(device)
        # 1. 序列编码（复用核心函数）
        user_emb = self.get_user_embedding(hist_indices, history_len)

        # 2. 正样本特征
        pos_indices = batch["target_indices"]
        pos_indices_list = self._tensor_to_indices_list(pos_indices, seq_len=1)
        pos_sem_feat = self.semantic_id_to_feat(pos_indices_list).squeeze(1)  # (B, D)

        return user_emb, pos_sem_feat

    def get_user_embedding(self, hist_indices, history_len):
        """
        抽取用户表征（100%复用forward中的SASRec序列建模逻辑）
        Args:
            hist_indices: (batch, max_len, num_layers) 历史语义ID
            history_len: (batch,) 历史序列有效长度
        Returns:
            user_emb: (batch, hidden_dim) 用户表征
        """
        # 1. 语义ID转特征
        hist_indices_list = self._tensor_to_indices_list(hist_indices)
        history_sem_feat = self.semantic_id_to_feat(hist_indices_list)  # (B, L, D)
        history_sem_feat = self.dropout_proj(history_sem_feat)

        device = history_sem_feat.device
        # 2. 位置编码
        batch_size = history_sem_feat.shape[0]
        positions = torch.arange(history_sem_feat.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
        history_sem_feat = history_sem_feat + self.position_embedding(positions)

        # 3. 掩码构造
        # 未来掩码
        bool_mask = torch.tril(torch.ones((history_sem_feat.shape[1], history_sem_feat.shape[1]), device=device)).bool()
        mask = torch.zeros_like(bool_mask, dtype=torch.float32).masked_fill(~bool_mask, -1e9)
        # Padding掩码
        max_len = history_sem_feat.shape[1]
        position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        src_key_padding_mask = position_ids >= history_len.unsqueeze(1)

        # 4. Transformer编码
        encoded_seq = self.transformer_encoder(history_sem_feat, mask=mask, src_key_padding_mask=src_key_padding_mask)
        encoded_seq = self.layer_norm(encoded_seq)

        # 5. 取最后有效位生成用户表征
        last_indices = torch.clamp(history_len - 1, min=0)
        user_emb = encoded_seq[torch.arange(batch_size, device=device), last_indices]  # (B, D)

        return user_emb

    def predict_all(self, hist_indices, indices_list, history_len):
        """
        全量物品打分：100%对齐forward中的打分逻辑
        Args:
            hist_indices 历史语义id，形状为 (batch, max_len, num_layers)
            indices_list: 所有物品的语义id，形状为 (item_num, num_layers)
            history_items: (batch, max_len) 历史物品ID（可选，用于确定有效序列长度）
        Returns:
            all_scores: (batch, item_num) 每个用户对所有物品的打分
        """
        self.eval()  # 评估模式

        with torch.no_grad():
            # Step 1: 抽取用户表征（和forward完全一致）
            user_emb = self.get_user_embedding(hist_indices, history_len)  # (batch, hidden_dim)


            # 语义ID → 语义特征（100%复用semantic_id_to_feat，和forward中正样本逻辑一致）
            # 将 indices_list 从 (item_num, num_layers) 转换为 list 格式
            indices_list_tensor = indices_list.unsqueeze(1)  # (item_num, 1, num_layers)
            all_indices_list = self._tensor_to_indices_list(indices_list_tensor, seq_len=1)
            all_sem_feat = self.semantic_id_to_feat(all_indices_list)  # (item_num, 1, hidden_dim)
            all_sem_feat = all_sem_feat.squeeze(1)  # (item_num, hidden_dim)

            # Step 3: 计算用户对所有物品的打分（复用语义ID embedding权重点积，与原生SASRec一致）
            # 扩展维度：(batch, 1, hidden_dim) × (1, item_num, hidden_dim) → (batch, item_num, hidden_dim)
            all_scores = (user_emb.unsqueeze(1) * all_sem_feat.unsqueeze(0)).sum(dim=-1)  # (batch, item_num)

        return all_scores
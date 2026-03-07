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
    def semantic_id_to_feat(self, indices_list):
        """
        核心：将AH-RQ生成的多层次语义ID映射为特征
        Args:
            indices_list: list of tensor，每层语义ID (batch, seq_len)
        Returns:
            semantic_feat: (batch, seq_len, hidden_dim) 语义特征
        """
        # 校验：indices数量必须等于总层数（论文维度均分要求）
        if len(indices_list) != self.num_layers:
            raise ValueError(f"语义ID层数{len(indices_list)}≠配置层数{self.num_layers}（违反分层量化维度均分原则）")


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
            pos_scores: (batch,) 正样本分数
            neg_scores: (batch, num_neg) 负样本分数
            quantized: (batch, hidden_dim) 正样本AH-RQ量化特征
            user_emb: (batch, hidden_dim) 用户表征
            indices_list: list 正样本多层次语义ID
            quantized_layers: list 正样本分层量化结果
        """

        # 历史序列语义ID
        hist_indices = batch["history_indices"]
        # 正样本语义id
        pos_indices = batch["target_indices"].float()
        # 负样本语义id
        neg_indices = batch["neg_indices_list"].float()

        # 负样本量化（展平处理多模态特征）
        neg_batch = neg_indices.shape[0]
        neg_num = neg_indices.shape[1]


        # Step 3: 多层次语义ID → 语义特征（替代原始Embedding）
        history_sem_feat = self.semantic_id_to_feat(hist_indices)  # (batch, max_len, hidden_dim)
        pos_sem_feat = self.semantic_id_to_feat(pos_indices)  # (batch, hidden_dim)
        neg_sem_feat = self.semantic_id_to_feat(neg_indices)  # (batch*num_neg, hidden_dim)
        neg_sem_feat = neg_sem_feat.view(neg_batch, neg_num, self.hidden_dim)  # (batch, num_neg, hidden_dim)

        # Step 4: SASRec序列建模（输入为语义特征，逻辑完全不变）
        batch_size = history_sem_feat.shape[0]
        device = history_sem_feat.device

        # 位置编码
        positions = torch.arange(history_sem_feat.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
        history_sem_feat = history_sem_feat + self.position_embedding(positions)
        assert self.position_embedding.weight.requires_grad, "位置编码必须可训练"


        # 掩码（防止看到未来）
        bool_mask = torch.tril(torch.ones((history_sem_feat.shape[1], history_sem_feat.shape[1]), device=device)).bool()
        # 2. 转换为数值型mask：True→0（保留），False→-1e9（屏蔽未来）
        mask = torch.zeros_like(bool_mask, dtype=torch.float32).masked_fill(~bool_mask, -1e9)

        # Transformer编码（先编码）
        src_key_padding_mask = torch.zeros((batch_size, history_sem_feat.shape[1]), dtype=torch.bool, device=device)
        encoded_seq = self.transformer_encoder(history_sem_feat, mask=mask, src_key_padding_mask=src_key_padding_mask)

        # 新增：输出LayerNorm（与原生SASRec一致）
        encoded_seq = self.layer_norm(encoded_seq)

        # 关键修复：从 batch 中获取真实的 history_len（在 encoded_seq 定义之后）
        history_lens = batch.get("history_len", None)
        if history_lens is not None:
            # 使用数据集中真实的序列长度
            history_lens = history_lens.to(device)
            last_indices = history_lens - 1
            last_indices = torch.clamp(last_indices, min=0)
        else:
            # 备用方案：基于历史物品ID判断（假设0是padding）
            history_items_batch = batch.get("history_items", torch.zeros(batch_size, history_sem_feat.shape[1], dtype=torch.long, device=device))
            non_zero_mask = (history_items_batch != 0)  # 物品ID为0的是padding
            last_indices = non_zero_mask.sum(dim=1) - 1
            last_indices = torch.clamp(last_indices, min=0)

        # 用户表征聚合
        user_emb = encoded_seq[torch.arange(batch_size, device=device), last_indices]  # (batch, hidden_dim)

        # Step 5: 推荐分数计算（复用语义ID embedding权重点积，与原生SASRec一致）
        pos_scores = (user_emb * pos_sem_feat).sum(dim=-1)  # (batch,)

        neg_scores = (user_emb.unsqueeze(1) * neg_sem_feat).sum(dim=-1)  # (batch, num_neg)

        # 返回结果（兼容原有接口）
        return (
            pos_scores, neg_scores, user_emb,
        )

    def get_user_embedding(self, hist_indices, history_items=None):
        """
        抽取用户表征（100%复用forward中的SASRec序列建模逻辑）
        Args:
            hist_indices 历史语义id
            history_items: (batch, max_len) 历史物品ID（可选，用于确定有效序列长度）
        Returns:
            user_emb: (batch, hidden_dim) 用户表征
        """
        self.eval()  # 评估模式，关闭Dropout/EMA
        with torch.no_grad():
            # Step 2: 语义ID → 语义特征（100%复用forward中的semantic_id_to_feat处理）
            # 注意：semantic_id_to_feat 内部已经包含了 gelu → normalize → scale → dropout
            history_sem_feat = self.semantic_id_to_feat(hist_indices)  # (batch, max_len, hidden_dim)

            # Step 3: 位置编码（和forward完全一致）
            batch_size = history_sem_feat.shape[0]
            device = history_sem_feat.device
            positions = torch.arange(history_sem_feat.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
            history_sem_feat = history_sem_feat + self.position_embedding(positions)

            # Step 4: Transformer编码（和forward完全一致）
            bool_mask = torch.tril(
                torch.ones((history_sem_feat.shape[1], history_sem_feat.shape[1]), device=device)).bool()
            mask = torch.zeros_like(bool_mask, dtype=torch.float32).masked_fill(~bool_mask, -1e9)
            encoded_seq = self.transformer_encoder(history_sem_feat, mask=mask)
            encoded_seq = self.layer_norm(encoded_seq)

            # Step 5: 聚合用户表征（关键修复：基于history_items确定有效序列长度）
            if history_items is not None:
                # 使用真实的序列长度
                non_zero_mask = (history_items != 0)  # 物品ID为0的是padding
                last_indices = non_zero_mask.sum(dim=1) - 1
                last_indices = torch.clamp(last_indices, min=0)
            else:
                # 备用方案：基于历史特征判断
                non_zero_mask = (history_sem_feat.sum(dim=-1) != 0)
                last_indices = non_zero_mask.sum(dim=1) - 1
                last_indices = torch.clamp(last_indices, min=0)
            user_emb = encoded_seq[torch.arange(batch_size, device=device), last_indices]  # (batch, hidden_dim)

            return user_emb

    def predict_all(self, hist_indices, indices_list, history_items=None):
        """
        全量物品打分：100%对齐forward中的打分逻辑
        Args:
            hist_indices 历史语义id
            indices_list: 所有语义id
            history_items: (batch, max_len) 历史物品ID（可选，用于确定有效序列长度）
        Returns:
            all_scores: (batch, item_num) 每个用户对所有物品的打分
        """
        self.eval()  # 评估模式
        device = hist_indices.device
        batch_size = hist_indices.shape[0]
        item_num = indices_list.shape[0]

        with torch.no_grad():
            # Step 1: 抽取用户表征（和forward完全一致）
            user_emb = self.get_user_embedding(hist_indices, history_items)  # (batch, hidden_dim)


            # 语义ID → 语义特征（100%复用semantic_id_to_feat，和forward中正样本逻辑一致）
            # 注意：semantic_id_to_feat 内部已经包含了完整的特征处理（gelu → normalize → scale → dropout）
            all_sem_feat = self.semantic_id_to_feat(indices_list)  # (item_num, hidden_dim)

            # Step 3: 计算用户对所有物品的打分（复用语义ID embedding权重点积，与原生SASRec一致）
            # 扩展维度：(batch, 1, hidden_dim) × (1, item_num, hidden_dim) → (batch, item_num, hidden_dim)
            all_scores = (user_emb.unsqueeze(1) * all_sem_feat.unsqueeze(0)).sum(dim=-1)  # (batch, item_num)

        return all_scores
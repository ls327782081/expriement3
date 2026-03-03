import torch
import torch.nn as nn
from config import new_config
from our_models.ah_rq import AdaptiveHierarchicalQuantizer  # 升级后的AH-RQ


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
    

        # 2. AH-RQ量化器（单模态，生成多层次语义ID）
        self.ahrq = AdaptiveHierarchicalQuantizer(
            hidden_dim=new_config.pmat_hidden_dim,
            semantic_hierarchy=new_config.semantic_hierarchy,
            use_multimodal=True,  
            text_dim=new_config.text_dim,
            visual_dim=new_config.visual_dim,
            beta=new_config.ahrq_beta,
            use_ema=new_config.ahrq_use_ema,
            ema_decay=0.99,
            reset_unused_codes=new_config.ahrq_reset_unused_codes,
            reset_threshold=new_config.ahrq_reset_threshold
        )

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
        self.num_layers = new_config.sasrec_num_layers
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
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 推荐打分层（保留）
        self.score_layer = nn.Linear(self.hidden_dim, 1)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
            block_feat = self.semantic_id_emb[cb_key](indices)  # (batch, seq_len, layer_dim)
            semantic_blocks.append(block_feat)
            layer_idx += 1
        # Style层语义ID映射
        for layer in new_config.semantic_hierarchy["style"]["layers"]:
            cb_key = f"style_{layer}"
            indices = indices_list[layer_idx]  # (batch, seq_len)
            block_feat = self.semantic_id_emb[cb_key](indices)  # (batch, seq_len, layer_dim)
            semantic_blocks.append(block_feat)
            layer_idx += 1
        # 拼接所有层特征
        semantic_feat = torch.cat(semantic_blocks, dim=-1)  # (batch, seq_len, hidden_dim)
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
        # Step 1: 提取多模态特征（直接使用传入的文本/视觉特征，不再从ID加载）
        # 历史序列多模态特征
        history_text = batch["history_text_feat"].float()  # (batch, max_len, text_dim)
        history_vision = batch["history_vision_feat"].float()  # (batch, max_len, visual_dim)
        # 正样本多模态特征
        target_text = batch["target_text_feat"].float()  # (batch, text_dim)
        target_vision = batch["target_vision_feat"].float()  # (batch, visual_dim)
        # 负样本多模态特征
        neg_text = batch["neg_text_feat"].float()  # (batch, num_neg, text_dim)
        neg_vision = batch["neg_vision_feat"].float()  # (batch, num_neg, visual_dim)

        # Step 2: AH-RQ量化（启用多模态对齐，直接处理文本+视觉特征）
        # 历史序列量化（多模态输入）
        hist_quantized, hist_indices, hist_quant_layers = self.ahrq(history_text, history_vision)
        # 正样本量化（多模态输入）
        pos_quantized, pos_indices, pos_quant_layers = self.ahrq(target_text, target_vision)
        # 负样本量化（展平处理多模态特征）
        neg_batch = neg_text.shape[0]
        neg_num = neg_text.shape[1]
        # 展平负样本特征
        neg_text_flat = neg_text.view(-1, neg_text.size(-1))  # (batch*num_neg, text_dim)
        neg_vision_flat = neg_vision.view(-1, neg_vision.size(-1))  # (batch*num_neg, visual_dim)
        # AH-RQ量化展平的负样本
        neg_quantized_flat, neg_indices, neg_quant_layers = self.ahrq(neg_text_flat, neg_vision_flat)
        # 恢复维度
        neg_quantized = neg_quantized_flat.view(neg_batch, neg_num, self.hidden_dim)  # (batch, num_neg, hidden_dim)

        # Step 3: 多层次语义ID → 语义特征（替代原始Embedding）
        history_sem_feat = self.semantic_id_to_feat(hist_indices)  # (batch, max_len, hidden_dim)
        pos_sem_feat = self.semantic_id_to_feat(pos_indices)  # (batch, hidden_dim)
        neg_sem_feat = self.semantic_id_to_feat(neg_indices)  # (batch*num_neg, hidden_dim)
        neg_sem_feat = neg_sem_feat.view(neg_batch, neg_num, self.hidden_dim)  # (batch, num_neg, hidden_dim)

        # Step 4: SASRec序列建模（输入为语义特征，逻辑完全不变）
        batch_size = history_sem_feat.shape[0]
        device = history_sem_feat.device

        # 位置编码
        positions = torch.arange(self.max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        history_sem_feat = history_sem_feat + self.position_embedding(positions)

        # 掩码（防止看到未来）
        mask = torch.tril(torch.ones((self.max_len, self.max_len), device=device)).bool()

        # Transformer编码
        encoded_seq = self.transformer_encoder(history_sem_feat, mask=mask)

        # 用户表征聚合
        non_zero_mask = (history_sem_feat.sum(dim=-1) != 0)  # 非padding掩码
        last_indices = non_zero_mask.sum(dim=1) - 1
        last_indices = torch.clamp(last_indices, min=0)
        user_emb = encoded_seq[torch.arange(batch_size), last_indices]  # (batch, hidden_dim)

        # Step 5: 推荐分数计算（逻辑完全不变）
        pos_interaction = user_emb * pos_sem_feat
        pos_scores = self.score_layer(pos_interaction).squeeze(-1)  # (batch,)

        neg_interaction = user_emb.unsqueeze(1) * neg_sem_feat
        neg_scores = self.score_layer(neg_interaction).squeeze(-1)  # (batch, num_neg)

        # 返回结果（兼容原有接口）
        return (
            pos_scores, neg_scores,
            pos_quantized, user_emb,
            pos_indices, pos_quant_layers
        )
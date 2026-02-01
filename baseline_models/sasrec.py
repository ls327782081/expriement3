import torch
import torch.nn as nn

class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation

    Args:
        num_items: 物品总数
        max_seq_len: 最大序列长度
        hidden_size: 隐藏层维度
        num_blocks: Transformer块数量
        num_heads: 注意力头数量
        dropout_rate: Dropout率
        initializer_range: 权重初始化范围
        use_causal_mask: 是否使用因果掩码（默认True，符合原始SASRec论文）
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_size: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
        initializer_range: float = 0.02,
        use_causal_mask: bool = True,
        **kwargs
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.use_causal_mask = use_causal_mask

        # Embedding layers
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])

        # LayerNorm before transformer (Pre-LN architecture)
        self.input_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        # 初始化权重以防止NaN
        self._init_weights()

        # 标记已初始化（防止被training_utils覆盖）
        self._weights_initialized = True

        # 缓存因果掩码以提高效率
        self._causal_mask_cache = {}

    def _init_weights(self):
        """初始化模型权重 - 参考RecBole实现"""
        # 使用normal初始化，std=0.02（与RecBole一致）
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=self.initializer_range)

        # 初始化LayerNorm
        nn.init.constant_(self.input_layer_norm.weight, 1.0)
        nn.init.constant_(self.input_layer_norm.bias, 0.0)

        # 初始化所有子模块（TransformerBlock中的Linear和LayerNorm）
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """获取因果掩码（带缓存）

        Args:
            seq_len: 序列长度
            device: 设备

        Returns:
            causal_mask: [seq_len, seq_len] 因果掩码，上三角为-inf
        """
        cache_key = (seq_len, device)
        if cache_key not in self._causal_mask_cache:
            # 创建因果掩码：上三角矩阵（不包括对角线）设为-inf
            # 这样每个位置只能看到自己和之前的位置
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )
            self._causal_mask_cache[cache_key] = causal_mask
        return self._causal_mask_cache[cache_key]

    def forward(self, seq_items, seq_lens=None):
        """前向传播

        Args:
            seq_items: [batch_size, seq_len] 序列物品ID
            seq_lens: 可选，序列长度

        Returns:
            seq_emb: [batch_size, seq_len, hidden_size] 序列表示
        """
        # seq_items: [batch_size, seq_len]
        batch_size, seq_len = seq_items.shape

        # Position embeddings
        positions = torch.arange(seq_len, device=seq_items.device).unsqueeze(0).expand(batch_size, -1)

        # Item + position embeddings
        seq_emb = self.item_emb(seq_items) + self.pos_emb(positions)

        # 关键修复：先LayerNorm再dropout
        seq_emb = self.input_layer_norm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        # Create padding mask (2D for MultiheadAttention)
        padding_mask = (seq_items == 0)  # True for padding tokens

        # Create causal mask if enabled
        causal_mask = None
        if self.use_causal_mask:
            causal_mask = self._get_causal_mask(seq_len, seq_items.device)

        # Apply transformer blocks
        for block in self.blocks:
            seq_emb = block(seq_emb, padding_mask=padding_mask, causal_mask=causal_mask)

        return seq_emb

    def predict(self, seq_items, seq_lens=None):
        """
        预测下一个物品（全排序）

        Args:
            seq_items: [batch_size, seq_len] 序列物品ID
            seq_lens: 可选，序列长度

        Returns:
            logits: [batch_size, num_items+1] 对所有物品的预测分数
        """
        # 调用forward获取序列表示
        seq_output = self.forward(seq_items, seq_lens)  # [batch_size, seq_len, hidden_size]

        # 修复：使用seq_lens获取最后一个有效位置的表示（与GRU4Rec一致）
        if seq_lens is not None:
            # seq_lens - 1 是最后一个有效位置的索引
            batch_size = seq_output.size(0)
            seq_lens_tensor = seq_lens if isinstance(seq_lens, torch.Tensor) else torch.tensor(seq_lens, device=seq_output.device)
            seq_lens_tensor = seq_lens_tensor.long() - 1  # 转换为索引
            seq_lens_tensor = torch.clamp(seq_lens_tensor, 0, seq_output.size(1) - 1)

            # 使用gather获取每个样本的最后一个有效位置
            seq_emb = seq_output[torch.arange(batch_size, device=seq_output.device), seq_lens_tensor]  # [batch_size, hidden_size]
        else:
            # 如果没有seq_lens，使用最后一个位置
            seq_emb = seq_output[:, -1, :]  # [batch_size, hidden_size]

        # 计算与所有物品的相似度
        # 使用完整的embedding权重（与RecBole的full_sort_predict一致）
        all_item_emb = self.item_emb.weight  # [num_items+1, hidden_size]
        logits = torch.matmul(seq_emb, all_item_emb.transpose(0, 1))  # [batch_size, num_items+1]

        return logits

class TransformerBlock(nn.Module):
    """Transformer Block for SASRec - Pre-LN架构（更稳定）

    支持因果掩码和填充掩码的组合使用
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float, **kwargs):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, padding_mask=None, causal_mask=None):
        """前向传播

        Args:
            x: [batch_size, seq_len, hidden_size] 输入序列
            padding_mask: [batch_size, seq_len] 填充掩码，True表示padding位置
            causal_mask: [seq_len, seq_len] 因果掩码，上三角为-inf

        Returns:
            x: [batch_size, seq_len, hidden_size] 输出序列
        """
        # Pre-LN架构：先LayerNorm再attention（更稳定，防止梯度爆炸）
        # Self-attention with residual connection
        normed_x = self.layer_norm1(x)

        # 同时使用因果掩码和填充掩码
        # attn_mask: [seq_len, seq_len] 用于因果掩码
        # key_padding_mask: [batch_size, seq_len] 用于填充掩码
        attn_output, _ = self.attention(
            normed_x, normed_x, normed_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask
        )
        x = x + self.dropout(attn_output)

        # Feed forward with residual connection
        normed_x = self.layer_norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + ff_output

        return x
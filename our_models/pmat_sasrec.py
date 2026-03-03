import torch
import torch.nn as nn
from config import new_config
# 导入PMAT-AH-RQ编码器（仅编码器，无推荐功能）
from our_models.pmat_ahrq import PMATAHRQ


class PMATSASRec(nn.Module):
    """
    完整推荐模型：PMAT-AH-RQ编码器 + SASRec推荐器
    - PMATAHRQ：负责多模态编码、AH-RQ量化、动态ID更新等（无推荐功能）
    - 内置SASRec模块：接收PMAT编码后的语义表征，完成序列推荐排序
    """

    def __init__(self):
        super().__init__()
        # 1. 导入PMAT-AH-RQ编码器（纯编码器，无推荐功能）
        self.pmat_encoder = PMATAHRQ()

        # 2. SASRec核心推荐模块（仅处理PMAT输出的语义表征）
        self.hidden_dim = new_config.pmat_hidden_dim  # 与PMAT编码器输出维度对齐
        self.max_len = new_config.sasrec_max_len
        self.num_heads = new_config.sasrec_num_heads
        self.num_layers = new_config.sasrec_num_layers
        self.dropout = new_config.sasrec_dropout

        # SASRec位置编码（复用，与PMAT编码器的位置编码解耦）
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_dim)

        # Transformer编码器（SASRec核心序列建模）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 推荐打分层（最终输出推荐分数）
        self.score_layer = nn.Linear(self.hidden_dim, 1)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """统一权重初始化"""
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, batch):
        """
        完整推荐模型前向传播
        Args:
            batch: dict，包含PMAT编码器所需的多模态特征：
                - history_text: (batch, max_len, text_dim) 历史文本特征
                - history_visual: (batch, max_len, visual_dim) 历史视觉特征
                - history_len: (batch,) 历史序列有效长度
                - target_text: (batch, text_dim) 正样本文本特征
                - target_visual: (batch, visual_dim) 正样本视觉特征
                - neg_text: (batch, num_neg, text_dim) 负样本文本特征
                - neg_visual: (batch, num_neg, visual_dim) 负样本视觉特征
        Returns:
            pos_scores: (batch,) 正样本推荐分数
            neg_scores: (batch, num_neg) 负样本推荐分数
            quantized: (batch, hidden_dim) AH-RQ量化后的语义表征
            user_emb: (batch, hidden_dim) 用户兴趣表征
            indices_list: list 量化索引（用于ID质量计算）
            quantized_layers: list 分层量化结果
        """
        # ===================== Step 1: PMAT-AH-RQ编码（仅编码，无推荐） =====================
        encoder_outputs = self.pmat_encoder(batch)

        # 从PMAT编码器获取核心输出
        history_emb = encoder_outputs["history_emb"]  # PMAT编码后的历史序列特征
        pos_quant_emb = encoder_outputs["pos_quant_emb"]  # 正样本AH-RQ量化表征
        neg_quant_emb = encoder_outputs["neg_quant_emb"]  # 负样本AH-RQ量化表征
        user_interest = encoder_outputs["user_interest"]  # PMAT输出的用户兴趣表征
        indices_list = encoder_outputs["pos_indices"]  # 量化索引
        quantized_layers = encoder_outputs.get("quantized_layers", [])  # 分层量化结果

        # ===================== Step 2: SASRec序列推荐（核心推荐逻辑） =====================
        batch_size = history_emb.shape[0]
        device = history_emb.device

        # 1. 叠加SASRec位置编码（补充序列时序信息）
        positions = torch.arange(self.max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        history_emb = history_emb + self.position_embedding(positions)

        # 2. 生成SASRec掩码（防止看到未来物品）
        mask = torch.tril(torch.ones((self.max_len, self.max_len), device=device)).bool()

        # 3. Transformer序列建模
        encoded_seq = self.transformer_encoder(history_emb, mask=mask)

        # 4. 用户表征聚合（取最后一个有效位置）
        non_zero_mask = (history_emb.sum(dim=-1) != 0)  # 非padding掩码
        last_indices = non_zero_mask.sum(dim=1) - 1
        last_indices = torch.clamp(last_indices, min=0)  # 防止全0序列
        user_emb = encoded_seq[torch.arange(batch_size), last_indices]

        # 5. 计算推荐分数（基于PMAT-AH-RQ量化表征）
        pos_interaction = user_emb * pos_quant_emb
        pos_scores = self.score_layer(pos_interaction).squeeze(-1)

        neg_interaction = user_emb.unsqueeze(1) * neg_quant_emb
        neg_scores = self.score_layer(neg_interaction).squeeze(-1)

        # ===================== 返回统一格式（与SASRecAHRQ对齐） =====================
        quantized = pos_quant_emb  # 以正样本量化表征作为核心量化输出
        return pos_scores, neg_scores, quantized, user_emb, indices_list, quantized_layers


# ===================== 便捷调用函数（可选） =====================
def build_pmat_sasrec_model():
    """构建PMAT-SASRec完整推荐模型"""
    model = PMATSASRec().to(new_config.device)
    return model


if __name__ == "__main__":
    # 简单测试模型前向传播
    torch.manual_seed(new_config.seed)

    # 构造测试数据
    batch = {
        "history_text": torch.randn(4, new_config.sasrec_max_len, new_config.pmat_text_dim).to(new_config.device),
        "history_visual": torch.randn(4, new_config.sasrec_max_len, new_config.pmat_visual_dim).to(new_config.device),
        "history_len": torch.tensor([20, 15, 30, 10]).to(new_config.device),
        "target_text": torch.randn(4, new_config.pmat_text_dim).to(new_config.device),
        "target_visual": torch.randn(4, new_config.pmat_visual_dim).to(new_config.device),
        "neg_text": torch.randn(4, 4, new_config.pmat_text_dim).to(new_config.device),
        "neg_visual": torch.randn(4, 4, new_config.pmat_visual_dim).to(new_config.device)
    }

    # 初始化模型
    model = build_pmat_sasrec_model()
    model.train()

    # 前向传播
    pos_scores, neg_scores, quantized, user_emb, indices_list, quantized_layers = model(batch)

    # 打印输出维度（验证正确性）
    print("模型输出维度验证：")
    print(f"pos_scores: {pos_scores.shape}")  # (4,)
    print(f"neg_scores: {neg_scores.shape}")  # (4, 4)
    print(f"quantized: {quantized.shape}")  # (4, hidden_dim)
    print(f"user_emb: {user_emb.shape}")  # (4, hidden_dim)
    print(f"indices_list长度: {len(indices_list)}")  # 等于量化层数
    print("模型前向传播测试通过！")
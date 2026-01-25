# DuoRec模型 - 用于Pctx第一阶段训练：生成个性化语义ID
# 基于原始Pctx论文的两阶段训练流程
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseModel  # 导入抽象基类
from config import config


class DuoRec(BaseModel):
    """
    DuoRec: Dual-Channel Recommendation Framework
    基于原始论文的双通道推荐模型
    """
    def __init__(self):
        super(DuoRec, self).__init__()  # 调用父类初始化
        # 用户和物品嵌入
        self.user_emb = nn.Embedding(config.user_vocab_size, config.hidden_dim)
        self.item_emb = nn.Embedding(config.item_vocab_size, config.hidden_dim)
        
        # 多模态编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(768, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.vision_encoder = nn.Sequential(
            nn.Linear(512, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 用户序列建模 - 使用Transformer
        self.user_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.attention_heads,
                dim_feedforward=config.mlp_dim,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # 语义ID生成器
        self.semantic_id_generator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.codebook_size * config.id_length)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, batch):
        """
        前向传播
        
        Args:
            batch: 包含以下键的字典
                - user_id: 用户ID张量
                - item_id: 物品ID张量
                - user_seq: 用户历史序列 (batch_size, seq_len)
                - text_feat: 文本特征 (batch_size, 768)
                - vision_feat: 视觉特征 (batch_size, 512)
                
        Returns:
            logits: 语义ID的logits (batch_size, id_length, codebook_size)
            user_repr: 用户表示 (batch_size, hidden_dim)
            item_repr: 物品表示 (batch_size, hidden_dim)
        """
        # 获取基础嵌入
        user_emb = self.user_emb(batch["user_id"])  # (batch_size, hidden_dim)
        item_emb = self.item_emb(batch["item_id"])  # (batch_size, hidden_dim)
        
        # 多模态特征编码
        text_emb = self.text_encoder(batch["text_feat"].float())  # (batch_size, hidden_dim)
        vision_emb = self.vision_encoder(batch["vision_feat"].float())  # (batch_size, hidden_dim)
        
        # 融合物品表示
        item_repr = item_emb + text_emb + vision_emb  # (batch_size, hidden_dim)
        
        # 用户序列建模
        if "user_seq" in batch:
            # 获取序列中的物品嵌入
            seq_item_emb = self.item_emb(batch["user_seq"])  # (batch_size, seq_len, hidden_dim)
            
            # 通过Transformer编码用户序列
            user_seq_repr = self.user_transformer(seq_item_emb)  # (batch_size, seq_len, hidden_dim)
            
            # 池化得到用户表示 (使用平均池化)
            user_repr = torch.mean(user_seq_repr, dim=1)  # (batch_size, hidden_dim)
        else:
            # 如果没有序列信息，直接使用用户嵌入
            user_repr = user_emb
        
        # 生成个性化语义ID
        user_item_concat = torch.cat([user_repr, item_repr], dim=1)  # (batch_size, hidden_dim*2)
        logits = self.semantic_id_generator(user_item_concat)  # (batch_size, id_length*codebook_size)
        logits = logits.reshape(-1, config.id_length, config.codebook_size)  # (batch_size, id_length, codebook_size)
        
        return logits, user_repr, item_repr
    
    def train_step(self, batch, optimizer, criterion, device):
        """
        单步训练方法
        
        Args:
            batch: 训练批次数据
            optimizer: 优化器
            criterion: 损失函数
            device: 计算设备
            
        Returns:
            loss: 损失值
        """
        # 移动数据到设备
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        logits, _, _ = self.forward(batch)
        
        # 计算损失
        target = torch.randint(0, config.codebook_size, (logits.size(0), config.id_length)).to(device)
        loss = 0
        for i in range(config.id_length):
            loss += criterion(logits[:, i, :], target[:, i])
        loss /= config.id_length
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def predict(self, batch, **kwargs):
        """
        预测方法
        
        Args:
            batch: 预测批次数据
            **kwargs: 其他参数
            
        Returns:
            predictions: 预测结果
        """
        # 生成语义ID
        semantic_ids, _, _ = self.forward(batch)
        
        # 获取最可能的ID
        predictions = torch.argmax(semantic_ids, dim=-1)  # (batch_size, id_length)
        
        return predictions
    
    def generate_semantic_ids(self, batch):
        """
        为物品生成个性化语义ID
        
        Args:
            batch: 包含用户和物品信息的字典
            
        Returns:
            semantic_ids: 生成的语义ID (batch_size, id_length)
        """
        logits, _, _ = self.forward(batch)
        
        # 使用argmax获取每个位置的token
        semantic_ids = torch.argmax(logits, dim=-1)  # (batch_size, id_length)
        
        return semantic_ids
"""
MMQ: Multi-Modal Quantization
基于原始论文的多模态量化推荐模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseModel  # 导入抽象基类
from config import config


class MMExpert(nn.Module):
    """多模态专家模块"""
    def __init__(self, input_dim, hidden_dim, expert_type='shared'):
        super().__init__()
        self.expert_type = expert_type
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate_network = nn.Linear(input_dim, 1) if expert_type == 'specific' else None

    def forward(self, x):
        expert_out = self.network(x)
        if self.gate_network is not None:
            gate = torch.sigmoid(self.gate_network(x))
            return gate * expert_out
        return expert_out


class MMQ(BaseModel):
    """
    MMQ: 多模态混合量化模型
    使用多专家架构和语义感知量化器生成离散语义ID
    """
    def __init__(self):
        super(MMQ, self).__init__()
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
        
        # 多专家架构：模态共享和模态特定专家
        self.shared_expert = MMExpert(config.hidden_dim, config.hidden_dim, 'shared')
        self.text_specific_expert = MMExpert(config.hidden_dim, config.hidden_dim, 'specific')
        self.vision_specific_expert = MMExpert(config.hidden_dim, config.hidden_dim, 'specific')
        
        # 门控网络用于模态特定专家
        self.text_gate = nn.Linear(config.hidden_dim, 1)
        self.vision_gate = nn.Linear(config.hidden_dim, 1)
        
        # 语义感知量化器
        self.vq_input = nn.Linear(config.hidden_dim * 2, config.hidden_dim)  # 用户+物品
        self.vq_output = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # 语义ID生成器
        self.semantic_id_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.codebook_size * config.id_length)
        )
        
        # 代码本 (codebook)
        self.codebook = nn.Embedding(config.codebook_size, config.hidden_dim)
        
        # 正交正则化参数
        self.orthogonal_reg_weight = 0.01
        
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
                - text_feat: 文本特征 (batch_size, 768)
                - vision_feat: 视觉特征 (batch_size, 512)
                
        Returns:
            logits: 语义ID的logits (batch_size, id_length, codebook_size)
        """
        # 获取基础嵌入
        user_emb = self.user_emb(batch["user_id"])  # (batch_size, hidden_dim)
        item_emb = self.item_emb(batch["item_id"])  # (batch_size, hidden_dim)
        
        # 多模态特征编码
        text_emb = self.text_encoder(batch["text_feat"].float())  # (batch_size, hidden_dim)
        vision_emb = self.vision_encoder(batch["vision_feat"].float())  # (batch_size, hidden_dim)
        
        # 多专家融合
        # 模态共享专家
        shared_text = self.shared_expert(text_emb)
        shared_vision = self.shared_expert(vision_emb)
        shared_fused = (shared_text + shared_vision) / 2  # 直接相加融合
        
        # 模态特定专家
        text_specific = self.text_specific_expert(text_emb)
        vision_specific = self.vision_specific_expert(vision_emb)
        
        # 门控机制
        text_gate = torch.sigmoid(self.text_gate(text_emb))  # (batch_size, 1)
        vision_gate = torch.sigmoid(self.vision_gate(vision_emb))  # (batch_size, 1)
        
        text_fused = text_gate * text_specific
        vision_fused = vision_gate * vision_specific
        
        # 最终融合表示
        fused_repr = torch.cat([user_emb, item_emb], dim=-1)  # (batch_size, hidden_dim*2)
        latent = self.vq_input(fused_repr)  # (batch_size, hidden_dim)
        
        # 融合多专家输出
        expert_output = shared_fused + text_fused + vision_fused  # 融合专家输出
        latent = latent + expert_output  # 添加专家输出到潜在表示

        # 语义感知量化：使用余弦相似度而非L2距离
        semantic_ids_logits = []
        residual = latent

        for i in range(config.id_length):
            # 计算余弦相似度
            # 归一化潜在向量和码本向量
            normalized_latent = F.normalize(residual.unsqueeze(1), p=2, dim=-1)  # (batch_size, 1, hidden_dim)
            normalized_codebook = F.normalize(self.codebook.weight.unsqueeze(0), p=2, dim=-1)  # (1, codebook_size, hidden_dim)
            
            # 计算余弦相似度
            cosine_similarities = torch.bmm(normalized_latent, normalized_codebook.transpose(1, 2)).squeeze(1)  # (batch_size, codebook_size)
            
            # 使用余弦相似度作为logits（相似度越高，logit越大）
            logits = cosine_similarities  # (batch_size, codebook_size)
            semantic_ids_logits.append(logits)
            
            # 选择最相似的码
            ids = torch.argmax(cosine_similarities, dim=-1)  # (batch_size,)
            
            # 获取量化嵌入
            quantized = self.codebook(ids)  # (batch_size, hidden_dim)
            
            # 计算残差
            residual = residual - quantized
        
        # 堆叠logits: (batch, id_length, codebook_size)
        semantic_ids_logits = torch.stack(semantic_ids_logits, dim=1)
        
        return semantic_ids_logits
    
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
        logits = self.forward(batch)
        
        # 生成目标语义ID
        target = torch.randint(0, config.codebook_size, (logits.size(0), config.id_length)).to(device)
        
        # 计算主损失
        loss = 0
        for i in range(config.id_length):
            loss += criterion(logits[:, i, :], target[:, i])
        loss /= config.id_length
        
        # 添加正交正则化损失（针对专家网络权重）
        orthogonal_loss = 0
        if hasattr(self, 'shared_expert') and hasattr(self.text_specific_expert, 'network'):
            # 计算专家网络权重的正交正则化
            shared_weights = self.shared_expert.network[0].weight
            text_weights = self.text_specific_expert.network[0].weight
            vision_weights = self.vision_specific_expert.network[0].weight
            
            # 计算权重矩阵之间的正交性损失
            shared_text_cos = F.cosine_similarity(
                shared_weights.view(-1), text_weights.view(-1), dim=0
            )
            shared_vision_cos = F.cosine_similarity(
                shared_weights.view(-1), vision_weights.view(-1), dim=0
            )
            text_vision_cos = F.cosine_similarity(
                text_weights.view(-1), vision_weights.view(-1), dim=0
            )
            
            orthogonal_loss = (shared_text_cos.pow(2) + shared_vision_cos.pow(2) + text_vision_cos.pow(2)) * self.orthogonal_reg_weight

        total_loss = loss + orthogonal_loss
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
    
    def predict(self, batch, **kwargs):
        """
        预测方法
        
        Args:
            batch: 预测批次数据
            **kwargs: 其他参数
            
        Returns:
            predictions: 预测结果
        """
        # 前向传播获取logits
        logits = self.forward(batch)
        
        # 获取最可能的语义ID
        predictions = torch.argmax(logits, dim=-1)  # (batch_size, id_length)
        
        return predictions
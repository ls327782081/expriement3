"""
MMQ: Multi-Modal Quantization
基于原始论文的多模态量化推荐模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import AbstractTrainableModel  # 导入抽象基类
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


class MMQ(AbstractTrainableModel):
    """
    MMQ: 多模态混合量化模型
    使用多专家架构和语义感知量化器生成离散语义ID
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(MMQ, self).__init__(device)
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

        # 优化器缓存
        self._optimizers = {}
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

    def _get_optimizer(self, stage_id: int, stage_kwargs: dict) -> torch.optim.Optimizer:
        """获取指定阶段的优化器"""
        lr = stage_kwargs.get('lr', 0.001)
        return torch.optim.Adam(self.parameters(), lr=lr)

    def _get_optimizer_state_dict(self) -> dict:
        """获取当前阶段优化器的状态字典"""
        optimizer_states = {}
        for stage_id, optimizer in self._optimizers.items():
            optimizer_states[stage_id] = optimizer.state_dict()
        return optimizer_states

    def _load_optimizer_state_dict(self, state_dict: dict):
        """加载当前阶段优化器的状态字典"""
        for stage_id, opt_state in state_dict.items():
            if stage_id in self._optimizers:
                self._optimizers[stage_id].load_state_dict(opt_state)

    def _train_one_batch(self, batch: any, stage_id: int, stage_kwargs: dict) -> tuple:
        """
        单batch训练逻辑
        Args:
            batch: 训练批次数据
            stage_id: 阶段ID
            stage_kwargs: 该阶段的自定义参数
        Returns:
            (batch_loss, batch_metrics)
        """
        # 移动数据到设备
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # 前向传播
        logits = self.forward(batch)
        
        # 生成目标语义ID
        target = torch.randint(0, config.codebook_size, (logits.size(0), config.id_length)).to(self.device)
        
        # 计算主损失
        criterion = torch.nn.CrossEntropyLoss()
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
        
        # 计算指标
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == target).float().sum()
        accuracy = correct / (target.size(0) * target.size(1))
        metrics = {'accuracy': accuracy}
        
        return total_loss, metrics

    def _validate_one_epoch(self, val_dataloader: torch.utils.data.DataLoader, stage_id: int, stage_kwargs: dict) -> dict:
        """单轮验证逻辑"""
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                # 移动数据到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # 前向传播
                logits = self.forward(batch)

                # 生成目标语义ID
                target = torch.randint(0, config.codebook_size, (logits.size(0), config.id_length)).to(self.device)

                # 计算损失
                criterion = torch.nn.CrossEntropyLoss()
                loss = 0
                for i in range(config.id_length):
                    loss += criterion(logits[:, i, :], target[:, i])
                loss /= config.id_length

                # 添加正交正则化损失
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

                total_loss += (loss.item() + orthogonal_loss)

                # 计算准确率
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == target).float().sum()
                total_correct += correct.item()
                total_samples += target.size(0) * target.size(1)

        avg_loss = total_loss / len(val_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {'loss': avg_loss, 'accuracy': accuracy}
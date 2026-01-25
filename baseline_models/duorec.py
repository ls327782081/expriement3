# DuoRec模型 - 用于Pctx第一阶段训练：生成个性化语义ID
# 基于原始Pctx论文的两阶段训练流程
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import AbstractTrainableModel  # 导入抽象基类
from config import config


class DuoRec(AbstractTrainableModel):
    """
    DuoRec: Dual-Channel Recommendation Framework
    基于原始论文的双通道推荐模型
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(DuoRec, self).__init__(device)  # 调用父类初始化
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
        logits, _, _ = self.forward(batch)
        
        # 生成目标语义ID
        target = torch.randint(0, config.codebook_size, (logits.size(0), config.id_length)).to(self.device)
        
        # 计算损失
        criterion = torch.nn.CrossEntropyLoss()
        loss = 0
        for i in range(config.id_length):
            loss += criterion(logits[:, i, :], target[:, i])
        loss /= config.id_length
        
        # 计算指标
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == target).float().sum()
        accuracy = correct / (target.size(0) * target.size(1))
        metrics = {'accuracy': accuracy}
        
        return loss, metrics

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
                logits, _, _ = self.forward(batch)

                # 生成目标语义ID
                target = torch.randint(0, config.codebook_size, (logits.size(0), config.id_length)).to(self.device)

                # 计算损失
                criterion = torch.nn.CrossEntropyLoss()
                loss = 0
                for i in range(config.id_length):
                    loss += criterion(logits[:, i, :], target[:, i])
                loss /= config.id_length

                total_loss += loss.item()

                # 计算准确率
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == target).float().sum()
                total_correct += correct.item()
                total_samples += target.size(0) * target.size(1)

        avg_loss = total_loss / len(val_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {'loss': avg_loss, 'accuracy': accuracy}
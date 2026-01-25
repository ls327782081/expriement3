
from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class BaseModel(nn.Module, ABC):
    """
    抽象模型基类，定义统一的训练和预测接口
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, batch):
        """
        前向传播方法，所有子类必须实现
        """
        pass

    @abstractmethod
    def train_step(self, batch, optimizer, criterion, device):
        """
        单步训练方法，返回损失值
        """
        pass

    @abstractmethod
    def predict(self, batch, **kwargs):
        """
        预测方法，返回预测结果
        """
        pass

    def train_epoch(self, dataloader, optimizer, criterion, device, logger=None):
        """
        默认的单轮训练逻辑
        """
        self.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # 移动数据到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # 执行单步训练
            batch_loss = self.train_step(batch, optimizer, criterion, device)
            total_loss += batch_loss
            num_batches += 1

            if logger and batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {batch_loss:.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def evaluate(self, dataloader, metrics, device):
        """
        默认的评估逻辑
        """
        self.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                # 移动数据到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                # 获取预测结果
                predictions = self.predict(batch)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch["item_id"].cpu().numpy())

        # 计算评估指标
        results = {}
        for metric_name, metric_func in metrics.items():
            results[metric_name] = metric_func(all_predictions, all_targets)

        return results
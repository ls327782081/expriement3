import abc
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List, Union
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


# ==================== 模型基类说明 ====================
#
# BaseModel: 简单的 nn.Module 别名
#   - 用于不需要多阶段训练的模型（如 PMAT, MCRL）
#   - 这些模型有自己的训练逻辑（如 train_step 方法）
#   - 适合研究型模型，灵活性高
#
# AbstractTrainableModel: 抽象训练模型基类
#   - 用于需要统一多阶段训练流程的模型（如基线模型）
#   - 提供完整的训练框架：检查点、优化器管理、钩子函数等
#   - 子类必须实现5个抽象方法
#   - 适合生产环境，规范性强
#
# ====================================================

BaseModel = nn.Module


@dataclass
class StageConfig:
    """
    阶段配置数据类：标准化每个训练阶段的参数
    :param stage_id: 阶段唯一标识（如1、2、3）
    :param epochs: 该阶段的训练轮数
    :param start_epoch: 该阶段的起始epoch（用于断点续训，默认0）
    :param kwargs: 该阶段的自定义参数（如学习率、冻结模块列表等）
    """
    stage_id: int
    epochs: int
    start_epoch: int = 0
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.kwargs = self.kwargs or {}

        # 设置调度器默认参数（可在外部传入kwargs覆盖）
        self.kwargs.setdefault('eta_min', 1e-5)  # 余弦退火最小学习率
        self.kwargs.setdefault('weight_decay', 1e-4)  # L2正则


class AbstractTrainableModel(nn.Module, abc.ABC):
    """
    抽象训练模型基类：支持任意多阶段训练，统一调用入口
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.current_stage_id = 0  # 当前阶段ID（0:未开始）
        self.current_stage_epoch = 0  # 当前阶段内的已训练epoch
        self.best_metric = 0.0  # 通用最优指标存储
        self._stage_optimizers = {}  # 缓存各阶段的优化器
        self._stage_schedulers = {} # 缓存各阶段的学习率调度器
        self.checkpoint_dir = './checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.lr_history = []

    # -------------------------- 通用工具方法（无需重写） --------------------------
    def save_checkpoint(self, path: str, additional_info: Optional[Dict] = None):
        """保存检查点：包含当前阶段、阶段内epoch、模型/优化器状态等"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self._get_optimizer_state_dict(),
            "scheduler_state_dict": self._get_scheduler_state_dict(),
            "current_stage_id": self.current_stage_id,
            "current_stage_epoch": self.current_stage_epoch,
            "best_metric": self.best_metric,
            **(additional_info or {})
        }
        torch.save(checkpoint, path)
        print(
            f"Checkpoint saved to {path} | Stage: {self.current_stage_id} | Epoch in stage: {self.current_stage_epoch}")

    def load_checkpoint(self, path: str, strict: bool = False) -> Dict:
        """加载检查点，恢复多阶段训练状态

        Args:
            path: 检查点文件路径
            strict: 是否严格匹配模型参数。默认False，允许加载不完全匹配的checkpoint
                   （例如动态创建的模块可能有不同的键名）
        """
        checkpoint = torch.load(path, map_location=self.device)
        # 使用strict=False允许加载不完全匹配的checkpoint
        # 这对于动态创建模块（如alignment_nets）很重要
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint (ignored): {unexpected_keys}")
        self._load_optimizer_state_dict(checkpoint["optimizer_state_dict"])
        self._load_scheduler_state_dict(checkpoint["scheduler_state_dict"])
        self.current_stage_id = checkpoint["current_stage_id"]
        self.current_stage_epoch = checkpoint["current_stage_epoch"]
        self.best_metric = checkpoint["best_metric"]
        print(f"Checkpoint loaded | Stage: {self.current_stage_id} | Epoch in stage: {self.current_stage_epoch}")
        return checkpoint

    def _freeze_modules(self, module_names: list):
        """冻结指定模块"""
        for name, param in self.named_parameters():
            if any(module_name in name for module_name in module_names):
                param.requires_grad = False

    def _unfreeze_modules(self, module_names: list):
        """解冻指定模块"""
        for name, param in self.named_parameters():
            if any(module_name in name for module_name in module_names):
                param.requires_grad = True

    def _update_params(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, scaler=None):
        """通用参数更新逻辑"""
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


    def _get_scheduler_state_dict(self) -> Dict:
        """获取当前阶段调度器的状态字典"""
        if self.current_stage_id in self._stage_schedulers:
            return self._stage_schedulers[self.current_stage_id].state_dict()
        return {}

    def _load_scheduler_state_dict(self, state_dict: Dict):
        """加载当前阶段调度器的状态字典"""
        if self.current_stage_id in self._stage_schedulers and state_dict:
            self._stage_schedulers[self.current_stage_id].load_state_dict(state_dict)

    # 训练结束后绘制学习率曲线
    def plot_lr_curve(self, save_path="./results/lr_curve.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.lr_history) + 1), self.lr_history, marker='o', label='Actual LR')
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        print(f"学习率曲线已保存到：{save_path}")

    # -------------------------- 抽象方法（实现类必须重写） --------------------------
    @abc.abstractmethod
    def _get_scheduler(self, optimizer: torch.optim.Optimizer, stage_id: int,
                       stage_kwargs: Dict) -> torch.optim.lr_scheduler.LRScheduler:
        """
        获取指定阶段的学习率调度器（子类实现）
        :param optimizer: 当前阶段的优化器
        :param stage_id: 阶段ID
        :param stage_kwargs: 阶段自定义参数
        :return: 学习率调度器
        """
        pass

    @abc.abstractmethod
    def _get_optimizer(self, stage_id: int, stage_kwargs: Dict) -> torch.optim.Optimizer:
        """
        获取指定阶段的优化器（支持自定义阶段参数）
        :param stage_id: 阶段ID（1、2、3...）
        :param stage_kwargs: 该阶段的自定义参数
        """
        pass

    @abc.abstractmethod
    def _get_optimizer_state_dict(self) -> Dict:
        """获取当前阶段优化器的状态字典"""
        pass

    @abc.abstractmethod
    def _load_optimizer_state_dict(self, state_dict: Dict):
        """加载当前阶段优化器的状态字典"""
        pass

    @abc.abstractmethod
    def _train_one_batch(self, batch: Any, stage_id: int, stage_kwargs: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        单batch训练逻辑（按阶段ID区分逻辑）
        :param batch: 训练批次数据
        :param stage_id: 阶段ID
        :param stage_kwargs: 该阶段的自定义参数
        :return: (batch_loss, batch_metrics)
        """
        pass

    @abc.abstractmethod
    def _validate_one_epoch(self, val_dataloader: torch.utils.data.DataLoader, stage_id: int,
                            stage_kwargs: Dict) -> Dict:
        """单轮验证逻辑（按阶段ID区分）"""
        pass

    # -------------------------- 钩子方法（实现类可选重写） --------------------------
    def on_epoch_start(self, epoch: int, stage_id: int, stage_kwargs: Dict):
        """epoch开始钩子"""
        pass

    def on_epoch_end(self, epoch: int, stage_id: int, stage_kwargs: Dict,
                     train_metrics: Dict, val_metrics: Dict):
        """epoch结束钩子：默认实现保存检查点"""
        if stage_id in self._stage_schedulers:
            scheduler = self._stage_schedulers[stage_id]
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"\nEpoch {epoch+1} | Stage {stage_id} | 当前学习率: {current_lr:.6f}")
            self.lr_history.append(current_lr)

        experiment_name = stage_kwargs.get('experiment_name', self.__class__.__name__)
        val_loss = val_metrics.get('loss', float('inf'))

        # 更新最优指标
        is_best = val_loss < self.best_metric if self.best_metric > 0 else True
        if is_best:
            self.best_metric = val_loss



        # 保存当前epoch的检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{experiment_name}_stage_{stage_id}_epoch_{epoch}.pth')
        self.save_checkpoint(checkpoint_path)

        # 如果是最优模型，额外保存一份
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f'{experiment_name}_best.pth')
            self.save_checkpoint(best_path)
            print(f"✅ 保存最优模型到 {best_path} | Val Loss: {val_loss:.4f}")

    def on_batch_start(self, batch: Any, batch_idx: int, stage_id: int, stage_kwargs: Dict):
        """batch开始钩子"""
        pass

    def on_batch_end(self, batch: Any, batch_idx: int, stage_id: int, stage_kwargs: Dict, loss: torch.Tensor,
                     metrics: Dict):
        """batch结束钩子"""
        pass

    def on_stage_start(self, stage_id: int, stage_kwargs: Dict):
        """阶段开始钩子（如初始化模块、冻结参数）"""
        pass

    def on_stage_end(self, stage_id: int, stage_kwargs: Dict):
        """阶段结束钩子"""
        experiment_name = stage_kwargs.get('experiment_name', self.__class__.__name__)
        self.plot_lr_curve(f"./results/{experiment_name}_stage_{stage_id}_lr_curve.png")


    def on_stage_switch(self, from_stage_id: int, to_stage_id: int, to_stage_kwargs: Dict):
        """阶段切换钩子（核心：处理模块冻结/解冻、学习率调整等）"""
        pass

    # -------------------------- 核心训练流程（基类封装，统一入口） --------------------------
    def _train_one_epoch(self, train_dataloader: torch.utils.data.DataLoader, stage_id: int,
                         stage_kwargs: Dict) -> Dict:
        """单轮训练（通用流程）"""
        self.train()
        total_loss = 0.0
        total_metrics = {}

        # 获取当前阶段的优化器（缓存避免重复创建）
        if stage_id not in self._stage_optimizers:
            self._stage_optimizers[stage_id] = self._get_optimizer(stage_id, stage_kwargs)
        optimizer = self._stage_optimizers[stage_id]
        if stage_id not in self._stage_schedulers:
            self._stage_schedulers[stage_id] = self._get_scheduler(optimizer, stage_id, stage_kwargs)

        # 混合精度训练配置
        # 注意：某些模型架构（如包含自定义层的模型）可能与AMP不兼容
        # 如果遇到dtype不匹配错误，可以设置use_amp=False
        use_amp = False  # 禁用混合精度训练以避免dtype不匹配问题
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        pbar = tqdm(train_dataloader, desc=f"Stage {stage_id} Train Epoch")
        for batch_idx, batch in enumerate(pbar):
            self.on_batch_start(batch, batch_idx, stage_id, stage_kwargs)

            # 数据移到设备（支持字典和列表两种格式）
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]

            # 核心训练逻辑（使用autocast进行混合精度）
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss, metrics = self._train_one_batch(batch, stage_id, stage_kwargs)
            else:
                loss, metrics = self._train_one_batch(batch, stage_id, stage_kwargs)

            # 参数更新
            self._update_params(loss, optimizer, scaler)

            # 累计指标
            total_loss += loss.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)

            self.on_batch_end(batch, batch_idx, stage_id, stage_kwargs, loss, metrics)
            pbar.set_postfix({"loss": loss.item()})

        pbar.close()
        # 计算平均指标
        avg_loss = total_loss / len(train_dataloader)
        avg_metrics = {k: v / len(train_dataloader) for k, v in total_metrics.items()}
        avg_metrics["loss"] = avg_loss
        return avg_metrics

    def customer_train(self,
              train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: torch.utils.data.DataLoader,
              stage_configs: List[StageConfig],
              skip_validation: bool = False):
        """
        统一训练入口：支持任意多阶段训练
        :param train_dataloader: 训练数据加载器
        :param val_dataloader: 验证数据加载器
        :param stage_configs: 阶段配置列表（1个元素=一阶段，N个元素=N阶段）
        :param skip_validation: 是否跳过训练过程中的验证（用于加速训练）
        """
        # 排序阶段配置（确保按stage_id升序执行）
        stage_configs = sorted(stage_configs, key=lambda x: x.stage_id)
        prev_stage_id = 0  # 初始前一阶段为0（未开始）

        # 迭代执行每个阶段
        for stage_cfg in stage_configs:
            stage_id = stage_cfg.stage_id
            stage_kwargs = stage_cfg.kwargs
            total_epochs = stage_cfg.epochs
            start_epoch = stage_cfg.start_epoch

            # 跳过已完成的阶段（断点续训）
            if self.current_stage_id > stage_id:
                continue
            # 调整当前阶段的起始epoch（断点续训）
            if self.current_stage_id == stage_id:
                start_epoch = self.current_stage_epoch

            # -------------------------- 阶段开始 --------------------------
            self.current_stage_id = stage_id
            self.on_stage_start(stage_id, stage_kwargs)
            # 阶段切换钩子（非第一个阶段时触发）
            if prev_stage_id != 0:
                self.on_stage_switch(prev_stage_id, stage_id, stage_kwargs)
            print(f"\n===== Start Stage {stage_id} Training (Epochs: {start_epoch + 1}~{total_epochs}) =====")

            experiment_name = stage_kwargs.get('experiment_name', self.__class__.__name__)
            # -------------------------- 阶段内训练 --------------------------
            for epoch in range(start_epoch, total_epochs):
                self.current_stage_epoch = epoch

                checkpoint_path = os.path.join(self.checkpoint_dir, f'{experiment_name}_stage_{stage_id}_epoch_{epoch}.pth')
                if os.path.exists(checkpoint_path):
                    self.load_checkpoint(checkpoint_path)
                    continue

                # Epoch开始钩子
                self.on_epoch_start(epoch, stage_id, stage_kwargs)

                # 训练一轮
                train_metrics = self._train_one_epoch(train_dataloader, stage_id, stage_kwargs)
                # 验证一轮（可选跳过以加速训练）
                if skip_validation:
                    val_metrics = {}  # 跳过验证时返回空指标
                else:
                    val_metrics = self._validate_one_epoch(val_dataloader, stage_id, stage_kwargs)

                # Epoch结束钩子
                self.on_epoch_end(epoch, stage_id, stage_kwargs, train_metrics,
                                  val_metrics)

            # -------------------------- 阶段结束 --------------------------
            self.on_stage_end(stage_id, stage_kwargs)
            self.current_stage_epoch = 0  # 重置阶段内epoch
            prev_stage_id = stage_id

        # 训练全部完成
        print("\n===== All Stages Training Completed =====")
        self.current_stage_id = 0  # 重置阶段ID

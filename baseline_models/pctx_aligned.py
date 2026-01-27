"""
Pctx 模型 - 对齐官方实现
官方源码: https://github.com/YoungZ365/Pctx

核心特点:
1. 简单的 T5 包装（所有复杂性在 Tokenizer 中）
2. forward 方法直接调用 self.t5(**batch)
3. 支持自定义 beam search 和 beam merging
"""
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from base_model import AbstractTrainableModel
from config import config
from collections import defaultdict
import numpy as np


class PctxAligned(AbstractTrainableModel):
    """
    Pctx: 个性化上下文感知推荐模型（对齐官方实现）

    与官方实现的一致性:
    - ✅ 简单的 T5 包装
    - ✅ forward 直接调用 t5(**batch)
    - ✅ 自定义 beam search
    - ⚠️ Tokenizer 需要单独实现（预处理 semantic IDs）
    """

    def __init__(self, vocab_size=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(PctxAligned, self).__init__(device)

        # 计算 vocab_size
        # 官方: 0(padding) + 256*4(4个codebook) + n_user_tokens + 1(eos)
        if vocab_size is None:
            codebook_size = getattr(config, 'codebook_size', 256)
            n_codebooks = getattr(config, 'n_codebooks', 3)
            n_user_tokens = getattr(config, 'n_user_tokens', 1)
            # base_user_token = sum(codebook_sizes) + 1
            # eos_token = base_user_token + n_user_tokens
            vocab_size = codebook_size * (n_codebooks + 1) + n_user_tokens + 1

        self.vocab_size = vocab_size

        # T5 配置（对齐官方参数）
        t5_config = T5Config(
            num_layers=getattr(config, 'num_layers', 4),
            num_decoder_layers=getattr(config, 'num_decoder_layers', 4),
            d_model=getattr(config, 'd_model', 128),  # 官方默认 128
            d_ff=getattr(config, 'd_ff', 1024),
            num_heads=getattr(config, 'num_heads', 6),
            d_kv=getattr(config, 'd_kv', 64),
            dropout_rate=getattr(config, 'dropout_rate', 0.1),
            activation_function=getattr(config, 'activation_function', 'relu'),
            vocab_size=self.vocab_size,
            pad_token_id=0,
            eos_token_id=self.vocab_size - 1,  # 最后一个token是eos
            decoder_start_token_id=0,
            feed_forward_proj=getattr(config, 'feed_forward_proj', 'relu'),
        )

        # T5 模型（官方命名为 self.t5）
        self.t5 = T5ForConditionalGeneration(config=t5_config)

        # 推理时的 ensemble 数量
        self.n_inference_ensemble = getattr(config, 'n_inference_ensemble', -1)

        print(f"[PctxAligned] Initialized with vocab_size={self.vocab_size}")
        print(f"[PctxAligned] T5 config: d_model={t5_config.d_model}, num_layers={t5_config.num_layers}")

    @property
    def n_parameters(self) -> str:
        """计算参数量（对齐官方）"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.t5.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
               f'#Non-embedding parameters: {total_params - emb_params}\n' \
               f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict) -> torch.Tensor:
        """
        前向传播（对齐官方实现）

        官方实现:
            def forward(self, batch: dict) -> torch.Tensor:
                outputs = self.t5(**batch)
                return outputs

        Args:
            batch: 应该包含以下键（由 Tokenizer 预处理）:
                - input_ids: (batch_size, seq_len)
                - attention_mask: (batch_size, seq_len)
                - labels: (batch_size, label_len) - 可选，训练时需要

        Returns:
            outputs: T5 的输出，包含 loss 和 logits
        """
        # 官方实现就这么简单！
        outputs = self.t5(**batch)
        return outputs

    def _get_optimizer(self, stage_id: int, stage_kwargs: dict) -> torch.optim.Optimizer:
        """获取优化器（对齐官方使用 AdamW）"""
        lr = stage_kwargs.get('lr', 0.003)  # 官方默认 0.003
        weight_decay = stage_kwargs.get('weight_decay', 0.1)  # 官方默认 0.1
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_optimizer_state_dict(self) -> dict:
        """获取优化器状态"""
        optimizer_states = {}
        for stage_id, optimizer in self._stage_optimizers.items():
            optimizer_states[stage_id] = optimizer.state_dict()
        return optimizer_states

    def _load_optimizer_state_dict(self, state_dict: dict):
        """加载优化器状态"""
        for stage_id, opt_state in state_dict.items():
            if stage_id in self._stage_optimizers:
                self._stage_optimizers[stage_id].load_state_dict(opt_state)

    def _train_one_batch(self, batch: any, stage_id: int, stage_kwargs: dict) -> tuple:
        """
        单batch训练（简化版）

        Args:
            batch: 应该是预处理好的字典，包含 input_ids, attention_mask, labels

        Returns:
            (loss, metrics)
        """
        # 确保数据在正确设备上
        if isinstance(batch, dict):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

        # 前向传播
        outputs = self.forward(batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # 计算指标（简化版）
        metrics = {}
        if hasattr(outputs, 'logits') and 'labels' in batch:
            with torch.no_grad():
                predictions = torch.argmax(outputs.logits, dim=-1)
                labels = batch['labels']

                # 忽略 padding token（T5 使用 -100 表示需要忽略的位置）
                mask = labels != -100
                if mask.sum() > 0:
                    correct = ((predictions == labels) & mask).float().sum()
                    total = mask.sum()
                    metrics['accuracy'] = (correct / total).item()
                else:
                    metrics['accuracy'] = 0.0

                # 添加 perplexity（语言模型常用指标）
                metrics['perplexity'] = torch.exp(loss.detach()).item()

        return loss, metrics


    def _validate_one_epoch(self, val_dataloader, stage_id: int, stage_kwargs: dict) -> dict:
        """单轮验证"""
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if isinstance(batch, dict):
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device)

                outputs = self.forward(batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()

                if hasattr(outputs, 'logits') and 'labels' in batch:
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct = (predictions == batch['labels']).float().sum()
                    total_correct += correct.item()
                    total_samples += batch['labels'].numel()

        metrics = {
            'val_loss': total_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0,
            'val_accuracy': total_correct / total_samples if total_samples > 0 else 0.0
        }

        return metrics

    def predict(self, batch, **kwargs):
        """预测方法"""
        self.eval()

        if isinstance(batch, dict):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

        with torch.no_grad():
            num_beams = kwargs.get('num_beams', getattr(config, 'num_beams', 50))
            max_length = kwargs.get('max_length', 6)

            generated = self.t5.generate(
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=kwargs.get('num_return_sequences', 10),
                early_stopping=True,
                pad_token_id=0,
                eos_token_id=self.vocab_size - 1
            )

        return generated

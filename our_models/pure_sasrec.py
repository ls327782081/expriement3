"""
Pure ID SASRec (完全对齐RecBole官方实现 + 补全抽象方法)
解决TypeError，确保能正常实例化和训练
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, List
import numpy as np
from base_model import AbstractTrainableModel  # 你的基类
from tqdm import tqdm
from utils.utils import check_tensor


class RecBoleSASRec(AbstractTrainableModel):
    """完全对齐RecBole v1.2.0 SASRec实现的纯ID版本（补全所有抽象方法）"""

    def __init__(
            self,
            config,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(device=device)  # 必须调用父类初始化
        self.config = config

        # 核心配置（和RecBole完全一致）
        self.hidden_dim = config.hidden_dim
        self.max_seq_len = getattr(config, 'max_history_len', 50)
        self.num_items = config.num_items  # 必须和RecBole一致
        self.num_heads = config.attention_heads
        self.num_blocks = getattr(config, 'num_transformer_blocks', 2)
        self.dropout = config.dropout
        self.loss_type = getattr(config, 'loss_type', 'CE')  # RecBole默认CE

        # 1. 物品Embedding（RecBole标准）
        self.item_embedding = nn.Embedding(
            self.num_items,
            self.hidden_dim,
            padding_idx=0  # PAD_ID=0（左对齐）
        )
        # 2. 位置编码（RecBole标准：max_seq_len）
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_dim)
        # 3. LayerNorm + Dropout
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.input_dropout = nn.Dropout(self.dropout)

        # 4. Transformer Encoder（RecBole Pre-LN架构）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # RecBole核心：Pre-LN
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_blocks
        )

        # 5. 输出层
        self.output_layer = nn.Linear(self.hidden_dim, self.num_items)

        # 初始化权重（RecBole官方逻辑）
        self._init_weights()

        # 移动模型到指定设备
        self.to(self.device)

    def _init_weights(self):
        """完全复刻RecBole权重初始化"""
        # 物品embedding
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.constant_(self.item_embedding.weight[self.item_embedding.padding_idx], 0.0)
        # 位置编码
        nn.init.xavier_uniform_(self.position_embedding.weight)
        # LayerNorm
        nn.init.constant_(self.input_norm.weight, 1.0)
        nn.init.constant_(self.input_norm.bias, 0.0)
        # Transformer层
        for m in self.transformer_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        # 输出层
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """生成RecBole标准因果掩码（左对齐）"""
        # 上三角为True（屏蔽未来位置），下三角为False
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        return mask

    def encode_sequence(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        完全对齐RecBole的左对齐序列编码
        batch包含：
        - history_items: [batch, max_seq_len] 左对齐，如[3,4,5,0,0]
        - history_len: [batch] 有效序列长度，如3
        """
        history_items = batch['history_items'].to(self.device)  # [B, L]
        history_len = batch['history_len'].to(self.device)  # [B]
        batch_size, max_seq_len = history_items.shape

        # 1. Item Embedding
        item_emb = self.item_embedding(history_items)  # [B, L, D]

        # 2. 位置编码（RecBole标准：仅有效位置有编码）
        position_ids = torch.arange(max_seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)  # [B, L]
        # PAD位置（item_id=0）的位置编码置0
        position_ids = position_ids * (history_items != 0).long()
        pos_emb = self.position_embedding(position_ids)  # [B, L, D]

        # 3. 输入融合 + Norm + Dropout
        seq_emb = item_emb + pos_emb
        seq_emb = self.input_norm(seq_emb)
        seq_emb = self.input_dropout(seq_emb)

        # 4. 生成RecBole标准掩码
        # 4.1 因果掩码（屏蔽未来位置）
        causal_mask = self._generate_causal_mask(max_seq_len)  # [L, L]
        # 4.2 Key Padding Mask（PAD位置为True，屏蔽PAD）
        padding_mask = (history_items == 0)  # [B, L]

        # 5. Transformer编码（RecBole核心调用方式）
        seq_output = self.transformer_encoder(
            src=seq_emb,
            mask=causal_mask,  # src_mask：因果掩码
            src_key_padding_mask=padding_mask  # src_key_padding_mask：PAD掩码
        )  # [B, L, D]

        # 6. 提取最后有效位置（RecBole核心：左对齐的最后有效位置）
        batch_idx = torch.arange(batch_size, device=self.device)
        # 取每个样本的有效长度-1（左对齐的最后一个有效位置）
        last_valid_idx = history_len - 1
        # 兜底：防止seq_len=0
        last_valid_idx = torch.clamp(last_valid_idx, min=0, max=max_seq_len - 1)
        user_repr = seq_output[batch_idx, last_valid_idx]  # [B, D]

        # 调试：验证最后有效位置是否正确
        # if self.training and torch.rand(1).item() < 0.01:
        #     print(f"\n===== RecBole序列编码验证 =====")
        #     print(f"history_items[0]: {history_items[0].tolist()}")
        #     print(f"history_len[0]: {history_len[0].item()}")
        #     print(f"last_valid_idx[0]: {last_valid_idx[0].item()}")
        #     print(f"取到的item_id: {history_items[0, last_valid_idx[0]].item()}")

        return user_repr

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播（完全对齐RecBole）"""
        # 1. 编码用户序列
        user_repr = self.encode_sequence(batch)  # [B, D]

        # 2. 计算所有物品得分（RecBole标准：无温度系数、不归一化）
        logits = self.output_layer(user_repr)  # [B, N_items]

        return {
            'user_repr': user_repr,
            'logits': logits,
            'target_item': batch['target_item'].to(self.device)
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """损失计算（完全对齐RecBole）"""
        logits = outputs['logits']
        target_items = outputs['target_item']

        # 过滤PAD的target（RecBole核心）
        valid_mask = (target_items != 0)
        if not valid_mask.any():
            return {
                'ce_loss': torch.tensor(0.0, device=self.device),
                'total_loss': torch.tensor(0.0, device=self.device)
            }

        # 仅计算有效样本的交叉熵损失
        ce_loss = F.cross_entropy(logits[valid_mask], target_items[valid_mask])

        return {
            'ce_loss': ce_loss,
            'total_loss': ce_loss
        }

    # ==================== 补全抽象方法（解决TypeError）====================
    def _get_optimizer(self, stage_id: int, stage_kwargs: Dict) -> torch.optim.Optimizer:
        """RecBole默认优化器：AdamW（实现抽象方法）"""
        lr = stage_kwargs.get('lr', 1e-4)
        weight_decay = stage_kwargs.get('weight_decay', 1e-4)
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_scheduler(self, optimizer: torch.optim.Optimizer, stage_id: int,
                       stage_kwargs: Dict) -> torch.optim.lr_scheduler.LRScheduler:
        """RecBole默认调度器：Warmup+余弦退火（实现抽象方法）"""
        total_epochs = self.config.epochs
        warmup_epochs = max(1, int(total_epochs * 0.1))

        # Warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        # 余弦退火
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )
        # 组合调度器
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        return scheduler

    def _get_optimizer_state_dict(self) -> Dict:
        """获取所有阶段优化器的状态字典（实现抽象方法）"""
        optimizer_states = {}
        for stage_id, optimizer in self._stage_optimizers.items():
            optimizer_states[stage_id] = optimizer.state_dict()
        return optimizer_states

    def _load_optimizer_state_dict(self, state_dict: Dict):
        """加载所有阶段优化器的状态字典（实现抽象方法）"""
        for stage_id, opt_state in state_dict.items():
            if stage_id in self._stage_optimizers:
                self._stage_optimizers[stage_id].load_state_dict(opt_state)

    def _train_one_batch(self, batch: Any, stage_id: int, stage_kwargs: Dict) -> Tuple[torch.Tensor, Dict]:
        """单batch训练逻辑（实现抽象方法）"""
        outputs = self.forward(batch)
        losses = self.compute_loss(outputs)
        metrics = {'ce_loss': losses['ce_loss'].item()}
        return losses['total_loss'], metrics

    def _validate_one_epoch(
            self,
            val_dataloader: torch.utils.data.DataLoader,
            stage_id: int,
            stage_kwargs: Dict
    ) -> Dict:
        """验证逻辑（完全对齐RecBole Full Ranking，实现抽象方法）"""
        self.eval()
        all_ranks = []
        val_pbar = tqdm(val_dataloader, desc="Validate", leave=False, total=len(val_dataloader))

        with torch.no_grad():
            for batch in val_pbar:
                # 数据移到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # 1. 编码用户（和训练逻辑一致）
                user_repr = self.encode_sequence(batch)  # [B, D]

                # 2. 计算所有物品得分（RecBole标准）
                logits = self.output_layer(user_repr)  # [B, N_items]

                # 3. 计算排名（RecBole核心逻辑）
                batch_size = logits.shape[0]
                target_items = batch['target_item']
                valid_mask = (target_items != 0)

                # 初始化排名
                ranks = torch.full((batch_size,), self.num_items, device=self.device, dtype=torch.float)

                if valid_mask.any():
                    # 仅计算有效样本的排名
                    valid_idx = torch.where(valid_mask)[0]
                    valid_logits = logits[valid_idx]
                    valid_targets = target_items[valid_idx]

                    # 计算每个目标物品的排名
                    for idx, (logit, target) in enumerate(zip(valid_logits, valid_targets)):
                        # 分数大于目标物品的数量 + 1 = 排名
                        rank = (logit > logit[target]).sum().item() + 1
                        ranks[valid_idx[idx]] = rank

                all_ranks.append(ranks.cpu())

        # 计算评估指标
        all_ranks = torch.cat(all_ranks, dim=0).float()
        metrics = self._compute_metrics(all_ranks)
        self.train()
        return metrics

    def _compute_metrics(self, ranks: torch.Tensor, k_list: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """RecBole标准评估指标计算"""
        metrics = {}
        for k in k_list:
            # HR@k
            hits = (ranks <= k).float()
            metrics[f'HR@{k}'] = hits.mean().item()

            # NDCG@k
            dcg = 1.0 / torch.log2(ranks.clamp(min=1).float() + 1)
            dcg = torch.where(ranks <= k, dcg, torch.zeros_like(dcg))
            metrics[f'NDCG@{k}'] = dcg.mean().item()

            # MRR@k
            rr = 1.0 / ranks.clamp(min=1).float()
            rr = torch.where(ranks <= k, rr, torch.zeros_like(rr))
            metrics[f'MRR@{k}'] = rr.mean().item()

        # 全局MRR和平均排名
        metrics['MRR'] = (1.0 / ranks.clamp(min=1).float()).mean().item()
        metrics['Mean_Rank'] = ranks.mean().item()

        return metrics
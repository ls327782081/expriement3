"""
终极修复：完全复刻RecBole v1.2.0 SASRec（解决梯度阻塞+假学习问题）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, List
from base_model import AbstractTrainableModel
from tqdm import tqdm


class RecBoleSASRec(AbstractTrainableModel):
    """1:1复刻RecBole v1.2.0 SASRec"""

    def __init__(
        self,
        config,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(device=device)
        self.config = config
        self.device = torch.device(device)

        # RecBole核心配置
        self.hidden_size = config.hidden_dim  # 64
        self.inner_size = config.inner_size  # 256
        self.max_seq_len = config.max_history_len  # 50
        self.n_items = config.num_items  # 物品总数
        self.n_heads = config.attention_heads  # 2
        self.n_layers = config.num_transformer_blocks  # 2
        self.dropout_prob = config.dropout  # 0.5
        self.layer_norm_eps = config.layer_norm_eps  # 1e-12
        self.initializer_range = config.initializer_range  # 0.02


        # 1. 物品Embedding（RecBole官方：共享权重）
        self.item_embedding = nn.Embedding(
            self.n_items,
            self.hidden_size,
            padding_idx=0
        )
        # 2. 位置编码
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)
        # 3. Dropout
        self.dropout = nn.Dropout(self.dropout_prob)

        # 4. Transformer Encoder（RecBole Pre-LN）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,  # inner_size=256
            dropout=self.dropout_prob,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN
            layer_norm_eps=self.layer_norm_eps
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # 5. 输出LayerNorm（RecBole核心缺失项）
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # 初始化权重（RecBole官方逻辑）
        self._init_weights()

        # 移到设备
        self.to(self.device)

    def _init_weights(self):
        """RecBole官方初始化"""
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=self.initializer_range)
        nn.init.constant_(self.item_embedding.weight[self.item_embedding.padding_idx], 0.0)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=self.initializer_range)
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=self.initializer_range)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _generate_mask(self, seq_len):
        """RecBole官方因果掩码"""
        mask = (torch.triu(torch.ones(seq_len, seq_len, device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """RecBole官方forward逻辑"""
        # 1. 取数据
        history = batch['history_items'].to(self.device)  # [B, L]
        hist_len = batch['history_len'].to(self.device)    # [B]
        target = batch['target_item'].to(self.device)      # [B]

        # 2. 生成位置ID（左对齐）
        position_ids = torch.arange(history.size(1), device=self.device).unsqueeze(0).repeat(history.size(0), 1)
        position_ids = position_ids * (history != 0).long()

        # 3. Embedding层
        item_emb = self.item_embedding(history)  # [B, L, D]
        pos_emb = self.position_embedding(position_ids)  # [B, L, D]
        seq_emb = item_emb + pos_emb
        seq_emb = self.dropout(seq_emb)

        # 4. 生成掩码
        padding_mask = (history == 0)  # [B, L]
        causal_mask = self._generate_mask(history.size(1))  # [L, L]

        # 5. Transformer编码
        seq_output = self.encoder(
            src=seq_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )  # [B, L, D]

        # 6. RecBole核心：取最后有效位置 + LayerNorm
        batch_idx = torch.arange(history.size(0), device=self.device)
        last_idx = hist_len - 1
        last_idx = torch.clamp(last_idx, min=0)
        seq_output = self.layer_norm(seq_output)  # 关键：输出LayerNorm
        user_emb = seq_output[batch_idx, last_idx, :]  # [B, D]

        # 7. 计算得分（RecBole官方：共享item_embedding权重）
        score = torch.matmul(user_emb, self.item_embedding.weight.t())  # [B, N]

        return {
            'user_emb': user_emb,
            'score': score,
            'target': target
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """RecBole官方Loss计算"""
        score = outputs['score']
        target = outputs['target']

        # 过滤PAD
        valid_mask = (target != 0)
        if not valid_mask.any():
            return {'total_loss': torch.tensor(0.0, device=self.device)}

        loss = F.cross_entropy(score[valid_mask], target[valid_mask])


        return {'total_loss': loss}

    # ==================== 抽象方法实现（RecBole官方优化器/调度器）====================
    def _get_optimizer(self, stage_id: int, stage_kwargs: Dict) -> torch.optim.Optimizer:
        """RecBole官方：Adam + 学习率1e-3"""
        lr = stage_kwargs.get('lr', 1e-3)
        weight_decay = stage_kwargs.get('weight_decay', 0.0)  # RecBole无权重衰减
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_scheduler(self, optimizer: torch.optim.Optimizer, stage_id: int, stage_kwargs: Dict) -> torch.optim.lr_scheduler.LRScheduler:
        """RecBole官方：StepLR（不是余弦退火）"""
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)

    def _get_optimizer_state_dict(self) -> Dict:
        return {k: v.state_dict() for k, v in self._stage_optimizers.items()}

    def _load_optimizer_state_dict(self, state_dict: Dict):
        for k, v in state_dict.items():
            if k in self._stage_optimizers:
                self._stage_optimizers[k].load_state_dict(v)

    def _train_one_batch(self, batch: Any, stage_id: int, stage_kwargs: Dict) -> Tuple[torch.Tensor, Dict]:
        outputs = self.forward(batch)
        loss_dict = self.compute_loss(outputs)
        return loss_dict['total_loss'], {'loss': loss_dict['total_loss'].item()}

    def _validate_one_epoch(
            self,
            val_dataloader: torch.utils.data.DataLoader,
            stage_id: int,
            stage_kwargs: Dict
    ) -> Dict:
        """RecBole官方验证逻辑（排除历史物品+正确排名）"""
        self.eval()
        all_ranks = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validate", leave=False):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # 1. 前向传播
                outputs = self.forward(batch)
                score = outputs['score']
                history = batch['history_items']
                target = batch['target_item']
                valid_mask = (target != 0)

                # 2. 初始化排名
                ranks = torch.full((score.size(0),), self.n_items, device=self.device, dtype=torch.float)

                if valid_mask.any():
                    # 2.1 排除历史物品
                    valid_idx = torch.where(valid_mask)[0]
                    valid_score = score[valid_idx]
                    valid_history = history[valid_idx]
                    valid_target = target[valid_idx]

                    # 标记历史物品
                    history_mask = torch.zeros_like(valid_score, dtype=torch.bool)
                    for i in range(len(valid_idx)):
                        hist = valid_history[i][valid_history[i] != 0]
                        if len(hist) > 0:
                            history_mask[i, hist] = True
                    valid_score = valid_score.masked_fill(history_mask, -float('inf'))
                    if i < 3:
                        print(f"样本{i}目标物品得分: {valid_score[i, valid_target[i]].item():.4f}")

                    # 2.2 计算排名（RecBole官方逻辑）
                    target_score = valid_score[torch.arange(len(valid_idx)), valid_target]
                    rank = (valid_score >= target_score.unsqueeze(1)).sum(dim=1).float()
                    ranks[valid_idx] = rank

                all_ranks.append(ranks.cpu())

        # 计算指标
        all_ranks = torch.cat(all_ranks, dim=0).float()
        metrics = self._compute_metrics(all_ranks)
        self.train()
        return metrics

    def _compute_metrics(self, ranks: torch.Tensor, k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
        metrics = {}
        for k in k_list:
            hits = (ranks <= k).float()
            metrics[f'HR@{k}'] = hits.mean().item()
            metrics[f'hit@{k}'] = hits.mean().item()  # 新增：RecBole原生hit@k
            metrics[f'recall@{k}'] = hits.mean().item()  # 新增：RecBole原生recall@k（单目标下=hit@k）

            dcg = 1.0 / torch.log2(ranks.clamp(min=1).float() + 1)
            dcg = torch.where(ranks <= k, dcg, torch.zeros_like(dcg))
            metrics[f'NDCG@{k}'] = dcg.mean().item()
            metrics[f'ndcg@{k}'] = dcg.mean().item()  # 新增：RecBole原生ndcg@k

            rr = 1.0 / ranks.clamp(min=1).float()
            rr = torch.where(ranks <= k, rr, torch.zeros_like(rr))
            metrics[f'MRR@{k}'] = rr.mean().item()
            metrics[f'mrr@{k}'] = rr.mean().item()  # 新增：RecBole原生mrr@k

            # 新增：RecBole原生precision@k
            precision = (ranks <= k).float() / k
            metrics[f'precision@{k}'] = precision.mean().item()

        metrics['MRR'] = (1.0 / ranks.clamp(min=1).float()).mean().item()
        metrics['Mean_Rank'] = ranks.mean().item()
        return metrics
"""
Pure Multimodal SASRec: 纯净的多模态序列推荐基线

这是一个最简化的实现，用于验证：
1. 数据加载是否正确
2. SASRec 骨架是否正常工作
3. Cross Entropy 损失是否正确

不包含任何创新组件：
- 无 SemanticIDQuantizer
- 无 UserModalAttention
- 无 DynamicIDUpdater
- 无对比学习
- 无两阶段训练

架构:
Item features (text_feat, vision_feat)
   ↓
简单线性融合 → item embedding
   ↓
SASRec (Transformer with causal mask)
   ↓
Cross Entropy Loss (softmax over all items)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List
import math

from base_model import AbstractTrainableModel
from baseline_models.sasrec import TransformerBlock


class SimpleItemEncoder(nn.Module):
    """最简单的多模态物品编码器
    
    只做线性投影和加权融合，不包含任何复杂组件
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # 简单的线性投影
        self.text_encoder = nn.Sequential(
            nn.Linear(config.text_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(config.visual_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 可学习的模态权重
        self.modal_weight = nn.Parameter(torch.ones(2) / 2)
        
    def forward(self, text_feat: torch.Tensor, vision_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feat: (..., text_dim)
            vision_feat: (..., visual_dim)
        
        Returns:
            item_emb: (..., hidden_dim)
        """
        text_feat = text_feat.float()
        vision_feat = vision_feat.float()
        
        # 编码各模态
        text_encoded = self.text_encoder(text_feat)
        visual_encoded = self.visual_encoder(vision_feat)
        
        # 加权融合
        weights = F.softmax(self.modal_weight, dim=0)
        item_emb = weights[0] * text_encoded + weights[1] * visual_encoded
        
        return item_emb


class PureSASRec(AbstractTrainableModel):
    """纯净的多模态 SASRec
    
    用于验证基础骨架是否正常工作
    """
    
    def __init__(
        self,
        config,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(device=device)
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.max_seq_len = getattr(config, 'max_history_len', 50)
        
        # ===== 物品编码器（最简单的版本） =====
        self.item_encoder = SimpleItemEncoder(config)
        
        # ===== 序列编码器 (SASRec) =====
        self.pos_emb = nn.Embedding(self.max_seq_len, config.hidden_dim)
        
        num_blocks = getattr(config, 'num_transformer_blocks', 2)
        num_heads = config.attention_heads
        dropout_rate = config.dropout
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        self.input_layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # ===== 投影层 =====
        self.user_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        self.item_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        # ===== 预计算的物品表征 =====
        self._all_item_repr = None
        
        # 缓存因果掩码
        self._causal_mask_cache = {}
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.input_layer_norm.weight, 1.0)
        nn.init.constant_(self.input_layer_norm.bias, 0.0)
        
        for module in [self.user_projection, self.item_projection]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """获取因果掩码"""
        cache_key = (seq_len, device)
        if cache_key not in self._causal_mask_cache:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )
            self._causal_mask_cache[cache_key] = causal_mask
        return self._causal_mask_cache[cache_key]
    
    def set_all_item_features(self, all_item_features: Dict[str, torch.Tensor]):
        """预计算所有物品的表征"""
        self.eval()
        
        all_text_feat = all_item_features['text'].to(self.device)
        all_visual_feat = all_item_features['visual'].to(self.device)
        num_items = all_text_feat.shape[0]
        
        print(f"预计算 {num_items} 个物品的表征...")
        
        item_batch_size = 256
        all_item_repr_list = []
        
        with torch.no_grad():
            for start_idx in range(0, num_items, item_batch_size):
                end_idx = min(start_idx + item_batch_size, num_items)
                item_text = all_text_feat[start_idx:end_idx]
                item_visual = all_visual_feat[start_idx:end_idx]
                
                item_emb = self.item_encoder(item_text, item_visual)
                item_repr = self.item_projection(item_emb)
                all_item_repr_list.append(item_repr.cpu())
            
            self._all_item_repr = torch.cat(all_item_repr_list, dim=0).to(self.device)
            # L2 归一化
            self._all_item_repr = F.normalize(self._all_item_repr, dim=-1)
        
        print(f"物品表征预计算完成，形状: {self._all_item_repr.shape}")
    
    def encode_sequence(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor,
        seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """编码用户历史序列
        
        Returns:
            user_repr: (batch, hidden_dim)
        """
        batch_size, seq_len, _ = text_feat.shape
        device = text_feat.device

        # 1. 创建掩码
        # 注意：数据是左 padding！格式为 [PAD, PAD, ..., item1, item2, item3]
        # padding 在前面，有效内容在后面
        # 例如 seq_len=5, history_len=3 时，位置 0,1 是 padding，位置 2,3,4 是有效的
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        # padding 位置：位置 < (seq_len - history_len)
        pad_start = seq_len - seq_lens.unsqueeze(1)  # (batch, 1)
        padding_mask = positions < pad_start  # (batch, seq_len), True 表示 padding

        # 2. 物品编码
        item_emb = self.item_encoder(text_feat, vision_feat)

        # 3. 添加位置嵌入
        # 对于左 padding，我们需要给有效位置分配正确的位置编码
        # 有效位置从 0 开始编号（相对位置）
        # 例如 seq_len=5, history_len=3: 位置 2,3,4 的相对位置是 0,1,2
        pos_indices = positions - pad_start  # 相对位置
        pos_indices = pos_indices.clamp(min=0, max=self.max_seq_len - 1)
        seq_emb = item_emb + self.pos_emb(pos_indices)

        # 4. 处理 padding
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(seq_emb)
        noise = torch.randn_like(seq_emb) * 0.01
        seq_emb = torch.where(padding_mask_expanded, noise.detach(), seq_emb)

        # 5. LayerNorm + Dropout
        seq_emb = self.input_layer_norm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        # 6. Causal mask
        causal_mask = self._get_causal_mask(seq_len, device)

        # 7. Transformer
        for block in self.transformer_blocks:
            seq_emb = block(seq_emb, padding_mask=padding_mask, causal_mask=causal_mask)

        # 处理可能的 NaN
        seq_emb = torch.nan_to_num(seq_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # 8. 获取用户表示（最后一个位置，即序列末尾）
        # 对于左 padding，最后一个有效位置就是 seq_len - 1
        user_repr = seq_emb[:, -1, :]  # (batch, hidden_dim)

        # 9. 用户投影
        user_repr = self.user_projection(user_repr)

        return user_repr
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 1. 编码用户序列
        user_repr = self.encode_sequence(
            batch['history_text_feat'],
            batch['history_vision_feat'],
            batch['history_len']
        )
        
        # 2. 计算 logits
        if self._all_item_repr is not None:
            temperature = getattr(self.config, 'logit_temperature', 0.1)
            user_repr_norm = F.normalize(user_repr, dim=-1)
            logits = torch.matmul(user_repr_norm, self._all_item_repr.T) / temperature
        else:
            logits = None
        
        return {
            'user_repr': user_repr,
            'logits': logits,
            'target_item': batch['target_item']
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算 Cross Entropy 损失"""
        logits = outputs['logits']
        target_items = outputs['target_item']
        
        ce_loss = F.cross_entropy(logits, target_items)
        
        return {
            'ce_loss': ce_loss,
            'total_loss': ce_loss
        }
    
    # ==================== AbstractTrainableModel 实现 ====================

    def _get_optimizer(self, stage_id: int, stage_kwargs: Dict) -> torch.optim.Optimizer:
        lr = stage_kwargs.get('lr', 0.001)
        weight_decay = stage_kwargs.get('weight_decay', 0.01)

        # 重要：只训练序列编码器，冻结 item_encoder
        # 因为物品表征是预计算的，如果 item_encoder 参数更新，
        # 预计算的表征就会与当前模型不一致
        seq_params = []
        for name, param in self.named_parameters():
            if name.startswith('item_encoder'):
                param.requires_grad = False  # 冻结 item_encoder
            else:
                seq_params.append(param)

        return torch.optim.AdamW(seq_params, lr=lr, weight_decay=weight_decay)

    def _get_optimizer_state_dict(self) -> Dict:
        """获取优化器状态"""
        optimizer_states = {}
        for stage_id, optimizer in self._stage_optimizers.items():
            optimizer_states[stage_id] = optimizer.state_dict()
        return optimizer_states

    def _load_optimizer_state_dict(self, state_dict: Dict):
        """加载优化器状态"""
        for stage_id, opt_state in state_dict.items():
            if stage_id in self._stage_optimizers:
                self._stage_optimizers[stage_id].load_state_dict(opt_state)

    def _get_scheduler(self, optimizer, stage_id: int, stage_kwargs: Dict):
        return None
    
    def _train_one_batch(self, batch: Any, stage_id: int, stage_kwargs: Dict) -> Tuple[torch.Tensor, Dict]:
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
        """验证 - Full Ranking"""
        self.eval()
        
        all_item_features = stage_kwargs.get('all_item_features', None)
        if all_item_features is None:
            raise ValueError("需要提供 all_item_features")
        
        # 确保物品表征已预计算
        if self._all_item_repr is None:
            self.set_all_item_features(all_item_features)
        
        all_ranks = []
        
        from tqdm import tqdm
        val_pbar = tqdm(val_dataloader, desc="Validate", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                batch_size = batch['history_text_feat'].shape[0]
                target_items = batch['target_item']
                
                # 编码用户
                user_repr = self.encode_sequence(
                    batch['history_text_feat'],
                    batch['history_vision_feat'],
                    batch['history_len']
                )
                
                # 计算分数
                temperature = getattr(self.config, 'logit_temperature', 0.1)
                user_repr_norm = F.normalize(user_repr, dim=-1)
                all_scores = torch.matmul(user_repr_norm, self._all_item_repr.T) / temperature
                
                # 计算排名
                target_scores = all_scores[torch.arange(batch_size, device=self.device), target_items]
                ranks = (all_scores >= target_scores.unsqueeze(1)).sum(dim=1)
                all_ranks.append(ranks.cpu())
        
        all_ranks = torch.cat(all_ranks, dim=0).float()
        
        # 计算指标
        metrics = self._compute_metrics(all_ranks)
        return metrics
    
    def _compute_metrics(self, ranks: torch.Tensor, k_list: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        
        for k in k_list:
            hits = (ranks <= k).float()
            metrics[f'HR@{k}'] = hits.mean().item()
            
            dcg = 1.0 / torch.log2(ranks.clamp(min=1).float() + 1)
            dcg = torch.where(ranks <= k, dcg, torch.zeros_like(dcg))
            metrics[f'NDCG@{k}'] = dcg.mean().item()
            
            rr = 1.0 / ranks.clamp(min=1).float()
            rr = torch.where(ranks <= k, rr, torch.zeros_like(rr))
            metrics[f'MRR@{k}'] = rr.mean().item()
        
        metrics['MRR'] = (1.0 / ranks.clamp(min=1).float()).mean().item()
        metrics['Mean_Rank'] = ranks.mean().item()
        
        return metrics


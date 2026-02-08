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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, List
import numpy as np

from base_model import AbstractTrainableModel
from tqdm import tqdm

from utils.utils import check_tensor


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
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(config.visual_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
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

        check_tensor(text_feat, "SimpleItemEncoder", "输入text_feat")
        check_tensor(vision_feat, "SimpleItemEncoder", "输入vision_feat")
        
        # 编码各模态
        text_encoded = self.text_encoder(text_feat)
        visual_encoded = self.visual_encoder(vision_feat)

        check_tensor(text_encoded, "SimpleItemEncoder", "编码后text_encoded")
        check_tensor(visual_encoded, "SimpleItemEncoder", "编码后visual_encoded")
        
        # 加权融合
        weights = F.softmax(self.modal_weight, dim=0)
        item_emb = weights[0] * text_encoded + weights[1] * visual_encoded

        check_tensor(weights, "SimpleItemEncoder", "模态权重weights")
        check_tensor(item_emb, "SimpleItemEncoder", "融合后item_emb")
        
        return item_emb


class PureSASRec(AbstractTrainableModel):
    """纯净的多模态 SASRec
    
    用于验证基础骨架是否正常工作
    """
    
    def __init__(
        self,
        config,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(device=device)
        self.num_items = None
        self.all_item_vision_feat = None
        self.all_item_text_feat = None
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

        # 校验参数
        if self.hidden_dim % self.config.attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by attention_heads ({self.config.attention_heads})")
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        self.input_layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # ===== 投影层 =====
        self.user_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        
        self.item_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
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

        # 初始化物品编码器（SimpleItemEncoder）
        for m in self.item_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        for module in [self.user_projection, self.item_projection]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        # 新增：初始化TransformerBlock
        for block in self.transformer_blocks:
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device,seq_lens: torch.Tensor) -> torch.Tensor:
        """
    获取因果掩码（适配左对齐+解决NaN+保留缓存）
    Args:
        seq_len: 序列总长度
        device: 设备
        seq_lens: 可选，每个样本的有效长度（左对齐必须传）
    Returns:
        causal_mask: 左对齐→[batch_size, seq_len, seq_len]
    """
        #  左对齐场景（传seq_lens）：生成适配左对齐的掩码（保留缓存+解决NaN）
        batch_size = seq_lens.shape[0]
        cache_key = (seq_len, str(device), tuple(seq_lens.cpu().numpy()))  # 按有效长度缓存
        if cache_key not in self._causal_mask_cache:
            # 步骤1：生成右对齐基础掩码
            base_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-1e4'),
                diagonal=1
            )
            # 步骤2：生成左对齐专属掩码（适配每个样本的有效长度）
            batch_mask = []
            for valid_len in seq_lens:
                # 左对齐：有效起始位置 = seq_len - 有效长度
                pad_start = max(0, seq_len - valid_len.item())
                # 初始化全禁止（-1e9）
                left_mask = torch.full((seq_len, seq_len), -1e4, device=device)
                # 有效区域：pad_start到seq_len-1，应用因果约束（下三角允许）
                if valid_len > 0:
                    # 提取有效区域的基础掩码，翻转后填充
                    valid_mask = base_mask[pad_start:, pad_start:]
                    left_mask[pad_start:, pad_start:] = valid_mask
                batch_mask.append(left_mask)
            causal_mask = torch.stack(batch_mask)  # [batch_size, seq_len, seq_len]
            # 最终防护：过滤NaN/Inf，确保无异常值
            causal_mask = torch.nan_to_num(causal_mask, nan=-1e9, posinf=1e9, neginf=-1e9)
            self._causal_mask_cache[cache_key] = causal_mask

        return self._causal_mask_cache[cache_key]
    
    def set_all_item_features(self, all_item_features: Dict[str, torch.Tensor]):
        """初始化所有物品的原始特征（训练前调用）"""
        self.eval()
        self.all_item_text_feat = all_item_features['text'].to(self.device)
        self.all_item_vision_feat = all_item_features['visual'].to(self.device)
        self.num_items = all_item_features['num_items']
        # 首次更新缓存
        self.update_item_repr_cache()

    def update_item_repr_cache(self):
        """更新物品表征缓存（每epoch训练后调用，保证和item_encoder同步）"""
        self.eval()
        with torch.no_grad():
            # 从config读取批次大小，无则用默认值
            item_batch_size = getattr(self.config, 'item_batch_size', 1024)
            item_repr_list = []
            for start in range(0, self.num_items, item_batch_size):
                end = min(start + item_batch_size, self.num_items)
                text_feat = self.all_item_text_feat[start:end]
                vision_feat = self.all_item_vision_feat[start:end]
                item_emb = self.item_encoder(text_feat, vision_feat)
                item_repr = self.item_projection(item_emb)
                item_repr_list.append(F.normalize(item_repr, dim=-1))
            self._all_item_repr = torch.cat(item_repr_list, dim=0)
        self.train()
    
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

        check_tensor(text_feat, "PureSASRec/encode_sequence", "输入history_text_feat")
        check_tensor(vision_feat, "PureSASRec/encode_sequence", "输入history_vision_feat")
        check_tensor(seq_lens, "PureSASRec/encode_sequence", "输入seq_lens")

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
        # 初始化全0的pos_emb（padding位置天然为0）
        pos_emb = torch.zeros((batch_size, seq_len, self.config.hidden_dim), device=device)
        # 计算相对位置：有效位置从0开始计数，padding位置设为-1
        relative_pos = positions - pad_start
        # 只提取有效位置计算pos_emb
        valid_mask = (relative_pos > -1)  # (batch, seq_len)
        valid_indices = relative_pos[valid_mask]  # 提取所有有效位置的relative_pos
        pos_emb[valid_mask] = self.pos_emb(valid_indices)


        seq_emb = item_emb + pos_emb  # 最终序列embedding

        check_tensor(pos_emb, "PureSASRec/encode_sequence", "pos_emb")
        check_tensor(seq_emb, "PureSASRec/encode_sequence", "seq_emb（item+pos）")

        # 4. LayerNorm + Dropout
        seq_emb = self.input_layer_norm(seq_emb)
        seq_emb = self.dropout(seq_emb)
        check_tensor(seq_emb, "PureSASRec/encode_sequence", "seq_emb（LayerNorm+Dropout）")
        # 5. Causal mask
        causal_mask = self._get_causal_mask(seq_len, device, seq_lens)

        # 新增head维度 → [batch, 1, seq_len, seq_len]
        causal_mask = causal_mask.unsqueeze(1)
        # 在head维度重复num_heads次 → [batch, num_heads, seq_len, seq_len]
        causal_mask = causal_mask.repeat(1, self.config.attention_heads, 1, 1)
        # 展平为[batch×num_heads, seq_len, seq_len] → 每个样本的num_heads个head都复用该样本的掩码
        causal_mask = causal_mask.view(batch_size * self.config.attention_heads, seq_len, seq_len)

        # 6. Transformer
        attn_weight_list = []
        for block in self.transformer_blocks:
            seq_emb, attn_weight = block(seq_emb, padding_mask=padding_mask, causal_mask=causal_mask)
            attn_weight_list.append(attn_weight)
            check_tensor(seq_emb, "PureSASRec/encode_sequence", f"TransformerBlock输出")

        # self.check_attention_weights(attn_weight_list[0], seq_lens, seq_len)
        # 处理可能的 NaN
        seq_emb = torch.nan_to_num(seq_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # 7. 获取用户表示（最后一个位置，即序列末尾）
        # 正确提取最后一个有效位置的表征
        # 计算每个样本的最后有效位置索引（左Padding下，有效位置结束于seq_len-1，长度为seq_lens）
        # 左Padding的序列中，有效内容在末尾，最后一个有效位置永远是seq_len-1
        last_valid_indices = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)
        # 批量索引
        batch_idx = torch.arange(batch_size, device=device)
        # 提取最后有效位置的表征
        user_repr = seq_emb[batch_idx, last_valid_indices]
        check_tensor(user_repr, "PureSASRec/encode_sequence", "原始user_repr")
        # 8. 用户投影
        user_repr = self.user_projection(user_repr)
        check_tensor(user_repr, "PureSASRec/encode_sequence", "投影后user_repr")

        return user_repr

    def check_attention_weights(self, attn_weights, seq_lens, seq_len, batch_idx=0):
        """
        修复后的注意力权重排查函数
        Args:
            attn_weights: 自定义Attention返回的权重 [batch, num_heads, seq_len, seq_len]
            seq_lens: 每个样本的有效长度 [batch]
            seq_len: 总长度
        """
        # 1. 分离梯度+移到CPU+转numpy（避免梯度干扰）
        attn_weights_np = attn_weights[batch_idx].cpu().detach().numpy()  # [num_heads, seq_len, seq_len]
        # 取所有head的平均权重（消除单head波动）
        avg_attn_weights = attn_weights_np.mean(axis=0)  # [seq_len, seq_len]

        # 2. 获取当前样本的有效信息
        valid_len = seq_lens[batch_idx].item()
        pad_start = seq_len - valid_len  # 有效起始位置
        last_valid_pos = seq_len - 1  # 最后有效位置（比如15）

        print(
            f"\n================================================== 注意力权重排查 ==================================================")
        print(f"第一个样本：有效长度={valid_len}，有效起始位置={pad_start}")
        print(
            f"最后有效位置({last_valid_pos})的原始权重范围：min={avg_attn_weights[last_valid_pos].min():.6f}, max={avg_attn_weights[last_valid_pos].max():.6f}")

        # 3. 核心修复：过滤浮点误差（只保留>1e-3的权重，排除PAD的浮点噪声）
        last_pos_weights = avg_attn_weights[last_valid_pos].copy()
        last_pos_weights[last_pos_weights < 1e-3] = 0.0  # 屏蔽浮点误差的PAD权重

        # 4. 强制清空PAD区域的权重（兜底）
        last_pos_weights[:pad_start] = 0.0  # PAD位置（0~pad_start-1）权重置0

        # 5. 筛选有效关注位置（只保留非0权重的位置）
        non_zero_idx = np.where(last_pos_weights > 0)[0]

        # 6. 验证是否在有效区域内
        is_in_valid = all([idx >= pad_start for idx in non_zero_idx])

        # 7. 打印结果
        print(f"最后有效位置({last_valid_pos})关注的位置：{non_zero_idx}")
        print(f"关注位置是否在有效区域内：{is_in_valid}")
        return is_in_valid

    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 1. 编码用户序列
        user_repr = self.encode_sequence(
            batch['history_text_feat'],
            batch['history_vision_feat'],
            batch['history_len']
        )
        user_repr_norm = F.normalize(user_repr, dim=-1)
        check_tensor(user_repr_norm, "PureSASRec/forward", "user_repr_norm")
        check_tensor(self._all_item_repr, "PureSASRec/forward", "_all_item_repr")
        
        # 2. 计算 logits
        logits = torch.matmul(user_repr_norm, self._all_item_repr.T) / self.config.logit_temperature
        check_tensor(logits, "PureSASRec/forward", f"logits（temperature={self.config.logit_temperature}）")
        return {
            'user_repr': user_repr,
            'logits': logits,
            'target_item': batch['target_item']
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算 Cross Entropy 损失"""
        logits = outputs['logits']
        target_items = outputs['target_item']
        check_tensor(logits, "PureSASRec/compute_loss", "logits（Loss输入）")
        check_tensor(target_items, "PureSASRec/compute_loss", "target_items")
        
        ce_loss = F.cross_entropy(logits, target_items)
        check_tensor(ce_loss, "PureSASRec/compute_loss", "ce_loss")
        return {
            'ce_loss': ce_loss,
            'total_loss': ce_loss
        }
    
    # ==================== AbstractTrainableModel 实现 ====================

    def _get_optimizer(self, stage_id: int, stage_kwargs: Dict) -> torch.optim.Optimizer:
        lr = stage_kwargs.get('lr', 0.001)
        weight_decay = stage_kwargs.get('weight_decay', 0.01)

        # 训练所有参数（包括item_encoder）
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # 实现调度器创建（核心：Warmup + 余弦退火）

    def _get_scheduler(self, optimizer: torch.optim.Optimizer, stage_id: int,
                       stage_kwargs: Dict) -> torch.optim.lr_scheduler.LRScheduler:
        """
        创建Warmup+余弦退火调度器
        - 前warmup_epochs轮：学习率从0.1*lr线性升到lr
        - 剩余轮数：余弦退火到eta_min
        """
        warmup_epochs = stage_kwargs.get('warmup_epochs', 5)
        eta_min = stage_kwargs.get('eta_min', 1e-5)
        total_epochs = stage_kwargs.get('total_epochs', 50)  # 当前阶段总轮数

        # 定义Warmup和余弦退火调度器
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,  # 初始学习率=0.1*lr
            total_iters=warmup_epochs
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,  # 余弦退火的总轮数
            eta_min=eta_min  # 最小学习率
        )
        # 组合调度器
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs]  # warmup结束后切换到余弦退火
        )
        return scheduler

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

    
    def _train_one_batch(self, batch: Any, stage_id: int, stage_kwargs: Dict) -> Tuple[torch.Tensor, Dict]:
        outputs = self.forward(batch)
        losses = self.compute_loss(outputs)
        metrics = {'ce_loss': losses['ce_loss'].item()}
        return losses['total_loss'], metrics

    def on_epoch_end(self, epoch: int, stage_id: int, stage_kwargs: Dict, train_metrics: Dict, val_metrics: Dict):
        self.update_item_repr_cache()
        super().on_epoch_end(epoch, stage_id, stage_kwargs, train_metrics, val_metrics)
    
    def _validate_one_epoch(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        stage_id: int,
        stage_kwargs: Dict
    ) -> Dict:
        """验证 - Full Ranking"""
        self.eval()
        
        all_ranks = []
        

        val_pbar = tqdm(val_dataloader, desc="Validate", leave=False, total=len(val_dataloader))
        
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
                user_repr_norm = F.normalize(user_repr, dim=-1)

                # 用缓存计算全量分数（RecBole逻辑）
                all_scores = torch.matmul(user_repr_norm, self._all_item_repr.T) / self.config.logit_temperature

                # 计算排名
                # 目标物品分数（用验证过的simple_target_scores）
                batch_idx = torch.arange(len(target_items), device=self.device)
                valid_mask = (target_items >= 0) & (target_items < self.num_items)
                target_scores = torch.zeros(len(target_items), device=self.device)
                target_scores[valid_mask] = all_scores[batch_idx[valid_mask], target_items[valid_mask]]

                # 正确计算排名（用argsort，避免分数相等导致的错误）
                # 步骤1：对all_scores降序排序，得到每个位置的物品索引
                sorted_indices = torch.argsort(all_scores, dim=1, descending=True)  # [batch, 6370]
                # 步骤2：遍历每个样本，找到目标物品的排名（仅有效样本）
                ranks = torch.ones(len(target_items), device=self.device) * self.num_items  # 无效样本默认排最后
                for i in range(len(target_items)):
                    if valid_mask[i]:
                        # 找到目标物品在排序后的位置（+1=排名）
                        rank_idx = (sorted_indices[i] == target_items[i]).nonzero(as_tuple=True)[0]
                        ranks[i] = rank_idx.item() + 1  # 排名从1开始
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

class TransformerBlock(nn.Module):
    """Transformer Block for SASRec - Pre-LN架构（更稳定）

    支持因果掩码和填充掩码的组合使用
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float, **kwargs):
        super().__init__()
        # self.attention = nn.MultiheadAttention(
        #     hidden_size, num_heads, dropout=dropout_rate,
        #     batch_first=True
        # )
        self.attention = CustomMultiheadSelfAttention(
            hidden_size, num_heads, dropout_rate,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, padding_mask=None, causal_mask=None):
        """前向传播

        Args:
            x: [batch_size, seq_len, hidden_size] 输入序列
            padding_mask: [batch_size, seq_len] 填充掩码，True表示padding位置
            causal_mask: [seq_len, seq_len] 因果掩码，上三角为-inf

        Returns:
            x: [batch_size, seq_len, hidden_size] 输出序列
        """
        check_tensor(x, "TransformerBlock", "输入x")

        # Pre-LN架构：先LayerNorm再attention（更稳定，防止梯度爆炸）
        # Self-attention with residual connection
        normed_x = self.layer_norm1(x)

        # 同时使用因果掩码和填充掩码
        # attn_mask: [seq_len, seq_len] 用于因果掩码
        # key_padding_mask: [batch_size, seq_len] 用于填充掩码
        # attn_output, attn_weight = self.attention(
        #     normed_x, normed_x, normed_x,
        #     attn_mask=causal_mask,
        #     key_padding_mask=padding_mask,
        #     need_weights=True,
        #     is_causal=True
        # )
        attn_output, attn_weight = self.attention(
            normed_x, causal_mask, padding_mask
        )
        check_tensor(attn_output, "TransformerBlock", "attn_output")
        # 处理Attention输出的NaN/Inf（避免梯度爆炸）
        attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=0.0, neginf=0.0)
        x = x + self.dropout(attn_output)

        # Feed forward with residual connection
        normed_x = self.layer_norm2(x)
        ff_output = self.feed_forward(normed_x)
        # 新增：处理FeedForward输出的NaN/Inf
        ff_output = torch.nan_to_num(ff_output, nan=0.0, posinf=0.0, neginf=0.0)
        check_tensor(ff_output, "TransformerBlock", "ff_output")
        x = x + ff_output
        check_tensor(x, "TransformerBlock", "输出x")
        return x, attn_weight


class CustomMultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim  # 总隐藏维度
        self.num_heads = num_heads  # 注意力头数
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first  # 适配你的输入维度[batch, seq_len, hidden]

        # 每个head的维度
        self.d_k = embed_dim // num_heads
        assert self.d_k * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # Q/K/V的线性层（自注意力：共享权重）
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        # 输出线性层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        自定义自注意力前向传播
        Args:
            x: 输入特征 [batch, seq_len, embed_dim]（batch_first=True）
            attn_mask: 因果掩码 [batch×num_heads, seq_len, seq_len] 或 [seq_len, seq_len]
            key_padding_mask: PAD掩码 [batch, seq_len]（bool型，True=PAD）
        Returns:
            attn_output: 注意力输出 [batch, seq_len, embed_dim]
            attn_weights: 注意力权重 [batch, num_heads, seq_len, seq_len]
        """
        # 1. 输入维度校验（确保batch_first）
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape
            x = x.transpose(0, 1)  # 转成[batch, seq_len, embed_dim]

        # 2. 线性层拆分Q/K/V → [batch, seq_len, 3*embed_dim]
        qkv = self.qkv_proj(x)
        # 拆分Q/K/V → 各[batch, seq_len, embed_dim]
        q, k, v = torch.split(qkv, self.embed_dim, dim=-1)

        # 3. 维度变换：[batch, seq_len, embed_dim] → [batch, num_heads, seq_len, d_k]
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 4. 强制清空PAD位置的Q/K/V（左对齐核心！）
        if key_padding_mask is not None:
            # 扩展mask维度：[batch, seq_len] → [batch, 1, seq_len, 1]
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)
            q = q.masked_fill(pad_mask, 0.0)
            k = k.masked_fill(pad_mask, 0.0)
            v = v.masked_fill(pad_mask, 0.0)

        # 5. 计算注意力分数：Q @ K^T / sqrt(d_k) → [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 6. 应用因果掩码（causal_mask）
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                # 3维掩码：[batch×num_heads, seq_len, seq_len] → 转成[batch, num_heads, seq_len, seq_len]
                attn_mask = attn_mask.view(batch_size, self.num_heads, seq_len, seq_len)
            elif attn_mask.dim() == 2:
                # 2维掩码：[seq_len, seq_len] → 广播到[batch, num_heads, seq_len, seq_len]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.num_heads, 1, 1)
            # 应用掩码（分数+掩码值）
            scores = scores + attn_mask

        # 7. 应用key_padding_mask（屏蔽PAD的key）
        if key_padding_mask is not None:
            # 扩展mask维度：[batch, seq_len] → [batch, 1, 1, seq_len]
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, -1e4)  # PAD位置分数置为-1e4

        # 8. 计算注意力权重（softmax + dropout）
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 验证有效区域权重：打印第一个样本head0的有效区域权重

        # 9. 计算注意力输出：权重 @ V → [batch, num_heads, seq_len, d_k]
        attn_output = torch.matmul(attn_weights, v)

        # 10. 拼接heads：[batch, num_heads, seq_len, d_k] → [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # 11. 输出线性层
        attn_output = self.out_proj(attn_output)

        # 12. 还原维度（如果batch_first=False，这里需要转回，但你用True）
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights


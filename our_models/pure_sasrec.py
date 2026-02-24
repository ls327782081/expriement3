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

        # 【极简编码器】只保留一层线性，不压特征
        self.text_proj = nn.Linear(config.text_dim, config.hidden_dim)
        self.visual_proj = nn.Linear(config.visual_dim, config.hidden_dim)

        # 简单加权，不学死
        self.modal_weight = nn.Parameter(torch.ones(2))
        
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

        # ===== 新增：输入特征兜底清理 =====
        text_feat = torch.nan_to_num(text_feat, nan=0.0, posinf=0.0, neginf=0.0)
        vision_feat = torch.nan_to_num(vision_feat, nan=0.0, posinf=0.0, neginf=0.0)

        # 【关键】只做线性投影，不 normalize、不缩放、不压方差
        text_h = self.text_proj(text_feat)
        vision_h = self.visual_proj(vision_feat)

        # 简单加权融合
        w = F.softmax(self.modal_weight, dim=0)
        item_emb = w[0] * text_h + w[1] * vision_h

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

        # ===== 核心修改：纯ID嵌入（替换多模态ItemEncoder）=====
        self.item_embedding = nn.Embedding(self.config.num_items, config.hidden_dim, padding_idx=0)
        
        # ===== 序列编码器 (SASRec) =====
        self.pos_emb = nn.Embedding(self.max_seq_len + 1, config.hidden_dim)
        
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

        
        # ===== 预计算的物品表征 =====
        self._all_item_repr = None
        
        # 缓存因果掩码
        self._causal_mask_cache = {}
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        # 新增：初始化item_embedding
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.05)
        # 处理padding_idx（如果有）
        if self.item_embedding.padding_idx is not None:
            nn.init.constant_(self.item_embedding.weight[self.item_embedding.padding_idx], 0.0)

        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.05)
        nn.init.constant_(self.input_layer_norm.weight, 1.0)
        nn.init.constant_(self.input_layer_norm.bias, 0.0)

        # 新增：初始化TransformerBlock
        for block in self.transformer_blocks:
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.05)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def _get_causal_mask(self, total_seq_len: int, device: torch.device, valid_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        获取因果掩码（适配左对齐+增强掩码值）
        Args:
            total_seq_len: 序列总长度（如50）
            device: 设备
            valid_seq_lens: 每个样本的有效长度（左对齐必须传）
        Returns:
            causal_mask: [batch_size, total_seq_len, total_seq_len]
        """
        batch_size = valid_seq_lens.shape[0]
        # 强制修正有效长度
        valid_seq_lens = valid_seq_lens.clamp(min=1, max=total_seq_len)

        # ===== 核心修改：增强掩码值（从-1e9→-1e4）=====
        mask_value = -1e4

        # 1. 生成标准因果掩码（上三角为-1e4）→ 只禁止关注未来位置
        causal_mask = torch.triu(torch.ones(total_seq_len, total_seq_len, device=device) * mask_value, diagonal=1)
        # 2. 扩展到batch维度
        causal_mask = causal_mask.unsqueeze(0).repeat(batch_size, 1, 1)

        # 兜底：过滤NaN/Inf（修正posinf值为mask_value）
        causal_mask = torch.nan_to_num(causal_mask, nan=mask_value, posinf=mask_value, neginf=mask_value)
        return causal_mask
    
    def set_all_item_features(self, all_item_features: Dict[str, torch.Tensor]):
        """初始化所有物品的原始特征（训练前调用）"""
        self.eval()
        # self.all_item_text_feat = all_item_features['text'].to(self.device)
        # self.all_item_vision_feat = all_item_features['visual'].to(self.device)
        self.num_items = all_item_features['num_items']

        # 校验：item_embedding的num_embeddings必须≥target_item的最大值
        if self.item_embedding.num_embeddings < self.num_items:
            raise ValueError(
                f"item_embedding的num_embeddings({self.item_embedding.num_embeddings}) "
                f"小于物品总数({self.num_items})！"
            )

        self.train()

        # 打印校验信息
        print(f"\n===== 纯ID版物品表征校验 =====")
        print(f"物品总数：{self.num_items}")
        print(f"item_embedding形状：{self.item_embedding.weight.shape}")

    # def update_item_repr_cache(self):
    #     """更新物品表征缓存（每epoch训练后调用，保证和item_encoder同步）"""
    #     self.eval()
    #     with torch.no_grad():
    #         # 仅在需要时重新计算（可选，保证缓存和item_embedding同步）
    #         item_ids = torch.arange(self.num_items, device=self.device)
    #         self._all_item_repr = self.item_embedding(item_ids)
    #         # 强制缩放到单位范数
    #         norm = self._all_item_repr.norm(dim=-1, keepdim=True)
    #         self._all_item_repr = self._all_item_repr / (norm + 1e-8)
    #         # 从config读取批次大小，无则用默认值
    #         # item_batch_size = getattr(self.config, 'item_batch_size', 1024)
    #         # item_repr_list = []
    #         # for start in range(0, self.num_items, item_batch_size):
    #         #     end = min(start + item_batch_size, self.num_items)
    #         #     text_feat = self.all_item_text_feat[start:end]
    #         #     vision_feat = self.all_item_vision_feat[start:end]
    #         #     item_emb = self.item_encoder(text_feat, vision_feat)
    #         #     item_repr = self.item_projection(item_emb)
    #         #     # 计算当前范数
    #         #     norm = item_repr.norm(dim=-1, keepdim=True)
    #         #     # 对范数过小的表征，缩放到平均范数1.0
    #         #     scale = torch.where(norm < 0.1, 1.0 / (norm + 1e-8), 1.0)
    #         #     item_repr = item_repr * scale
    #         #
    #         #     item_repr_list.append(item_repr)
    #         # self._all_item_repr = torch.cat(item_repr_list, dim=0)
    #
    #
    #     self.train()

    def encode_sequence(
            self,
            batch_item_ids: torch.Tensor,
            seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """编码用户历史序列

        Returns:
            user_repr: (batch, hidden_dim)
        """
        batch_size, total_seq_len = batch_item_ids.shape  # 重命名：total_seq_len=50（总长度）
        device = batch_item_ids.device

        # ===== 1. 强制修正seq_lens（有效长度）=====
        # 限制有效长度在1~total_seq_len之间
        valid_seq_lens = seq_lens.clamp(min=1, max=total_seq_len)

        r = torch.rand(1).item()

        # 关键调试：打印总长度和有效长度
        if self.training and r < 0.01:
            print(f"\n===== 序列长度调试 =====")
            print(f"序列总长度(total_seq_len)：{total_seq_len}")
            print(f"第一个样本有效长度(valid_seq_lens)：{valid_seq_lens[0].item()}")
            print(f"有效长度范围：min={valid_seq_lens.min().item()}, max={valid_seq_lens.max().item()}")

        # ===== 2. 正确计算pad_start（左对齐PAD的起始位置）=====
        # 左对齐逻辑：PAD在序列左侧，有效内容在右侧
        # 示例：total_seq_len=50，valid_len=18 → pad_start=50-18=32 → 前32位PAD，后18位有效
        pad_start = total_seq_len - valid_seq_lens.unsqueeze(1)  # [batch, 1]
        pad_start = pad_start.clamp(min=0, max=total_seq_len)  # 防止负数/越界

        # ===== 3. 正确生成padding_mask =====
        # positions：[1,2,...,50] → 对应序列位置1~50
        positions = torch.arange(1, total_seq_len + 1, device=device).unsqueeze(0)  # [1, 50]
        # padding_mask：True=PAD（位置≤pad_start），False=有效（位置>pad_start）
        padding_mask = positions <= pad_start

        # ===== 4. 关键校验：有效位置数必须等于有效长度 =====
        if self.training and r < 0.01:
            first_valid_len = valid_seq_lens[0].item()
            first_pad_start = pad_start[0].item()
            first_padding_mask = padding_mask[0]
            actual_valid_count = (~first_padding_mask).sum().item()

            print(f"\n===== Padding Mask 最终调试 =====")
            print(f"序列总长度：{total_seq_len}")
            print(f"样本有效长度：{first_valid_len}")
            print(f"PAD起始位置：{first_pad_start}")
            print(f"padding_mask前10位：{first_padding_mask[:10]}")
            print(f"padding_mask后10位：{first_padding_mask[-10:]}")  # 新增：打印最后10位（有效区域）
            print(f"计算的有效位置数：{actual_valid_count}")
            print(f"预期的有效位置数：{first_valid_len}")

            # 强制校验：不匹配则报错，直接定位问题
            assert actual_valid_count == first_valid_len, \
                f"有效位置数不匹配！计算值={actual_valid_count}，预期值={first_valid_len}"

        # ===== 5. 物品编码（纯ID嵌入）=====
        item_emb = self.item_embedding(batch_item_ids)

        # ===== 6. 位置嵌入（和RecBole对齐：从1开始，不屏蔽PAD）=====
        # RecBole：pos_indices从1开始（1~total_seq_len）
        pos_indices = torch.arange(1, total_seq_len + 1, device=device)  # [50]
        pos_emb = self.pos_emb(pos_indices)  # [50, hidden_dim]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, 50, hidden_dim]
        # 移除：屏蔽PAD位置的位置嵌入（RecBole不做这一步）
        # pos_emb = pos_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        seq_emb = item_emb + pos_emb

        # ===== 7. LayerNorm + Dropout =====
        seq_emb = self.input_layer_norm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        # ===== 8. 因果掩码（使用修正后的有效长度）=====
        causal_mask = self._get_causal_mask(total_seq_len, device, valid_seq_lens)  # [batch, 50, 50]

        # ===== 9. Transformer编码 =====
        for block in self.transformer_blocks:
            seq_emb, _ = block(seq_emb, padding_mask=padding_mask, causal_mask=causal_mask)
        seq_emb = torch.nan_to_num(seq_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # ===== 10. 获取最后有效位置（左对齐逻辑）=====
        last_valid_indices = []
        for i in range(batch_size):
            valid_len = valid_seq_lens[i].item()
            # 左对齐：最后有效位置 = pad_start + valid_len - 1 = (50-18) + 18 -1 = 49
            last_valid_idx = pad_start[i].item() + valid_len - 1
            # 兜底：防止越界
            last_valid_idx = min(max(last_valid_idx, 0), total_seq_len - 1)
            last_valid_indices.append(last_valid_idx)

        last_valid_indices = torch.tensor(last_valid_indices, device=device, dtype=torch.long)
        batch_idx = torch.arange(batch_size, device=device)
        user_repr = seq_emb[batch_idx, last_valid_indices]

        # 归一化
        # norm = user_repr.norm(dim=-1, keepdim=True)
        # user_repr = user_repr / (norm + 1e-8)

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
        # 核心修改：输入是item_id序列，而非text/vision特征
        user_repr = self.encode_sequence(
            batch['history_items'],  # 改为item_id序列
            batch['history_len']
        )

        # RecBole：动态获取所有item embedding，不预计算、不归一化
        all_item_emb = self.item_embedding.weight  # [num_items, hidden_dim]
        # 移除温度系数（RecBole的SASRec无温度系数）
        logits = user_repr @ all_item_emb.T

        return {
            'user_repr': user_repr,
            'logits': logits,
            'target_item': batch['target_item']
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = outputs['logits']
        target_items = outputs['target_item']
        check_tensor(logits, "PureSASRec/compute_loss", "logits（Loss输入）")
        check_tensor(target_items, "PureSASRec/compute_loss", "target_items")

        # RecBole核心逻辑：过滤target_item=0（padding）
        valid_mask = (target_items != 0)  # 仅保留非padding的target
        if not valid_mask.any():
            # 无有效样本时返回0损失（避免报错）
            return {'ce_loss': torch.tensor(0.0, device=logits.device),
                    'total_loss': torch.tensor(0.0, device=logits.device)}

        # 仅计算有效样本的Loss
        ce_loss = F.cross_entropy(logits[valid_mask], target_items[valid_mask])
        check_tensor(ce_loss, "PureSASRec/compute_loss", "ce_loss")
        return {
            'ce_loss': ce_loss,
            'total_loss': ce_loss
        }
    
    # ==================== AbstractTrainableModel 实现 ====================

    def _get_optimizer(self, stage_id: int, stage_kwargs: Dict) -> torch.optim.Optimizer:
        lr = stage_kwargs.get('lr',  1e-4)
        weight_decay = stage_kwargs.get('weight_decay', 0.0)

        print(f"\n===== 优化器参数 =====")
        print(f"stage_kwargs['lr']: {stage_kwargs.get('lr', '未传，使用默认5e-4')}")
        print(f"实际使用的base_lr: {lr:.6f}")

        # 训练所有参数（包括item_encoder）
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # 实现调度器创建（核心：Warmup + 余弦退火）

    def _get_scheduler(self, optimizer: torch.optim.Optimizer, stage_id: int,
                       stage_kwargs: Dict) -> torch.optim.lr_scheduler.LRScheduler:
        """临时：关闭调度器，用固定学习率"""
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        # """
        # 创建Warmup+余弦退火调度器
        # - 前warmup_epochs轮：学习率从0.1*lr线性升到lr
        # - 剩余轮数：余弦退火到eta_min
        # """
        # lr = stage_kwargs.get('lr', 1e-4)
        # warmup_epochs = max(1, int(self.config.epochs * 0.15))  # 15%的轮数用于warmup
        # eta_min = stage_kwargs.get('eta_min', 1e-5)
        #
        # total_epochs = self.config.epochs
        # warmup_epochs = min(warmup_epochs, total_epochs - 1)  # 至少留1轮退火
        # cosine_T_max = max(1, (total_epochs - warmup_epochs) // 2)  # 退火轮数=剩余轮数的1/2
        #
        # # 3. 定义Warmup调度器（0.2*base_lr → base_lr，3轮完成）
        # scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        #     optimizer,
        #     start_factor=0.2,  # 初始学习率=0.2*0.0001=0.00002（而非0.1，避免过低）
        #     total_iters=warmup_epochs
        # )
        #
        # # 4. 定义余弦退火调度器（快速下降到eta_min）
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=cosine_T_max,  # 比如20轮总训练→(20-3)//2=8轮退火
        #     eta_min=eta_min
        # )
        #
        # # 5. 定义收尾调度器（稳定在eta_min）
        # scheduler_constant = torch.optim.lr_scheduler.ConstantLR(
        #     optimizer,
        #     factor=1.0,
        #     total_iters=total_epochs - warmup_epochs - cosine_T_max
        # )
        #
        # # 6. 组合调度器
        # scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer,
        #     schedulers=[scheduler_warmup, scheduler_cosine, scheduler_constant],
        #     milestones=[warmup_epochs, warmup_epochs + cosine_T_max]
        # )
        #
        # # 打印配置（验证参数是否匹配）
        # print(f"\n===== 调度器参数（base_lr=1e-4） =====")
        # print(f"warmup_epochs={warmup_epochs}, total_epochs={total_epochs}")
        # print(f"cosine_T_max={cosine_T_max}, eta_min={eta_min}")
        # print(f"Warmup初始LR：{0.2 * lr:.6f} → Warmup结束LR：{lr:.6f}")
        # print(f"余弦退火结束LR：{eta_min:.6f}")
        # return scheduler

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
        # self.update_item_repr_cache()
        super().on_epoch_end(epoch, stage_id, stage_kwargs, train_metrics, val_metrics)

    def _validate_one_epoch(
            self,
            val_dataloader: torch.utils.data.DataLoader,
            stage_id: int,
            stage_kwargs: Dict
    ) -> Dict:
        """验证 - Full Ranking（和RecBole完全对齐）"""
        self.eval()
        all_ranks = []
        val_pbar = tqdm(val_dataloader, desc="Validate", leave=False, total=len(val_dataloader))

        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # 1. 编码用户（和训练逻辑一致）
                user_repr = self.encode_sequence(
                    batch['history_items'],
                    batch['history_len']
                )

                # 2. 计算分数（和forward逻辑完全一致：无预计算、无温度系数、无归一化）
                all_item_emb = self.item_embedding.weight  # 动态获取
                all_scores = user_repr @ all_item_emb.T

                # 3. 计算排名（逻辑不变，仅优化校验）
                batch_size = len(batch['target_item'])
                batch_idx = torch.arange(batch_size, device=self.device)
                valid_mask = (batch['target_item'] >= 1) & (batch['target_item'] < self.num_items)

                # 初始化排名为物品总数（无效样本默认最后）
                ranks = torch.full((batch_size,), self.num_items, device=self.device, dtype=torch.float)

                if valid_mask.any():
                    # 仅计算有效样本的分数和排名
                    target_scores = all_scores[batch_idx[valid_mask], batch['target_item'][valid_mask]]
                    # 计算每个有效样本的排名（分数>目标分数的数量+1）
                    for i in batch_idx[valid_mask]:
                        i = i.item()
                        target = batch['target_item'][i].item()
                        rank = (all_scores[i] > all_scores[i][target]).sum().item() + 1
                        ranks[i] = rank

                all_ranks.append(ranks.cpu())

        all_ranks = torch.cat(all_ranks, dim=0).float()
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
            causal_mask: [batch_size, seq_len, seq_len] 因果掩码，上三角为-inf

        Returns:
            x: [batch_size, seq_len, hidden_size] 输出序列
        """
        check_tensor(x, "TransformerBlock", "输入x")

        # Self-attention with residual connection（Post-LN）
        attn_output, attn_weight = self.attention(
            x,  # 直接用x，不先做LayerNorm
            attn_mask=causal_mask,
            key_padding_mask=padding_mask
        )
        attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=0.0, neginf=0.0)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)  # Add之后做LayerNorm

        # Feed forward with residual connection（Post-LN）
        ff_output = self.feed_forward(x)
        ff_output = torch.nan_to_num(ff_output, nan=0.0, posinf=0.0, neginf=0.0)
        x = x + ff_output
        x = self.layer_norm2(x)  # Add之后做LayerNorm

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
            attn_mask: 因果掩码 [batch, seq_len, seq_len] 或 [seq_len, seq_len]
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

        # 4. 计算注意力分数：Q @ K^T / sqrt(d_k) → [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ===== 核心修复1：提前限制分数范围（缩小到合理区间）=====
        scores = torch.clamp(scores, min=-10.0, max=10.0)  # 从±1e4→±10，增强掩码效果

        # ===== 核心修复2：增强掩码值（从-1e9→-1e4）=====
        mask_value = -1e4

        # 5. 正确处理因果掩码（核心！）
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # 2维掩码：[seq_len, seq_len] → 广播到[batch, num_heads, seq_len, seq_len]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.num_heads, 1, 1)
            elif attn_mask.dim() == 3:
                # 3维掩码：[batch, seq_len, seq_len] → 广播到[batch, num_heads, seq_len, seq_len]
                attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            # 替换掩码值为增强版
            attn_mask = torch.where(attn_mask == -1e9, torch.tensor(mask_value, device=attn_mask.device), attn_mask)
            # 应用掩码（分数+掩码值）
            scores = scores + attn_mask

        # 6. 应用key_padding_mask（屏蔽PAD的key）
        if key_padding_mask is not None:
            # 扩展mask维度：[batch, seq_len] → [batch, 1, 1, seq_len]
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # 应用增强版掩码值
            scores = scores.masked_fill(pad_mask, mask_value)  # PAD位置分数置为-1e4

        # ===== 核心修复3：再次限制分数范围（确保掩码生效）=====
        scores = torch.clamp(scores, min=-10.0, max=10.0)

        # 7. 计算注意力权重（仅一次！）
        attn_weights = torch.softmax(scores, dim=-1)

        # ===== 核心修复4：强制清空PAD区域权重（兜底）=====
        if key_padding_mask is not None:
            # 扩展mask到[batch, num_heads, seq_len, seq_len]
            pad_mask_full = pad_mask.repeat(1, self.num_heads, seq_len, 1)
            attn_weights = attn_weights.masked_fill(pad_mask_full, 0.0)

        # 兜底：强制归一化（解决数值误差导致的权重和≠1）
        attn_weights = self.dropout(attn_weights)
        attn_weights_sum = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_weights = attn_weights / attn_weights_sum


        # 8. 计算注意力输出：权重 @ V → [batch, num_heads, seq_len, d_k]
        attn_output = torch.matmul(attn_weights, v)

        # 9. 拼接heads：[batch, num_heads, seq_len, d_k] → [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # 10. 输出线性层
        attn_output = self.out_proj(attn_output)

        # 调试信息（优化：打印有效区域的权重）
        p = torch.rand(1).item() < 0.01
        p = False
        if self.training and p:  # 1%概率打印，避免刷屏
            print("\n===== 注意力权重调试 =====")
            print(f"第一个样本head0的注意力权重形状：{attn_weights[0, 0].shape}")
            # 优化：打印最后10行最后10列（有效区域）
            print(f"第一个样本head0的注意力权重最后10行最后10列：\n{attn_weights[0, 0][-10:, -10:]}")
            print(
                f"第一个样本padding_mask前10位：{key_padding_mask[0][:10] if key_padding_mask is not None else 'None'}")
            # 新增：打印padding_mask最后10位
            print(
                f"第一个样本padding_mask最后10位：{key_padding_mask[0][-10:] if key_padding_mask is not None else 'None'}")
            print(f"注意力权重最大值：{attn_weights.max().item()}, 最小值：{attn_weights.min().item()}")
            print(f"第一个样本head0最后一行权重和：{attn_weights[0, 0, -1].sum().item():.4f}（预期≈1.0）")

        return attn_output, attn_weights
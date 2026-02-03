"""
MCRL-SASRec: MCRL对比学习增强 + SASRec强排序骨架

架构:
Item raw features (text_feat, vision_feat)
   ↓
多模态编码器 → item embedding
   ↓
SASRec (Transformer with causal mask)
   ↓
用户表示 + 对比学习辅助任务
   ↓
BPR Loss + Contrastive Loss

核心创新:
1. 使用SASRec的自回归序列建模（causal mask）
2. 保留MCRL的三层对比学习作为辅助任务
3. 与PMAT-SASRec形成对照：语义ID vs 对比学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List
import math
import numpy as np

from base_model import AbstractTrainableModel
from config import config

# 复用SASRec的Transformer块
from baseline_models.sasrec import TransformerBlock

# 复用MCRL的对比学习模块
from our_models.mcrl import (
    UserPreferenceContrastive,
    IntraModalContrastive,
    InterModalContrastive,
    MCRLItemEncoder
)


class MCRLSASRecItemEncoder(nn.Module):
    """MCRL-SASRec的物品编码器

    简化版多模态融合，不使用语义ID量化
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        # 模态编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(config.text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.visual_encoder = nn.Sequential(
            nn.Linear(config.visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 模态融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 可学习的模态权重
        self.modal_weight = nn.Parameter(torch.ones(2) / 2)

    def forward(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            text_feat: (..., text_dim)
            vision_feat: (..., visual_dim)

        Returns:
            item_emb: (..., hidden_dim)
        """
        # 确保float32
        text_feat = text_feat.float()
        vision_feat = vision_feat.float()

        # 编码各模态
        text_encoded = self.text_encoder(text_feat)
        visual_encoded = self.visual_encoder(vision_feat)

        # 加权融合
        weights = F.softmax(self.modal_weight, dim=0)
        weighted = weights[0] * text_encoded + weights[1] * visual_encoded

        # 拼接融合
        combined = torch.cat([text_encoded, visual_encoded], dim=-1)
        fused = self.fusion(combined)

        # 残差连接
        return fused + weighted


class MCRL_SASRec(AbstractTrainableModel):
    """MCRL-SASRec: 对比学习增强的序列推荐模型

    结合:
    1. SASRec的自回归序列建模
    2. MCRL的三层对比学习
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

        # ===== 物品编码器 =====
        self.item_encoder = MCRLSASRecItemEncoder(config)

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

        # ===== 预测层 =====
        self.prediction_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )

        # ===== 用户-物品匹配层 =====
        self.matcher = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

        # ===== 对比学习模块 (来自MCRL) =====
        self.temperature = getattr(config, 'mcrl_temperature', 0.07)
        self.alpha = getattr(config, 'mcrl_alpha', 1.0)  # 模态内对比权重
        self.beta = getattr(config, 'mcrl_beta', 0.5)    # 模态间对比权重

        self.user_preference_cl = UserPreferenceContrastive(
            hidden_dim=config.hidden_dim,
            temperature=self.temperature
        )

        self.intra_modal_cl = IntraModalContrastive(
            hidden_dim=config.hidden_dim,
            num_modalities=config.num_modalities,
            temperature=self.temperature
        )

        self.inter_modal_cl = InterModalContrastive(
            hidden_dim=config.hidden_dim,
            num_modalities=config.num_modalities,
            temperature=self.temperature
        )

        # 注册模态
        self._register_modalities()

        # ===== 损失权重 =====
        self.rec_loss_weight = getattr(config, 'rec_loss_weight', 1.0)
        self.cl_loss_weight = getattr(config, 'mcrl_loss_weight', 0.3)  # 降低对比学习权重

        # 缓存因果掩码
        self._causal_mask_cache = {}

        # 初始化权重
        self._init_weights()

    def _register_modalities(self):
        """注册模态到对比学习模块"""
        modal_dims = {
            'visual': self.config.visual_dim,
            'text': self.config.text_dim
        }
        for modality, dim in modal_dims.items():
            self.intra_modal_cl.add_modality(modality, dim)
            self.inter_modal_cl.add_modality(modality, dim)

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.input_layer_norm.weight, 1.0)
        nn.init.constant_(self.input_layer_norm.bias, 0.0)

        for module in self.prediction_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """获取因果掩码（带缓存）"""
        cache_key = (seq_len, device)
        if cache_key not in self._causal_mask_cache:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )
            self._causal_mask_cache[cache_key] = causal_mask
        return self._causal_mask_cache[cache_key]

    def compute_match_score(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor
    ) -> torch.Tensor:
        """计算用户-物品匹配分数

        Args:
            user_repr: (batch, hidden_dim) 或需要扩展
            item_repr: (batch, hidden_dim) 或 (batch, num_items, hidden_dim)

        Returns:
            scores: (batch,) 或 (batch, num_items)
        """
        if item_repr.dim() == 2:
            # 单个物品: (batch, hidden_dim)
            combined = torch.cat([user_repr, item_repr], dim=-1)
            scores = self.matcher(combined).squeeze(-1)
        else:
            # 多个物品: (batch, num_items, hidden_dim)
            batch_size, num_items, _ = item_repr.shape
            user_expanded = user_repr.unsqueeze(1).expand(-1, num_items, -1)
            combined = torch.cat([user_expanded, item_repr], dim=-1)
            scores = self.matcher(combined).squeeze(-1)
        return scores

    def encode_sequence(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor,
        seq_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码历史序列（SASRec风格）

        Args:
            text_feat: (batch, seq_len, text_dim)
            vision_feat: (batch, seq_len, visual_dim)
            seq_lens: (batch,) 实际序列长度

        Returns:
            user_repr: (batch, hidden_dim) 用户表示
            seq_output: (batch, seq_len, hidden_dim) 序列输出
        """
        batch_size, seq_len, _ = text_feat.shape
        device = text_feat.device

        # 1. 创建掩码
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        padding_mask = positions >= seq_lens.unsqueeze(1)  # (batch, seq_len)

        # 2. 物品编码
        item_emb = self.item_encoder(text_feat, vision_feat)  # (batch, seq_len, hidden_dim)

        # 3. 添加位置嵌入
        pos_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_indices = pos_indices.clamp(max=self.max_seq_len - 1)
        seq_emb = item_emb + self.pos_emb(pos_indices)

        # 4. 处理padding位置
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(seq_emb)
        noise = torch.randn_like(seq_emb) * 0.01
        seq_emb = torch.where(padding_mask_expanded, noise.detach(), seq_emb)

        # 5. LayerNorm + Dropout
        seq_emb = self.input_layer_norm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        # 6. Causal mask
        causal_mask = self._get_causal_mask(seq_len, device)

        # 7. Transformer编码
        for block in self.transformer_blocks:
            seq_emb = block(seq_emb, padding_mask=padding_mask, causal_mask=causal_mask)

        # 处理NaN
        seq_emb = torch.nan_to_num(seq_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # 8. 获取用户表示（最后一个有效位置）
        seq_lens_idx = (seq_lens - 1).clamp(min=0, max=seq_len - 1).long()
        user_repr = seq_emb[torch.arange(batch_size, device=device), seq_lens_idx]

        # 9. 预测层投影
        user_repr = self.prediction_layer(user_repr)

        return user_repr, seq_emb

    def encode_items(
        self,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor
    ) -> torch.Tensor:
        """编码候选物品

        Args:
            text_feat: (..., text_dim)
            vision_feat: (..., visual_dim)

        Returns:
            item_repr: (..., hidden_dim)
        """
        item_emb = self.item_encoder(text_feat, vision_feat)
        item_repr = self.prediction_layer(item_emb)
        return item_repr

    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """验证batch数据完整性"""
        required_keys = [
            'history_text_feat', 'history_vision_feat', 'history_len',
            'target_text_feat', 'target_vision_feat',
            'neg_text_feat', 'neg_vision_feat'
        ]
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise KeyError(f"Batch缺少必要的键: {missing_keys}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            batch: 包含历史序列和候选物品的字典

        Returns:
            outputs: 包含分数和对比学习损失的字典
        """
        self._validate_batch(batch)

        # 1. 编码历史序列 → 用户表示
        user_repr, seq_output = self.encode_sequence(
            batch['history_text_feat'],
            batch['history_vision_feat'],
            batch['history_len']
        )

        # 2. 编码正样本
        pos_repr = self.encode_items(
            batch['target_text_feat'],
            batch['target_vision_feat']
        )

        # 3. 编码负样本
        neg_repr = self.encode_items(
            batch['neg_text_feat'],
            batch['neg_vision_feat']
        )

        # 4. 计算匹配分数
        pos_scores = self.compute_match_score(user_repr, pos_repr)
        neg_scores = self.compute_match_score(user_repr, neg_repr)

        # 5. 编码历史物品表征（用于对比学习）
        batch_size, seq_len, _ = batch['history_text_feat'].shape
        history_text_flat = batch['history_text_feat'].view(batch_size * seq_len, -1)
        history_vision_flat = batch['history_vision_feat'].view(batch_size * seq_len, -1)
        history_item_repr = self.item_encoder(history_text_flat, history_vision_flat)
        history_item_repr = history_item_repr.view(batch_size, seq_len, -1)

        # 创建历史序列mask
        history_mask = torch.arange(seq_len, device=batch['history_len'].device).unsqueeze(0) < batch['history_len'].unsqueeze(1)

        # 6. 准备模态特征（用于对比学习）
        modal_features = {
            'visual': batch['target_vision_feat'],
            'text': batch['target_text_feat']
        }

        # 可学习的模态权重
        modal_weights = F.softmax(self.item_encoder.modal_weight, dim=0)
        modal_weights = modal_weights.unsqueeze(0).expand(batch_size, -1)

        # 7. 计算对比学习损失
        # Layer 1: 用户偏好对比
        L_user = self.user_preference_cl(
            id_embeddings=pos_repr,
            user_embeddings=user_repr,
            positive_ids=pos_repr.unsqueeze(1),
            negative_ids=neg_repr,
            history_item_repr=history_item_repr,
            history_mask=history_mask
        )

        # Layer 2: 模态内对比
        L_intra = self.intra_modal_cl(
            id_embeddings=pos_repr,
            modal_features=modal_features,
            modal_weights=modal_weights
        )

        # Layer 3: 模态间对比
        L_inter = self.inter_modal_cl(modal_features=modal_features)

        return {
            'user_repr': user_repr,
            'pos_repr': pos_repr,
            'neg_repr': neg_repr,
            'pos_scores': pos_scores,
            'neg_scores': neg_scores,
            'contrastive_losses': {
                'user_preference_loss': L_user,
                'intra_modal_loss': L_intra,
                'inter_modal_loss': L_inter
            }
        }


    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算多任务损失

        主损失: BPR推荐损失
        辅助损失: 三层对比学习损失
        """
        pos_scores = outputs['pos_scores']
        neg_scores = outputs['neg_scores']

        # 1. BPR推荐损失
        if neg_scores.numel() == 0 or neg_scores.size(1) == 0:
            bpr_loss = F.relu(1.0 - pos_scores).mean()
        else:
            pos_scores_expanded = pos_scores.unsqueeze(1)
            score_diff = torch.clamp(pos_scores_expanded - neg_scores, min=-50, max=50)
            bpr_loss = -F.logsigmoid(score_diff).mean()

        # 2. 对比学习损失
        cl_losses = outputs['contrastive_losses']
        contrastive_loss = (
            cl_losses['user_preference_loss'] +
            self.alpha * cl_losses['intra_modal_loss'] +
            self.beta * cl_losses['inter_modal_loss']
        )

        # 3. 总损失
        total_loss = self.rec_loss_weight * bpr_loss + self.cl_loss_weight * contrastive_loss

        return {
            'bpr_loss': bpr_loss,
            'contrastive_loss': contrastive_loss,
            'user_pref_loss': cl_losses['user_preference_loss'],
            'intra_modal_loss': cl_losses['intra_modal_loss'],
            'inter_modal_loss': cl_losses['inter_modal_loss'],
            'total_loss': total_loss
        }

    # ==================== AbstractTrainableModel 抽象方法实现 ====================

    def _get_optimizer(self, stage_id: int, stage_kwargs: Dict) -> torch.optim.Optimizer:
        """获取优化器"""
        lr = stage_kwargs.get('lr', 0.001)
        weight_decay = stage_kwargs.get('weight_decay', 0.01)
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

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

    def _update_params(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, scaler=None):
        """参数更新（带梯度裁剪）"""
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

    def _train_one_batch(self, batch: Any, stage_id: int, stage_kwargs: Dict) -> Tuple[torch.Tensor, Dict]:
        """单batch训练"""
        outputs = self.forward(batch)
        losses = self.compute_loss(outputs)

        pos_scores = outputs['pos_scores']
        neg_scores = outputs['neg_scores']
        pos_expanded = pos_scores.unsqueeze(1)
        auc_approx = (pos_expanded > neg_scores).float().mean().item()

        metrics = {
            'bpr_loss': losses['bpr_loss'].item(),
            'contrastive_loss': losses['contrastive_loss'].item(),
            'user_pref_loss': losses['user_pref_loss'].item(),
            'intra_modal_loss': losses['intra_modal_loss'].item(),
            'inter_modal_loss': losses['inter_modal_loss'].item(),
            'auc_approx': auc_approx
        }

        return losses['total_loss'], metrics

    def _validate_one_epoch(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        stage_id: int,
        stage_kwargs: Dict
    ) -> Dict:
        """单轮验证"""
        self.eval()

        total_loss = 0.0
        total_bpr_loss = 0.0
        total_cl_loss = 0.0
        all_pos_scores = []
        all_neg_scores = []

        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = self.forward(batch)
                losses = self.compute_loss(outputs)

                total_loss += losses['total_loss'].item()
                total_bpr_loss += losses['bpr_loss'].item()
                total_cl_loss += losses['contrastive_loss'].item()

                all_pos_scores.append(outputs['pos_scores'].cpu())
                all_neg_scores.append(outputs['neg_scores'].cpu())

        num_batches = len(val_dataloader)
        avg_loss = total_loss / num_batches
        avg_bpr_loss = total_bpr_loss / num_batches
        avg_cl_loss = total_cl_loss / num_batches

        all_pos_scores = torch.cat(all_pos_scores, dim=0)
        all_neg_scores = torch.cat(all_neg_scores, dim=0)

        metrics = self._compute_recommendation_metrics(all_pos_scores, all_neg_scores)
        metrics['loss'] = avg_loss
        metrics['bpr_loss'] = avg_bpr_loss
        metrics['contrastive_loss'] = avg_cl_loss

        return metrics

    def _compute_recommendation_metrics(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        k_list: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """计算推荐指标

        Args:
            pos_scores: (num_samples,) 正样本分数
            neg_scores: (num_samples, num_neg) 负样本分数
            k_list: 计算指标的K值列表

        Returns:
            metrics: 各项推荐指标
        """
        metrics = {}
        num_samples = pos_scores.size(0)
        num_neg = neg_scores.size(1)

        # 将正样本分数与负样本分数拼接
        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        # 获取排名
        _, indices = torch.sort(all_scores, dim=1, descending=True)
        ranks = (indices == 0).nonzero(as_tuple=True)[1] + 1  # 正样本排名

        for k in k_list:
            # HR@K
            hits = (ranks <= k).float()
            metrics[f'HR@{k}'] = hits.mean().item()

            # NDCG@K
            dcg = (ranks <= k).float() / torch.log2(ranks.float() + 1)
            metrics[f'NDCG@{k}'] = dcg.mean().item()

            # MRR@K
            mrr = (ranks <= k).float() / ranks.float()
            metrics[f'MRR@{k}'] = mrr.mean().item()

        # 全局MRR
        metrics['MRR'] = (1.0 / ranks.float()).mean().item()

        # AUC
        auc = (pos_scores.unsqueeze(1) > neg_scores).float().mean().item()
        metrics['AUC'] = auc

        return metrics

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        all_item_text_feat: torch.Tensor,
        all_item_vision_feat: torch.Tensor,
        top_k: int = 20
    ) -> torch.Tensor:
        """预测top-k物品

        Args:
            batch: 包含用户历史的batch
            all_item_text_feat: (num_items, text_dim) 所有物品的文本特征
            all_item_vision_feat: (num_items, visual_dim) 所有物品的视觉特征
            top_k: 返回的物品数量

        Returns:
            predictions: (batch_size, top_k) 预测的物品ID
        """
        self.eval()
        with torch.no_grad():
            # 编码用户
            user_repr, _ = self.encode_sequence(
                batch['history_text_feat'],
                batch['history_vision_feat'],
                batch['history_len']
            )

            # 编码所有物品
            all_item_repr = self.encode_items(all_item_text_feat, all_item_vision_feat)

            # 计算分数
            batch_size = user_repr.size(0)
            num_items = all_item_repr.size(0)

            # 扩展用户表示
            user_expanded = user_repr.unsqueeze(1).expand(-1, num_items, -1)
            item_expanded = all_item_repr.unsqueeze(0).expand(batch_size, -1, -1)

            combined = torch.cat([user_expanded, item_expanded], dim=-1)
            scores = self.matcher(combined).squeeze(-1)  # (batch_size, num_items)

            # 获取top-k
            _, top_indices = torch.topk(scores, k=min(top_k, num_items), dim=1)

            return top_indices

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from config import new_config


# ========== 从 simple_rqvae.py 复制的模块 ==========

def kmeans(data: torch.Tensor, k: int, max_iters: int = 100) -> torch.Tensor:
    """
    K-means聚类算法

    Args:
        data: (N, D) 数据点
        k: 聚类中心数量
        max_iters: 最大迭代次数

    Returns:
        centers: (k, D) 聚类中心
    """
    N, D = data.shape

    # 随机初始化中心
    indices = torch.randperm(N)[:k]
    centers = data[indices].clone()

    for _ in range(max_iters):
        # 计算距离
        distances = torch.cdist(data, centers)  # (N, k)

        # 分配到最近的中心
        assignments = torch.argmin(distances, dim=1)  # (N,)

        # 更新中心
        new_centers = torch.zeros_like(centers)
        for i in range(k):
            mask = assignments == i
            if mask.sum() > 0:
                new_centers[i] = data[mask].mean(dim=0)
            else:
                # 如果某个中心没有分配到点，随机重新初始化
                new_centers[i] = data[torch.randint(0, N, (1,))].squeeze(0)

        # 检查收敛
        if torch.allclose(centers, new_centers, atol=1e-6):
            break

        centers = new_centers

    return centers


def sinkhorn_algorithm(cost_matrix: torch.Tensor, epsilon: float, max_iters: int = 100) -> torch.Tensor:
    """
    Sinkhorn算法用于最优传输

    Args:
        cost_matrix: (N, K) 代价矩阵
        epsilon: 正则化参数
        max_iters: 最大迭代次数

    Returns:
        Q: (N, K) 传输矩阵
    """
    N, K = cost_matrix.shape

    # 初始化
    Q = torch.exp(-cost_matrix / epsilon)

    for _ in range(max_iters):
        # 行归一化
        Q = Q / Q.sum(dim=1, keepdim=True)

        # 列归一化
        Q = Q / Q.sum(dim=0, keepdim=True)

    return Q


class VectorQuantizer(nn.Module):
    """
    向量量化器（从simple_rqvae复制）
    """

    def __init__(
        self,
        n_e: int,  # 码本大小
        e_dim: int,  # 嵌入维度
        beta: float = 0.25,  # commitment loss权重
        kmeans_init: bool = False,  # 是否使用K-means初始化
        kmeans_iters: int = 10,  # K-means迭代次数
        sk_epsilon: float = 0.0,  # Sinkhorn epsilon（0表示不使用）
        sk_iters: int = 100,  # Sinkhorn迭代次数
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        # 码本
        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self) -> torch.Tensor:
        """获取码本"""
        return self.embedding.weight

    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[Tuple] = None) -> torch.Tensor:
        """根据索引获取码本向量"""
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data: torch.Tensor):
        """使用K-means初始化码本"""
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances: torch.Tensor) -> torch.Tensor:
        """中心化距离用于Sinkhorn约束"""
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x: torch.Tensor, use_sk: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入 (batch_size, e_dim)
            use_sk: 是否使用Sinkhorn算法

        Returns:
            x_q: 量化后的向量
            loss: 量化损失
            indices: 码本索引
        """
        # 展平输入
        latent = x.view(-1, self.e_dim)

        # K-means初始化（仅在训练时第一次）
        if not self.initted and self.training:
            self.init_emb(latent)

        # 计算L2距离
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.weight.t())

        # 选择最近的码本向量
        if not use_sk or self.sk_epsilon <= 0:
            # 标准最近邻
            indices = torch.argmin(d, dim=-1)
        else:
            # 使用Sinkhorn算法
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # 获取量化向量
        x_q = self.embedding(indices).view(x.shape)

        # 计算损失
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        x_q = x + (x_q - x).detach()

        # 恢复索引形状
        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


class ResidualVectorQuantizer(nn.Module):
    """
    残差向量量化器（从simple_rqvae复制）
    """

    def __init__(
        self,
        n_e_list: List[int],  # 每层的码本大小
        e_dim: int,  # 嵌入维度
        sk_epsilons: List[float],  # 每层的Sinkhorn epsilon
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_iters: int = 100,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        # 创建多层量化器
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(
                n_e, e_dim,
                beta=self.beta,
                kmeans_init=self.kmeans_init,
                kmeans_iters=self.kmeans_iters,
                sk_epsilon=sk_epsilon,
                sk_iters=sk_iters
            )
            for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
        ])

    def get_codebook(self) -> torch.Tensor:
        """获取所有层的码本"""
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x: torch.Tensor, use_sk: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播（残差量化）

        Args:
            x: 输入 (batch_size, e_dim)
            use_sk: 是否使用Sinkhorn算法

        Returns:
            x_q: 累积的量化向量
            mean_losses: 平均量化损失
            all_indices: 所有层的索引 (batch_size, num_layers)
        """
        all_losses = []
        all_indices = []

        x_q = 0  # 累积量化向量
        residual = x  # 当前残差

        for quantizer in self.vq_layers:
            # 量化当前残差
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)

            # 更新残差
            residual = residual - x_res

            # 累积量化向量
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        # 平均损失
        mean_losses = torch.stack(all_losses).mean()

        # 堆叠索引
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices


class MLPLayers(nn.Module):
    """MLP层（从simple_rqvae复制）"""

    def __init__(self, layers: List[int], dropout: float = 0.0, bn: bool = False):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.bn = bn

        mlp_modules = []
        for i in range(len(layers) - 1):
            mlp_modules.append(nn.Linear(layers[i], layers[i + 1]))
            if bn:
                mlp_modules.append(nn.BatchNorm1d(layers[i + 1]))
            if i < len(layers) - 2:  # 最后一层不加激活
                mlp_modules.append(nn.ReLU())
                if dropout > 0:
                    mlp_modules.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ========== 抽取的多模态融合模块 ==========

class MultimodalFusion(nn.Module):
    """
    多模态融合模块（抽取自AdaptiveHierarchicalQuantizer）
    支持单模态和多模态输入
    """

    def __init__(
        self,
        hidden_dim: int,
        text_dim: Optional[int] = None,
        visual_dim: Optional[int] = None,
        use_multimodal: bool = False,
        dropout: float = 0.1,
        bn: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_multimodal = use_multimodal
        self.text_dim = text_dim
        self.visual_dim = visual_dim

        if self.use_multimodal:
            # 多模态：使用注意力融合
            self.text_proj = MLPLayers([text_dim, hidden_dim], dropout=dropout, bn=bn)
            self.visual_proj = MLPLayers([visual_dim, hidden_dim], dropout=dropout, bn=bn)
            self.modal_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),
                nn.Softmax(dim=-1)
            )
        else:
            # 单模态：直接投影
            combined_dim = (text_dim or 0) + (visual_dim or 0)
            self.feature_proj = MLPLayers([combined_dim, hidden_dim], dropout=dropout, bn=bn)

    def forward(self, *args) -> torch.Tensor:
        """
        前向传播

        Args:
            *args: 单模态输入或(text_feat, visual_feat)双模态输入

        Returns:
            x: 融合后的特征 (batch_size, hidden_dim)
        """
        if self.use_multimodal:
            text_feat, visual_feat = args
            text_proj = self.text_proj(text_feat)
            visual_proj = self.visual_proj(visual_feat)

            feat_mean = (text_proj + visual_proj).mean(dim=-2)
            modal_weights = self.modal_attention(feat_mean)

            modal_weights_exp = modal_weights.unsqueeze(-2)
            fused_feat = text_proj * modal_weights_exp[..., 0:1] + visual_proj * modal_weights_exp[..., 1:2]

            return fused_feat
        else:
            if len(args) == 2:
                text_feat, visual_feat = args
                combined = torch.cat([text_feat, visual_feat], dim=-1)
            else:
                combined = args[0]
            return self.feature_proj(combined)


class AdaptiveHierarchicalQuantizer(nn.Module):
    """
    自适应层次化量化器

    支持：
    - 单模态/多模态输入（通过MultimodalFusion模块）
    - 层次化语义分组（Topic和Style）
    - EMA更新和死码重置
    """

    def __init__(
        self,
        hidden_dim: int,
        semantic_hierarchy: dict,
        use_multimodal: bool = False,
        text_dim: Optional[int] = None,
        visual_dim: Optional[int] = None,
        beta: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        reset_unused_codes: bool = True,
        reset_threshold: int = 100,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        sk_epsilon: float = 0.0,
        sk_iters: int = 100,
        dropout: float = 0.1,
        bn: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.semantic_hierarchy = semantic_hierarchy
        self.use_multimodal = use_multimodal
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.reset_unused_codes = reset_unused_codes
        self.reset_threshold = reset_threshold

        # 多模态融合模块（使用MLPLayers）
        self.fusion = MultimodalFusion(
            hidden_dim=hidden_dim,
            text_dim=text_dim,
            visual_dim=visual_dim,
            use_multimodal=use_multimodal,
            dropout=dropout,
            bn=bn
        )

        # 量化器初始化 - 动态计算总层数（支持任意配置的语义层级）
        self.num_layers = sum(
            len(config["layers"])
            for config in semantic_hierarchy.values()
        )
        self.layer_dim = hidden_dim // self.num_layers
        assert hidden_dim % self.num_layers == 0, "hidden_dim必须是总层数的整数倍"

        # 构建 n_e_list 和 sk_epsilons（用于 ResidualVectorQuantizer）
        # 动态遍历所有语义层级，支持任意配置
        n_e_list = []
        sk_epsilons_list = []

        for semantic_type, config in semantic_hierarchy.items():
            for layer in config["layers"]:
                n_e_list.append(config["codebook_size"])
                sk_epsilons_list.append(sk_epsilon)

        # 根据use_ema参数选择使用带EMA的量化器或普通量化器
        if use_ema:
            # 使用带EMA更新和死码重置的量化器
            self.rq = ResidualVectorQuantizerEMA(
                n_e_list=n_e_list,
                e_dim=hidden_dim,
                beta=beta,
                use_ema=True,
                ema_decay=ema_decay,
                reset_unused_codes=reset_unused_codes,
                reset_threshold=reset_threshold
            )
        else:
            # 使用普通残差量化器（无EMA功能）
            self.rq = ResidualVectorQuantizer(
                n_e_list=n_e_list,
                e_dim=hidden_dim,
                sk_epsilons=sk_epsilons_list,
                beta=beta,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                sk_iters=sk_iters
            )

        # 码本使用统计
        self.code_usage_count = {}
        self.temperature = nn.Parameter(torch.tensor(new_config.ahrq_temperature))
        self._last_quant_output = {}

    def forward(self, *args):
        """
        前向传播：使用ResidualVectorQuantizer进行量化

        Returns:
            quantized: 量化后的特征
            indices: 量化索引 (batch_size, num_layers)
            raw_feat: 原始融合特征
            quant_loss: 量化损失
        """
        # Step 1: 使用融合模块处理输入
        x = self.fusion(*args)
        raw_feat = x.clone()

        # Step 2: 使用ResidualVectorQuantizer进行量化
        x_q, quant_loss, indices = self.rq(x, use_sk=(getattr(self.rq, 'sk_epsilons', [0])[0] > 0))

        self._last_quant_output = {
            "quantized": x_q,
            "indices": indices
        }

        return x_q, indices, raw_feat, quant_loss

    def collect_code_usage(self, indices_list):
        for layer_idx, indices in enumerate(indices_list):
            # 动态查找该 layer 属于哪个语义层级
            cb_type = None
            for semantic_type, config in self.semantic_hierarchy.items():
                if layer_idx in config["layers"]:
                    cb_type = semantic_type
                    break
            if cb_type is None:
                cb_type = f"layer_{layer_idx}"
            cb_key = f"{cb_type}_{layer_idx}"

            if cb_key not in self.code_usage_count:
                self.code_usage_count[cb_key] = {}
            unique_indices, counts = torch.unique(indices, return_counts=True)
            for idx, cnt in zip(unique_indices, counts):
                idx_item = idx.item()
                self.code_usage_count[cb_key][idx_item] = self.code_usage_count[cb_key].get(idx_item, 0) + cnt.item()

# ========== 新增：VectorQuantizerEMA（带EMA更新的量化器） ==========

class VectorQuantizerEMA(nn.Module):
    """
    带EMA更新的向量量化器

    特性：
    - EMA更新码本（避免梯度消失）
    - 死码重置（解决码本崩溃）
    - 码本使用率跟踪
    """

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        reset_unused: bool = True,
        reset_threshold: int = 100,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.use_ema = use_ema
        self.decay = decay
        self.reset_unused = reset_unused
        self.reset_threshold = reset_threshold

        # 码本
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        if use_ema:
            # EMA统计量
            self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
            self.register_buffer('ema_w', self.embedding.weight.data.clone())
            self.register_buffer('codebook_usage', torch.zeros(codebook_size))

        # 死码跟踪
        if reset_unused:
            self.register_buffer('steps_since_used', torch.zeros(codebook_size))

    def forward(self, x: torch.Tensor) -> dict:
        """前向传播"""
        batch_size = x.size(0)

        # 计算距离
        distances = torch.cdist(x, self.embedding.weight)

        # 找到最近的码本向量
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices)

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        # 计算损失
        commitment_loss = F.mse_loss(x, quantized.detach())
        final_loss = self.beta * commitment_loss

        # EMA更新（仅在训练时）
        if self.training and self.use_ema:
            self._ema_update(x, indices)
            if self.reset_unused:
                self._reset_dead_codes(x)

        # 计算码本使用率
        usage_rate = (self.codebook_usage > 0).float().mean().item() if self.use_ema else 0.0

        return {
            'quantized': quantized,
            'indices': indices,
            'loss': final_loss,
            'usage_rate': usage_rate
        }

    def _ema_update(self, x: torch.Tensor, indices: torch.Tensor):
        """EMA更新码本"""
        encodings = F.one_hot(indices, self.codebook_size).float()
        self.ema_cluster_size.data.mul_(self.decay).add_(
            encodings.sum(0), alpha=1 - self.decay
        )
        dw = encodings.t() @ x
        self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
        self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
        self.codebook_usage.data.mul_(0.99).add_(encodings.sum(0), alpha=0.01)

    def _reset_dead_codes(self, x: torch.Tensor):
        """重置死码"""
        used_codes = torch.unique(torch.argmin(
            torch.cdist(x, self.embedding.weight), dim=1
        ))
        self.steps_since_used += 1
        self.steps_since_used[used_codes] = 0
        dead_codes = (self.steps_since_used > self.reset_threshold).nonzero(as_tuple=True)[0]
        if len(dead_codes) > 0:
            random_indices = torch.randint(0, x.size(0), (len(dead_codes),), device=x.device)
            self.embedding.weight.data[dead_codes] = x[random_indices].detach()
            self.steps_since_used[dead_codes] = 0


# ========== 新增：层次化语义一致性模块 (HSCL) ==========

class HierarchicalSemanticConsistency(nn.Module):
    """
    层次化语义一致性学习模块 (HSCL)
    解决MACRec指出的深层语义损失问题
    """

    def __init__(
        self,
        hidden_dim: int,
        semantic_hierarchy: dict,
        predictor_type: str = "mlp",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.semantic_hierarchy = semantic_hierarchy

        # 构建层次预测器：Topic -> Style, Style -> Emotion
        self.predictors = nn.ModuleDict()

        semantic_types = list(semantic_hierarchy.keys())
        for i in range(len(semantic_types) - 1):
            source_type = semantic_types[i]
            target_type = semantic_types[i + 1]

            source_layers = semantic_hierarchy[source_type]['layers']
            target_codebook_size = semantic_hierarchy[target_type]['codebook_size']
            input_dim = len(source_layers) * hidden_dim

            predictor_name = f"{source_type}_to_{target_type}"

            if predictor_type == "mlp":
                self.predictors[predictor_name] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, target_codebook_size)
                )

    def compute_consistency_loss(
        self,
        quantized_layers: List[torch.Tensor],
        indices: torch.Tensor
    ) -> dict:
        """计算层次一致性损失"""
        losses = {}
        semantic_types = list(self.semantic_hierarchy.keys())

        for i in range(len(semantic_types) - 1):
            source_type = semantic_types[i]
            target_type = semantic_types[i + 1]

            source_layers = self.semantic_hierarchy[source_type]['layers']
            target_layers = self.semantic_hierarchy[target_type]['layers']

            source_quantized = [quantized_layers[idx] for idx in source_layers]
            source_concat = torch.cat(source_quantized, dim=-1)

            predictor_name = f"{source_type}_to_{target_type}"
            target_pred = self.predictors[predictor_name](source_concat)

            target_idx = target_layers[0]
            target_true = indices[:, target_idx]

            consistency_loss = F.cross_entropy(target_pred, target_true)
            losses[f"consistency_{source_type}_to_{target_type}"] = consistency_loss

        total_consistency_loss = sum(losses.values())
        losses['total_consistency_loss'] = total_consistency_loss

        return losses


# ========== 新增：基于EMA的残差量化器 ==========

class ResidualVectorQuantizerEMA(nn.Module):
    """基于EMA的残差向量量化器"""

    def __init__(
        self,
        n_e_list: List[int],
        e_dim: int,
        beta: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        reset_unused_codes: bool = True,
        reset_threshold: int = 100,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.reset_unused_codes = reset_unused_codes
        self.reset_threshold = reset_threshold

        self.vq_layers = nn.ModuleList([
            VectorQuantizerEMA(
                n_e, e_dim,
                beta=self.beta,
                use_ema=self.use_ema,
                decay=self.ema_decay,
                reset_unused=self.reset_unused_codes,
                reset_threshold=self.reset_threshold
            )
            for n_e in n_e_list
        ])

    def get_codebook(self) -> torch.Tensor:
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.embedding.weight.data
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x

        for quantizer in self.vq_layers:
            quant_output = quantizer(residual)
            x_res = quant_output['quantized']
            loss = quant_output['loss']
            indices = quant_output['indices']

            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices
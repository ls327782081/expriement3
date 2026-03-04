import torch
import torch.nn as nn
import torch.nn.functional as F
from config import new_config


class AdaptiveHierarchicalQuantizer(nn.Module):
    """
    升级后：支持单/多模态特征输入，内置多模态对齐层
    - 单模态：直接量化（适配创新点1）
    - 多模态：先对齐融合 → 再量化（适配创新点2）
    """

    def __init__(self, hidden_dim, semantic_hierarchy,
                 use_multimodal=False,  # 是否启用多模态对齐
                 text_dim=None, visual_dim=None,  # 多模态特征维度
                 beta=0.25, use_ema=True, ema_decay=0.99,
                 reset_unused_codes=True, reset_threshold=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.semantic_hierarchy = semantic_hierarchy
        self.use_multimodal = use_multimodal  # 核心开关：是否处理多模态

        # ========== 新增：多模态对齐层（仅use_multimodal=True时生效） ==========
        if self.use_multimodal:
            # 文本特征映射到统一维度
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            # 视觉特征映射到统一维度
            self.visual_proj = nn.Sequential(
                nn.Linear(visual_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            # 模态融合权重（个性化）
            self.modal_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),
                nn.Softmax(dim=-1)
            )

        # ========== 原有AH-RQ量化逻辑（完全保留） ==========
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.reset_unused_codes = reset_unused_codes
        self.reset_threshold = reset_threshold

        self.num_layers = len(semantic_hierarchy["topic"]["layers"]) + len(semantic_hierarchy["style"]["layers"])
        self.layer_dim = hidden_dim // self.num_layers
        assert hidden_dim % self.num_layers == 0, "hidden_dim必须是总层数的整数倍"

        self.codebooks = nn.ParameterDict()
        self.ema_codebooks = nn.ParameterDict()
        self.code_usage_count = {}

        # Topic层码本
        for layer in semantic_hierarchy["topic"]["layers"]:
            cb_size = semantic_hierarchy["topic"]["codebook_size"]
            self.codebooks[f"topic_{layer}"] = nn.Parameter(torch.randn(cb_size, self.layer_dim))
            self.ema_codebooks[f"topic_{layer}"] = nn.Parameter(torch.randn(cb_size, self.layer_dim),
                                                                requires_grad=False)

        # Style层码本
        for layer in semantic_hierarchy["style"]["layers"]:
            cb_size = semantic_hierarchy["style"]["codebook_size"]
            self.codebooks[f"style_{layer}"] = nn.Parameter(torch.randn(cb_size, self.layer_dim))
            self.ema_codebooks[f"style_{layer}"] = nn.Parameter(torch.randn(cb_size, self.layer_dim),
                                                                requires_grad=False)

        self.temperature = nn.Parameter(torch.tensor(new_config.ahrq_temperature))
        self._last_quant_output = {}

    def multimodal_align(self, text_feat, visual_feat):
        """
        多模态对齐融合（新增核心函数）
        Args:
            text_feat: (batch, ..., text_dim) 文本特征
            visual_feat: (batch, ..., visual_dim) 视觉特征
        Returns:
            fused_feat: (batch, ..., hidden_dim) 对齐融合后的特征
        """
        # 映射到统一维度
        text_proj = self.text_proj(text_feat)
        visual_proj = self.visual_proj(visual_feat)

        # 计算个性化模态权重（基于特征均值）
        feat_mean = (text_proj + visual_proj).mean(dim=-2)  # (batch, hidden_dim)
        modal_weights = self.modal_attention(feat_mean)  # (batch, 2)

        # 融合（适配任意序列长度）
        modal_weights_exp = modal_weights.unsqueeze(-2)  # (batch, 1, 2)
        fused_feat = text_proj * modal_weights_exp[..., 0:1] + visual_proj * modal_weights_exp[..., 1:2]

        return fused_feat

    def forward(self, *args):
        """
        重载forward：支持单/多模态输入
        - 单模态（创新点1）：forward(feat) → 直接量化
        - 多模态（创新点2）：forward(text_feat, visual_feat) → 先对齐融合 → 再量化
        """
        # Step 1: 处理输入，得到统一特征
        if self.use_multimodal:
            # 多模态输入：text_feat, visual_feat
            text_feat, visual_feat = args
            x = self.multimodal_align(text_feat, visual_feat)
        else:
            # 单模态输入：仅特征
            x = args[0]

        raw_feat = x.clone()

        # Step 2: 原有AH-RQ量化逻辑（完全复用）
        x_blocks = torch.chunk(x, self.num_layers, dim=-1)
        quantized_blocks = []
        indices_list = []
        quantized_layers = []
        code_probs_list = []  # 存储每层的码本选择概率

        for layer_idx, block in enumerate(x_blocks):
            if layer_idx in self.semantic_hierarchy["topic"]["layers"]:
                cb_type = "topic"
                cb_size = self.semantic_hierarchy["topic"]["codebook_size"]
                ema_decay = self.semantic_hierarchy["topic"]["ema_decay"]
            else:
                cb_type = "style"
                cb_size = self.semantic_hierarchy["style"]["codebook_size"]
                ema_decay = self.semantic_hierarchy["style"]["ema_decay"]

            cb_key = f"{cb_type}_{layer_idx}"
            codebook = self.codebooks[cb_key]

            similarity = torch.matmul(block, codebook.T) / torch.clamp(self.temperature, min=0.04)

            if self.training:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarity) + 1e-8) + 1e-8)
                similarity = similarity + gumbel_noise * 0.05

            indices = torch.argmax(similarity, dim=-1)
            indices_list.append(indices)

            code_probs = F.softmax(similarity, dim=-1)  # [batch, ..., cb_size]
            code_probs_list.append(code_probs)

            soft_indices = F.softmax(similarity, dim=-1)
            quant_block_soft = torch.matmul(soft_indices, codebook)
            quant_block_hard = F.embedding(indices, codebook)
            quant_block = quant_block_hard + (quant_block_soft - quant_block_soft.detach())

            quantized_blocks.append(quant_block)
            quantized_layers.append(quant_block)

            # EMA更新、死码重置（原有逻辑完全保留）
            if self.training and self.use_ema:
                if cb_key not in self.code_usage_count:
                    self.code_usage_count[cb_key] = {}
                unique_indices, counts = torch.unique(indices, return_counts=True)
                for idx, cnt in zip(unique_indices, counts):
                    idx_item = idx.item()
                    self.code_usage_count[cb_key][idx_item] = self.code_usage_count[cb_key].get(idx_item,
                                                                                                0) + cnt.item()

                dw = torch.zeros_like(self.ema_codebooks[cb_key])
                for idx in unique_indices:
                    mask = indices == idx
                    dw[idx] = (block[mask].mean(dim=0) - self.ema_codebooks[cb_key][idx]) * (1 - ema_decay)
                self.ema_codebooks[cb_key].data.add_(dw)
                self.codebooks[cb_key].data = self.ema_codebooks[cb_key].data.clone()

            if self.training and self.reset_unused_codes:
                if cb_key in self.code_usage_count:
                    dead_codes = [k for k, v in self.code_usage_count[cb_key].items() if v < self.reset_threshold]
                    if len(dead_codes) > 0:

                        max_idx = block.shape[0] - 1
                        random_idx = torch.randint(0, max_idx, (len(dead_codes),), device=block.device)
                        random_feat = block[random_idx].detach()

                        for i, code_idx in enumerate(dead_codes):
                            self.codebooks[cb_key].data[code_idx] = (
                                    self.codebooks[cb_key].data[code_idx] * 0.7 + random_feat[i].mean(dim=0) * 0.3
                            )
                            self.code_usage_count[cb_key][code_idx] = self.reset_threshold

        quantized = torch.cat(quantized_blocks, dim=-1)

        code_probs_stack = torch.stack(code_probs_list, dim=-2)  # [batch, ..., num_layers, cb_size]
        avg_code_probs = code_probs_stack.mean(dim=-2)  # 所有层取平均 → [batch, ..., cb_size]

        self._last_quant_output = {
            "quantized": quantized,
            "indices": indices_list,
            "quantized_layers": quantized_layers
        }

        return quantized, indices_list, quantized_layers, avg_code_probs, raw_feat

    # 在AdaptiveHierarchicalQuantizer类中新增
    def collect_code_usage(self, indices_list):
        """
        批量统计ID使用情况（供外部重置调用）
        Args:
            indices_list: list of tensor，多层ID (batch, seq_len)
        """
        for layer_idx, indices in enumerate(indices_list):
            # 匹配cb_key的命名规则
            if layer_idx in self.semantic_hierarchy["topic"]["layers"]:
                cb_type = "topic"
            else:
                cb_type = "style"
            cb_key = f"{cb_type}_{layer_idx}"

            if cb_key not in self.code_usage_count:
                self.code_usage_count[cb_key] = {}
            unique_indices, counts = torch.unique(indices, return_counts=True)
            for idx, cnt in zip(unique_indices, counts):
                idx_item = idx.item()
                self.code_usage_count[cb_key][idx_item] = self.code_usage_count[cb_key].get(idx_item, 0) + cnt.item()
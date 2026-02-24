import torch
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BaseConfig:
    """基础配置类"""
    # 设备配置
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    num_workers: int = 4

    # 数据配置
    category: str = "Video_Games"  # 数据集类别
    max_samples: int = 5000  # 控制样本量
    item_vocab_size: int = None # 物品库规模
    user_vocab_size: int = None   # 用户库规模
    num_items: int = None

    # 训练配置
    batch_size: int = 64  # 增加batch size以加快训练
    gradient_accumulation_steps: int = 1  # 不使用梯度累积
    epochs: int = 5  # 减少到5个epoch以节省时间
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    eta_min: float = 1e-5

    # 语义ID配置
    id_length: int = 8  # ID长度
    codebook_size: int = 1024  # 码本规模

    # 模型配置
    attention_heads: int = 4
    hidden_dim: int = 64
    mlp_dim: int = 512
    dropout: float = 0.5

    # 多模态配置
    num_modalities: int = 2  # 视觉+文本（实际使用的模态数量）
    visual_dim: int = 512   # 视觉特征维度 (CLIP ViT-B/32)
    text_dim: int = 768     # 文本特征维度 (BERT-base)
    audio_dim: int = 128    # 音频特征维度（预留）

    # PMAT特定配置
    drift_threshold: float = 0.3  # 兴趣漂移阈值
    consistency_weight: float = 0.1  # 语义一致性权重
    short_history_len: int = 10  # 短期历史长度
    long_history_len: int = 50   # 长期历史长度

    # PMAT推荐模型配置（多任务学习）
    rec_loss_weight: float = 1.0  # 推荐损失（BPR）权重
    semantic_loss_weight: float = 0.1  # 语义ID生成损失权重
    max_history_len: int = 50  # 用户历史序列最大长度
    num_negative_samples: int = 4  # 训练时每个正样本对应的负样本数量
    # 注意：评估使用Full Ranking模式（对所有物品排序），不再使用负采样评估

    # MCRL特定配置
    mcrl_alpha: float = 1.0  # 模态内对比权重
    mcrl_beta: float = 0.5   # 模态间对比权重
    mcrl_temperature: float = 0.07  # 对比学习温度
    num_positive_samples: int = 5   # 正样本数量

    # 两阶段训练配置（通用）
    two_stage_training: bool = True  # 是否启用两阶段训练
    stage1_epochs: int = 3  # 阶段1（预训练物品编码器）的epoch数
    stage2_epochs: int = 5  # 阶段2（训练序列模型）的epoch数
    stage1_lr: float = 1e-3  # 阶段1学习率（预训练通常用较大学习率）
    stage2_lr: float = 1e-4  # 阶段2学习率

    # 阶段1：物品编码器预训练（对比学习）
    pretrain_temperature: float = 0.07  # 对比学习温度
    pretrain_intra_weight: float = 1.0  # 模态内对比损失权重
    pretrain_inter_weight: float = 0.5  # 模态间对比损失权重

    # 阶段2：序列模型训练
    logit_temperature: float = 0.05  # Cross Entropy logits 温度缩放（避免数值不稳定）

    # 联合训练配置
    pmat_loss_weight: float = 1.0  # PMAT损失权重
    mcrl_loss_weight: float = 0.5  # MCRL损失权重

    # 实验配置
    experiment_name: str = "PMAT_MCRL_AmazonBooks"
    checkpoint_dir: str = "./checkpoints"
    result_dir: str = "./results"
    log_dir: str = "./logs"
    save_interval: int = 1  # 每N个epoch保存一次

    # 基线模型配置
    # 当前框架基线（多模态推荐）
    baseline_models: List[str] = field(default_factory=lambda: [
        "DGMRec",    # SIGIR 2025: 模态解耦和生成 - 已验证正确
    ])

    # RecBole基线（序列推荐，需要单独运行）
    # 使用 recbole_baselines/ 目录下的脚本运行
    recbole_baselines: List[str] = field(default_factory=lambda: [
        "SASRec",    # 序列推荐经典 (Transformer)
        "BERT4Rec",  # 双向序列推荐 (Bidirectional Transformer)
        "GRU4Rec",   # RNN序列推荐基线
    ])

    # PMAT消融实验模块
    pmat_ablation_modules: List[str] = field(default_factory=lambda: [
        "no_personalization",   # 移除个性化模态权重
        "no_dynamic_update",    # 移除动态更新机制
    ])

    # MCRL消融实验模块
    mcrl_ablation_modules: List[str] = field(default_factory=lambda: [
        "no_user_cl",          # 移除用户偏好对比学习
        "no_intra_cl",         # 移除模态内对比学习
        "no_inter_cl"          # 移除模态间对比学习
    ])

    # 超参数搜索空间
    hyper_param_search: Dict = field(default_factory=lambda: {
        "id_length": [4, 8, 12],
        "lr": [5e-5, 1e-4, 2e-4],
        "codebook_size": [512, 1024, 2048],
        "mcrl_alpha": [0.5, 1.0, 2.0],
        "mcrl_beta": [0.25, 0.5, 1.0]
    })

    # 评估配置
    eval_batch_size: int = 64
    eval_interval: int = 1  # 每N个epoch评估一次
    top_k_list: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    eval_top_k: int = 10


# 实例化配置
config = BaseConfig()
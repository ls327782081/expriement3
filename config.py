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
    max_samples: int = 50000  # 控制样本量
    item_vocab_size: int = 5000  # 物品库规模
    user_vocab_size: int = 10000  # 用户库规模

    # 训练配置
    batch_size: int = 64  # 增加batch size以加快训练
    gradient_accumulation_steps: int = 1  # 不使用梯度累积
    epochs: int = 5  # 减少到5个epoch以节省时间
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # 语义ID配置
    id_length: int = 8  # ID长度
    codebook_size: int = 1024  # 码本规模

    # 模型配置
    attention_heads: int = 4
    hidden_dim: int = 256
    mlp_dim: int = 512
    dropout: float = 0.1

    # 多模态配置
    num_modalities: int = 3  # 默认：视觉+文本+音频
    visual_dim: int = 1280  # 视觉特征维度
    text_dim: int = 768     # 文本特征维度
    audio_dim: int = 128    # 音频特征维度

    # PMAT特定配置
    drift_threshold: float = 0.3  # 兴趣漂移阈值
    consistency_weight: float = 0.1  # 语义一致性权重
    short_history_len: int = 10  # 短期历史长度
    long_history_len: int = 50   # 长期历史长度

    # MCRL特定配置
    mcrl_alpha: float = 1.0  # 模态内对比权重
    mcrl_beta: float = 0.5   # 模态间对比权重
    mcrl_temperature: float = 0.07  # 对比学习温度
    num_positive_samples: int = 5   # 正样本数量
    num_negative_samples: int = 20  # 负样本数量

    # 联合训练配置
    pmat_loss_weight: float = 1.0  # PMAT损失权重
    mcrl_loss_weight: float = 0.5  # MCRL损失权重

    # 实验配置
    experiment_name: str = "PMAT_MCRL_AmazonBooks"
    checkpoint_dir: str = "./checkpoints"
    result_dir: str = "./results"
    log_dir: str = "./logs"
    save_interval: int = 1  # 每N个epoch保存一次

    # 基线模型（更新为2025年最新）
    baseline_models: List[str] = field(default_factory=lambda: [
        "Pctx",      # 上下文感知
        "MMQ",       # 多模态量化
        "FusID",     # 融合ID
        "RPG",       # 检索增强生成
        "PRISM",     # 2025: 个性化多模态融合
        "AMMRM",     # 2025: 自适应多模态推荐
        "CoFiRec",   # 2025: 粗细粒度token化
        "LETTER"     # 2024: 可学习token化
    ])

    # 消融实验模块
    ablation_modules: List[str] = field(default_factory=lambda: [
        "no_personalization",   # 移除个性化模态权重
        "no_dynamic_update",    # 移除动态更新
        "no_user_cl",          # 移除用户偏好对比
        "no_intra_cl",         # 移除模态内对比
        "no_inter_cl"          # 移除模态间对比
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

    # 数据集配置
    datasets: Dict = field(default_factory=lambda: {
        "Amazon-Beauty": {
            "path": "./data/amazon_beauty",
            "modalities": ["text", "visual"],
            "num_items": 12101,
            "num_users": 22363
        },
        "MM-Rec": {
            "path": "./data/mm_rec",
            "modalities": ["text", "visual", "audio"],
            "num_items": 8000,
            "num_users": 15000
        },
        "MovieLens-25M": {
            "path": "./data/movielens_25m",
            "modalities": ["text", "visual"],
            "num_items": 62138,
            "num_users": 162541
        }
    })

    # 当前使用的数据集
    current_dataset: str = "Amazon-Beauty"


@dataclass
class ExperimentConfig(BaseConfig):
    """实验专用配置"""

    # 创新点1实验配置
    exp1_name: str = "Innovation1_PMAT"
    exp1_baselines: List[str] = field(default_factory=lambda: [
        "FusID", "MMQ", "PRISM", "AMMRM"
    ])

    # 创新点2实验配置
    exp2_name: str = "Innovation2_MCRL"
    exp2_baselines: List[str] = field(default_factory=lambda: [
        "LETTER", "CoFiRec"
    ])

    # 整体系统实验配置
    full_system_name: str = "FullSystem_PMAT_MCRL"
    full_system_baselines: List[str] = field(default_factory=lambda: [
        "Pctx", "MMQ", "FusID", "PRISM", "AMMRM", "CoFiRec", "LETTER"
    ])

    # ID质量评估指标
    id_quality_metrics: List[str] = field(default_factory=lambda: [
        "id_uniqueness",           # ID唯一性
        "semantic_consistency",    # 语义一致性
        "personalization_discrimination"  # 个性化区分度
    ])

    # 推荐性能指标
    recommendation_metrics: List[str] = field(default_factory=lambda: [
        "recall",      # 召回率
        "ndcg",        # NDCG
        "mrr",         # MRR
        "hit_rate",    # 命中率
        "coverage",    # 覆盖率
        "diversity"    # 多样性
    ])

    # 效率指标
    efficiency_metrics: List[str] = field(default_factory=lambda: [
        "id_generation_time",   # ID生成时间
        "retrieval_latency",    # 检索延迟
        "memory_usage",         # 内存占用
        "throughput"            # 吞吐量
    ])


# 实例化配置
config = BaseConfig()
exp_config = ExperimentConfig()
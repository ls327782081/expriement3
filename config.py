import torch
from dataclasses import dataclass


@dataclass
class BaseConfig:
    # 基础配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 42
    max_samples = 50000  # 适配L4：控制Amazon_Books样本量
    item_vocab_size = 5000  # 物品库规模
    batch_size = 32
    gradient_accumulation_steps = 2  # 梯度累积（等效batch=64）
    epochs = 10
    lr = 1e-4
    weight_decay = 1e-5

    # 语义ID配置（轻量化）
    id_length = 8  # ID长度（适配L4）
    codebook_size = 1024  # 码本规模
    fine_codebook_size = 512
    coarse_codebook_size = 256

    # 模型轻量化配置
    attention_heads = 4
    hidden_dim = 256
    mlp_dim = 512

    # 实验配置
    experiment_name = "PMAT_AmazonBooks"
    checkpoint_dir = "./checkpoints"
    result_dir = "./results"
    baseline_models = ["Pctx", "MMQ", "FusID", "RPG", "COBRA"]  # 基线模型
    ablation_modules = ["modal_attention", "dynamic_update"]  # 消融模块
    # 超参实验配置
    hyper_param_search = {
        "id_length": [4, 8, 12],
        "lr": [5e-5, 1e-4, 2e-5],
        "codebook_size": [512, 1024, 2048]
    }


# 实例化配置
config = BaseConfig()
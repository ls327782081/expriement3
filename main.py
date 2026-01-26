import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import logging
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_utils import get_dataloader, get_pctx_dataloader
from baseline_models import Pctx, PctxAligned, PRISM, DGMRec
from our_models.pmat import PMAT
from our_models.mcrl import MCRL
from metrics import calculate_metrics
from util import item_id_to_semantic_id, save_checkpoint, load_checkpoint, save_results
from base_model import AbstractTrainableModel, StageConfig

# 混合精度训练
scaler = GradScaler()

# 工作进程数（Windows下设置为0避免多进程问题）
NUM_WORKS = 0 if os.name == 'nt' else 4


def sample_positive_negative_ids(batch, train_loader, num_pos=5, num_neg=20):
    """采样正负样本ID（简化版本）"""
    hidden_dim = config.hidden_dim

    # 简化版本：随机采样
    # 正样本：同一用户的其他物品
    positive_ids = torch.randn(batch["user_id"].size(0), num_pos, hidden_dim).to(config.device)

    # 负样本：其他用户的物品
    negative_ids = torch.randn(batch["user_id"].size(0), num_neg, hidden_dim).to(config.device)

    return positive_ids, negative_ids


def train_model(model, train_loader, val_loader, experiment_name, ablation_module=None, logger=None):
    """通用训练函数（使用 AbstractTrainableModel 统一训练框架）

    Args:
        model: 模型实例（必须继承 AbstractTrainableModel）
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        experiment_name: 实验名称
        ablation_module: 消融模块名称（可选）
        logger: 日志记录器

    Returns:
        训练后的模型
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    # 检查模型是否继承自 AbstractTrainableModel
    if not isinstance(model, AbstractTrainableModel):
        raise TypeError(f"模型 {model.__class__.__name__} 必须继承 AbstractTrainableModel")

    # ✅ 使用 AbstractTrainableModel 的统一训练框架
    logger.info(f"✅ 使用 AbstractTrainableModel 统一训练框架训练 {model.__class__.__name__}")

    # 配置单阶段训练
    stage_config = StageConfig(
        stage_id=1,
        epochs=config.epochs,
        start_epoch=0,
        kwargs={
            'lr': config.lr,
            'weight_decay': config.weight_decay,
            'experiment_name': experiment_name,
            'num_positive_samples': getattr(config, 'num_positive_samples', 5),
            'num_negative_samples': getattr(config, 'num_negative_samples', 20),
        }
    )

    # 调用模型的统一训练方法
    model.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        stage_configs=[stage_config]
    )

    return model


def train_mcrl_model(model, train_loader, val_loader, experiment_name, logger=None):
    """MCRL模型训练函数（现在使用统一训练框架）

    注意：此函数现在只是 train_model 的别名，保留是为了向后兼容

    Args:
        model: MCRL模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        experiment_name: 实验名称
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    logger.info(f"\n===== 开始训练MCRL模型: {experiment_name} =====")

    # 使用统一训练框架
    return train_model(model, train_loader, val_loader, experiment_name, logger=logger)


def evaluate_model(model, test_loader, model_name, logger=None):
    """评估模型"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    logger.info(f"开始评估模型: {model_name}")

    model.eval()
    all_predictions = []
    all_ground_truth = []
    all_user_ids = []

    # 检查模型是否继承自BaseModel
    from base_model import BaseModel
    is_basemodel = isinstance(model, BaseModel)
    
    with torch.no_grad():
        for batch in test_loader:
            # 移动数据到设备
            batch = {k: v.to(config.device) for k, v in batch.items()}

            # 检查模型是否继承自BaseModel
            if is_basemodel:
                # 使用模型的预测方法
                predictions = model.predict(batch, top_k=config.eval_top_k)
            else:
                # 使用传统的前向传播
                logits = model(batch)
                # 获取top-k预测
                _, predictions = torch.topk(logits, k=config.eval_top_k, dim=-1)

            # 收集预测结果
            all_predictions.extend(predictions.cpu().numpy())
            all_ground_truth.extend(batch["item_id"].cpu().numpy())
            all_user_ids.extend(batch["user_id"].cpu().numpy())

    # 计算指标
    metrics = calculate_metrics(all_user_ids, all_predictions, all_ground_truth, logger=logger)
    metrics["model"] = model_name
    return metrics


# 1. 基线实验
def run_baseline_experiment(logger:logging.Logger, quick_mode:bool=False):
    """运行基线模型实验"""

    logger.info("===== 开始基线实验 =====")

    results = []
    # 训练并评估基线模型
    for baseline_name in config.baseline_models:
        logger.info(f"\n训练基线模型：{baseline_name}")

        # PctxAligned 需要特殊的数据加载流程
        if baseline_name == "PctxAligned":
            logger.info("Using specialized dataloader for PctxAligned...")
            tokenizer_path = f"checkpoints/pctx_tokenizer_{config.category}.pkl"
            train_loader, val_loader, test_loader, tokenizer = get_pctx_dataloader(
                cache_dir="./data",
                category=config.category,
                batch_size=config.batch_size,
                shuffle=True,
                quick_mode=quick_mode,
                logger=logger,
                num_workers=0,
                tokenizer_path=tokenizer_path,
                device=config.device
            )
            model = PctxAligned(vocab_size=tokenizer.vocab_size, device=config.device).to(config.device)
        else:
            # 其他模型使用标准数据加载器
            if baseline_name == config.baseline_models[0] and baseline_name != "PctxAligned":
                # 只在第一个非PctxAligned模型时加载数据
                train_loader, val_loader, test_loader = get_dataloader(
                    "./data",
                    category=config.category,
                    shuffle=True,
                    quick_mode=quick_mode,
                    logger=logger,
                    num_workers=0
                )

            if baseline_name == "Pctx":
                model = Pctx().to(config.device)
            elif baseline_name == "PRISM":
                model = PRISM(config).to(config.device)
            elif baseline_name == "DGMRec":
                base_model = DGMRec(config).to(config.device)
                # Wrapper for DGMRec to accept batch dict
                class DGMRecWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        self.fc = torch.nn.Linear(config.hidden_dim, config.codebook_size * config.id_length)
                    def forward(self, batch):
                        result = self.model(batch["user_id"], batch["item_id"], batch["vision_feat"].float(), batch["text_feat"].float())
                        logits = self.fc(result['user_embeddings'])
                        return logits.reshape(-1, config.id_length, config.codebook_size)
                model = DGMRecWrapper(base_model).to(config.device)
            else:
                logger.warning(f"未知基线模型: {baseline_name}，跳过")
                continue

        # 训练
        model = train_model(model, train_loader, val_loader, f"baseline_{baseline_name}", logger=logger)
        # 评估
        metrics = evaluate_model(model, test_loader, baseline_name, logger=logger)
        results.append(metrics)

    # 训练并评估原创模型PMAT
    logger.info("\n训练原创模型：PMAT")
    pmat_model = PMAT(config).to(config.device)
    pmat_model = train_model(pmat_model, train_loader, val_loader, "PMAT", logger=logger)
    pmat_metrics = evaluate_model(pmat_model, test_loader, "PMAT", logger=logger)
    results.append(pmat_metrics)

    # 保存结果
    save_results(results, "baseline", logger=logger)


# 2. 消融实验
def run_ablation_experiment(logger=None,quick_mode:bool=False):
    """运行消融实验"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")
        
    logger.info("\n===== 开始消融实验 =====")
    train_loader, val_loader, test_loader = get_dataloader("./data", category=config.category, shuffle=True, logger=logger,
                                                           quick_mode=quick_mode, num_workers=NUM_WORKS)

    results = []
    # 完整模型
    logger.info("训练完整PMAT模型")
    full_model = PMAT(config).to(config.device)
    full_model = train_model(full_model, train_loader, val_loader, "PMAT_full", logger=logger)
    full_metrics = evaluate_model(full_model, test_loader, "PMAT_full", logger=logger)
    results.append(full_metrics)

    # 消融实验：逐个移除模块
    for ablation_module in config.ablation_modules:
        logger.info(f"训练消融模型（移除{ablation_module}）")
        ablation_model = get_pmat_ablation_model(ablation_module, config).to(config.device)
        ablation_model = train_model(
            ablation_model, train_loader, val_loader, f"PMAT_ablation_{ablation_module}", logger=logger
        )
        ablation_metrics = evaluate_model(ablation_model, test_loader, f"PMAT_ablation_{ablation_module}", logger=logger)
        results.append(ablation_metrics)

    # 保存结果
    save_results(results, "ablation", logger=logger)


# 3. 超参实验
def run_hyper_param_experiment(logger=None,quick_mode:bool=False):
    """运行超参实验"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")
        
    logger.info("\n===== 开始超参实验 =====")
    train_loader,val_loader, test_loader  = get_dataloader("./data", category=config.category, shuffle=True, logger=logger,
                                                           quick_mode=quick_mode, num_workers=NUM_WORKS)


    results = []
    # 遍历超参组合
    for id_length in config.hyper_param_search["id_length"]:
        for lr in config.hyper_param_search["lr"]:
            for codebook_size in config.hyper_param_search["codebook_size"]:
                logger.info(f"\n超参组合：id_length={id_length}, lr={lr}, codebook_size={codebook_size}")
                # 更新配置
                config.id_length = id_length
                config.lr = lr
                config.codebook_size = codebook_size

                # 训练模型
                model = PMAT(config).to(config.device)
                exp_name = f"PMAT_hyper_id{id_length}_lr{lr}_cb{codebook_size}"
                model = train_model(model, train_loader, val_loader, exp_name, logger=logger)

                # 评估
                metrics = evaluate_model(model, test_loader, exp_name, logger=logger)
                # 添加超参信息
                metrics["id_length"] = id_length
                metrics["lr"] = lr
                metrics["codebook_size"] = codebook_size
                results.append(metrics)

    # 保存结果
    save_results(results, "hyper_param", logger=logger)


# 4. MCRL实验
def run_mcrl_experiment(logger=None,quick_model:bool=False):
    """运行MCRL完整实验

    包括：
    1. MCRL单独训练
    2. 消融实验（三层对比学习）
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    logger.info("\n" + "="*70)
    logger.info("===== 开始MCRL完整实验 =====")
    logger.info("="*70 + "\n")

    # 加载数据
    train_loader, val_loader, test_loader = get_dataloader(
        "./data",
        category=config.category,
        shuffle=True,
        logger=logger,
        quick_mode=quick_model,
        num_workers=NUM_WORKS
    )

    results = []

    # ========== 实验1: MCRL模型训练 ==========
    logger.info("\n【实验1】训练MCRL模型")
    logger.info("-" * 70)
    mcrl_model = MCRL(config).to(config.device)
    mcrl_model = train_mcrl_model(mcrl_model, train_loader, val_loader, "MCRL_Exp_MCRL_only", logger=logger)
    mcrl_metrics = evaluate_model(mcrl_model, test_loader, "MCRL_only", logger=logger)
    results.append(mcrl_metrics)

    # 保存结果
    save_results(results, "mcrl_experiment", logger=logger)

    logger.info("\n" + "="*70)
    logger.info("MCRL实验完成")
    logger.info("="*70 + "\n")


def get_pmat_ablation_model(ablation_module, config=None):
    """获取PMAT消融模型"""
    from our_models.pmat import PMAT
    if config is None:
        from config import config as default_config
        config = default_config
    return PMAT(config, ablation_module=ablation_module)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PMAT & MCRL 实验')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full', 'baseline', 'ablation', 'hyper', 'mcrl'],
                        help='实验模式: quick(快速测试)/full(完整实验)/specific(特定实验)')
    
    # 数据集
    parser.add_argument('--dataset', type=str, default='mock',
                        choices=['mock', 'amazon', 'movielens'],
                        help='数据集类型')
    
    # 模型选择
    parser.add_argument('--model', type=str, default='pmat',
                        choices=['pmat', 'mcrl', 'all'],
                        help='模型选择')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='设备 (默认: 自动检测)')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='结果保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')

    return parser.parse_args()


def apply_args_to_config(args):
    """将命令行参数应用到config"""
    # 根据模式设置默认参数
    if args.mode == 'quick':
        config.epochs = args.epochs if args.epochs is not None else 25
        config.batch_size = args.batch_size if args.batch_size is not None else 32
        # quick模式使用真实数据集，但会抽样
        config.category = "Video_Games"
    elif args.mode == 'full':
        config.epochs = args.epochs if args.epochs is not None else 50
        config.batch_size = args.batch_size if args.batch_size is not None else 512
    else:
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size

    # 学习率
    if args.lr is not None:
        config.lr = args.lr

    # 设备
    if args.device is not None:
        config.device = torch.device(args.device)

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"配置已更新: epochs={config.epochs}, batch_size={config.batch_size}, lr={config.lr}, device={config.device}")


if __name__ == "__main__":
    # 设置日志
    os.makedirs(config.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, f'experiment_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("PMAT_Experiment")

    # 解析命令行参数
    args = parse_args()

    # 应用参数到config
    apply_args_to_config(args)

    logger.info("="*70)
    logger.info(f"实验模式: {args.mode}")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"模型: {args.model}")
    logger.info("="*70)

    # 根据模式运行实验
    if args.mode == 'quick':
        logger.info("快速测试模式 - 运行所有实验（抽样数据）")
        run_baseline_experiment(logger=logger, quick_mode=True)
        run_ablation_experiment(logger=logger, quick_mode=True)
        run_hyper_param_experiment(logger=logger, quick_mode=True)
        run_mcrl_experiment(logger=logger, quick_model=True)
    elif args.mode == 'full':
        logger.info("完整实验模式 - 运行所有实验")
        run_baseline_experiment(logger=logger, quick_mode=False)
        run_ablation_experiment(logger=logger, quick_mode=False)
        run_hyper_param_experiment(logger=logger, quick_mode=False)
        run_mcrl_experiment(logger=logger, quick_model=False)
    elif args.mode == 'baseline':
        logger.info("基线实验模式")
        run_baseline_experiment(logger=logger, quick_mode=(args.dataset=='mock'))
    elif args.mode == 'ablation':
        logger.info("消融实验模式")
        run_ablation_experiment(logger=logger, quick_mode=(args.dataset=='mock'))
    elif args.mode == 'hyper':
        logger.info("超参实验模式")
        run_hyper_param_experiment(logger=logger, quick_mode=(args.dataset=='mock'))
    elif args.mode == 'mcrl':
        logger.info("MCRL实验模式")
        run_mcrl_experiment(logger=logger, quick_model=(args.dataset=='mock'))

    logger.info("所有实验完成！")
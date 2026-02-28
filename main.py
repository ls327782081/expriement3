import os
import sys
import gc
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
from data_utils import get_dataloader, get_pctx_dataloader, get_pmat_dataloader, get_dgmrec_dataloader, \
    get_all_item_pretrain_dataloader
from baseline_models import DGMRec
# 注意：PRISM 和 PctxAligned 已从基线中移除
# PRISM: 实现存在错误（使用随机目标训练）
# PctxAligned: 需要特殊tokenizer，与实验设置不兼容
from our_models.pmat import PMAT
from our_models.mcrl import MCRL
from our_models.pmat_sasrec import PMAT_SASRec
from our_models.mcrl_sasrec import MCRL_SASRec
from our_models.pure_sasrec import RecBoleSASRec
from metrics import calculate_metrics
from util import item_id_to_semantic_id, save_checkpoint, load_checkpoint, save_results
from base_model import AbstractTrainableModel, StageConfig

# 混合精度训练
scaler = GradScaler()

# 工作进程数（Windows下设置为0避免多进程问题）
NUM_WORKS = 0 if os.name == 'nt' else 4


def train_model(model, train_loader, val_loader, experiment_name, ablation_module=None, logger=None, eval_kwargs=None, skip_validation=True):
    """通用训练函数（使用 AbstractTrainableModel 统一训练框架）

    Args:
        model: 模型实例（必须继承 AbstractTrainableModel）
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        experiment_name: 实验名称
        ablation_module: 消融模块名称（可选）
        logger: 日志记录器
        eval_kwargs: 评估时需要的额外参数（如 all_item_features）
        skip_validation: 是否跳过训练过程中的验证（默认True，加速训练）

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
    if skip_validation:
        logger.info("⏭️ 训练过程中跳过验证（仅在最后进行测试评估）")

    # 构建 stage_kwargs
    stage_kwargs = {
        'lr': config.lr,
        'weight_decay': config.weight_decay,
        'experiment_name': experiment_name,
        'num_positive_samples': getattr(config, 'num_positive_samples', 5),
        'num_negative_samples': getattr(config, 'num_negative_samples', 20),
    }

    # 添加评估参数（用于 Full Ranking 评估）
    if eval_kwargs:
        stage_kwargs.update(eval_kwargs)

    # 配置单阶段训练
    stage_config = StageConfig(
        stage_id=1,
        epochs=config.epochs,
        start_epoch=0,
        kwargs=stage_kwargs
    )

    # 调用模型的统一训练方法
    model.customer_train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        stage_configs=[stage_config],
        skip_validation=skip_validation
    )

    return model


def train_mcrl_two_stage(model, train_loader, val_loader, experiment_name, logger=None, eval_kwargs=None, skip_validation=True):
    """MCRL两阶段训练函数

    Stage 1 (表征塑形): 专注对比学习，低推荐权重，冻结matcher
    Stage 2 (推荐对齐): 专注推荐，低对比权重，解冻所有模块

    Args:
        model: MCRL模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        experiment_name: 实验名称
        logger: 日志记录器
        eval_kwargs: 评估时需要的额外参数（如 all_item_features）
        skip_validation: 是否跳过训练过程中的验证（默认True，加速训练）

    Returns:
        训练后的模型
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    # 检查是否启用两阶段训练
    two_stage_enabled = getattr(config, 'mcrl_two_stage', True)

    if not two_stage_enabled:
        logger.info("MCRL两阶段训练未启用，使用单阶段训练")
        return train_model(model, train_loader, val_loader, experiment_name, logger=logger, eval_kwargs=eval_kwargs, skip_validation=skip_validation)

    logger.info("="*60)
    logger.info("MCRL两阶段训练")
    if skip_validation:
        logger.info("⏭️ 训练过程中跳过验证（仅在最后进行测试评估）")
    logger.info("="*60)

    # 计算两阶段的epoch分配
    total_epochs = config.epochs
    stage1_ratio = getattr(config, 'mcrl_stage1_epochs_ratio', 0.3)
    stage1_epochs = max(1, int(total_epochs * stage1_ratio))
    stage2_epochs = total_epochs - stage1_epochs

    logger.info(f"总Epochs: {total_epochs}")
    logger.info(f"Stage 1 (表征塑形): {stage1_epochs} epochs")
    logger.info(f"  - rec_weight: {getattr(config, 'mcrl_stage1_rec_weight', 0.1)}")
    logger.info(f"  - cl_weight: {getattr(config, 'mcrl_stage1_cl_weight', 1.0)}")
    logger.info(f"Stage 2 (推荐对齐): {stage2_epochs} epochs")
    logger.info(f"  - rec_weight: {getattr(config, 'mcrl_stage2_rec_weight', 1.0)}")
    logger.info(f"  - cl_weight: {getattr(config, 'mcrl_stage2_cl_weight', 0.3)}")
    logger.info("="*60)

    # 构建基础 stage_kwargs
    base_kwargs = {
        'num_positive_samples': getattr(config, 'num_positive_samples', 5),
        'num_negative_samples': getattr(config, 'num_negative_samples', 20),
    }
    # 添加评估参数（用于 Full Ranking 评估）
    if eval_kwargs:
        base_kwargs.update(eval_kwargs)

    # 配置两阶段训练
    stage_configs = [
        # Stage 1: 表征塑形
        StageConfig(
            stage_id=1,
            epochs=stage1_epochs,
            start_epoch=0,
            kwargs={
                **base_kwargs,
                'lr': config.lr,
                'weight_decay': config.weight_decay,
                'experiment_name': f"{experiment_name}_stage1",
            }
        ),
        # Stage 2: 推荐对齐
        StageConfig(
            stage_id=2,
            epochs=stage2_epochs,
            start_epoch=0,
            kwargs={
                **base_kwargs,
                'lr': config.lr * 0.5,  # Stage 2使用较小学习率，避免破坏已学习的表征
                'weight_decay': config.weight_decay,
                'experiment_name': f"{experiment_name}_stage2",
            }
        )
    ]

    # 调用模型的统一训练方法
    model.customer_train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        stage_configs=stage_configs,
        skip_validation=skip_validation
    )

    return model



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
    from base_model import BaseModel, AbstractTrainableModel
    is_basemodel = isinstance(model, BaseModel)
    is_trainable_model = isinstance(model, AbstractTrainableModel)

    with torch.no_grad():
        for batch in test_loader:
            # 处理不同的数据格式
            if isinstance(batch, tuple):
                # DGMRec 格式: (users, pos_items, neg_items)
                users, pos_items, neg_items = batch
                users = users.to(config.device)
                pos_items = pos_items.to(config.device)
                neg_items = neg_items.to(config.device)
                batch_tuple = (users, pos_items, neg_items)

                # 使用模型的 predict 方法
                if hasattr(model, 'predict'):
                    predictions = model.predict(batch_tuple, top_k=config.eval_top_k)
                else:
                    # 使用 full_sort_predict
                    scores = model.full_sort_predict((users, pos_items))
                    _, predictions = torch.topk(scores, k=config.eval_top_k, dim=-1)

                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_ground_truth.extend(pos_items.cpu().numpy())
                all_user_ids.extend(users.cpu().numpy())
            else:
                # 字典格式
                batch = {k: v.to(config.device) for k, v in batch.items()}

                # 检查模型是否继承自BaseModel或AbstractTrainableModel
                if is_basemodel or (is_trainable_model and hasattr(model, 'predict')):
                    # 使用模型的预测方法
                    predictions = model.predict(batch, top_k=config.eval_top_k)
                else:
                    # 使用传统的前向传播
                    logits = model(batch)
                    # 获取top-k预测
                    _, predictions = torch.topk(logits, k=config.eval_top_k, dim=-1)

                # 收集预测结果
                all_predictions.extend(predictions.cpu().numpy())
                # 支持不同的键名
                item_key = "item_id" if "item_id" in batch else "pos_item"
                all_ground_truth.extend(batch[item_key].cpu().numpy())
                all_user_ids.extend(batch["user_id"].cpu().numpy())

    # 计算指标
    metrics = calculate_metrics(all_user_ids, all_predictions, all_ground_truth, logger=logger)
    metrics["model"] = model_name
    return metrics


# 1. 基线实验
def run_baseline_experiment(logger:logging.Logger, quick_mode:bool=False):
    """
    运行基线模型实验

    当前框架基线：
    - DGMRec: 多模态解耦和生成 (SIGIR 2025)

    RecBole基线（需要单独运行）：
    - SASRec, BERT4Rec, GRU4Rec
    - 使用 recbole_baselines/run_baselines.py 运行
    """

    logger.info("===== 开始基线实验 =====")
    logger.info("当前框架基线: " + ", ".join(config.baseline_models))
    logger.info("RecBole基线 (需单独运行): " + ", ".join(config.recbole_baselines))

    results = []
    train_loader, val_loader, test_loader = None, None, None

    # 训练并评估基线模型
    for baseline_name in config.baseline_models:
        logger.info(f"\n训练基线模型：{baseline_name}")

        if baseline_name == "DGMRec":
            # DGMRec 需要专用的数据加载器和数据适配器
            logger.info("为 DGMRec 加载专用数据...")
            dgmrec_train_loader, dgmrec_val_loader, dgmrec_test_loader, dataset_adapter = get_dgmrec_dataloader(
                "./data",
                category=config.category,
                batch_size=config.batch_size,
                shuffle=True,
                quick_mode=quick_mode,
                logger=logger,
                num_workers=0
            )
            # 使用数据适配器初始化 DGMRec（已继承 AbstractTrainableModel）
            model = DGMRec(config, dataset=dataset_adapter).to(config.device)
            # 使用 DGMRec 专用的数据加载器
            train_loader, val_loader, test_loader = dgmrec_train_loader, dgmrec_val_loader, dgmrec_test_loader
        else:
            logger.warning(f"未知基线模型: {baseline_name}，跳过")
            continue

        # 训练
        model = train_model(model, train_loader, val_loader, f"baseline_{baseline_name}", logger=logger)
        # 评估
        metrics = evaluate_model(model, test_loader, baseline_name, logger=logger)
        results.append(metrics)

        # 清理内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 训练并评估原创模型PMAT（如果有数据加载器）
    if train_loader is not None:
        logger.info("\n训练原创模型：PMAT")
        pmat_model = PMAT(config).to(config.device)
        pmat_model = train_model(pmat_model, train_loader, val_loader, "PMAT", logger=logger)
        pmat_metrics = evaluate_model(pmat_model, test_loader, "PMAT", logger=logger)
        results.append(pmat_metrics)

    # 保存结果
    save_results(results, "baseline", logger=logger)

    logger.info("\n" + "=" * 60)
    logger.info("提示：RecBole基线需要单独运行：")
    logger.info("  1. python recbole_baselines/convert_to_recbole.py --category Video_Games")
    logger.info("  2. pip install recbole")
    logger.info("  3. python recbole_baselines/run_baselines.py --dataset Video_Games")
    logger.info("=" * 60)


# 2. 消融实验
def run_ablation_experiment(logger=None, quick_mode: bool = False):
    """运行消融实验（PMAT和MCRL）

    PMAT消融模块：
    - no_personalization: 移除个性化模态权重
    - no_dynamic_update: 移除动态更新机制

    MCRL消融模块：
    - no_user_cl: 移除用户偏好对比学习
    - no_intra_cl: 移除模态内对比学习
    - no_inter_cl: 移除模态间对比学习
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    logger.info("\n" + "=" * 70)
    logger.info("===== 开始消融实验 =====")
    logger.info("=" * 70 + "\n")

    # 使用PMAT专用数据加载器（MCRL复用相同格式）
    train_loader, val_loader, test_loader, all_item_features = get_pmat_dataloader(
        cache_dir="./data",
        category=config.category,
        batch_size=config.batch_size,
        max_history_len=config.max_history_len,
        num_negative_samples=config.num_negative_samples,
        shuffle=True,
        quick_mode=quick_mode,
        num_workers=NUM_WORKS,
        logger=logger
    )

    # Full Ranking评估需要的stage_kwargs
    eval_kwargs = {'all_item_features': all_item_features}

    results = []

    # ==================== PMAT消融实验 ====================
    logger.info("\n" + "-" * 50)
    logger.info("PMAT消融实验")
    logger.info("-" * 50)

    # 完整PMAT模型
    logger.info("\n训练完整PMAT模型")
    pmat_full = PMAT(config).to(config.device)
    pmat_full = train_model(pmat_full, train_loader, val_loader, "PMAT_full", logger=logger, eval_kwargs=eval_kwargs)
    pmat_full.eval()
    pmat_full_metrics = pmat_full._validate_one_epoch(test_loader, stage_id=1, stage_kwargs=eval_kwargs)
    pmat_full_metrics["model"] = "PMAT_full"
    results.append(pmat_full_metrics)
    logger.info(f"PMAT完整模型: HR@10={pmat_full_metrics.get('HR@10', 0):.4f}, NDCG@10={pmat_full_metrics.get('NDCG@10', 0):.4f}")

    # PMAT消融实验
    from our_models.pmat import get_pmat_ablation_model
    for ablation_module in config.pmat_ablation_modules:
        logger.info(f"\n训练PMAT消融模型（移除{ablation_module}）")
        ablation_model = get_pmat_ablation_model(ablation_module, config).to(config.device)
        ablation_model = train_model(
            ablation_model, train_loader, val_loader, f"PMAT_ablation_{ablation_module}", logger=logger, eval_kwargs=eval_kwargs
        )
        ablation_model.eval()
        ablation_metrics = ablation_model._validate_one_epoch(test_loader, stage_id=1, stage_kwargs=eval_kwargs)
        ablation_metrics["model"] = f"PMAT_w/o_{ablation_module}"
        results.append(ablation_metrics)
        logger.info(f"PMAT w/o {ablation_module}: HR@10={ablation_metrics.get('HR@10', 0):.4f}, NDCG@10={ablation_metrics.get('NDCG@10', 0):.4f}")

    # ==================== MCRL消融实验（两阶段训练） ====================
    logger.info("\n" + "-" * 50)
    logger.info("MCRL消融实验（两阶段训练）")
    logger.info("-" * 50)

    # 完整MCRL模型（两阶段训练）
    logger.info("\n训练完整MCRL模型（两阶段训练）")
    mcrl_full = MCRL(config).to(config.device)
    mcrl_full = train_mcrl_two_stage(mcrl_full, train_loader, val_loader, "MCRL_full", logger=logger, eval_kwargs=eval_kwargs)
    mcrl_full.eval()
    mcrl_full_metrics = mcrl_full._validate_one_epoch(test_loader, stage_id=2, stage_kwargs=eval_kwargs)
    mcrl_full_metrics["model"] = "MCRL_full_TwoStage"
    results.append(mcrl_full_metrics)
    logger.info(f"MCRL完整模型: HR@10={mcrl_full_metrics.get('HR@10', 0):.4f}, NDCG@10={mcrl_full_metrics.get('NDCG@10', 0):.4f}")

    # MCRL消融实验（两阶段训练）
    from our_models.mcrl import get_mcrl_ablation_model
    for ablation_module in config.mcrl_ablation_modules:
        logger.info(f"\n训练MCRL消融模型（移除{ablation_module}，两阶段训练）")
        ablation_model = get_mcrl_ablation_model(ablation_module, config).to(config.device)
        ablation_model = train_mcrl_two_stage(
            ablation_model, train_loader, val_loader, f"MCRL_ablation_{ablation_module}", logger=logger, eval_kwargs=eval_kwargs
        )
        ablation_model.eval()
        ablation_metrics = ablation_model._validate_one_epoch(test_loader, stage_id=2, stage_kwargs=eval_kwargs)
        ablation_metrics["model"] = f"MCRL_w/o_{ablation_module}_TwoStage"
        results.append(ablation_metrics)
        logger.info(f"MCRL w/o {ablation_module}: HR@10={ablation_metrics.get('HR@10', 0):.4f}, NDCG@10={ablation_metrics.get('NDCG@10', 0):.4f}")

    # 保存结果
    save_results(results, "ablation", logger=logger)

    logger.info("\n" + "=" * 70)
    logger.info("消融实验完成")
    logger.info("=" * 70 + "\n")

    return results


# 3. 超参实验
def run_hyper_param_experiment(logger=None, quick_mode: bool = False):
    """运行超参实验（PMAT和MCRL）

    PMAT超参：id_length, lr, codebook_size
    MCRL超参：mcrl_alpha, mcrl_beta, lr
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    logger.info("\n" + "=" * 70)
    logger.info("===== 开始超参实验 =====")
    logger.info("=" * 70 + "\n")

    # 使用PMAT专用数据加载器
    train_loader, val_loader, test_loader, all_item_features = get_pmat_dataloader(
        cache_dir="./data",
        category=config.category,
        batch_size=config.batch_size,
        max_history_len=config.max_history_len,
        num_negative_samples=config.num_negative_samples,
        shuffle=True,
        quick_mode=quick_mode,
        num_workers=NUM_WORKS,
        logger=logger
    )

    # Full Ranking评估需要的stage_kwargs
    eval_kwargs = {'all_item_features': all_item_features}

    results = []

    # ==================== PMAT超参实验 ====================
    logger.info("\n" + "-" * 50)
    logger.info("PMAT超参实验")
    logger.info("-" * 50)

    # 保存原始配置
    original_id_length = config.id_length
    original_lr = config.lr
    original_codebook_size = config.codebook_size

    for id_length in config.hyper_param_search["id_length"]:
        for lr in config.hyper_param_search["lr"]:
            for codebook_size in config.hyper_param_search["codebook_size"]:
                logger.info(f"\nPMAT超参组合：id_length={id_length}, lr={lr}, codebook_size={codebook_size}")
                # 更新配置
                config.id_length = id_length
                config.lr = lr
                config.codebook_size = codebook_size

                # 训练模型
                model = PMAT(config).to(config.device)
                exp_name = f"PMAT_hyper_id{id_length}_lr{lr}_cb{codebook_size}"
                model = train_model(model, train_loader, val_loader, exp_name, logger=logger, eval_kwargs=eval_kwargs)

                # 评估
                model.eval()
                metrics = model._validate_one_epoch(test_loader, stage_id=1, stage_kwargs=eval_kwargs)
                metrics["model"] = exp_name
                metrics["id_length"] = id_length
                metrics["lr"] = lr
                metrics["codebook_size"] = codebook_size
                results.append(metrics)
                logger.info(f"  HR@10={metrics.get('HR@10', 0):.4f}, NDCG@10={metrics.get('NDCG@10', 0):.4f}")

                # 清理内存，防止内存溢出
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    # 恢复原始配置
    config.id_length = original_id_length
    config.lr = original_lr
    config.codebook_size = original_codebook_size

    # ==================== MCRL超参实验（两阶段训练） ====================
    logger.info("\n" + "-" * 50)
    logger.info("MCRL超参实验（两阶段训练）")
    logger.info("-" * 50)

    # 保存原始配置
    original_alpha = config.mcrl_alpha
    original_beta = config.mcrl_beta
    original_lr = config.lr

    for alpha in config.hyper_param_search["mcrl_alpha"]:
        for beta in config.hyper_param_search["mcrl_beta"]:
            for lr in config.hyper_param_search["lr"]:
                logger.info(f"\nMCRL超参组合：alpha={alpha}, beta={beta}, lr={lr}（两阶段训练）")
                # 更新配置
                config.mcrl_alpha = alpha
                config.mcrl_beta = beta
                config.lr = lr

                # 训练模型（两阶段训练）
                model = MCRL(config).to(config.device)
                exp_name = f"MCRL_hyper_a{alpha}_b{beta}_lr{lr}_TwoStage"
                model = train_mcrl_two_stage(model, train_loader, val_loader, exp_name, logger=logger, eval_kwargs=eval_kwargs)

                # 评估
                model.eval()
                metrics = model._validate_one_epoch(test_loader, stage_id=2, stage_kwargs=eval_kwargs)
                metrics["model"] = exp_name
                metrics["mcrl_alpha"] = alpha
                metrics["mcrl_beta"] = beta
                metrics["lr"] = lr
                results.append(metrics)
                logger.info(f"  HR@10={metrics.get('HR@10', 0):.4f}, NDCG@10={metrics.get('NDCG@10', 0):.4f}")

                # 清理内存，防止内存溢出
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    # 恢复原始配置
    config.mcrl_alpha = original_alpha
    config.mcrl_beta = original_beta
    config.lr = original_lr

    # 保存结果
    save_results(results, "hyper_param", logger=logger)

    logger.info("\n" + "=" * 70)
    logger.info("超参实验完成")
    logger.info("=" * 70 + "\n")

    return results


# ==================== 新增：PMAT/MCRL 推荐模型实验 ====================

def run_pmat_recommendation_experiment(logger=None, quick_mode=False):
    """运行PMAT-SASRec推荐模型实验（两阶段训练）

    两阶段训练流程：
    - 阶段1：预训练物品编码器（对比学习）
    - 阶段2：冻结物品编码器，训练序列模型（Cross Entropy）
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    logger.info("\n" + "="*70)
    logger.info("===== PMAT-SASRec推荐模型实验（两阶段训练） =====")
    logger.info("="*70 + "\n")

    # 在 CPU 模式下使用较小的 batch_size
    effective_batch_size = config.batch_size
    if config.device == torch.device('cpu') or str(config.device) == 'cpu':
        effective_batch_size = min(config.batch_size, 256)
        logger.info(f"CPU模式：将batch_size从{config.batch_size}调整为{effective_batch_size}以节省内存")


    # 创建模型
    logger.info("创建PMAT-SASRec推荐模型...")
    model = PMAT_SASRec(config, device=config.device).to(config.device)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数总数: {total_params:,}")

    # ==================== 两阶段训练 ====================
    use_two_stage = getattr(config, 'two_stage_training', True)

    if use_two_stage:
        logger.info("\n" + "-"*50)
        logger.info("===== 阶段1：预训练物品编码器（对比学习） =====")
        logger.info("-"*50)

        pretrain_loader, item_meta = get_all_item_pretrain_dataloader(
            cache_dir="./data",
            category="Video_Games",
            batch_size=effective_batch_size,  # 预训练用大batch
            quick_mode=quick_mode,
            logger=logger
        )
        config.num_items = item_meta["num_items"]

        # 冻结序列编码器，只训练物品编码器
        model.freeze_sequence_encoder()

        stage1_epochs = getattr(config, 'stage1_epochs', 5)
        stage1_lr = getattr(config, 'stage1_lr', 1e-3)

        stage1_config = StageConfig(
            stage_id=0,  # 预训练阶段
            epochs=stage1_epochs,
            start_epoch=0,
            kwargs={'lr': stage1_lr, 'weight_decay': config.weight_decay}
        )

        logger.info(f"阶段1配置: epochs={stage1_epochs}, lr={stage1_lr}")
        model.customer_train(
            train_dataloader=pretrain_loader,
            val_dataloader=None,
            stage_configs=[stage1_config],
            skip_validation=True  # 预训练阶段跳过验证
        )

        logger.info("\n" + "-"*50)
        logger.info("===== 阶段2：训练序列模型（Cross Entropy） =====")
        logger.info("-"*50)

        # 解冻序列编码器，冻结物品编码器
        model.unfreeze_sequence_encoder()
        model.unfreeze_item_encoder()

        logger.info("Stage 2 加载数据...")
        train_loader, val_loader, test_loader, all_item_features = get_pmat_dataloader(
            cache_dir="./data",
            category=config.category,
            batch_size=effective_batch_size,
            max_history_len=config.max_history_len,
            num_negative_samples=config.num_negative_samples,
            shuffle=True,
            quick_mode=quick_mode,
            num_workers=NUM_WORKS,
            logger=logger
        )

        eval_kwargs = {'all_item_features': all_item_features}

        # 预计算所有物品表征（使用训练好的物品编码器）
        logger.info("预计算所有物品表征...")
        model.set_all_item_features(all_item_features)

        stage2_epochs = getattr(config, 'stage2_epochs', 10)
        stage2_lr = getattr(config, 'stage2_lr', 1e-4)

        stage2_config = StageConfig(
            stage_id=1,  # 推荐阶段
            epochs=stage2_epochs,
            start_epoch=0,
            kwargs={'lr': stage2_lr, 'weight_decay': config.weight_decay, 'all_item_features': all_item_features}
        )

        logger.info(f"阶段2配置: epochs={stage2_epochs}, lr={stage2_lr}")
        model.customer_train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            stage_configs=[stage2_config],
            skip_validation=True
        )
    else:
        # 单阶段训练（原有逻辑）
        logger.info("使用单阶段训练...")
        model.set_all_item_features(all_item_features)
        model = train_model(model, train_loader, val_loader, "PMAT_SASRec", logger=logger, eval_kwargs=eval_kwargs)

    # ==================== 评估 ====================
    logger.info("\n" + "-"*50)
    logger.info("===== 最终评估 =====")
    logger.info("-"*50)

    model.eval()
    metrics = model._validate_one_epoch(test_loader, stage_id=1, stage_kwargs=eval_kwargs)

    logger.info("\nPMAT-SASRec推荐模型评估结果:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # 保存结果
    results = [{"model": "PMAT_SASRec", **metrics}]
    save_results(results, "pmat_sasrec_experiment", logger=logger)

    logger.info("\n" + "="*70)
    logger.info("PMAT-SASRec推荐模型实验完成")
    logger.info("="*70 + "\n")

    return metrics


def run_mcrl_sasrec_experiment(logger=None, quick_mode=False):
    """运行MCRL-SASRec推荐模型实验（两阶段训练）

    两阶段训练流程：
    - 阶段1：预训练物品编码器（对比学习）
    - 阶段2：冻结物品编码器，训练序列模型（Cross Entropy）
    """
    if logger is None:
        logger = logging.getLogger("MCRL_Experiment")

    logger.info("\n" + "="*70)
    logger.info("===== MCRL-SASRec推荐模型实验（两阶段训练） =====")
    logger.info("="*70 + "\n")

    # 加载数据
    logger.info("加载数据...")
    train_loader, val_loader, test_loader, all_item_features = get_pmat_dataloader(
        cache_dir="./data",
        category=config.category,
        batch_size=config.batch_size,
        max_history_len=config.max_history_len,
        num_negative_samples=config.num_negative_samples,
        shuffle=True,
        quick_mode=quick_mode,
        num_workers=NUM_WORKS,
    )

    eval_kwargs = {'all_item_features': all_item_features}

    # 创建模型
    logger.info("创建MCRL-SASRec推荐模型...")
    model = MCRL_SASRec(config).to(config.device)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数总数: {total_params:,}")

    # ==================== 两阶段训练 ====================
    use_two_stage = getattr(config, 'two_stage_training', True)

    if use_two_stage:
        logger.info("\n" + "-"*50)
        logger.info("===== 阶段1：预训练物品编码器（对比学习） =====")
        logger.info("-"*50)

        # 冻结序列编码器，只训练物品编码器
        model.freeze_sequence_encoder()

        stage1_epochs = getattr(config, 'stage1_epochs', 3)
        stage1_lr = getattr(config, 'stage1_lr', 1e-3)

        stage1_config = StageConfig(
            stage_id=0,  # 预训练阶段
            epochs=stage1_epochs,
            start_epoch=0,
            kwargs={'lr': stage1_lr, 'weight_decay': config.weight_decay}
        )

        logger.info(f"阶段1配置: epochs={stage1_epochs}, lr={stage1_lr}")
        model.customer_train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            stage_configs=[stage1_config],
            skip_validation=True
        )

        logger.info("\n" + "-"*50)
        logger.info("===== 阶段2：训练序列模型（Cross Entropy） =====")
        logger.info("-"*50)

        # 解冻序列编码器，冻结物品编码器
        model.unfreeze_sequence_encoder()
        model.freeze_item_encoder()

        # 预计算所有物品表征
        logger.info("预计算所有物品表征...")
        model.set_all_item_features(all_item_features)

        stage2_epochs = getattr(config, 'stage2_epochs', 5)
        stage2_lr = getattr(config, 'stage2_lr', 1e-4)

        stage2_config = StageConfig(
            stage_id=1,  # 推荐阶段
            epochs=stage2_epochs,
            start_epoch=0,
            kwargs={'lr': stage2_lr, 'weight_decay': config.weight_decay, 'all_item_features': all_item_features}
        )

        logger.info(f"阶段2配置: epochs={stage2_epochs}, lr={stage2_lr}")
        model.customer_train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            stage_configs=[stage2_config],
            skip_validation=True
        )
    else:
        # 单阶段训练（原有逻辑）
        logger.info("使用单阶段训练...")
        model.set_all_item_features(all_item_features)

        stage_config = StageConfig(
            stage_id=1,
            epochs=config.epochs,
            start_epoch=0,
            kwargs={'lr': config.lr, 'weight_decay': config.weight_decay, 'all_item_features': all_item_features}
        )

        model.customer_train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            stage_configs=[stage_config],
            skip_validation=True
        )

    # ==================== 评估 ====================
    logger.info("\n" + "-"*50)
    logger.info("===== 最终评估 =====")
    logger.info("-"*50)

    model.eval()
    metrics = model._validate_one_epoch(test_loader, stage_id=1, stage_kwargs=eval_kwargs)

    logger.info("\nMCRL-SASRec推荐模型评估结果:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # 保存结果
    results = [{"model": "MCRL_SASRec", **metrics}]
    save_results(results, "mcrl_sasrec_experiment", logger=logger)

    logger.info("\n" + "="*70)
    logger.info("MCRL-SASRec推荐模型实验完成")
    logger.info("="*70 + "\n")

    return metrics


def run_pure_sasrec_experiment(logger=None, quick_mode=False):
    """运行纯净SASRec基线实验（验证骨架）

    这是一个最简化的实现，用于验证：
    1. 数据加载是否正确
    2. SASRec 骨架是否正常工作
    3. Cross Entropy 损失是否正确

    不包含任何创新组件。
    """
    if logger is None:
        logger = logging.getLogger("PureSASRec_Experiment")

    logger.info("\n" + "="*70)
    logger.info("===== 纯净SASRec基线实验（验证骨架） =====")
    logger.info("="*70 + "\n")

    # batch_size
    effective_batch_size = config.batch_size
    if config.device == torch.device('cpu') or str(config.device) == 'cpu':
        effective_batch_size = min(config.batch_size, 16)
        logger.info(f"CPU模式：将batch_size从{config.batch_size}调整为{effective_batch_size}")

    # 加载数据
    logger.info("加载数据...")
    train_loader, val_loader, test_loader, all_item_features = get_pmat_dataloader(
        cache_dir="./data",
        category=config.category,
        batch_size=effective_batch_size,
        max_history_len=config.max_history_len,
        num_negative_samples=config.num_negative_samples,
        shuffle=True,
        quick_mode=quick_mode,
        num_workers=NUM_WORKS,
        logger=logger
    )

    eval_kwargs = {'all_item_features': all_item_features}

    # 创建模型
    logger.info("创建纯净SASRec模型...")
    config.num_items = all_item_features["num_items"]
    model = RecBoleSASRec(config, device=config.device).to(config.device)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数总数: {total_params:,}")

    # 预计算物品表征
    logger.info("预计算所有物品表征...")


    # 单阶段训练
    logger.info("\n" + "-"*50)
    logger.info("===== 训练（单阶段，Cross Entropy） =====")
    logger.info("-"*50)

    stage_config = StageConfig(
        stage_id=0,
        epochs=config.epochs,
        start_epoch=0,
        kwargs={
            'lr': config.lr,
            'weight_decay': config.weight_decay,
            "eta_min": config.eta_min,
            'all_item_features': all_item_features
        }
    )

    logger.info(f"训练配置: epochs={config.epochs}, lr={config.lr}")
    model.customer_train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        stage_configs=[stage_config],
        skip_validation=False
    )

    # 评估
    logger.info("\n" + "-"*50)
    logger.info("===== 最终评估 =====")
    logger.info("-"*50)

    model.eval()
    metrics = model._validate_one_epoch(test_loader, stage_id=0, stage_kwargs=eval_kwargs)

    logger.info("\n纯净SASRec评估结果:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # 保存结果
    results = [{"model": "PureSASRec", **metrics}]
    save_results(results, "pure_sasrec_experiment", logger=logger)

    logger.info("\n" + "="*70)
    logger.info("纯净SASRec基线实验完成")
    logger.info("="*70 + "\n")

    return metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PMAT & MCRL 实验')

    # 模式选择
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'baseline', 'ablation', 'hyper',
                                'pmat_rec', 'mcrl_sasrec', 'pure_sasrec'],
                        help='实验模式: all(运行PMAT和MCRL)/baseline(基线对比)/ablation(消融实验)/hyper(超参实验)/pmat_rec(PMAT推荐)/mcrl_sasrec(MCRL-SASRec推荐)')

    # 快速模式（独立参数）
    parser.add_argument('--quick', action='store_true', default=False,
                        help='快速模式：使用抽样数据进行快速验证')

    # 数据集
    parser.add_argument('--dataset', type=str, default='amazon',
                        choices=['amazon', 'movielens'],
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
    # 根据quick模式设置默认参数
    if args.quick:
        config.epochs = args.epochs if args.epochs is not None else 50
        config.batch_size = args.batch_size if args.batch_size is not None else 32
        config.category = "Video_Games"
    else:
        config.epochs = args.epochs if args.epochs is not None else 50
        config.batch_size = args.batch_size if args.batch_size is not None else 512

    # 学习率
    if args.lr is not None:
        config.lr = args.lr

    # 设备
    if args.device is not None:
        config.device = torch.device(args.device)

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info(f"配置已更新: epochs={config.epochs}, batch_size={config.batch_size}, lr={config.lr}, device={config.device}, quick_mode={args.quick}")


if __name__ == "__main__":
    # 设置日志
    os.makedirs(config.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, f'experiment_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'), encoding='utf-8'),
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
    logger.info(f"快速模式: {args.quick}")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"模型: {args.model}")
    logger.info("="*70)

    # quick_mode 由 --quick 参数控制
    quick_mode = args.quick

    # 根据模式运行实验
    if args.mode == 'all':
        logger.info(f"运行PMAT-SASRec和MCRL-SASRec推荐实验 (quick_mode={quick_mode})")
        run_pmat_recommendation_experiment(logger=logger, quick_mode=quick_mode)
        run_mcrl_sasrec_experiment(logger=logger, quick_mode=quick_mode)
    elif args.mode == 'baseline':
        logger.info(f"基线实验模式 (quick_mode={quick_mode})")
        run_baseline_experiment(logger=logger, quick_mode=quick_mode)
    elif args.mode == 'ablation':
        logger.info(f"消融实验模式 (quick_mode={quick_mode})")
        run_ablation_experiment(logger=logger, quick_mode=quick_mode)
    elif args.mode == 'hyper':
        logger.info(f"超参实验模式 (quick_mode={quick_mode})")
        run_hyper_param_experiment(logger=logger, quick_mode=quick_mode)
    elif args.mode == 'pmat_rec':
        logger.info(f"PMAT-SASRec推荐模型实验模式 (quick_mode={quick_mode})")
        run_pmat_recommendation_experiment(logger=logger, quick_mode=quick_mode)
    elif args.mode == 'mcrl_sasrec':
        logger.info(f"MCRL-SASRec推荐模型实验模式 (quick_mode={quick_mode})")
        run_mcrl_sasrec_experiment(logger=logger, quick_mode=quick_mode)
    elif args.mode == 'pure_sasrec':
        logger.info(f"纯净SASRec基线实验模式 (quick_mode={quick_mode})")
        run_pure_sasrec_experiment(logger=logger, quick_mode=quick_mode)

    logger.info("所有实验完成！")
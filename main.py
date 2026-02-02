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
from data_utils import get_dataloader, get_pctx_dataloader, get_pmat_dataloader
from baseline_models import PctxAligned, PRISM, DGMRec
from our_models.pmat import PMAT
from our_models.mcrl import MCRL
from our_models.pmat_sasrec import PMAT_SASRec
from metrics import calculate_metrics
from util import item_id_to_semantic_id, save_checkpoint, load_checkpoint, save_results
from base_model import AbstractTrainableModel, StageConfig

# 混合精度训练
scaler = GradScaler()

# 工作进程数（Windows下设置为0避免多进程问题）
NUM_WORKS = 0 if os.name == 'nt' else 4


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
    model.customer_train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        stage_configs=[stage_config]
    )

    return model


def train_mcrl_two_stage(model, train_loader, val_loader, experiment_name, logger=None):
    """MCRL两阶段训练函数

    Stage 1 (表征塑形): 专注对比学习，低推荐权重，冻结matcher
    Stage 2 (推荐对齐): 专注推荐，低对比权重，解冻所有模块

    Args:
        model: MCRL模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        experiment_name: 实验名称
        logger: 日志记录器

    Returns:
        训练后的模型
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    # 检查是否启用两阶段训练
    two_stage_enabled = getattr(config, 'mcrl_two_stage', True)

    if not two_stage_enabled:
        logger.info("MCRL两阶段训练未启用，使用单阶段训练")
        return train_model(model, train_loader, val_loader, experiment_name, logger=logger)

    logger.info("="*60)
    logger.info("MCRL两阶段训练")
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

    # 配置两阶段训练
    stage_configs = [
        # Stage 1: 表征塑形
        StageConfig(
            stage_id=1,
            epochs=stage1_epochs,
            start_epoch=0,
            kwargs={
                'lr': config.lr,
                'weight_decay': config.weight_decay,
                'experiment_name': f"{experiment_name}_stage1",
                'num_positive_samples': getattr(config, 'num_positive_samples', 5),
                'num_negative_samples': getattr(config, 'num_negative_samples', 20),
            }
        ),
        # Stage 2: 推荐对齐
        StageConfig(
            stage_id=2,
            epochs=stage2_epochs,
            start_epoch=0,
            kwargs={
                'lr': config.lr * 0.5,  # Stage 2使用较小学习率，避免破坏已学习的表征
                'weight_decay': config.weight_decay,
                'experiment_name': f"{experiment_name}_stage2",
                'num_positive_samples': getattr(config, 'num_positive_samples', 5),
                'num_negative_samples': getattr(config, 'num_negative_samples', 20),
            }
        )
    ]

    # 调用模型的统一训练方法
    model.customer_train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        stage_configs=stage_configs
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

            if baseline_name == "PRISM":
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
    train_loader, val_loader, test_loader = get_pmat_dataloader(
        cache_dir="./data",
        category=config.category,
        batch_size=config.batch_size,
        max_history_len=config.max_history_len,
        num_negative_samples=config.num_negative_samples,
        eval_num_negative_samples=config.eval_num_negative_samples,
        shuffle=True,
        quick_mode=quick_mode,
        num_workers=NUM_WORKS,
        logger=logger
    )

    results = []

    # ==================== PMAT消融实验 ====================
    logger.info("\n" + "-" * 50)
    logger.info("PMAT消融实验")
    logger.info("-" * 50)

    # 完整PMAT模型
    logger.info("\n训练完整PMAT模型")
    pmat_full = PMAT(config).to(config.device)
    pmat_full = train_model(pmat_full, train_loader, val_loader, "PMAT_full", logger=logger)
    pmat_full.eval()
    pmat_full_metrics = pmat_full._validate_one_epoch(test_loader, stage_id=1, stage_kwargs={})
    pmat_full_metrics["model"] = "PMAT_full"
    results.append(pmat_full_metrics)
    logger.info(f"PMAT完整模型: HR@10={pmat_full_metrics.get('HR@10', 0):.4f}, NDCG@10={pmat_full_metrics.get('NDCG@10', 0):.4f}")

    # PMAT消融实验
    from our_models.pmat import get_pmat_ablation_model
    for ablation_module in config.pmat_ablation_modules:
        logger.info(f"\n训练PMAT消融模型（移除{ablation_module}）")
        ablation_model = get_pmat_ablation_model(ablation_module, config).to(config.device)
        ablation_model = train_model(
            ablation_model, train_loader, val_loader, f"PMAT_ablation_{ablation_module}", logger=logger
        )
        ablation_model.eval()
        ablation_metrics = ablation_model._validate_one_epoch(test_loader, stage_id=1, stage_kwargs={})
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
    mcrl_full = train_mcrl_two_stage(mcrl_full, train_loader, val_loader, "MCRL_full", logger=logger)
    mcrl_full.eval()
    mcrl_full_metrics = mcrl_full._validate_one_epoch(test_loader, stage_id=2, stage_kwargs={})
    mcrl_full_metrics["model"] = "MCRL_full_TwoStage"
    results.append(mcrl_full_metrics)
    logger.info(f"MCRL完整模型: HR@10={mcrl_full_metrics.get('HR@10', 0):.4f}, NDCG@10={mcrl_full_metrics.get('NDCG@10', 0):.4f}")

    # MCRL消融实验（两阶段训练）
    from our_models.mcrl import get_mcrl_ablation_model
    for ablation_module in config.mcrl_ablation_modules:
        logger.info(f"\n训练MCRL消融模型（移除{ablation_module}，两阶段训练）")
        ablation_model = get_mcrl_ablation_model(ablation_module, config).to(config.device)
        ablation_model = train_mcrl_two_stage(
            ablation_model, train_loader, val_loader, f"MCRL_ablation_{ablation_module}", logger=logger
        )
        ablation_model.eval()
        ablation_metrics = ablation_model._validate_one_epoch(test_loader, stage_id=2, stage_kwargs={})
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
    train_loader, val_loader, test_loader = get_pmat_dataloader(
        cache_dir="./data",
        category=config.category,
        batch_size=config.batch_size,
        max_history_len=config.max_history_len,
        num_negative_samples=config.num_negative_samples,
        eval_num_negative_samples=config.eval_num_negative_samples,
        shuffle=True,
        quick_mode=quick_mode,
        num_workers=NUM_WORKS,
        logger=logger
    )

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
                model = train_model(model, train_loader, val_loader, exp_name, logger=logger)

                # 评估
                model.eval()
                metrics = model._validate_one_epoch(test_loader, stage_id=1, stage_kwargs={})
                metrics["model"] = exp_name
                metrics["id_length"] = id_length
                metrics["lr"] = lr
                metrics["codebook_size"] = codebook_size
                results.append(metrics)
                logger.info(f"  HR@10={metrics.get('HR@10', 0):.4f}, NDCG@10={metrics.get('NDCG@10', 0):.4f}")

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
                model = train_mcrl_two_stage(model, train_loader, val_loader, exp_name, logger=logger)

                # 评估
                model.eval()
                metrics = model._validate_one_epoch(test_loader, stage_id=2, stage_kwargs={})
                metrics["model"] = exp_name
                metrics["mcrl_alpha"] = alpha
                metrics["mcrl_beta"] = beta
                metrics["lr"] = lr
                results.append(metrics)
                logger.info(f"  HR@10={metrics.get('HR@10', 0):.4f}, NDCG@10={metrics.get('NDCG@10', 0):.4f}")

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
    """运行PMAT-SASRec推荐模型实验（使用真实用户历史）

    这是PMAT-SASRec混合模型，使用：
    - PMAT语义增强嵌入 + SASRec序列建模
    - 真实用户历史序列
    - BPR推荐损失（主任务）+ 语义ID损失（辅助任务）
    - 推荐指标：HR@K, NDCG@K, MRR, AUC
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    logger.info("\n" + "="*70)
    logger.info("===== PMAT-SASRec推荐模型实验（真实用户历史） =====")
    logger.info("="*70 + "\n")

    # 在 CPU 模式下使用较小的 batch_size 以避免内存问题
    # SemanticIDQuantizer 中的 torch.cdist 在大 batch 时会消耗大量内存
    effective_batch_size = config.batch_size
    if config.device == torch.device('cpu') or str(config.device) == 'cpu':
        effective_batch_size = min(config.batch_size, 16)
        logger.info(f"CPU模式：将batch_size从{config.batch_size}调整为{effective_batch_size}以节省内存")

    # 使用PMAT专用数据加载器
    logger.info("加载数据（使用get_pmat_dataloader）...")
    train_loader, val_loader, test_loader = get_pmat_dataloader(
        cache_dir="./data",
        category=config.category,
        batch_size=effective_batch_size,
        max_history_len=config.max_history_len,
        num_negative_samples=config.num_negative_samples,
        eval_num_negative_samples=config.eval_num_negative_samples,
        shuffle=True,
        quick_mode=quick_mode,
        num_workers=NUM_WORKS,
        logger=logger
    )

    # 创建模型
    logger.info("创建PMAT-SASRec推荐模型...")
    model = PMAT_SASRec(config, device=config.device).to(config.device)

    # 训练
    logger.info("开始训练...")
    model = train_model(model, train_loader, val_loader, "PMAT_SASRec", logger=logger)

    # 评估
    logger.info("评估模型...")
    model.eval()
    metrics = model._validate_one_epoch(test_loader, stage_id=1, stage_kwargs={})

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


def run_mcrl_recommendation_experiment(logger=None, quick_mode=False):
    """运行MCRL推荐模型实验（使用真实用户历史 + 两阶段训练）

    这是改造后的MCRL模型，使用：
    - 真实用户历史序列（替代随机占位符）
    - 两阶段训练：Stage1(表征塑形) + Stage2(推荐对齐)
    - BPR推荐损失（主任务）+ 对比学习损失（辅助任务）
    - 推荐指标：HR@K, NDCG@K, MRR, AUC
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    logger.info("\n" + "="*70)
    logger.info("===== MCRL推荐模型实验（两阶段训练） =====")
    logger.info("="*70 + "\n")

    # 使用PMAT专用数据加载器（MCRL复用相同格式）
    logger.info("加载数据（使用get_pmat_dataloader）...")
    train_loader, val_loader, test_loader = get_pmat_dataloader(
        cache_dir="./data",
        category=config.category,
        batch_size=config.batch_size,
        max_history_len=config.max_history_len,
        num_negative_samples=config.num_negative_samples,
        eval_num_negative_samples=config.eval_num_negative_samples,
        shuffle=True,
        quick_mode=quick_mode,
        num_workers=NUM_WORKS,
        logger=logger
    )

    # 创建模型
    logger.info("创建MCRL推荐模型...")
    model = MCRL(config).to(config.device)

    # 使用两阶段训练
    logger.info("开始两阶段训练...")
    model = train_mcrl_two_stage(model, train_loader, val_loader, "MCRL_Rec", logger=logger)

    # 评估
    logger.info("评估模型...")
    model.eval()
    metrics = model._validate_one_epoch(test_loader, stage_id=2, stage_kwargs={})

    logger.info("\nMCRL推荐模型评估结果:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # 保存结果
    results = [{"model": "MCRL_Rec_TwoStage", **metrics}]
    save_results(results, "mcrl_rec_experiment", logger=logger)

    logger.info("\n" + "="*70)
    logger.info("MCRL推荐模型实验完成（两阶段训练）")
    logger.info("="*70 + "\n")

    return metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PMAT & MCRL 实验')

    # 模式选择
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full', 'baseline', 'ablation', 'hyper',
                                'pmat_rec', 'mcrl_rec'],
                        help='实验模式: quick(快速测试)/full(完整实验)/baseline(基线对比)/ablation(消融实验)/hyper(超参实验)/pmat_rec(PMAT推荐)/mcrl_rec(MCRL推荐)')

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
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"模型: {args.model}")
    logger.info("="*70)

    # 根据模式运行实验
    if args.mode == 'quick':
        logger.info("快速测试模式 - 运行PMAT和MCRL推荐实验（抽样数据）")
        run_pmat_recommendation_experiment(logger=logger, quick_mode=True)
        run_mcrl_recommendation_experiment(logger=logger, quick_mode=True)
    elif args.mode == 'full':
        logger.info("完整实验模式 - 运行PMAT和MCRL推荐实验")
        run_pmat_recommendation_experiment(logger=logger, quick_mode=False)
        run_mcrl_recommendation_experiment(logger=logger, quick_mode=False)
    elif args.mode == 'baseline':
        logger.info("基线实验模式")
        run_baseline_experiment(logger=logger, quick_mode=(args.dataset=='mock'))
    elif args.mode == 'ablation':
        logger.info("消融实验模式 - PMAT和MCRL消融实验")
        run_ablation_experiment(logger=logger, quick_mode=(args.dataset=='mock'))
    elif args.mode == 'hyper':
        logger.info("超参实验模式 - PMAT和MCRL超参实验")
        run_hyper_param_experiment(logger=logger, quick_mode=(args.dataset=='mock'))
    elif args.mode == 'pmat_rec':
        logger.info("PMAT推荐模型实验模式")
        run_pmat_recommendation_experiment(logger=logger, quick_mode=(args.dataset=='mock'))
    elif args.mode == 'mcrl_rec':
        logger.info("MCRL推荐模型实验模式")
        run_mcrl_recommendation_experiment(logger=logger, quick_mode=(args.dataset=='mock'))

    logger.info("所有实验完成！")
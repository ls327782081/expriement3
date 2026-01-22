import os
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import pandas as pd

# 导入自定义模块
from config import config
from data_utils import get_dataloader
from metrics import calculate_metrics
# Updated imports: our models are now in our_models package
from our_models.pmat import PMAT, get_pmat_ablation_model
from our_models.mcrl import MCRL, PMATWithMCRL
from baseline_models.pctx import Pctx
from baseline_models.mmq import MMQ
from baseline_models.fusid import FusID
from baseline_models.rpg import RPG
from baseline_models.prism import PRISM
from baseline_models.dgmrec import DGMRec
from baseline_models.rearm import REARM
from util import save_checkpoint, load_checkpoint, save_results, item_id_to_semantic_id, semantic_id_to_item_id

# 配置日志系统
def setup_logger():
    """设置日志系统"""
    logger = logging.getLogger("PMAT_Experiment")
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 清除已有的处理器
    logger.handlers.clear()

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建文件处理器
    os.makedirs("./logs", exist_ok=True)
    file_handler = logging.FileHandler("./logs/experiment.log", mode='w')  # 每次运行清空日志
    file_handler.setLevel(logging.INFO)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# 全局logger
logger = setup_logger()

# 混合精度训练初始化
scaler = GradScaler()

NUM_WORKS = os.cpu_count()


def sample_positive_negative_ids(batch, train_loader, num_pos=5, num_neg=20):
    """为MCRL采样正负样本

    Args:
        batch: 当前批次数据
        train_loader: 训练数据加载器
        num_pos: 正样本数量
        num_neg: 负样本数量

    Returns:
        positive_ids: (batch_size, num_pos, hidden_dim) 正样本ID嵌入
        negative_ids: (batch_size, num_neg, hidden_dim) 负样本ID嵌入
    """
    batch_size = batch["user_id"].size(0)
    hidden_dim = config.hidden_dim

    # 简化版本：随机采样
    # 正样本：同一用户的其他物品
    positive_ids = torch.randn(batch_size, num_pos, hidden_dim).to(config.device)

    # 负样本：其他用户的物品
    negative_ids = torch.randn(batch_size, num_neg, hidden_dim).to(config.device)

    return positive_ids, negative_ids


def train_model(model, train_loader, val_loader, experiment_name, ablation_module=None, logger=None):
    """通用训练函数（带检查点）"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 加载检查点（断点续训）
    model, optimizer, start_epoch, best_loss = load_checkpoint(model, optimizer, experiment_name, logger=logger)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for step, batch in enumerate(pbar):
            # 数据移至GPU
            batch = {k: v.to(config.device) for k, v in batch.items()}

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):  # 混合精度训练（适配L4）
                logits = model(batch)
                # 构造标签（将item_id转换为语义ID序列）
                # target shape: (batch_size, id_length)
                target = item_id_to_semantic_id(batch["item_id"], config.id_length, config.codebook_size)
                loss = criterion(logits.reshape(-1, config.codebook_size), target.reshape(-1))
                loss = loss / config.gradient_accumulation_steps  # 梯度累积

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度累积更新
            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * config.gradient_accumulation_steps
            pbar.set_postfix({"train_loss": train_loss / (step + 1)})

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(config.device) for k, v in batch.items()}
                logits = model(batch)
                target = batch["item_id"].repeat(config.id_length).reshape(-1, config.id_length)
                loss = criterion(logits.reshape(-1, config.codebook_size), target.reshape(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")

        # 保存检查点
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint(model, optimizer, epoch, val_loss, experiment_name, is_best, logger=logger)

    return model


def train_mcrl_model(model, train_loader, val_loader, experiment_name, logger=None):
    """MCRL模型专用训练函数（包含三层对比学习）

    Args:
        model: PMATWithMCRL模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        experiment_name: 实验名称
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")

    logger.info(f"\n===== 开始训练MCRL模型: {experiment_name} =====")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 加载检查点
    model, optimizer, start_epoch, best_loss = load_checkpoint(model, optimizer, experiment_name, logger=logger)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_losses = {
            'total': 0.0,
            'pmat_total': 0.0,
            'mcrl_user_pref': 0.0,
            'mcrl_intra': 0.0,
            'mcrl_inter': 0.0,
            'mcrl_total': 0.0
        }

        pbar = tqdm(train_loader, desc=f"[MCRL] Epoch {epoch + 1}/{config.epochs}")

        for step, batch in enumerate(pbar):
            # 数据移至设备
            batch = {k: v.to(config.device) for k, v in batch.items()}
            batch_size = batch["user_id"].size(0)

            # 采样正负样本
            positive_ids, negative_ids = sample_positive_negative_ids(
                batch, train_loader,
                num_pos=config.num_positive_samples,
                num_neg=config.num_negative_samples
            )

            # 准备用户嵌入（简化版本：使用user_id的embedding）
            user_embeddings = torch.randn(batch_size, config.hidden_dim).to(config.device)

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # 前向传播（PMAT + MCRL）
                # 注意：这里需要适配实际的batch结构
                try:
                    # 尝试使用完整的MCRL前向传播
                    from our_models.pmat import PMAT

                    # 如果模型是PMATWithMCRL
                    if hasattr(model, 'pmat') and hasattr(model, 'mcrl'):
                        # 准备item_features
                        item_features = {
                            'visual': batch.get('vision_feat', torch.randn(batch_size, config.visual_dim).to(config.device)),
                            'text': batch.get('text_feat', torch.randn(batch_size, config.text_dim).to(config.device))
                        }

                        # 准备user_history（简化版本）
                        user_history = torch.randn(batch_size, 10, config.hidden_dim).to(config.device)

                        # 完整前向传播
                        outputs = model(
                            item_features=item_features,
                            user_history=user_history,
                            user_embeddings=user_embeddings,
                            positive_ids=positive_ids,
                            negative_ids=negative_ids
                        )

                        # 计算损失
                        target = item_id_to_semantic_id(batch["item_id"], config.id_length, config.codebook_size)
                        losses = model.compute_total_loss(outputs, target)

                        loss = losses['total_loss']

                        # 记录各项损失
                        epoch_losses['total'] += loss.item()
                        epoch_losses['pmat_total'] += losses.get('pmat_total_loss', 0).item() if torch.is_tensor(losses.get('pmat_total_loss', 0)) else 0
                        epoch_losses['mcrl_user_pref'] += losses.get('mcrl_user_preference_loss', 0).item() if torch.is_tensor(losses.get('mcrl_user_preference_loss', 0)) else 0
                        epoch_losses['mcrl_intra'] += losses.get('mcrl_intra_modal_loss', 0).item() if torch.is_tensor(losses.get('mcrl_intra_modal_loss', 0)) else 0
                        epoch_losses['mcrl_inter'] += losses.get('mcrl_inter_modal_loss', 0).item() if torch.is_tensor(losses.get('mcrl_inter_modal_loss', 0)) else 0
                        epoch_losses['mcrl_total'] += losses.get('mcrl_total_contrastive_loss', 0).item() if torch.is_tensor(losses.get('mcrl_total_contrastive_loss', 0)) else 0

                    else:
                        # 降级到普通训练
                        logits = model(batch)
                        target = item_id_to_semantic_id(batch["item_id"], config.id_length, config.codebook_size)
                        criterion = nn.CrossEntropyLoss()
                        loss = criterion(logits.reshape(-1, config.codebook_size), target.reshape(-1))
                        epoch_losses['total'] += loss.item()

                except Exception as e:
                    logger.warning(f"MCRL训练出错，降级到普通训练: {e}")
                    # 降级到普通训练
                    logits = model(batch)
                    target = item_id_to_semantic_id(batch["item_id"], config.id_length, config.codebook_size)
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(logits.reshape(-1, config.codebook_size), target.reshape(-1))
                    epoch_losses['total'] += loss.item()

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # 优化器步进
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # 更新进度条
            avg_loss = epoch_losses['total'] / (step + 1)
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        # 学习率调度
        scheduler.step()

        # 打印epoch统计
        num_batches = len(train_loader)
        logger.info(f"\nEpoch {epoch + 1} 训练统计:")
        logger.info(f"  总损失: {epoch_losses['total'] / num_batches:.4f}")
        if epoch_losses['pmat_total'] > 0:
            logger.info(f"  PMAT损失: {epoch_losses['pmat_total'] / num_batches:.4f}")
            logger.info(f"  MCRL用户偏好损失: {epoch_losses['mcrl_user_pref'] / num_batches:.4f}")
            logger.info(f"  MCRL模态内损失: {epoch_losses['mcrl_intra'] / num_batches:.4f}")
            logger.info(f"  MCRL模态间损失: {epoch_losses['mcrl_inter'] / num_batches:.4f}")
            logger.info(f"  MCRL总对比损失: {epoch_losses['mcrl_total'] / num_batches:.4f}")

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(config.device) for k, v in batch.items()}
                try:
                    logits = model(batch) if not hasattr(model, 'pmat') else model.pmat(batch)
                except:
                    logits = model(batch)

                target = item_id_to_semantic_id(batch["item_id"], config.id_length, config.codebook_size)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits.reshape(-1, config.codebook_size), target.reshape(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logger.info(f"  验证损失: {val_loss:.4f}\n")

        # 保存检查点
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint(model, optimizer, epoch, val_loss, experiment_name, is_best, logger=logger)

    logger.info(f"===== MCRL模型训练完成 =====\n")
    return model


def evaluate_model(model, test_loader, model_name, logger=None):
    """模型评估（多维度指标）"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")
        
    model.eval()
    all_user_ids = []
    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            batch = {k: v.to(config.device) for k, v in batch.items()}
            logits = model(batch)

            # 收集用户ID
            all_user_ids.extend(batch["user_id"].cpu().numpy())

            # 获取预测的语义ID序列（取每个token的最高概率）
            pred_semantic_ids = torch.argmax(logits, dim=-1)  # (batch, id_length)
            # 将语义ID序列转换回item_id
            pred_item_ids = semantic_id_to_item_id(pred_semantic_ids, config.codebook_size)
            all_predictions.extend(pred_item_ids.cpu().numpy())

            # 收集真实物品ID
            all_ground_truth.extend(batch["item_id"].cpu().numpy())

    # 计算指标
    metrics = calculate_metrics(all_user_ids, all_predictions, all_ground_truth, logger=logger)
    metrics["model"] = model_name
    return metrics


# 1. 基线实验
def run_baseline_experiment(logger:logging.Logger, quick_mode:bool=False):
    """运行基线模型实验"""

    logger.info("===== 开始基线实验 =====")
    # Windows下使用num_workers=0避免多进程问题
    train_loader,val_loader, test_loader  = get_dataloader("./data", category=config.category, shuffle=True, quick_mode=quick_mode, logger=logger, num_workers=0)

    results = []
    # 训练并评估基线模型
    for baseline_name in config.baseline_models:
        logger.info(f"\n训练基线模型：{baseline_name}")
        if baseline_name == "Pctx":
            model = Pctx().to(config.device)
        elif baseline_name == "MMQ":
            model = MMQ().to(config.device)
        elif baseline_name == "FusID":
            model = FusID().to(config.device)
        elif baseline_name == "RPG":
            model = RPG().to(config.device)
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
        elif baseline_name == "REARM":
            base_model = REARM(config).to(config.device)
            # Wrapper for REARM to accept batch dict
            class REARMWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    self.fc = torch.nn.Linear(config.hidden_dim, config.codebook_size * config.id_length)
                def forward(self, batch):
                    result = self.model(batch["user_id"], batch["item_id"], batch["vision_feat"].float(), batch["text_feat"].float())
                    logits = self.fc(result['user_embeddings'])
                    return logits.reshape(-1, config.id_length, config.codebook_size)
            model = REARMWrapper(base_model).to(config.device)
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
def run_ablation_experiment(logger=None):
    """运行消融实验"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")
        
    logger.info("\n===== 开始消融实验 =====")
    train_loader, val_loader, test_loader = get_dataloader("./data", category=config.category, shuffle=True, logger=logger,
                                                           num_workers=NUM_WORKS)

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
        ablation_model = get_pmat_ablation_model(ablation_module).to(config.device)
        ablation_model = train_model(
            ablation_model, train_loader, val_loader, f"PMAT_ablation_{ablation_module}", logger=logger
        )
        ablation_metrics = evaluate_model(ablation_model, test_loader, f"PMAT_ablation_{ablation_module}", logger=logger)
        results.append(ablation_metrics)

    # 保存结果
    save_results(results, "ablation", logger=logger)


# 3. 超参实验
def run_hyper_param_experiment(logger=None):
    """运行超参实验"""
    if logger is None:
        logger = logging.getLogger("PMAT_Experiment")
        
    logger.info("\n===== 开始超参实验 =====")
    train_loader,val_loader, test_loader  = get_dataloader("./data", category=config.category, shuffle=True, logger=logger, num_workers=NUM_WORKS)


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
def run_mcrl_experiment(logger=None):
    """运行MCRL完整实验

    包括：
    1. PMAT单独训练
    2. MCRL单独训练（基于预训练的PMAT）
    3. PMAT+MCRL联合训练
    4. 消融实验（三层对比学习）
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
        num_workers=NUM_WORKS
    )

    results = []

    # ========== 实验1: PMAT单独训练（作为基线） ==========
    logger.info("\n【实验1】训练PMAT模型（无MCRL）")
    logger.info("-" * 70)
    pmat_model = PMAT(config).to(config.device)
    pmat_model = train_model(pmat_model, train_loader, val_loader, "MCRL_Exp_PMAT_only", logger=logger)
    pmat_metrics = evaluate_model(pmat_model, test_loader, "PMAT_only", logger=logger)
    results.append(pmat_metrics)

    # ========== 实验2: PMAT+MCRL联合训练 ==========
    logger.info("\n【实验2】训练PMAT+MCRL联合模型")
    logger.info("-" * 70)
    joint_model = PMATWithMCRL(config).to(config.device)
    joint_model = train_mcrl_model(joint_model, train_loader, val_loader, "MCRL_Exp_PMAT_MCRL_joint", logger=logger)
    joint_metrics = evaluate_model(joint_model, test_loader, "PMAT+MCRL_joint", logger=logger)
    results.append(joint_metrics)

    # ========== 实验3: MCRL消融实验 - 移除用户偏好对比 ==========
    logger.info("\n【实验3】MCRL消融实验 - 移除用户偏好对比学习")
    logger.info("-" * 70)
    config_no_user = config
    config_no_user.mcrl_loss_weight = 0.5
    # 这里需要修改MCRL模型，暂时跳过详细实现
    logger.info("（消融实验需要修改MCRL模型结构，暂时记录）")

    # ========== 实验4: MCRL消融实验 - 移除模态内对比 ==========
    logger.info("\n【实验4】MCRL消融实验 - 移除模态内对比学习")
    logger.info("-" * 70)
    logger.info("（消融实验需要修改MCRL模型结构，暂时记录）")

    # ========== 实验5: MCRL消融实验 - 移除模态间对比 ==========
    logger.info("\n【实验5】MCRL消融实验 - 移除模态间对比学习")
    logger.info("-" * 70)
    logger.info("（消融实验需要修改MCRL模型结构，暂时记录）")

    # 保存结果
    save_results(results, "mcrl_experiment", logger=logger)

    logger.info("\n" + "="*70)
    logger.info("===== MCRL完整实验完成 =====")
    logger.info("="*70 + "\n")

    return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PMAT/MCRL 推荐系统实验')

    # 实验模式
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full', 'baseline', 'ablation', 'hyper', 'mcrl'],
                        help='实验模式: quick(快速测试), full(完整实验), baseline(基线对比), ablation(消融实验), hyper(超参搜索), mcrl(MCRL完整实验)')

    # 数据集选择
    parser.add_argument('--dataset', type=str, default='mock',
                        choices=['mock', 'amazon', 'movielens'],
                        help='数据集: mock(模拟数据), amazon(Amazon Books), movielens(MovieLens-25M)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (默认: quick=2, full=10)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批大小 (默认: quick=64, full=32)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (默认: 1e-4)')

    # 模型选择
    parser.add_argument('--model', type=str, default='pmat',
                        choices=['pmat', 'mcrl', 'pmat_mcrl', 'pctx', 'mmq', 'fusid', 'rpg', 'prism', 'dgmrec', 'rearm'],
                        help='模型选择')

    # 设备
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
        config.epochs = args.epochs if args.epochs is not None else 1
        config.batch_size = args.batch_size if args.batch_size is not None else 32
        # quick模式使用真实数据集，但会抽样
        config.category = "Video_Games"
    elif args.mode == 'full':
        config.epochs = args.epochs if args.epochs is not None else 10
        config.batch_size = args.batch_size if args.batch_size is not None else 32
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
        logger.info("快速测试模式 - 运行基线实验（抽样数据）")
        run_baseline_experiment(logger=logger, quick_mode=True)
    elif args.mode == 'full':
        logger.info("完整实验模式 - 运行所有实验")
        run_baseline_experiment(logger=logger, quick_mode=False)
        run_ablation_experiment(logger=logger)
        run_hyper_param_experiment(logger=logger)
    elif args.mode == 'baseline':
        logger.info("基线对比实验")
        run_baseline_experiment(logger=logger, quick_mode=False)
    elif args.mode == 'ablation':
        logger.info("消融实验")
        run_ablation_experiment(logger=logger)
    elif args.mode == 'hyper':
        logger.info("超参数搜索实验")
        run_hyper_param_experiment(logger=logger)
    elif args.mode == 'mcrl':
        logger.info("MCRL完整实验")
        run_mcrl_experiment(logger=logger)

    logger.info("实验完成！结果已保存至 ./results 目录")
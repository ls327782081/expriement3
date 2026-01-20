import os
import logging

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
from models import PMAT, get_pmat_ablation_model
from baseline_models.pctx import Pctx
from baseline_models.mmq import MMQ
from baseline_models.fusid import FusID
from baseline_models.rpg import RPG
from baseline_models.pctx import Pctx
from baseline_models.mmq import MMQ
from util import save_checkpoint, load_checkpoint, save_results

# 配置日志系统
def setup_logger():
    """设置日志系统"""
    logger = logging.getLogger("PMAT_Experiment")
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()

    
    # 创建文件处理器
    os.makedirs("./logs", exist_ok=True)
    file_handler = logging.FileHandler("./logs/experiment.log")

    
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

            with autocast():  # 混合精度训练（适配L4）
                logits = model(batch)
                # 构造标签（使用语义ID）
                target = batch["item_id"].repeat(config.id_length).reshape(-1, config.id_length)
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
            
            # 获取预测的物品ID（取每个token的最高概率）
            pred_ids = torch.argmax(logits, dim=-1)  # (batch, id_length)
            # 将多token ID转换为单一物品ID（简化：取第一个token）
            predictions = pred_ids[:, 0].cpu().numpy()
            all_predictions.extend(predictions)
            
            # 收集真实物品ID
            all_ground_truth.extend(batch["item_id"].cpu().numpy())

    # 计算指标
    metrics = calculate_metrics(all_user_ids, all_predictions, all_ground_truth, logger=logger)
    metrics["model"] = model_name
    return metrics


# 1. 基线实验
def run_baseline_experiment(logger:logging.Logger):
    """运行基线模型实验"""
        
    logger.info("===== 开始基线实验 =====")
    train_loader,val_loader, test_loader  = get_dataloader("./data/train.pkl", shuffle=True, logger=logger, num_workers=NUM_WORKS)

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
    pmat_model = PMAT().to(config.device)
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
    train_loader, val_loader, test_loader = get_dataloader("./data/train.pkl", shuffle=True, logger=logger,
                                                           num_workers=NUM_WORKS)

    results = []
    # 完整模型
    logger.info("训练完整PMAT模型")
    full_model = PMAT().to(config.device)
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
    train_loader,val_loader, test_loader  = get_dataloader("./data/train.pkl", shuffle=True, logger=logger, num_workers=NUM_WORKS)


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
                model = PMAT().to(config.device)
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


if __name__ == "__main__":
    # 依次运行所有实验
    run_baseline_experiment(logger=logger)
    run_ablation_experiment(logger=logger)
    run_hyper_param_experiment(logger=logger)
    logger.info("所有实验完成！结果已保存至 ./results 目录")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import pandas as pd

# 导入自定义模块
from config import config
from data_utils import get_dataloader
from metrics import calculate_metrics
from models import PMAT, get_pmat_ablation_model
from baseline_models.pctx import Pctx
from baseline_models.mmq import MMQ
from utils import save_checkpoint, load_checkpoint, save_results

# 混合精度训练初始化
scaler = GradScaler()


def train_model(model, train_loader, val_loader, experiment_name, ablation_module=None):
    """通用训练函数（带检查点）"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 加载检查点（断点续训）
    model, optimizer, start_epoch, best_loss = load_checkpoint(model, optimizer, experiment_name)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for step, batch in enumerate(pbar):
            # 数据移至GPU
            batch = {k: v.to(config.device) for k, v in batch.items()}

            with autocast():  # 混合精度训练（适配L4）
                logits = model(batch)
                # 构造标签（简化：item_id作为目标）
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
        print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}")

        # 保存检查点
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint(model, optimizer, epoch, val_loss, experiment_name, is_best)

    return model


def evaluate_model(model, test_loader, model_name):
    """模型评估（多维度指标）"""
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            batch = {k: v.to(config.device) for k, v in batch.items()}
            logits = model(batch)

            # 构造真实标签（one-hot）
            true = torch.zeros(len(batch["item_id"]), config.item_vocab_size).to(config.device)
            true[torch.arange(len(batch["item_id"])), batch["item_id"]] = 1
            all_true.append(true.cpu().numpy())

            # 预测得分（取平均）
            pred = torch.mean(logits, dim=1)  # (batch, codebook_size)
            all_pred.append(pred.cpu().numpy())

    # 计算指标
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    metrics = calculate_metrics(all_true, all_pred)
    metrics["model"] = model_name
    return metrics


# 1. 基线实验
def run_baseline_experiment():
    """运行基线模型实验"""
    print("===== 开始基线实验 =====")
    train_loader = get_dataloader("./data/train.pkl", shuffle=True)
    val_loader = get_dataloader("./data/val.pkl", shuffle=False)
    test_loader = get_dataloader("./data/test.pkl", shuffle=False)

    results = []
    # 训练并评估基线模型
    for baseline_name in config.baseline_models:
        print(f"\n训练基线模型：{baseline_name}")
        if baseline_name == "Pctx":
            model = Pctx().to(config.device)
        elif baseline_name == "MMQ":
            model = MMQ().to(config.device)

        # 训练
        model = train_model(model, train_loader, val_loader, f"baseline_{baseline_name}")
        # 评估
        metrics = evaluate_model(model, test_loader, baseline_name)
        results.append(metrics)

    # 训练并评估原创模型PMAT
    print("\n训练原创模型：PMAT")
    pmat_model = PMAT().to(config.device)
    pmat_model = train_model(pmat_model, train_loader, val_loader, "PMAT")
    pmat_metrics = evaluate_model(pmat_model, test_loader, "PMAT")
    results.append(pmat_metrics)

    # 保存结果
    save_results(results, "baseline")


# 2. 消融实验
def run_ablation_experiment():
    """运行消融实验"""
    print("\n===== 开始消融实验 =====")
    train_loader = get_dataloader("./data/train.pkl", shuffle=True)
    val_loader = get_dataloader("./data/val.pkl", shuffle=False)
    test_loader = get_dataloader("./data/test.pkl", shuffle=False)

    results = []
    # 完整模型
    print("训练完整PMAT模型")
    full_model = PMAT().to(config.device)
    full_model = train_model(full_model, train_loader, val_loader, "PMAT_full")
    full_metrics = evaluate_model(full_model, test_loader, "PMAT_full")
    results.append(full_metrics)

    # 消融实验：逐个移除模块
    for ablation_module in config.ablation_modules:
        print(f"训练消融模型（移除{ablation_module}）")
        ablation_model = get_pmat_ablation_model(ablation_module).to(config.device)
        ablation_model = train_model(
            ablation_model, train_loader, val_loader, f"PMAT_ablation_{ablation_module}"
        )
        ablation_metrics = evaluate_model(ablation_model, test_loader, f"PMAT_ablation_{ablation_module}")
        results.append(ablation_metrics)

    # 保存结果
    save_results(results, "ablation")


# 3. 超参实验
def run_hyper_param_experiment():
    """运行超参实验"""
    print("\n===== 开始超参实验 =====")
    train_loader = get_dataloader("./data/train.pkl", shuffle=True)
    val_loader = get_dataloader("./data/val.pkl", shuffle=False)
    test_loader = get_dataloader("./data/test.pkl", shuffle=False)

    results = []
    # 遍历超参组合
    for id_length in config.hyper_param_search["id_length"]:
        for lr in config.hyper_param_search["lr"]:
            for codebook_size in config.hyper_param_search["codebook_size"]:
                print(f"\n超参组合：id_length={id_length}, lr={lr}, codebook_size={codebook_size}")
                # 更新配置
                config.id_length = id_length
                config.lr = lr
                config.codebook_size = codebook_size

                # 训练模型
                model = PMAT().to(config.device)
                exp_name = f"PMAT_hyper_id{id_length}_lr{lr}_cb{codebook_size}"
                model = train_model(model, train_loader, val_loader, exp_name)

                # 评估
                metrics = evaluate_model(model, test_loader, exp_name)
                # 添加超参信息
                metrics["id_length"] = id_length
                metrics["lr"] = lr
                metrics["codebook_size"] = codebook_size
                results.append(metrics)

    # 保存结果
    save_results(results, "hyper_param")


if __name__ == "__main__":
    # 依次运行所有实验
    run_baseline_experiment()
    run_ablation_experiment()
    run_hyper_param_experiment()
    print("所有实验完成！结果已保存至 ./results 目录")
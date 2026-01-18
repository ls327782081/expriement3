import os
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from config import config


# 检查点保存/加载
def save_checkpoint(model, optimizer, epoch, loss, experiment_name, is_best=False):
    """保存检查点（避免实验中断）"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"{experiment_name}_epoch_{epoch}.pth"
    )
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config.__dict__
    }
    torch.save(checkpoint, checkpoint_path)
    # 保存最优模型
    if is_best:
        torch.save(checkpoint, os.path.join(config.checkpoint_dir, f"{experiment_name}_best.pth"))
    print(f"检查点已保存：{checkpoint_path}")


def load_checkpoint(model, optimizer, experiment_name, epoch=None):
    """加载检查点"""
    if epoch is None:
        checkpoint_path = os.path.join(config.checkpoint_dir, f"{experiment_name}_best.pth")
    else:
        checkpoint_path = os.path.join(config.checkpoint_dir, f"{experiment_name}_epoch_{epoch}.pth")

    if not os.path.exists(checkpoint_path):
        print("检查点不存在，从头训练")
        return model, optimizer, 0, float("inf")

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["loss"]
    print(f"加载检查点成功：从第{start_epoch}轮继续训练")
    return model, optimizer, start_epoch, best_loss


# 实验结果保存
def save_results(results, experiment_type):
    """保存实验结果（csv/json）"""
    os.makedirs(config.result_dir, exist_ok=True)

    # 保存为CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(config.result_dir, f"{experiment_type}_results.csv")
    df.to_csv(csv_path, index=False)

    # 保存为JSON
    json_path = os.path.join(config.result_dir, f"{experiment_type}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # 可视化结果
    plot_results(df, experiment_type)
    print(f"实验结果已保存：{csv_path} | {json_path}")


def plot_results(df, experiment_type):
    """可视化实验结果"""
    # 1. Top-10指标对比
    metrics = ["Precision@10", "Recall@10", "NDCG@10", "MRR@10"]
    plt.figure(figsize=(12, 8))
    for metric in metrics:
        sns.barplot(x="model", y=metric, data=df, label=metric)
    plt.title(f"{experiment_type} - Top-10 Metrics")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.result_dir, f"{experiment_type}_top10_metrics.png"))

    # 2. 超参实验热力图（仅超参实验）
    if experiment_type == "hyper_param":
        plt.figure(figsize=(10, 8))
        pivot_df = df.pivot(index="id_length", columns="lr", values="NDCG@10")
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu")
        plt.title("Hyper-Param Search - NDCG@10")
        plt.tight_layout()
        plt.savefig(os.path.join(config.result_dir, "hyper_param_heatmap.png"))
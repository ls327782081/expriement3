"""
AH-RQ消融实验可视化脚本

从消融实验结果生成图表：
1. 各层码本利用率对比
2. 重构损失对比
3. 按语义层次分组的利用率
4. 指标汇总表格

用法: python plot_ahrq_ablation.py
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List, Dict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def plot_codebook_usage_by_layer(results: List[dict], output_dir: str):
    """绘制各层码本利用率对比（分组柱状图）"""
    fig, ax = plt.subplots(figsize=(12, 5))

    experiments = [r['experiment_name'] for r in results]
    max_layers = max(len(r['metrics']['codebook_usage']['by_layer']) for r in results)

    x = np.arange(len(experiments))
    width = 0.8 / max_layers

    for layer_idx in range(max_layers):
        layer_rates = []
        for r in results:
            by_layer = r['metrics']['codebook_usage']['by_layer']
            layer_rates.append(by_layer[layer_idx] if layer_idx < len(by_layer) else 0)
        ax.bar(x + layer_idx * width, layer_rates, width, label=f'Layer {layer_idx+1}')

    ax.set_ylabel('Codebook Usage Rate')
    ax.set_xlabel('Experiment')
    ax.set_title('Codebook Usage by Layer')
    ax.set_xticks(x + width * (max_layers - 1) / 2)
    ax.set_xticklabels(experiments, rotation=15)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/codebook_usage_by_layer.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/codebook_usage_by_layer.png")


def plot_reconstruction_loss(results: List[dict], output_dir: str):
    """绘制重构损失对比"""
    fig, ax = plt.subplots(figsize=(8, 5))

    experiments = [r['experiment_name'] for r in results]
    recon_losses = [r['metrics']['val_recon_loss'] for r in results]

    bars = ax.bar(experiments, recon_losses, color=COLORS[:len(results)])
    ax.set_ylabel('Validation Reconstruction Loss')
    ax.set_title('Reconstruction Loss Comparison')

    for bar, loss in zip(bars, recon_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{loss:.5f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/reconstruction_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/reconstruction_loss.png")


def plot_semantic_hierarchy_usage(results: List[dict], output_dir: str):
    """绘制按语义层次分组的码本利用率"""
    fig, ax = plt.subplots(figsize=(10, 5))

    experiments = [r['experiment_name'] for r in results]

    # 收集所有语义层次
    all_groups = set()
    for r in results:
        by_group = r['metrics']['codebook_usage'].get('by_group', {})
        all_groups.update(by_group.keys())
    all_groups = sorted(all_groups)

    if not all_groups:
        print("No semantic hierarchy data found, skipping plot.")
        return

    x = np.arange(len(experiments))
    width = 0.8 / len(all_groups)

    for i, group in enumerate(all_groups):
        group_rates = []
        for r in results:
            by_group = r['metrics']['codebook_usage'].get('by_group', {})
            group_rates.append(by_group.get(group, {}).get('avg', 0))
        ax.bar(x + i * width, group_rates, width, label=group.capitalize())

    ax.set_ylabel('Codebook Usage Rate')
    ax.set_xlabel('Experiment')
    ax.set_title('Codebook Usage by Semantic Hierarchy')
    ax.set_xticks(x + width * (len(all_groups) - 1) / 2)
    ax.set_xticklabels(experiments, rotation=15)
    ax.legend(title='Semantic Level')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/semantic_hierarchy_usage.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/semantic_hierarchy_usage.png")


def plot_silhouette_by_layer(results: List[dict], output_dir: str):
    """绘制各层Silhouette系数对比"""
    fig, ax = plt.subplots(figsize=(12, 5))

    experiments = [r['experiment_name'] for r in results]

    # 收集所有层
    all_layers = set()
    for r in results:
        cluster_analysis = r['metrics'].get('cluster_analysis', {})
        all_layers.update(cluster_analysis.keys())
    all_layers = sorted(all_layers, key=lambda x: int(x[1:]))  # 按L0, L1, L2...排序

    if not all_layers:
        print("No silhouette data found, skipping plot.")
        return

    x = np.arange(len(experiments))
    width = 0.8 / len(all_layers)

    for i, layer in enumerate(all_layers):
        silhouette_scores = []
        for r in results:
            cluster_analysis = r['metrics'].get('cluster_analysis', {})
            layer_data = cluster_analysis.get(layer, {})
            silhouette_scores.append(layer_data.get('silhouette_score', 0))

        color = plt.cm.viridis(i / max(len(all_layers) - 1, 1))
        ax.bar(x + i * width, silhouette_scores, width, label=layer, color=color)

    ax.set_ylabel('Silhouette Score')
    ax.set_xlabel('Experiment')
    ax.set_title('Silhouette Score by Layer')
    ax.set_xticks(x + width * (len(all_layers) - 1) / 2)
    ax.set_xticklabels(experiments, rotation=15)
    ax.legend(title='Layer', loc='upper right', fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/silhouette_by_layer.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/silhouette_by_layer.png")


def plot_metrics_summary_table(results: List[dict], output_dir: str):
    """生成指标汇总表格"""
    import pandas as pd

    table_data = []
    for r in results:
        row = {
            "Experiment": r['experiment_name'],
            "EMA": "Yes" if r['config']['use_ema'] else "No",
            "HSCL": "Yes" if r['config']['use_hscl'] else "No",
            "Emotion": "Yes" if r['config']['use_emotion'] else "No",
            "Codebook Usage": f"{r['metrics']['codebook_usage']['overall']:.4f}",
            "Recon Loss": f"{r['metrics']['val_recon_loss']:.6f}",
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # 保存CSV
    csv_path = f"{output_dir}/metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # 保存为PNG表格
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(weight='bold')

    png_path = f"{output_dir}/metrics_table.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")


def plot_overall_quality_summary(results: List[dict], output_dir: str):
    """绘制综合质量总览图（3个子图）"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    experiments = [r['experiment_name'] for r in results]

    # (a) 码本利用率
    ax = axes[0]
    usage_rates = [r['metrics']['codebook_usage']['overall'] for r in results]
    bars = ax.bar(experiments, usage_rates, color=COLORS[:len(results)])
    ax.set_ylabel('Codebook Usage Rate')
    ax.set_title('(a) Overall Codebook Usage')
    ax.set_ylim(0, 1)
    for bar, rate in zip(bars, usage_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.3f}', ha='center', va='bottom', fontsize=9)

    # (b) 各组平均利用率
    ax = axes[1]
    group_rates = []
    for r in results:
        by_group = r['metrics']['codebook_usage'].get('by_group', {})
        if by_group:
            group_rates.append(np.mean([v.get('avg', 0) for v in by_group.values()]))
        else:
            group_rates.append(r['metrics']['codebook_usage']['overall'])
    bars = ax.bar(experiments, group_rates, color=COLORS[:len(results)])
    ax.set_ylabel('Avg Group Usage Rate')
    ax.set_title('(b) Semantic Group Usage')
    ax.set_ylim(0, 1)
    for bar, rate in zip(bars, group_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.3f}', ha='center', va='bottom', fontsize=9)

    # (c) 重构损失
    ax = axes[2]
    recon_losses = [r['metrics']['val_recon_loss'] for r in results]
    bars = ax.bar(experiments, recon_losses, color=COLORS[:len(results)])
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('(c) Reconstruction Loss')
    for bar, loss in zip(bars, recon_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{loss:.5f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_quality_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/overall_quality_summary.png")


def main():
    """主函数：加载结果并生成所有可视化"""
    results_dir = "./results/ahrq_ablation"
    os.makedirs(results_dir, exist_ok=True)

    # 加载实验结果
    results = []
    fname_map = {
        "baseline_rq_results.json": "Baseline-RQ",
        "ahrq_ema_results.json": "AHRQ-EMA",
        "ahrq_hscl_results.json": "AHRQ-HSCL",
        "ahrq_full_results.json": "AHRQ-Full"
    }

    for fname in fname_map.keys():
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
                print(f"Loaded: {fname}")

    if not results:
        print("No results found!")
        print(f"Please run train_ahrq_ablation.py first to generate results in {results_dir}")
        return

    print(f"\nLoaded {len(results)} experiment results")
    print("="*50)

    # 按实验名称排序（确保顺序一致）
    results.sort(key=lambda x: x['experiment_name'])

    # 生成所有可视化
    print("\nGenerating visualizations...")
    plot_codebook_usage_by_layer(results, results_dir)
    plot_reconstruction_loss(results, results_dir)
    plot_semantic_hierarchy_usage(results, results_dir)
    plot_silhouette_by_layer(results, results_dir)
    plot_metrics_summary_table(results, results_dir)
    plot_overall_quality_summary(results, results_dir)

    print("\n" + "="*50)
    print("All visualizations generated successfully!")
    print(f"Output directory: {results_dir}")
    print("="*50)
    print("\nGenerated files:")
    print("  - codebook_usage_by_layer.png")
    print("  - reconstruction_loss.png")
    print("  - semantic_hierarchy_usage.png")
    print("  - silhouette_by_layer.png")
    print("  - metrics_summary.csv")
    print("  - metrics_table.png")
    print("  - overall_quality_summary.png")


if __name__ == "__main__":
    main()
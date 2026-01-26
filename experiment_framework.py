"""
完整的实验框架
包括：多数据集验证、消融实验、效率分析、鲁棒性分析
"""

import os
import time
import json
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from config import config
from metrics import calculate_metrics


class ExperimentFramework:
    """实验框架类"""

    def __init__(self, results_dir="./results", logger=None):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        if logger is None:
            self.logger = logging.getLogger("ExperimentFramework")
        else:
            self.logger = logger

        # 实验结果存储
        self.results = defaultdict(dict)

    def run_multi_dataset_experiments(self, model_class, datasets, **kwargs):
        """多数据集验证实验

        Args:
            model_class: 模型类
            datasets: 数据集列表 [(name, train_loader, val_loader), ...]
            **kwargs: 模型初始化参数
        """
        self.logger.info("=" * 80)
        self.logger.info("开始多数据集验证实验")
        self.logger.info("=" * 80)

        results = {}

        for dataset_name, train_loader, val_loader in datasets:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"数据集: {dataset_name}")
            self.logger.info(f"{'='*60}")

            # 初始化模型
            model = model_class(**kwargs)

            # 训练和评估
            metrics = self._train_and_evaluate(
                model, train_loader, val_loader,
                experiment_name=f"{model_class.__name__}_{dataset_name}"
            )

            results[dataset_name] = metrics

        # 保存结果
        self._save_results(results, "multi_dataset_results.json")
        self._plot_multi_dataset_results(results)

        return results

    def run_ablation_experiments(self, model_class, train_loader, val_loader,
                                 ablation_configs, **kwargs):
        """消融实验

        Args:
            model_class: 模型类
            train_loader: 训练数据
            val_loader: 验证数据
            ablation_configs: 消融配置列表 [(name, config_dict), ...]
            **kwargs: 基础模型参数
        """
        self.logger.info("=" * 80)
        self.logger.info("开始消融实验")
        self.logger.info("=" * 80)

        results = {}

        for ablation_name, ablation_config in ablation_configs:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"消融配置: {ablation_name}")
            self.logger.info(f"{'='*60}")

            # 合并配置
            model_kwargs = {**kwargs, **ablation_config}

            # 初始化模型
            model = model_class(**model_kwargs)

            # 训练和评估
            metrics = self._train_and_evaluate(
                model, train_loader, val_loader,
                experiment_name=f"{model_class.__name__}_ablation_{ablation_name}"
            )

            results[ablation_name] = metrics

        # 保存结果
        self._save_results(results, f"{model_class.__name__}_ablation_results.json")
        self._plot_ablation_results(results, model_class.__name__)

        return results

    def run_efficiency_analysis(self, models, train_loader, val_loader):
        """效率分析实验

        Args:
            models: 模型列表 [(name, model_instance), ...]
            train_loader: 训练数据
            val_loader: 验证数据
        """
        self.logger.info("=" * 80)
        self.logger.info("开始效率分析实验")
        self.logger.info("=" * 80)

        results = {}

        for model_name, model in models:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"模型: {model_name}")
            self.logger.info(f"{'='*60}")

            # 测量训练时间
            train_time = self._measure_training_time(model, train_loader)

            # 测量推理时间
            inference_time = self._measure_inference_time(model, val_loader)

            # 测量内存占用
            memory_usage = self._measure_memory_usage(model)

            # 统计参数量
            num_params = sum(p.numel() for p in model.parameters())

            results[model_name] = {
                "train_time_per_epoch": train_time,
                "inference_time_per_sample": inference_time,
                "memory_usage_mb": memory_usage,
                "num_parameters": num_params
            }

            self.logger.info(f"训练时间: {train_time:.2f}s/epoch")
            self.logger.info(f"推理时间: {inference_time*1000:.2f}ms/sample")
            self.logger.info(f"内存占用: {memory_usage:.2f}MB")
            self.logger.info(f"参数量: {num_params:,}")

        # 保存结果
        self._save_results(results, "efficiency_analysis_results.json")
        self._plot_efficiency_results(results)

        return results

    def run_robustness_analysis(self, model_class, datasets_by_sparsity, **kwargs):
        """鲁棒性分析"""
        self.logger.info("开始鲁棒性分析实验")
        results = {}

        for sparsity_level, train_loader, val_loader in datasets_by_sparsity:
            model = model_class(**kwargs)
            metrics = self._train_and_evaluate(
                model, train_loader, val_loader,
                experiment_name=f"{model_class.__name__}_sparsity_{sparsity_level}"
            )
            results[sparsity_level] = metrics

        self._save_results(results, f"{model_class.__name__}_robustness_results.json")
        return results

    def _train_and_evaluate(self, model, train_loader, val_loader, experiment_name):
        """训练并评估"""
        from main import train_model
        trained_model = train_model(model, train_loader, val_loader, experiment_name, logger=self.logger)
        metrics = self._evaluate_model(trained_model, val_loader)
        return metrics

    def _evaluate_model(self, model, val_loader):
        """评估模型"""
        model.eval()
        all_predictions, all_ground_truth, all_user_ids = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch)
                predictions = outputs.get('predictions', outputs.get('logits')) if isinstance(outputs, dict) else outputs
                all_predictions.extend(predictions.cpu().numpy())
                all_ground_truth.extend(batch['target_item_id'].cpu().numpy())
                all_user_ids.extend(batch['user_id'].cpu().numpy())

        return calculate_metrics(all_user_ids, all_predictions, all_ground_truth, k_list=[5, 10, 20])

    def _measure_training_time(self, model, train_loader, num_epochs=1):
        """测量训练时间"""
        model.train()
        start_time = time.time()
        for epoch in range(num_epochs):
            for batch in train_loader:
                outputs = model(batch)
                if isinstance(outputs, dict) and 'loss' in outputs:
                    outputs['loss'].backward()
        return (time.time() - start_time) / num_epochs

    def _measure_inference_time(self, model, val_loader, num_samples=100):
        """测量推理时间"""
        model.eval()
        times, count = [], 0
        with torch.no_grad():
            for batch in val_loader:
                if count >= num_samples:
                    break
                start = time.time()
                _ = model(batch)
                times.append((time.time() - start) / batch['user_id'].size(0))
                count += batch['user_id'].size(0)
        return np.mean(times)

    def _measure_memory_usage(self, model):
        """测量内存"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            dummy = {'user_id': torch.randint(0, 1000, (32,)).to(config.device),
                    'item_id': torch.randint(0, 1000, (32,)).to(config.device)}
            _ = model(dummy)
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

    def _save_results(self, results, filename):
        """保存结果"""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"结果已保存: {filepath}")

    def _plot_multi_dataset_results(self, results):
        """绘制多数据集结果"""
        metrics = ['Recall@10', 'NDCG@10', 'MRR@10']
        datasets = list(results.keys())

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, metric in enumerate(metrics):
            values = [results[ds].get(metric, 0) for ds in datasets]
            axes[i].bar(datasets, values)
            axes[i].set_title(metric)
            axes[i].set_ylabel('Score')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'multi_dataset_comparison.png'))
        plt.close()

    def _plot_ablation_results(self, results, model_name):
        """绘制消融实验结果"""
        configs = list(results.keys())
        metrics = ['Recall@10', 'NDCG@10']

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, metric in enumerate(metrics):
            values = [results[cfg].get(metric, 0) for cfg in configs]
            axes[i].bar(configs, values)
            axes[i].set_title(f'{model_name} - {metric}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_ablation.png'))
        plt.close()

    def _plot_efficiency_results(self, results):
        """绘制效率分析结果"""
        models = list(results.keys())

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 训练时间
        train_times = [results[m]['train_time_per_epoch'] for m in models]
        axes[0, 0].bar(models, train_times)
        axes[0, 0].set_title('Training Time per Epoch')
        axes[0, 0].set_ylabel('Time (s)')

        # 推理时间
        infer_times = [results[m]['inference_time_per_sample'] * 1000 for m in models]
        axes[0, 1].bar(models, infer_times)
        axes[0, 1].set_title('Inference Time per Sample')
        axes[0, 1].set_ylabel('Time (ms)')

        # 内存占用
        memory = [results[m]['memory_usage_mb'] for m in models]
        axes[1, 0].bar(models, memory)
        axes[1, 0].set_title('Memory Usage')
        axes[1, 0].set_ylabel('Memory (MB)')

        # 参数量
        params = [results[m]['num_parameters'] / 1e6 for m in models]
        axes[1, 1].bar(models, params)
        axes[1, 1].set_title('Number of Parameters')
        axes[1, 1].set_ylabel('Parameters (M)')

        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'efficiency_analysis.png'))
        plt.close()

    def _plot_robustness_results(self, results, model_name):
        """绘制鲁棒性分析结果"""
        sparsity_levels = list(results.keys())
        metrics = ['Recall@10', 'NDCG@10']

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, metric in enumerate(metrics):
            values = [results[sp].get(metric, 0) for sp in sparsity_levels]
            axes[i].plot(sparsity_levels, values, marker='o')
            axes[i].set_title(f'{model_name} - {metric} vs Sparsity')
            axes[i].set_xlabel('Sparsity Level')
            axes[i].set_ylabel('Score')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_robustness.png'))
        plt.close()


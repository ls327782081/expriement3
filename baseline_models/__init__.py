"""
Baseline Models Package
=======================

This package contains implementations of baseline models for comparison.

当前框架基线:
-------------
- DGMRec: Disentangling and Generating Modalities (SIGIR 2025) - 已验证正确

已移除的基线:
-------------
- PRISM: 实现存在错误（使用随机目标训练），已移除
- PctxAligned: 需要特殊tokenizer，与实验设置不兼容，已移除

RecBole基线 (需单独运行):
------------------------
- SASRec: Self-Attentive Sequential Recommendation
- BERT4Rec: Bidirectional Encoder Representations from Transformer
- GRU4Rec: Session-based Recommendations with RNN

使用 recbole_baselines/ 目录下的脚本运行RecBole基线

Usage:
------
    from baseline_models import DGMRec

    # Initialize model
    dgmrec = DGMRec(config, dataset=dataset_adapter)

Author: Graduate Student
Date: 2026-02-04
"""

from .dgmrec import DGMRec

__all__ = [
    'DGMRec',
]

__version__ = '3.0.0'
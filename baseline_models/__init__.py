"""
Baseline Models Package
=======================

This package contains implementations of baseline models for comparison.

Implemented Baselines:
---------------------
- PctxAligned: Context-aware recommendation (SIGIR 2023) - Official alignment
- PRISM: Personalized multimodal fusion (WWW 2026) - Official alignment
- DGMRec: Disentangling and Generating Modalities (SIGIR 2025) - Official alignment

Usage:
------
    from baseline_models import PctxAligned, PRISM, DGMRec

    # Initialize models
    pctx = PctxAligned(config)
    prism = PRISM(config)
    dgmrec = DGMRec(config)

Author: Graduate Student
Date: 2026-01-26
"""

from .pctx_aligned import PctxAligned
from .prism import PRISM
from .dgmrec import DGMRec

__all__ = [
    'PctxAligned',
    'PRISM',
    'DGMRec',
]

__version__ = '2.0.0'
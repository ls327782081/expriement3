"""
Baseline Models Package
=======================

This package contains implementations of baseline models for comparison.

Classic Baselines:
------------------
- Pctx: Context-aware recommendation
- MMQ: Multimodal Quantization
- FusID: Fusion-based Semantic ID
- PRISM: Personalized multimodal fusion (WWW 2026)

2025 Latest Baselines (GitHub-sourced):
---------------------------------------
- DGMRec: Disentangling and Generating Modalities (SIGIR 2025)

To be implemented:
------------------
- AMMRM: Adaptive Multimodal Recommendation
- CoFiRec: Coarse-to-Fine Tokenization
- LETTER: Learnable Item Tokenization

Usage:
------
    from baseline_models.pctx import Pctx
    from baseline_models.mmq import MMQ
    from baseline_models.fusid import FusID
    from baseline_models.prism import PRISM
    from baseline_models.dgmrec import DGMRec

Author: Graduate Student
Date: 2026-01-21
"""

from .pctx import Pctx
from .duorec import DuoRec
from .mmq import MMQ
from .fusid import FusID
from .prism import PRISM
from .dgmrec import DGMRec

__all__ = [
    'Pctx',
    'DuoRec',
    'MMQ',
    'FusID',
    'PRISM',
    'DGMRec',
]

__version__ = '1.0.0'
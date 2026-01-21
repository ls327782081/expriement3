"""
Baseline Models Package
=======================

This package contains implementations of baseline models for comparison.

Classic Baselines:
------------------
- Pctx: Context-aware recommendation
- MMQ: Multimodal Quantization
- FusID: Fusion-based Semantic ID
- RPG: Recurrent Personalized Generation

2025 Latest Baselines (GitHub-sourced):
---------------------------------------
- PRISM: Personalized multimodal fusion (WWW 2026)
- DGMRec: Disentangling and Generating Modalities (SIGIR 2025)
- REARM: Relation-Enhanced Adaptive Representation (MM 2025)

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
    from baseline_models.rpg import RPG
    from baseline_models.prism import PRISM
    from baseline_models.dgmrec import DGMRec
    from baseline_models.rearm import REARM

Author: Graduate Student
Date: 2026-01-21
"""

from .pctx import Pctx
from .mmq import MMQ
from .fusid import FusID
from .rpg import RPG
from .prism import PRISM
from .dgmrec import DGMRec
from .rearm import REARM

__all__ = [
    'Pctx',
    'MMQ',
    'FusID',
    'RPG',
    'PRISM',
    'DGMRec',
    'REARM',
]

__version__ = '1.0.0'


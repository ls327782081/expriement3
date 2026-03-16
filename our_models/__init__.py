"""
Our Models Package
==================

This package contains our innovative models for personalized multimodal semantic ID generation.

Models:
-------
- PMAT: Personalized Multimodal Adaptive Tokenizer (Innovation Point 1)
- MCRL: Multi-task Contrastive Representation Learning (Innovation Point 2)
- PMAT_SASRec: PMAT语义增强嵌入 + SASRec强排序骨架 (Innovation Point 3)

Usage:
------
    from our_models.pmat import PMAT
    from our_models.mcrl import MCRL
    from our_models.pmat_sasrec import PMAT_SASRec

    # Initialize PMAT
    pmat = PMAT(config)

    # Initialize MCRL
    mcrl = MCRL(config)

    # Initialize PMAT_SASRec
    pmat_sasrec = PMAT_SASRec(config)

Author: Graduate Student
Date: 2026-01-21
"""

from .mcrl import (
    MCRL,
    UserPreferenceContrastive,
    IntraModalContrastive,
    InterModalContrastive
)


__all__ = [

    # MCRL components
    'MCRL',
    'UserPreferenceContrastive',
    'IntraModalContrastive',
    'InterModalContrastive',

]

__version__ = '1.0.0'
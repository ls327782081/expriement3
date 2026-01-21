"""
Our Models Package
==================

This package contains our innovative models for personalized multimodal semantic ID generation.

Models:
-------
- PMAT: Personalized Multimodal Adaptive Tokenizer (Innovation Point 1)
- MCRL: Multi-task Contrastive Representation Learning (Innovation Point 2)
- PMATWithMCRL: Joint model combining PMAT and MCRL

Usage:
------
    from our_models.pmat import PMAT
    from our_models.mcrl import MCRL, PMATWithMCRL
    
    # Initialize PMAT
    pmat = PMAT(config)
    
    # Initialize MCRL
    mcrl = MCRL(config)
    
    # Initialize joint model
    joint_model = PMATWithMCRL(config)

Author: Graduate Student
Date: 2026-01-21
"""

from .pmat import (
    PMAT,
    UserModalAttention,
    MultiModalEncoder,
    PersonalizedFusion,
    DynamicIDUpdater,
    SemanticIDQuantizer,
    get_pmat_ablation_model
)

from .mcrl import (
    MCRL,
    PMATWithMCRL,
    UserPreferenceContrastive,
    IntraModalContrastive,
    InterModalContrastive
)

__all__ = [
    # PMAT components
    'PMAT',
    'UserModalAttention',
    'MultiModalEncoder',
    'PersonalizedFusion',
    'DynamicIDUpdater',
    'SemanticIDQuantizer',
    'get_pmat_ablation_model',

    # MCRL components
    'MCRL',
    'PMATWithMCRL',
    'UserPreferenceContrastive',
    'IntraModalContrastive',
    'InterModalContrastive',
]

__version__ = '1.0.0'


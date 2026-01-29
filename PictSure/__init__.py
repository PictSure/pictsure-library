"""
PictSure - Few-shot image classification library.

This module provides the main interface for few-shot image classification
using various vision encoders (ResNet, ViT, DINOv2, CLIP).
"""

from .model_PictSure import PictSure
from .model_embeddings import (
    ResNetWrapper,
    VitNetWrapper,
    DINOV2Wrapper,
    CLIPWrapper,
    get_encoder,
    load_encoder,
)

__all__ = [
    'PictSure',
    'ResNetWrapper',
    'VitNetWrapper',
    'DINOV2Wrapper',
    'CLIPWrapper',
    'get_encoder',
    'load_encoder',
]

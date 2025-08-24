"""
Settings package for ML Manager.

This package contains all configuration classes for the ML Manager module.
"""

from .weights_config import ModelWeightsConfig
from .yolo_config import YOLOTrainingConfig
from .videomae_config import VideoMAETrainingConfig

__all__ = [
    "ModelWeightsConfig",
    "YOLOTrainingConfig", 
    "VideoMAETrainingConfig"
]

"""
Models package for ML Manager.

This package contains specialized model classes for different ML tasks:
- YOLOModule: Unified YOLO model wrapper for detection and segmentation
- ActionDetector: Specialized action detection model
- BallDetector: Specialized ball detection/segmentation model
- CourtSegmentation: Specialized court segmentation model
- PlayerModule: Unified player detection/segmentation/pose estimation
- GameStatusClassifier: VideoMAE-based game state classification
"""

from .YoloModule import YOLOModule
from .ActionDetectorModule import ActionDetectorModule
from .BallDetectorModule import BallDetectorModule
from .CourtSegmentationModule import CourtSegmentationModule
from .PlayerDetectorModule import PlayerDetectorModule
from .GameStatusClassifierModule import GameStatusClassifierModule

__all__ = [
    "YOLOModule",
    "ActionDetectorModule",
    "BallDetectorModule",
    "CourtSegmentationModule",
    "PlayerDetectorModule",
    "GameStatusClassifierModule"
]

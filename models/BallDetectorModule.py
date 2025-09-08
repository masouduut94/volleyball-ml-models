"""
Ball detection and segmentation model for volleyball.

This module provides specialized ball detection functionality using YOLO models
trained for volleyball ball recognition and segmentation.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
from .YoloModule import YOLOModule
from ..core.data_structures import Detection
from ..enums import DetectorModel
from ..utils.logger import logger


class BallDetectorModule:
    """
    Specialized ball detection model for volleyball.
    
    This class wraps the YOLOModule specifically for ball detection tasks,
    providing volleyball-specific utilities and filtering.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize ball detector.
        
        Args:
            model_path: Path to ball detection model weights
            device: Device to run inference on
        """
        logger.info(f"Initializing BallDetector with model: {model_path}")
        # Note: YOLOModule will automatically detect the model type
        self.yolo_module = YOLOModule(
            model_path=model_path,
            device=device
        )
        self.ball_class_names = ("ball",)
    
    def detect_ball(self, 
                   image: Union[str, np.ndarray],
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45,
                   **kwargs) -> Detection:
        """
        Detect volleyball ball in a single frame.
        
        Args:
            image: Input image (single frame)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional arguments for detection
            
        Returns:
            List of Detection objects with ball detection results
        """
        detections = self.yolo_module.detect(
            image,
            conf_threshold,
            iou_threshold,
            detector_model=DetectorModel.BALL_DETECTOR.value,
            **kwargs
        )

        # Filter to only ball detections
        ball = max(detections, key=lambda x: x.confidence) if detections else []

        return ball
    


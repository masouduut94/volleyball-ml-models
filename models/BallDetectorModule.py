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
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 use_segmentation: bool = False):
        """
        Initialize ball detector.
        
        Args:
            model_path: Path to ball detection model weights
            device: Device to run inference on
            use_segmentation: Whether to use segmentation model
        """
        logger.info(f"Initializing BallDetector with model: {model_path}")
        # Note: YOLOModule will automatically detect the model type
        self.yolo_module = YOLOModule(
            model_path=model_path,
            device=device
        )
        
        self.use_segmentation = use_segmentation
        self.ball_class_names = ["ball", "volleyball"]
    
    def detect_ball(self, 
                   image: Union[str, np.ndarray],
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45,
                   **kwargs) -> List[Detection]:
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
        ball_detections = []
        for det in detections:
            if det.class_name.lower() in self.ball_class_names:
                ball_detections.append(det)
        
        return ball_detections
    
    def get_ball_trajectory(self, 
                           detections_list: List[List[Detection]]) -> List[Tuple[float, float]]:
        """
        Extract ball trajectory from a sequence of detections.
        
        Args:
            detections_list: List of detection lists from consecutive frames
            
        Returns:
            List of (x, y) coordinates representing ball trajectory
        """
        trajectory = []
        
        for detections in detections_list:
            for det in detections:
                if det.class_name.lower() in self.ball_class_names:
                    center = det.bbox.center
                    trajectory.append(center)
                    break  # Only take first ball detection per frame
        
        return trajectory
    
    def plot_ball_trajectory(self, 
                           image: np.ndarray,
                           detections: List[Detection],
                           show_labels: bool = True,
                           show_conf: bool = True,
                           line_thickness: int = 2) -> np.ndarray:
        """
        Plot ball detection results on image.
        
        Args:
            image: Input image
            detections: Ball detection results
            show_labels: Whether to show labels
            show_conf: Whether to show confidence scores
            line_thickness: Line thickness for annotations
            
        Returns:
            Annotated image
        """
        # For now, return the original image
        # TODO: Implement proper plotting without DetectionBatch
        return image

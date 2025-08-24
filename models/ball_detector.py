"""
Ball detection and segmentation model for volleyball.

This module provides specialized ball detection functionality using YOLO models
trained for volleyball ball recognition and segmentation.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
from .yolo_module import YOLOModule
from .data_structures import DetectionBatch


class BallDetector:
    """
    Specialized ball detection model for volleyball.
    
    This class wraps the YOLOModule specifically for ball detection tasks,
    providing volleyball-specific utilities and filtering.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 verbose: bool = False,
                 use_segmentation: bool = False):
        """
        Initialize ball detector.
        
        Args:
            model_path: Path to ball detection model weights
            device: Device to run inference on
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            verbose: Whether to print verbose output
            use_segmentation: Whether to use segmentation model
        """
        # Note: YOLOModule will automatically detect the model type
        self.yolo_module = YOLOModule(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            verbose=verbose
        )
        
        self.use_segmentation = use_segmentation
        self.ball_class_names = ["ball", "volleyball"]
    
    def detect_ball(self, 
                   image: Union[str, np.ndarray, List[str], List[np.ndarray]],
                   **kwargs) -> DetectionBatch:
        """
        Detect volleyball ball in image(s).
        
        Args:
            image: Input image(s)
            **kwargs: Additional arguments for detection
            
        Returns:
            DetectionBatch with ball detection results
        """
        detections = self.yolo_module.detect(image, **kwargs)
        
        # Filter to only ball detections
        ball_detections = []
        for det in detections.detections:
            if det.class_name.lower() in self.ball_class_names:
                ball_detections.append(det)
        
        return DetectionBatch(
            detections=ball_detections,
            image_shape=detections.image_shape,
            processing_time=detections.processing_time,
            metadata=detections.metadata
        )
    
    def get_ball_trajectory(self, 
                           detections_list: List[DetectionBatch]) -> List[Tuple[float, float]]:
        """
        Extract ball trajectory from a sequence of detections.
        
        Args:
            detections_list: List of DetectionBatch from consecutive frames
            
        Returns:
            List of (x, y) coordinates representing ball trajectory
        """
        trajectory = []
        
        for detections in detections_list:
            for det in detections.detections:
                if det.class_name.lower() in self.ball_class_names:
                    center = det.bbox.center
                    trajectory.append(center)
                    break  # Only take first ball detection per frame
        
        return trajectory
    
    def plot_ball_trajectory(self, 
                           image: np.ndarray,
                           detections: DetectionBatch,
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
        return self.yolo_module.plot_results(
            image, detections, show_labels, show_conf, line_thickness
        )

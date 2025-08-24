"""
Action detection model for volleyball actions.

This module provides specialized action detection functionality using YOLO models
trained for volleyball action recognition.
"""

from typing import List, Optional, Union, Dict
import numpy as np
from .yolo_module import YOLOModule
from ..core.data_structures import DetectionBatch


class ActionDetector:
    """
    Specialized action detection model for volleyball actions.
    
    This class wraps the YOLOModule specifically for action detection tasks,
    providing volleyball-specific utilities and filtering.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize action detector.
        
        Args:
            model_path: Path to action detection model weights
            device: Device to run inference on
            verbose: Whether to print verbose output
        """
        self.yolo_module = YOLOModule(
            model_path=model_path,
            device=device,
            verbose=verbose
        )
        
        # Common volleyball actions
        self.volleyball_actions = [
            "serve", "receive", "set", "spike", "block", "dig"
        ]
    
    def detect_actions(self, 
                      image: Union[str, np.ndarray, List[str], List[np.ndarray]],
                      conf_threshold: float = 0.25,
                      iou_threshold: float = 0.45,
                      **kwargs) -> DetectionBatch:
        """
        Detect volleyball actions in image(s).
        
        Args:
            image: Input image(s)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional arguments for detection
            
        Returns:
            DetectionBatch with action detection results
        """
        return self.yolo_module.detect(image, conf_threshold, iou_threshold, **kwargs)
    
    def filter_by_action_type(self, 
                             detections: DetectionBatch,
                             action_types: List[str]) -> DetectionBatch:
        """
        Filter detections by specific action types.
        
        Args:
            detections: Input detection batch
            action_types: List of action types to keep
            
        Returns:
            Filtered DetectionBatch
        """
        return self.yolo_module.filter_results(
            detections, 
            class_names=action_types
        )
    
    def get_action_counts(self, detections: DetectionBatch) -> Dict[str, int]:
        """
        Get count of each action type detected.
        
        Args:
            detections: Detection results
            
        Returns:
            Dictionary mapping action types to counts
        """
        return self.yolo_module.get_class_counts(detections)
    
    def plot_actions(self, 
                    image: np.ndarray,
                    detections: DetectionBatch,
                    show_labels: bool = True,
                    show_conf: bool = True,
                    line_thickness: int = 2) -> np.ndarray:
        """
        Plot action detection results on image.
        
        Args:
            image: Input image
            detections: Action detection results
            show_labels: Whether to show action labels
            show_conf: Whether to show confidence scores
            line_thickness: Line thickness for annotations
            
        Returns:
            Annotated image
        """
        return self.yolo_module.plot_results(
            image, detections, show_labels, show_conf, line_thickness
        )
    
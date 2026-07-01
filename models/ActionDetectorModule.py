"""
Action detection model for volleyball actions.

This module provides specialized action detection functionality using YOLO models
trained for volleyball action recognition.
"""

from typing import List, Optional, Union, Dict
import numpy as np
from .YoloModule import YOLOModule
from ..core.data_structures import Detection
from ..enums import DetectorModel
from ..utils.logger import logger


class ActionDetectorModule:
    """
    Specialized action detection model for volleyball actions.
    
    This class wraps the YOLOModule specifically for action detection tasks,
    providing volleyball-specific utilities and filtering.
    """

    def __init__(self,
                 model_path: str,
                 device: Optional[str] = None):
        """
        Initialize action detector.
        
        Args:
            model_path: Path to action detection model weights
            device: Device to run inference on
        """
        logger.info(f"Initializing ActionDetector with model: {model_path}")
        self.yolo_module = YOLOModule(
            model_path=model_path,
            device=device
        )
        self.labels = self.yolo_module.labels

    def action_id2name(self, class_id: int):
        return self.yolo_module.id2class(class_id)

    def detect_actions(self,
                       image: Union[str, np.ndarray],
                       conf_threshold: float = 0.25,
                       iou_threshold: float = 0.45,
                       **kwargs) -> List[Detection]:
        """
        Detect volleyball actions in a single frame.
        
        Args:
            image: Input image (single frame)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional arguments for detection
            
        Returns:
            List of Detection objects with action detection results
        """

        detections = self.yolo_module.detect(
            image,
            conf_threshold,
            iou_threshold,
            detector_model=DetectorModel.ACTION_DETECTOR.value,
            **kwargs
        )
        # The model trained for actions have two extra objects (ball, serve) which we should exclude.
        detections = [det for det in detections if det.class_name not in ['ball', 'serve']]

        return detections

    @staticmethod
    def keep_actions(detections: List[Detection], action_types: List[str]) -> List[Detection]:
        """
         Keep the actions given as input.
        
        Args:
            detections: Input detection list
            action_types: List of action types to keep
            
        Returns:
            Filtered list of detections
        """
        return [det for det in detections if det.class_name in action_types]

    @staticmethod
    def get_action_counts(detections: List[Detection]) -> Dict[str, int]:
        """
        Get count of each action type detected.
        
        Args:
            detections: Detection results
            
        Returns:
            Dictionary mapping action types to counts
        """
        counts = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    @staticmethod
    def plot_actions(image: np.ndarray,
                     detections: List[Detection],
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
        # For now, return the original image
        # TODO: Implement proper plotting without DetectionBatch
        return image

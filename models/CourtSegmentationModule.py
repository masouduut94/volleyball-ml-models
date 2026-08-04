"""
Court segmentation model for volleyball.

This module provides specialized court segmentation functionality using YOLO models
trained for volleyball court recognition and segmentation.
"""

import numpy as np
from typing import List, Optional, Union

from .YoloModule import YOLOModule
from ..utils.logger import logger
from ..core.data_structures import Detection, SegmentationDetection


class CourtSegmentationModule:
    """
    Specialized court segmentation model for volleyball.
    
    This class wraps the YOLOModule specifically for court segmentation tasks,
    providing volleyball-specific utilities and filtering.
    """

    def __init__(self,
                 model_path: str,
                 device: Optional[str] = None):
        """
        Initialize court segmentation model.
        
        Args:
            model_path: Path to court segmentation model weights
            device: Device to run inference on
        """
        logger.info(f"Initializing CourtSegmentation with model: {model_path}")
        # Note: YOLOModule will automatically detect the model type
        self.yolo_module = YOLOModule(
            model_path=model_path,
            device=device
        )

    def segment_court(self,
                      image: Union[str, np.ndarray],
                      conf_threshold: float = 0.25,
                      iou_threshold: float = 0.45,
                      **kwargs) -> Optional[SegmentationDetection]:
        """
        Segment volleyball court in a single frame.
        
        Args:
            image: Input image (single frame)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional arguments for segmentation
            
        Returns:
            List of Detection objects with court segmentation results
        """
        detections = self.yolo_module.detect(image, conf_threshold, iou_threshold, **kwargs)
        h, w, _ = image.shape
        # Keep the biggest one which is presumably the court
        court: SegmentationDetection = max(
            detections,
            key=lambda x: x.confidence
        ) if detections else None
        if court:
            court.get_court_corners_from_mask(original_width=w, original_height=h)
            return court

        return None

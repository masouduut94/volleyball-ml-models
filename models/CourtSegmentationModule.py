"""
Court segmentation model for volleyball.

This module provides specialized court segmentation functionality using YOLO models
trained for volleyball court recognition and segmentation.
"""

import numpy as np
from typing import List, Optional, Union

from .YoloModule import YOLOModule
from ..enums import DetectorModel
from ..utils.logger import logger
from ..core.data_structures import Detection


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
        
        self.court_class_names = ["court", "volleyball_court", "field"]
    
    def segment_court(self, 
                     image: Union[str, np.ndarray],
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45,
                     **kwargs) -> List[Detection]:
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
        detections = self.yolo_module.detect(
            image, 
            conf_threshold, 
            iou_threshold, 
            detector_model=DetectorModel.COURT_DETECTOR.value,
            **kwargs
        )
        
        # Filter to only court detections
        court_detections = []
        for det in detections:
            if det.class_name.lower() in self.court_class_names:
                court_detections.append(det)
        
        return court_detections
    
    def get_court_mask(self, detections: List[Detection]) -> Optional[np.ndarray]:
        """
        Get the court segmentation mask from detections.
        
        Args:
            detections: Court segmentation results
            
        Returns:
            Court segmentation mask or None if no detections
        """
        if not detections:
            return None
        
        # Get the first court detection with a mask
        for det in detections:
            if hasattr(det, 'mask') and det.mask is not None:
                return np.array(det.mask)
        
        return None


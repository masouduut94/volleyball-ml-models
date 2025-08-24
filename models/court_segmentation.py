"""
Court segmentation model for volleyball.

This module provides specialized court segmentation functionality using YOLO models
trained for volleyball court recognition and segmentation.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
import cv2
from .yolo_module import YOLOModule
from .data_structures import DetectionBatch
from ..enums import YOLOModelType


class CourtSegmentation:
    """
    Specialized court segmentation model for volleyball.
    
    This class wraps the YOLOModule specifically for court segmentation tasks,
    providing volleyball-specific utilities and filtering.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 verbose: bool = False):
        """
        Initialize court segmentation model.
        
        Args:
            model_path: Path to court segmentation model weights
            device: Device to run inference on
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            verbose: Whether to print verbose output
        """
        # Note: YOLOModule will automatically detect the model type
        self.yolo_module = YOLOModule(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            verbose=verbose
        )
        
        self.court_class_names = ["court", "volleyball_court", "field"]
    
    def segment_court(self, 
                     image: Union[str, np.ndarray, List[str], List[np.ndarray]],
                     **kwargs) -> DetectionBatch:
        """
        Segment volleyball court in image(s).
        
        Args:
            image: Input image(s)
            **kwargs: Additional arguments for segmentation
            
        Returns:
            DetectionBatch with court segmentation results
        """
        detections = self.yolo_module.detect(image, **kwargs)
        
        # Filter to only court detections
        court_detections = []
        for det in detections.detections:
            if det.class_name.lower() in self.court_class_names:
                court_detections.append(det)
        
        return DetectionBatch(
            detections=court_detections,
            image_shape=detections.image_shape,
            processing_time=detections.processing_time,
            metadata=detections.metadata
        )
    
    def get_court_mask(self, detections: DetectionBatch) -> Optional[np.ndarray]:
        """
        Get the court segmentation mask from detections.
        
        Args:
            detections: Court segmentation results
            
        Returns:
            Court segmentation mask or None if no detections
        """
        if not detections.detections:
            return None
        
        # Get the first court detection with a mask
        for det in detections.detections:
            if hasattr(det, 'mask') and det.mask is not None:
                return np.array(det.mask)
        
        return None
    
    def plot_court_segmentation(self, 
                              image: np.ndarray,
                              detections: DetectionBatch,
                              show_labels: bool = True,
                              show_conf: bool = True,
                              line_thickness: int = 2) -> np.ndarray:
        """
        Plot court segmentation results on image.
        
        Args:
            image: Input image
            detections: Court segmentation results
            show_labels: Whether to show labels
            show_conf: Whether to show confidence scores
            line_thickness: Line thickness for annotations
            
        Returns:
            Annotated image
        """
        return self.yolo_module.plot_results(
            image, detections, show_labels, show_conf, line_thickness
        )

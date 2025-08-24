"""
Unified YOLO module for detection, segmentation, and pose estimation.

This module provides a unified interface for different YOLO models while maintaining
the same API for detection, plotting, and result filtering.
"""

import time
from typing import List, Optional, Union, Tuple, Dict, Any
import numpy as np
import cv2
from ultralytics import YOLO
from supervision import Detections, BoxAnnotator, MaskAnnotator, KeypointAnnotator
from supervision.detection.core import Detections as SupervisionDetections

from .data_structures import (
    Detection, SegmentationDetection, PoseDetection, DetectionBatch,
    BoundingBox, KeyPoint
)
from ..enums import YOLOModelType


class YOLOModule:
    """
    Unified YOLO module for different model types.
    
    This class can be initialized with different YOLO models (detection, segmentation, pose)
    and provides consistent interfaces for inference, plotting, and result processing.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 verbose: bool = False):
        """
        Initialize YOLO module.
        
        Args:
            model_path: Path to YOLO model weights
            model_type: Type of model (optional, will be auto-detected if None)
            device: Device to run inference on ("cpu", "cuda", etc.)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            verbose: Whether to print verbose output
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Automatically determine model_type from YOLO model attributes
        yolo_task = getattr(self.model, 'task', None)
        
        if yolo_task == 'detect':
            self.model_type = YOLOModelType.DETECTION
        elif yolo_task == 'segment':
            self.model_type = YOLOModelType.SEGMENTATION
        elif yolo_task == 'pose':
            self.model_type = YOLOModelType.POSE
        elif yolo_task == 'classify':
            self.model_type = YOLOModelType.CLASSIFICATION
        elif yolo_task == 'obb':
            self.model_type = YOLOModelType.OBB
        else:
            self.model_type = YOLOModelType.DETECTION  # default fallback
        
        if device:
            self.model.to(device)
        
        # Initialize annotators based on model type
        self._init_annotators()
        
        # Class names from model
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
    
    def _init_annotators(self):
        """Initialize appropriate annotators based on model type."""
        if self.model_type == YOLOModelType.DETECTION:
            self.annotator = BoxAnnotator()
        elif self.model_type == YOLOModelType.SEGMENTATION:
            self.annotator = MaskAnnotator()
        elif self.model_type == YOLOModelType.POSE:
            self.annotator = KeypointAnnotator()
        else:
            self.annotator = BoxAnnotator()
    
    def detect(self, 
               image: Union[str, np.ndarray, List[str], List[np.ndarray]],
               **kwargs) -> DetectionBatch:
        """
        Perform detection on input image(s).
        
        Args:
            image: Input image(s) - can be path, numpy array, or list of either
            **kwargs: Additional arguments for YOLO inference
            
        Returns:
            DetectionBatch containing detection results
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=self.verbose,
            **kwargs
        )
        
        processing_time = time.time() - start_time
        
        # Process results
        if isinstance(image, (str, np.ndarray)):
            # Single image
            detections = self._process_single_result(results, image)
            image_shape = self._get_image_shape(image)
        else:
            # Multiple images
            detections = []
            image_shape = (0, 0)
            for i, result in enumerate(results):
                img = image[i] if isinstance(image, list) else image
                dets = self._process_single_result(result, img)
                detections.extend(dets)
                if i == 0:  # Use first image shape
                    image_shape = self._get_image_shape(img)
        
        return DetectionBatch(
            detections=detections,
            image_shape=image_shape,
            processing_time=processing_time
        )
    
    def _process_single_result(self, result, image) -> List[Detection]:
        """Process single YOLO result into Detection objects."""
        detections = []
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes
        for i in range(len(boxes)):
            # Get box coordinates
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box
            
            # Get confidence and class
            conf = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            
            if self.model_type == YOLOModelType.SEGMENTATION and hasattr(result, 'masks'):
                # Segmentation detection
                mask = result.masks.data[i].cpu().numpy() if result.masks is not None else None
                det = SegmentationDetection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask
                    )
            elif self.model_type == YOLOModelType.POSE and hasattr(result, 'keypoints'):
                # Pose detection
                keypoints = self._extract_keypoints(result.keypoints.data[i])
                det = PoseDetection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                    keypoints=keypoints
                    )
            else:
                # Regular detection
                det = Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name
                )
            
            detections.append(det)
        
        return detections
    
    def _extract_keypoints(self, keypoints_data) -> List[KeyPoint]:
        """Extract keypoints from pose estimation result."""
        keypoints = []
        if keypoints_data is None:
            return keypoints
        
        keypoints_array = keypoints_data.cpu().numpy()
        for i in range(0, len(keypoints_array), 3):
            if i + 2 < len(keypoints_array):
                x, y, conf = keypoints_array[i:i+3]
                if conf > 0:  # Only add visible keypoints
                    keypoint = KeyPoint(x=float(x), y=float(y), confidence=float(conf))
                    keypoints.append(keypoint)
        
        return keypoints
    
    def _get_image_shape(self, image) -> Tuple[int, int]:
        """Get image shape from image input."""
        if isinstance(image, str):
            img = cv2.imread(image)
            return img.shape[:2] if img is not None else (0, 0)
        elif isinstance(image, np.ndarray):
            return image.shape[:2]
        return (0, 0)
    
    def plot_results(self, 
                    image: np.ndarray,
                    detections: DetectionBatch,
                    show_labels: bool = True,
                    show_conf: bool = True,
                    line_thickness: int = 2) -> np.ndarray:
        """
        Plot detection results on image using supervision library.
        
        Args:
            image: Input image
            detections: Detection results to plot
            show_labels: Whether to show class labels
            show_conf: Whether to show confidence scores
            line_thickness: Line thickness for annotations
            
        Returns:
            Annotated image
        """
        if not detections.detections:
            return image
        
        # Convert to supervision format
        supervision_detections = self._to_supervision_format(detections)
        
        # Annotate image
        annotated_image = self.annotator.annotate(
            scene=image.copy(),
            detections=supervision_detections,
            labels=self._get_labels(detections, show_labels, show_conf)
        )
        
        return annotated_image
    
    def _to_supervision_format(self, detections: DetectionBatch) -> SupervisionDetections:
        """Convert DetectionBatch to supervision Detections format."""
        if not detections.detections:
            return SupervisionDetections.empty()
        
        # Extract bounding boxes
        boxes = []
        confidences = []
        class_ids = []
        
        for det in detections.detections:
            bbox = det.bbox
            boxes.append([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
            confidences.append(det.confidence)
            class_ids.append(det.class_id)
        
        # Create supervision Detections object
        supervision_detections = SupervisionDetections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )
        
        return supervision_detections
    
    def _get_labels(self, 
                    detections: DetectionBatch, 
                    show_labels: bool, 
                    show_conf: bool) -> List[str]:
        """Generate labels for annotations."""
        if not show_labels and not show_conf:
            return []
        
        labels = []
        for det in detections.detections:
            label_parts = []
            
            if show_labels:
                label_parts.append(det.class_name)
            
            if show_conf:
                label_parts.append(f"{det.confidence:.2f}")
            
            labels.append(" ".join(label_parts))
        
        return labels
    
    def filter_results(self, 
                      detections: DetectionBatch,
                      class_names: Optional[List[str]] = None,
                      min_confidence: Optional[float] = None,
                      max_confidence: Optional[float] = None) -> DetectionBatch:
        """
        Filter detection results based on criteria.
        
        Args:
            detections: Input detection batch
            class_names: Filter by class names
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            
        Returns:
            Filtered DetectionBatch
        """
        filtered = detections.detections.copy()
        
        if class_names:
            filtered = [d for d in filtered if d.class_name in class_names]
        
        if min_confidence is not None:
            filtered = [d for d in filtered if d.confidence >= min_confidence]
        
        if max_confidence is not None:
            filtered = [d for d in filtered if d.confidence <= max_confidence]
        
        return DetectionBatch(
            detections=filtered,
            image_shape=detections.image_shape,
            processing_time=detections.processing_time,
            metadata=detections.metadata
        )

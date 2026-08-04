"""
Unified YOLO module for detection, segmentation, and pose estimation.

This module provides a unified interface for different YOLO models while maintaining
the same API for detection, plotting, and result filtering.
"""

import cv2
import numpy as np
from ultralytics.engine.results import Results
from ultralytics import YOLO
from typing import List, Optional, Union, Tuple, Dict
from supervision import BoxAnnotator, MaskAnnotator, VertexAnnotator
from supervision.detection.core import Detections as SupervisionDetections

from ..utils.logger import logger
from ..core.data_structures import (
    Detection, SegmentationDetection, PoseDetection,
    BoundingBox, KeyPoint
)
from ..enums import YOLOModelType


class YOLOModule:
    """
    Unified YOLO module for different model types.
    
    This class can be initialized with different YOLO models (detection, segmentation, pose)
    and provides consistent interfaces for inference, plotting, and result processing.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize YOLO module.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on ("cpu", "cuda", etc.)
        """
        self.model_path = model_path

        # Load YOLO model
        logger.info(f"Loading YOLO model from: {model_path}")
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
            logger.info(f"Moving YOLO model to device: {device}")
            self.model.to(device)

        # Initialize annotators based on model type
        self._init_annotators()

        # Class names from model
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        self.id2classes = {v: k for k, v in self.class_names.items()}
        self.labels = list(self.model.names.values())

    def class2id(self, class_name) -> int:
        return self.id2classes[class_name]

    def id2class(self, class_id) -> str:
        return self.class_names[class_id]

    def _init_annotators(self):
        """Initialize appropriate annotators based on model type."""
        if self.model_type == YOLOModelType.DETECTION:
            self.annotator = BoxAnnotator()
        elif self.model_type == YOLOModelType.SEGMENTATION:
            self.annotator = MaskAnnotator()
        elif self.model_type == YOLOModelType.POSE:
            self.annotator = VertexAnnotator()
        else:
            self.annotator = BoxAnnotator()

    def detect(self,
               image: Union[str, np.ndarray],
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45,
               **kwargs) -> List[Detection | SegmentationDetection | KeyPoint]:
        """
        Perform detection on a single frame.
        
        Args:
            image: Input image - can be path or numpy array
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional arguments for YOLO inference
            
        Returns:
            List of Detection objects
        """
        logger.debug(f"Running detection with conf={conf_threshold}, iou={iou_threshold}")

        # Run inference on single image only
        results: Results | List[Results] = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            **kwargs
        )

        # Process results for single image
        detections = self._process_single_result(results[0] if isinstance(results, list) else results)

        logger.debug(f"Detection completed. Found {len(detections)} detections")
        return detections

    def _process_single_result(self, result: Results) -> List[Detection]:
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
                    class_name=class_name,
                )

            detections.append(det)

        return detections

    @staticmethod
    def _extract_keypoints(keypoints_data) -> List[KeyPoint]:
        """Extract keypoints from pose estimation result."""
        keypoints = []
        if keypoints_data is None:
            return keypoints

        keypoints_array = keypoints_data.cpu().numpy()
        for i in range(0, len(keypoints_array)):
            x, y, conf = keypoints_array[i]
            if conf > 0:  # Only add visible keypoints
                keypoint = KeyPoint(x=float(x), y=float(y), confidence=float(conf))
                keypoints.append(keypoint)

        return keypoints

    @staticmethod
    def to_sv_format(detections: List[Detection]) -> SupervisionDetections:
        """Convert List[Detection] to supervision Detections format."""
        if not detections:
            return SupervisionDetections.empty()

        # Extract bounding boxes
        boxes = []
        confidences = []
        class_ids = []

        for det in detections:
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

    @staticmethod
    def _get_labels(detections: List[Detection],
                    show_labels: bool,
                    show_conf: bool) -> List[str]:
        """Generate labels for annotations."""
        if not show_labels and not show_conf:
            return []

        labels = []
        for det in detections:
            label_parts = []

            if show_labels:
                label_parts.append(det.class_name)

            if show_conf:
                label_parts.append(f"{det.confidence:.2f}")

            labels.append(" ".join(label_parts))

        return labels

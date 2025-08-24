"""
Data structures for ML model outputs.

This module defines the data classes used to represent detection and classification results
from various ML models. All classes are decorated with dataclasses_json for easy JSON serialization.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from dataclasses_json import dataclass_json
import numpy as np


@dataclass_json
@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x1: float  # Left coordinate
    y1: float  # Top coordinate
    x2: float  # Right coordinate
    y2: float  # Bottom coordinate
    
    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y2 - self.y1
    
    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass_json
@dataclass
class KeyPoint:
    """Key point for pose estimation."""
    x: float
    y: float
    confidence: float
    name: Optional[str] = None


@dataclass_json
@dataclass
class Detection:
    """Base detection result."""
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    score: float = field(default=0.0)
    
    def __post_init__(self):
        """Set score to confidence if not provided."""
        if self.score == 0.0:
            self.score = self.confidence


@dataclass_json
@dataclass
class SegmentationDetection(Detection):
    """Segmentation detection result with mask."""
    mask: Optional[np.ndarray] = None
    polygon: Optional[List[tuple[float, float]]] = None
    
    def __post_init__(self):
        """Convert numpy array to list for JSON serialization."""
        super().__post_init__()
        if isinstance(self.mask, np.ndarray):
            self.mask = self.mask.tolist()


@dataclass_json
@dataclass
class PoseDetection(Detection):
    """Pose detection result with keypoints."""
    keypoints: List[KeyPoint] = field(default_factory=list)
    skeleton: Optional[List[tuple[int, int]]] = None  # Connections between keypoints


@dataclass_json
@dataclass
class ActionDetection(Detection):
    """Action detection result."""
    action_type: str = ""
    duration: Optional[float] = None  # Duration of the action in seconds


@dataclass_json
@dataclass
class GameStateResult:
    """Game state classification result."""
    predicted_class: str
    confidence: float
    class_id: int
    all_probabilities: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[float] = None


@dataclass_json
@dataclass
class DetectionBatch:
    """Batch of detection results."""
    detections: List[Detection]
    image_shape: tuple[int, int] = (0, 0)  # (height, width)
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def filter_by_class(self, class_names: List[str]) -> 'DetectionBatch':
        """Filter detections by class names."""
        filtered = [d for d in self.detections if d.class_name in class_names]
        return DetectionBatch(
            detections=filtered,
            image_shape=self.image_shape,
            processing_time=self.processing_time,
            metadata=self.metadata
        )
    
    def filter_by_confidence(self, min_confidence: float) -> 'DetectionBatch':
        """Filter detections by minimum confidence."""
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return DetectionBatch(
            detections=filtered,
            image_shape=self.image_shape,
            processing_time=self.processing_time,
            metadata=self.metadata
        )
    
    def get_top_k(self, k: int) -> 'DetectionBatch':
        """Get top-k detections by confidence."""
        sorted_detections = sorted(self.detections, key=lambda x: x.confidence, reverse=True)
        return DetectionBatch(
            detections=sorted_detections[:k],
            image_shape=self.image_shape,
            processing_time=self.processing_time,
            metadata=self.metadata
        )

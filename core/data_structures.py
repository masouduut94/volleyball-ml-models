"""
Data structures for ML model outputs.

This module defines the data classes used to represent detection and classification results
from various ML models. All classes are decorated with dataclasses_json for easy JSON serialization.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from dataclasses_json import dataclass_json
import numpy as np
from ..enums import DetectorModel


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
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


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
class PlayerKeyPoints:
    """Player keypoints for pose estimation containing all body parts."""

    # Head keypoints
    nose: Optional[KeyPoint] = None
    left_eye: Optional[KeyPoint] = None
    right_eye: Optional[KeyPoint] = None
    left_ear: Optional[KeyPoint] = None
    right_ear: Optional[KeyPoint] = None

    # Upper body keypoints
    left_shoulder: Optional[KeyPoint] = None
    right_shoulder: Optional[KeyPoint] = None
    left_elbow: Optional[KeyPoint] = None
    right_elbow: Optional[KeyPoint] = None
    left_wrist: Optional[KeyPoint] = None
    right_wrist: Optional[KeyPoint] = None

    # Lower body keypoints
    left_hip: Optional[KeyPoint] = None
    right_hip: Optional[KeyPoint] = None
    left_knee: Optional[KeyPoint] = None
    right_knee: Optional[KeyPoint] = None
    left_ankle: Optional[KeyPoint] = None
    right_ankle: Optional[KeyPoint] = None

    # Additional metadata
    confidence: float = 0.0
    bbox: Optional[BoundingBox] = None
    player_id: Optional[int] = None

    def get_head_keypoints(self) -> List[KeyPoint]:
        """Get all head-related keypoints."""
        keypoints = []
        for kp in [self.nose, self.left_eye, self.right_eye, self.left_ear, self.right_ear]:
            if kp is not None:
                keypoints.append(kp)
        return keypoints

    def get_upper_body_keypoints(self) -> List[KeyPoint]:
        """Get all upper body keypoints (shoulders, elbows, wrists)."""
        keypoints = []
        for kp in [self.left_shoulder, self.right_shoulder, self.left_elbow,
                   self.right_elbow, self.left_wrist, self.right_wrist]:
            if kp is not None:
                keypoints.append(kp)
        return keypoints

    def get_lower_body_keypoints(self) -> List[KeyPoint]:
        """Get all lower body keypoints (hips, knees, ankles)."""
        keypoints = []
        for kp in [self.left_hip, self.right_hip, self.left_knee,
                   self.right_knee, self.left_ankle, self.right_ankle]:
            if kp is not None:
                keypoints.append(kp)
        return keypoints

    def get_arm_keypoints(self, side: str = "both") -> List[KeyPoint]:
        """
        Get arm keypoints for specified side.
        
        Args:
            side: "left", "right", or "both"
        """
        keypoints = []
        if side in ["left", "both"]:
            for kp in [self.left_shoulder, self.left_elbow, self.left_wrist]:
                if kp is not None:
                    keypoints.append(kp)
        if side in ["right", "both"]:
            for kp in [self.right_shoulder, self.right_elbow, self.right_wrist]:
                if kp is not None:
                    keypoints.append(kp)
        return keypoints

    def get_leg_keypoints(self, side: str = "both") -> List[KeyPoint]:
        """
        Get leg keypoints for specified side.
        
        Args:
            side: "left", "right", or "both"
        """
        keypoints = []
        if side in ["left", "both"]:
            for kp in [self.left_hip, self.left_knee, self.left_ankle]:
                if kp is not None:
                    keypoints.append(kp)
        if side in ["right", "both"]:
            for kp in [self.right_hip, self.right_knee, self.right_ankle]:
                if kp is not None:
                    keypoints.append(kp)
        return keypoints

    def get_hand_positions(self) -> List[KeyPoint]:
        """Get hand/wrist positions."""
        keypoints = []
        for kp in [self.left_wrist, self.right_wrist]:
            if kp is not None:
                keypoints.append(kp)
        return keypoints

    def get_shoulder_positions(self) -> List[KeyPoint]:
        """Get shoulder positions."""
        keypoints = []
        for kp in [self.left_shoulder, self.right_shoulder]:
            if kp is not None:
                keypoints.append(kp)
        return keypoints

    def get_all_keypoints(self) -> List[KeyPoint]:
        """Get all available keypoints."""
        keypoints = []
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, KeyPoint):
                    keypoints.append(attr_value)
        return keypoints

    def is_visible(self, keypoint_name: str, min_confidence: float = 0.5) -> bool:
        """
        Check if a specific keypoint is visible with sufficient confidence.
        
        Args:
            keypoint_name: Name of the keypoint (e.g., 'left_wrist', 'nose')
            min_confidence: Minimum confidence threshold
        """
        if hasattr(self, keypoint_name):
            kp = getattr(self, keypoint_name)
            return kp is not None and kp.confidence >= min_confidence
        return False

    @classmethod
    def from_yolo_output(cls, 
                        bbox_coords: List[float], 
                        keypoints: np.ndarray, 
                        confidence: float,
                        player_id: Optional[int] = None) -> 'PlayerKeyPoints':
        """
        Initialize PlayerKeyPoints from YOLO pose model output.
        
        Args:
            bbox_coords: Bounding box coordinates [x1, y1, x2, y2]
            keypoints: YOLO keypoints array of shape (17, 3) where each row is [x, y, confidence]
            confidence: Overall detection confidence
            player_id: Optional player ID
            
        Returns:
            PlayerKeyPoints instance
        """
        # Create bounding box
        bbox = BoundingBox(x1=bbox_coords[0], y1=bbox_coords[1], 
                          x2=bbox_coords[2], y2=bbox_coords[3])
        
        # COCO keypoint names in order
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Create keypoint dictionary
        keypoint_dict = {}
        for i, name in enumerate(keypoint_names):
            if i < len(keypoints):
                x, y, kp_conf = keypoints[i]
                if kp_conf > 0:  # Only add visible keypoints
                    keypoint_dict[name] = KeyPoint(x=float(x), y=float(y), 
                                                  confidence=float(kp_conf), name=name)
                else:
                    keypoint_dict[name] = None
            else:
                keypoint_dict[name] = None
        
        return cls(
            bbox=bbox,
            confidence=confidence,
            player_id=player_id,
            **keypoint_dict
        )


@dataclass_json
@dataclass
class Detection:
    """Base detection result."""
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    model: str


@dataclass_json
@dataclass
class SegmentationDetection(Detection):
    """Segmentation detection result with mask."""
    mask: Optional[np.ndarray] = None
    polygon: Optional[List[tuple[float, float]]] = None

    def __post_init__(self):
        """Convert numpy array to list for JSON serialization."""
        if isinstance(self.mask, np.ndarray):
            self.mask = self.mask.tolist()


@dataclass_json
@dataclass
class PoseDetection(Detection):
    """Pose detection result with keypoints."""
    keypoints: List[KeyPoint] = field(default_factory=list)


@dataclass_json
@dataclass
class GameStateResult:
    """Game state classification result."""
    predicted_class: str
    confidence: float




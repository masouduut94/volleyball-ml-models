"""
Player keypoint detection module for volleyball.

This module provides specialized player pose estimation functionality using YOLO pose models
for volleyball player keypoint detection and analysis.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
import cv2
from .YoloModule import YOLOModule
from ..core.data_structures import PlayerKeyPoints, KeyPoint, BoundingBox, PoseDetection, Detection
from ..enums import YOLOModelType, DetectorModel
from ..utils.logger import logger


class PlayerDetectorModule:
    """
    Player keypoint detection module for volleyball.
    
    This class provides pose estimation functionality using YOLO pose models
    specifically designed for volleyball player keypoint detection.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None):
        """
        Initialize player keypoint detection module.
        
        Args:
            model_path: Path to YOLO pose model weights (e.g., yolo11n-pose.pt)
            device: Device to run inference on
        """
        self.model_path = model_path
        logger.info(f"Initializing PlayerModule with model: {model_path}")
        
        self.yolo_module = YOLOModule(
            model_path=model_path,
            device=device
        )
        
        # Standard COCO keypoint names in order
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def detect(self, 
               image: Union[str, np.ndarray],
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45,
               **kwargs) -> List[PlayerKeyPoints]:
        """
        Detect player keypoints in a single frame.
        
        Args:
            image: Input image (single frame)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional arguments for detection
            
        Returns:
            List of PlayerKeyPoints objects containing pose information
        """
        detections = self.yolo_module.detect(
            image, 
            conf_threshold, 
            iou_threshold,
            detector_model=DetectorModel.PLAYER_DETECTOR.value,
            **kwargs
        )
        
        player_keypoints = []
        for i, detection in enumerate(detections):
            # Only process pose detections
            if isinstance(detection, PoseDetection) and detection.keypoints:
                player_kp = self._convert_to_player_keypoints(detection, player_id=i)
                player_keypoints.append(player_kp)
        
        return player_keypoints
    
    def _convert_to_player_keypoints(self, pose_detection: PoseDetection, player_id: int) -> PlayerKeyPoints:
        """
        Convert a PoseDetection to PlayerKeyPoints.
        
        Args:
            pose_detection: PoseDetection object from YOLO
            player_id: Unique identifier for the player
            
        Returns:
            PlayerKeyPoints object with structured keypoint data
        """
        # Create a dictionary to map keypoint names to KeyPoint objects
        keypoint_dict = {}
        
        # Map YOLO keypoints to our structure
        for i, keypoint in enumerate(pose_detection.keypoints):
            if i < len(self.keypoint_names):
                keypoint_name = self.keypoint_names[i]
                keypoint_dict[keypoint_name] = keypoint
        
        # Create PlayerKeyPoints object
        player_kp = PlayerKeyPoints(
            # Head keypoints
            nose=keypoint_dict.get('nose'),
            left_eye=keypoint_dict.get('left_eye'),
            right_eye=keypoint_dict.get('right_eye'),
            left_ear=keypoint_dict.get('left_ear'),
            right_ear=keypoint_dict.get('right_ear'),
            
            # Upper body keypoints
            left_shoulder=keypoint_dict.get('left_shoulder'),
            right_shoulder=keypoint_dict.get('right_shoulder'),
            left_elbow=keypoint_dict.get('left_elbow'),
            right_elbow=keypoint_dict.get('right_elbow'),
            left_wrist=keypoint_dict.get('left_wrist'),
            right_wrist=keypoint_dict.get('right_wrist'),
            
            # Lower body keypoints
            left_hip=keypoint_dict.get('left_hip'),
            right_hip=keypoint_dict.get('right_hip'),
            left_knee=keypoint_dict.get('left_knee'),
            right_knee=keypoint_dict.get('right_knee'),
            left_ankle=keypoint_dict.get('left_ankle'),
            right_ankle=keypoint_dict.get('right_ankle'),
            
            # Additional metadata
            confidence=pose_detection.confidence,
            bbox=pose_detection.bbox,
            player_id=player_id
        )
        
        return player_kp
    
    def get_player_count(self, player_keypoints: List[PlayerKeyPoints]) -> int:
        """
        Get total number of players detected.
        
        Args:
            player_keypoints: List of PlayerKeyPoints
            
        Returns:
            Number of players detected
        """
        return len(player_keypoints)
    
    def get_player_positions(self, player_keypoints: List[PlayerKeyPoints]) -> List[Tuple[float, float]]:
        """
        Get center positions of all detected players based on their bounding boxes.
        
        Args:
            player_keypoints: List of PlayerKeyPoints
            
        Returns:
            List of (x, y) coordinates representing player positions
        """
        positions = []
        for player_kp in player_keypoints:
            if player_kp.bbox:
                positions.append(player_kp.bbox.center)
            else:
                # If no bbox, calculate center from visible keypoints
                visible_kps = [kp for kp in player_kp.get_all_keypoints() if kp.confidence > 0.5]
                if visible_kps:
                    avg_x = sum(kp.x for kp in visible_kps) / len(visible_kps)
                    avg_y = sum(kp.y for kp in visible_kps) / len(visible_kps)
                    positions.append((avg_x, avg_y))
                else:
                    positions.append((0.0, 0.0))
        return positions


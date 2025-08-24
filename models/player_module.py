"""
Player detection, segmentation, and pose estimation module for volleyball.

This module provides specialized player analysis functionality using YOLO models
trained for volleyball player recognition, segmentation, and pose estimation.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
import cv2
from .yolo_module import YOLOModule
from ..core.data_structures import DetectionBatch
from ..enums import PlayerDetectionMode, YOLOModelType


class PlayerModule:
    """
    Unified player analysis module for volleyball.
    
    This class can switch between different player analysis modes:
    - detection: Basic player bounding box detection
    - segmentation: Player segmentation with masks
    - pose: Player pose estimation with keypoints
    
    It wraps the YOLOModule and provides volleyball-specific utilities.
    """
    
    def __init__(self, 
                 model_path: str,
                 mode: PlayerDetectionMode = PlayerDetectionMode.POSE,
                 device: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize player module.
        
        Args:
            model_path: Path to player model weights
            mode: Analysis mode (detection, segmentation, or pose)
            device: Device to run inference on
            verbose: Whether to print verbose output
        """
        self.mode = mode
        self.model_path = model_path
        
        # Map mode to YOLO model type
        mode_to_type = {
            PlayerDetectionMode.DETECTION: YOLOModelType.DETECTION,
            PlayerDetectionMode.SEGMENTATION: YOLOModelType.SEGMENTATION, 
            PlayerDetectionMode.POSE: YOLOModelType.POSE
        }
        
        self.yolo_module = YOLOModule(
            model_path=model_path,
            device=device,
            verbose=verbose
        )
        
        # Player-related class names
        self.player_class_names = ["person", "player", "volleyball_player"]
    
    def switch_mode(self, new_mode: PlayerDetectionMode, new_model_path: Optional[str] = None) -> None:
        """
        Switch to a different analysis mode.
        
        Args:
            new_mode: New analysis mode
            new_model_path: Optional new model path for the mode
        """
        if new_mode == self.mode and not new_model_path:
            return  # No change needed
        
        # Update mode
        self.mode = new_mode
        
        # Update model path if provided
        if new_model_path:
            self.model_path = new_model_path
        
        # Reinitialize YOLO module with new mode
        mode_to_type = {
            PlayerDetectionMode.DETECTION: YOLOModelType.DETECTION,
            PlayerDetectionMode.SEGMENTATION: YOLOModelType.SEGMENTATION,
            PlayerDetectionMode.POSE: YOLOModelType.POSE
        }
        
        self.yolo_module = YOLOModule(
            model_path=self.model_path,
            device=self.yolo_module.model.device if hasattr(self.yolo_module, 'model') else None,
            verbose=self.yolo_module.verbose
        )
    
    def detect_players(self, 
                      image: Union[str, np.ndarray, List[str], List[np.ndarray]],
                      conf_threshold: float = 0.25,
                      iou_threshold: float = 0.45,
                      **kwargs) -> DetectionBatch:
        """
        Detect players in image(s) using current mode.
        
        Args:
            image: Input image(s)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional arguments for detection
            
        Returns:
            DetectionBatch with player detection results
        """
        detections = self.yolo_module.detect(image, conf_threshold, iou_threshold, **kwargs)
        
        # Filter to only player detections
        player_detections = []
        for det in detections.detections:
            if det.class_name.lower() in self.player_class_names:
                player_detections.append(det)
        
        return DetectionBatch(
            detections=player_detections,
            image_shape=detections.image_shape,
            processing_time=detections.processing_time,
            metadata=detections.metadata
        )
    
    def get_player_positions(self, detections: DetectionBatch) -> List[Tuple[float, float]]:
        """
        Get positions of all detected players.
        
        Args:
            detections: Player detection results
            
        Returns:
            List of (x, y) coordinates representing player positions
        """
        positions = []
        for det in detections.detections:
            positions.append(det.bbox.center)
        return positions
    
    def get_player_count(self, detections: DetectionBatch) -> int:
        """
        Get total number of players detected.
        
        Args:
            detections: Player detection results
            
        Returns:
            Number of players detected
        """
        return len(detections.detections)
    
    def get_player_poses(self, detections: DetectionBatch) -> List[Dict[str, Any]]:
        """
        Get pose information for all players (only in pose mode).
        
        Args:
            detections: Player detection results
            
        Returns:
            List of pose dictionaries with keypoints and skeleton info
        """
        if self.mode != PlayerDetectionMode.POSE:
            return []
        
        poses = []
        for det in detections.detections:
            if hasattr(det, 'keypoints') and det.keypoints:
                pose_info = {
                    'player_id': len(poses),
                    'keypoints': [(kp.x, kp.y, kp.confidence) for kp in det.keypoints],
                    'bbox': det.bbox,
                    'confidence': det.confidence
                }
                poses.append(pose_info)
        
        return poses
    
    def get_player_segments(self, detections: DetectionBatch) -> List[np.ndarray]:
        """
        Get segmentation masks for all players (only in segmentation mode).
        
        Args:
            detections: Player detection results
            
        Returns:
            List of segmentation masks
        """
        if self.mode != PlayerDetectionMode.SEGMENTATION:
            return []
        
        masks = []
        for det in detections.detections:
            if hasattr(det, 'mask') and det.mask is not None:
                masks.append(np.array(det.mask))
        
        return masks
    
    def plot_players(self, 
                    image: np.ndarray,
                    detections: DetectionBatch,
                    show_labels: bool = True,
                    show_conf: bool = True,
                    show_keypoints: bool = True,
                    line_thickness: int = 2) -> np.ndarray:
        """
        Plot player detection results on image.
        
        Args:
            image: Input image
            detections: Player detection results
            show_labels: Whether to show labels
            show_conf: Whether to show confidence scores
            show_keypoints: Whether to show keypoints (pose mode only)
            line_thickness: Line thickness for annotations
            
        Returns:
            Annotated image with player detections
        """
        # Plot basic detections
        annotated_image = self.yolo_module.plot_results(
            image, detections, show_labels, show_conf, line_thickness
        )
        
        # Add pose-specific visualizations
        if self.mode == PlayerDetectionMode.POSE and show_keypoints:
            for det in detections.detections:
                if hasattr(det, 'keypoints') and det.keypoints:
                    # Draw keypoints
                    for kp in det.keypoints:
                        if kp.confidence > 0.3:  # Only show confident keypoints
                            cv2.circle(
                                annotated_image,
                                (int(kp.x), int(kp.y)),
                                3,
                                (0, 255, 255),
                                -1
                            )
        
        return annotated_image
    
    def get_mode_info(self) -> Dict[str, Any]:
        """
        Get information about current mode and capabilities.
        
        Returns:
            Dictionary with mode information
        """
        return {
            'current_mode': self.mode,
            'model_path': self.model_path,
            'capabilities': {
                'detection': self.mode in [PlayerDetectionMode.DETECTION, PlayerDetectionMode.SEGMENTATION, PlayerDetectionMode.POSE],
                'segmentation': self.mode == PlayerDetectionMode.SEGMENTATION,
                'pose_estimation': self.mode == PlayerDetectionMode.POSE,
                'keypoints': self.mode == PlayerDetectionMode.POSE
            },
            'class_names': self.player_class_names
        }

"""
Multi-object tracking module for volleyball analytics.

This module provides tracking capabilities for balls and players using multiple
tracking algorithms including Norfair, SORT, and custom volleyball-specific trackers.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from norfair import Detection, Tracker, Video
    NORFAIR_AVAILABLE = True
except ImportError:
    NORFAIR_AVAILABLE = False
    logging.warning("Norfair not available. Install with: pip install norfair")

try:
    from sort import Sort
    SORT_AVAILABLE = True
except ImportError:
    SORT_AVAILABLE = False
    logging.warning("SORT not available. Install with: pip install sort-python")

from .data_structures import DetectionObject


class TrackerType(Enum):
    """Available tracking algorithms."""
    NORFAIR = "norfair"
    SORT = "sort"
    CUSTOM = "custom"


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    class_id: int
    frame_count: int
    last_seen: int
    trajectory: List[Tuple[float, float]]  # List of (x, y) center points
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy)
    is_active: bool = True


@dataclass
class TrackingConfig:
    """Configuration for tracking algorithms."""
    tracker_type: TrackerType = TrackerType.NORFAIR
    max_disappeared: int = 30  # Frames before marking track as lost
    min_hits: int = 3  # Minimum detections before confirming track
    iou_threshold: float = 0.3  # IoU threshold for association
    distance_threshold: float = 50.0  # Distance threshold for association
    velocity_weight: float = 0.1  # Weight for velocity in association
    trajectory_length: int = 30  # Number of trajectory points to keep


class VolleyballTracker:
    """
    Multi-object tracker specialized for volleyball.
    
    Tracks balls and players with volleyball-specific logic including:
    - Ball trajectory analysis
    - Player movement patterns
    - Court boundary constraints
    - Action-based tracking updates
    """
    
    def __init__(self, 
                 config: Optional[TrackingConfig] = None,
                 verbose: bool = False):
        """
        Initialize volleyball tracker.
        
        Args:
            config: Tracking configuration
            verbose: Whether to print verbose output
        """
        self.config = config or TrackingConfig()
        self.verbose = verbose
        
        # Initialize trackers
        self._init_trackers()
        
        # Tracking state
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_track_id = 0
        self.frame_count = 0
        
        # Volleyball-specific tracking
        self.ball_tracks: Dict[int, TrackedObject] = {}
        self.player_tracks: Dict[int, TrackedObject] = {}
        
        if self.verbose:
            print(f"VolleyballTracker initialized with {self.config.tracker_type.value}")
    
    def _init_trackers(self):
        """Initialize tracking algorithms."""
        if self.config.tracker_type == TrackerType.NORFAIR and NORFAIR_AVAILABLE:
            self._init_norfair_tracker()
        elif self.config.tracker_type == TrackerType.SORT and SORT_AVAILABLE:
            self._init_sort_tracker()
        else:
            self._init_custom_tracker()
    
    def _init_norfair_tracker(self):
        """Initialize Norfair tracker."""
        self.norfair_tracker = Tracker(
            distance_threshold=self.config.distance_threshold,
            hit_counter_max=self.config.min_hits,
            filter_setup_time=0,
            period=1
        )
    
    def _init_sort_tracker(self):
        """Initialize SORT tracker."""
        self.sort_tracker = Sort(
            max_age=self.config.max_disappeared,
            min_hits=self.config.min_hits,
            iou_threshold=self.config.iou_threshold
        )
    
    def _init_custom_tracker(self):
        """Initialize custom tracking logic."""
        self.custom_tracker = None
        if self.verbose:
            print("Using custom tracking logic")
    
    def update(self, 
               detections: List[DetectionObject],
               frame: np.ndarray,
               frame_number: int) -> List[TrackedObject]:
        """
        Update tracking with new detections.
        
        Args:
            detections: List of detection objects
            frame: Current frame
            frame_number: Current frame number
            
        Returns:
            List of currently tracked objects
        """
        self.frame_count = frame_number
        
        # Separate ball and player detections
        ball_detections = [d for d in detections if d.get('name') == 'ball']
        player_detections = [d for d in detections if d.get('name') == 'person']
        
        # Update tracking based on tracker type
        if self.config.tracker_type == TrackerType.NORFAIR:
            tracked_objects = self._update_norfair(detections)
        elif self.config.tracker_type == TrackerType.SORT:
            tracked_objects = self._update_sort(detections)
        else:
            tracked_objects = self._update_custom(detections)
        
        # Update volleyball-specific tracking
        self._update_volleyball_tracking(tracked_objects, ball_detections, player_detections)
        
        # Clean up old tracks
        self._cleanup_tracks()
        
        return list(self.tracked_objects.values())
    
    def _update_norfair(self, detections: List[DetectionObject]) -> List[TrackedObject]:
        """Update tracking using Norfair."""
        if not NORFAIR_AVAILABLE:
            return []
        
        # Convert detections to Norfair format
        norfair_detections = []
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            norfair_detections.append(Detection(
                points=np.array(center),
                scores=np.array([det.get('confidence', 0.5)]),
                label=det.get('name', 'unknown')
            ))
        
        # Update Norfair tracker
        tracked_objects = self.norfair_tracker.update(detections=norfair_detections)
        
        # Convert back to our format
        result = []
        for track in tracked_objects:
            if track.hit_counter >= self.config.min_hits:
                bbox = self._get_bbox_from_center(track.estimate, track.last_detection)
                tracked_obj = TrackedObject(
                    track_id=track.id,
                    bbox=bbox,
                    confidence=track.last_detection.scores[0],
                    class_name=track.last_detection.label,
                    class_id=0,  # Will be updated based on class name
                    frame_count=self.frame_count,
                    last_seen=self.frame_count,
                    trajectory=[(p[0], p[1]) for p in track.estimate],
                    is_active=True
                )
                result.append(tracked_obj)
        
        return result
    
    def _update_sort(self, detections: List[DetectionObject]) -> List[TrackedObject]:
        """Update tracking using SORT."""
        if not SORT_AVAILABLE:
            return []
        
        # Convert detections to SORT format
        detection_array = []
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            detection_array.append([
                bbox[0], bbox[1], bbox[2], bbox[3], 
                det.get('confidence', 0.5)
            ])
        
        if not detection_array:
            detection_array = np.empty((0, 5))
        else:
            detection_array = np.array(detection_array)
        
        # Update SORT tracker
        tracked_objects = self.sort_tracker.update(detection_array)
        
        # Convert back to our format
        result = []
        for track in tracked_objects:
            if track[4] > 0:  # SORT returns [x1, y1, x2, y2, track_id]
                tracked_obj = TrackedObject(
                    track_id=int(track[4]),
                    bbox=track[:4].tolist(),
                    confidence=0.8,  # SORT doesn't provide confidence
                    class_name='unknown',
                    class_id=0,
                    frame_count=self.frame_count,
                    last_seen=self.frame_count,
                    trajectory=[((track[0] + track[2]) / 2, (track[1] + track[3]) / 2)],
                    is_active=True
                )
                result.append(tracked_obj)
        
        return result
    
    def _update_custom(self, detections: List[DetectionObject]) -> List[TrackedObject]:
        """Update tracking using custom logic."""
        # Simple custom tracking based on IoU and distance
        tracked_objects = []
        
        for det in detections:
            best_match = None
            best_score = 0
            
            for track_id, tracked_obj in self.tracked_objects.items():
                if tracked_obj.class_name == det.get('name'):
                    # Calculate similarity score
                    iou_score = self._calculate_iou(tracked_obj.bbox, det.get('bbox', [0, 0, 0, 0]))
                    distance_score = self._calculate_distance(tracked_obj.bbox, det.get('bbox', [0, 0, 0, 0]))
                    
                    # Combined score
                    score = iou_score * 0.7 + distance_score * 0.3
                    
                    if score > best_score and score > 0.3:
                        best_score = score
                        best_match = track_id
            
            if best_match is not None:
                # Update existing track
                tracked_obj = self.tracked_objects[best_match]
                tracked_obj.bbox = det.get('bbox', [0, 0, 0, 0])
                tracked_obj.confidence = det.get('confidence', 0.5)
                tracked_obj.last_seen = self.frame_count
                
                # Update trajectory
                center = ((tracked_obj.bbox[0] + tracked_obj.bbox[2]) / 2, 
                         (tracked_obj.bbox[1] + tracked_obj.bbox[3]) / 2)
                tracked_obj.trajectory.append(center)
                
                # Keep only recent trajectory points
                if len(tracked_obj.trajectory) > self.config.trajectory_length:
                    tracked_obj.trajectory = tracked_obj.trajectory[-self.config.trajectory_length:]
                
                tracked_objects.append(tracked_obj)
            else:
                # Create new track
                new_track = TrackedObject(
                    track_id=self.next_track_id,
                    bbox=det.get('bbox', [0, 0, 0, 0]),
                    confidence=det.get('confidence', 0.5),
                    class_name=det.get('name', 'unknown'),
                    class_id=0,
                    frame_count=self.frame_count,
                    last_seen=self.frame_count,
                    trajectory=[((det.get('bbox', [0, 0, 0, 0])[0] + det.get('bbox', [0, 0, 0, 0])[2]) / 2,
                               (det.get('bbox', [0, 0, 0, 0])[1] + det.get('bbox', [0, 0, 0, 0])[3]) / 2)],
                    is_active=True
                )
                
                self.tracked_objects[self.next_track_id] = new_track
                tracked_objects.append(new_track)
                self.next_track_id += 1
        
        return tracked_objects
    
    def _update_volleyball_tracking(self, 
                                  tracked_objects: List[TrackedObject],
                                  ball_detections: List[DetectionObject],
                                  player_detections: List[DetectionObject]):
        """Update volleyball-specific tracking state."""
        # Update ball tracks
        for obj in tracked_objects:
            if obj.class_name == 'ball':
                self.ball_tracks[obj.track_id] = obj
            elif obj.class_name == 'person':
                self.player_tracks[obj.track_id] = obj
    
    def _cleanup_tracks(self):
        """Remove old and inactive tracks."""
        current_time = self.frame_count
        tracks_to_remove = []
        
        for track_id, tracked_obj in self.tracked_objects.items():
            # Mark track as inactive if not seen recently
            if current_time - tracked_obj.last_seen > self.config.max_disappeared:
                tracked_obj.is_active = False
                tracks_to_remove.append(track_id)
        
        # Remove inactive tracks
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
            if track_id in self.ball_tracks:
                del self.ball_tracks[track_id]
            if track_id in self.player_tracks:
                del self.player_tracks[track_id]
    
    def _get_bbox_from_center(self, center_points: np.ndarray) -> List[float]:
        """Convert center points to bounding box."""
        if len(center_points) == 0:
            return [0, 0, 0, 0]
        
        # Use last center point
        center = center_points[-1]
        
        # Estimate bbox size (this could be improved with detection size)
        bbox_size = 50  # Default size
        x1 = max(0, center[0] - bbox_size // 2)
        y1 = max(0, center[1] - bbox_size // 2)
        x2 = center[0] + bbox_size // 2
        y2 = center[1] + bbox_size // 2
        
        return [x1, y1, x2, y2]
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between two bounding box centers."""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def get_ball_trajectory(self, track_id: Optional[int] = None) -> List[Tuple[float, float]]:
        """Get ball trajectory for analysis."""
        if track_id is None and self.ball_tracks:
            # Return trajectory of most recent ball track
            track_id = max(self.ball_tracks.keys(), key=lambda k: self.ball_tracks[k].last_seen)
        
        if track_id in self.ball_tracks:
            return self.ball_tracks[track_id].trajectory
        return []
    
    def get_player_tracks(self) -> Dict[int, TrackedObject]:
        """Get all currently tracked players."""
        return {k: v for k, v in self.player_tracks.items() if v.is_active}
    
    def get_ball_tracks(self) -> Dict[int, TrackedObject]:
        """Get all currently tracked balls."""
        return {k: v for k, v in self.ball_tracks.items() if v.is_active}
    
    def reset(self):
        """Reset all tracking state."""
        self.tracked_objects.clear()
        self.ball_tracks.clear()
        self.player_tracks.clear()
        self.next_track_id = 0
        self.frame_count = 0
        
        # Reinitialize trackers
        self._init_trackers()
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            'total_tracks': len(self.tracked_objects),
            'active_tracks': len([t for t in self.tracked_objects.values() if t.is_active]),
            'ball_tracks': len(self.ball_tracks),
            'player_tracks': len(self.player_tracks),
            'frame_count': self.frame_count
        }

"""
Multi-object tracking module for volleyball analytics.

This module provides tracking capabilities for balls and players using the
Norfair tracking algorithm optimized for volleyball-specific tracking.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from norfair import Detection, Tracker
from norfair.distances import mean_euclidean


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
    """Configuration for Norfair tracking algorithm."""
    max_disappeared: int = 30  # Frames before marking track as lost
    min_hits: int = 3  # Minimum detections before confirming track
    distance_threshold: float = 50.0  # Distance threshold for association
    trajectory_length: int = 30  # Number of trajectory points to keep


class VolleyballTracker:
    """
    Norfair-based multi-object tracker specialized for volleyball.
    
    Tracks balls and players with volleyball-specific logic including:
    - Ball trajectory analysis using Norfair
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
            print("VolleyballTracker initialized with Norfair tracker")
    
    def _init_trackers(self):
        """Initialize Norfair tracking algorithm."""
        self._init_norfair_tracker()
    
    def _init_norfair_tracker(self):
        """Initialize Norfair tracker."""
        self.norfair_tracker = Tracker(
            distance_function=mean_euclidean,
            distance_threshold=self.config.distance_threshold,
            hit_counter_max=self.config.min_hits,
            initialization_delay=0,
            pointwise_hit_counter_max=self.config.min_hits
        )
    
    def update(self, 
               detections: List[Detection],
               frame_number: int) -> List[TrackedObject]:
        """
        Update tracking with new detections.
        
        Args:
            detections: List of Norfair Detection objects
            frame_number: Current frame number
            
        Returns:
            List of currently tracked objects
        """
        self.frame_count = frame_number
        
        # Update tracking using Norfair
        tracked_objects = self._update_norfair(detections)
        
        # Update volleyball-specific tracking
        self._update_volleyball_tracking(tracked_objects)
        
        # Clean up old tracks
        self._cleanup_tracks()
        
        return list(self.tracked_objects.values())
    
    def _update_norfair(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracking using Norfair."""
        # Update Norfair tracker with detections
        tracked_objects = self.norfair_tracker.update(detections=detections)
        
        # Convert to our TrackedObject format
        result = []
        for track in tracked_objects:
            if track.hit_counter >= self.config.min_hits:
                # Extract bounding box from detection points
                bbox = self._get_bbox_from_points(track.estimate[-1])
                
                tracked_obj = TrackedObject(
                    track_id=track.id,
                    bbox=bbox,
                    confidence=track.last_detection.scores[0] if track.last_detection.scores.size > 0 else 0.5,
                    class_name=getattr(track.last_detection, 'label', 'unknown'),
                    class_id=0,
                    frame_count=self.frame_count,
                    last_seen=self.frame_count,
                    trajectory=[(p[0], p[1]) for p in track.estimate],
                    is_active=True
                )
                result.append(tracked_obj)
        
        return result
    
    def _update_volleyball_tracking(self, tracked_objects: List[TrackedObject]):
        """Update volleyball-specific tracking state."""
        # Update ball and player tracks
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
    
    def _get_bbox_from_points(self, points: np.ndarray) -> List[float]:
        """Convert detection points to bounding box."""
        if points.size == 0:
            return [0, 0, 0, 0]
        
        # If points is a center point, create bounding box around it
        if points.ndim == 1 and len(points) == 2:
            center_x, center_y = points
            bbox_size = 50  # Default size
            x1 = max(0, center_x - bbox_size // 2)
            y1 = max(0, center_y - bbox_size // 2)
            x2 = center_x + bbox_size // 2
            y2 = center_y + bbox_size // 2
            return [x1, y1, x2, y2]
        
        # If multiple points, use bounding box of all points
        x_coords = points[:, 0] if points.ndim > 1 else [points[0]]
        y_coords = points[:, 1] if points.ndim > 1 else [points[1]]
        
        x1, x2 = float(np.min(x_coords)), float(np.max(x_coords))
        y1, y2 = float(np.min(y_coords)), float(np.max(y_coords))
        
        return [x1, y1, x2, y2]
    
    
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

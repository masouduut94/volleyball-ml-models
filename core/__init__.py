"""
Core package for ML Manager.

This package contains core data structures and utilities that are not specific
to deep learning models but are used throughout the ML Manager system.
"""

from .data_structures import (
    Detection, SegmentationDetection, PoseDetection,
    BoundingBox, KeyPoint, GameStateResult, PlayerKeyPoints
)
from .tracking_module import (
    VolleyballTracker, TrackedObject, TrackingConfig
)

__all__ = [
    # Data structures
    "Detection",
    "SegmentationDetection", 
    "PoseDetection",
    "BoundingBox",
    "KeyPoint",
    "GameStateResult",
    "PlayerKeyPoints",
    
    # Tracking
    "VolleyballTracker",
    "TrackedObject", 
    "TrackingConfig"
]

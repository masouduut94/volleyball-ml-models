"""
Enums for ML Manager module.

This module contains string enums for various ML model types and configurations
to provide type safety and autocomplete support.
"""

from enum import StrEnum


class YOLOModelType(StrEnum):
    """String enum for YOLO model types."""
    
    DETECTION = "detect"
    SEGMENTATION = "segment"
    POSE = "pose"
    CLASSIFICATION = "classify"
    OBB = "obb"  # Oriented bounding box


class PlayerDetectionMode(StrEnum):
    """String enum for player detection modes."""
    
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    POSE = "pose"


class GameState(StrEnum):
    """String enum for game states."""
    
    PLAY = "play"
    NO_PLAY = "no_play"
    CHALLENGE = "challenge"
    UNKNOWN = "unknown"


class VolleyballAction(StrEnum):
    """String enum for volleyball actions."""
    
    SPIKE = "spike"
    BLOCK = "block"
    RECEPTION = "reception"
    SERVICE = "service"
    SETTER = "setter"
    UNKNOWN = "unknown"

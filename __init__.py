"""
Volleyball Analytics ML Manager Module

This module provides a unified interface for all machine learning models
used in volleyball analytics, including object detection, segmentation,
action recognition, and game state classification.
"""

from .ml_manager import MLManager

__version__ = "1.0.0"
__author__ = "Volleyball Analytics Team"

__all__ = [
    "MLManager"
]

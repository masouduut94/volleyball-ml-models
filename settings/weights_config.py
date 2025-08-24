"""
Model weights configuration for ML Manager.

This module contains the Pydantic settings for model weights configuration.
"""

from typing import Optional
from pydantic import BaseSettings, Field


class ModelWeightsConfig(BaseSettings):
    """
    Pydantic settings for model weights configuration.
    
    Attributes:
        ball_detection: Path to ball detection model weights
        action_detection: Path to action detection model weights
        game_status: Path to game status classification model weights
        court_detection: Path to court detection model weights
        player_detection: Path to player detection model weights (None for default YOLO pose)
    """
    
    ball_detection: Optional[str] = Field(
        default="../weights/ball_segment/model1/weights/best.pt",
        description="Path to ball detection model weights"
    )
    action_detection: Optional[str] = Field(
        default="../weights/action_detection/6_class/1/weights/best.pt",
        description="Path to action detection model weights"
    )
    game_status: Optional[str] = Field(
        default="../weights/game-state/3-states/checkpoint",
        description="Path to game status classification model weights"
    )
    court_detection: Optional[str] = Field(
        default="../weights/court_segment/weights/best.pt",
        description="Path to court detection model weights"
    )
    player_detection: Optional[str] = Field(
        default="None",
        description="Path to player detection model weights (None for default YOLO pose)"
    )
    
    class Config:
        env_prefix = "ML_"
        case_sensitive = False

"""
Model weights configuration for ML Manager.

This module contains the Pydantic settings for model weights configuration.
"""

import yaml
from pathlib import Path
from typing import Optional, Union, Dict
from pydantic import Field
from pydantic_settings import BaseSettings


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
        default="./weights/ball/weights/best.pt",
        description="Path to ball detection model weights"
    )
    action_detection: Optional[str] = Field(
        default="./weights/action/weights/best.pt",
        description="Path to action detection model weights"
    )
    game_status: Optional[str] = Field(
        default="./weights/game_state",
        description="Path to game status classification model weights"
    )
    court_detection: Optional[str] = Field(
        default="./weights/court/weights/best.pt",
        description="Path to court detection model weights"
    )
    player_detection: Optional[str] = Field(
        default=None,
        description="Path to player detection model weights (None for default YOLO pose)"
    )

    class Config:
        env_prefix = "ML_"
        case_sensitive = False

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'ModelWeightsConfig':
        """Load ModelWeightsConfig from a YAML file."""
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to a YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)

    @staticmethod
    def download_missing_weights(weights_dir: Optional[Union[str, Path]] = None, force: bool = False) -> bool:
        """
        Download missing model weights from Google Drive.
        
        Args:
            weights_dir: Base directory for weights (defaults to parent of configured paths)
            force: Whether to force download even if weights exist
            
        Returns:
            True if download was successful, False otherwise
        """
        from ..utils.downloader import download_all_models

        return download_all_models(weights_dir=weights_dir, force_download=force)

    @staticmethod
    def check_weights_availability(weights_dir: Optional[Union[str, Path]] = None) -> Dict[str, bool]:
        """
        Check which model weights are available locally.
        
        Args:
            weights_dir: Base directory for weights (defaults to parent of configured paths)
            
        Returns:
            Dictionary mapping model names to availability status
        """
        from ..utils.downloader import check_model_weights

        return check_model_weights(weights_dir=weights_dir)

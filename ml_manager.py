"""
Main ML Manager Class

This module provides the unified MLManager class that handles all deep learning models
for volleyball analytics without requiring external configuration files.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

# Import model classes
from .core import GameStateResult, VolleyballTracker
from .visualization import VolleyballVisualizer
from .core.data_structures import PlayerKeyPoints, Detection
from .settings import ModelWeightsConfig
from .utils.logger import logger
from .models import (
    ActionDetectorModule, BallDetectorModule, CourtSegmentationModule,
    PlayerDetectorModule, GameStatusClassifierModule
)


class MLManager:
    """
    Unified ML Manager for Volleyball Analytics
    
    This class manages all deep learning models including:
    - Action Detection (YOLO)
    - Ball Segmentation (YOLO)
    - Court Segmentation (YOLO)
    - Player Detection (YOLO)
    - Game State Classification (VideoMAE)
    
    Models can be initialized with hardcoded paths, Pydantic settings, or YAML configuration.
    """

    def __init__(self,
                 weights_config: Optional[Union[ModelWeightsConfig, str]] = None,
                 device: Optional[str] = None):
        """
        Initialize the ML Manager with all models.
        
        Args:
            weights_config: Can be one of:
                - ModelWeightsConfig instance: Use provided configuration directly
                - str: Path to YAML configuration file (will be loaded automatically)
                - None: Use default configuration with hardcoded paths
            device: Device to run models on ('cuda', 'cpu', or None for auto)
            
        Example:
            # Using default configuration
            ml_manager = MLManager()
            
            # Using YAML file
            ml_manager = MLManager(weights_config="config/models.yaml")
            
            # Using ModelWeightsConfig instance
            config = ModelWeightsConfig(ball_detection="custom/ball.pt")
            ml_manager = MLManager(weights_config=config)
        """

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"ML Manager initialized on device: {self.device}")

        # Initialize weights configuration
        self.weights_config = self._initialize_weights_config(weights_config)

        # Initialize all models
        self._initialize_models()

    @staticmethod
    def _initialize_weights_config(weights_config: Optional[Union[ModelWeightsConfig, str]]) -> ModelWeightsConfig:
        """
        Initialize weights configuration from various sources.
        
        Args:
            weights_config: ModelWeightsConfig instance, YAML file path, or None
            
        Returns:
            Initialized ModelWeightsConfig instance
        """
        if weights_config is None:
            logger.info("Using default weights configuration")
            return ModelWeightsConfig()

        elif isinstance(weights_config, ModelWeightsConfig):
            logger.info("Using provided ModelWeightsConfig instance")
            return weights_config

        elif isinstance(weights_config, str):
            logger.info(f"Loading weights configuration from YAML: {weights_config}")
            try:
                return ModelWeightsConfig.from_yaml(weights_config)
            except Exception as e:
                logger.warning(f"Failed to load YAML configuration: {e}")
                logger.info("Falling back to default configuration")
                return ModelWeightsConfig()

        else:
            raise ValueError(f"Unsupported weights_config type: {type(weights_config)}. "
                             f"Expected ModelWeightsConfig instance, YAML file path string, or None.")

    def _initialize_models(self):
        """Initialize all deep learning models."""
        logger.info("Initializing deep learning models...")

        # Initialize YOLO models
        # Auto-download weights if they don't exist
        self._auto_download_weights()
        self._init_action_detection()
        self._init_ball_segmentation()
        self._init_court_segmentation()
        self._init_player_detection()

        # Initialize tracking module
        self._init_tracking()

        # Initialize VideoMAE model
        self._init_game_state_classification()

        # Initialize visualization module
        self._init_visualization()

        logger.success("All models initialized successfully!")

    def _auto_download_weights(self):
        """Automatically download missing model weights."""
        try:
            # Check if we need to download weights
            missing_weights = self.weights_config.check_weights_availability()
            missing_models = [name for name, available in missing_weights.items() if not available]

            if missing_models:
                logger.info(f"Missing model weights detected: {missing_models}")
                logger.info("Downloading missing weights from Google Drive...")

                # Download missing weights
                success = self.weights_config.download_missing_weights(force=False)

                # Log results
                if success:
                    logger.success("Successfully downloaded all model weights")
                else:
                    logger.warning("Failed to download some model weights")

            else:
                logger.info("All model weights are available locally")

        except Exception as e:
            logger.warning(f"Auto-download failed: {e}")
            logger.info("Please download weights manually using the download_weights.py script")

    def _init_action_detection(self):
        """Initialize action detection model."""
        try:
            if self.weights_config.action_detection:
                # Convert to absolute path
                model_path = Path.cwd() / self.weights_config.action_detection
                self.action_detector = ActionDetectorModule(
                    model_path=str(model_path),
                    device=self.device
                )
                logger.success(f"Action detection model loaded: {model_path}")
            else:
                self.action_detector = None
                logger.warning("Action detection model not configured")
        except Exception as e:
            self.action_detector = None
            logger.error(f"Failed to load action detection model: {e}")

    def _init_ball_segmentation(self):
        """Initialize ball segmentation model."""
        try:
            if self.weights_config.ball_detection:
                # Convert to absolute path
                model_path = Path.cwd() / self.weights_config.ball_detection
                self.ball_detector = BallDetectorModule(
                    model_path=str(model_path),
                    device=self.device
                )
                logger.success(f"Ball segmentation model loaded: {model_path}")
            else:
                self.ball_detector = None
                logger.warning("Ball segmentation model not configured")
        except Exception as e:
            self.ball_detector = None
            logger.error(f"Failed to load ball segmentation model: {e}")

    def _init_court_segmentation(self):
        """Initialize court segmentation model."""
        try:
            if self.weights_config.court_detection:
                # Convert to absolute path
                model_path = Path.cwd() / self.weights_config.court_detection
                self.court_detector = CourtSegmentationModule(
                    model_path=str(model_path),
                    device=self.device
                )
                logger.success(f"Court segmentation model loaded: {model_path}")
            else:
                self.court_detector = None
                logger.warning("Court segmentation model not configured")
        except Exception as e:
            self.court_detector = None
            logger.error(f"Failed to load court segmentation model: {e}")

    def _init_player_detection(self):
        """Initialize player keypoint detection model."""
        try:
            if self.weights_config.player_detection:
                # Use custom player keypoint detection model with absolute path
                model_path = Path.cwd() / self.weights_config.player_detection
                self.player_detector = PlayerDetectorModule(
                    model_path=str(model_path),
                    device=self.device
                )
                logger.success(f"Custom player keypoint detection model loaded: {model_path}")
            else:
                # Use default YOLO pose estimation model
                self.player_detector = PlayerDetectorModule(
                    model_path="yolo11n-pose.pt",
                    device=self.device
                )
                logger.success("Default YOLO pose estimation model loaded for player keypoint detection")
        except Exception as e:
            self.player_detector = None
            logger.error(f"Failed to load player keypoint detection model: {e}")

    def _init_tracking(self):
        """Initialize tracking module."""
        try:
            from .core.tracking_module import TrackingConfig

            # Initialize with default configuration
            self.tracker = VolleyballTracker(
                config=TrackingConfig()
            )
            logger.success("Tracking module initialized successfully")
        except Exception as e:
            self.tracker = None
            logger.error(f"Failed to initialize tracking module: {e}")

    def _init_visualization(self):
        """Initialize visualization module."""
        try:
            self.visualizer = VolleyballVisualizer()
            logger.success("Visualization module initialized successfully")
        except Exception as e:
            self.visualizer = None
            logger.error(f"Failed to initialize visualization module: {e}")

    def _init_game_state_classification(self):
        """Initialize game state classification model."""
        try:
            if self.weights_config.game_status:
                model_path = Path.cwd() / self.weights_config.game_status
                self.game_state_detector = GameStatusClassifierModule(
                    model_path=model_path.as_posix(),
                    device=self.device
                )
                logger.success(f"Game state classification model loaded: {model_path}")
            else:
                self.game_state_detector = None
                logger.warning("Game state classification model not configured")
        except Exception as e:
            self.game_state_detector = None
            logger.error(f"Failed to load game state classification model: {e}")

    def get_weights_config(self) -> ModelWeightsConfig:
        """
        Get the current weights configuration.
        
        Returns:
            Current ModelWeightsConfig instance
        """
        return self.weights_config

    def update_weights_config(self, new_config: Union[ModelWeightsConfig, str]):
        """
        Update weights configuration and reinitialize models.
        
        Args:
            new_config: New ModelWeightsConfig instance or YAML file path
        """
        logger.info("Updating weights configuration...")

        # Update configuration
        self.weights_config = self._initialize_weights_config(new_config)

        # Reinitialize models
        self._initialize_models()

        logger.success("Weights configuration updated and models reinitialized!")

    def save_weights_config_to_yaml(self, output_path: str):
        """
        Save current weights configuration to YAML file.
        
        Args:
            output_path: Path where to save the YAML file
        """
        try:
            self.weights_config.to_yaml(output_path)
            logger.success(f"Weights configuration saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save weights configuration: {e}")

    # Action Detection Methods
    def detect_actions(self,
                       frame: np.ndarray,
                       exclude: Optional[List[str]] = None,
                       conf_threshold: float = 0.25,
                       iou_threshold: float = 0.45) -> List[Detection]:
        """
        Detect volleyball actions in a frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            exclude: List of action types to exclude from detection
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            List of Detection objects with action detection results
            
        Raises:
            RuntimeError: If action detection model is not available
        """
        if self.action_detector is None:
            raise RuntimeError("Action detection model not available")

        # Use the new ActionDetector class
        detections = self.action_detector.detect_actions(frame, conf_threshold, iou_threshold)

        # Filter by excluded actions if specified
        if exclude:
            detections = self.action_detector.filter_by_action_type(detections,
                                                                    [action for action in
                                                                     self.action_detector.volleyball_actions if
                                                                     action not in exclude])

        return detections

    # Ball Detection Methods
    def detect_ball(self,
                    frame: np.ndarray,
                    conf_threshold: float = 0.25,
                    iou_threshold: float = 0.45) -> Detection:
        """
        Detect ball in a frame using segmentation.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            List of Detection objects with ball detection results
            
        Raises:
            RuntimeError: If ball detection model is not available
        """
        if self.ball_detector is None:
            raise RuntimeError("Ball detection model not available")

        # Use the new BallDetector class
        return self.ball_detector.detect_ball(frame, conf_threshold, iou_threshold)

    # Court Segmentation Methods
    def segment_court(self,
                      frame: np.ndarray,
                      conf_threshold: float = 0.25,
                      iou_threshold: float = 0.45) -> List[Detection]:
        """
        Segment volleyball court in a frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            List of Detection objects with court segmentation results
            
        Raises:
            RuntimeError: If court segmentation model is not available
        """
        if self.court_detector is None:
            raise RuntimeError("Court segmentation model is not available")

        # Use the new CourtSegmentation class
        return self.court_detector.segment_court(frame, conf_threshold, iou_threshold)

    # Player Detection Methods
    def detect_players(self,
                       frame: Union[str, np.ndarray],
                       conf_threshold: float = 0.25,
                       iou_threshold: float = 0.45) -> List[PlayerKeyPoints]:
        """
        Detect player keypoints in a single frame.
        
        Args:
            frame: Input frame as numpy array or file path (H, W, C)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            List of PlayerKeyPoints with player pose information
            
        Raises:
            RuntimeError: If player detection model is not available
        """
        if self.player_detector is None:
            raise RuntimeError("Player detection model not available")

        # Use the new PlayerModule detect method that returns List[PlayerKeyPoints]
        return self.player_detector.detect(frame, conf_threshold, iou_threshold)

    def detect_all(self,
                   frame: np.ndarray,
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45) -> Tuple[List[Detection], Detection, List[PlayerKeyPoints]]:
        """
        Detect all objects (actions, ball, players) in a single frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            Tuple containing:
            - List of action detections
            - List of ball detections  
            - List of player keypoints
            
        Note:
            Court segmentation is not included as it's typically done once per video, not per frame.
        """
        action_detections = []
        ball_detection = None
        player_keypoints = []

        # Detect actions
        if self.action_detector is not None:
            try:
                action_detections = self.detect_actions(frame, conf_threshold=conf_threshold,
                                                        iou_threshold=iou_threshold)
            except Exception as e:
                logger.warning(f"Action detection failed: {e}")

        # Detect ball
        if self.ball_detector is not None:
            try:
                ball_detection = self.detect_ball(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            except Exception as e:
                logger.warning(f"Ball detection failed: {e}")

        # Detect players
        if self.player_detector is not None:
            try:
                player_keypoints = self.detect_players(frame, conf_threshold=conf_threshold,
                                                       iou_threshold=iou_threshold)
            except Exception as e:
                logger.warning(f"Player detection failed: {e}")

        return action_detections, ball_detection, player_keypoints

    # Tracking Methods
    def track_objects(self,
                      detections: List[Any],
                      frame_number: int) -> List[Dict[str, Any]]:
        """
        Track objects across frames.
        
        Args:
            detections: List of Norfair Detection objects
            frame_number: Current frame number
            
        Returns:
            List of tracked objects with trajectory information
        """
        if self.tracker is None:
            raise RuntimeError("Tracking module not available")

        return self.tracker.update(detections, frame_number)

    def get_tracking_stats(self) -> Dict[str, Any]:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking statistics
        """
        if self.tracker is None:
            return {}

        return self.tracker.get_tracking_stats()

    def get_ball_trajectory(self, track_id: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        Get ball trajectory for analysis.
        
        Args:
            track_id: Specific track ID, or None for most recent
            
        Returns:
            List of trajectory points (x, y)
        """
        if self.tracker is None:
            return []

        return self.tracker.get_ball_trajectory(track_id)

    def get_player_tracks(self) -> Dict[int, Any]:
        """
        Get all currently tracked players.
        
        Returns:
            Dictionary of player tracks
        """
        if self.tracker is None:
            return {}

        return self.tracker.get_player_tracks()

    def get_ball_tracks(self) -> Dict[int, Any]:
        """
        Get all currently tracked balls.
        
        Returns:
            Dictionary of ball tracks
        """
        if self.tracker is None:
            return {}

        return self.tracker.get_ball_tracks()

    # Visualization Methods
    def visualize_frame(self,
                        frame: np.ndarray,
                        detections: List[Dict[str, Any]] = None,
                        tracked_objects: List[Dict[str, Any]] = None,
                        game_state: str = "",
                        frame_info: str = "") -> np.ndarray:
        """
        Visualize frame with detections, tracking, and game state.
        
        Args:
            frame: Input frame
            detections: List of detection objects
            tracked_objects: List of tracked objects
            game_state: Current game state
            frame_info: Additional frame information
            
        Returns:
            Frame with visualization overlays
        """
        if self.visualizer is None:
            return frame

        result_frame = frame.copy()

        # Draw detections
        if detections:
            result_frame = self.visualizer.draw_detections(result_frame, detections)

        # Draw tracking
        if tracked_objects:
            result_frame = self.visualizer.draw_tracking(result_frame, tracked_objects)

        # Draw game state
        if game_state:
            result_frame = self.visualizer.draw_game_state(result_frame, game_state, frame_info=frame_info)

        return result_frame

    def create_trajectory_plot(self,
                               trajectory: List[Tuple[float, float]],
                               title: str = "Ball Trajectory",
                               save_path: Optional[str] = None) -> Any:
        """
        Create a trajectory plot.
        
        Args:
            trajectory: List of trajectory points
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.visualizer is None:
            return None

        return self.visualizer.create_trajectory_plot(trajectory, title, save_path)

    def create_tracking_summary(self,
                                save_path: Optional[str] = None) -> Any:
        """
        Create a tracking summary visualization.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.visualizer is None:
            return None

        tracking_stats = self.get_tracking_stats()
        return self.visualizer.create_tracking_summary(tracking_stats, save_path)

    # Game State Classification Methods
    def classify_game_state(self,
                            frames: List[np.ndarray]) -> GameStateResult:
        """
        Classify the current game state using VideoMAE.
        
        Args:
            frames: List of consecutive frames for temporal analysis
            
        Returns:
            GameStateResult with classification results
            
        Raises:
            RuntimeError: If game state classification model is not available
        """
        if self.game_state_detector is None:
            raise RuntimeError("Game state classification model not available")

        return self.game_state_detector.classify(frames)

    # Utility Methods
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all models.
        
        Returns:
            Dictionary containing status information for each model
        """
        status = {}

        # Action detection
        status['action_detection'] = {
            'available': self.action_detector is not None,
            'labels': self.action_labels if hasattr(self, 'action_labels') else None
        }

        # Ball detection
        status['ball_detection'] = {
            'available': self.ball_detector is not None,
            'labels': self.ball_labels if hasattr(self, 'ball_labels') else None
        }

        # Court segmentation
        status['court_segmentation'] = {
            'available': self.court_detector is not None,
            'labels': self.court_labels if hasattr(self, 'court_labels') else None
        }

        # Player detection
        status['player_detection'] = {
            'available': self.player_detector is not None,
            'labels': self.player_labels if hasattr(self, 'player_labels') else None
        }

        # Game state classification
        status['game_state_classification'] = {
            'available': self.game_state_detector is not None
        }

        # Tracking
        status['tracking'] = {
            'available': self.tracker is not None,
            'stats': self.get_tracking_stats() if self.tracker is not None else {}
        }

        # Visualization
        status['visualization'] = {
            'available': self.visualizer is not None
        }

        return status

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        model_map = {
            'action_detection': self.action_detector,
            'ball_detection': self.ball_detector,
            'court_segmentation': self.court_detector,
            'player_detection': self.player_detector,
            'game_state_classification': self.game_state_detector,
            'tracking': self.tracker,
            'visualization': self.visualizer
        }

        return model_map.get(model_name) is not None

    def cleanup(self):
        """Clean up model resources."""
        # YOLO models are automatically cleaned up

        # Clean up tracker
        if hasattr(self, 'tracker') and self.tracker is not None:
            self.tracker.reset()

        # Clean up visualizer
        if hasattr(self, 'visualizer') and self.visualizer is not None:
            # Close any open matplotlib figures
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except ImportError:
                pass

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()

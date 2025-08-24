"""
Main ML Manager Class

This module provides the unified MLManager class that handles all deep learning models
for volleyball analytics without requiring external configuration files.
"""

import os
import torch
import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from pydantic import BaseSettings, Field
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from torchvision.transforms import Compose, Lambda, Resize
from pytorchvideo.transforms import Normalize, UniformTemporalSubsample

# Import model classes
from .models import (
    ActionDetector, BallDetector, CourtSegmentation, 
    PlayerModule, GameStatusClassifier
)
from .core import DetectionBatch, GameStateResult, VolleyballTracker
from .visualization import VolleyballVisualizer
from .enums import PlayerDetectionMode
from .settings import ModelWeightsConfig




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
                 device: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize the ML Manager with all models.
        
        Args:
            weights_config: ModelWeightsConfig instance, YAML file path, or None for defaults
            device: Device to run models on ('cuda', 'cpu', or None for auto)
            verbose: Whether to print initialization messages
        """
        self.verbose = verbose
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if self.verbose:
            print(f"ML Manager initialized on device: {self.device}")
        
        # Initialize weights configuration
        self.weights_config = self._initialize_weights_config(weights_config)
        
        # Initialize all models
        self._initialize_models()
    
    def _initialize_weights_config(self, weights_config: Optional[Union[ModelWeightsConfig, str]]) -> ModelWeightsConfig:
        """
        Initialize weights configuration from various sources.
        
        Args:
            weights_config: ModelWeightsConfig instance, YAML file path, or None
            
        Returns:
            Initialized ModelWeightsConfig instance
        """
        if weights_config is None:
            # Use default configuration
            if self.verbose:
                print("Using default weights configuration")
            return ModelWeightsConfig()
        
        elif isinstance(weights_config, ModelWeightsConfig):
            # Use provided configuration
            if self.verbose:
                print("Using provided weights configuration")
            return weights_config
        
        elif isinstance(weights_config, str):
            # Load from YAML file
            if self.verbose:
                print(f"Loading weights configuration from: {weights_config}")
            return self._load_config_from_yaml(weights_config)
        
        else:
            raise ValueError(f"Unsupported weights_config type: {type(weights_config)}")
    
    def _load_config_from_yaml(self, yaml_path: str) -> ModelWeightsConfig:
        """
        Load weights configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            ModelWeightsConfig instance loaded from YAML
        """
        try:
            yaml_path = Path(yaml_path)
            if not yaml_path.exists():
                raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
            
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if self.verbose:
                print(f"Loaded configuration from: {yaml_path}")
                print(f"Configuration data: {config_data}")
            
            # Create ModelWeightsConfig from YAML data
            return ModelWeightsConfig(**config_data)
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to load YAML configuration: {e}")
                print("Falling back to default configuration")
            return ModelWeightsConfig()
    
    def _initialize_models(self):
        """Initialize all deep learning models."""
        if self.verbose:
            print("Initializing deep learning models...")
        
        # Initialize YOLO models
        #TODO: If yolo model weights don't exist, download it from google drive.
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
        
        if self.verbose:
            print("All models initialized successfully!")
    
    def _init_action_detection(self):
        """Initialize action detection model."""
        try:
            if self.weights_config.action_detection:
                self.action_detector = ActionDetector(
                    model_path=self.weights_config.action_detection,
                    device=self.device,
                    verbose=self.verbose
                )
                if self.verbose:
                    print(f"✓ Action detection model loaded: {self.weights_config.action_detection}")
            else:
                self.action_detector = None
                if self.verbose:
                    print("⚠ Action detection model not configured")
        except Exception as e:
            self.action_detector = None
            if self.verbose:
                print(f"✗ Failed to load action detection model: {e}")
    
    def _init_ball_segmentation(self):
        """Initialize ball segmentation model."""
        try:
            if self.weights_config.ball_detection:
                self.ball_detector = BallDetector(
                    model_path=self.weights_config.ball_detection,
                    device=self.device,
                    verbose=self.verbose
                )
                if self.verbose:
                    print(f"✓ Ball segmentation model loaded: {self.weights_config.ball_detection}")
            else:
                self.ball_detector = None
                if self.verbose:
                    print("⚠ Ball segmentation model not configured")
        except Exception as e:
            self.ball_detector = None
            if self.verbose:
                print(f"✗ Failed to load ball segmentation model: {e}")
    
    def _init_court_segmentation(self):
        """Initialize court segmentation model."""
        try:
            if self.weights_config.court_detection:
                self.court_detector = CourtSegmentation(
                    model_path=self.weights_config.court_detection,
                    device=self.device,
                    verbose=self.verbose
                )
                if self.verbose:
                    print(f"✓ Court segmentation model loaded: {self.weights_config.court_detection}")
            else:
                self.court_detector = None
                if self.verbose:
                    print("⚠ Court segmentation model not configured")
        except Exception as e:
            self.court_detector = None
            if self.verbose:
                print(f"✗ Failed to load court segmentation model: {e}")
    
    def _init_player_detection(self):
        """Initialize player detection model."""
        try:
            if self.weights_config.player_detection:
                # Use custom player detection model
                self.player_detector = PlayerModule(
                    model_path=self.weights_config.player_detection,
                    mode=PlayerDetectionMode.POSE,  # Default to pose estimation
                    device=self.device,
                    verbose=self.verbose
                )
                if self.verbose:
                    print(f"✓ Custom player detection model loaded: {self.weights_config.player_detection}")
            else:
                # Use default YOLO pose estimation model
                self.player_detector = PlayerModule(
                    model_path="yolov8n-pose.pt",
                    mode=PlayerDetectionMode.POSE,
                    device=self.device,
                    verbose=self.verbose
                )
                if self.verbose:
                    print("✓ Default YOLO pose estimation model loaded for player detection")
        except Exception as e:
            self.player_detector = None
            if self.verbose:
                print(f"✗ Failed to load player detection model: {e}")
    
    def _init_tracking(self):
        """Initialize tracking module."""
        try:
            from .models.tracking_module import TrackingConfig, TrackerType
            
            # Initialize with default configuration
            self.tracker = VolleyballTracker(
                config=TrackingConfig(tracker_type=TrackerType.NORFAIR),
                verbose=self.verbose
            )
            if self.verbose:
                print("✓ Tracking module initialized successfully")
        except Exception as e:
            self.tracker = None
            if self.verbose:
                print(f"✗ Failed to initialize tracking module: {e}")
    
    def _init_visualization(self):
        """Initialize visualization module."""
        try:
            self.visualizer = VolleyballVisualizer(verbose=self.verbose)
            if self.verbose:
                print("✓ Visualization module initialized successfully")
        except Exception as e:
            self.visualizer = None
            if self.verbose:
                print(f"✗ Failed to initialize visualization module: {e}")
    
    def _init_game_state_classification(self):
        """Initialize game state classification model."""
        try:
            if self.weights_config.game_status:
                self.game_state_detector = GameStatusClassifier(
                    model_path=self.weights_config.game_status,
                    device=self.device,
                    verbose=self.verbose
                )
                if self.verbose:
                    print(f"✓ Game state classification model loaded: {self.weights_config.game_status}")
            else:
                self.game_state_detector = None
                if self.verbose:
                    print("⚠ Game state classification model not configured")
        except Exception as e:
            self.game_state_detector = None
            if self.verbose:
                print(f"✗ Failed to load game state classification model: {e}")
    
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
        if self.verbose:
            print("Updating weights configuration...")
        
        # Update configuration
        self.weights_config = self._initialize_weights_config(new_config)
        
        # Reinitialize models
        self._initialize_models()
        
        if self.verbose:
            print("Weights configuration updated and models reinitialized!")
    
    def save_weights_config_to_yaml(self, output_path: str):
        """
        Save current weights configuration to YAML file.
        
        Args:
            output_path: Path where to save the YAML file
        """
        try:
            config_dict = self.weights_config.dict()
            
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            if self.verbose:
                print(f"✓ Weights configuration saved to: {output_path}")
                
        except Exception as e:
            if self.verbose:
                print(f"✗ Failed to save weights configuration: {e}")
    
    # Action Detection Methods
    def detect_actions(self, 
                      frame: np.ndarray, 
                      exclude: Optional[List[str]] = None,
                      conf_threshold: float = 0.25,
                      iou_threshold: float = 0.45) -> DetectionBatch:
        """
        Detect volleyball actions in a frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            exclude: List of action types to exclude from detection
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            DetectionBatch with action detection results
            
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
                [action for action in self.action_detector.volleyball_actions if action not in exclude])
        
        return detections
    
    def detect_actions_batch(self, 
                            frames: List[np.ndarray], 
                            exclude: Optional[List[str]] = None,
                            conf_threshold: float = 0.25,
                            iou_threshold: float = 0.45) -> List[Dict[str, List[Dict[str, Any]]]]:
        """
        Detect volleyball actions in multiple frames.
        
        Args:
            frames: List of input frames
            exclude: List of action types to exclude from detection
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            List of detection results for each frame
        """
        if self.action_detector is None:
            raise RuntimeError("Action detection model not available")
        
        if exclude is None:
            exclude = []
        
        # Filter classes to detect
        available_classes = [k for k, v in self.action_labels.items() 
                           if v not in exclude]
        
        # Run batch inference
        results = self.action_detector.detect_actions(
            frames, 
            conf=conf_threshold, 
            iou=iou_threshold,
            classes=available_classes,
            verbose=False
        )
        
        # Process results for each frame
        batch_detections = []
        
        for result in results:
            detections = {label: [] for label in self.action_labels.values()}
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    label = self.action_labels[class_id]
                    if label not in exclude:
                        detection = {
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class_id': int(class_id)
                        }
                        detections[label].append(detection)
            
            batch_detections.append(detections)
        
        return batch_detections
    
    # Ball Detection Methods
    def detect_ball(self, 
                   frame: np.ndarray,
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45) -> DetectionBatch:
        """
        Detect ball in a frame using segmentation.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            DetectionBatch with ball detection results
            
        Raises:
            RuntimeError: If ball detection model is not available
        """
        if self.ball_detector is None:
            raise RuntimeError("Ball detection model not available")
        
        # Use the new BallDetector class
        return self.ball_detector.detect_ball(frame, conf_threshold, iou_threshold)
    
    def detect_ball_batch(self, 
                          frames: List[np.ndarray],
                          conf_threshold: float = 0.25,
                          iou_threshold: float = 0.45) -> List[List[Dict[str, Any]]]:
        """
        Detect ball in multiple frames.
        
        Args:
            frames: List of input frames
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            List of ball detection results for each frame
        """
        if self.ball_detector is None:
            raise RuntimeError("Ball detection model not available")
        
        # Run batch inference
        results = self.ball_detector(
            frames, 
            conf=conf_threshold, 
            iou=iou_threshold,
            verbose=False
        )
        
        batch_detections = []
        
        for result in results:
            detections = []
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                # Get segmentation masks if available
                masks = None
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    detection = {
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'mask': masks[i].tolist() if masks is not None else None
                    }
                    detections.append(detection)
            
            batch_detections.append(detections)
        
        return batch_detections
    
    # Court Segmentation Methods
    def segment_court(self, 
                     frame: np.ndarray,
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45) -> DetectionBatch:
        """
        Segment volleyball court in a frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            DetectionBatch with court segmentation results
            
        Raises:
            RuntimeError: If court segmentation model is not available
        """
        if self.court_detector is None:
            raise RuntimeError("Court segmentation model is not available")
        
        # Use the new CourtSegmentation class
        return self.court_detector.segment_court(frame, conf_threshold, iou_threshold)
    
    # Player Detection Methods
    def detect_players(self, 
                      frame: np.ndarray,
                      conf_threshold: float = 0.25,
                      iou_threshold: float = 0.45) -> DetectionBatch:
        """
        Detect players in a frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            
        Returns:
            DetectionBatch with player detection results
            
        Raises:
            RuntimeError: If player detection model is not available
        """
        if self.player_detector is None:
            raise RuntimeError("Player detection model not available")
        
        # Use the new PlayerModule class
        return self.player_detector.detect_players(frame, conf_threshold, iou_threshold)
    
    # Tracking Methods
    def track_objects(self, 
                     frame: np.ndarray,
                     detections: List[Dict[str, Any]],
                     frame_number: int) -> List[Dict[str, Any]]:
        """
        Track objects across frames.
        
        Args:
            frame: Current frame
            detections: List of detection objects
            frame_number: Current frame number
            
        Returns:
            List of tracked objects with trajectory information
        """
        if self.tracker is None:
            raise RuntimeError("Tracking module not available")
        
        return self.tracker.update(detections, frame, frame_number)
    
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
        
        return self.game_state_detector.classify_frames(frames)

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
        # VideoMAE models need explicit cleanup
        if hasattr(self, 'game_state_detector') and self.game_state_detector is not None:
            self.game_state_detector.cleanup()
        
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


class GameStateDetector:
    """
    Game State Detection using VideoMAE
    
    This class handles the initialization and inference for the VideoMAE
    model used for game state classification.
    """
    
    def __init__(self, checkpoint_path: str):
        """
        Initialize the game state detector.
        
        Args:
            checkpoint_path: Path to the model checkpoint
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.feature_extractor = VideoMAEImageProcessor.from_pretrained(checkpoint_path)
        self.model = VideoMAEForVideoClassification.from_pretrained(checkpoint_path).to(self.device)
        
        # Get labels from model config
        self.labels = list(self.model.config.label2id.keys())
        
        # Create transforms
        sample_size = self.model.config.num_frames
        mean = self.feature_extractor.image_mean
        std = self.feature_extractor.image_std
        resize_to = 224
        
        self.transforms = Compose([
            UniformTemporalSubsample(sample_size),
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            Resize((resize_to, resize_to)),
        ])
        
        # State mappings
        self.label2state = {'service': 1, 'play': 2, 'no-play': 3}
        self.state2label = {1: 'service', 2: 'play', 3: 'no-play'}
    
    def predict(self, frames: List[np.ndarray]) -> str:
        """
        Predict game state from a sequence of frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            Predicted game state label
        """
        # Preprocess frames
        video_tensor = torch.tensor(np.array(frames).astype(frames[0].dtype))
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (num_channels, num_frames, height, width)
        video_tensor_pp = self.transforms(video_tensor)
        video_tensor_pp = video_tensor_pp.permute(1, 0, 2, 3)  # (num_frames, num_channels, height, width)
        video_tensor_pp = video_tensor_pp.unsqueeze(0)  # Add batch dimension
        
        # Run inference
        inputs = {"pixel_values": video_tensor_pp.to(self.device)}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get predicted class
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predicted_label = self.labels[predicted_class_id]
        
        return predicted_label
    
    def predict_with_confidence(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Predict game state with confidence scores.
        
        Args:
            frames: List of input frames
            
        Returns:
            Dictionary with predicted label and confidence scores
        """
        # Preprocess frames
        video_tensor = torch.tensor(np.array(frames).astype(frames[0].dtype))
        video_tensor = video_tensor.permute(3, 0, 1, 2)
        video_tensor_pp = self.transforms(video_tensor)
        video_tensor_pp = video_tensor_pp.permute(1, 0, 2, 3)
        video_tensor_pp = video_tensor_pp.unsqueeze(0)
        
        # Run inference
        inputs = {"pixel_values": video_tensor_pp.to(self.device)}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get softmax probabilities
        softmax_scores = torch.nn.functional.softmax(logits, dim=1).squeeze(0)
        
        # Get predicted class
        predicted_class_id = torch.argmax(softmax_scores, dim=0).item()
        predicted_label = self.labels[predicted_class_id]
        
        # Get confidence scores for all classes
        confidences = {
            self.labels[i]: float(softmax_scores[i]) 
            for i in range(len(self.labels))
        }
        
        return {
            'predicted_label': predicted_label,
            'predicted_class_id': predicted_class_id,
            'confidences': confidences,
            'max_confidence': float(softmax_scores[predicted_class_id])
        }
    
    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'feature_extractor'):
            del self.feature_extractor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

"""
Game state classification model for volleyball using VideoMAE.

This module provides specialized game state classification functionality using
VideoMAE models trained for volleyball game state recognition.
"""

import time
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
import cv2
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms._functional_video import normalize

from ..enums import GameState
from ..core.data_structures import GameStateResult
from ..utils.logger import logger


class GameStatusClassifierModule:
    """
    Specialized game state classifier for volleyball.

    This class provides game state classification using VideoMAE models,
    with volleyball-specific utilities for analyzing game sequences.
    """

    def __init__(self,
                 model_path: str,
                 device: Optional[str] = None,
                 num_frames: int = 16,
                 resize_to: int = 224,
                 mean: Tuple[float] = (0.485, 0.456, 0.406),
                 std: Tuple[float] = (0.229, 0.224, 0.225)):
        """
        Initialize game status classifier.

        Args:
            model_path: Path to VideoMAE model checkpoint
            device: Device to run inference on
            num_frames: Number of frames to use for classification
            resize_to: Spatial size to resize frames to (height, width)
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.model_path = model_path
        self.num_frames = num_frames
        self.resize_to = resize_to
        self.mean = mean
        self.std = std
        self.class_to_gamestate = {
            "play": GameState.PLAY,
            "no_play": GameState.NO_PLAY,
            "serve": GameState.SERVE,
        }
        logger.info(f"Initializing GameStatusClassifier with model: {model_path}")

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize the transform pipeline
        self._setup_transforms()

        # Load model (without the processor since we're handling preprocessing ourselves)
        self._load_model()

        # Common volleyball game states
        self.game_states = [
            "serve", "receive", "rally", "point_scored", "timeout",
            "game_over", "celebration", "substitution"
        ]

    def _setup_transforms(self):
        """Setup the video transformation pipeline."""
        self.vid_transforms = Compose([
            UniformTemporalSubsample(self.num_frames),
            Lambda(lambda x: x / 255.0),  # Scale to [0, 1]
            Lambda(lambda x: normalize(x, self.mean, self.std)),  # Normalize
            # Resize would go here if needed, but we're handling it separately per-frame
        ])
        logger.debug("Video transforms pipeline initialized")

    def _load_model(self):
        """Load VideoMAE model (without the processor)."""
        try:
            # Load model without the processor since we're handling preprocessing
            self.model = VideoMAEForVideoClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                ignore_mismatched_sizes=True  # In case preprocessing differs
            )
            self.model.to(self.device)
            self.model.eval()

            logger.success(f"Loaded VideoMAE model from {self.model_path} on device {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load VideoMAE model: {e}")

    def _preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocess individual frames: convert BGR to RGB and resize.

        Args:
            frames: List of frames as numpy arrays in BGR format

        Returns:
            List of preprocessed frames in RGB format
        """
        processed_frames = []
        for frame in frames:
            if frame is not None:
                # Convert BGR to RGB (crucial for correct color interpretation)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to the expected spatial dimensions
                frame_resized = cv2.resize(frame_rgb, (self.resize_to, self.resize_to))
                processed_frames.append(frame_resized)
            else:
                logger.warning("[ERR] Frame is None!")
        return processed_frames

    def _preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Full preprocessing pipeline for VideoMAE model input using the custom transform pipeline.

        Args:
            frames: List of frames as numpy arrays (BGR format)

        Returns:
            Preprocessed tensor ready for model inference
        """
        if not frames:
            raise ValueError("No frames provided for preprocessing")

        logger.debug(f"Preprocessing {len(frames)} frames for VideoMAE")

        # Step 1: Preprocess individual frames (color conversion + resize)
        processed_frames = self._preprocess_frames(frames)

        # Step 2: Convert to tensor and rearrange dimensions
        # Stack frames and convert to tensor: [T, H, W, C] -> [T, C, H, W]
        video_tensor = torch.tensor(np.stack(processed_frames)).float()
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]

        # Add batch dimension: [T, C, H, W] -> [1, T, C, H, W]
        video_tensor = video_tensor.unsqueeze(0)

        # Step 3: Apply the transform pipeline (UniformTemporalSubsample + Normalization)
        processed_video = self.vid_transforms(video_tensor)

        # The model expects [batch_size, num_channels, num_frames, height, width]
        # Our processed_video is [1, C, T, H, W], but the model might expect [1, T, C, H, W]
        # Check what the specific model expects and permute if necessary
        # For VideoMAE, it typically expects [batch_size, num_frames, num_channels, height, width]
        processed_video = processed_video.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W] -> [1, T, C, H, W]

        logger.debug(f"Final tensor shape: {processed_video.shape}")

        return processed_video

    def _infer(self, preprocessed_input: torch.Tensor) -> Tuple[GameState, float]:
        """
        Run inference on preprocessed inputs.

        Args:
            preprocessed_input: Preprocessed tensor from _preprocess method

        Returns:
            Tuple of (predicted_game_state, confidence)
        """
        logger.debug("Running VideoMAE inference")

        # Move input to device
        preprocessed_input = preprocessed_input.to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(pixel_values=preprocessed_input)
            logits = outputs.logits

        # Get predictions
        probs = torch.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class_id].item()

        # Get class label
        predicted_class = self.model.config.id2label[predicted_class_id]

        logger.debug(f"Predicted class: {predicted_class} with confidence: {confidence:.4f}")

        # Return mapped GameState or UNKNOWN if not found
        game_state = self.class_to_gamestate.get(predicted_class.lower(), GameState.UNKNOWN)
        return game_state, confidence

    def classify(self, frames: List[np.ndarray]) -> GameStateResult:
        """
        Classify game state from a list of frames (combines preprocess and inference).

        Args:
            frames: List of frames as numpy arrays (BGR format from OpenCV)

        Returns:
            GameStateResult with predicted class and confidence
        """
        if not frames:
            logger.warning(f"No frames for video classification.")
            return GameStateResult(predicted_class=GameState.UNKNOWN, confidence=0.0)

        try:
            # Preprocess frames using our custom pipeline
            preprocessed_input = self._preprocess(frames)

            # Run inference
            game_state, confidence = self._infer(preprocessed_input)

            return GameStateResult(predicted_class=game_state, confidence=confidence)

        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return GameStateResult(predicted_class=GameState.UNKNOWN, confidence=0.0)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self.model_path,
            'model_type': 'VideoMAE',
            'device': self.device,
            'num_frames': self.num_frames,
            'resize_to': self.resize_to,
            'mean': self.mean,
            'std': self.std,
            'available_classes': list(self.model.config.id2label.values()) if hasattr(self.model, 'config') else [],
            'game_states': self.game_states
        }
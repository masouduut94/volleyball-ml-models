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
                 resize_to: int = 224):
        """
        Initialize game status classifier.

        Args:
            model_path: Path to VideoMAE model checkpoint
            device: Device to run inference on
            num_frames: Number of frames to use for classification
            resize_to: Spatial size to resize frames to (height, width)
        """
        self.model_path = model_path
        self.num_frames = num_frames
        self.resize_to = resize_to
        self.class_to_gamestate = {
            "play": GameState.PLAY,
            "no-play": GameState.NO_PLAY,
            "service": GameState.SERVICE,
        }
        logger.info(f"Initializing GameStatusClassifier with model: {model_path}")

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and processor
        self._load_model()

        # Common volleyball game states
        self.game_states = [
            "serve", "receive", "rally", "point_scored", "timeout",
            "game_over", "celebration", "substitution"
        ]

    def _load_model(self):
        """Load VideoMAE model and processor."""
        try:
            self.processor = VideoMAEImageProcessor.from_pretrained(self.model_path)
            self.model = VideoMAEForVideoClassification.from_pretrained(self.model_path, dtype=torch.float16)
            self.model.to(self.device)
            self.model.eval()

            logger.success(f"Loaded VideoMAE model from {self.model_path} on device {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load VideoMAE model: {e}")

    def _preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocess individual frames: convert BGR to RGB and resize.
        This prepares the frames for the VideoMAEImageProcessor.

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

    def _select_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply uniform temporal subsampling to select the desired number of frames.

        Args:
            frames: List of preprocessed frames

        Returns:
            List of selected frames for processing
        """
        if len(frames) <= self.num_frames:
            # If we have fewer frames than required, use all available frames
            return frames
        else:
            # Calculate step size to evenly distribute frame selection (UniformTemporalSubsample)
            step = (len(frames) - 1) / (self.num_frames - 1)
            selected_frames = []

            for i in range(self.num_frames):
                frame_idx = int(round(i * step))
                frame_idx = min(frame_idx, len(frames) - 1)  # Ensure we don't go out of bounds
                selected_frames.append(frames[frame_idx])

            return selected_frames

    def _preprocess(self, frames: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Full preprocessing pipeline for VideoMAE model input.

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

        # Step 2: Apply temporal subsampling
        selected_frames = self._select_frames(processed_frames)

        logger.debug(f"Selected {len(selected_frames)} frames after subsampling")

        # Step 3: Process with VideoMAE processor (handles normalization, etc.)
        # The processor expects a list of frames in RGB format
        inputs = self.processor(
            selected_frames,  # Pass the list of selected, preprocessed frames
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(self.device).half() for k, v in inputs.items()}

        return inputs

    def _infer(self, preprocessed_inputs: Dict[str, torch.Tensor]) -> Tuple[GameState, float]:
        """
        Run inference on preprocessed inputs.

        Args:
            preprocessed_inputs: Preprocessed tensor from preprocess_frames method

        Returns:
            Tuple of (predicted_game_state, confidence)
        """
        logger.debug("Running VideoMAE inference")

        # Run inference
        with torch.no_grad():
            outputs = self.model(**preprocessed_inputs)
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
            # Preprocess frames
            preprocessed_inputs = self._preprocess(frames)

            # Run inference
            game_state, confidence = self._infer(preprocessed_inputs)

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
            'available_classes': list(self.model.config.id2label.values()) if hasattr(self.model, 'config') else [],
            'game_states': self.game_states
        }
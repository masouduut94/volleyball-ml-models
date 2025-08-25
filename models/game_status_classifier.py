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

class GameStatusClassifier:
    """
    Specialized game state classifier for volleyball.
    
    This class provides game state classification using VideoMAE models,
    with volleyball-specific utilities for analyzing game sequences.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 num_frames: int = 16,
                 verbose: bool = False):
        """
        Initialize game status classifier.
        
        Args:
            model_path: Path to VideoMAE model checkpoint
            device: Device to run inference on
            frame_interval: Interval between frames to sample
            num_frames: Number of frames to use for classification
            verbose: Whether to print verbose output
        """
        self.model_path = model_path
        self.num_frames = num_frames
        self.verbose = verbose
        
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
            self.model = VideoMAEForVideoClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            if self.verbose:
                print(f"Loaded VideoMAE model from {self.model_path}")
                print(f"Model device: {self.device}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load VideoMAE model: {e}")
    
    def classify_frames(self, frames: List[np.ndarray]) -> GameState:
        """
        Classify game state from a list of frames.
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            GameState enum value representing the predicted game state
        """
        if not frames:
            return GameState.UNKNOWN
        
        # Calculate step size for linearly increasing frame selection
        if len(frames) <= self.num_frames:
            # If we have fewer frames than required, use all available frames
            selected_frames = frames
        else:
            # Calculate step size to evenly distribute frame selection
            step = (len(frames) - 1) / (self.num_frames - 1)
            selected_frames = []
            
            for i in range(self.num_frames):
                frame_idx = int(round(i * step))
                frame_idx = min(frame_idx, len(frames) - 1)  # Ensure we don't go out of bounds
                selected_frames.append(frames[frame_idx])
        
        if self.verbose:
            print(f"Selected {len(selected_frames)} frames from {len(frames)} input frames")
        
        try:
            # Convert frames to the format expected by VideoMAE
            # VideoMAE expects frames in RGB format with shape (num_frames, height, width, 3)
            processed_frames = []
            for frame in selected_frames:
                # Ensure frame is in RGB format
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Convert BGR to RGB if needed (OpenCV uses BGR)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    # Handle grayscale or other formats
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if len(frame.shape) == 2 else frame
                
                processed_frames.append(frame_rgb)
            
            # Stack frames into a single array
            video_tensor = np.stack(processed_frames)
            
            # Process with VideoMAE processor
            inputs = self.processor(
                list(video_tensor), 
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Get predictions
            probs = torch.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class_id].item()
            
            # Get class label
            predicted_class = self.model.config.id2label[predicted_class_id]
            
            if self.verbose:
                print(f"Predicted class: {predicted_class} with confidence: {confidence:.4f}")
            
            # Map the predicted class to GameState enum
            # This mapping should be adjusted based on your specific model's output classes
            class_to_gamestate = {
                "play": GameState.PLAY,
                "no_play": GameState.NO_PLAY,
                "serve": GameState.SERVE,
            }
            
            # Return mapped GameState or UNKNOWN if not found
            return class_to_gamestate.get(predicted_class.lower(), GameState.UNKNOWN)
            
        except Exception as e:
            if self.verbose:
                print(f"Error during classification: {e}")
            return GameState.UNKNOWN

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
            'available_classes': list(self.model.config.id2label.values()) if hasattr(self.model, 'config') else [],
            'game_states': self.game_states
        }

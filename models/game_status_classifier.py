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
from PIL import Image

from ..core.data_structures import GameStateResult


class GameStatusClassifier:
    """
    Specialized game state classifier for volleyball.
    
    This class provides game state classification using VideoMAE models,
    with volleyball-specific utilities for analyzing game sequences.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 frame_interval: int = 16,
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
        self.frame_interval = frame_interval
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
    
    def classify_game_state(self, 
                           video_path: str,
                           start_frame: int = 0,
                           end_frame: Optional[int] = None) -> GameStateResult:
        """
        Classify game state from video file.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame number
            end_frame: Ending frame number (None for end of video)
            
        Returns:
            GameStateResult with classification results
        """
        start_time = time.time()
        
        # Extract frames
        frames = self._extract_frames(video_path, start_frame, end_frame)
        
        if not frames:
            raise ValueError("No frames could be extracted from video")
        
        # Classify frames
        result = self._classify_frames(frames)
        
        processing_time = time.time() - start_time
        
        # Add processing time to result
        result.processing_time = processing_time
        
        return result
    
    def classify_frames(self, 
                       frames: List[np.ndarray],
                       **kwargs) -> GameStateResult:
        """
        Classify game state from list of frames.
        
        Args:
            frames: List of frames as numpy arrays
            **kwargs: Additional arguments for classification
            
        Returns:
            GameStateResult with classification results
        """
        start_time = time.time()
        
        result = self._classify_frames(frames, **kwargs)
        result.processing_time = time.time() - start_time
        
        return result
    
    def _extract_frames(self, 
                        video_path: str,
                        start_frame: int = 0,
                        end_frame: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame number
            end_frame: Ending frame number
            
        Returns:
            List of frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if end_frame is None:
                end_frame = total_frames
            
            # Validate frame range
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)
            
            if start_frame >= end_frame:
                raise ValueError("Start frame must be before end frame")
            
            # Calculate frame sampling
            frame_count = end_frame - start_frame
            if frame_count < self.num_frames:
                # If not enough frames, reduce interval
                self.frame_interval = max(1, frame_count // self.num_frames)
            
            frames = []
            frame_indices = []
            
            for i in range(start_frame, end_frame, self.frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    frame_indices.append(i)
                    
                    if len(frames) >= self.num_frames:
                        break
            
            if self.verbose:
                print(f"Extracted {len(frames)} frames from video")
                print(f"Frame indices: {frame_indices}")
            
            return frames
            
        finally:
            cap.release()
    
    def _classify_frames(self, 
                         frames: List[np.ndarray],
                         **kwargs) -> GameStateResult:
        """
        Classify game state from frames.
        
        Args:
            frames: List of frames as numpy arrays
            **kwargs: Additional arguments for classification
            
        Returns:
            GameStateResult with classification results
        """
        if len(frames) < self.num_frames:
            # Pad frames if we don't have enough
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            while len(frames) < self.num_frames:
                frames.append(last_frame.copy())
        
        # Convert frames to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Process frames with VideoMAE processor
        inputs = self.processor(
            pil_frames, 
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get predictions
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0, predicted_class_id].item()
        
        # Get class name
        predicted_class = self.model.config.id2label.get(
            predicted_class_id, f"class_{predicted_class_id}"
        )
        
        # Get all probabilities
        all_probabilities = {}
        for i, prob in enumerate(probabilities[0]):
            class_name = self.model.config.id2label.get(i, f"class_{i}")
            all_probabilities[class_name] = prob.item()
        
        return GameStateResult(
            predicted_class=predicted_class,
            confidence=confidence,
            class_id=predicted_class_id,
            all_probabilities=all_probabilities
        )
    
    def classify_video_sequence(self, 
                               video_path: str,
                               segment_duration: float = 2.0,
                               overlap: float = 0.5) -> List[GameStateResult]:
        """
        Classify game state for overlapping video segments.
        
        Args:
            video_path: Path to video file
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments (0.0 to 1.0)
            
        Returns:
            List of GameStateResult for each segment
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = total_frames / fps
            
            # Calculate segment parameters
            frames_per_segment = int(segment_duration * fps)
            overlap_frames = int(frames_per_segment * overlap)
            step_frames = frames_per_segment - overlap_frames
            
            results = []
            current_frame = 0
            
            while current_frame < total_frames:
                end_frame = min(current_frame + frames_per_segment, total_frames)
                
                # Extract frames for this segment
                frames = self._extract_frames_from_range(
                    cap, current_frame, end_frame
                )
                
                if frames:
                    # Classify segment
                    result = self._classify_frames(frames)
                    result.timestamp = current_frame / fps
                    results.append(result)
                
                current_frame += step_frames
            
            return results
            
        finally:
            cap.release()
    
    def _extract_frames_from_range(self, 
                                   cap: cv2.VideoCapture,
                                   start_frame: int,
                                   end_frame: int) -> List[np.ndarray]:
        """
        Extract frames from a specific range using an open video capture.
        
        Args:
            cap: Open video capture object
            start_frame: Starting frame number
            end_frame: Ending frame number
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        
        for i in range(start_frame, end_frame, self.frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if len(frames) >= self.num_frames:
                    break
        
        return frames
    
    def get_game_state_timeline(self, 
                                video_path: str,
                                window_size: float = 3.0,
                                step_size: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get game state timeline with sliding window analysis.
        
        Args:
            video_path: Path to video file
            window_size: Size of analysis window in seconds
            step_size: Step size between windows in seconds
            
        Returns:
            List of timeline entries with timestamps and classifications
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = total_frames / fps
            
            # Calculate frame parameters
            window_frames = int(window_size * fps)
            step_frames = int(step_size * fps)
            
            timeline = []
            current_frame = 0
            
            while current_frame + window_frames <= total_frames:
                end_frame = current_frame + window_frames
                
                # Extract frames for this window
                frames = self._extract_frames_from_range(
                    cap, current_frame, end_frame
                )
                
                if frames:
                    # Classify window
                    result = self._classify_frames(frames)
                    
                    timeline_entry = {
                        'start_time': current_frame / fps,
                        'end_time': end_frame / fps,
                        'duration': window_size,
                        'predicted_class': result.predicted_class,
                        'confidence': result.confidence,
                        'class_id': result.class_id,
                        'all_probabilities': result.all_probabilities
                    }
                    
                    timeline.append(timeline_entry)
                
                current_frame += step_frames
            
            return timeline
            
        finally:
            cap.release()
    
    def analyze_game_flow(self, 
                          timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze game flow from timeline data.
        
        Args:
            timeline: List of timeline entries from get_game_state_timeline
            
        Returns:
            Analysis results including state transitions and patterns
        """
        if not timeline:
            return {}
        
        # Count state occurrences
        state_counts = {}
        state_durations = {}
        transitions = {}
        
        for entry in timeline:
            state = entry['predicted_class']
            duration = entry['duration']
            confidence = entry['confidence']
            
            # Count states
            state_counts[state] = state_counts.get(state, 0) + 1
            
            # Accumulate durations
            if state not in state_durations:
                state_durations[state] = []
            state_durations[state].append(duration)
            
            # Track transitions
            if len(timeline) > 1:
                for i in range(len(timeline) - 1):
                    current_state = timeline[i]['predicted_class']
                    next_state = timeline[i + 1]['predicted_class']
                    transition = f"{current_state}->{next_state}"
                    transitions[transition] = transitions.get(transition, 0) + 1
        
        # Calculate average durations
        avg_durations = {}
        for state, durations in state_durations.items():
            avg_durations[state] = float(np.mean(durations))
        
        # Find dominant states
        dominant_states = sorted(
            state_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        analysis = {
            'total_segments': len(timeline),
            'state_counts': state_counts,
            'average_durations': avg_durations,
            'dominant_states': dominant_states,
            'state_transitions': transitions,
            'most_common_transition': max(transitions.items(), key=lambda x: x[1]) if transitions else None
        }
        
        return analysis
    
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
            'frame_interval': self.frame_interval,
            'num_frames': self.num_frames,
            'available_classes': list(self.model.config.id2label.values()) if hasattr(self.model, 'config') else [],
            'game_states': self.game_states
        }

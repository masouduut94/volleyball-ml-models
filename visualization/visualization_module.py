"""
Visualization module for volleyball analytics.

This module provides comprehensive visualization capabilities for:
- Object tracking with trajectories
- Detection bounding boxes and labels
- Game state information
- Player pose estimation
- Ball trajectory analysis
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass

from ..core.tracking_module import TrackedObject
from ..core.data_structures import Detection


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    # Colors
    ball_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    player_color: Tuple[int, int, int] = (255, 0, 0)  # Blue
    action_colors: Dict[str, Tuple[int, int, int]] = None
    court_color: Tuple[int, int, int] = (128, 128, 128)  # Gray
    
    # Drawing settings
    line_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    trajectory_length: int = 30
    show_trajectories: bool = True
    show_confidence: bool = True
    show_track_ids: bool = True
    
    # Game state visualization
    game_state_font_scale: float = 1.5
    game_state_thickness: int = 3
    show_frame_info: bool = True
    
    def __post_init__(self):
        if self.action_colors is None:
            self.action_colors = {
                'spike': (255, 0, 255),      # Magenta
                'set': (255, 255, 0),        # Cyan
                'receive': (128, 0, 128),    # Purple
                'block': (0, 255, 255),      # Yellow
                'serve': (255, 165, 0),      # Orange
                'unknown': (128, 128, 128)   # Gray
            }


class VolleyballVisualizer:
    """
    Comprehensive visualizer for volleyball analytics.
    
    Provides methods for visualizing:
    - Object tracking with trajectories
    - Detection results
    - Game state information
    - Player poses
    - Ball trajectories
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize volleyball visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Initialize color maps
        self._init_color_maps()
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def _init_color_maps(self):
        """Initialize color maps for different object types."""
        # Generate distinct colors for track IDs
        self.track_colors = {}
        np.random.seed(42)  # For consistent colors
        
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get a consistent color for a track ID."""
        if track_id not in self.track_colors:
            # Generate a new color
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            self.track_colors[track_id] = color
        return self.track_colors[track_id]
    
    def draw_detections(self, 
                       frame: np.ndarray,
                       detections: List[Detection],
                       show_labels: bool = True) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detection objects
            show_labels: Whether to show labels
            
        Returns:
            Frame with detections drawn
        """
        result_frame = frame.copy()
        
        for detection in detections:
            bbox = detection.bbox
            class_name = detection.class_name
            confidence = detection.confidence
            
            # Get color based on class
            if class_name == 'ball':
                color = self.config.ball_color
            elif class_name == 'person':
                color = self.config.player_color
            else:
                color = self.config.action_colors.get(class_name, self.config.action_colors['unknown'])
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, self.config.line_thickness)
            
            # Draw label
            if show_labels:
                label = f"{class_name}"
                if self.config.show_confidence:
                    label += f": {confidence:.2f}"
                
                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, self.font, self.config.font_scale, self.config.font_thickness
                )
                
                # Draw label background
                cv2.rectangle(result_frame, 
                            (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1),
                            color, -1)
                
                # Draw label text
                cv2.putText(result_frame, label, (x1, y1 - baseline - 5),
                           self.font, self.config.font_scale, (255, 255, 255), 
                           self.config.font_thickness)
        
        return result_frame
    
    def draw_tracking(self, 
                     frame: np.ndarray,
                     tracked_objects: List[TrackedObject],
                     show_trajectories: bool = None) -> np.ndarray:
        """
        Draw tracking results on frame.
        
        Args:
            frame: Input frame
            tracked_objects: List of tracked objects
            show_trajectories: Whether to show trajectories (overrides config)
            
        Returns:
            Frame with tracking visualization
        """
        if show_trajectories is None:
            show_trajectories = self.config.show_trajectories
            
        result_frame = frame.copy()
        
        for tracked_obj in tracked_objects:
            if not tracked_obj.is_active:
                continue
                
            bbox = tracked_obj.bbox
            track_id = tracked_obj.track_id
            class_name = tracked_obj.class_name
            
            # Get color for this track
            color = self.get_track_color(track_id)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, self.config.line_thickness)
            
            # Draw track ID
            if self.config.show_track_ids:
                track_label = f"ID: {track_id}"
                cv2.putText(result_frame, track_label, (x1, y1 - 10),
                           self.font, 0.5, color, 1)
            
            # Draw trajectory
            if show_trajectories and len(tracked_obj.trajectory) > 1:
                trajectory_points = tracked_obj.trajectory[-self.config.trajectory_length:]
                
                # Convert to integer points
                points = np.array([(int(x), int(y)) for x, y in trajectory_points], dtype=np.int32)
                
                # Draw trajectory line
                if len(points) > 1:
                    cv2.polylines(result_frame, [points], False, color, 2)
                    
                    # Draw trajectory points
                    for point in points[-5:]:  # Last 5 points
                        cv2.circle(result_frame, tuple(point), 3, color, -1)
        
        return result_frame
    
    def draw_game_state(self, 
                       frame: np.ndarray,
                       game_state: str,
                       confidence: float = 1.0,
                       frame_info: str = "") -> np.ndarray:
        """
        Draw game state information on frame.
        
        Args:
            frame: Input frame
            game_state: Current game state
            confidence: Confidence in the prediction
            frame_info: Additional frame information
            
        Returns:
            Frame with game state information
        """
        result_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Game state color mapping
        state_colors = {
            'service': (0, 255, 0),      # Green
            'play': (255, 255, 0),       # Yellow
            'no-play': (0, 0, 255),      # Red
            'unknown': (255, 255, 255)   # White
        }
        
        color = state_colors.get(game_state.lower(), state_colors['unknown'])
        
        # Draw game state
        state_text = f"Game State: {game_state.upper()}"
        if self.config.show_confidence:
            state_text += f" ({confidence:.2f})"
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            state_text, self.font, self.config.game_state_font_scale, 
            self.config.game_state_thickness
        )
        
        # Position at top-left
        x, y = 20, 50
        
        # Draw background rectangle
        cv2.rectangle(result_frame, 
                    (x - 10, y - text_height - baseline - 10),
                    (x + text_width + 10, y + baseline + 10),
                    (0, 0, 0), -1)
        
        # Draw game state text
        cv2.putText(result_frame, state_text, (x, y),
                   self.font, self.config.game_state_font_scale, color,
                   self.config.game_state_thickness)
        
        # Draw frame information
        if self.config.show_frame_info and frame_info:
            frame_text = frame_info
            
            # Calculate frame text size
            (frame_text_width, frame_text_height), frame_baseline = cv2.getTextSize(
                frame_text, self.font, 0.6, 2
            )
            
            # Position at top-right
            frame_x = w - frame_text_width - 20
            frame_y = 50
            
            # Draw frame info background
            cv2.rectangle(result_frame, 
                        (frame_x - 10, frame_y - frame_text_height - frame_baseline - 10),
                        (frame_x + frame_text_width + 10, frame_y + frame_baseline + 10),
                        (0, 0, 0), -1)
            
            # Draw frame info text
            cv2.putText(result_frame, frame_text, (frame_x, frame_y),
                       self.font, 0.6, (255, 255, 255), 2)
        
        return result_frame
    
    def draw_ball_trajectory(self, 
                            frame: np.ndarray,
                            trajectory: List[Tuple[float, float]],
                            color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Draw ball trajectory on frame.
        
        Args:
            frame: Input frame
            trajectory: List of trajectory points (x, y)
            color: Trajectory color
            
        Returns:
            Frame with ball trajectory drawn
        """
        if not trajectory or len(trajectory) < 2:
            return frame
            
        result_frame = frame.copy()
        
        if color is None:
            color = self.config.ball_color
        
        # Convert to integer points
        points = np.array([(int(x), int(y)) for x, y in trajectory], dtype=np.int32)
        
        # Draw trajectory line
        cv2.polylines(result_frame, [points], False, color, 3)
        
        # Draw trajectory points
        for i, point in enumerate(points):
            # Make recent points larger
            radius = 5 if i >= len(points) - 5 else 3
            cv2.circle(result_frame, tuple(point), radius, color, -1)
            
            # Add velocity arrows for recent points
            if i > 0 and i >= len(points) - 3:
                prev_point = points[i-1]
                dx = point[0] - prev_point[0]
                dy = point[1] - prev_point[1]
                
                # Normalize and scale
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx = int(dx * 20 / length)
                    dy = int(dy * 20 / length)
                    
                    # Draw arrow
                    cv2.arrowedLine(result_frame, 
                                   (prev_point[0], prev_point[1]),
                                   (point[0] + dx, point[1] + dy),
                                   color, 2, tipLength=0.3)
        
        return result_frame
    
    @staticmethod
    def draw_player_poses(frame: np.ndarray,
                          pose_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw player pose keypoints on frame.
        
        Args:
            frame: Input frame
            pose_data: List of pose detection results
            
        Returns:
            Frame with pose keypoints drawn
        """
        result_frame = frame.copy()
        
        # Pose keypoint connections (simplified skeleton)
        skeleton = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Head to right hand
            (1, 5), (5, 6), (6, 7),           # Left shoulder to left hand
            (1, 8), (8, 9), (9, 10),          # Right hip to right foot
            (8, 11), (11, 12), (12, 13),      # Left hip to left foot
            (1, 8)                            # Shoulder to hip
        ]
        
        for pose in pose_data:
            keypoints = pose.get('keypoints', [])
            if not keypoints:
                continue
                
            # Draw keypoints
            for kp in keypoints:
                if len(kp) >= 2 and kp[2] > 0.5:  # Check confidence
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(result_frame, (x, y), 4, (0, 255, 255), -1)
            
            # Draw skeleton
            for connection in skeleton:
                if (connection[0] < len(keypoints) and connection[1] < len(keypoints) and
                    keypoints[connection[0]][2] > 0.5 and keypoints[connection[1]][2] > 0.5):
                    
                    pt1 = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
                    pt2 = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
                    
                    cv2.line(result_frame, pt1, pt2, (255, 255, 0), 2)
        
        return result_frame
    
    @staticmethod
    def create_trajectory_plot(trajectory: List[Tuple[float, float]],
                               title: str = "Ball Trajectory",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a matplotlib plot of ball trajectory.
        
        Args:
            trajectory: List of trajectory points (x, y)
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not trajectory:
            print("No trajectory data to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract x and y coordinates
        x_coords = [point[0] for point in trajectory]
        y_coords = [point[1] for point in trajectory]
        
        # Create trajectory plot
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        ax.scatter(x_coords, y_coords, c=range(len(trajectory)), 
                  cmap='viridis', s=50, alpha=0.8)
        
        # Add start and end markers
        if len(trajectory) > 0:
            ax.scatter(x_coords[0], y_coords[0], c='green', s=100, 
                      marker='o', label='Start', zorder=5)
            ax.scatter(x_coords[-1], y_coords[-1], c='red', s=100, 
                      marker='x', label='End', zorder=5)
        
        # Customize plot
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Invert y-axis for image coordinates
        ax.invert_yaxis()
        
        # Add colorbar for trajectory progression
        if len(trajectory) > 1:
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                      norm=plt.Normalize(0, len(trajectory)-1))
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Frame Number')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory plot saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def create_tracking_summary(tracking_stats: Dict[str, Any],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a summary visualization of tracking statistics.
        
        Args:
            tracking_stats: Dictionary of tracking statistics
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total tracks over time
        if 'total_tracks' in tracking_stats:
            ax1.bar(['Total Tracks'], [tracking_stats['total_tracks']], 
                   color='skyblue', alpha=0.7)
            ax1.set_title('Total Tracks')
            ax1.set_ylabel('Count')
        
        # Active vs inactive tracks
        if 'active_tracks' in tracking_stats and 'total_tracks' in tracking_stats:
            active = tracking_stats['active_tracks']
            inactive = tracking_stats['total_tracks'] - active
            ax2.pie([active, inactive], labels=['Active', 'Inactive'], 
                   autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            ax2.set_title('Track Status')
        
        # Ball vs player tracks
        if 'ball_tracks' in tracking_stats and 'player_tracks' in tracking_stats:
            track_types = ['Ball Tracks', 'Player Tracks']
            track_counts = [tracking_stats['ball_tracks'], tracking_stats['player_tracks']]
            ax3.bar(track_types, track_counts, color=['orange', 'blue'], alpha=0.7)
            ax3.set_title('Track Types')
            ax3.set_ylabel('Count')
        
        # Frame count
        if 'frame_count' in tracking_stats:
            ax4.text(0.5, 0.5, f'Frames Processed:\n{tracking_stats["frame_count"]}', 
                    ha='center', va='center', transform=ax4.transAxes, 
                    fontsize=14, fontweight='bold')
            ax4.set_title('Processing Progress')
            ax4.axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tracking summary saved to: {save_path}")
        
        return fig

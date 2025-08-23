"""
Example Usage of the Volleyball ML Manager

This file demonstrates how to use the MLManager class for volleyball analytics,
including the new Pydantic settings and YAML configuration features.
"""

import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

# Import the ML Manager
from .ml_manager import MLManager, ModelWeightsConfig


def basic_usage_example():
    """Basic usage example of the ML Manager with default configuration."""
    print("=== Basic ML Manager Usage Example ===\n")
    
    # Initialize the ML Manager with default configuration
    print("1. Initializing ML Manager with default configuration...")
    ml_manager = MLManager(verbose=True)
    
    # Check model status
    print("\n2. Checking Model Status...")
    status = ml_manager.get_model_status()
    
    for model_name, info in status.items():
        available = "✓ Available" if info['available'] else "✗ Not Available"
        print(f"  - {model_name}: {available}")
    
    # Create a dummy frame for demonstration
    print("\n3. Creating dummy frame for testing...")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"   Frame shape: {dummy_frame.shape}")
    
    # Test ball detection
    print("\n4. Testing Ball Detection...")
    try:
        ball_results = ml_manager.detect_ball(dummy_frame)
        print(f"   Ball detections: {len(ball_results)}")
        if ball_results:
            for i, detection in enumerate(ball_results):
                bbox = detection['bbox']
                conf = detection['confidence']
                print(f"     Ball {i+1}: bbox={bbox}, confidence={conf:.3f}")
    except RuntimeError as e:
        print(f"   Error: {e}")
    
    # Test action detection
    print("\n5. Testing Action Detection...")
    try:
        action_results = ml_manager.detect_actions(dummy_frame)
        total_actions = sum(len(detections) for detections in action_results.values())
        print(f"   Total actions detected: {total_actions}")
        
        for action_type, detections in action_results.items():
            if detections:
                print(f"     {action_type}: {len(detections)} detections")
    except RuntimeError as e:
        print(f"   Error: {e}")
    
    # Test player detection
    print("\n6. Testing Player Detection...")
    try:
        player_results = ml_manager.detect_players(dummy_frame)
        print(f"   Players detected: {len(player_results)}")
        if player_results:
            for i, detection in enumerate(player_results):
                bbox = detection['bbox']
                conf = detection['confidence']
                print(f"     Player {i+1}: bbox={bbox}, confidence={conf:.3f}")
    except RuntimeError as e:
        print(f"   Error: {e}")
    
    # Test court segmentation
    print("\n7. Testing Court Segmentation...")
    try:
        court_results = ml_manager.segment_court(dummy_frame)
        print(f"   Court segments: {len(court_results)}")
        if court_results:
            for i, result in enumerate(court_results):
                bbox = result['bbox']
                conf = result['confidence']
                print(f"     Court {i+1}: bbox={bbox}, confidence={conf:.3f}")
    except RuntimeError as e:
        print(f"   Error: {e}")
    
    # Test game state classification
    print("\n8. Testing Game State Classification...")
    try:
        # Create a sequence of dummy frames
        dummy_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                       for _ in range(16)]
        
        game_state = ml_manager.classify_game_state(dummy_frames)
        print(f"   Predicted game state: {game_state}")
        
        # Get confidence scores
        confidence_results = ml_manager.classify_game_state_with_confidence(dummy_frames)
        print(f"   Confidence: {confidence_results['max_confidence']:.3f}")
        print(f"   All confidences: {confidence_results['confidences']}")
        
    except RuntimeError as e:
        print(f"   Error: {e}")
    
    # Cleanup
    print("\n9. Cleaning up...")
    ml_manager.cleanup()
    print("   ML Manager cleaned up successfully!")
    
    print("\n=== Example Completed ===")


def pydantic_config_example():
    """Example using Pydantic configuration."""
    print("=== Pydantic Configuration Example ===\n")
    
    # Create custom configuration
    print("1. Creating custom Pydantic configuration...")
    custom_config = ModelWeightsConfig(
        ball_detection="custom_weights/ball_model.pt",
        action_detection="custom_weights/action_model.pt",
        game_status="custom_weights/game_state_checkpoint",
        court_detection="custom_weights/court_model.pt",
        player_detection=None  # Use default YOLO pose
    )
    
    print("   Configuration created:")
    print(f"     Ball detection: {custom_config.ball_detection}")
    print(f"     Action detection: {custom_config.action_detection}")
    print(f"     Game status: {custom_config.game_status}")
    print(f"     Court detection: {custom_config.court_detection}")
    print(f"     Player detection: {custom_config.player_detection}")
    
    # Initialize ML Manager with custom configuration
    print("\n2. Initializing ML Manager with custom configuration...")
    ml_manager = MLManager(weights_config=custom_config, verbose=True)
    
    # Get current configuration
    print("\n3. Getting current configuration...")
    current_config = ml_manager.get_weights_config()
    print(f"   Current ball detection path: {current_config.ball_detection}")
    
    # Cleanup
    ml_manager.cleanup()
    print("\n=== Pydantic Configuration Example Completed ===")


def yaml_config_example():
    """Example using YAML configuration file."""
    print("=== YAML Configuration Example ===\n")
    
    # Create a temporary YAML configuration file
    print("1. Creating temporary YAML configuration...")
    yaml_content = """
ball_detection: "temp_weights/ball_model.pt"
action_detection: "temp_weights/action_model.pt"
game_status: "temp_weights/game_state_checkpoint"
court_detection: "temp_weights/court_model.pt"
player_detection: null
"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_yaml_path = f.name
    
    print(f"   Temporary YAML file created: {temp_yaml_path}")
    
    try:
        # Initialize ML Manager with YAML configuration
        print("\n2. Initializing ML Manager with YAML configuration...")
        ml_manager = MLManager(weights_config=temp_yaml_path, verbose=True)
        
        # Get current configuration
        print("\n3. Getting current configuration...")
        current_config = ml_manager.get_weights_config()
        print(f"   Ball detection path: {current_config.ball_detection}")
        print(f"   Action detection path: {current_config.action_detection}")
        print(f"   Game status path: {current_config.game_status}")
        print(f"   Court detection path: {current_config.court_detection}")
        print(f"   Player detection path: {current_config.player_detection}")
        
        # Save current configuration to a new YAML file
        print("\n4. Saving current configuration to new YAML file...")
        output_yaml_path = "current_config.yaml"
        ml_manager.save_weights_config_to_yaml(output_yaml_path)
        
        if Path(output_yaml_path).exists():
            print(f"   Configuration saved to: {output_yaml_path}")
            
            # Show the saved content
            with open(output_yaml_path, 'r') as f:
                saved_content = f.read()
            print("   Saved content:")
            print(saved_content)
            
            # Clean up the saved file
            os.remove(output_yaml_path)
            print("   Temporary saved file cleaned up")
        
        # Cleanup
        ml_manager.cleanup()
        
    finally:
        # Clean up temporary YAML file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)
            print(f"   Temporary YAML file cleaned up: {temp_yaml_path}")
    
    print("\n=== YAML Configuration Example Completed ===")


def configuration_update_example():
    """Example of updating configuration dynamically."""
    print("=== Configuration Update Example ===\n")
    
    # Initialize with default configuration
    print("1. Initializing ML Manager with default configuration...")
    ml_manager = MLManager(verbose=False)
    
    # Get initial configuration
    print("\n2. Initial configuration:")
    initial_config = ml_manager.get_weights_config()
    print(f"   Ball detection: {initial_config.ball_detection}")
    
    # Update configuration
    print("\n3. Updating configuration...")
    new_config = ModelWeightsConfig(
        ball_detection="updated_weights/ball_model.pt",
        action_detection="updated_weights/action_model.pt",
        game_status="updated_weights/game_state_checkpoint",
        court_detection="updated_weights/court_model.pt",
        player_detection="updated_weights/player_model.pt"
    )
    
    ml_manager.update_weights_config(new_config)
    
    # Get updated configuration
    print("\n4. Updated configuration:")
    updated_config = ml_manager.get_weights_config()
    print(f"   Ball detection: {updated_config.ball_detection}")
    print(f"   Action detection: {updated_config.action_detection}")
    print(f"   Game status: {updated_config.game_status}")
    print(f"   Court detection: {updated_config.court_detection}")
    print(f"   Player detection: {updated_config.player_detection}")
    
    # Cleanup
    ml_manager.cleanup()
    print("\n=== Configuration Update Example Completed ===")


def environment_variables_example():
    """Example using environment variables for configuration."""
    print("=== Environment Variables Example ===\n")
    
    print("1. Setting environment variables...")
    print("   Note: In a real scenario, you would set these in your environment")
    print("   export ML_BALL_DETECTION='env_weights/ball_model.pt'")
    print("   export ML_ACTION_DETECTION='env_weights/action_model.pt'")
    print("   export ML_GAME_STATUS='env_weights/game_state_checkpoint'")
    print("   export ML_COURT_DETECTION='env_weights/court_model.pt'")
    print("   export ML_PLAYER_DETECTION='env_weights/player_model.pt'")
    
    print("\n2. When you set these environment variables, the ML Manager will")
    print("   automatically use them instead of the default values.")
    print("   This is useful for deployment and CI/CD pipelines.")
    
    print("\n3. The environment variable prefix is 'ML_' and the field names")
    print("   are automatically converted to uppercase with underscores.")
    
    print("\n=== Environment Variables Example Completed ===")


def video_processing_example():
    """Example of processing a video file."""
    print("=== Video Processing Example ===\n")
    
    # Initialize the ML Manager
    ml_manager = MLManager(verbose=False)
    
    # Example video path (replace with actual path)
    video_path = "path/to/your/volleyball_video.mp4"
    
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")
        return
    
    print(f"Processing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Process frames
    frame_idx = 0
    frames_for_state = []  # Buffer for game state classification
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Process every 10th frame for demonstration
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{frame_count}")
            
            try:
                # Ball detection
                ball_results = ml_manager.detect_ball(frame)
                
                # Action detection
                action_results = ml_manager.detect_actions(frame)
                
                # Player detection
                player_results = ml_manager.detect_players(frame)
                
                # Collect frames for game state classification
                frames_for_state.append(frame)
                if len(frames_for_state) == 16:
                    try:
                        game_state = ml_manager.classify_game_state(frames_for_state)
                        print(f"  Game state: {game_state}")
                    except RuntimeError:
                        pass
                    frames_for_state = frames_for_state[8:]  # Keep last 8 frames
                
                # Print results
                print(f"  Balls: {len(ball_results)}, Actions: {sum(len(d) for d in action_results.values())}, Players: {len(player_results)}")
                
            except RuntimeError as e:
                print(f"  Error processing frame: {e}")
        
        # Limit processing for demonstration
        if frame_idx > 100:
            break
    
    cap.release()
    ml_manager.cleanup()
    print("Video processing completed!")


def batch_processing_example():
    """Example of batch processing multiple frames."""
    print("=== Batch Processing Example ===\n")
    
    # Initialize the ML Manager
    ml_manager = MLManager(verbose=False)
    
    # Create multiple dummy frames
    print("Creating batch of frames...")
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
              for _ in range(5)]
    
    print(f"Created {len(frames)} frames")
    
    # Batch ball detection
    print("\nRunning batch ball detection...")
    try:
        ball_batch_results = ml_manager.detect_ball_batch(frames)
        print(f"Ball detection results for {len(ball_batch_results)} frames:")
        
        for i, frame_results in enumerate(ball_batch_results):
            print(f"  Frame {i+1}: {len(frame_results)} balls detected")
            for j, detection in enumerate(frame_results):
                bbox = detection['bbox']
                conf = detection['confidence']
                print(f"    Ball {j+1}: bbox={bbox}, confidence={conf:.3f}")
                
    except RuntimeError as e:
        print(f"Error in batch ball detection: {e}")
    
    # Batch action detection
    print("\nRunning batch action detection...")
    try:
        action_batch_results = ml_manager.detect_actions_batch(frames)
        print(f"Action detection results for {len(action_batch_results)} frames:")
        
        for i, frame_results in enumerate(action_batch_results):
            total_actions = sum(len(detections) for detections in frame_results.values())
            print(f"  Frame {i+1}: {total_actions} total actions")
            
            for action_type, detections in frame_results.items():
                if detections:
                    print(f"    {action_type}: {len(detections)} detections")
                    
    except RuntimeError as e:
        print(f"Error in batch action detection: {e}")
    
    # Cleanup
    ml_manager.cleanup()
    print("\nBatch processing completed!")


if __name__ == "__main__":
    # Run examples
    basic_usage_example()
    print("\n" + "="*50 + "\n")
    pydantic_config_example()
    print("\n" + "="*50 + "\n")
    yaml_config_example()
    print("\n" + "="*50 + "\n")
    configuration_update_example()
    print("\n" + "="*50 + "\n")
    environment_variables_example()
    print("\n" + "="*50 + "\n")
    batch_processing_example()
    print("\n" + "="*50 + "\n")
    video_processing_example()

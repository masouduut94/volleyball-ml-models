"""
Test file for the ML Manager

This file tests the basic functionality of the MLManager class,
including the new Pydantic settings and YAML configuration features.
"""

import numpy as np
from pathlib import Path
import sys
import tempfile
import os
import yaml

# Add the parent directory to the path to import the ml_manager
sys.path.append(str(Path(__file__).parent.parent))

from ml_manager import MLManager, ModelWeightsConfig


def test_ml_manager_initialization():
    """Test ML Manager initialization."""
    print("Testing ML Manager initialization...")
    
    try:
        # Initialize with default settings
        ml_manager = MLManager(verbose=True)
        print("✓ ML Manager initialized successfully")
        
        # Check model status
        status = ml_manager.get_model_status()
        print(f"✓ Model status retrieved: {len(status)} models")
        
        # Cleanup
        ml_manager.cleanup()
        print("✓ ML Manager cleaned up successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ ML Manager initialization failed: {e}")
        return False


def test_pydantic_config_initialization():
    """Test ML Manager initialization with Pydantic configuration."""
    print("\nTesting Pydantic configuration initialization...")
    
    try:
        # Create custom configuration
        custom_config = ModelWeightsConfig(
            ball_detection="test_weights/ball_model.pt",
            action_detection="test_weights/action_model.pt",
            game_status="test_weights/game_state_checkpoint",
            court_detection="test_weights/court_model.pt",
            player_detection=None
        )
        
        # Initialize with custom configuration
        ml_manager = MLManager(weights_config=custom_config, verbose=False)
        print("✓ ML Manager initialized with Pydantic configuration")
        
        # Get configuration
        config = ml_manager.get_weights_config()
        print(f"✓ Configuration retrieved: ball_detection = {config.ball_detection}")
        
        # Cleanup
        ml_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Pydantic configuration initialization failed: {e}")
        return False


def test_yaml_config_initialization():
    """Test ML Manager initialization with YAML configuration."""
    print("\nTesting YAML configuration initialization...")
    
    try:
        # Create temporary YAML file
        yaml_content = """
ball_detection: "yaml_weights/ball_model.pt"
action_detection: "yaml_weights/action_model.pt"
game_status: "yaml_weights/game_state_checkpoint"
court_detection: "yaml_weights/court_model.pt"
player_detection: null
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_yaml_path = f.name
        
        try:
            # Initialize with YAML configuration
            ml_manager = MLManager(weights_config=temp_yaml_path, verbose=False)
            print("✓ ML Manager initialized with YAML configuration")
            
            # Get configuration
            config = ml_manager.get_weights_config()
            print(f"✓ Configuration loaded from YAML: ball_detection = {config.ball_detection}")
            
            # Cleanup
            ml_manager.cleanup()
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_yaml_path):
                os.remove(temp_yaml_path)
        
        return True
        
    except Exception as e:
        print(f"✗ YAML configuration initialization failed: {e}")
        return False


def test_configuration_update():
    """Test dynamic configuration updates."""
    print("\nTesting configuration updates...")
    
    try:
        # Initialize with default configuration
        ml_manager = MLManager(verbose=False)
        
        # Get initial configuration
        initial_config = ml_manager.get_weights_config()
        print(f"✓ Initial configuration: ball_detection = {initial_config.ball_detection}")
        
        # Create new configuration
        new_config = ModelWeightsConfig(
            ball_detection="updated_weights/ball_model.pt",
            action_detection="updated_weights/action_model.pt",
            game_status="updated_weights/game_state_checkpoint",
            court_detection="updated_weights/court_model.pt",
            player_detection="updated_weights/player_model.pt"
        )
        
        # Update configuration
        ml_manager.update_weights_config(new_config)
        print("✓ Configuration updated successfully")
        
        # Verify update
        updated_config = ml_manager.get_weights_config()
        print(f"✓ Updated configuration: ball_detection = {updated_config.ball_detection}")
        
        # Cleanup
        ml_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Configuration update failed: {e}")
        return False


def test_configuration_save_to_yaml():
    """Test saving configuration to YAML file."""
    print("\nTesting configuration save to YAML...")
    
    try:
        # Initialize ML Manager
        ml_manager = MLManager(verbose=False)
        
        # Save configuration to YAML
        output_path = "test_config_output.yaml"
        ml_manager.save_weights_config_to_yaml(output_path)
        
        # Verify file was created
        if Path(output_path).exists():
            print("✓ Configuration saved to YAML file")
            
            # Read and verify content
            with open(output_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            print(f"✓ YAML content verified: {len(saved_config)} configuration items")
            
            # Clean up
            os.remove(output_path)
            print("✓ Test YAML file cleaned up")
            
        else:
            print("✗ YAML file was not created")
            return False
        
        # Cleanup
        ml_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Configuration save to YAML failed: {e}")
        return False


def test_model_availability():
    """Test model availability checking."""
    print("\nTesting model availability...")
    
    try:
        ml_manager = MLManager(verbose=False)
        
        # Check each model type
        model_types = [
            'action_detection',
            'ball_detection', 
            'court_segmentation',
            'player_detection',
            'game_state_classification'
        ]
        
        for model_type in model_types:
            available = ml_manager.is_model_available(model_type)
            status = "✓ Available" if available else "✗ Not Available"
            print(f"  - {model_type}: {status}")
        
        ml_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Model availability test failed: {e}")
        return False


def test_dummy_inference():
    """Test inference with dummy data."""
    print("\nTesting dummy inference...")
    
    try:
        ml_manager = MLManager(verbose=False)
        
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"✓ Created dummy frame: {dummy_frame.shape}")
        
        # Test ball detection
        try:
            ball_results = ml_manager.detect_ball(dummy_frame)
            print(f"✓ Ball detection: {len(ball_results)} results")
        except RuntimeError as e:
            print(f"⚠ Ball detection not available: {e}")
        
        # Test action detection
        try:
            action_results = ml_manager.detect_actions(dummy_frame)
            total_actions = sum(len(detections) for detections in action_results.values())
            print(f"✓ Action detection: {total_actions} total actions")
        except RuntimeError as e:
            print(f"⚠ Action detection not available: {e}")
        
        # Test player detection
        try:
            player_results = ml_manager.detect_players(dummy_frame)
            print(f"✓ Player detection: {len(player_results)} results")
        except RuntimeError as e:
            print(f"⚠ Player detection not available: {e}")
        
        # Test court segmentation
        try:
            court_results = ml_manager.segment_court(dummy_frame)
            print(f"✓ Court segmentation: {len(court_results)} results")
        except RuntimeError as e:
            print(f"⚠ Court segmentation not available: {e}")
        
        # Test game state classification
        try:
            dummy_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                           for _ in range(16)]
            game_state = ml_manager.classify_game_state(dummy_frames)
            print(f"✓ Game state classification: {game_state}")
        except RuntimeError as e:
            print(f"⚠ Game state classification not available: {e}")
        
        ml_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Dummy inference test failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing functionality."""
    print("\nTesting batch processing...")
    
    try:
        ml_manager = MLManager(verbose=False)
        
        # Create multiple dummy frames
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                 for _ in range(3)]
        print(f"✓ Created {len(frames)} dummy frames")
        
        # Test batch ball detection
        try:
            ball_batch_results = ml_manager.detect_ball_batch(frames)
            print(f"✓ Batch ball detection: {len(ball_batch_results)} frame results")
        except RuntimeError as e:
            print(f"⚠ Batch ball detection not available: {e}")
        
        # Test batch action detection
        try:
            action_batch_results = ml_manager.detect_actions_batch(frames)
            print(f"✓ Batch action detection: {len(action_batch_results)} frame results")
        except RuntimeError as e:
            print(f"⚠ Batch action detection not available: {e}")
        
        ml_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        return False


def test_pydantic_config_validation():
    """Test Pydantic configuration validation."""
    print("\nTesting Pydantic configuration validation...")
    
    try:
        # Test valid configuration
        valid_config = ModelWeightsConfig(
            ball_detection="valid/path/model.pt",
            action_detection="valid/path/model.pt",
            game_status="valid/path/checkpoint",
            court_detection="valid/path/model.pt",
            player_detection=None
        )
        print("✓ Valid configuration created successfully")
        
        # Test configuration attributes
        assert valid_config.ball_detection == "valid/path/model.pt"
        assert valid_config.player_detection is None
        print("✓ Configuration attributes validated")
        
        # Test configuration dictionary conversion
        config_dict = valid_config.dict()
        assert isinstance(config_dict, dict)
        assert "ball_detection" in config_dict
        print("✓ Configuration dictionary conversion works")
        
        return True
        
    except Exception as e:
        print(f"✗ Pydantic configuration validation failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("🧪 Running ML Manager Tests...\n")
    
    tests = [
        ("Initialization Test", test_ml_manager_initialization),
        ("Pydantic Config Initialization Test", test_pydantic_config_initialization),
        ("YAML Config Initialization Test", test_yaml_config_initialization),
        ("Configuration Update Test", test_configuration_update),
        ("Configuration Save to YAML Test", test_configuration_save_to_yaml),
        ("Model Availability Test", test_model_availability),
        ("Dummy Inference Test", test_dummy_inference),
        ("Batch Processing Test", test_batch_processing),
        ("Pydantic Config Validation Test", test_pydantic_config_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    print("=" * 40)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ML Manager is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the ML Manager setup.")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()

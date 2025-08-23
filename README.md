# Volleyball ML Manager

A unified, professional machine learning management system for volleyball analytics that brings together all ML models under a single, maintainable architecture.

## 🚀 Features

- **Flexible Configuration**: Support for hardcoded paths, Pydantic settings, and YAML configuration files
- **No Configuration Files Required**: Models work out of the box with sensible defaults
- **Unified Interface**: Single entry point for all ML operations
- **Professional Architecture**: Clean separation of concerns with modular design
- **Comprehensive Model Support**: YOLO object detection, VideoMAE classification, and more
- **Type Hints**: Full type annotations for better development experience
- **Error Handling**: Robust error handling and fallbacks
- **Easy Integration**: Simple import and usage pattern
- **Environment Variable Support**: Configure via environment variables for deployment

## 🏗️ Architecture

```
src/ml_manager/
├── __init__.py          # Main module interface
├── ml_manager.py        # Core MLManager class with Pydantic settings
├── .gitignore          # Git ignore rules
├── pyproject.toml      # Project configuration
├── README.md           # This file
├── SETUP.md            # Git submodule setup guide
├── sample_config.yaml  # Sample YAML configuration
├── example_usage.py    # Usage examples
└── test_ml_manager.py  # Testing framework
```

## 📋 Supported Models

### YOLO Models
- **Action Detection**: 6-class volleyball action recognition (serve, spike, block, receive, set, ball)
- **Ball Segmentation**: Ball detection with segmentation masks
- **Court Segmentation**: Volleyball court boundary detection
- **Player Detection**: Person detection and localization (with default YOLO pose fallback)

### VideoMAE Models
- **Game State Classification**: 3-state game phase classification (service, play, no-play)

## 🛠️ Installation

### Prerequisites
```bash
pip install torch torchvision ultralytics transformers pytorchvideo opencv-python pillow numpy pydantic pyyaml
```

### From Source
```bash
git clone https://github.com/volleyball-analytics/ml-manager.git
cd ml-manager
pip install -e .
```

## 📖 Quick Start

### Basic Usage (Default Configuration)
```python
from src.ml_manager import MLManager

# Initialize the manager with default configuration
ml_manager = MLManager()

# Detect ball in a frame
ball_results = ml_manager.detect_ball(frame)

# Detect actions
action_results = ml_manager.detect_actions(frame)

# Classify game state
game_state = ml_manager.classify_game_state(frames)
```

### Advanced Usage (Custom Configuration)
```python
from src.ml_manager import MLManager, ModelWeightsConfig

# Create custom configuration
custom_config = ModelWeightsConfig(
    ball_detection="custom/weights/ball_model.pt",
    action_detection="custom/weights/action_model.pt",
    game_status="custom/weights/game_state_checkpoint",
    court_detection="custom/weights/court_model.pt",
    player_detection=None  # Use default YOLO pose
)

# Initialize with custom configuration
ml_manager = MLManager(weights_config=custom_config, verbose=False)

# Use for inference
ball_results = ml_manager.detect_ball(frame)
```

### YAML Configuration
```python
from src.ml_manager import MLManager

# Initialize from YAML file
ml_manager = MLManager(weights_config="config/weights.yaml")

# Or update configuration dynamically
ml_manager.update_weights_config("new_config.yaml")
```

## 🔧 Configuration

### Pydantic Settings

The ML Manager uses Pydantic for configuration management with the following fields:

```python
class ModelWeightsConfig(BaseSettings):
    ball_detection: Optional[str] = "weights/ball_segment/model1/weights/best.pt"
    action_detection: Optional[str] = "weights/action_detection/6_class/1/weights/best.pt"
    game_status: Optional[str] = "weights/game-state/3-states/checkpoint"
    court_detection: Optional[str] = "weights/court_segment/weights/best.pt"
    player_detection: Optional[str] = None  # None = use default YOLO pose
```

### YAML Configuration Format

Create a YAML file with the following structure:

```yaml
# config/weights.yaml
ball_detection: "weights/ball_segment/model1/weights/best.pt"
action_detection: "weights/action_detection/6_class/1/weights/best.pt"
game_status: "weights/game-state/3-states/checkpoint"
court_detection: "weights/court_segment/weights/best.pt"
player_detection: null  # Use default YOLO pose
```

### Environment Variables

Set environment variables to override configuration:

```bash
export ML_BALL_DETECTION="custom/weights/ball_model.pt"
export ML_ACTION_DETECTION="custom/weights/action_model.pt"
export ML_GAME_STATUS="custom/weights/game_state_checkpoint"
export ML_COURT_DETECTION="custom/weights/court_model.pt"
export ML_PLAYER_DETECTION="custom/weights/player_model.pt"
```

## 🔧 API Reference

### MLManager Class

#### Constructor
```python
MLManager(
    weights_config: Optional[Union[ModelWeightsConfig, str]] = None,
    device: Optional[str] = None,
    verbose: bool = True
)
```

**Parameters:**
- `weights_config`: ModelWeightsConfig instance, YAML file path, or None for defaults
- `device`: Device to run models on ('cuda', 'cpu', or None for auto)
- `verbose`: Whether to print initialization messages

#### Configuration Methods
```python
def get_weights_config(self) -> ModelWeightsConfig
def update_weights_config(self, new_config: Union[ModelWeightsConfig, str])
def save_weights_config_to_yaml(self, output_path: str)
```

#### Model Methods

##### Action Detection
```python
def detect_actions(
    self, 
    frame: np.ndarray, 
    exclude: Optional[List[str]] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Dict[str, List[Dict[str, Any]]]
```

**Returns:** Dictionary mapping action types to lists of detection results
- Each detection contains: `bbox`, `confidence`, `class_id`

##### Ball Detection
```python
def detect_ball(
    self, 
    frame: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> List[Dict[str, Any]]
```

**Returns:** List of ball detections
- Each detection contains: `bbox`, `confidence`, `mask`

##### Court Segmentation
```python
def segment_court(
    self, 
    frame: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> List[Dict[str, Any]]
```

**Returns:** List of court segmentation results
- Each result contains: `bbox`, `confidence`, `mask`

##### Player Detection
```python
def detect_players(
    self, 
    frame: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> List[Dict[str, Any]]
```

**Returns:** List of player detections
- Each detection contains: `bbox`, `confidence`

##### Game State Classification
```python
def classify_game_state(
    self, 
    frames: List[np.ndarray]
) -> str
```

**Returns:** Predicted game state ('service', 'play', or 'no-play')

```python
def classify_game_state_with_confidence(
    self, 
    frames: List[np.ndarray]
) -> Dict[str, Any]
```

**Returns:** Dictionary with predicted state and confidence scores for all classes

##### Utility Methods
```python
def get_model_status(self) -> Dict[str, Dict[str, Any]]
def is_model_available(self, model_name: str) -> bool
def cleanup(self)
```

## 📁 Expected Directory Structure

The ML Manager expects the following directory structure for model weights:

```
weights/
├── action_detection/
│   └── 6_class/
│       └── 1/
│           └── weights/
│               └── best.pt
├── ball_segment/
│   └── model1/
│       └── weights/
│           └── best.pt
├── court_segment/
│   └── weights/
│       └── best.pt
├── game-state/
│   └── 3-states/
│       └── checkpoint/
│           ├── config.json
│           ├── model.safetensors
│           └── ...
└── yolov8n.pt
```

## 🔍 Model Availability

The ML Manager automatically checks for model availability and provides informative messages:

- ✅ **Model loaded successfully**: Model weights found and loaded
- ⚠ **Weights not found**: Model weights file missing
- ✗ **Failed to load**: Error during model initialization

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src/ml_manager tests/
```

Run the included test file:
```bash
python src/ml_manager/test_ml_manager.py
```

## 📊 Performance

### Model Loading
- **YOLO Models**: Fast loading with Ultralytics backend
- **VideoMAE Models**: Moderate loading time due to transformer architecture

### Inference Speed
- **YOLO Models**: Real-time performance on GPU
- **VideoMAE Models**: Batch processing recommended for optimal performance

### Memory Usage
- **YOLO Models**: Efficient memory usage
- **VideoMAE Models**: Higher memory requirements for transformer models

## 🚨 Error Handling

The ML Manager provides robust error handling:

```python
try:
    results = ml_manager.detect_actions(frame)
except RuntimeError as e:
    print(f"Action detection failed: {e}")
    # Handle gracefully
```

## 🔧 Customization

### Adding New Models
To add new models, extend the MLManager class:

```python
class CustomMLManager(MLManager):
    def _init_custom_model(self):
        # Custom initialization logic
        pass
    
    def custom_inference(self, input_data):
        # Custom inference logic
        pass
```

### Custom Device Management
```python
# Force CPU usage
ml_manager = MLManager(device="cpu")

# Force CUDA usage
ml_manager = MLManager(device="cuda")
```

### Dynamic Configuration Updates
```python
# Update configuration at runtime
new_config = ModelWeightsConfig(
    ball_detection="new/weights/ball_model.pt"
)
ml_manager.update_weights_config(new_config)

# Save current configuration
ml_manager.save_weights_config_to_yaml("current_config.yaml")
```

## 📝 Examples

### Complete Pipeline Example
```python
import cv2
import numpy as np
from src.ml_manager import MLManager

# Initialize with YAML configuration
ml_manager = MLManager(weights_config="config/weights.yaml")

# Load video
cap = cv2.VideoCapture("volleyball_match.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run all detections
    try:
        # Ball detection
        ball_results = ml_manager.detect_ball(frame)
        
        # Action detection
        action_results = ml_manager.detect_actions(frame)
        
        # Player detection
        player_results = ml_manager.detect_players(frame)
        
        # Process results
        print(f"Frame: {ball_results} balls, {len(action_results)} actions")
        
    except RuntimeError as e:
        print(f"Detection failed: {e}")
    
    # Display frame
    cv2.imshow("Volleyball Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ml_manager.cleanup()
```

### Configuration Management Example
```python
from src.ml_manager import MLManager, ModelWeightsConfig

# Initialize with default configuration
ml_manager = MLManager()

# Get current configuration
current_config = ml_manager.get_weights_config()
print(f"Ball detection path: {current_config.ball_detection}")

# Update configuration
new_config = ModelWeightsConfig(
    ball_detection="custom/weights/ball_model.pt",
    player_detection=None  # Use default YOLO pose
)
ml_manager.update_weights_config(new_config)

# Save configuration to YAML
ml_manager.save_weights_config_to_yaml("updated_config.yaml")
```

### Game State Analysis
```python
# Extract frames for temporal analysis
frames = []
for i in range(16):  # VideoMAE typically needs 16 frames
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

# Classify game state
if len(frames) == 16:
    game_state = ml_manager.classify_game_state(frames)
    confidence_results = ml_manager.classify_game_state_with_confidence(frames)
    
    print(f"Game State: {game_state}")
    print(f"Confidence: {confidence_results['max_confidence']:.3f}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/volleyball-analytics/ml-manager.git
cd ml-manager
pip install -e ".[dev]"
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: [https://volleyball-analytics.github.io/ml-manager](https://volleyball-analytics.github.io/ml-manager)
- **Issues**: [GitHub Issues](https://github.com/volleyball-analytics/ml-manager/issues)
- **Discussions**: [GitHub Discussions](https://github.com/volleyball-analytics/ml-manager/discussions)

## 🙏 Acknowledgments

- **Ultralytics**: For YOLO model support
- **Hugging Face**: For VideoMAE transformers
- **PyTorch**: For deep learning framework
- **Pydantic**: For configuration management
- **Volleyball Analytics Community**: For feedback and contributions

---

**Built with ❤️ for Volleyball Analytics**

*This module is designed to be a git submodule in the main volleyball analytics project.*

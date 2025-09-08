# 🏐 ML Manager - Volleyball Analytics

Unified machine learning manager for volleyball analytics, providing seamless integration of multiple deep learning models for comprehensive video analysis.

## 🎯 Features

- **🚀 Automatic Model Downloads** - No manual setup required
- **🎭 Action Detection** - 6-class volleyball action recognition using YOLO
- **🏐 Ball Detection** - Real-time ball tracking and segmentation  
- **🏟️ Court Segmentation** - Volleyball court boundary detection
- **🎮 Game State Classification** - Play/no-play state detection using VideoMAE
- **👥 Player Tracking** - Multi-player detection and tracking
- **📊 Unified Interface** - Single class manages all models

## 🚀 Quick Start

```python
from ml_manager import MLManager

# Initialize with automatic weight download
manager = MLManager()

# Process a video frame
results = manager.process_frame(frame)

# Get specific detections
ball_detections = manager.detect_ball(frame)
actions = manager.detect_actions(frame)
court_mask = manager.segment_court(frame)
```

## 📦 Installation

```bash
# Install dependencies
pip install torch torchvision ultralytics transformers opencv-python pillow numpy pydantic pyyaml gdown

# Fix PyTorchVideo compatibility (IMPORTANT!)
pip uninstall pytorchvideo -y
pip install git+https://github.com/facebookresearch/pytorchvideo

# Or install from pyproject.toml
pip install -e .
```

## 🎯 Model Weights

Weights are **automatically downloaded** on first use from [this Google Drive ZIP](https://drive.google.com/file/d/1__zkTmGwZo2z0EgbJvC14I_3kOpgQx3o/view).

Manual download:
```python
from ml_manager.utils.downloader import download_all_models
download_all_models()
```

## 🔧 Configuration

### Using Default Settings
```python
from ml_manager import MLManager

# Uses default weight paths and auto-download
manager = MLManager()
```

### Custom Configuration
```python
from ml_manager import MLManager
from ml_manager.settings import ModelWeightsConfig

# Custom weights configuration
config = ModelWeightsConfig(
    ball_detection="custom/path/ball.pt",
    action_detection="custom/path/action.pt",
    court_detection="custom/path/court.pt",
    game_state="custom/path/game_state/"
)

manager = MLManager(weights_config=config)
```

### YAML Configuration
```python
from ml_manager import MLManager

# Load from YAML file
manager = MLManager(weights_config="config.yaml")
```

## 🎬 Usage Examples

### Basic Video Processing
```python
import cv2
from ml_manager import MLManager

manager = MLManager()

cap = cv2.VideoCapture("volleyball_video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame with all models
    results = manager.process_frame(frame)
    
    # Visualize results
    annotated_frame = manager.visualize_results(frame, results)
    cv2.imshow("Results", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Individual Model Usage
```python
# Ball detection only
ball_results = manager.detect_ball(frame)

# Action detection only  
action_results = manager.detect_actions(frame)

# Court segmentation only
court_mask = manager.segment_court(frame)

# Game state classification (requires video sequence)
game_state_result = manager.classify_game_state(video_frames)
print(f"Predicted state: {game_state_result.predicted_class}")
print(f"Confidence: {game_state_result.confidence}")
```

### Batch Processing
```python
# Process multiple frames
frames = [frame1, frame2, frame3]
batch_results = manager.process_batch(frames)
```

## 📊 Output Format

```python
{
    "ball_detection": DetectionBatch,      # Ball positions and confidence
    "action_detection": DetectionBatch,    # Player actions and bounding boxes  
    "court_segmentation": np.ndarray,      # Court mask
    "game_state": GameStateResult,         # Play/no-play classification
    "player_tracking": List[PlayerKeyPoints]  # Player positions and keypoints
}
```

## 🔧 Advanced Configuration

### Device Selection
```python
# Use GPU
manager = MLManager(device="cuda")

# Use specific GPU
manager = MLManager(device="cuda:1")

# Use CPU
manager = MLManager(device="cpu")
```

### Performance Optimization
```python
# Initialize only needed models
from ml_manager.models import BallDetector, ActionDetector

ball_detector = BallDetector("weights/ball/weights/best.pt")
action_detector = ActionDetector("weights/action/weights/best.pt")
```

## 📝 Model Details

| Model | Type | Classes | Input Size | Framework |
|-------|------|---------|------------|-----------|
| Ball Detection | YOLOv8 Segmentation | 1 (ball) | 640x640 | Ultralytics |
| Action Detection | YOLOv8 Detection | 6 actions | 640x640 | Ultralytics |
| Court Segmentation | YOLOv8 Segmentation | 1 (court) | 640x640 | Ultralytics |
| Game State | VideoMAE | 2 states | 16 frames | Transformers |

### Action Classes
1. **Serve** - Player serving the ball
2. **Spike** - Attacking/spiking motion
3. **Block** - Defensive blocking
4. **Dig** - Defensive digging/receiving
5. **Set** - Setting the ball for attack
6. **Pass** - General passing/bumping

## 🛠️ Development

### Project Structure
```
ml_manager/
├── models/           # Individual model classes
├── core/             # Data structures and tracking
├── settings/         # Configuration management
├── utils/            # Utilities (downloader, logger)
├── visualization/    # Visualization tools
└── training/         # Training utilities
```

### Testing
```bash
python -m pytest tests/
```

### Linting
```bash
black .
isort .
flake8 .
```

## 🆘 Troubleshooting

### Common Issues

1. **🎯 PyTorchVideo Compatibility Error** (⚡ **EASIEST SOLUTION**)
   ```
   ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
   ```
   
   **💡 Quick Fix** (Recommended):
   ```bash
   # Uninstall old pytorchvideo
   pip uninstall pytorchvideo -y
   
   # Install latest version from GitHub (fixes compatibility)
   pip install git+https://github.com/facebookresearch/pytorchvideo
   ```
   
   **📝 Alternative Manual Fix**:
   If you encounter this error in your own code, replace:
   ```python
   import torchvision.transforms.functional_tensor as F_t
   ```
   with:
   ```python
   import torchvision.transforms.functional as F_t
   ```

2. **🎯 Model Weights Missing**
   - Check internet connection for auto-download
   - Verify weights directory structure
   - Try manual download using `download_all_models()`

3. **🐍 Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -e .
   
   # Or check specific imports
   python -c "from ml_manager import MLManager; print('Success!')"
   ```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the example code
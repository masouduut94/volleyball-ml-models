# Volleyball Analytics ML Manager

A unified machine learning module for volleyball analytics, providing object detection, segmentation, action recognition, and game state classification capabilities.

## Features

- **Unified ML Manager**: Single interface for all ML models
- **YOLO Integration**: Object detection, segmentation, and pose estimation
- **VideoMAE Integration**: Game state classification
- **Flexible Configuration**: Pydantic-based settings with YAML support
- **Comprehensive Training**: Full training support for both YOLO and VideoMAE models
- **Quick Training**: Simple training commands with minimal configuration
- **Resume Training**: Automatic checkpoint detection and resumption

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.ml_manager import MLManager

# Initialize with default models
ml_manager = MLManager()

# Detect actions in a frame
actions = ml_manager.detect_actions(frame)

# Detect ball
ball_detections = ml_manager.detect_ball(frame)

# Classify game state
game_state = ml_manager.classify_game_state(frames)
```

### Configuration

```python
from src.ml_manager import MLManager, ModelWeightsConfig

# Custom weights configuration
weights_config = ModelWeightsConfig(
    ball_detection="path/to/ball_model.pt",
    action_detection="path/to/action_model.pt",
    game_status="path/to/gamestate_model",
    court_detection="path/to/court_model.pt",
    player_detection=None  # Use default YOLO pose
)

ml_manager = MLManager(weights_config=weights_config)
```

## Training Models

### Quick Training (Recommended for most users)

```bash
# Quick YOLO training (detection)
python -m src.ml_manager.train --quick-yolo data.yaml --task detect --size n --epochs 100

# Quick YOLO training (segmentation)
python -m src.ml_manager.train --quick-yolo data.yaml --task segment --size m --epochs 50

# Quick YOLO training (pose estimation)
python -m src.ml_manager.train --quick-yolo data.yaml --task pose --size s --epochs 80

# Quick VideoMAE training
python -m src.ml_manager.train --quick-videomae data_dir --classes 3 --epochs 100

# Resume training from checkpoint
python -m src.ml_manager.train --quick-yolo data.yaml --resume
```

### Configuration File Training (Advanced users)

```bash
# Train YOLO model with configuration file
python -m src.ml_manager.train --config configs/yolo_config.yaml

# Train VideoMAE model with configuration file
python -m src.ml_manager.train --config configs/videomae_config.yaml

# Resume training
python -m src.ml_manager.train --config configs/yolo_config.yaml --resume

# Create configuration templates
python -m src.ml_manager.train --create-template yolo --output configs/yolo_template.yaml
python -m src.ml_manager.train --create-template videomae --output configs/videomae_template.yaml
```

### Python API Training

```python
from src.ml_manager.training import UnifiedTrainer

# Create trainer
trainer = UnifiedTrainer(verbose=True)

# Quick YOLO training
success = trainer.quick_train_yolo(
    data_yaml="data/ball_detection/data.yaml",
    model_size="n",
    task="detect",
    epochs=100,
    device="0"
)

# Quick VideoMAE training
success = trainer.quick_train_videomae(
    data_dir="data/game-status",
    num_classes=3,
    epochs=100,
    batch_size=8
)

# Full configuration training
from src.ml_manager.settings import YOLOTrainingConfig
config = YOLOTrainingConfig(...)
success = trainer.train_yolo(config, resume=False)
```

## Architecture

### Core Components

- **MLManager**: Main interface class that orchestrates all ML models
- **Model Classes**: Specialized classes for different ML tasks
- **Settings**: Configuration management using Pydantic
- **Training**: Comprehensive training interface for all model types

### Model Classes

- **YOLOModule**: Unified wrapper for YOLO models (detection, segmentation, pose)
- **ActionDetector**: Specialized for volleyball action detection
- **BallDetector**: Specialized for ball detection and tracking
- **CourtSegmentation**: Specialized for court segmentation
- **PlayerModule**: Unified player analysis (detection, segmentation, pose)
- **GameStatusClassifier**: VideoMAE-based game state classification

### Settings Structure

```
src/ml_manager/settings/
├── __init__.py
├── weights_config.py      # Model weights configuration
├── yolo_config.py         # YOLO training configuration
└── videomae_config.py     # VideoMAE training configuration
```

### Training Structure

```
src/ml_manager/training/
├── __init__.py
├── trainer.py             # Unified Trainer class
└── utils.py               # Training utilities and helpers
```

## Configuration

### Model Weights Configuration

```yaml
# Example weights configuration
ball_detection: "weights/ball_segment/model1/weights/best.pt"
action_detection: "weights/action_detection/6_class/1/weights/best.pt"
game_status: "weights/game-state/3-states/checkpoint"
court_detection: "weights/court_segment/weights/best.pt"
player_detection: null  # Use default YOLO pose
```

### YOLO Training Configuration

```yaml
# Example YOLO training configuration
model:
  size: "n"              # n, s, m, l, x
  type: "detect"          # detect, segment, pose
  num_classes: 6
  imgsz: 640
  batch: 16
  epochs: 100

dataset:
  data: "path/to/your/dataset.yaml"
  train: "images/train"
  val: "images/val"

training:
  lr0: 0.01
  momentum: 0.937
  weight_decay: 0.0005
```

### VideoMAE Training Configuration

```yaml
# Example VideoMAE training configuration
model:
  model_name: "MCG-NJU/videomae-base"
  num_classes: 3
  image_size: 224
  num_frames: 16

dataset:
  data_dir: "path/to/your/dataset"
  train_split: 0.8

training:
  num_epochs: 100
  batch_size: 8
  learning_rate: 1e-4
```

## Training Examples

### Ball Detection Training

```bash
# Quick training for ball detection
python -m src.ml_manager.train --quick-yolo data/ball_detection/data.yaml --task detect --size n --epochs 100

# With custom parameters
python -m src.ml_manager.train --quick-yolo data/ball_detection/data.yaml --task detect --size m --epochs 50 --batch-size 32 --device 0
```

### Action Detection Training

```bash
# Quick training for action detection (6 classes)
python -m src.ml_manager.train --quick-yolo data/action_detection/data.yaml --task detect --size s --epochs 100

# Resume interrupted training
python -m src.ml_manager.train --quick-yolo data/action_detection/data.yaml --task detect --size s --epochs 100 --resume
```

### Court Segmentation Training

```bash
# Quick training for court segmentation
python -m src.ml_manager.train --quick-yolo data/court_segmentation/data.yaml --task segment --size m --epochs 40
```

### Game State Classification Training

```bash
# Quick training for game state classification
python -m src.ml_manager.train --quick-videomae data/game-status --classes 3 --epochs 100 --batch-size 8
```

## Usage Examples

### Action Detection

```python
from src.ml_manager import MLManager

ml_manager = MLManager()

# Detect actions in a frame
actions = ml_manager.detect_actions(
    frame,
    exclude=["serve"],  # Exclude specific actions
    conf_threshold=0.5
)

# Filter high-confidence actions
high_conf_actions = actions.filter_by_confidence(min_confidence=0.7)

# Get action counts
action_counts = actions.get_class_counts()
```

### Ball Detection

```python
# Detect ball
ball_detections = ml_manager.detect_ball(frame)

# Get ball trajectory from multiple frames
trajectory = ml_manager.ball_detector.get_ball_trajectory(frames_list)

# Plot ball detections
annotated_frame = ml_manager.ball_detector.plot_ball_trajectory(
    frame, ball_detections
)
```

### Player Analysis

```python
# Detect players
players = ml_manager.detect_players(frame)

# Get player positions
positions = ml_manager.player_detector.get_player_positions(players)

# Switch to pose estimation mode
ml_manager.player_detector.switch_mode("pose")

# Get player poses
poses = ml_manager.player_detector.get_player_poses(players)
```

### Game State Classification

```python
# Classify game state from video frames
game_state = ml_manager.classify_game_state(
    frames,
    num_frames=16
)

# Get confidence scores
result = ml_manager.game_state_classifier.classify_game_state_with_confidence(frames)
print(f"Predicted: {result['predicted_label']}")
print(f"Confidence: {result['max_confidence']:.2f}")
```

## Training Utilities

### Automatic Batch Size Optimization

The training system automatically determines optimal batch sizes based on:
- Model size (n, s, m, l, x)
- Task type (detect, segment, pose)
- Available hardware (GPU/CPU)

### Dataset Validation

Automatic validation of dataset structure:
- YOLO: Checks for `images/train`, `images/val`, `labels/train`, `labels/val`
- VideoMAE: Checks for `train/*/*.mp4`, `test/*/*.mp4`

### Checkpoint Management

Automatic checkpoint detection and resumption:
- YOLO: Looks for `.pt` files in `runs/train`
- VideoMAE: Looks for `checkpoint-*` directories in output folder

## Environment Variables

You can override configuration values using environment variables:

```bash
export ML_BALL_DETECTION="path/to/custom/ball_model.pt"
export ML_ACTION_DETECTION="path/to/custom/action_model.pt"
export ML_GAME_STATUS="path/to/custom/gamestate_model"
```

## Dependencies

- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO model implementation
- **Transformers**: VideoMAE model implementation
- **Pydantic**: Data validation and settings management
- **PyYAML**: YAML configuration file support
- **Supervision**: Computer vision utilities
- **OpenCV**: Image processing
- **NumPy**: Numerical computing
- **PyTorchVideo**: Video data loading and transforms
- **Scikit-learn**: Metrics computation
- **Matplotlib & Seaborn**: Visualization and plotting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

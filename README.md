# 🏐 Volleyball Analytics ML Manager

A unified machine learning module for volleyball analytics, providing object detection, segmentation, action recognition, and game state classification capabilities.

## 🚀 Features

- **🎯 Unified ML Manager**: Single interface for all ML models
- **🔍 YOLO Integration**: Object detection, segmentation, and pose estimation
- **🎬 VideoMAE Integration**: Game state classification
- **⚙️ Flexible Configuration**: Pydantic-based settings with YAML support
- **🏋️ Comprehensive Training**: Full training support for both YOLO and VideoMAE models
- **⚡ Quick Training**: Simple training commands with minimal configuration
- **🔄 Resume Training**: Automatic checkpoint detection and resumption

## 📁 Project Structure

```
src/ml_manager/
├── 🧠 core/                           # Core utilities and data structures
│   ├── __init__.py                    # Core package exports
│   ├── data_structures.py             # Data classes for ML outputs
│   └── tracking_module.py             # Multi-object tracking logic
│
├── 🎨 visualization/                  # Visualization utilities
│   ├── __init__.py                    # Visualization package exports
│   └── visualization_module.py        # Comprehensive visualization tools
│
├── 🤖 models/                         # Actual ML model implementations
│   ├── __init__.py                    # Models package exports
│   ├── yolo_module.py                 # Unified YOLO wrapper
│   ├── action_detector.py             # Action detection model
│   ├── ball_detector.py               # Ball detection/segmentation
│   ├── court_segmentation.py          # Court segmentation model
│   ├── player_module.py               # Player detection/pose estimation
│   └── game_status_classifier.py      # Game state classification
│
├── ⚙️ settings/                       # Configuration management
├── 🏋️ training/                       # Training utilities
├── 🎯 weights/                        # Model weights storage
├── 📚 examples/                       # Usage examples
├── 📋 conf/                           # Configuration files
├── 🔢 enums.py                        # Enumerations
├── 🎮 ml_manager.py                   # Main ML Manager class
└── 📖 README.md                       # Main documentation
```

## 🎯 Model Weights Structure

The ML Manager expects model weights in the following organized structure:

```
weights/
├── 🏐 ball/                           # Ball detection & segmentation
│   ├── weights/
│   │   ├── best.pt                    # Best model weights (6.5MB)
│   │   └── last.pt                    # Latest model weights (6.5MB)
│   ├── args.yaml                      # Training configuration
│   ├── results.png                    # Training results visualization
│   ├── confusion_matrix.png           # Confusion matrix
│   └── *.jpg                          # Training/validation samples
│
├── 🎭 action/                         # Action detection (6 classes)
│   ├── weights/
│   │   ├── best.pt                    # Best model weights (6.0MB)
│   │   └── last.pt                    # Latest model weights (6.0MB)
│   ├── results.csv                    # Training metrics
│   └── val_batch0_labels.jpg         # Validation samples
│
├── 🏟️ court/                          # Court segmentation
│   ├── weights/
│   │   ├── best.pt                    # Best model weights (52MB)
│   │   └── last.pt                    # Latest model weights (52MB)
│   ├── args.yaml                      # Training configuration
│   ├── results.png                    # Training results
│   ├── confusion_matrix.png           # Confusion matrix
│   └── *.jpg                          # Training samples
│
└── 🎮 game_state/                     # Game state classification (VideoMAE)
    ├── model.safetensors              # Main model weights (329MB)
    ├── optimizer.pt                   # Optimizer state (658MB)
    ├── config.json                    # Model configuration
    ├── preprocessor_config.json       # Preprocessor settings
    └── trainer_state.json             # Training state
```

### 📊 Model Details

- **🏐 Ball Detection**: YOLOv8 segmentation model (6.5MB)
- **🎭 Action Detection**: YOLOv8 detection model for 6 volleyball actions (6.0MB)
- **🏟️ Court Segmentation**: YOLOv8 segmentation model for court boundaries (52MB)
- **🎮 Game State**: VideoMAE model for 3-state classification (329MB)

## 🚀 Installation

```bash
# Install the package with dependencies (recommended)
pip install -e .

# Or install dependencies manually
pip install torch torchvision ultralytics transformers pytorchvideo numpy opencv-python pillow pydantic pyyaml supervision dataclasses-json

# Or using uv (if available)
uv sync
```

## ⚡ Quick Start

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
    ball_detection="weights/ball/weights/best.pt",
    action_detection="weights/action/weights/best.pt",
    game_status="weights/game_state",
    court_detection="weights/court/weights/best.pt",
    player_detection=None  # Use default YOLO pose
)

ml_manager = MLManager(weights_config=weights_config)
```

## 🏋️ Training Models

### ⚡ Quick Training (Recommended for most users)

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

### 📋 Configuration File Training (Advanced users)

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

### 🐍 Python API Training

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

## 🏗️ Architecture

### 🧠 Core Components

- **🎮 MLManager**: Main interface class that orchestrates all ML models
- **🤖 Model Classes**: Specialized classes for different ML tasks
- **⚙️ Settings**: Configuration management using Pydantic
- **🏋️ Training**: Comprehensive training interface for all model types

### 🤖 Model Classes

- **🎯 YOLOModule**: Unified wrapper for YOLO models (detection, segmentation, pose)
- **🎭 ActionDetector**: Specialized for volleyball action detection
- **🏐 BallDetector**: Specialized for ball detection and tracking
- **🏟️ CourtSegmentation**: Specialized for court segmentation
- **👥 PlayerModule**: Unified player analysis (detection, segmentation, pose)
- **🎮 GameStatusClassifier**: VideoMAE-based game state classification

### ⚙️ Settings Structure

```
src/ml_manager/settings/
├── __init__.py
├── weights_config.py      # Model weights configuration
├── yolo_config.py         # YOLO training configuration
└── videomae_config.py     # VideoMAE training configuration
```

### 🏋️ Training Structure

```
src/ml_manager/training/
├── __init__.py
├── trainer.py             # Unified Trainer class
└── utils.py               # Training utilities and helpers
```

## ⚙️ Configuration

### 🎯 Model Weights Configuration

```yaml
# Example weights configuration
ball_detection: "weights/ball/weights/best.pt"
action_detection: "weights/action/weights/best.pt"
game_status: "weights/game_state"
court_detection: "weights/court/weights/best.pt"
player_detection: null  # Use default YOLO pose
```

### 🎯 YOLO Training Configuration

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

### 🎬 VideoMAE Training Configuration

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

## 🏋️ Training Examples

### 🏐 Ball Detection Training

```bash
# Quick training for ball detection
python -m src.ml_manager.train --quick-yolo data/ball_detection/data.yaml --task detect --size n --epochs 100

# With custom parameters
python -m src.ml_manager.train --quick-yolo data/ball_detection/data.yaml --task detect --size m --epochs 50 --batch-size 32 --device 0
```

### 🎭 Action Detection Training

```bash
# Quick training for action detection (6 classes)
python -m src.ml_manager.train --quick-yolo data/action_detection/data.yaml --task detect --size s --epochs 100

# Resume interrupted training
python -m src.ml_manager.train --quick-yolo data/action_detection/data.yaml --task detect --size s --epochs 100 --resume
```

### 🏟️ Court Segmentation Training

```bash
# Quick training for court segmentation
python -m src.ml_manager.train --quick-yolo data/court_segmentation/data.yaml --task segment --size m --epochs 40
```

### 🎮 Game State Classification Training

```bash
# Quick training for game state classification
python -m src.ml_manager.train --quick-videomae data/game-status --classes 3 --epochs 100 --batch-size 8
```

## 💻 Usage Examples

### 🎭 Action Detection

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

### 🏐 Ball Detection

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

### 👥 Player Analysis

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

### 🎮 Game State Classification

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

## 🛠️ Training Utilities

### ⚡ Automatic Batch Size Optimization

The training system automatically determines optimal batch sizes based on:
- Model size (n, s, m, l, x)
- Task type (detect, segment, pose)
- Available hardware (GPU/CPU)

### ✅ Dataset Validation

Automatic validation of dataset structure:
- YOLO: Checks for `images/train`, `images/val`, `labels/train`, `labels/val`
- VideoMAE: Checks for `train/*/*.mp4`, `test/*/*.mp4`

### 💾 Checkpoint Management

Automatic checkpoint detection and resumption:
- YOLO: Looks for `.pt` files in `runs/train`
- VideoMAE: Looks for `checkpoint-*` directories in output folder

## 🔧 Environment Variables

You can override configuration values using environment variables:

```bash
export ML_BALL_DETECTION="path/to/custom/ball_model.pt"
export ML_ACTION_DETECTION="path/to/custom/action_model.pt"
export ML_GAME_STATUS="path/to/custom/gamestate_model"
```

## 📦 Dependencies

- **🔥 PyTorch**: Deep learning framework
- **🎯 Ultralytics**: YOLO model implementation
- **🤗 Transformers**: VideoMAE model implementation
- **⚙️ Pydantic**: Data validation and settings management
- **📋 PyYAML**: YAML configuration file support
- **👁️ Supervision**: Computer vision utilities
- **📹 OpenCV**: Image processing
- **🔢 NumPy**: Numerical computing
- **🎬 PyTorchVideo**: Video data loading and transforms
- **📊 Scikit-learn**: Metrics computation
- **📈 Matplotlib & Seaborn**: Visualization and plotting

## 🔄 Git Submodule Setup

### 🚀 Quick Setup

```bash
# Add the ML Manager as a submodule
git submodule add https://github.com/volleyball-analytics/ml-manager.git src/ml_manager

# Initialize and update the submodule
git submodule update --init --recursive
```

### 📥 Clone Project with Submodules

```bash
# Clone the main project with submodules
git clone --recursive https://github.com/your-username/volleyball-analytics.git

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### 🔄 Update Submodule

```bash
# Navigate to the submodule directory
cd src/ml_manager

# Pull latest changes
git pull origin main

# Go back to main project
cd ../..

# Commit the submodule update
git add src/ml_manager
git commit -m "Update ML Manager submodule"
```

## 🧪 Testing the Setup

### 1. Test Basic Functionality

```bash
cd src/ml_manager
python test_ml_manager.py
```

### 2. Test Integration

```bash
cd ../..
python src/ml_manager/example_usage.py
```

### 3. Test Main Project

```bash
python src/demo.py --video_path path/to/your/video.mp4
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd volleyball-analytics
   python -c "from src.ml_manager import MLManager; print('Import successful')"
   ```

2. **Submodule Not Initialized**
   ```bash
   git submodule update --init --recursive
   ```

3. **Model Weights Not Found**
   - Check that weights directory exists
   - Verify model weight file paths
   - Check file permissions

4. **Dependencies Missing**
   ```bash
   # Install the package with all dependencies
   pip install -e .
   
   # Or install core dependencies manually
   pip install torch torchvision ultralytics transformers pytorchvideo numpy opencv-python pillow pydantic pyyaml supervision dataclasses-json
   ```

### Debug Mode

```python
# Enable verbose mode for debugging
ml_manager = MLManager(verbose=True)

# Check model status
status = ml_manager.get_model_status()
for model_name, info in status.items():
    print(f"{model_name}: {'Available' if info['available'] else 'Not Available'}")
```

## 🔄 Maintenance

### Regular Updates

```bash
# Update submodule to latest version
cd src/ml_manager
git pull origin main
cd ../..
git add src/ml_manager
git commit -m "Update ML Manager submodule"
```

### Contributing to ML Manager

```bash
# Make changes in the submodule
cd src/ml_manager

# Create a new branch
git checkout -b feature/new-feature

# Make your changes and commit
git add .
git commit -m "Add new feature"

# Push to your fork
git push origin feature/new-feature

# Create pull request on GitHub
```

## 📚 Additional Resources

- [Example Usage](src/ml_manager/example_usage.py) - Usage examples
- [Test Suite](src/ml_manager/test_ml_manager.py) - Testing framework
- [Git Submodules Documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the ML Manager README
3. Check model weight file paths
4. Verify Python dependencies
5. Open an issue on the ML Manager repository

---

**Happy Volleyball Analytics! 🏐✨**

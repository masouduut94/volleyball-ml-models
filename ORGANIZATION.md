# ML Manager File Organization

This document describes the organized structure of the ML Manager module for better maintainability and clarity.

## Directory Structure

```
src/ml_manager/
├── core/                           # Core utilities and data structures
│   ├── __init__.py                # Core package exports
│   ├── data_structures.py         # Data classes for ML outputs
│   └── tracking_module.py         # Multi-object tracking logic
│
├── visualization/                  # Visualization utilities
│   ├── __init__.py                # Visualization package exports
│   └── visualization_module.py    # Comprehensive visualization tools
│
├── models/                         # Actual ML model implementations
│   ├── __init__.py                # Models package exports
│   ├── yolo_module.py             # Unified YOLO wrapper
│   ├── action_detector.py         # Action detection model
│   ├── ball_detector.py           # Ball detection/segmentation
│   ├── court_segmentation.py      # Court segmentation model
│   ├── player_module.py           # Player detection/pose estimation
│   └── game_status_classifier.py  # Game state classification
│
├── settings/                       # Configuration management
├── training/                       # Training utilities
├── weights/                        # Model weights storage
├── examples/                       # Usage examples
├── conf/                          # Configuration files
├── enums.py                       # Enumerations
├── ml_manager.py                  # Main ML Manager class
└── README.md                      # Main documentation
```

## Organization Principles

### 1. **Core Package** (`core/`)
- **Purpose**: Contains fundamental data structures and utilities used throughout the system
- **Contents**:
  - `data_structures.py`: Data classes for detection results, bounding boxes, keypoints, etc.
  - `tracking_module.py`: Multi-object tracking logic (not a deep learning model)
- **Why here**: These are foundational utilities that don't represent ML models

### 2. **Visualization Package** (`visualization/`)
- **Purpose**: Tools for displaying and analyzing results
- **Contents**:
  - `visualization_module.py`: Comprehensive visualization for detections, tracking, and analytics
- **Why here**: Visualization is a separate concern from ML model implementation

### 3. **Models Package** (`models/`)
- **Purpose**: Contains actual deep learning model implementations
- **Contents**:
  - YOLO-based models for detection, segmentation, and pose estimation
  - VideoMAE-based game state classification
- **Why here**: These are the actual ML models that perform inference

### 4. **Settings Package** (`settings/`)
- **Purpose**: Configuration management using Pydantic
- **Why here**: Configuration is a cross-cutting concern

### 5. **Training Package** (`training/`)
- **Purpose**: Training utilities and trainers
- **Why here**: Training is a separate concern from inference

## Benefits of This Organization

1. **Clear Separation of Concerns**: ML models, utilities, and visualization are clearly separated
2. **Easier Maintenance**: Related functionality is grouped together
3. **Better Imports**: Clear import paths make dependencies obvious
4. **Logical Grouping**: Files are organized by their purpose, not just by being "ML-related"
5. **Scalability**: Easy to add new models, utilities, or visualization tools

## Import Patterns

### From Core
```python
from .core import DetectionBatch, VolleyballTracker
```

### From Visualization
```python
from .visualization import VolleyballVisualizer
```

### From Models
```python
from .models import YOLOModule, ActionDetector
```

### From Settings
```python
from .settings import ModelWeightsConfig
```

## Adding New Components

### New ML Model
- Place in `models/` directory
- Update `models/__init__.py`
- Follow existing naming conventions

### New Utility
- Place in `core/` directory
- Update `core/__init__.py`
- Ensure it's truly a utility, not a model

### New Visualization Tool
- Place in `visualization/` directory
- Update `visualization/__init__.py`
- Keep visualization logic separate from ML logic

This organization makes the codebase more maintainable and easier to understand for new developers.

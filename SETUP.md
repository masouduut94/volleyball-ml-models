# ML Manager Git Submodule Setup Guide

This guide explains how to set up the ML Manager as a git submodule in your main volleyball analytics project.

## 🚀 Quick Setup

### 1. Add as Submodule (from main project root)

```bash
# Add the ML Manager as a submodule
git submodule add https://github.com/volleyball-analytics/ml-manager.git src/ml_manager

# Initialize and update the submodule
git submodule update --init --recursive
```

### 2. Clone Project with Submodules

If you're cloning a project that already has this submodule:

```bash
# Clone the main project with submodules
git clone --recursive https://github.com/your-username/volleyball-analytics.git

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### 3. Update Submodule

To update the ML Manager to the latest version:

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

## 📁 Directory Structure

After setup, your project should look like this:

```
volleyball-analytics/
├── src/
│   ├── ml_manager/          # Git submodule
│   │   ├── .git/           # Submodule git info
│   │   ├── __init__.py
│   │   ├── ml_manager.py
│   │   ├── .gitignore
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── example_usage.py
│   │   └── test_ml_manager.py
│   ├── main.py             # Updated to use ML Manager
│   ├── demo.py             # Updated to use ML Manager
│   └── ...
├── weights/                 # Model weights directory
├── .gitmodules             # Git submodule configuration
└── ...
```

## 🔧 Usage in Main Project

### Import the ML Manager

```python
# In your main project files
from src.ml_manager import MLManager

# Initialize
ml_manager = MLManager()

# Use for inference
ball_results = ml_manager.detect_ball(frame)
action_results = ml_manager.detect_actions(frame)
game_state = ml_manager.classify_game_state(frames)
```

### Example: Updated main.py

```python
# Old way (with YAML config)
# import yaml
# cfg = yaml.load(open('conf/ml_models.yaml'), Loader=yaml.SafeLoader)

# New way (no config needed)
from src.ml_manager import MLManager

ml_manager = MLManager(verbose=True)
# Models are automatically initialized with hardcoded paths
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

## 📋 Model Weights Setup

The ML Manager expects model weights in the following structure:

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
   pip install torch torchvision ultralytics transformers pytorchvideo opencv-python pillow numpy
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

- [ML Manager README](src/ml_manager/README.md) - Comprehensive documentation
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

**Happy Volleyball Analytics! 🏐**

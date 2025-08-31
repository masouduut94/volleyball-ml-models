# ML Manager Utils - Downloader

Simple utility to download all model weights from Google Drive in one ZIP file.

## Quick Start

```python
from ml_manager.utils.downloader import download_all_models

# Download all models
success = download_all_models()
```

## Available Models

The ZIP file contains:
- **Ball Detection**: `weights/ball/weights/best.pt`
- **Action Detection**: `weights/action/weights/best.pt`  
- **Court Detection**: `weights/court/weights/best.pt`
- **Game State**: `weights/game_state/` (checkpoint files)

## Functions

### `download_all_models(weights_dir=None, force_download=False, quiet=False)`
Downloads and extracts all model weights from a single ZIP file.

### `check_model_weights(weights_dir=None)`
Returns dictionary showing which models are available locally.

### `download_from_google_drive(file_id, output_path, quiet=False)`
Direct download from Google Drive using file ID.

## Auto-Download

Model weights are automatically downloaded when you initialize `MLManager`:

```python
from ml_manager import MLManager

# Automatically downloads missing weights
manager = MLManager()
```
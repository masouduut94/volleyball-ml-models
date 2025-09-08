"""
Google Drive downloader utility for ML model weights.

This module provides functionality to download model weights from Google Drive
using a single ZIP file containing all models.
"""

import os
import zipfile
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urlparse, parse_qs

try:
    import gdown
except ImportError:
    gdown = None

from .logger import logger


# Google Drive file ID for the complete weights ZIP file
WEIGHTS_ZIP_DRIVE_ID = "1__zkTmGwZo2z0EgbJvC14I_3kOpgQx3o"

# Expected file paths for each model
MODEL_PATHS = {
    "ball_detection": "ball/weights/best.pt",
    "action_detection": "action/weights/best.pt",
    "court_detection": "court/weights/best.pt",
    "game_state": "game_state/"
}


def extract_drive_id(url: str) -> Optional[str]:
    """
    Extract Google Drive file ID from various Google Drive URL formats.
    
    Args:
        url: Google Drive URL in various formats
        
    Returns:
        Google Drive file ID or None if not found
    """
    # Direct file ID
    if len(url) == 33 and not url.startswith('http'):
        return url
    
    # Parse different Google Drive URL formats
    parsed = urlparse(url)
    
    # Format: https://drive.google.com/file/d/FILE_ID/view
    if 'drive.google.com' in parsed.netloc and '/file/d/' in parsed.path:
        parts = parsed.path.split('/')
        try:
            file_id_index = parts.index('d') + 1
            return parts[file_id_index]
        except (ValueError, IndexError):
            pass
    
    # Format: https://drive.google.com/open?id=FILE_ID
    if 'drive.google.com' in parsed.netloc and parsed.path == '/open':
        query_params = parse_qs(parsed.query)
        if 'id' in query_params:
            return query_params['id'][0]
    
    return None


def download_from_google_drive(file_id: str, output_path: Union[str, Path], quiet: bool = False) -> bool:
    """
    Download a file from Google Drive using its file ID.
    
    Args:
        file_id: Google Drive file ID
        output_path: Local path where the file should be saved
        quiet: Whether to suppress download progress output
        
    Returns:
        True if download was successful, False otherwise
    """
    if gdown is None:
        raise ImportError(
            "gdown is required for downloading from Google Drive. "
            "Install it with: pip install gdown"
        )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Downloading from Google Drive (ID: {file_id}) to {output_path}")
        
        # Download file
        gdown.download(id=file_id, output=str(output_path), quiet=quiet)
        
        if not output_path.exists():
            logger.error(f"Download failed: {output_path} does not exist")
            return False
            
        logger.info(f"Successfully downloaded to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download from Google Drive: {e}")
        return False


def download_all_models(weights_dir: Optional[Union[str, Path]] = None, force_download: bool = False,
                        quiet: bool = False) -> bool:
    """
    Download all model weights from the complete ZIP file.
    
    Args:
        weights_dir: Base directory for weights (defaults to 'weights' in current directory)
        force_download: Whether to download even if files already exist
        quiet: Whether to suppress download progress output
        
    Returns:
        True if download and extraction was successful, False otherwise
    """
    if weights_dir is None:
        weights_dir = Path.cwd() / "weights"
    else:
        weights_dir = Path(weights_dir)
    
    # Check if all models already exist
    if not force_download:
        existing_models = check_model_weights(weights_dir)
        if all(existing_models.values()):
            logger.info("All model weights already exist")
            return True
    
    # Create weights directory
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Download ZIP file
    zip_path = weights_dir / "all_weights.zip"
    
    try:
        logger.info("Downloading complete weights ZIP file from Google Drive...")
        success = download_from_google_drive(
            file_id=WEIGHTS_ZIP_DRIVE_ID,
            output_path=zip_path,
            quiet=quiet
        )
        
        if not success:
            logger.error("Failed to download weights ZIP file")
            return False
        
        # Extract ZIP file
        logger.info(f"Extracting weights to {weights_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)
        
        # Remove ZIP file after extraction
        zip_path.unlink()
        logger.info("ZIP file extracted and removed")
        
        # Verify extraction
        extracted_models = check_model_weights(weights_dir)
        successful_extractions = sum(extracted_models.values())
        total_models = len(extracted_models)
        
        if successful_extractions == total_models:
            logger.success(f"Successfully extracted all {total_models} models")
            return True
        else:
            missing = [name for name, exists in extracted_models.items() if not exists]
            logger.warning(f"Some models missing after extraction: {missing}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to download/extract weights ZIP: {e}")
        # Clean up partial download
        if zip_path.exists():
            zip_path.unlink()
        return False


def check_model_weights(weights_dir: Optional[Union[str, Path]] = None) -> Dict[str, bool]:
    """
    Check which model weights are available locally.
    
    Args:
        weights_dir: Base directory for weights (defaults to 'weights' in current directory)
        
    Returns:
        Dictionary mapping model names to availability status
    """
    if weights_dir is None:
        weights_dir = Path.cwd()
    else:
        weights_dir = Path(weights_dir)
    
    results = {}
    
    for model_name, relative_path in MODEL_PATHS.items():
        target_path = weights_dir / relative_path
        
        if model_name == "game_state":
            # For game_state, check if directory exists and has files
            exists = target_path.exists() and any(target_path.iterdir())
        else:
            # For other models, check if the .pt file exists
            exists = target_path.exists()
        
        results[model_name] = exists
    
    return results
"""
Training utilities for ML Manager.

This module contains common utility functions used by both YOLO and VideoMAE training.
"""

import os
import torch
import numpy as np
import pandas as pd
import seaborn as sn
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import Dict, Any, List, Tuple, Optional

import pytorchvideo.data
from torchvision.transforms import Compose, Lambda, Resize
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, Normalize
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor


def compute_videomae_metrics(pred, class_labels: List[str]) -> Dict[str, float]:
    """
    Compute metrics for VideoMAE training.
    
    Args:
        pred: Prediction object from trainer
        class_labels: List of class labels
        
    Returns:
        Dictionary with accuracy, f1, precision, and recall
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    predictions = np.argmax(pred.predictions, axis=1)

    # Create confusion matrix
    cm = confusion_matrix(pred.label_ids, predictions)
    df_cfm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 7))
    cfm_plot = sn.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig("confusion_matrix_model.jpg")
    plt.close()

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def create_videomae_collate_fn():
    """
    Create collate function for VideoMAE training.
    
    Returns:
        Collate function for data batching
    """
    def collate_fn(examples):
        """The collation function to be used by `Trainer` to prepare data batches."""
        # permute to (num_frames, num_channels, height, width)
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    return collate_fn


def create_videomae_transforms(
    feature_extractor: VideoMAEImageProcessor,
    num_frames: int,
    resize_to: Tuple[int, int] = (224, 224)
) -> Tuple[Any, Any]:
    """
    Create training and validation transforms for VideoMAE.
    
    Args:
        feature_extractor: VideoMAE image processor
        num_frames: Number of frames to sample
        resize_to: Target image size
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    mean = feature_extractor.image_mean
    std = feature_extractor.image_std

    # Training dataset transformations
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to)
                    ]
                ),
            ),
        ]
    )

    # Validation and evaluation datasets' transformations
    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )
    
    return train_transform, val_transform


def create_videomae_datasets(
    data_path: str,
    num_frames: int,
    sample_rate: int = 3,
    fps: int = 30
) -> Tuple[Any, Any, List[str], Dict[str, int], Dict[int, str]]:
    """
    Create VideoMAE training and validation datasets.
    
    Args:
        data_path: Path to dataset directory
        num_frames: Number of frames to sample
        sample_rate: Frame sampling rate
        fps: Frames per second
        
    Returns:
        Tuple of (train_dataset, val_dataset, class_labels, label2id, id2label)
    """
    dataset_root_path = Path(data_path)
    train_files = list(dataset_root_path.glob("train/*/*.mp4"))
    test_files = list(dataset_root_path.glob("test/*/*.mp4"))

    video_count_train = len(train_files)
    video_count_val = len(test_files)
    video_total = video_count_train + video_count_val
    
    print(f"Total videos: {video_total}")
    print(f"Training videos: {video_count_train}")
    print(f"Validation videos: {video_count_val}")

    all_video_file_paths = train_files + test_files
    class_labels = sorted({path.parent.stem for path in all_video_file_paths})
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print("Class labels:", class_labels)

    # Calculate clip duration
    clip_duration = num_frames * sample_rate / fps
    print(f"Clip duration: {clip_duration} seconds")

    # Create transforms
    feature_extractor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    train_transform, val_transform = create_videomae_transforms(feature_extractor, num_frames)

    # Create datasets
    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    val_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    
    return train_dataset, val_dataset, class_labels, label2id, id2label


def create_videomae_model(
    model_name: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str]
) -> Tuple[VideoMAEForVideoClassification, VideoMAEImageProcessor]:
    """
    Create VideoMAE model and processor.
    
    Args:
        model_name: HuggingFace model name
        label2id: Label to ID mapping
        id2label: ID to label mapping
        
    Returns:
        Tuple of (model, processor)
    """
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    
    return model, processor


def create_yolo_training_args(
    model_size: str = "n",
    task: str = "detect",
    data_yaml: str = "",
    epochs: int = 100,
    batch_size: int = 32,
    imgsz: int = 640,
    device: str = "0",
    optimizer: str = "AdamW",
    learning_rate: float = 0.001,
    final_lr_factor: float = 0.01,
    workers: int = 16,
    **kwargs
) -> Dict[str, Any]:
    """
    Create YOLO training arguments.
    
    Args:
        model_size: YOLO model size (n, s, m, l, x)
        task: Training task (detect, segment, pose)
        data_yaml: Path to data YAML file
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Input image size
        device: Device to use
        optimizer: Optimizer name
        learning_rate: Initial learning rate
        final_lr_factor: Final learning rate factor
        workers: Number of workers
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of training arguments
    """
    args = {
        'data': data_yaml,
        'epochs': epochs,
        'task': task,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': device,
        'optimizer': optimizer,
        'lr0': learning_rate,
        'lrf': final_lr_factor,
        'workers': workers,
        'plots': True,
        'cos_lr': True,
        'seed': 1368,
        **kwargs
    }
    
    return args


def validate_dataset_path(data_path: str) -> bool:
    """
    Validate that dataset path exists and has required structure.
    
    Args:
        data_path: Path to dataset
        
    Returns:
        True if valid, False otherwise
    """
    path = Path(data_path)
    
    if not path.exists():
        print(f"Dataset path does not exist: {data_path}")
        return False
    
    # Check for YOLO dataset structure
    if (path / "images").exists() and (path / "labels").exists():
        train_images = list((path / "images" / "train").glob("*.jpg")) + list((path / "images" / "train").glob("*.png"))
        val_images = list((path / "images" / "val").glob("*.jpg")) + list((path / "images" / "val").glob("*.png"))
        
        if len(train_images) > 0 and len(val_images) > 0:
            print(f"YOLO dataset found: {len(train_images)} training, {len(val_images)} validation images")
            return True
    
    # Check for VideoMAE dataset structure
    if (path / "train").exists() and (path / "test").exists():
        train_videos = list((path / "train").rglob("*.mp4")) + list((path / "train").rglob("*.avi"))
        test_videos = list((path / "test").rglob("*.mp4")) + list((path / "test").rglob("*.avi"))
        
        if len(train_videos) > 0 and len(test_videos) > 0:
            print(f"VideoMAE dataset found: {len(train_videos)} training, {len(test_videos)} test videos")
            return True
    
    print(f"Dataset structure not recognized at: {data_path}")
    return False


def save_training_results(results: Dict[str, Any], output_dir: str, model_name: str):
    """
    Save training results and metrics.
    
    Args:
        results: Training results dictionary
        output_dir: Output directory
        model_name: Name of the model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results to JSON
    import json
    results_file = output_path / f"{model_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Training results saved to: {results_file}")


def get_optimal_batch_size(model_size: str, task: str, device: str = "0") -> int:
    """
    Get optimal batch size based on model size and task.
    
    Args:
        model_size: YOLO model size (n, s, m, l, x)
        task: Training task (detect, segment, pose)
        device: Device to use
        
    Returns:
        Recommended batch size
    """
    # Base batch sizes for different model sizes
    base_batch_sizes = {
        'n': 32,
        's': 24,
        'm': 16,
        'l': 8,
        'x': 4
    }
    
    # Adjust for different tasks
    task_multipliers = {
        'detect': 1.0,
        'segment': 0.75,
        'pose': 0.5
    }
    
    base_batch = base_batch_sizes.get(model_size, 16)
    multiplier = task_multipliers.get(task, 1.0)
    
    # Adjust for device
    if device == "cpu":
        multiplier *= 0.5
    
    optimal_batch = int(base_batch * multiplier)
    
    print(f"Recommended batch size for YOLOv8{model_size} {task}: {optimal_batch}")
    return optimal_batch

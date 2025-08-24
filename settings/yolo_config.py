"""
YOLO training configuration.

This module provides a Pydantic-based configuration class for YOLO training
parameters, with validation and default values.
"""

from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from ..enums import YOLOModelType


class YOLOModelConfig(BaseModel):
    """YOLO model configuration."""
    size: str = Field(default="n", description="YOLO model size (n, s, m, l, x)")
    type: YOLOModelType = Field(default=YOLOModelType.DETECTION, description="Model type")
    num_classes: int = Field(default=6, description="Number of classes")
    imgsz: int = Field(default=640, description="Input image size")
    batch: int = Field(default=16, description="Batch size")
    epochs: int = Field(default=100, description="Number of epochs")


class YOLODatasetConfig(BaseModel):
    """YOLO dataset configuration."""
    data: str = Field(description="Path to dataset YAML file")
    train: str = Field(default="images/train", description="Training images path")
    val: str = Field(default="images/val", description="Validation images path")
    workers: int = Field(default=8, description="Number of workers for data loading")
    cache: bool = Field(default=False, description="Cache images in memory")


class YOLOTrainingParams(BaseModel):
    """YOLO training parameters."""
    lr0: float = Field(default=0.01, description="Initial learning rate")
    lrf: float = Field(default=0.01, description="Final learning rate")
    momentum: float = Field(default=0.937, description="Momentum")
    weight_decay: float = Field(default=0.0005, description="Weight decay")
    warmup_epochs: float = Field(default=3.0, description="Warmup epochs")
    warmup_momentum: float = Field(default=0.8, description="Warmup momentum")
    warmup_bias_lr: float = Field(default=0.1, description="Warmup bias learning rate")


class YOLOAugmentationConfig(BaseModel):
    """YOLO augmentation configuration."""
    hsv_h: float = Field(default=0.015, description="HSV hue augmentation")
    hsv_s: float = Field(default=0.7, description="HSV saturation augmentation")
    hsv_v: float = Field(default=0.4, description="HSV value augmentation")
    degrees: float = Field(default=0.0, description="Rotation degrees")
    translate: float = Field(default=0.1, description="Translation")
    scale: float = Field(default=0.5, description="Scale")
    shear: float = Field(default=0.0, description="Shear")
    perspective: float = Field(default=0.0, description="Perspective")
    flipud: float = Field(default=0.0, description="Flip up-down probability")
    fliplr: float = Field(default=0.5, description="Flip left-right probability")
    mosaic: float = Field(default=1.0, description="Mosaic augmentation probability")
    mixup: float = Field(default=0.0, description="MixUp augmentation probability")
    copy_paste: float = Field(default=0.0, description="Copy-paste augmentation probability")


class YOLOLossConfig(BaseModel):
    """YOLO loss configuration."""
    box: float = Field(default=0.05, description="Box loss gain")
    cls: float = Field(default=0.5, description="Classification loss gain")
    dfl: float = Field(default=1.5, description="DFL loss gain")
    pose: float = Field(default=12.0, description="Pose keypoint loss gain")
    kobj: float = Field(default=1.0, description="Keypoint loss gain")
    label_smoothing: float = Field(default=0.0, description="Label smoothing")


class YOLOOptimizerConfig(BaseModel):
    """YOLO optimizer configuration."""
    type: Literal["SGD", "Adam", "AdamW"] = Field(default="SGD", description="Optimizer type")
    nesterov: bool = Field(default=False, description="Use Nesterov momentum")
    beta1: float = Field(default=0.937, description="Adam beta1")
    beta2: float = Field(default=0.999, description="Adam beta2")


class YOLOSchedulerConfig(BaseModel):
    """YOLO scheduler configuration."""
    type: Literal["cosine", "linear", "step"] = Field(default="cosine", description="Scheduler type")
    lrf: float = Field(default=0.01, description="Minimum learning rate factor")


class YOLOValidationConfig(BaseModel):
    """YOLO validation configuration."""
    val_freq: int = Field(default=1, description="Validation frequency (epochs)")
    save_best: bool = Field(default=True, description="Save best model")
    save_last: bool = Field(default=True, description="Save last model")
    save_period: int = Field(default=-1, description="Save period")


class YOLOOutputConfig(BaseModel):
    """YOLO output configuration."""
    project: str = Field(default="volleyball_analytics", description="Project name")
    name: str = Field(default="yolo_training", description="Experiment name")
    save_dir: str = Field(default="runs/train", description="Output directory")
    save: bool = Field(default=True, description="Save results")
    save_conf: bool = Field(default=True, description="Save confusion matrix")
    save_txt: bool = Field(default=False, description="Save predictions")
    save_labels: bool = Field(default=True, description="Save labels")
    save_crop: bool = Field(default=False, description="Save crop")


class YOLOHardwareConfig(BaseModel):
    """YOLO hardware configuration."""
    device: str = Field(default="auto", description="Device (cpu, 0, 1, etc.)")
    workers: int = Field(default=8, description="Number of workers")
    amp: bool = Field(default=True, description="Mixed precision training")
    half: bool = Field(default=False, description="Half precision")


class YOLOTrainingConfig(BaseModel):
    """Complete YOLO training configuration."""
    model: YOLOModelConfig = Field(default_factory=YOLOModelConfig)
    dataset: YOLODatasetConfig = Field(description="Dataset configuration")
    training: YOLOTrainingParams = Field(default_factory=YOLOTrainingParams)
    augmentation: YOLOAugmentationConfig = Field(default_factory=YOLOAugmentationConfig)
    loss: YOLOLossConfig = Field(default_factory=YOLOLossConfig)
    optimizer: YOLOOptimizerConfig = Field(default_factory=YOLOOptimizerConfig)
    scheduler: YOLOSchedulerConfig = Field(default_factory=YOLOSchedulerConfig)
    validation: YOLOValidationConfig = Field(default_factory=YOLOValidationConfig)
    output: YOLOOutputConfig = Field(default_factory=YOLOOutputConfig)
    hardware: YOLOHardwareConfig = Field(default_factory=YOLOHardwareConfig)
    
    def to_yolo_args(self) -> List[str]:
        """Convert configuration to YOLO command-line arguments."""
        args = []
        
        # Model args
        args.extend(["--model", f"yolov8{self.model.size}.pt"])
        args.extend(["--data", self.dataset.data])
        args.extend(["--epochs", str(self.model.epochs)])
        args.extend(["--imgsz", str(self.model.imgsz)])
        args.extend(["--batch", str(self.model.batch)])
        
        # Training args
        args.extend(["--lr0", str(self.training.lr0)])
        args.extend(["--lrf", str(self.training.lrf)])
        args.extend(["--momentum", str(self.training.momentum)])
        args.extend(["--weight-decay", str(self.training.weight_decay)])
        args.extend(["--warmup-epochs", str(self.training.warmup_epochs)])
        
        # Hardware args
        args.extend(["--device", self.hardware.device])
        args.extend(["--workers", str(self.hardware.workers)])
        if self.hardware.amp:
            args.append("--amp")
        if self.hardware.half:
            args.append("--half")
        
        # Output args
        args.extend(["--project", self.output.project])
        args.extend(["--name", self.output.name])
        args.extend(["--save-dir", self.output.save_dir])
        
        return args
    
    def get_summary(self) -> str:
        """Get a summary of the configuration."""
        return f"""YOLO Training Configuration:
Model: YOLOv8{self.model.size} ({self.model.type.value})
Classes: {self.model.num_classes}
Epochs: {self.model.epochs}
Batch Size: {self.model.batch}
Image Size: {self.model.imgsz}
Learning Rate: {self.training.lr0}
Device: {self.hardware.device}
Output: {self.output.project}/{self.output.name}"""

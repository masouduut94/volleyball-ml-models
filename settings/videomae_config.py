"""
VideoMAE training configuration.

This module provides a Pydantic-based configuration class for VideoMAE training
parameters, with validation and default values.
"""

from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field


class VideoMAEModelConfig(BaseModel):
    """VideoMAE model configuration."""
    model_name: str = Field(default="MCG-NJU/videomae-base", description="HuggingFace model name")
    num_classes: int = Field(default=3, description="Number of classes")
    image_size: int = Field(default=224, description="Input image size")
    num_frames: int = Field(default=16, description="Number of frames per video")
    frame_interval: int = Field(default=16, description="Frame sampling interval")
    dropout_rate: float = Field(default=0.1, description="Dropout rate")
    attention_dropout: float = Field(default=0.0, description="Attention dropout rate")


class VideoMAEDatasetConfig(BaseModel):
    """VideoMAE dataset configuration."""
    type: str = Field(default="custom", description="Dataset type")
    data_dir: str = Field(description="Path to dataset directory")
    train_split: float = Field(default=0.8, description="Training split ratio")
    video_extensions: List[str] = Field(default=[".mp4", ".avi", ".mov", ".mkv"], description="Supported video extensions")
    max_duration: float = Field(default=30.0, description="Maximum video duration in seconds")
    min_duration: float = Field(default=2.0, description="Minimum video duration in seconds")
    num_workers: int = Field(default=4, description="Number of data loading workers")
    cache_dir: Optional[str] = Field(default=None, description="Cache directory for processed videos")


class VideoMAETrainingParams(BaseModel):
    """VideoMAE training parameters."""
    num_epochs: int = Field(default=100, description="Number of training epochs")
    batch_size: int = Field(default=8, description="Training batch size")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    weight_decay: float = Field(default=0.01, description="Weight decay")
    warmup_steps: int = Field(default=1000, description="Warmup steps")
    max_grad_norm: float = Field(default=1.0, description="Maximum gradient norm")
    gradient_accumulation_steps: int = Field(default=1, description="Gradient accumulation steps")


class VideoMAEOptimizerConfig(BaseModel):
    """VideoMAE optimizer configuration."""
    type: Literal["AdamW", "Adam", "SGD"] = Field(default="AdamW", description="Optimizer type")
    beta1: float = Field(default=0.9, description="Adam beta1")
    beta2: float = Field(default=0.999, description="Adam beta2")
    epsilon: float = Field(default=1e-8, description="Adam epsilon")


class VideoMAESchedulerConfig(BaseModel):
    """VideoMAE scheduler configuration."""
    type: Literal["linear", "cosine", "polynomial"] = Field(default="linear", description="Scheduler type")
    num_warmup_steps: int = Field(default=1000, description="Number of warmup steps")
    num_training_steps: Optional[int] = Field(default=None, description="Total number of training steps")


class VideoMAEValidationConfig(BaseModel):
    """VideoMAE validation configuration."""
    eval_strategy: Literal["no", "steps", "epoch"] = Field(default="epoch", description="Evaluation strategy")
    eval_steps: Optional[int] = Field(default=None, description="Evaluation steps (if strategy is 'steps')")
    save_strategy: Literal["no", "steps", "epoch"] = Field(default="epoch", description="Save strategy")
    save_steps: Optional[int] = Field(default=None, description="Save steps (if strategy is 'steps')")
    save_total_limit: Optional[int] = Field(default=None, description="Maximum number of checkpoints to save")
    load_best_model_at_end: bool = Field(default=True, description="Load best model at end of training")
    metric_for_best_model: str = Field(default="eval_loss", description="Metric for best model selection")
    greater_is_better: bool = Field(default=False, description="Whether greater metric values are better")


class VideoMAEOutputConfig(BaseModel):
    """VideoMAE output configuration."""
    output_dir: str = Field(default="./output", description="Output directory")
    logging_dir: Optional[str] = Field(default=None, description="Logging directory")
    logging_steps: int = Field(default=100, description="Logging frequency")
    report_to: List[str] = Field(default=["tensorboard"], description="Reporting integrations")
    run_name: Optional[str] = Field(default=None, description="Run name for tracking")


class VideoMAEHardwareConfig(BaseModel):
    """VideoMAE hardware configuration."""
    device: str = Field(default="auto", description="Device (cpu, cuda, etc.)")
    fp16: bool = Field(default=False, description="Use mixed precision training")
    bf16: bool = Field(default=False, description="Use bfloat16 precision")
    dataloader_pin_memory: bool = Field(default=True, description="Pin memory for data loading")
    dataloader_num_workers: int = Field(default=4, description="Number of data loading workers")


class VideoMAETrainingConfig(BaseModel):
    """Complete VideoMAE training configuration."""
    model: VideoMAEModelConfig = Field(default_factory=VideoMAEModelConfig)
    dataset: VideoMAEDatasetConfig = Field(description="Dataset configuration")
    training: VideoMAETrainingParams = Field(default_factory=VideoMAETrainingParams)
    optimizer: VideoMAEOptimizerConfig = Field(default_factory=VideoMAEOptimizerConfig)
    scheduler: VideoMAESchedulerConfig = Field(default_factory=VideoMAESchedulerConfig)
    validation: VideoMAEValidationConfig = Field(default_factory=VideoMAEValidationConfig)
    output: VideoMAEOutputConfig = Field(default_factory=VideoMAEOutputConfig)
    hardware: VideoMAEHardwareConfig = Field(default_factory=VideoMAEHardwareConfig)
    
    def to_training_args(self) -> dict:
        """Convert configuration to training arguments dictionary."""
        return {
            "output_dir": self.output.output_dir,
            "num_train_epochs": self.training.num_epochs,
            "per_device_train_batch_size": self.training.batch_size,
            "per_device_eval_batch_size": self.training.batch_size,
            "learning_rate": self.training.learning_rate,
            "weight_decay": self.training.weight_decay,
            "warmup_steps": self.training.warmup_steps,
            "max_grad_norm": self.training.max_grad_norm,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "eval_strategy": self.validation.eval_strategy,
            "eval_steps": self.validation.eval_steps,
            "save_strategy": self.validation.save_strategy,
            "save_steps": self.validation.save_steps,
            "save_total_limit": self.validation.save_total_limit,
            "load_best_model_at_end": self.validation.load_best_model_at_end,
            "metric_for_best_model": self.validation.metric_for_best_model,
            "greater_is_better": self.validation.greater_is_better,
            "logging_dir": self.output.logging_dir,
            "logging_steps": self.output.logging_steps,
            "report_to": self.output.report_to,
            "run_name": self.output.run_name,
            "fp16": self.hardware.fp16,
            "bf16": self.hardware.bf16,
            "dataloader_pin_memory": self.hardware.dataloader_pin_memory,
            "dataloader_num_workers": self.hardware.dataloader_num_workers,
        }
    
    def get_summary(self) -> str:
        """Get a summary of the configuration."""
        return f"""VideoMAE Training Configuration:
Model: {self.model.model_name}
Classes: {self.model.num_classes}
Epochs: {self.training.num_epochs}
Batch Size: {self.training.batch_size}
Image Size: {self.model.image_size}
Frames: {self.model.num_frames}
Learning Rate: {self.training.learning_rate}
Device: {self.hardware.device}
Output: {self.output.output_dir}"""

"""
Unified Trainer for ML Manager.

This module provides a Trainer class that can handle training for both YOLO and VideoMAE models
using configuration files and the existing training patterns.
"""

import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any
from transformers import TrainingArguments, Trainer
from ultralytics import YOLO

from ..settings import YOLOTrainingConfig, VideoMAETrainingConfig
from .utils import (
    create_videomae_datasets, create_videomae_model, create_videomae_collate_fn,
    compute_videomae_metrics, create_yolo_training_args, validate_dataset_path,
    save_training_results, get_optimal_batch_size
)


class UnifiedTrainer:
    """
    Unified trainer for YOLO and VideoMAE models.
    
    This class can train both types of models using their respective configurations.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the trainer.
        
        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the trainer."""
        logger = logging.getLogger("MLManagerTrainer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger
    
    def train_yolo(self, config: YOLOTrainingConfig, resume: bool = False) -> bool:
        """
        Train a YOLO model using the provided configuration.
        
        Args:
            config: YOLO training configuration
            resume: Whether to resume from a previous checkpoint
            
        Returns:
            True if training was successful, False otherwise
        """
        try:
            self.logger.info("Starting YOLO training...")
            self.logger.info(config.get_summary())
            
            # Validate dataset path
            if not validate_dataset_path(config.dataset.data):
                self.logger.error(f"Invalid dataset path: {config.dataset.data}")
                return False
            
            # Get optimal batch size if not specified
            if config.model.batch == 16:  # Default value
                optimal_batch = get_optimal_batch_size(
                    config.model.size, 
                    config.model.type.value, 
                    config.hardware.device
                )
                config.model.batch = optimal_batch
            
            # Create YOLO model
            model_name = f"yolov8{config.model.size}.pt"
            if config.model.type.value == "segment":
                model_name = f"yolov8{config.model.size}-seg.pt"
            elif config.model.type.value == "pose":
                model_name = f"yolov8{config.model.size}-pose.pt"
            
            self.logger.info(f"Loading YOLO model: {model_name}")
            model = YOLO(model_name)
            
            # Create training arguments
            train_args = create_yolo_training_args(
                model_size=config.model.size,
                task=config.model.type.value,
                data_yaml=config.dataset.data,
                epochs=config.model.epochs,
                batch_size=config.model.batch,
                imgsz=config.model.imgsz,
                device=config.hardware.device,
                optimizer=config.optimizer.type,
                learning_rate=config.training.lr0,
                final_lr_factor=config.training.lrf,
                workers=config.hardware.workers,
                cos_lr=True,
                plots=True,
                seed=1368
            )
            
            # Add resume checkpoint if requested
            if resume:
                # Look for latest checkpoint
                runs_dir = Path(config.output.save_dir)
                if runs_dir.exists():
                    checkpoints = list(runs_dir.rglob("*.pt"))
                    if checkpoints:
                        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                        train_args['resume'] = str(latest_checkpoint)
                        self.logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
            
            self.logger.info(f"Training arguments: {train_args}")
            
            # Start training
            results = model.train(**train_args)
            
            # Save results
            save_training_results(
                results, 
                config.output.save_dir, 
                f"yolov8{config.model.size}_{config.model.type.value}"
            )
            
            self.logger.info("YOLO training completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"YOLO training failed: {e}")
            return False
    
    def train_videomae(self, config: VideoMAETrainingConfig, resume: bool = False) -> bool:
        """
        Train a VideoMAE model using the provided configuration.
        
        Args:
            config: VideoMAE training configuration
            resume: Whether to resume from a previous checkpoint
            
        Returns:
            True if training was successful, False otherwise
        """
        try:
            self.logger.info("Starting VideoMAE training...")
            self.logger.info(config.get_summary())
            
            # Validate dataset path
            if not validate_dataset_path(config.dataset.data_dir):
                self.logger.error(f"Invalid dataset path: {config.dataset.data_dir}")
                return False
            
            # Create datasets
            self.logger.info("Creating VideoMAE datasets...")
            train_dataset, val_dataset, class_labels, label2id, id2label = create_videomae_datasets(
                config.dataset.data_dir,
                config.model.num_frames,
                sample_rate=16,  # Default sample rate
                fps=30
            )
            
            # Create model and processor
            self.logger.info("Creating VideoMAE model...")
            model, processor = create_videomae_model(
                config.model.model_name,
                label2id,
                id2label
            )
            
            # Create training arguments
            training_args = TrainingArguments(
                output_dir=config.output.output_dir,
                num_train_epochs=config.training.num_epochs,
                per_device_train_batch_size=config.training.batch_size,
                per_device_eval_batch_size=config.training.batch_size,
                learning_rate=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                warmup_steps=config.training.warmup_steps,
                max_grad_norm=config.training.max_grad_norm,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                eval_strategy=config.validation.eval_strategy,
                eval_steps=config.validation.eval_steps,
                save_strategy=config.validation.save_strategy,
                save_steps=config.validation.save_steps,
                save_total_limit=config.validation.save_total_limit,
                load_best_model_at_end=config.validation.load_best_model_at_end,
                metric_for_best_model=config.validation.metric_for_best_model,
                greater_is_better=config.validation.greater_is_better,
                logging_dir=config.output.logging_dir,
                logging_steps=config.output.logging_steps,
                report_to=config.output.report_to,
                run_name=config.output.run_name,
                fp16=config.hardware.fp16,
                bf16=config.hardware.bf16,
                dataloader_pin_memory=config.hardware.dataloader_pin_memory,
                dataloader_num_workers=config.hardware.dataloader_num_workers,
                remove_unused_columns=False,
                push_to_hub=False,
                seed=1368
            )
            
            # Add resume checkpoint if requested
            if resume:
                # Look for latest checkpoint
                output_dir = Path(config.output.output_dir)
                if output_dir.exists():
                    checkpoints = list(output_dir.glob("checkpoint-*"))
                    if checkpoints:
                        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
                        training_args.resume_from_checkpoint = str(latest_checkpoint)
                        self.logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
            
            # Create HuggingFace trainer
            hf_trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=processor,
                compute_metrics=lambda pred: compute_videomae_metrics(pred, class_labels),
                data_collator=create_videomae_collate_fn(),
            )
            
            # Start training
            self.logger.info("Starting VideoMAE training...")
            train_results = hf_trainer.train()
            
            # Save model
            hf_trainer.save_model()
            processor.save_pretrained(config.output.output_dir)
            
            # Final evaluation
            test_results = hf_trainer.evaluate(val_dataset)
            hf_trainer.log_metrics("test", test_results)
            hf_trainer.save_metrics("test", test_results)
            hf_trainer.save_state()
            
            # Save results
            save_training_results(
                test_results, 
                config.output.output_dir, 
                "videomae_model"
            )
            
            self.logger.info("VideoMAE training completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"VideoMAE training failed: {e}")
            return False
    
    def train(self, 
              config: Union[YOLOTrainingConfig, VideoMAETrainingConfig], 
              resume: bool = False) -> bool:
        """
        Train a model using the provided configuration.
        
        Args:
            config: Training configuration (YOLO or VideoMAE)
            resume: Whether to resume from a previous checkpoint
            
        Returns:
            True if training was successful, False otherwise
        """
        if isinstance(config, YOLOTrainingConfig):
            return self.train_yolo(config, resume)
        elif isinstance(config, VideoMAETrainingConfig):
            return self.train_videomae(config, resume)
        else:
            raise ValueError(f"Unsupported configuration type: {type(config)}")
    
    @staticmethod
    def validate_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a training configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with validation results
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            return {
                "valid": False,
                "error": f"Configuration file not found: {config_path}"
            }
        
        try:
            # Try to load the configuration
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                import yaml
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Determine config type based on content
                if "model" in config_data and "model_name" in config_data.get("model", {}):
                    if "videomae" in config_data["model"]["model_name"].lower():
                        config_type = "VideoMAE"
                        config = VideoMAETrainingConfig(**config_data)
                    else:
                        config_type = "YOLO"
                        config = YOLOTrainingConfig(**config_data)
                else:
                    config_type = "YOLO"
                    config = YOLOTrainingConfig(**config_data)
                
                return {
                    "valid": True,
                    "config_type": config_type,
                    "config": config,
                    "summary": config.get_summary()
                }
            else:
                return {
                    "valid": False,
                    "error": f"Unsupported file format: {config_path.suffix}"
                }
                
        except Exception as e:
            return {
                "valid": False,
                "error": f"Configuration validation failed: {e}"
            }
    
    def quick_train_yolo(self, 
                         data_yaml: str,
                         model_size: str = "n",
                         task: str = "detect",
                         epochs: int = 100,
                         batch_size: Optional[int] = None,
                         device: str = "0") -> bool:
        """
        Quick YOLO training with minimal configuration.
        
        Args:
            data_yaml: Path to data YAML file
            model_size: YOLO model size (n, s, m, l, x)
            task: Training task (detect, segment, pose)
            epochs: Number of epochs
            batch_size: Batch size (auto-determined if None)
            device: Device to use
            
        Returns:
            True if training was successful, False otherwise
        """
        try:
            self.logger.info(f"Starting quick YOLO training: {model_size} {task}")
            
            # Auto-determine batch size if not specified
            if batch_size is None:
                batch_size = get_optimal_batch_size(model_size, task, device)
            
            # Create minimal config
            config = YOLOTrainingConfig(
                dataset=YOLOTrainingConfig.__fields__['dataset'].type(
                    data=data_yaml
                ),
                model=YOLOTrainingConfig.__fields__['model'].type(
                    size=model_size,
                    type=task,
                    epochs=epochs,
                    batch=batch_size
                ),
                hardware=YOLOTrainingConfig.__fields__['hardware'].type(
                    device=device
                )
            )
            
            return self.train_yolo(config, resume=False)
            
        except Exception as e:
            self.logger.error(f"Quick YOLO training failed: {e}")
            return False
    
    def quick_train_videomae(self,
                             data_dir: str,
                             num_classes: int = 3,
                             epochs: int = 100,
                             batch_size: int = 8,
                             device: str = "auto") -> bool:
        """
        Quick VideoMAE training with minimal configuration.
        
        Args:
            data_dir: Path to dataset directory
            num_classes: Number of classes
            epochs: Number of epochs
            batch_size: Batch size
            device: Device to use
            
        Returns:
            True if training was successful, False otherwise
        """
        try:
            self.logger.info(f"Starting quick VideoMAE training: {num_classes} classes")
            
            # Create minimal config
            config = VideoMAETrainingConfig(
                dataset=VideoMAETrainingConfig.__fields__['dataset'].type(
                    data_dir=data_dir
                ),
                model=VideoMAETrainingConfig.__fields__['model'].type(
                    num_classes=num_classes
                ),
                training=VideoMAETrainingConfig.__fields__['training'].type(
                    num_epochs=epochs,
                    batch_size=batch_size
                ),
                hardware=VideoMAETrainingConfig.__fields__['hardware'].type(
                    device=device
                )
            )
            
            return self.train_videomae(config, resume=False)
            
        except Exception as e:
            self.logger.error(f"Quick VideoMAE training failed: {e}")
            return False

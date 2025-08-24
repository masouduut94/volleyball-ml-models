#!/usr/bin/env python3
"""
Main training script for ML Manager.

This script provides a command-line interface for training both YOLO and VideoMAE models
using configuration files or quick training options.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .training.trainer import UnifiedTrainer
from .settings import YOLOTrainingConfig, VideoMAETrainingConfig


def create_yolo_config_template(output_path: Path) -> None:
    """Create a YOLO training configuration template."""
    config = YOLOTrainingConfig()

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to YAML
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False, indent=2, sort_keys=False)

    print(f"YOLO configuration template created at: {output_path}")


def create_videomae_config_template(output_path: Path) -> None:
    """Create a VideoMAE training configuration template."""
    config = VideoMAETrainingConfig()

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to YAML
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False, indent=2, sort_keys=False)

    print(f"VideoMAE configuration template created at: {output_path}")


def main():
    """Main function for the training script."""
    parser = argparse.ArgumentParser(
        description="Train YOLO or VideoMAE models using configuration files or quick training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train YOLO model with configuration file
  python -m src.ml_manager.train --config configs/yolo_config.yaml
  
  # Train VideoMAE model with configuration file
  python -m src.ml_manager.train --config configs/videomae_config.yaml
  
  # Quick YOLO training (detection)
  python -m src.ml_manager.train --quick-yolo data.yaml --task detect --size n --epochs 100
  
  # Quick YOLO training (segmentation)
  python -m src.ml_manager.train --quick-yolo data.yaml --task segment --size m --epochs 50
  
  # Quick VideoMAE training
  python -m src.ml_manager.train --quick-videomae data_dir --classes 3 --epochs 100
  
  # Resume training
  python -m src.ml_manager.train --config configs/yolo_config.yaml --resume
  
  # Create configuration templates
  python -m src.ml_manager.train --create-template yolo --output configs/yolo_template.yaml
  python -m src.ml_manager.train --create-template videomae --output configs/videomae_template.yaml
        """
    )

    # Configuration file training
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to training configuration file (YAML)"
    )

    # Quick training options
    parser.add_argument(
        "--quick-yolo",
        type=str,
        help="Quick YOLO training with data YAML path"
    )

    parser.add_argument(
        "--quick-videomae",
        type=str,
        help="Quick VideoMAE training with data directory path"
    )

    # YOLO quick training parameters
    parser.add_argument(
        "--task",
        choices=["detect", "segment", "pose"],
        default="detect",
        help="YOLO task type (default: detect)"
    )

    parser.add_argument(
        "--size",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="YOLO model size (default: n)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (auto-determined if not specified)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (default: 0)"
    )

    # VideoMAE quick training parameters
    parser.add_argument(
        "--classes",
        type=int,
        default=3,
        help="Number of classes for VideoMAE (default: 3)"
    )

    # Common arguments
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume training from checkpoint"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose output"
    )

    # Template creation
    parser.add_argument(
        "--create-template",
        choices=["yolo", "videomae"],
        help="Create a configuration template"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for template creation"
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle template creation
    if args.create_template:
        if not args.output:
            print("Error: --output is required when creating templates")
            sys.exit(1)

        if args.create_template == "yolo":
            create_yolo_config_template(args.output)
        else:
            create_videomae_config_template(args.output)
        return

    # Initialize trainer
    trainer = UnifiedTrainer(verbose=args.verbose)

    # Handle quick training
    if args.quick_yolo:
        print(f"Starting quick YOLO training: {args.task} {args.size}")
        success = trainer.quick_train_yolo(
            data_yaml=args.quick_yolo,
            model_size=args.size,
            task=args.task,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )

        if success:
            print("Quick YOLO training completed successfully!")
            sys.exit(0)
        else:
            print("Quick YOLO training failed!")
            sys.exit(1)

    elif args.quick_videomae:
        print(f"Starting quick VideoMAE training: {args.classes} classes")
        success = trainer.quick_train_videomae(
            data_dir=args.quick_videomae,
            num_classes=args.classes,
            epochs=args.epochs,
            batch_size=args.batch_size or 8,
            device=args.device
        )

        if success:
            print("Quick VideoMAE training completed successfully!")
            sys.exit(0)
        else:
            print("Quick VideoMAE training failed!")
            sys.exit(1)

    # Handle configuration file training
    elif args.config:
        if not args.config.exists():
            print(f"Error: Configuration file not found: {args.config}")
            sys.exit(1)

        # Validate configuration
        print(f"Validating configuration: {args.config}")
        validation_result = trainer.validate_config(args.config)

        if not validation_result["valid"]:
            print(f"Configuration validation failed: {validation_result['error']}")
            sys.exit(1)

        print(f"Configuration is valid ({validation_result['config_type']})")
        print(validation_result["summary"])

        # Start training
        print(f"\nStarting {validation_result['config_type']} training...")
        success = trainer.train(validation_result["config"], resume=args.resume)

        if success:
            print("Training completed successfully!")
            sys.exit(0)
        else:
            print("Training failed!")
            sys.exit(1)

    else:
        print("Error: No training method specified")
        print("Use --config for configuration file training")
        print("Use --quick-yolo for quick YOLO training")
        print("Use --quick-videomae for quick VideoMAE training")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

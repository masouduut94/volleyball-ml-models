#!/usr/bin/env python3
"""
Training examples for ML Manager.

This script demonstrates how to use the training system for both YOLO and VideoMAE models.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import ml_manager
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ml_manager.training import UnifiedTrainer
from src.ml_manager.settings import YOLOTrainingConfig, VideoMAETrainingConfig


def example_yolo_training():
    """Example of YOLO training with configuration."""
    print("=== YOLO Training Example ===")
    
    # Create trainer
    trainer = UnifiedTrainer(verbose=True)
    
    # Example 1: Quick training (minimal configuration)
    print("\n1. Quick YOLO Training (Detection)")
    success = trainer.quick_train_yolo(
        data_yaml="data/processed/ball_detection/data.yaml",
        model_size="n",
        task="detect",
        epochs=50,
        device="0"
    )
    print(f"Quick training {'succeeded' if success else 'failed'}")
    
    # Example 2: Quick training (segmentation)
    print("\n2. Quick YOLO Training (Segmentation)")
    success = trainer.quick_train_yolo(
        data_yaml="data/processed/court_segmentation/data.yaml",
        model_size="m",
        task="segment",
        epochs=40,
        device="0"
    )
    print(f"Quick training {'succeeded' if success else 'failed'}")
    
    # Example 3: Full configuration training
    print("\n3. Full Configuration YOLO Training")
    
    # Create configuration
    config = YOLOTrainingConfig(
        dataset=YOLOTrainingConfig.__fields__['dataset'].type(
            data="data/processed/action_detection/data.yaml"
        ),
        model=YOLOTrainingConfig.__fields__['model'].type(
            size="s",
            type="detect",
            num_classes=6,
            epochs=100,
            batch=24,
            imgsz=640
        ),
        training=YOLOTrainingConfig.__fields__['training'].type(
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005
        ),
        hardware=YOLOTrainingConfig.__fields__['hardware'].type(
            device="0",
            workers=8,
            amp=True
        ),
        output=YOLOTrainingConfig.__fields__['output'].type(
            project="volleyball_analytics",
            name="action_detection_v2",
            save_dir="runs/train"
        )
    )
    
    success = trainer.train_yolo(config, resume=False)
    print(f"Full configuration training {'succeeded' if success else 'failed'}")


def example_videomae_training():
    """Example of VideoMAE training with configuration."""
    print("\n=== VideoMAE Training Example ===")
    
    # Create trainer
    trainer = UnifiedTrainer(verbose=True)
    
    # Example 1: Quick training
    print("\n1. Quick VideoMAE Training")
    success = trainer.quick_train_videomae(
        data_dir="data/processed/game-status",
        num_classes=3,
        epochs=100,
        batch_size=8,
        device="auto"
    )
    print(f"Quick training {'succeeded' if success else 'failed'}")
    
    # Example 2: Full configuration training
    print("\n2. Full Configuration VideoMAE Training")
    
    # Create configuration
    config = VideoMAETrainingConfig(
        dataset=VideoMAETrainingConfig.__fields__['dataset'].type(
            data_dir="data/processed/game-status",
            train_split=0.8,
            video_extensions=[".mp4", ".avi"],
            max_duration=30.0,
            min_duration=2.0
        ),
        model=VideoMAETrainingConfig.__fields__['model'].type(
            model_name="MCG-NJU/videomae-base",
            num_classes=3,
            image_size=224,
            num_frames=16,
            frame_interval=16
        ),
        training=VideoMAETrainingConfig.__fields__['training'].type(
            num_epochs=100,
            batch_size=8,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=1000
        ),
        hardware=VideoMAETrainingConfig.__fields__['hardware'].type(
            device="auto",
            fp16=True,
            dataloader_num_workers=4
        ),
        output=VideoMAETrainingConfig.__fields__['output'].type(
            output_dir="./output/game_status_classifier",
            logging_dir="./logs/game_status",
            report_to=["tensorboard"]
        )
    )
    
    success = trainer.train_videomae(config, resume=False)
    print(f"Full configuration training {'succeeded' if success else 'failed'}")


def example_resume_training():
    """Example of resuming training from checkpoints."""
    print("\n=== Resume Training Example ===")
    
    trainer = UnifiedTrainer(verbose=True)
    
    # Resume YOLO training
    print("\n1. Resume YOLO Training")
    config = YOLOTrainingConfig(
        dataset=YOLOTrainingConfig.__fields__['dataset'].type(
            data="data/processed/ball_detection/data.yaml"
        ),
        output=YOLOTrainingConfig.__fields__['output'].type(
            save_dir="runs/train"
        )
    )
    
    success = trainer.train_yolo(config, resume=True)
    print(f"Resume training {'succeeded' if success else 'failed'}")
    
    # Resume VideoMAE training
    print("\n2. Resume VideoMAE Training")
    config = VideoMAETrainingConfig(
        dataset=VideoMAETrainingConfig.__fields__['dataset'].type(
            data_dir="data/processed/game-status"
        ),
        output=VideoMAETrainingConfig.__fields__['output'].type(
            output_dir="./output/game_status_classifier"
        )
    )
    
    success = trainer.train_videomae(config, resume=True)
    print(f"Resume training {'succeeded' if success else 'failed'}")


def main():
    """Main function to run all examples."""
    print("ML Manager Training Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_yolo_training()
        example_videomae_training()
        example_resume_training()
        
        print("\n" + "=" * 50)
        print("All examples completed!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies and dataset paths configured.")


if __name__ == "__main__":
    main()

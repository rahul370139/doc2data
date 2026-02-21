#!/usr/bin/env python3
"""
YOLO Training Script for CMS-1500 Field Detection

This script trains YOLOv8 to detect form fields on CMS-1500 documents.
Designed to run on DGX GPU cluster.

Usage:
    python scripts/train_yolo_gpu.py --epochs 100 --batch 16
"""

import argparse
import os
from pathlib import Path
import yaml


def create_dataset_yaml(data_dir: Path, output_path: Path, classes: list):
    """Create dataset.yaml for YOLO training."""
    config = {
        'path': str(data_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created dataset config at {output_path}")
    return output_path


def train_yolo(
    data_yaml: str,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 1280,
    model: str = 'yolov8m.pt',
    project: str = 'runs/detect',
    name: str = 'cms1500_gpu',
    device: str = '0',
    workers: int = 8,
    resume: bool = False,
):
    """Train YOLOv8 model on GPU."""
    from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print(f"YOLO Training Configuration")
    print(f"{'='*60}")
    print(f"  Model:        {model}")
    print(f"  Dataset:      {data_yaml}")
    print(f"  Epochs:       {epochs}")
    print(f"  Batch size:   {batch}")
    print(f"  Image size:   {imgsz}")
    print(f"  Device:       {device}")
    print(f"  Workers:      {workers}")
    print(f"{'='*60}\n")
    
    # Load model
    yolo = YOLO(model)
    
    # Training parameters optimized for form detection
    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        
        # Augmentation (conservative for forms - too much rotation hurts)
        degrees=3.0,       # Slight rotation only
        translate=0.05,    # Minor translation
        scale=0.15,        # Minor scale variation
        shear=1.0,         # Minimal shear
        perspective=0.0,   # No perspective (forms are flat)
        flipud=0.0,        # No vertical flip (forms have orientation)
        fliplr=0.0,        # No horizontal flip (forms have orientation)
        mosaic=0.3,        # Reduced mosaic (forms shouldn't be cut)
        mixup=0.0,         # No mixup
        
        # Training params
        lr0=0.01,          # Initial learning rate
        lrf=0.01,          # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=30,       # Early stopping patience
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Other
        plots=True,
        save=True,
        verbose=True,
    )
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best model: {project}/{name}/weights/best.pt")
    print(f"{'='*60}\n")
    
    return results


def validate_dataset(data_dir: Path):
    """Check if dataset structure is valid."""
    required_dirs = [
        data_dir / 'images' / 'train',
        data_dir / 'images' / 'val',
        data_dir / 'labels' / 'train',
        data_dir / 'labels' / 'val',
    ]
    
    for d in required_dirs:
        if not d.exists():
            print(f"⚠️  Missing directory: {d}")
            return False
        
        # Count files
        n_files = len(list(d.glob('*')))
        print(f"  {d.relative_to(data_dir)}: {n_files} files")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for CMS-1500 field detection')
    parser.add_argument('--data-dir', type=str, default='datasets/cms1500_yolo',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size')
    parser.add_argument('--model', type=str, default='yolov8m.pt', help='Base model')
    parser.add_argument('--name', type=str, default='cms1500_gpu', help='Run name')
    parser.add_argument('--device', type=str, default='0', help='GPU device(s)')
    parser.add_argument('--workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    
    print(f"\nProject root: {project_root}")
    print(f"Dataset dir:  {data_dir}")
    
    # Validate dataset
    if not data_dir.exists():
        print(f"\n❌ Dataset directory not found: {data_dir}")
        print("\nTo create training data, you need to:")
        print("  1. Create images/train/ and images/val/ with form images")
        print("  2. Create labels/train/ and labels/val/ with YOLO format labels")
        print("     (one .txt per image with: class_id x_center y_center width height)")
        return
    
    print("\nValidating dataset structure...")
    if not validate_dataset(data_dir):
        print("\n❌ Dataset validation failed")
        return
    
    # Check/create dataset.yaml
    dataset_yaml = data_dir / 'dataset.yaml'
    if not dataset_yaml.exists():
        classes = ['field', 'table', 'checkbox', 'header', 'signature']
        create_dataset_yaml(data_dir, dataset_yaml, classes)
    
    # Train
    train_yolo(
        data_yaml=str(dataset_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        model=args.model,
        project=str(project_root / 'runs' / 'detect'),
        name=args.name,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
    )


if __name__ == '__main__':
    main()


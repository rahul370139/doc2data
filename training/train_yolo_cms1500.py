"""
Fine-tune YOLOv8 for CMS-1500 layout detection.

This script provides comprehensive training options:
- Multi-scale training for better small object detection
- Data augmentation presets for form documents
- Learning rate scheduling
- Model selection (nano to xlarge)
- Export to multiple formats

Dataset structure (YOLO format):
    dataset/
      images/train/*.jpg|png
      images/val/*.jpg|png
      labels/train/*.txt
      labels/val/*.txt

Each label file: <class_id> <cx> <cy> <w> <h> (normalized 0-1).

Recommended classes for CMS-1500:
    0: field    - Text entry fields
    1: table    - Service line tables
    2: checkbox - Checkbox fields
    3: header   - Section headers
    4: signature - Signature lines

Usage:
    # Basic training
    python training/train_yolo_cms1500.py --data datasets/cms1500_yolo/dataset.yaml --epochs 100

    # Advanced training with multi-scale
    python training/train_yolo_cms1500.py \\
        --data datasets/cms1500_yolo/dataset.yaml \\
        --epochs 100 \\
        --model yolov8m.pt \\
        --imgsz 1280 \\
        --batch 8 \\
        --multi-scale \\
        --name cms1500_yolo_v1

    # Quick test run
    python training/train_yolo_cms1500.py --data datasets/cms1500_yolo/dataset.yaml --epochs 5 --model yolov8n.pt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for CMS-1500 layout detection")
    
    # Required
    parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (-1 for auto)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size (higher = better for forms)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    # Model selection
    parser.add_argument("--model", default="yolov8m.pt", 
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                        help="Base YOLOv8 model (n=nano, s=small, m=medium, l=large, x=xlarge)")
    
    # Advanced training
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lr-final", type=float, default=0.0001, help="Final learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--multi-scale", action="store_true", help="Enable multi-scale training")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic in last N epochs")
    
    # Augmentation
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV-Hue augmentation")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV-Saturation augmentation")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV-Value augmentation")
    parser.add_argument("--degrees", type=float, default=5.0, help="Rotation degrees")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation fraction")
    parser.add_argument("--scale", type=float, default=0.2, help="Scale augmentation")
    parser.add_argument("--shear", type=float, default=2.0, help="Shear degrees")
    parser.add_argument("--perspective", type=float, default=0.0001, help="Perspective augmentation")
    parser.add_argument("--flipud", type=float, default=0.0, help="Flip up-down probability")
    parser.add_argument("--fliplr", type=float, default=0.0, help="Flip left-right probability")
    parser.add_argument("--mosaic", type=float, default=0.5, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.1, help="Mixup augmentation probability")
    
    # Output
    parser.add_argument("--name", default=None, help="Run name (default: cms1500_yolo_TIMESTAMP)")
    parser.add_argument("--project", default="runs/detect", help="Project directory")
    parser.add_argument("--device", default="", help="Device (cuda:0, cpu, or empty for auto)")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    
    # Extras
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing run")
    parser.add_argument("--export", action="store_true", help="Export best model to ONNX after training")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check ultralytics
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "❌ ultralytics not installed.\n"
            "Install with: pip install ultralytics\n"
            "Or for full support: pip install ultralytics[export]"
        ) from exc
    
    # Generate run name
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"cms1500_yolo_{timestamp}"
    
    # Validate dataset
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"❌ Dataset not found: {data_path}")
    
    print("=" * 60)
    print("🏥 CMS-1500 YOLOv8 Training")
    print("=" * 60)
    print(f"  Dataset: {data_path}")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Run Name: {args.name}")
    print("=" * 60)
    
    # Load model
    model = YOLO(args.model)
    
    # Training arguments
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "name": args.name,
        "project": args.project,
        "device": args.device if args.device else None,
        "workers": args.workers,
        "exist_ok": args.exist_ok,
        "resume": args.resume,
        "verbose": args.verbose,
        "pretrained": True,
        "optimizer": "AdamW",
        
        # Learning rate
        "lr0": args.lr,
        "lrf": args.lr_final / args.lr,  # Final LR as fraction of initial
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        
        # Augmentation - tuned for form documents
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "close_mosaic": args.close_mosaic,
        
        # Detection specific
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.1,
        "val": True,
        "plots": True,
        "save": True,
        "save_period": -1,  # Save only best
    }
    
    # Multi-scale training
    if args.multi_scale:
        train_args["multi_scale"] = True
        print("📐 Multi-scale training enabled")
    
    # Train
    print("\n🚀 Starting training...\n")
    results = model.train(**train_args)
    
    # Output paths
    run_dir = Path(args.project) / args.name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    
    if best_weights.exists():
        print(f"  Best weights: {best_weights}")
    if last_weights.exists():
        print(f"  Last weights: {last_weights}")
    print(f"  Results: {run_dir}")
    
    # Export to ONNX if requested
    if args.export and best_weights.exists():
        print("\n📦 Exporting to ONNX...")
        try:
            export_model = YOLO(str(best_weights))
            export_model.export(format="onnx", imgsz=args.imgsz)
            print(f"  ONNX exported: {run_dir / 'weights' / 'best.onnx'}")
        except Exception as e:
            print(f"  ⚠️ ONNX export failed: {e}")
    
    # Usage instructions
    print("\n" + "=" * 60)
    print("📌 Next Steps:")
    print("=" * 60)
    print(f"""
1. Enable in your app:
   export YOLO_MODEL_PATH={best_weights}
   export YOLO_CONFIDENCE=0.25
   export YOLO_IOU=0.6

2. Or update utils/config.py:
   YOLO_MODEL_PATH = "{best_weights}"

3. Run the app:
   streamlit run app/streamlit_main.py

4. Test inference:
   from ultralytics import YOLO
   model = YOLO("{best_weights}")
   results = model.predict("path/to/cms1500.pdf", imgsz={args.imgsz})
""")


if __name__ == "__main__":
    main()

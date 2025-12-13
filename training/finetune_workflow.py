#!/usr/bin/env python3
"""
Complete CMS-1500 Fine-Tuning Workflow

This script orchestrates the entire fine-tuning process:
1. Collect and prepare training data from sample documents
2. Auto-label using OCR or schema-based methods
3. Train YOLOv8 model for layout detection
4. Evaluate on validation set
5. Deploy the fine-tuned model

Usage:
    # Full workflow (prepare + train + evaluate)
    python training/finetune_workflow.py --input data/sample_docs/cms1500_test/ --epochs 100

    # Prepare data only
    python training/finetune_workflow.py --input data/sample_docs/cms1500_test/ --prepare-only

    # Train only (using existing dataset)
    python training/finetune_workflow.py --train-only --dataset datasets/cms1500_yolo/dataset.yaml

    # Evaluate only
    python training/finetune_workflow.py --eval-only --model runs/detect/cms1500_yolo/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class FineTuneWorkflow:
    """Orchestrates the complete CMS-1500 fine-tuning workflow."""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        epochs: int = 100,
        batch_size: int = 8,
        image_size: int = 1280,
        model_base: str = "yolov8m.pt",
        val_split: float = 0.2,
        num_augments: int = 3,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.model_base = model_base
        self.val_split = val_split
        self.num_augments = num_augments
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"cms1500_yolo_{self.timestamp}"
        
        # Paths
        self.dataset_dir = output_dir / "datasets" / "cms1500_yolo"
        self.runs_dir = output_dir / "runs" / "detect"
        self.logs_dir = output_dir / "logs"
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
        print(f"[{timestamp}] {prefix} {message}")
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and capture output."""
        self.log(f"Running: {description}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                self.log(f"Command failed: {result.stderr}", "ERROR")
                return False
            return True
        except Exception as e:
            self.log(f"Command error: {e}", "ERROR")
            return False
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        self.log("Checking dependencies...")
        
        required = ["ultralytics", "cv2", "paddleocr"]
        missing = []
        
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        
        if missing:
            self.log(f"Missing packages: {missing}", "WARNING")
            self.log("Install with: pip install ultralytics opencv-python paddleocr")
            return False
        
        self.log("All dependencies available", "SUCCESS")
        return True
    
    def prepare_dataset(self) -> Optional[Path]:
        """Prepare YOLOv8 dataset from input documents."""
        self.log("=" * 60)
        self.log("STEP 1: Preparing Dataset")
        self.log("=" * 60)
        
        if not self.input_dir.exists():
            self.log(f"Input directory not found: {self.input_dir}", "ERROR")
            return None
        
        # Count input files
        docs = list(self.input_dir.glob("*.pdf")) + list(self.input_dir.glob("*.png")) + list(self.input_dir.glob("*.jpg"))
        if not docs:
            self.log(f"No documents found in {self.input_dir}", "ERROR")
            return None
        
        self.log(f"Found {len(docs)} documents")
        
        # Run prepare_dataset.py
        cmd = [
            sys.executable, str(PROJECT_ROOT / "training" / "prepare_dataset.py"),
            "--input", str(self.input_dir),
            "--output", str(self.dataset_dir),
            "--mode", "ocr",
            "--val-split", str(self.val_split),
            "--num-augments", str(self.num_augments),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode != 0:
            self.log("Dataset preparation failed", "ERROR")
            return None
        
        dataset_yaml = self.dataset_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            self.log("dataset.yaml not created", "ERROR")
            return None
        
        self.log(f"Dataset prepared: {dataset_yaml}", "SUCCESS")
        return dataset_yaml
    
    def train_model(self, dataset_yaml: Path) -> Optional[Path]:
        """Train YOLOv8 model."""
        self.log("=" * 60)
        self.log("STEP 2: Training YOLOv8 Model")
        self.log("=" * 60)
        
        self.log(f"Dataset: {dataset_yaml}")
        self.log(f"Base model: {self.model_base}")
        self.log(f"Epochs: {self.epochs}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Image size: {self.image_size}")
        
        cmd = [
            sys.executable, str(PROJECT_ROOT / "training" / "train_yolo_cms1500.py"),
            "--data", str(dataset_yaml),
            "--epochs", str(self.epochs),
            "--batch", str(self.batch_size),
            "--imgsz", str(self.image_size),
            "--model", self.model_base,
            "--name", self.run_name,
            "--multi-scale",
        ]
        
        # Run training (this will take a while)
        self.log("Starting training (this may take a while)...")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        
        # Check for best weights
        best_weights = self.runs_dir / self.run_name / "weights" / "best.pt"
        if not best_weights.exists():
            # Try default location
            best_weights = PROJECT_ROOT / "runs" / "detect" / self.run_name / "weights" / "best.pt"
        
        if best_weights.exists():
            self.log(f"Training complete: {best_weights}", "SUCCESS")
            return best_weights
        else:
            self.log("Training completed but weights not found", "WARNING")
            return None
    
    def evaluate_model(self, model_path: Path, dataset_yaml: Path) -> Dict:
        """Evaluate trained model on validation set."""
        self.log("=" * 60)
        self.log("STEP 3: Evaluating Model")
        self.log("=" * 60)
        
        try:
            from ultralytics import YOLO
            
            model = YOLO(str(model_path))
            results = model.val(data=str(dataset_yaml), imgsz=self.image_size)
            
            metrics = {
                "mAP50": float(results.box.map50),
                "mAP50-95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
            }
            
            self.log(f"mAP50: {metrics['mAP50']:.4f}", "SUCCESS")
            self.log(f"mAP50-95: {metrics['mAP50-95']:.4f}")
            self.log(f"Precision: {metrics['precision']:.4f}")
            self.log(f"Recall: {metrics['recall']:.4f}")
            
            return metrics
        except Exception as e:
            self.log(f"Evaluation failed: {e}", "ERROR")
            return {}
    
    def deploy_model(self, model_path: Path) -> bool:
        """Deploy the trained model by updating config."""
        self.log("=" * 60)
        self.log("STEP 4: Deploying Model")
        self.log("=" * 60)
        
        # Create deployment script
        deploy_script = f"""#!/bin/bash
# CMS-1500 YOLO Model Deployment
# Generated: {datetime.now().isoformat()}

export YOLO_MODEL_PATH="{model_path}"
export YOLO_CONFIDENCE=0.25
export YOLO_IOU=0.6

echo "Model deployed: $YOLO_MODEL_PATH"
echo "Run: streamlit run app/streamlit_main.py"
"""
        
        deploy_path = PROJECT_ROOT / "deploy_yolo.sh"
        with open(deploy_path, "w") as f:
            f.write(deploy_script)
        os.chmod(deploy_path, 0o755)
        
        self.log(f"Deployment script created: {deploy_path}", "SUCCESS")
        self.log("")
        self.log("To use the fine-tuned model, run:")
        self.log(f"  source {deploy_path}")
        self.log("  streamlit run app/streamlit_main.py")
        self.log("")
        self.log("Or set environment variables manually:")
        self.log(f"  export YOLO_MODEL_PATH={model_path}")
        
        # Also create a .env entry
        env_entry = f"""
# CMS-1500 Fine-tuned YOLO Model (added {datetime.now().strftime('%Y-%m-%d')})
YOLO_MODEL_PATH={model_path}
YOLO_CONFIDENCE=0.25
YOLO_IOU=0.6
"""
        
        env_file = PROJECT_ROOT / ".env.yolo"
        with open(env_file, "w") as f:
            f.write(env_entry)
        
        self.log(f"Environment file created: {env_file}", "SUCCESS")
        
        return True
    
    def run_full_workflow(self) -> bool:
        """Run the complete fine-tuning workflow."""
        self.log("=" * 60)
        self.log("CMS-1500 Fine-Tuning Workflow")
        self.log(f"Started: {datetime.now().isoformat()}")
        self.log("=" * 60)
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Prepare dataset
        dataset_yaml = self.prepare_dataset()
        if not dataset_yaml:
            return False
        
        # Train model
        model_path = self.train_model(dataset_yaml)
        if not model_path:
            return False
        
        # Evaluate
        metrics = self.evaluate_model(model_path, dataset_yaml)
        
        # Deploy
        self.deploy_model(model_path)
        
        # Save summary
        summary = {
            "timestamp": self.timestamp,
            "run_name": self.run_name,
            "input_dir": str(self.input_dir),
            "dataset_yaml": str(dataset_yaml),
            "model_path": str(model_path),
            "epochs": self.epochs,
            "metrics": metrics,
        }
        
        summary_path = self.runs_dir / self.run_name / "summary.json" if (self.runs_dir / self.run_name).exists() else PROJECT_ROOT / f"finetune_summary_{self.timestamp}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.log("=" * 60)
        self.log("WORKFLOW COMPLETE", "SUCCESS")
        self.log("=" * 60)
        self.log(f"Summary saved: {summary_path}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="CMS-1500 Fine-Tuning Workflow")
    
    # Input/output
    parser.add_argument("--input", "-i", type=Path, default=Path("data/sample_docs/cms1500_test"),
                        help="Input directory with CMS-1500 PDFs/images")
    parser.add_argument("--output", "-o", type=Path, default=Path("."),
                        help="Output directory for datasets and models")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    parser.add_argument("--model", default="yolov8m.pt", help="Base YOLO model")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--augments", type=int, default=3, help="Augmentations per image")
    
    # Workflow modes
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare dataset")
    parser.add_argument("--train-only", action="store_true", help="Only train (requires --dataset)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate (requires --model-path)")
    parser.add_argument("--dataset", type=Path, help="Existing dataset.yaml for train-only mode")
    parser.add_argument("--model-path", type=Path, help="Existing model for eval-only mode")
    
    args = parser.parse_args()
    
    workflow = FineTuneWorkflow(
        input_dir=args.input,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        model_base=args.model,
        val_split=args.val_split,
        num_augments=args.augments,
    )
    
    if args.prepare_only:
        dataset_yaml = workflow.prepare_dataset()
        if dataset_yaml:
            print(f"\n✅ Dataset ready: {dataset_yaml}")
            print(f"Next: python training/finetune_workflow.py --train-only --dataset {dataset_yaml}")
        return
    
    if args.train_only:
        if not args.dataset:
            print("Error: --train-only requires --dataset")
            return
        model_path = workflow.train_model(args.dataset)
        if model_path:
            workflow.deploy_model(model_path)
        return
    
    if args.eval_only:
        if not args.model_path or not args.dataset:
            print("Error: --eval-only requires --model-path and --dataset")
            return
        workflow.evaluate_model(args.model_path, args.dataset)
        return
    
    # Full workflow
    workflow.run_full_workflow()


if __name__ == "__main__":
    main()


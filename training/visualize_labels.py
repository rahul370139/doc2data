"""
Visualize YOLO labels on images for verification.

Usage:
    python training/visualize_labels.py --input datasets/cms1500_yolo/ --output tmp/labeled_preview/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

# Class colors (BGR)
CLASS_COLORS: Dict[int, tuple] = {
    0: (0, 255, 0),    # field - green
    1: (255, 0, 0),    # table - blue
    2: (0, 255, 255),  # checkbox - yellow
    3: (255, 0, 255),  # header - magenta
    4: (0, 165, 255),  # signature - orange
}

CLASS_NAMES = {
    0: "field",
    1: "table",
    2: "checkbox",
    3: "header",
    4: "signature",
}


def yolo_to_bbox(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple:
    """Convert YOLO format to (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return (x1, y1, x2, y2)


def visualize_image(img_path: Path, label_path: Path, output_path: Path):
    """Draw YOLO labels on image and save."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load: {img_path}")
        return
    
    h, w = img.shape[:2]
    
    # Read labels
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                
                x1, y1, x2, y2 = yolo_to_bbox(cx, cy, bw, bh, w, h)
                
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                label = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w + 5, y1), color, -1)
                
                # Draw label text
                cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO labels")
    parser.add_argument("--input", required=True, help="Dataset folder with images/ and labels/")
    parser.add_argument("--output", required=True, help="Output folder for visualized images")
    parser.add_argument("--split", default="train", help="Split to visualize (train/val)")
    parser.add_argument("--max-images", type=int, default=50, help="Max images to process")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    images_dir = input_dir / "images" / args.split
    labels_dir = input_dir / "labels" / args.split
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return
    
    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Found {len(images)} images in {args.split}")
    
    for i, img_path in enumerate(images[:args.max_images]):
        label_path = labels_dir / f"{img_path.stem}.txt"
        output_path = output_dir / f"{img_path.stem}_labeled.jpg"
        
        visualize_image(img_path, label_path, output_path)
        print(f"[{i+1}/{min(len(images), args.max_images)}] {img_path.name}")
    
    print(f"\n✅ Visualized {min(len(images), args.max_images)} images to {output_dir}")


if __name__ == "__main__":
    main()


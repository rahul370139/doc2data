#!/usr/bin/env python3
"""
Import Reducto JSON outputs to create a YOLOv8 training dataset.

Maps Reducto's rich semantic labels to our visual layout classes for YOLO training.
The SLM Agent will later refine these into specific semantic tags.

Usage:
    python training/prepare_dataset_reducto.py --json_dir data/reducto_json --image_dir data/raw --output_dir datasets/reducto_yolo
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np

# Map Reducto types to YOLO visual classes
# YOLO focuses on visual structure. SLM adds semantic meaning later.
REDUCTO_TO_YOLO = {
    "Text": 0,
    "Section Header": 3,
    "Title": 3,
    "Footer": 0,          # Visually looks like text
    "Key Value": 0,       # Visually text/field group
    "Form Field": 0,
    "Table": 1,
    "Figure": 2,
    "Checkbox": 2,        # Re-using class 2 for checkbox? No, let's make it distinct.
    "Signature": 4        # Distinct visual feature
}

# Final YOLO Class Map
YOLO_CLASSES = {
    0: "text",
    1: "table",
    2: "figure",
    3: "header",
    4: "signature",
    5: "checkbox"  # Added distinct class
}

def parse_reducto_json(json_path: Path) -> List[Dict[str, Any]]:
    """Extract blocks from Reducto JSON."""
    with open(json_path) as f:
        data = json.load(f)
    
    result = data.get("result", data)
    if "chunks" in result:
        blocks = []
        for chunk in result["chunks"]:
            blocks.extend(chunk.get("blocks", []))
        return blocks
    elif isinstance(result, list):
        return result
    return []

def convert_bbox(bbox: Dict[str, float]) -> str:
    """Convert Reducto bbox (left, top, width, height) to YOLO (xc, yc, w, h)."""
    # Reducto coords are normalized 0-1
    l = bbox.get("left", 0)
    t = bbox.get("top", 0)
    w = bbox.get("width", 0)
    h = bbox.get("height", 0)
    
    xc = l + w / 2
    yc = t + h / 2
    
    return f"{xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

def process_file(json_path: Path, image_dir: Path, output_dir: Path):
    """Process a single JSON file."""
    stem = json_path.stem
    # Find matching image (pdf, png, jpg)
    image_files = list(image_dir.glob(f"{stem}.*"))
    if not image_files:
        # Try fuzzy match if needed, or skip
        print(f"⚠️ Image not found for {json_path.name}")
        return

    img_path = image_files[0]
    blocks = parse_reducto_json(json_path)
    
    if not blocks:
        print(f"⚠️ No blocks in {json_path.name}")
        return

    yolo_labels = []
    for block in blocks:
        b_type = block.get("type", "Text")
        bbox = block.get("bbox")
        
        if not bbox:
            continue
            
        # Map to YOLO class
        class_id = 0 # Default to text
        
        if b_type in ["Table"]:
            class_id = 1
        elif b_type in ["Figure", "Image", "Chart"]:
            class_id = 2
        elif b_type in ["Section Header", "Title"]:
            class_id = 3
        elif b_type in ["Signature"]:
            class_id = 4
        elif b_type in ["Checkbox"]:
            class_id = 5
        elif b_type in ["Key Value", "Form Field"]:
            # Reducto distinguishes these, but visually they are text/input pairs.
            # We can map them to Text (0) and let SLM parse, OR
            # map to Form Field if we want specific detection.
            # Let's map to Text (0) for now to keep YOLO robust for general layout.
            class_id = 0 
            
        yolo_labels.append(f"{class_id} {convert_bbox(bbox)}")

    # Save
    out_img = output_dir / "images" / "train" / img_path.name
    out_lbl = output_dir / "labels" / "train" / f"{img_path.stem}.txt"
    
    shutil.copy(img_path, out_img)
    with open(out_lbl, "w") as f:
        f.write("\n".join(yolo_labels))
        
    print(f"✅ Processed {stem}: {len(yolo_labels)} labels")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/reducto_yolo"))
    args = parser.parse_args()

    # Setup dirs
    for split in ["train", "val"]:
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Create dataset.yaml
    yaml_content = f"""path: {args.output_dir.absolute()}
train: images/train
val: images/train
nc: {len(YOLO_CLASSES)}
names:
{chr(10).join([f"  {k}: {v}" for k, v in YOLO_CLASSES.items()])}
"""
    with open(args.output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    # Run
    for json_file in args.json_dir.glob("*.json"):
        process_file(json_file, args.image_dir, args.output_dir)

if __name__ == "__main__":
    main()

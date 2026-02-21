#!/usr/bin/env python3
"""
Import Reducto JSON outputs to create a YOLOv8 training dataset.

This script takes Reducto JSON responses (which contain high-quality layout detection)
and converts them into YOLO format labels so we can train our own model to replicate
Reducto's layout detection capabilities.

Usage:
    python training/import_reducto_data.py --json_dir data/reducto_json --image_dir data/raw --output_dir datasets/reducto_yolo
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np

# Map Reducto types to our YOLO classes
# We want to detect: Table, Form, Text, Figure, Key Value
# Plus sub-classes if possible
REDUCTO_TO_YOLO = {
    "Key Value": 0,
    "Form Field": 0,  # Map to Key Value
    "Table": 1,
    "Figure": 2,
    "Text": 3,
    "Section Header": 4,
    "Title": 4,       # Map to Header
    "Footer": 5,
    "Signature": 6,
    "Checkbox": 7
}

YOLO_CLASSES = {v: k for k, v in REDUCTO_TO_YOLO.items()}

def parse_reducto_json(json_path: Path) -> List[Dict[str, Any]]:
    """Extract blocks from Reducto JSON."""
    with open(json_path) as f:
        data = json.load(f)
    
    # Handle different Reducto response formats
    result = data.get("result", data)
    if "chunks" in result:
        # Full response format
        blocks = []
        for chunk in result["chunks"]:
            blocks.extend(chunk.get("blocks", []))
        return blocks
    elif isinstance(result, list):
        # Direct list of blocks
        return result
    return []

def convert_bbox(bbox: Dict[str, float], width: int, height: int) -> str:
    """Convert Reducto bbox (left, top, width, height) to YOLO (xc, yc, w, h)."""
    # Reducto gives normalized coordinates (0-1)
    l = bbox.get("left", 0)
    t = bbox.get("top", 0)
    w = bbox.get("width", 0)
    h = bbox.get("height", 0)
    
    # YOLO format: center_x, center_y, width, height (all normalized)
    xc = l + w / 2
    yc = t + h / 2
    
    return f"{xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

def process_file(json_path: Path, image_dir: Path, output_dir: Path):
    """Process a single JSON file and its corresponding image."""
    # Find matching image
    stem = json_path.stem
    image_files = list(image_dir.glob(f"{stem}.*"))
    if not image_files:
        # Try finding by 'pdf_url' or other metadata if needed, 
        # but for now assume filename matching
        print(f"⚠️ No matching image found for {json_path.name}")
        return

    img_path = image_files[0]
    
    # Load image to get dimensions (though Reducto bboxes are normalized)
    # We mainly need to copy the image to the dataset
    
    blocks = parse_reducto_json(json_path)
    if not blocks:
        print(f"⚠️ No blocks found in {json_path.name}")
        return

    # Prepare labels
    yolo_labels = []
    for block in blocks:
        b_type = block.get("type", "Text")
        # Map specific Reducto types to our YOLO schema
        class_id = REDUCTO_TO_YOLO.get(b_type, REDUCTO_TO_YOLO.get("Text"))
        
        bbox = block.get("bbox")
        if not bbox:
            continue
            
        label_line = f"{class_id} {convert_bbox(bbox, 1000, 1000)}" # Dims don't matter for normalized
        yolo_labels.append(label_line)

    # Save to dataset
    # Copy image
    out_img_path = output_dir / "images" / "train" / img_path.name
    shutil.copy(img_path, out_img_path)
    
    # Save labels
    out_lbl_path = output_dir / "labels" / "train" / f"{img_path.stem}.txt"
    with open(out_lbl_path, "w") as f:
        f.write("\n".join(yolo_labels))
    
    print(f"✅ Processed {stem}: {len(yolo_labels)} labels")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=Path, required=True, help="Folder containing Reducto JSONs")
    parser.add_argument("--image_dir", type=Path, required=True, help="Folder containing source images/PDFs")
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/reducto_yolo"))
    args = parser.parse_args()

    # Setup directories
    (args.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Create dataset.yaml
    yaml_content = f"""path: {args.output_dir.absolute()}
train: images/train
val: images/train  # Use train for val if small dataset
nc: {len(YOLO_CLASSES)}
names:
{chr(10).join([f"  {k}: {v}" for k, v in YOLO_CLASSES.items()])}
"""
    with open(args.output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    # Process files
    json_files = list(args.json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        process_file(json_file, args.image_dir, args.output_dir)

if __name__ == "__main__":
    main()


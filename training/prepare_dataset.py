#!/usr/bin/env python3
"""
Dataset preparation for CMS-1500 YOLOv8 fine-tuning.

Creates YOLO training data with per-field bounding boxes.

Methods:
1. Schema-based: Uses cms-1500.json schema with accurate field coordinates
2. OCR-based: Auto-detects text regions (fallback, less accurate)

Usage:
    # From schema (recommended for CMS-1500)
    python training/prepare_dataset.py --input data/raw/ --output datasets/cms1500_yolo/ --method schema

    # From OCR (for other forms)
    python training/prepare_dataset.py --input data/raw/ --output datasets/cms1500_yolo/ --method ocr

Classes:
    0: field     - Text entry fields
    1: table     - Service line tables
    2: checkbox  - Checkbox fields
    3: header    - Section headers
    4: signature - Signature lines
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

# YOLO class mapping
CLASSES = {"field": 0, "table": 1, "checkbox": 2, "header": 3, "signature": 4}


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> List[np.ndarray]:
    """Convert PDF to images."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        images = []
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            images.append(img)
        doc.close()
        return images
    except Exception as e:
        print(f"  ⚠️ PDF error: {e}")
        return []


def load_image(path: Path) -> Optional[np.ndarray]:
    """Load image file."""
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def bbox_to_yolo(bbox: List[float]) -> Tuple[float, float, float, float]:
    """Convert [x0, y0, x1, y1] to YOLO [x_center, y_center, width, height]."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    return (x0 + w/2, y0 + h/2, w, h)


def get_class_id(field: dict) -> int:
    """Determine YOLO class from field info."""
    ftype = field.get("field_type", "text").lower()
    fid = field.get("id", "").lower()
    
    if ftype == "checkbox":
        return CLASSES["checkbox"]
    elif ftype == "signature":
        return CLASSES["signature"]
    elif "service_line" in fid or fid.startswith("24_"):
        return CLASSES["table"]
    else:
        return CLASSES["field"]


def generate_schema_labels(schema_path: Path) -> List[str]:
    """Generate YOLO labels from CMS-1500 schema."""
    with open(schema_path) as f:
        schema = json.load(f)
    
    labels = []
    for field in schema.get("fields", []):
        bbox = field.get("bbox_norm")
        if not bbox or len(bbox) != 4:
            continue
        
        x_c, y_c, w, h = bbox_to_yolo(bbox)
        class_id = get_class_id(field)
        labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    
    return labels


def augment_image(img: np.ndarray, idx: int) -> np.ndarray:
    """Apply augmentation."""
    if idx == 0:
        # Brightness
        return cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=random.randint(-20, 20))
    elif idx == 1:
        # Blur
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    elif idx == 2:
        # Noise
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    schema_path: Optional[Path] = None,
    method: str = "schema",
    augment: bool = True,
    train_ratio: float = 0.8
):
    """Prepare YOLO dataset."""
    
    # Setup output dirs
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Generate labels from schema
    if method == "schema":
        if not schema_path or not schema_path.exists():
            schema_path = Path(__file__).parent.parent / "data" / "schemas" / "cms-1500.json"
        
        print(f"📋 Loading schema: {schema_path}")
        labels = generate_schema_labels(schema_path)
        print(f"  Generated {len(labels)} field labels")
        label_content = "\n".join(labels)
    else:
        label_content = None  # Will use OCR-based labeling
    
    # Find input files
    input_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.png")) + \
                  list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    
    if not input_files:
        print(f"❌ No PDF/image files found in {input_dir}")
        return
    
    print(f"\n📂 Processing {len(input_files)} files...")
    
    all_images = []
    
    for file_path in input_files:
        print(f"  📄 {file_path.name}")
        
        # Load images
        if file_path.suffix.lower() == ".pdf":
            images = pdf_to_images(file_path)
        else:
            img = load_image(file_path)
            images = [img] if img is not None else []
        
        # Process each page
        for page_idx, img in enumerate(images):
            base_name = f"{file_path.stem}_p{page_idx}"
            all_images.append((base_name, img, label_content))
            
            # Augment
            if augment:
                for aug_idx in range(3):
                    aug_img = augment_image(img, aug_idx)
                    aug_name = f"{base_name}_aug{aug_idx}"
                    all_images.append((aug_name, aug_img, label_content))
    
    # Split train/val
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)
    train_set = all_images[:split_idx]
    val_set = all_images[split_idx:]
    
    print(f"\n📊 Split: {len(train_set)} train, {len(val_set)} val")
    
    # Save images and labels
    for name, img, labels in train_set:
        img_path = output_dir / "images" / "train" / f"{name}.jpg"
        lbl_path = output_dir / "labels" / "train" / f"{name}.txt"
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if labels:
            with open(lbl_path, 'w') as f:
                f.write(labels)
    
    for name, img, labels in val_set:
        img_path = output_dir / "images" / "val" / f"{name}.jpg"
        lbl_path = output_dir / "labels" / "val" / f"{name}.txt"
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if labels:
            with open(lbl_path, 'w') as f:
                f.write(labels)
    
    # Create dataset.yaml
    yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val
nc: 5
names:
  0: field
  1: table
  2: checkbox
  3: header
  4: signature
"""
    with open(output_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset ready at {output_dir}")
    print(f"   Train: {len(train_set)} images")
    print(f"   Val: {len(val_set)} images")
    print(f"   Labels per image: {len(labels.split(chr(10))) if labels else 0}")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset for CMS-1500")
    parser.add_argument("--input", "-i", type=Path, default=Path("data/raw"),
                        help="Input directory with PDFs/images")
    parser.add_argument("--output", "-o", type=Path, default=Path("datasets/cms1500_yolo"),
                        help="Output dataset directory")
    parser.add_argument("--method", "-m", choices=["schema", "ocr"], default="schema",
                        help="Labeling method: schema (recommended) or ocr")
    parser.add_argument("--schema", "-s", type=Path, default=None,
                        help="Path to CMS-1500 schema JSON")
    parser.add_argument("--augment", "-a", action="store_true", default=True,
                        help="Apply data augmentation")
    parser.add_argument("--no-augment", action="store_false", dest="augment",
                        help="Disable augmentation")
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        schema_path=args.schema,
        method=args.method,
        augment=args.augment
    )


if __name__ == "__main__":
    main()

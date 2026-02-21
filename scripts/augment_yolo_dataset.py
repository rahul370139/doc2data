#!/usr/bin/env python3
"""
Augment YOLO dataset for CMS-1500 field detection.

This script generates additional training data by:
1. Applying various augmentations to existing images
2. Generating synthetic variations (blur, noise, rotation, contrast)
3. Properly preserving bounding box coordinates

Run on DGX:
    python scripts/augment_yolo_dataset.py --input datasets/cms1500_yolo --output datasets/cms1500_yolo_augmented
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import random
import shutil
from typing import List, Tuple


def load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO format labels: class_id x_center y_center width height (normalized)."""
    labels = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    labels.append((cls_id, x, y, w, h))
    return labels


def save_yolo_labels(labels: List[Tuple[int, float, float, float, float]], label_path: Path):
    """Save YOLO format labels."""
    with open(label_path, 'w') as f:
        for cls_id, x, y, w, h in labels:
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def augment_image_with_labels(
    image: np.ndarray, 
    labels: List[Tuple[int, float, float, float, float]],
    aug_type: str
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Apply augmentation to image and update labels accordingly."""
    
    h, w = image.shape[:2]
    new_labels = labels.copy()
    
    if aug_type == 'blur':
        # Gaussian blur (simulates low quality scan)
        k = random.choice([3, 5, 7])
        image = cv2.GaussianBlur(image, (k, k), 0)
        
    elif aug_type == 'noise':
        # Add Gaussian noise (simulates scanner noise)
        noise = np.random.normal(0, random.uniform(5, 20), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    elif aug_type == 'brightness':
        # Random brightness adjustment
        factor = random.uniform(0.7, 1.3)
        image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        
    elif aug_type == 'contrast':
        # Random contrast adjustment
        factor = random.uniform(0.8, 1.2)
        mean = np.mean(image)
        image = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
        
    elif aug_type == 'rotate_small':
        # Small rotation (-3 to +3 degrees) - common for scanned docs
        angle = random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # For small angles, bounding boxes are approximately preserved
        # (YOLO training handles minor misalignment via augmentation)
        
    elif aug_type == 'translate':
        # Small translation (simulates scanning misalignment)
        tx = random.randint(-20, 20)
        ty = random.randint(-20, 20)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # Update label positions
        new_labels = []
        for cls_id, x, y, bw, bh in labels:
            new_x = x + tx / w
            new_y = y + ty / h
            # Only keep if still mostly visible
            if 0.1 < new_x < 0.9 and 0.1 < new_y < 0.9:
                new_labels.append((cls_id, new_x, new_y, bw, bh))
        
    elif aug_type == 'jpeg_compress':
        # JPEG compression artifacts (common in faxed docs)
        quality = random.randint(50, 85)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
    elif aug_type == 'sharpen':
        # Sharpen (opposite of blur)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
        
    elif aug_type == 'grayscale_back':
        # Convert to grayscale and back (lose color info)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
    return image, new_labels


def generate_augmentations(
    input_dir: Path, 
    output_dir: Path, 
    num_augments_per_image: int = 10
):
    """Generate augmented dataset."""
    
    # Setup output directories
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    aug_types = ['blur', 'noise', 'brightness', 'contrast', 'rotate_small', 
                 'translate', 'jpeg_compress', 'sharpen', 'grayscale_back']
    
    for split in ['train', 'val']:
        img_dir = input_dir / 'images' / split
        lbl_dir = input_dir / 'labels' / split
        
        if not img_dir.exists():
            continue
            
        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        print(f"\nProcessing {split}: {len(images)} images")
        
        for img_path in images:
            # Copy original
            label_path = lbl_dir / f"{img_path.stem}.txt"
            labels = load_yolo_labels(label_path)
            
            # Copy original
            shutil.copy(img_path, output_dir / 'images' / split / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, output_dir / 'labels' / split / label_path.name)
            
            # Generate augmentations
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            for i in range(num_augments_per_image):
                # Apply 1-3 random augmentations
                aug_img = image.copy()
                aug_labels = labels.copy()
                
                n_augs = random.randint(1, 3)
                chosen_augs = random.sample(aug_types, n_augs)
                
                for aug in chosen_augs:
                    aug_img, aug_labels = augment_image_with_labels(aug_img, aug_labels, aug)
                
                # Save augmented version
                aug_name = f"{img_path.stem}_gen{i}"
                cv2.imwrite(str(output_dir / 'images' / split / f"{aug_name}.jpg"), aug_img)
                save_yolo_labels(aug_labels, output_dir / 'labels' / split / f"{aug_name}.txt")
        
        # Count results
        n_out = len(list((output_dir / 'images' / split).glob('*')))
        print(f"  Output: {n_out} images")


def main():
    parser = argparse.ArgumentParser(description='Augment YOLO dataset')
    parser.add_argument('--input', type=str, default='datasets/cms1500_yolo',
                        help='Input dataset directory')
    parser.add_argument('--output', type=str, default='datasets/cms1500_yolo_augmented',
                        help='Output dataset directory')
    parser.add_argument('--num-aug', type=int, default=15,
                        help='Number of augmentations per image')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentations per image: {args.num_aug}")
    
    generate_augmentations(input_dir, output_dir, args.num_aug)
    
    # Create dataset.yaml
    classes = ['field', 'table', 'checkbox', 'header', 'signature']
    config = f"""path: {output_dir.absolute()}
train: images/train
val: images/val
nc: {len(classes)}
names:
"""
    for i, name in enumerate(classes):
        config += f"  {i}: {name}\n"
    
    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(config)
    
    print(f"\nDataset config written to {output_dir / 'dataset.yaml'}")
    print("\nNext steps:")
    print(f"  python scripts/train_yolo_gpu.py --data-dir {output_dir} --epochs 100 --device 0")


if __name__ == '__main__':
    main()


"""
Image Registration Module for Template Alignment.

This module provides functions to align a scanned document to a reference template
using feature matching and homography. This allows using schema coordinates
accurately even if the scan is rotated, scaled, or translated.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import os

# Cache for template features to avoid re-computing
_TEMPLATE_CACHE: Dict[str, Dict[str, Any]] = {}

def get_reference_image_path(template_name: str) -> Optional[str]:
    """Resolve path to reference image for a template."""
    # Assuming standard location in data/sample_docs
    # Adjust this path relative to the project root
    base_dir = Path(__file__).parent.parent.parent / "data" / "sample_docs"
    
    mapping = {
        "cms-1500": "cms1500_blank.pdf",
        "cms1500": "cms1500_blank.pdf",
        "ub-04": "ub04_clean.pdf", 
        "ub04": "ub04_clean.pdf"
    }
    
    filename = mapping.get(template_name.lower())
    if not filename:
        return None
        
    path = base_dir / filename
    return str(path) if path.exists() else None

def load_and_process_reference(template_name: str) -> Optional[Dict[str, Any]]:
    """
    Load reference image and compute features.
    Returns dictionary with 'keypoints', 'descriptors', 'shape'.
    """
    global _TEMPLATE_CACHE
    
    if template_name in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[template_name]
    
    path = get_reference_image_path(template_name)
    if not path:
        return None
    
    try:
        # Handle PDF reference
        if path.lower().endswith('.pdf'):
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            page = doc[0]
            # Render at 300 DPI for good feature detection
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            doc.close()
        else:
            # Handle Image reference
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
        if img is None:
            return None
            
        # Initialize ORB detector
        # Increase features for better matching on dense forms
        orb = cv2.ORB_create(nfeatures=5000)
        kp, des = orb.detectAndCompute(img, None)
        
        if des is None:
            return None
            
        data = {
            "keypoints": kp,
            "descriptors": des,
            "shape": img.shape, # h, w
            "image": img # Keep image for visualization/debugging if needed
        }
        _TEMPLATE_CACHE[template_name] = data
        return data
        
    except Exception as e:
        print(f"Error loading reference template {template_name}: {e}")
        return None

def compute_alignment_matrix(
    input_image: np.ndarray, 
    template_name: str
) -> Optional[np.ndarray]:
    """
    Compute homography matrix to align input_image to template.
    
    Returns 3x3 matrix H such that:
    Template_Point = H * Input_Point
    
    Note: To map Schema (Template) coordinates TO Input Image,
    we typically need the inverse (Input = H_inv * Template).
    """
    ref_data = load_and_process_reference(template_name)
    if not ref_data:
        return None
        
    # Convert input to grayscale if needed
    if len(input_image.shape) == 3:
        gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = input_image
        
    # Detect features in input image
    orb = cv2.ORB_create(nfeatures=5000)
    kp_img, des_img = orb.detectAndCompute(gray, None)
    
    if des_img is None or len(kp_img) < 10:
        return None
        
    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ref_data["descriptors"], des_img)
    
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Keep top % matches
    keep_percent = 0.2
    num_keep = int(len(matches) * keep_percent)
    good_matches = matches[:num_keep]
    
    if len(good_matches) < 10:
        print(f"Not enough matches for alignment: {len(good_matches)}")
        return None
        
    # Extract point coordinates
    src_pts = np.float32([ref_data["keypoints"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find Homography
    # H maps Reference(src) -> Input(dst)
    # This means: Input_Coord = H * Reference_Coord
    # This is exactly what we need to transform Schema BBoxes (Reference) to Input Image
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H

def transform_bbox(bbox: Tuple[float, float, float, float], H: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Transform a bounding box [x0, y0, x1, y1] using homography H.
    """
    x0, y0, x1, y1 = bbox
    
    # Transform all 4 corners to handle rotation/skew
    points = np.float32([
        [[x0, y0]],
        [[x1, y0]],
        [[x1, y1]],
        [[x0, y1]]
    ])
    
    transformed = cv2.perspectiveTransform(points, H)
    
    # Get new bounding box wrapping the transformed points
    pts = transformed.reshape(-1, 2)
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    
    return (min_x, min_y, max_x, max_y)

def transform_normalized_bbox(
    norm_bbox: List[float], 
    H: np.ndarray, 
    ref_width: int, 
    ref_height: int
) -> Tuple[float, float, float, float]:
    """
    Transform normalized bbox from schema to absolute coords in input image.
    """
    # 1. Denormalize to Reference dimensions
    x0 = norm_bbox[0] * ref_width
    y0 = norm_bbox[1] * ref_height
    x1 = norm_bbox[2] * ref_width
    y1 = norm_bbox[3] * ref_height
    
    # 2. Transform using H
    return transform_bbox((x0, y0, x1, y1), H)


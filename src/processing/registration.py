"""
Image Registration Module for Template Alignment.

This module provides functions to align a scanned document to a reference template
using feature matching and homography. This allows using schema coordinates
accurately even if the scan is rotated, scaled, or translated.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import os

# Cache for template features to avoid re-computing
_TEMPLATE_CACHE: Dict[str, Dict[str, Any]] = {}


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order quad points as [tl, tr, br, bl]."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _detect_form_quad(gray: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    Detect outer form boundary as a quadrilateral for stable perspective alignment.
    Returns (quad, score 0..1).
    """
    try:
        g = gray
        if g.dtype != np.uint8:
            g = g.astype(np.uint8)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        edges = cv2.Canny(g, 50, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0
        h, w = g.shape[:2]
        img_area = float(h * w)
        best = None
        best_score = 0.0

        def _angle(a, b, c) -> float:
            ba = a - b
            bc = c - b
            denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
            return float(np.degrees(np.arccos(cosang)))

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < img_area * 0.15:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            quad = _order_quad_points(approx.reshape(4, 2))
            area_ratio = area / img_area
            angs = [
                _angle(quad[3], quad[0], quad[1]),
                _angle(quad[0], quad[1], quad[2]),
                _angle(quad[1], quad[2], quad[3]),
                _angle(quad[2], quad[3], quad[0]),
            ]
            ang_err = float(np.mean([abs(a - 90.0) for a in angs]))
            angle_score = float(max(0.0, 1.0 - (ang_err / 25.0)))
            score = 0.7 * min(1.0, area_ratio / 0.60) + 0.3 * angle_score
            if score > best_score:
                best_score = score
                best = quad
        return (best, float(best_score)) if best is not None else (None, 0.0)
    except Exception:
        return None, 0.0


def _validate_homography_ref_to_input(H: np.ndarray, ref_shape: Tuple[int, int], input_shape: Tuple[int, int]) -> bool:
    """
    Validate a homography that maps REFERENCE -> INPUT.
    Reject wild warps but allow mild perspective (scans).
    """
    if H is None:
        return False
    try:
        ref_h, ref_w = ref_shape[:2]
        in_h, in_w = input_shape[:2]

        ref_corners = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(ref_corners, H).reshape(-1, 2)
        if not np.isfinite(warped).all():
            return False

        # Area sanity (in input space)
        area = float(cv2.contourArea(warped.astype(np.float32)))
        in_area = float(in_w * in_h)
        if area < in_area * 0.15 or area > in_area * 1.10:
            return False

        # Keep corners within a reasonable padded boundary
        pad_x = in_w * 0.15
        pad_y = in_h * 0.15
        if (warped[:, 0].min() < -pad_x or warped[:, 0].max() > in_w + pad_x or
            warped[:, 1].min() < -pad_y or warped[:, 1].max() > in_h + pad_y):
            return False

        if not cv2.isContourConvex(warped.astype(np.float32)):
            return False
        return True
    except Exception:
        return False

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

    # 0) Stable primary: outer quad -> compute input->ref then invert to ref->input
    try:
        quad, q_score = _detect_form_quad(gray)
        ref_h, ref_w = ref_data["shape"][:2]
        if quad is not None and q_score >= 0.55:
            dst = np.array([[0, 0], [ref_w - 1, 0], [ref_w - 1, ref_h - 1], [0, ref_h - 1]], dtype=np.float32)
            H_in_to_ref = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
            if H_in_to_ref is not None:
                H_ref_to_in = np.linalg.inv(H_in_to_ref)
                if _validate_homography_ref_to_input(H_ref_to_in, (ref_h, ref_w), gray.shape):
                    return H_ref_to_in
    except Exception:
        pass
        
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
    
    # Find Homography (feature fallback)
    # H maps Reference(src) -> Input(dst)
    # This means: Input_Coord = H * Reference_Coord
    # This is exactly what we need to transform Schema BBoxes (Reference) to Input Image
    H_ref_to_in, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H_ref_to_in is None:
        return None
    ref_h, ref_w = ref_data["shape"][:2]
    if not _validate_homography_ref_to_input(H_ref_to_in, (ref_h, ref_w), gray.shape):
        return None
    return H_ref_to_in

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


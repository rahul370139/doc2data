"""
Image preprocessing utilities for document pipeline.

PURPOSE: Prepares images before OCR and alignment. Includes deskew (correct
rotation), denoise, contrast enhancement (CLAHE), and red template text
removal. Uses GPU when available for faster processing.

USE CASE: Called before OCR on scanned documents to improve text recognition.
Red removal helps separate handwritten ink from pre-printed form labels.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image
from utils.config import Config
from src.processing.gpu_utils import GPUUtils


def deskew_image(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    De-skew image. Robustly estimate small skew and correct it.
    Returns the original image if:
    - The estimated angle is tiny (<0.5°)
    - The angle is too large (>5°) - likely a bad estimate from grid lines
    - There's not enough consensus among detected lines
    
    IMPORTANT: This function should NOT be applied to digital PDFs or forms
    with complex grid structures (like UB-04) as the grid lines can cause
    false skew detection.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Tuple of (deskewed_image, rotation_angle)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Contrast enhancement + binarization (GPU accelerated when available)
    try:
        if GPUUtils.is_available():
            gray = GPUUtils.gaussian_blur(gray, (3, 3), 0)
        else:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
    except Exception:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )

    # Emphasize text lines (focus on horizontal text lines, not grid)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))  # Longer kernel to favor text lines
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # Hough transform to detect predominant line angles
    lines = cv2.HoughLines(horiz, 1, np.pi / 180.0, threshold=200)  # Higher threshold
    if lines is None or len(lines) < 10:  # Require more lines for confidence
        # No reliable skew
        return image, 0.0

    # Convert angles to degrees around horizontal axis [-90, 90]
    # Focus only on near-horizontal lines (±10° from 0 or 180)
    angles = []
    for rho_theta in lines:
        theta = rho_theta[0][1]
        deg = (theta * 180.0 / np.pi)
        # Map to [-90, 90]
        if deg > 90:
            deg -= 180
        # Only consider angles close to horizontal
        if -10 <= deg <= 10:
            angles.append(deg)

    if len(angles) < 5:  # Need minimum consensus
        return image, 0.0

    # Use median to be robust to outliers
    angle = float(np.median(angles))
    
    # Check standard deviation - high variance means unreliable estimate
    std = float(np.std(angles))
    if std > 2.0:  # Too much variance, lines are not aligned
        return image, 0.0

    # If angle is tiny or too large (likely wrong), do nothing
    # Be VERY conservative: only correct clear skew between 0.5° and 5°
    if abs(angle) < 0.5 or abs(angle) > 5.0:
        return image, 0.0

    print(f"[Deskew] Detected skew angle: {angle:.2f}° (std: {std:.2f}°) - applying correction")
    
    # Rotate by the negative of the skew angle to deskew
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, -angle


def denoise_image(image: np.ndarray, method: str = "median") -> np.ndarray:
    """
    De-noise image using median blur or Gaussian blur.
    
    Args:
        image: Input image as numpy array
        method: Denoising method ('median' or 'gaussian')
        
    Returns:
        Denoised image
    """
    if method == "median":
        if len(image.shape) == 3:
            denoised = cv2.medianBlur(image, 3)
        else:
            denoised = cv2.medianBlur(image, 3)
    elif method == "gaussian":
        try:
            if GPUUtils.is_available():
                denoised = GPUUtils.gaussian_blur(image, (3, 3), 0)
            else:
                denoised = cv2.GaussianBlur(image, (3, 3), 0)
        except Exception:
            denoised = cv2.GaussianBlur(image, (3, 3), 0)
    else:
        denoised = image
    
    return denoised


def enhance_camera_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image captured by camera (lighting correction, unsharp mask).
    """
    # 1. White balance / Lighting correction via morphological opening (background est)
    if len(image.shape) == 3:
        # Working in LAB space for brightness
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Estimate background illumination
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        bg = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        
        # Normalize brightness: (L / bg) * 200
        # Avoid division by zero
        bg = bg.astype(float)
        bg[bg == 0] = 1
        l_float = l.astype(float)
        l_norm = (l_float / bg) * 220.0
        l_norm = np.clip(l_norm, 0, 255).astype(np.uint8)
        
        # Merge back
        lab_norm = cv2.merge((l_norm, a, b))
        corrected = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale approach
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        bg = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        bg = bg.astype(float)
        bg[bg == 0] = 1
        norm = (image.astype(float) / bg) * 220.0
        corrected = np.clip(norm, 0, 255).astype(np.uint8)

    # 2. Denoise lightly
    corrected = cv2.fastNlMeansDenoising(corrected, None, 10, 7, 21) if len(image.shape) != 3 else \
                cv2.fastNlMeansDenoisingColored(corrected, None, 10, 10, 7, 21)

    # 3. Unsharp mask for edge crispness
    gaussian_3 = cv2.GaussianBlur(corrected, (0, 0), 2.0)
    unsharp = cv2.addWeighted(corrected, 1.5, gaussian_3, -0.5, 0)
    
    return unsharp


def preprocess_image(
    image: np.ndarray,
    deskew: bool = True,
    denoise: bool = True,
    denoise_method: str = "median",
    doc_type: str = "generic"
) -> Tuple[np.ndarray, dict]:
    """
    Apply preprocessing to image.
    
    Args:
        image: Input image as numpy array
        deskew: Whether to apply de-skewing
        denoise: Whether to apply de-noising
        denoise_method: Method for de-noising ('median' or 'gaussian')
        
    Returns:
        Tuple of (processed_image, metadata)
    """
    metadata = {
        "deskewed": False,
        "denoised": False,
        "rotation_angle": 0.0
    }
    
    processed = image.copy()
    hint = (doc_type or Config.DOC_TYPE_HINT or "generic").lower()
    
    # Camera image enhancement
    if hint in {"camera", "photo", "mobile"}:
        processed = enhance_camera_image(processed)
        metadata["camera_enhanced"] = True

    # Adaptive upscaling for handwritten/low-res scans
    if hint in {"handwritten", "scan", "scanned"}:
        scale = 1.3
        new_w = int(processed.shape[1] * scale)
        new_h = int(processed.shape[0] * scale)
        processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    if deskew:
        processed, angle = deskew_image(processed)
        metadata["deskewed"] = True
        metadata["rotation_angle"] = angle
    
    if denoise:
        processed = denoise_image(processed, method=denoise_method)
        metadata["denoised"] = True

    # Form-specific tweak: stronger contrast for forms
    if hint in {"form", "cms1500", "ub04", "ncpdp"}:
        gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY) if len(processed.shape) == 3 else processed
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # NOTE:
    # We intentionally do NOT remove form lines here for CMS-1500.
    # De-lining before template alignment can destroy SIFT/ECC keypoints and ruin alignment.
    # De-lining is applied later in the pipeline *after alignment* and *before OCR*.

    return processed, metadata


def remove_red_template_text(image: np.ndarray) -> np.ndarray:
    """
    Adaptive red removal for CMS-1500 per-field crops.

    Scanned/compressed PDFs often desaturate the CMS-1500 dropout-red so it
    falls below a fixed HSV saturation threshold.  This version uses two
    complementary color-space strategies so muted reds are caught too.

    Strategy 1 — LAB 'a' channel:
        The 'a' axis encodes red-vs-green independent of lightness.  Template
        red sits at a > ~138 (out of 255, where 128 = neutral).  Robust to
        JPEG compression and scanner colour-profile shifts.
    Strategy 2 — HSV with *relaxed* saturation (S >= 35 instead of 70):
        Catches any remaining vivid reds.  The lower threshold is safe because
        dark ink is excluded by the brightness guard below.
    Safety guards:
        1. Pixels darker than gray=100 are NEVER removed — protects faded
           handwriting, gray pencil, and light-blue ink (gray 80-120 range).
        2. Canny edge strokes are NEVER removed — structural text contours
           are preserved regardless of their color, preventing accidental
           erasure of handwriting that overlaps red detection thresholds.
    Rules preserved from the original:
        - NO dilation (would eat adjacent handwriting strokes)
        - Never binarise — preserve grayscale strokes for OCR
    """
    img = image.copy()
    if img.ndim == 2:
        return img

    # --- Strategy 1: LAB colour space (most robust for muted scans) --------
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    a_ch = lab[:, :, 1]
    l_ch = lab[:, :, 0]
    lab_red = (a_ch > 133) & (l_ch > 55)

    # --- Strategy 2: HSV with relaxed saturation ---------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, np.array([0,  25, 45], dtype=np.uint8),
                           np.array([15, 255, 255], dtype=np.uint8))
    m2 = cv2.inRange(hsv, np.array([165, 25, 45], dtype=np.uint8),
                           np.array([180, 255, 255], dtype=np.uint8))
    hsv_red = (cv2.bitwise_or(m1, m2) > 0)

    # --- Strategy 3: Pinkish/light-red that scanners produce ───────────────
    pink_red = (a_ch > 130) & (l_ch > 100)

    # --- Combine & safety guards -------------------------------------------
    combined = lab_red | hsv_red | pink_red

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Guard 1: Protect ALL dark-ish pixels.  Raised from 70 → 100 because
    # faded handwriting, gray pencil, and light-blue ink sit at gray 80-120
    # and were being erased.
    combined[gray < 100] = False

    # Guard 2: Edge-based text stroke protection.  Canny detects structural
    # contours (pen strokes).  Any pixel that is part of a text stroke is
    # preserved, even if its colour looks "red" to the detectors above.
    edges = cv2.Canny(gray, 40, 120)
    stroke_mask = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )
    combined[stroke_mask > 0] = False

    img[combined] = [255, 255, 255]
    return img



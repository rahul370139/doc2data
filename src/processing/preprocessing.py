"""
Image preprocessing utilities: de-skew and de-noise.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image
from utils.config import Config
from src.processing.gpu_utils import GPUUtils


def deskew_image(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    De-skew image. Robustly estimate small skew (±3° max) and correct it.
    Returns the original image if the estimated angle is tiny (<0.3°) or
    implausibly large (>3°), which usually indicates a bad estimate.
    
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

    # Emphasize text lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    mask = cv2.bitwise_or(horiz, vert)

    # Hough transform to detect predominant line angles
    lines = cv2.HoughLines(mask, 1, np.pi / 180.0, threshold=160)
    if lines is None or len(lines) < 5:
        # No reliable skew
        return image, 0.0

    # Convert angles to degrees around horizontal axis [-90, 90]
    angles = []
    for rho_theta in lines:
        theta = rho_theta[0][1]
        deg = (theta * 180.0 / np.pi)
        # Map to [-90, 90]
        if deg > 90:
            deg -= 180
        if -89 <= deg <= 89:
            angles.append(deg)

    if not angles:
        return image, 0.0

    # Use median to be robust to outliers
    angle = float(np.median(angles))

    # If angle is tiny or too large (likely wrong), do nothing
    if abs(angle) < 0.3 or abs(angle) > 3.0:
        return image, 0.0

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

    return processed, metadata

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
    Safe red removal for CMS-1500 per-field crops.
    
    Designed to be safe on small crops where handwriting may touch red grid lines.
    Rules:
    - HSV only, high saturation (S >= 70) to catch only true dropout reds
    - NO low-saturation ranges (would catch brown ink, shadows, paper yellowing)
    - NO RGB dominance check (would catch warm-toned handwriting)
    - NO dilation (would eat into adjacent handwriting strokes)
    - Never binarize — preserve grayscale strokes for OCR
    - Only whiten strongly-red pixels, leave everything else untouched
    """
    img = image.copy()
    if img.ndim == 2:
        return img
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Only strongly saturated reds (S >= 70). This catches the CMS-1500
    # dropout-red ink but NOT dark handwriting, brown ink, or scan artifacts.
    # Red hue wraps around 0/180 in OpenCV HSV.
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50], dtype=np.uint8),
                              np.array([12, 255, 255], dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array([168, 70, 50], dtype=np.uint8),
                              np.array([180, 255, 255], dtype=np.uint8))
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # NO dilation — do not expand mask into adjacent handwriting
    # NO binarization — keep original pixel values for non-red areas
    
    img[red_mask > 0] = [255, 255, 255]
    return img


def remove_form_lines(image: np.ndarray) -> np.ndarray:
    """
    Remove CMS-1500 ruling lines (especially red grid) to improve OCR.
    Approach:
      1) Remove all red content (labels + lines).
      2) Extract long horizontal/vertical lines from a binary image and inpaint.
    """
    # Step 1: Remove all red template content
    img = remove_red_template_text(image)

    # Step 2: Remove any remaining long ruling lines via morphology + inpaint
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 11)

    h, w = bw.shape[:2]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 25), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 25)))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)
    line_mask = cv2.bitwise_or(horiz, vert)

    # Slight dilation so we inpaint full stroke width
    line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    if np.count_nonzero(line_mask) > 0:
        img = cv2.inpaint(img, line_mask, 3, cv2.INPAINT_TELEA)

    return img

def extract_ink_by_subtraction(
    image: np.ndarray, 
    template: np.ndarray, 
    threshold: int = 25,
    return_ink_image: bool = False
) -> np.ndarray:
    """
    Extract ink from a filled form by subtracting the blank template.
    Requires 'image' to be aligned to 'template'.
    
    Args:
        image: Aligned filled form (grayscale or RGB)
        template: Blank form template (grayscale or RGB)
        threshold: Difference threshold to consider pixel as ink
        return_ink_image: If True, return image with non-ink pixels whitened (better for OCR)
        
    Returns:
        If return_ink_image=False: Mask where ink is white (255), background black (0)
        If return_ink_image=True: RGB image with only ink visible (template text whitened)
    """
    # 1. Ensure Grayscale for comparison
    if len(image.shape) == 3:
        g_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        g_img = image
        
    if len(template.shape) == 3:
        g_temp = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    else:
        g_temp = template
        
    # 2. Resize image to match template if needed (should be aligned already)
    if g_img.shape != g_temp.shape:
        g_img = cv2.resize(g_img, (g_temp.shape[1], g_temp.shape[0]))
        
    # 3. Compute absolute difference
    diff = cv2.absdiff(g_img, g_temp)
    
    # 4. Multi-level thresholding for better ink detection
    # Lower threshold for dark ink, higher for faint marks
    _, mask_strong = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    _, mask_weak = cv2.threshold(diff, max(10, threshold - 10), 255, cv2.THRESH_BINARY)
    
    # Combine: strong ink or weak ink that's actually dark in original
    dark_in_original = g_img < 180  # Handwritten ink is typically dark
    mask = np.where(dark_in_original, mask_weak, mask_strong).astype(np.uint8)
    
    # 5. Morphological cleanup
    # Small dilation to connect broken strokes
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    # Remove small noise (dots, specks)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    if not return_ink_image:
        return mask
    
    # Return an OCR-friendly image with template text removed
    if len(image.shape) == 3:
        ink_image = image.copy()
        # Whiten non-ink pixels
        ink_image[mask == 0] = [255, 255, 255]
    else:
        ink_image = g_img.copy()
        ink_image[mask == 0] = 255
        ink_image = cv2.cvtColor(ink_image, cv2.COLOR_GRAY2RGB)
    
    return ink_image


def extract_ink_advanced(
    aligned_image: np.ndarray,
    template_image: np.ndarray,
    enhance_contrast: bool = True
) -> np.ndarray:
    """
    Advanced ink extraction for handwritten forms.
    Uses multiple techniques to isolate handwritten ink from pre-printed template.
    
    Args:
        aligned_image: Scanned form aligned to template space
        template_image: Blank template image
        enhance_contrast: Apply contrast enhancement to improve ink visibility
        
    Returns:
        RGB image with only handwritten ink visible (template whitened)
    """
    # Ensure RGB
    if len(aligned_image.shape) == 2:
        aligned_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2RGB)
    else:
        aligned_rgb = aligned_image.copy()
        
    if len(template_image.shape) == 2:
        template_rgb = cv2.cvtColor(template_image, cv2.COLOR_GRAY2RGB)
    else:
        template_rgb = template_image.copy()
    
    # Resize if needed
    if aligned_rgb.shape[:2] != template_rgb.shape[:2]:
        aligned_rgb = cv2.resize(aligned_rgb, (template_rgb.shape[1], template_rgb.shape[0]))
    
    # Convert to LAB for better color separation
    aligned_lab = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2LAB)
    template_lab = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2LAB)
    
    # Compute difference in L channel (luminance)
    l_aligned = aligned_lab[:, :, 0].astype(np.float32)
    l_template = template_lab[:, :, 0].astype(np.float32)
    l_diff = np.abs(l_aligned - l_template)
    
    # Also check if pixel is darker than template (ink adds darkness)
    is_darker = l_aligned < (l_template - 5)
    
    # Ink mask: significant difference AND darker than template
    ink_mask = ((l_diff > 15) & is_darker).astype(np.uint8) * 255
    
    # Also detect very dark pixels (pen ink is typically very dark)
    very_dark = (l_aligned < 120).astype(np.uint8) * 255
    
    # Combine masks
    combined_mask = cv2.bitwise_or(ink_mask, very_dark)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Create output image
    output = np.full_like(aligned_rgb, 255)  # White background
    output[combined_mask > 0] = aligned_rgb[combined_mask > 0]
    
    if enhance_contrast:
        # Enhance contrast of the ink
        output_gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(output_gray)
        output = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        # Restore white background
        output[combined_mask == 0] = [255, 255, 255]
    
    return output

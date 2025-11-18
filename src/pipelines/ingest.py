"""
Document ingestion pipeline: PDF/image loading and preprocessing.
"""
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import pdf2image
from io import BytesIO
import re

from utils.models import PageImage, WordBox
from utils.config import Config
from src.processing.preprocessing import preprocess_image
from src.processing.gpu_utils import GPUUtils


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale."""
    if image is None:
        return image
    if len(image.shape) == 3:
        import cv2
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def _estimate_orientation_from_masks(horizontal_mask: np.ndarray, vertical_mask: np.ndarray) -> Tuple[float, float]:
    """
    Estimate coarse page orientation (0 or 90 degrees) using horizontal vs vertical line energy.
    
    Returns orientation_degrees and confidence (0-1).
    """
    if horizontal_mask is None or vertical_mask is None:
        return 0.0, 0.0
    horiz_score = float(np.sum(horizontal_mask) / 255.0) if horizontal_mask.size else 0.0
    vert_score = float(np.sum(vertical_mask) / 255.0) if vertical_mask.size else 0.0
    total = horiz_score + vert_score
    if total <= 0:
        return 0.0, 0.0
    if vert_score > horiz_score * 1.25:
        orientation = 90.0
    else:
        orientation = 0.0
    confidence = abs(vert_score - horiz_score) / total
    return orientation, float(min(max(confidence, 0.0), 1.0))


def _generate_analysis_layers(image: np.ndarray) -> Dict[str, Any]:
    """
    Generate analysis-friendly layers (high-contrast, binary, line masks, box masks).
    """
    result: Dict[str, Any] = {}
    if image is None:
        return result
    try:
        import cv2
    except ImportError:
        return result
    
    gray = _ensure_gray(image)
    if gray is None:
        return result
    
    use_gpu = GPUUtils.is_available()
    if use_gpu:
        try:
            high_contrast = GPUUtils.clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8))
        except Exception:
            high_contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        high_contrast = clahe.apply(gray)
    
    # Adaptive threshold for binary analysis image
    binary = cv2.adaptiveThreshold(
        high_contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )
    
    height, width = binary.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(15, width // 80), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(15, height // 80))
    )
    
    try:
        if use_gpu:
            horizontal_lines = GPUUtils.morphology(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = GPUUtils.morphology(binary, cv2.MORPH_OPEN, vertical_kernel)
        else:
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    except Exception:
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    line_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    box_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    try:
        if use_gpu:
            box_mask = GPUUtils.morphology(binary, cv2.MORPH_CLOSE, box_kernel)
        else:
            box_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, box_kernel, iterations=1)
    except Exception:
        box_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, box_kernel, iterations=1)
    
    orientation, orientation_conf = _estimate_orientation_from_masks(horizontal_lines, vertical_lines)
    
    result.update({
        "analysis_image": high_contrast,
        "binary_image": binary,
        "line_mask": line_mask,
        "box_mask": box_mask,
        "orientation": orientation,
        "orientation_confidence": orientation_conf
    })
    return result


def detect_and_correct_rotation(image: np.ndarray, page_id: int = 0) -> np.ndarray:
    """
    Detect and correct 90/180/270 degree page rotation.
    DISABLED BY DEFAULT - only rotates if explicitly needed.
    Most PDFs are already correctly oriented, so we don't auto-rotate.
    
    Args:
        image: Input image
        page_id: Page number for logging
        
    Returns:
        Corrected image (usually unchanged)
    """
    # DISABLED: Auto-rotation is causing issues with correctly oriented PDFs
    # If you need rotation detection, enable it explicitly via environment variable
    import os
    enable_rotation = os.getenv("ENABLE_ROTATION_DETECTION", "false").lower() == "true"
    
    if not enable_rotation:
        # Skip rotation detection - assume PDFs are already correctly oriented
        return image
    
    # Only proceed if explicitly enabled
    try:
        import cv2
    except ImportError:
        return image
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape
    
    # Use Tesseract OSD (Orientation and Script Detection) - most reliable
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(image)
        # Parse rotation from OSD output
        import re
        rotation_match = re.search(r'Rotate: (\d+)', osd)
        if rotation_match:
            rotation = int(rotation_match.group(1))
            # Only rotate if rotation is significant (not 0)
            if rotation == 90:
                print(f"  🔄 Page {page_id}: Detected 90° rotation, correcting...")
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                print(f"  🔄 Page {page_id}: Detected 180° rotation, correcting...")
                return cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 270:
                print(f"  🔄 Page {page_id}: Detected 270° rotation, correcting...")
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # If rotation is 0, page is already correct
            return image
    except Exception as e:
        # Tesseract OSD not available or failed
        pass
    
    # Fallback disabled - too error-prone
    return image


def extract_digital_text(pdf_path: str) -> List[dict]:
    """
    Extract digital text layer from PDF with word bounding boxes.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of pages with word boxes
    """
    doc = fitz.open(pdf_path)
    pages_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        words = page.get_text("words")  # Get words with bboxes in PDF coordinate space (points)
        rect = page.rect  # page rectangle in points
        page_w_pts = float(rect.width)
        page_h_pts = float(rect.height)

        words_data = []
        for word in words:
            words_data.append({
                "text": word[4],  # Text content
                "bbox": [word[0], word[1], word[2], word[3]],  # [x0, y0, x1, y1] in points
                "page": page_num
            })

        pages_data.append({
            "page": page_num,
            "page_width_pts": page_w_pts,
            "page_height_pts": page_h_pts,
            "words": words_data,
            "has_digital_text": len(words_data) > 0
        })
    
    doc.close()
    return pages_data


def pdf_to_images_pymupdf(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    """
    Convert PDF to images using PyMuPDF.
    PyMuPDF honors PDF /Rotate metadata, then we auto-rotate if bitmap is still sideways.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution in DPI
        
    Returns:
        List of page images as numpy arrays (all pages upright)
    """
    doc = fitz.open(pdf_path)
    images = []
    
    zoom = dpi / 72.0  # PyMuPDF uses 72 DPI as base
    mat = fitz.Matrix(zoom, zoom)
    
    for page_id, page in enumerate(doc):
        # Get pixmap - PyMuPDF automatically honors PDF /Rotate metadata
        # This is CORRECT - we trust PyMuPDF's rendering
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img_data = pix.tobytes("ppm")
        img = Image.open(BytesIO(img_data))
        img_array = np.array(img.convert('RGB'))
        
        # NO autorotate - PyMuPDF already handles orientation correctly
        # Adding autorotate was causing correctly oriented PDFs to be rotated incorrectly
        images.append(img_array)
    
    doc.close()
    return images


def pdf_to_images_pdf2image(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    """
    Convert PDF to images using pdf2image.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution in DPI
        
    Returns:
        List of page images as numpy arrays
    """
    images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
    # NO autorotate - pdf2image handles orientation correctly
    return [np.array(img.convert("RGB")) for img in images]


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array
    """
    img = Image.open(image_path)
    return np.array(img.convert('RGB'))


def ingest_document(
    file_path: str,
    dpi: int = None,
    deskew: bool = None,
    denoise: bool = None,
    use_pymupdf: bool = True
) -> List[PageImage]:
    """
    Ingest document (PDF or image) and return list of PageImage objects.
    
    Args:
        file_path: Path to document file
        dpi: Resolution for PDF conversion (default: Config.DPI)
        deskew: Whether to apply de-skewing (default: Config.DESKEW_ENABLED)
        denoise: Whether to apply de-noising (default: Config.DENOISE_ENABLED)
        use_pymupdf: Whether to use PyMuPDF (True) or pdf2image (False) for PDF
        
    Returns:
        List of PageImage objects
    """
    if dpi is None:
        dpi = Config.DPI
    if deskew is None:
        deskew = Config.DESKEW_ENABLED
    if denoise is None:
        denoise = Config.DENOISE_ENABLED
    
    file_path = Path(file_path)
    pages = []
    digital_text_data = None
    
    # Handle PDF
    if file_path.suffix.lower() == '.pdf':
        # Try to extract digital text layer
        try:
            digital_text_data = extract_digital_text(str(file_path))
        except Exception as e:
            print(f"Warning: Could not extract digital text from PDF: {e}")
        
        # Convert PDF to images
        try:
            if use_pymupdf:
                images = pdf_to_images_pymupdf(str(file_path), dpi=dpi)
            else:
                images = pdf_to_images_pdf2image(str(file_path), dpi=dpi)
        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            raise
    
    # Handle images
    elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        images = [load_image(str(file_path))]
        digital_text_data = None
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Process each page
    for page_id, image in enumerate(images):
        # NO rotation detection - PDFs are ingested exactly as they are
        # The PDF conversion already handles orientation correctly
        
        # Preprocess image
        processed_image, preprocess_meta = preprocess_image(
            image,
            deskew=deskew,
            denoise=denoise
        )
        
        analysis_layers = _generate_analysis_layers(processed_image)
        orientation = analysis_layers.get("orientation", 0.0)
        orientation_conf = analysis_layers.get("orientation_confidence", 0.0)
        preprocess_meta["analysis_generated"] = bool(analysis_layers)
        preprocess_meta["orientation_estimate"] = orientation
        preprocess_meta["orientation_confidence"] = orientation_conf
        
        # Check if page has digital text and store word boxes if available
        has_digital_text = False
        page_digital_words: List[WordBox] = []
        if digital_text_data and page_id < len(digital_text_data):
            page_info = digital_text_data[page_id]
            has_digital_text = page_info.get("has_digital_text", False)
            # Scale PDF point-space word boxes to pixel coordinates of rendered image
            words_list = list(page_info.get("words", []))
            pw = float(page_info.get("page_width_pts") or 0.0) or 0.0
            ph = float(page_info.get("page_height_pts") or 0.0) or 0.0
            scale_x = (processed_image.shape[1] / pw) if pw > 0 else 1.0
            scale_y = (processed_image.shape[0] / ph) if ph > 0 else 1.0
            scaled_words: List[WordBox] = []
            for w in words_list:
                bbox = w.get("bbox") if isinstance(w, dict) else None
                text = w.get("text") if isinstance(w, dict) else None
                if not bbox or len(bbox) < 4:
                    continue
                x0, y0, x1, y1 = bbox[:4]
                sx0 = float(x0) * scale_x
                sy0 = float(y0) * scale_y
                sx1 = float(x1) * scale_x
                sy1 = float(y1) * scale_y
                scaled_words.append(WordBox(text=str(text or ""), bbox=(sx0, sy0, sx1, sy1), confidence=1.0))
            page_digital_words = scaled_words
            preprocess_meta["digital_text_extracted"] = has_digital_text and len(scaled_words) > 0
        
        # Create PageImage object
        height, width = processed_image.shape[:2]
        page_image = PageImage(
            image=processed_image,
            page_id=page_id,
            width=width,
            height=height,
            dpi=dpi,
            digital_text=has_digital_text,
            digital_words=page_digital_words,
            preprocess_metadata=preprocess_meta,
            analysis_image=analysis_layers.get("analysis_image"),
            binary_image=analysis_layers.get("binary_image"),
            line_mask=analysis_layers.get("line_mask"),
            box_mask=analysis_layers.get("box_mask"),
            orientation=orientation,
            orientation_confidence=orientation_conf
        )
        
        pages.append(page_image)
    
    return pages

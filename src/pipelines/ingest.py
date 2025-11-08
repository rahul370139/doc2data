"""
Document ingestion pipeline: PDF/image loading and preprocessing.
"""
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple
import pdf2image
from io import BytesIO
import re

from utils.models import PageImage
from utils.config import Config
from src.processing.preprocessing import preprocess_image


def _autorotate(image: np.ndarray, page_id: int = 0) -> np.ndarray:
    """
    Detect if bitmap is rotated 90/180/270° and fix it.
    Returns the original image if orientation is already correct.
    
    This handles cases where the PDF page bitmap itself is stored sideways,
    regardless of PDF /Rotate metadata.
    """
    try:
        import cv2
    except ImportError:
        return image
    
    # Robust check: score 0/90/180/270 with geometry + tiny OCR and only rotate
    # if the best orientation clearly improves over 0°. This avoids flipping
    # forms that have strong vertical structure.
    try:
        import pytesseract
        import cv2
        # Downscale for speed
        h, w = image.shape[:2]
        scale = 1200.0 / max(h, w)
        if scale < 1.0:
            small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = image
        rotations = {
            0: small,
            90: cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(small, cv2.ROTATE_180),
            270: cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }

        def line_score(img: np.ndarray) -> Tuple[float, float]:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=120,
                minLineLength=max(img.shape[1] // 5, 80),
                maxLineGap=15
            )
            if lines is None:
                return 0.0, 0.0
            horiz_len = 0.0
            vert_len = 0.0
            for l in lines:
                x1, y1, x2, y2 = l[0]
                dx, dy = x2 - x1, y2 - y1
                length = float(np.hypot(dx, dy))
                ang = abs(np.degrees(np.arctan2(dy, dx)))
                if ang < 10 or ang > 170:
                    horiz_len += length
                elif 80 < ang < 100:
                    vert_len += length
            return horiz_len, vert_len

        def ocr_score(img: np.ndarray) -> float:
            txt = pytesseract.image_to_string(img, config="--psm 6")
            alnum = sum(ch.isalnum() for ch in txt)
            length = max(len(txt), 1)
            return alnum / float(length)

        combined_scores = {}
        for ang, img in rotations.items():
            hlen, vlen = line_score(img)
            ls = (hlen - 0.5 * vlen) / max(img.shape[0] * img.shape[1] / 2000.0, 1.0)
            os = ocr_score(img)
            combined = 0.75 * ls + 0.25 * os
            combined_scores[ang] = combined

        # Choose best angle but require strong margin over 0°
        best_angle = max(combined_scores, key=lambda k: combined_scores[k])
        gain = combined_scores[best_angle] - combined_scores.get(0, 0.0)
        # Absolute and relative margins
        margin_abs = 0.15
        if best_angle != 0 and gain > margin_abs:
            clockwise = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE
            }
            print(f"  🔄 Page {page_id}: combined scoring {combined_scores} → rotate {best_angle}° (gain={gain:.3f})")
            return cv2.rotate(image, clockwise[best_angle])
    except Exception:
        pass
    
    # Fallback: Use Hough line detection to detect text orientation
    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None and len(lines) > 5:
            # Count horizontal vs vertical lines
            horizontal = sum(1 for l in lines if abs(l[0][1]) < np.pi/8)  # ~0° or ~180°
            vertical = sum(1 for l in lines if abs(l[0][1] - np.pi/2) < np.pi/8)  # ~90°
            
            # If predominantly vertical lines, page is sideways
            if vertical > horizontal * 2:
                print(f"  🔄 Page {page_id}: detected sideways bitmap (vertical lines: {vertical}, horizontal: {horizontal}), correcting…")
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    except Exception:
        pass
    
    return image


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
        words = page.get_text("words")  # Get words with bboxes
        
        words_data = []
        for word in words:
            words_data.append({
                "text": word[4],  # Text content
                "bbox": [word[0], word[1], word[2], word[3]],  # [x0, y0, x1, y1]
                "page": page_num
            })
        
        pages_data.append({
            "page": page_num,
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
        
        # Check if page has digital text
        has_digital_text = False
        if digital_text_data and page_id < len(digital_text_data):
            has_digital_text = digital_text_data[page_id]["has_digital_text"]
            preprocess_meta["digital_text_extracted"] = has_digital_text
        
        # Create PageImage object
        height, width = processed_image.shape[:2]
        page_image = PageImage(
            image=processed_image,
            page_id=page_id,
            width=width,
            height=height,
            dpi=dpi,
            digital_text=has_digital_text,
            preprocess_metadata=preprocess_meta
        )
        
        pages.append(page_image)
    
    return pages


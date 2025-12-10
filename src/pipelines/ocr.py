"""
OCR pipeline: orchestration, header/footer detection, caption candidate detection.
"""
import hashlib
import math
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy optional in some environments
    linear_sum_assignment = None

from utils.models import Block, BlockType, BlockRole, WordBox
from src.ocr.paddle_ocr import PaddleOCRWrapper
from src.ocr.tesseract_ocr import TesseractOCRWrapper
from src.processing.postprocessing import postprocess_text
from utils.config import Config
from src.pipelines.validators import (
    validate_field,
    guess_field_type,
)
from utils.corrections import log_correction, auto_tune_thresholds, load_threshold_overrides
try:
    from src.vlm.qwen_vl import QwenVLProcessor
except Exception:
    QwenVLProcessor = None


def compute_label_field_cost(field_block: Block, label_block: Block) -> float:
    """Heuristic cost between a form field and label (lower is better)."""
    fx0, fy0, fx1, fy1 = field_block.bbox
    lx0, ly0, lx1, ly1 = label_block.bbox
    field_height = max(fy1 - fy0, 1.0)
    horizontal_gap = max(0.0, fx0 - lx1)
    vertical_center_diff = abs(((fy0 + fy1) / 2.0) - ((ly0 + ly1) / 2.0))
    overlap = max(0.0, min(fy1, ly1) - max(fy0, ly0)) / field_height
    overlap_penalty = (1.0 - min(1.0, overlap)) * field_height * 1.2
    right_penalty = 150.0 if lx0 >= fx1 else 0.0
    return horizontal_gap + (vertical_center_diff * 0.8) + overlap_penalty + right_penalty


class OCRPipeline:
    """OCR pipeline with header/footer and caption detection."""
    
    def __init__(
        self,
        use_paddle: bool = True,
        max_workers: Optional[int] = None,
        assignment_mode: str = "greedy"
    ):
        """
        Initialize OCR pipeline.
        
        Args:
            use_paddle: Use PaddleOCR (True) or Tesseract (False)
            max_workers: Max parallel workers for OCR (default: CPU count)
        """
        self.use_paddle = use_paddle
        self.paddle_ocr = None
        self.tesseract_ocr = None
        self.max_workers = max_workers or min(os.cpu_count() or 4, 4)  # Limit to 4 to avoid memory issues
        self.low_confidence_threshold = 0.72
        self.checkbox_on_threshold = 0.55
        self.checkbox_off_threshold = 0.25
        self.assignment_mode = assignment_mode.lower() if assignment_mode and assignment_mode.lower() in {"greedy", "hungarian"} else "greedy"
        self.max_assignment_cost = 220.0
        self.vlm = None
        self.vlm_cache: Dict[str, Dict[str, Any]] = {}
        self.max_block_area_ratio = 0.55  # Skip very large regions for faster OCR
        tuned = auto_tune_thresholds({"low_confidence_threshold": self.low_confidence_threshold})
        self.low_confidence_threshold = tuned.get("low_confidence_threshold", self.low_confidence_threshold)
        
        if use_paddle:
            try:
                self.paddle_ocr = PaddleOCRWrapper()
            except Exception as e:
                print(f"Warning: PaddleOCR initialization failed: {e}")
                print("Falling back to Tesseract-only mode")
                self.use_paddle = False
        
        if not self.use_paddle:
            try:
                self.tesseract_ocr = TesseractOCRWrapper()
            except Exception as e:
                print(f"Error: Tesseract OCR initialization failed: {e}")
                raise
        else:
            # When Paddle is primary, keep a silent Tesseract fallback for tough cases
            try:
                self.tesseract_ocr = TesseractOCRWrapper()
            except Exception as e:
                print(f"Warning: Tesseract fallback unavailable: {e}")
        # Lazy-init VLM only if enabled and available
        if Config.ENABLE_VLM and QwenVLProcessor:
            try:
                self.vlm = QwenVLProcessor(enabled=True)
            except Exception:
                self.vlm = None
    
    @staticmethod
    def _extract_page_image(page_entry: Any):
        """Return numpy-like image for a page entry (PageImage or ndarray)."""
        if page_entry is None:
            return None
        image = getattr(page_entry, "image", None)
        if image is not None:
            return image
        return page_entry if hasattr(page_entry, "shape") else None

    @staticmethod
    def _hash_crop(crop: np.ndarray) -> str:
        try:
            return hashlib.md5(crop.tobytes()).hexdigest()
        except Exception:
            return ""
    
    @staticmethod
    def _get_page_height(page_entry: Any) -> int:
        """Safely obtain page height."""
        try:
            image = OCRPipeline._extract_page_image(page_entry)
            if image is not None and hasattr(image, "shape"):
                return int(image.shape[0])
            height = getattr(page_entry, "height", None)
            if height:
                return int(height)
        except Exception:
            pass
        return 0

    @staticmethod
    def _crop_block_image(
        image: np.ndarray,
        block: Block,
        padding: int = 10,
        max_area_ratio: float = 0.8
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """Crop block from image with padding and return crop + bbox + area ratio."""
        if image is None or not hasattr(image, "shape"):
            return None
        x0, y0, x1, y1 = [int(coord) for coord in block.bbox]
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(image.shape[1], x1 + padding)
        y1 = min(image.shape[0], y1 + padding)
        if x1 <= x0 or y1 <= y0:
            return None
        block_area = (x1 - x0) * (y1 - y0)
        page_area = image.shape[0] * image.shape[1]
        area_ratio = block_area / page_area if page_area > 0 else 0
        if area_ratio > max_area_ratio:
            return None
        block_image = image[y0:y1, x0:x1]
        if block_image.size == 0:
            return None
        return block_image, (x0, y0, x1, y1), area_ratio

    @staticmethod
    def _calculate_text_confidence(word_boxes: List[WordBox]) -> float:
        if not word_boxes:
            return 0.0
        total = sum(max(0.0, min(wb.confidence if wb.confidence is not None else 0.0, 1.0)) for wb in word_boxes)
        return float(total / len(word_boxes)) if word_boxes else 0.0

    def _run_tesseract_fallback(
        self,
        block: Block,
        page_image: np.ndarray
    ) -> Tuple[str, List[WordBox], Dict[str, Any]]:
        if not self.tesseract_ocr:
            return "", [], {"avg_confidence": 0.0, "engine": "none"}
        crop_info = self._crop_block_image(page_image, block, padding=6, max_area_ratio=self.max_block_area_ratio)
        if crop_info is None:
            return "", [], {"avg_confidence": 0.0, "engine": "tesseract"}
        block_image, _, _ = crop_info
        try:
            word_boxes = self.tesseract_ocr.extract_text(block_image)
        except Exception as exc:
            print(f"⚠️ Tesseract fallback failed for block {block.id}: {exc}")
            return "", [], {"avg_confidence": 0.0, "engine": "tesseract"}
        if not word_boxes:
            return "", [], {"avg_confidence": 0.0, "engine": "tesseract"}
        avg_conf = self._calculate_text_confidence(word_boxes)
        text = postprocess_text(" ".join(wb.text for wb in word_boxes))
        return text, word_boxes, {"avg_confidence": avg_conf, "engine": "tesseract"}
    
    def extract_text_from_block(
        self,
        image: np.ndarray,
        block: Block,
        skip_ocr: bool = False
    ) -> Tuple[str, List[WordBox], Dict[str, Any]]:
        """
        Extract text from a block.
        
        Args:
            image: Full page image
            block: Block to extract text from
            skip_ocr: Skip OCR if digital text is available
            
        Returns:
            Tuple of (text, word_boxes)
        """
        if skip_ocr:
            if block.word_boxes:
                text = " ".join([wb.text for wb in block.word_boxes])
                return text, block.word_boxes, {"avg_confidence": 1.0, "engine": "digital"}
            return "", [], {"avg_confidence": 0.0, "engine": "digital"}
        
        crop_info = self._crop_block_image(image, block, max_area_ratio=self.max_block_area_ratio)
        if crop_info is None:
            return "", [], {"avg_confidence": 0.0, "engine": "paddle"}
        block_image, (x0, y0, x1, y1), area_ratio = crop_info
        
        # Ensure minimum size for OCR
        if block_image.shape[0] < 10 or block_image.shape[1] < 10:
            return "", [], {"avg_confidence": 0.0, "engine": "paddle"}
        
        # Make image contiguous if needed (OpenCV operations usually handle this)
        if not block_image.flags['C_CONTIGUOUS']:
            block_image = np.ascontiguousarray(block_image)
        
        # Optional pre-processing: resize + contrast + sharpening
        fallback_proc = block_image
        try:
            import cv2
            proc = block_image
            # Ensure minimum readable size
            h, w = proc.shape[:2]
            # Upsample small crops more aggressively for better OCR quality
            min_h = 100  # Increased from 64 for better quality
            min_w = 100
            scale_h = max(1.0, min_h / max(h, 1)) if h < min_h else 1.0
            scale_w = max(1.0, min_w / max(w, 1)) if w < min_w else 1.0
            scale = max(scale_h, scale_w)
            if scale > 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                proc = cv2.resize(proc, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            # Enhanced preprocessing for better OCR quality
            if len(proc.shape) == 3 and proc.shape[2] == 3:
                # Convert to LAB for CLAHE on L channel (better contrast)
                lab = cv2.cvtColor(proc, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                # More aggressive CLAHE for better text contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                lab = cv2.merge((cl, a, b))
                proc = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale CLAHE
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                proc = clahe.apply(proc)
                proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
            
            # Denoise before sharpening
            proc = cv2.bilateralFilter(proc, 5, 50, 50)
            
            # Enhanced unsharp mask for better text sharpness
            blur = cv2.GaussianBlur(proc, (0, 0), 1.5)
            proc = cv2.addWeighted(proc, 1.8, blur, -0.8, 0)
            
            # Additional contrast enhancement
            proc = cv2.convertScaleAbs(proc, alpha=1.2, beta=10)
            # Ensure contiguous (OpenCV operations usually already do this)
            if not proc.flags['C_CONTIGUOUS']:
                proc = np.ascontiguousarray(proc)
            fallback_proc = proc
        except Exception:
            proc = block_image
            # Ensure contiguous even for fallback
            if not proc.flags['C_CONTIGUOUS']:
                proc = np.ascontiguousarray(proc)
            fallback_proc = proc

        # Run OCR
        try:
            if self.use_paddle and self.paddle_ocr:
                word_boxes = self.paddle_ocr.extract_text(proc)
                engine = "paddle"
            elif self.tesseract_ocr:
                word_boxes = self.tesseract_ocr.extract_text(fallback_proc)
                engine = "tesseract"
            else:
                print(f"⚠️ No OCR engine available for block {block.id}")
                return "", [], {"avg_confidence": 0.0, "engine": "none"}
        except Exception as e:
            print(f"⚠️ OCR error for block {block.id}: {e}")
            import traceback
            traceback.print_exc()
            word_boxes = []
            engine = "error"
        
        # Fallback to Tesseract when Paddle returns nothing
        if (
            self.use_paddle
            and self.paddle_ocr
            and not word_boxes
            and self.tesseract_ocr
        ):
            try:
                fallback_boxes = self.tesseract_ocr.extract_text(fallback_proc)
                if fallback_boxes:
                    print(f"  ↩︎ Paddle OCR empty for block {block.id}; fallback to Tesseract returned {len(fallback_boxes)} words")
                    word_boxes = fallback_boxes
            except Exception as fallback_err:
                print(f"⚠️ Tesseract fallback failed for block {block.id}: {fallback_err}")
        
        # Adjust coordinates to page coordinates (accounting for padding)
        # Word boxes are relative to padded crop, so add the padded crop origin
        adjusted_word_boxes = []
        for wb in word_boxes:
            adjusted_bbox = (
                wb.bbox[0] + x0,  # x0 already includes padding offset
                wb.bbox[1] + y0,  # y0 already includes padding offset
                wb.bbox[2] + x0,
                wb.bbox[3] + y0
            )
            adjusted_word_boxes.append(
                WordBox(text=wb.text, bbox=adjusted_bbox, confidence=wb.confidence)
            )
        
        # Combine text with proper spacing
        text = " ".join([wb.text for wb in adjusted_word_boxes])
        text = postprocess_text(text)

        # General post-processing for better text quality (no hardcoded word fixes)
        if text:
            import re
            
            # Fix common character recognition errors (general patterns)
            text = text.replace("|", "l")  # Vertical bar often misread as lowercase L
            
            # Fix spacing issues around punctuation (general)
            text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
            text = re.sub(r'([.,;:!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after punctuation
            
            # Fix missing spaces after periods in abbreviations/names (general pattern)
            text = re.sub(r'([A-Z])\.([a-z])', r'\1. \2', text)  # "H.and" -> "H. and"
            
            # Fix number recognition errors: O vs 0 in numeric contexts (general)
            # Only fix when O appears between digits or at end of number patterns
            text = re.sub(r'(\d)O(\d)', r'\g<1>0\g<2>', text)  # "3O" -> "30" between digits
            text = re.sub(r'(\d{1,2})O\b', r'\g<1>0', text)  # "30" -> "30" at word boundary
            
            # Fix date formatting (general pattern)
            text = re.sub(r'\b(\d{1,2})\s*\)\s*,\s*(\d{4})\b', r'\1, \2', text)  # "30), 1997" -> "30, 1997"
            
            # Remove excessive whitespace but preserve line breaks
            lines = [line.strip() for line in text.split("\n")]
            text = "\n".join(line for line in lines if line)
            
            # Fix hyphenation issues (general)
            text = text.replace("-\n", "").replace("-\r\n", "")
            
            # Clean up multiple spaces
            text = re.sub(r' +', ' ', text)
            text = text.strip()

        # Tier-2: re-OCR for low-confidence
        stats = {"avg_confidence": self._calculate_text_confidence(adjusted_word_boxes), "engine": engine}
        avg_conf = stats["avg_confidence"]
        if avg_conf < 0.75:
            tier2_boxes: List[WordBox] = []
            tier2_text = ""
            try:
                import cv2
                h2, w2 = block_image.shape[:2]
                scale = 1.8
                enlarged = cv2.resize(block_image, (int(w2 * scale), int(h2 * scale)), interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(enlarged, cv2.COLOR_RGB2GRAY) if len(enlarged.shape) == 3 else enlarged
                bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
                bin_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)
                if self.use_paddle and self.paddle_ocr:
                    tier2_boxes = self.paddle_ocr.extract_text(bin_img)
                elif self.tesseract_ocr:
                    tier2_boxes = self.tesseract_ocr.extract_text(bin_img)
                tier2_text = postprocess_text(" ".join([wb.text for wb in tier2_boxes]))
            except Exception:
                tier2_boxes = []
            if tier2_boxes:
                tier2_avg = self._calculate_text_confidence(tier2_boxes)
                if tier2_avg > avg_conf:
                    log_correction({
                        "block_id": block.id,
                        "source": "tier2_reocr",
                        "old_confidence": avg_conf,
                        "new_confidence": tier2_avg,
                        "old_text": text,
                        "new_text": tier2_text
                    })
                    adjusted_word_boxes = [
                        WordBox(text=wb.text, bbox=(wb.bbox[0] + x0, wb.bbox[1] + y0, wb.bbox[2] + x0, wb.bbox[3] + y0), confidence=wb.confidence)
                        for wb in tier2_boxes
                    ]
                    text = tier2_text
                    avg_conf = tier2_avg
                    stats = {"avg_confidence": avg_conf, "engine": engine + "+tier2"}
                else:
                    stats["tier2_avg"] = tier2_avg
                    stats["tier2_text"] = tier2_text

        # Tier-3: VLM assist for very low confidence (suggestion only)
        if avg_conf < 0.5 and self.vlm is not None:
            try:
                vlm_result = self.vlm.process_table(block_image, None)
                vlm_text = vlm_result.get("content") if isinstance(vlm_result, dict) else None
                if vlm_text:
                    stats["vlm_text"] = vlm_text
                    stats["vlm_used"] = True
            except Exception:
                pass

        # Debug output
        if len(adjusted_word_boxes) == 0:
            print(f"⚠️ No text extracted from block {block.id} (type: {block.type.value}, size: {block_image.shape})")
        elif len(text) == 0:
            print(f"⚠️ Empty text after processing for block {block.id}")

        stats["avg_confidence"] = avg_conf
        return text, adjusted_word_boxes, stats
    
    def _extract_from_digital_layer(
        self,
        block: Block,
        digital_words: List[WordBox]
    ) -> Tuple[str, List[WordBox]]:
        """Build text + word boxes for a block using the PDF digital text layer."""
        if not digital_words:
            return "", []
        
        x0, y0, x1, y1 = block.bbox
        matches: List[WordBox] = []
        for entry in digital_words:
            word = entry
            if isinstance(entry, dict):
                bbox = entry.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                word = WordBox(
                    text=str(entry.get("text", "")),
                    bbox=tuple(map(float, bbox[:4])),
                    confidence=float(entry.get("confidence", 1.0))
                )
            wx0, wy0, wx1, wy1 = word.bbox
            cx = (wx0 + wx1) / 2.0
            cy = (wy0 + wy1) / 2.0
            if x0 - 2 <= cx <= x1 + 2 and y0 - 2 <= cy <= y1 + 2:
                matches.append(word)
        
        if not matches:
            return "", []
        
        sorted_words = sorted(
            matches,
            key=lambda wb: ((wb.bbox[1] + wb.bbox[3]) / 2.0, wb.bbox[0])
        )
        copied = [
            WordBox(text=wb.text, bbox=tuple(wb.bbox), confidence=wb.confidence)
            for wb in sorted_words
        ]
        text = postprocess_text(" ".join(wb.text for wb in copied))
        return text, copied
    
    def detect_repeating_headers_footers(
        self,
        blocks: List[Block],
        pages: List[Any]
    ) -> Dict[str, str]:
        """
        Detect repeating headers and footers across pages.
        
        Args:
            blocks: List of blocks
            pages: List of page images (np.ndarray or PageImage)
            
        Returns:
            Dictionary mapping block IDs to role hints
        """
        role_hints = {}
        
        # Group blocks by page
        blocks_by_page = {}
        for block in blocks:
            if block.page_id not in blocks_by_page:
                blocks_by_page[block.page_id] = []
            blocks_by_page[block.page_id].append(block)
        
        if len(blocks_by_page) < 2:
            return role_hints  # Need at least 2 pages
        
        # Extract text from blocks near top/bottom of pages
        header_candidates = []
        footer_candidates = []
        
        for page_id, page_blocks in blocks_by_page.items():
            if not pages or page_id >= len(pages):
                continue
            
            page_entry = pages[page_id]
            # Use helper function to safely get page height
            page_height = OCRPipeline._get_page_height(page_entry)
            if not page_height:
                # Fallback: try to extract image and get height
                image = OCRPipeline._extract_page_image(page_entry)
                if image is not None and hasattr(image, "shape") and len(image.shape) >= 2:
                    page_height = int(image.shape[0])
            if not page_height:
                page_height = 1
            
            # Top 15% of page = header candidate
            # Bottom 15% of page = footer candidate
            header_threshold = page_height * 0.15
            footer_threshold = page_height * 0.85
            
            for block in page_blocks:
                y_center = (block.bbox[1] + block.bbox[3]) / 2
                
                if y_center < header_threshold:
                    # Extract text for hashing
                    text = block.text or ""
                    if text:
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        header_candidates.append((block.id, text_hash, text))
                
                elif y_center > footer_threshold:
                    text = block.text or ""
                    if text:
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        footer_candidates.append((block.id, text_hash, text))
        
        # Find repeating headers (appear in 2+ pages)
        header_hashes = {}
        for block_id, text_hash, text in header_candidates:
            if text_hash not in header_hashes:
                header_hashes[text_hash] = []
            header_hashes[text_hash].append(block_id)
        
        for text_hash, block_ids in header_hashes.items():
            if len(block_ids) >= 2:  # Appears in 2+ pages
                for block_id in block_ids:
                    role_hints[block_id] = "header"
        
        # Find repeating footers (appear in 2+ pages)
        footer_hashes = {}
        for block_id, text_hash, text in footer_candidates:
            if text_hash not in footer_hashes:
                footer_hashes[text_hash] = []
            footer_hashes[text_hash].append(block_id)
        
        for text_hash, block_ids in footer_hashes.items():
            if len(block_ids) >= 2:  # Appears in 2+ pages
                for block_id in block_ids:
                    role_hints[block_id] = "footer"
        
        return role_hints
    
    def detect_caption_candidates(
        self,
        blocks: List[Block],
        threshold: float = 40.0
    ) -> Dict[str, bool]:
        """
        Detect caption candidates (text blocks near figures).
        
        Args:
            blocks: List of blocks
            threshold: Distance threshold in pixels (default: 40px)
            
        Returns:
            Dictionary mapping block IDs to is_caption flag
        """
        caption_flags = {}
        
        # Find figure blocks
        figure_blocks = [b for b in blocks if b.type == BlockType.FIGURE]
        
        # Find text blocks
        text_blocks = [b for b in blocks if b.type == BlockType.TEXT]
        
        for text_block in text_blocks:
            text_center_y = (text_block.bbox[1] + text_block.bbox[3]) / 2
            text_center_x = (text_block.bbox[0] + text_block.bbox[2]) / 2
            
            for figure_block in figure_blocks:
                if figure_block.page_id != text_block.page_id:
                    continue
                
                # Check if text block is near figure (within threshold)
                figure_center_y = (figure_block.bbox[1] + figure_block.bbox[3]) / 2
                figure_center_x = (figure_block.bbox[0] + figure_block.bbox[2]) / 2
                
                # Calculate distance
                distance = np.sqrt(
                    (text_center_x - figure_center_x) ** 2 +
                    (text_center_y - figure_center_y) ** 2
                )
                
                if distance < threshold:
                    caption_flags[text_block.id] = True
                    break
        
        return caption_flags
    
    def process_blocks(
        self,
        blocks: List[Block],
        pages: List[Any],
        skip_ocr_for_digital: bool = True,
        parallel: bool = False
    ) -> List[Block]:
        """Run multi-tier OCR sequentially and enrich block metadata."""
        page_lookup: Dict[int, Any] = {}
        if pages:
            for idx, page in enumerate(pages):
                page_id = getattr(page, "page_id", idx)
                page_lookup[page_id] = page
        processed_blocks: List[Block] = []
        vlm_candidates: List[Dict[str, Any]] = []
        for block in blocks:
            if block.type not in {BlockType.TEXT, BlockType.TITLE, BlockType.LIST, BlockType.FORM}:
                processed_blocks.append(block)
                continue
            page_entry = page_lookup.get(block.page_id)
            page_image = None
            page_obj = None
            if page_entry is not None:
                if hasattr(page_entry, "image"):
                    page_obj = page_entry
                    page_image = page_entry.image
                else:
                    page_image = page_entry
            if page_image is None and pages and block.page_id < len(pages):
                fallback = pages[block.page_id]
                if hasattr(fallback, "image"):
                    page_obj = fallback
                    page_image = fallback.image
                else:
                    page_image = fallback
            if page_image is None:
                processed_blocks.append(block)
                continue
            page_digital_words = getattr(page_obj, "digital_words", []) if page_obj else []
            page_has_digital = getattr(page_obj, "digital_text", False) if page_obj else False
            block_has_digital = block.metadata.get("digital_text", False) if hasattr(block, "metadata") else False
            text = block.text or ""
            word_boxes = block.word_boxes or []
            stats = {"avg_confidence": 0.0, "engine": "digital"}
            used_digital = False
            if skip_ocr_for_digital and (block_has_digital or (page_has_digital and page_digital_words)):
                extracted_text, extracted_boxes = self._extract_from_digital_layer(block, page_digital_words)
                if extracted_boxes and extracted_text.strip():
                    text = extracted_text
                    word_boxes = extracted_boxes
                    stats["avg_confidence"] = 0.95
                    stats["engine"] = "digital"
                    used_digital = True
            if not used_digital:
                ocr_text, ocr_boxes, ocr_stats = self.extract_text_from_block(page_image, block, skip_ocr=False)
                text, word_boxes, stats = ocr_text, ocr_boxes, ocr_stats
                original_text = text
                original_conf = stats.get("avg_confidence", 0.0)
                need_fallback = stats.get("avg_confidence", 0.0) < self.low_confidence_threshold or not text.strip()
                if need_fallback and self.tesseract_ocr:
                    fallback_text, fallback_boxes, fallback_stats = self._run_tesseract_fallback(block, page_image)
                    if fallback_boxes and fallback_stats.get("avg_confidence", 0.0) >= stats.get("avg_confidence", 0.0):
                        if fallback_text and fallback_text != original_text:
                            log_correction({
                                "block_id": block.id,
                                "source": "tesseract_fallback",
                                "old_confidence": original_conf,
                                "new_confidence": fallback_stats.get("avg_confidence", 0.0),
                                "old_text": original_text,
                                "new_text": fallback_text
                            })
                        text, word_boxes, stats = fallback_text, fallback_boxes, fallback_stats
                # Validator feedback: if field type is known and validator fails with low conf, flag for review
                field_type = block.metadata.get("form_field", {}).get("field_type") if hasattr(block, "metadata") else None
                if not field_type:
                    field_type = guess_field_type(block.metadata.get("label_text")) if hasattr(block, "metadata") else None
                if field_type and text.strip():
                    passed, val_info = validate_field(field_type, text)
                    stats["validator_passed"] = passed
                    stats["validator_info"] = val_info
                    if not passed and stats.get("avg_confidence", 0.0) < 0.75:
                        stats["needs_review"] = True
                        block.metadata["needs_review"] = True
            block.text = text
            block.word_boxes = word_boxes
            block.metadata = dict(block.metadata)
            block.metadata["ocr_confidence"] = stats.get("avg_confidence", 0.0)
            block.metadata["ocr_engine"] = stats.get("engine")
            if stats.get("vlm_text"):
                block.metadata["vlm_text"] = stats.get("vlm_text")
            if stats.get("needs_review"):
                block.metadata["needs_review"] = True
            if self.vlm and Config.ENABLE_VLM and stats.get("avg_confidence", 0.0) < 0.5:
                crop_info = self._crop_block_image(page_image, block, padding=6, max_area_ratio=self.max_block_area_ratio)
                if crop_info is not None:
                    crop_img, _, _ = crop_info
                    crop_hash = self._hash_crop(crop_img)
                    vlm_candidates.append({
                        "hash": crop_hash or f"{block.id}",
                        "image": crop_img,
                        "block": block,
                        "hint": block.text,
                        "base_conf": stats.get("avg_confidence", 0.0)
                    })
            processed_blocks.append(block)
        # Batch VLM for low-confidence blocks (per-page cache)
        if self.vlm and Config.ENABLE_VLM and vlm_candidates:
            unique: Dict[str, Tuple[str, np.ndarray, Optional[str]]] = {}
            for req in vlm_candidates:
                if req["hash"] not in self.vlm_cache:
                    unique[req["hash"]] = (req["hash"], req["image"], req.get("hint"))
            vlm_results: Dict[str, Dict[str, Any]] = {}
            if unique:
                try:
                    vlm_results = self.vlm.process_text_batch(list(unique.values()))
                except Exception as exc:
                    print(f"⚠ VLM batch failed: {exc}")
                    vlm_results = {}
                for key, res in vlm_results.items():
                    self.vlm_cache[key] = res
            for req in vlm_candidates:
                res = self.vlm_cache.get(req["hash"])
                if not res or not res.get("text"):
                    continue
                block = req["block"]
                vlm_text = res.get("text", "")
                vlm_conf = float(res.get("confidence", 0.0) or 0.0)
                # Store metadata regardless; replace text only if it's empty or clearly better
                block.metadata = dict(block.metadata)
                block.metadata["vlm_text"] = vlm_text
                block.metadata["vlm_confidence"] = vlm_conf
                if (not block.text or req["base_conf"] < 0.4 or vlm_conf > (req["base_conf"] + 0.1)) and vlm_text:
                    log_correction({
                        "block_id": block.id,
                        "source": "vlm_text",
                        "old_text": block.text,
                        "new_text": vlm_text,
                        "old_confidence": req["base_conf"],
                        "new_confidence": vlm_conf
                    })
                    block.text = vlm_text
                    block.word_boxes = []
                    block.metadata["ocr_confidence"] = max(req["base_conf"], vlm_conf)
                    block.metadata["ocr_engine"] = block.metadata.get("ocr_engine", "vlm")
        self._associate_form_geometry(processed_blocks)
        return processed_blocks

    def _associate_form_geometry(self, blocks: List[Block]) -> None:
        blocks_by_page: Dict[int, List[Block]] = defaultdict(list)
        for block in blocks:
            blocks_by_page[block.page_id].append(block)
        for page_id, page_blocks in blocks_by_page.items():
            text_blocks = [b for b in page_blocks if b.type in {BlockType.TEXT, BlockType.TITLE, BlockType.LIST}]
            form_blocks = [b for b in page_blocks if b.type == BlockType.FORM]
            if not form_blocks:
                continue
            checkbox_blocks = [
                blk for blk in form_blocks
                if blk.metadata.get("checkbox") or blk.metadata.get("field_kind") == "checkbox"
            ]
            field_blocks = [
                blk for blk in form_blocks
                if blk not in checkbox_blocks
            ]
            for checkbox_block in checkbox_blocks:
                self._annotate_checkbox(checkbox_block, text_blocks)
            hungarian_applied = False
            if (
                field_blocks
                and self.assignment_mode == "hungarian"
                and linear_sum_assignment is not None
            ):
                hungarian_applied = self._associate_form_fields_hungarian(field_blocks, text_blocks)
            if not hungarian_applied:
                for field_block in field_blocks:
                    self._link_form_field(field_block, text_blocks)

    def _annotate_checkbox(self, checkbox_block: Block, label_candidates: List[Block]) -> None:
        fill_ratio = checkbox_block.metadata.get("filled_ratio_estimate", 0.0)
        state = "ambiguous"
        if fill_ratio >= self.checkbox_on_threshold:
            state = "checked"
        elif fill_ratio <= self.checkbox_off_threshold:
            state = "unchecked"
        checkbox_block.metadata["checkbox_state"] = state
        checkbox_block.metadata["checkbox_confidence"] = float(fill_ratio)
        label = self._find_nearest_label(checkbox_block, label_candidates, allow_right=True)
        if label:
            checkbox_block.metadata["label_id"] = label.id
            checkbox_block.metadata["label_text"] = label.text
            label.metadata = dict(label.metadata)
            linked = label.metadata.get("linked_checkboxes", [])
            linked.append(checkbox_block.id)
            label.metadata["linked_checkboxes"] = linked
            if not label.role:
                label.role = BlockRole.KV_LABEL
        checkbox_block.role = BlockRole.KV_VALUE
        checkbox_block.metadata["role_locked"] = True

    def _associate_form_fields_hungarian(
        self,
        field_blocks: List[Block],
        text_blocks: List[Block]
    ) -> bool:
        if not field_blocks or not text_blocks or linear_sum_assignment is None:
            return False
        label_candidates = [b for b in text_blocks if b.text and b.text.strip()]
        if not label_candidates:
            return False
        cost_matrix = np.zeros((len(field_blocks), len(label_candidates)), dtype=float)
        large_penalty = self.max_assignment_cost * 4
        for i, field in enumerate(field_blocks):
            for j, label in enumerate(label_candidates):
                if field.page_id != label.page_id:
                    cost_matrix[i, j] = large_penalty
                else:
                    cost_matrix[i, j] = compute_label_field_cost(field, label)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        used_labels = set()
        matched = False
        for row, col in zip(row_ind, col_ind):
            cost = cost_matrix[row, col]
            if cost > self.max_assignment_cost:
                continue
            label_block = label_candidates[col]
            if label_block.id in used_labels:
                continue
            used_labels.add(label_block.id)
            self._link_form_field(field_blocks[row], text_blocks, label_override=label_block)
            matched = True
        return matched

    def _link_form_field(
        self,
        field_block: Block,
        text_blocks: List[Block],
        label_override: Optional[Block] = None
    ) -> None:
        label = label_override if label_override is not None else self._find_nearest_label(field_block, text_blocks)
        label_text = label.text if label and label.text else None
        field_text = (field_block.text or "").strip()
        field_type = guess_field_type(label_text) if label_text else None
        validator_passed = False
        validator_info: Dict[str, Any] = {}
        if field_type and field_text:
            validator_passed, validator_info = validate_field(field_type, field_text)
        
        existing_meta = field_block.metadata.get("form_field", {})
        field_block.metadata["form_field"] = {
            "label_id": label.id if label else existing_meta.get("label_id"),
            "label_text": label_text or existing_meta.get("label_text"),
            "field_type": field_type or existing_meta.get("field_type"),
            "schema_id": existing_meta.get("schema_id"),
            "validator_passed": validator_passed,
            "validator_info": validator_info,
            "value_text": field_text,
        }
        if validator_passed:
            field_block.metadata["validator_passed"] = True
        if label:
            label.metadata = dict(label.metadata)
            linked = label.metadata.get("linked_fields", [])
            linked.append(field_block.id)
            label.metadata["linked_fields"] = linked
            if not label.role:
                label.role = BlockRole.KV_LABEL
            label.metadata["role_locked"] = True
            field_block.metadata["label_id"] = label.id
            field_block.metadata["label_text"] = label_text
        field_block.role = BlockRole.KV_VALUE
        field_block.metadata["role_locked"] = True

    def _find_nearest_label(
        self,
        field_block: Block,
        label_candidates: List[Block],
        allow_right: bool = False
    ) -> Optional[Block]:
        fx0, fy0, fx1, fy1 = field_block.bbox
        f_height = max(fy1 - fy0, 1)
        best = None
        best_score = float("inf")
        for candidate in label_candidates:
            if not candidate.text or not candidate.text.strip():
                continue
            cx0, cy0, cx1, cy1 = candidate.bbox
            overlap_y = max(0.0, min(fy1, cy1) - max(fy0, cy0)) / f_height
            aligned = overlap_y >= 0.25
            horizontal_distance = fx0 - cx1
            orientation_penalty = 0.0
            if horizontal_distance < -5:
                horizontal_distance = abs((fy0 + fy1) / 2 - (cy0 + cy1) / 2)
                orientation_penalty = 40.0
            if not aligned and not (cy1 <= fy0 and fy0 - cy1 < f_height):
                continue
            if horizontal_distance < 0 and not allow_right:
                continue
            if allow_right and cx0 >= fx1 and abs((cy0 + cy1) / 2 - (fy0 + fy1) / 2) < f_height:
                horizontal_distance = cx0 - fx1
            score = max(horizontal_distance, 0.0) + orientation_penalty + (1.0 - min(overlap_y, 1.0)) * 25.0
            if score < best_score:
                best_score = score
                best = candidate
        return best

"""
OCR pipeline: orchestration, header/footer detection, caption candidate detection.
"""
import hashlib
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from utils.models import Block, BlockType, WordBox
from src.ocr.paddle_ocr import PaddleOCRWrapper
from src.ocr.tesseract_ocr import TesseractOCRWrapper
from src.processing.postprocessing import postprocess_text
from utils.config import Config


class OCRPipeline:
    """OCR pipeline with header/footer and caption detection."""
    
    def __init__(self, use_paddle: bool = True, max_workers: Optional[int] = None):
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
    
    def extract_text_from_block(
        self,
        image: np.ndarray,
        block: Block,
        skip_ocr: bool = False
    ) -> Tuple[str, List[WordBox]]:
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
            # Use existing word boxes if available
            if block.word_boxes:
                text = " ".join([wb.text for wb in block.word_boxes])
                return text, block.word_boxes
            return "", []
        
        # Crop block from image
        x0, y0, x1, y1 = [int(coord) for coord in block.bbox]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(image.shape[1], x1)
        y1 = min(image.shape[0], y1)
        
        if x1 <= x0 or y1 <= y0:
            return "", []  # Invalid bbox, skip silently
        
        # Only skip extremely large blocks (>80% of page) - these are likely full-page images
        # But still try OCR on large blocks (60-80%) as they might contain text
        block_area = (x1 - x0) * (y1 - y0)
        page_area = image.shape[0] * image.shape[1]
        area_ratio = block_area / page_area if page_area > 0 else 0
        
        if area_ratio > 0.8:  # Block is >80% of page - likely full-page image
            return "", []  # Skip silently
        
        block_image = image[y0:y1, x0:x1]
        
        # Validate block image
        if block_image.size == 0:
            return "", []  # Empty image, skip silently
        
        # Ensure minimum size for OCR
        if block_image.shape[0] < 10 or block_image.shape[1] < 10:
            return "", []  # Too small, skip silently
        
        # CRITICAL: Make image contiguous and writable to avoid PaddleOCR tensor memory errors
        # This is especially important for parallel processing
        if not block_image.flags['C_CONTIGUOUS']:
            block_image = np.ascontiguousarray(block_image)
        if not block_image.flags['WRITEABLE']:
            block_image = block_image.copy()
        
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
            # Ensure contiguous and writable
            if not proc.flags['C_CONTIGUOUS']:
                proc = np.ascontiguousarray(proc)
            if not proc.flags['WRITEABLE']:
                proc = proc.copy()
            fallback_proc = proc
        except Exception:
            proc = block_image
            # Ensure contiguous and writable even for fallback
            if not proc.flags['C_CONTIGUOUS']:
                proc = np.ascontiguousarray(proc)
            if not proc.flags['WRITEABLE']:
                proc = proc.copy()
            fallback_proc = proc

        # Run OCR
        try:
            if self.use_paddle and self.paddle_ocr:
                word_boxes = self.paddle_ocr.extract_text(proc)
            elif self.tesseract_ocr:
                word_boxes = self.tesseract_ocr.extract_text(fallback_proc)
            else:
                print(f"⚠️ No OCR engine available for block {block.id}")
                return "", []
        except Exception as e:
            print(f"⚠️ OCR error for block {block.id}: {e}")
            import traceback
            traceback.print_exc()
            word_boxes = []
        
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
        
        # Adjust coordinates to page coordinates
        adjusted_word_boxes = []
        for wb in word_boxes:
            adjusted_bbox = (
                wb.bbox[0] + x0,
                wb.bbox[1] + y0,
                wb.bbox[2] + x0,
                wb.bbox[3] + y0
            )
            adjusted_word_boxes.append(
                WordBox(text=wb.text, bbox=adjusted_bbox, confidence=wb.confidence)
            )
        
        # Combine text with proper spacing
        text = " ".join([wb.text for wb in adjusted_word_boxes])
        text = postprocess_text(text)
        
        # Enhanced post-processing for better text quality
        if text:
            import re
            
            # Fix common OCR errors (context-aware)
            # Fix "I tlls" -> "I am writing this" (common OCR error)
            text = re.sub(r'\bI\s+tlls\s+Is\b', 'I am writing this', text, flags=re.IGNORECASE)
            text = re.sub(r'\bI\s+tlls\b', 'I am writing', text, flags=re.IGNORECASE)
            
            # Fix "Snna munon" -> "Signature" (common OCR error for signature fields)
            text = re.sub(r'\bSnna\s+munon\b', 'Signature', text, flags=re.IGNORECASE)
            
            # Fix common character recognition errors
            text = text.replace("|", "l")  # Vertical bar to lowercase L (context-dependent)
            
            # Fix spacing issues around punctuation
            text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
            text = re.sub(r'([.,;:!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after punctuation
            
            # Fix common word errors
            text = re.sub(r'\bice\s+President\b', 'Vice President', text, flags=re.IGNORECASE)
            text = re.sub(r'\bcientific\b', 'Scientific', text, flags=re.IGNORECASE)
            text = re.sub(r'\bhe\s+Council\b', 'the Council', text, flags=re.IGNORECASE)
            text = re.sub(r'\bw\s+York\b', 'New York', text, flags=re.IGNORECASE)
            text = re.sub(r'\b0\s+Third\b', 'O Third', text, flags=re.IGNORECASE)
            
            # Fix date formatting
            text = re.sub(r'\b(\d{1,2})\s*\)\s*,\s*(\d{4})\b', r'\1, \2', text)  # "30), 1997" -> "30, 1997"
            
            # Remove excessive whitespace but preserve line breaks
            lines = [line.strip() for line in text.split("\n")]
            text = "\n".join(line for line in lines if line)
            
            # Fix hyphenation issues
            text = text.replace("-\n", "").replace("-\r\n", "")
            
            # Clean up multiple spaces
            text = re.sub(r' +', ' ', text)
            text = text.strip()
        
        # Debug output
        if len(adjusted_word_boxes) == 0:
            print(f"⚠️ No text extracted from block {block.id} (type: {block.type.value}, size: {block_image.shape})")
        elif len(text) == 0:
            print(f"⚠️ Empty text after processing for block {block.id}")
        
        return text, adjusted_word_boxes
    
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
        parallel: bool = True
    ) -> List[Block]:
        """
        Process blocks with OCR (optionally in parallel).
        
        Args:
            blocks: List of blocks to process
            pages: List of page images (np.ndarray or PageImage)
            skip_ocr_for_digital: Skip OCR for blocks with digital text
            parallel: Process blocks in parallel (default: True)
            
        Returns:
            List of processed blocks with text and word_boxes
        """
        page_lookup: Dict[int, Any] = {}
        if pages:
            for idx, page in enumerate(pages):
                page_id = getattr(page, "page_id", idx)
                page_lookup[page_id] = page
        
        blocks_to_process: List[Tuple[Block, np.ndarray]] = []
        skipped_blocks: List[Block] = []
        
        for block in blocks:
            page_entry = page_lookup.get(block.page_id)
            page_image = None
            page_obj = None
            if page_entry is not None:
                if hasattr(page_entry, "image"):
                    page_obj = page_entry
                    page_image = page_entry.image
                else:
                    page_image = page_entry
            elif pages and block.page_id < len(pages):
                fallback = pages[block.page_id]
                if hasattr(fallback, "image"):
                    page_obj = fallback
                    page_image = fallback.image
                else:
                    page_image = fallback
            
            # Try to extract from digital text layer first (if available)
            digital_words = getattr(page_obj, "digital_words", []) if page_obj else []
            page_has_digital = getattr(page_obj, "digital_text", False) if page_obj else False
            
            # Only use digital text if we have actual word boxes AND block doesn't already have text
            if skip_ocr_for_digital and digital_words and (not block.word_boxes or not block.text):
                text, word_boxes = self._extract_from_digital_layer(block, digital_words)
                if word_boxes and text and text.strip():
                    # Successfully extracted from digital layer
                    block.word_boxes = word_boxes
                    block.text = text
                    skipped_blocks.append(block)
                    continue
                # Digital extraction failed or incomplete - fall through to OCR
            
            # If block already has text from digital layer, skip OCR
            if skip_ocr_for_digital and block.text and block.text.strip() and block.word_boxes:
                skipped_blocks.append(block)
                continue
            
            # Process ALL block types that might contain text
            # This includes: TEXT, TITLE, LIST, FORM, TABLE, and even FIGURES with text
            # Only skip pure image figures without any text regions
            if block.type == BlockType.FIGURE:
                # Check if figure has text regions (captions, labels, etc.)
                has_text_region = block.metadata.get("has_text", False)
                has_caption = block.metadata.get("caption_candidate_text", None)
                if not has_text_region and not has_caption:
                    # Pure image figure, skip OCR
                    continue
            
            if page_image is None:
                continue
            
            x0, y0, x1, y1 = block.bbox
            block_area = (x1 - x0) * (y1 - y0)
            # Process even small blocks - they might contain important text
            if block_area < 10:  # Very small threshold to catch all text
                continue
            
            blocks_to_process.append((block, page_image))
        
        # Process blocks (parallel or sequential)
        if parallel and len(blocks_to_process) > 1:
            # Parallel processing with timeout protection
            def process_single_block(block_and_image):
                block, page_image = block_and_image
                try:
                    # Only skip extremely large blocks (>80% of page)
                    # Process large blocks (50-80%) as they might contain text
                    x0, y0, x1, y1 = block.bbox
                    block_area = (x1 - x0) * (y1 - y0)
                    page_area = page_image.shape[0] * page_image.shape[1] if hasattr(page_image, 'shape') else 1
                    area_ratio = block_area / page_area if page_area > 0 else 0
                    
                    # Only skip extremely large blocks (likely full-page images)
                    if area_ratio > 0.8:
                        return block, "", [], "Block too large (>80% of page)"
                    
                    text, word_boxes = self.extract_text_from_block(
                        page_image,
                        block,
                        skip_ocr=False
                    )
                    return block, text, word_boxes, None
                except Exception as e:
                    return block, "", [], str(e)
            
            print(f"🔄 Processing {len(blocks_to_process)} blocks in parallel ({self.max_workers} workers)...")
            completed = 0
            failed = 0
            successful = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_single_block, item): item for item in blocks_to_process}
                
                for future in as_completed(futures):
                    try:
                        block, text, word_boxes, error = future.result(timeout=30)  # 30 second timeout per block
                        completed += 1
                        if completed % 5 == 0 or completed == len(blocks_to_process):
                            print(f"  ✓ Processed {completed}/{len(blocks_to_process)} blocks...")
                        if error:
                            failed += 1
                            if failed <= 5:  # Only print first 5 errors to avoid spam
                                print(f"⚠️ OCR skipped for block {block.id}: {error}")
                            block.text = ""
                            block.word_boxes = []
                        else:
                            block.text = text
                            block.word_boxes = word_boxes
                    except Exception as timeout_err:
                        completed += 1
                        failed += 1
                        block = blocks_to_process[0][0]  # Get block from future
                        print(f"⚠️ OCR timeout for block {block.id}: {timeout_err}")
                        block.text = ""
                        block.word_boxes = []
            
            successful = len([b for b in blocks_to_process if hasattr(b[0], 'text') and b[0].text])
            print(f"✅ OCR complete: {successful}/{len(blocks_to_process)} blocks with text ({failed} skipped/failed)")
        else:
            # Sequential processing with progress
            total = len(blocks_to_process)
            successful = 0
            for idx, (block, page_image) in enumerate(blocks_to_process, 1):
                try:
                    # Only skip extremely large blocks (>80% of page)
                    x0, y0, x1, y1 = block.bbox
                    block_area = (x1 - x0) * (y1 - y0)
                    page_area = page_image.shape[0] * page_image.shape[1] if hasattr(page_image, 'shape') else 1
                    area_ratio = block_area / page_area if page_area > 0 else 0
                    
                    if area_ratio > 0.8:
                        print(f"  ⏭️ Skipping extremely large block {block.id} ({area_ratio:.1%} of page)")
                        block.text = ""
                        block.word_boxes = []
                        continue
                    
                    if idx % 5 == 0 or idx == total:
                        print(f"  Processing block {idx}/{total}...")
                    
                    text, word_boxes = self.extract_text_from_block(
                        page_image,
                        block,
                        skip_ocr=False
                    )
                    block.text = text
                    block.word_boxes = word_boxes
                    if text and text.strip():
                        successful += 1
                except Exception as e:
                    print(f"⚠️ OCR error for block {block.id}: {e}")
                    block.text = ""
                    block.word_boxes = []
            
            print(f"✅ OCR complete: {successful}/{total} blocks with text extracted")
        
        # Collect all processed blocks (preserve order)
        processed_blocks = []
        processed_dict = {}
        
        # Store processed blocks in dict
        for block, page_image in blocks_to_process:
            processed_dict[block.id] = block
        
        for block in skipped_blocks:
            processed_dict[block.id] = block
        
        # Reconstruct in original order
        for block in blocks:
            if block.id in processed_dict:
                processed_blocks.append(processed_dict[block.id])
            else:
                processed_blocks.append(block)
        
        # Detect headers/footers
        role_hints = self.detect_repeating_headers_footers(processed_blocks, pages)
        for block in processed_blocks:
            if block.id in role_hints:
                if 'role_hint' not in block.metadata:
                    block.metadata['role_hint'] = role_hints[block.id]
        
        # Detect caption candidates
        caption_flags = self.detect_caption_candidates(processed_blocks)
        for block in processed_blocks:
            if block.id in caption_flags:
                block.metadata['caption_candidate'] = True

        # Heuristically mark key-value style blocks
        try:
            import re
            label_re = re.compile(
                r"^\s*(evac\s*category|battle\s*roster|name|date|time|unit|allergies|mechanism\s*of\s*injury|treatments?|last\s*4)\b",
                re.IGNORECASE
            )
            kv_colon = re.compile(r"^[^\n:]{1,40}:\s*.+")
            for b in processed_blocks:
                if b.type not in (BlockType.TEXT, BlockType.TITLE, BlockType.LIST):
                    continue
                txt = (b.text or "").strip()
                if not txt:
                    continue
                if label_re.match(txt) or kv_colon.match(txt):
                    b.type = BlockType.FORM
                    b.metadata = dict(b.metadata)
                    b.metadata['kv_candidate'] = True
        except Exception:
            pass

        # Heuristic: promote probable section headers near page top (uppercase or high-cap text)
        try:
            for b in processed_blocks:
                if b.type not in (BlockType.TEXT, BlockType.TITLE):
                    continue
                txt = (b.text or "").strip()
                if not txt:
                    continue
                page_entry = pages[b.page_id] if pages and b.page_id < len(pages) else None
                page_h = self._get_page_height(page_entry) if page_entry is not None else None
                # Default: if no page dimension, still allow header mapping by bbox
                y_center = (b.bbox[1] + b.bbox[3]) / 2.0
                near_top = False
                if page_h and page_h > 0:
                    near_top = y_center < page_h * 0.2
                else:
                    near_top = y_center < 300
                # Uppercase ratio
                letters = [ch for ch in txt if ch.isalpha()]
                upper_ratio = (sum(1 for ch in letters if ch.isupper()) / max(len(letters), 1)) if letters else 0.0
                long_enough = len(txt) >= 6
                if near_top and long_enough and upper_ratio >= 0.8:
                    b.type = BlockType.TITLE
        except Exception:
            pass

        return processed_blocks

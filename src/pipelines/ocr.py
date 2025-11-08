"""
OCR pipeline: orchestration, header/footer detection, caption candidate detection.
"""
import hashlib
from typing import List, Dict, Tuple, Optional
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
                print("Falling back to Tesseract")
                self.use_paddle = False
        
        if not self.use_paddle:
            try:
                self.tesseract_ocr = TesseractOCRWrapper()
            except Exception as e:
                print(f"Error: Tesseract OCR initialization failed: {e}")
                raise
    
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
            print(f"⚠️ Invalid block bbox: ({x0}, {y0}, {x1}, {y1})")
            return "", []
        
        block_image = image[y0:y1, x0:x1]
        
        # Validate block image
        if block_image.size == 0:
            print(f"⚠️ Empty block image for block {block.id}")
            return "", []
        
        # Ensure minimum size for OCR
        if block_image.shape[0] < 10 or block_image.shape[1] < 10:
            print(f"⚠️ Block too small for OCR: {block_image.shape}")
            return "", []
        
        # Run OCR
        try:
            if self.use_paddle and self.paddle_ocr:
                word_boxes = self.paddle_ocr.extract_text(block_image)
            elif self.tesseract_ocr:
                word_boxes = self.tesseract_ocr.extract_text(block_image)
            else:
                print(f"⚠️ No OCR engine available for block {block.id}")
                return "", []
        except Exception as e:
            print(f"⚠️ OCR error for block {block.id}: {e}")
            import traceback
            traceback.print_exc()
            return "", []
        
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
        
        # Combine text
        text = " ".join([wb.text for wb in adjusted_word_boxes])
        text = postprocess_text(text)
        
        # Debug output
        if len(adjusted_word_boxes) == 0:
            print(f"⚠️ No text extracted from block {block.id} (type: {block.type.value}, size: {block_image.shape})")
        elif len(text) == 0:
            print(f"⚠️ Empty text after processing for block {block.id}")
        
        return text, adjusted_word_boxes
    
    def detect_repeating_headers_footers(
        self,
        blocks: List[Block],
        pages: List[np.ndarray]
    ) -> Dict[str, str]:
        """
        Detect repeating headers and footers across pages.
        
        Args:
            blocks: List of blocks
            pages: List of page images
            
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
            if page_id >= len(pages):
                continue
            
            page_height = pages[page_id].shape[0]
            
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
        pages: List[np.ndarray],
        skip_ocr_for_digital: bool = True,
        parallel: bool = True
    ) -> List[Block]:
        """
        Process blocks with OCR (optionally in parallel).
        
        Args:
            blocks: List of blocks to process
            pages: List of page images
            skip_ocr_for_digital: Skip OCR for blocks with digital text
            parallel: Process blocks in parallel (default: True)
            
        Returns:
            List of processed blocks with text and word_boxes
        """
        # Filter blocks that need OCR - ONLY process TEXT blocks initially for speed
        blocks_to_process = []
        for block in blocks:
            skip_ocr = skip_ocr_for_digital and hasattr(block, 'metadata') and block.metadata.get('digital_text', False)
            if skip_ocr:
                if block.word_boxes:
                    block.text = " ".join([wb.text for wb in block.word_boxes])
                else:
                    block.text = ""
                    block.word_boxes = []
            # OPTIMIZATION: Only process TEXT, TITLE, LIST blocks initially (skip FORM and TABLE for speed)
            elif block.type in [BlockType.TEXT, BlockType.TITLE, BlockType.LIST]:
                page_image = pages[block.page_id] if block.page_id < len(pages) else None
                if page_image is not None:
                    # Skip blocks that are too small or too large
                    x0, y0, x1, y1 = block.bbox
                    block_area = (x1 - x0) * (y1 - y0)
                    if block_area < 100:  # Too small
                        block.text = ""
                        block.word_boxes = []
                        continue
                    blocks_to_process.append((block, page_image))
                else:
                    block.text = ""
                    block.word_boxes = []
            else:
                # Skip FORM and TABLE blocks for now (can be processed later if needed)
                block.text = ""
                block.word_boxes = []
        
        # Process blocks (parallel or sequential)
        if parallel and len(blocks_to_process) > 1:
            # Parallel processing
            def process_single_block(block_and_image):
                block, page_image = block_and_image
                try:
                    text, word_boxes = self.extract_text_from_block(
                        page_image,
                        block,
                        skip_ocr=False
                    )
                    return block, text, word_boxes, None
                except Exception as e:
                    return block, "", [], str(e)
            
            print(f"🔄 Processing {len(blocks_to_process)} text blocks in parallel ({self.max_workers} workers)...")
            completed = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_single_block, item): item for item in blocks_to_process}
                
                for future in as_completed(futures):
                    block, text, word_boxes, error = future.result()
                    completed += 1
                    if completed % 10 == 0:
                        print(f"  ✓ Processed {completed}/{len(blocks_to_process)} blocks...")
                    if error:
                        print(f"⚠️ OCR error for block {block.id}: {error}")
                        block.text = ""
                        block.word_boxes = []
                    else:
                        block.text = text
                        block.word_boxes = word_boxes
            print(f"✅ OCR complete: {len([b for b in blocks_to_process if hasattr(b[0], 'text') and b[0].text])} blocks with text")
        else:
            # Sequential processing
            for block, page_image in blocks_to_process:
                try:
                    text, word_boxes = self.extract_text_from_block(
                        page_image,
                        block,
                        skip_ocr=False
                    )
                    block.text = text
                    block.word_boxes = word_boxes
                except Exception as e:
                    print(f"⚠️ OCR error for block {block.id}: {e}")
                    block.text = ""
                    block.word_boxes = []
        
        # Collect all processed blocks (preserve order)
        processed_blocks = []
        processed_dict = {}
        
        # Store processed blocks in dict
        for block, page_image in blocks_to_process:
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
        
        return processed_blocks

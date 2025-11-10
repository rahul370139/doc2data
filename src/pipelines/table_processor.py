"""
Table processing pipeline: Path A (TATR/heuristics) and Path B (Qwen-VL).
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import cv2

from utils.models import Block, TableBlock, BlockType
from utils.config import Config
from src.vlm.qwen_vl import QwenVLProcessor
from src.ocr.paddle_ocr import PaddleOCRWrapper
from src.ocr.tesseract_ocr import TesseractOCRWrapper


class TableProcessor:
    """Table extraction with Path A (deterministic) and Path B (VLM)."""
    
    def __init__(self, use_paddle: bool = True, enable_vlm: Optional[bool] = None):
        """
        Initialize table processor.
        
        Args:
            use_paddle: Use PaddleOCR (True) or Tesseract (False)
        """
        vlm_enabled = Config.ENABLE_VLM if enable_vlm is None else enable_vlm
        self.qwen_vl = QwenVLProcessor(enabled=vlm_enabled)
        self.use_paddle = use_paddle
        self.paddle_ocr = None
        self.tesseract_ocr = None
        
        if use_paddle:
            try:
                self.paddle_ocr = PaddleOCRWrapper()
            except:
                self.use_paddle = False
        
        if not self.use_paddle:
            try:
                self.tesseract_ocr = TesseractOCRWrapper()
            except:
                pass
    
    def extract_table_structure_heuristics(
        self,
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extract table structure using heuristics (Path A).
        
        Args:
            image: Table image
            
        Returns:
            Dictionary with table structure
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Detect lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return {
                "shape": [0, 0],
                "headers": [],
                "body": [],
                "units": None,
                "confidence": 0.0
            }
        
        # Detect horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # Horizontal
                horizontal_lines.append((min(x1, x2), max(x1, x2), y1))
            elif abs(x2 - x1) < 5:  # Vertical
                vertical_lines.append((x1, min(y1, y2), max(y1, y2)))
        
        # Sort lines
        horizontal_lines = sorted(horizontal_lines, key=lambda l: l[2])
        vertical_lines = sorted(vertical_lines, key=lambda l: l[0])
        
        # Estimate rows and columns
        rows = len(set([l[2] for l in horizontal_lines]))
        cols = len(set([l[0] for l in vertical_lines]))
        
        # Extract text from cells (simplified)
        # This is a basic implementation - can be improved
        if self.use_paddle and self.paddle_ocr:
            ocr_wrapper = self.paddle_ocr
        elif self.tesseract_ocr:
            ocr_wrapper = self.tesseract_ocr
        else:
            return {
                "shape": [rows, cols],
                "headers": [],
                "body": [],
                "units": None,
                "confidence": 0.3
            }
        
        # Extract all text
        word_boxes = ocr_wrapper.extract_text(image)
        
        # Simple cell assignment (this is very basic - would need proper cell detection)
        headers = []
        body = []
        
        # For now, return basic structure
        return {
            "shape": [rows, cols],
            "headers": headers,
            "body": body,
            "units": None,
            "confidence": 0.5
        }
    
    def extract_table_structure_vlm(
        self,
        image: np.ndarray,
        ocr_tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract table structure using Qwen-VL (Path B).
        
        Args:
            image: Table image
            ocr_tokens: Optional OCR tokens
            
        Returns:
            Dictionary with table structure
        """
        return self.qwen_vl.process_table(image, ocr_tokens)
    
    def process_table_block(
        self,
        block: Block,
        page_image: np.ndarray,
        use_vlm: bool = False
    ) -> TableBlock:
        """
        Process a table block.
        
        Args:
            block: Table block
            page_image: Full page image
            use_vlm: Use VLM (Path B) or heuristics (Path A)
            
        Returns:
            TableBlock with extracted data
        """
        # Crop table from page
        x0, y0, x1, y1 = [int(coord) for coord in block.bbox]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(page_image.shape[1], x1)
        y1 = min(page_image.shape[0], y1)
        
        if x1 <= x0 or y1 <= y0:
            return TableBlock(
                id=block.id,
                type=BlockType.TABLE,
                bbox=block.bbox,
                page_id=block.page_id,
                confidence=0.0
            )
        
        table_image = page_image[y0:y1, x0:x1]
        
        # Extract OCR tokens if needed
        ocr_tokens = None
        if use_vlm:
            if self.use_paddle and self.paddle_ocr:
                word_boxes = self.paddle_ocr.extract_text(table_image)
                ocr_tokens = [wb.text for wb in word_boxes]
            elif self.tesseract_ocr:
                word_boxes = self.tesseract_ocr.extract_text(table_image)
                ocr_tokens = [wb.text for wb in word_boxes]
        
        # Extract structure
        if use_vlm:
            structure = self.extract_table_structure_vlm(table_image, ocr_tokens)
        else:
            structure = self.extract_table_structure_heuristics(table_image)
            # If confidence is low, try VLM
            if structure.get("confidence", 0.0) < 0.5:
                structure = self.extract_table_structure_vlm(table_image, ocr_tokens)
        
        # Create TableBlock
        table_block = TableBlock(
            id=block.id,
            type=BlockType.TABLE,
            bbox=block.bbox,
            page_id=block.page_id,
            confidence=structure.get("confidence", 0.5)
        )
        
        # Set table data
        shape = structure.get("shape", [0, 0])
        if shape and len(shape) == 2:
            table_block.shape = tuple(shape)
        
        table_block.headers = structure.get("headers", [])
        table_block.body = structure.get("body", [])
        table_block.units = structure.get("units")
        
        # Add citations
        table_block.add_citation(block.page_id, block.bbox)
        
        return table_block

"""
Figure processing pipeline: Qwen-VL classification and extraction.
"""
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import cv2

from utils.models import Block, FigureBlock, BlockType
from src.vlm.qwen_vl import QwenVLProcessor
from src.ocr.paddle_ocr import PaddleOCRWrapper


class FigureProcessor:
    """Figure processing with Qwen-VL."""
    
    def __init__(self):
        """Initialize figure processor."""
        # Stubbed: VLM disabled by default for local testing
        self.qwen_vl = QwenVLProcessor(enabled=False)
        self.paddle_ocr = None
        try:
            self.paddle_ocr = PaddleOCRWrapper()
        except:
            pass
    
    def process_figure_block(
        self,
        block: Block,
        page_image: np.ndarray,
        caption: Optional[str] = None
    ) -> FigureBlock:
        """
        Process a figure block.
        
        Args:
            block: Figure block
            page_image: Full page image
            caption: Optional caption text
            
        Returns:
            FigureBlock with classification and data
        """
        # Crop figure from page
        x0, y0, x1, y1 = [int(coord) for coord in block.bbox]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(page_image.shape[1], x1)
        y1 = min(page_image.shape[0], y1)
        
        if x1 <= x0 or y1 <= y0:
            return FigureBlock(
                id=block.id,
                type=BlockType.FIGURE,
                bbox=block.bbox,
                page_id=block.page_id,
                confidence=0.0
            )
        
        figure_image = page_image[y0:y1, x0:x1]
        
        # Step 1: Classify figure type using Qwen-VL
        classification = self.qwen_vl.classify_figure(figure_image, caption)
        figure_type = classification.get("figure_type", "other")
        confidence = classification.get("confidence", 0.5)
        
        # Create FigureBlock
        figure_block = FigureBlock(
            id=block.id,
            type=BlockType.FIGURE,
            bbox=block.bbox,
            page_id=block.page_id,
            figure_type=figure_type,
            caption=caption,
            confidence=confidence
        )
        
        # Process based on figure type
        if figure_type in ["bar", "line", "pie", "scatter"]:
            # Chart - extract data
            chart_data = self.qwen_vl.extract_chart_data(figure_image, figure_type)
            figure_block.series = chart_data.get("series", [])
            figure_block.metadata["axes"] = chart_data.get("axes", {})
            figure_block.metadata["units"] = chart_data.get("units", {})
        else:
            # Non-chart image
            # Store thumbnail path
            thumbnail_path = self._save_thumbnail(figure_image, block.id)
            figure_block.data_or_link = thumbnail_path
            
            # Extract embedded text if any
            if self.paddle_ocr:
                try:
                    word_boxes = self.paddle_ocr.extract_text(figure_image)
                    embedded_text = " ".join([wb.text for wb in word_boxes])
                    if embedded_text.strip():
                        figure_block.alt_text = embedded_text
                except:
                    pass
        
        # Add citations
        figure_block.add_citation(block.page_id, block.bbox)
        
        return figure_block
    
    def _save_thumbnail(
        self,
        image: np.ndarray,
        block_id: str,
        max_size: int = 512
    ) -> str:
        """
        Save figure thumbnail.
        
        Args:
            image: Figure image
            block_id: Block ID
            max_size: Maximum size for thumbnail
            
        Returns:
            Path to saved thumbnail
        """
        # Resize if needed
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Save thumbnail
        thumb_dir = Path("data/thumbnails")
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumb_dir / f"{block_id}.png"
        
        cv2.imwrite(str(thumb_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return str(thumb_path)
    
    def find_caption_for_figure(
        self,
        figure_block: Block,
        text_blocks: List[Block],
        threshold: float = 40.0
    ) -> Optional[str]:
        """
        Find caption for a figure from nearby text blocks.
        
        Args:
            figure_block: Figure block
            text_blocks: List of text blocks
            threshold: Distance threshold in pixels
            
        Returns:
            Caption text or None
        """
        figure_center_y = (figure_block.bbox[1] + figure_block.bbox[3]) / 2
        figure_center_x = (figure_block.bbox[0] + figure_block.bbox[2]) / 2
        
        for text_block in text_blocks:
            if text_block.page_id != figure_block.page_id:
                continue
            
            text_center_y = (text_block.bbox[1] + text_block.bbox[3]) / 2
            text_center_x = (text_block.bbox[0] + text_block.bbox[2]) / 2
            
            # Check if text is below figure (typical caption position)
            if text_center_y > figure_block.bbox[3]:
                distance = np.sqrt(
                    (text_center_x - figure_center_x) ** 2 +
                    (text_center_y - figure_center_y) ** 2
                )
                
                if distance < threshold:
                    return text_block.text
        
        return None


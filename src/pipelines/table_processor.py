"""
Table processing pipeline: Path A (TATR/heuristics) and Path B (Qwen-VL).
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import cv2
from pathlib import Path
from PIL import Image
try:
    import torch
except Exception:
    torch = None

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
        self.vlm_cache: Dict[str, Dict[str, Any]] = {}
        self.tatr_model = None
        self.tatr_processor = None
        self.tatr_device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        # Enable TATR if model path is set (can be HuggingFace model name or local path)
        self.enable_tatr = bool(Config.TATR_MODEL_PATH and torch is not None)
        
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

    def _ensure_tatr_model(self) -> None:
        """Lazy-load TATR model if a checkpoint is provided."""
        if not self.enable_tatr or self.tatr_model is not None:
            return
        try:
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
            model_path = Config.TATR_MODEL_PATH or "microsoft/table-transformer-structure-recognition"
            print(f"  Loading TATR model: {model_path} ...")
            self.tatr_processor = AutoImageProcessor.from_pretrained(model_path)
            self.tatr_model = TableTransformerForObjectDetection.from_pretrained(model_path)
            if torch and self.tatr_device == "cuda":
                self.tatr_model.to(self.tatr_device)
            self.tatr_model.eval()
            print(f"✓ Loaded TATR model from {model_path} on {self.tatr_device}")
        except Exception as exc:
            print(f"⚠ Failed to load TATR model: {exc}")
            import traceback
            traceback.print_exc()
            self.tatr_model = None
            self.enable_tatr = False

    @staticmethod
    def _cluster_positions(values: List[float], tolerance: float) -> List[float]:
        clusters: List[List[float]] = []
        for v in sorted(values):
            if not clusters or abs(v - clusters[-1][-1]) > tolerance:
                clusters.append([v])
            else:
                clusters[-1].append(v)
        return [float(sum(c) / len(c)) for c in clusters]

    def _process_table_tatr(self, table_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run TATR if available to estimate table structure with simple cell grid + OCR fill."""
        if not self.enable_tatr:
            return None
        self._ensure_tatr_model()
        if self.tatr_model is None or self.tatr_processor is None:
            return None
        try:
            pil_image = Image.fromarray(table_image) if len(table_image.shape) == 3 else Image.fromarray(table_image).convert("RGB")
            inputs = self.tatr_processor(images=pil_image, return_tensors="pt")
            if torch and self.tatr_device == "cuda":
                inputs = {k: v.to(self.tatr_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.tatr_model(**inputs)
            target_sizes = torch.tensor([pil_image.size[::-1]])
            if torch and self.tatr_device == "cuda":
                target_sizes = target_sizes.to(self.tatr_device)
            results = self.tatr_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
            scores = results.get("scores", [])
            labels = results.get("labels", [])
            boxes = results.get("boxes", [])
            id2label = getattr(self.tatr_model.config, "id2label", {}) or {}
            cell_centers_x: List[float] = []
            cell_centers_y: List[float] = []
            cell_boxes: List[Tuple[float, float, float, float]] = []
            for lbl, box in zip(labels, boxes):
                name = id2label.get(int(lbl), str(lbl))
                if "cell" in str(name).lower():
                    cx = float((box[0] + box[2]) / 2)
                    cy = float((box[1] + box[3]) / 2)
                    cell_centers_x.append(cx)
                    cell_centers_y.append(cy)
                    cell_boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
            if cell_centers_x and cell_centers_y:
                tol_x = max(6.0, pil_image.width * 0.015)
                tol_y = max(6.0, pil_image.height * 0.015)
                cols = self._cluster_positions(cell_centers_x, tol_x)
                rows = self._cluster_positions(cell_centers_y, tol_y)
                # Boundaries for span detection
                col_edges = [0.0] + [float((cols[i] + cols[i + 1]) / 2.0) for i in range(len(cols) - 1)] + [float(pil_image.width)]
                row_edges = [0.0] + [float((rows[i] + rows[i + 1]) / 2.0) for i in range(len(rows) - 1)] + [float(pil_image.height)]
                confidence = float(scores.mean().item()) if hasattr(scores, "mean") else 0.65
                # Build grid and fill using lightweight OCR
                body: List[List[str]] = [["" for _ in cols] for _ in rows]
                spans: List[Dict[str, Any]] = []
                # Map cell boxes to nearest row/col cluster
                for bx0, by0, bx1, by1 in cell_boxes:
                    cx = (bx0 + bx1) / 2.0
                    cy = (by0 + by1) / 2.0
                    col_idx = int(np.argmin([abs(cx - c) for c in cols]))
                    row_idx = int(np.argmin([abs(cy - r) for r in rows]))
                    # Span detection: find all row/col bands overlapped by box
                    col_span_indices = [i for i in range(len(cols)) if bx1 > col_edges[i] and bx0 < col_edges[i + 1]]
                    row_span_indices = [j for j in range(len(rows)) if by1 > row_edges[j] and by0 < row_edges[j + 1]]
                    col_start, col_end = min(col_span_indices), max(col_span_indices)
                    row_start, row_end = min(row_span_indices), max(row_span_indices)
                    cell_crop = np.array(pil_image.crop((bx0, by0, bx1, by1)))
                    cell_text = ""
                    try:
                        if self.use_paddle and self.paddle_ocr:
                            wbs = self.paddle_ocr.extract_text(cell_crop)
                            cell_text = " ".join(w.text for w in wbs if w.text)
                        elif self.tesseract_ocr:
                            wbs = self.tesseract_ocr.extract_text(cell_crop)
                            cell_text = " ".join(w.text for w in wbs if w.text)
                    except Exception:
                        cell_text = ""
                    if cell_text:
                        body[row_idx][col_idx] = cell_text
                    spans.append({
                        "row_start": row_start,
                        "row_end": row_end,
                        "col_start": col_start,
                        "col_end": col_end,
                        "text": cell_text,
                        "bbox": [bx0, by0, bx1, by1],
                    })
                # Detect header rows (simple heuristic: top row with text becomes header)
                header_rows = 0
                headers: List[List[str]] = []
                if rows:
                    non_empty_counts = [sum(1 for c in r if c) for r in body]
                    if non_empty_counts:
                        max_count = max(non_empty_counts)
                        header_rows = 0
                        for idx, cnt in enumerate(non_empty_counts):
                            if cnt >= max(1, int(0.6 * max_count)):
                                header_rows += 1
                            else:
                                break
                        if header_rows > 0:
                            headers = body[:header_rows]
                            body = body[header_rows:]
                return {
                    "shape": [len(rows), len(cols)],
                    "headers": headers,
                    "header_rows": header_rows,
                    "body": body,
                    "units": None,
                    "confidence": confidence,
                    "cells": cell_boxes,
                    "spans": spans,
                }
        except Exception as exc:
            print(f"⚠ TATR inference failed: {exc}")
        return None
    
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
        structure = None
        tatr_used = False
        if self.enable_tatr:
            structure = self._process_table_tatr(table_image)
            tatr_used = bool(structure)
        if structure is None:
            if use_vlm:
                cache_key = None
                try:
                    cache_key = str(hash(table_image.tobytes()))
                except Exception:
                    cache_key = None
                if cache_key and cache_key in self.vlm_cache:
                    structure = self.vlm_cache[cache_key]
                else:
                    structure = self.extract_table_structure_vlm(table_image, ocr_tokens)
                    if cache_key:
                        self.vlm_cache[cache_key] = structure
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
        
        # Add citations and metadata
        table_block.add_citation(block.page_id, block.bbox)
        table_block.metadata = dict(block.metadata or {})
        table_block.metadata.update({
            "structure_confidence": structure.get("confidence", table_block.confidence),
            "ocr_engine": "paddle" if self.use_paddle else "tesseract",
            "vlm_used": use_vlm,
            "tatr_used": tatr_used,
            "structure_source": "tatr" if tatr_used else ("vlm" if use_vlm else "heuristic"),
            "ocr_tokens": ocr_tokens if ocr_tokens else [],
            "spans": structure.get("spans", []),
        })
        return table_block

    def process_table_blocks_batch(
        self,
        blocks: List[Block],
        page_image: np.ndarray,
        use_vlm: bool = False
    ) -> List[TableBlock]:
        """Batch-friendly wrapper that reuses VLM cache across tables on a page."""
        processed: List[TableBlock] = []
        for blk in blocks:
            processed.append(self.process_table_block(blk, page_image, use_vlm=use_vlm))
        return processed

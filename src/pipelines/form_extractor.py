"""
Form Extractor Pipeline using Hybrid OCR + VLM/SLM with Grounding.

This module implements a State-of-the-Art Intelligent Document Processing (IDP) pipeline.
Instead of relying on brittle hardcoded coordinates, it uses a "Read, Understand, Ground" approach:

1. READ (OCR): Detect all text and precise bounding boxes.
2. UNDERSTAND (SLM/VLM): Use LLM to extract structured data based on Schema.
3. GROUND: Map extracted values back to OCR boxes to provide visual evidence.

This approach generalizes to ANY form (CMS-1500, UB-04, Invoices) by simply changing the Schema.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher

from utils.config import Config
from utils.models import Block
from src.pipelines.business_schema import map_to_business_schema, merge_business_with_ocr


def load_form_schema(form_type: str) -> Dict[str, Any]:
    """Load a form schema by type."""
    # Try multiple paths to be robust
    paths = [
        Path(__file__).parent.parent.parent / "data" / "schemas" / f"{form_type}.json",
        Path("data/schemas") / f"{form_type}.json",
    ]
    
    for path in paths:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    
    return {"fields": [], "form_type": form_type}


class FormExtractor:
    """
    Intelligent Form Extractor.
    
    Combines OCR (for precision/localization) with SLM/VLM (for semantic understanding).
    """
    
    def __init__(self, enabled: Optional[bool] = None):
        """Initialize the form extractor."""
        self.enabled = enabled if enabled is not None else Config.ENABLE_VLM
        self.client = None
        self.model = None
        self.ollama_working = False
        self.model_candidates = [
            Config.OLLAMA_MODEL_SLM,
            "mistral:latest",
            "qwen2.5:7b-instruct",
        ]
        self.available_models: List[str] = []
        
        if self.enabled:
            try:
                from src.vlm.ollama_client import get_ollama_client
                self.client = get_ollama_client()
                self.model = Config.OLLAMA_MODEL_SLM 
                
                # Test connection with a simple request
                print(f"[FormExtractor] Testing Ollama connection to {Config.OLLAMA_HOST}...")
                import requests
                test_url = f"http://{Config.OLLAMA_HOST}/api/tags"
                resp = requests.get(test_url, timeout=5)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    print(f"[FormExtractor] Available models: {model_names}")
                    self.available_models = [m for m in self.model_candidates if any(m in name for name in model_names)]
                    if any(self.model in name for name in model_names):
                        self.ollama_working = True
                        print(f"✓ Form extractor enabled using model: {self.model}")
                    else:
                        print(f"⚠ Model {self.model} not found. Available: {model_names}")
                        # Try to use whatever is available
                        if self.available_models:
                            self.model = self.available_models[0]
                            self.ollama_working = True
                            print(f"✓ Using available fallback model: {self.model}")
                        elif model_names:
                            self.model = model_names[0]
                            self.available_models = [self.model]
                            self.ollama_working = True
                            print(f"✓ Using first discovered model: {self.model}")
                else:
                    print(f"⚠ Ollama returned status {resp.status_code}")
            except Exception as e:
                print(f"⚠ Form extractor: Ollama connection failed: {e}")
                self.ollama_working = False

    def _fuzzy_find_bbox(self, target_text: str, blocks: List[Block]) -> Optional[List[float]]:
        """
        Find the bounding box of text that best matches the target_text.
        
        Args:
            target_text: The text value extracted by LLM.
            blocks: List of OCR blocks.
            
        Returns:
            [x0, y0, x1, y1] or None if no match found.
        """
        if not target_text or not blocks:
            return None
            
        target = target_text.lower().strip()
        best_ratio = 0.0
        best_bbox = None
        
        # 1. Exact match check
        for block in blocks:
            if not block.text: continue
            cleaned = block.text.lower().strip()
            
            if target == cleaned:
                return block.bbox
            if target in cleaned:
                # target is a substring of the block
                return block.bbox

        # 2. Fuzzy match
        for block in blocks:
            if not block.text: continue
            cleaned = block.text.lower().strip()
            
            # similarity ratio
            ratio = SequenceMatcher(None, target, cleaned).ratio()
            if ratio > best_ratio and ratio > 0.7: # Threshold
                best_ratio = ratio
                best_bbox = block.bbox
                
        return best_bbox

    def extract_with_ocr_context(
        self,
        schema: Dict[str, Any],
        blocks: List[Block]
    ) -> Dict[str, Any]:
        """
        Extract fields using OCR text as context for the LLM.
        This is often more accurate for dense forms than raw VLM.
        """
        print(f"[FormExtractor] extract_with_ocr_context called with {len(blocks)} blocks")
        print(f"[FormExtractor] enabled={self.enabled}, ollama_working={self.ollama_working}")
        
        if not self.enabled or not self.ollama_working:
            print("[FormExtractor] Using heuristic fallback (Ollama not available)")
            return self._heuristic_fallback(schema, blocks)

        # 1. Prepare Context (OCR Text)
        # Sort blocks by vertical position to maintain reading order
        sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
        doc_text = "\n".join([b.text for b in sorted_blocks if b.text])
        
        # 2. Prepare Schema Prompt
        fields_prompt = []
        for field in schema.get("fields", []):
            fields_prompt.append(f"- {field.get('id')}: {field.get('label')} ({field.get('description', '')})")
        
        fields_str = "\n".join(fields_prompt)
        
        prompt = f"""You are a specialized data extraction AI. 
Your task is to extract structured data from the provided document text based on a specific schema.

SCHEMA (Fields to extract):
{fields_str}

DOCUMENT TEXT:
{doc_text[:12000]}  # Limit context window if needed

INSTRUCTIONS:
1. Extract the EXACT value for each field from the document text.
2. If a field is not found, use null or empty string.
3. For checkboxes, infer 'checked' or 'unchecked' from context (e.g. "Sex: [x] Male" -> Male).
4. Return ONLY a valid JSON object mapping field_ids to values.

JSON OUTPUT:
"""
        # 3. Call LLM
        try:
            extracted_dict: Dict[str, Any] = {}
            models_to_try = self.available_models if self.available_models else [self.model] if self.model else []
            for mdl in models_to_try:
                try:
                    response = self.client.generate(
                        model=mdl,
                        prompt=prompt,
                        format="json"
                    )
                    response_text = response.get("response", "")
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    if start != -1 and end != -1:
                        extracted_dict = json.loads(response_text[start:end])
                    else:
                        extracted_dict = json.loads(response_text)
                    if extracted_dict:
                        self.model = mdl
                        break
                except Exception as inner_exc:
                    print(f"[FormExtractor] Model {mdl} failed: {inner_exc}")
                    continue
            if not extracted_dict:
                print("[FormExtractor] No LLM output, falling back to heuristics")
                return self._heuristic_fallback(schema, blocks)
            
            # 4. Grounding: Find BBoxes for extracted values
            field_details = []
            grounded_count = 0
            for field in schema.get("fields", []):
                f_id = field.get("id")
                val = extracted_dict.get(f_id, "")
                
                # Grounding step: Find where this value came from
                bbox_raw = self._fuzzy_find_bbox(str(val), blocks) if val else None
                # Convert tuple to list for JSON serialization and consistency
                bbox = list(bbox_raw) if bbox_raw else None
                
                if bbox:
                    grounded_count += 1
                
                field_details.append({
                    "id": f_id,
                    "label": field.get("label"),
                    "value": val,
                    "bbox": bbox,
                    "confidence": 0.9 if bbox else 0.5, # Higher confidence if grounded
                    "detected_by": "slm_grounded" if bbox else "slm_inferred"
                })
            
            print(f"[FormExtractor] Grounded {grounded_count}/{len(field_details)} fields with bboxes")
                
            return {
                "extracted_fields": extracted_dict,
                "field_details": field_details,
                "extraction_method": "slm_grounded",
                "confidence": 0.85
            }

        except Exception as e:
            print(f"SLM Extraction failed: {e}")
            return self._heuristic_fallback(schema, blocks)

    def _heuristic_fallback(self, schema: Dict[str, Any], blocks: List[Block]) -> Dict[str, Any]:
        """
        Fallback using OCR data directly when LLM is unavailable.
        This extracts all detected text with their bounding boxes.
        """
        print(f"[FormExtractor] Heuristic fallback with {len(blocks)} blocks")
        
        extracted = {}
        details = []
        
        # Build a list of all OCR text with bboxes
        all_ocr_results = []
        for block in blocks:
            if block.text and block.text.strip():
                all_ocr_results.append({
                    "text": block.text.strip(),
                    "bbox": list(block.bbox) if block.bbox else None,
                    "confidence": block.metadata.get("ocr_confidence", 0.8) if hasattr(block, "metadata") else 0.8
                })
        
        print(f"[FormExtractor] Found {len(all_ocr_results)} text blocks")
        
        # For CMS-1500 and similar forms, try to match labels to values
        # Sort blocks by position (top-to-bottom, left-to-right)
        sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]) if b.bbox else (0, 0))
        
        # Create field details from schema, trying to match with OCR
        for field in schema.get("fields", []):
            f_id = field.get("id")
            label = field.get("label", "")
            
            # Try to find matching text using label keywords
            matched_value = ""
            matched_bbox = None
            matched_conf = 0.0
            
            # Search for label in OCR text
            label_lower = label.lower()
            for i, block in enumerate(sorted_blocks):
                if not block.text:
                    continue
                text_lower = block.text.lower()
                
                # Check if this block contains the label
                if any(kw in text_lower for kw in label_lower.split()[:2]):
                    # Look for value in nearby blocks (next block or same block after colon)
                    if ":" in block.text:
                        # Value might be after colon
                        parts = block.text.split(":", 1)
                        if len(parts) > 1 and parts[1].strip():
                            matched_value = parts[1].strip()
                            matched_bbox = list(block.bbox) if block.bbox else None
                            matched_conf = block.metadata.get("ocr_confidence", 0.7) if hasattr(block, "metadata") else 0.7
                            break
                    # Check next block for value
                    elif i + 1 < len(sorted_blocks):
                        next_block = sorted_blocks[i + 1]
                        if next_block.text and next_block.bbox:
                            # Check if next block is close (same row or just below)
                            if abs(next_block.bbox[1] - block.bbox[1]) < 50:
                                matched_value = next_block.text.strip()
                                matched_bbox = list(next_block.bbox)
                                matched_conf = next_block.metadata.get("ocr_confidence", 0.7) if hasattr(next_block, "metadata") else 0.7
                                break
            
            extracted[f_id] = matched_value
            details.append({
                "id": f_id,
                "label": label,
                "value": matched_value,
                "bbox": matched_bbox,
                "confidence": matched_conf,
                "detected_by": "heuristic_ocr" if matched_value else "heuristic_fallback"
            })
        
        # Also add all raw OCR blocks for visualization
        fields_found = sum(1 for d in details if d.get("value"))
        avg_conf = np.mean([d.get("confidence", 0) for d in details if d.get("confidence")]) if details else 0
        
        return {
            "extracted_fields": extracted,
            "field_details": details,
            "extraction_method": "heuristic_fallback",
            "confidence": avg_conf,
            "ocr_blocks": all_ocr_results,  # Raw OCR for visualization
            "fields_found": fields_found
        }


def extract_with_full_pipeline(
    pdf_path: str,
        schema: Optional[Dict[str, Any]] = None,
    use_vlm: bool = True,
    map_business: bool = True
) -> Dict[str, Any]:
    """
    Main entry point for the "Smart" pipeline.
    
    Steps:
    1. Ingest: Convert PDF to images
    2. Full-page OCR: Extract ALL text with bounding boxes from full page
    3. Extract: Use LLM (or heuristics) to map text to schema fields
    4. Ground: Link extracted values back to bounding boxes
    
    NOTE: We do FULL-PAGE OCR first to avoid missing text in small crops.
    """
    print(f"\n{'='*60}")
    print(f"[Pipeline] Starting extraction for: {pdf_path}")
    print(f"{'='*60}")
    
    from src.pipelines.ingest import ingest_document
    from src.ocr.paddle_ocr import PaddleOCRWrapper
    from utils.models import Block, BlockType
    
    if schema is None:
        schema = load_form_schema("cms-1500")
    
    # 1. Ingest
    print("\n[Step 1/3] Ingesting document...")
    pages = ingest_document(pdf_path, dpi=300, deskew=True)
    if not pages:
        return {"error": "Ingestion failed", "field_details": [], "ocr_blocks": []}
    page = pages[0]
    img_h, img_w = page.image.shape[:2]
    print(f"  ✓ Loaded page 0, image shape: {page.image.shape}")
    
    # 2. Full-page OCR - Get ALL text with bounding boxes
    print("\n[Step 2/3] Running full-page OCR...")
    ocr = PaddleOCRWrapper()
    word_boxes = ocr.extract_text(page.image)
    print(f"  ✓ OCR detected {len(word_boxes)} text elements")
    
    # Debug: Print first few detections
    for i, wb in enumerate(word_boxes[:10]):
        print(f"    [{i}] '{wb.text[:40]}' @ {wb.bbox} (conf: {wb.confidence:.2f})")
    
    # Convert WordBoxes to Blocks for the extractor
    blocks = []
    for i, wb in enumerate(word_boxes):
        block = Block(
            id=f"ocr-{i}",
            page_id=0,
            bbox=wb.bbox,
            type=BlockType.TEXT,
            text=wb.text,
            word_boxes=[wb],
            metadata={"ocr_confidence": wb.confidence}
        )
        blocks.append(block)
    
    # 3. Extract & Ground (The "Brain")
    print("\n[Step 3/3] Extracting structured data...")
    extractor = FormExtractor(enabled=use_vlm)
    result = extractor.extract_with_ocr_context(schema, blocks)
    
    # Add metadata
    result["total_blocks_detected"] = len(blocks)
    result["blocks_with_text"] = len(blocks)
    
    # Add all OCR blocks for visualization
    result["ocr_blocks"] = [
        {
            "text": wb.text,
            "bbox": list(wb.bbox),
            "confidence": wb.confidence
        }
        for wb in word_boxes
    ]
    
    # Add page dimensions for coordinate scaling
    result["page_height"] = img_h
    result["page_width"] = img_w
    
    fields_found = sum(1 for d in result.get("field_details", []) if d.get("value"))
    print(f"\n{'='*60}")
    print(f"[Pipeline] Extraction complete!")
    print(f"  - Method: {result.get('extraction_method', 'unknown')}")
    print(f"  - Fields found: {fields_found}/{len(schema.get('fields', []))}")
    print(f"  - OCR blocks: {len(result.get('ocr_blocks', []))}")
    print(f"{'='*60}\n")
    
    if map_business and schema:
        business = map_to_business_schema(result, form_type=schema.get("form_type", "cms1500"))
        result = merge_business_with_ocr(result, business)
    
    return result

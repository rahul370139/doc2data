"""
Labeling Agent - semantic labeling using SLM/VLM.

PURPOSE: Assigns field_type (e.g. patient_name, diagnosis) to blocks using
Ollama (Qwen, etc.). Handles text, tables (TATR + SLM), and figures (VLM).
Returns blocks with metadata for schema mapping.

USE CASE: Pipeline calls this after OCR when enable_slm_labeling is on.
Maps raw text to schema field IDs. No manual use; part of MultiAgentPipeline.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.pipelines.core import BaseAgent, BlockType, DetectedBlock, PipelineConfig

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.config import Config


class LabelingAgent(BaseAgent):
    """
    Semantic labeling using SLM/VLM:
    - SLM (Llama 3.2) for text field labeling
    - TATR + SLM for tables
    - VLM (MiniCPM-V) for figures/charts
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__("LabelingAgent")
        self.config = config
        self._ollama_available: Optional[bool] = None
        self._available_models: List[str] = []
    
    async def initialize(self):
        if self._initialized:
            return
        # Check if Ollama has models available (prevent hallucination from empty Ollama)
        try:
            import requests
            resp = requests.get(f"http://{Config.OLLAMA_HOST}/api/tags", timeout=5)
            if resp.ok:
                models = resp.json().get("models", [])
                self._available_models = [m.get("name", "") for m in models]
                self._ollama_available = len(self._available_models) > 0
                if self._ollama_available:
                    self.log(f"Ollama models available: {self._available_models}")
                else:
                    self.log("WARNING: Ollama has no models — SLM labeling will be skipped")
            else:
                self._ollama_available = False
                self.log("WARNING: Ollama not responding — SLM labeling disabled")
        except Exception as e:
            self._ollama_available = False
            self.log(f"WARNING: Ollama check failed: {e} — SLM labeling disabled")
        self._initialized = True
    
    def _call_slm(self, prompt: str) -> str:
        """Call SLM via Ollama. Returns empty string if Ollama unavailable."""
        # Guard: skip if no models available (prevents hallucination)
        if self._ollama_available is False:
            return ""
        try:
            import requests
            response = requests.post(
                f"http://{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": self.config.slm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 500}
                },
                timeout=60
            )
            if response.ok:
                return response.json().get("response", "")
        except Exception as e:
            self.log(f"SLM call failed: {e}")
        return ""
    
    def _call_vlm(self, prompt: str, image: np.ndarray) -> str:
        """Call VLM via Ollama with image."""
        try:
            import requests
            import base64
            
            # Encode image
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            response = requests.post(
                f"http://{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": self.config.vlm_model,
                    "prompt": prompt,
                    "images": [img_base64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 500}
                },
                timeout=60
            )
            if response.ok:
                return response.json().get("response", "")
        except Exception as e:
            self.log(f"VLM call failed: {e}")
        return ""
    
    def clean_cms1500_field_with_slm(
        self, 
        raw_ocr_text: str, 
        field_name: str, 
        field_type: str = "text"
    ) -> str:
        """
        Use SLM to intelligently clean OCR output for a CMS-1500 field.
        
        This helps extract just the handwritten value from OCR that may include
        garbled template labels like "(Last Name, First Name, Middle)" -> "Khan Shah Rukh"
        
        Args:
            raw_ocr_text: Raw OCR output (may include template labels)
            field_name: Human-readable field name (e.g., "Patient Name")
            field_type: Field type hint (text, date, phone, etc.)
            
        Returns:
            Cleaned value or original if SLM unavailable/fails
        """
        if not raw_ocr_text or not raw_ocr_text.strip():
            return ""
            
        # Skip SLM for very clean-looking values (no noise)
        clean_chars = sum(1 for c in raw_ocr_text if c.isalnum() or c.isspace())
        if clean_chars / max(len(raw_ocr_text), 1) > 0.95:
            return raw_ocr_text.strip()
        
        # Guard: skip if no models available
        if self._ollama_available is False:
            return raw_ocr_text
        
        # Build a focused prompt
        type_hints = {
            "text": "a name, word, or phrase",
            "date": "a date in MM/DD/YYYY or similar format",
            "phone": "a phone number",
            "address": "a street address",
            "npi": "a 10-digit NPI number",
            "money": "a dollar amount",
            "icd10": "an ICD-10 diagnosis code",
        }
        type_hint = type_hints.get(field_type, "text")
        
        prompt = f"""Extract ONLY the handwritten value from this OCR text.
The field is "{field_name}" which should contain {type_hint}.
Remove any pre-printed form labels, OCR artifacts, or template text.

OCR Text: {raw_ocr_text}

Rules:
- Output ONLY the actual handwritten/filled-in value
- Remove template labels like "(Last Name, First Name)" or "MM DD YY"
- Remove garbled OCR text that doesn't look like real data
- If the OCR is too garbled to extract a value, output "UNCLEAR"
- Keep the value brief and clean

Extracted Value:"""

        try:
            result = self._call_slm(prompt)
            if result:
                # Clean up SLM output
                result = result.strip()
                # Remove common SLM response prefixes
                for prefix in ["Extracted Value:", "Value:", "The value is", "Answer:"]:
                    if result.lower().startswith(prefix.lower()):
                        result = result[len(prefix):].strip()
                # If SLM says unclear, return empty
                if result.upper() in ["UNCLEAR", "N/A", "NONE", "EMPTY"]:
                    return ""
                # Don't return if SLM added more noise
                if len(result) > len(raw_ocr_text) * 2:
                    return raw_ocr_text
                return result
        except Exception as e:
            self.log(f"SLM field cleaning failed: {e}")
        
        return raw_ocr_text
    
    async def label_text_block(self, block: DetectedBlock, context: str = "") -> DetectedBlock:
        """Label a text block and clean its value using SLM."""
        if not self.config.enable_slm_labeling or not block.text:
            return block

        # Guardrails: never let the SLM rewrite already-structured fields.
        # For CMS-1500, schema/zone-matched values are the source of truth. An LLM can:
        # - hallucinate text that isn't on the page
        # - drop leading characters (e.g., "Baltimore" -> "altimore")
        # - rename semantic labels causing key collisions/overwrites
        meta = block.metadata or {}
        src = str(meta.get("source") or "").strip().lower()
        form_type = str(meta.get("form_type") or "").strip().lower()
        existing_sem = str(meta.get("semantic_label") or "").strip()
        try:
            is_schema_like = bool(re.match(r"^\d+[a-z]?\_", str(block.id or ""))) or bool(re.match(r"^\d+[a-z]?\_", existing_sem))
        except Exception:
            is_schema_like = False
        if src in {"ocr_zone_matching", "schema_zones", "cms1500_production"}:
            return block
        if form_type in {"cms-1500", "cms1500"} and is_schema_like:
            return block
        
        # Enhanced prompt for semantic tagging and value extraction
        prompt = f"""Analyze this text block from a medical form.
Block Text: "{block.text}"
Context: {context}

1. Classify the Semantic Role: [Title, Section Header, Footer, Page Number, Key-Value Pair, List Item, Signature, Comment, Other]
2. Identify the Field Name (if Key-Value Pair).
3. Extract ONLY the Clean Value (remove printed labels, instructions).

Respond in JSON format: 
{{
  "role": "...", 
  "field_name": "...", 
  "clean_value": "..."
}}
"""
        
        response = self._call_slm(prompt)
        try:
            import json
            from difflib import SequenceMatcher
            clean_resp = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_resp)
            
            # Store fine-grained semantic role
            role = (data.get("role") or "text").lower()
            field_name = str(data.get("field_name", "Unknown") or "Unknown").strip()
            if block.metadata is None:
                block.metadata = {}
            block.metadata["semantic_role"] = role
            block.metadata["semantic_field_name"] = field_name
            # Keep extracted_fields keys stable/unique by default.
            # If we do assign a semantic_label for readability, suffix with block.id to avoid collisions.
            if field_name and field_name.lower() not in {"unknown", "text", "form_field"}:
                block.metadata["semantic_label"] = f"{field_name}::{block.id}"

            # Map semantic role to block type
            if "title" in role:
                block.block_type = BlockType.TITLE
            elif "header" in role:
                block.block_type = BlockType.HEADER
            elif "footer" in role:
                block.block_type = BlockType.FOOTER
            elif "page" in role:
                block.block_type = BlockType.PAGE_NUM
            elif "signature" in role:
                block.block_type = BlockType.SIGNATURE
            elif "list" in role:
                block.block_type = BlockType.LIST
            else:
                # keep as text/form_field; semantic label captured in metadata
                pass

            cleaned = data.get("clean_value")
            if cleaned and str(cleaned).strip().lower() not in ["null", "none", ""]:
                cleaned_str = str(cleaned).strip()
                orig_str = str(block.text or "").strip()
                # Only accept "clean_value" if it is grounded in the original text.
                # This prevents hallucinations from leaking into the final output.
                o = re.sub(r"\s+", " ", orig_str.lower())
                c = re.sub(r"\s+", " ", cleaned_str.lower())
                grounded = False
                try:
                    grounded = (c in o) or (SequenceMatcher(None, c, o).ratio() >= 0.72)
                except Exception:
                    grounded = False
                if grounded:
                    block.metadata["original_text"] = block.text
                    block.text = cleaned_str
                else:
                    block.metadata["slm_clean_value_rejected"] = cleaned_str
        except Exception:
            pass
            
        return block
    
    async def process_table(self, image: np.ndarray, block: DetectedBlock) -> Dict[str, Any]:
        """Process table block using VLM for structured extraction.
        
        For CMS-1500 service lines (Box 24), extract:
        - Date of service (FROM/TO)
        - Place of service
        - EMG
        - CPT/HCPCS codes
        - Diagnosis pointer
        - Charges
        - Days/Units
        - NPI
        """
        h, w = image.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in block.bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        crop = image[y0:y1, x0:x1]
        
        table_data = {
            "type": "table",
            "bbox": block.bbox,
            "rows": [],
            "raw_text": block.text
        }
        
        # Use VLM for table extraction (more accurate for handwritten forms)
        use_vlm = getattr(self.config, 'enable_vlm_tables', True) or self.config.enable_vlm_figures
        if use_vlm and crop.size > 0:
            prompt = """Extract service line data from this CMS-1500 form table (Box 24).
For each row, extract:
- date_from: MM/DD/YY format
- date_to: MM/DD/YY format  
- place_of_service: 2-digit code
- cpt_code: 5-digit procedure code
- modifier: optional modifier codes
- diagnosis_pointer: letter A-L
- charges: dollar amount
- days_units: number of units
- npi: 10-digit provider number

Return as JSON array of objects, one per service line. Only include rows with actual data."""
            
            try:
                response = self._call_vlm(prompt, crop)
                if response:
                    # Try to parse JSON from response
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', response)
                    if json_match:
                        rows = json.loads(json_match.group())
                        table_data["rows"] = rows
                        table_data["extraction_method"] = "vlm"
                    else:
                        table_data["vlm_response"] = response
                        table_data["extraction_method"] = "vlm_text"
            except Exception as e:
                table_data["vlm_error"] = str(e)
        
        # Fallback: use OCR text + SLM parsing
        if not table_data["rows"] and self.config.enable_slm_labeling and block.text:
            prompt = f"""Parse CMS-1500 service line data from this OCR text:
"{block.text}"

Extract each service line with: date_from, date_to, place_of_service, cpt_code, charges, days_units.
Return as JSON array."""
            
            try:
                response = self._call_slm(prompt)
                if response:
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', response)
                    if json_match:
                        table_data["rows"] = json.loads(json_match.group())
                        table_data["extraction_method"] = "slm"
            except:
                pass
        
        return table_data
    
    async def process_figure(self, image: np.ndarray, block: DetectedBlock) -> Dict[str, Any]:
        """Process figure/chart using VLM."""
        figure_data = {
            "type": "figure",
            "bbox": block.bbox,
            "description": ""
        }
        
        if self.config.enable_vlm_figures:
            h, w = image.shape[:2]
            x0, y0, x1, y1 = block.bbox
            crop = image[int(y0):int(y1), int(x0):int(x1)]
            
            prompt = "Describe this image/chart from a medical document. What does it show?"
            description = self._call_vlm(prompt, crop)
            figure_data["description"] = description
        
        return figure_data
    
    async def process(self, image: np.ndarray, blocks: List[DetectedBlock]) -> List[DetectedBlock]:
        """Process all blocks with appropriate labeling."""
        await self.initialize()
        
        for block in blocks:
            if block.block_type == BlockType.TABLE:
                table_data = await self.process_table(image, block)
                block.metadata["table_data"] = table_data
            elif block.block_type == BlockType.FIGURE:
                figure_data = await self.process_figure(image, block)
                block.metadata["figure_data"] = figure_data
            else:
                block = await self.label_text_block(block)
        
        return blocks

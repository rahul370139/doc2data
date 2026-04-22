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
    
    @property
    def VLM_TABLE_MODEL(self):
        return Config.VLM_MODEL_TABLE

    @property
    def VLM_TABLE_FALLBACK(self):
        return Config.VLM_MODEL_TABLE_FALLBACK

    def _call_vlm(self, prompt: str, image: np.ndarray, max_tokens: int = 500,
                   timeout: int = 180, model: str = "", temperature: float = 0.1) -> str:
        """Call VLM via Ollama with image. ``model`` overrides config default."""
        if not self._ollama_available:
            self.log("VLM skipped: Ollama has no models")
            return ""
        try:
            import requests
            import base64

            use_model = model or self.config.vlm_model

            if image.ndim == 3 and image.shape[2] == 3:
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr = image
            _, buffer = cv2.imencode('.jpg', bgr)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            self.log(f"VLM call: model={use_model}, max_tokens={max_tokens}, timeout={timeout}, temp={temperature}")
            response = requests.post(
                f"http://{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": use_model,
                    "prompt": prompt,
                    "images": [img_base64],
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}
                },
                timeout=timeout,
            )
            if response.ok:
                return response.json().get("response", "")
        except Exception as e:
            self.log(f"VLM call failed ({model or 'default'}): {e}")
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
        """Process table block: VLM for Box 24 service lines extraction.

        Box 24 goes DIRECTLY to VLM (minicpm-v / llava) with a structured
        extraction prompt.  Florence-2 only supports fixed task prompts like
        ``<OCR>`` — it cannot do structured table extraction.  Florence-2 is
        used elsewhere for text-field OCR rescue only.
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
            "raw_text": block.text,
        }

        if crop.size == 0:
            table_data["extraction_method"] = "skipped_empty_crop"
            return table_data

        try:
            from src.processing.preprocessing import remove_red_template_text
            clean_crop = remove_red_template_text(crop)
        except Exception:
            clean_crop = crop

        # Upscale small table crops for better VLM accuracy
        ch, cw = clean_crop.shape[:2]
        if ch < 400 or cw < 800:
            scale = max(2, min(3, 800 // max(1, cw)))
            clean_crop = cv2.resize(clean_crop, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)
            self.log(f"Table crop upscaled {scale}x to {clean_crop.shape[1]}x{clean_crop.shape[0]}")

        # ── Box 24 → VLM directly (minicpm-v understands custom prompts) ──
        is_box24 = "service_lines" in (block.id or "")

        if is_box24:
            self.log("Table Box 24 → VLM direct (structured extraction)")
            # Prompt design notes:
            # - We ask for pipe-separated output (not JSON) because smaller
            #   VLMs like minicpm-v often break JSON syntax on long tables.
            # - We drop the leading "ROW" column from the required format —
            #   the model used to confuse it with literal text headers and
            #   emit "ROW | date | ..." which then had to be filtered out.
            # - The example line is explicit about what "empty" looks like
            #   (two adjacent pipes, no space) so the parser doesn't
            #   mis-align columns when the modifier is blank.
            prompt = (
                "You are reading CMS-1500 Box 24 — the service lines table. "
                "It has 6 rows. Some rows have handwritten data, the rest are blank.\n\n"
                "For EACH row that has ANY handwritten data, output ONE line "
                "in this format (pipe-separated, 8 columns, empty values "
                "kept as empty strings):\n"
                "date_from | date_to | place | cpt | modifier | charges | units | npi\n\n"
                "Column rules:\n"
                "- date_from, date_to: MM/DD/YY or MM/DD/YYYY\n"
                "- place: 2-digit place-of-service code (e.g. 11, 21, 12)\n"
                "- cpt: 5-character CPT/HCPCS code (e.g. 99213, V5259, H2557)\n"
                "- modifier: 2-char modifier if present, else empty\n"
                "- charges: dollar amount as plain number (e.g. 83.40)\n"
                "- units: integer day/unit count\n"
                "- npi: 10-digit NPI on the far right, else empty\n\n"
                "Example (row 1 has data, modifier is empty):\n"
                "09/20/24 | 09/20/24 | 11 | 99213 |  | 150.00 | 1 | 1234567893\n\n"
                "Output ONLY pipe-separated data lines. No headers, no "
                "commentary, no markdown, no JSON."
            )

            def _count_pipe_rows(resp: str) -> int:
                if not resp:
                    return 0
                return sum(
                    1 for ln in resp.strip().split("\n")
                    if "|" in ln and any(c.isdigit() for c in ln)
                )

            resp_primary = ""
            resp_fallback = ""
            try:
                resp_primary = self._call_vlm(
                    prompt, clean_crop, max_tokens=2000, timeout=300,
                    model=self.VLM_TABLE_MODEL, temperature=0.0,
                ) or ""
                primary_lines = _count_pipe_rows(resp_primary)
                self.log(
                    f"Primary VLM ({self.VLM_TABLE_MODEL}): {primary_lines} "
                    f"pipe-rows, {len(resp_primary)} chars"
                )
            except Exception as e:
                table_data["vlm_primary_error"] = str(e)
                self.log(f"Primary VLM table call failed: {e}")

            try:
                resp_fallback = self._call_vlm(
                    prompt, clean_crop, max_tokens=2000, timeout=300,
                    model=self.VLM_TABLE_FALLBACK, temperature=0.0,
                ) or ""
                fallback_lines = _count_pipe_rows(resp_fallback)
                self.log(
                    f"Fallback VLM ({self.VLM_TABLE_FALLBACK}): {fallback_lines} "
                    f"pipe-rows, {len(resp_fallback)} chars"
                )
            except Exception as e:
                table_data["vlm_fallback_error"] = str(e)
                self.log(f"Fallback VLM table call failed: {e}")

            primary_lines = _count_pipe_rows(resp_primary)
            fallback_lines = _count_pipe_rows(resp_fallback)
            response = resp_fallback if fallback_lines > primary_lines else resp_primary

            # Always preserve the raw VLM output so the debug panel (and
            # us, next iteration) can see exactly what the model said
            # when zero rows get parsed.
            table_data["vlm_raw_primary"] = resp_primary[:800]
            table_data["vlm_raw_fallback"] = resp_fallback[:800]
            table_data["vlm_model_used"] = (
                self.VLM_TABLE_FALLBACK
                if fallback_lines > primary_lines
                else self.VLM_TABLE_MODEL
            )

            if response:
                # First try JSON — some VLMs ignore the pipe-format
                # instruction and return JSON anyway; still valid input.
                json_match = re.search(r'\[[\s\S]*?\]', response)
                if json_match:
                    try:
                        rows = json.loads(json_match.group())
                        if rows and isinstance(rows, list):
                            table_data["rows"] = rows
                            table_data["extraction_method"] = "vlm_direct"
                    except json.JSONDecodeError:
                        pass

                # Pipe-line parser — tolerant of the model dropping the
                # NPI column, adding a leading row number, or leaving
                # trailing whitespace/unicode bars.
                if not table_data["rows"]:
                    parsed_rows = []
                    for line in response.strip().split("\n"):
                        line = line.strip().rstrip("|").lstrip("|").strip()
                        if not line:
                            continue
                        low = line.lower()
                        # Skip ANY line that looks like a header/example
                        # from our own prompt bleeding into the output.
                        if (
                            low.startswith("#")
                            or low.startswith("example")
                            or (
                                "date_from" in low
                                and "date_to" in low
                            )
                            or low.startswith("row ")
                            or low.startswith("---")
                        ):
                            continue
                        # Require at least one pipe and one digit.  This
                        # keeps commentary lines out without dropping
                        # rows where modifier/NPI are missing.
                        if "|" not in line or not any(c.isdigit() for c in line):
                            continue
                        parts = [p.strip() for p in line.split("|")]
                        # Drop leading row-number if the model insisted
                        # on including it (e.g. "1 | 09/20/24 | …").
                        offset = 1 if parts and re.fullmatch(r"\d{1,2}", parts[0] or "") else 0
                        padded = parts + [""] * 8
                        row = {
                            "date_from": padded[offset] if len(padded) > offset else "",
                            "date_to": padded[offset + 1] if len(padded) > offset + 1 else "",
                            "place_of_service": padded[offset + 2] if len(padded) > offset + 2 else "",
                            "cpt_code": padded[offset + 3] if len(padded) > offset + 3 else "",
                            "modifier": padded[offset + 4] if len(padded) > offset + 4 else "",
                            "charges": padded[offset + 5] if len(padded) > offset + 5 else "",
                            "units": padded[offset + 6] if len(padded) > offset + 6 else "",
                            "npi": padded[offset + 7] if len(padded) > offset + 7 else "",
                        }
                        # Keep rows that carry SOMETHING substantive.
                        # CPT used to be required, which silently dropped
                        # rows where only dates + charges were read.
                        if any(row[k] for k in ("cpt_code", "charges", "date_from", "npi")):
                            parsed_rows.append(row)
                    if parsed_rows:
                        table_data["rows"] = parsed_rows
                        table_data["extraction_method"] = "vlm_direct_parsed"

                if table_data["rows"]:
                    # Dedup on (cpt, charges) — if both are empty we keep
                    # the row (the dates/place/NPI still carry info).
                    seen = set()
                    unique_rows = []
                    for r in table_data["rows"]:
                        cpt = re.sub(r"\s+", "", str(r.get("cpt_code", ""))).upper()
                        chg = re.sub(r"[^\d.]", "", str(r.get("charges", "")))
                        key = (cpt, chg)
                        if cpt == "" and chg == "":
                            unique_rows.append(r)
                            continue
                        if key not in seen:
                            seen.add(key)
                            unique_rows.append(r)
                    if len(unique_rows) < len(table_data["rows"]):
                        self.log(
                            f"Deduped {len(table_data['rows'])} → "
                            f"{len(unique_rows)} rows"
                        )
                    table_data["rows"] = unique_rows
                    table_data["total_rows"] = len(unique_rows)
                    parts = []
                    for r in unique_rows:
                        p = []
                        if r.get("date_from"):
                            p.append(str(r["date_from"]))
                        if r.get("date_to"):
                            p.append(str(r["date_to"]))
                        if r.get("place_of_service"):
                            p.append(str(r["place_of_service"]))
                        if r.get("cpt_code"):
                            p.append(str(r["cpt_code"]))
                        if r.get("modifier"):
                            p.append(str(r["modifier"]))
                        if r.get("charges"):
                            p.append(f"${r.get('charges', '')}")
                        if r.get("units"):
                            p.append(f"{r.get('units', '')} units")
                        if r.get("npi"):
                            p.append(str(r["npi"]))
                        if p:
                            parts.append(" | ".join(p))
                    table_data["summary"] = "\n".join(parts)
                    self.log(
                        f"VLM extracted {len(unique_rows)} service line rows"
                    )
        
        # ── Fallback for non-Box24 tables or when VLM fails: SLM parsing ────
        if not table_data["rows"] and self.config.enable_slm_labeling and block.text:
            prompt = f"""Extract structured table data from this OCR text of a medical claim form.
The text represents service line entries from a table. Parse every distinct row.

OCR text:
\"\"\"
{block.text}
\"\"\"

For each row, extract these fields (leave empty if not found):
- date_from: service start date (MM/DD/YY)
- date_to: service end date (MM/DD/YY)
- place_of_service: 2-digit code
- cpt_code: CPT/HCPCS procedure code
- modifier: modifier code if any
- charges: dollar amount
- units: days or units
- npi: rendering provider NPI (10 digits)

Return ONLY a JSON array of objects. No explanation.
Example: [{{"date_from":"09/20/12","date_to":"09/20/12","place_of_service":"12","cpt_code":"99213","modifier":"","charges":"150.00","units":"1","npi":"1234567890"}}]"""

            try:
                response = self._call_slm(prompt)
                if response:
                    json_match = re.search(r'\[[\s\S]*?\]', response)
                    if json_match:
                        rows = json.loads(json_match.group())
                        if isinstance(rows, list):
                            table_data["rows"] = rows
                            table_data["extraction_method"] = "slm_fallback"
                            table_data["total_rows"] = len(rows)
            except Exception:
                pass
        
        if not table_data.get("total_rows") and table_data.get("rows"):
            table_data["total_rows"] = len(table_data["rows"])
        
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

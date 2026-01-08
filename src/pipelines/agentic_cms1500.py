"""
Agentic CMS-1500 Pipeline
=========================

PURPOSE:
    Primary pipeline for CMS-1500 form extraction. Uses a multi-agent architecture
    to maximize accuracy on this specific form type.

INPUT:
    - PDF or image file path containing a CMS-1500 form
    - Configuration options for ICR, LLM, template alignment

OUTPUT:
    - Dict with:
        - "extracted_fields": {field_id: value}
        - "field_details": [{id, label, value, bbox, confidence, detected_by}]
        - "business_fields": {patient_name, dob, insurance_id, ...}
        - "ocr_blocks": [{text, bbox, confidence}]
        - Metadata (page dimensions, processing info)

ARCHITECTURE (following your diagram):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Document Ingestion → 300 DPI RGB Array                          │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ RegistrationAgent: ORB/SIFT feature matching → Homography warp  │
    │   - Aligns scanned form to reference CMS-1500 template          │
    │   - Handles rotation, skew, scale variations                    │
    │   - Falls back to ML detection if alignment fails               │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ ZoneAgent: Projects schema zones → aligned coordinates          │
    │   - Uses CMS-1500 field definitions with bbox_norm              │
    │   - Refines zones using detected lines/boxes                    │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ OCRAgent (Tiered): Full-page OCR → Zone matching                │
    │   - PaddleOCR for printed text                                  │
    │   - TrOCR for handwriting/signatures                            │
    │   - Checkbox density for check marks                            │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ LLMExtractionAgent: Semantic extraction with SLM                │
    │   - Uses full OCR text context                                  │
    │   - Fallback chain: Llama 3.2 → Mistral → Qwen                  │
    │   - Grounds extracted values back to OCR bboxes                 │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ BusinessMapper: Schema IDs → Business fields + validation       │
    │   - Maps field_ids to semantic names (patient_name, dob, etc.)  │
    │   - Applies validators (NPI, date, phone, ICD-10)               │
    └─────────────────────────────────────────────────────────────────┘

USAGE:
    from src.pipelines.agentic_cms1500 import run_cms1500_agentic
    
    result = run_cms1500_agentic(
        "path/to/cms1500.pdf",
        use_icr=True,      # Enable TrOCR for handwriting
        use_llm=True,      # Enable LLM for semantic extraction
        align_template=True # Enable template alignment
    )
    print(result["business_fields"])

AGENTS:
    1) RegistrationAgent - Template alignment via homography
    2) ZoneAgent - Schema zone projection and refinement
    3) OCRAgent - Tiered OCR (Paddle → TrOCR → checkbox)
    4) LLMExtractionAgent - SLM-based semantic extraction
    5) BusinessMapper - Field mapping and validation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
import json
import textwrap

from utils.config import Config
from utils.models import Block, BlockType, WordBox
from src.pipelines.ingest import ingest_document
from src.processing.registration import (
    compute_alignment_matrix,
    transform_normalized_bbox,
    load_and_process_reference,
)
from src.ocr.paddle_ocr import PaddleOCRWrapper
from src.ocr.trocr_wrapper import TrOCRWrapper
from src.pipelines.business_schema import map_to_business_schema, merge_business_with_ocr
from src.pipelines.form_extractor import load_form_schema
from src.vlm.ollama_client import get_ollama_client


REF_SIZE = (2550, 3300)  # Width, Height used when schema bboxes are normalized


@dataclass
class ZoneProposal:
    field_id: str
    label: str
    bbox: Tuple[float, float, float, float]
    field_type: str
    bbox_source: str = "schema"
    refined: bool = False


@dataclass
class ZonePrediction:
    field_id: str
    label: str
    value: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    engine: str
    word_boxes: List[WordBox]
    field_type: str
    notes: Optional[str] = None


def _clamp_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(float(x0), float(width)))
    x1 = max(0.0, min(float(x1), float(width)))
    y0 = max(0.0, min(float(y0), float(height)))
    y1 = max(0.0, min(float(y1), float(height)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def _crop(image: np.ndarray, bbox: Tuple[float, float, float, float], pad: int = 8) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    h, w = image.shape[:2]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return image[0:0, 0:0]
    return image[y0:y1, x0:x1]


def _fuzzy_match_bbox(value: str, all_boxes: List[WordBox]) -> Optional[Tuple[float, float, float, float]]:
    """Find bbox for a value by matching against OCR results."""
    if not value or not all_boxes:
        return None
    
    value_lower = value.lower().strip()
    value_words = value_lower.split()
    
    best_match = None
    best_score = 0.0
    
    for wb in all_boxes:
        text_lower = wb.text.lower().strip()
        # Check for substring match
        if value_lower in text_lower or text_lower in value_lower:
            score = len(text_lower) / max(len(value_lower), 1)
            if score > best_score:
                best_score = score
                best_match = wb.bbox
        # Check for word overlap
        elif any(w in text_lower for w in value_words if len(w) > 2):
            score = 0.5
            if score > best_score:
                best_score = score
                best_match = wb.bbox
    
    return best_match


def _estimate_handwriting_score(crop: np.ndarray) -> float:
    """Rough handwriting detector using stroke variance."""
    if crop is None or crop.size == 0:
        return 0.0
    try:
        gray = crop
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (min(220, gray.shape[1]), min(80, gray.shape[0])), interpolation=cv2.INTER_AREA)
        var_lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        edge_density = float(np.mean(gray < 180))
        return float(min(max((var_lap * 0.002) + (edge_density * 0.6), 0.0), 1.2))
    except Exception:
        return 0.0


class RegistrationAgent:
    """Align page to reference template using ORB homography."""

    def __init__(self, template_name: str = "cms-1500"):
        self.template_name = template_name
        self.ref_data = load_and_process_reference(template_name)

    def align(self, page_image: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        if self.ref_data is None:
            return None, REF_SIZE[::-1]
        H = compute_alignment_matrix(page_image, self.template_name)
        ref_shape = self.ref_data.get("shape", (REF_SIZE[1], REF_SIZE[0])) if isinstance(self.ref_data, dict) else (REF_SIZE[1], REF_SIZE[0])
        return H, (int(ref_shape[1]), int(ref_shape[0]))


class ZoneAgent:
    """Project schema zones to the current page and refine with masks."""

    def __init__(self, padding: float = 12.0):
        self.padding = padding

    def propose(self, schema: Dict[str, Any], page: Any, H: Optional[np.ndarray], ref_size: Tuple[int, int]) -> List[ZoneProposal]:
        proposals: List[ZoneProposal] = []
        if not schema:
            return proposals

        page_h, page_w = page.image.shape[:2]
        ref_w, ref_h = ref_size

        for field in schema.get("fields", []):
            norm = field.get("bbox_norm")
            if not norm or len(norm) != 4:
                continue
            bbox = transform_normalized_bbox(norm, H, ref_w, ref_h) if H is not None else (
                norm[0] * page_w,
                norm[1] * page_h,
                norm[2] * page_w,
                norm[3] * page_h,
            )
            bbox = _clamp_bbox(bbox, page_w, page_h)
            bbox = (
                bbox[0] - self.padding,
                bbox[1] - self.padding,
                bbox[2] + self.padding,
                bbox[3] + self.padding,
            )
            refined_bbox = self._refine_with_masks(bbox, page)
            proposals.append(
                ZoneProposal(
                    field_id=field.get("id"),
                    label=field.get("label", field.get("id", "")),
                    bbox=refined_bbox,
                    field_type=field.get("field_type", "text"),
                    bbox_source="schema+align" if H is not None else "schema",
                    refined=refined_bbox != bbox,
                )
            )
        return proposals

    def _refine_with_masks(self, bbox: Tuple[float, float, float, float], page: Any) -> Tuple[float, float, float, float]:
        """Snap bbox to nearby lines/boxes to absorb minor template differences."""
        if page is None or page.box_mask is None:
            return bbox
        try:
            mask = page.box_mask
            x0, y0, x1, y1 = [int(round(v)) for v in _clamp_bbox(bbox, mask.shape[1], mask.shape[0])]
            crop = mask[max(0, y0 - 6):min(mask.shape[0], y1 + 6), max(0, x0 - 6):min(mask.shape[1], x1 + 6)]
            if crop is None or crop.size == 0:
                return bbox
            contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return bbox
            # Pick the largest contour
            largest = max(contours, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(largest)
            refined = (
                x0 + rx - 2,
                y0 + ry - 2,
                x0 + rx + rw + 2,
                y0 + ry + rh + 2,
            )
            return _clamp_bbox(refined, mask.shape[1], mask.shape[0])
        except Exception:
            return bbox


class OCRAgent:
    """Route each zone to the best OCR/ICR reader - using full-page OCR first."""

    def __init__(self, use_icr: bool = True):
        self.paddle = PaddleOCRWrapper()
        self.icr = None
        self._icr_enabled = use_icr
        self._fullpage_cache: Optional[List[WordBox]] = None

    def run_fullpage_ocr(self, page: Any) -> List[WordBox]:
        """Run OCR on full page once, cache results."""
        if self._fullpage_cache is not None:
            return self._fullpage_cache
        
        image = getattr(page, "image", None)
        if image is None:
            self._fullpage_cache = []
            return self._fullpage_cache
        
        try:
            print("[OCRAgent] Running full-page PaddleOCR...")
            self._fullpage_cache = self.paddle.extract_text(image)
            print(f"[OCRAgent] Full-page OCR found {len(self._fullpage_cache)} word boxes")
        except Exception as e:
            print(f"[OCRAgent] Full-page OCR failed: {e}")
            self._fullpage_cache = []
        
        return self._fullpage_cache

    def _boxes_in_zone(self, all_boxes: List[WordBox], bbox: Tuple[float, float, float, float], expand: int = 10) -> List[WordBox]:
        """Find word boxes that fall within or overlap with zone."""
        x0, y0, x1, y1 = bbox
        x0 -= expand
        y0 -= expand
        x1 += expand
        y1 += expand
        matches = []
        for wb in all_boxes:
            bx0, by0, bx1, by1 = wb.bbox
            # Check overlap (not strict containment)
            if bx1 >= x0 and bx0 <= x1 and by1 >= y0 and by0 <= y1:
                matches.append(wb)
        return matches

    def _digital_words_in_zone(self, page: Any, bbox: Tuple[float, float, float, float]) -> List[WordBox]:
        words: List[WordBox] = []
        if not getattr(page, "digital_words", None):
            return words
        x0, y0, x1, y1 = bbox
        for wb in page.digital_words:
            bx0, by0, bx1, by1 = wb.bbox
            if bx0 >= x0 - 2 and bx1 <= x1 + 2 and by0 >= y0 - 2 and by1 <= y1 + 2:
                words.append(wb)
        return words

    def _get_icr(self) -> Optional[TrOCRWrapper]:
        if not self._icr_enabled:
            return None
        if self.icr is None:
            try:
                self.icr = TrOCRWrapper(use_gpu=Config.USE_GPU)
            except Exception:
                self.icr = None
        return self.icr

    def predict(self, zone: ZoneProposal, page: Any, all_boxes: Optional[List[WordBox]] = None) -> ZonePrediction:
        image = getattr(page, "image", None)
        if image is None:
            return ZonePrediction(zone.field_id, zone.label, "", zone.bbox, 0.0, "none", [], zone.field_type, notes="no_image")

        # Prefer digital text when available
        digital_words = self._digital_words_in_zone(page, zone.bbox)
        if digital_words:
            text = " ".join(wb.text for wb in digital_words)
            conf = float(np.mean([wb.confidence for wb in digital_words])) if digital_words else 0.95
            return ZonePrediction(zone.field_id, zone.label, text, zone.bbox, conf, "digital", digital_words, zone.field_type)

        # Checkbox handling
        if zone.field_type == "checkbox":
            crop = _crop(image, zone.bbox)
            return self._predict_checkbox(zone, crop)

        # Use full-page OCR boxes (more reliable than per-zone crop+OCR)
        if all_boxes is None:
            all_boxes = self.run_fullpage_ocr(page)
        
        zone_boxes = self._boxes_in_zone(all_boxes, zone.bbox)
        
        if zone_boxes:
            # Sort boxes left-to-right, top-to-bottom for proper reading order
            zone_boxes_sorted = sorted(zone_boxes, key=lambda wb: (wb.bbox[1], wb.bbox[0]))
            text = " ".join(wb.text for wb in zone_boxes_sorted).strip()
            conf = float(np.mean([wb.confidence for wb in zone_boxes_sorted]))
            return ZonePrediction(zone.field_id, zone.label, text, zone.bbox, conf, "paddle_fullpage", zone_boxes_sorted, zone.field_type)

        # Fallback: Try TrOCR on crop for handwriting
        crop = _crop(image, zone.bbox)
        if crop.size > 0:
            handwriting_score = _estimate_handwriting_score(crop)
            icr_model = self._get_icr()
            if handwriting_score > 0.3 and icr_model:
                icr_text = icr_model.predict(crop)
                if icr_text:
                    return ZonePrediction(zone.field_id, zone.label, icr_text, zone.bbox, 0.65, "trocr", [], zone.field_type, notes=f"handwriting={handwriting_score:.2f}")

        return ZonePrediction(zone.field_id, zone.label, "", zone.bbox, 0.0, "none", [], zone.field_type, notes="no_text_found")

    def _predict_checkbox(self, zone: ZoneProposal, crop: np.ndarray) -> ZonePrediction:
        if crop is None or crop.size == 0:
            return ZonePrediction(zone.field_id, zone.label, "", zone.bbox, 0.0, "checkbox", [], zone.field_type)
        gray = crop
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fill_ratio = float(np.sum(thresh > 0) / max(thresh.size, 1))
        state = "checked" if fill_ratio > 0.15 else "unchecked"
        return ZonePrediction(zone.field_id, zone.label, state, zone.bbox, 0.6, "checkbox_density", [], zone.field_type, notes=f"fill_ratio={fill_ratio:.2f}")


class LLMExtractionAgent:
    """Fill low-confidence/missing fields using fallback chain."""

    def __init__(self):
        self.client = None
        self.available_models: List[str] = []
        try:
            self.client = get_ollama_client()
            candidates = [
                Config.OLLAMA_MODEL_SLM,
                "mistral:latest",
                "qwen2.5:7b-instruct",
            ]
            for model in candidates:
                if self.client.is_model_available(model):
                    self.available_models.append(model)
        except Exception as exc:
            self.client = None
            self.available_models = []

    def extract_all(self, schema: Dict[str, Any], ocr_text: str) -> Dict[str, str]:
        """Extract ALL fields using LLM semantic understanding."""
        if not self.client or not self.available_models:
            return {}
        
        fields = schema.get("fields", [])
        if not fields:
            return {}
        
        field_list = "\n".join([f"- {f.get('id')}: {f.get('label')}" for f in fields])
        
        prompt = textwrap.dedent(
            f"""
            You are an expert CMS-1500 medical form extractor. Extract the PATIENT-FILLED values from this form.
            
            IMPORTANT:
            - Extract ONLY values that appear to be filled in by a patient/provider (handwritten or typed in fields)
            - DO NOT extract form template text like "PATIENT'S NAME", "INSURED'S I.D. NUMBER", etc.
            - If a field appears empty or only has template text, return empty string
            - For checkboxes, return "checked" or "unchecked"
            - For names, extract the actual name written (e.g., "KHAN SHAH RUKH" not "PATIENT'S NAME")
            
            FIELDS TO EXTRACT:
            {field_list}
            
            OCR TEXT FROM FORM:
            {ocr_text[:8000]}
            
            Return JSON with field_id -> extracted_value. Example:
            {{"2_patient_name": "John Smith", "3_patient_dob": "01/15/1985", "1a_insured_id": "ABC123456"}}
            """
        ).strip()
        
        for model in self.available_models:
            try:
                print(f"[LLMAgent] Trying model: {model}")
                resp = self.client.generate(model=model, prompt=prompt, format="json")
                text = resp.get("response", "")
                start = text.find("{")
                end = text.rfind("}") + 1
                parsed = json.loads(text[start:end] if start != -1 else text)
                if isinstance(parsed, dict):
                    print(f"[LLMAgent] Extracted {len(parsed)} fields with {model}")
                    return parsed
            except Exception as e:
                print(f"[LLMAgent] Model {model} failed: {e}")
                continue
        return {}

    def fill(self, schema: Dict[str, Any], field_predictions: List[ZonePrediction]) -> Dict[str, str]:
        if not self.client or not self.available_models:
            return {}
        missing_fields = [f for f in schema.get("fields", []) if not any(p.field_id == f.get("id") and p.value for p in field_predictions)]
        if not missing_fields:
            return {}

        # Build context from available text
        context_lines = []
        for pred in field_predictions:
            if pred.value:
                context_lines.append(f"{pred.label}: {pred.value}")
        context = "\n".join(context_lines)[:12000]

        field_prompt = "\n".join([f"- {fld.get('id')}: {fld.get('label')}" for fld in missing_fields])
        prompt = textwrap.dedent(
            f"""
            You are an expert CMS-1500 form extractor. Fill only the missing fields.
            Use the provided OCR snippets to infer values. Return JSON with field_id -> value.

            MISSING FIELDS:
            {field_prompt}

            OCR SNIPPETS:
            {context}
            """
        ).strip()

        for model in self.available_models:
            try:
                resp = self.client.generate(model=model, prompt=prompt, format="json")
                text = resp.get("response", "")
                start = text.find("{")
                end = text.rfind("}") + 1
                parsed = json.loads(text[start:end] if start != -1 else text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return {}


def run_cms1500_agentic(
    pdf_path: str,
    use_icr: bool = True,
    use_llm: bool = True,
    align_template: bool = True,
) -> Dict[str, Any]:
    """
    Agentic, template-aware CMS-1500 pipeline entrypoint.

    Returns:
        Dict with OCR schema result + business schema mapping.
    """
    schema = load_form_schema("cms-1500")
    pages = ingest_document(pdf_path, dpi=Config.DPI, doc_type_hint="cms1500")
    if not pages:
        return {"error": "ingest_failed"}
    page = pages[0]

    reg_agent = RegistrationAgent("cms-1500") if align_template else None
    H = None
    ref_size = (REF_SIZE[0], REF_SIZE[1])
    if reg_agent:
        H, ref_size = reg_agent.align(page.image)

    zone_agent = ZoneAgent(padding=10.0)
    zones = zone_agent.propose(schema, page, H, ref_size)

    ocr_agent = OCRAgent(use_icr=use_icr)
    
    # Run full-page OCR ONCE first (key fix - don't crop+OCR each zone)
    all_boxes = ocr_agent.run_fullpage_ocr(page)
    print(f"[Agentic] Full-page OCR: {len(all_boxes)} words detected")
    
    # Build full OCR text for LLM context
    ocr_full_text = " ".join(wb.text for wb in all_boxes).strip()
    print(f"[Agentic] OCR text preview: {ocr_full_text[:500]}...")
    
    # Primary extraction via LLM (semantic understanding > zone cropping)
    llm_extracted = {}
    if use_llm and ocr_full_text:
        llm_agent = LLMExtractionAgent()
        llm_extracted = llm_agent.extract_all(schema, ocr_full_text)
        print(f"[Agentic] LLM extracted {len(llm_extracted)} fields")
    
    zone_preds: List[ZonePrediction] = []
    for zone in zones:
        # If LLM found a value, use it; otherwise fall back to zone OCR
        llm_value = llm_extracted.get(zone.field_id, "")
        
        if llm_value:
            # Ground LLM value: find bbox in full OCR
            bbox_match = _fuzzy_match_bbox(str(llm_value), all_boxes)
            zone_preds.append(ZonePrediction(
                zone.field_id,
                zone.label,
                llm_value,
                bbox_match if bbox_match else zone.bbox,
                0.85,
                "llm_grounded",
                [],
                zone.field_type,
                notes="llm_primary"
            ))
        else:
            # Fallback: OCR-based zone prediction
            pred = ocr_agent.predict(zone, page, all_boxes)
            zone_preds.append(pred)

    extracted_fields: Dict[str, Any] = {p.field_id: p.value for p in zone_preds}
    field_details: List[Dict[str, Any]] = []
    all_word_boxes: List[Dict[str, Any]] = []
    for pred in zone_preds:
        field_details.append(
            {
                "id": pred.field_id,
                "label": pred.label,
                "value": pred.value,
                "bbox": [float(v) for v in pred.bbox],
                "confidence": pred.confidence,
                "detected_by": pred.engine,
                "notes": pred.notes,
            }
        )
        for wb in pred.word_boxes:
            all_word_boxes.append({"text": wb.text, "bbox": list(wb.bbox), "confidence": wb.confidence})

    ocr_result = {
        "success": True,
        "form_type": "CMS-1500",
        "extraction_method": "agentic_template",
        "extracted_fields": extracted_fields,
        "field_details": field_details,
        "ocr_blocks": all_word_boxes,
        "page_width": page.width,
        "page_height": page.height,
        "zones_used": len(zones),
        "llm_used": use_llm,
        "total_blocks_detected": len(zone_preds),
        "blocks_with_text": sum(1 for p in zone_preds if p.value),
        "metadata": {
            "alignment_success": bool(H is not None),
            "alignment_matrix_present": bool(H is not None),
            "ref_size": list(ref_size) if ref_size else None,
        },
    }

    business = map_to_business_schema(ocr_result, form_type="cms1500")
    merged = merge_business_with_ocr(ocr_result, business)
    return merged

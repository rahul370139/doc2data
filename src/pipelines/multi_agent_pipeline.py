"""
Multi-Agent Document Processing Pipeline - main orchestrator.

PURPOSE: Orchestrates 3-lane extraction (Lane A: widgets, B: digital text,
C: scanned). Runs form ID, alignment, layout, OCR, labeling, validation.
Assembles JSON with extracted_fields and business_fields.

USE CASE: from src.pipelines.multi_agent_pipeline import MultiAgentPipeline
pipeline = MultiAgentPipeline(config); result = pipeline.process_sync(path)

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOCUMENT INGESTION                                 │
│                     (PDF/Image → 300 DPI RGB Array)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FORM IDENTIFICATION AGENT                               │
│  - OCR header for "CMS-1500", "UB-04", etc.                                 │
│  - Layout fingerprint matching (Sensible-style)                             │
│  - Returns: form_type, confidence, version                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
        ┌───────────────────┐             ┌───────────────────────┐
        │   CMS-1500 PATH   │             │   GENERAL FORM PATH   │
        │  (Template-based) │             │   (ML Detection)      │
        └───────────────────┘             └───────────────────────┘
                    │                                 │
                    ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TEMPLATE ALIGNMENT AGENT                               │
│  (CMS-1500 only)                                                             │
│  - ORB/SIFT feature matching                                                 │
│  - Homography warp to reference template                                     │
│  - Fallback: ML detection if alignment fails                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LAYOUT DETECTION AGENT                                 │
│  CMS-1500:              │  General Forms:                                   │
│  - YOLOv8 (fine-tuned)  │  - LayoutLMv3 / Detectron2 (PubLayNet)            │
│  - Template zones       │  - Donut (end-to-end)                             │
│  Returns: blocks with type (text, table, figure, form fields, checkbox)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
        ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
        │  TEXT BLOCKS  │  │ TABLE BLOCKS  │  │ FIGURE BLOCKS │ 
        └───────────────┘  └───────────────┘  └───────────────┘
                    │                │                │
                    ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OCR AGENT (Tiered)                                   │
│  - PaddleOCR (printed text)                                                  │
│  - TrOCR (handwriting, signatures)                                           │
│  - Checkbox density detector                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                │                │
                    ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SLM/VLM LABELING AGENT                                 │
│  TEXT:    SLM (Llama 3.2) → semantic field labels                           │
│  TABLES:  TATR + SLM → structured row/column extraction                     │
│  FIGURES: VLM (MiniCPM-V) → chart/image understanding                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VALIDATION AGENT                                        │
│  - Field validators (NPI, date, phone, ICD-10, etc.)                         │
│  - Cross-field consistency checks                                            │
│  - LLM QA sanity check                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ASSEMBLY AGENT                                          │
│  - OCR Schema (raw extraction)                                               │
│  - Business Schema (mapped fields)                                           │
│  - Reducto-style JSON export                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    from src.pipelines.multi_agent_pipeline import MultiAgentPipeline
    
    pipeline = MultiAgentPipeline()
    result = await pipeline.process("document.pdf")
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import concurrent.futures

import cv2
import numpy as np

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.config import Config

from src.pipelines.core import (
    AlignmentResult,
    BlockType,
    DetectedBlock,
    FormIdentification,
    FormType,
    PipelineConfig,
)
from src.pipelines.agents import (
    FormIdentificationAgent,
    LabelingAgent,
    LayoutDetectionAgent,
    OCRAgent,
    TemplateAlignmentAgent,
    ValidationAgent,
)


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

class MultiAgentPipeline:
    """
    Main orchestrator for the multi-agent document processing pipeline.
    
    CMS-1500 uses a 3-lane extraction strategy:
      Lane A (best):  AcroForm widgets → values (no OCR needed for fillable PDFs)
      Lane B (good):  PDF text layer → word boxes → zone match (flattened digital)
      Lane C (scan):  align → template subtraction → OCR → zone match → ICR fallback
    """
    
    # ── Widget name → schema field ID mapping ──────────────────────────
    # Built from actual Cigna CMS-1500 AcroForm field names.
    # This is the authoritative mapping for Lane A extraction.
    WIDGET_TO_SCHEMA: Dict[str, str] = {
        # Top header / insurance
        "insurance_name":           "header_top_right_notes",  # insurance company name
        "insurance_id":             "1a_insured_id",
        # Patient
        "pt_name":                  "2_patient_name",
        "pt_street":                "5_patient_address",
        "pt_city":                  "5_patient_city",
        "pt_state":                 "5_patient_state",
        "pt_zip":                   "5_patient_zip",
        "pt_AreaCode":              "_pt_area_code",  # composed into phone
        "pt_phone":                 "_pt_phone_num",  # composed into phone
        # Insured
        "ins_name":                 "4_insured_name",
        "ins_street":               "7_insured_address",
        "ins_city":                 "7_insured_city",
        "ins_state":                "7_insured_state",
        "ins_zip":                  "7_insured_zip",
        "ins_phone area":           "_ins_area_code",
        "ins_phone":                "_ins_phone_num",
        # Other insured
        "other_ins_name":           "9_other_insured_name",
        "other_ins_policy":         "9a_other_insured_policy",
        # Policy / plan
        "ins_policy":               "11_insured_policy_group",
        "ins_plan_name":            "11c_insurance_plan_name",
        "other_ins_plan_name":      "_other_ins_plan_name",
        # DOB (composed)
        "birth_mm":                 "_birth_mm",
        "birth_dd":                 "_birth_dd",
        "birth_yy":                 "_birth_yy",
        "ins_dob_mm":               "_ins_dob_mm",
        "ins_dob_dd":               "_ins_dob_dd",
        "ins_dob_yy":               "_ins_dob_yy",
        # Signatures / dates
        "pt_signature":             "12_patient_signature",
        "pt_date":                  "12_signature_date",
        "ins_signature":            "13_insured_signature",
        # Additional
        "96":                       "19_additional_claim_info",
        "charge":                   "20_charges",
        # Diagnosis
        "diagnosis1":               "21_diagnosis_a",
        "diagnosis2":               "21_diagnosis_b",
        "diagnosis3":               "_diagnosis_c",
        "diagnosis4":               "_diagnosis_d",
        "prior_auth":               "_prior_auth",
        # Bottom fields (these were broken by bad bboxes before)
        "tax_id":                   "25_federal_tax_id",
        "pt_account":               "26_patient_account",
        "t_charge":                 "28_total_charge",
        "amt_paid":                 "29_amount_paid",
        "physician_signature":      "31_physician_signature",
        "physician_date":           "_physician_date",
        # Service facility
        "fac_name":                 "32_service_facility_name",
        "fac_street":               "32_service_facility_address",
        "fac_location":             "_fac_location",
        "pin1":                     "32a_npi",
        # Billing provider
        "doc_name":                 "33_billing_provider_name",
        "doc_street":               "33_billing_provider_address",
        "doc_location":             "_doc_location",
        "doc_phone area":           "_doc_phone_area",
        "doc_phone":                "_doc_phone_num",
        "pin":                      "33a_npi",
    }

    # Widget names for radio/checkbox groups mapped to schema
    WIDGET_CHECKBOX_MAP: Dict[str, str] = {
        "sex":           "3_patient_sex",
        "ins_sex":       "11a_insured_sex",
        "insurance_type": "_insurance_type",
        "rel_to_ins":    "_rel_to_insured",
        "employment":    "_employment_related",
        "pt_auto_accident": "_auto_accident",
        "other_accident":   "_other_accident",
        "assignment":    "27_accept_assignment",
        "lab":           "20_outside_lab",
        "ssn":           "_ssn_ein",
    }

    # Service-line widget prefix patterns (lines 1-6)
    SVC_LINE_FIELDS = [
        "sv{n}_mm_from", "sv{n}_dd_from", "sv{n}_yy_from",
        "sv{n}_mm_end",  "sv{n}_dd_end",  "sv{n}_yy_end",
        "place{n}", "type{n}", "cpt{n}",
        "mod{n}", "mod{n}a", "mod{n}b", "mod{n}c",
        "diag{n}", "ch{n}", "day{n}", "local{n}",
    ]
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize agents
        self.form_id_agent = FormIdentificationAgent()
        self.alignment_agent = TemplateAlignmentAgent()
        self.layout_agent = LayoutDetectionAgent(self.config)
        self.ocr_agent = OCRAgent(self.config)
        self.labeling_agent = LabelingAgent(self.config)
        self.validation_agent = ValidationAgent(self.config)
        self._template_word_blacklist: Dict[str, set] = {}

    # ──────────────────────────────────────────────────────────────────
    # LANE A: AcroForm widget extraction (highest accuracy for fillable PDFs)
    # ──────────────────────────────────────────────────────────────────

    def _extract_widgets(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Try to extract AcroForm widget values from a PDF.
        Returns None if the PDF has no widgets or too few filled values.
        Returns a dict of {widget_field_name: field_value} for all non-empty widgets.
        """
        if not str(path).lower().endswith(".pdf"):
            return None
        try:
            import fitz
            doc = fitz.open(str(path))
            page = doc[0]
            widgets = list(page.widgets())
            if len(widgets) < 10:
                doc.close()
                return None

            raw: Dict[str, Any] = {}
            # Track all values + rects for bbox generation
            widget_data: List[Dict[str, Any]] = []
            for w in widgets:
                fn = str(getattr(w, "field_name", "") or "").strip()
                fv = str(getattr(w, "field_value", "") or "").strip()
                ft = getattr(w, "field_type", -1)
                r = getattr(w, "rect", None)
                if not fn:
                    continue
                widget_data.append({
                    "name": fn, "value": fv, "type": ft,
                    "rect": r,
                    "page_w": page.rect.width, "page_h": page.rect.height,
                })
                # For radio/checkbox groups, only store if value is non-empty
                # and keep the first non-empty value per group name
                if ft == 2:  # radio/checkbox
                    if fv and fn not in raw:
                        raw[fn] = fv
                elif ft == 7:  # text
                    if fv:
                        # Some widget names repeat (e.g. insurance_type); keep first non-empty
                        if fn not in raw:
                            raw[fn] = fv
                elif ft == 1:  # pushbutton
                    pass
                else:
                    if fv and fn not in raw:
                        raw[fn] = fv

            doc.close()

            # Only proceed if we got a meaningful number of filled fields
            filled = sum(1 for v in raw.values() if v)
            if filled < 5:
                print(f"[Lane A] Only {filled} filled widgets — too few, skipping widget path")
                return None

            print(f"[Lane A] Extracted {filled} filled widget values from {len(widgets)} total widgets")
            return {"raw": raw, "widget_data": widget_data, "total_widgets": len(widgets), "filled": filled}
        except Exception as e:
            print(f"[Lane A] Widget extraction failed: {e}")
            return None

    def _map_widgets_to_schema(
        self,
        widget_info: Dict[str, Any],
        form_type: Optional[FormType] = None
    ) -> Tuple[Dict[str, str], List[DetectedBlock]]:
        """
        Map extracted widget values to schema field IDs.
        Returns (extracted_fields dict, list of DetectedBlocks for UI overlay).
        """
        if form_type is None:
            form_type = FormType.CMS1500
        if form_type == FormType.UB04:
            return self._map_widgets_to_schema_ub04(widget_info)

        raw = widget_info["raw"]
        widget_data = widget_info.get("widget_data", [])

        # Load schema labels + field types for block typing (Lane A UI consistency)
        schema_info: Dict[str, Dict[str, str]] = {}
        try:
            import json as _json
            schema_path = Path(__file__).parent.parent.parent / "data" / "schemas" / "cms-1500.json"
            if schema_path.exists():
                with open(schema_path) as _f:
                    _schema = _json.load(_f)
                for _field in _schema.get("fields", []):
                    fid = _field.get("id")
                    if fid:
                        schema_info[fid] = {
                            "label": _field.get("label", fid),
                            "field_type": _field.get("field_type", "text")
                        }
        except Exception:
            schema_info = {}

        # Build name→rect lookup (for UI bboxes)
        name_to_rect: Dict[str, Any] = {}
        page_w = 612.0
        page_h = 792.0
        for wd in widget_data:
            n = wd["name"]
            if n not in name_to_rect and wd.get("rect"):
                name_to_rect[n] = wd["rect"]
                page_w = wd.get("page_w", 612.0)
                page_h = wd.get("page_h", 792.0)

        extracted: Dict[str, str] = {}
        blocks: List[DetectedBlock] = []

        def _normalize_widget_value(v: str) -> str:
            """Clean up common widget value artifacts (extra internal spaces around hyphens, etc.)."""
            import re
            v = v.strip()
            # Collapse spaces around hyphens: "ID10- 45678" → "ID10-45678"
            v = re.sub(r'\s*-\s*', '-', v)
            # Collapse double spaces
            v = re.sub(r' {2,}', ' ', v)
            return v

        # ── Map simple text fields ──
        for widget_name, schema_id in self.WIDGET_TO_SCHEMA.items():
            val = raw.get(widget_name, "")
            if val and not schema_id.startswith("_"):
                extracted[schema_id] = _normalize_widget_value(val)

        # ── Compose multi-part fields ──
        # Patient DOB
        mm = raw.get("birth_mm", "")
        dd = raw.get("birth_dd", "")
        yy = raw.get("birth_yy", "")
        if mm and dd and yy:
            extracted["3_patient_dob"] = f"{mm}/{dd}/{yy}"

        # Insured DOB
        imm = raw.get("ins_dob_mm", "")
        idd = raw.get("ins_dob_dd", "")
        iyy = raw.get("ins_dob_yy", "")
        if imm and idd and iyy:
            extracted["11a_insured_dob"] = f"{imm}/{idd}/{iyy}"

        # Patient phone
        area = raw.get("pt_AreaCode", "")
        num = raw.get("pt_phone", "")
        if area and num:
            extracted["5_patient_phone"] = f"({area}) {num}"
        elif num:
            extracted["5_patient_phone"] = num

        # Insured phone
        iarea = raw.get("ins_phone area", "")
        inum = raw.get("ins_phone", "")
        if iarea and inum:
            extracted["7_insured_phone"] = f"({iarea}) {inum}"
        elif inum:
            extracted["7_insured_phone"] = inum

        # Billing provider phone
        darea = raw.get("doc_phone area", "")
        dnum = raw.get("doc_phone", "")
        if darea and dnum:
            extracted["33_billing_provider_phone"] = f"({darea}) {dnum}"
        elif dnum:
            extracted["33_billing_provider_phone"] = dnum

        # Header notes (compose from insurance name/address/city)
        ins_parts = [raw.get(k, "") for k in ["insurance_name", "insurance_address", "insurance_address2", "insurance_city_state_zip"]]
        ins_header = " ".join(p for p in ins_parts if p).strip()
        if ins_header:
            extracted["header_top_right_notes"] = ins_header

        # ── Checkboxes ──
        for widget_name, schema_id in self.WIDGET_CHECKBOX_MAP.items():
            val = raw.get(widget_name, "")
            if val and not schema_id.startswith("_"):
                extracted[schema_id] = val

        # Patient sex: widget stores "M" or "F" as the radio value
        sex_val = raw.get("sex", "")
        if sex_val:
            extracted["3_patient_sex"] = sex_val
            if sex_val.upper() == "M":
                extracted["3_patient_sex_m"] = "X"
            elif sex_val.upper() == "F":
                extracted["3_patient_sex_f"] = "X"

        # Insured sex
        isex = raw.get("ins_sex", "")
        if isex:
            extracted["11a_insured_sex"] = isex

        # ── Service lines (compose into a summary) ──
        svc_lines = []
        for n in range(1, 7):
            mm_from = raw.get(f"sv{n}_mm_from", "")
            if not mm_from:
                continue  # no more lines
            parts = []
            # Date from
            df = "/".join(filter(None, [raw.get(f"sv{n}_mm_from"), raw.get(f"sv{n}_dd_from"), raw.get(f"sv{n}_yy_from")]))
            dt = "/".join(filter(None, [raw.get(f"sv{n}_mm_end"), raw.get(f"sv{n}_dd_end"), raw.get(f"sv{n}_yy_end")]))
            cpt = raw.get(f"cpt{n}", "")
            mod = raw.get(f"mod{n}", "")
            diag = raw.get(f"diag{n}", "")
            ch = raw.get(f"ch{n}", "")
            line = f"{df}-{dt} {cpt} {mod} {diag} ${ch}".strip()
            svc_lines.append(line)
        if svc_lines:
            extracted["24_service_lines"] = " | ".join(svc_lines)

        # ── Build DetectedBlocks for UI overlay ──
        zoom = 300.0 / 72.0

        def _rect_to_bbox(r) -> Tuple[float, float, float, float]:
            return (r.x0 * zoom, r.y0 * zoom, r.x1 * zoom, r.y1 * zoom)

        def _combine_widget_rects(widget_names: list) -> Optional[Tuple[float, float, float, float]]:
            """Combine multiple widget rects into one encompassing bbox."""
            rects = [name_to_rect[n] for n in widget_names if n in name_to_rect]
            if not rects:
                return None
            x0 = min(r.x0 for r in rects) * zoom
            y0 = min(r.y0 for r in rects) * zoom
            x1 = max(r.x1 for r in rects) * zoom
            y1 = max(r.y1 for r in rects) * zoom
            return (x0, y0, x1, y1)

        # Pre-build schema_id → bbox for composed fields
        composed_bboxes: Dict[str, Tuple[float, float, float, float]] = {}

        # DOB fields: combine mm+dd+yy rects
        dob_bbox = _combine_widget_rects(["birth_mm", "birth_dd", "birth_yy"])
        if dob_bbox:
            composed_bboxes["3_patient_dob"] = dob_bbox
        idob_bbox = _combine_widget_rects(["ins_dob_mm", "ins_dob_dd", "ins_dob_yy"])
        if idob_bbox:
            composed_bboxes["11a_insured_dob"] = idob_bbox

        # Phone fields: combine area + number
        for sid, widgets in [
            ("5_patient_phone", ["pt_AreaCode", "pt_phone"]),
            ("7_insured_phone", ["ins_phone area", "ins_phone"]),
            ("33_billing_provider_phone", ["doc_phone area", "doc_phone"]),
        ]:
            b = _combine_widget_rects(widgets)
            if b:
                composed_bboxes[sid] = b

        # Header: combine insurance name/address/city
        hdr_bbox = _combine_widget_rects(["insurance_name", "insurance_address", "insurance_address2", "insurance_city_state_zip"])
        if hdr_bbox:
            composed_bboxes["header_top_right_notes"] = hdr_bbox

        # Service lines: combine all service line widgets for lines 1-6
        svc_widgets = []
        for n in range(1, 7):
            svc_widgets.extend([f"sv{n}_mm_from", f"local{n}"])
        svc_bbox = _combine_widget_rects(svc_widgets)
        if svc_bbox:
            composed_bboxes["24_service_lines"] = svc_bbox

        # Sex checkboxes: combine both options
        sex_bbox = _combine_widget_rects(["sex"])  # radio widget covers both
        if "sex" in name_to_rect:
            composed_bboxes["3_patient_sex"] = _rect_to_bbox(name_to_rect["sex"])
            composed_bboxes["3_patient_sex_m"] = _rect_to_bbox(name_to_rect["sex"])

        for schema_id, value in extracted.items():
            if not value:
                continue

            # 1. Check composed bboxes first
            bbox = composed_bboxes.get(schema_id)

            # 2. Check direct widget mapping
            if bbox is None:
                for wn, sid in self.WIDGET_TO_SCHEMA.items():
                    if sid == schema_id and wn in name_to_rect:
                        bbox = _rect_to_bbox(name_to_rect[wn])
                        break

            # 3. Check checkbox mapping
            if bbox is None:
                for wn, sid in self.WIDGET_CHECKBOX_MAP.items():
                    if sid == schema_id and wn in name_to_rect:
                        bbox = _rect_to_bbox(name_to_rect[wn])
                        break

            # 4. Try to find from schema JSON as last resort
            if bbox is None:
                try:
                    import json as _json
                    schema_path = Path(__file__).parent.parent.parent / "data" / "schemas" / "cms-1500.json"
                    if schema_path.exists():
                        with open(schema_path) as _f:
                            _schema = _json.load(_f)
                        for _field in _schema.get("fields", []):
                            if _field.get("id") == schema_id:
                                bn = _field.get("bbox_norm", [])
                                if len(bn) == 4:
                                    pw_px = page_w * zoom
                                    ph_px = page_h * zoom
                                    bbox = (bn[0]*pw_px, bn[1]*ph_px, bn[2]*pw_px, bn[3]*ph_px)
                                break
                except Exception:
                    pass

            if bbox is None:
                bbox = (0.0, 0.0, 1.0, 1.0)

            # Schema label + field type (for consistent UI labeling with Lane B)
            info = schema_info.get(schema_id, {})
            label = info.get("label", schema_id)
            field_type = info.get("field_type", "text")

            # Infer block type (same logic as Lane B zone matching)
            label_l = (label or "").lower()
            field_id_l = (schema_id or "").lower()
            if field_type == "checkbox":
                block_type = BlockType.CHECKBOX
            elif field_type == "signature" or "signature" in field_id_l or "signature" in label_l:
                block_type = BlockType.SIGNATURE
            elif "table" in field_id_l or "table" in label_l:
                block_type = BlockType.TABLE
            elif "figure" in field_id_l or "figure" in label_l or "image" in label_l or "graphic" in label_l:
                block_type = BlockType.FIGURE
            elif "title" in field_id_l or "title" in label_l:
                block_type = BlockType.TITLE
            elif "header" in field_id_l or "header" in label_l:
                block_type = BlockType.HEADER
            elif "footer" in field_id_l or "footer" in label_l:
                block_type = BlockType.FOOTER
            elif ("page" in label_l and "number" in label_l) or "page_num" in field_id_l or "pagenum" in field_id_l:
                block_type = BlockType.PAGE_NUM
            elif "list" in field_id_l or "list" in label_l:
                block_type = BlockType.LIST
            else:
                block_type = BlockType.FORM_FIELD

            blocks.append(DetectedBlock(
                id=schema_id,
                block_type=block_type,
                bbox=bbox,
                text=value,
                confidence=0.99,  # widget values are ground truth
                metadata={
                    "source": "acroform_widget",
                    "label": label,
                    "semantic_label": label,
                    "field_type": field_type,
                    "ocr_engine": "widget",
                }
            ))

        print(f"[Lane A] Mapped {len(extracted)} schema fields from widgets")
        return extracted, blocks

    # ── UB-04 Widget name → schema field ID mapping ──────────────────────────
    # Built from actual UB-04 XFA widget names found in fillable PDFs.
    UB04_WIDGET_TO_SCHEMA: Dict[str, str] = {
        # Provider info (FL 1)
        "Address1": "fl1_provider_name",
        "address2": "fl1_provider_address1",
        "address3": "fl1_provider_city_state_zip",
        "address4": "_provider_phone",
        
        # Patient control / bill type (FL 3-4)
        "patctrl": "fl3a_patient_control",
        "typebill": "fl4_type_of_bill",
        
        # Tax ID and dates (FL 5-6)
        "provtaxID": "fl5_federal_tax_id",
        "fromdte": "fl6_from_date",
        "thrudate": "fl6_thru_date",
        
        # Patient info (FL 8-11)
        "patientIDnum": "fl8a_patient_id",
        "pataddrstreet": "fl9_patient_address",
        "pataddresscity": "fl9_patient_city",
        "pataddrState": "fl9_patient_state",
        "pataddresszip": "fl9_patient_zip",
        "DOB": "fl10_patient_dob",
        "sex": "fl11_patient_sex",
        
        # Admission info (FL 12-17)
        "lrd": "fl12_admission_date",
        "17date": "fl13_admission_hour",
        "18type": "fl14_admission_type",
        "19source": "fl15_admission_source",
        "20src": "fl16_discharge_hour",
        "21dhr": "fl17_patient_status",
        
        # Occurrence codes (FL 31-34)
        "32_.1": "fl31_occurrence_code_1",
        "32occ_.1": "fl31_occurrence_date_1",
        "33_.1": "fl32_occurrence_code_2",
        "33code_.1": "fl32_occurrence_date_2",
        "34_.1": "fl33_occurrence_code_3",
        "34code_.1": "fl33_occurrence_date_3",
        "35_.1": "fl34_occurrence_code_4",
        "35code_.1": "fl34_occurrence_date_4",
        "32_.2": "_occurrence_code_5",
        "32occ_.2": "_occurrence_date_5",
        "33_.2": "_occurrence_code_6",
        "33code_.2": "_occurrence_date_6",
        "34_.2": "_occurrence_code_7",
        "34code_.2": "_occurrence_date_7",
        "35_.2": "_occurrence_code_8",
        "35code_.2": "_occurrence_date_8",
        
        # Occurrence span codes (FL 35-36)
        "36_.1": "fl35_occurrence_span_code_1",
        "37_.3_.0_.0_.0": "_occurrence_span_code_2",
        "37_.3_.0_.1": "fl35_occurrence_span_from_1",
        "37_.3_.0_.1_1": "_occurrence_span_from_2",
        "37_.3_.0_.1_2": "_occurrence_span_from_3",
        "37_.3_.1_.1": "fl35_occurrence_span_thru_1",
        
        # Responsible party (FL 38)
        "38_.1": "fl38_responsible_party",
        
        # Value codes (FL 39-41)
        "39code_.1": "fl39a_value_code_1",
        "val_.1": "fl39a_value_amount_1",
        "39code_.2": "fl39b_value_code_2",
        "val_.1_1": "fl39b_value_amount_2",
        "39code_.3": "fl39c_value_code_3",
        "val_.1_2": "fl39c_value_amount_3",
        "39code_.4": "fl39d_value_code_4",
        "val_.1_3": "fl39d_value_amount_4",
        
        # Revenue codes and charges (FL 42-47) - Line 1
        "revcd42_.1": "fl42_revenue_code_1",
        "43desc_.1": "fl43_description_1",
        "44hcps_.1": "fl44_hcpcs_1",
        "45servdate_.1": "fl45_service_date_1",
        "46servunits_.1": "fl46_units_1",
        "47totalcharges_.1": "fl47_charges_1",
        
        # Revenue codes and charges - Line 2
        "revcd42_.2": "fl42_revenue_code_2",
        "43desc_.2": "fl43_description_2",
        "44hcps_.2": "fl44_hcpcs_2",
        "45servdate_.2": "fl45_service_date_2",
        "46servunits_.2": "fl46_units_2",
        "47totalcharges_.2": "fl47_charges_2",
        
        # Total charges (FL 47 totals row)
        "47totalcharges_.23": "fl47_total_charges",
        "revcd42_.22_.1": "_revenue_totals_code",
        
        # Payer info (FL 50-56)
        "50payer_.1": "fl50_payer_name_a",
        "51providernum_.1": "fl51_health_plan_id_a",
        "52relinfo_.1": "fl52_release_info_a",
        "52asgben_.1": "fl53_assignment_a",
        "54prior_.1": "fl54_prior_payments_a",
        "55est_.1": "fl55_estimated_due_a",
        "56_.1": "fl56_npi_a",
        
        # Insured info (FL 58-62)
        "57": "fl58_insured_name_a",
        "59prel_.1": "fl59_patient_rel_a",
        "60cert_.1": "fl60_insured_id_a",
        "61groupname_.1": "fl61_group_name_a",
        "62insgroup_.1": "fl62_group_number_a",
        
        # Treatment / employer (FL 63-66)
        "63treatment_.1": "fl63_treatment_auth_a",
        "65empname_.1": "fl65_employer_name_a",
        "66emploc_.1": "fl66_employer_loc_a",
        
        # Diagnosis codes (FL 67-72)
        "67prin": "fl67_principal_diagnosis",
        "68code": "fl67a_diagnosis_a",
        "69code": "fl67b_diagnosis_b",
        "71code": "fl71_pps_code",
        "72code": "fl72_eci_code",
        "73code": "fl67c_diagnosis_c",
        "75code": "fl67d_diagnosis_d",
        "76admdiag": "fl69_admitting_diagnosis",
        "77ecode": "fl70_patient_reason",
        "79pc": "fl79_admitting_dx_code",
        
        # Procedure codes (FL 74)
        "princode_.0": "fl74_principal_procedure",
        "prindate": "fl74_principal_proc_date",
        "other2": "fl74a_other_procedure_1",
        "other3": "fl74a_other_proc_date_1",
        "other3date": "_other_proc_date_2",
        "74code_.0": "fl74_principal_procedure",
        "74code_.1_.0_.0_.1": "_other_procedure_4",
        
        # Principal code dates
        "princode_.1_.1": "fl74_principal_proc_date",
        "princode_.0_.1_.0_.1": "_proc_date_2",
        "princode_.1_.0_.0": "_proc_code_2",
        "princode_.0_.1_.0": "_proc_code_3",
        "princode_.0_.1_.1_.0_.0": "_proc_code_4",
        "princode_.0_.1_.1_.0_.0_.1": "_proc_date_4",
        "princode_.1_.0_.1": "_proc_code_5",
        "princode_.1_.0_.0_.1": "_proc_code_6",
        "princode_.0_.1_.1_.0_.1": "_proc_code_7",
        "princode_.0_.1_.1_.1_.0": "_proc_date_7",
        "princode_.1_.0_.1_.1": "_proc_date_5",
        "princode_.0_.1_.1_.0_.1_.1": "_proc_date_8",
        "princode_.0_.1_.1_.1_.1_.0_.0": "_proc_code_taxonomy",
        "princode_.0_.1_.1_.1_.0_.1_.0": "_proc_taxonomy_code",
        
        # Attending physician (FL 76)
        "NPI_.0": "fl76_attending_npi",
        "82attend_.0": "fl76_attending_qual",
        "NPI_.1_.0_.0": "fl76_attending_last",
        "NPI_.1_.1": "fl76_attending_first",
        
        # Operating physician (FL 77)
        "NPI_.1_.0_.1_.1": "fl77_operating_npi",
        "NPI_.1_.0_.1_.0_.0": "fl77_operating_last",
        "NPI_.1_.0_.0_.1": "fl77_operating_first",
        
        # Other provider (FL 78-79)
        "NPI_.1_.0_.1_.0_.1": "fl78_other_npi_1",
        "NPI_.1_.0_.1_.0_.0_.1": "fl78_other_first_1",
        
        # Remarks (FL 80)
        "remarks": "fl80_remarks",
        
        # Attending name (FL 82-83)
        "83id": "_attending_id",
    }

    def _map_widgets_to_schema_ub04(self, widget_info: Dict[str, Any]) -> Tuple[Dict[str, str], List[DetectedBlock]]:
        """
        Map UB-04 widget values to schema fields using explicit name mapping + spatial fallback.
        This supports fillable UB-04 PDFs with XFA-style widget names.
        """
        import re
        
        raw = widget_info.get("raw", {}) or {}
        widget_data = widget_info.get("widget_data", []) or []
        if not widget_data:
            return {}, []

        # Build name→rect lookup
        name_to_rect: Dict[str, Any] = {}
        page_w = 612.0
        page_h = 792.0
        for wd in widget_data:
            n = wd.get("name")
            if n and n not in name_to_rect and wd.get("rect"):
                name_to_rect[n] = wd["rect"]
                page_w = wd.get("page_w", page_w)
                page_h = wd.get("page_h", page_h)

        # Normalize XFA widget names to simpler form
        def _normalize_widget_name(raw_name: str) -> str:
            """Simplify XFA widget name to a clean identifier."""
            name = raw_name
            prefixes = [
                "topmostSubform[0].Page1[0].topmostSubform_0_\\.Page1_0_\\.",
                "topmostSubform[0].Page1[0]."
            ]
            for p in prefixes:
                if name.startswith(p):
                    name = name[len(p):]
                    break
            name = re.sub(r'\[0\]', '', name)
            name = name.replace('\\', '')
            name = re.sub(r'_0_$', '', name)
            name = re.sub(r'\.0$', '', name)
            return name.strip('_.')

        # Load UB-04 schema for metadata
        schema_fields = []
        schema_info: Dict[str, Dict[str, str]] = {}
        try:
            import json as _json
            schema_path = Path(__file__).parent.parent.parent / "data" / "schemas" / "ub-04.json"
            if schema_path.exists():
                with open(schema_path) as _f:
                    schema = _json.load(_f)
                for field in schema.get("fields", []):
                    fid = field.get("id")
                    if fid:
                        schema_fields.append(field)
                        schema_info[fid] = {
                            "label": field.get("label", fid),
                            "field_type": field.get("field_type", "text"),
                            "bbox_norm": field.get("bbox_norm"),
                            "business_key": field.get("business_key")
                        }
        except Exception:
            pass

        def _normalize_widget_value(v: str) -> str:
            v = str(v).strip()
            v = re.sub(r"\s*-\s*", "-", v)
            v = re.sub(r" {2,}", " ", v)
            return v

        def _bbox_iou(a, b) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            inter_x0 = max(ax0, bx0)
            inter_y0 = max(ay0, by0)
            inter_x1 = min(ax1, bx1)
            inter_y1 = min(ay1, by1)
            inter_w = max(0.0, inter_x1 - inter_x0)
            inter_h = max(0.0, inter_y1 - inter_y0)
            inter = inter_w * inter_h
            area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
            area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
            denom = area_a + area_b - inter
            return inter / denom if denom > 0 else 0.0

        # Process widgets and map to schema
        extracted: Dict[str, str] = {}
        blocks: List[DetectedBlock] = []
        zoom = 300.0 / 72.0
        assigned_fields: set = set()

        def _rect_to_bbox(r) -> Tuple[float, float, float, float]:
            return (r.x0 * zoom, r.y0 * zoom, r.x1 * zoom, r.y1 * zoom)

        # First pass: direct name mapping
        for wd in widget_data:
            raw_name = wd.get("name") or ""
            rect = wd.get("rect")
            value = raw.get(raw_name, "")
            if not value or str(value).strip() == "" or value in ("Off", "off"):
                continue

            norm_name = _normalize_widget_name(raw_name)
            schema_id = self.UB04_WIDGET_TO_SCHEMA.get(norm_name)
            
            # Skip internal fields (starting with _)
            if schema_id and schema_id.startswith("_"):
                schema_id = None
            
            if schema_id and schema_id not in assigned_fields:
                assigned_fields.add(schema_id)
                clean_value = _normalize_widget_value(value)
                extracted[schema_id] = clean_value
                
                meta = schema_info.get(schema_id, {})
                label = meta.get("label", schema_id)
                field_type = meta.get("field_type", "text")
                schema_block_type = meta.get("block_type", "form_field")
                
                bbox = _rect_to_bbox(rect) if rect else (0.0, 0.0, 1.0, 1.0)
                
                # Map schema block_type to BlockType enum
                block_type_map = {
                    "header": BlockType.HEADER,
                    "table_cell": BlockType.TABLE,
                    "table": BlockType.TABLE,
                    "checkbox": BlockType.CHECKBOX,
                    "signature": BlockType.SIGNATURE,
                    "form_field": BlockType.FORM_FIELD,
                }
                block_type = block_type_map.get(schema_block_type, BlockType.FORM_FIELD)
                
                # Override with field_type if checkbox
                if field_type == "checkbox":
                    block_type = BlockType.CHECKBOX

                blocks.append(DetectedBlock(
                    id=schema_id,
                    block_type=block_type,
                    bbox=bbox,
                    text=clean_value,
                    confidence=0.99,
                    metadata={
                        "source": "acroform_widget",
                        "label": label,
                        "semantic_label": label,
                        "field_type": field_type,
                        "block_type": schema_block_type,
                        "ocr_engine": "widget",
                        "widget_name": norm_name,
                        "business_key": meta.get("business_key")
                    }
                ))

        # Second pass: spatial matching for unmapped widgets with values
        for wd in widget_data:
            raw_name = wd.get("name") or ""
            rect = wd.get("rect")
            value = raw.get(raw_name, "")
            if not value or str(value).strip() == "" or value in ("Off", "off"):
                continue
            if not rect:
                continue

            norm_name = _normalize_widget_name(raw_name)
            # Skip if already mapped
            if self.UB04_WIDGET_TO_SCHEMA.get(norm_name):
                continue

            # Spatial matching
            wx0, wy0, wx1, wy1 = rect.x0 / page_w, rect.y0 / page_h, rect.x1 / page_w, rect.y1 / page_h
            cx = (wx0 + wx1) / 2.0
            cy = (wy0 + wy1) / 2.0

            best_id = None
            best_score = 0.0

            for field in schema_fields:
                fid = field.get("id")
                bbox_norm = field.get("bbox_norm") or []
                if not fid or fid in assigned_fields or len(bbox_norm) != 4:
                    continue
                fx0, fy0, fx1, fy1 = bbox_norm
                inside = fx0 <= cx <= fx1 and fy0 <= cy <= fy1
                iou = _bbox_iou((wx0, wy0, wx1, wy1), (fx0, fy0, fx1, fy1))
                score = (1.0 + iou) if inside else iou
                if score > best_score:
                    best_score = score
                    best_id = fid

            if best_id and best_score >= 0.1:
                assigned_fields.add(best_id)
                clean_value = _normalize_widget_value(value)
                extracted[best_id] = clean_value
                
                meta = schema_info.get(best_id, {})
                label = meta.get("label", best_id)
                field_type = meta.get("field_type", "text")
                bbox = _rect_to_bbox(rect)

                blocks.append(DetectedBlock(
                    id=best_id,
                    block_type=BlockType.FORM_FIELD,
                    bbox=bbox,
                    text=clean_value,
                    confidence=0.95,
                    metadata={
                        "source": "acroform_widget_spatial",
                        "label": label,
                        "semantic_label": label,
                        "field_type": field_type,
                        "ocr_engine": "widget",
                        "widget_name": norm_name,
                        "match_score": best_score,
                        "business_key": meta.get("business_key")
                    }
                ))

        print(f"[Lane A] Mapped {len(extracted)} UB-04 fields from widgets")
        return extracted, blocks

    # Comprehensive CMS-1500 template word blacklist.
    # Every pre-printed word on the standard CMS-1500 (02/12) form.
    # Used to filter template text from OCR results AFTER recognition,
    # so handwriting quality is preserved (no red removal needed).
    _CMS1500_TEMPLATE_WORDS = {
        # Form title and header
        "health", "insurance", "claim", "form", "approved", "national",
        "uniform", "committee", "nucc", "pica", "02/12", "02/2",
        # Insurance type labels
        "medicare", "medicaid", "tricare", "champva", "group", "feca",
        "blk", "lung", "other",
        # Field labels — patient info
        "patient's", "patients", "patient", "name", "last", "first",
        "middle", "initial", "birth", "date", "sex", "address",
        "city", "state", "zip", "code", "telephone", "include",
        "area", "relationship", "insured", "insured's", "insureds",
        "self", "spouse", "child",
        # Field labels — other insured
        "other", "policy", "number", "group", "feca",
        "employment", "current", "previous", "auto", "accident",
        "place", "reserved", "nucc", "use",
        # Field labels — insurance
        "plan", "program", "another", "benefit", "complete",
        "items", "designated",
        # Field labels — signatures
        "signature", "authorize", "release", "medical", "information",
        "necessary", "process", "request", "payment", "government",
        "benefits", "myself", "party", "accepts", "assignment",
        "below", "signed", "read", "back", "before", "completing",
        "signing",
        # Field labels — dates and medical
        "illness", "injury", "pregnancy", "lmp", "qual",
        "dates", "unable", "work", "occupation", "from",
        "referring", "provider", "source", "hospitalization",
        "services", "related", "additional", "information",
        "outside", "lab", "charges", "diagnosis", "nature",
        "relate", "service", "line", "icd", "ind",
        "resubmission", "original", "ref", "prior",
        "authorization",
        # Field labels — service lines
        "procedures", "supplies", "explain", "unusual",
        "circumstances", "cpt", "hcpcs", "modifier",
        "pointer", "days", "units", "epsdt", "family",
        "qual", "rendering", "npi",
        # Field labels — bottom section
        "federal", "tax", "i.d.", "i.d", "ssn", "ein",
        "account", "accept", "total", "charge", "amount",
        "paid", "rsvd", "physician", "supplier", "degrees",
        "credentials", "certify", "statements", "reverse",
        "apply", "bill", "made", "part", "thereof",
        "facility", "location", "billing", "info",
        # Instructions and misc
        "no.", "street", "print", "type", "please",
        "instruction", "manual", "available", "www.nucc.org",
        "omb-0938-1197", "1500",
        # Field numbers (OCR may read these)
        "1a.", "1a", "2.", "3.", "4.", "5.", "6.", "7.", "8.",
        "9.", "9a.", "9d.", "10.", "10a.", "10b.", "10c.", "10d.",
        "11.", "11a.", "11b.", "11c.", "11d.", "12.", "13.",
        "14.", "15.", "16.", "17.", "17a.", "17b.", "18.",
        "19.", "20.", "21.", "22.", "23.", "24.", "25.",
        "26.", "27.", "28.", "29.", "30.", "31.", "32.",
        "32a.", "32b.", "33.", "33a.", "33b.",
        # Common OCR misreads of template labels
        "patent's", "patents", "patent", "nsured's", "nsured",
        "nsurance", "atient", "atient's", "ddress", "elephone",
        "ignature", "hysician", "iagnosis", "rocedures",
        "ederal", "illing", "acility", "ertify",
        # Parenthetical instructions
        "(last", "(first", "(middle", "(no.,", "(include",
        "(designated", "(current", "(for", "program",
        "item", "(medicare#)", "(medicaid#)", "(id#/dod#)",
        "(member", "id#)", "(id#)",
    }
    
    def _get_template_word_blacklist(self, template_key: str) -> set:
        """
        Return comprehensive blacklist of pre-printed template words.
        
        For CMS-1500: returns a hardcoded set of ALL words printed on the
        standard form. This is more reliable than OCR-ing the template image
        because it includes common OCR misreads of template labels.
        
        Used in the OCR-first-clean-after pipeline: PaddleOCR reads the
        raw aligned image (with red template visible for max handwriting
        quality), then template words are filtered from the results.
        """
        key = (template_key or "").lower().strip()
        if not key:
            return set()
        if key in self._template_word_blacklist:
            return self._template_word_blacklist[key]
        
        if key in {"cms-1500", "cms1500"}:
            # Use the hardcoded comprehensive blacklist
            # Also add dynamically OCR'd template words as supplement
            words = set(self._CMS1500_TEMPLATE_WORDS)
            try:
                from src.processing.registration import load_and_process_reference
                from src.ocr.paddle_ocr import PaddleOCRWrapper
                ref_data = load_and_process_reference(key)
                if ref_data and ref_data.get("image") is not None:
                    paddle = PaddleOCRWrapper()
                    word_boxes = paddle.extract_text(ref_data["image"])
                    for wb in word_boxes or []:
                        text = (wb.text or "").strip().lower()
                        if len(text) >= 2:
                            words.add(text)
                    print(f"[Pipeline] CMS-1500 blacklist: {len(words)} words (hardcoded + OCR'd template)")
            except Exception:
                print(f"[Pipeline] CMS-1500 blacklist: {len(words)} words (hardcoded only)")
            self._template_word_blacklist[key] = words
            return words
        
        # Other templates: OCR-based blacklist (fallback)
        try:
            from src.processing.registration import load_and_process_reference
            from src.ocr.paddle_ocr import PaddleOCRWrapper
            ref_data = load_and_process_reference(key)
            if not ref_data or ref_data.get("image") is None:
                return set()
            paddle = PaddleOCRWrapper()
            word_boxes = paddle.extract_text(ref_data["image"])
            words = set()
            for wb in word_boxes or []:
                text = (wb.text or "").strip().lower()
                if len(text) >= 3:
                    words.add(text)
            self._template_word_blacklist[key] = words
            return words
        except Exception:
            return set()

    def _extract_pdf_digital_words(self, page, zoom: float) -> Optional[List[Any]]:
        """Extract word-level boxes from a PDF text layer and scale into rendered pixel space."""
        try:
            from utils.models import WordBox
            words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,word_no (PDF points)
            if not words or len(words) < 30:
                return None
            scaled = []
            for w in words:
                if len(w) < 5:
                    continue
                x0, y0, x1, y1, text = w[:5]
                text = str(text or "").strip()
                if len(text) < 1:
                    continue
                sx0 = float(x0) * zoom
                sy0 = float(y0) * zoom
                sx1 = float(x1) * zoom
                sy1 = float(y1) * zoom
                scaled.append(WordBox(text=text, bbox=(sx0, sy0, sx1, sy1), confidence=1.0))
            return scaled if len(scaled) >= 30 else None
        except Exception:
            return None

    def _digital_layer_matches_visual(self, image_rgb: np.ndarray, word_boxes: List[Any], max_samples: int = 40) -> bool:
        """
        Validate that the PDF text layer matches visible pixels.
        This prevents the "Rahul vs Rohit" bug caused by hidden/incorrect OCR layers.
        """
        if image_rgb is None or image_rgb.size == 0 or not word_boxes:
            return False
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
            h, w = gray.shape[:2]
            # Sample medium-length words (more reliable)
            candidates = [wb for wb in word_boxes if 3 <= len(getattr(wb, "text", "") or "") <= 20]
            if len(candidates) < 15:
                candidates = list(word_boxes)
            # Uniform sampling
            step = max(1, len(candidates) // max_samples)
            samples = candidates[::step][:max_samples]

            ink_scores = []
            for wb in samples:
                x0, y0, x1, y1 = [int(round(v)) for v in wb.bbox]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 <= x0 + 2 or y1 <= y0 + 2:
                    continue
                crop = gray[y0:y1, x0:x1]
                if crop.size < 50:
                    continue
                # Ink-ness: fraction of dark pixels
                thr = int(np.clip(np.median(crop) - 15, 100, 210))
                dark = np.count_nonzero(crop < thr)
                ink = dark / float(crop.size)
                ink_scores.append(float(ink))
            if len(ink_scores) < 10:
                return False
            med = float(np.median(ink_scores))
            # If median ink in word boxes is too low, those words are not actually visible.
            return med >= 0.015
        except Exception:
            return False
    
    def _load_image(self, path: str) -> Tuple[np.ndarray, int, int, Optional[List[Any]]]:
        """Load document image."""
        path = Path(path)
        digital_word_boxes = None

        if path.suffix.lower() == ".pdf":
            import fitz
            doc = fitz.open(str(path))
            page = doc[0]
            zoom = 300 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            # Extract digital text layer, but only keep it if it matches visible pixels.
            # This prevents using hidden/incorrect layers (the "Rahul vs Rohit" bug).
            maybe_words = self._extract_pdf_digital_words(page, zoom)
            if maybe_words and self._digital_layer_matches_visual(img, maybe_words):
                digital_word_boxes = maybe_words

            doc.close()
        else:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img, img.shape[1], img.shape[0], digital_word_boxes
    
    def _get_block_label(self, block: DetectedBlock) -> str:
        """Get display label for a block (shows block TYPE, not field name)."""
        # Priority 1: YOLO class name
        class_name = block.metadata.get("class_name", "")
        if class_name:
            return class_name.upper()
        
        # Priority 2: Block type enum
        btype = block.block_type.value if hasattr(block.block_type, 'value') else str(block.block_type)
        return btype.upper()
    
    def _has_ink_in_bbox(self, image: np.ndarray, bbox: Tuple[float, float, float, float], 
                         threshold: float = 0.018) -> bool:
        """
        Detect if a crop contains handwritten/printed ink (dark pixels).
        Used to filter empty fields — only show bounding boxes where there is actual content.
        threshold: minimum ratio of dark pixels (0.018 ≈ 1.8% catches thin handwriting).
        """
        if image is None or image.size == 0:
            return False
        try:
            x0, y0, x1, y1 = [int(round(v)) for v in bbox]
            h, w = image.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 <= x0 + 2 or y1 <= y0 + 2:
                return False
            crop = image[int(y0):int(y1), int(x0):int(x1)]
            if crop.size < 50:
                return False
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
            # Use adaptive threshold (ink = dark pixels below median)
            thr = int(np.clip(np.median(gray) - 20, 80, 200))
            dark = np.count_nonzero(gray < thr)
            ink_ratio = dark / float(crop.size)
            return ink_ratio >= threshold
        except Exception:
            return False

    def _normalize_vlm_candidate(self, text: str, field_type: str) -> str:
        """Normalize VLM output into field-specific canonical shapes."""
        t = re.sub(r"\s+", " ", str(text or "")).strip()
        if not t:
            return ""

        ft = (field_type or "").lower()
        # Normalize common OCR substitutions first.
        t = (
            t.replace("O", "0")
            .replace("o", "0")
            .replace("Q", "0")
            .replace("q", "0")
            .replace("U", "0")
            .replace("u", "0")
            .replace("I", "1")
            .replace("l", "1")
        )

        if ft == "date":
            m = re.search(r"(\d{1,2})[\/\-\.\s](\d{1,2})[\/\-\.\s](\d{2,4})", t)
            if m:
                mm, dd, yy = m.groups()
                return f"{int(mm):02d}/{int(dd):02d}/{yy}"
            digits = re.sub(r"\D", "", t)
            if len(digits) >= 6:
                mm, dd, yy = digits[:2], digits[2:4], digits[4:8]
                return f"{int(mm):02d}/{int(dd):02d}/{yy}"
            return t

        if ft == "phone":
            digits = re.sub(r"\D", "", t)
            if len(digits) >= 10:
                digits = digits[:10]
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            if len(digits) >= 9:
                return digits
            return t

        if ft == "npi":
            digits = re.sub(r"\D", "", t)
            return digits[:10] if len(digits) >= 10 else digits

        if ft == "zip":
            digits = re.sub(r"\D", "", t)
            if len(digits) >= 9:
                return f"{digits[:5]}-{digits[5:9]}"
            return digits[:5] if len(digits) >= 5 else digits

        if ft == "tax_id":
            digits = re.sub(r"\D", "", t)
            if len(digits) >= 9:
                return f"{digits[:2]}-{digits[2:9]}"
            return digits

        if ft == "money":
            m = re.search(r"\$?\s*\d[\d,]*(?:\.\d{1,2})?", t)
            return m.group().replace(" ", "") if m else t

        if ft == "icd10":
            # Keep only ICD-friendly characters.
            return re.sub(r"[^A-Za-z0-9\.]", "", t).upper()

        return t

    def _apply_targeted_vlm_rescue(
        self,
        image: np.ndarray,
        blocks: List[DetectedBlock],
        extracted_fields: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run a conservative VLM rescue pass only for clearly-bad fields.
        This mirrors "grounded correction" behavior: do not rewrite good fields.
        """
        if image is None or image.size == 0:
            return {"count": 0, "fields": []}
        if not getattr(self.config, "enable_vlm_ocr_fallback", True):
            return {"count": 0, "fields": []}

        try:
            from src.pipelines.validators import validate_field as typed_validate
        except Exception:
            typed_validate = None

        # Structured validators we can trust for rescue acceptance.
        validator_map = {
            "date": "date",
            "phone": "phone",
            "npi": "npi",
            "zip": "zip",
            "icd10": "icd10",
            "money": "money",
            "tax_id": "tax_id",
        }
        # Important free-text fields where we still allow rescue if text looks broken.
        priority_text_ids = {
            "header_top_right_notes",
            "9c_reserved_nucc_use",
            "11c_insurance_plan_name",
            "19_additional_claim_info",
        }
        invalid_ids = {e.get("field_id", "") for e in (validation or {}).get("errors", [])}

        corrected: List[str] = []
        h, w = image.shape[:2]
        max_rescues = 12

        for block in blocks:
            if len(corrected) >= max_rescues:
                break

            fid = str(block.id or "")
            meta = block.metadata or {}
            src = str(meta.get("source", ""))
            if src not in ("schema_zones", "ocr_zone_matching", "full_page_ocr"):
                continue

            field_type = str(meta.get("field_type", "") or "").lower()
            validator_name = validator_map.get(field_type)
            current = str(block.text or "").strip()
            bbox = meta.get("original_bbox") or block.bbox

            needs_rescue = False
            old_valid = True

            if validator_name and typed_validate is not None:
                old_valid, _ = typed_validate(validator_name, current or "")
                if (fid in invalid_ids) or (not old_valid):
                    needs_rescue = True
                elif (not current or float(block.confidence or 0.0) < 0.45) and self._has_ink_in_bbox(image, bbox, threshold=0.012):
                    needs_rescue = True
            elif fid in priority_text_ids:
                noisy = (not current) or self.ocr_agent._is_hallucination(current) or (len(current) < 6)
                if noisy and self._has_ink_in_bbox(image, bbox, threshold=0.012):
                    needs_rescue = True

            if not needs_rescue:
                continue

            try:
                x0, y0, x1, y1 = [int(round(v)) for v in bbox]
                pad = max(6, int(max(x1 - x0, y1 - y0) * 0.06))
                x0p, y0p = max(0, x0 - pad), max(0, y0 - pad)
                x1p, y1p = min(w, x1 + pad), min(h, y1 + pad)
                if x1p <= x0p + 2 or y1p <= y0p + 2:
                    continue
                crop = image[y0p:y1p, x0p:x1p]
                if crop.size == 0:
                    continue

                cand_raw, cand_conf = self.ocr_agent._vlm_ocr(crop)
                candidate = self._normalize_vlm_candidate(cand_raw, field_type)
                if not candidate:
                    continue

                accept = False
                if validator_name and typed_validate is not None:
                    new_valid, _ = typed_validate(validator_name, candidate)
                    # Allow near-valid phone rescue (9 digits) when OCR dropped one digit.
                    if validator_name == "phone" and not new_valid:
                        old_digits = re.sub(r"\D", "", current)
                        new_digits = re.sub(r"\D", "", candidate)
                        if len(new_digits) >= 9 and len(new_digits) > len(old_digits):
                            accept = True
                    if new_valid and (not old_valid or float(cand_conf or 0.0) >= float(block.confidence or 0.0) - 0.05):
                        accept = True
                elif fid in priority_text_ids:
                    if not self.ocr_agent._is_hallucination(candidate):
                        if (not current) or (len(candidate) >= len(current) + 2):
                            accept = True

                if not accept:
                    continue

                block.metadata["vlm_rescued"] = True
                block.metadata["vlm_previous_text"] = current
                block.metadata["vlm_candidate_text"] = candidate
                block.metadata["ocr_engine"] = "vlm_rescue"
                block.text = candidate
                block.confidence = max(float(block.confidence or 0.0), float(cand_conf or 0.72))
                extracted_fields[fid] = candidate
                corrected.append(fid)
            except Exception:
                continue

        return {"count": len(corrected), "fields": corrected}
    
    # CMS-1500 template label patterns - comprehensive list for stripping
    CMS1500_TEMPLATE_LABELS = {
        # Name field hints (OCR may garble these)
        r"\(?L[AE]ST\s*N[AE]ME[,\s]*F[I1]RST\s*N[AE]ME[,\s]*M[I1]DDL?E?\s*(INITIAL|NAME|BITA|NARE|INIT)?\)?",
        r"\(?LAST[,\s]*FIRST[,\s]*M\.?I\.?\)?",
        r"\(?F[I1]RST[,\s]*M[I1]DDLE[,\s]*LAST\)?",
        r"N[AE]ME\s*\([^)]*\)",  # "NAME (Last, First, Middle)"
        r"N[SU]?[AU]N?[CG]?[OD]?O?\s*NA?ME?",  # OCR garbled "NAME" -> "NSUngOo NAMe"
        
            # Field numbers and labels
        r"^\s*\d{1,2}\s*[a-z]?\s*\.?\s*",  # "1", "1.", "1a", "21a", etc.
        
        # Patient labels (with OCR error tolerance)
        r"P?A?T[I1]?EN?T'?S?\s*(NAME|BIRTH|ADDRESS|SEX|PHONE|RELATIONSHIP|ACCOUNT)",
        r"I?N?S?U?R?E?D'?S?\s*(NAME|I\.?D\.?\s*N|ADDRESS|DATE|POLICY|GROUP|SIGNATURE)",
            r"OTHER\s+INSURED'?S?\s*(NAME|POLICY)",
        
        # Common form labels
            r"INSURANCE\s+PLAN\s+NAME",
            r"EMPLOYER'?S?\s*NAME",
            r"REFERRING\s+PROVIDER",
        r"SIGNATURE\s+OF\s*(PHYSICIAN|PATIENT)?",
            r"BILLING\s+PROVIDER",
            r"SERVICE\s+FACILITY",
        r"DATE\s+OF\s+(CURRENT|BIRTH|SERVICE)",
            r"DIAGNOSIS\s+OR\s+NATURE",
        r"FEDERAL\s+TAX\s+I\.?D\.?",
        r"HEALTH\s+INSURANCE\s+CLAIM",
        r"TOTAL\s+CHARGE",
        r"AMOUNT\s+PAID",
        r"ACCEPT\s+ASSIGNMENT",
        r"OUTSIDE\s+LAB",
        r"ADDITIONAL\s+CLAIM\s+INFO",
        
        # Date format hints
        r"\(?\s*MM\s*[\/-]?\s*DD\s*[\/-]?\s*Y{2,4}\s*\)?",
            r"MM\s*DD\s*YY",
        r"\(?\s*MONTH\s*DAY\s*YEAR\s*\)?",
        
        # Address hints
        r"\(?\s*No\.[,\s]*Street\s*\)?",
        r"\(?\s*INCLUDE\s+AREA\s+CODE\s*\)?",
        r"\(?\s*AREA\s+CODE\s*\)?",
        
        # Checkbox hints
        r"\(?\s*YES\s*/?\s*NO\s*\)?",
        r"\[\s*\]\s*YES",
        r"\[\s*\]\s*NO",
        
        # Program hints
        r"OR\s+PROGR?A?M\s+NAME",
        r"\(?\s*For\s+Program.*?\)?",
        r"FECA\s+NUMBER",
        
        # Box labels (standalone)
        r"(?<![A-Z])CITY(?![A-Z])",
        r"(?<![A-Z])STATE(?![A-Z])",
            r"ZIP\s*CODE",
        r"(?<![A-Z])TELEPHONE(?![A-Z])",
        r"(?<![A-Z])SEX(?![A-Z])\s*[MF]?",
        r"(?<![A-Z])NPI(?![A-Z])",
        r"(?<![A-Z])DOB(?![A-Z])",
        
        # OCR garbage patterns (common misreads)
        r"N[SU]U?n?[gC]?[OD]?o?\s*",  # "NSUngOo" etc.
        r"L[AE]ET\s*N[E3]ME",  # "Laet Neme"
        r"M[I1]DD[E3]?\s*B[I1]TA",  # "Midde bita"
        r"F[I1]RST\s*N[AE]R[E3]",  # "First Nare"
    }
    
    def _clean_field_value(self, raw_text: str, field_label: str, field_id: str) -> str:
        """
        Strip pre-printed labels from OCR text to get just the filled-in value.
        
        CMS-1500 forms have pre-printed labels like "PATIENT'S NAME (Last, First, Middle)"
        that get captured by OCR along with the actual handwritten values.
        This method removes them using comprehensive pattern matching.
        """
        import re
        
        if not raw_text:
            return ""
        
        text = raw_text
        
        # Apply all CMS-1500 template label patterns
        for pattern in self.CMS1500_TEMPLATE_LABELS:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        
        # Also try to remove the specific field label
        if field_label:
            # Escape special regex chars and make flexible
            label_pattern = re.escape(field_label).replace(r"\ ", r"\s*")
            text = re.sub(label_pattern, " ", text, flags=re.IGNORECASE)
        
        # Remove field ID patterns (e.g., "2_patient_name" -> remove "patient name")
        if field_id:
            # Extract meaningful words from field_id
            id_words = re.findall(r"[a-zA-Z]+", field_id)
            for word in id_words:
                if len(word) >= 4:  # Skip short words like "id", "of"
                    text = re.sub(rf"\b{word}\b", " ", text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Remove leading/trailing punctuation and junk
        text = re.sub(r"^[\s\(\)\[\]\.,\-:;]+", "", text)
        text = re.sub(r"[\s\(\)\[\]\.,\-:;]+$", "", text)
        
        # Remove standalone single characters (OCR artifacts)
        text = re.sub(r"\s+[A-Za-z]\s+", " ", text)
        
        # Final cleanup
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    async def _match_ocr_to_zones(
        self, 
        word_boxes: List, 
        fields: List[dict], 
        width: int, 
        height: int,
        image: np.ndarray,
        word_level: bool = False,
        template_word_blacklist: Optional[set] = None,
        require_ink: bool = False,
        ink_threshold: float = 0.02,
        skip_label_cleaning: bool = False
    ) -> List[DetectedBlock]:
        """
        Match OCR word boxes to schema field zones.
        
        Strategy:
        1. For each schema field, calculate expected pixel coordinates from bbox_norm
        2. Find all OCR words that overlap with the field region
        3. Concatenate overlapping text as the field value
        
        Template word blacklist filters out pre-printed form labels AFTER OCR,
        preserving full image quality for handwriting recognition.
        """
        blocks = []

        # Precompute zones — direct bbox_norm to pixel conversion, no offsets
        zones = []
        for field_def in fields:
            field_id = field_def.get("id")
            if not field_id:
                continue
            bbox_norm = field_def.get("bbox_norm")
            if not bbox_norm or len(bbox_norm) != 4:
                continue

            x0 = int(bbox_norm[0] * width)
            y0 = int(bbox_norm[1] * height)
            x1 = int(bbox_norm[2] * width)
            y1 = int(bbox_norm[3] * height)

            # Expand matching region (ratio + px padding)
            pad_x = max(int((x1 - x0) * float(self.config.zone_padding_ratio)), int(self.config.zone_padding_px))
            pad_y = max(int((y1 - y0) * float(self.config.zone_padding_ratio)), int(self.config.zone_padding_px))
            x0_exp = max(0, x0 - pad_x)
            y0_exp = max(0, y0 - pad_y)
            x1_exp = min(width, x1 + pad_x)
            y1_exp = min(height, y1 + pad_y)

            field_type = field_def.get("field_type", "text")
            label = field_def.get("label", field_id)

            zones.append({
                "field_id": field_id,
                "label": label,
                "field_type": field_type,
                "bbox": (x0, y0, x1, y1),
                "bbox_exp": (x0_exp, y0_exp, x1_exp, y1_exp),
                "words": []
            })

        # Helper: intersection area
        def _inter_area(a, b) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            oy = max(0.0, min(ay1, by1) - max(ay0, by0))
            return ox * oy

        # Assign each OCR "word" box to the best zone (prevents duplicates when zones overlap).
        # Only safe when we truly have word-level boxes (e.g. digital PDF text layer).
        # OCR detectors often output *line-level* boxes that span multiple fields, and forcing unique
        # assignment on those will mis-route text.
        unique_ok = bool(self.config.enforce_unique_word_assignment and word_level)
        blacklist = template_word_blacklist or set()
        gray_for_ink = None
        if require_ink and image is not None and hasattr(image, "shape"):
            try:
                gray_for_ink = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            except Exception:
                gray_for_ink = None

        def _has_ink(bbox: Tuple[float, float, float, float]) -> bool:
            if gray_for_ink is None:
                return True
            try:
                gx0, gy0, gx1, gy1 = [int(round(v)) for v in bbox]
                h, w = gray_for_ink.shape[:2]
                gx0, gy0 = max(0, gx0), max(0, gy0)
                gx1, gy1 = min(w, gx1), min(h, gy1)
                if gx1 <= gx0 + 2 or gy1 <= gy0 + 2:
                    return False
                crop = gray_for_ink[gy0:gy1, gx0:gx1]
                if crop.size < 50:
                    return False
                thr = int(np.clip(np.median(crop) - 15, 90, 210))
                dark = np.count_nonzero(crop < thr)
                ink = dark / float(crop.size)
                return ink >= ink_threshold
            except Exception:
                return True
        def _is_template_word(text: str) -> bool:
            """Check if an OCR word is from the pre-printed template.
            Returns True if >50% of the tokens in the text are blacklisted."""
            if not text or not blacklist:
                return False
            t = text.strip().lower()
            # Exact match
            if t in blacklist:
                return True
            # Token-level: if majority of tokens are template words, skip
            tokens = t.replace(".", " ").replace(",", " ").replace("(", " ").replace(")", " ").split()
            if not tokens:
                return False
            bl_count = sum(1 for tok in tokens if tok in blacklist or (len(tok) >= 3 and tok.rstrip("'s") in blacklist))
            # Skip if >50% of tokens are template words
            return bl_count > len(tokens) * 0.5
        
        if unique_ok:
            for wb in word_boxes:
                if _is_template_word(wb.text):
                    continue
                if require_ink and not _has_ink(wb.bbox):
                    continue
                wx0, wy0, wx1, wy1 = wb.bbox
                word_bbox = (float(wx0), float(wy0), float(wx1), float(wy1))
                cx = (word_bbox[0] + word_bbox[2]) / 2.0
                cy = (word_bbox[1] + word_bbox[3]) / 2.0
                # Use a top-anchored point for assignment. OCR sometimes returns tall word boxes
                # that span multiple adjacent fields (e.g., patient name + address rows).
                # Using the center can mis-assign such boxes to the lower field.
                word_h = max(word_bbox[3] - word_bbox[1], 1.0)
                anchor_y = word_bbox[1] + min(2.0, word_h * 0.2)
                anchor_x = cx

                def _contains(b, x, y) -> bool:
                    x0, y0, x1, y1 = b
                    return (x0 <= x <= x1) and (y0 <= y <= y1)

                def _area(b) -> float:
                    x0, y0, x1, y1 = b
                    return max((x1 - x0) * (y1 - y0), 1.0)

                # Prefer original bbox containment (more precise), fall back to expanded bbox containment.
                candidates = []
                for i, z in enumerate(zones):
                    if _contains(z["bbox"], anchor_x, anchor_y):
                        candidates.append((i, _area(z["bbox"])))

                if not candidates:
                    for i, z in enumerate(zones):
                        if _contains(z["bbox_exp"], anchor_x, anchor_y):
                            candidates.append((i, _area(z["bbox_exp"])))

                if candidates:
                    # Choose smallest containing zone (most specific) to avoid bleeding into neighbors
                    candidates.sort(key=lambda t: t[1])
                    best_idx = candidates[0][0]
                    zones[best_idx]["words"].append({
                        "text": wb.text,
                        "confidence": wb.confidence,
                        "x": cx,
                        "y": cy
                    })
                # NOTE: In word-level mode we do NOT perform an “intersection fallback”.
                # If a word doesn't land in (bbox or bbox_exp), leaving it unassigned is safer
                # than contaminating a neighboring field.
        else:
            # Legacy behavior: each zone collects any overlapping words
            for z in zones:
                x0_exp, y0_exp, x1_exp, y1_exp = z["bbox_exp"]
                for wb in word_boxes:
                    if _is_template_word(wb.text):
                        continue
                    if require_ink and not _has_ink(wb.bbox):
                        continue
                    wx0, wy0, wx1, wy1 = wb.bbox
                    inter = _inter_area((float(wx0), float(wy0), float(wx1), float(wy1)), (x0_exp, y0_exp, x1_exp, y1_exp))
                    if inter > 0:
                        cx = (wx0 + wx1) / 2.0
                        cy = (wy0 + wy1) / 2.0
                        z["words"].append({"text": wb.text, "confidence": wb.confidence, "x": cx, "y": cy})

        # Build blocks from zones
        for z in zones:
            field_id = z["field_id"]
            label = z["label"]
            field_type = z["field_type"]
            x0, y0, x1, y1 = z["bbox"]

            # Sort words by position (top-to-bottom, left-to-right)
            words = sorted(z["words"], key=lambda w: (w["y"], w["x"]))

            raw_text = " ".join([w["text"] for w in words]).strip()
            avg_conf = sum(w["confidence"] for w in words) / len(words) if words else 0.0

            # For scan/OCR text, strip printed labels. For digital text, skip
            # (digital values are ground truth — cleaning corrupts them,
            #  e.g. "8340 Baltimore Aveune" → "altimore Aveune").
            if skip_label_cleaning:
                cleaned_text = raw_text
            else:
                cleaned_text = self._clean_field_value(raw_text, label, field_id)

            # Assign block type based on schema hints
            label_l = (label or "").lower()
            field_id_l = (field_id or "").lower()
            if field_type == "checkbox":
                block_type = BlockType.CHECKBOX
            elif field_type == "signature" or "signature" in field_id_l or "signature" in label_l:
                block_type = BlockType.SIGNATURE
            elif "table" in field_id_l or "table" in label_l:
                block_type = BlockType.TABLE
            elif "figure" in field_id_l or "figure" in label_l or "image" in label_l or "graphic" in label_l:
                block_type = BlockType.FIGURE
            elif "title" in field_id_l or "title" in label_l:
                block_type = BlockType.TITLE
            elif "header" in field_id_l or "header" in label_l:
                block_type = BlockType.HEADER
            elif "footer" in field_id_l or "footer" in label_l:
                block_type = BlockType.FOOTER
            elif ("page" in label_l and "number" in label_l) or "page_num" in field_id_l or "pagenum" in field_id_l:
                block_type = BlockType.PAGE_NUM
            elif "list" in field_id_l or "list" in label_l:
                block_type = BlockType.LIST
            else:
                block_type = BlockType.FORM_FIELD

            blocks.append(DetectedBlock(
                id=field_id,
                block_type=block_type,
                bbox=(x0, y0, x1, y1),
                text=cleaned_text,
                confidence=avg_conf,
                metadata={
                    "label": label,
                    "semantic_label": label,  # Use schema label as semantic label
                    "source": "ocr_zone_matching",
                    "field_type": field_type,
                    "skip_label_cleaning": bool(skip_label_cleaning),
                    "digital_text": bool(skip_label_cleaning),
                    "num_matched_words": len(words),
                    "raw_ocr_text": raw_text,
                    "zone_padding_px": int(self.config.zone_padding_px),
                    "zone_padding_ratio": float(self.config.zone_padding_ratio),
                    "unique_word_assignment": bool(unique_ok),
                }
            ))
        
        # Filter out empty fields (optional - keep for completeness)
        # blocks = [b for b in blocks if b.text]
        
        return blocks
    
    def _group_words_into_blocks(self, word_boxes: List, width: int, height: int) -> List[DetectedBlock]:
        """
        Group OCR word boxes into logical text blocks based on spatial proximity.
        This provides meaningful structure instead of one giant text block.
        
        Enhanced algorithm:
        1. Filter out garbage/low-confidence words
        2. Sort words by Y position (top to bottom)
        3. Group words into lines based on Y proximity
        4. Merge nearby lines into paragraphs for better structure
        """
        if not word_boxes:
            return []
        
        # STEP 1: Minimal filtering - only remove truly garbage
        min_conf = getattr(self.config, 'min_ocr_confidence', 0.20)
        filtered_words = []
        for word in word_boxes:
            # Skip very low confidence
            if word.confidence < min_conf:
                continue
            text = str(word.text or "").strip()
            # Skip empty
            if not text:
                continue
            # Skip truly garbage patterns only
            if self._is_garbage_text(text):
                continue
            filtered_words.append(word)
        
        if not filtered_words:
            # If filtering removed everything, return all words with text
            filtered_words = [w for w in word_boxes if str(w.text or "").strip()]
            if not filtered_words:
                return []
        
        print(f"[Pipeline] OCR: {len(word_boxes)} words -> {len(filtered_words)} after minimal filter")
        
        # Sort by Y position (top to bottom), then X (left to right)
        sorted_words = sorted(filtered_words, key=lambda w: (w.bbox[1], w.bbox[0]))
        
        # STEP 2: Group into lines
        lines = []
        current_line_words = []
        current_y = None
        line_threshold = max(15, height * 0.015)  # Adaptive threshold based on page height
        
        for word in sorted_words:
            word_y = (word.bbox[1] + word.bbox[3]) / 2  # Center Y
            
            if current_y is None:
                current_y = word_y
                current_line_words = [word]
            elif abs(word_y - current_y) <= line_threshold:
                # Same line
                current_line_words.append(word)
            else:
                # New line - save current line
                if current_line_words:
                    lines.append(current_line_words)
                current_line_words = [word]
                current_y = word_y
        
        # Don't forget last line
        if current_line_words:
            lines.append(current_line_words)
        
        # STEP 3: Convert lines to blocks with smart merging
        # For general forms, merge nearby lines into paragraph-like blocks
        blocks = []
        para_gap = max(30, height * 0.03)  # Gap threshold for paragraph grouping
        current_para_lines = []
        last_y = None
        
        for line_words in lines:
            line_y = min(w.bbox[1] for w in line_words)
            
            if last_y is None or (line_y - last_y) <= para_gap:
                # Same paragraph
                current_para_lines.append(line_words)
                last_y = max(w.bbox[3] for w in line_words)
            else:
                # New paragraph - save current
                if current_para_lines:
                    block = self._lines_to_block(current_para_lines, len(blocks))
                    if block.text and len(block.text.strip()) >= 3:
                        blocks.append(block)
                current_para_lines = [line_words]
                last_y = max(w.bbox[3] for w in line_words)
        
        # Save last paragraph
        if current_para_lines:
            block = self._lines_to_block(current_para_lines, len(blocks))
            if block.text and len(block.text.strip()) >= 3:
                blocks.append(block)
        
        print(f"[Pipeline] Created {len(blocks)} text blocks from {len(lines)} lines")
        return blocks
    
    def _is_garbage_text(self, text: str) -> bool:
        """Check if text is truly garbage (very minimal filtering)."""
        import re
        
        # Empty or whitespace
        if not text or not text.strip():
            return True
        
        text = text.strip()
        
        # Only filter TRULY garbage patterns - be very conservative
        # Single character
        if len(text) == 1 and not text.isalnum():
            return True
        
        # Only special characters (no letters/numbers at all)
        if re.match(r'^[\|\-\_\=\+\#\*\.\/\\]+$', text):
            return True
        
        # Repeated single character (like "----" or "====")
        if re.match(r'^(.)\1{5,}$', text):
            return True
        
        return False
    
    def _lines_to_block(self, lines: List[List], block_idx: int) -> DetectedBlock:
        """Convert multiple lines of words into a DetectedBlock (paragraph)."""
        # Flatten all words
        all_words = []
        for line in lines:
            all_words.extend(sorted(line, key=lambda w: w.bbox[0]))
        
        if not all_words:
            return DetectedBlock(
                id=f"block_{block_idx}",
                block_type=BlockType.TEXT,
                bbox=(0, 0, 1, 1),
                text="",
                confidence=0.0
            )
        
        # Compute bounding box
        x0 = min(w.bbox[0] for w in all_words)
        y0 = min(w.bbox[1] for w in all_words)
        x1 = max(w.bbox[2] for w in all_words)
        y1 = max(w.bbox[3] for w in all_words)
        
        # Build text line by line
        text_parts = []
        for line in lines:
            line_sorted = sorted(line, key=lambda w: w.bbox[0])
            line_text = " ".join(w.text for w in line_sorted)
            text_parts.append(line_text)
        text = "\n".join(text_parts)
        
        avg_conf = sum(w.confidence for w in all_words) / len(all_words)
        
        return DetectedBlock(
            id=f"block_{block_idx}",
            block_type=BlockType.TEXT,
            bbox=(x0, y0, x1, y1),
            text=text,
            confidence=avg_conf,
            metadata={
                "source": "full_page_ocr",
                "word_count": len(all_words),
                "line_count": len(lines),
                "ocr_engine": "paddleocr"
            }
        )
    
    def _words_to_block(self, words: List, block_idx: int) -> DetectedBlock:
        """Convert a list of word boxes into a DetectedBlock (single line)."""
        # Sort words left to right
        words = sorted(words, key=lambda w: w.bbox[0])
        
        # Compute bounding box
        x0 = min(w.bbox[0] for w in words)
        y0 = min(w.bbox[1] for w in words)
        x1 = max(w.bbox[2] for w in words)
        y1 = max(w.bbox[3] for w in words)
        
        # Concatenate text
        text = " ".join(w.text for w in words)
        avg_conf = sum(w.confidence for w in words) / len(words)
        
        return DetectedBlock(
            id=f"line_{block_idx}",
            block_type=BlockType.TEXT,
            bbox=(x0, y0, x1, y1),
            text=text,
            confidence=avg_conf,
            metadata={
                "source": "full_page_ocr",
                "word_count": len(words),
                "ocr_engine": "paddleocr"
            }
        )

    async def _load_schema_zones(self, image: np.ndarray, width: int, height: int, is_scan: bool = True) -> List[DetectedBlock]:
        """Load CMS-1500 schema zones with mode-aware field selection.
        
        For scanned forms (Lane C): uses composite address fields, skips digital-only sub-fields.
        For digital forms: uses individual address sub-fields, skips scan-only composites.
        """
        import json
        from pathlib import Path
        
        schema_path = Path(__file__).parent.parent.parent / "data" / "schemas" / "cms-1500.json"
        if not schema_path.exists():
            print(f"[Pipeline] Schema not found: {schema_path}")
            return []
        
        try:
            with open(schema_path) as f:
                schema = json.load(f)
            
            x_offset = getattr(self.config, "alignment_x_offset", -0.008)
            y_offset = getattr(self.config, "alignment_y_offset", 0.0)
            pad_ratio = getattr(self.config, "zone_padding_ratio", 0.08)
            pad_px = getattr(self.config, "zone_padding_px", 6)
            
            blocks = []
            skipped_mode = 0
            for field in schema.get("fields", []):
                bbox_norm = field.get("bbox_norm")
                if not bbox_norm or len(bbox_norm) != 4:
                    continue
                
                field_id = field.get("id", f"field_{len(blocks)}")
                field_type = field.get("field_type", "text")
                mode = field.get("mode", "both")
                
                # Mode filtering: skip fields not applicable to current lane
                if is_scan and mode == "digital":
                    skipped_mode += 1
                    continue
                if not is_scan and mode == "scan":
                    skipped_mode += 1
                    continue
                
                x0_norm = bbox_norm[0] + x_offset
                y0_norm = bbox_norm[1] + y_offset
                x1_norm = bbox_norm[2] + x_offset
                y1_norm = bbox_norm[3] + y_offset
                
                x0 = int(x0_norm * width)
                y0 = int(y0_norm * height)
                x1 = int(x1_norm * width)
                y1 = int(y1_norm * height)
                
                box_w = x1 - x0
                box_h = y1 - y0
                pad_x = max(int(box_w * pad_ratio), pad_px)
                pad_y = max(int(box_h * pad_ratio), pad_px)
                
                x0_padded = max(0, x0 - pad_x)
                y0_padded = max(0, y0 - pad_y)
                x1_padded = min(width, x1 + pad_x)
                y1_padded = min(height, y1 + pad_y)
                
                if field_type == "table" or "service_lines" in field_id.lower():
                    block_type = BlockType.TABLE
                elif field_type == "signature":
                    block_type = BlockType.SIGNATURE
                elif field_type == "checkbox":
                    block_type = BlockType.CHECKBOX
                else:
                    block_type = BlockType.FORM_FIELD
                
                blocks.append(DetectedBlock(
                    id=field_id,
                    block_type=block_type,
                    bbox=(x0_padded, y0_padded, x1_padded, y1_padded),
                    confidence=0.9,
                    metadata={
                        "label": block_type.value.upper(),
                        "field_name": field.get("label", ""),
                        "source": "schema_zones",
                        "field_type": field_type,
                        "class_name": block_type.value,
                        "original_bbox": (x0, y0, x1, y1),
                        "alignment_offset": (x_offset, y_offset),
                        "padding_applied": (pad_x, pad_y),
                        "mode": mode
                    }
                ))
            
            table_count = sum(1 for b in blocks if b.block_type == BlockType.TABLE)
            print(f"[Pipeline] Loaded {len(blocks)} zones (skipped {skipped_mode} mode-filtered, tables={table_count})")
            return blocks
        except Exception as e:
            print(f"[Pipeline] Failed to load schema: {e}")
            return []
    
    def _get_schema_field_order(self, form_type: FormType) -> List[str]:
        """Return field IDs in schema order for serial output (Reducto-style reading order)."""
        schema_paths = {
            FormType.CMS1500: Path(__file__).parent.parent.parent / "data" / "schemas" / "cms-1500.json",
            FormType.UB04: Path(__file__).parent.parent.parent / "data" / "schemas" / "ub-04.json",
        }
        path = schema_paths.get(form_type)
        if not path or not path.exists():
            return []
        try:
            with open(path) as f:
                schema = json.load(f)
            return [f.get("id") for f in schema.get("fields", []) if f.get("id")]
        except Exception:
            return []
    
    def _parse_address_block(self, text: str, prefix: str) -> Dict[str, str]:
        """Parse a composite address block OCR result into sub-fields.
        
        CMS-1500 address blocks have a known layout:
        Line 1: Street address (e.g., "825 Lynn Ogden Lane")
        Line 2: City  State  (e.g., "Beaumont TX")
        Line 3: ZIP  Phone  (e.g., "77701 (409) 853-3240")
        
        Returns dict with keys like {prefix}_city, {prefix}_state, etc.
        """
        import re
        result = {}
        if not text:
            return result
        
        lines = [l.strip() for l in text.replace('\n', ' | ').split('|') if l.strip()]
        if not lines:
            lines = [text.strip()]
        
        all_text = text.strip()
        
        # US state abbreviations
        states = {
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
            "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
            "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
            "TX","UT","VT","VA","WA","WV","WI","WY","DC"
        }
        
        # Extract phone number pattern (strict first, then OCR-tolerant: 0/O/Q/U, 1/I/l)
        phone_match = re.search(r'\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}', all_text)
        if not phone_match:
            # OCR-tolerant: e.g. 409-853-32U (U->0); (?<!\d) avoids matching from middle of ZIP (e.g. 77701)
            phone_match = re.search(r'(?<!\d)\(?[\dOoQqUu]{3}\)?[\s\-]?[\dOoIlUu]{3}[\s\-]?[\dOoIlUu]{3,4}', all_text)
        if phone_match:
            raw_phone = phone_match.group().strip()
            # Normalize OCR substitutions for valid output
            norm = raw_phone.replace('O', '0').replace('o', '0').replace('Q', '0').replace('q', '0')
            norm = norm.replace('U', '0').replace('u', '0').replace('I', '1').replace('l', '1')
            result[f"{prefix}_phone"] = norm
            all_text = all_text[:phone_match.start()] + all_text[phone_match.end():]
        
        # Extract ZIP code (5 digits, optionally followed by -4 digits)
        zip_match = re.search(r'\b(\d{5})(?:-\d{4})?\b', all_text)
        if zip_match:
            result[f"{prefix}_zip"] = zip_match.group().strip()
            all_text = all_text[:zip_match.start()] + all_text[zip_match.end():]
        
        # Extract state (2-letter abbreviation)
        for token in all_text.split():
            if token.upper().rstrip('.,') in states:
                result[f"{prefix}_state"] = token.upper().rstrip('.,')
                all_text = all_text.replace(token, '', 1)
                break
        
        # Remaining text: try to split into address and city
        remaining = re.sub(r'\s+', ' ', all_text).strip().rstrip(',. ')
        
        if remaining:
            # Heuristic: if there's a comma, split at last comma
            if ',' in remaining:
                parts = remaining.rsplit(',', 1)
                result[f"{prefix}_address"] = parts[0].strip()
                if len(parts) > 1 and parts[1].strip():
                    result[f"{prefix}_city"] = parts[1].strip()
            else:
                # First line is usually the street address
                words = remaining.split()
                # If we have many words, assume first part is address and last 1-2 words are city
                if len(words) > 3:
                    # Look for common address suffixes to find the split point
                    addr_suffixes = {"lane", "road", "rd", "st", "ave", "drive", "dr", "blvd", "way", "ct", "pl", "ln"}
                    split_idx = len(words)
                    for i, w in enumerate(words):
                        if w.lower().rstrip('.,') in addr_suffixes and i > 0:
                            split_idx = i + 1
                            break
                    result[f"{prefix}_address"] = ' '.join(words[:split_idx]).strip()
                    city_part = ' '.join(words[split_idx:]).strip()
                    if city_part:
                        result[f"{prefix}_city"] = city_part
                else:
                    result[f"{prefix}_address"] = remaining
        
        return result
    
    async def _extract_service_lines_ocr(self, image: np.ndarray, table_bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """Extract Box 24 service lines using cell-by-cell OCR.
        
        CMS-1500 Box 24 has exactly 6 data rows with a fixed column layout.
        Instead of VLM (which hallucinates), we divide the table into cells
        and run PaddleOCR + TrOCR on each cell individually.
        """
        from src.processing.preprocessing import remove_red_template_text
        
        h, w = image.shape[:2]
        tx0, ty0, tx1, ty1 = [int(v) for v in table_bbox]
        tx0, ty0 = max(0, tx0), max(0, ty0)
        tx1, ty1 = min(w, tx1), min(h, ty1)
        
        table_crop = image[ty0:ty1, tx0:tx1]
        if table_crop.size == 0:
            return {"type": "table", "rows": [], "extraction_method": "cell_ocr"}
        
        # Remove red template lines from table crop
        try:
            clean_crop = remove_red_template_text(table_crop)
        except Exception:
            clean_crop = table_crop
        
        th, tw = clean_crop.shape[:2]
        
        # CMS-1500 Box 24 column layout (relative to table width)
        columns = [
            ("date_from",    0.00, 0.13),
            ("date_to",      0.13, 0.21),
            ("place",        0.21, 0.26),
            ("cpt_code",     0.30, 0.48),
            ("modifier",     0.48, 0.54),
            ("dx_pointer",   0.54, 0.59),
            ("charges",      0.59, 0.73),
            ("days_units",   0.73, 0.79),
            ("provider_id",  0.80, 1.00),
        ]
        
        # Skip header row (~18% of table height), then 6 equal data rows
        header_frac = 0.18
        data_start = int(th * header_frac)
        data_height = th - data_start
        row_height = data_height // 6
        
        rows = []
        from src.ocr.paddle_ocr import PaddleOCRWrapper
        paddle = PaddleOCRWrapper()
        
        for row_idx in range(6):
            ry0 = data_start + row_idx * row_height
            ry1 = min(th, ry0 + row_height)
            
            if ry1 - ry0 < 5:
                continue
            
            row_data = {"line_number": row_idx + 1}
            has_content = False
            
            for col_name, cx0_frac, cx1_frac in columns:
                cx0 = int(tw * cx0_frac)
                cx1 = int(tw * cx1_frac)
                
                # Crop the cell with a small vertical padding
                cell_pad_y = max(2, int(row_height * 0.05))
                cell_y0 = max(0, ry0 - cell_pad_y)
                cell_y1 = min(th, ry1 + cell_pad_y)
                cell_crop = clean_crop[cell_y0:cell_y1, cx0:cx1]
                
                if cell_crop.size == 0:
                    row_data[col_name] = ""
                    continue
                
                # Run PaddleOCR on the cell
                try:
                    word_boxes = paddle.extract_text(cell_crop)
                    if word_boxes:
                        cell_text = " ".join(wb.text for wb in word_boxes if wb.confidence >= 0.2)
                        cell_conf = max(wb.confidence for wb in word_boxes)
                    else:
                        cell_text = ""
                        cell_conf = 0.0
                except Exception:
                    cell_text = ""
                    cell_conf = 0.0
                
                # TrOCR fallback for low-confidence cells
                if (not cell_text or cell_conf < 0.3) and getattr(self.config, 'enable_trocr', False):
                    try:
                        trocr_text, trocr_conf = self.ocr_agent._trocr_ocr(cell_crop)
                        if trocr_text and len(trocr_text.strip()) > 0:
                            if not self.ocr_agent._is_hallucination(trocr_text.strip()):
                                if trocr_conf > cell_conf or not cell_text:
                                    cell_text = trocr_text.strip()
                    except Exception:
                        pass
                
                cell_text = cell_text.strip()
                if cell_text:
                    has_content = True
                row_data[col_name] = cell_text
            
            if has_content:
                rows.append(row_data)
        
        # Build summary text for the block
        summary_parts = []
        for row in rows:
            parts = []
            if row.get("date_from"):
                parts.append(row["date_from"])
            if row.get("cpt_code"):
                parts.append(row["cpt_code"])
            if row.get("charges"):
                parts.append(f"${row['charges']}")
            if row.get("provider_id"):
                parts.append(row["provider_id"])
            if parts:
                summary_parts.append(" | ".join(parts))
        
        return {
            "type": "table",
            "rows": rows,
            "summary": "\n".join(summary_parts),
            "extraction_method": "cell_ocr",
            "total_rows": len(rows)
        }
    
    def _to_reducto_format(self, result: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
        """Convert pipeline result to Reducto-like JSON format with full enrichment."""
        import uuid
        
        field_details = result.get("field_details", [])
        page_w = float(width) if width > 0 else 1.0
        page_h = float(height) if height > 0 else 1.0
        
        # 1. Build Blocks (fine-grained regions)
        blocks = []
        all_text_lines = []  # Collect all text for content
        
        for field in field_details:
            x0, y0, x1, y1 = field.get("bbox", [0, 0, 0, 0])
            fid = str(field.get("id", "")).lower()
            text = str(field.get("value") or field.get("text") or "").strip()
            
            if not text:
                continue
                
            # Collect text for content
            all_text_lines.append(text)
            
            # Reducto Type Mapping
            block_type = "Text"
            source = field.get("metadata", {}).get("source", "")
            if "table" in fid or field.get("type") == "table":
                block_type = "Table"
            elif "figure" in fid or field.get("type") == "figure":
                block_type = "Figure"
            elif "title" in fid or field.get("type") == "title":
                block_type = "Title"
            elif "header" in fid or field.get("type") == "header":
                block_type = "Header"
            elif source == "full_page_ocr":
                block_type = "Text"
            elif ":" in text or "=" in text:
                block_type = "Key Value"
            
            conf_score = field.get("confidence", 0.0)
            conf_str = "high" if conf_score > 0.85 else ("medium" if conf_score > 0.6 else "low")
            
            blocks.append({
                "type": block_type,
                "bbox": {
                    "left": x0 / page_w,
                    "top": y0 / page_h,
                    "width": (x1 - x0) / page_w,
                    "height": (y1 - y0) / page_h,
                    "page": 1,
                    "original_page": 1
                },
                "content": text,
                "image_url": None,
                "chart_data": None,
                "confidence": conf_str,
                "granular_confidence": {
                    "extract_confidence": None,
                    "parse_confidence": conf_score
                }
            })

        # 2. Build the Main Content (Reducto returns the full OCR text organized by reading order)
        form_name = result.get("form_type", "Document").upper().replace("-", " ")
        
        # For full-page OCR, just return the text in reading order
        if all_text_lines:
            full_content = f"# {form_name}\n\n" + "\n".join(all_text_lines)
        else:
            full_content = f"# {form_name}\n\n(No text extracted)"

        # 3. Assemble Reducto-like structure
        return {
            "job_id": str(uuid.uuid4()),
            "duration": result.get("processing_time", 0.0),
            "pdf_url": None,
            "studio_link": None,
            "usage": {"num_pages": 1, "credits": 4},
            "result": {
                "type": "full",
                "chunks": [
                    {
                        "content": full_content,
                        "embed": full_content,
                        "enriched": full_content,
                        "enrichment_success": True,
                        "blocks": blocks
                    }
                ],
                "ocr": None,
                "custom": None
            }
        }
    
    async def process(self, path: str) -> Dict[str, Any]:
        """
        Process a document through the full pipeline.
        
        Returns comprehensive extraction result.
        """
        start_time = time.time()
        print(f"[Pipeline] Processing {path}")
        
        # Load image (+ optional digital text layer boxes)
        image, width, height, digital_words = self._load_image(path)
        digital_words_present = bool(digital_words) if digital_words is not None else False
        pre_meta = {}
        
        # Step 1: Form Identification
        if self.config.enable_form_detection and not self.config.form_type_override:
            form_id = await self.form_id_agent.process(image)
            
            # Fallback: check filename if detection failed
            if form_id.form_type == FormType.GENERIC:
                fname = Path(path).name.lower()
                if "cms1500" in fname or "cms-1500" in fname:
                    print(f"[Pipeline] Filename hint override: {fname} -> CMS-1500")
                    form_id.form_type = FormType.CMS1500
                    form_id.confidence = 0.8
                elif "ub04" in fname or "ub-04" in fname or "ub_04" in fname:
                    print(f"[Pipeline] Filename hint override: {fname} -> UB-04")
                    form_id.form_type = FormType.UB04
                    form_id.confidence = 0.8
        else:
            form_id = FormIdentification(
                form_type=self.config.form_type_override or FormType.GENERIC,
                confidence=1.0,
                detection_method="override"
            )
        
        print(f"[Pipeline] Detected Form Type: {form_id.form_type}")

        # ══════════════════════════════════════════════════════════════
        # LANE A: AcroForm widget extraction (fillable PDFs)
        # This is the HIGHEST accuracy path for machine-filled CMS-1500.
        # If widgets provide enough data, we skip OCR entirely.
        # ══════════════════════════════════════════════════════════════
        if form_id.form_type in (FormType.CMS1500, FormType.UB04):
            widget_info = self._extract_widgets(path)
            if widget_info is not None:
                extracted_fields, blocks = self._map_widgets_to_schema(widget_info, form_id.form_type)

                # Validate: if we got a good set of fields, return immediately
                filled_count = sum(1 for v in extracted_fields.values() if v and len(str(v).strip()) > 0)
                print(f"[Lane A] Widget extraction produced {filled_count} non-empty fields")

                min_fields = 10 if form_id.form_type == FormType.CMS1500 else 3
                if filled_count >= min_fields:  # Good enough — skip OCR entirely
                    processing_time = time.time() - start_time

                    # Business mapping
                    from src.pipelines.schemas import map_to_business_schema, merge_business_with_ocr
                    temp_result = {
                        "extracted_fields": extracted_fields,
                        "field_details": [
                            {"id": b.id, "bbox": list(b.bbox), "confidence": b.confidence, "metadata": b.metadata}
                            for b in blocks
                        ],
                        "page_width": width, "page_height": height,
                    }
                    if form_id.form_type == FormType.CMS1500:
                        business_result = map_to_business_schema(temp_result, "cms-1500")
                    elif form_id.form_type == FormType.UB04:
                        business_result = map_to_business_schema(temp_result, "ub-04")
                    else:
                        business_result = {
                            "business_fields": {},
                            "business_field_details": [],
                            "business_coverage": 0.0,
                        }

                    # Validation
                    validation = await self.validation_agent.process(blocks, extracted_fields)

                    final_result = {
                        "success": True,
                        "form_type": form_id.form_type.value,
                        "form_confidence": form_id.confidence,
                        "form_version": form_id.version,
                        "alignment_quality": 1.0,
                        "extracted_fields": extracted_fields,
                        "field_details": [
                            {
                                "id": b.id,
                                "label": self._get_block_label(b),
                                "type": b.block_type.value if hasattr(b.block_type, 'value') else str(b.block_type),
                                "bbox": list(b.bbox),
                                "value": b.text or "",
                                "text": b.text or "",
                                "confidence": b.confidence,
                                "detected_by": "acroform_widget",
                                "metadata": b.metadata,
                            }
                            for b in blocks
                        ],
                        "page_width": width, "page_height": height,
                        "processing_time": processing_time,
                        "validation": validation,
                        "ocr_blocks": [
                            {"text": b.text, "bbox": list(b.bbox), "confidence": b.confidence}
                            for b in blocks if b.text
                        ],
                        "extraction_method": "lane_a_acroform_widgets",
                        "config": {
                            "layout_model": "none (widget extraction)",
                            "enable_trocr": False,
                            "enable_slm": False,
                            "enable_vlm": False,
                        },
                        "debug": {
                            "lane": "A",
                            "widgets_total": widget_info.get("total_widgets", 0),
                            "widgets_filled": widget_info.get("filled", 0),
                            "digital_text_used": False,
                            "alignment_used": False,
                        },
                    }
                    final_merged = merge_business_with_ocr(final_result, business_result)
                    final_merged["reducto_format"] = self._to_reducto_format(final_merged, width, height)
                    print(f"[Pipeline] ✅ Lane A complete: {filled_count} fields in {processing_time:.1f}s (no OCR needed)")
                    return final_merged
                else:
                    print(f"[Pipeline] Lane A yielded only {filled_count} fields — falling through to Lane B/C")

        # Decide whether to use digital text layer:
        # - Only available for PDFs
        # - Only trust it when it contains *real filled values*, not just the pre-printed template layer.
        use_digital_text = False
        
        # For non-CMS1500 forms (UB-04, generic, etc.):
        # If _digital_layer_matches_visual passed (meaning digital_words_present is True),
        # trust it and skip deskew - digital PDFs are already straight.
        if digital_words_present and form_id.form_type != FormType.CMS1500:
            use_digital_text = True
            print(f"[Pipeline] ✅ Digital text layer present for {form_id.form_type} - using it (skip deskew).")
        
        # For CMS-1500: do additional validation to catch template-only layers
        elif digital_words_present and form_id.form_type == FormType.CMS1500:
            try:
                schema_path = Config.PROJECT_ROOT / "data" / "schemas" / "cms-1500.json"
                if schema_path.exists():
                    import json
                    with open(schema_path) as f:
                        schema = json.load(f)
                    fields_list = schema.get("fields", [])
                    candidate_blocks = await self._match_ocr_to_zones(
                        digital_words or [], fields_list, width, height, image, word_level=True, skip_label_cleaning=True
                    )
                    by_id = {b.id: ((b.metadata or {}).get("raw_ocr_text") or b.text or "").strip() for b in candidate_blocks}
                    insured_id = by_id.get("1a_insured_id", "")
                    patient_name = by_id.get("2_patient_name", "")
                    patient_dob = by_id.get("3_patient_dob", "")

                    import re
                    def looks_like_member_id(s: str) -> bool:
                        s = (s or "").strip()
                        return bool(re.search(r"[A-Za-z]{0,4}\d{4,}", s)) and len(s) >= 5
                    def looks_like_name(s: str) -> bool:
                        s = (s or "").strip()
                        return bool(re.search(r"[A-Za-z]{2,}", s)) and ("," in s or " " in s)
                    def looks_like_dob(s: str) -> bool:
                        s = (s or "").strip()
                        nums = re.findall(r"\d{2,4}", s)
                        return len(nums) >= 3

                    score = sum([
                        1 if looks_like_member_id(insured_id) else 0,
                        1 if looks_like_name(patient_name) else 0,
                        1 if looks_like_dob(patient_dob) else 0,
                    ])
                    print(f"[Pipeline] Digital QA: insured_id='{insured_id[:40]}', patient_name='{patient_name[:40]}', dob='{patient_dob[:40]}', score={score}/3")
                    # Require at least 2/3 anchor fields to look sane; otherwise treat as scan.
                    use_digital_text = score >= 2
                    if use_digital_text:
                        print("[Pipeline] ✅ Digital text layer validated - using it (skip preprocess/alignment).")
                    else:
                        print("[Pipeline] ⚠️ Digital text layer present but looks like template-only; using scan OCR path.")
            except Exception as e:
                print(f"[Pipeline] Digital layer validation error: {e}")
                use_digital_text = False

        # Preprocess ONLY for scan/camera path.
        # For CMS-1500 scans: SKIP all preprocessing (no deskew, no contrast).
        # The alignment agent handles rotation via homography, and aggressive
        # contrast/CLAHE degrades handwritten text quality.
        # For general forms: light preprocessing only (deskew, no heavy contrast).
        if not use_digital_text:
            from src.processing.preprocessing import preprocess_image
            will_align = bool(self.config.enable_alignment and form_id.form_type == FormType.CMS1500)
            if will_align:
                # CMS-1500 scan: SKIP preprocessing entirely — alignment handles it
                pre_meta = {"skipped": True, "reason": "cms1500_will_align"}
                print("[Pipeline] Skipping preprocessing for CMS-1500 scan (alignment will handle rotation)")
            else:
                image, pre_meta = preprocess_image(
                    image,
                    deskew=True,
                    denoise=False,
                    doc_type="generic"
                )
            height, width = image.shape[:2]
        
        # Step 2: Template Alignment (scan path only)
        aligned_image = image
        alignment_result = None
        aligned_preview_path = None
        scan_ocr_source = None
        alignment_quality_override = None
        prod_aligned_shape = None
        if (not use_digital_text) and self.config.enable_alignment and form_id.form_type == FormType.CMS1500:
            try:
                alignment_result = await self.alignment_agent.process(image, form_id.form_type)
                if alignment_result.success and alignment_result.aligned_image is not None:
                    aligned_image = alignment_result.aligned_image
                    height, width = aligned_image.shape[:2]
                    print(f"[Pipeline] Alignment succeeded, quality: {alignment_result.alignment_quality:.2f}")
                else:
                    print("[Pipeline] Alignment failed; will NOT use schema zones on raw scan.")
            except Exception as e:
                print(f"[Pipeline] Alignment exception: {e}")

        # Write an aligned preview image for UI overlays (optional but very useful for debugging)
        try:
            import uuid
            cache_dir = Config.PROJECT_ROOT / "cache" / "previews"
            cache_dir.mkdir(parents=True, exist_ok=True)
            aligned_preview_path = cache_dir / f"aligned_{uuid.uuid4().hex}.png"
            # cv2.imwrite expects BGR; our pipeline images are RGB
            if aligned_image is not None and aligned_image.ndim == 3 and aligned_image.shape[2] == 3:
                bgr = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aligned_preview_path), bgr)
            elif aligned_image is not None:
                cv2.imwrite(str(aligned_preview_path), aligned_image)
        except Exception:
            aligned_preview_path = None
        
        # Step 3: Layout Detection
        # CMS-1500 scanned forms use schema zones below; general forms use layout agent
        blocks = []
        
        if form_id.form_type == FormType.CMS1500 and not blocks:
            schema_path = Config.PROJECT_ROOT / "data" / "schemas" / "cms-1500.json"
            if use_digital_text and schema_path.exists():
                # DIGITAL CMS-1500: zone match using validated digital words (best quality)
                import json
                with open(schema_path) as f:
                    schema = json.load(f)
                fields_list = schema.get("fields", [])
                blocks = await self._match_ocr_to_zones(digital_words or [], fields_list, width, height, aligned_image, word_level=True, skip_label_cleaning=True)
                print(f"[Pipeline] CMS-1500 digital (Lane B): matched {len(blocks)} fields — label cleaning SKIPPED")

                # REALITY CHECK: If too few zones have meaningful text, template is wrong
                filled_zones = sum(1 for b in blocks if (b.text or '').strip() and len((b.text or '').strip()) >= 3)
                if filled_zones < 10:  # < ~20% of 48 zones
                    print(f"[Pipeline] ⚠️ Template mismatch (digital): only {filled_zones}/48 zones have text - falling back to layout model")
                    blocks = []  # Force fallback
                    use_digital_text = False  # Disable digital path
            else:
                # SCANNED CMS-1500: Per-field crop OCR
                if alignment_result is not None and alignment_result.success:
                    blocks = await self._load_schema_zones(aligned_image, width, height)
                    for b in blocks:
                        if b.metadata is None:
                            b.metadata = {}
                        b.metadata["form_type"] = "cms-1500"
                    scan_ocr_source = aligned_image
                    print(f"[Pipeline] CMS-1500 scan (Lane C): loaded {len(blocks)} schema zones for per-field crop OCR")
                else:
                    # Alignment failed — still try per-field crop OCR on raw image
                    blocks = await self._load_schema_zones(aligned_image, width, height)
                    for b in blocks:
                        if b.metadata is None:
                            b.metadata = {}
                        b.metadata["form_type"] = "cms-1500"
                    scan_ocr_source = aligned_image
                    print(f"[Pipeline] CMS-1500 scan (no alignment): loaded {len(blocks)} schema zones for per-field crop OCR")
        elif form_id.form_type == FormType.UB04 and not blocks and use_digital_text:
            schema_path = Config.PROJECT_ROOT / "data" / "schemas" / "ub-04.json"
            if schema_path.exists():
                import json
                with open(schema_path) as f:
                    schema = json.load(f)
                fields_list = schema.get("fields", [])
                blocks = await self._match_ocr_to_zones(
                    digital_words or [],
                    fields_list,
                    width,
                    height,
                    aligned_image,
                    word_level=True,
                    skip_label_cleaning=True
                )
                filled_zones = sum(1 for b in blocks if (b.text or '').strip() and len((b.text or '').strip()) >= 2)
                print(f"[Pipeline] UB-04 digital (Lane B): matched {len(blocks)} fields, filled={filled_zones}")
                if filled_zones < 2:
                    print("[Pipeline] ⚠️ UB-04 digital match too sparse; falling back to layout model")
                    blocks = []
        
        # GENERAL FORM PATH (only if no blocks yet)
        if not blocks:
            # Try Detectron2/PaddleDetection. Drop giant blocks; fallback to OCR line grouping.
            print("[Pipeline] Running layout detection for general form...")
            try:
                blocks = await self.layout_agent.process(aligned_image, form_id.form_type)
                print(f"[Pipeline] Detected {len(blocks)} blocks")
                
                # Drop blocks that are basically "the whole page"
                page_area = float(width * height) if width > 0 and height > 0 else 1.0
                filtered = []
                for b in blocks:
                    x0, y0, x1, y1 = b.bbox
                    area = float(max(0.0, x1 - x0) * max(0.0, y1 - y0))
                    if area / page_area > 0.85:
                        continue
                    filtered.append(b)
                if len(filtered) != len(blocks):
                    print(f"[Pipeline] Dropped {len(blocks) - len(filtered)} giant blocks")
                blocks = filtered

                if len(blocks) < 3:
                    print("[Pipeline] ⚠️ Layout too coarse (<3 blocks), using OCR line grouping fallback")
                    from src.ocr.paddle_ocr import PaddleOCRWrapper
                    paddle = PaddleOCRWrapper()
                    word_boxes = paddle.extract_text(aligned_image)
                    if word_boxes:
                        blocks = self._group_words_into_blocks(word_boxes, width, height)
                        print(f"[Pipeline] Full-page OCR: {len(word_boxes)} words -> {len(blocks)} blocks")
            except Exception as e:
                print(f"[Pipeline] Layout detection failed: {e}")
                # Fallback to OCR
                from src.ocr.paddle_ocr import PaddleOCRWrapper
                paddle = PaddleOCRWrapper()
                word_boxes = paddle.extract_text(aligned_image)
                if word_boxes:
                    blocks = self._group_words_into_blocks(word_boxes, width, height)
        
        # If still no blocks, use full-page OCR with intelligent word grouping
        if not blocks:
            print("[Pipeline] No layout blocks detected, using full-page OCR with word grouping")
            from src.ocr.paddle_ocr import PaddleOCRWrapper
            paddle = PaddleOCRWrapper()
            word_boxes = paddle.extract_text(aligned_image)
            
            if word_boxes:
                # Use word grouping instead of one giant block
                blocks = self._group_words_into_blocks(word_boxes, width, height)
                if not blocks:
                    # Final fallback: concatenate all text if grouping fails
                    all_text = " ".join([wb.text for wb in word_boxes if wb.confidence >= self.config.min_ocr_confidence])
                    if all_text.strip():
                        blocks = [DetectedBlock(
                            id="full_page",
                            block_type=BlockType.TEXT,
                            bbox=(0, 0, width, height),
                            text=all_text,
                            confidence=sum(wb.confidence for wb in word_boxes) / len(word_boxes)
                        )]
        
        # Step 4: OCR
        # Attach form_type to block metadata for OCR routing (template-diff, etc.)
        for b in blocks:
            if b.metadata is None:
                b.metadata = {}
            b.metadata.setdefault("form_type", form_id.form_type.value if hasattr(form_id.form_type, "value") else str(form_id.form_type))
        
        # Per-field OCR uses the raw aligned image (no red removal).
        # For CMS-1500 scans, zone-matched blocks already have text from
        # full-page OCR and skip per-field re-OCR (line 1442).
        # This mainly affects checkboxes and signatures.
        ocr_image = aligned_image

        blocks = await self.ocr_agent.process_blocks(ocr_image, blocks)

        # Post-clean CMS-1500 zone OCR: strip printed labels from OCR text.
        # ONLY for scan/OCR path (not use_digital_text). Digital text layer values
        # are already clean ground truth. _clean_field_value corrupts them:
        #   "8340 Baltimore Aveune" → regex strips "8340 B" → "altimore Aveune"
        if form_id.form_type == FormType.CMS1500 and not use_digital_text:
            for b in blocks:
                try:
                    src = (b.metadata or {}).get("source")
                    if src not in ("schema_zones", "ocr_zone_matching"):
                        continue
                    if not b.text:
                        continue
                    field_label = (b.metadata or {}).get("field_name") or (b.metadata or {}).get("label") or b.id
                    cleaned = self._clean_field_value(str(b.text), str(field_label), str(b.id))
                    if cleaned and len(cleaned) <= len(str(b.text)) + 2:
                        b.metadata["raw_ocr_text"] = b.metadata.get("raw_ocr_text") or b.text
                        b.text = cleaned
                        b.metadata["post_cleaned"] = True
                except Exception:
                    continue
        
        # Step 5: SLM/VLM Labeling
        # For CMS-1500: 
        #   - Skip SLM for text/form_field blocks (already schema-identified)
        #   - ALWAYS process table blocks (service lines) via VLM for structured extraction
        #   - Process figure blocks via VLM if enabled
        # For general forms: full SLM labeling on all blocks.
        if form_id.form_type == FormType.CMS1500:
            # Box 24 service lines: use cell-by-cell OCR (not VLM which hallucinates)
            for block in blocks:
                if block.block_type == BlockType.TABLE and "service_lines" in block.id:
                    print(f"[Pipeline] Extracting service lines via cell-by-cell OCR: {block.id}")
                    table_data = await self._extract_service_lines_ocr(aligned_image, block.bbox)
                    block.metadata["table_data"] = table_data
                    if table_data.get("summary"):
                        block.text = table_data["summary"]
                        block.metadata["cell_ocr_extracted"] = True
                    print(f"[Pipeline] Service lines: {table_data.get('total_rows', 0)} rows extracted")
                elif block.block_type == BlockType.FIGURE and self.config.enable_vlm_figures:
                    figure_data = await self.labeling_agent.process_figure(aligned_image, block)
                    block.metadata["figure_data"] = figure_data
        elif self.config.enable_slm_labeling:
            blocks = await self.labeling_agent.process(aligned_image, blocks)
        
        # Step 6: Build extracted data
        # SORT blocks by Y position (top to bottom) for logical ordering
        blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))  # Sort by Y, then X
        
        # For CMS-1500, always use block.id as the key (schema field ID).
        # For general forms, use semantic_label if SLM provided one.
        extracted_fields = {}
        for block in blocks:
            if block.text:
                val = str(block.text).strip()
                if not val or val.lower() in ("null", "none", "n/a", "-", ""):
                    continue
                if form_id.form_type == FormType.CMS1500:
                    label = block.id
                else:
                    label = block.metadata.get("semantic_label", block.id)
                extracted_fields[label] = val
        
        # Post-process composite address blocks for CMS-1500 scans:
        # Parse full-address OCR into individual sub-fields (city, state, zip, phone)
        if form_id.form_type == FormType.CMS1500 and not use_digital_text:
            for composite_id, prefix in [("5_patient_address_full", "5_patient"),
                                          ("7_insured_address_full", "7_insured")]:
                raw_id = composite_id.replace("_full", "_raw")
                phone_key = f"{prefix}_phone"
                # Parse from _full (schema zone) or _raw (if already renamed)
                source_text = extracted_fields.get(composite_id) or extracted_fields.get(raw_id)
                if source_text:
                    parsed = self._parse_address_block(source_text, prefix)
                    for sub_key, sub_val in parsed.items():
                        if sub_key not in extracted_fields and sub_val:
                            extracted_fields[sub_key] = sub_val
                    # Keep composite as _raw for reference
                    if composite_id in extracted_fields:
                        extracted_fields[raw_id] = extracted_fields.pop(composite_id)
                    # Fallback: if phone still missing, try loose extraction from raw (9-10 digits, OCR-tolerant)
                    if not extracted_fields.get(phone_key) and source_text:
                        import re
                        loose = re.search(r'(?<!\d)[\dOoIlUu]{3}[\s\-\.]?[\dOoIlUu]{3}[\s\-\.]?[\dOoIlUu]{3,4}', source_text)
                        if loose:
                            ph = loose.group().replace('O', '0').replace('o', '0').replace('U', '0').replace('u', '0').replace('I', '1').replace('l', '1')
                            extracted_fields[phone_key] = ph
        
        # Filter: show boxes that have (a) ink in crop OR (b) meaningful OCR text.
        # Ink-only filter was too strict — faint handwriting, low contrast can fail ink detection.
        # We must include all fields with extracted values so they appear in field_details and OCR JSON.
        def _field_bbox_for_check(b: DetectedBlock):
            orig = (b.metadata or {}).get("original_bbox")
            if orig and len(orig) == 4:
                return orig
            return b.bbox

        def _has_meaningful_text(b: DetectedBlock) -> bool:
            """True if block has non-empty, non-placeholder OCR text."""
            text = (b.text or "").strip()
            if not text or len(text) < 2:
                return False
            # Checkbox "X" or signature placeholder without ink = likely false positive
            if b.block_type == BlockType.CHECKBOX and text == "X":
                return False  # Require ink for checkbox
            if b.block_type == BlockType.SIGNATURE and text.upper() in ("SIGNED", "[SIGNED]", "N/A"):
                return False  # Placeholder without real signature
            return True

        def _should_show_field(b: DetectedBlock) -> bool:
            src = (b.metadata or {}).get("source", "")
            if src not in ("schema_zones", "ocr_zone_matching"):
                return True
            if use_digital_text:
                return True
            # Reducto-style: always show schema zones for full form coverage (every field gets a bbox)
            if src == "schema_zones":
                return True
            check_bbox = _field_bbox_for_check(b)
            has_ink = self._has_ink_in_bbox(aligned_image, check_bbox)
            if has_ink:
                return True
            if _has_meaningful_text(b):
                return True
            return False

        filtered_blocks = [b for b in blocks if _should_show_field(b)]
        # Use FULL extracted_fields for OCR JSON, validation, business mapping — includes parsed sub-fields.
        # field_details stays filtered (only boxes with ink or meaningful text) for visualization.
        
        # Step 7: Validation (full extracted data)
        validation = await self.validation_agent.process(filtered_blocks, extracted_fields)

        # Step 7b: Targeted VLM rescue on clearly-bad fields (validator failed / critical noisy fields).
        # This is intentionally conservative to avoid degrading already-good fields.
        vlm_rescue = {"count": 0, "fields": []}
        if not use_digital_text and getattr(self.config, "enable_vlm_ocr_fallback", True):
            vlm_rescue = self._apply_targeted_vlm_rescue(
                aligned_image, filtered_blocks, extracted_fields, validation
            )
            if vlm_rescue.get("count", 0) > 0:
                # Re-validate after rescue updates so business mapping gets corrected values.
                validation = await self.validation_agent.process(filtered_blocks, extracted_fields)
        
        # Step 8: Business Mapping (Canonical Schema)
        from src.pipelines.schemas import map_to_business_schema, merge_business_with_ocr
        
        def _display_bbox(b: DetectedBlock):
            return list(_field_bbox_for_check(b))
        
        # Prepare intermediate result — use full extracted_fields so all values map to business schema
        temp_result = {
            "extracted_fields": extracted_fields,
            "field_details": [
                {"id": b.id, "bbox": _display_bbox(b), "confidence": b.confidence, "metadata": b.metadata}
                for b in filtered_blocks
            ],
            "page_width": width,
            "page_height": height
        }
        
        # Map to business schema
        business_result = map_to_business_schema(temp_result, form_id.form_type.value)
        
        # Build final result dict
        processing_time = time.time() - start_time
        if use_digital_text:
            alignment_quality_value = 1.0
        elif alignment_quality_override is not None:
            alignment_quality_value = float(alignment_quality_override)
        elif alignment_result:
            alignment_quality_value = float(alignment_result.alignment_quality)
        else:
            alignment_quality_value = 0.0
        if alignment_quality_override is not None:
            alignment_success_value = alignment_quality_value >= float(self.config.alignment_quality_threshold)
        else:
            alignment_success_value = bool(alignment_result.success) if alignment_result else False
        
        # Build field_details: blocks with boxes + synthetic entries for parsed sub-fields (no bbox)
        block_ids = {b.id for b in filtered_blocks}
        field_details_list = [
                {
                    "id": b.id,
                "label": self._get_block_label(b),
                    "type": b.block_type.value if hasattr(b.block_type, 'value') else str(b.block_type),
                "bbox": _display_bbox(b),
                    "value": b.text or "",
                    "text": b.text or "",
                    "confidence": b.confidence,
                    "detected_by": b.metadata.get("source", "yolo"),
                    "metadata": b.metadata
                }
            for b in filtered_blocks
        ]
        # Add parsed sub-fields (e.g. 5_patient_city) so they appear in Fields table and match OCR JSON
        for fid, val in extracted_fields.items():
            if fid not in block_ids and val and str(val).strip():
                field_details_list.append({
                    "id": fid,
                    "label": fid.replace("_", " ").title(),
                    "type": "form_field",
                    "bbox": [0, 0, 0, 0],  # No box — derived from composite
                    "value": val,
                    "text": val,
                    "confidence": 0.8,
                    "detected_by": "parsed",
                    "metadata": {"source": "parsed", "derived": True}
                })
        
        # Sort extracted_fields and field_details by schema order (Reducto-style serial reading order)
        schema_order = self._get_schema_field_order(form_id.form_type)
        if schema_order:
            order_idx = {fid: i for i, fid in enumerate(schema_order)}
            def _sort_key(item):
                fid = item.get("id", "") if isinstance(item, dict) else item
                return order_idx.get(fid, 9999)
            extracted_fields = dict(
                sorted(extracted_fields.items(), key=lambda kv: order_idx.get(kv[0], 9999))
            )
            field_details_list = sorted(field_details_list, key=lambda x: _sort_key(x))
        
        final_result = {
            "success": True,
            "form_type": form_id.form_type.value,
            "form_confidence": form_id.confidence,
            "form_version": form_id.version,
            "alignment_quality": alignment_quality_value,
            "extracted_fields": extracted_fields,
            "field_details": field_details_list,
            "page_width": width,
            "page_height": height,
            "processing_time": processing_time,
            "validation": validation,
            "ocr_blocks": [
                {"text": b.text, "bbox": _display_bbox(b), "confidence": b.confidence}
                for b in filtered_blocks if b.text
            ],
            "extraction_method": "multi_agent",
            "config": {
                "layout_model": self.config.layout_model,
                "enable_trocr": self.config.enable_trocr,
                "enable_slm": self.config.enable_slm_labeling,
                "enable_vlm": self.config.enable_vlm_figures
            },
            "debug": {
                "alignment_used": bool(self.config.enable_alignment),
                "alignment_success": alignment_success_value,
                "aligned_image_shape": prod_aligned_shape or (list(aligned_image.shape[:2]) if aligned_image is not None else None),
                "alignment_quality": alignment_quality_value,
                "alignment_method": (
                    (alignment_result.metadata or {}).get("alignment_method")
                    if alignment_result is not None
                    else None
                ),
                "alignment_profile": (
                    ((alignment_result.metadata or {}).get("registrar_debug") or {}).get("profile")
                    if alignment_result is not None
                    else None
                ),
                "aligned_preview_path": str(aligned_preview_path) if aligned_preview_path else None,
                "digital_text_used": bool(use_digital_text),
                "digital_words_count": int(len(digital_words)) if digital_words is not None else 0,
                "vlm_rescue_count": int(vlm_rescue.get("count", 0)),
                "vlm_rescue_fields": vlm_rescue.get("fields", []),
            },
        }
        
        # Merge business data
        final_merged = merge_business_with_ocr(final_result, business_result)
        
        # Add Reducto-style output
        final_merged["reducto_format"] = self._to_reducto_format(final_merged, width, height)
        
        return final_merged
    
    def process_sync(self, path: str) -> Dict[str, Any]:
        """Synchronous wrapper for process()."""
        return asyncio.run(self.process(path))


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Document Pipeline")
    parser.add_argument("input", help="Input document path")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--form-type", choices=["cms-1500", "ub-04", "generic"], help="Override form type")
    parser.add_argument("--no-alignment", action="store_true", help="Disable template alignment")
    parser.add_argument("--no-trocr", action="store_true", help="Disable TrOCR")
    parser.add_argument("--no-slm", action="store_true", help="Disable SLM labeling")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM for figures")
    args = parser.parse_args()
    
    config = PipelineConfig(
        form_type_override=FormType(args.form_type) if args.form_type else None,
        enable_alignment=not args.no_alignment,
        enable_trocr=not args.no_trocr,
        enable_slm_labeling=not args.no_slm,
        enable_vlm_figures=not args.no_vlm
    )
    
    pipeline = MultiAgentPipeline(config)
    result = pipeline.process_sync(args.input)
    
    print(f"\n✅ Processing complete in {result['processing_time']:.2f}s")
    print(f"   Form type: {result['form_type']} ({result['form_confidence']:.0%})")
    print(f"   Fields extracted: {len(result['extracted_fields'])}")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()

"""
Adapter to emit doc2data results in a Reducto-like JSON structure.

This does **not** call Reducto; it transforms our agentic CMS-1500 output
into a similar schema so downstream consumers can switch with minimal changes.
"""
from __future__ import annotations

import time
import uuid
from typing import Dict, Any, List, Optional

from src.pipelines.agentic_cms1500 import run_cms1500_agentic


def _confidence_band(value: float) -> str:
    if value >= 0.9:
        return "high"
    if value >= 0.7:
        return "medium"
    return "low"


def _normalize_bbox(bbox: List[float], page_w: float, page_h: float) -> Dict[str, float]:
    x0, y0, x1, y1 = bbox
    return {
        "left": x0 / page_w if page_w else 0.0,
        "top": y0 / page_h if page_h else 0.0,
        "width": (x1 - x0) / page_w if page_w else 0.0,
        "height": (y1 - y0) / page_h if page_h else 0.0,
        "page": 1,
        "original_page": 1,
    }


def _field_to_block(field: Dict[str, Any], page_w: float, page_h: float) -> Dict[str, Any]:
    value = field.get("value", "")
    label = field.get("label", field.get("id", ""))
    block_type = _infer_block_type(field)
    bbox = field.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    conf = float(field.get("confidence", 0.0) or 0.0)
    section = _infer_section(field)
    block: Dict[str, Any] = {
        "type": block_type,
        "bbox": _normalize_bbox(bbox, page_w, page_h),
        "content": f"{label}: {value}".strip(": "),
        "image_url": None,
        "chart_data": None,
        "confidence": _confidence_band(conf),
        "granular_confidence": {
            "extract_confidence": None,
            "parse_confidence": conf,
        },
    }
    if section:
        block["section"] = section
    # If table, add minimal table metadata placeholder to mimic Reducto assets
    if block_type == "Table":
        block["table_meta"] = {"source": "schema_zone", "rows": None, "cols": None}
    return block


def _infer_block_type(field: Dict[str, Any]) -> str:
    fid = (field.get("id") or "").lower()
    label = (field.get("label") or "").lower()
    val = str(field.get("value") or "").lower()
    if "service_line" in fid or "service lines" in label or "table" in fid or "table" in label:
        return "Table"
    if "diagnosis" in fid or "icd" in fid:
        return "Table"  # treat diagnosis grid as tabular list
    if "signature" in label or "statement" in label:
        return "Section Header"
    if fid.startswith("1a") or "id number" in label:
        return "Key Value"
    if "checkbox" in val or val in {"checked", "unchecked"}:
        return "Key Value"
    return "Key Value"


def _infer_section(field: Dict[str, Any]) -> Optional[str]:
    fid = (field.get("id") or "").lower()
    label = (field.get("label") or "").lower()
    section_map = [
        ("patient", ["2_", "3_", "5_", "12_", "15_", "16_"]),
        ("insured", ["1a_", "4_", "7_", "9_", "11_", "13_"]),
        ("provider", ["25_", "26_", "27_", "28_", "29_", "32_", "33_"]),
        ("diagnosis", ["21_"]),
        ("service_lines", ["24_"]),
    ]
    for section, prefixes in section_map:
        if any(fid.startswith(p) for p in prefixes):
            return section
    if "diagnosis" in label:
        return "diagnosis"
    if "service" in label and "line" in label:
        return "service_lines"
    return None


def adapt_result_to_reducto(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an agentic CMS-1500 extraction result into a Reducto-style JSON.
    """
    page_w = float(ocr_result.get("page_width") or 1.0)
    page_h = float(ocr_result.get("page_height") or 1.0)
    field_details: List[Dict[str, Any]] = ocr_result.get("field_details", []) or []

    # Build blocks
    blocks = []
    for field in field_details:
        if not field.get("id"):
            continue
        blocks.append(_field_to_block(field, page_w, page_h))

    # Build plain-text content from field ordering
    lines = []
    for f in field_details:
        lbl = f.get("label", f.get("id", ""))
        val = f.get("value", "")
        lines.append(f"{lbl}: {val}")

    business = ocr_result.get("business_fields") or {}
    if business:
        lines.append("\nBusiness Fields:")
        for k, v in business.items():
            lines.append(f"{k}: {v}")

    content_text = "\n".join(lines).strip()

    chunk = {
        "content": content_text,
        "embed": content_text,
        "enriched": content_text,
        "enrichment_success": True,
        "blocks": blocks,
    }

    reducto_like = {
        "job_id": str(uuid.uuid4()),
        "duration": ocr_result.get("duration_seconds", 0.0),
        "pdf_url": None,
        "studio_link": None,
        "usage": {
            "num_pages": 1,
            "credits": 0,
        },
        "result": {
            "type": "full",
            "chunks": [chunk],
            "ocr": None,
            "custom": None,
        },
    }
    return reducto_like


def run_reducto_style(
    pdf_path: str,
    use_icr: bool = True,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Run agentic CMS-1500 pipeline and emit Reducto-like JSON in one call.
    """
    start = time.time()
    base = run_cms1500_agentic(pdf_path, use_icr=use_icr, use_llm=use_llm, align_template=True)
    base["duration_seconds"] = time.time() - start
    return adapt_result_to_reducto(base)

"""
Business schema mapper for CMS-1500 and related forms.

The mapper converts OCR/schema-level field IDs into business-friendly keys
(`patient_name`, `insurance_id`, `billing_npi`, etc.) and applies validators
to normalize values. It is intentionally deterministic so it can run both as a
post-processing step for the agentic pipeline and as a fallback for the
existing Read→Understand→Ground flow.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import re

from src.pipelines.validators import validate_field, guess_field_type

# US state/territory abbreviations for address sanity checks
US_STATE_CODES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN",
    "MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA",
    "WV","WI","WY","PR","VI","GU","AS","MP"
}


# Mapping from business field -> schema field ids (ordered by preference)
CMS1500_BUSINESS_MAPPING: Dict[str, Dict[str, Any]] = {
    "patient_name": {"sources": ["2_patient_name"]},
    "patient_dob": {"sources": ["3_patient_dob"], "validator": "date"},
    "patient_sex": {"sources": ["3_patient_sex_m", "3_patient_sex_f", "3_patient_sex"], "composer": "sex"},
    "patient_address": {
        "sources": ["5_patient_address", "5_patient_city", "5_patient_state", "5_patient_zip"],
        "composer": "address",
    },
    "patient_phone": {"sources": ["5_patient_phone"], "validator": "phone"},
    "insured_name": {"sources": ["4_insured_name"]},
    "insured_id": {"sources": ["1a_insured_id"], "validator": "member_id"},
    "insured_dob": {"sources": ["11a_insured_dob"], "validator": "date"},
    "insured_sex": {"sources": ["11a_insured_sex_m", "11a_insured_sex_f", "11a_insured_sex"], "composer": "sex"},
    "insured_address": {
        "sources": ["7_insured_address", "7_insured_city", "7_insured_state", "7_insured_zip"],
        "composer": "address",
    },
    "insured_phone": {"sources": ["7_insured_phone"], "validator": "phone"},
    "insurance_plan_name": {"sources": ["11c_insurance_plan_name"]},
    "claim_number": {"sources": ["26_patient_account"]},
    "service_facility_name": {"sources": ["32_service_facility_name"]},
    "service_facility_address": {"sources": ["32_service_facility_address"], "composer": "address"},
    "billing_provider_name": {"sources": ["33_billing_provider_name"]},
    "billing_provider_address": {"sources": ["33_billing_provider_address"], "composer": "address"},
    "billing_provider_phone": {"sources": ["33_billing_provider_phone"], "validator": "phone"},
    "billing_npi": {"sources": ["33a_npi", "32a_npi"], "validator": "npi"},
    "total_charge": {"sources": ["28_total_charge"], "validator": "money"},
    "amount_paid": {"sources": ["29_amount_paid"], "validator": "money"},
    "diagnosis_codes": {"sources": ["21_diagnosis_a", "21_diagnosis_b"], "validator": "icd"},
}


@dataclass
class BusinessFieldDetail:
    """Detailed business field with provenance and validation status."""
    business_id: str
    value: Any
    source_field_id: Optional[str] = None
    confidence: float = 0.0
    bbox: Optional[List[float]] = None
    validator: Optional[str] = None
    validator_passed: Optional[bool] = None
    normalized_value: Optional[Any] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "business_id": self.business_id,
            "value": self.value,
            "source_field_id": self.source_field_id,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "validator": self.validator,
            "validator_passed": self.validator_passed,
            "normalized_value": self.normalized_value,
            "notes": self.notes,
        }


def _find_detail_for_field(field_id: str, field_details: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the first detail dict matching a field id."""
    for detail in field_details:
        if detail.get("id") == field_id:
            return detail
    return None


def _extract_value_from_labeled_text(text: str, field_type: str = "") -> str:
    """
    Extract the actual value from OCR text that includes pre-printed labels.
    
    Example:
        "2. PATIENT'S NAME (La Sharma, Rahul" -> "Sharma, Rahul"
        "ATIENT'S BIRTH MM DD 11 08 98" -> "11 08 98"
        "1 a. INSURED'SI.D. N ID10-45678" -> "ID10-45678"
    """
    if not text:
        return ""
    
    result = text
    
    # Phase 1: Remove field numbers at start
    result = re.sub(r"^\d+\s*[a-z]?\.\s*", "", result, flags=re.IGNORECASE)
    
    # Phase 2: Remove specific CMS-1500 labels (non-greedy, careful patterns)
    label_patterns = [
        # Patient fields
        r"P?ATIENT'?S?\s+NAME\s*\(La",  # "PATIENT'S NAME (La" -> remove, keep rest
        r"P?ATIENT'?S?\s+NAME\s*",
        r"P?ATIENT'?S?\s+BIRTH\s*",
        r"P?ATIENT'?S?\s+ADDRESS\s*",
        r"P?ATIENT'?S?\s+PHONE\s*",
        r"P?ATIENT'?S?\s+ACCOUNT\s*",
        
        # Insured fields
        r"I?NSURED'?S?\s*I\.?D\.?\s*N\w*\s*",  # "INSURED'S I.D. NUMBER"
        r"I?NSURED'?S?\s+NAME\s*\(La",
        r"I?NSURED'?S?\s+NAME\s*",
        r"I?NSURED'?S?\s+ADDRESS\s*",
        r"I?NSURED'?S?\s+DATE\s+OF\s+BIRT\w*\s*",
        r"I?NSURED'?S?\s+POLICY\s*",
        
        # Other fields
        r"INSURANCE\s+PLAN\s+NAME\s*",
        r"SIGNATURE\s+OF\s+PHYSICIAN[^,]*",
        r"SIGNATURE\s+OF\s*",
        r"BILLING\s+PROVIDER[^,]*",
        r"SERVICE\s+FACILITY[^,]*",
        
        # Date hints
        r"MM\s+DD\s+Y+",
        r"MM\s+DD\s+",
        r"^MM\s+DD\s*",
    ]
    
    for pattern in label_patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    
    # Phase 3: Remove parenthetical instructions
    result = re.sub(r"\(Last[^)]*\)", "", result)
    result = re.sub(r"\(No\.[^)]*\)", "", result)
    result = re.sub(r"\(Include[^)]*\)", "", result)
    
    # Phase 4: Clean up
    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"^[\s\(\)\[\]\.,\-:]+", "", result)
    result = re.sub(r"[\s\(\)\[\]\.,\-:]+$", "", result)
    
    return result.strip()


def _compose_address(values: Dict[str, str]) -> str:
    """Join address components into a single line."""
    parts = []
    address = values.get("address")
    if address:
        parts.append(address)
    city = values.get("city")
    state = values.get("state")
    zip_code = values.get("zip")
    # Basic sanitization: ignore garbage values that often come from nearby printed labels
    if state:
        st = str(state).strip().upper()
        state = st if (re.fullmatch(r"[A-Z]{2}", st) and st in US_STATE_CODES) else None
    if zip_code:
        z = re.sub(r"[^0-9]", "", str(zip_code))
        zip_code = z if len(z) in (5, 9) else None
    locality = " ".join(p for p in [city, state] if p)
    if locality:
        parts.append(locality.strip())
    if zip_code:
        parts[-1] = (parts[-1] + " " + zip_code).strip()
    return ", ".join([p for p in parts if p]).strip(", ")


def _compose_sex(values: Dict[str, str]) -> str:
    """
    Compose sex from checkbox sources.
    We treat any non-empty (or 'X') as checked.
    """
    m = (values.get("sex_m") or values.get("m") or "").strip().lower()
    f = (values.get("sex_f") or values.get("f") or "").strip().lower()
    def checked(v: str) -> bool:
        return v not in ("", "none", "null", "0") and ("x" in v or v == "checked" or v == "true" or v == "1")
    if checked(m):
        return "M"
    if checked(f):
        return "F"
    # Fallback: sometimes OCR captures 'M'/'F' as text
    raw = " ".join([values.get("sex") or "", values.get("raw") or ""]).lower()
    if " m" in raw or raw.strip() == "m":
        return "M"
    if " f" in raw or raw.strip() == "f":
        return "F"
    return ""


def _pick_value_from_sources(
    sources: List[str],
    extracted_fields: Dict[str, Any],
    field_details: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str], float, Optional[List[float]]]:
    """
    Pick the best value among multiple source fields.

    Returns: value, source_field_id, confidence, bbox
    """
    best_value: Optional[str] = None
    best_source: Optional[str] = None
    best_conf: float = 0.0
    best_bbox: Optional[List[float]] = None

    for source in sources:
        raw_value = extracted_fields.get(source)
        detail = _find_detail_for_field(source, field_details)
        conf = float(detail.get("confidence", 0.0)) if detail else 0.0
        bbox = detail.get("bbox") if detail else None

        if raw_value is None or raw_value == "":
            continue
        
        # Clean the value - extract actual data from labeled text
        value = _extract_value_from_labeled_text(str(raw_value))
        
        if not value:
            continue

        # Prefer validated or higher-confidence values
        if conf > best_conf or best_value is None:
            best_value = value
            best_source = source
            best_conf = conf
            best_bbox = bbox if bbox else best_bbox

    return best_value, best_source, best_conf, best_bbox


def _validator_for_field(field_id: str, default_validator: Optional[str]) -> Optional[str]:
    """Resolve validator using schema hints when explicit validator is missing."""
    if default_validator:
        return default_validator
    inferred = guess_field_type(field_id)
    return inferred


def map_to_business_schema(
    ocr_result: Dict[str, Any],
    form_type: str = "cms1500"
) -> Dict[str, Any]:
    """
    Map OCR/schema output to business schema.

    Args:
        ocr_result: Dict containing `extracted_fields` and `field_details`
        form_type: Currently only `cms1500` is supported

    Returns:
        Dict with business_fields and detailed metadata
    """
    if not ocr_result:
        return {"business_fields": {}, "business_field_details": []}

    extracted_fields: Dict[str, Any] = ocr_result.get("extracted_fields", {}) or {}
    field_details: List[Dict[str, Any]] = ocr_result.get("field_details", []) or []

    mapping = CMS1500_BUSINESS_MAPPING if form_type.lower().startswith("cms") else {}
    business_fields: Dict[str, Any] = {}
    details: List[BusinessFieldDetail] = []

    # Pre-cache address pieces to avoid repeated lookups
    address_cache = {
        "address": extracted_fields.get("5_patient_address"),
        "city": extracted_fields.get("5_patient_city"),
        "state": extracted_fields.get("5_patient_state"),
        "zip": extracted_fields.get("5_patient_zip"),
    }
    insured_addr_cache = {
        "address": extracted_fields.get("7_insured_address"),
        "city": extracted_fields.get("7_insured_city"),
        "state": extracted_fields.get("7_insured_state"),
        "zip": extracted_fields.get("7_insured_zip"),
    }
    facility_addr_cache = {
        "address": extracted_fields.get("32_service_facility_address"),
        "city": None,
        "state": None,
        "zip": None,
    }
    billing_addr_cache = {
        "address": extracted_fields.get("33_billing_provider_address"),
        "city": None,
        "state": None,
        "zip": None,
    }

    for biz_field, cfg in mapping.items():
        sources = cfg.get("sources", [])
        composer = cfg.get("composer")
        validator_name = _validator_for_field(biz_field, cfg.get("validator"))

        value = None
        source_field = None
        conf = 0.0
        bbox = None
        normalized = None
        validator_passed = None
        notes = None

        if composer == "address":
            cache = address_cache
            if biz_field == "service_facility_address":
                cache = facility_addr_cache
            elif biz_field == "billing_provider_address":
                cache = billing_addr_cache
            elif biz_field == "insured_address":
                cache = insured_addr_cache
            value = _compose_address(cache)
            source_field = sources[0] if sources else None
            conf = 0.65 if value else 0.0
            bbox = (_find_detail_for_field(source_field, field_details) or {}).get("bbox") if source_field else None
        elif composer == "sex":
            # Build a small cache from expected checkbox ids
            sex_cache = {
                "sex_m": extracted_fields.get("3_patient_sex_m") if biz_field == "patient_sex" else extracted_fields.get("11a_insured_sex_m"),
                "sex_f": extracted_fields.get("3_patient_sex_f") if biz_field == "patient_sex" else extracted_fields.get("11a_insured_sex_f"),
                "sex": extracted_fields.get("3_patient_sex") if biz_field == "patient_sex" else extracted_fields.get("11a_insured_sex"),
            }
            value = _compose_sex(sex_cache)
            source_field = sources[0] if sources else None
            conf = 0.8 if value else 0.0
            bbox = (_find_detail_for_field(source_field, field_details) or {}).get("bbox") if source_field else None
        elif biz_field == "diagnosis_codes":
            diag_values = []
            for sid in sources:
                diag_val = extracted_fields.get(sid)
                if diag_val:
                    diag_values.append(str(diag_val))
            value = diag_values
            source_field = sources[0] if diag_values else None
            conf = 0.55 if diag_values else 0.0
            bbox = (_find_detail_for_field(source_field, field_details) or {}).get("bbox") if source_field else None
        else:
            value, source_field, conf, bbox = _pick_value_from_sources(sources, extracted_fields, field_details)

        # Apply validator if available
        if validator_name and value not in (None, "", []):
            # Handle list values (e.g., diagnosis_codes) by validating each item
            if isinstance(value, list):
                validated_items = []
                all_passed = True
                for item in value:
                    if item:
                        passed, info = validate_field(validator_name, str(item))
                        if passed:
                            validated_items.append(info.get("normalized") or item)
                        else:
                            validated_items.append(item)
                            all_passed = False
                value = validated_items
                validator_passed = all_passed
                if all_passed:
                    conf = max(conf, 0.75)
                else:
                    notes = f"partial_{validator_name}"
            else:
                passed, info = validate_field(validator_name, value)
                validator_passed = bool(passed)
                normalized = info.get("normalized")
            if passed:
                value = normalized if normalized else value
                conf = max(conf, 0.75)
            else:
                notes = f"failed_{validator_name}"
                conf = min(conf, 0.55)

        # Final assignment
        business_fields[biz_field] = value
        details.append(
            BusinessFieldDetail(
                business_id=biz_field,
                value=value,
                source_field_id=source_field,
                confidence=conf,
                bbox=bbox,
                validator=validator_name,
                validator_passed=validator_passed,
                normalized_value=normalized,
                notes=notes,
            )
        )

    # Coverage metric
    filled = sum(1 for v in business_fields.values() if v not in (None, "", []))
    coverage = filled / max(len(mapping), 1)

    return {
        "business_fields": business_fields,
        "business_field_details": [d.to_dict() for d in details],
        "business_coverage": coverage,
    }


def merge_business_with_ocr(
    ocr_result: Dict[str, Any],
    business_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge business mapping output back into the main response payload.
    """
    merged = dict(ocr_result or {})
    if business_result:
        merged["business_fields"] = business_result.get("business_fields", {})
        merged["business_field_details"] = business_result.get("business_field_details", [])
        merged["business_coverage"] = business_result.get("business_coverage")
    return merged

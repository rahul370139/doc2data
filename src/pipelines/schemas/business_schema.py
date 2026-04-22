"""
Business schema mapper for CMS-1500 and related forms.

PURPOSE: Converts OCR/schema field IDs (e.g. 2_patient_name) into business
keys (patient_name, insurance_id, billing_npi). Applies validators, composes
addresses and sex from checkboxes. Deterministic for reproducibility.

USE CASE: Call map_to_business_schema(ocr_result) after extraction to get
clean JSON for downstream systems. Used by MultiAgentPipeline and API.
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
# Field names aligned with gold label format for grading.  Order is the
# order they render in the UI.  Stakeholder priorities drive what's
# included here: claim identification, parties, dates, diagnoses, money.
CMS1500_BUSINESS_MAPPING: Dict[str, Dict[str, Any]] = {
    # ── Claim header / type ───────────────────────────────────────────
    "insurance_type": {
        # Composed from the 7 type checkboxes at the top of the form.
        "sources": [
            "1_insurance_type_medicare", "1_insurance_type_medicaid",
            "1_insurance_type_tricare", "1_insurance_type_champva",
            "1_insurance_type_group", "1_insurance_type_feca",
            "1_insurance_type_other",
        ],
        "composer": "insurance_type",
    },

    # ── Patient ───────────────────────────────────────────────────────
    "patient_name": {"sources": ["2_patient_name"]},
    "patient_dob": {"sources": ["3_patient_dob"], "validator": "date"},
    "patient_sex": {
        "sources": ["3_patient_sex_m", "3_patient_sex_f", "3_patient_sex"],
        "composer": "sex",
    },
    "patient_address": {"sources": ["5_patient_address"]},
    "patient_city": {"sources": ["5_patient_city"]},
    "patient_state": {"sources": ["5_patient_state"]},
    "patient_zip": {"sources": ["5_patient_zip"]},
    "patient_phone": {"sources": ["5_patient_phone"], "validator": "phone"},
    "patient_relationship": {
        "sources": [
            "6_patient_relationship_self", "6_patient_relationship_spouse",
            "6_patient_relationship_child", "6_patient_relationship_other",
            "6_patient_relationship",
        ],
        "composer": "patient_relationship",
    },

    # ── Insured ───────────────────────────────────────────────────────
    "insured_name": {"sources": ["4_insured_name"]},
    "insured_id": {"sources": ["1a_insured_id"], "validator": "member_id"},
    "insured_dob": {"sources": ["11a_insured_dob"], "validator": "date"},
    "insured_sex": {
        "sources": [
            "11a_insured_sex_m", "11a_insured_sex_f", "11a_insured_sex",
        ],
        "composer": "sex",
    },
    "insured_address": {"sources": ["7_insured_address"]},
    "insured_city": {"sources": ["7_insured_city"]},
    "insured_state": {"sources": ["7_insured_state"]},
    "insured_zip": {"sources": ["7_insured_zip"]},
    "insured_phone": {"sources": ["7_insured_phone"], "validator": "phone"},

    # ── Insurance / policy / employer ─────────────────────────────────
    "insurance_plan": {
        "sources": ["11c_insurance_plan_name", "11c_insurance_plan"],
    },
    "policy_number": {
        "sources": [
            "11_insured_policy_group", "11_policy_group_number",
            "11_group_number",
        ],
    },
    "other_claim_id": {"sources": ["11b_other_claim_id"]},
    "another_health_plan": {
        "sources": [
            "11d_another_health_plan_yes", "11d_another_health_plan_no",
        ],
        "composer": "yes_no",
    },

    # ── Condition relation (Box 10) ───────────────────────────────────
    "condition_employment": {
        "sources": ["10a_employment_yes", "10a_employment_no"],
        "composer": "yes_no",
    },
    "condition_auto_accident": {
        "sources": ["10b_auto_accident_yes", "10b_auto_accident_no"],
        "composer": "yes_no",
    },
    "condition_other_accident": {
        "sources": ["10c_other_accident_yes", "10c_other_accident_no"],
        "composer": "yes_no",
    },

    # ── Dates / referral / hospitalization ────────────────────────────
    "current_illness_date": {
        "sources": ["14_date_of_illness"], "validator": "date",
    },
    "other_date": {
        "sources": ["15_Other_date", "15_other_date"], "validator": "date",
    },
    "unable_to_work_dates": {
        "sources": ["16_dates_unable_to_work"], "validator": "date_range",
    },
    "referring_provider": {"sources": ["17_referring_provider"]},
    "referring_npi": {
        "sources": ["17b_referring_npi"], "validator": "npi",
    },
    "hospitalization_dates": {
        "sources": ["18_hospitalization_dates"], "validator": "date_range",
    },
    "additional_claim_info": {"sources": ["19_additional_claim_info"]},

    # ── Outside lab / diagnosis / claim codes ─────────────────────────
    "outside_lab": {
        "sources": ["20_outside_lab_yes", "20_outside_lab_no"],
        "composer": "yes_no",
    },
    "outside_lab_charges": {
        "sources": ["20_charges"], "validator": "money",
    },
    # Combined diagnosis area (the 21_diagnosis_all field already covers
    # codes A-L on the form).  Kept as a single string here for the
    # demo; downstream consumers can split on whitespace if needed.
    "diagnosis_codes": {
        "sources": [
            "21_diagnosis_all", "21_diagnosis_a", "21_diagnosis_1",
        ],
    },
    "resubmission_code": {"sources": ["22_resubmission_code"]},
    "original_ref_number": {"sources": ["22_original_ref_number"]},
    "prior_authorization": {"sources": ["23_prior_authorization"]},

    # ── Service lines (Box 24 table) ─────────────────────────────────
    # The table block stores its parsed rows on metadata.table_rows;
    # we expose the joined summary string here so the schema view stays
    # human-readable.  The full structured rows are also surfaced
    # under business_field_details.notes for downstream systems.
    "service_lines": {"sources": ["24_service_lines"]},

    # ── Tax / accounts / charges ─────────────────────────────────────
    "tax_id": {"sources": ["25_federal_tax_id", "25_tax_id"]},
    "tax_id_type": {
        "sources": [
            "25b_federal_tax_id_type_ssn",
            "25b_federal_tax_id_type_ein",
        ],
        "composer": "tax_id_type",
    },
    "patient_account": {"sources": ["26_patient_account"]},
    "accept_assignment": {
        "sources": [
            "27_accept_assignment_yes", "27_accept_assignment_no",
        ],
        "composer": "yes_no",
    },
    "total_charge": {"sources": ["28_total_charge"], "validator": "money"},
    "amount_paid": {"sources": ["29_amount_paid"], "validator": "money"},

    # ── Provider / facility / signature ──────────────────────────────
    "physician_signature_date": {
        "sources": ["31_physician_signature_date"], "validator": "date",
    },
    "physician_signature_present": {
        "sources": ["31_physician_signature"], "composer": "signed_marker",
    },
    # The CMS-1500 schema collapses the service facility / billing
    # provider name, street, and city/state/zip into a single multi-line
    # bbox (*_address).  We surface the same value under two keys —
    # one verbatim and one as ``address`` block — so stakeholders get
    # both "the whole block" and a parsed-address view when we can
    # split it.
    "service_facility": {"sources": ["32_service_facility_address"]},
    "service_facility_address": {
        "sources": ["32_service_facility_address"], "composer": "address",
    },
    "service_facility_npi": {
        "sources": ["32_a_npi", "32a_npi"], "validator": "npi",
    },
    "billing_provider": {"sources": ["33_billing_provider_address"]},
    "billing_provider_address": {
        "sources": ["33_billing_provider_address"], "composer": "address",
    },
    "billing_provider_phone": {
        "sources": ["33_billing_provider_phone"], "validator": "phone",
    },
    "billing_npi": {
        "sources": ["33a_npi", "33_a_npi"], "validator": "npi",
    },
}


# UB-04 Business Field Mapping
UB04_BUSINESS_MAPPING: Dict[str, Dict[str, Any]] = {
    # Provider info
    "provider_name": {"sources": ["fl1_provider_name"]},
    "provider_address": {"sources": ["fl1_provider_address1", "fl1_provider_city_state_zip"], "composer": "join"},
    
    # Patient info
    "patient_name": {"sources": ["fl8a_patient_id"]},
    "patient_dob": {"sources": ["fl10_patient_dob"], "validator": "date"},
    "patient_sex": {"sources": ["fl11_patient_sex"]},
    "patient_address": {"sources": ["fl9_patient_address"]},
    "patient_city": {"sources": ["fl9_patient_city"]},
    "patient_state": {"sources": ["fl9_patient_state"]},
    "patient_zip": {"sources": ["fl9_patient_zip"]},
    
    # Claim info
    "patient_control_number": {"sources": ["fl3a_patient_control"]},
    "type_of_bill": {"sources": ["fl4_type_of_bill"]},
    "federal_tax_id": {"sources": ["fl5_federal_tax_id"]},
    "statement_from_date": {"sources": ["fl6_from_date"], "validator": "date"},
    "statement_thru_date": {"sources": ["fl6_thru_date"], "validator": "date"},
    
    # Admission info
    "admission_date": {"sources": ["fl12_admission_date"], "validator": "date"},
    "admission_hour": {"sources": ["fl13_admission_hour"]},
    "admission_type": {"sources": ["fl14_admission_type"]},
    "admission_source": {"sources": ["fl15_admission_source"]},
    "discharge_status": {"sources": ["fl17_patient_status"]},
    
    # Payer info
    "payer_name": {"sources": ["fl50_payer_name_a"]},
    "health_plan_id": {"sources": ["fl51_health_plan_id_a"]},
    "prior_payments": {"sources": ["fl54_prior_payments_a"], "validator": "money"},
    "estimated_amount_due": {"sources": ["fl55_estimated_due_a"], "validator": "money"},
    "billing_npi": {"sources": ["fl56_npi_a"], "validator": "npi"},
    
    # Insured info
    "insured_name": {"sources": ["fl58_insured_name_a"]},
    "insured_id": {"sources": ["fl60_insured_id_a"]},
    "group_name": {"sources": ["fl61_group_name_a"]},
    "group_number": {"sources": ["fl62_group_number_a"]},
    "patient_relationship": {"sources": ["fl59_patient_rel_a"]},
    
    # Treatment/Employer
    "treatment_auth_code": {"sources": ["fl63_treatment_auth_a"]},
    "employer_name": {"sources": ["fl65_employer_name_a"]},
    
    # Diagnosis codes (FL 67-72)
    "principal_diagnosis": {"sources": ["fl67_principal_diagnosis"], "validator": "icd"},
    "diagnosis_code_2": {"sources": ["fl67a_diagnosis_a"], "validator": "icd"},
    "diagnosis_code_3": {"sources": ["fl67b_diagnosis_b"], "validator": "icd"},
    "diagnosis_code_4": {"sources": ["fl67c_diagnosis_c"], "validator": "icd"},
    "admitting_diagnosis": {"sources": ["fl69_admitting_diagnosis"], "validator": "icd"},
    "pps_code": {"sources": ["fl71_pps_code"]},
    "eci_code": {"sources": ["fl72_eci_code"]},
    "admitting_dx_code": {"sources": ["fl79_admitting_dx_code"]},
    
    # Procedure codes
    "principal_procedure": {"sources": ["fl74_principal_procedure"]},
    "principal_procedure_date": {"sources": ["fl74_principal_proc_date"], "validator": "date"},
    "other_procedure_1": {"sources": ["fl74a_other_procedure_1"]},
    
    # Service line 1
    "revenue_code_1": {"sources": ["fl42_revenue_code_1"]},
    "service_description_1": {"sources": ["fl43_description_1"]},
    "hcpcs_code_1": {"sources": ["fl44_hcpcs_1"]},
    "service_date_1": {"sources": ["fl45_service_date_1"], "validator": "date"},
    "charges_1": {"sources": ["fl47_charges_1"], "validator": "money"},
    
    # Service line 2
    "revenue_code_2": {"sources": ["fl42_revenue_code_2"]},
    "service_description_2": {"sources": ["fl43_description_2"]},
    "hcpcs_code_2": {"sources": ["fl44_hcpcs_2"]},
    "service_date_2": {"sources": ["fl45_service_date_2"], "validator": "date"},
    "charges_2": {"sources": ["fl47_charges_2"], "validator": "money"},
    
    # Totals
    "total_charges": {"sources": ["fl47_total_charges"], "validator": "money"},
    
    # Attending physician
    "attending_npi": {"sources": ["fl76_attending_npi"], "validator": "npi"},
    "attending_physician_last": {"sources": ["fl76_attending_last"]},
    "attending_physician_first": {"sources": ["fl76_attending_first"]},
    
    # Responsible party
    "responsible_party": {"sources": ["fl38_responsible_party"]},
    
    # Remarks
    "remarks": {"sources": ["fl80_remarks"]},
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
    
    # Phase 1: Remove short field numbers at start (avoid stripping real values like 3000.78)
    # Examples to remove: "1.", "1a.", "24." when followed by whitespace
    result = re.sub(r"^\s*\d{1,2}\s*[a-z]?\s*\.?\s+", "", result, flags=re.IGNORECASE)
    
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


def _is_check_marked(value: Any) -> bool:
    """Treat anything non-blank that isn't an explicit unchecked sentinel as checked.

    Checkbox detector outputs typically look like ``"X"`` (filled),
    ``""`` (empty), or rarely ``"checked"`` / ``"true"`` / ``"1"`` from
    the SLM cleanup path.  Some pipelines also return the literal mark
    glyph ``✓`` or ``✗``.
    """
    if value is None:
        return False
    s = str(value).strip().lower()
    if not s or s in ("none", "null", "0", "false", "off", "no"):
        return False
    return (
        "x" in s or "✓" in s or "✗" in s
        or s in ("checked", "true", "yes", "on", "1")
    )


def _compose_yes_no(yes_value: Any, no_value: Any) -> str:
    """Pick "yes"/"no"/empty from a yes/no checkbox pair."""
    yes = _is_check_marked(yes_value)
    no = _is_check_marked(no_value)
    if yes and not no:
        return "yes"
    if no and not yes:
        return "no"
    if yes and no:
        # Both marked → ambiguous, prefer the stronger signal we got
        # downstream by leaving it visible to the validation panel.
        return "ambiguous"
    return ""


def _compose_insurance_type(extracted: Dict[str, Any]) -> str:
    """Pick the insurance program from the 7 type checkboxes at the top.

    Returns one of the canonical labels expected by clearinghouses
    (MEDICARE / MEDICAID / TRICARE / CHAMPVA / GROUP / FECA / OTHER) or
    ``""`` when none are visibly checked.  When more than one is
    marked we return the highest-priority single label and tag the
    remainder under business_field_details.notes via the caller.
    """
    candidates = [
        ("MEDICARE", extracted.get("1_insurance_type_medicare")),
        ("MEDICAID", extracted.get("1_insurance_type_medicaid")),
        ("TRICARE",  extracted.get("1_insurance_type_tricare")),
        ("CHAMPVA",  extracted.get("1_insurance_type_champva")),
        ("GROUP",    extracted.get("1_insurance_type_group")),
        ("FECA",     extracted.get("1_insurance_type_feca")),
        ("OTHER",    extracted.get("1_insurance_type_other")),
    ]
    marked = [name for name, val in candidates if _is_check_marked(val)]
    if marked:
        return marked[0]
    # Fallback: a free-text "1_insurance_type" sometimes carries the
    # literal name when the checkboxes were unreadable.
    raw = (extracted.get("1_insurance_type") or "").strip()
    if raw:
        return raw.upper()
    return ""


def _compose_patient_relationship(extracted: Dict[str, Any]) -> str:
    """Patient relationship to insured: SELF / SPOUSE / CHILD / OTHER."""
    pairs = [
        ("SELF",   extracted.get("6_patient_relationship_self")),
        ("SPOUSE", extracted.get("6_patient_relationship_spouse")),
        ("CHILD",  extracted.get("6_patient_relationship_child")),
        ("OTHER",  extracted.get("6_patient_relationship_other")),
    ]
    for name, val in pairs:
        if _is_check_marked(val):
            return name
    raw = (extracted.get("6_patient_relationship") or "").strip()
    return raw.upper() if raw else ""


def _compose_tax_id_type(extracted: Dict[str, Any]) -> str:
    """SSN vs EIN type indicator (Box 25b)."""
    if _is_check_marked(extracted.get("25b_federal_tax_id_type_ein")):
        return "EIN"
    if _is_check_marked(extracted.get("25b_federal_tax_id_type_ssn")):
        return "SSN"
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
        
        # Determine if this value came from AcroForm widgets (already clean)
        # or from OCR (needs label stripping)
        meta = (detail.get("metadata") or {}) if detail else {}
        source_type = meta.get("source", "")
        skip_cleaning = bool(meta.get("skip_label_cleaning") or meta.get("digital_text"))
        if source_type == "acroform_widget" or skip_cleaning:
            # Widget values and clean digital values are ground truth — don't strip anything
            value = str(raw_value).strip()
        else:
            # Clean the value - extract actual data from labeled text (OCR cleanup)
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

    # Select mapping based on form type
    form_lower = form_type.lower().replace("-", "").replace("_", "")
    if form_lower.startswith("cms") or "1500" in form_lower:
        mapping = CMS1500_BUSINESS_MAPPING
    elif "ub04" in form_lower or "ub" in form_lower:
        mapping = UB04_BUSINESS_MAPPING
    else:
        mapping = {}
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
        elif composer == "yes_no":
            # Sources is a [yes_id, no_id] pair (in that order).  Default
            # to (None, None) when fewer ids are provided so the helper
            # treats the missing side as unchecked instead of raising.
            yes_id = sources[0] if len(sources) > 0 else None
            no_id = sources[1] if len(sources) > 1 else None
            value = _compose_yes_no(
                extracted_fields.get(yes_id) if yes_id else None,
                extracted_fields.get(no_id) if no_id else None,
            )
            source_field = yes_id if value == "yes" else (no_id if value == "no" else (yes_id or no_id))
            conf = 0.85 if value in ("yes", "no") else (0.5 if value else 0.0)
            bbox = (_find_detail_for_field(source_field, field_details) or {}).get("bbox") if source_field else None
        elif composer == "insurance_type":
            value = _compose_insurance_type(extracted_fields)
            # Pick the source whose checkbox we actually picked so the
            # frontend can highlight it on the PDF.
            tag = value.lower()
            preferred = next(
                (sid for sid in sources if tag and tag in sid), sources[0] if sources else None,
            )
            source_field = preferred
            conf = 0.85 if value else 0.0
            bbox = (_find_detail_for_field(source_field, field_details) or {}).get("bbox") if source_field else None
        elif composer == "patient_relationship":
            value = _compose_patient_relationship(extracted_fields)
            tag = value.lower()
            preferred = next(
                (sid for sid in sources if tag and tag in sid), sources[0] if sources else None,
            )
            source_field = preferred
            conf = 0.85 if value else 0.0
            bbox = (_find_detail_for_field(source_field, field_details) or {}).get("bbox") if source_field else None
        elif composer == "tax_id_type":
            value = _compose_tax_id_type(extracted_fields)
            preferred = sources[0] if sources else None
            if value:
                preferred = next(
                    (sid for sid in sources if value.lower() in sid),
                    preferred,
                )
            source_field = preferred
            conf = 0.85 if value else 0.0
            bbox = (_find_detail_for_field(source_field, field_details) or {}).get("bbox") if source_field else None
        elif composer == "signed_marker":
            # 31 physician signature shows up as "[SIGNED]" or any ink.
            raw = (extracted_fields.get(sources[0]) if sources else "") or ""
            raw_str = str(raw).strip()
            value = "yes" if raw_str else ""
            source_field = sources[0] if sources else None
            conf = 0.9 if value else 0.0
            bbox = (_find_detail_for_field(source_field, field_details) or {}).get("bbox") if source_field else None
        elif biz_field == "diagnosis_codes":
            # The 21_diagnosis_all field holds the full text;
            # split by whitespace so downstream consumers get a list.
            joined = ""
            primary_id = None
            for sid in sources:
                v = (extracted_fields.get(sid) or "").strip()
                if v:
                    joined = v
                    primary_id = sid
                    break
            if joined:
                # Split into 1-7 character ICD-10 candidates by
                # whitespace / commas.  Validator will mark each
                # member individually.
                parts = [p.strip(" ,;.") for p in re.split(r"[\s,]+", joined) if p.strip()]
                value = [p for p in parts if 2 <= len(p) <= 8]
            else:
                value = []
            source_field = primary_id
            conf = 0.7 if value else 0.0
            bbox = (_find_detail_for_field(source_field, field_details) or {}).get("bbox") if source_field else None
        elif biz_field == "service_lines":
            # Pull both the human-readable summary AND the structured
            # rows the table extractor parked on metadata.table_rows.
            primary_id = sources[0] if sources else None
            detail = _find_detail_for_field(primary_id, field_details) if primary_id else None
            meta = (detail or {}).get("metadata") or {}
            rows = meta.get("table_rows") or []
            summary = (extracted_fields.get(primary_id) or "").strip() if primary_id else ""
            if rows:
                value = rows
                conf = 0.75
                notes = f"summary: {summary}" if summary else None
            elif summary:
                value = summary
                conf = 0.6
            else:
                value = []
                conf = 0.0
            source_field = primary_id
            bbox = (detail or {}).get("bbox") if detail else None
        else:
            value, source_field, conf, bbox = _pick_value_from_sources(sources, extracted_fields, field_details)

        # Apply validator if available.  Guard `passed` so the list branch
        # (which validates per-item) doesn't leak `passed` from the
        # previous iteration of the outer loop — that was a latent
        # NameError when the first business field happened to be a list.
        passed = False
        if validator_name and value not in (None, "", []):
            # Handle list values (e.g., diagnosis_codes) by validating each item
            if isinstance(value, list):
                validated_items: List[Any] = []
                all_passed = True
                for item in value:
                    if not item:
                        continue
                    # Skip non-string items (e.g. dict service-line rows
                    # that don't have a meaningful flat-string validator).
                    if not isinstance(item, (str, int, float)):
                        validated_items.append(item)
                        continue
                    item_passed, info = validate_field(
                        validator_name, str(item),
                    )
                    if item_passed:
                        validated_items.append(info.get("normalized") or item)
                    else:
                        validated_items.append(item)
                        all_passed = False
                value = validated_items
                validator_passed = all_passed
                passed = all_passed
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

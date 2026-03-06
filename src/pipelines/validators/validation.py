"""
Field validators and ValidationAgent — single module for all validation.

PURPOSE:
- Validates and normalizes extracted field values (NPI, date, phone, ICD-10,
  HCPCS, SSN, zip, money, etc.). Each validator returns (passed, info).
- ValidationAgent runs field-level validation over blocks and optional LLM QA.

USE CASE: Pipeline calls ValidationAgent.process() after assembly. Other
modules (business_schema, multi_agent_pipeline) use validate_field() directly.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.pipelines.core import BaseAgent, DetectedBlock, PipelineConfig
from utils.config import Config

# ---------------------------------------------------------------------------
# Validator functions (field-level format checks)
# ---------------------------------------------------------------------------
NPI_PATTERN = re.compile(r"^[0-9]{10}$")
NDC_PATTERN = re.compile(r"^[0-9]{4,5}-?[0-9]{3,4}-?[0-9]{1,2}$")
ICD_PATTERN = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?$", re.IGNORECASE)
HCPCS_PATTERN = re.compile(r"^[A-V][0-9]{4}$", re.IGNORECASE)
DATE_PATTERN = re.compile(r"^(0?[1-9]|1[0-2])[\-/](0?[1-9]|[12][0-9]|3[01])[\-/](\d{2}|\d{4})$")
PHONE_PATTERN = re.compile(r"^\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$")
NUMERIC_PATTERN = re.compile(r"^[0-9]+$")
ALPHANUM_PATTERN = re.compile(r"^[A-Z0-9]+$", re.IGNORECASE)
SSN_PATTERN = re.compile(r"^(\d{3}-?\d{2}-?\d{4})$")
ZIP_PATTERN = re.compile(r"^\d{5}(?:-\d{4})?$")
MONEY_PATTERN = re.compile(r"^[\$]?\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?$")
TAXID_PATTERN = re.compile(r"^\d{2}-?\d{7}$")


def validate_npi(value: str) -> Tuple[bool, Dict[str, Any]]:
    digits = re.sub(r"[^0-9]", "", value)
    if len(digits) != 10:
        return False, {"reason": "length"}
    total = 0
    for idx, ch in enumerate(digits[::-1]):
        num = int(ch)
        if idx % 2 == 1:
            num *= 2
            if num > 9:
                num -= 9
        total += num
    if total % 10 != 0:
        return False, {"reason": "checksum"}
    return True, {"normalized": digits}


def validate_ndc(value: str) -> Tuple[bool, Dict[str, Any]]:
    if not value:
        return False, {"reason": "empty"}
    cleaned = value.replace(" ", "").upper()
    if NDC_PATTERN.match(cleaned):
        return True, {"normalized": cleaned}
    return False, {"reason": "format"}


def validate_icd(value: str) -> Tuple[bool, Dict[str, Any]]:
    if not value:
        return False, {"reason": "empty"}
    cleaned = value.replace(" ", "").upper()
    if ICD_PATTERN.match(cleaned):
        return True, {"normalized": cleaned}
    return False, {"reason": "format"}


def validate_hcpcs(value: str) -> Tuple[bool, Dict[str, Any]]:
    if not value:
        return False, {"reason": "empty"}
    cleaned = value.replace(" ", "").upper()
    if HCPCS_PATTERN.match(cleaned):
        return True, {"normalized": cleaned}
    return False, {"reason": "format"}


def validate_date(value: str) -> Tuple[bool, Dict[str, Any]]:
    if not value:
        return False, {"reason": "empty"}
    cleaned = value.strip()
    match = DATE_PATTERN.match(cleaned)
    if not match:
        return False, {"reason": "format"}
    month, day, year = match.groups()
    if len(year) == 2:
        year = "20" + year if int(year) < 50 else "19" + year
    normalized = f"{int(month):02d}/{int(day):02d}/{year}"
    return True, {"normalized": normalized}


def validate_phone(value: str) -> Tuple[bool, Dict[str, Any]]:
    if not value:
        return False, {"reason": "empty"}
    cleaned = value.strip()
    if PHONE_PATTERN.match(cleaned):
        digits = re.sub(r"[^0-9]", "", cleaned)
        normalized = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}" if len(digits) == 10 else cleaned
        return True, {"normalized": normalized}
    return False, {"reason": "format"}


def validate_member_id(value: str) -> Tuple[bool, Dict[str, Any]]:
    if not value:
        return False, {"reason": "empty"}
    cleaned = value.replace(" ", "").upper()
    if len(cleaned) >= 6 and len(cleaned) <= 15 and ALPHANUM_PATTERN.match(cleaned):
        return True, {"normalized": cleaned}
    return False, {"reason": "format"}


def validate_ssn(value: str) -> Tuple[bool, Dict[str, Any]]:
    if not value:
        return False, {"reason": "empty"}
    cleaned = value.strip()
    if SSN_PATTERN.match(cleaned):
        digits = re.sub(r"[^0-9]", "", cleaned)
        return True, {"normalized": f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"}
    return False, {"reason": "format"}


def validate_zip(value: str) -> Tuple[bool, Dict[str, Any]]:
    if not value:
        return False, {"reason": "empty"}
    cleaned = value.strip()
    if ZIP_PATTERN.match(cleaned):
        return True, {"normalized": cleaned}
    return False, {"reason": "format"}


def validate_money(value: str) -> Tuple[bool, Dict[str, Any]]:
    if not value:
        return False, {"reason": "empty"}
    cleaned = value.replace(" ", "").strip()
    if MONEY_PATTERN.match(cleaned):
        norm = cleaned
        if not norm.startswith("$"):
            norm = "$" + norm
        return True, {"normalized": norm}
    return False, {"reason": "format"}


def validate_numeric(value: str, min_len: int = 1, max_len: int = 20) -> Tuple[bool, Dict[str, Any]]:
    digits = re.sub(r"[^0-9]", "", value)
    if min_len <= len(digits) <= max_len:
        return True, {"normalized": digits}
    return False, {"reason": "length"}


def validate_tax_id(value: str) -> Tuple[bool, Dict[str, Any]]:
    if not value:
        return False, {"reason": "empty"}
    cleaned = value.strip()
    if TAXID_PATTERN.match(cleaned):
        digits = re.sub(r"[^0-9]", "", cleaned)
        return True, {"normalized": f"{digits[:2]}-{digits[2:]}"}
    return False, {"reason": "format"}


def validate_cpt(value: str) -> Tuple[bool, Dict[str, Any]]:
    if not value:
        return False, {"reason": "empty"}
    digits = re.sub(r"[^0-9]", "", value)
    if len(digits) == 5:
        return True, {"normalized": digits}
    return False, {"reason": "format"}


FIELD_VALIDATORS = {
    "npi": validate_npi,
    "ndc": validate_ndc,
    "icd": validate_icd,
    "icd10": validate_icd,
    "hcpcs": validate_hcpcs,
    "cpt": validate_cpt,
    "date": validate_date,
    "phone": validate_phone,
    "member_id": validate_member_id,
    "numeric": validate_numeric,
    "ssn": validate_ssn,
    "zip": validate_zip,
    "money": validate_money,
    "tax_id": validate_tax_id,
}


def validate_field(field_type: str, value: str) -> Tuple[bool, Dict[str, Any]]:
    """Run validator by type. Returns (passed, info)."""
    validator = FIELD_VALIDATORS.get(field_type)
    if not validator:
        return False, {"reason": "unknown_validator"}
    return validator(value or "")


def guess_field_type(label_text: Optional[str]) -> Optional[str]:
    """Heuristic mapping from label text to validator type."""
    if not label_text:
        return None
    text = label_text.lower()
    if "npi" in text:
        return "npi"
    if "ndc" in text or "drug" in text:
        return "ndc"
    if "icd" in text or "diagnosis" in text:
        return "icd"
    if "hcpcs" in text or "procedure" in text:
        return "hcpcs"
    if "date" in text or "dob" in text:
        return "date"
    if "phone" in text or "contact" in text or "tel" in text:
        return "phone"
    if "member" in text or "id" in text:
        return "member_id"
    if "zip" in text or "postal" in text:
        return "numeric"
    if "ssn" in text or "social" in text:
        return "ssn"
    if "amount" in text or "paid" in text or "charge" in text:
        return "money"
    if "tax" in text and "id" in text:
        return "tax_id"
    return None


# ---------------------------------------------------------------------------
# ValidationAgent
# ---------------------------------------------------------------------------

class ValidationAgent(BaseAgent):
    """Field validation and optional LLM QA. Runs after assembly."""

    def __init__(self, config: PipelineConfig):
        super().__init__("ValidationAgent")
        self.config = config

    async def initialize(self):
        self._initialized = True

    def validate_field(self, value: str, field_type: str) -> Tuple[bool, str]:
        """Validate a field value using validators."""
        if not value or not field_type:
            return True, ""
        passed, info = validate_field(field_type, value)
        if passed:
            return True, ""
        if info.get("reason") == "unknown_validator":
            return True, ""
        reason = info.get("reason", "format")
        return False, f"Invalid {field_type} ({reason})"

    async def llm_qa_check(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Run LLM QA check on extracted data."""
        if not self.config.enable_llm_qa:
            return []
        notes = []
        try:
            import requests
            fields_str = "\n".join([f"- {k}: {v}" for k, v in extracted_data.items() if v])
            prompt = f"""Review this medical form extraction for errors:
{fields_str}

List only obvious errors (max 3). If all looks good, say "OK"."""
            response = requests.post(
                f"http://{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": self.config.slm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200}
                },
                timeout=30
            )
            if response.ok:
                result = response.json().get("response", "").strip()
                if result and "ok" not in result.lower():
                    notes.append(result)
        except Exception:
            pass
        return notes

    async def process(self, blocks: List[DetectedBlock], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all data."""
        await self.initialize()
        validation_results = {
            "errors": [],
            "warnings": [],
            "qa_notes": []
        }
        for block in blocks:
            field_type = block.metadata.get("field_type")
            if field_type:
                valid, msg = self.validate_field(block.text, field_type)
                if not valid:
                    validation_results["errors"].append({
                        "field_id": block.id,
                        "message": msg
                    })
        qa_notes = await self.llm_qa_check(extracted_data)
        validation_results["qa_notes"] = qa_notes
        return validation_results

"""Field validators and normalizers for forms pipeline."""
from __future__ import annotations
import re
from typing import Dict, Any, Optional, Tuple

# Precompiled regex patterns
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


FIELD_VALIDATORS = {
    "npi": validate_npi,
    "ndc": validate_ndc,
    "icd": validate_icd,
    "icd10": validate_icd,
    "hcpcs": validate_hcpcs,
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
    validator = FIELD_VALIDATORS.get(field_type)
    if not validator:
        return False, {"reason": "unknown_validator"}
    return validator(value or "")


def guess_field_type(label_text: Optional[str]) -> Optional[str]:
    """Heuristic mapping from label text to field type."""
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
    if "zip" in text:
        return "zip"
    if "amount" in text or "paid" in text or "charge" in text:
        return "money"
    if "tax" in text and "id" in text:
        return "tax_id"
    return None

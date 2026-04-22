"""Quick offline validation of the blank-guard decision logic.

We replicate the EXACT decision tree from run_rescue_attempt in
rescue_strategies.py and exercise it against the 4 regressions the
user flagged in screenshots.  This avoids importing the full
pipeline (which pulls in torch/opencv and can segfault on some
builds).
"""
from __future__ import annotations
import sys


_DIGIT_RANGES = {
    "npi", "phone", "zip", "date", "date_range", "money", "currency",
    "tax_id", "cpt", "icd", "icd10", "hcpcs", "ndc", "member_id",
    "numeric", "ssn", "ein",
}
_TYPED = _DIGIT_RANGES | {"state"}


def _fake_validate(field_type: str, text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if field_type == "date":
        import re
        return bool(re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", t))
    if field_type == "phone":
        digits = [c for c in t if c.isdigit()]
        return len(digits) >= 10
    return True


def rescue_decision(
    field_type: str,
    old_text: str,
    candidate: str,
    blank_status: str = "uncertain",
    is_blank: bool = False,
    inner_ink: float = 0.0,
    ocr_engine: str = "florence2",
    old_confidence: float = 0.0,
):
    """Mirrors run_rescue_attempt's accept/reject branches."""
    new_valid = _fake_validate(field_type, candidate)

    # --- Degenerate-candidate guard -------------------------------
    stripped = (candidate or "").strip()
    alnum = "".join(ch for ch in stripped if ch.isalnum())
    if stripped and len(stripped) <= 3 and len(alnum) == 0:
        return False, "degenerate_candidate"

    if not candidate:
        return False, "no_candidate"

    # --- Blank guard ----------------------------------------------
    likely_blank_explicit = blank_status in {
        "blank", "blank_structural", "blank_ink",
        "blank_high", "blank_med",
        "cleared_as_template_leak", "cleared_as_unrescuable",
    }
    _INK_EMPTY = 0.08
    ink_available = inner_ink > 0.0
    ink_says_empty = ink_available and inner_ink < _INK_EMPTY
    engine_says_blank = any(
        tag in ocr_engine
        for tag in ("blank_fast", "blank_confirmed", "blank_structural")
    )
    v1_empty_signal = (not old_text) and (old_confidence < 0.05) and engine_says_blank
    likely_blank = (
        likely_blank_explicit
        or is_blank
        or v1_empty_signal
        or (not old_text and ink_says_empty)
    )
    if likely_blank and not old_text:
        is_typed = field_type in _TYPED
        typed_allowed = (
            is_typed
            and new_valid
            and (not ink_available or inner_ink >= _INK_EMPTY)
            and not likely_blank_explicit
            and not is_blank
            and not v1_empty_signal
        )
        if not typed_allowed:
            return False, "blank_guard"

    return True, "would_accept"


def main():
    # (case name, kwargs, expected_accept)
    cases = [
        (
            "11b_other_claim_id — Florence aggressive leaked DOB text",
            dict(field_type="text", old_text="", candidate="07 13 1984 X",
                 blank_status="uncertain", is_blank=True,
                 ocr_engine="blank_confirmed", old_confidence=0.0),
            False,
        ),
        (
            "9_other_insured_name — aggressive Florence leaked neighbors",
            dict(field_type="name", old_text="", candidate="44136 3011 F73-4413",
                 blank_status="uncertain", is_blank=True,
                 ocr_engine="blank_confirmed", old_confidence=0.0),
            False,
        ),
        (
            "14_date_of_illness — VLM hallucinated valid date",
            dict(field_type="date", old_text="", candidate="03/29/2018",
                 blank_status="uncertain", is_blank=True,
                 ocr_engine="blank_confirmed", old_confidence=0.0),
            False,
        ),
        (
            "22_resubmission_code — '...' degenerate",
            dict(field_type="text", old_text="", candidate="...",
                 blank_status="uncertain", is_blank=False,
                 ocr_engine="florence2", old_confidence=0.0),
            False,
        ),
        (
            "sanity: typed field with real ink should accept",
            dict(field_type="phone", old_text="", candidate="(555) 123-4567",
                 blank_status="uncertain", is_blank=False,
                 inner_ink=0.15),
            True,
        ),
        (
            "sanity: V1 low-conf empty without is_blank (engine says blank)",
            dict(field_type="date", old_text="", candidate="03/29/2018",
                 blank_status="uncertain", is_blank=False,
                 ocr_engine="blank_confirmed", old_confidence=0.0),
            False,
        ),
        (
            "sanity: typed field with real ink but engine tagged blank → reject",
            dict(field_type="date", old_text="", candidate="03/29/2018",
                 blank_status="uncertain", is_blank=False,
                 inner_ink=0.20,
                 ocr_engine="blank_confirmed", old_confidence=0.0),
            False,
        ),
    ]

    fails = 0
    for name, kw, expected in cases:
        accepted, reason = rescue_decision(**kw)
        ok = accepted == expected
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}")
        print(f"       accepted={accepted} expected={expected} reason={reason}")
        if not ok:
            fails += 1

    if fails == 0:
        print(f"\nAll {len(cases)} cases passed.")
        return 0
    print(f"\n{fails}/{len(cases)} FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

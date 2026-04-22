"""
Agentic rescue ladder.

Each field that fails validation gets walked down an ordered list of
strategies, one per rescue iteration.  The order is *per field type*
so handwriting-heavy fields and numeric fields try the engine that
actually helps them first:

    default / handwriting (text, name, address, city, diagnosis, ...)
        iter 0  →  VLM rescue (context disambiguation)
        iter 1  →  Florence-2 on template-subtracted crop + 2× upscale
        iter 2  →  Florence-2 aggressive (3× + CLAHE on grayscale)
        iter 3  →  deterministic cleaner + SLM symbolic normalisation

    numeric / enum (npi, phone, date, zip, money, tax_id, state, ...)
        iter 0  →  VLM rescue (digit context helps most)
        iter 1  →  Florence-2 on template-subtracted crop + 2× upscale
        iter 2  →  Florence-2 aggressive
        iter 3  →  PARSeq (scene-text, 23M params, ~30ms) — printed
                    digits/caps tend to agree with VLM cheaply
        iter 4  →  deterministic cleaner + SLM normalisation

CRITICAL BEHAVIOUR — every vision rescue strategy now runs on a
*template-subtracted* crop (``_get_subtracted_crop``) when the form
is a CMS-1500 scan.  This was previously disabled — the comment on
``_strategy_florence2_raw_upscale`` used to say "subtraction damages
stroke continuity" — but the April-2026 benchmark proved the
opposite: without subtraction Florence-2 / GOT-OCR read the red
printed labels ("PATIENT'S NAME", "NPI", "PICA", etc.) instead of
the handwriting, yielding 0/46 valid fields.  Alignment quality is
high enough today (AKAZE + outer-quad + piecewise) that subtraction
cleanly zeroes template pixels without shredding ink.

TrOCR and GOT-OCR 2.0 were retired from the default ladders based
on the same benchmark:
  * TrOCR over-regularised numeric fields into English words
  * GOT-OCR 2.0 was 2× slower than Florence-2 with no post-cleaning
    accuracy gain
Both stay registered in ``_STRATEGY_FN`` so the benchmark harness
and manual method overrides keep working.

After every attempt we re-run the typed validator.  We ACCEPT the
new value only if it validates OR if it looks more correct by
structural signals (longer, fewer hallucinations, higher confidence).
We REJECT otherwise and let the next iteration try a different
method — the field-type ladder guarantees successive calls hit
different architectures.

This module is deliberately decoupled from the graph nodes — it
exposes a single ``run_rescue_attempt`` function the node calls per
candidate, and a pair of registries (``RESCUE_LADDER`` for the
default order and ``ladder_for_field_type`` for the per-type overrides)
that the node uses to track exhaustion.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger("graph.rescue")


# Ordered ladder — iteration index → strategy name.  Keep this short;
# the graph caps iterations at ``len(RESCUE_LADDER)``.  This is the
# *default* order; for handwriting-heavy or typed fields we re-order
# via ``ladder_for_field_type`` below so the right model fires first.
#
# 2026-04 update — ladder slimmed down based on benchmark data:
#   * TrOCR: dropped (over-regularised numeric fields into English
#     words on the IAM prior).  Kept in ``_STRATEGY_FN`` for the
#     benchmark harness; no ladder references it.
#   * GOT-OCR 2.0: dropped from ladders.  On the 2-PDF benchmark it
#     was 2× slower than Florence-2, produced raw output for 100% of
#     fields but 0% valid after template/cleaning filters — same
#     failure mode as Florence-2 (reads template labels), no
#     compensating accuracy gain.  Kept in ``_STRATEGY_FN`` so the
#     benchmark and the `got_ocr` method override still work.
#   * PARSeq: kept implemented for experiments/benchmarking, but removed
#     from the default production ladders because it underperforms on our
#     handwritten CMS-1500 crops and can degrade numeric stability.
RESCUE_LADDER: List[str] = [
    "vlm",
    "florence2_raw_upscale",
    "florence2_aggressive",
    "slm_normalize",
]


# Handwriting-heavy fields: names, addresses, cities, diagnoses,
# notes.  Florence-2's caption-head on a template-subtracted crop is
# our best open model on this form's handwriting; a second Florence-2
# pass with different preprocessing (CLAHE, wider pad) gives diverse
# errors cheaply.  VLM sits first because context ("this is a patient
# name") resolves a lot of ambiguous strokes at zero extra cost.
# PARSeq is NOT in this ladder — it was trained on scene text, not
# handwriting, so it underperforms Florence-2 on cursive.
_LADDER_HANDWRITING: List[str] = [
    "vlm",
    "florence2_raw_upscale",
    "florence2_aggressive",
    "slm_normalize",
]


# Short numeric / highly-structured fields: NPI, phone, zip, date,
# money, tax-id, CPT, ICD-10.  The VLM wins because it uses language
# context to pick between visually-ambiguous digits (0↔6, 1↔7, etc).
# Florence-2 variants follow. GOT-OCR 2.0 is used as a diverse fallback
# (different architecture/training corpus) before the final SLM clean-up.
_LADDER_NUMERIC: List[str] = [
    "vlm",
    "florence2_raw_upscale",
    "florence2_aggressive",
    "got_ocr",
    "slm_normalize",
]


# Two-letter enum fields (state).  VLM first (it knows "FL" is
# Florida), then Florence-2, then GOT-OCR 2.0, finally SLM fallback.
# ``_clean_state`` downstream extracts the
# 2-letter code from noisy OCR so the ladder only needs a rough
# candidate here.
_LADDER_STATE: List[str] = [
    "vlm",
    "florence2_raw_upscale",
    "florence2_aggressive",
    "got_ocr",
    "slm_normalize",
]


_FIELD_TYPE_HANDWRITING = {
    "text", "address", "name", "city", "diagnosis", "notes", "signature",
}
_FIELD_TYPE_NUMERIC = {
    "npi", "phone", "zip", "date", "date_range", "money", "currency",
    "tax_id", "cpt", "icd", "icd10", "hcpcs", "ndc", "member_id",
    "numeric", "ssn", "ein",
}
_FIELD_TYPE_ENUM = {"state"}


def ladder_for_field_type(field_type: str) -> List[str]:
    """Return the strategy order to try for a given field type.

    Falls back to the default ``RESCUE_LADDER`` when the field has no
    type, an unknown type, or a checkbox/table type (those never reach
    here in practice).  Keeping this keyed off the schema's
    ``field_type`` means adding a new form never requires touching
    rescue code — populate the schema and the right ladder is picked.
    """
    ft = (field_type or "").strip().lower()
    if ft in _FIELD_TYPE_HANDWRITING:
        return _LADDER_HANDWRITING
    if ft in _FIELD_TYPE_NUMERIC:
        return _LADDER_NUMERIC
    if ft in _FIELD_TYPE_ENUM:
        return _LADDER_STATE
    return RESCUE_LADDER


@dataclass
class RescueAttempt:
    field_id: str
    method: str
    iteration: int
    old_text: str
    new_text: str
    old_valid: bool
    new_valid: bool
    accepted: bool
    reason: str
    confidence: float


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

_SENTINEL_TOKEN_RE = re.compile(
    r"</?\s*(pad|s|eos|bos|unk|mask|sep|cls)\s*/?>", re.IGNORECASE,
)


def _strip_sentinels(text: str) -> str:
    if not text:
        return ""
    cleaned = _SENTINEL_TOKEN_RE.sub("", text)
    cleaned = cleaned.replace("\x00", "").replace("\ufffd", "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


# Generic STRUCTURAL patterns for template chrome that leaks in when
# the printed template is not fully subtracted from a crop.  These
# describe *shapes* that templates tend to take (field number only,
# number + all-caps caption, run of ALL-CAPS short tokens) rather than
# any particular form's vocabulary.  Works for CMS-1500, UB-04, W-2,
# 1040, or anything else we plug a schema in for.
_TEMPLATE_SHAPE_PATTERNS = [
    # Standalone field number: "1.", "17a", "28."
    re.compile(r"^\s*\d{1,2}[a-z]?\.?\s*$", re.MULTILINE),
    # Field-number + ALL-CAPS caption: "28. TOTAL CHARGE"
    re.compile(r"^\s*\d{1,2}[a-z]?\.?\s+[A-Z][A-Z\s&/\-]{3,}\s*$",
               re.MULTILINE),
    # Parenthetical hint that templates use, e.g. "(Include Area Code)",
    # "(Designed by ...)".  A raw parenthetical, no alphanumeric outside
    # the parens, is almost never user data.
    re.compile(r"^\s*\([A-Za-z][^)]{3,}\)\s*$", re.MULTILINE),
]


# Token-overlap threshold: if the candidate text shares at least this
# many tokens with any schema label (after normalisation), treat as a
# template leak.  Tuned so "36. Rsvd.tor NUCC US" (tokens: {36, rsvd,
# tor, nucc, us}) matches "RESERVED FOR NUCC USE (30)" label tokens
# ({reserved, for, nucc, use, 30}) via the 2-token overlap {nucc, use}
# — Florence's OCR of "US" for "USE" is counted as a match below.
_LABEL_OVERLAP_MIN_TOKENS = 2

# Schema label tokens, pre-split per form.  Populated lazily next to
# the regex cache.
_SCHEMA_LABEL_TOKEN_SETS: Dict[str, List[set]] = {}

# Per-form page-level "chrome" vocabulary — tokens from the
# ``template_chrome`` array in each schema JSON.  Populated lazily and
# cached per form.  This is what REPLACES the hardcoded PICA/CARRIER
# keyword list that used to live in code: the template vocabulary now
# ships as DATA next to the schema, and adding a new form = dropping
# a schema file, nothing to change in Python.
_CHROME_TOKEN_CACHE: Dict[str, set] = {}


# Schema-driven label cache.  We lazily build a compiled regex union
# from the active form's field labels the first time we need it, so
# "TELEPHONE (Include Area Code)" isn't baked into the rescue module —
# it comes from data/schemas/<form>.json.  Keeps the logic generic.
_SCHEMA_LABEL_CACHE: Dict[str, "re.Pattern"] = {}


def _normalize_for_label_match(text: str) -> str:
    """Normalise a candidate string so punctuation + case + whitespace
    differences between the schema label and the OCR output stop
    defeating the match.

    The schema has "ORIGINAL REF NO" but Florence-2 outputs
    "ORIGINAL REF. NO" (extra period). ``re.escape`` on the schema label
    then fails to match.  We collapse everything to lower-case alnum
    tokens joined by single spaces so both sides become
    ``"original ref no"``.
    """
    if not text:
        return ""
    # Drop everything that isn't a letter or a digit; collapse runs of
    # those drops into a single space.  Keeps tokens apart but erases
    # punctuation noise.
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    return " ".join(tokens).lower().strip()


def _load_schema_for_form(form_type: str) -> Optional[Dict[str, Any]]:
    """Load and return the parsed schema dict for ``form_type``, or None."""
    if not form_type:
        return None
    key = form_type.lower().replace("_", "-")
    try:
        from pathlib import Path as _Path
        import json as _json
        schema_path = _Path(__file__).parent.parent.parent.parent \
            / "data" / "schemas" / f"{key}.json"
        if not schema_path.exists():
            alt = schema_path.with_name(
                schema_path.stem.replace("-", "") + ".json",
            )
            schema_path = alt if alt.exists() else schema_path
        if not schema_path.exists():
            return None
        with open(schema_path) as fh:
            return _json.load(fh)
    except Exception:
        return None


def _schema_label_pattern(form_type: str) -> Optional["re.Pattern"]:
    """Compile a union regex of schema field labels for `form_type`.

    Returns None if the schema isn't available.  Called once per form
    per process (result is cached).

    The candidate text is expected to have been passed through
    ``_normalize_for_label_match`` before being matched against this
    pattern, so both sides are already lowercase alnum-only
    space-separated.  That's what lets "ORIGINAL REF NO" (schema) match
    "ORIGINAL REF. NO" (Florence output).
    """
    if not form_type:
        return None
    key = form_type.lower().replace("_", "-")
    if key in _SCHEMA_LABEL_CACHE:
        return _SCHEMA_LABEL_CACHE[key]

    schema = _load_schema_for_form(form_type)
    if schema is None:
        _SCHEMA_LABEL_CACHE[key] = None  # type: ignore[assignment]
        _SCHEMA_LABEL_TOKEN_SETS[key] = []
        return None

    try:
        labels: List[str] = []
        token_sets: List[set] = []
        seen: set = set()
        for f in schema.get("fields", []):
            lab = (f.get("label") or "").strip()
            if not lab or len(lab) < 3:
                continue
            # Keep only the SHORT leading chunk of each label — the
            # part that actually appears as template chrome inside a
            # crop.  "PATIENT'S NAME (Last Name, First Name, Middle
            # Initial)" → "PATIENT'S NAME".
            short = re.split(r"\(|,|\|", lab, maxsplit=1)[0].strip()
            normalised = _normalize_for_label_match(short)
            # Require at least 2 tokens OR >= 4 chars so we don't match
            # common single words like "SEX" or "NO" inside real data.
            if not normalised:
                continue
            tokens = [t for t in normalised.split() if len(t) >= 2]
            token_count = len(tokens)
            if token_count < 2 and len(normalised) < 4:
                continue
            if normalised in seen:
                continue
            seen.add(normalised)
            # Escape the NORMALISED form so any remaining
            # regex-metacharacters (there shouldn't be any after
            # normalisation, but belt-and-braces) are safe.
            labels.append(re.escape(normalised))
            # Token set (for fuzzy overlap), excluding the most generic
            # filler words that show up in many labels and would over-
            # trigger ("the", "of", "or", "and", "for"...).
            tok_set = set(tokens) - {
                "the", "of", "or", "and", "for", "to", "by", "no",
                "from", "name", "last", "first", "middle", "initial",
                "include", "area", "code", "any", "other", "use",
            }
            if tok_set:
                token_sets.append(tok_set)
        if not labels:
            _SCHEMA_LABEL_CACHE[key] = None  # type: ignore[assignment]
            _SCHEMA_LABEL_TOKEN_SETS[key] = []
            return None
        # Longest-first alternation so "billing provider info"
        # matches before the shorter "billing provider" prefix.
        labels.sort(key=len, reverse=True)
        # Anchored to word boundary (space or string edge) on both
        # sides of the normalised candidate.
        pat = re.compile(
            r"(?:^|\s)(?:" + "|".join(labels) + r")(?:$|\s)",
        )
        _SCHEMA_LABEL_CACHE[key] = pat
        _SCHEMA_LABEL_TOKEN_SETS[key] = token_sets
        return pat
    except Exception:
        _SCHEMA_LABEL_CACHE[key] = None  # type: ignore[assignment]
        _SCHEMA_LABEL_TOKEN_SETS[key] = []
        return None


def _schema_label_token_sets(form_type: str) -> List[set]:
    """Return the cached per-label token sets for `form_type`."""
    if not form_type:
        return []
    key = form_type.lower().replace("_", "-")
    if key not in _SCHEMA_LABEL_TOKEN_SETS:
        # Side effect: populates _SCHEMA_LABEL_TOKEN_SETS as well.
        _schema_label_pattern(form_type)
    return _SCHEMA_LABEL_TOKEN_SETS.get(key, [])


def _template_chrome_tokens(form_type: str) -> set:
    """Return the set of lowercase page-level chrome tokens for the form.

    Reads ``template_chrome`` from ``data/schemas/<form>.json``.  The
    file explicitly enumerates every piece of printed-on-template text
    that isn't tied to a single field bbox (page headers, footers,
    column headings, boilerplate).  Lives in the schema next to the
    field definitions so adding a new form = adding one JSON file.

    Returns an empty set when the schema or the key is missing — the
    filter then falls back to the structural + schema-label stages
    alone, which is still safe on any form.
    """
    if not form_type:
        return set()
    key = form_type.lower().replace("_", "-")
    if key in _CHROME_TOKEN_CACHE:
        return _CHROME_TOKEN_CACHE[key]
    schema = _load_schema_for_form(form_type) or {}
    raw = schema.get("template_chrome") or []
    tokens: set = set()
    for phrase in raw:
        if not isinstance(phrase, str):
            continue
        # Split each chrome phrase into alphanumeric tokens the same way
        # we tokenise candidates, so a phrase like "APPROVED BY NUCC"
        # adds {approved, by, nucc} — the candidate side then strips
        # common fillers (by/of/the...) automatically via
        # ``_candidate_token_set``'s abbrev map.
        for tok in re.findall(r"[A-Za-z0-9]+", phrase.lower()):
            if len(tok) >= 2:
                tokens.add(tok)
    # A handful of stop-tokens that appear in chrome phrases but are
    # ALSO common in real values — e.g. an address might literally
    # contain "GROUP HEALTH" so we don't want "group" alone to flag
    # it.  Drop them so the chrome check requires co-occurrence of
    # a more distinctive token.
    tokens -= {
        "the", "of", "or", "and", "for", "to", "by", "from",
        "in", "on", "at", "a", "an",
    }
    _CHROME_TOKEN_CACHE[key] = tokens
    return tokens


# Cached per-form list of normalised chrome PHRASES (not just tokens).
# Used by ``_looks_like_template_label`` for phrase containment, which
# catches chrome like "APPROVED BY NUCC" that contains a stop-token
# ("by") and therefore can't be matched by the stripped token set
# alone.  Populated lazily next to ``_template_chrome_tokens``.
_CHROME_PHRASE_CACHE: Dict[str, List[str]] = {}


def _template_chrome_phrases(form_type: str) -> List[str]:
    """Return normalised (lowercase alnum) chrome phrases for the form.

    Same data source as ``_template_chrome_tokens``.  Returned strings
    are ``_normalize_for_label_match``-formatted so phrase containment
    is punctuation-insensitive on both sides.
    """
    if not form_type:
        return []
    key = form_type.lower().replace("_", "-")
    if key in _CHROME_PHRASE_CACHE:
        return _CHROME_PHRASE_CACHE[key]
    schema = _load_schema_for_form(form_type) or {}
    raw = schema.get("template_chrome") or []
    phrases: List[str] = []
    for phrase in raw:
        if not isinstance(phrase, str):
            continue
        norm = _normalize_for_label_match(phrase)
        # Skip noise-length phrases; a 2-char chrome like "DD" would
        # otherwise flag every date-containing candidate.
        if len(norm) >= 4:
            phrases.append(norm)
    _CHROME_PHRASE_CACHE[key] = phrases
    return phrases


def _candidate_token_set(text: str) -> set:
    """Tokenise the candidate the same way schema labels are tokenised.

    Uses lowercase alnum tokens, drops 1-character tokens, and treats
    common OCR-shortened forms as their full counterparts so Florence
    misreads ("US" → "USE", "ADDR" → "ADDRESS") still join the right
    label.  This is what lets "36. Rsvd.tor NUCC US" overlap with the
    label "RESERVED FOR NUCC USE".
    """
    if not text:
        return set()
    # Map common Florence-2 truncations / abbreviations back to the
    # full schema token they almost always come from.  Conservative:
    # every entry is unambiguous in form-chrome context.
    abbrev = {
        "us": "use",
        "addr": "address",
        "no": "number",
        "tel": "telephone",
        "ph": "phone",
        "rsvd": "reserved",
        "rsv": "reserved",
        "rsd": "reserved",
        "rsrvd": "reserved",
        "nucc": "nucc",
        "ins": "insured",
        "pat": "patient",
        "dob": "date",
        "ref": "reference",
    }
    out: set = set()
    for tok in re.findall(r"[A-Za-z]+", text.lower()):
        if len(tok) < 2:
            continue
        out.add(abbrev.get(tok, tok))
    return out


def _looks_like_template_label(text: str, form_type: str = "") -> bool:
    """True if the candidate looks like leaked template chrome, not data.

    Three DATA-DRIVEN stages (cheapest first), no form-specific
    keywords baked into Python:

      1. Generic structural shapes that templates take on any form
         (field-number only, number + ALL-CAPS caption, lone
         parenthetical hints) — form-agnostic.
      2. Schema-driven page-chrome vocabulary from
         ``schema.template_chrome`` — e.g. "PICA", "CARRIER",
         "APPROVED BY NUCC".  The list lives in the schema JSON so
         adding a new form means adding a JSON file, not a Python
         edit.
      3. Schema-driven EXACT label match on the normalised candidate
         (handles OCR punctuation noise like "ORIGINAL REF. NO").
      4. Schema-driven FUZZY token-overlap match — covers OCR
         distortions like "36. Rsvd.tor NUCC US" sharing {nucc, use}
         with the "RESERVED FOR NUCC USE" label.
      5. Structural "mostly-short ALL-CAPS multi-line" fallback for
         multi-line leaks like "QUAL / 17a / 17b / NPI".
    """
    if not text:
        return False
    normalized = text.strip()
    if not normalized:
        return False

    stripped = normalized.strip(" ()[]{}.,:;-_*\"'")
    candidate = stripped or normalized

    # (1) Structural shapes — run on the ORIGINAL (pre-strip) text so
    # parenthetical-hint patterns like "(Include Area Code)" still
    # match.  The stripped form is used by later stages.
    for pat in _TEMPLATE_SHAPE_PATTERNS:
        if pat.search(normalized) or pat.search(candidate):
            return True

    # (2) DATA-DRIVEN page-chrome vocabulary for this form.  Two checks:
    #     (a) Phrase containment: does the normalised candidate contain
    #         any full chrome phrase from the schema?  Catches
    #         "APPROVED BY NUCC" even though "by" is a stop-token in
    #         the per-token set — the multi-word phrase as a whole is
    #         still unique to the template.
    #     (b) Token-overlap: for leaks where the candidate is a
    #         single-word chrome token ("PICA") or a lightly-mangled
    #         multi-word phrase — we require the non-chrome residue
    #         to be empty (allowing only digits) so a real address
    #         like "123 Health St" still passes.
    cand_tokens = _candidate_token_set(candidate)
    chrome_tokens = _template_chrome_tokens(form_type)
    if chrome_tokens and len(candidate) <= 80:
        # (a) Phrase containment — compare on normalised alnum space.
        cand_alnum = _normalize_for_label_match(candidate)
        chrome_phrases = _template_chrome_phrases(form_type)
        for phrase_alnum in chrome_phrases:
            if len(phrase_alnum) >= 6 and phrase_alnum in cand_alnum:
                return True
        # (b) Token-overlap with residue check.
        if cand_tokens:
            non_chrome = cand_tokens - chrome_tokens
            non_chrome = {t for t in non_chrome if not t.isdigit()}
            if not non_chrome and (cand_tokens & chrome_tokens):
                return True

    # (3) Schema-driven exact label match on the normalised form.
    if form_type:
        sp = _schema_label_pattern(form_type)
        if sp is not None:
            alnum = _normalize_for_label_match(candidate)
            if alnum and sp.search(" " + alnum + " "):
                return True

    # (4) Schema-driven fuzzy token overlap — catches OCR distortions
    # of printed labels.  Cheap because the candidate set is small
    # (typically < 8 alnum tokens) and we cap label sets ahead of time.
    if form_type and cand_tokens:
        token_sets = _schema_label_token_sets(form_type)
        if token_sets:
            for tok_set in token_sets:
                # Only consider labels whose footprint is comparable to
                # the candidate's — a 2-token overlap with a 12-token
                # label is too lenient.
                if not tok_set:
                    continue
                overlap = cand_tokens & tok_set
                if len(overlap) >= _LABEL_OVERLAP_MIN_TOKENS:
                    # Require the overlap to dominate either side so a
                    # real value like "Florida Medical Group of NUCC"
                    # doesn't get flagged on a single shared token.
                    cand_share = len(overlap) / max(1, len(cand_tokens))
                    label_share = len(overlap) / max(1, len(tok_set))
                    if cand_share >= 0.5 or label_share >= 0.5:
                        return True

    # (5) Multi-line mostly-short-ALL-CAPS → probably template chrome.
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
    if len(lines) >= 2:
        short_count = sum(1 for ln in lines if len(ln) <= 5)
        if short_count / len(lines) >= 0.75:
            return True

    return False


# Per-field-type plausible digit count ranges.  We only accept a candidate
# via the `more_digits` heuristic when its digit count lands in this range;
# otherwise we're likely picking up a longer but meaningless string
# (template labels, multi-field leaks, etc.).
_DIGIT_RANGES: Dict[str, Tuple[int, int]] = {
    "phone": (7, 14),
    "npi": (9, 10),
    "tax_id": (9, 10),
    "zip": (5, 9),
    "date": (4, 8),
    "date_range": (4, 16),
    "money": (1, 10),
}


def _digit_count(text: str) -> int:
    return len(re.sub(r"\D", "", text or ""))


# ---------------------------------------------------------------------- #
# Deterministic, type-aware cleaners — the cheapest rescue tier.
# Run BEFORE any LLM call.  If a cleaner yields a validator-passing
# value, the SLM call is skipped entirely.
# ---------------------------------------------------------------------- #

_DATE_DIGIT_RUN = re.compile(r"(\d{1,2})\D(\d{1,2})\D(\d{2,4})")
_PHONE_ALL_DIGITS = re.compile(r"\d+")


def _clean_date(text: str) -> str:
    """Normalize whatever a rescue produced into MM/DD/YYYY (or ``""``)."""
    if not text:
        return ""
    # Replace common separator garbage with slashes, strip trailing junk.
    t = re.sub(r"[_\-\.\s]+", "/", text.strip())
    t = re.sub(r"[^\d/]+$", "", t)  # drop trailing letters like trailing 'M'
    t = re.sub(r"^[^\d]+", "", t)
    # Also try the digit-run fallback.
    m = _DATE_DIGIT_RUN.search(t)
    if not m:
        m = _DATE_DIGIT_RUN.search(re.sub(r"\D", "/", text))
    if not m:
        return ""
    mm, dd, yy = m.group(1), m.group(2), m.group(3)
    try:
        mm_i, dd_i = int(mm), int(dd)
        if not (1 <= mm_i <= 12 and 1 <= dd_i <= 31):
            return ""
    except ValueError:
        return ""
    if len(yy) == 2:
        # Mirror the validator's 2-digit-year rollover (< 50 → 20xx).
        yy_i = int(yy)
        yy = ("20" if yy_i < 50 else "19") + yy
    if len(yy) != 4:
        return ""
    return f"{int(mm):02d}/{int(dd):02d}/{yy}"


def _clean_phone(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    if not digits:
        return ""
    # Drop a leading '1' country code if present.
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        # Try to pick the best 10-digit substring.
        best = None
        for i in range(len(digits) - 9):
            cand = digits[i:i + 10]
            if cand[:3] != "000":
                best = cand
                break
        if not best:
            return ""
        digits = best
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def _clean_npi(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    return digits if len(digits) == 10 else ""


def _clean_zip(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    if len(digits) == 5:
        return digits
    if len(digits) == 9:
        return f"{digits[:5]}-{digits[5:]}"
    return ""


_MONEY_TOKEN_RE = re.compile(
    r"\$?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)",
)


def _clean_money(text: str) -> str:
    if not text:
        return ""
    # Find the FIRST plausible money-shaped token.  Handles cases like
    # '230$5000100' (two concatenated values), '$230.00 junk', '28.0$',
    # '1,234.56 extra text', etc. by picking the left-most match.
    m = _MONEY_TOKEN_RE.search(text)
    if m:
        raw = m.group(1).replace(",", "")
    else:
        raw = re.sub(r"[^0-9.]", "", text)
    if not raw:
        return ""
    # If we have multiple dots, keep the last one.
    if raw.count(".") > 1:
        parts = raw.split(".")
        raw = "".join(parts[:-1]) + "." + parts[-1]
    # Left-pad missing cents.
    if "." in raw:
        whole, cents = raw.split(".", 1)
        cents = (cents + "00")[:2]
        raw = f"{whole or '0'}.{cents}"
    else:
        # Guard against absurdly long integer-only strings (6+ digits
        # in a CMS-1500 charge field almost always means concatenated
        # garbage; keep only the first 6 digits in that case).
        if len(raw) > 6:
            raw = raw[:6]
        raw = f"{raw}.00"
    return raw


def _clean_tax_id(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    if len(digits) == 9:
        return f"{digits[:2]}-{digits[2:]}"
    return ""


def _clean_state(text: str) -> str:
    """Extract a valid US state abbreviation from OCR noise.

    Florence-2 and TrOCR both mangle the tiny state boxes — we've seen
    ``"سہ"`` (arabic-looking ligatures), ``"FI."``, ``"F1"``, ``"fl "``
    etc.  The typed validator will flag any of those, so here we try
    to rescue a real code from what's still recognisable:

      1. If the cleaned A-Z-only form is already a valid 2-letter code,
         keep it.
      2. Otherwise try the first pair of contiguous letters in the
         text — e.g. ``"FL12345"`` → ``"FL"``.
      3. Otherwise try each 2-letter window and return the first one
         that's a valid US state code.

    Returning ``""`` means "no candidate found"; the rescue ladder will
    then leave the raw OCR in place so the user can see what the OCR
    read.
    """
    if not text:
        return ""
    # Import locally to avoid circulars — validators imports rescue
    # helpers via the agent wiring.
    try:
        from src.pipelines.validators.validation import US_STATE_CODES
    except Exception:
        return ""
    cleaned = re.sub(r"[^A-Za-z]", "", text).upper()
    if len(cleaned) == 2 and cleaned in US_STATE_CODES:
        return cleaned
    # Try a sliding 2-character window over the alphabetic chars —
    # first match wins.
    for i in range(len(cleaned) - 1):
        pair = cleaned[i:i + 2]
        if pair in US_STATE_CODES:
            return pair
    return ""


_CLEANERS: Dict[str, Callable[[str], str]] = {
    "date": _clean_date,
    "date_range": _clean_date,
    "phone": _clean_phone,
    "npi": _clean_npi,
    "zip": _clean_zip,
    "money": _clean_money,
    "tax_id": _clean_tax_id,
    "state": _clean_state,
}


def _deterministic_clean(text: str, field_type: str) -> str:
    fn = _CLEANERS.get(field_type)
    if not fn or not text:
        return ""
    try:
        return fn(text) or ""
    except Exception:
        return ""


# NOTE: This function is intentionally CONSERVATIVE.  Earlier versions
# used digit-count and alpha-ratio heuristics to clear "garbage" values
# after the rescue ladder, but that turned out to destroy too much
# legitimate partial OCR output (e.g. a 4-digit year when the scan cut
# off the day).  We now clear ONLY on exact template-label hits, which
# is the one case where we know for certain that the text is leaked
# chrome rather than real data.
def _looks_like_garbage(text: str, field_type: str, form_type: str = "") -> bool:
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    if _looks_like_template_label(t, form_type):
        return True
    return False


def _post_rescue_sanitize(
    blocks, extracted: Dict[str, str],
    rescue_history: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Final sweep over typed fields after the rescue ladder is done.

    Three behaviors, in increasing severity:

      1. REPAIR — if the deterministic cleaner can turn the current
         text into a validator-passing value (e.g. '12_10_2025' →
         '12/10/2025'), accept the repair.

      2. CLEAR (template leak) — if the remaining text looks like a
         printed form label (punctuation-insensitive, schema-driven),
         drop it.

      3. CLEAR (exhausted rescue + impossible digit count) — if the
         field is typed (npi / phone / date / zip / tax_id / money),
         the rescue ladder already tried ≥ 1 method (so we're not
         short-circuiting real OCR), the validator still rejects the
         current text, AND the digit count is well below the minimum
         required for that field type, the current value is almost
         certainly OCR garbage (e.g. "1%" for an NPI).  We clear it
         instead of shipping a confidently-wrong value.

    Everything else is left alone so the user still sees the
    extraction attempt and the validator error pinpoints the issue.
    """
    rescue_history = rescue_history or {}
    cleared: List[str] = []
    for b in blocks:
        btype = getattr(b, "block_type", None)
        if btype is not None and hasattr(btype, "value"):
            if btype.value in ("checkbox", "signature", "table"):
                continue
        meta = b.metadata or {}
        field_type = str(meta.get("field_type", "") or "").lower()
        form_type = str(meta.get("form_type", "") or "")
        text = str(b.text or "").strip()
        if not text:
            continue

        validator_name = _validator_name_for(field_type)
        if validator_name and _run_validator(validator_name, text):
            continue

        if field_type in _CLEANERS:
            repaired = _deterministic_clean(text, field_type)
            if repaired and _run_validator(validator_name, repaired):
                b.text = repaired
                b.metadata["rescue_method"] = "final_cleanup"
                b.metadata["ocr_engine"] = "rescue_final_cleanup"
                extracted[b.id] = repaired
                continue

        # Template-leak clear (punctuation-insensitive schema match).
        if _looks_like_template_label(text, form_type):
            cleared.append(b.id)
            b.metadata["rescue_previous_text"] = text
            b.metadata["blank_status"] = "cleared_as_template_leak"
            b.metadata["rescue_method"] = "final_cleanup_clear"
            b.text = ""
            extracted.pop(b.id, None)
            continue

        # Exhausted-rescue clear for typed fields with impossible
        # digit counts.  Only fires when the rescue ladder has already
        # tried at least one method — otherwise we'd clear fields
        # before Florence-2 had a chance at them.
        attempts = len(rescue_history.get(b.id, []))
        if attempts == 0 or field_type not in _DIGIT_RANGES:
            continue
        digits = _digit_count(text)
        min_digits = _DIGIT_RANGES[field_type][0]
        # We require the current value to be at LESS than half the
        # minimum digit count, AND short in length overall, before we
        # call it garbage.  "1%" has 1 digit for an NPI that needs 9
        # → cleared.  "12345" for an NPI is at 5, which is more than
        # half of 9 → left alone for the user to notice.
        if digits < max(1, min_digits // 2) and len(text) <= 6:
            cleared.append(b.id)
            b.metadata["rescue_previous_text"] = text
            b.metadata["blank_status"] = "cleared_as_unrescuable"
            b.metadata["rescue_method"] = "final_cleanup_clear_typed"
            b.metadata["rescue_attempts_before_clear"] = attempts
            b.text = ""
            extracted.pop(b.id, None)
    return extracted, cleared


def _validator_name_for(field_type: str) -> Optional[str]:
    # Maps schema ``field_type`` strings onto validator names in the
    # shared ``FIELD_VALIDATORS`` registry.  ``state`` routes to the
    # USPS code validator so OCR noise like broken glyphs automatically
    # fails validation and triggers the rescue ladder, even though the
    # rescue ladder itself doesn't (yet) do a custom cleaner for
    # state codes.
    return {
        "date": "date",
        "date_range": "date",
        "phone": "phone",
        "npi": "npi",
        "zip": "zip",
        "icd10": "icd10",
        "money": "money",
        "tax_id": "tax_id",
        "state": "state",
    }.get(field_type)


def _run_validator(validator_name: Optional[str], value: str) -> bool:
    if not validator_name:
        return True
    try:
        from src.pipelines.validators import validate_field as typed_validate
        ok, _ = typed_validate(validator_name, value or "")
        return bool(ok)
    except Exception:
        return True


def _get_crop(
    image: np.ndarray, bbox: Tuple[float, float, float, float], pad_ratio: float = 0.06,
) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    pad = max(6, int(max(x1 - x0, y1 - y0) * pad_ratio))
    x0p, y0p = max(0, x0 - pad), max(0, y0 - pad)
    x1p, y1p = min(w, x1 + pad), min(h, y1 + pad)
    if x1p <= x0p + 2 or y1p <= y0p + 2:
        return None
    crop = image[y0p:y1p, x0p:x1p]
    if crop.size == 0:
        return None
    return crop


def _upscale(crop: np.ndarray, factor: float = 2.0) -> np.ndarray:
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    new_w = int(round(w * factor))
    new_h = int(round(h * factor))
    if new_w < 8 or new_h < 8:
        return crop
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _get_subtracted_crop(
    pipeline,
    block,
    image: np.ndarray,
    pad_ratio: float = 0.04,
) -> Optional[np.ndarray]:
    """Return a template-subtracted crop for this block, or the raw crop.

    Why this exists:
      The V1 ``OCRAgent.process_block`` already runs ``_template_subtract``
      before calling Florence-2 (see agents/ocr.py).  The rescue ladder
      historically did NOT — earlier strategies passed the raw aligned
      crop to Florence-2/VLM, and the comment on ``_strategy_florence2_raw_upscale``
      even notes "template-subtraction damages stroke continuity" as a
      justification for skipping it.  That comment was written when
      alignment quality was much lower; today (with AKAZE + outer-quad
      fallback + piecewise warp) the alignment is tight enough that
      subtraction reliably whitens template pixels without shredding
      strokes.  The recent 2-PDF benchmark showed Florence-2 producing
      *zero* valid outputs on raw crops because it kept reading the
      printed field labels ("PATIENT'S NAME", "INSURED'S ID NUMBER",
      etc.) instead of the handwriting.

      So: when we're on a CMS-1500 scan AND a template is available,
      feed the rescue engine the subtracted crop.  Otherwise fall back
      to the raw crop (e.g. non-CMS forms, or crops outside the template).

    The caller can still do its own upscale / preprocessing afterwards.
    We also keep the deterministic cleaner + template-label filter in
    place downstream so if subtraction IS hurting a given field we can
    still reject its garbage output.

    ``pad_ratio`` is kept small by default because we don't want to pad
    INTO neighbouring template labels after subtraction (the
    neighbouring label pixels wouldn't be whitened — they'd still read
    as ink to Florence-2).
    """
    bbox = (block.metadata or {}).get("original_bbox") or block.bbox
    raw = _get_crop(image, bbox, pad_ratio=pad_ratio)
    if raw is None:
        return None

    cfg = getattr(pipeline, "config", None)
    if not getattr(cfg, "rescue_use_template_subtract", True):
        return raw

    form_type = str((block.metadata or {}).get("form_type", "") or "").lower()
    is_cms = form_type in ("cms-1500", "cms1500")
    if not is_cms:
        return raw

    try:
        template_rgb = pipeline.ocr_agent._get_template_rgb(form_type)
    except Exception:
        template_rgb = None
    if template_rgb is None:
        return raw

    # We have to re-derive the bbox WITH the same padding _get_crop used
    # so the template region aligns with the raw crop.  _get_crop used
    # pad = max(6, int(max(w, h) * pad_ratio)); recompute it.
    try:
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        pad = max(6, int(max(x1 - x0, y1 - y0) * pad_ratio))
        h_img, w_img = image.shape[:2]
        x0p = max(0, x0 - pad)
        y0p = max(0, y0 - pad)
        x1p = min(w_img, x1 + pad)
        y1p = min(h_img, y1 + pad)
        bbox_px = (x0p, y0p, x1p, y1p)
        subtracted, _ink_ratio = pipeline.ocr_agent._template_subtract(
            raw, template_rgb, bbox_px, diff_threshold=25,
        )
    except Exception as e:
        logger.debug("template_subtract in rescue failed: %s", e)
        return raw

    # Sanity check: subtraction must leave enough ink to be useful.  If
    # it whitened everything out (e.g. because the user's scan was
    # perfectly aligned with a blank field), fall back to raw.
    try:
        if subtracted.ndim == 3:
            g = cv2.cvtColor(subtracted, cv2.COLOR_RGB2GRAY)
        else:
            g = subtracted
        ink_pixels = int(np.count_nonzero(g < 200))
        if ink_pixels < 8:
            return subtracted  # probably blank; still return subtracted
    except Exception:
        pass
    return subtracted


def _is_text_better(
    old: str, new: str, old_valid: bool, new_valid: bool,
    field_type: str, form_type: str = "",
) -> Tuple[bool, str]:
    """Decide whether the rescue candidate should replace the old text."""
    old = (old or "").strip()
    new = (new or "").strip()
    if not new:
        return False, "new_empty"
    if new == old:
        return False, "identical"

    # HARD REJECT: template-label leakage.  A rescue that returns chrome
    # from adjacent cells / captions is ALWAYS worse than what we had.
    if _looks_like_template_label(new, form_type):
        return False, "template_label"

    if new_valid and not old_valid:
        return True, "validator_passes"
    if new_valid and old_valid:
        # Both valid — prefer the longer / more informative one.
        if len(new) > len(old):
            return True, "validator_both_longer"
        return False, "validator_both_no_gain"
    if not new_valid and old_valid:
        return False, "worse_validation"

    # Neither validates — fall back to structural heuristics.
    if field_type in _DIGIT_RANGES:
        old_digits = _digit_count(old)
        new_digits = _digit_count(new)
        lo, hi = _DIGIT_RANGES[field_type]
        # Require the candidate to land in a plausible digit-count range
        # for this field type.  This kills "more_digits" accepts of things
        # with way too few or way too many digits for the type.
        in_range = lo <= new_digits <= hi
        old_in_range = lo <= old_digits <= hi
        if in_range and not old_in_range:
            return True, "digits_in_range"
        if in_range and new_digits > old_digits:
            return True, "more_digits"
        if new_digits > old_digits and not _looks_like_template_label(new, form_type):
            # Accept weaker improvement only if old has almost no digits.
            if old_digits <= 1 and new_digits >= lo // 2:
                return True, "more_digits"
        return False, "not_enough_digits"

    if len(new) >= len(old) + 2 and not re.match(r"^[\W_]+$", new):
        return True, "structurally_longer"
    return False, "no_gain"


# ---------------------------------------------------------------------- #
# Strategies
# ---------------------------------------------------------------------- #

def _strategy_vlm(
    pipeline, block, image: np.ndarray, field_type: str,
) -> Tuple[str, float]:
    """Iter 0 — Ollama VLM rescue.

    VLMs handle template chrome well because we tell them in the prompt
    ("extract only the handwritten value, ignore the printed label"),
    so passing a slightly wider crop is fine here.  We still prefer a
    template-subtracted crop when available — the model sees cleaner
    pixels and spends fewer tokens explaining away the red ink.
    """
    try:
        from utils.config import Config
    except Exception:
        return "", 0.0
    crop = _get_subtracted_crop(pipeline, block, image, pad_ratio=0.06)
    if crop is None:
        return "", 0.0
    try:
        text, conf = pipeline.ocr_agent._vlm_ocr_field(
            crop, field_name=block.id, field_type=field_type,
            model=Config.VLM_MODEL_RESCUE, timeout=90,
        )
    except Exception as e:
        logger.debug("VLM rescue failed for %s: %s", block.id, e)
        return "", 0.0
    if text and pipeline.ocr_agent._is_vlm_template_text(text):
        return "", 0.0
    cleaned = _strip_sentinels(text)
    form_type = str((block.metadata or {}).get("form_type", "") or "")
    if _looks_like_template_label(cleaned, form_type):
        return "", 0.0
    return cleaned, float(conf or 0.0)


def _strategy_florence2_raw_upscale(
    pipeline, block, image: np.ndarray, field_type: str,
) -> Tuple[str, float]:
    """Iter 1 — Florence-2 on the template-subtracted crop + 2× upscale.

    Why template-subtracted:
      The April-2026 benchmark (scripts/analyse_bench_pdf1_pdf2.py) showed
      Florence-2 getting 0/46 valid fields on raw crops because it kept
      reading the printed template labels ("PATIENT'S NAME", "NPI",
      "PICA", etc.) instead of the handwriting.  The original reason
      this strategy skipped subtraction — "damages stroke continuity" —
      was relevant back when our AKAZE alignment was noisy; with the
      current outer-quad + piecewise alignment the template pixels line
      up tightly enough that subtraction reliably zeroes the label.
      See ``_get_subtracted_crop`` for the full argument + safety net.

    We use a tight 0.02 pad because, even after subtraction, residual
    ink near the edge can pull Florence-2's attention toward adjacent
    fields.  The deterministic cleaner + ``_looks_like_template_label``
    filter downstream catch anything that still slips through.
    """
    crop = _get_subtracted_crop(pipeline, block, image, pad_ratio=0.02)
    if crop is None:
        return "", 0.0
    upscaled = _upscale(crop, factor=2.0)
    try:
        text, conf = pipeline.ocr_agent._florence2_ocr(upscaled)
    except Exception as e:
        logger.debug("Florence-2 raw+upscale failed for %s: %s", block.id, e)
        return "", 0.0
    text = _strip_sentinels(text)
    if not text:
        return "", 0.0
    if pipeline.ocr_agent._is_hallucination(text, field_type):
        return "", 0.0
    form_type = str((block.metadata or {}).get("form_type", "") or "")
    if _looks_like_template_label(text, form_type):
        return "", 0.0
    return text, float(conf or 0.0)


def _strategy_trocr(
    pipeline, block, image: np.ndarray, field_type: str,
) -> Tuple[str, float]:
    """Legacy TrOCR handwriting specialist.

    No longer referenced by any default ladder — see the comment on
    ``RESCUE_LADDER`` for why.  Kept registered so the benchmark
    harness can still measure TrOCR vs Florence-2 / GOT-OCR on the
    same crops, and so the frontend can force it via the manual method
    override for debugging.
    """
    if not getattr(pipeline.config, "enable_trocr", False):
        return "", 0.0
    bbox = (block.metadata or {}).get("original_bbox") or block.bbox
    crop = _get_crop(image, bbox, pad_ratio=0.04)
    if crop is None:
        return "", 0.0
    upscaled = _upscale(crop, factor=1.5)
    try:
        text, conf = pipeline.ocr_agent._trocr_ocr(upscaled, preprocess=True)
    except Exception as e:
        logger.debug("TrOCR rescue failed for %s: %s", block.id, e)
        return "", 0.0
    text = _strip_sentinels(text)
    if not text:
        return "", 0.0
    if pipeline.ocr_agent._is_hallucination(text, field_type):
        return "", 0.0
    form_type = str((block.metadata or {}).get("form_type", "") or "")
    if _looks_like_template_label(text, form_type):
        return "", 0.0
    return text, float(conf or 0.0)


def _strategy_florence2_aggressive(
    pipeline, block, image: np.ndarray, field_type: str,
) -> Tuple[str, float]:
    """Second Florence-2 pass with different preprocessing.

    Why two Florence-2 strategies?  We used to rely on TrOCR for a
    "diverse" second opinion, but TrOCR's IAM prior mangled numeric
    fields into English words (see the 'S4 21 Balnac H' bug).
    Firing Florence-2 twice with DIFFERENT preprocessing gives us
    genuinely different outputs most of the time:

      * ``florence2_raw_upscale`` — tight 0.02 pad, 2× resize.  Good
        when template-subtraction damaged strokes.
      * ``florence2_aggressive`` (this one) — wider 0.06 pad, 3×
        resize, CLAHE-boosted grayscale.  Good when handwriting is
        very faint (pencil, worn scans) or when upstream preproc
        washed out low-contrast ink.

    The rescue ladder walks both so different failure modes get
    distinct attempts.  Same subtraction rationale as
    ``_strategy_florence2_raw_upscale`` — template pixels are removed
    first so Florence-2 sees ink, not red label text.
    """
    crop = _get_subtracted_crop(pipeline, block, image, pad_ratio=0.06)
    if crop is None:
        return "", 0.0

    try:
        upscaled = _upscale(crop, factor=3.0)

        # Grayscale + CLAHE to boost faint strokes.  Florence-2's
        # ViT encoder handles RGB input fine; we round-trip gray→RGB
        # after the contrast boost so the stroke detail is preserved
        # but red/blue template residue gets flattened out.
        if upscaled.ndim == 3:
            gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)
        else:
            gray = upscaled
        try:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    except Exception as e:
        logger.debug("Florence-2 aggressive preproc failed for %s: %s",
                     block.id, e)
        return "", 0.0

    try:
        text, conf = pipeline.ocr_agent._florence2_ocr(processed)
    except Exception as e:
        logger.debug("Florence-2 aggressive OCR failed for %s: %s",
                     block.id, e)
        return "", 0.0

    text = _strip_sentinels(text)
    if not text:
        return "", 0.0
    if pipeline.ocr_agent._is_hallucination(text, field_type):
        return "", 0.0
    form_type = str((block.metadata or {}).get("form_type", "") or "")
    if _looks_like_template_label(text, form_type):
        return "", 0.0
    return text, float(conf or 0.0)


def _strategy_got_ocr(
    pipeline, block, image: np.ndarray, field_type: str,
) -> Tuple[str, float]:
    """GOT-OCR 2.0 rescue — unified transformer OCR (stepfun-ai).

    Gate: ``config.enable_got_ocr`` must be True AND the HF weights
    must be loadable (transformers>=4.49 with bf16 GPU).  A missing
    model returns ``("", 0.0)`` silently so the ladder moves on to
    the next strategy.

    The crop pad (0.04) matches the TrOCR pad we replaced, so behaviour
    is roughly interchangeable from the caller's side.  GOT-OCR does
    its own internal resizing to 1024px so we don't need to upscale.
    """
    if not getattr(pipeline.config, "enable_got_ocr", False):
        return "", 0.0
    got_agent = getattr(pipeline, "got_ocr_agent", None)
    if got_agent is None:
        return "", 0.0

    bbox = (block.metadata or {}).get("original_bbox") or block.bbox
    crop = _get_crop(image, bbox, pad_ratio=0.04)
    if crop is None:
        return "", 0.0

    try:
        text, conf = got_agent.recognize(crop, max_new_tokens=96)
    except Exception as e:
        logger.debug("GOT-OCR rescue failed for %s: %s", block.id, e)
        return "", 0.0

    text = _strip_sentinels(text)
    if not text:
        return "", 0.0
    if pipeline.ocr_agent._is_hallucination(text, field_type):
        return "", 0.0
    form_type = str((block.metadata or {}).get("form_type", "") or "")
    if _looks_like_template_label(text, form_type):
        return "", 0.0
    return text, float(conf or 0.0)


def _strategy_parseq(
    pipeline, block, image: np.ndarray, field_type: str,
) -> Tuple[str, float]:
    """PARSeq rescue — scene-text transformer (baudm/parseq, 23M params).

    PARSeq shines on PRINTED short lines — exactly the shape of most
    numeric CMS-1500 fields (NPI, phone, zip, tax-id, CPT).  The model
    was trained on scene text (street signs, printed signage) so it's
    less useful for cursive handwriting, but for typewriter/laser-
    printed data entries it tends to agree with the VLM while costing
    only ~30-50ms per crop (vs Florence-2's 600ms + VLM's 2-15s).

    We also feed it the template-subtracted crop so the red chrome
    doesn't compete for the decoder's attention.  PARSeq expects
    tight, single-line crops, so we use pad_ratio=0.02 (same as
    florence2_raw_upscale) and a 2× upscale to help the STR backbone
    see the ink clearly.
    """
    if not getattr(pipeline.config, "enable_parseq", True):
        return "", 0.0
    agent = getattr(pipeline, "parseq_agent", None)
    if agent is None:
        return "", 0.0

    crop = _get_subtracted_crop(pipeline, block, image, pad_ratio=0.02)
    if crop is None:
        return "", 0.0
    upscaled = _upscale(crop, factor=2.0)

    try:
        text, conf = agent.recognize(upscaled)
    except Exception as e:
        logger.debug("PARSeq rescue failed for %s: %s", block.id, e)
        return "", 0.0

    text = _strip_sentinels(text)
    if not text:
        return "", 0.0
    if pipeline.ocr_agent._is_hallucination(text, field_type):
        return "", 0.0
    form_type = str((block.metadata or {}).get("form_type", "") or "")
    if _looks_like_template_label(text, form_type):
        return "", 0.0
    return text, float(conf or 0.0)


def _strategy_slm_normalize(
    pipeline, block, image: np.ndarray, field_type: str,
) -> Tuple[str, float]:
    """Iter 3 — deterministic cleaner FIRST, SLM as a fallback.

    We feed the cleaners/SLM the most digit-rich text we've seen so far
    for this field, walking through ``block.metadata['rescue_history_texts']``
    (populated by ``run_rescue_attempt``).  That way even if the previous
    strategy accepted something with `more_digits` that still failed
    validation, this stage gets a shot at cleaning it up.
    """
    current = _strip_sentinels(str(block.text or ""))
    history_texts: List[str] = list(
        (block.metadata or {}).get("rescue_history_texts") or []
    )
    history_texts.append(current)
    # Pick the candidate with the most digits (for numeric fields) or
    # longest non-empty string (for text fields).
    if field_type in _DIGIT_RANGES:
        best = max(
            (t for t in history_texts if t),
            key=lambda t: (_digit_count(t), len(t)), default="",
        )
    else:
        best = max((t for t in history_texts if t), key=len, default="")
    if not best:
        return "", 0.0

    # 1) Deterministic cleaner — cheap, zero-ML.
    cleaned = _deterministic_clean(best, field_type)
    validator_name = _validator_name_for(field_type)
    if cleaned and _run_validator(validator_name, cleaned):
        return cleaned, max(float(block.confidence or 0.0), 0.90)

    # 2) Deterministic cleaner on *every* history item — sometimes an
    #    earlier rejected candidate cleans up better than the "best".
    for cand in history_texts:
        c = _deterministic_clean(cand, field_type)
        if c and _run_validator(validator_name, c):
            return c, max(float(block.confidence or 0.0), 0.88)

    # 3) SLM fallback.
    try:
        normalized = pipeline.ocr_agent._llm_normalize_field(best, field_type)
    except Exception as e:
        logger.debug("SLM normalize failed for %s: %s", block.id, e)
        return "", 0.0
    normalized = _strip_sentinels(normalized or "")
    if normalized and normalized != current:
        # Try deterministic cleaner on the SLM output too.
        secondary = _deterministic_clean(normalized, field_type)
        if secondary and _run_validator(validator_name, secondary):
            return secondary, max(float(block.confidence or 0.0), 0.80)
        return normalized, max(float(block.confidence or 0.0), 0.70)

    return "", 0.0


_STRATEGY_FN: Dict[str, Callable] = {
    "vlm": _strategy_vlm,
    "florence2_raw_upscale": _strategy_florence2_raw_upscale,
    "florence2_aggressive": _strategy_florence2_aggressive,
    "parseq": _strategy_parseq,
    # Kept registered for the benchmark harness / manual method overrides
    # but NOT referenced by any default ladder:
    "got_ocr": _strategy_got_ocr,
    "trocr": _strategy_trocr,
    "slm_normalize": _strategy_slm_normalize,
}


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #

def run_rescue_attempt(
    pipeline,
    block,
    image: np.ndarray,
    iteration: int,
    history: List[str],
) -> Optional[RescueAttempt]:
    """Attempt ONE rescue step on a single block.

    Returns the attempt record whether or not we accepted the new value.
    Returns ``None`` if every strategy has already been tried for this field.
    """
    field_type = str((block.metadata or {}).get("field_type", "") or "").lower()
    validator_name = _validator_name_for(field_type)

    old_text = _strip_sentinels(str(block.text or ""))
    old_valid = _run_validator(validator_name, old_text)

    # FAST PATH — deterministic cleaner on the current text.  If the
    # old value is just a light rewrite away from validating (e.g.
    # '12_10_2025' → '12/10/2025'), apply it and return early.  This
    # short-circuits the ML ladder for trivially fixable fields.
    if not old_valid and field_type in _CLEANERS:
        repaired = _deterministic_clean(old_text, field_type)
        if repaired and _run_validator(validator_name, repaired):
            block.metadata.setdefault("rescue_history", []).append("deterministic")
            block.metadata["ocr_engine"] = "rescue_deterministic"
            block.metadata["rescue_method"] = "deterministic"
            block.metadata["rescue_previous_text"] = old_text
            block.metadata["rescue_accepted"] = True
            block.text = repaired
            block.confidence = max(float(block.confidence or 0.0), 0.92)
            return RescueAttempt(
                field_id=block.id,
                method="deterministic",
                iteration=int(iteration),
                old_text=old_text,
                new_text=repaired,
                old_valid=False,
                new_valid=True,
                accepted=True,
                reason="deterministic_clean",
                confidence=0.92,
            )

    # Choose the next UNTRIED strategy for this field.  The ladder
    # is picked by ``field_type`` so handwriting-heavy fields get
    # Florence-2 / VLM / GOT-OCR first, numeric fields prefer VLM +
    # GOT-OCR, etc.  Iteration offsets into the per-type ladder so
    # different fields attempt different strategies first and we
    # don't pile up on the same engine when many fields rescue at
    # once.  Disabled strategies (e.g. GOT-OCR when its weights are
    # missing) silently return no candidate — we just walk past them.
    ladder = ladder_for_field_type(field_type)
    chosen = None
    for offset in range(len(ladder)):
        cand = ladder[(iteration + offset) % len(ladder)]
        if cand not in history:
            chosen = cand
            break
    if chosen is None:
        return None

    fn = _STRATEGY_FN.get(chosen)
    if fn is None:
        return None

    try:
        candidate, conf = fn(pipeline, block, image, field_type)
    except Exception as e:
        logger.debug("rescue strategy %s crashed: %s", chosen, e)
        candidate, conf = "", 0.0

    # After each ML call, try a deterministic polish on the raw candidate
    # — common case: VLM returns '05/19/45M' → clean to '05/19/1945'.
    if candidate and field_type in _CLEANERS:
        polished = _deterministic_clean(candidate, field_type)
        if polished and _run_validator(validator_name, polished):
            candidate = polished

    candidate = _strip_sentinels(candidate)
    new_valid = _run_validator(validator_name, candidate) if candidate else False

    # Record EVERY non-empty candidate in a per-block history for the SLM
    # to consult later (see _strategy_slm_normalize).
    if candidate:
        block.metadata.setdefault("rescue_history_texts", []).append(candidate)

    form_type = str((block.metadata or {}).get("form_type", "") or "")
    accepted, reason = False, "no_candidate"

    # Degenerate-candidate guard: single punctuation mark, ellipsis, or a
    # couple of template chars slipping through.  Florence sometimes emits
    # "...", "-", "—", etc. from blank crops and they pass naive checks.
    if candidate:
        _stripped = candidate.strip()
        _alnum = "".join(ch for ch in _stripped if ch.isalnum())
        if len(_stripped) <= 3 and len(_alnum) == 0:
            return RescueAttempt(
                field_id=block.id,
                method=chosen,
                iteration=int(iteration),
                old_text=old_text,
                new_text=candidate,
                old_valid=bool(old_valid),
                new_valid=False,
                accepted=False,
                reason="degenerate_candidate",
                confidence=float(conf or 0.0),
            )

    if candidate:
        # Blank-guard: reject speculative rescues on cells that are
        # genuinely empty.  We combine several signals because
        # different OCR paths populate different metadata:
        #
        #   * V2 blank detector writes ``blank_status`` + ``inner_ink``
        #   * V1 OCR agent writes ``is_blank`` (bool)
        #   * Post-rescue sanitizers tag ``cleared_as_*``
        #
        # The policy:
        #   - Free-text fields (name / address / text / generic) are
        #     NEVER filled from an empty starting point when the crop
        #     looks empty.  Hallucination risk from neighbors / bleed
        #     is simply too high.
        #   - Typed fields (date, npi, phone, zip, money, tax_id,
        #     state, cpt, icd*, hcpcs, ndc, member_id, ssn) may be
        #     filled only if the candidate passes its validator AND
        #     the crop shows meaningful ink.
        md = block.metadata or {}
        blank_status = str(md.get("blank_status", "") or "").lower()
        inner_ink = float(md.get("inner_ink", 0.0) or 0.0)
        is_blank_flag = bool(md.get("is_blank", False))
        old_conf = float(block.confidence or 0.0)
        ocr_engine = str(md.get("ocr_engine", "") or "").lower()
        likely_blank_explicit = blank_status in {
            "blank",
            "blank_structural",
            "blank_ink",
            "blank_high",
            "blank_med",
            "cleared_as_template_leak",
            "cleared_as_unrescuable",
        }
        # Below this ink ratio the crop is almost certainly empty — the
        # signal comes from template residue / bleed-through rather
        # than user writing.  Only trusted when inner_ink is actually
        # populated (V2 path).  Otherwise we fall back to the
        # ``is_blank`` flag from V1 and the blank_status label.
        _INK_EMPTY = 0.08
        ink_available = inner_ink > 0.0  # V2 populated this
        ink_says_empty = ink_available and inner_ink < _INK_EMPTY
        # V1 fallback signal: when the original OCR returned empty text
        # with essentially zero confidence AND the engine tag looks like
        # a blank path (``blank_fast``, ``blank_confirmed``).  Safer
        # than trusting any single field.
        engine_says_blank = any(
            tag in ocr_engine
            for tag in ("blank_fast", "blank_confirmed", "blank_structural")
        )
        v1_empty_signal = (not old_text) and (old_conf < 0.05) and engine_says_blank
        likely_blank = (
            likely_blank_explicit
            or is_blank_flag
            or v1_empty_signal
            or (not old_text and ink_says_empty)
        )
        if likely_blank and not old_text:
            is_typed = field_type in _DIGIT_RANGES or field_type in {"state"}
            # Typed + empty crop: only allow if validator passes AND
            # EITHER we don't have ink evidence (legacy V1 path) OR
            # ink is above the empty threshold.  Short valid-looking
            # strings (e.g. a fabricated date) from a crop with
            # essentially no writing are indistinguishable from
            # hallucinations.
            typed_allowed = (
                is_typed
                and new_valid
                and (not ink_available or inner_ink >= _INK_EMPTY)
                and not likely_blank_explicit
                and not is_blank_flag
                and not v1_empty_signal
            )
            if not typed_allowed:
                return RescueAttempt(
                    field_id=block.id,
                    method=chosen,
                    iteration=int(iteration),
                    old_text=old_text,
                    new_text="",
                    old_valid=bool(old_valid),
                    new_valid=False,
                    accepted=False,
                    reason="blank_guard",
                    confidence=float(conf or 0.0),
                )
        accepted, reason = _is_text_better(
            old_text, candidate, old_valid, new_valid, field_type, form_type,
        )

    if accepted:
        block.metadata.setdefault("rescue_history", []).append(chosen)
        block.metadata["ocr_engine"] = f"rescue_{chosen}"
        block.metadata["rescue_method"] = chosen
        block.metadata["rescue_previous_text"] = old_text
        block.metadata["rescue_accepted"] = True
        block.text = candidate
        block.confidence = max(float(block.confidence or 0.0), float(conf or 0.72))

    return RescueAttempt(
        field_id=block.id,
        method=chosen,
        iteration=int(iteration),
        old_text=old_text,
        new_text=candidate,
        old_valid=bool(old_valid),
        new_valid=bool(new_valid),
        accepted=bool(accepted),
        reason=reason,
        confidence=float(conf or 0.0),
    )

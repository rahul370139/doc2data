"""Per-section VLM extraction.

For each ``Section`` produced by ``src.pipelines.layout.sections``, this
module:

  1. Crops the aligned page image to the section bbox.
  2. Builds a compact schema-guided JSON prompt containing ONLY that
     section's fields (id, label, type, short description, accepted
     values for checkboxes).
  3. Calls Ollama with a mid-size VLM (MiniCPM-o4.5 by default, overrides
     via ``VLM_MODEL_SECTION``) using ``/api/generate`` with the crop as
     the image.
  4. Parses the VLM's JSON response permissively (fenced code blocks,
     trailing prose, etc.) and normalises values.
  5. Returns ``{field_id: (value, confidence)}`` with per-field metadata.

Design constraints (explicitly not a whole-page VLM):
  * ≤ 10 fields per call → small prompt + small output → reliable parsing.
  * Checkbox groups answered as "CHECKED" | "UNCHECKED" (or the symbol the
    VLM sees, post-normalised).
  * Refuses to answer if the crop looks blank / empty — empty fields stay
    empty so the downstream blank-detector + validator behave normally.
  * Robust to Ollama errors / timeouts — returns an empty result on any
    failure so the existing Florence-2 path still runs.

Output plugs straight into the graph: ``extract_sections_node`` seeds
the schema-zone blocks with these values BEFORE Florence-2 per-field OCR.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────

@dataclass
class FieldExtraction:
    field_id: str
    value: str
    confidence: float
    raw: str = ""


@dataclass
class SectionExtractionResult:
    section_id: str
    model: str
    success: bool
    latency_s: float = 0.0
    values: Dict[str, FieldExtraction] = _dc_field(default_factory=dict)
    raw_response: str = ""
    error: str = ""


# ── Prompt assembly ───────────────────────────────────────────────────────

_FIELD_TYPE_HINTS: Dict[str, str] = {
    "text":       "free-text (name, address, description as written)",
    "name":       "person name",
    "address":    "street/city/state/zip composite, as written",
    "date":       "date, MM/DD/YYYY or MM DD YYYY as it appears",
    "date_range": "two dates (from .. to), MM/DD/YYYY each",
    "phone":      "phone number (10 digits, any formatting)",
    "zip":        "5- or 9-digit US zip code",
    "npi":        "10-digit National Provider Identifier",
    "tax_id":     "federal tax id (9 digits, SSN or EIN)",
    "account":    "alphanumeric account number",
    "money":      "dollar amount with 2 decimal places, e.g. 152.00",
    "checkbox":   "return CHECKED or UNCHECKED (or empty string if unsure)",
    "signature":  "handwritten signature; leave empty",
    "table":      "structured table — return empty string here",
}


def build_prompt(
    section_id: str,
    section_label: str,
    section_fields: List[Dict[str, Any]],
    form_type: str = "CMS-1500",
    *,
    include_description: bool = True,
    max_desc_chars: int = 140,
) -> str:
    """Build a compact schema-guided prompt for one section.

    Uses a plain bulleted list (not JSON-templated) so value hints that
    contain quotes (e.g. checkbox guidance) don't poison the grammar the
    VLM is trying to mimic.  The response contract is the JSON object at
    the bottom — the VLM produces one JSON object with the listed field
    ids as keys.

    Kept under ~350 prompt tokens so even small VLMs have headroom for
    image tokens and the JSON answer.
    """
    field_lines: List[str] = []
    for f in section_fields:
        fid = f.get("id")
        if not fid:
            continue
        ftype = (f.get("field_type") or "text").lower()
        hint = _FIELD_TYPE_HINTS.get(ftype, "value as written")
        label = (f.get("label") or fid).replace('"', "'")
        line = f'- {fid}   ({ftype})   "{label}"   → {hint}'
        if include_description:
            raw_desc = (f.get("description") or "").strip().replace('"', "'")
            if raw_desc:
                if len(raw_desc) > max_desc_chars:
                    raw_desc = raw_desc[:max_desc_chars].rstrip() + "..."
                line += f"\n    note: {raw_desc}"
        field_lines.append(line)
    fields_block = "\n".join(field_lines)

    # Flat JSON skeleton — no field-description text inside braces, so the
    # VLM has a pristine JSON template to imitate.
    json_skeleton_lines = [f'  "{f["id"]}": ""' for f in section_fields if f.get("id")]
    json_skeleton = "{\n" + ",\n".join(json_skeleton_lines) + "\n}"

    prompt = f"""You are an expert form-reading assistant.

Form: {form_type}
Section: {section_label} (id={section_id})

The image above is a crop of this section of the form.  Read what is
actually written and extract the value of every listed field.  Reply
with STRICT JSON only — no commentary, no markdown fences.

Rules:
- If a field is BLANK, return an empty string.
- For checkboxes, return the exact word CHECKED when the box has a
  visible mark (X, tick, filled, handwritten cross), UNCHECKED when it
  is clearly empty, and an empty string when you cannot tell.
- For dates, preserve the format you actually see (do not invent
  separators).
- Never return placeholder template text (MM DD YY, Last Name First
  Name, Designed by NUCC, APPROVED BY ...).  If you only see template
  labels, the field is blank — return an empty string.
- Do not add keys that are not listed below.

Fields:
{fields_block}

Return JSON of the form:
{json_skeleton}
"""
    return prompt


# ── Response parsing (permissive) ─────────────────────────────────────────

_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

_NULL_LIKE = {"", "null", "none", "n/a", "na", "n.a.", "-", "—", "not visible",
              "not applicable", "not shown", "not present", "blank"}

_SENTINEL_RE = re.compile(r"<(pad|unk|s|/s|bos|eos|mask)>", re.IGNORECASE)

# Structural template-label patterns.  These are the shapes printed BY the
# form itself (not filled-in data).  We reject them in the VLM output so the
# downstream block seeder never trusts them.  All patterns are deliberately
# generic: they trigger on any value that looks like boilerplate rather than
# a real entry.
_TEMPLATE_PHRASES_RE = re.compile(
    r"""^\s*(
        reserved\s+for\s+nucc(\s+use)?(\s*\(.*\))?   # "RESERVED FOR NUCC USE (9b)"
      | claim\s+codes?                               # "CLAIM CODES"
      | designated\s+by\s+nucc                       # "(Designated by NUCC)"
      | insurance\s+plan\s+name(\s+or\s+program\s+name)?(\s*\(.*\))?
      | \$\s*charges?                                # "$ CHARGES"
      | amount\s+paid
      | total\s+charge
      | federal\s+tax\s+i\.?d\.?\s+number
      | patient.?s?\s+account\s+no\.?
      | other\s+claim\s+id
      | last\s+name\s+first\s+name\s+middle\s+initial
      | mm\s*[/ ]?\s*dd\s*[/ ]?\s*yy(yy)?            # date template "MM DD YY"
      | approved\s+by\s+nucc
      | service\s+facility\s+(location\s+)?information
      | ordinal\s+ref\.?\s+no\.?
      | prior\s+authorization\s+number
      | signed
      | date(\s+of\s+service)?
      | rendering\s+provider\s+id\.?
      | billing\s+provider\s+info\s*&?\s*ph\s*#?
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _looks_like_template_label(value: str) -> bool:
    """True if the string is boilerplate printed on the form, not real data."""
    if not value:
        return False
    # Strip wrapping punctuation (parens, quotes, etc.) so
    # "(Designated by NUCC)" matches the bare phrase.
    stripped = re.sub(r"^[\s\"'(\[]+|[\s\"'\]\)]+$", "", value.strip())
    return bool(_TEMPLATE_PHRASES_RE.match(stripped))


def _strip_code_fence(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s).strip()


def _first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extract the first valid JSON object from ``text``.

    Handles fenced code blocks, leading/trailing commentary, and mild
    trailing garbage.  Returns None if no JSON object can be parsed.
    """
    if not text:
        return None
    s = _strip_code_fence(text.strip())
    # Fast path: whole thing is JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Scan for the first {...} block and try to parse incrementally
    depth = 0
    start = -1
    for i, c in enumerate(s):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = s[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    start = -1  # keep scanning
                    continue
    return None


def _normalise_value(raw: Any, field_type: str) -> str:
    if raw is None:
        return ""
    v = str(raw).strip()
    if not v:
        return ""
    v = _SENTINEL_RE.sub("", v).strip()
    v = re.sub(r"\s+", " ", v)
    if v.lower() in _NULL_LIKE:
        return ""
    if field_type == "checkbox":
        low = v.lower()
        if low in ("checked", "true", "yes", "x", "✓", "marked", "filled"):
            return "CHECKED"
        if low in ("unchecked", "false", "no", "empty", "blank"):
            return "UNCHECKED"
        return ""  # unknown / refused
    # Template-label leak — VLM returned the printed form label instead of
    # an actual entry.  Drop it so the block is left empty for Florence-2.
    if _looks_like_template_label(v):
        return ""
    return v


# ── Model-error classification ────────────────────────────────────────────
#
# Ollama returns a JSON body with an ``error`` key for a wide range of
# failure modes.  For fallback routing we only need a coarse bucket:
# "retryable_on_another_model" vs "not_retryable".  Examples we've seen:
#
#   * "llama runner process has terminated: GGML_ASSERT(false && \"unsupported
#      minicpmv version\") failed"             → runtime incompatible with
#                                                  *this* model → retry fallback
#   * "model not found"                        → retry fallback
#   * "unexpected EOF", "context deadline exceeded" → transient, retry
#
# JSON-parse failures (``non_json_body``) are NOT retried: a different
# model is unlikely to emit valid JSON if the primary couldn't.
_RETRYABLE_ERROR_TAGS = {
    "timeout",
    "http_error",            # network-level (connection reset, etc.)
    "unsupported_model",
    "model_not_found",
    "runtime_error",
    "server_error",
}


def _classify_model_error(msg: str) -> str:
    """Map a raw error string from Ollama to a stable, low-cardinality tag."""
    low = (msg or "").lower()
    if "unsupported" in low and ("minicpm" in low or "version" in low or "arch" in low):
        return "unsupported_model"
    if "ggml_assert" in low or "llama runner" in low:
        return "runtime_error"
    if "model" in low and ("not found" in low or "does not exist" in low):
        return "model_not_found"
    if "timeout" in low or "deadline" in low:
        return "timeout"
    if "http_" in low or low.startswith("5") or "server error" in low:
        return "server_error"
    if low.startswith("http_error"):
        return "http_error"
    return msg.strip()[:80] or "unknown_error"


def _is_retryable_error(err: Optional[str]) -> bool:
    if not err:
        return False
    tag = err.split(":", 1)[0]
    return tag in _RETRYABLE_ERROR_TAGS


# ── Extractor ─────────────────────────────────────────────────────────────

class SectionVLMExtractor:
    """Thin wrapper around Ollama for per-section structured extraction.

    Designed to be instantiated once and reused across sections / docs.
    The underlying HTTP call is blocking; callers run it in a thread
    (see ``extract_sections_node`` which uses ``asyncio.to_thread``).
    """

    def __init__(
        self,
        ollama_host: str,
        model: str,
        *,
        timeout: int = 120,
        # Per-section answers are short: ~10 fields × ~20 chars = ~200 tokens
        # budget is plenty.  Dropping from 600 → 400 cuts Ollama eval time
        # noticeably without truncating any real response.
        max_tokens: int = 400,
        temperature: float = 0.0,
        jpeg_quality: int = 92,
        # Keep crops below Ollama's internal slice threshold (~1024px) so
        # each section runs as one tile instead of 2-3.
        max_crop_side: int = 1000,
        seed: int = 17,
        fallback_models: Optional[List[str]] = None,
    ):
        self.ollama_host = ollama_host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.jpeg_quality = jpeg_quality
        self.max_crop_side = max_crop_side
        self.seed = seed
        # Deduplicate while preserving order; drop any that equal the primary
        # so we don't try the same model twice.
        seen = {self.model}
        self.fallback_models: List[str] = []
        for m in (fallback_models or []):
            if m and m not in seen:
                seen.add(m)
                self.fallback_models.append(m)

    # ── Public API ────────────────────────────────────────────────────────

    def extract(
        self,
        section_image: np.ndarray,
        section_id: str,
        section_label: str,
        section_fields: List[Dict[str, Any]],
        form_type: str = "CMS-1500",
    ) -> SectionExtractionResult:
        """Extract all fields in a section from a crop image.

        ``section_fields`` is the raw schema dicts for just the fields in
        this section (so we preserve ``field_type``, ``description``, etc.
        for the prompt).
        """
        t0 = time.time()
        if section_image is None or getattr(section_image, "size", 0) == 0:
            return SectionExtractionResult(
                section_id=section_id, model=self.model,
                success=False, latency_s=0.0, error="empty_crop",
            )

        # Skip sections that have only signatures/tables — nothing useful
        # a mid-size VLM can do here.
        usable_fields = [
            f for f in section_fields
            if (f.get("field_type") or "text").lower() not in ("signature", "table")
        ]
        if not usable_fields:
            return SectionExtractionResult(
                section_id=section_id, model=self.model,
                success=True, latency_s=time.time() - t0,
                error="no_usable_fields",
            )

        img_b64 = self._encode_image(section_image)
        if not img_b64:
            return SectionExtractionResult(
                section_id=section_id, model=self.model,
                success=False, latency_s=time.time() - t0, error="encode_failed",
            )

        prompt = build_prompt(
            section_id=section_id,
            section_label=section_label,
            section_fields=usable_fields,
            form_type=form_type,
        )

        candidates = [self.model, *self.fallback_models]
        last_error = "no_candidate_models"
        for idx, model_name in enumerate(candidates):
            raw_text, err = self._call_model(model_name, prompt, img_b64)
            if raw_text is not None:
                values = self._parse_values(raw_text, usable_fields)
                latency = time.time() - t0
                filled = sum(1 for v in values.values() if (v.value or "").strip())
                logger.info(
                    "Section %-30s model=%-30s latency=%5.2fs filled=%2d/%2d%s",
                    section_id, model_name, latency, filled, len(usable_fields),
                    "" if idx == 0 else f" (fallback:{idx})",
                )
                return SectionExtractionResult(
                    section_id=section_id,
                    model=model_name,
                    success=True,
                    latency_s=latency,
                    values=values,
                    raw_response=raw_text,
                )
            last_error = err or "unknown_error"
            # Only fall through to the next model for errors that are likely
            # model-specific (runtime unsupported, 4xx/5xx, timeout) — not
            # for JSON parse errors that would repeat on any model.
            if not _is_retryable_error(err):
                break
            logger.warning(
                "Section %s on model=%s failed (%s); trying next candidate.",
                section_id, model_name, err,
            )

        return SectionExtractionResult(
            section_id=section_id, model=self.model,
            success=False, latency_s=time.time() - t0, error=last_error,
        )

    # Single HTTP call to one model.  Returns (raw_text, None) on success,
    # (None, error_tag) on any failure we want the caller to see.  The
    # error_tag is normalised so the fallback policy can decide whether to
    # retry with a different model (e.g. "unsupported_model" → retry;
    # "timeout" → retry; "non_json_body" → do not retry).
    def _call_model(
        self,
        model: str,
        prompt: str,
        img_b64: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        try:
            resp = requests.post(
                f"http://{self.ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                    "format": "json",
                    "options": {
                        # temperature=0 + fixed seed → same crop returns
                        # the same JSON every run.  Critical for reproducible
                        # rescue loops and avoiding spurious per-run diffs.
                        "temperature": self.temperature,
                        "top_p": 1.0,
                        "seed": self.seed,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            return None, "timeout"
        except Exception as e:
            return None, f"http_error:{e}"

        # Inspect body even on 500s — Ollama returns {"error": "..."} on
        # runtime failures (e.g. unsupported model).  We classify those as
        # retryable so a fallback can kick in.
        try:
            payload = resp.json()
        except Exception:
            return None, "non_json_body"

        if not resp.ok:
            err_msg = str(payload.get("error") or f"http_{resp.status_code}")
            return None, _classify_model_error(err_msg)

        err_msg = payload.get("error")
        if err_msg:
            return None, _classify_model_error(str(err_msg))

        raw_text = payload.get("response", "") or ""
        return raw_text, None

    # ── Internals ─────────────────────────────────────────────────────────

    def _encode_image(self, image: np.ndarray) -> str:
        try:
            img = image
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 3:
                # Assume BGR — if the caller had RGB, cvtColor is still cheap.
                # Keep as-is; cv2.imencode writes BGR.
                pass
            # Downscale very large crops; VLMs don't benefit from extreme res
            # and big JPEGs push context budget without improving accuracy.
            h, w = img.shape[:2]
            long_side = max(h, w)
            if long_side > self.max_crop_side:
                scale = self.max_crop_side / float(long_side)
                img = cv2.resize(
                    img, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                return ""
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception as e:
            logger.warning("Image encode failed: %s", e)
            return ""

    def _parse_values(
        self,
        raw_text: str,
        section_fields: List[Dict[str, Any]],
    ) -> Dict[str, FieldExtraction]:
        obj = _first_json_object(raw_text) or {}
        out: Dict[str, FieldExtraction] = {}
        field_type_by_id = {f["id"]: (f.get("field_type") or "text").lower()
                            for f in section_fields if f.get("id")}

        for fid, ftype in field_type_by_id.items():
            raw = obj.get(fid, "")
            normalised = _normalise_value(raw, ftype)
            # Confidence heuristic: a non-empty, non-null answer that we
            # didn't strip to "" → 0.85.  Empty answer → 0.0 (the downstream
            # Florence-2 pass may still fill it).  Template/sentinel → 0.0.
            if normalised:
                conf = 0.85
            else:
                conf = 0.0
            out[fid] = FieldExtraction(
                field_id=fid,
                value=normalised,
                confidence=conf,
                raw=str(raw) if raw is not None else "",
            )
        return out


# ── Convenience helpers for downstream nodes ─────────────────────────────

def crop_section(
    image: np.ndarray,
    bbox_norm: Tuple[float, float, float, float],
) -> np.ndarray:
    """Crop ``image`` to normalised bbox. Safe against degenerate boxes."""
    if image is None or image.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    x0 = max(0, int(round(bbox_norm[0] * w)))
    y0 = max(0, int(round(bbox_norm[1] * h)))
    x1 = min(w, int(round(bbox_norm[2] * w)))
    y1 = min(h, int(round(bbox_norm[3] * h)))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return image[y0:y1, x0:x1].copy()

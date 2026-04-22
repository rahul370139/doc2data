"""
High-level field OCR driver — batched, adaptive, low-latency.

This module is the core of the latency fix.  Instead of walking every
field through multi-pass Florence-2 + raw fallback + upscale retry +
consistency check, we:

1. Pre-classify every field as BLANK / UNCERTAIN / FILLED using the
   structural blank detector.
2. Skip Florence-2 entirely for BLANK fields.
3. Batch Florence-2 calls for UNCERTAIN + FILLED fields.
4. Post-filter Florence-2 outputs using the same hallucination / template
   keyword rules the OCR agent already owns.

Latency budget for an 86-field CMS-1500 on CPU goes from ~50-90 seconds
(serial F2) to ~8-15 seconds (batched), with zero Ollama VLM calls for
the common case.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.pipelines.ocr_v2.blank_detector import (
    BlankDecision, BlankDetector, BlankStatus,
)
from src.pipelines.ocr_v2.florence_batch import (
    BatchedFlorence2, get_batched_florence2,
)


logger = logging.getLogger("ocr_v2.field_ocr")


# ---------------------------------------------------------------------- #
# Template text filters (shared with OCR agent — duplicated intentionally
# so field_ocr can stand alone for unit testing).
# ---------------------------------------------------------------------- #

# Template text filters are now entirely DATA-DRIVEN: the per-form
# vocabulary (page chrome + field labels) is loaded from
# ``data/schemas/<form>.json`` by
# ``src.pipelines.graph.rescue_strategies._looks_like_template_label``,
# so the regex-of-keywords that used to live here has been removed.
#
# What remains below is a *structural* pre-filter that works on ANY
# form even when no schema is provided — it catches sentinel tokens,
# pure placeholder chars, and box-number-only outputs.  Anything more
# specific (PICA, CARRIER, schema labels) is handled via the schema-
# aware pass inside ``_filter_ocr_text``.

_PLACEHOLDER_CHARS = frozenset({"-", "–", "—", ".", "_", "|", "~", "*", ":", ";"})

# Defensive strip for Florence-2 / TrOCR / VLM sentinel tokens that should
# never reach the consumer.
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


# ---------------------------------------------------------------------- #
# Data classes
# ---------------------------------------------------------------------- #

@dataclass
class FieldOCRRequest:
    """One OCR request: a crop + meta used for post-filtering."""
    field_id: str
    field_type: str
    crop_for_ocr: np.ndarray   # template-subtracted
    crop_raw: np.ndarray       # pre-subtraction (fallback)
    ink_ratio_hint: float = 1.0
    # Optional form-type tag (e.g. "cms-1500") so the post-filter can
    # consult the schema-driven template-label matcher in addition to
    # the local keyword regex.  Falls back to local-only filtering
    # when empty, so the unit tests don't need a schema present.
    form_type: str = ""


@dataclass
class FieldOCRResult:
    field_id: str
    text: str
    confidence: float
    engine: str
    blank: bool
    escalation: str = "none"
    decision: Optional[BlankDecision] = None
    extra: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Batch driver
# ---------------------------------------------------------------------- #

class FieldOCRBatch:
    """Runs blank detection + batched Florence-2 over a list of fields.

    The caller supplies a ``BlankDetector`` that has already been
    calibrated for the current form page.  This lets us reuse the same
    detector across the pipeline and share the same baseline.
    """

    def __init__(
        self,
        blank_detector: BlankDetector,
        florence: Optional[BatchedFlorence2] = None,
        max_new_tokens: int = 128,
    ):
        self.blank_detector = blank_detector
        self.florence = florence or get_batched_florence2()
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    def run(self, requests: List[FieldOCRRequest]) -> List[FieldOCRResult]:
        """Process a list of field OCR requests.

        Returns results in the same order.  Empty / blank crops short
        circuit; everything else goes through a single batched F2 call.
        """
        if not requests:
            return []

        t_start = time.time()

        # 1) Blank pre-classify every request
        decisions: List[BlankDecision] = []
        for req in requests:
            decisions.append(
                self.blank_detector.classify(
                    req.crop_for_ocr,
                    ink_ratio_hint=req.ink_ratio_hint,
                )
            )

        # 2) Collect batched indices
        to_ocr_indices: List[int] = []
        to_ocr_crops: List[np.ndarray] = []
        for idx, (req, decision) in enumerate(zip(requests, decisions)):
            if decision.status is BlankStatus.BLANK:
                continue
            to_ocr_indices.append(idx)
            to_ocr_crops.append(req.crop_for_ocr)

        # 3) Run batched F2
        f2_texts: List[str] = []
        if to_ocr_crops:
            f2_texts = self.florence.run_batch(to_ocr_crops)
        f2_map: Dict[int, str] = {
            i: t for i, t in zip(to_ocr_indices, f2_texts)
        }

        # 4) Collect crops that need the RAW-crop retry (F2 empty but
        #    structural signal present).  We retry for BOTH FILLED and
        #    UNCERTAIN crops — UNCERTAIN is typical of thin/faint
        #    handwriting where template subtraction shaved enough ink to
        #    leave the crop sparse but still recognisable on the raw
        #    (un-subtracted) pixels.
        raw_indices: List[int] = []
        raw_crops: List[np.ndarray] = []
        for idx in to_ocr_indices:
            req = requests[idx]
            decision = decisions[idx]
            text = f2_map.get(idx, "").strip()
            has_any_structure = (
                decision.status is not BlankStatus.BLANK
                and (decision.component_count >= 1
                     or decision.inner_ink > decision.blank_threshold)
            )
            needs_raw = (
                not text
                and has_any_structure
                and req.crop_raw is not None and req.crop_raw.size > 0
            )
            if needs_raw:
                raw_indices.append(idx)
                raw_crops.append(req.crop_raw)

        raw_texts: List[str] = []
        if raw_crops:
            raw_texts = self.florence.run_batch(raw_crops)
        raw_map: Dict[int, str] = {
            i: t for i, t in zip(raw_indices, raw_texts)
        }

        # 5) Assemble results
        results: List[FieldOCRResult] = []
        for idx, req in enumerate(requests):
            decision = decisions[idx]
            results.append(self._assemble_result(
                req, decision,
                f2_text=f2_map.get(idx, ""),
                raw_text=raw_map.get(idx, ""),
                ran_raw=idx in raw_map,
            ))

        elapsed_ms = int((time.time() - t_start) * 1000)
        blanks = sum(1 for r in results if r.blank)
        filled = len(results) - blanks
        logger.info(
            "FieldOCRBatch: %d fields in %d ms (blank=%d filled=%d raw_retries=%d)",
            len(requests), elapsed_ms, blanks, filled, len(raw_crops),
        )
        return results

    # ------------------------------------------------------------------ #
    # Filters + assembly
    # ------------------------------------------------------------------ #

    def _assemble_result(
        self,
        req: FieldOCRRequest,
        decision: BlankDecision,
        f2_text: str,
        raw_text: str,
        ran_raw: bool,
    ) -> FieldOCRResult:
        # Status BLANK ⇒ always return empty.  Note: we do NOT overrule
        # BLANK with F2 text; if the detector says blank and F2 still
        # produced something, it's template residue.
        if decision.status is BlankStatus.BLANK:
            return FieldOCRResult(
                field_id=req.field_id,
                text="",
                confidence=0.0,
                engine="blank_fast",
                blank=True,
                escalation="none",
                decision=decision,
            )

        text = (f2_text or "").strip()
        engine = "florence2"
        escalation = "none"

        text = self._filter_ocr_text(
            text, req.field_type, form_type=req.form_type,
        )

        if not text and ran_raw:
            raw = self._filter_ocr_text(
                (raw_text or "").strip(), req.field_type,
                form_type=req.form_type,
            )
            if raw:
                text = raw
                engine = "florence2_raw_fallback"
                escalation = "raw_crop_retry"

        if not text:
            # Detector thought there was content but we couldn't read it.
            # Record as blank_confirmed so we don't display garbage; but
            # mark escalation="noisy" so the rescue step can pick it up.
            status = BlankStatus.BLANK if decision.inner_ink < decision.blank_threshold \
                else BlankStatus.UNCERTAIN
            return FieldOCRResult(
                field_id=req.field_id,
                text="",
                confidence=0.0,
                engine="blank_confirmed",
                blank=(status is BlankStatus.BLANK),
                escalation="f2_empty",
                decision=decision,
            )

        # Text-based confidence — just F2's baseline modulated by filters
        raw_conf = 0.82
        composite = self._compute_composite_confidence(
            raw_conf, text, req.field_type, regex_ok=True
        )
        return FieldOCRResult(
            field_id=req.field_id,
            text=text,
            confidence=composite,
            engine=engine,
            blank=False,
            escalation=escalation,
            decision=decision,
        )

    @staticmethod
    def _filter_ocr_text(text: str, field_type: str,
                         form_type: str = "") -> str:
        """Strip placeholder / template residue from an OCR output.

        Two STAGES, both data-driven (no form-specific keyword lists
        live here):

        1. **Structural**: strip sentinels, one-char placeholders,
           lone box-numbers (``"23"``, ``"24a"``).  Form-agnostic.
        2. **Schema-aware**: when ``form_type`` is supplied, delegate
           the chrome/label decision to
           ``rescue_strategies._looks_like_template_label``, which in
           turn reads the ``template_chrome`` array and field labels
           from ``data/schemas/<form>.json``.  Adding a new form is a
           JSON edit, not a Python edit.
        """
        if not text:
            return ""
        clean = _strip_sentinels(text)
        clean = clean.strip().strip('"\'`')
        if not clean:
            return ""
        if clean in _PLACEHOLDER_CHARS:
            return ""
        if len(clean) < 2:
            return ""
        if field_type == "text" and re.match(r"^\d{1,3}[a-z]?\.?$", clean):
            return ""
        # Schema-aware leak detector — catches page chrome (PICA,
        # CARRIER, APPROVED BY NUCC…) and leaked field labels
        # ("ORIGINAL REF. NO", "RESERVED FOR NUCC USE") using
        # fuzzy token-overlap against the schema vocabulary.  When
        # ``form_type`` is empty (e.g. unit tests) we skip this stage
        # and accept the candidate — the structural filter above is
        # enough to catch the universally-bad cases.
        if form_type:
            try:
                from src.pipelines.graph.rescue_strategies import (
                    _looks_like_template_label as _llt,
                )
                if _llt(clean, form_type):
                    return ""
            except Exception:
                pass
        return clean

    @staticmethod
    def _compute_composite_confidence(
        ocr_conf: float, text: str, field_type: str, regex_ok: bool,
    ) -> float:
        t = (text or "").strip()
        char_valid = 0.5
        if t:
            if field_type in ("date", "date_range", "phone", "tax_id",
                              "npi", "money", "zip", "account"):
                ok = sum(c.isdigit() or c in "/-.()$ " for c in t)
                char_valid = ok / max(1, len(t))
            elif field_type == "address":
                ok = sum(
                    c.isalnum() or c.isspace() or c in ",.-'#/" for c in t
                )
                char_valid = ok / max(1, len(t))
            elif field_type in ("text", "icd10"):
                ok = sum(
                    c.isalpha() or c.isspace() or c in ",.-'" for c in t
                )
                char_valid = ok / max(1, len(t))
        regex_score = 1.0 if regex_ok else 0.0
        context_score = 1.0 if t and len(t) >= 2 else 0.3
        return float(
            0.4 * ocr_conf + 0.2 * regex_score
            + 0.2 * char_valid + 0.2 * context_score
        )

"""
Graph state definition.

Kept small and flat so reducers are cheap.  Fields that grow large
(aligned image, block list, validation dict) are kept as plain Python
objects — LangGraph handles them fine as long as they pickle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np

from src.pipelines.core.models import (
    DetectedBlock, FormType, PipelineConfig,
)


Lane = Literal["A", "B", "C", "unknown"]


class GraphState(TypedDict, total=False):
    """State shared across all graph nodes.

    ``total=False`` lets nodes populate fields incrementally.  Keys that
    are conceptually required at the end of the graph are asserted in
    ``finalize``.
    """
    # ── Input ────────────────────────────────────────────────────────
    file_path: str
    config: PipelineConfig

    # ── Load ────────────────────────────────────────────────────────
    image: np.ndarray
    width: int
    height: int
    digital_words: List[Dict[str, Any]]
    has_digital_text: bool

    # ── Form ID ────────────────────────────────────────────────────
    form_type: FormType
    form_confidence: float

    # ── Plan ───────────────────────────────────────────────────────
    lane: Lane
    plan_reason: str

    # ── Align (Lane C) ─────────────────────────────────────────────
    aligned_image: np.ndarray
    alignment_used: bool
    alignment_quality: float
    alignment_success: bool
    # Forward homography (original → template).  Stored so ``finalize``
    # can compute the INVERSE and project per-field bboxes back onto
    # the original (unwarped) image — that's what lets the frontend
    # show the user's uploaded PDF with overlays that line up, instead
    # of a noticeably-warped aligned render that always looks "off".
    homography_matrix: Optional[List[List[float]]]
    aligned_size: Optional[Tuple[int, int]]   # (width, height) of aligned space

    # ── Sections (Lane C Tier-1) ───────────────────────────────────
    # Layout segmentation groups the schema into 8–22 semantic
    # regions (row-bands with optional L/R column split).  Each
    # section is read by ONE VLM call with ONLY its fields in the
    # prompt — small model, small schema, high accuracy.  The
    # results seed the per-field blocks BEFORE Florence-2 runs, so
    # Florence-2 only has to handle the residual (fields the VLM
    # couldn't read confidently).
    #
    # sections:         List[dict]  — detected sections (id, bbox, field_ids)
    # section_values:   {section_id: {field_id: value}}
    # section_confidences: {section_id: {field_id: conf}}
    # section_meta:     {section_id: {"model", "latency_s", "error", ...}}
    sections: List[Dict[str, Any]]
    section_values: Dict[str, Dict[str, str]]
    section_confidences: Dict[str, Dict[str, float]]
    section_meta: Dict[str, Dict[str, Any]]
    sections_used: bool

    # ── Extract ────────────────────────────────────────────────────
    blocks: List[DetectedBlock]
    extraction_method: str

    # ── Validation ─────────────────────────────────────────────────
    extracted_fields: Dict[str, Any]
    validation: Dict[str, Any]

    # ── Reflect / Rescue loop ──────────────────────────────────────
    rescue_candidates: List[str]
    rescue_iterations: int
    vlm_rescue_count: int
    # Per-field escalation ladder state — which methods have been tried
    # against a given field across rescue iterations.  Used by
    # ``reflect_node`` to pick the NEXT method, so the loop keeps
    # attempting new strategies until validation passes.
    rescue_history: Dict[str, List[str]]
    # Rescue log — every attempt with method + old/new text + accepted bool
    rescue_log: List[Dict[str, Any]]

    # ── Finalize ───────────────────────────────────────────────────
    business_fields: Dict[str, Any]
    reducto_format: Dict[str, Any]

    # ── Meta ───────────────────────────────────────────────────────
    timings: Dict[str, float]
    errors: List[str]
    trace: List[str]           # node execution trace
    final_response: Dict[str, Any]


def create_initial_state(
    file_path: str,
    config: Optional[PipelineConfig] = None,
) -> GraphState:
    """Create a clean initial state for a single file."""
    from src.pipelines.core.models import PipelineConfig as DefaultConfig
    return GraphState(
        file_path=file_path,
        config=config or DefaultConfig(),
        width=0,
        height=0,
        digital_words=[],
        has_digital_text=False,
        form_type=FormType.UNKNOWN,
        form_confidence=0.0,
        lane="unknown",
        plan_reason="",
        alignment_used=False,
        alignment_quality=0.0,
        alignment_success=False,
        homography_matrix=None,
        aligned_size=None,
        sections=[],
        section_values={},
        section_confidences={},
        section_meta={},
        sections_used=False,
        blocks=[],
        extraction_method="",
        extracted_fields={},
        validation={"errors": [], "warnings": [], "qa_notes": []},
        rescue_candidates=[],
        rescue_iterations=0,
        vlm_rescue_count=0,
        rescue_history={},
        rescue_log=[],
        business_fields={},
        reducto_format={},
        timings={},
        errors=[],
        trace=[],
        final_response={},
    )

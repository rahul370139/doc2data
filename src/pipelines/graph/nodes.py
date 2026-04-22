"""
Graph nodes — each is an async function ``(state) → partial state``.

A node is expected to:
  1. Read the state it needs.
  2. Call into the underlying agent(s).
  3. Return only the keys it actually modified (LangGraph merges them in).

We reuse the battle-tested ``MultiAgentPipeline`` internals — the graph is
just the orchestrator, not a re-implementation of every agent.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.pipelines.core.models import (
    AlignmentResult, BlockType, DetectedBlock, FormType, PipelineConfig,
)
from src.pipelines.graph.state import GraphState


logger = logging.getLogger("graph.nodes")


# ---------------------------------------------------------------------- #
# Pipeline handle — a single MultiAgentPipeline instance is reused across
# graph runs.  It owns expensive models (Florence-2, PaddleOCR, TrOCR,
# template images) that should not be reloaded per request.
# ---------------------------------------------------------------------- #

_pipeline_lock = asyncio.Lock()
_pipeline_instance = None  # type: ignore
_pipeline_config_hash = None


async def _get_pipeline(config: PipelineConfig):
    """Return a shared MultiAgentPipeline; rebuild only if config changes."""
    global _pipeline_instance, _pipeline_config_hash
    from src.pipelines.multi_agent_pipeline import MultiAgentPipeline

    # Small stable signature of the config — enough to detect reconfig
    sig = (
        config.layout_model, config.enable_trocr, config.ocr_engine_mode,
        config.enable_vlm_ocr_fallback, config.zone_padding_px,
        config.zone_padding_ratio, getattr(config, "extra_ocr_padding_px", 0),
        getattr(config, "use_ocr_v2", True),
        config.enable_slm_labeling, config.enable_vlm_tables,
        config.enable_validators, config.enable_alignment,
    )

    async with _pipeline_lock:
        if _pipeline_instance is None or _pipeline_config_hash != sig:
            _pipeline_instance = MultiAgentPipeline(config)
            _pipeline_config_hash = sig
            # Kick off model loads concurrently.  ``labeling_agent`` is the
            # one that owns VLM table extraction — historically it was only
            # initialised on the legacy (non-graph) path, so in the graph
            # flow ``_ollama_available`` stayed ``None``, ``_call_vlm``
            # short-circuited with "VLM skipped: Ollama has no models", and
            # Box 24 came back empty every single run.  Initialising it
            # here is what actually unblocks table extraction.
            try:
                await _pipeline_instance.form_id_agent.initialize()
                await _pipeline_instance.ocr_agent.initialize()
                await _pipeline_instance.labeling_agent.initialize()
                # GOT-OCR is lazy-loaded on first use (see GOTOCRAgent),
                # but its ``initialize`` just marks the agent as ready
                # so ``pipeline.got_ocr_agent.is_available`` works for
                # the benchmark harness and debug UI.
                got_agent = getattr(_pipeline_instance, "got_ocr_agent", None)
                if got_agent is not None:
                    await got_agent.initialize()
                # PARSeq agent — same lazy-load pattern; ``initialize``
                # just flips the ready flag so ``is_available`` works
                # without triggering a torch.hub download at startup.
                parseq_agent = getattr(_pipeline_instance, "parseq_agent", None)
                if parseq_agent is not None and getattr(state["config"], "enable_parseq", False):
                    await parseq_agent.initialize()
            except Exception as e:
                logger.warning("Pipeline pre-warm failed: %s", e)
    return _pipeline_instance


def _mark(state: GraphState, node: str, elapsed: float) -> None:
    """Record trace + timing."""
    trace = list(state.get("trace") or [])
    trace.append(node)
    timings = dict(state.get("timings") or {})
    timings[node] = round(elapsed, 3)
    state["trace"] = trace
    state["timings"] = timings


# ---------------------------------------------------------------------- #
# Nodes
# ---------------------------------------------------------------------- #

async def load_node(state: GraphState) -> Dict[str, Any]:
    """Load PDF/image bytes, render first page, extract digital text words."""
    t0 = time.time()
    pipeline = await _get_pipeline(state["config"])
    image, width, height, digital_words = await asyncio.to_thread(
        pipeline._load_image, state["file_path"]
    )
    if image is None:
        return {
            "errors": [f"Failed to load file: {state['file_path']}"],
            "trace": list(state.get("trace") or []) + ["load"],
        }
    elapsed = time.time() - t0
    out = {
        "image": image,
        "width": int(width),
        "height": int(height),
        "digital_words": digital_words or [],
        "has_digital_text": bool(digital_words),
    }
    _update_trace_in_place(out, state, "load", elapsed)
    return out


async def identify_node(state: GraphState) -> Dict[str, Any]:
    """Identify form type (CMS-1500 / UB-04 / generic)."""
    t0 = time.time()
    pipeline = await _get_pipeline(state["config"])
    config = state["config"]
    if config.form_type_override is not None:
        form = type("F", (), {})()
        form.form_type = config.form_type_override
        form.confidence = 1.0
    else:
        form = await pipeline.form_id_agent.process(state["image"])
    elapsed = time.time() - t0
    out = {
        "form_type": form.form_type,
        "form_confidence": float(getattr(form, "confidence", 1.0)),
    }
    _update_trace_in_place(out, state, "identify", elapsed)
    return out


async def plan_node(state: GraphState) -> Dict[str, Any]:
    """Select extraction lane.

    Auto rules (in order):
      * CMS-1500 / UB-04 + fillable PDF widgets                → Lane A
      * CMS-1500 with NO widgets                               → Lane C (scan)
          (reason: digital text rarely sits on template coords; alignment +
           schema-zone OCR is more reliable than blind word-to-bbox matching.)
      * Non-CMS/UB-04 form with digital text layer             → Lane B
      * Otherwise                                              → Lane C

    Override:
      When ``config.method_override`` is set to anything other than
      "auto", plan_node honours it and records the reason so users can
      see in the trace why the usual routing was bypassed.  Unknown
      overrides fall back to "auto" with a warning.
    """
    t0 = time.time()
    config = state["config"]
    pipeline = await _get_pipeline(config)
    form_type = state["form_type"]
    file_path = state["file_path"]

    lane = "C"
    reason = "scan default"

    # Method override short-circuits the auto routing.
    override = str(getattr(config, "method_override", "auto") or "auto").lower()
    if override and override != "auto":
        override_map = {
            "cms1500_scan": ("C", "override: cms1500_scan (force align + Florence-2)"),
            "scan": ("C", "override: scan (force align + Florence-2)"),
            "sections": ("C", "override: sections (force section-VLM Tier 1)"),
            "digital": ("B", "override: digital (force text-layer zone match)"),
            "widgets": ("A", "override: widgets (force AcroForm extraction)"),
        }
        picked = override_map.get(override)
        if picked is None:
            logger.warning(
                "plan_node: unknown method_override %r, falling back to auto",
                override,
            )
        else:
            lane, reason = picked
            # If user forced Lane A but there are no widgets, downgrade
            # so we don't return an empty response.  Same for Lane B.
            if lane == "A" and form_type in (FormType.CMS1500, FormType.UB04):
                widget_info = await asyncio.to_thread(
                    pipeline._extract_widgets, file_path,
                )
                any_filled = int((widget_info or {}).get("filled", 0))
                if any_filled == 0:
                    lane = "C"
                    reason = "override:widgets requested but no filled widgets → Lane C"
            elif lane == "B" and not state.get("has_digital_text"):
                lane = "C"
                reason = "override:digital requested but no text layer → Lane C"
            elapsed = time.time() - t0
            out = {"lane": lane, "plan_reason": reason}
            _update_trace_in_place(out, state, "plan", elapsed)
            return out

    # Lane A: AcroForm widgets.
    #
    # NOTE: ``_extract_widgets`` returns the shape:
    #   {"raw": {name→value}, "widget_data": [...], "total_widgets": int, "filled": int}
    # (It does NOT return a "widgets" key.)  Prior code read
    # ``widget_info.get("widgets")`` and therefore never entered Lane A —
    # every fillable CMS-1500 PDF silently fell through to the
    # align+OCR path, which is both slower and less accurate.
    #
    # We also prefer counting only TEXT-type widgets (pymupdf field_type==7)
    # toward the Lane-A threshold.  A form full of unchecked/checked boxes
    # with no text fields gives us no data-bearing content and should go
    # through Lane C instead.
    if form_type in (FormType.CMS1500, FormType.UB04):
        widget_info = await asyncio.to_thread(
            pipeline._extract_widgets, file_path
        )
        if widget_info:
            widget_data = widget_info.get("widget_data") or []
            text_filled = sum(
                1
                for w in widget_data
                if w.get("type") == 7 and (w.get("value") or "").strip()
            )
            any_filled = int(widget_info.get("filled", 0)) or sum(
                1 for w in widget_data if (w.get("value") or "").strip()
            )
            threshold = 10 if form_type == FormType.CMS1500 else 3
            if text_filled >= threshold:
                lane = "A"
                reason = (
                    f"{text_filled} filled TEXT widgets "
                    f"(>= {threshold}; total filled={any_filled})"
                )

    # Lane B: digital text layer for non-structured forms ONLY.
    # For CMS-1500/UB-04, Lane B's word-to-zone matching is unreliable because
    # schema bboxes are calibrated to the standard template — if the PDF is a
    # custom layout (digital PDF with different coords) the extraction produces
    # duplicated/garbled text. Route to Lane C which performs template-aware
    # alignment + per-zone OCR instead.
    if lane == "C" and state.get("has_digital_text"):
        if form_type in (FormType.CMS1500, FormType.UB04):
            reason = "digital text present but form is structured → Lane C (align+OCR)"
        else:
            lane = "B"
            reason = "digital text layer present (non-structured form)"

    elapsed = time.time() - t0
    out = {"lane": lane, "plan_reason": reason}
    _update_trace_in_place(out, state, "plan", elapsed)
    return out


async def extract_widgets_node(state: GraphState) -> Dict[str, Any]:
    """Lane A — fillable PDF widgets."""
    t0 = time.time()
    pipeline = await _get_pipeline(state["config"])
    widget_info = await asyncio.to_thread(
        pipeline._extract_widgets, state["file_path"]
    )
    blocks: List[DetectedBlock] = []
    extracted: Dict[str, Any] = {}
    method = "lane_a_acroform_widgets"
    if widget_info:
        # _map_widgets_to_schema returns (fields_dict, blocks_list)
        result = await asyncio.to_thread(
            pipeline._map_widgets_to_schema,
            widget_info,
            state["form_type"],
        )
        if isinstance(result, tuple) and len(result) == 2:
            extracted, blocks = result
        elif isinstance(result, dict):
            blocks = result.get("blocks", [])
            extracted = result.get("fields", {})
    elapsed = time.time() - t0
    out = {
        "blocks": blocks,
        "extracted_fields": extracted,
        "extraction_method": method,
    }
    _update_trace_in_place(out, state, "widgets", elapsed)
    return out


async def align_node(state: GraphState) -> Dict[str, Any]:
    """Lane C — align scan to template.

    Also updates ``width`` / ``height`` to match the aligned image.  The
    template warp renders into ``(tw, th)`` pixel space, and when the
    uploaded scan was captured at a different DPI (e.g. 600 dpi scans vs
    the template's 300 dpi render) the original image dimensions do NOT
    equal the aligned image dimensions.  Every downstream consumer —
    schema-zone pixel lookups, the Reducto exporter, and the frontend
    overlay math — needs the aligned dims to avoid the "bboxes shifted
    left/up" look the user keeps seeing.
    """
    t0 = time.time()
    pipeline = await _get_pipeline(state["config"])
    image = state["image"]
    aligned = image
    used, success, quality = False, False, 0.0
    homography_matrix: Optional[List[List[float]]] = None
    if state["config"].enable_alignment \
            and state["form_type"] == FormType.CMS1500:
        try:
            align_res = await pipeline.alignment_agent.process(
                image, state["form_type"],
            )
            if align_res is not None and align_res.success \
                    and align_res.aligned_image is not None:
                aligned = align_res.aligned_image
                used, success = True, True
                quality = float(align_res.alignment_quality or 0.0)
                # Capture the forward homography (original → template).
                # finalize_node uses this to compute per-bbox quads in the
                # ORIGINAL image's pixel space so the frontend can overlay
                # on the user's uploaded PDF instead of the visibly-warped
                # aligned render.
                H = getattr(align_res, "homography_matrix", None)
                if H is not None:
                    try:
                        homography_matrix = np.asarray(H, dtype=float).tolist()
                    except Exception:
                        homography_matrix = None
            else:
                used = True
                quality = float(getattr(align_res, "alignment_quality", 0.0) or 0.0)
        except Exception as e:
            logger.warning("Alignment failed: %s", e)
    elapsed = time.time() - t0
    out: Dict[str, Any] = {
        "aligned_image": aligned,
        "alignment_used": used,
        "alignment_success": success,
        "alignment_quality": quality,
        "homography_matrix": homography_matrix,
    }
    # Keep width/height in sync with the image we actually hand downstream.
    # When alignment succeeded the aligned image may have different dims
    # than the original (template-space warp into (tw, th)).
    try:
        if aligned is not None and hasattr(aligned, "shape"):
            out["width"] = int(aligned.shape[1])
            out["height"] = int(aligned.shape[0])
            out["aligned_size"] = (int(aligned.shape[1]), int(aligned.shape[0]))
    except Exception:
        pass
    _update_trace_in_place(out, state, "align", elapsed)
    return out


async def extract_sections_node(state: GraphState) -> Dict[str, Any]:
    """Lane C Tier-1 — layout-first section extraction.

    Runs AFTER ``align_node`` and BEFORE ``extract_scan_node`` on scanned
    CMS/UB forms.  Does:

      1. Groups the form schema into ~8–22 semantic sections (row bands
         with optional L/R column split) — this is generic spatial
         clustering, no form-specific rules.
      2. Crops the aligned image to each section.
      3. Calls a MID-size VLM (default ``openbmb/minicpm-o4.5``) on each
         crop with ONLY that section's fields in the prompt.  Small
         model, small prompt, small output → accurate.
      4. Stores ``{section_id: {field_id: value}}`` in state.

    Downstream:
      * ``extract_scan_node`` seeds Florence-2 blocks with any values the
        VLM already filled, and ONLY runs Florence-2 on fields the VLM
        left blank (so per-field OCR cost drops ~80%).
      * ``validate_node`` / ``reflect_node`` behave as before — if a VLM
        answer fails a validator, the rescue ladder still fires.

    Fail-safe:
      * Any Ollama/HTTP error leaves ``section_values`` empty and
        ``sections_used=False`` — downstream degrades to the pure
        Florence-2 path exactly as today.
    """
    t0 = time.time()
    config = state["config"]
    form_type = state["form_type"]

    out: Dict[str, Any] = {
        "sections": [],
        "section_values": {},
        "section_confidences": {},
        "section_meta": {},
        "sections_used": False,
    }

    # Resolve enable_sections in priority order:
    #   1. method_override="sections" → force ON (user explicitly picked it).
    #   2. ENABLE_SECTIONS env var if set.
    #   3. config.enable_sections.
    override = str(
        getattr(config, "method_override", "auto") or "auto",
    ).lower()
    if override == "sections":
        enable_sections = True
    else:
        env_flag = os.environ.get("ENABLE_SECTIONS")
        if env_flag is not None:
            enable_sections = env_flag.strip().lower() in (
                "1", "true", "yes", "on",
            )
        else:
            enable_sections = bool(getattr(config, "enable_sections", False))
    if not enable_sections:
        out["section_meta"]["_global"] = {"skipped": "disabled_in_config"}
        _update_trace_in_place(out, state, "extract_sections", time.time() - t0)
        return out

    if form_type not in (FormType.CMS1500, FormType.UB04):
        out["section_meta"]["_global"] = {"skipped": f"unsupported_form:{form_type}"}
        _update_trace_in_place(out, state, "extract_sections", time.time() - t0)
        return out

    aligned_image = state.get("aligned_image")
    if aligned_image is None or getattr(aligned_image, "size", 0) == 0:
        aligned_image = state.get("image")
    if aligned_image is None or getattr(aligned_image, "size", 0) == 0:
        out["section_meta"]["_global"] = {"skipped": "no_image"}
        _update_trace_in_place(out, state, "extract_sections", time.time() - t0)
        return out

    # ── Load schema ──────────────────────────────────────────────────────
    import json as _json
    from pathlib import Path as _Path
    schema_file_map = {
        FormType.CMS1500: _Path(__file__).parent.parent.parent.parent
            / "data" / "schemas" / "cms-1500.json",
        FormType.UB04: _Path(__file__).parent.parent.parent.parent
            / "data" / "schemas" / "ub-04.json",
    }
    schema_path = schema_file_map.get(form_type)
    schema: Dict[str, Any] = {}
    if schema_path and schema_path.exists():
        try:
            with open(schema_path) as fh:
                schema = _json.load(fh) or {}
        except Exception as e:
            logger.warning("extract_sections_node: schema load failed: %s", e)
    if not schema.get("fields"):
        out["section_meta"]["_global"] = {"skipped": "no_schema_fields"}
        _update_trace_in_place(out, state, "extract_sections", time.time() - t0)
        return out

    # ── Detect sections (spatial clustering of schema fields) ────────────
    from src.pipelines.layout.sections import detect_sections, sections_to_blocks
    try:
        sections = detect_sections(schema)
    except Exception as e:
        logger.warning("detect_sections failed: %s", e)
        out["section_meta"]["_global"] = {"error": f"detect_failed: {e}"}
        _update_trace_in_place(out, state, "extract_sections", time.time() - t0)
        return out
    if not sections:
        out["section_meta"]["_global"] = {"skipped": "no_sections"}
        _update_trace_in_place(out, state, "extract_sections", time.time() - t0)
        return out

    img_h = int(aligned_image.shape[0])
    img_w = int(aligned_image.shape[1])
    section_blocks = sections_to_blocks(sections, img_w, img_h)
    out["sections"] = section_blocks

    # ── Instantiate VLM extractor (one per request) ──────────────────────
    from src.pipelines.vlm.section_extractor import (
        SectionVLMExtractor, crop_section,
    )
    from utils.config import Config

    model = (
        os.environ.get("VLM_MODEL_SECTION")
        or getattr(config, "section_vlm_model", "")
        or getattr(Config, "VLM_MODEL_SECTION", "openbmb/minicpm-o4.5:latest")
    )
    # Comma-separated list of fallback models, tried in order if the primary
    # errors out (e.g. Ollama runtime can't serve it).  Typical config:
    #   VLM_MODEL_SECTION_FALLBACK=minicpm-v,llava:latest
    fallback_raw = (
        os.environ.get("VLM_MODEL_SECTION_FALLBACK")
        or getattr(config, "section_vlm_fallback_models", "")
        or getattr(Config, "VLM_MODEL_SECTION_FALLBACK", "minicpm-v")
    )
    if isinstance(fallback_raw, (list, tuple)):
        fallback_models = [str(m).strip() for m in fallback_raw if str(m).strip()]
    else:
        fallback_models = [m.strip() for m in str(fallback_raw).split(",") if m.strip()]
    timeout = int(getattr(config, "section_vlm_timeout_s", 120))
    max_crop = int(getattr(config, "section_max_crop_side_px", 1400))

    extractor = SectionVLMExtractor(
        ollama_host=Config.OLLAMA_HOST,
        model=model,
        fallback_models=fallback_models,
        timeout=timeout,
        max_crop_side=max_crop,
    )

    # Index schema fields by id for fast lookup when building per-section
    # prompts.
    fields_by_id = {f["id"]: f for f in schema["fields"] if f.get("id")}
    form_key = form_type.value if hasattr(form_type, "value") else str(form_type)

    # ── Run sections with bounded concurrency ────────────────────────────
    max_parallel = max(1, int(getattr(config, "section_vlm_max_parallel", 3)))
    sem = asyncio.Semaphore(max_parallel)

    async def _run_one(sec) -> Tuple[str, Any]:
        async with sem:
            section_fields = [
                fields_by_id[fid] for fid in sec.field_ids if fid in fields_by_id
            ]
            crop = crop_section(aligned_image, sec.bbox_norm)
            try:
                res = await asyncio.to_thread(
                    extractor.extract,
                    crop, sec.id, sec.label, section_fields, form_key,
                )
            except Exception as e:
                logger.warning(
                    "Section VLM crashed for %s: %s", sec.id, e,
                )
                return sec.id, None
            return sec.id, res

    results = await asyncio.gather(*(_run_one(s) for s in sections))

    n_sections_ok = 0
    n_filled_total = 0
    for sec_id, res in results:
        if res is None:
            out["section_meta"][sec_id] = {"error": "exception"}
            continue
        meta = {
            "model": res.model,
            "latency_s": res.latency_s,
            "success": res.success,
            "error": res.error or "",
        }
        out["section_meta"][sec_id] = meta
        if not res.success:
            continue
        values: Dict[str, str] = {}
        confs: Dict[str, float] = {}
        for fid, ex in res.values.items():
            v = (ex.value or "").strip()
            if v:
                values[fid] = v
                confs[fid] = float(ex.confidence or 0.0)
                n_filled_total += 1
        if values:
            out["section_values"][sec_id] = values
            out["section_confidences"][sec_id] = confs
            n_sections_ok += 1

    out["sections_used"] = n_filled_total > 0
    out["section_meta"]["_global"] = {
        "n_sections": len(sections),
        "n_sections_with_values": n_sections_ok,
        "n_fields_filled": n_filled_total,
        "total_latency_s": time.time() - t0,
    }
    logger.info(
        "extract_sections: %d/%d sections returned values, %d fields filled, %.2fs",
        n_sections_ok, len(sections), n_filled_total, time.time() - t0,
    )

    _update_trace_in_place(out, state, "extract_sections", time.time() - t0)
    return out


async def extract_scan_node(state: GraphState) -> Dict[str, Any]:
    """Lane C — schema zones + batched OCR.

    Alignment-gate: if template alignment failed or produced a low-quality
    result we CAP the number of zones processed by the OCR agent. Running
    Florence-2 over 130+ misaligned bboxes is both expensive (~4 min) and
    unreliable — the blank-detector can't calibrate against known-blanks when
    the bbox coordinates don't correspond to actual field regions.
    """
    t0 = time.time()
    pipeline = await _get_pipeline(state["config"])
    aligned_image = state.get("aligned_image") if state.get("aligned_image") is not None \
        else state["image"]

    form_type = state["form_type"]
    alignment_used = bool(state.get("alignment_used", False))
    alignment_ok = bool(state.get("alignment_success", False))
    alignment_q = float(state.get("alignment_quality", 0.0))

    # For CMS-1500 we REQUIRE alignment — without it the schema zones are
    # meaningless. Record the failure so the caller can surface QA notes.
    low_alignment = (
        form_type == FormType.CMS1500
        and alignment_used
        and (not alignment_ok or alignment_q < 0.50)
    )

    # ALWAYS normalize schema zones against the image we're OCR-ing.
    # align_node now keeps state["width"/"height"] in sync, but we pull
    # straight from aligned_image.shape as a belt-and-braces guarantee —
    # if these ever disagree with the image we pass to `_load_schema_zones`
    # every subsequent bbox is off by the ratio, which is exactly the
    # visual shift the user reported.
    zone_w = int(aligned_image.shape[1]) if aligned_image is not None \
        and hasattr(aligned_image, "shape") else int(state["width"])
    zone_h = int(aligned_image.shape[0]) if aligned_image is not None \
        and hasattr(aligned_image, "shape") else int(state["height"])
    blocks = await pipeline._load_schema_zones(
        aligned_image, zone_w, zone_h, is_scan=True,
    )
    for b in blocks:
        b.metadata["form_type"] = str(form_type.value)

    if low_alignment:
        # Short-circuit heavy OCR. Keep the (empty) zones so VLM rescue can
        # still target a small priority subset in the reflect/rescue pass.
        logger.warning(
            "Low-alignment CMS-1500 (%.3f) — skipping full Florence-2 pass.",
            alignment_q,
        )
        elapsed = time.time() - t0
        out = {
            "blocks": blocks,
            "extracted_fields": {},
            "extraction_method": "lane_c_alignment_failed",
        }
        _update_trace_in_place(out, state, "extract_scan", elapsed)
        return out

    # ── Section-VLM seeding ──────────────────────────────────────────────
    #
    # If ``extract_sections_node`` already produced values for some fields,
    # write them onto the corresponding blocks and EXCLUDE those blocks
    # from the Florence-2 batch.  Net effect: Florence-2 only runs on the
    # residual (typically 5-15 fields instead of all ~50), and per-field
    # OCR latency drops proportionally.
    #
    # Never overwrite TABLE blocks (Box 24) — the VLM doesn't handle those
    # and the dedicated labeling_agent does below.  Never overwrite
    # SIGNATURE blocks either — those are intentionally left for the
    # "[SIGNED]" marker.  CHECKBOX blocks use the fill-ratio detector
    # (more reliable than VLM vision on small marks), so we also skip
    # seeding checkboxes.
    section_values = state.get("section_values") or {}
    section_confs = state.get("section_confidences") or {}
    section_meta = state.get("section_meta") or {}
    section_threshold = float(
        getattr(state["config"], "section_min_confidence", 0.55)
    )

    # Flatten {section_id: {field_id: value}} → {field_id: (value, conf, section_id)}
    flat_values: Dict[str, Tuple[str, float, str]] = {}
    for sid, vals in section_values.items():
        if not isinstance(vals, dict):
            continue
        conf_map = section_confs.get(sid) or {}
        for fid, v in vals.items():
            if not v:
                continue
            c = float(conf_map.get(fid, 0.0) or 0.0)
            # Prefer the highest-confidence section if a field happens to
            # appear in more than one (shouldn't happen with our detector
            # but the code is cheap and safe).
            cur = flat_values.get(fid)
            if cur is None or c > cur[1]:
                flat_values[fid] = (v, c, sid)

    seeded_ids: set = set()
    for b in blocks:
        seed = flat_values.get(b.id)
        if seed is None:
            continue
        v, c, sid = seed
        if not v or c < section_threshold:
            continue
        if b.block_type in (BlockType.TABLE, BlockType.SIGNATURE, BlockType.CHECKBOX):
            continue
        b.text = v
        b.confidence = max(float(b.confidence or 0.0), c)
        meta = b.metadata or {}
        meta["source"] = f"section_vlm:{sid}"
        meta["ocr_engine"] = (
            f"section_vlm:{section_meta.get(sid, {}).get('model', '')}"
        )
        meta["blank_status"] = "filled"
        meta["section_id"] = sid
        b.metadata = meta
        seeded_ids.add(b.id)

    if seeded_ids:
        logger.info(
            "extract_scan: section VLM seeded %d/%d blocks (Florence-2 will run on %d residual)",
            len(seeded_ids), len(blocks),
            len([b for b in blocks if b.id not in seeded_ids]),
        )

    residual = [b for b in blocks if b.id not in seeded_ids]
    if residual:
        # process_blocks mutates blocks in-place, so the order in `blocks`
        # is preserved even though we only pass a subset.
        await pipeline.ocr_agent.process_blocks(aligned_image, residual)

    # NOTE: Table extraction has moved to its own node
    # ``extract_tables_node`` which runs right after this one in the
    # graph.  Keeping it there means Box 24 gets a dedicated route,
    # its own retry ladder (VLM → raw-OCR + regex row parser), and its
    # own trace entry so we can reason about table latency in
    # isolation.  TABLE blocks are untouched by Florence-2 here (they
    # were already filtered out of the residual batch earlier).

    # Template-label leak guard.  Florence-2 sometimes returns the
    # printed form label (e.g. "ORIGINAL REF. NO", "CLAIM CODES") as
    # the value when the underlying pixel crop was essentially empty.
    # We clear those values here BEFORE they hit extracted_fields, so
    # validation and the rescue ladder see an empty field instead of a
    # confidently-wrong one.  Detection is schema-driven and
    # punctuation-insensitive, so it adapts to any form we have a
    # schema for.
    from src.pipelines.graph.rescue_strategies import _looks_like_template_label
    form_type_str = ""
    ft = state.get("form_type")
    if ft is not None:
        form_type_str = getattr(ft, "value", "") or str(ft)
    n_cleared_template_leak = 0
    for b in blocks:
        if b.block_type in (BlockType.CHECKBOX, BlockType.SIGNATURE,
                            BlockType.TABLE):
            continue
        text = (b.text or "").strip()
        if text and _looks_like_template_label(text, form_type_str):
            b.metadata = b.metadata or {}
            b.metadata["template_leak_cleared"] = text
            b.metadata["blank_status"] = "cleared_as_template_leak"
            b.text = ""
            n_cleared_template_leak += 1
    if n_cleared_template_leak:
        logger.info(
            "extract_scan: cleared %d template-label leak(s) from Florence output",
            n_cleared_template_leak,
        )

    # Assemble extracted_fields: take first non-empty value per id
    extracted: Dict[str, Any] = {}
    for b in blocks:
        txt = (b.text or "").strip()
        if txt:
            extracted[b.id] = txt

    elapsed = time.time() - t0
    method = "lane_c_scan_ocr_v2"
    if seeded_ids:
        # Make it visible in traces that sections shaved work off OCR.
        method = f"lane_c_sections+ocr_v2 (seeded={len(seeded_ids)})"
    out = {
        "blocks": blocks,
        "extracted_fields": extracted,
        "extraction_method": method,
    }
    _update_trace_in_place(out, state, "extract_scan", elapsed)
    return out


async def extract_tables_node(state: GraphState) -> Dict[str, Any]:
    """Dedicated table-extraction route.

    Historically tables (CMS-1500 Box 24) were processed inline inside
    ``extract_scan_node`` and silently failed: ``labeling_agent`` was
    never initialised in the graph path, so ``_ollama_available`` stayed
    ``None`` and ``_call_vlm`` short-circuited with an empty string.
    The block's text stayed blank and nothing wrote back into
    ``extracted_fields``.

    Splitting this out into its own node buys us three things:

    1. Clear routing — tables follow their own lane:
       VLM primary → VLM fallback → raw-OCR row parser fallback.
    2. Separate tracing/timing so we can tell how much Box 24 costs vs
       the rest of the page.
    3. An explicit place to write table rows back into
       ``extracted_fields`` (previously only the summary string landed
       there, and only when VLM worked).

    This node is a no-op when ``config.enable_vlm_tables`` is False or
    no TABLE blocks are present.
    """
    t0 = time.time()
    cfg = state["config"]
    out: Dict[str, Any] = {"table_debug": list(state.get("table_debug") or [])}

    if not cfg.enable_vlm_tables:
        logger.info("extract_tables: enable_vlm_tables=False — skipping")
        _update_trace_in_place(out, state, "extract_tables", time.time() - t0)
        return out

    blocks = state.get("blocks") or []
    table_blocks = [b for b in blocks if b.block_type == BlockType.TABLE]
    if not table_blocks:
        logger.info("extract_tables: no TABLE blocks in schema — skipping")
        _update_trace_in_place(out, state, "extract_tables", time.time() - t0)
        return out

    pipeline = await _get_pipeline(cfg)
    aligned_image = state.get("aligned_image")
    if aligned_image is None:
        aligned_image = state.get("image")
    if aligned_image is None:
        logger.warning("extract_tables: no image available, aborting")
        _update_trace_in_place(out, state, "extract_tables", time.time() - t0)
        return out

    extracted = dict(state.get("extracted_fields") or {})

    for tb in table_blocks:
        tb_dbg: Dict[str, Any] = {
            "block_id": tb.id,
            "bbox": list(tb.bbox),
            "status": "pending",
        }
        try:
            tb_out = await pipeline.labeling_agent.process_table(
                aligned_image, tb,
            )
        except Exception as e:
            logger.warning("extract_tables: process_table raised for %s: %s",
                           tb.id, e)
            tb_dbg["status"] = "exception"
            tb_dbg["error"] = str(e)
            out["table_debug"].append(tb_dbg)
            continue

        if tb_out is None:
            tb_dbg["status"] = "returned_none"
            out["table_debug"].append(tb_dbg)
            continue

        # process_table returns a dict for TABLE blocks; assimilate.
        if isinstance(tb_out, dict):
            summary = (tb_out.get("summary") or "").strip()
            raw_text = (tb_out.get("raw_text") or "").strip()
            rows = tb_out.get("rows") or []
            method = tb_out.get("extraction_method") or ""

            # Fallback: VLM returned nothing usable.  Try a Florence-2
            # raw-OCR sweep on the table crop and run a tolerant regex
            # parser on the line output.  This is our last-chance row
            # extractor when neither VLM model can parse the service
            # lines — it won't be as accurate as VLM but beats zero.
            if not rows:
                try:
                    rows_fb, summary_fb, method_fb = await _florence2_table_fallback(
                        pipeline, aligned_image, tb,
                    )
                    if rows_fb:
                        rows = rows_fb
                        summary = summary_fb or summary
                        method = method_fb or "florence2_row_fallback"
                        tb_dbg["fallback_used"] = method
                except Exception as e:
                    logger.warning(
                        "extract_tables: florence2 row fallback failed for %s: %s",
                        tb.id, e,
                    )
                    tb_dbg["fallback_error"] = str(e)

            # Write back onto the block so downstream (finalize) picks it up.
            if summary:
                tb.text = summary
            elif raw_text:
                tb.text = raw_text
            tb.metadata = tb.metadata or {}
            tb.metadata.update({
                "table_rows": rows,
                "table_total_rows": int(
                    tb_out.get("total_rows") or len(rows)
                ),
                "table_extraction_method": method,
            })
            if tb_out.get("vlm_error"):
                tb.metadata["table_vlm_error"] = str(tb_out["vlm_error"])
            if rows:
                tb.metadata["ocr_engine"] = (
                    f"vlm_table:{method}" if method else "vlm_table"
                )
                tb.confidence = max(float(tb.confidence or 0.0), 0.7)
                # Make sure the field is not blank-flagged.
                tb.metadata["blank_status"] = "filled"

            # Get into extracted_fields — used by validate/finalize.
            if tb.text and tb.text.strip():
                extracted[tb.id] = tb.text

            tb_dbg.update({
                "status": "ok" if rows else "no_rows_parsed",
                "extraction_method": method,
                "row_count": len(rows),
                "summary_head": (summary or raw_text)[:200],
                "vlm_error": tb_out.get("vlm_error"),
                "vlm_primary_error": tb_out.get("vlm_primary_error"),
                "vlm_fallback_error": tb_out.get("vlm_fallback_error"),
                "vlm_model_used": tb_out.get("vlm_model_used"),
                "vlm_raw_primary": (tb_out.get("vlm_raw_primary") or "")[:400],
                "vlm_raw_fallback": (tb_out.get("vlm_raw_fallback") or "")[:400],
            })
        else:
            # Legacy return type (DetectedBlock).  Kept for safety.
            tb.text = getattr(tb_out, "text", None) or tb.text
            extra = getattr(tb_out, "metadata", None) or {}
            if extra:
                tb.metadata = tb.metadata or {}
                tb.metadata.update(extra)
            if tb.text and tb.text.strip():
                extracted[tb.id] = tb.text
            tb_dbg["status"] = "legacy_return_type"

        out["table_debug"].append(tb_dbg)

    out["blocks"] = blocks
    out["extracted_fields"] = extracted
    _update_trace_in_place(out, state, "extract_tables", time.time() - t0)
    return out


async def _florence2_table_fallback(
    pipeline: Any,
    aligned_image: np.ndarray,
    tb: "DetectedBlock",
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Last-resort row extractor using Florence-2 on a horizontally-sliced
    table crop.

    Approach:
      * Crop the table bbox.
      * Split vertically into N bands (6 rows for CMS-1500 Box 24).
      * Run Florence-2 ``<OCR>`` on each band.
      * Regex-extract date / place-of-service / CPT / money tokens
        from the line text — populate a partial row for whatever the
        OCR could pick up.

    Returns ``(rows, summary, method)``.  ``rows`` is empty if nothing
    was recovered.  The caller decides whether to keep the result.
    """
    rows: List[Dict[str, Any]] = []
    h, w = aligned_image.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in tb.bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return rows, "", ""

    table_crop = aligned_image[y0:y1, x0:x1]
    if table_crop.size == 0:
        return rows, "", ""

    # Clean the red template if available — same helper the VLM path uses.
    try:
        from src.processing.preprocessing import remove_red_template_text
        table_crop = remove_red_template_text(table_crop)
    except Exception:
        pass

    # CMS-1500 Box 24 is a 6-row table.  For generic tables we default to
    # 6 bands which is a sensible upper bound for a claim-form service
    # section.  The parser tolerates partial output, so over-slicing is
    # cheap.
    num_rows = 6
    band_h = (y1 - y0) // num_rows
    if band_h < 20:
        return rows, "", ""

    # Lazy import because OCRAgent is heavyweight.
    ocr_agent = getattr(pipeline, "ocr_agent", None)
    if ocr_agent is None:
        return rows, "", ""

    # Each row gets a small DetectedBlock so we can reuse the agent's
    # single-block process path (handles Florence-2, fallbacks, etc).
    date_re = re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b")
    money_re = re.compile(r"\$?\s*(\d{1,4}(?:[,.]\d{2})?)")
    cpt_re = re.compile(r"\b([A-Z]?\d{4,5}[A-Z]?)\b")
    pos_re = re.compile(r"\b(1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|0[1-9])\b")

    for i in range(num_rows):
        ry0 = y0 + i * band_h
        ry1 = y1 if i == num_rows - 1 else (y0 + (i + 1) * band_h)
        band_block = DetectedBlock(
            id=f"{tb.id}_row{i+1}",
            block_type=BlockType.TEXT,
            bbox=(x0, ry0, x1, ry1),
            text="",
            confidence=0.0,
            metadata={"field_type": "text", "source": "table_row_fallback"},
        )
        try:
            await asyncio.wait_for(
                ocr_agent.process(aligned_image, band_block),
                timeout=30.0,
            )
        except Exception as e:
            logger.debug("florence2 table row %d OCR failed: %s", i + 1, e)
            continue

        txt = (band_block.text or "").strip()
        if not txt or len(txt) < 4:
            continue

        # Regex-extract the things we can detect structurally.  This is
        # intentionally conservative — we only populate fields where the
        # token shape is unambiguous.
        dates = date_re.findall(txt)
        moneys = money_re.findall(txt)
        cpts = cpt_re.findall(txt)
        # POS is tricky (many codes match digit pairs).  Prefer a POS
        # code that appears BEFORE a CPT in the text.
        pos_matches = pos_re.findall(txt)

        row: Dict[str, Any] = {
            "date_from": "", "date_to": "", "place_of_service": "",
            "cpt_code": "", "modifier": "", "charges": "",
            "units": "", "npi": "",
        }
        if dates:
            row["date_from"] = "/".join(dates[0])
            if len(dates) > 1:
                row["date_to"] = "/".join(dates[1])
            else:
                row["date_to"] = row["date_from"]
        if pos_matches:
            row["place_of_service"] = pos_matches[0]
        if cpts:
            # Pick the token that looks like a CPT/HCPCS (5 digits or 1
            # letter + 4 digits).  Skip anything that's part of a date.
            for c in cpts:
                if len(c) >= 4 and c not in (row["date_from"], row["date_to"]):
                    row["cpt_code"] = c
                    break
        if moneys:
            # Charges are usually the largest number on the row.
            try:
                largest = max(moneys, key=lambda s: float(s.replace(",", "")))
                row["charges"] = largest
            except Exception:
                row["charges"] = moneys[0]

        if any(row[k] for k in ("date_from", "cpt_code", "charges")):
            rows.append(row)

    if not rows:
        return rows, "", ""

    summary_parts: List[str] = []
    for r in rows:
        p: List[str] = []
        if r["date_from"]:
            p.append(r["date_from"])
        if r["date_to"] and r["date_to"] != r["date_from"]:
            p.append(r["date_to"])
        if r["place_of_service"]:
            p.append(r["place_of_service"])
        if r["cpt_code"]:
            p.append(r["cpt_code"])
        if r["charges"]:
            p.append(f"${r['charges']}")
        if p:
            summary_parts.append(" | ".join(p))

    return rows, "\n".join(summary_parts), "florence2_row_fallback"


async def extract_digital_node(state: GraphState) -> Dict[str, Any]:
    """Lane B — digital text layer zone matching.

    Safety nets:
      * If `fields` is empty (no schema loaded for form_type) → downgrade to C.
      * If matched blocks are empty or too sparse → downgrade to C.
      * If too many blocks share identical text (sign of misaligned template
        vs custom layout) → downgrade to C.
    """
    t0 = time.time()
    pipeline = await _get_pipeline(state["config"])
    words = state.get("digital_words") or []
    import json as _json
    from pathlib import Path as _Path
    schema_file_map = {
        FormType.CMS1500: _Path(__file__).parent.parent.parent.parent
            / "data" / "schemas" / "cms-1500.json",
        FormType.UB04: _Path(__file__).parent.parent.parent.parent
            / "data" / "schemas" / "ub-04.json",
    }
    schema_path = schema_file_map.get(state["form_type"])
    fields: List[Dict[str, Any]] = []
    if schema_path and schema_path.exists():
        try:
            with open(schema_path) as f:
                fields = (_json.load(f) or {}).get("fields", []) or []
        except Exception:
            fields = []

    # Guard: no schema → cannot match to zones → downgrade to Lane C
    if not fields:
        out = {
            "blocks": [],
            "extracted_fields": {},
            "extraction_method": "lane_b_downgraded_to_c",
            "lane": "C",
            "plan_reason": "no schema for form_type → scan lane",
        }
        _update_trace_in_place(out, state, "digital_downgrade", time.time() - t0)
        return out

    blocks = await pipeline._match_ocr_to_zones(
        words, fields, state["width"], state["height"], state["image"],
    )

    filled = sum(1 for b in blocks if (b.text or "").strip())

    # Sparsity downgrade (CMS-1500/UB-04 specifically)
    min_fill = 10 if state["form_type"] == FormType.CMS1500 else 5
    if state["form_type"] in (FormType.CMS1500, FormType.UB04) and filled < min_fill:
        out = {
            "blocks": [], "extracted_fields": {},
            "extraction_method": "lane_b_downgraded_to_c", "lane": "C",
            "plan_reason": f"digital match too sparse ({filled} filled < {min_fill})",
        }
        _update_trace_in_place(out, state, "digital_downgrade", time.time() - t0)
        return out

    # Template-mismatch downgrade — if >30% of filled blocks share the same
    # text value, the schema bboxes don't line up with this layout and we're
    # just copying the same words across many zones.
    texts = [(b.text or "").strip() for b in blocks if (b.text or "").strip()]
    if len(texts) >= 10:
        from collections import Counter
        top_count = Counter(texts).most_common(1)[0][1]
        if top_count / len(texts) > 0.30:
            out = {
                "blocks": [], "extracted_fields": {},
                "extraction_method": "lane_b_downgraded_to_c", "lane": "C",
                "plan_reason": (
                    f"template mismatch: {top_count}/{len(texts)} blocks "
                    "share identical text → scan lane"
                ),
            }
            _update_trace_in_place(
                out, state, "digital_downgrade", time.time() - t0,
            )
            return out

    extracted = {
        b.id: (b.text or "").strip() for b in blocks if (b.text or "").strip()
    }
    elapsed = time.time() - t0
    out = {
        "blocks": blocks,
        "extracted_fields": extracted,
        "extraction_method": "lane_b_digital_text",
    }
    _update_trace_in_place(out, state, "extract_digital", elapsed)
    return out


async def validate_node(state: GraphState) -> Dict[str, Any]:
    """Run the ValidationAgent over the current blocks."""
    t0 = time.time()
    pipeline = await _get_pipeline(state["config"])
    blocks = state.get("blocks") or []
    extracted = state.get("extracted_fields") or {}
    method = state.get("extraction_method", "")

    if not state["config"].enable_validators:
        validation: Dict[str, Any] = {"errors": [], "warnings": [], "qa_notes": []}
    else:
        validation = await pipeline.validation_agent.process(blocks, extracted) or {}

    # Attach pipeline-level QA notes surfaced from routing decisions.
    qa = list(validation.get("qa_notes") or [])
    if method == "lane_c_alignment_failed":
        qa.append({
            "code": "alignment_failed",
            "message": (
                "Template alignment did not meet quality threshold — "
                "field-level OCR was skipped to save cost. Consider "
                "uploading a cleaner scan or enabling VLM fallback."
            ),
            "alignment_quality": float(state.get("alignment_quality", 0.0)),
        })
    if method == "lane_b_downgraded_to_c":
        qa.append({
            "code": "digital_downgrade",
            "message": state.get("plan_reason", "digital lane downgraded"),
        })
    validation["qa_notes"] = qa

    elapsed = time.time() - t0
    out = {"validation": validation}
    _update_trace_in_place(out, state, "validate", elapsed)
    return out


async def reflect_node(state: GraphState) -> Dict[str, Any]:
    """Identify fields that need rescue, picking the NEXT untried method.

    This is the agentic 'revise' step — after each rescue iteration we
    re-inspect validation, confidence, and ink signals and build a fresh
    candidate list.  A field that has already tried every method on the
    ladder — or where the ladder has converged on the same invalid answer
    from independent models — is dropped so we don't thrash.

    Everything here is derived from block metadata (validator result, ink
    ratio, confidence, type) so it works on any form we plug a schema in
    for — no form-specific field-id lists.

    Rescue triggers (in priority order):
      - typed validator failure
      - empty value + visible ink (blank_status != blank)
      - low confidence on a non-blank cell
      - rescued-but-still-invalid (needs next method)

    Rescue STOP conditions (generic — no hardcoded words/ids):
      - every strategy in the ladder has been tried for the field
      - two or more independent rescue methods returned the same
        non-validating candidate — strong signal the crop is wrong, not
        the OCR, so more model calls won't help
    """
    t0 = time.time()
    blocks = state.get("blocks") or []
    validation = state.get("validation") or {}
    history: Dict[str, List[str]] = dict(state.get("rescue_history") or {})

    invalid_ids = {
        e.get("field_id")
        for e in (validation.get("errors") or [])
        if e.get("field_id")
    }

    from src.pipelines.graph.rescue_strategies import (
        RESCUE_LADDER, ladder_for_field_type, _run_validator,
        _validator_name_for,
    )

    # Build an id → field_type lookup once so ``_already_exhausted``
    # can consult the correct per-type ladder length.  Handwriting
    # fields have a slightly different ladder than numeric ones, and
    # using ``RESCUE_LADDER`` (the default) as the universal yardstick
    # would let some fields burn extra rescue iterations while
    # prematurely declaring others exhausted.
    _block_field_type: Dict[str, str] = {
        b.id: str((b.metadata or {}).get("field_type", "") or "").lower()
        for b in blocks
    }

    def _already_exhausted(fid: str) -> bool:
        tried = history.get(fid) or []
        ladder = ladder_for_field_type(_block_field_type.get(fid, ""))
        return len(tried) >= len(ladder)

    def _norm(s: str) -> str:
        # Case + whitespace + surrounding punctuation-insensitive compare,
        # so "$ 1,908.00" and "1,908.00" count as the same candidate when
        # we're looking for multi-model agreement.
        import re as _re
        return _re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    def _consensus_invalid(b) -> bool:
        """True iff ≥2 distinct methods returned the same failing
        candidate.  Totally schema-agnostic — uses the per-block history
        that every strategy already writes to ``rescue_history_texts``.
        """
        meta = b.metadata or {}
        texts = [t for t in (meta.get("rescue_history_texts") or []) if t]
        if len(texts) < 2:
            return False
        field_type = str(meta.get("field_type", "") or "").lower()
        validator = _validator_name_for(field_type)
        # Count identical candidates.  If ≥2 agree AND that candidate fails
        # typed validation, the bbox is the problem.
        seen: Dict[str, int] = {}
        for t in texts:
            k = _norm(t)
            if not k:
                continue
            seen[k] = seen.get(k, 0) + 1
            if seen[k] >= 2 and not _run_validator(validator, t):
                return True
        return False

    scored: List[tuple] = []
    for b in blocks:
        if b.block_type in (BlockType.CHECKBOX, BlockType.SIGNATURE, BlockType.TABLE):
            continue
        fid = b.id
        if _already_exhausted(fid):
            continue
        if _consensus_invalid(b):
            # Mark so the UI / debug tab can show why we stopped.
            b.metadata["rescue_stopped"] = "consensus_invalid"
            continue
        text = (b.text or "").strip()
        conf = float(b.confidence or 0.0)
        meta = b.metadata or {}
        blank_status = meta.get("blank_status", "uncertain")
        field_type = str(meta.get("field_type", "") or "").lower()
        is_text_field = field_type in ("", "text", "address", "name")

        priority = -1
        _TYPED_FT = {
            "npi", "phone", "zip", "date", "date_range", "money", "currency",
            "tax_id", "cpt", "icd", "icd10", "hcpcs", "ndc", "member_id",
            "numeric", "ssn", "ein", "state",
        }
        is_typed = field_type in _TYPED_FT
        inner_ink = float(meta.get("inner_ink", 0.0) or 0.0)
        is_blank_flag = bool(meta.get("is_blank", False))
        _BLANK_STATUSES = (
            "blank", "blank_structural", "blank_ink",
            "blank_high", "blank_med",
            "cleared_as_template_leak", "cleared_as_unrescuable",
        )
        blank_like = (blank_status in _BLANK_STATUSES) or is_blank_flag
        if fid in invalid_ids and not (blank_like and not text):
            # Don't rescue fields the upstream OCR already confirmed blank
            # even if the validator flagged them as invalid (e.g. NPI
            # "required" errors on truly empty cells).  Otherwise rescue
            # would hallucinate a value on top of nothing.
            priority = 3
        elif (
            not text
            and not blank_like
            and is_typed
            and inner_ink > 0.08
        ):
            # ONLY rescue genuinely-empty cells when the field is strongly
            # typed (date/npi/phone/etc.) AND there is substantial ink
            # (not just template bleed-through).  Previously a 1% ink
            # threshold was enough to queue free-text cells like
            # ``9_other_insured_name`` or ``22_resubmission_code``, and
            # Florence/VLM would hallucinate text from adjacent cells.
            priority = 2
        elif is_text_field and conf < 0.55 and blank_status != "blank":
            # Free-text fields generally carry more information than typed
            # ones (names, addresses, descriptions) — give them a small
            # boost when confidence is shaky but the crop clearly has ink.
            priority = 1
        elif conf < 0.4 and blank_status != "blank":
            priority = 0
        if priority >= 0:
            scored.append((priority, fid))

    scored.sort(key=lambda t: -t[0])
    max_rescues = 12
    candidates = [fid for _, fid in scored[:max_rescues]]

    elapsed = time.time() - t0
    out = {"rescue_candidates": candidates}
    _update_trace_in_place(out, state, "reflect", elapsed)
    return out


async def rescue_node(state: GraphState) -> Dict[str, Any]:
    """Run ONE rescue iteration — per-field escalation ladder.

    For each candidate field we pick the next untried strategy
    (VLM → Florence-2 raw+upscale → TrOCR → SLM normalize), run it, and
    accept the result only if it improves the field.
    """
    t0 = time.time()
    pipeline = await _get_pipeline(state["config"])
    iteration = int(state.get("rescue_iterations", 0))

    blocks = state.get("blocks") or []
    candidates = set(state.get("rescue_candidates") or [])
    extracted = dict(state.get("extracted_fields") or {})
    aligned_image = state.get("aligned_image") \
        if state.get("aligned_image") is not None else state["image"]
    history: Dict[str, List[str]] = dict(state.get("rescue_history") or {})
    log: List[Dict[str, Any]] = list(state.get("rescue_log") or [])

    from src.pipelines.graph.rescue_strategies import (
        run_rescue_attempt, RESCUE_LADDER,
    )

    target_blocks = [b for b in blocks if b.id in candidates]

    def _run_all() -> int:
        nonlocal history, log, extracted
        accepted_n = 0
        for b in target_blocks:
            tried = list(history.get(b.id, []))
            attempt = run_rescue_attempt(
                pipeline, b, aligned_image, iteration=iteration, history=tried,
            )
            if attempt is None:
                continue
            history.setdefault(b.id, list(tried)).append(attempt.method)
            log.append({
                "field_id": attempt.field_id,
                "iteration": attempt.iteration,
                "method": attempt.method,
                "old_text": attempt.old_text,
                "new_text": attempt.new_text,
                "old_valid": attempt.old_valid,
                "new_valid": attempt.new_valid,
                "accepted": attempt.accepted,
                "reason": attempt.reason,
                "confidence": attempt.confidence,
            })
            if attempt.accepted:
                accepted_n += 1
                extracted[b.id] = b.text or ""
        return accepted_n

    accepted = await asyncio.to_thread(_run_all)

    elapsed = time.time() - t0
    out = {
        "rescue_iterations": iteration + 1,
        "vlm_rescue_count": int(state.get("vlm_rescue_count", 0)) + int(accepted),
        "extracted_fields": extracted,
        "rescue_history": history,
        "rescue_log": log,
    }
    _update_trace_in_place(out, state, "rescue", elapsed)
    logger.info(
        "Rescue iter=%d candidates=%d accepted=%d ladder=%s",
        iteration, len(target_blocks), accepted, RESCUE_LADDER,
    )
    return out


async def finalize_node(state: GraphState) -> Dict[str, Any]:
    """Map to business schema, build reducto format, assemble response."""
    t0 = time.time()
    pipeline = await _get_pipeline(state["config"])
    blocks = state.get("blocks") or []
    extracted = state.get("extracted_fields") or {}
    form_type = state["form_type"]

    # Final sanitization: strip any sentinel tokens that slipped through
    # the various OCR / rescue paths.  This is the single source of truth
    # for "clean values that ever leave the graph".
    from src.pipelines.graph.rescue_strategies import (
        _strip_sentinels, _post_rescue_sanitize,
    )
    extracted = {k: _strip_sentinels(str(v)) if isinstance(v, str) else v
                 for k, v in extracted.items()}
    # Drop keys whose sanitized value is now empty
    extracted = {k: v for k, v in extracted.items() if v}
    for b in blocks:
        if isinstance(b.text, str):
            b.text = _strip_sentinels(b.text)

    # Final rescue sweep: if typed fields still hold garbage or
    # template-leak text after the full ladder, apply one more
    # deterministic-repair attempt.  If that also fails we clear them
    # and mark as blank.  This keeps the response honest — we'd rather
    # show "empty" than a confidently-wrong value.  We pass the
    # rescue_history so the sweep can tell "the ladder already gave
    # up" from "we haven't tried yet".
    rescue_history = dict(state.get("rescue_history") or {})
    extracted, cleared_ids = _post_rescue_sanitize(
        blocks, extracted, rescue_history=rescue_history,
    )

    # Re-run validation on the cleaned-up data so errors for cleared
    # fields are dropped from the final response.
    cleared_set = set(cleared_ids or [])
    if cleared_set and state["config"].enable_validators:
        try:
            new_validation = await pipeline.validation_agent.process(
                blocks, extracted,
            ) or {}
            state_validation = state.get("validation") or {}
            # Preserve any qa_notes attached earlier by validate_node.
            new_validation["qa_notes"] = list(
                state_validation.get("qa_notes") or [],
            ) + list(new_validation.get("qa_notes") or [])
            # Record the cleanup action in qa_notes so the UI can show it.
            if cleared_ids:
                new_validation["qa_notes"].append({
                    "code": "post_rescue_cleanup",
                    "message": (
                        f"Cleared {len(cleared_ids)} typed field(s) that "
                        "still held unrescuable garbage after the full "
                        "4-step rescue ladder."
                    ),
                    "cleared_field_ids": list(cleared_ids),
                })
            state["validation"] = new_validation
        except Exception as e:
            logger.warning("Post-cleanup re-validation failed: %s", e)

    from src.pipelines.schemas.business_schema import (
        map_to_business_schema, merge_business_with_ocr,
    )
    business_details = {}
    business: Dict[str, Any] = {}
    try:
        form_key = form_type.value if hasattr(form_type, "value") \
            else str(form_type)
        ocr_shape = {
            "extracted_fields": extracted,
            "field_details": [{
                "id": b.id,
                "value": b.text or "",
                "confidence": float(b.confidence or 0.0),
                "bbox": list(b.bbox),
                "metadata": b.metadata or {},
            } for b in blocks],
        }
        business_details = map_to_business_schema(ocr_shape, form_key) or {}
        business = business_details.get("business_fields", {}) or {}
    except Exception as e:
        logger.warning("Business mapping failed: %s", e)

    # ── Per-field quads in ORIGINAL-image space ─────────────────────────
    # The bboxes currently live in aligned (template) pixel space.  If we
    # have the forward homography (original → template), we can invert it
    # and project each bbox's 4 corners back into the ORIGINAL image.
    # This lets the frontend show the user's uploaded PDF with overlays
    # that actually line up — which the user explicitly asked for after
    # finding the warped aligned view visually jarring.
    H_forward = state.get("homography_matrix")
    H_inv_np: Optional[np.ndarray] = None
    original_dims: Tuple[int, int] = (0, 0)
    try:
        orig_img = state.get("image")
        if orig_img is not None and hasattr(orig_img, "shape"):
            original_dims = (int(orig_img.shape[1]), int(orig_img.shape[0]))
    except Exception:
        original_dims = (0, 0)
    if H_forward is not None:
        try:
            import cv2 as _cv2_inv
            H_np = np.asarray(H_forward, dtype=np.float64)
            ok, H_inv_np = _cv2_inv.invert(H_np)
            if ok == 0:
                H_inv_np = None
        except Exception as e:
            logger.warning("Homography inversion failed: %s", e)
            H_inv_np = None

    def _project_bbox_to_original(
        bbox: Tuple[float, float, float, float],
    ) -> Optional[List[List[float]]]:
        """Project an aligned-space (x1,y1,x2,y2) rect onto the original
        image using H_inv.  Returns a 4-corner quad [[x,y], …] in
        original-image pixel space, or None if the homography isn't
        available or the projection leaves the image canvas.
        """
        if H_inv_np is None:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        pts = np.array([
            [[x1, y1]],
            [[x2, y1]],
            [[x2, y2]],
            [[x1, y2]],
        ], dtype=np.float32)
        try:
            import cv2 as _cv2_proj
            proj = _cv2_proj.perspectiveTransform(pts, H_inv_np)
        except Exception:
            return None
        out_pts: List[List[float]] = []
        for i in range(4):
            px = float(proj[i][0][0])
            py = float(proj[i][0][1])
            if not (np.isfinite(px) and np.isfinite(py)):
                return None
            # Clamp gently to the image canvas to avoid wild overhangs
            # when H_inv is a bit off near the margins.
            if original_dims[0] > 0:
                px = max(-20.0, min(original_dims[0] + 20.0, px))
            if original_dims[1] > 0:
                py = max(-20.0, min(original_dims[1] + 20.0, py))
            out_pts.append([px, py])
        return out_pts

    field_details = []
    for b in blocks:
        md = b.metadata or {}
        bbox_list = list(b.bbox)
        quad = _project_bbox_to_original(b.bbox)
        field_details.append({
            "id": b.id,
            "value": b.text or "",
            "confidence": float(b.confidence or 0.0),
            "bbox": bbox_list,
            "bbox_original_quad": quad,
            "block_type": b.block_type.value,
            "metadata": {
                "field_name": md.get("field_name", ""),
                "field_type": md.get("field_type", ""),
                "source": md.get("source", ""),
                "ocr_engine": md.get("ocr_engine", ""),
                "blank_status": md.get("blank_status", ""),
                "escalation": md.get("escalation", ""),
                "ink_ratio": md.get("ink_ratio", None),
                "inner_ink": md.get("inner_ink", None),
                "weighted_ink": md.get("weighted_ink", None),
                "components": md.get("components", None),
                "table_rows": md.get("table_rows", None),
                "table_total_rows": md.get("table_total_rows", None),
                "table_extraction_method": md.get("table_extraction_method", None),
            },
        })

    reducto = {}
    try:
        reducto = pipeline._to_reducto_format(
            {
                "field_details": field_details,
                "extracted_fields": extracted,
                "form_type": form_type.value if hasattr(form_type, "value")
                             else str(form_type),
            },
            state.get("width", 0),
            state.get("height", 0),
        ) or {}
    except Exception as e:
        logger.warning("Reducto export failed: %s", e)

    # Encode BOTH the aligned and the original image as base64.  The
    # aligned image is what the bboxes were computed against (axis-
    # aligned rects line up there), but the user finds the visibly-warped
    # aligned render jarring.  The original image plus the
    # ``bbox_original_quad`` we compute above lets the frontend overlay
    # on what the user actually uploaded.  We ship both so the frontend
    # can toggle between them without a round-trip.
    def _encode_jpeg(img: Optional[np.ndarray]) -> str:
        if img is None or getattr(img, "size", 0) == 0:
            return ""
        try:
            import base64 as _b64
            import cv2 as _cv2
            bgr = _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR) \
                if img.ndim == 3 and img.shape[2] == 3 \
                else img
            ok, enc = _cv2.imencode(
                ".jpg", bgr, [int(_cv2.IMWRITE_JPEG_QUALITY), 85],
            )
            if not ok:
                return ""
            return "data:image/jpeg;base64," + \
                _b64.b64encode(enc.tobytes()).decode("ascii")
        except Exception as e:
            logger.warning("Image encode failed: %s", e)
            return ""

    aligned_image_b64 = _encode_jpeg(state.get("aligned_image"))
    original_image_b64 = _encode_jpeg(state.get("image"))
    if not aligned_image_b64:
        aligned_image_b64 = original_image_b64

    response = {
        "success": True,
        "form_type": form_type.value if hasattr(form_type, "value") else str(form_type),
        "extraction_method": state.get("extraction_method", ""),
        "lane": state.get("lane", ""),
        "plan_reason": state.get("plan_reason", ""),
        "page_width": int(state.get("width", 0)),
        "page_height": int(state.get("height", 0)),
        "original_width": int(original_dims[0]),
        "original_height": int(original_dims[1]),
        "aligned_image_b64": aligned_image_b64,
        "original_image_b64": original_image_b64,
        "has_original_quads": bool(H_inv_np is not None),
        "extracted_fields": extracted,
        "field_details": field_details,
        "business_fields": business,
        "validation": state.get("validation", {"errors": [], "warnings": []}),
        "debug": {
            "alignment_used": bool(state.get("alignment_used", False)),
            "alignment_success": bool(state.get("alignment_success", False)),
            "alignment_quality": float(state.get("alignment_quality", 0.0)),
            "digital_text_used": state.get("lane", "") == "B",
            "vlm_rescue_count": int(state.get("vlm_rescue_count", 0)),
            "rescue_iterations": int(state.get("rescue_iterations", 0)),
            "rescue_history": state.get("rescue_history", {}),
            "rescue_log": state.get("rescue_log", []),
            "cleared_field_ids": sorted(cleared_set),
            "sections_used": bool(state.get("sections_used", False)),
            "sections": state.get("sections", []),
            "section_meta": state.get("section_meta", {}),
            "table_debug": state.get("table_debug", []),
            "trace": state.get("trace", []),
            "timings": state.get("timings", {}),
        },
        "reducto_format": reducto,
    }

    elapsed = time.time() - t0
    out = {
        "business_fields": business,
        "reducto_format": reducto,
        "final_response": response,
    }
    _update_trace_in_place(out, state, "finalize", elapsed)
    return out


# ---------------------------------------------------------------------- #
# Routing helpers
# ---------------------------------------------------------------------- #

def route_from_plan(state: GraphState) -> str:
    lane = state.get("lane", "C")
    if lane == "A":
        return "extract_widgets"
    if lane == "B":
        return "extract_digital"
    return "align"


def route_after_digital(state: GraphState) -> str:
    """If Lane B downgraded to C, re-enter align; else validate."""
    if state.get("lane") == "C" and state.get("extraction_method") \
            == "lane_b_downgraded_to_c":
        return "align"
    return "validate"


def route_after_reflect(state: GraphState) -> str:
    """Rescue if candidates exist and we haven't looped too many times.

    The agentic loop is bounded by the length of the rescue ladder —
    each iteration tries a different method per field.  We also stop
    early if the previous iteration produced no validation errors,
    short-circuiting unnecessary iterations.
    """
    candidates = state.get("rescue_candidates") or []
    iters = int(state.get("rescue_iterations", 0))
    from src.pipelines.graph.rescue_strategies import RESCUE_LADDER
    max_iters = len(RESCUE_LADDER)

    # Stop if validation is clean.
    validation = state.get("validation") or {}
    errors = validation.get("errors") or []
    if not errors and iters > 0:
        return "finalize"

    if candidates and iters < max_iters:
        return "rescue"
    return "finalize"


# ---------------------------------------------------------------------- #
# Utilities
# ---------------------------------------------------------------------- #

def _update_trace_in_place(out: Dict[str, Any], state: GraphState,
                           node: str, elapsed: float) -> None:
    trace = list(state.get("trace") or [])
    trace.append(node)
    timings = dict(state.get("timings") or {})
    timings[node] = round(float(elapsed), 3)
    out["trace"] = trace
    out["timings"] = timings

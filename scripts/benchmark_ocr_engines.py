"""Benchmark Florence-2 / Florence-2-aggressive / GOT-OCR 2.0 / TrOCR / VLM
head-to-head on CMS-1500 field crops.

Why this exists
---------------

The rescue ladder was recently rewired: TrOCR is out, Florence-2 raw
and a new Florence-2-aggressive variant are in, GOT-OCR 2.0 joins as
the third-opinion diverse-architecture OCR, and VLM sits on top for
context-aware rescue.  That reorder was motivated by published
handwriting benchmarks (IAM CER) and visual inspection on two PDFs,
but *our* data is messy mixed-print/cursive CMS-1500 crops, not IAM.

This script runs every engine on every field crop, records:

  * text, confidence, validator verdict, latency_ms
  * per-field-type agreement, per-engine win rate, best-engine majority

and writes a JSON report plus a human-readable markdown summary.  The
``ladder_for_field_type`` constants in ``rescue_strategies.py`` can
then be promoted/demoted based on hard numbers per field type instead
of a single-PDF gut feeling.

Typical usage (on DGX after deploy)
----------------------------------

    docker exec doc2data-server \
        python3 scripts/benchmark_ocr_engines.py \
            --pdf data/raw/cms1500_1.pdf data/raw/cms1500_2.pdf \
                  data/raw/cms1500_3.pdf data/raw/cms1500_4.pdf \
                  data/raw/cms1500_5.pdf data/raw/cms1500_6.pdf \
            --engines florence2_raw_upscale florence2_aggressive \
                      got_ocr trocr vlm \
            --out /tmp/ocr_bench

Run locally without the VLM (no Ollama) or without GOT-OCR (missing
weights) by omitting them from ``--engines``.

The script deliberately does NOT re-run the full graph per engine —
we align the PDF once, load the schema once, and run each engine on
the SAME crop so any difference is attributable to the OCR model,
not to alignment jitter.
"""
from __future__ import annotations

# Silence the TF import path before anything from transformers loads.
import os as _os
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Make ``src.*`` imports work whether this is run from the repo root or
# from /app inside the container.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------- #
# Engine wrappers
# ---------------------------------------------------------------------- #

@dataclass
class EngineResult:
    """One engine's output on one crop."""
    engine: str
    text: str
    confidence: float
    latency_ms: float
    valid: Optional[bool] = None
    error: str = ""


@dataclass
class FieldRun:
    """All engines' outputs for one field on one PDF."""
    pdf: str
    field_id: str
    field_type: str
    field_name: str
    reference_text: str        # what the live pipeline put there (post-rescue)
    reference_engine: str      # which engine the live pipeline chose
    bbox: Tuple[int, int, int, int]
    blank: bool
    per_engine: Dict[str, EngineResult] = field(default_factory=dict)


def _call_florence2_raw_upscale(pipeline, crop: np.ndarray) -> EngineResult:
    """Same call as the ``florence2_raw_upscale`` rescue strategy.

    We re-import ``_upscale`` from the rescue module so any future
    tweak to the preprocessing lands in both places.
    """
    from src.pipelines.graph.rescue_strategies import _upscale
    import cv2  # noqa: F401 — imported transitively but keep explicit
    t0 = time.time()
    try:
        up = _upscale(crop, factor=2.0)
        text, conf = pipeline.ocr_agent._florence2_ocr(up)
    except Exception as e:
        return EngineResult("florence2_raw_upscale", "", 0.0,
                            (time.time() - t0) * 1000, error=str(e))
    return EngineResult(
        "florence2_raw_upscale", (text or "").strip(), float(conf or 0.0),
        (time.time() - t0) * 1000,
    )


def _call_florence2_aggressive(pipeline, crop: np.ndarray) -> EngineResult:
    """Same preprocessing pipeline as the ``florence2_aggressive`` rescue
    strategy — 3× upscale + CLAHE-boosted grayscale."""
    from src.pipelines.graph.rescue_strategies import _upscale
    import cv2
    t0 = time.time()
    try:
        up = _upscale(crop, factor=3.0)
        if up.ndim == 3:
            gray = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY)
        else:
            gray = up
        try:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        text, conf = pipeline.ocr_agent._florence2_ocr(processed)
    except Exception as e:
        return EngineResult("florence2_aggressive", "", 0.0,
                            (time.time() - t0) * 1000, error=str(e))
    return EngineResult(
        "florence2_aggressive", (text or "").strip(), float(conf or 0.0),
        (time.time() - t0) * 1000,
    )


def _call_trocr(pipeline, crop: np.ndarray) -> EngineResult:
    """Keep TrOCR in the benchmark for the historical comparison —
    the rescue ladder doesn't use it anymore but we want the numbers."""
    from src.pipelines.graph.rescue_strategies import _upscale
    t0 = time.time()
    try:
        up = _upscale(crop, factor=1.5)
        text, conf = pipeline.ocr_agent._trocr_ocr(up, preprocess=True)
    except Exception as e:
        return EngineResult("trocr", "", 0.0,
                            (time.time() - t0) * 1000, error=str(e))
    return EngineResult(
        "trocr", (text or "").strip(), float(conf or 0.0),
        (time.time() - t0) * 1000,
    )


def _call_got_ocr(pipeline, crop: np.ndarray) -> EngineResult:
    """GOT-OCR 2.0.  Returns empty on missing model (no weights / old
    transformers) so the script still runs on stripped-down envs."""
    t0 = time.time()
    got = getattr(pipeline, "got_ocr_agent", None)
    if got is None or not got.is_available:
        return EngineResult("got_ocr", "", 0.0, 0.0,
                            error="agent_unavailable")
    try:
        text, conf = got.recognize(crop, max_new_tokens=96)
    except Exception as e:
        return EngineResult("got_ocr", "", 0.0,
                            (time.time() - t0) * 1000, error=str(e))
    return EngineResult(
        "got_ocr", (text or "").strip(), float(conf or 0.0),
        (time.time() - t0) * 1000,
    )


def _call_vlm(pipeline, crop: np.ndarray,
              field_name: str, field_type: str) -> EngineResult:
    """Direct Ollama VLM call.  Uses the rescue model, which is the
    same one the live ladder uses at iter 0."""
    t0 = time.time()
    try:
        from utils.config import Config
        text, conf = pipeline.ocr_agent._vlm_ocr_field(
            crop, field_name=field_name, field_type=field_type,
            model=Config.VLM_MODEL_RESCUE, timeout=90,
        )
    except Exception as e:
        return EngineResult("vlm", "", 0.0,
                            (time.time() - t0) * 1000, error=str(e))
    return EngineResult(
        "vlm", (text or "").strip(), float(conf or 0.0),
        (time.time() - t0) * 1000,
    )


def _call_parseq(pipeline, crop: np.ndarray) -> EngineResult:
    """PARSeq — scene-text transformer, 23M params.

    Uses the same 2× upscale the rescue strategy does so the benchmark
    reflects the real ladder behaviour.  Returns ``agent_unavailable``
    if pytorch_lightning isn't installed (we still want the script to
    run on envs without PARSeq — the other engines will report numbers).
    """
    from src.pipelines.graph.rescue_strategies import _upscale
    t0 = time.time()
    agent = getattr(pipeline, "parseq_agent", None)
    if agent is None or not agent.is_available:
        return EngineResult("parseq", "", 0.0, 0.0,
                            error="agent_unavailable")
    try:
        up = _upscale(crop, factor=2.0)
        text, conf = agent.recognize(up)
    except Exception as e:
        return EngineResult("parseq", "", 0.0,
                            (time.time() - t0) * 1000, error=str(e))
    return EngineResult(
        "parseq", (text or "").strip(), float(conf or 0.0),
        (time.time() - t0) * 1000,
    )


ENGINE_CALLERS: Dict[str, Callable[..., EngineResult]] = {
    "florence2_raw_upscale": lambda p, c, *_: _call_florence2_raw_upscale(p, c),
    "florence2_aggressive": lambda p, c, *_: _call_florence2_aggressive(p, c),
    "trocr": lambda p, c, *_: _call_trocr(p, c),
    "got_ocr": lambda p, c, *_: _call_got_ocr(p, c),
    "parseq": lambda p, c, *_: _call_parseq(p, c),
    "vlm": _call_vlm,
}


# ---------------------------------------------------------------------- #
# Graph → field list
# ---------------------------------------------------------------------- #

async def _run_graph_for_bench(pdf_path: Path):
    """Run the graph once to get an aligned image + block list.

    We DON'T run the engines inside the graph — that would couple the
    benchmark to the rescue ladder.  Instead we grab the aligned image
    plus the blocks (with resolved original_bbox) and let the
    per-engine calls downstream do their own crop + OCR.
    """
    from src.pipelines.core import PipelineConfig
    from src.pipelines.graph import build_graph
    from src.pipelines.graph.state import create_initial_state

    # Make everything cheap EXCEPT the alignment + block synthesis so
    # we get a clean block list to crop from.  We turn the rescue
    # ladder off so the benchmark is measuring each engine cold.
    cfg = PipelineConfig(
        enable_form_detection=True,
        enable_alignment=True,
        enable_trocr=False,       # skip built-in TrOCR
        enable_got_ocr=False,     # we'll call it ourselves below
        enable_parseq=True,       # agent still loads; _call_parseq invokes it
        enable_vlm_ocr_fallback=False,
        enable_vlm_tables=False,
        use_ocr_v2=True,
        method_override="cms1500_scan",
    )

    graph = build_graph()
    state = create_initial_state(str(pdf_path), cfg)
    final_state: Dict[str, Any] = {}
    async for delta in graph.astream(state, stream_mode="updates"):
        for _, patch in (delta or {}).items():
            if isinstance(patch, dict):
                final_state.update(patch)
    # ``build_graph`` ends at ``finalize``; the aligned image is on
    # the node delta, so merge anything we missed from the "end"
    # event when ``ainvoke`` returns.  For belt-and-braces, also call
    # ``ainvoke`` if the stream missed a state.
    if "blocks" not in final_state:
        final_state = await graph.ainvoke(state)
    return final_state


def _aligned_image(state: Dict[str, Any]) -> np.ndarray:
    img = state.get("aligned_image")
    if img is None:
        img = state.get("image")
    if img is None:
        raise RuntimeError("No aligned image in graph state")
    return img


def _field_type_of(block) -> str:
    meta = getattr(block, "metadata", None) or {}
    return str(meta.get("field_type", "") or "").lower()


def _field_name_of(block) -> str:
    meta = getattr(block, "metadata", None) or {}
    return str(meta.get("field_name", "") or block.id)


def _is_blank(block) -> bool:
    """Use the pipeline's own blank flag — don't re-derive it."""
    meta = getattr(block, "metadata", None) or {}
    status = str(meta.get("blank_status", "") or "").lower()
    if status in ("blank", "blank_structural", "blank_ink"):
        return True
    if not (getattr(block, "text", "") or "").strip():
        return True
    return False


def _validator_verdict(field_type: str, text: str) -> Optional[bool]:
    """Return True/False if we have a validator for this field, else None."""
    try:
        from src.pipelines.graph.rescue_strategies import (
            _run_validator, _validator_name_for,
        )
    except Exception:
        return None
    name = _validator_name_for(field_type)
    if not name or not text:
        return None
    try:
        return bool(_run_validator(name, text))
    except Exception:
        return None


# ---------------------------------------------------------------------- #
# Scoring
# ---------------------------------------------------------------------- #

def _norm(s: str) -> str:
    """Loose compare: strip non-alnum, lowercase."""
    import re
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _summarise(runs: List[FieldRun]) -> Dict[str, Any]:
    """Aggregate engine-level stats across all field runs."""
    engines: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "non_empty": 0,
        "valid": 0,
        "agrees_with_reference": 0,
        "winner_on_type": defaultdict(int),
        "latency_ms_p50": [],
    })

    per_field_type: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for run in runs:
        if run.blank:
            continue                          # skip truly blank crops
        ref = _norm(run.reference_text)
        # For "winner on this field": whichever non-empty engine matches
        # the reference text exactly (loose norm).  If none match, skip
        # scoring a winner for this field but still log per-engine.
        winners: List[str] = []
        for eng_name, res in run.per_engine.items():
            bucket = engines[eng_name]
            bucket["total"] += 1
            if res.text:
                bucket["non_empty"] += 1
            if res.valid is True:
                bucket["valid"] += 1
            if res.text and _norm(res.text) == ref and ref:
                bucket["agrees_with_reference"] += 1
                winners.append(eng_name)
            bucket["latency_ms_p50"].append(res.latency_ms)

        for w in winners:
            engines[w]["winner_on_type"][run.field_type] += 1
            per_field_type[run.field_type][w] += 1

    # Finalise latency p50
    for bucket in engines.values():
        lats = sorted(bucket["latency_ms_p50"])
        if lats:
            bucket["latency_ms_p50"] = round(lats[len(lats) // 2], 1)
        else:
            bucket["latency_ms_p50"] = None
        bucket["winner_on_type"] = dict(bucket["winner_on_type"])

    return {
        "engines": dict(engines),
        "winners_per_field_type": {k: dict(v) for k, v in per_field_type.items()},
    }


def _render_summary_md(summary: Dict[str, Any]) -> str:
    """Human-readable markdown of the aggregate summary."""
    lines: List[str] = []
    lines.append("# OCR engine benchmark — CMS-1500 crops\n")
    lines.append("## Per-engine totals\n")
    lines.append("| engine | crops | non_empty | valid | agree_ref | p50 ms |")
    lines.append("|---|---|---|---|---|---|")
    for eng, s in sorted(summary["engines"].items()):
        lines.append(
            f"| `{eng}` | {s['total']} | {s['non_empty']} | {s['valid']} | "
            f"{s['agrees_with_reference']} | {s['latency_ms_p50']} |"
        )

    lines.append("\n## Winners by field type\n")
    lines.append("| field_type | " + " | ".join(
        sorted(summary["engines"].keys())) + " |")
    lines.append("|---|" + "|".join(["---"] * len(summary["engines"])) + "|")
    for ft, wins in sorted(summary["winners_per_field_type"].items()):
        cells = [str(wins.get(e, 0)) for e in sorted(summary["engines"].keys())]
        lines.append(f"| {ft or '(untyped)'} | " + " | ".join(cells) + " |")

    lines.append("\n(`agree_ref` = exact-match to the live pipeline's "
                 "post-rescue value; use as a proxy for ground truth "
                 "when no labels are available.)\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Main loop
# ---------------------------------------------------------------------- #

async def _bench_one_pdf(
    pdf_path: Path,
    engines: List[str],
    pipeline,
    max_fields: Optional[int] = None,
    skip_blanks: bool = True,
) -> List[FieldRun]:
    """Run all selected engines on every non-blank field in a PDF.

    ``skip_blanks`` (default on) prevents the VLM engine from spending
    7-30 seconds per crop asking about a blank cell — the live
    pipeline never sends those to rescue anyway, so the benchmark
    shouldn't either.  Blank entries are still recorded in the output
    so we can see how each engine handles a visually blank crop, but
    we only call the engines on non-blank ones.
    """
    from src.pipelines.graph.rescue_strategies import _get_subtracted_crop

    print(f"→ aligning {pdf_path.name} …", flush=True)
    state = await _run_graph_for_bench(pdf_path)
    image = _aligned_image(state)
    blocks = state.get("blocks") or []
    # Skip checkboxes, tables, signatures — OCR engines don't apply.
    text_blocks = [
        b for b in blocks
        if (getattr(b, "block_type", None) and b.block_type.name not in
            ("CHECKBOX", "TABLE", "SIGNATURE"))
    ]
    if max_fields is not None:
        text_blocks = text_blocks[:max_fields]

    runs: List[FieldRun] = []
    for idx, b in enumerate(text_blocks, 1):
        field_type = _field_type_of(b)
        field_name = _field_name_of(b)
        bbox = tuple((b.metadata or {}).get("original_bbox") or b.bbox)
        # Benchmark on the SAME crop path the live rescue ladder uses:
        # template-subtracted for CMS-1500 (when available), raw otherwise.
        crop = _get_subtracted_crop(pipeline, b, image, pad_ratio=0.04)
        if crop is None:
            continue

        blank = _is_blank(b)
        run = FieldRun(
            pdf=pdf_path.name,
            field_id=b.id,
            field_type=field_type,
            field_name=field_name,
            reference_text=(b.text or "").strip(),
            reference_engine=str((b.metadata or {}).get("ocr_engine") or ""),
            bbox=tuple(int(x) for x in bbox),
            blank=blank,
        )

        if blank and skip_blanks:
            # Record the field but don't burn VLM seconds on it.
            print(
                f"    [{idx:>2}/{len(text_blocks)}] {b.id:<32s} "
                f"{'(blank)':<22s} skipped",
                flush=True,
            )
            runs.append(run)
            continue

        for eng in engines:
            caller = ENGINE_CALLERS.get(eng)
            if caller is None:
                continue
            res = caller(pipeline, crop, field_name, field_type)
            res.valid = _validator_verdict(field_type, res.text)
            run.per_engine[eng] = res
            print(
                f"    [{idx:>2}/{len(text_blocks)}] {b.id:<32s} "
                f"{eng:<22s} → {res.text[:32]!r:<34s} "
                f"conf={res.confidence:.2f} lat={res.latency_ms:>5.0f}ms "
                f"valid={res.valid}"
                + (f" err={res.error}" if res.error else ""),
                flush=True,
            )
        runs.append(run)
    return runs


async def _amain(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger().setLevel(logging.WARNING)

    from src.pipelines.graph.nodes import _get_pipeline
    from src.pipelines.core import PipelineConfig

    # Pipeline singleton — we'll flip the flags so each engine is
    # usable even though we're not going through the rescue ladder.
    cfg = PipelineConfig(
        enable_trocr=("trocr" in args.engines),
        enable_got_ocr=("got_ocr" in args.engines),
        enable_vlm_ocr_fallback=("vlm" in args.engines),
    )
    pipeline = await _get_pipeline(cfg)

    def _flush_summary(acc: List[FieldRun], suffix: str = "") -> None:
        """Write summary.{json,md} on disk from whatever we have so far.

        Re-writing after every PDF means a mid-benchmark kill still
        leaves us with a useful aggregate; the previous version only
        wrote at the very end, so any interruption lost all the
        per-engine aggregates.
        """
        if not acc:
            return
        summary = _summarise(acc)
        (out_dir / f"summary{suffix}.json").write_text(
            json.dumps(summary, indent=2)
        )
        md = _render_summary_md(summary)
        (out_dir / f"summary{suffix}.md").write_text(md)

    all_runs: List[FieldRun] = []
    for pdf_str in args.pdf:
        pdf = Path(pdf_str)
        if not pdf.exists():
            print(f"!! skipping {pdf} (not found)", flush=True)
            continue
        try:
            runs = await _bench_one_pdf(
                pdf, args.engines, pipeline, args.max_fields,
                skip_blanks=not args.include_blanks,
            )
        except Exception as e:
            print(f"!! {pdf.name} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
        # Persist raw-per-field data for later deep-dives.
        (out_dir / f"{pdf.stem}_runs.json").write_text(json.dumps(
            [
                {
                    **asdict(r),
                    "per_engine": {k: asdict(v) for k, v in r.per_engine.items()},
                }
                for r in runs
            ],
            indent=2,
            default=str,
        ))
        all_runs.extend(runs)
        _flush_summary(all_runs, suffix="_partial")
        print(
            f"  … rolling summary: {len(all_runs)} fields across "
            f"{sum(1 for p in args.pdf[:args.pdf.index(pdf_str)+1])} PDFs",
            flush=True,
        )

    summary = _summarise(all_runs)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    md = _render_summary_md(summary)
    (out_dir / "summary.md").write_text(md)
    print("\n" + md)
    print(f"\n→ report: {out_dir}/summary.md")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", nargs="+", required=True)
    ap.add_argument(
        "--engines",
        nargs="+",
        default=["florence2_raw_upscale", "florence2_aggressive",
                 "got_ocr", "parseq", "vlm"],
        help="Subset of engines to bench.  Default includes PARSeq "
             "(scene-text, 23M params) and GOT-OCR (kept for benchmark "
             "comparison even though it's no longer in the default "
             "rescue ladder).  TrOCR is excluded by default — include "
             "it explicitly if you want the historical number.",
    )
    ap.add_argument(
        "--max-fields", type=int, default=None,
        help="Cap fields per PDF for quick smoke runs.",
    )
    ap.add_argument(
        "--include-blanks", action="store_true",
        help="Call engines on visually-blank fields too.  Off by "
             "default — VLM calls on blanks waste minutes and the "
             "live pipeline never rescues them anyway.",
    )
    ap.add_argument("--out", default="/tmp/ocr_bench")
    return asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test harness for the new LangGraph orchestrator.

Runs the graph on a list of CMS-1500 PDFs and prints:
  - end-to-end latency
  - node trace + per-node timings
  - number of non-blank fields extracted
  - validation error count
  - F1 against gold labels (when available)

Usage:
    python scripts/test_graph_pipeline.py
    python scripts/test_graph_pipeline.py data/raw/cms1500_1.pdf
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _tokens(s: str) -> List[str]:
    return [t.lower() for t in str(s).split() if t.strip()]


def _f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(set(p) & set(g))
    if inter == 0:
        return 0.0
    prec = inter / len(p)
    rec = inter / len(g)
    return 2 * prec * rec / (prec + rec)


async def _run_one(file_path: Path) -> Dict[str, Any]:
    from src.pipelines.graph import run_graph
    from src.pipelines.core.models import PipelineConfig

    cfg = PipelineConfig(
        enable_alignment=True,
        enable_trocr=False,
        enable_vlm_ocr_fallback=False,   # focus on the batched OCR v2 path
        enable_vlm_tables=False,
        use_ocr_v2=True,
        enable_validators=True,
    )
    t0 = time.time()
    result = await run_graph(str(file_path), cfg)
    elapsed = time.time() - t0

    fields = result.get("extracted_fields", {}) or {}
    non_blank = {k: v for k, v in fields.items() if str(v or "").strip()}
    debug = result.get("debug", {}) or {}

    stem = file_path.stem
    gold_path = PROJECT_ROOT / "data" / "gold_labels" / f"{stem}.json"
    f1_avg = None
    if gold_path.exists():
        try:
            gold = json.loads(gold_path.read_text())
            business = result.get("business_fields", {}) or {}
            if business:
                f1s = []
                for k, v in gold.items():
                    pred = str(business.get(k, "") or "")
                    f1s.append(_f1(pred, str(v or "")))
                if f1s:
                    f1_avg = sum(f1s) / len(f1s)
        except Exception:
            pass

    summary = {
        "file": str(file_path),
        "elapsed_sec": round(elapsed, 2),
        "lane": result.get("lane"),
        "extraction_method": result.get("extraction_method"),
        "non_blank_fields": len(non_blank),
        "validation_errors": len(result.get("validation", {}).get("errors", [])),
        "vlm_rescue_count": debug.get("vlm_rescue_count", 0),
        "rescue_iterations": debug.get("rescue_iterations", 0),
        "alignment_quality": round(float(debug.get("alignment_quality", 0.0)), 3),
        "trace": debug.get("trace", []),
        "timings": debug.get("timings", {}),
        "business_f1_avg": round(f1_avg, 3) if f1_avg is not None else None,
    }
    return {"summary": summary, "result": result}


async def main(paths: List[Path]) -> None:
    print(f"\n{'=' * 72}")
    print(f"Doc2Data Graph Pipeline Test — {len(paths)} file(s)")
    print(f"{'=' * 72}\n")
    for p in paths:
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        try:
            out = await _run_one(p)
            s = out["summary"]
            print(f"[{p.name}]")
            print(f"  lane={s['lane']} method={s['extraction_method']}")
            print(f"  elapsed={s['elapsed_sec']}s "
                  f"non_blank={s['non_blank_fields']} "
                  f"val_errors={s['validation_errors']} "
                  f"vlm_rescue={s['vlm_rescue_count']}")
            if s['business_f1_avg'] is not None:
                print(f"  business_f1_avg={s['business_f1_avg']}")
            print(f"  trace={s['trace']}")
            print(f"  timings={s['timings']}")
            # Sample 5 extracted fields
            fields = out["result"].get("extracted_fields", {}) or {}
            sample = {k: v for k, v in list(fields.items())[:5]}
            print(f"  sample_fields={sample}")
            print()
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = [Path(a) for a in sys.argv[1:]]
    else:
        files = [
            PROJECT_ROOT / "data" / "raw" / f
            for f in (
                "cms1500.pdf",
                "cms1500_1.pdf",
                "cms1500_2.pdf",
                "cms1500_3.pdf",
                "cms1500_6.pdf",
            )
        ]
    asyncio.run(main(files))

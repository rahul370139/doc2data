"""Post-hoc analysis of the OCR benchmark — apply the same filters the
live rescue ladder would apply and see who actually wins per field.

The raw benchmark results give us per-engine text without cleanup.  The
live pipeline runs that text through:
  1. ``_looks_like_template_label`` → drop if pure template chrome
  2. ``_deterministic_clean`` for numeric types (date/phone/zip/npi/…)
  3. Field-specific post-processing (e.g. _clean_state)

So comparing raw engine outputs apples-to-apples requires us to
re-apply those same functions and then ask: how close does each engine
get to the reference?

This is a standalone script — run it with the repo on sys.path.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make repo imports work — the script lives at REPO/scripts/ so parent.parent is REPO.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Suppress TF/Paddle noise before imports
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.pipelines.graph.rescue_strategies import (  # noqa: E402
    _looks_like_template_label,
    _deterministic_clean,
    _run_validator,
    _validator_name_for,
    _clean_state,
    _strip_sentinels,
)


def _norm(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()


def _post_pipeline_clean(raw: str, field_type: str, form_type: str = "cms-1500") -> str:
    """Mimic what the rescue ladder's ``_strategy_*`` stages do.

    Order matches ``_strategy_florence2_raw_upscale`` /
    ``_strategy_got_ocr``:
      1. strip model sentinels
      2. reject if looks like template chrome  → empty
      3. deterministic clean (dates, phones, zips, npis…)
      4. state-code extraction if type==state
    Returns the final cleaned string or "" if it would be rejected.
    """
    if not raw:
        return ""
    text = _strip_sentinels(raw).strip()
    if not text:
        return ""
    if _looks_like_template_label(text, form_type):
        return ""
    text = _deterministic_clean(text, field_type)
    if (field_type or "").lower() == "state":
        text = _clean_state(text) or text
    return (text or "").strip()


def _load_runs(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def analyse(files: list[Path]) -> None:
    all_runs: list[dict] = []
    for f in files:
        all_runs.extend(_load_runs(f))

    print(f"Loaded {len(all_runs)} field-runs from {len(files)} PDFs")
    print(f"  non-blank: {sum(1 for r in all_runs if not r['blank'])}")
    print()

    # ----------------------------------------------------------------
    # Per-engine tally after applying the rescue-ladder cleanup
    # ----------------------------------------------------------------
    engines = ["florence2_raw_upscale", "florence2_aggressive", "got_ocr", "vlm"]
    stats: dict[str, Counter] = {e: Counter() for e in engines}
    by_type: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {e: Counter() for e in engines}
    )
    head_to_head_rows: list[dict] = []

    for run in all_runs:
        if run["blank"]:
            continue
        ft = (run.get("field_type") or "text").lower()
        ref = _norm(run.get("reference_text", ""))
        per_engine = run.get("per_engine", {})

        row = {
            "field_id": run["field_id"],
            "field_type": ft,
            "field_name": run["field_name"],
            "reference": run.get("reference_text", ""),
        }

        for eng in engines:
            entry = per_engine.get(eng, {}) or {}
            raw = entry.get("text", "") or ""
            cleaned = _post_pipeline_clean(raw, ft)
            row[f"{eng}_raw"] = raw
            row[f"{eng}_cleaned"] = cleaned

            s = stats[eng]
            s["total"] += 1
            if raw:
                s["raw_non_empty"] += 1
            if cleaned:
                s["clean_non_empty"] += 1

            # Validate the cleaned text
            vname = _validator_name_for(ft)
            if cleaned and vname:
                try:
                    if _run_validator(vname, cleaned):
                        s["clean_valid"] += 1
                        by_type[ft][eng]["valid"] += 1
                except Exception:
                    pass

            # Agreement with reference (normalised string match)
            if cleaned and _norm(cleaned) == ref and ref:
                s["clean_agrees_ref"] += 1
                by_type[ft][eng]["agrees_ref"] += 1
            elif cleaned and ref and _norm(cleaned) in ref or (ref and _norm(cleaned) and ref in _norm(cleaned)):
                s["clean_substring_ref"] += 1

        head_to_head_rows.append(row)

    # ----------------------------------------------------------------
    # Print per-engine totals
    # ----------------------------------------------------------------
    print("=" * 90)
    print("PER-ENGINE (after applying rescue-ladder cleanup)")
    print("=" * 90)
    print(f"{'engine':<24}  total  raw≠∅  clean≠∅  valid  agree-ref  substr-ref")
    for eng in engines:
        s = stats[eng]
        print(f"{eng:<24}  {s['total']:>5}  {s['raw_non_empty']:>5}  "
              f"{s['clean_non_empty']:>7}  {s['clean_valid']:>5}  "
              f"{s['clean_agrees_ref']:>9}  {s['clean_substring_ref']:>10}")
    print()

    # ----------------------------------------------------------------
    # Per-field-type winners
    # ----------------------------------------------------------------
    print("=" * 90)
    print("PER-FIELD-TYPE (agree-with-reference counts after cleanup)")
    print("=" * 90)
    print(f"{'field_type':<14} {' | '.join(f'{e:>22}' for e in engines)}")
    for ft in sorted(by_type.keys()):
        row = [f"{by_type[ft][e]['agrees_ref']:>22}" for e in engines]
        print(f"{ft:<14} {' | '.join(row)}")
    print()

    # ----------------------------------------------------------------
    # Head-to-head: fields where ONE engine wins and others fail
    # ----------------------------------------------------------------
    print("=" * 90)
    print("FIELDS where engines DIVERGE (showing cleaned outputs)")
    print("=" * 90)
    for row in head_to_head_rows:
        ref = row["reference"]
        f_raw = row["florence2_raw_upscale_cleaned"]
        f_agg = row["florence2_aggressive_cleaned"]
        got = row["got_ocr_cleaned"]
        vlm = row["vlm_cleaned"]
        # only show fields with at least some divergence among non-empty
        outs = [f_raw, f_agg, got, vlm]
        non_empty = [o for o in outs if o]
        if len(non_empty) < 2:
            continue
        if len({_norm(o) for o in non_empty}) == 1:
            continue  # all agree — skip
        print(f"\n— {row['field_id']:<32s} ({row['field_type']})")
        print(f"    REF      : {ref!r}")
        print(f"    florence2_raw : {f_raw!r}")
        print(f"    florence2_agg : {f_agg!r}")
        print(f"    got_ocr       : {got!r}")
        print(f"    vlm           : {vlm!r}")

    # ----------------------------------------------------------------
    # GOT-OCR unique wins: cases where GOT-OCR gets something useful
    # after cleaning and other engines returned empty
    # ----------------------------------------------------------------
    print()
    print("=" * 90)
    print("GOT-OCR 'unique output' fields (GOT-OCR clean non-empty, "
          "Florence-2 both empty)")
    print("=" * 90)
    for row in head_to_head_rows:
        got = row["got_ocr_cleaned"]
        f_raw = row["florence2_raw_upscale_cleaned"]
        f_agg = row["florence2_aggressive_cleaned"]
        if got and not f_raw and not f_agg:
            print(f"\n— {row['field_id']:<32s} ({row['field_type']})")
            print(f"    REF      : {row['reference']!r}")
            print(f"    got_ocr  : {got!r}")
            print(f"    vlm      : {row['vlm_cleaned']!r}")


if __name__ == "__main__":
    files = [
        Path(__file__).parent.parent / "artifacts/ocr_bench_pdf1_pdf2/cms1500_1_runs.json",
        Path(__file__).parent.parent / "artifacts/ocr_bench_pdf1_pdf2/cms1500_2_runs.json",
    ]
    analyse([Path(f) for f in files])

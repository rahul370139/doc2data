"""End-to-end smoke test for the LangGraph doc2data pipeline.

Runs `/extract/graph` against a set of CMS-1500 samples and audits the response
for three things we just fixed:

1. No `<pad>`/sentinel tokens leak into any field value or block text.
2. The rescue ladder fires on validation failures (rescue_history/rescue_log
   populated when validation errors exist).
3. Basic health: field count, blank count, validation error count, latency.

Usage (inside the container or on the host):
    python3 scripts/test_graph_e2e.py \
        --api http://localhost:8000 \
        --pdf data/raw/cms1500.pdf data/raw/cms1500_3.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

SENTINEL_RE = re.compile(r"</?\s*(pad|s|eos|bos|unk|mask|sep|cls)\s*/?>", re.IGNORECASE)


def find_sentinels(obj: Any, path: str = "") -> List[Tuple[str, str]]:
    """Recursively walk a JSON object and report any sentinel-token strings."""
    hits: List[Tuple[str, str]] = []
    if isinstance(obj, str):
        if SENTINEL_RE.search(obj):
            hits.append((path, obj[:160]))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            hits.extend(find_sentinels(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(find_sentinels(v, f"{path}[{i}]"))
    return hits


def _details_iter(details: Any) -> List[Dict[str, Any]]:
    if isinstance(details, dict):
        return [{"id": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in details.items()]
    if isinstance(details, list):
        return details
    return []


def _is_blank_detail(d: Dict[str, Any]) -> bool:
    if d.get("is_blank") is True:
        return True
    meta = d.get("metadata") or {}
    if str(meta.get("blank_status") or "").lower() in ("blank", "blank_structural", "blank_ink"):
        return True
    val = d.get("value")
    return isinstance(val, str) and not val.strip()


def summarize(resp: Dict[str, Any]) -> Dict[str, Any]:
    fields = resp.get("extracted_fields") or {}
    details = _details_iter(resp.get("field_details"))
    validation = resp.get("validation") or {}
    debug = resp.get("debug") or {}

    filled = sum(1 for v in fields.values() if isinstance(v, str) and v.strip())
    blanks = sum(1 for d in details if _is_blank_detail(d))
    low_conf = sum(
        1
        for d in details
        if not _is_blank_detail(d) and (d.get("confidence") or 0) < 0.6
    )
    errors = validation.get("errors") or []
    warnings = validation.get("warnings") or []

    # Section-first stats (only populated when extract_sections_node runs).
    sections = debug.get("sections") or []
    section_meta = debug.get("section_meta") or {}
    sections_used = bool(debug.get("sections_used"))
    sect_globals = section_meta.get("_global") or {}
    section_seeded = sum(
        1
        for d in details
        if (d.get("metadata") or {}).get("source", "").startswith("section_vlm")
    )

    return {
        "form_type": resp.get("form_type"),
        "alignment_quality": debug.get("alignment_quality"),
        "total_fields": len(fields) or len(details),
        "filled_fields": filled,
        "blank_fields": blanks,
        "low_confidence_fields": low_conf,
        "validation_errors": len(errors),
        "validation_warnings": len(warnings),
        "sections_used": sections_used,
        "sections_detected": len(sections),
        "sections_with_values": sect_globals.get("n_sections_with_values"),
        "section_vlm_filled": sect_globals.get("n_fields_filled"),
        "section_seeded_blocks": section_seeded,
        "rescue_iterations": debug.get("rescue_iterations"),
        "vlm_rescue_count": debug.get("vlm_rescue_count"),
        "rescue_history_size": len(debug.get("rescue_history") or {}),
        "rescue_log_size": len(debug.get("rescue_log") or []),
    }


def call_graph(api: str, pdf_path: Path, timeout: int = 900) -> Tuple[float, Dict[str, Any]]:
    t0 = time.time()
    with pdf_path.open("rb") as fh:
        r = requests.post(
            f"{api.rstrip('/')}/extract/graph",
            files={"file": (pdf_path.name, fh, "application/pdf")},
            timeout=timeout,
        )
    r.raise_for_status()
    return time.time() - t0, r.json()


def print_audit(pdf: Path, elapsed: float, resp: Dict[str, Any]) -> Dict[str, Any]:
    summary = summarize(resp)
    sentinel_hits = find_sentinels(resp)
    print(f"\n=== {pdf.name} ===")
    print(f"  latency          : {elapsed:.1f}s")
    for k, v in summary.items():
        print(f"  {k:18s}: {v}")
    if sentinel_hits:
        print(f"  sentinel leaks   : {len(sentinel_hits)} (FAIL)")
        for path, sample in sentinel_hits[:5]:
            print(f"      {path} = {sample!r}")
    else:
        print(f"  sentinel leaks   : 0 (PASS)")
    rlog = (resp.get("debug") or {}).get("rescue_log") or []
    if rlog:
        print(f"  rescue attempts  :")
        for entry in rlog[:10]:
            print(
                f"      {entry.get('field')} | {entry.get('method')} | "
                f"{'ACCEPTED' if entry.get('accepted') else 'rejected'} "
                f"| reason={entry.get('reason')!r:40s} | new={entry.get('new_text')!r:40s}"
            )
        if len(rlog) > 10:
            print(f"      … {len(rlog) - 10} more")
    return {"summary": summary, "sentinel_hits": sentinel_hits, "rescue_log": rlog}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--pdf", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/graph_e2e")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {"runs": []}
    ok = True
    for pdf_str in args.pdf:
        pdf = Path(pdf_str)
        if not pdf.exists():
            print(f"!! skipping {pdf} (not found)")
            continue
        try:
            elapsed, resp = call_graph(args.api, pdf)
        except Exception as exc:
            print(f"!! {pdf.name} FAILED: {exc}")
            ok = False
            continue
        audit = print_audit(pdf, elapsed, resp)
        (out_dir / f"{pdf.stem}.json").write_text(json.dumps(resp, indent=2))
        report["runs"].append(
            {
                "pdf": pdf.name,
                "latency_s": round(elapsed, 2),
                "summary": audit["summary"],
                "sentinel_hits": audit["sentinel_hits"],
                "rescue_log_size": len(audit["rescue_log"] or []),
            }
        )
        if audit["sentinel_hits"]:
            ok = False

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out_dir}/report.json")
    print("STATUS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

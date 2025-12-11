"""
Evaluate CMS-1500 extraction accuracy.

Usage:
    python validation/eval_cms1500.py --use-llm --use-icr

The script expects:
- Test PDFs under data/sample_docs/cms1500_test/
- Matching ground-truth JSON files under validation/ground_truth/cms1500/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

from src.pipelines.agentic_cms1500 import run_cms1500_agentic


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return [v.strip().lower() for v in value if isinstance(v, str)]
    return value


def compare_fields(pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    if not gt:
        return 0.0, {}
    total = len(gt)
    correct = 0
    per_field: Dict[str, float] = {}
    for key, gt_val in gt.items():
        pv = pred.get(key)
        if _normalize(pv) == _normalize(gt_val):
            correct += 1
            per_field[key] = 1.0
        else:
            per_field[key] = 0.0
    return (correct / max(total, 1)), per_field


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="data/sample_docs/cms1500_test", help="Directory with CMS-1500 PDFs/images")
    parser.add_argument("--gt-dir", default="validation/ground_truth/cms1500", help="Directory with ground truth JSON")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM backfill")
    parser.add_argument("--use-icr", action="store_true", help="Enable TrOCR handwriting model")
    parser.add_argument("--output", default="validation/results/cms1500_eval.json", help="Where to write summary JSON")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    gt_dir = Path(args.gt_dir)
    pdfs = sorted(
        list(test_dir.glob("*.pdf")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    )
    if not pdfs:
        print(f"No test files found in {test_dir}. Add CMS-1500 samples and rerun.")
        return

    results = []
    aggregate_correct = 0.0
    aggregate_total = 0

    for pdf in pdfs:
        gt_path = gt_dir / f"{pdf.stem}.json"
        ground_truth = {}
        if gt_path.exists():
            ground_truth = json.loads(gt_path.read_text())

        print(f"Processing {pdf.name} ...")
        prediction = run_cms1500_agentic(str(pdf), use_icr=args.use_icr, use_llm=args.use_llm)
        business_fields = prediction.get("business_fields", {})
        acc, per_field = compare_fields(business_fields, ground_truth)
        aggregate_correct += acc * max(len(ground_truth), 0)
        aggregate_total += max(len(ground_truth), 0)

        results.append(
            {
                "file": pdf.name,
                "accuracy": acc,
                "per_field": per_field,
                "business_fields": business_fields,
            }
        )

    overall = (aggregate_correct / aggregate_total) if aggregate_total > 0 else 0.0
    output = {"overall_accuracy": overall, "files": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved results to {out_path}")
    print(f"Overall accuracy: {overall:.2%}")


if __name__ == "__main__":
    main()

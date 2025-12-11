"""
Export a PDF/image to a Reducto-like JSON file using the agentic CMS-1500 pipeline.

Usage:
    python scripts/export_reducto.py --input data/sample_docs/cms1500.pdf --output out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.reducto_adapter import run_reducto_style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CMS-1500 PDF/image")
    parser.add_argument("--output", required=True, help="Where to write Reducto-like JSON")
    parser.add_argument("--no-icr", action="store_true", help="Disable TrOCR handwriting model")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM backfill")
    args = parser.parse_args()

    reducto_json = run_reducto_style(
        args.input,
        use_icr=not args.no_icr,
        use_llm=not args.no_llm,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reducto_json, indent=2))
    print(f"Saved Reducto-style JSON to {out_path}")


if __name__ == "__main__":
    main()

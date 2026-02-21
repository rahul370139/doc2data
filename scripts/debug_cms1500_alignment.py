"""
Debug utility for deterministic CMS-1500 registration.

Usage:
  python3 scripts/debug_cms1500_alignment.py --input /path/to/form.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import cv2
import numpy as np

from utils.config import Config
from src.pipelines.cms1500_register import get_cms1500_registrar


def _load_input(path: Path, dpi: int = 300) -> np.ndarray:
    if path.suffix.lower() == ".pdf":
        import fitz

        doc = fitz.open(str(path))
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif pix.n == 3:
            img = arr
        else:
            img = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        doc.close()
        return img

    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise RuntimeError(f"Failed to read input file: {path}")
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug CMS-1500 alignment quality.")
    parser.add_argument("--input", required=True, help="Input PDF/image")
    parser.add_argument("--output-dir", default=str(Config.PROJECT_ROOT / "cache" / "alignment_debug"), help="Output directory")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = _load_input(in_path)
    registrar = get_cms1500_registrar()
    result = registrar.register(image)

    run_id = uuid.uuid4().hex[:10]
    report = {
        "input": str(in_path),
        "success": bool(result.success),
        "quality": float(result.quality),
        "method": result.method,
        "debug": result.debug,
    }

    report_path = out_dir / f"alignment_report_{run_id}.json"
    report_path.write_text(json.dumps(report, indent=2))

    if result.aligned_image is not None:
        aligned_bgr = cv2.cvtColor(result.aligned_image, cv2.COLOR_RGB2BGR)
        aligned_path = out_dir / f"aligned_{run_id}.png"
        cv2.imwrite(str(aligned_path), aligned_bgr)
        report["aligned_image"] = str(aligned_path)

    print(json.dumps(report, indent=2))
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()

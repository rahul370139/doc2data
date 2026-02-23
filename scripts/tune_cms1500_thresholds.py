"""
Grid-search threshold tuner for CMS-1500 alignment.

PURPOSE: Runs alignment on multiple PDFs with different CMS1500_* threshold
combinations. Finds best params for handwritten/noisy scans. Prints
recommended env vars for DGX deployment.

USE CASE: python scripts/tune_cms1500_thresholds.py --input-dir data/sample_docs
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import itertools
import os

import cv2
import numpy as np

from src.pipelines.registration import CMS1500Registrar


@dataclass
class TrialResult:
    params: Dict[str, float]
    success_rate: float
    avg_quality: float
    score: float


def _load_input(path: Path, dpi: int = 300) -> np.ndarray:
    if path.suffix.lower() == ".pdf":
        import fitz

        doc = fitz.open(str(path))
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif pix.n == 3:
            rgb = arr
        else:
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        doc.close()
        return rgb

    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise RuntimeError(f"Failed to read file: {path}")
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def _collect_files(input_dir: Path, pattern: str) -> List[Path]:
    files = sorted(list(input_dir.glob(pattern)))
    return [p for p in files if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}]


def _run_trial(files: List[Path], params: Dict[str, float]) -> TrialResult:
    # Apply env overrides for this trial.
    for k, v in params.items():
        os.environ[k] = str(v)

    registrar = CMS1500Registrar()
    success = 0
    qualities: List[float] = []
    for f in files:
        try:
            image = _load_input(f)
            out = registrar.register(image)
            if out.success:
                success += 1
            qualities.append(float(out.quality))
        except Exception:
            qualities.append(0.0)

    success_rate = success / float(max(len(files), 1))
    avg_quality = float(np.mean(qualities)) if qualities else 0.0
    # Weighted objective: prioritize consistency first, quality second.
    score = (0.7 * success_rate) + (0.3 * avg_quality)
    return TrialResult(params=params, success_rate=success_rate, avg_quality=avg_quality, score=score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CMS-1500 alignment thresholds.")
    parser.add_argument("--input-dir", required=True, help="Folder with handwritten failure docs")
    parser.add_argument("--glob", default="*", help="Filename glob (default: *)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = _collect_files(input_dir, args.glob)
    if not files:
        raise RuntimeError(f"No input files found in {input_dir} with glob '{args.glob}'")

    print(f"Found {len(files)} files for tuning")

    # Focused tuning around handwritten-sensitive parameters.
    grid = {
        "CMS1500_RED_S_MIN": [0.30, 0.34, 0.38],
        "CMS1500_RED_V_MIN": [0.40, 0.45, 0.50],
        "CMS1500_RANSAC_REPROJ_HANDWRITTEN": [4.8, 5.5, 6.2],
        "CMS1500_QUAD_MIN_SCORE_HANDWRITTEN": [0.28, 0.34, 0.40],
    }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    all_trials = []
    for combo in itertools.product(*values):
        params = {k: v for k, v in zip(keys, combo)}
        trial = _run_trial(files, params)
        all_trials.append(trial)
        print(
            f"trial {params} => success={trial.success_rate:.3f}, "
            f"quality={trial.avg_quality:.3f}, score={trial.score:.3f}"
        )

    best = max(all_trials, key=lambda t: t.score)
    print("\nBest threshold set:")
    for k, v in best.params.items():
        print(f"  {k}={v}")
    print(f"  success_rate={best.success_rate:.3f}")
    print(f"  avg_quality={best.avg_quality:.3f}")
    print(f"  score={best.score:.3f}")

    print("\nExport these on DGX:")
    for k, v in best.params.items():
        print(f"export {k}={v}")


if __name__ == "__main__":
    main()

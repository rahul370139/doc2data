"""
Unit tests for the adaptive BlankDetector.

These tests intentionally avoid running any model — they synthesise crops
that mimic the situations the detector must handle:
  - Truly blank scan patch (just paper noise)
  - Field with a handwritten word in the middle
  - Field with template-residue specks at the edges
  - Field with a thin horizontal line stripe (alignment bleed)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from src.pipelines.ocr_v2 import BlankDetector, BlankStatus


def _blank_crop(h: int = 60, w: int = 320) -> np.ndarray:
    """Uniform light-gray paper with a tiny bit of noise."""
    rng = np.random.default_rng(42)
    base = rng.integers(245, 255, size=(h, w), dtype=np.uint8)
    return np.stack([base, base, base], axis=-1)


def _handwriting_crop(h: int = 60, w: int = 320) -> np.ndarray:
    """Blank crop with a 'word' drawn in the middle as dark pixels."""
    img = _blank_crop(h, w)
    # Draw a 4px-wide horizontal stroke shaped like text at 50% height
    cy = h // 2
    stroke = np.zeros((6, 120), dtype=np.uint8)
    # Fake word: 3 letters with gaps
    for seg_x in (5, 45, 85):
        stroke[1:5, seg_x:seg_x + 25] = 255
    # Paint dark (value 40) into image where stroke is non-zero
    y0 = cy - 3
    x0 = (w - 120) // 2
    for i in range(6):
        for j in range(120):
            if stroke[i, j]:
                img[y0 + i, x0 + j] = [40, 40, 40]
    return img


def _edge_noise_crop(h: int = 60, w: int = 320) -> np.ndarray:
    """Blank-looking center with scattered specks along the edges."""
    img = _blank_crop(h, w)
    rng = np.random.default_rng(7)
    # Put 40 small dark specks only in the outer ring
    for _ in range(40):
        side = rng.choice(["top", "bottom", "left", "right"])
        if side == "top":
            y = rng.integers(0, 5)
            x = rng.integers(0, w)
        elif side == "bottom":
            y = rng.integers(h - 5, h)
            x = rng.integers(0, w)
        elif side == "left":
            y = rng.integers(0, h)
            x = rng.integers(0, 5)
        else:
            y = rng.integers(0, h)
            x = rng.integers(w - 5, w)
        img[max(0, y - 1):y + 1, max(0, x - 1):x + 1] = 50
    return img


def _horizontal_line_crop(h: int = 60, w: int = 320) -> np.ndarray:
    """Blank crop with a thin dark horizontal line near the top edge."""
    img = _blank_crop(h, w)
    img[3:5, 5:w - 5] = 30
    return img


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #


def test_blank_crop_is_blank() -> None:
    det = BlankDetector()
    det._baseline = 0.004  # simulate a calibrated low baseline
    d = det.classify(_blank_crop())
    assert d.status is BlankStatus.BLANK, (
        f"expected BLANK for empty crop, got {d.status} "
        f"(inner={d.inner_ink:.4f} weighted={d.weighted_score:.4f})"
    )


def test_handwriting_crop_is_filled() -> None:
    det = BlankDetector()
    det._baseline = 0.004
    d = det.classify(_handwriting_crop())
    assert d.status is BlankStatus.FILLED, (
        f"expected FILLED for handwriting, got {d.status} "
        f"(inner={d.inner_ink:.4f} components={d.component_count})"
    )


def test_edge_noise_is_not_filled() -> None:
    """Scanner/alignment dust on the edges must NOT be classified FILLED."""
    det = BlankDetector()
    det._baseline = 0.004
    d = det.classify(_edge_noise_crop())
    assert d.status is not BlankStatus.FILLED, (
        f"edge noise should not be FILLED, got {d.status} "
        f"(inner={d.inner_ink:.4f} edge={d.edge_ink:.4f})"
    )
    # Edge ink must be higher than inner ink
    assert d.edge_ink >= d.inner_ink


def test_horizontal_line_is_not_filled() -> None:
    """A single line stripe (alignment bleed) must not be FILLED either."""
    det = BlankDetector()
    det._baseline = 0.004
    d = det.classify(_horizontal_line_crop())
    assert d.status is not BlankStatus.FILLED


def test_calibration_changes_threshold() -> None:
    det = BlankDetector()
    # Strongly artificial baseline — blank threshold should shift up.
    det._baseline = 0.05
    d = det.classify(_blank_crop())
    # Both thresholds must move with baseline.
    assert d.blank_threshold > 0.05
    assert d.filled_threshold > d.blank_threshold


def test_decision_metadata_keys() -> None:
    det = BlankDetector()
    d = det.classify(_handwriting_crop())
    md = d.to_metadata()
    for k in (
        "blank_status", "inner_ink", "edge_ink", "weighted_ink",
        "components", "baseline_noise", "blank_threshold", "filled_threshold",
    ):
        assert k in md

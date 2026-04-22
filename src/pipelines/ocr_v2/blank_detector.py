"""
Structural, adaptive blank detection for CMS-1500 fields.

Problem with the previous detector
----------------------------------
The original pipeline uses a single absolute threshold:

    if ink_ratio < 0.005:
        return blank
    if center_ink > 0.05:
        run raw fallback

These thresholds were tuned on ``cms1500_6`` and fail on ``cms1500_1/2/3``
because those scans have slightly different red-removal artefacts, paper
tint and subtraction residue.  The *absolute* ink ratio drifts enough to
push filled fields under the threshold (missed content) or lift blank
fields above it (false text from template residue).

What this detector does differently
-----------------------------------
1. **Per-form calibration.**  At the start of a page, we sample a number
   of known-blank-or-sparse regions (the template image itself, plus an
   aligned-scan margin sample) and compute a baseline noise level.  The
   blank threshold becomes ``baseline + margin`` instead of a hard 0.05.

2. **Center-weighted ink measurement.**  We trust the center of the
   bounding box far more than the edges.  Edges receive template bleed
   from neighbouring fields and alignment residue.  The reported score is

       score = 0.75 * inner_ink + 0.25 * edge_ink

   where ``inner_ink`` comes from the center 55% of the crop and
   ``edge_ink`` from the outer ring.

3. **Structural confirmation via connected components.**  Real handwriting
   contains >= 1 CC inside the center region with area above a dynamic
   threshold (scaled by crop height).  Template residue after subtraction
   is either scattered specks (many tiny CCs) or stripes at the edges
   (extreme aspect ratio).  Both are filtered out.

4. **Three-level decision:**

       BLANK    — skip OCR entirely, return empty
       UNCERTAIN — run OCR but require explicit text evidence
       FILLED   — run OCR, trust output within confidence bounds

The downstream OCR code branches on this tri-state instead of the binary
``ink_ratio`` gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np


class BlankStatus(Enum):
    BLANK = "blank"
    UNCERTAIN = "uncertain"
    FILLED = "filled"


@dataclass
class BlankDecision:
    status: BlankStatus
    inner_ink: float
    edge_ink: float
    weighted_score: float
    component_count: int
    baseline: float
    # Keeping the threshold that fired so tests + debug UIs can show "why".
    blank_threshold: float
    filled_threshold: float

    def is_blank(self) -> bool:
        return self.status is BlankStatus.BLANK

    def is_filled(self) -> bool:
        return self.status is BlankStatus.FILLED

    def to_metadata(self) -> dict:
        return {
            "blank_status": self.status.value,
            "inner_ink": round(self.inner_ink, 4),
            "edge_ink": round(self.edge_ink, 4),
            "weighted_ink": round(self.weighted_score, 4),
            "components": self.component_count,
            "baseline_noise": round(self.baseline, 4),
            "blank_threshold": round(self.blank_threshold, 4),
            "filled_threshold": round(self.filled_threshold, 4),
        }


class BlankDetector:
    """Stateful per-form blank detector.

    Life cycle:

        det = BlankDetector()
        det.calibrate(template_image_rgb, aligned_image_rgb)  # once per form
        for crop in crops:
            decision = det.classify(crop)
    """

    INNER_MARGIN = 0.225  # 22.5% → center ~55%
    EDGE_INNER_MARGIN = 0.08
    MIN_CROP_SIZE = 8
    DEFAULT_BASELINE = 0.006
    # Margins above baseline to decide blank vs filled.
    BLANK_MARGIN = 0.008
    FILLED_MARGIN = 0.025
    MIN_COMPONENT_AREA_FACTOR = 0.0008  # 0.08% of crop area
    MIN_COMPONENT_ABS_AREA = 18

    def __init__(self,
                 blank_margin: float = BLANK_MARGIN,
                 filled_margin: float = FILLED_MARGIN,
                 inner_margin: float = INNER_MARGIN):
        self.blank_margin = blank_margin
        self.filled_margin = filled_margin
        self.inner_margin = inner_margin
        self._baseline: float = self.DEFAULT_BASELINE
        self._calibrated: bool = False

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        template_rgb: Optional[np.ndarray] = None,
        aligned_scan_rgb: Optional[np.ndarray] = None,
        sample_rects: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> float:
        """Compute a per-form baseline ink ratio from empty regions.

        ``template_rgb`` and ``aligned_scan_rgb`` are optional.  When both
        are provided, we compute the absolute-diff between them on a set of
        known-empty rectangles (default: the right-margin strips).  When
        only one is provided, we estimate noise from low-std regions in
        that image.

        Returns the baseline (also stored on self).
        """
        samples: List[float] = []

        if sample_rects is None:
            # Default: stripes where CMS-1500 has no printed content.
            # Normalised (x0, y0, x1, y1) for 300 DPI template / aligned scan.
            sample_rects = [
                (0.02, 0.02, 0.08, 0.06),   # top-left empty
                (0.92, 0.02, 0.98, 0.06),   # top-right empty
                (0.02, 0.96, 0.08, 0.99),   # bottom-left empty
                (0.92, 0.96, 0.98, 0.99),   # bottom-right empty
            ]

        if template_rgb is not None and aligned_scan_rgb is not None:
            try:
                th, tw = template_rgb.shape[:2]
                sh, sw = aligned_scan_rgb.shape[:2]
                tmpl_gray = _to_gray(template_rgb)
                scan_gray = _to_gray(aligned_scan_rgb)
                # Resize template to scan size if needed
                if tmpl_gray.shape != scan_gray.shape:
                    tmpl_gray = cv2.resize(
                        tmpl_gray, (sw, sh), interpolation=cv2.INTER_AREA
                    )
                diff = cv2.absdiff(scan_gray, tmpl_gray)
                _, ink_mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
                h, w = ink_mask.shape[:2]
                for nx0, ny0, nx1, ny1 in sample_rects:
                    x0 = max(0, int(nx0 * w))
                    y0 = max(0, int(ny0 * h))
                    x1 = min(w, int(nx1 * w))
                    y1 = min(h, int(ny1 * h))
                    if x1 - x0 < 4 or y1 - y0 < 4:
                        continue
                    patch = ink_mask[y0:y1, x0:x1]
                    if patch.size == 0:
                        continue
                    samples.append(
                        float(np.count_nonzero(patch)) / max(1, patch.size)
                    )
            except Exception:
                pass

        if not samples and aligned_scan_rgb is not None:
            # Estimate baseline from very-low-std regions of the scan.
            gray = _to_gray(aligned_scan_rgb)
            h, w = gray.shape[:2]
            step = max(20, h // 40)
            patches = []
            for y in range(0, h - step, step):
                for x in range(0, w - step, step):
                    patch = gray[y:y + step, x:x + step]
                    if float(np.std(patch)) < 6.0:
                        patches.append(patch)
            for patch in patches[:50]:
                _, bin_ = cv2.threshold(
                    patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                samples.append(float(np.count_nonzero(bin_)) / max(1, bin_.size))

        if samples:
            # Use upper quartile to be robust to actual content bleeding
            # into one of our sample rects.
            arr = np.asarray(samples, dtype=np.float32)
            self._baseline = float(np.quantile(arr, 0.75))
        else:
            self._baseline = self.DEFAULT_BASELINE

        # Clamp to sane range
        self._baseline = min(max(self._baseline, 0.001), 0.025)
        self._calibrated = True
        return self._baseline

    @property
    def baseline(self) -> float:
        return self._baseline

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #

    def classify(self, crop: np.ndarray,
                 ink_ratio_hint: Optional[float] = None) -> BlankDecision:
        """Classify a template-subtracted crop as BLANK / UNCERTAIN / FILLED.

        ``ink_ratio_hint`` is the caller's own computed ink ratio (e.g. from
        ``_template_subtract``).  We don't use it directly for the decision,
        but we do store it in the returned decision's debug fields if the
        crop is too small to re-measure.
        """
        baseline = self._baseline
        blank_t = baseline + self.blank_margin
        filled_t = baseline + self.filled_margin

        if crop is None or crop.size == 0:
            return BlankDecision(
                status=BlankStatus.BLANK,
                inner_ink=0.0, edge_ink=0.0, weighted_score=0.0,
                component_count=0, baseline=baseline,
                blank_threshold=blank_t, filled_threshold=filled_t,
            )

        h, w = crop.shape[:2]
        if h < self.MIN_CROP_SIZE or w < self.MIN_CROP_SIZE:
            return BlankDecision(
                status=BlankStatus.BLANK,
                inner_ink=float(ink_ratio_hint or 0.0),
                edge_ink=0.0,
                weighted_score=float(ink_ratio_hint or 0.0),
                component_count=0, baseline=baseline,
                blank_threshold=blank_t, filled_threshold=filled_t,
            )

        gray = _to_gray(crop)

        # Short-circuit: extremely uniform region ⇒ blank, skip Otsu.
        if float(np.std(gray)) < 5.0:
            return BlankDecision(
                status=BlankStatus.BLANK,
                inner_ink=0.0, edge_ink=0.0, weighted_score=0.0,
                component_count=0, baseline=baseline,
                blank_threshold=blank_t, filled_threshold=filled_t,
            )

        # Otsu binarisation — global, so it correctly reports nearly-blank
        # crops as ~0% ink.
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Inner / edge regions
        ix, iy = max(1, int(w * self.inner_margin)), max(1, int(h * self.inner_margin))
        inner = binary[iy:h - iy, ix:w - ix]
        if inner.size == 0:
            inner = binary

        inner_ink = float(np.count_nonzero(inner)) / max(1, inner.size)

        # Edge ring: everything that is NOT the inner rect
        edge_mask = np.ones_like(binary)
        edge_mask[iy:h - iy, ix:w - ix] = 0
        edge_pixels = int(np.sum(edge_mask))
        edge_ink_count = int(np.sum(binary[edge_mask == 1] > 0))
        edge_ink = edge_ink_count / max(1, edge_pixels)

        weighted = 0.75 * inner_ink + 0.25 * edge_ink

        # Count components in inner region
        component_count = _count_significant_components(
            inner,
            min_area=max(
                self.MIN_COMPONENT_ABS_AREA,
                int(inner.size * self.MIN_COMPONENT_AREA_FACTOR),
            ),
        )

        # Decision logic
        if weighted < blank_t and component_count < 1:
            status = BlankStatus.BLANK
        elif weighted > filled_t and component_count >= 1:
            status = BlankStatus.FILLED
        elif inner_ink > blank_t and component_count >= 2:
            # Lots of structured ink in the middle even if weighted is low
            status = BlankStatus.FILLED
        else:
            status = BlankStatus.UNCERTAIN

        return BlankDecision(
            status=status,
            inner_ink=inner_ink,
            edge_ink=edge_ink,
            weighted_score=weighted,
            component_count=component_count,
            baseline=baseline,
            blank_threshold=blank_t,
            filled_threshold=filled_t,
        )


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _count_significant_components(binary: np.ndarray, min_area: int) -> int:
    """Count CCA components whose area >= min_area and aspect ratio < 12.

    We filter extreme aspect ratios to drop the horizontal/vertical line
    residue left after template subtraction on partially-aligned scans.
    """
    if binary is None or binary.size == 0:
        return 0

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    count = 0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cw = max(int(stats[i, cv2.CC_STAT_WIDTH]), 1)
        ch = max(int(stats[i, cv2.CC_STAT_HEIGHT]), 1)
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if aspect > 12:
            continue
        count += 1
    return count

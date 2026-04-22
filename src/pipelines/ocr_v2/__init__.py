"""
OCR v2 — low-latency, adaptive field extraction for CMS-1500.

Replaces the multi-pass Florence-2 + raw-fallback + upscale-retry pipeline
with a single batched Florence-2 pass plus a structural blank detector that
calibrates thresholds per-form.

Modules:
- florence_batch: batched Florence-2 inference (N crops per generate call).
- blank_detector: structural + adaptive blank detection (center-weighted).
- field_ocr: high-level per-field OCR that ties batching + blank detection.
"""

from src.pipelines.ocr_v2.florence_batch import BatchedFlorence2, get_batched_florence2
from src.pipelines.ocr_v2.blank_detector import BlankDetector, BlankDecision, BlankStatus
from src.pipelines.ocr_v2.field_ocr import FieldOCRBatch, FieldOCRResult, FieldOCRRequest

__all__ = [
    "BatchedFlorence2",
    "get_batched_florence2",
    "BlankDetector",
    "BlankDecision",
    "BlankStatus",
    "FieldOCRBatch",
    "FieldOCRRequest",
    "FieldOCRResult",
]

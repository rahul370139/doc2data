"""Section-level VLM extraction for structured forms.

This package calls a MID-SIZE VLM (MiniCPM-V 5.5GB / MiniCPM-o4.5 6.1GB) on
a SINGLE section of the page at a time, with ONLY that section's schema
fields in the prompt.

Why mid-size + per-section and not whole-page VLM:
  * A mid-size VLM reading 4 fields in a small crop is much more accurate
    than the same VLM reading 50 fields in a full page.
  * Small prompts = small output = less hallucination.
  * Memory fits on a single consumer GPU (MiniCPM-V is 5.5GB).
  * Traceability: every extracted value has a section bbox as provenance.
  * Reducto / Extend / Landing-AI all do per-region (not per-page) for
    the same reason.

Entry point: ``SectionVLMExtractor``.
"""

from .section_extractor import SectionVLMExtractor, SectionExtractionResult

__all__ = ["SectionVLMExtractor", "SectionExtractionResult"]

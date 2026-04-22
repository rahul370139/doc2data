"""Layout segmentation for structured forms.

The goal of this package is to turn a schema (``data/schemas/<form>.json``)
into a small number of SEMANTIC SECTIONS — ~8-14 rectangular regions that
each contain a handful of related fields (e.g. "patient row", "diagnosis",
"services table", "provider block").

Instead of running OCR on 50+ tiny per-field crops (where a few pixels of
template misalignment destroy the result), the section-first pipeline runs
ONE mid-size VLM call per section with just that section's schema fields
in the prompt.  The VLM has full context, the crop is big enough to be
robust to minor alignment drift, and the prompt is small enough (5-8
fields) for the VLM to answer reliably.

This is the pattern Reducto/Extend/Landing-AI use: layout-first → small
targeted model calls → structured JSON per region.

The section detector itself is PURELY spatial — no per-form heuristics,
no learned layout model, no hardcoded vocabulary.  It clusters fields by
Y-position into row-bands and, within wide row-bands, by X-position into
column blocks.  Works on any form that has a schema with per-field bboxes.
"""

from .sections import Section, detect_sections, sections_to_blocks

__all__ = ["Section", "detect_sections", "sections_to_blocks"]

"""Generic layout section detection from a schema.

Algorithm (form-agnostic, uses ONLY field bboxes):

1. Row-band clustering by Y-center
   Sort fields by Y-center; start a new row-band whenever the next field's
   Y-center jumps by more than ``y_eps`` (default 2.5% of page height) from
   the running mean of the current band.  This naturally produces rows like
   "Box 2/3/4 row", "Box 5/7 row", "Box 24 row", etc.

2. Column splitting within wide row-bands
   Some row-bands are actually two independent semantic blocks side-by-side
   (e.g. patient column + insured column).  If a band spans nearly the full
   page width AND its fields cluster into 2+ X groups with a clear gap
   (> ``x_gap``), split into separate sections.

3. Merge very small adjacent bands
   A row-band that contains only 1 checkbox or a single tiny strip is
   merged up/down into the neighbouring band so we don't burn a whole VLM
   call on one checkbox.  (Exception: fields that span > 40% of page
   height/width — e.g. Box 24 services table — stay as their own section.)

4. Enclosing bbox + padding
   Each section's bbox is the tight bounding box of its member fields,
   expanded by a small padding (default 0.8%) on every side so the crop is
   tolerant of minor alignment error.

Output: ``List[Section]`` with
  - id         : stable id (derived from member field-id prefixes)
  - label      : human-readable label
  - field_ids  : list of schema field ids that live in this section
  - bbox_norm  : (x0, y0, x1, y1) normalized to [0, 1]
  - field_types: set of schema field_types present

The output is consumed by
  - ``vlm.section_extractor.SectionVLMExtractor`` — for the actual VLM call
  - ``graph.nodes.extract_sections_node`` — to crop and dispatch
  - the UI overlay — to visualize section boundaries for debugging
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


BBox = Tuple[float, float, float, float]  # x0, y0, x1, y1 normalized


# ── Section dataclass ─────────────────────────────────────────────────────

@dataclass
class Section:
    """A semantic region of the page containing a group of related fields."""
    id: str
    label: str
    field_ids: List[str]
    bbox_norm: BBox
    field_types: List[str] = _dc_field(default_factory=list)
    # Small bookkeeping for trace/debug output
    metadata: Dict[str, Any] = _dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "field_ids": list(self.field_ids),
            "bbox_norm": list(self.bbox_norm),
            "field_types": list(self.field_types),
            "metadata": dict(self.metadata),
        }


# ── Helpers ───────────────────────────────────────────────────────────────

# A generic "box number" prefix extractor.  Pulls "1", "1a", "24", "header",
# etc. from field ids like "1a_insured_id", "24_service_lines",
# "header_top_right_notes".  If no numeric leader, returns the part before
# the first underscore.
_PREFIX_RE = re.compile(r"^(\d+[a-zA-Z]?|[A-Za-z]+?)[_.\-]")


def _field_prefix(field_id: str) -> str:
    m = _PREFIX_RE.match(field_id)
    if m:
        return m.group(1)
    # fallback: first token
    return (field_id.split("_", 1)[0] or field_id).lower()


def _pick_bbox(field: Dict[str, Any]) -> Optional[BBox]:
    """Prefer ``bbox_norm_new`` (manually refined) over ``bbox_norm``."""
    b = field.get("bbox_norm_new") or field.get("bbox_norm")
    if not b or len(b) != 4:
        return None
    try:
        x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x0 <= x1 <= 1.0 and 0.0 <= y0 <= y1 <= 1.0):
        # Skip malformed bboxes (keep signed zero etc. but clamp)
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(1.0, max(x0, x1)), min(1.0, max(y0, y1))
    return (x0, y0, x1, y1)


def _ymid(bbox: BBox) -> float:
    return 0.5 * (bbox[1] + bbox[3])


def _xmid(bbox: BBox) -> float:
    return 0.5 * (bbox[0] + bbox[2])


def _enclose(bboxes: List[BBox]) -> BBox:
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _pad(bbox: BBox, pad_x: float, pad_y: float) -> BBox:
    x0, y0, x1, y1 = bbox
    return (
        max(0.0, x0 - pad_x),
        max(0.0, y0 - pad_y),
        min(1.0, x1 + pad_x),
        min(1.0, y1 + pad_y),
    )


# ── Core clustering ───────────────────────────────────────────────────────

def _cluster_rows(
    items: List[Tuple[str, BBox, Dict[str, Any]]],
    y_eps: float,
) -> List[List[Tuple[str, BBox, Dict[str, Any]]]]:
    """Y-band clustering.

    Sort by y-center, then grow a band while the next field's y-center is
    within ``y_eps`` of the running mean.  This is robust to slight row
    misalignment (±1% of page height) and doesn't need a k.
    """
    if not items:
        return []
    sorted_items = sorted(items, key=lambda t: _ymid(t[1]))
    bands: List[List[Tuple[str, BBox, Dict[str, Any]]]] = [[sorted_items[0]]]
    # Running mean of the current band's y-center (small CMS-1500 rows have
    # ~8 fields, so the mean moves slowly — this is stable).
    running_mean = _ymid(sorted_items[0][1])
    n_in_band = 1
    for it in sorted_items[1:]:
        y = _ymid(it[1])
        if abs(y - running_mean) <= y_eps:
            bands[-1].append(it)
            n_in_band += 1
            running_mean = ((running_mean * (n_in_band - 1)) + y) / n_in_band
        else:
            bands.append([it])
            running_mean = y
            n_in_band = 1
    return bands


def _split_by_column(
    band: List[Tuple[str, BBox, Dict[str, Any]]],
    x_gap_min: float,
    min_left_fraction: float,
) -> List[List[Tuple[str, BBox, Dict[str, Any]]]]:
    """Split a row-band into left/right columns if there's a clear X-gap.

    A band gets split when:
      * It contains at least 3 fields (single/double don't need splitting)
      * The band's total X-span is >= 0.65 of the page (i.e. genuinely wide)
      * There is an X-gap of at least ``x_gap_min`` somewhere in the middle
        40% of the page

    The split is done at the widest gap in the middle of the band.
    """
    if len(band) < 3:
        return [band]
    bboxes = [it[1] for it in band]
    enc = _enclose(bboxes)
    band_width = enc[2] - enc[0]
    if band_width < 0.65:
        return [band]
    # Compute the midpoints of fields and find the biggest gap
    mids = sorted((_xmid(b), it) for it, b in zip(band, bboxes))
    best_gap = 0.0
    split_x: Optional[float] = None
    for i in range(1, len(mids)):
        gap = mids[i][0] - mids[i - 1][0]
        mid_point = 0.5 * (mids[i][0] + mids[i - 1][0])
        # Require the gap to be somewhere in the middle third of the page —
        # avoids cutting off a single-field header on one side.
        if gap >= x_gap_min and 0.30 <= mid_point <= 0.70 and gap > best_gap:
            best_gap = gap
            split_x = mid_point
    if split_x is None:
        return [band]
    left: List[Tuple[str, BBox, Dict[str, Any]]] = []
    right: List[Tuple[str, BBox, Dict[str, Any]]] = []
    for it in band:
        if _xmid(it[1]) <= split_x:
            left.append(it)
        else:
            right.append(it)
    if len(left) == 0 or len(right) == 0:
        return [band]
    # Only accept the split if both sides have enough content to justify a
    # VLM call.  Otherwise keep the band whole.
    if len(left) < max(1, int(min_left_fraction * len(band))):
        return [band]
    return [left, right]


# ── Section id / label derivation (form-agnostic) ─────────────────────────

def _section_id_from_prefixes(field_ids: List[str]) -> str:
    prefixes = [_field_prefix(fid) for fid in field_ids]
    # Keep the order of first appearance, drop duplicates
    seen = []
    for p in prefixes:
        if p not in seen:
            seen.append(p)
    if not seen:
        return "section"
    if len(seen) == 1:
        return f"section_{seen[0]}"
    if len(seen) <= 3:
        return "section_" + "_".join(seen)
    return f"section_{seen[0]}__{seen[-1]}"


def _section_label(field_ids: List[str]) -> str:
    prefixes = [_field_prefix(fid) for fid in field_ids]
    counter = Counter(prefixes)
    top = [p for p, _ in counter.most_common(4)]
    if not top:
        return "Section"
    # Human-ish formatting: numbers become "Box N", words become title case
    parts = []
    for p in top:
        if p.isdigit() or (p[:-1].isdigit() and p[-1].isalpha()):
            parts.append(f"Box {p}")
        else:
            parts.append(p.replace("_", " ").title())
    return " + ".join(parts)


# ── Public API ────────────────────────────────────────────────────────────

def detect_sections(
    schema: Dict[str, Any],
    *,
    y_eps: float = 0.025,
    x_gap_min: float = 0.07,
    min_left_fraction: float = 0.20,
    pad_x: float = 0.010,
    pad_y: float = 0.012,
    exclude_field_types: Tuple[str, ...] = ("signature",),
    include_checkboxes: bool = True,
) -> List[Section]:
    """Derive a list of sections from a schema.

    Args:
        schema: Parsed schema dict (must have a ``"fields"`` list).
        y_eps: Row-band Y-tolerance (fraction of page height).  Default 2.5%
            matches CMS-1500 row spacing; increase for looser forms.
        x_gap_min: Minimum gap (fraction of page width) required to split a
            row-band into two columns.
        min_left_fraction: When splitting, require the left column to have
            at least this fraction of the row's fields.  Prevents splitting
            off single-field stubs.
        pad_x, pad_y: Padding added around each section's bbox (fraction of
            page).  Keep small — too much pad → neighbour bleed.
        exclude_field_types: Field types to NOT include in any section's
            prompt.  Signatures are excluded by default because no VLM can
            reliably transcribe handwriting that isn't constrained to a
            known character set.  They're still processed by the normal
            Florence-2 path.
        include_checkboxes: If True, checkboxes are grouped with their
            parent section (so the VLM sees the whole "insurance type" row
            as one block).  If False, checkboxes are skipped entirely.

    Returns:
        List of ``Section`` in top-to-bottom reading order.
    """
    fields = schema.get("fields") or []
    if not fields:
        return []

    items: List[Tuple[str, BBox, Dict[str, Any]]] = []
    for f in fields:
        fid = f.get("id")
        if not fid:
            continue
        ftype = (f.get("field_type") or "text").lower()
        if ftype in exclude_field_types:
            continue
        if not include_checkboxes and ftype == "checkbox":
            continue
        bbox = _pick_bbox(f)
        if bbox is None:
            continue
        items.append((fid, bbox, f))

    if not items:
        return []

    # ── 1) Row bands ──────────────────────────────────────────────────────
    bands = _cluster_rows(items, y_eps=y_eps)

    # ── 2) Optional column split within very wide bands ───────────────────
    sections_items: List[List[Tuple[str, BBox, Dict[str, Any]]]] = []
    for band in bands:
        sections_items.extend(
            _split_by_column(band, x_gap_min=x_gap_min, min_left_fraction=min_left_fraction)
        )

    # ── 3) Merge trivial singleton bands into neighbours ──────────────────
    #
    # If a band contains only 1 field AND that field is small (<2% page
    # area), merge into the previous band so we don't burn a VLM call on a
    # checkbox by itself.  Exception: if the neighbour is also trivial,
    # keep them separate.
    merged: List[List[Tuple[str, BBox, Dict[str, Any]]]] = []
    for idx, cluster in enumerate(sections_items):
        bboxes = [it[1] for it in cluster]
        enc = _enclose(bboxes)
        area = (enc[2] - enc[0]) * (enc[3] - enc[1])
        is_tiny = len(cluster) == 1 and area < 0.015
        if is_tiny and merged:
            merged[-1].extend(cluster)
        else:
            merged.append(cluster)

    # ── 4) Build Section objects ──────────────────────────────────────────
    # Sort clusters top-to-bottom, left-to-right BEFORE assigning IDs so the
    # ordinal prefix reflects reading order. Makes debug output stable across
    # runs and avoids confusing "section_5" appearing after "section_7".
    def _cluster_reading_key(cluster):
        bxs = [it[1] for it in cluster]
        env = _enclose(bxs)
        return (_ymid(env), _xmid(env))

    merged_sorted = sorted(merged, key=_cluster_reading_key)

    out: List[Section] = []
    used_raw_ids: Dict[str, int] = {}
    for ordinal, cluster in enumerate(merged_sorted, start=1):
        field_ids = [it[0] for it in cluster]
        bboxes = [it[1] for it in cluster]
        ftypes = sorted({
            (it[2].get("field_type") or "text").lower()
            for it in cluster
        })
        enc = _enclose(bboxes)
        padded = _pad(enc, pad_x=pad_x, pad_y=pad_y)

        # ID format: sec_{ordinal:02}_{prefix_summary}
        # Ordinal guarantees uniqueness and reading order; prefix summary
        # keeps the ID human-debuggable (sec_05_5_6 = the 5th section, fields
        # from Box 5 / 6).
        raw_prefix = _section_id_from_prefixes(field_ids)
        # Strip the leading "section_" so we don't emit sec_01_section_1_2_3
        if raw_prefix.startswith("section_"):
            raw_prefix = raw_prefix[len("section_"):]
        sec_id = f"sec_{ordinal:02d}_{raw_prefix}"
        used_raw_ids[raw_prefix] = used_raw_ids.get(raw_prefix, 0) + 1

        sec = Section(
            id=sec_id,
            label=_section_label(field_ids),
            field_ids=field_ids,
            bbox_norm=padded,
            field_types=list(ftypes),
            metadata={
                "raw_bbox_norm": list(enc),
                "padding": [pad_x, pad_y],
                "n_fields": len(field_ids),
                "reading_ordinal": ordinal,
                "prefix_summary": raw_prefix,
            },
        )
        out.append(sec)

    logger.info(
        "detect_sections: %d fields → %d sections (avg %.1f fields/section)",
        sum(len(s.field_ids) for s in out),
        len(out),
        sum(len(s.field_ids) for s in out) / max(1, len(out)),
    )
    return out


def sections_to_blocks(sections: List[Section], width: int, height: int) -> List[Dict[str, Any]]:
    """Convert sections to pixel-space dicts for UI overlay / tracing.

    Each returned dict has ``id``, ``label``, ``bbox`` (pixel-space), and
    ``field_ids``.
    """
    out = []
    for s in sections:
        x0, y0, x1, y1 = s.bbox_norm
        out.append({
            "id": s.id,
            "label": s.label,
            "bbox": [
                int(x0 * width), int(y0 * height),
                int(x1 * width), int(y1 * height),
            ],
            "bbox_norm": list(s.bbox_norm),
            "field_ids": list(s.field_ids),
            "field_types": list(s.field_types),
        })
    return out

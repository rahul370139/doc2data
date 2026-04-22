"""
Shared enums and data classes for the multi-agent document processing pipeline.

PURPOSE: Single source of truth for FormType, BlockType, DetectedBlock,
PipelineConfig, etc. Used by all agents and the orchestrator.

USE CASE: Import these types when building blocks, configuring the pipeline,
or extending the pipeline with new form types or block types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.config import Config


class FormType(Enum):
    CMS1500 = "cms-1500"
    UB04 = "ub-04"
    NCPDP = "ncpdp"
    GENERIC = "generic"
    UNKNOWN = "unknown"


class BlockType(Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CHECKBOX = "checkbox"
    SIGNATURE = "signature"
    HEADER = "header"
    FOOTER = "footer"
    TITLE = "title"
    PAGE_NUM = "page_num"
    LIST = "list"
    FORM_FIELD = "form_field"


@dataclass
class DetectedBlock:
    """A detected region in the document."""
    id: str
    block_type: BlockType
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) absolute pixels
    confidence: float
    page_id: int = 0
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormIdentification:
    """Result of form identification."""
    form_type: FormType
    confidence: float
    version: Optional[str] = None
    fingerprint_matched: bool = False
    detection_method: str = ""


@dataclass
class AlignmentResult:
    """Result of template alignment."""
    success: bool
    aligned_image: Optional[Any]
    homography_matrix: Optional[Any]
    alignment_quality: float
    fallback_used: bool = False
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for the multi-agent pipeline."""
    # Form detection
    enable_form_detection: bool = True
    form_type_override: Optional[FormType] = None

    # Layout detection
    layout_model: str = "auto"
    yolo_confidence: float = 0.25
    detectron_threshold: float = 0.30

    # OCR (v1.3: multi-engine consensus — PaddleOCR + TrOCR + Florence-2 + VLM rescue)
    enable_trocr: bool = True
    ocr_engine_mode: str = "consensus"  # "consensus" | "trocr_only"
    trocr_model: str = "large"
    trocr_confidence_threshold: float = 0.85  # TrOCR fallback threshold
    # GOT-OCR 2.0 (stepfun-ai/GOT-OCR-2.0-hf) — third-opinion rescue engine.
    # Adds a ~580M-param unified OCR model whose error modes are decorrelated
    # from Florence-2 and TrOCR (different architecture, different training
    # data).  Useful on numeric/short-text fields that VLM + TrOCR disagree
    # on; the ladder will consider its candidate alongside the others.
    # Disabled by default so environments without transformers>=4.49 (or
    # without the HF weights cached) don't fail.  Flip to True after the
    # weights are pre-downloaded on DGX.
    enable_got_ocr: bool = True
    got_ocr_model_id: str = "stepfun-ai/GOT-OCR-2.0-hf"
    # PARSeq (baudm/parseq) — scene-text transformer used by the
    # NUMERIC / STATE rescue ladders for printed digits/caps.  ~23M
    # params, ~30-50ms/crop on GPU, loads via torch.hub on first use.
    # Disabled by default in production ladders; keep available for
    # experiments via benchmark/feature flags.
    enable_parseq: bool = False
    parseq_variant: str = "parseq"  # "parseq" | "parseq_tiny" | "parseq_patch16_224"
    # Rescue strategies run Florence-2 / VLM / PARSeq on template-
    # subtracted crops by default (see _get_subtracted_crop in
    # graph/rescue_strategies.py).  Set to False only if you want to
    # debug alignment — raw crops let template labels bleed through.
    rescue_use_template_subtract: bool = True
    ocr_confidence_threshold: float = 0.5
    handwriting_threshold: float = 0.35
    enable_vlm_ocr_fallback: bool = True  # Enable VLM for all text fields
    vlm_ocr_model: str = field(default_factory=lambda: Config.OLLAMA_MODEL_VLM_OCR)
    vlm_parallel_calls: int = 8  # Max concurrent VLM calls for parallel processing
    # PADDING v2 (2026): schema padding is applied ONCE in _load_schema_zones.
    # Default is small, fixed absolute padding to absorb ±2px alignment jitter.
    # zone_padding_ratio is disabled (0.0) by default — ratio-based padding
    # caused up to 8% bleed into adjacent fields on tall rows.
    zone_padding_px: int = 3
    zone_padding_ratio: float = 0.0
    # Optional extra padding in the OCR stage — OFF by default (no double pad).
    extra_ocr_padding_px: int = 0
    # OCR v2: batched Florence-2 + adaptive blank detection
    use_ocr_v2: bool = True
    ocr_v2_batch_size: int = 8
    ocr_v2_blank_margin: float = 0.008
    ocr_v2_filled_margin: float = 0.025

    # Alignment
    alignment_x_offset: float = -0.008
    alignment_y_offset: float = 0.0
    use_advanced_ink_extraction: bool = True
    enforce_unique_word_assignment: bool = True
    enable_alignment: bool = True
    alignment_fallback_to_ml: bool = True
    alignment_quality_threshold: float = 0.35

    use_cms1500_production_pipeline: bool = False

    # SLM/VLM
    enable_slm_labeling: bool = False
    enable_vlm_figures: bool = False
    enable_vlm_tables: bool = True
    slm_model: str = field(default_factory=lambda: Config.OLLAMA_MODEL_SLM)
    vlm_model: str = field(default_factory=lambda: Config.OLLAMA_MODEL_VLM)
    enable_slm_field_cleaning: bool = False

    # Section-first extraction (Tier 1 in Lane C):
    #   • Split the schema into ~8–22 semantic sections (layout-first CV).
    #   • One VLM call per section using a MID-size model (MiniCPM-o4.5 by
    #     default).  Each call sees only that section's 2–12 fields, so
    #     the prompt is small and the VLM is accurate.
    #   • Florence-2 per-field OCR still runs AFTER sections but only on
    #     fields the section VLM left blank.  This is the residual path.
    #   • Rescue ladder (validator → vlm → florence2_raw_upscale → trocr)
    #     runs AFTER validation as before — nothing removed.
    # Disabled by default.  Section-first Tier-1 produced a single-VLM-call
    # per section which, on Ollama, serialises on the GPU and balloons wall
    # time to 10+ minutes per doc.  Florence-2 per-field (Tier-2) is much
    # faster on this hardware and already covers 59–76% fill.  Toggle back
    # on only when you want the extra VLM read (e.g. for field-level
    # residual fill-in) and have tuned concurrency to match the GPU.
    #   Environment override: ENABLE_SECTIONS=true|false
    enable_sections: bool = False
    section_vlm_model: str = field(
        default_factory=lambda: Config.VLM_MODEL_SECTION
    )
    section_vlm_timeout_s: int = 200
    # 4 concurrent Ollama calls: matches the OLLAMA_NUM_PARALLEL env
    # setting we ship in the Docker launch.  On the GB10 (128GB unified,
    # ~80GB free after Florence-2 + PaddleOCR) this is the sweet spot —
    # 5+ concurrent minicpm-v calls push per-call latency back up as
    # Ollama starts queueing on the GPU memory bus.  Section Box-24
    # (tables) is kept on its own pass because one crop uses most of
    # the VLM context window and is meaningfully slower.
    section_vlm_max_parallel: int = 4
    # MiniCPM-V / LLaVA auto-slice images wider than ~1024px into 2-3 tiles,
    # which triples per-call latency.  Capping the long side at 1000 keeps
    # each crop in a single tile and roughly halves wall-clock time with no
    # measurable accuracy loss on cms1500_6.pdf.
    section_max_crop_side_px: int = 1000
    section_min_confidence: float = 0.55       # drop below → Florence-2 fallback

    min_ocr_confidence: float = 0.20

    # Validation
    enable_validators: bool = True
    enable_llm_qa: bool = False

    # Method override — "auto" means let plan_node choose based on form
    # detection + widget presence.  Other values force a specific path
    # even if plan_node would have routed elsewhere, which is useful
    # for:
    #   • A/B testing different extraction strategies on the same PDF
    #     without re-uploading (frontend dropdown wires to this).
    #   • Falling back to a known-good pipeline when automatic routing
    #     misfires (e.g. a hybrid fillable PDF that should have been
    #     scan-OCR'd but got widget-routed).
    # Accepted values:
    #   "auto"          → current heuristic routing.
    #   "cms1500_scan"  → force Lane C (template-align + Florence-2),
    #                     the default for scanned CMS-1500 / UB-04.
    #   "sections"      → force Lane C with enable_sections=True
    #                     (section-VLM Tier 1 + Florence-2 residual).
    #   "digital"       → force Lane B (digital text layer → zone
    #                     matching).  Downgrades to C if no text layer.
    #   "widgets"       → force Lane A (AcroForm widgets).  Downgrades
    #                     to C if the PDF has none.
    # Unknown values are treated as "auto" and logged.
    method_override: str = "auto"

    # Output
    include_ocr_schema: bool = True
    include_business_schema: bool = True

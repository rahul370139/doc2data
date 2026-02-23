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

    # OCR
    enable_trocr: bool = True
    trocr_model: str = "large"
    ocr_confidence_threshold: float = 0.5
    handwriting_threshold: float = 0.35
    enable_vlm_ocr_fallback: bool = True
    vlm_ocr_model: str = field(default_factory=lambda: Config.OLLAMA_MODEL_VLM)
    zone_padding_px: int = 6
    zone_padding_ratio: float = 0.08

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

    min_ocr_confidence: float = 0.20

    # Validation
    enable_validators: bool = True
    enable_llm_qa: bool = False

    # Output
    include_ocr_schema: bool = True
    include_business_schema: bool = True

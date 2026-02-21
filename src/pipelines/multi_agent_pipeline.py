"""
Multi-Agent Document Processing Pipeline

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOCUMENT INGESTION                                 │
│                     (PDF/Image → 300 DPI RGB Array)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FORM IDENTIFICATION AGENT                               │
│  - OCR header for "CMS-1500", "UB-04", etc.                                 │
│  - Layout fingerprint matching (Sensible-style)                             │
│  - Returns: form_type, confidence, version                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
        ┌───────────────────┐             ┌───────────────────────┐
        │   CMS-1500 PATH   │             │   GENERAL FORM PATH   │
        │  (Template-based) │             │   (ML Detection)      │
        └───────────────────┘             └───────────────────────┘
                    │                                 │
                    ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TEMPLATE ALIGNMENT AGENT                               │
│  (CMS-1500 only)                                                             │
│  - ORB/SIFT feature matching                                                 │
│  - Homography warp to reference template                                     │
│  - Fallback: ML detection if alignment fails                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LAYOUT DETECTION AGENT                                 │
│  CMS-1500:              │  General Forms:                                   │
│  - YOLOv8 (fine-tuned)  │  - LayoutLMv3 / Detectron2 (PubLayNet)            │
│  - Template zones       │  - Donut (end-to-end)                             │
│  Returns: blocks with type (text, table, figure, form fields, checkbox)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
        ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
        │  TEXT BLOCKS  │  │ TABLE BLOCKS  │  │ FIGURE BLOCKS │ 
        └───────────────┘  └───────────────┘  └───────────────┘
                    │                │                │
                    ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OCR AGENT (Tiered)                                   │
│  - PaddleOCR (printed text)                                                  │
│  - TrOCR (handwriting, signatures)                                           │
│  - Checkbox density detector                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                │                │
                    ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SLM/VLM LABELING AGENT                                 │
│  TEXT:    SLM (Llama 3.2) → semantic field labels                           │
│  TABLES:  TATR + SLM → structured row/column extraction                     │
│  FIGURES: VLM (MiniCPM-V) → chart/image understanding                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VALIDATION AGENT                                        │
│  - Field validators (NPI, date, phone, ICD-10, etc.)                         │
│  - Cross-field consistency checks                                            │
│  - LLM QA sanity check                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ASSEMBLY AGENT                                          │
│  - OCR Schema (raw extraction)                                               │
│  - Business Schema (mapped fields)                                           │
│  - Reducto-style JSON export                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    from src.pipelines.multi_agent_pipeline import MultiAgentPipeline
    
    pipeline = MultiAgentPipeline()
    result = await pipeline.process("document.pdf")
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import concurrent.futures

import cv2
import numpy as np

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.config import Config


# ============================================================================
# Enums & Data Classes
# ============================================================================

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
    aligned_image: Optional[np.ndarray]
    homography_matrix: Optional[np.ndarray]
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
    layout_model: str = "auto"  # auto, yolo, layoutlm, detectron2, donut
    yolo_confidence: float = 0.25
    # PubLayNet models often need a lower threshold to avoid returning only 1 giant region.
    detectron_threshold: float = 0.30
    
    # OCR
    # TrOCR enabled as fallback for handwritten fields (PaddleOCR first, TrOCR if conf < 0.30)
    enable_trocr: bool = True
    ocr_confidence_threshold: float = 0.5
    handwriting_threshold: float = 0.35
    # Schema-zone matching / OCR tuning
    zone_padding_px: int = 15  # Increased for handwriting overshoot
    zone_padding_ratio: float = 0.20  # 20% expansion — enough for handwriting overshoot
    
    # Alignment offset correction (fraction of image width/height)
    # Positive values shift boxes RIGHT/DOWN, negative shift LEFT/UP
    # Use negative x_offset if boxes appear shifted right (text cut off on left)
    alignment_x_offset: float = -0.008  # Shift boxes ~0.8% left
    alignment_y_offset: float = 0.0  # No vertical adjustment needed
    
    # Advanced ink extraction for better OCR on scanned forms
    use_advanced_ink_extraction: bool = True
    # If True, each OCR word box is assigned to at most one schema zone to avoid duplicates/overlaps
    enforce_unique_word_assignment: bool = True
    
    # Template alignment - ENABLED for CMS-1500 scans (safe alignment implementation below)
    enable_alignment: bool = True
    alignment_fallback_to_ml: bool = True
    # Lower threshold - schema zones are still better than one giant block
    alignment_quality_threshold: float = 0.35
    
    # Use CMS-1500 Production Pipeline (includes ink extraction, Track 2C)
    # DISABLED: per-field crop OCR is unreliable; use full-page OCR + zone matching instead
    use_cms1500_production_pipeline: bool = False
    
    # SLM/VLM
    # DISABLED by default: SLM labeling causes hallucinations (garbage text in results)
    # Only enable if you have a well-tuned local model AND want semantic classification
    enable_slm_labeling: bool = False
    enable_vlm_figures: bool = False  # VLM for general figures - disabled by default
    enable_vlm_tables: bool = True  # VLM for CMS-1500 tables (Box 24) - enabled for accuracy
    slm_model: str = "llama3.2:3b"
    vlm_model: str = "llama3.2:3b"  # or minicpm-v
    
    # SLM-based field cleaning for handwritten forms
    # When enabled, uses SLM to intelligently extract values from noisy OCR
    enable_slm_field_cleaning: bool = False  # Disabled by default - adds latency
    
    # OCR confidence filtering
    # Minimum confidence to accept OCR text (lowered to avoid over-filtering)
    min_ocr_confidence: float = 0.20
    
    # Validation
    enable_validators: bool = True
    enable_llm_qa: bool = False  # Disabled - LLM QA is slow and hallucinates
    
    # Output
    include_ocr_schema: bool = True
    include_business_schema: bool = True


# ============================================================================
# Base Agent Class
# ============================================================================

class BaseAgent(ABC):
    """Base class for all pipeline agents."""
    
    def __init__(self, name: str):
        self.name = name
        self._initialized = False
    
    @abstractmethod
    async def initialize(self):
        """Initialize the agent (lazy loading)."""
        pass
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """Process input and return result."""
        pass
    
    def log(self, message: str):
        """Log agent activity."""
        print(f"[{self.name}] {message}")


# ============================================================================
# Form Identification Agent
# ============================================================================

class FormIdentificationAgent(BaseAgent):
    """
    Identifies the type of form using multiple methods:
    1. OCR header text matching
    2. Layout fingerprint (Sensible-style)
    3. Visual feature matching
    """
    
    # Form fingerprints (text patterns to identify forms)
    # Each pattern has a weight (higher = more discriminative)
    FINGERPRINTS = {
        FormType.CMS1500: [
            "cms-1500", "cms 1500", "health insurance claim form",
            "approved by national uniform claim committee",
            "hcfa-1500", "please print or type", "form 1500"
        ],
        FormType.UB04: [
            "ub-04", "ub04", "uniform bill",
            "patient control no", "med rec no",
            "type of bill", "statement covers period",
            "occurrence span", "value codes",
            "revenue description", "serv date"
        ],
        FormType.NCPDP: [
            "ncpdp", "universal claim form",
            "pharmacy claim"
        ],
        FormType.GENERIC: [
            "tactical combat casualty care", "tccc", "casualty",
            "mechanism of injury", "evac category"
        ]
    }
    
    # High-confidence discriminative tokens (if found, strongly indicates that form)
    # UB-04 patterns are more extensive to prevent CMS-1500 misclassification
    STRONG_TOKENS = {
        FormType.UB04: ["ub-04", "ub04", "uniform bill", "ub 04", "type of bill", 
                        "fl 42", "revenue code", "occurrence span"],
        FormType.CMS1500: ["cms-1500", "cms 1500", "cms1500", "hcfa-1500", "form 1500 02-12"],
        FormType.NCPDP: ["ncpdp"],
    }
    
    # Version fingerprints for CMS-1500
    CMS1500_VERSIONS = {
        "02/12": "2012 revision",
        "08/05": "2005 revision",
        "12/90": "1990 revision"
    }
    
    def __init__(self):
        super().__init__("FormIdentificationAgent")
        self._ocr = None
    
    async def initialize(self):
        if self._initialized:
            return
        try:
            from paddleocr import PaddleOCR
            try:
                self._ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            except Exception:
                # Fallback for newer/older versions
                self._ocr = PaddleOCR(lang='en')
        except Exception as e:
            self.log(f"OCR init failed: {e}")
        self._initialized = True
    
    def _ocr_region(self, image: np.ndarray, region: Tuple[float, float, float, float]) -> str:
        """OCR a specific region of the image."""
        if self._ocr is None:
            return ""
        
        h, w = image.shape[:2]
        x0, y0, x1, y1 = region
        crop = image[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
        
        if crop.size == 0:
            return ""
            
        try:
            try:
                result = self._ocr.ocr(crop, cls=True)
            except TypeError:
                result = self._ocr.ocr(crop)
                
            # Handle PaddleX OCRResult
            first_item = result[0] if isinstance(result, list) and len(result) > 0 else result
            if "OCRResult" in str(type(first_item)) or "OCRResult" in str(type(result)):
                try:
                    # Try to get text directly
                    json_data = {}
                    if hasattr(first_item, 'json'):
                        j = first_item.json
                        json_data = j() if callable(j) else j
                    elif hasattr(result, 'json'):
                        j = result.json
                        json_data = j() if callable(j) else j
                        
                    if isinstance(json_data, str):
                        json_data = json.loads(json_data)
                    
                    # PaddleX v3 wraps data under "res" key
                    if isinstance(json_data, dict):
                        res = json_data.get('res', json_data)
                        texts = res.get('rec_texts', res.get('rec_text', []))
                        if texts:
                            text = " ".join(str(t) for t in texts).lower()
                            return text
                except Exception as e:
                    self.log(f"OCRResult parse error: {e}")

            # Handle list of results
            if isinstance(result, list) and len(result) > 0:
                lines = result[0] if isinstance(result[0], list) else result
                texts = []
                for line in lines:
                    if isinstance(line, list) and len(line) >= 2:
                        # line[1] is (text, conf)
                        text_conf = line[1]
                        if isinstance(text_conf, (list, tuple)):
                            texts.append(text_conf[0])
                        else:
                            texts.append(str(text_conf))
                text = " ".join(texts).lower()
                return text
        except Exception as e:
            self.log(f"OCR region failed: {e}")
        return ""
    
    def _match_fingerprint(self, text: str) -> Tuple[FormType, float]:
        """Match text against form fingerprints with strong-token priority."""
        text_lower = text.lower()
        
        # First check strong discriminative tokens (override generic matches)
        for form_type, strong_tokens in self.STRONG_TOKENS.items():
            for token in strong_tokens:
                if token in text_lower:
                    # Strong token match: high confidence
                    return form_type, 0.70
        
        best_match = FormType.UNKNOWN
        best_score = 0.0
        
        for form_type, patterns in self.FINGERPRINTS.items():
            matches = sum(1 for p in patterns if p in text_lower)
            # Boost score if multiple unique patterns match
            if matches > 0:
                score = matches / len(patterns) + 0.15  # Reduced base boost
                if score > best_score:
                    best_score = score
                    best_match = form_type
        
        # Raised threshold for acceptance (0.45 instead of 0.15)
        # Prevents "CLAIM FORM" alone from matching CMS-1500
        if best_score < 0.45:
            return FormType.GENERIC, best_score
            
        return best_match, best_score
    
    def _detect_version(self, text: str) -> Optional[str]:
        """Detect form version from text."""
        for version_code, version_name in self.CMS1500_VERSIONS.items():
            if version_code in text:
                return version_name
        return None
    
    async def process(self, image: np.ndarray) -> FormIdentification:
        """Identify the form type."""
        await self.initialize()
        
        # OCR header region (top 15% of page)
        header_text = self._ocr_region(image, (0.0, 0.0, 1.0, 0.15))
        
        # Also check footer for form identifiers
        footer_text = self._ocr_region(image, (0.0, 0.85, 1.0, 1.0))
        
        combined_text = header_text + " " + footer_text
        print(f"[FormID] Detected text: {combined_text[:200]}...")
        
        # Match fingerprint
        form_type, confidence = self._match_fingerprint(combined_text)
        print(f"[FormID] Matched: {form_type} (conf: {confidence:.2f})")
        
        # Detect version if CMS-1500
        version = None
        if form_type == FormType.CMS1500:
            version = self._detect_version(combined_text)
        
        # If no match, default to generic
        if confidence < 0.2:
            form_type = FormType.GENERIC
            confidence = 0.5
        
        return FormIdentification(
            form_type=form_type,
            confidence=confidence,
            version=version,
            fingerprint_matched=confidence > 0.3,
            detection_method="ocr_fingerprint"
        )


# ============================================================================
# Template Alignment Agent
# ============================================================================

class TemplateAlignmentAgent(BaseAgent):
    """
    Aligns scanned forms to a reference template using feature matching.
    Implements Sensible-style alignment with fallback to ML detection.
    """
    
    def __init__(self):
        super().__init__("TemplateAlignmentAgent")
        self._templates: Dict[FormType, np.ndarray] = {}
    
    async def initialize(self):
        if self._initialized:
            return
        # Load reference templates
        template_dir = Path(__file__).parent.parent.parent / "data" / "templates"
        if template_dir.exists():
            for template_file in template_dir.glob("*.png"):
                form_name = template_file.stem.replace("_template", "")
                try:
                    form_type = FormType(form_name)
                    self._templates[form_type] = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                except:
                    pass
        # Fallback: load reference images from sample_docs if template PNGs missing
        if FormType.CMS1500 not in self._templates:
            try:
                from src.processing.registration import load_and_process_reference
                ref_data = load_and_process_reference("cms-1500")
                if ref_data and ref_data.get("image") is not None:
                    self._templates[FormType.CMS1500] = ref_data["image"]
            except Exception:
                pass
        if FormType.UB04 not in self._templates:
            try:
                from src.processing.registration import load_and_process_reference
                ref_data = load_and_process_reference("ub-04")
                if ref_data and ref_data.get("image") is not None:
                    self._templates[FormType.UB04] = ref_data["image"]
            except Exception:
                pass
        self._initialized = True

    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 points as: top-left, top-right, bottom-right, bottom-left.
        pts: shape (4,2)
        """
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _detect_form_quad(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect the outer form rectangle as a quadrilateral.
        Returns (quad_points, score) where score is 0..1.
        This is a stable alignment primitive that will not "tilt" a straight scan
        if the quad is detected correctly.
        """
        try:
            g = gray
            if g.dtype != np.uint8:
                g = g.astype(np.uint8)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            # Strong edges
            edges = cv2.Canny(g, 50, 150)
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, 0.0

            h, w = g.shape[:2]
            img_area = float(h * w)

            best = None
            best_score = 0.0
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < img_area * 0.15:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) != 4:
                    continue

                quad = approx.reshape(4, 2).astype(np.float32)
                quad = self._order_quad_points(quad)
                # Score: large area + rectangular-ish shape
                area_ratio = area / img_area

                # Angle score: corners should be near 90 deg
                def _angle(a, b, c) -> float:
                    ba = a - b
                    bc = c - b
                    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
                    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
                    return float(np.degrees(np.arccos(cosang)))

                angs = [
                    _angle(quad[3], quad[0], quad[1]),
                    _angle(quad[0], quad[1], quad[2]),
                    _angle(quad[1], quad[2], quad[3]),
                    _angle(quad[2], quad[3], quad[0]),
                ]
                ang_err = float(np.mean([abs(a - 90.0) for a in angs]))
                angle_score = float(max(0.0, 1.0 - (ang_err / 25.0)))  # 25deg avg error -> 0

                score = 0.7 * min(1.0, area_ratio / 0.60) + 0.3 * angle_score
                if score > best_score:
                    best_score = score
                    best = quad

            return best, float(best_score)
        except Exception:
            return None, 0.0
    
    def _try_yolo_alignment(self, image: np.ndarray, template_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], float]:
        """
        Try to align using the fine-tuned YOLO model for form boundary detection.
        The YOLO model can detect the CMS-1500 form region more reliably than contours
        on noisy/handwritten scans.
        Returns (homography_matrix, quality_score) or (None, 0.0).
        """
        try:
            from pathlib import Path
            from src.pipelines.yolo_layout import YOLOLayoutDetector

            # Find the best YOLO model
            model_paths = []
            if Config.YOLO_MODEL_PATH:
                mp = Path(Config.YOLO_MODEL_PATH)
                model_paths.append(mp)
                if not mp.is_absolute():
                    model_paths.append(Config.PROJECT_ROOT / mp)
            model_paths.extend([
                Config.PROJECT_ROOT / "runs" / "detect" / "cms1500_yolo_run13" / "weights" / "best.pt",
                Path("/app/runs/detect/cms1500_yolo_run13/weights/best.pt"),
                Config.PROJECT_ROOT / "models" / "cms1500_yolo_v1.pt",
                Path("/app/models/cms1500_yolo_v1.pt"),
                Config.PROJECT_ROOT / "models" / "yolo" / "cms1500_best.pt",
            ])

            yolo = None
            for mp in model_paths:
                if mp and mp.exists():
                    yolo = YOLOLayoutDetector(str(mp), conf=0.15, iou=0.5)
                    self.log(f"YOLO alignment: loaded {mp.name}")
                    break

            if yolo is None:
                return None, 0.0

            # Run YOLO detection to find form blocks
            blocks = yolo.predict(image, page_id=0)
            if not blocks:
                return None, 0.0

            # Find the largest block (the form boundary)
            h, w = image.shape[:2]
            img_area = float(h * w)
            best_block = None
            best_area = 0.0
            for b in blocks:
                bx0, by0, bx1, by1 = b.bbox
                area = float((bx1 - bx0) * (by1 - by0))
                if area > best_area and area > img_area * 0.15:
                    best_area = area
                    best_block = b

            if best_block is None:
                return None, 0.0

            # Use YOLO bbox as ROI, then try to recover a rotated quad inside the ROI.
            bx0, by0, bx1, by1 = best_block.bbox
            bx0, by0, bx1, by1 = int(bx0), int(by0), int(bx1), int(by1)
            pad_x = int(max(2, (bx1 - bx0) * 0.02))
            pad_y = int(max(2, (by1 - by0) * 0.02))
            rx0 = max(0, bx0 - pad_x)
            ry0 = max(0, by0 - pad_y)
            rx1 = min(w, bx1 + pad_x)
            ry1 = min(h, by1 + pad_y)
            roi = image[ry0:ry1, rx0:rx1]
            if roi.size == 0:
                return None, 0.0
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi

            src_quad = None
            quad_score = None
            quad, q_score = self._detect_form_quad(roi_gray)
            if quad is not None:
                quad = quad.astype(np.float32)
                quad[:, 0] += rx0
                quad[:, 1] += ry0
                src_quad = quad
                quad_score = float(q_score)
                self.log(f"YOLO alignment: refined quad in ROI (score={q_score:.2f})")
            else:
                # Fallback: min-area rectangle inside ROI
                try:
                    edges = cv2.Canny(roi_gray, 50, 150)
                    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        area = float(cv2.contourArea(cnt))
                        roi_area = float((rx1 - rx0) * (ry1 - ry0))
                        if area > roi_area * 0.15:
                            rect = cv2.minAreaRect(cnt)
                            box = cv2.boxPoints(rect)
                            quad = self._order_quad_points(box)
                            quad[:, 0] += rx0
                            quad[:, 1] += ry0
                            src_quad = quad.astype(np.float32)
                            quad_score = 0.5
                            self.log("YOLO alignment: used minAreaRect fallback in ROI")
                except Exception:
                    pass

            if src_quad is None:
                return None, 0.0

            th, tw = template_shape
            dst_quad = np.array([
                [0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]
            ], dtype=np.float32)

            H = cv2.getPerspectiveTransform(src_quad, dst_quad)
            if H is not None:
                # Validate the homography
                if self._validate_homography(H, (h, w)):
                    quality = float(best_block.confidence)
                    if quad_score is not None:
                        quality = 0.6 * quality + 0.4 * float(quad_score)
                    self.log(f"YOLO alignment: form bbox ({bx0:.0f},{by0:.0f})-({bx1:.0f},{by1:.0f}), conf={best_block.confidence:.2f}")
                    return H, max(0.6, quality)

        except Exception as e:
            self.log(f"YOLO alignment failed: {e}")

        return None, 0.0

    def _compute_homography(self, image: np.ndarray, template: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Compute alignment from input image -> template space.

        Priority order:
        1. YOLO-based form detection (most robust for noisy scans)
        2. Contour-based quad detection (good for clean scans)
        3. SIFT/ORB feature matching (fallback)
        NO ECC refinement (prevents tilt on handwritten forms).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        th, tw = template.shape[:2]

        # 0) YOLO-based alignment (best for handwritten/noisy scans)
        H_yolo, q_yolo = self._try_yolo_alignment(image, (th, tw))
        if H_yolo is not None:
            self.log(f"Using YOLO alignment (quality={q_yolo:.2f})")
            return H_yolo, q_yolo

        # 1) Contour-based quad detection (good for clean digital scans)
        quad, q_score = self._detect_form_quad(gray)
        if quad is not None and q_score >= 0.55:
            dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
            if H is not None:
                self.log(f"Using contour quad alignment (quality={q_score:.2f})")
                return H, float(q_score)

        # Build structure masks for feature matching fallback
        def _structure_mask(g: np.ndarray) -> np.ndarray:
            g = cv2.GaussianBlur(g, (3, 3), 0)
            bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 11)
            h, w = bw.shape[:2]
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, w // 18), 1))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(50, h // 18)))
            horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
            vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)
            mask = cv2.bitwise_or(horiz, vert)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
            return mask

        h, w = gray.shape[:2]

        # Template is already grayscale; build structure masks for both (feature fallback)
        template_mask = _structure_mask(template)
        image_mask = _structure_mask(gray)
        
        h, w = gray.shape[:2]
        
        # 1. Coarse Alignment using SIFT (Handles large rotation/scale)
        kp1, des1 = [], None
        kp2, des2 = [], None
        
        # Try SIFT first (robust)
        try:
            sift = cv2.SIFT_create(nfeatures=8000)
            kp1, des1 = sift.detectAndCompute(image_mask, None)
            kp2, des2 = sift.detectAndCompute(template_mask, None)
        except Exception as e:
            print(f"[Alignment] SIFT failed: {e}")
            
        # Fallback to ORB if SIFT fails or returns few keypoints
        if des1 is None or len(kp1) < 10:
            try:
                orb = cv2.ORB_create(nfeatures=8000)
                kp1, des1 = orb.detectAndCompute(image_mask, None)
                kp2, des2 = orb.detectAndCompute(template_mask, None)
            except Exception as e:
                print(f"[Alignment] ORB failed: {e}")

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, 0.0
            
        bf = cv2.BFMatcher()
        good_matches = []
        try:
            # Check descriptor type to choose norm
            norm = cv2.NORM_L2
            if des1.dtype == np.uint8: # ORB uses uint8 descriptors
                norm = cv2.NORM_HAMMING
                bf = cv2.BFMatcher(norm, crossCheck=False)
            
            matches = bf.knnMatch(des1, des2, k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except Exception:
            pass
        
        if len(good_matches) < 10:
            return None, 0.0
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find Homography (Global)
        H_coarse, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # If Homography fails or is invalid, try Affine (simpler, more stable)
        if H_coarse is None or not self._validate_homography(H_coarse, (h, w)):
            try:
                # estimateAffinePartial2D covers rotation, translation, scaling (no shear)
                affine_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if affine_matrix is not None:
                    # Convert 2x3 affine to 3x3 homography
                    H_coarse = np.vstack([affine_matrix, [0, 0, 1]])
                else:
                    return None, 0.0
            except Exception:
                return None, 0.0
            
        # No ECC refinement (prevents tilt)
        return H_coarse, 0.55
    
    def _validate_homography(self, H: np.ndarray, img_shape: Tuple[int, int]) -> bool:
        """
        Validate homography is sane and won't cause distortion.
        STRICT validation to prevent zoom/skew/distortion artifacts.
        """
        if H is None:
            return False
        try:
            h, w = img_shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            if not np.isfinite(warped).all():
                return False
            
            # STRICT: Reject if warped polygon area differs by more than 30%
            # (prevents zoom/scale distortion)
            area = float(cv2.contourArea(warped.astype(np.float32)))
            orig = float(w * h)
            if area < orig * 0.70 or area > orig * 1.30:
                print(f"[Alignment] Rejected: area ratio {area/orig:.2f} out of bounds [0.70, 1.30]")
                return False
            
            # Reject flips (polygon should be convex)
            if not cv2.isContourConvex(warped.astype(np.float32)):
                print("[Alignment] Rejected: warped corners not convex (flip detected)")
                return False
            
            # STRICT: Check for excessive skew/perspective
            # Compute edge lengths of warped quad
            edge_lengths = []
            for i in range(4):
                p1, p2 = warped[i], warped[(i + 1) % 4]
                edge_lengths.append(np.linalg.norm(p2 - p1))
            
            # Top/bottom edges should be similar, left/right should be similar
            top_bottom_ratio = max(edge_lengths[0], edge_lengths[2]) / max(1, min(edge_lengths[0], edge_lengths[2]))
            left_right_ratio = max(edge_lengths[1], edge_lengths[3]) / max(1, min(edge_lengths[1], edge_lengths[3]))
            
            if top_bottom_ratio > 1.25 or left_right_ratio > 1.25:
                print(f"[Alignment] Rejected: excessive perspective (TB={top_bottom_ratio:.2f}, LR={left_right_ratio:.2f})")
                return False
            
            # Check diagonal condition - should form near-rectangle
            diag1 = np.linalg.norm(warped[2] - warped[0])
            diag2 = np.linalg.norm(warped[3] - warped[1])
            diag_ratio = max(diag1, diag2) / max(1, min(diag1, diag2))
            if diag_ratio > 1.15:
                print(f"[Alignment] Rejected: diagonal ratio {diag_ratio:.2f} too high (distortion)")
                return False
            
            return True
        except Exception as e:
            print(f"[Alignment] Validation exception: {e}")
            return False
    
    async def process(self, image: np.ndarray, form_type: FormType) -> AlignmentResult:
        """Align image to template."""
        await self.initialize()

        # Deterministic CMS-1500 alignment path (classical CV only).
        if form_type == FormType.CMS1500:
            try:
                from src.pipelines.cms1500_register import get_cms1500_registrar

                registrar = get_cms1500_registrar()
                reg = registrar.register(image)
                if reg.success and reg.aligned_image is not None and reg.homography_input_to_template is not None:
                    return AlignmentResult(
                        success=True,
                        aligned_image=reg.aligned_image,
                        homography_matrix=reg.homography_input_to_template,
                        alignment_quality=float(reg.quality),
                        fallback_used=False,
                        metadata={
                            "alignment_method": reg.method,
                            "registrar_debug": reg.debug,
                        },
                    )
                self.log(
                    f"CMS registrar fallback to legacy aligner (method={getattr(reg, 'method', 'unknown')}, q={float(getattr(reg, 'quality', 0.0)):.2f})"
                )
            except Exception as e:
                self.log(f"CMS registrar exception: {e}")
        
        # Check if we have a template for this form type
        if form_type not in self._templates:
            # Last-chance: try dynamic load from reference assets
            try:
                from src.processing.registration import load_and_process_reference
                ref_data = load_and_process_reference(form_type.value if hasattr(form_type, "value") else str(form_type))
                if ref_data and ref_data.get("image") is not None:
                    self._templates[form_type] = ref_data["image"]
            except Exception:
                pass
        if form_type not in self._templates:
            return AlignmentResult(
                success=False,
                aligned_image=image,
                homography_matrix=None,
                alignment_quality=0.0,
                fallback_used=True,
                error_message="No template available",
                metadata={"alignment_method": "none"},
            )
        
        template = self._templates[form_type]
        th, tw = template.shape[:2]
        
        # Compute homography
        H, quality = self._compute_homography(image, template)
        
        # Validate homography
        if H is not None and self._validate_homography(H, image.shape):
            # IMPORTANT:
            # H is estimated in the *template pixel coordinate system* (dst points come from template).
            # Therefore the aligned output should be rendered in (tw, th), not the original (w, h).
            aligned = cv2.warpPerspective(image, H, (tw, th))

            # Optional fine translation refinement to reduce small shifts
            try:
                if len(aligned.shape) == 3:
                    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
                else:
                    aligned_gray = aligned
                tmpl_gray = template
                # Edge maps for stable correlation
                a_edges = cv2.Canny(aligned_gray, 50, 150)
                t_edges = cv2.Canny(tmpl_gray, 50, 150)
                (shift_x, shift_y), response = cv2.phaseCorrelate(
                    np.float32(t_edges), np.float32(a_edges)
                )
                # Apply only small translations to avoid new drift
                if abs(shift_x) <= 20 and abs(shift_y) <= 20 and response > 0.1:
                    M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
                    aligned = cv2.warpAffine(aligned, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                # Edge correlation score for alignment quality (0..1)
                if a_edges.shape == t_edges.shape:
                    corr = cv2.matchTemplate(a_edges, t_edges, cv2.TM_CCOEFF_NORMED)[0][0]
                    edge_score = float((corr + 1.0) / 2.0)
                    if edge_score > quality:
                        quality = edge_score
            except Exception:
                pass
            
            return AlignmentResult(
                success=True,
                aligned_image=aligned,
                homography_matrix=H,
                alignment_quality=quality,
                fallback_used=False,
                metadata={"alignment_method": "legacy_homography"},
            )
        
        # Fallback: return original image
        return AlignmentResult(
            success=False,
            aligned_image=image,
            homography_matrix=None,
            alignment_quality=0.0,
            fallback_used=True,
            error_message="Homography validation failed",
            metadata={"alignment_method": "legacy_failed"},
        )


# ============================================================================
# Layout Detection Agent
# ============================================================================

class LayoutDetectionAgent(BaseAgent):
    """
    Detects document layout using appropriate model based on form type.
    
    CMS-1500: YOLOv8 (fine-tuned) or template zones
    General: LayoutLMv3 / Detectron2 (PubLayNet) / Donut
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__("LayoutDetectionAgent")
        self.config = config
        self._yolo = None
        self._detectron = None
        self._layoutlm = None
        self._layout_backend = None
    
    async def initialize(self):
        if self._initialized:
            return
        
        # Initialize YOLO if available
        print(f"[LayoutAgent] Config.YOLO_MODEL_PATH = {Config.YOLO_MODEL_PATH}")
        
        if Config.YOLO_MODEL_PATH:
            try:
                from pathlib import Path
                from src.pipelines.yolo_layout import YOLOLayoutDetector
                
                # Resolve model path (could be relative or absolute)
                model_path = Path(Config.YOLO_MODEL_PATH)
                print(f"[LayoutAgent] model_path = {model_path}, exists = {model_path.exists()}")
                
                if not model_path.is_absolute():
                    model_path = Config.PROJECT_ROOT / model_path
                    print(f"[LayoutAgent] Resolved to {model_path}, exists = {model_path.exists()}")
                
                if model_path.exists():
                    # Use lower confidence (0.1) to get more detections from fine-tuned model
                    conf = min(self.config.yolo_confidence, 0.10)
                    print(f"[LayoutAgent] Creating YOLO detector with conf={conf}")
                    self._yolo = YOLOLayoutDetector(
                        str(model_path),
                        conf=conf,
                        iou=Config.YOLO_IOU
                    )
                    print(f"[LayoutAgent] ✅ YOLO detector initialized from {model_path}")
                else:
                    print(f"[LayoutAgent] ❌ YOLO model not found at {model_path}")
            except Exception as e:
                import traceback
                print(f"[LayoutAgent] ❌ YOLO init failed: {e}")
                traceback.print_exc()
        else:
            print("[LayoutAgent] ❌ Config.YOLO_MODEL_PATH is None or empty")
        
        # Initialize layout detection model for general forms.
        # Prefer Detectron2 weights if available (better for some docs), otherwise use PaddleDetection.
        # PaddleDetection ppyolov2 (PubLayNet) is cached at:
        #   /root/.torch/iopath_cache/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet/
        # It produces 5 classes: Text, Title, List, Table, Figure
        from pathlib import Path
        layout_pref = (self.config.layout_model or "auto").lower().strip()

        detectron_candidates = [
            Path("/root/.detectron2/models/publaynet_faster_rcnn_R_50_FPN_3x.pth"),
            Path("/app/models/publaynet_faster_rcnn_R_50_FPN_3x.pth"),
            Config.PROJECT_ROOT / "models" / "publaynet_faster_rcnn_R_50_FPN_3x.pth",
            Config.PROJECT_ROOT / "models" / "detectron2" / "publaynet_faster_rcnn_R_50_FPN_3x.pth",
        ]
        detectron_model = next((p for p in detectron_candidates if p.exists()), None)

        def _init_detectron(model_path: Path) -> bool:
            try:
                import layoutparser as lp
                config_uri = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
                self._detectron = lp.Detectron2LayoutModel(
                    config_uri,
                    model_path=str(model_path),
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.config.detectron_threshold],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                )
                self._layout_backend = "detectron2"
                self.log(f"✅ Detectron2 initialized ({model_path})")
                return True
            except Exception as e:
                self.log(f"⚠️ Detectron2 init failed: {e}")
                return False

        def _init_paddle() -> bool:
            try:
                import layoutparser as lp
                cached_paddle = Path("/root/.torch/iopath_cache/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet/inference.pdiparams")
                if cached_paddle.exists():
                    self.log(f"Found cached PaddleDetection weights at {cached_paddle.parent}")
                self._detectron = lp.PaddleDetectionLayoutModel(
                    config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config",
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                    extra_config={"threshold": self.config.detectron_threshold},
                )
                self._layout_backend = "paddle"
                self.log("✅ PaddleDetection (PubLayNet ppyolov2) initialized for general forms")
                return True
            except Exception as e:
                self.log(f"⚠️ PaddleDetection init failed: {e}")
                return False

        prefer_detectron = layout_pref in ("detectron2", "detectron")
        prefer_paddle = layout_pref in ("paddle", "layoutparser", "publaynet")

        initialized = False
        if prefer_detectron or (layout_pref == "auto" and detectron_model is not None):
            if detectron_model is None:
                self.log("❌ No Detectron2 weights found locally, skipping Detectron2 init")
            else:
                initialized = _init_detectron(detectron_model)

        if not initialized and (prefer_paddle or layout_pref == "auto"):
            initialized = _init_paddle()

        if not initialized and not prefer_detectron and detectron_model is not None:
            initialized = _init_detectron(detectron_model)
        
        self._initialized = True
    
    def _yolo_detect(self, image: np.ndarray) -> List[DetectedBlock]:
        """Detect using YOLOv8."""
        if self._yolo is None:
            return []
        
        blocks = self._yolo.predict(image, page_id=0)
        
        return [
            DetectedBlock(
                id=b.id,
                block_type=self._map_yolo_type(b.type.value if hasattr(b.type, 'value') else str(b.type)),
                bbox=b.bbox,
                confidence=b.confidence,
                metadata=b.metadata or {}
            )
            for b in blocks
        ]
    
    def _detectron_detect(self, image: np.ndarray) -> List[DetectedBlock]:
        """Detect using Detectron2/LayoutParser."""
        if self._detectron is None:
            return []
        
        try:
            layout = self._detectron.detect(image)
            blocks = []
            backend = self._layout_backend or "layoutparser"
            
            for i, element in enumerate(layout):
                block_type = self._map_detectron_type(element.type)
                bbox = (element.block.x_1, element.block.y_1, 
                       element.block.x_2, element.block.y_2)
                
                blocks.append(DetectedBlock(
                    id=f"det-{i}",
                    block_type=block_type,
                    bbox=bbox,
                    confidence=element.score,
                    metadata={"model": backend}
                ))
            
            return blocks
        except Exception as e:
            self.log(f"Detectron detection error: {e}")
            return []
    
    def _map_yolo_type(self, type_str: str) -> BlockType:
        """Map YOLO class to BlockType."""
        mapping = {
            "field": BlockType.FORM_FIELD,
            "form": BlockType.FORM_FIELD,
            "table": BlockType.TABLE,
            "figure": BlockType.FIGURE,
            "checkbox": BlockType.CHECKBOX,
            "header": BlockType.HEADER,
            "signature": BlockType.SIGNATURE,
            "text": BlockType.TEXT,
        }
        return mapping.get(type_str.lower(), BlockType.TEXT)
    
    def _map_detectron_type(self, type_str: str) -> BlockType:
        """Map Detectron2 class to BlockType."""
        mapping = {
            "Text": BlockType.TEXT,
            "Title": BlockType.HEADER,
            "List": BlockType.TEXT,
            "Table": BlockType.TABLE,
            "Figure": BlockType.FIGURE,
        }
        return mapping.get(type_str, BlockType.TEXT)
    
    async def process(self, image: np.ndarray, form_type: FormType) -> List[DetectedBlock]:
        """Detect layout blocks."""
        await self.initialize()
        
        # Choose detection strategy based on form type and config
        if form_type == FormType.CMS1500 and self._yolo is not None:
            self.log("Using YOLO for CMS-1500")
            return self._yolo_detect(image)
        elif self._detectron is not None:
            backend = self._layout_backend or "layoutparser"
            self.log(f"Using {backend} for general detection")
            return self._detectron_detect(image)
        else:
            self.log("No layout model available, returning empty")
            return []


# ============================================================================
# OCR Agent (Tiered)
# ============================================================================

class OCRAgent(BaseAgent):
    """
    Tiered OCR agent:
    - PaddleOCR for printed text
    - TrOCR for handwriting/signatures
    - Checkbox density detector
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__("OCRAgent")
        self.config = config
        self._paddle = None
        self._trocr_model = None
        self._trocr_processor = None
        self._templates: Dict[str, np.ndarray] = {}
    
    async def initialize(self):
        if self._initialized:
            return
        
        # Use our PaddleOCRWrapper which handles PaddleX properly
        try:
            from src.ocr.paddle_ocr import PaddleOCRWrapper
            self._paddle = PaddleOCRWrapper()
            self.log("PaddleOCRWrapper initialized")
        except Exception as e:
            self.log(f"PaddleOCR init failed: {e}")
        
        # TrOCR (lazy load on first use)
        self._initialized = True

    def _get_template_gray(self, form_type: Optional[str]) -> Optional[np.ndarray]:
        """Load and cache a grayscale template image for template-diff OCR."""
        if not form_type:
            return None
        key = str(form_type)
        if key in self._templates:
            return self._templates[key]
        try:
            from utils.config import Config
            from pathlib import Path
            tmpl_dir = Path(Config.PROJECT_ROOT) / "data" / "templates"
            # We expect templates like cms-1500.png or cms1500.png
            candidates = [
                tmpl_dir / f"{key}.png",
                tmpl_dir / f"{key.replace('_','-')}.png",
                tmpl_dir / "cms-1500.png",
                tmpl_dir / "cms1500.png",
            ]
            for p in candidates:
                if p.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None and img.size > 0:
                        self._templates[key] = img
                        return img
        except Exception:
            return None
        return None

    def _template_diff_crop(self, crop_rgb: np.ndarray, template_gray: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Compute a template-diff image for a crop (aligned to template coordinate space).
        Returns an RGB image emphasizing filled-in ink, suppressing printed form text/lines.
        """
        try:
            x0, y0, x1, y1 = [int(round(v)) for v in bbox]
            th, tw = template_gray.shape[:2]
            if x0 < 0 or y0 < 0 or x1 > tw or y1 > th or x1 <= x0 or y1 <= y0:
                return None
            if crop_rgb.ndim == 3:
                crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
            else:
                crop_gray = crop_rgb
            tmpl_crop = template_gray[y0:y1, x0:x1]
            if tmpl_crop.shape[:2] != crop_gray.shape[:2]:
                return None
            diff = cv2.absdiff(crop_gray, tmpl_crop)
            # Boost contrast of differences
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            # Binarize and invert to get black text on white background
            _, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Slight dilation to connect strokes
            bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
            # Convert to RGB for OCR engines
            return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
        except Exception:
            return None
    
    def _load_trocr(self):
        """Lazy load TrOCR."""
        if self._trocr_model is not None:
            return
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self._trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self._trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            self.log("TrOCR loaded")
        except Exception as e:
            self.log(f"TrOCR load failed: {e}")
    
    def _paddle_ocr(self, image: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """Run PaddleOCR using new PaddleX API (predict instead of ocr)."""
        if self._paddle is None:
            return "", 0.0, []
        
        try:
            # Use PaddleOCRWrapper which handles the new PaddleX API
            word_boxes = self._paddle.extract_text(image)
            
            if word_boxes:
                texts = [wb.text for wb in word_boxes]
                confs = [wb.confidence for wb in word_boxes]
                boxes = [
                    {
                        "text": wb.text,
                        "bbox": list(wb.bbox),
                        "confidence": wb.confidence
                    }
                    for wb in word_boxes
                ]
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                return " ".join(texts), avg_conf, boxes
        except Exception as e:
            self.log(f"PaddleOCR error: {e}")
        
        return "", 0.0, []
    
    def _trocr_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Run TrOCR for handwriting."""
        if not self.config.enable_trocr:
            return "", 0.0
        
        self._load_trocr()
        
        if self._trocr_model is None:
            return "", 0.0
        
        try:
            from PIL import Image
            import torch
            
            # Convert to PIL
            if len(image.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image)
            
            # Process
            pixel_values = self._trocr_processor(images=pil_img, return_tensors="pt").pixel_values
            
            with torch.no_grad():
                generated_ids = self._trocr_model.generate(pixel_values, max_length=128)
            
            text = self._trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text, 0.85
        except Exception as e:
            self.log(f"TrOCR error: {e}")
        
        return "", 0.0
    
    def _detect_checkbox(self, image: np.ndarray) -> Tuple[bool, float]:
        """Detect if checkbox is checked."""
        if image is None or image.size == 0:
            return False, 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9
        )

        # Ignore the checkbox border by using an inner crop (captures light X marks better)
        h, w = binary.shape[:2]
        pad_x = int(w * 0.18)
        pad_y = int(h * 0.18)
        inner = binary[pad_y:max(pad_y + 1, h - pad_y), pad_x:max(pad_x + 1, w - pad_x)]
        if inner.size == 0:
            inner = binary

        ink = np.count_nonzero(inner)
        area = max(inner.size, 1)
        ink_ratio = ink / area

        # Empirically: empty boxes have very low inner-ink; checked boxes have higher ink.
        # Keep threshold low to catch thin hand-drawn X marks.
        is_checked = ink_ratio > 0.020
        confidence = float(min(1.0, max(0.0, (ink_ratio - 0.01) / 0.05)))
        
        return is_checked, confidence
    
    def _is_handwritten(self, image: np.ndarray, paddle_conf: float, paddle_text: str) -> bool:
        """Heuristic to detect if text is handwritten.
        
        TrOCR is prone to hallucinating on blank/noisy crops.
        Only consider TrOCR if there is visible ink AND PaddleOCR is low-quality.
        """
        def _has_ink(img: np.ndarray) -> bool:
            if img is None or img.size == 0:
                return False
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
            g = cv2.GaussianBlur(g, (3, 3), 0)
            bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9)
            ink_ratio = float(np.count_nonzero(bw) / max(1, bw.size))
            return ink_ratio > 0.015  # very low threshold, but filters blank boxes

        if not _has_ink(image):
            return False
        if paddle_conf < 0.35:
            return True
        if (not paddle_text) or len(paddle_text.strip()) < 2:
            return True
        return False
    
    async def process(self, image: np.ndarray, block: DetectedBlock) -> DetectedBlock:
        """OCR a single block with appropriate method."""
        await self.initialize()

        # If this block already has text from full-page OCR zone matching, do NOT re-OCR tiny crops.
        # Per-field crop OCR often *reduces* quality (tight crops, partial words, missing context),
        # and it also causes duplicates when overlapping zones exist.
        src = (block.metadata or {}).get("source")
        if src == "ocr_zone_matching" and block.block_type not in (BlockType.CHECKBOX, BlockType.SIGNATURE):
            if block.text and len(str(block.text).strip()) > 0:
                block.metadata["ocr_engine"] = "full_page_zone_matching"
                return block
        
        h, w = image.shape[:2]
        x0, y0, x1, y1 = block.bbox
        
        # Expand bbox to capture edges; controlled by config (UI slider `ocr_padding`)
        pad = int(getattr(self.config, "zone_padding_px", 10))
        x0_p, y0_p = max(0, x0 - pad), max(0, y0 - pad)
        x1_p, y1_p = min(w, x1 + pad), min(h, y1 + pad)
        crop = image[int(y0_p):int(y1_p), int(x0_p):int(x1_p)]
        
        if crop.size == 0:
            block.text = ""
            return block
        
        # Checkbox detection
        if block.block_type == BlockType.CHECKBOX:
            # If we have a template (aligned forms), prefer diff-based checkbox detection.
            tmpl = self._get_template_gray((block.metadata or {}).get("form_type"))
            diff_img = self._template_diff_crop(crop, tmpl, (x0_p, y0_p, x1_p, y1_p)) if tmpl is not None else None
            is_checked, conf = self._detect_checkbox(diff_img if diff_img is not None else crop)
            block.text = "X" if is_checked else ""
            block.confidence = conf
            block.metadata["ocr_engine"] = "checkbox_detector"
            return block
        
        # Signature detection
        if block.block_type == BlockType.SIGNATURE:
            text, conf = self._trocr_ocr(crop)
            if text:
                block.text = text
                block.confidence = conf
                block.metadata["ocr_engine"] = "trocr"
            else:
                # Check if signature present by density
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                density = np.count_nonzero(binary) / binary.size
                if density > 0.05:
                    block.text = "[SIGNED]"
                    block.confidence = min(1.0, density * 5)
                block.metadata["ocr_engine"] = "signature_detector"
            return block
        
        # Standard text — for CMS-1500 scans, apply safe red removal on the crop.
        # This strips red template labels/lines while preserving handwriting strokes.
        crop_for_ocr = crop
        form_type_meta = (block.metadata or {}).get("form_type", "")
        if form_type_meta in ("cms-1500", "CMS1500", "cms1500"):
            try:
                from src.processing.preprocessing import remove_red_template_text
                crop_for_ocr = remove_red_template_text(crop)
                block.metadata["red_removal_used"] = True
            except Exception:
                pass
        else:
            # Non-CMS forms: try template-diff if available
            tmpl = self._get_template_gray(form_type_meta)
            if tmpl is not None and image.shape[0] == tmpl.shape[0] and image.shape[1] == tmpl.shape[1]:
                diff_img = self._template_diff_crop(crop, tmpl, (x0_p, y0_p, x1_p, y1_p))
                if diff_img is not None:
                    crop_for_ocr = diff_img
                    block.metadata["template_diff_used"] = True

        # Tiered OCR (PaddleOCR first, TrOCR fallback for handwritten forms)
        paddle_text, paddle_conf, paddle_boxes = self._paddle_ocr(crop_for_ocr)
        
        # Accuracy boost: If low confidence or very short text, try zoomed crop
        if paddle_conf < 0.5 and paddle_text:
            zoomed = cv2.resize(crop_for_ocr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            z_text, z_conf, z_boxes = self._paddle_ocr(zoomed)
            if z_conf > paddle_conf:
                paddle_text, paddle_conf, paddle_boxes = z_text, z_conf, z_boxes

        block.metadata["ocr_boxes"] = paddle_boxes

        def _alnum_ratio(s: str) -> float:
            s = (s or "").strip()
            if not s:
                return 0.0
            an = sum(ch.isalnum() for ch in s)
            return an / max(1, len(s))

        def _looks_noisy(s: str) -> bool:
            s = (s or "").strip()
            if not s:
                return True
            if len(s) <= 2:
                return True
            return _alnum_ratio(s) < 0.55
        
        def _looks_truncated(s: str) -> bool:
            """Check if text looks truncated (starts with lowercase, missing first letter)"""
            s = (s or "").strip()
            if not s or len(s) < 3:
                return False
            # If starts with lowercase and has comma (likely name), might be truncated
            if s[0].islower() and ',' in s:
                return True
            return False

        # TrOCR for handwritten forms
        # Use TrOCR more aggressively for CMS-1500 scans where PaddleOCR struggles with handwriting
        use_trocr = getattr(self.config, 'enable_trocr', False) and self.config.enable_trocr
        is_cms_scan = form_type_meta in ("cms-1500", "CMS1500", "cms1500")
        
        # Trigger TrOCR for CMS-1500 scans when:
        # 1. PaddleOCR got nothing
        # 2. PaddleOCR confidence is low (< 0.50 for scans)
        # 3. Result looks noisy or truncated
        trocr_threshold = 0.50 if is_cms_scan else 0.30
        should_try_trocr = (
            use_trocr and 
            (not paddle_text or 
             paddle_conf < trocr_threshold or 
             _looks_noisy(paddle_text) or
             _looks_truncated(paddle_text))
        )
        
        if should_try_trocr:
            if self._is_handwritten(crop_for_ocr, paddle_conf, paddle_text):
                trocr_text, trocr_conf = self._trocr_ocr(crop_for_ocr)
                if trocr_text and len(trocr_text.strip()) > 0:
                    tt = trocr_text.strip()
                    # Reject obvious TrOCR hallucinations
                    if not self._is_hallucination(tt):
                        # For CMS-1500, prefer TrOCR if it's more complete
                        # (captures first letters that PaddleOCR missed)
                        if is_cms_scan:
                            paddle_len = len((paddle_text or "").strip())
                            trocr_len = len(tt)
                            # TrOCR wins if it's longer AND has decent confidence
                            if trocr_len > paddle_len or trocr_conf > paddle_conf:
                                block.text = tt
                                block.confidence = trocr_conf
                                block.metadata["ocr_engine"] = "trocr"
                                block.metadata["paddle_text"] = paddle_text  # Keep for comparison
                                return block
                        else:
                            block.text = tt
                            block.confidence = trocr_conf
                            block.metadata["ocr_engine"] = "trocr"
                            return block

        # Default: PaddleOCR result
        block.text = (paddle_text or "").strip()
        block.confidence = paddle_conf
        block.metadata["ocr_engine"] = "paddleocr"
        return block
    
    def _is_hallucination(self, text: str) -> bool:
        """Detect OCR/TrOCR hallucinations - garbage text patterns."""
        import re
        
        if not text or len(text.strip()) < 2:
            return False
        
        text = text.strip()
        
        # Pattern 1: Repeated letters/chars (e.g., "C.C.C.C.C", "U.U.U.U")
        if re.search(r'(.)\1{4,}', text.replace('.', '').replace(' ', '')):
            return True
        
        # Pattern 2: Too many periods/dots in sequence (e.g., "P.P.P", "S.S.S")
        dot_count = text.count('.')
        if dot_count > 3 and dot_count / max(1, len(text)) > 0.15:
            return True
        
        # Pattern 3: Nonsense phrases (common TrOCR hallucinations)
        hallucination_phrases = [
            r"parliament",
            r"american.*parliament",
            r"biographical.*directory",
            r"special.*announcement",
            r"member.*of.*the",
            r"throughout.*the",
            r"what.*is.*a.*member",
            r"application.*of.*[a-z]\.[a-z]\.[a-z]",
            r"manance",
            r"[a-z]\.[a-z]\.[a-z]\.[a-z]\.[a-z]",  # Like P.P.P.P.P
        ]
        text_lower = text.lower()
        for pattern in hallucination_phrases:
            if re.search(pattern, text_lower):
                return True
        
        # Pattern 4: Very long text with low alphanumeric ratio
        alphanum = sum(1 for c in text if c.isalnum())
        if len(text) > 20 and alphanum / len(text) < 0.5:
            return True
        
        # Pattern 5: Repeating word patterns
        words = text.split()
        if len(words) >= 4:
            unique_words = set(w.lower() for w in words)
            if len(unique_words) / len(words) < 0.4:  # Too repetitive
                return True
        
        return False
    
    async def process_blocks(self, image: np.ndarray, blocks: List[DetectedBlock]) -> List[DetectedBlock]:
        """Process multiple blocks concurrently."""
        await self.initialize()
        
        # Process blocks (could be parallelized with asyncio.gather)
        results = []
        for block in blocks:
            result = await self.process(image, block)
            results.append(result)
        
        # MINIMAL filtering - only remove truly empty blocks
        # Trust PaddleOCR results - the hallucination issue was from TrOCR which is now disabled
        filtered = []
        for block in results:
            # Keep ALL schema-matched zones (CMS-1500, UB-04)
            src = (block.metadata or {}).get("source", "")
            if src in ("schema_zones", "ocr_zone_matching", "cms1500_production", "full_page_ocr"):
                filtered.append(block)
                continue
            
            # Keep checkboxes and signatures
            if block.block_type in (BlockType.CHECKBOX, BlockType.SIGNATURE):
                filtered.append(block)
                continue
            
            # Keep blocks with any text
            if block.text and len(str(block.text).strip()) > 0:
                filtered.append(block)
                continue
            
            # Keep blocks with decent confidence even if text extraction pending
            if block.confidence >= 0.3:
                filtered.append(block)
        
        return filtered


# ============================================================================
# SLM/VLM Labeling Agent
# ============================================================================

class LabelingAgent(BaseAgent):
    """
    Semantic labeling using SLM/VLM:
    - SLM (Llama 3.2) for text field labeling
    - TATR + SLM for tables
    - VLM (MiniCPM-V) for figures/charts
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__("LabelingAgent")
        self.config = config
        self._ollama_available: Optional[bool] = None
        self._available_models: List[str] = []
    
    async def initialize(self):
        if self._initialized:
            return
        # Check if Ollama has models available (prevent hallucination from empty Ollama)
        try:
            import requests
            resp = requests.get(f"http://{Config.OLLAMA_HOST}/api/tags", timeout=5)
            if resp.ok:
                models = resp.json().get("models", [])
                self._available_models = [m.get("name", "") for m in models]
                self._ollama_available = len(self._available_models) > 0
                if self._ollama_available:
                    self.log(f"Ollama models available: {self._available_models}")
                else:
                    self.log("WARNING: Ollama has no models — SLM labeling will be skipped")
            else:
                self._ollama_available = False
                self.log("WARNING: Ollama not responding — SLM labeling disabled")
        except Exception as e:
            self._ollama_available = False
            self.log(f"WARNING: Ollama check failed: {e} — SLM labeling disabled")
        self._initialized = True
    
    def _call_slm(self, prompt: str) -> str:
        """Call SLM via Ollama. Returns empty string if Ollama unavailable."""
        # Guard: skip if no models available (prevents hallucination)
        if self._ollama_available is False:
            return ""
        try:
            import requests
            response = requests.post(
                f"http://{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": self.config.slm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 500}
                },
                timeout=60
            )
            if response.ok:
                return response.json().get("response", "")
        except Exception as e:
            self.log(f"SLM call failed: {e}")
        return ""
    
    def _call_vlm(self, prompt: str, image: np.ndarray) -> str:
        """Call VLM via Ollama with image."""
        try:
            import requests
            import base64
            
            # Encode image
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            response = requests.post(
                f"http://{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": self.config.vlm_model,
                    "prompt": prompt,
                    "images": [img_base64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 500}
                },
                timeout=60
            )
            if response.ok:
                return response.json().get("response", "")
        except Exception as e:
            self.log(f"VLM call failed: {e}")
        return ""
    
    def clean_cms1500_field_with_slm(
        self, 
        raw_ocr_text: str, 
        field_name: str, 
        field_type: str = "text"
    ) -> str:
        """
        Use SLM to intelligently clean OCR output for a CMS-1500 field.
        
        This helps extract just the handwritten value from OCR that may include
        garbled template labels like "(Last Name, First Name, Middle)" -> "Khan Shah Rukh"
        
        Args:
            raw_ocr_text: Raw OCR output (may include template labels)
            field_name: Human-readable field name (e.g., "Patient Name")
            field_type: Field type hint (text, date, phone, etc.)
            
        Returns:
            Cleaned value or original if SLM unavailable/fails
        """
        if not raw_ocr_text or not raw_ocr_text.strip():
            return ""
            
        # Skip SLM for very clean-looking values (no noise)
        clean_chars = sum(1 for c in raw_ocr_text if c.isalnum() or c.isspace())
        if clean_chars / max(len(raw_ocr_text), 1) > 0.95:
            return raw_ocr_text.strip()
        
        # Guard: skip if no models available
        if self._ollama_available is False:
            return raw_ocr_text
        
        # Build a focused prompt
        type_hints = {
            "text": "a name, word, or phrase",
            "date": "a date in MM/DD/YYYY or similar format",
            "phone": "a phone number",
            "address": "a street address",
            "npi": "a 10-digit NPI number",
            "money": "a dollar amount",
            "icd10": "an ICD-10 diagnosis code",
        }
        type_hint = type_hints.get(field_type, "text")
        
        prompt = f"""Extract ONLY the handwritten value from this OCR text.
The field is "{field_name}" which should contain {type_hint}.
Remove any pre-printed form labels, OCR artifacts, or template text.

OCR Text: {raw_ocr_text}

Rules:
- Output ONLY the actual handwritten/filled-in value
- Remove template labels like "(Last Name, First Name)" or "MM DD YY"
- Remove garbled OCR text that doesn't look like real data
- If the OCR is too garbled to extract a value, output "UNCLEAR"
- Keep the value brief and clean

Extracted Value:"""

        try:
            result = self._call_slm(prompt)
            if result:
                # Clean up SLM output
                result = result.strip()
                # Remove common SLM response prefixes
                for prefix in ["Extracted Value:", "Value:", "The value is", "Answer:"]:
                    if result.lower().startswith(prefix.lower()):
                        result = result[len(prefix):].strip()
                # If SLM says unclear, return empty
                if result.upper() in ["UNCLEAR", "N/A", "NONE", "EMPTY"]:
                    return ""
                # Don't return if SLM added more noise
                if len(result) > len(raw_ocr_text) * 2:
                    return raw_ocr_text
                return result
        except Exception as e:
            self.log(f"SLM field cleaning failed: {e}")
        
        return raw_ocr_text
    
    async def label_text_block(self, block: DetectedBlock, context: str = "") -> DetectedBlock:
        """Label a text block and clean its value using SLM."""
        if not self.config.enable_slm_labeling or not block.text:
            return block

        # Guardrails: never let the SLM rewrite already-structured fields.
        # For CMS-1500, schema/zone-matched values are the source of truth. An LLM can:
        # - hallucinate text that isn't on the page
        # - drop leading characters (e.g., "Baltimore" -> "altimore")
        # - rename semantic labels causing key collisions/overwrites
        meta = block.metadata or {}
        src = str(meta.get("source") or "").strip().lower()
        form_type = str(meta.get("form_type") or "").strip().lower()
        existing_sem = str(meta.get("semantic_label") or "").strip()
        try:
            is_schema_like = bool(re.match(r"^\d+[a-z]?\_", str(block.id or ""))) or bool(re.match(r"^\d+[a-z]?\_", existing_sem))
        except Exception:
            is_schema_like = False
        if src in {"ocr_zone_matching", "schema_zones", "cms1500_production"}:
            return block
        if form_type in {"cms-1500", "cms1500"} and is_schema_like:
            return block
        
        # Enhanced prompt for semantic tagging and value extraction
        prompt = f"""Analyze this text block from a medical form.
Block Text: "{block.text}"
Context: {context}

1. Classify the Semantic Role: [Title, Section Header, Footer, Page Number, Key-Value Pair, List Item, Signature, Comment, Other]
2. Identify the Field Name (if Key-Value Pair).
3. Extract ONLY the Clean Value (remove printed labels, instructions).

Respond in JSON format: 
{{
  "role": "...", 
  "field_name": "...", 
  "clean_value": "..."
}}
"""
        
        response = self._call_slm(prompt)
        try:
            import json
            from difflib import SequenceMatcher
            clean_resp = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_resp)
            
            # Store fine-grained semantic role
            role = (data.get("role") or "text").lower()
            field_name = str(data.get("field_name", "Unknown") or "Unknown").strip()
            if block.metadata is None:
                block.metadata = {}
            block.metadata["semantic_role"] = role
            block.metadata["semantic_field_name"] = field_name
            # Keep extracted_fields keys stable/unique by default.
            # If we do assign a semantic_label for readability, suffix with block.id to avoid collisions.
            if field_name and field_name.lower() not in {"unknown", "text", "form_field"}:
                block.metadata["semantic_label"] = f"{field_name}::{block.id}"

            # Map semantic role to block type
            if "title" in role:
                block.block_type = BlockType.TITLE
            elif "header" in role:
                block.block_type = BlockType.HEADER
            elif "footer" in role:
                block.block_type = BlockType.FOOTER
            elif "page" in role:
                block.block_type = BlockType.PAGE_NUM
            elif "signature" in role:
                block.block_type = BlockType.SIGNATURE
            elif "list" in role:
                block.block_type = BlockType.LIST
            else:
                # keep as text/form_field; semantic label captured in metadata
                pass

            cleaned = data.get("clean_value")
            if cleaned and str(cleaned).strip().lower() not in ["null", "none", ""]:
                cleaned_str = str(cleaned).strip()
                orig_str = str(block.text or "").strip()
                # Only accept "clean_value" if it is grounded in the original text.
                # This prevents hallucinations from leaking into the final output.
                o = re.sub(r"\s+", " ", orig_str.lower())
                c = re.sub(r"\s+", " ", cleaned_str.lower())
                grounded = False
                try:
                    grounded = (c in o) or (SequenceMatcher(None, c, o).ratio() >= 0.72)
                except Exception:
                    grounded = False
                if grounded:
                    block.metadata["original_text"] = block.text
                    block.text = cleaned_str
                else:
                    block.metadata["slm_clean_value_rejected"] = cleaned_str
        except Exception:
            pass
            
        return block
    
    async def process_table(self, image: np.ndarray, block: DetectedBlock) -> Dict[str, Any]:
        """Process table block using VLM for structured extraction.
        
        For CMS-1500 service lines (Box 24), extract:
        - Date of service (FROM/TO)
        - Place of service
        - EMG
        - CPT/HCPCS codes
        - Diagnosis pointer
        - Charges
        - Days/Units
        - NPI
        """
        h, w = image.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in block.bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        crop = image[y0:y1, x0:x1]
        
        table_data = {
            "type": "table",
            "bbox": block.bbox,
            "rows": [],
            "raw_text": block.text
        }
        
        # Use VLM for table extraction (more accurate for handwritten forms)
        use_vlm = getattr(self.config, 'enable_vlm_tables', True) or self.config.enable_vlm_figures
        if use_vlm and crop.size > 0:
            prompt = """Extract service line data from this CMS-1500 form table (Box 24).
For each row, extract:
- date_from: MM/DD/YY format
- date_to: MM/DD/YY format  
- place_of_service: 2-digit code
- cpt_code: 5-digit procedure code
- modifier: optional modifier codes
- diagnosis_pointer: letter A-L
- charges: dollar amount
- days_units: number of units
- npi: 10-digit provider number

Return as JSON array of objects, one per service line. Only include rows with actual data."""
            
            try:
                response = self._call_vlm(prompt, crop)
                if response:
                    # Try to parse JSON from response
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', response)
                    if json_match:
                        rows = json.loads(json_match.group())
                        table_data["rows"] = rows
                        table_data["extraction_method"] = "vlm"
                    else:
                        table_data["vlm_response"] = response
                        table_data["extraction_method"] = "vlm_text"
            except Exception as e:
                table_data["vlm_error"] = str(e)
        
        # Fallback: use OCR text + SLM parsing
        if not table_data["rows"] and self.config.enable_slm_labeling and block.text:
            prompt = f"""Parse CMS-1500 service line data from this OCR text:
"{block.text}"

Extract each service line with: date_from, date_to, place_of_service, cpt_code, charges, days_units.
Return as JSON array."""
            
            try:
                response = self._call_slm(prompt)
                if response:
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', response)
                    if json_match:
                        table_data["rows"] = json.loads(json_match.group())
                        table_data["extraction_method"] = "slm"
            except:
                pass
        
        return table_data
    
    async def process_figure(self, image: np.ndarray, block: DetectedBlock) -> Dict[str, Any]:
        """Process figure/chart using VLM."""
        figure_data = {
            "type": "figure",
            "bbox": block.bbox,
            "description": ""
        }
        
        if self.config.enable_vlm_figures:
            h, w = image.shape[:2]
            x0, y0, x1, y1 = block.bbox
            crop = image[int(y0):int(y1), int(x0):int(x1)]
            
            prompt = "Describe this image/chart from a medical document. What does it show?"
            description = self._call_vlm(prompt, crop)
            figure_data["description"] = description
        
        return figure_data
    
    async def process(self, image: np.ndarray, blocks: List[DetectedBlock]) -> List[DetectedBlock]:
        """Process all blocks with appropriate labeling."""
        await self.initialize()
        
        for block in blocks:
            if block.block_type == BlockType.TABLE:
                table_data = await self.process_table(image, block)
                block.metadata["table_data"] = table_data
            elif block.block_type == BlockType.FIGURE:
                figure_data = await self.process_figure(image, block)
                block.metadata["figure_data"] = figure_data
            else:
                block = await self.label_text_block(block)
        
        return blocks


# ============================================================================
# Validation Agent
# ============================================================================

class ValidationAgent(BaseAgent):
    """Field validation and QA checks."""
    
    VALIDATORS = {
        "date": r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})$",
        "phone": r"^[\d\(\)\-\s]{10,14}$",
        "npi": r"^\d{10}$",
        "icd10": r"^[A-Z]\d{2}\.?\d{0,4}$",
        "cpt": r"^\d{5}$",
        "zip": r"^\d{5}(-\d{4})?$",
    }
    
    def __init__(self, config: PipelineConfig):
        super().__init__("ValidationAgent")
        self.config = config
    
    async def initialize(self):
        self._initialized = True
    
    def validate_field(self, value: str, field_type: str) -> Tuple[bool, str]:
        """Validate a field value."""
        if not value or field_type not in self.VALIDATORS:
            return True, ""
        
        pattern = self.VALIDATORS[field_type]
        if re.match(pattern, value.upper().strip()):
            return True, ""
        return False, f"Invalid {field_type} format"
    
    async def llm_qa_check(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Run LLM QA check on extracted data."""
        if not self.config.enable_llm_qa:
            return []
        
        notes = []
        try:
            import requests
            
            fields_str = "\n".join([f"- {k}: {v}" for k, v in extracted_data.items() if v])
            prompt = f"""Review this medical form extraction for errors:
{fields_str}

List only obvious errors (max 3). If all looks good, say "OK"."""
            
            response = requests.post(
                f"http://{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": self.config.slm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200}
                },
                timeout=30
            )
            
            if response.ok:
                result = response.json().get("response", "").strip()
                if result and "ok" not in result.lower():
                    notes.append(result)
        except:
            pass
        
        return notes
    
    async def process(self, blocks: List[DetectedBlock], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all data."""
        await self.initialize()
        
        validation_results = {
            "errors": [],
            "warnings": [],
            "qa_notes": []
        }
        
        # Field-level validation
        for block in blocks:
            field_type = block.metadata.get("field_type")
            if field_type:
                valid, msg = self.validate_field(block.text, field_type)
                if not valid:
                    validation_results["errors"].append({
                        "field_id": block.id,
                        "message": msg
                    })
        
        # LLM QA
        qa_notes = await self.llm_qa_check(extracted_data)
        validation_results["qa_notes"] = qa_notes
        
        return validation_results


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

class MultiAgentPipeline:
    """
    Main orchestrator for the multi-agent document processing pipeline.
    
    CMS-1500 uses a 3-lane extraction strategy:
      Lane A (best):  AcroForm widgets → values (no OCR needed for fillable PDFs)
      Lane B (good):  PDF text layer → word boxes → zone match (flattened digital)
      Lane C (scan):  align → template subtraction → OCR → zone match → ICR fallback
    """
    
    # ── Widget name → schema field ID mapping ──────────────────────────
    # Built from actual Cigna CMS-1500 AcroForm field names.
    # This is the authoritative mapping for Lane A extraction.
    WIDGET_TO_SCHEMA: Dict[str, str] = {
        # Top header / insurance
        "insurance_name":           "header_top_right_notes",  # insurance company name
        "insurance_id":             "1a_insured_id",
        # Patient
        "pt_name":                  "2_patient_name",
        "pt_street":                "5_patient_address",
        "pt_city":                  "5_patient_city",
        "pt_state":                 "5_patient_state",
        "pt_zip":                   "5_patient_zip",
        "pt_AreaCode":              "_pt_area_code",  # composed into phone
        "pt_phone":                 "_pt_phone_num",  # composed into phone
        # Insured
        "ins_name":                 "4_insured_name",
        "ins_street":               "7_insured_address",
        "ins_city":                 "7_insured_city",
        "ins_state":                "7_insured_state",
        "ins_zip":                  "7_insured_zip",
        "ins_phone area":           "_ins_area_code",
        "ins_phone":                "_ins_phone_num",
        # Other insured
        "other_ins_name":           "9_other_insured_name",
        "other_ins_policy":         "9a_other_insured_policy",
        # Policy / plan
        "ins_policy":               "11_insured_policy_group",
        "ins_plan_name":            "11c_insurance_plan_name",
        "other_ins_plan_name":      "_other_ins_plan_name",
        # DOB (composed)
        "birth_mm":                 "_birth_mm",
        "birth_dd":                 "_birth_dd",
        "birth_yy":                 "_birth_yy",
        "ins_dob_mm":               "_ins_dob_mm",
        "ins_dob_dd":               "_ins_dob_dd",
        "ins_dob_yy":               "_ins_dob_yy",
        # Signatures / dates
        "pt_signature":             "12_patient_signature",
        "pt_date":                  "12_signature_date",
        "ins_signature":            "13_insured_signature",
        # Additional
        "96":                       "19_additional_claim_info",
        "charge":                   "20_charges",
        # Diagnosis
        "diagnosis1":               "21_diagnosis_a",
        "diagnosis2":               "21_diagnosis_b",
        "diagnosis3":               "_diagnosis_c",
        "diagnosis4":               "_diagnosis_d",
        "prior_auth":               "_prior_auth",
        # Bottom fields (these were broken by bad bboxes before)
        "tax_id":                   "25_federal_tax_id",
        "pt_account":               "26_patient_account",
        "t_charge":                 "28_total_charge",
        "amt_paid":                 "29_amount_paid",
        "physician_signature":      "31_physician_signature",
        "physician_date":           "_physician_date",
        # Service facility
        "fac_name":                 "32_service_facility_name",
        "fac_street":               "32_service_facility_address",
        "fac_location":             "_fac_location",
        "pin1":                     "32a_npi",
        # Billing provider
        "doc_name":                 "33_billing_provider_name",
        "doc_street":               "33_billing_provider_address",
        "doc_location":             "_doc_location",
        "doc_phone area":           "_doc_phone_area",
        "doc_phone":                "_doc_phone_num",
        "pin":                      "33a_npi",
    }

    # Widget names for radio/checkbox groups mapped to schema
    WIDGET_CHECKBOX_MAP: Dict[str, str] = {
        "sex":           "3_patient_sex",
        "ins_sex":       "11a_insured_sex",
        "insurance_type": "_insurance_type",
        "rel_to_ins":    "_rel_to_insured",
        "employment":    "_employment_related",
        "pt_auto_accident": "_auto_accident",
        "other_accident":   "_other_accident",
        "assignment":    "27_accept_assignment",
        "lab":           "20_outside_lab",
        "ssn":           "_ssn_ein",
    }

    # Service-line widget prefix patterns (lines 1-6)
    SVC_LINE_FIELDS = [
        "sv{n}_mm_from", "sv{n}_dd_from", "sv{n}_yy_from",
        "sv{n}_mm_end",  "sv{n}_dd_end",  "sv{n}_yy_end",
        "place{n}", "type{n}", "cpt{n}",
        "mod{n}", "mod{n}a", "mod{n}b", "mod{n}c",
        "diag{n}", "ch{n}", "day{n}", "local{n}",
    ]
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize agents
        self.form_id_agent = FormIdentificationAgent()
        self.alignment_agent = TemplateAlignmentAgent()
        self.layout_agent = LayoutDetectionAgent(self.config)
        self.ocr_agent = OCRAgent(self.config)
        self.labeling_agent = LabelingAgent(self.config)
        self.validation_agent = ValidationAgent(self.config)
        self._template_word_blacklist: Dict[str, set] = {}

    # ──────────────────────────────────────────────────────────────────
    # LANE A: AcroForm widget extraction (highest accuracy for fillable PDFs)
    # ──────────────────────────────────────────────────────────────────

    def _extract_widgets(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Try to extract AcroForm widget values from a PDF.
        Returns None if the PDF has no widgets or too few filled values.
        Returns a dict of {widget_field_name: field_value} for all non-empty widgets.
        """
        if not str(path).lower().endswith(".pdf"):
            return None
        try:
            import fitz
            doc = fitz.open(str(path))
            page = doc[0]
            widgets = list(page.widgets())
            if len(widgets) < 10:
                doc.close()
                return None

            raw: Dict[str, Any] = {}
            # Track all values + rects for bbox generation
            widget_data: List[Dict[str, Any]] = []
            for w in widgets:
                fn = str(getattr(w, "field_name", "") or "").strip()
                fv = str(getattr(w, "field_value", "") or "").strip()
                ft = getattr(w, "field_type", -1)
                r = getattr(w, "rect", None)
                if not fn:
                    continue
                widget_data.append({
                    "name": fn, "value": fv, "type": ft,
                    "rect": r,
                    "page_w": page.rect.width, "page_h": page.rect.height,
                })
                # For radio/checkbox groups, only store if value is non-empty
                # and keep the first non-empty value per group name
                if ft == 2:  # radio/checkbox
                    if fv and fn not in raw:
                        raw[fn] = fv
                elif ft == 7:  # text
                    if fv:
                        # Some widget names repeat (e.g. insurance_type); keep first non-empty
                        if fn not in raw:
                            raw[fn] = fv
                elif ft == 1:  # pushbutton
                    pass
                else:
                    if fv and fn not in raw:
                        raw[fn] = fv

            doc.close()

            # Only proceed if we got a meaningful number of filled fields
            filled = sum(1 for v in raw.values() if v)
            if filled < 5:
                print(f"[Lane A] Only {filled} filled widgets — too few, skipping widget path")
                return None

            print(f"[Lane A] Extracted {filled} filled widget values from {len(widgets)} total widgets")
            return {"raw": raw, "widget_data": widget_data, "total_widgets": len(widgets), "filled": filled}
        except Exception as e:
            print(f"[Lane A] Widget extraction failed: {e}")
            return None

    def _map_widgets_to_schema(
        self,
        widget_info: Dict[str, Any],
        form_type: Optional[FormType] = None
    ) -> Tuple[Dict[str, str], List[DetectedBlock]]:
        """
        Map extracted widget values to schema field IDs.
        Returns (extracted_fields dict, list of DetectedBlocks for UI overlay).
        """
        if form_type is None:
            form_type = FormType.CMS1500
        if form_type == FormType.UB04:
            return self._map_widgets_to_schema_ub04(widget_info)

        raw = widget_info["raw"]
        widget_data = widget_info.get("widget_data", [])

        # Load schema labels + field types for block typing (Lane A UI consistency)
        schema_info: Dict[str, Dict[str, str]] = {}
        try:
            import json as _json
            schema_path = Path(__file__).parent.parent.parent / "data" / "schemas" / "cms-1500.json"
            if schema_path.exists():
                with open(schema_path) as _f:
                    _schema = _json.load(_f)
                for _field in _schema.get("fields", []):
                    fid = _field.get("id")
                    if fid:
                        schema_info[fid] = {
                            "label": _field.get("label", fid),
                            "field_type": _field.get("field_type", "text")
                        }
        except Exception:
            schema_info = {}

        # Build name→rect lookup (for UI bboxes)
        name_to_rect: Dict[str, Any] = {}
        page_w = 612.0
        page_h = 792.0
        for wd in widget_data:
            n = wd["name"]
            if n not in name_to_rect and wd.get("rect"):
                name_to_rect[n] = wd["rect"]
                page_w = wd.get("page_w", 612.0)
                page_h = wd.get("page_h", 792.0)

        extracted: Dict[str, str] = {}
        blocks: List[DetectedBlock] = []

        def _normalize_widget_value(v: str) -> str:
            """Clean up common widget value artifacts (extra internal spaces around hyphens, etc.)."""
            import re
            v = v.strip()
            # Collapse spaces around hyphens: "ID10- 45678" → "ID10-45678"
            v = re.sub(r'\s*-\s*', '-', v)
            # Collapse double spaces
            v = re.sub(r' {2,}', ' ', v)
            return v

        # ── Map simple text fields ──
        for widget_name, schema_id in self.WIDGET_TO_SCHEMA.items():
            val = raw.get(widget_name, "")
            if val and not schema_id.startswith("_"):
                extracted[schema_id] = _normalize_widget_value(val)

        # ── Compose multi-part fields ──
        # Patient DOB
        mm = raw.get("birth_mm", "")
        dd = raw.get("birth_dd", "")
        yy = raw.get("birth_yy", "")
        if mm and dd and yy:
            extracted["3_patient_dob"] = f"{mm}/{dd}/{yy}"

        # Insured DOB
        imm = raw.get("ins_dob_mm", "")
        idd = raw.get("ins_dob_dd", "")
        iyy = raw.get("ins_dob_yy", "")
        if imm and idd and iyy:
            extracted["11a_insured_dob"] = f"{imm}/{idd}/{iyy}"

        # Patient phone
        area = raw.get("pt_AreaCode", "")
        num = raw.get("pt_phone", "")
        if area and num:
            extracted["5_patient_phone"] = f"({area}) {num}"
        elif num:
            extracted["5_patient_phone"] = num

        # Insured phone
        iarea = raw.get("ins_phone area", "")
        inum = raw.get("ins_phone", "")
        if iarea and inum:
            extracted["7_insured_phone"] = f"({iarea}) {inum}"
        elif inum:
            extracted["7_insured_phone"] = inum

        # Billing provider phone
        darea = raw.get("doc_phone area", "")
        dnum = raw.get("doc_phone", "")
        if darea and dnum:
            extracted["33_billing_provider_phone"] = f"({darea}) {dnum}"
        elif dnum:
            extracted["33_billing_provider_phone"] = dnum

        # Header notes (compose from insurance name/address/city)
        ins_parts = [raw.get(k, "") for k in ["insurance_name", "insurance_address", "insurance_address2", "insurance_city_state_zip"]]
        ins_header = " ".join(p for p in ins_parts if p).strip()
        if ins_header:
            extracted["header_top_right_notes"] = ins_header

        # ── Checkboxes ──
        for widget_name, schema_id in self.WIDGET_CHECKBOX_MAP.items():
            val = raw.get(widget_name, "")
            if val and not schema_id.startswith("_"):
                extracted[schema_id] = val

        # Patient sex: widget stores "M" or "F" as the radio value
        sex_val = raw.get("sex", "")
        if sex_val:
            extracted["3_patient_sex"] = sex_val
            if sex_val.upper() == "M":
                extracted["3_patient_sex_m"] = "X"
            elif sex_val.upper() == "F":
                extracted["3_patient_sex_f"] = "X"

        # Insured sex
        isex = raw.get("ins_sex", "")
        if isex:
            extracted["11a_insured_sex"] = isex

        # ── Service lines (compose into a summary) ──
        svc_lines = []
        for n in range(1, 7):
            mm_from = raw.get(f"sv{n}_mm_from", "")
            if not mm_from:
                continue  # no more lines
            parts = []
            # Date from
            df = "/".join(filter(None, [raw.get(f"sv{n}_mm_from"), raw.get(f"sv{n}_dd_from"), raw.get(f"sv{n}_yy_from")]))
            dt = "/".join(filter(None, [raw.get(f"sv{n}_mm_end"), raw.get(f"sv{n}_dd_end"), raw.get(f"sv{n}_yy_end")]))
            cpt = raw.get(f"cpt{n}", "")
            mod = raw.get(f"mod{n}", "")
            diag = raw.get(f"diag{n}", "")
            ch = raw.get(f"ch{n}", "")
            line = f"{df}-{dt} {cpt} {mod} {diag} ${ch}".strip()
            svc_lines.append(line)
        if svc_lines:
            extracted["24_service_lines"] = " | ".join(svc_lines)

        # ── Build DetectedBlocks for UI overlay ──
        zoom = 300.0 / 72.0

        def _rect_to_bbox(r) -> Tuple[float, float, float, float]:
            return (r.x0 * zoom, r.y0 * zoom, r.x1 * zoom, r.y1 * zoom)

        def _combine_widget_rects(widget_names: list) -> Optional[Tuple[float, float, float, float]]:
            """Combine multiple widget rects into one encompassing bbox."""
            rects = [name_to_rect[n] for n in widget_names if n in name_to_rect]
            if not rects:
                return None
            x0 = min(r.x0 for r in rects) * zoom
            y0 = min(r.y0 for r in rects) * zoom
            x1 = max(r.x1 for r in rects) * zoom
            y1 = max(r.y1 for r in rects) * zoom
            return (x0, y0, x1, y1)

        # Pre-build schema_id → bbox for composed fields
        composed_bboxes: Dict[str, Tuple[float, float, float, float]] = {}

        # DOB fields: combine mm+dd+yy rects
        dob_bbox = _combine_widget_rects(["birth_mm", "birth_dd", "birth_yy"])
        if dob_bbox:
            composed_bboxes["3_patient_dob"] = dob_bbox
        idob_bbox = _combine_widget_rects(["ins_dob_mm", "ins_dob_dd", "ins_dob_yy"])
        if idob_bbox:
            composed_bboxes["11a_insured_dob"] = idob_bbox

        # Phone fields: combine area + number
        for sid, widgets in [
            ("5_patient_phone", ["pt_AreaCode", "pt_phone"]),
            ("7_insured_phone", ["ins_phone area", "ins_phone"]),
            ("33_billing_provider_phone", ["doc_phone area", "doc_phone"]),
        ]:
            b = _combine_widget_rects(widgets)
            if b:
                composed_bboxes[sid] = b

        # Header: combine insurance name/address/city
        hdr_bbox = _combine_widget_rects(["insurance_name", "insurance_address", "insurance_address2", "insurance_city_state_zip"])
        if hdr_bbox:
            composed_bboxes["header_top_right_notes"] = hdr_bbox

        # Service lines: combine all service line widgets for lines 1-6
        svc_widgets = []
        for n in range(1, 7):
            svc_widgets.extend([f"sv{n}_mm_from", f"local{n}"])
        svc_bbox = _combine_widget_rects(svc_widgets)
        if svc_bbox:
            composed_bboxes["24_service_lines"] = svc_bbox

        # Sex checkboxes: combine both options
        sex_bbox = _combine_widget_rects(["sex"])  # radio widget covers both
        if "sex" in name_to_rect:
            composed_bboxes["3_patient_sex"] = _rect_to_bbox(name_to_rect["sex"])
            composed_bboxes["3_patient_sex_m"] = _rect_to_bbox(name_to_rect["sex"])

        for schema_id, value in extracted.items():
            if not value:
                continue

            # 1. Check composed bboxes first
            bbox = composed_bboxes.get(schema_id)

            # 2. Check direct widget mapping
            if bbox is None:
                for wn, sid in self.WIDGET_TO_SCHEMA.items():
                    if sid == schema_id and wn in name_to_rect:
                        bbox = _rect_to_bbox(name_to_rect[wn])
                        break

            # 3. Check checkbox mapping
            if bbox is None:
                for wn, sid in self.WIDGET_CHECKBOX_MAP.items():
                    if sid == schema_id and wn in name_to_rect:
                        bbox = _rect_to_bbox(name_to_rect[wn])
                        break

            # 4. Try to find from schema JSON as last resort
            if bbox is None:
                try:
                    import json as _json
                    schema_path = Path(__file__).parent.parent.parent / "data" / "schemas" / "cms-1500.json"
                    if schema_path.exists():
                        with open(schema_path) as _f:
                            _schema = _json.load(_f)
                        for _field in _schema.get("fields", []):
                            if _field.get("id") == schema_id:
                                bn = _field.get("bbox_norm", [])
                                if len(bn) == 4:
                                    pw_px = page_w * zoom
                                    ph_px = page_h * zoom
                                    bbox = (bn[0]*pw_px, bn[1]*ph_px, bn[2]*pw_px, bn[3]*ph_px)
                                break
                except Exception:
                    pass

            if bbox is None:
                bbox = (0.0, 0.0, 1.0, 1.0)

            # Schema label + field type (for consistent UI labeling with Lane B)
            info = schema_info.get(schema_id, {})
            label = info.get("label", schema_id)
            field_type = info.get("field_type", "text")

            # Infer block type (same logic as Lane B zone matching)
            label_l = (label or "").lower()
            field_id_l = (schema_id or "").lower()
            if field_type == "checkbox":
                block_type = BlockType.CHECKBOX
            elif field_type == "signature" or "signature" in field_id_l or "signature" in label_l:
                block_type = BlockType.SIGNATURE
            elif "table" in field_id_l or "table" in label_l:
                block_type = BlockType.TABLE
            elif "figure" in field_id_l or "figure" in label_l or "image" in label_l or "graphic" in label_l:
                block_type = BlockType.FIGURE
            elif "title" in field_id_l or "title" in label_l:
                block_type = BlockType.TITLE
            elif "header" in field_id_l or "header" in label_l:
                block_type = BlockType.HEADER
            elif "footer" in field_id_l or "footer" in label_l:
                block_type = BlockType.FOOTER
            elif ("page" in label_l and "number" in label_l) or "page_num" in field_id_l or "pagenum" in field_id_l:
                block_type = BlockType.PAGE_NUM
            elif "list" in field_id_l or "list" in label_l:
                block_type = BlockType.LIST
            else:
                block_type = BlockType.FORM_FIELD

            blocks.append(DetectedBlock(
                id=schema_id,
                block_type=block_type,
                bbox=bbox,
                text=value,
                confidence=0.99,  # widget values are ground truth
                metadata={
                    "source": "acroform_widget",
                    "label": label,
                    "semantic_label": label,
                    "field_type": field_type,
                    "ocr_engine": "widget",
                }
            ))

        print(f"[Lane A] Mapped {len(extracted)} schema fields from widgets")
        return extracted, blocks

    # ── UB-04 Widget name → schema field ID mapping ──────────────────────────
    # Built from actual UB-04 XFA widget names found in fillable PDFs.
    UB04_WIDGET_TO_SCHEMA: Dict[str, str] = {
        # Provider info (FL 1)
        "Address1": "fl1_provider_name",
        "address2": "fl1_provider_address1",
        "address3": "fl1_provider_city_state_zip",
        "address4": "_provider_phone",
        
        # Patient control / bill type (FL 3-4)
        "patctrl": "fl3a_patient_control",
        "typebill": "fl4_type_of_bill",
        
        # Tax ID and dates (FL 5-6)
        "provtaxID": "fl5_federal_tax_id",
        "fromdte": "fl6_from_date",
        "thrudate": "fl6_thru_date",
        
        # Patient info (FL 8-11)
        "patientIDnum": "fl8a_patient_id",
        "pataddrstreet": "fl9_patient_address",
        "pataddresscity": "fl9_patient_city",
        "pataddrState": "fl9_patient_state",
        "pataddresszip": "fl9_patient_zip",
        "DOB": "fl10_patient_dob",
        "sex": "fl11_patient_sex",
        
        # Admission info (FL 12-17)
        "lrd": "fl12_admission_date",
        "17date": "fl13_admission_hour",
        "18type": "fl14_admission_type",
        "19source": "fl15_admission_source",
        "20src": "fl16_discharge_hour",
        "21dhr": "fl17_patient_status",
        
        # Occurrence codes (FL 31-34)
        "32_.1": "fl31_occurrence_code_1",
        "32occ_.1": "fl31_occurrence_date_1",
        "33_.1": "fl32_occurrence_code_2",
        "33code_.1": "fl32_occurrence_date_2",
        "34_.1": "fl33_occurrence_code_3",
        "34code_.1": "fl33_occurrence_date_3",
        "35_.1": "fl34_occurrence_code_4",
        "35code_.1": "fl34_occurrence_date_4",
        "32_.2": "_occurrence_code_5",
        "32occ_.2": "_occurrence_date_5",
        "33_.2": "_occurrence_code_6",
        "33code_.2": "_occurrence_date_6",
        "34_.2": "_occurrence_code_7",
        "34code_.2": "_occurrence_date_7",
        "35_.2": "_occurrence_code_8",
        "35code_.2": "_occurrence_date_8",
        
        # Occurrence span codes (FL 35-36)
        "36_.1": "fl35_occurrence_span_code_1",
        "37_.3_.0_.0_.0": "_occurrence_span_code_2",
        "37_.3_.0_.1": "fl35_occurrence_span_from_1",
        "37_.3_.0_.1_1": "_occurrence_span_from_2",
        "37_.3_.0_.1_2": "_occurrence_span_from_3",
        "37_.3_.1_.1": "fl35_occurrence_span_thru_1",
        
        # Responsible party (FL 38)
        "38_.1": "fl38_responsible_party",
        
        # Value codes (FL 39-41)
        "39code_.1": "fl39a_value_code_1",
        "val_.1": "fl39a_value_amount_1",
        "39code_.2": "fl39b_value_code_2",
        "val_.1_1": "fl39b_value_amount_2",
        "39code_.3": "fl39c_value_code_3",
        "val_.1_2": "fl39c_value_amount_3",
        "39code_.4": "fl39d_value_code_4",
        "val_.1_3": "fl39d_value_amount_4",
        
        # Revenue codes and charges (FL 42-47) - Line 1
        "revcd42_.1": "fl42_revenue_code_1",
        "43desc_.1": "fl43_description_1",
        "44hcps_.1": "fl44_hcpcs_1",
        "45servdate_.1": "fl45_service_date_1",
        "46servunits_.1": "fl46_units_1",
        "47totalcharges_.1": "fl47_charges_1",
        
        # Revenue codes and charges - Line 2
        "revcd42_.2": "fl42_revenue_code_2",
        "43desc_.2": "fl43_description_2",
        "44hcps_.2": "fl44_hcpcs_2",
        "45servdate_.2": "fl45_service_date_2",
        "46servunits_.2": "fl46_units_2",
        "47totalcharges_.2": "fl47_charges_2",
        
        # Total charges (FL 47 totals row)
        "47totalcharges_.23": "fl47_total_charges",
        "revcd42_.22_.1": "_revenue_totals_code",
        
        # Payer info (FL 50-56)
        "50payer_.1": "fl50_payer_name_a",
        "51providernum_.1": "fl51_health_plan_id_a",
        "52relinfo_.1": "fl52_release_info_a",
        "52asgben_.1": "fl53_assignment_a",
        "54prior_.1": "fl54_prior_payments_a",
        "55est_.1": "fl55_estimated_due_a",
        "56_.1": "fl56_npi_a",
        
        # Insured info (FL 58-62)
        "57": "fl58_insured_name_a",
        "59prel_.1": "fl59_patient_rel_a",
        "60cert_.1": "fl60_insured_id_a",
        "61groupname_.1": "fl61_group_name_a",
        "62insgroup_.1": "fl62_group_number_a",
        
        # Treatment / employer (FL 63-66)
        "63treatment_.1": "fl63_treatment_auth_a",
        "65empname_.1": "fl65_employer_name_a",
        "66emploc_.1": "fl66_employer_loc_a",
        
        # Diagnosis codes (FL 67-72)
        "67prin": "fl67_principal_diagnosis",
        "68code": "fl67a_diagnosis_a",
        "69code": "fl67b_diagnosis_b",
        "71code": "fl71_pps_code",
        "72code": "fl72_eci_code",
        "73code": "fl67c_diagnosis_c",
        "75code": "fl67d_diagnosis_d",
        "76admdiag": "fl69_admitting_diagnosis",
        "77ecode": "fl70_patient_reason",
        "79pc": "fl79_admitting_dx_code",
        
        # Procedure codes (FL 74)
        "princode_.0": "fl74_principal_procedure",
        "prindate": "fl74_principal_proc_date",
        "other2": "fl74a_other_procedure_1",
        "other3": "fl74a_other_proc_date_1",
        "other3date": "_other_proc_date_2",
        "74code_.0": "fl74_principal_procedure",
        "74code_.1_.0_.0_.1": "_other_procedure_4",
        
        # Principal code dates
        "princode_.1_.1": "fl74_principal_proc_date",
        "princode_.0_.1_.0_.1": "_proc_date_2",
        "princode_.1_.0_.0": "_proc_code_2",
        "princode_.0_.1_.0": "_proc_code_3",
        "princode_.0_.1_.1_.0_.0": "_proc_code_4",
        "princode_.0_.1_.1_.0_.0_.1": "_proc_date_4",
        "princode_.1_.0_.1": "_proc_code_5",
        "princode_.1_.0_.0_.1": "_proc_code_6",
        "princode_.0_.1_.1_.0_.1": "_proc_code_7",
        "princode_.0_.1_.1_.1_.0": "_proc_date_7",
        "princode_.1_.0_.1_.1": "_proc_date_5",
        "princode_.0_.1_.1_.0_.1_.1": "_proc_date_8",
        "princode_.0_.1_.1_.1_.1_.0_.0": "_proc_code_taxonomy",
        "princode_.0_.1_.1_.1_.0_.1_.0": "_proc_taxonomy_code",
        
        # Attending physician (FL 76)
        "NPI_.0": "fl76_attending_npi",
        "82attend_.0": "fl76_attending_qual",
        "NPI_.1_.0_.0": "fl76_attending_last",
        "NPI_.1_.1": "fl76_attending_first",
        
        # Operating physician (FL 77)
        "NPI_.1_.0_.1_.1": "fl77_operating_npi",
        "NPI_.1_.0_.1_.0_.0": "fl77_operating_last",
        "NPI_.1_.0_.0_.1": "fl77_operating_first",
        
        # Other provider (FL 78-79)
        "NPI_.1_.0_.1_.0_.1": "fl78_other_npi_1",
        "NPI_.1_.0_.1_.0_.0_.1": "fl78_other_first_1",
        
        # Remarks (FL 80)
        "remarks": "fl80_remarks",
        
        # Attending name (FL 82-83)
        "83id": "_attending_id",
    }

    def _map_widgets_to_schema_ub04(self, widget_info: Dict[str, Any]) -> Tuple[Dict[str, str], List[DetectedBlock]]:
        """
        Map UB-04 widget values to schema fields using explicit name mapping + spatial fallback.
        This supports fillable UB-04 PDFs with XFA-style widget names.
        """
        import re
        
        raw = widget_info.get("raw", {}) or {}
        widget_data = widget_info.get("widget_data", []) or []
        if not widget_data:
            return {}, []

        # Build name→rect lookup
        name_to_rect: Dict[str, Any] = {}
        page_w = 612.0
        page_h = 792.0
        for wd in widget_data:
            n = wd.get("name")
            if n and n not in name_to_rect and wd.get("rect"):
                name_to_rect[n] = wd["rect"]
                page_w = wd.get("page_w", page_w)
                page_h = wd.get("page_h", page_h)

        # Normalize XFA widget names to simpler form
        def _normalize_widget_name(raw_name: str) -> str:
            """Simplify XFA widget name to a clean identifier."""
            name = raw_name
            prefixes = [
                "topmostSubform[0].Page1[0].topmostSubform_0_\\.Page1_0_\\.",
                "topmostSubform[0].Page1[0]."
            ]
            for p in prefixes:
                if name.startswith(p):
                    name = name[len(p):]
                    break
            name = re.sub(r'\[0\]', '', name)
            name = name.replace('\\', '')
            name = re.sub(r'_0_$', '', name)
            name = re.sub(r'\.0$', '', name)
            return name.strip('_.')

        # Load UB-04 schema for metadata
        schema_fields = []
        schema_info: Dict[str, Dict[str, str]] = {}
        try:
            import json as _json
            schema_path = Path(__file__).parent.parent.parent / "data" / "schemas" / "ub-04.json"
            if schema_path.exists():
                with open(schema_path) as _f:
                    schema = _json.load(_f)
                for field in schema.get("fields", []):
                    fid = field.get("id")
                    if fid:
                        schema_fields.append(field)
                        schema_info[fid] = {
                            "label": field.get("label", fid),
                            "field_type": field.get("field_type", "text"),
                            "bbox_norm": field.get("bbox_norm"),
                            "business_key": field.get("business_key")
                        }
        except Exception:
            pass

        def _normalize_widget_value(v: str) -> str:
            v = str(v).strip()
            v = re.sub(r"\s*-\s*", "-", v)
            v = re.sub(r" {2,}", " ", v)
            return v

        def _bbox_iou(a, b) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            inter_x0 = max(ax0, bx0)
            inter_y0 = max(ay0, by0)
            inter_x1 = min(ax1, bx1)
            inter_y1 = min(ay1, by1)
            inter_w = max(0.0, inter_x1 - inter_x0)
            inter_h = max(0.0, inter_y1 - inter_y0)
            inter = inter_w * inter_h
            area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
            area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
            denom = area_a + area_b - inter
            return inter / denom if denom > 0 else 0.0

        # Process widgets and map to schema
        extracted: Dict[str, str] = {}
        blocks: List[DetectedBlock] = []
        zoom = 300.0 / 72.0
        assigned_fields: set = set()

        def _rect_to_bbox(r) -> Tuple[float, float, float, float]:
            return (r.x0 * zoom, r.y0 * zoom, r.x1 * zoom, r.y1 * zoom)

        # First pass: direct name mapping
        for wd in widget_data:
            raw_name = wd.get("name") or ""
            rect = wd.get("rect")
            value = raw.get(raw_name, "")
            if not value or str(value).strip() == "" or value in ("Off", "off"):
                continue

            norm_name = _normalize_widget_name(raw_name)
            schema_id = self.UB04_WIDGET_TO_SCHEMA.get(norm_name)
            
            # Skip internal fields (starting with _)
            if schema_id and schema_id.startswith("_"):
                schema_id = None
            
            if schema_id and schema_id not in assigned_fields:
                assigned_fields.add(schema_id)
                clean_value = _normalize_widget_value(value)
                extracted[schema_id] = clean_value
                
                meta = schema_info.get(schema_id, {})
                label = meta.get("label", schema_id)
                field_type = meta.get("field_type", "text")
                schema_block_type = meta.get("block_type", "form_field")
                
                bbox = _rect_to_bbox(rect) if rect else (0.0, 0.0, 1.0, 1.0)
                
                # Map schema block_type to BlockType enum
                block_type_map = {
                    "header": BlockType.HEADER,
                    "table_cell": BlockType.TABLE,
                    "table": BlockType.TABLE,
                    "checkbox": BlockType.CHECKBOX,
                    "signature": BlockType.SIGNATURE,
                    "form_field": BlockType.FORM_FIELD,
                }
                block_type = block_type_map.get(schema_block_type, BlockType.FORM_FIELD)
                
                # Override with field_type if checkbox
                if field_type == "checkbox":
                    block_type = BlockType.CHECKBOX

                blocks.append(DetectedBlock(
                    id=schema_id,
                    block_type=block_type,
                    bbox=bbox,
                    text=clean_value,
                    confidence=0.99,
                    metadata={
                        "source": "acroform_widget",
                        "label": label,
                        "semantic_label": label,
                        "field_type": field_type,
                        "block_type": schema_block_type,
                        "ocr_engine": "widget",
                        "widget_name": norm_name,
                        "business_key": meta.get("business_key")
                    }
                ))

        # Second pass: spatial matching for unmapped widgets with values
        for wd in widget_data:
            raw_name = wd.get("name") or ""
            rect = wd.get("rect")
            value = raw.get(raw_name, "")
            if not value or str(value).strip() == "" or value in ("Off", "off"):
                continue
            if not rect:
                continue

            norm_name = _normalize_widget_name(raw_name)
            # Skip if already mapped
            if self.UB04_WIDGET_TO_SCHEMA.get(norm_name):
                continue

            # Spatial matching
            wx0, wy0, wx1, wy1 = rect.x0 / page_w, rect.y0 / page_h, rect.x1 / page_w, rect.y1 / page_h
            cx = (wx0 + wx1) / 2.0
            cy = (wy0 + wy1) / 2.0

            best_id = None
            best_score = 0.0

            for field in schema_fields:
                fid = field.get("id")
                bbox_norm = field.get("bbox_norm") or []
                if not fid or fid in assigned_fields or len(bbox_norm) != 4:
                    continue
                fx0, fy0, fx1, fy1 = bbox_norm
                inside = fx0 <= cx <= fx1 and fy0 <= cy <= fy1
                iou = _bbox_iou((wx0, wy0, wx1, wy1), (fx0, fy0, fx1, fy1))
                score = (1.0 + iou) if inside else iou
                if score > best_score:
                    best_score = score
                    best_id = fid

            if best_id and best_score >= 0.1:
                assigned_fields.add(best_id)
                clean_value = _normalize_widget_value(value)
                extracted[best_id] = clean_value
                
                meta = schema_info.get(best_id, {})
                label = meta.get("label", best_id)
                field_type = meta.get("field_type", "text")
                bbox = _rect_to_bbox(rect)

                blocks.append(DetectedBlock(
                    id=best_id,
                    block_type=BlockType.FORM_FIELD,
                    bbox=bbox,
                    text=clean_value,
                    confidence=0.95,
                    metadata={
                        "source": "acroform_widget_spatial",
                        "label": label,
                        "semantic_label": label,
                        "field_type": field_type,
                        "ocr_engine": "widget",
                        "widget_name": norm_name,
                        "match_score": best_score,
                        "business_key": meta.get("business_key")
                    }
                ))

        print(f"[Lane A] Mapped {len(extracted)} UB-04 fields from widgets")
        return extracted, blocks

    # Comprehensive CMS-1500 template word blacklist.
    # Every pre-printed word on the standard CMS-1500 (02/12) form.
    # Used to filter template text from OCR results AFTER recognition,
    # so handwriting quality is preserved (no red removal needed).
    _CMS1500_TEMPLATE_WORDS = {
        # Form title and header
        "health", "insurance", "claim", "form", "approved", "national",
        "uniform", "committee", "nucc", "pica", "02/12", "02/2",
        # Insurance type labels
        "medicare", "medicaid", "tricare", "champva", "group", "feca",
        "blk", "lung", "other",
        # Field labels — patient info
        "patient's", "patients", "patient", "name", "last", "first",
        "middle", "initial", "birth", "date", "sex", "address",
        "city", "state", "zip", "code", "telephone", "include",
        "area", "relationship", "insured", "insured's", "insureds",
        "self", "spouse", "child",
        # Field labels — other insured
        "other", "policy", "number", "group", "feca",
        "employment", "current", "previous", "auto", "accident",
        "place", "reserved", "nucc", "use",
        # Field labels — insurance
        "plan", "program", "another", "benefit", "complete",
        "items", "designated",
        # Field labels — signatures
        "signature", "authorize", "release", "medical", "information",
        "necessary", "process", "request", "payment", "government",
        "benefits", "myself", "party", "accepts", "assignment",
        "below", "signed", "read", "back", "before", "completing",
        "signing",
        # Field labels — dates and medical
        "illness", "injury", "pregnancy", "lmp", "qual",
        "dates", "unable", "work", "occupation", "from",
        "referring", "provider", "source", "hospitalization",
        "services", "related", "additional", "information",
        "outside", "lab", "charges", "diagnosis", "nature",
        "relate", "service", "line", "icd", "ind",
        "resubmission", "original", "ref", "prior",
        "authorization",
        # Field labels — service lines
        "procedures", "supplies", "explain", "unusual",
        "circumstances", "cpt", "hcpcs", "modifier",
        "pointer", "days", "units", "epsdt", "family",
        "qual", "rendering", "npi",
        # Field labels — bottom section
        "federal", "tax", "i.d.", "i.d", "ssn", "ein",
        "account", "accept", "total", "charge", "amount",
        "paid", "rsvd", "physician", "supplier", "degrees",
        "credentials", "certify", "statements", "reverse",
        "apply", "bill", "made", "part", "thereof",
        "facility", "location", "billing", "info",
        # Instructions and misc
        "no.", "street", "print", "type", "please",
        "instruction", "manual", "available", "www.nucc.org",
        "omb-0938-1197", "1500",
        # Field numbers (OCR may read these)
        "1a.", "1a", "2.", "3.", "4.", "5.", "6.", "7.", "8.",
        "9.", "9a.", "9d.", "10.", "10a.", "10b.", "10c.", "10d.",
        "11.", "11a.", "11b.", "11c.", "11d.", "12.", "13.",
        "14.", "15.", "16.", "17.", "17a.", "17b.", "18.",
        "19.", "20.", "21.", "22.", "23.", "24.", "25.",
        "26.", "27.", "28.", "29.", "30.", "31.", "32.",
        "32a.", "32b.", "33.", "33a.", "33b.",
        # Common OCR misreads of template labels
        "patent's", "patents", "patent", "nsured's", "nsured",
        "nsurance", "atient", "atient's", "ddress", "elephone",
        "ignature", "hysician", "iagnosis", "rocedures",
        "ederal", "illing", "acility", "ertify",
        # Parenthetical instructions
        "(last", "(first", "(middle", "(no.,", "(include",
        "(designated", "(current", "(for", "program",
        "item", "(medicare#)", "(medicaid#)", "(id#/dod#)",
        "(member", "id#)", "(id#)",
    }
    
    def _get_template_word_blacklist(self, template_key: str) -> set:
        """
        Return comprehensive blacklist of pre-printed template words.
        
        For CMS-1500: returns a hardcoded set of ALL words printed on the
        standard form. This is more reliable than OCR-ing the template image
        because it includes common OCR misreads of template labels.
        
        Used in the OCR-first-clean-after pipeline: PaddleOCR reads the
        raw aligned image (with red template visible for max handwriting
        quality), then template words are filtered from the results.
        """
        key = (template_key or "").lower().strip()
        if not key:
            return set()
        if key in self._template_word_blacklist:
            return self._template_word_blacklist[key]
        
        if key in {"cms-1500", "cms1500"}:
            # Use the hardcoded comprehensive blacklist
            # Also add dynamically OCR'd template words as supplement
            words = set(self._CMS1500_TEMPLATE_WORDS)
            try:
                from src.processing.registration import load_and_process_reference
                from src.ocr.paddle_ocr import PaddleOCRWrapper
                ref_data = load_and_process_reference(key)
                if ref_data and ref_data.get("image") is not None:
                    paddle = PaddleOCRWrapper()
                    word_boxes = paddle.extract_text(ref_data["image"])
                    for wb in word_boxes or []:
                        text = (wb.text or "").strip().lower()
                        if len(text) >= 2:
                            words.add(text)
                    print(f"[Pipeline] CMS-1500 blacklist: {len(words)} words (hardcoded + OCR'd template)")
            except Exception:
                print(f"[Pipeline] CMS-1500 blacklist: {len(words)} words (hardcoded only)")
            self._template_word_blacklist[key] = words
            return words
        
        # Other templates: OCR-based blacklist (fallback)
        try:
            from src.processing.registration import load_and_process_reference
            from src.ocr.paddle_ocr import PaddleOCRWrapper
            ref_data = load_and_process_reference(key)
            if not ref_data or ref_data.get("image") is None:
                return set()
            paddle = PaddleOCRWrapper()
            word_boxes = paddle.extract_text(ref_data["image"])
            words = set()
            for wb in word_boxes or []:
                text = (wb.text or "").strip().lower()
                if len(text) >= 3:
                    words.add(text)
            self._template_word_blacklist[key] = words
            return words
        except Exception:
            return set()

    def _extract_pdf_digital_words(self, page, zoom: float) -> Optional[List[Any]]:
        """Extract word-level boxes from a PDF text layer and scale into rendered pixel space."""
        try:
            from utils.models import WordBox
            words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,word_no (PDF points)
            if not words or len(words) < 30:
                return None
            scaled = []
            for w in words:
                if len(w) < 5:
                    continue
                x0, y0, x1, y1, text = w[:5]
                text = str(text or "").strip()
                if len(text) < 1:
                    continue
                sx0 = float(x0) * zoom
                sy0 = float(y0) * zoom
                sx1 = float(x1) * zoom
                sy1 = float(y1) * zoom
                scaled.append(WordBox(text=text, bbox=(sx0, sy0, sx1, sy1), confidence=1.0))
            return scaled if len(scaled) >= 30 else None
        except Exception:
            return None

    def _digital_layer_matches_visual(self, image_rgb: np.ndarray, word_boxes: List[Any], max_samples: int = 40) -> bool:
        """
        Validate that the PDF text layer matches visible pixels.
        This prevents the "Rahul vs Rohit" bug caused by hidden/incorrect OCR layers.
        """
        if image_rgb is None or image_rgb.size == 0 or not word_boxes:
            return False
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
            h, w = gray.shape[:2]
            # Sample medium-length words (more reliable)
            candidates = [wb for wb in word_boxes if 3 <= len(getattr(wb, "text", "") or "") <= 20]
            if len(candidates) < 15:
                candidates = list(word_boxes)
            # Uniform sampling
            step = max(1, len(candidates) // max_samples)
            samples = candidates[::step][:max_samples]

            ink_scores = []
            for wb in samples:
                x0, y0, x1, y1 = [int(round(v)) for v in wb.bbox]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 <= x0 + 2 or y1 <= y0 + 2:
                    continue
                crop = gray[y0:y1, x0:x1]
                if crop.size < 50:
                    continue
                # Ink-ness: fraction of dark pixels
                thr = int(np.clip(np.median(crop) - 15, 100, 210))
                dark = np.count_nonzero(crop < thr)
                ink = dark / float(crop.size)
                ink_scores.append(float(ink))
            if len(ink_scores) < 10:
                return False
            med = float(np.median(ink_scores))
            # If median ink in word boxes is too low, those words are not actually visible.
            return med >= 0.015
        except Exception:
            return False
    
    def _load_image(self, path: str) -> Tuple[np.ndarray, int, int, Optional[List[Any]]]:
        """Load document image.
        """
        path = Path(path)
        digital_word_boxes = None
        
        if path.suffix.lower() == ".pdf":
            import fitz
            doc = fitz.open(str(path))
            page = doc[0]
            zoom = 300 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            # Extract digital text layer, but only keep it if it matches visible pixels.
            # This prevents using hidden/incorrect layers (the "Rahul vs Rohit" bug).
            maybe_words = self._extract_pdf_digital_words(page, zoom)
            if maybe_words and self._digital_layer_matches_visual(img, maybe_words):
                digital_word_boxes = maybe_words
            
            doc.close()
        else:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img, img.shape[1], img.shape[0], digital_word_boxes
    
    def _get_block_label(self, block: DetectedBlock) -> str:
        """Get display label for a block (shows block TYPE, not field name)."""
        # Priority 1: YOLO class name
        class_name = block.metadata.get("class_name", "")
        if class_name:
            return class_name.upper()
        
        # Priority 2: Block type enum
        btype = block.block_type.value if hasattr(block.block_type, 'value') else str(block.block_type)
        return btype.upper()
    
    # CMS-1500 template label patterns - comprehensive list for stripping
    CMS1500_TEMPLATE_LABELS = {
        # Name field hints (OCR may garble these)
        r"\(?L[AE]ST\s*N[AE]ME[,\s]*F[I1]RST\s*N[AE]ME[,\s]*M[I1]DDL?E?\s*(INITIAL|NAME|BITA|NARE|INIT)?\)?",
        r"\(?LAST[,\s]*FIRST[,\s]*M\.?I\.?\)?",
        r"\(?F[I1]RST[,\s]*M[I1]DDLE[,\s]*LAST\)?",
        r"N[AE]ME\s*\([^)]*\)",  # "NAME (Last, First, Middle)"
        r"N[SU]?[AU]N?[CG]?[OD]?O?\s*NA?ME?",  # OCR garbled "NAME" -> "NSUngOo NAMe"
        
        # Field numbers and labels
        r"^\s*\d{1,2}\s*[a-z]?\s*\.?\s*",  # "1", "1.", "1a", "21a", etc.
        
        # Patient labels (with OCR error tolerance)
        r"P?A?T[I1]?EN?T'?S?\s*(NAME|BIRTH|ADDRESS|SEX|PHONE|RELATIONSHIP|ACCOUNT)",
        r"I?N?S?U?R?E?D'?S?\s*(NAME|I\.?D\.?\s*N|ADDRESS|DATE|POLICY|GROUP|SIGNATURE)",
        r"OTHER\s+INSURED'?S?\s*(NAME|POLICY)",
        
        # Common form labels
        r"INSURANCE\s+PLAN\s+NAME",
        r"EMPLOYER'?S?\s*NAME",
        r"REFERRING\s+PROVIDER",
        r"SIGNATURE\s+OF\s*(PHYSICIAN|PATIENT)?",
        r"BILLING\s+PROVIDER",
        r"SERVICE\s+FACILITY",
        r"DATE\s+OF\s+(CURRENT|BIRTH|SERVICE)",
        r"DIAGNOSIS\s+OR\s+NATURE",
        r"FEDERAL\s+TAX\s+I\.?D\.?",
        r"HEALTH\s+INSURANCE\s+CLAIM",
        r"TOTAL\s+CHARGE",
        r"AMOUNT\s+PAID",
        r"ACCEPT\s+ASSIGNMENT",
        r"OUTSIDE\s+LAB",
        r"ADDITIONAL\s+CLAIM\s+INFO",
        
        # Date format hints
        r"\(?\s*MM\s*[\/-]?\s*DD\s*[\/-]?\s*Y{2,4}\s*\)?",
        r"MM\s*DD\s*YY",
        r"\(?\s*MONTH\s*DAY\s*YEAR\s*\)?",
        
        # Address hints
        r"\(?\s*No\.[,\s]*Street\s*\)?",
        r"\(?\s*INCLUDE\s+AREA\s+CODE\s*\)?",
        r"\(?\s*AREA\s+CODE\s*\)?",
        
        # Checkbox hints
        r"\(?\s*YES\s*/?\s*NO\s*\)?",
        r"\[\s*\]\s*YES",
        r"\[\s*\]\s*NO",
        
        # Program hints
        r"OR\s+PROGR?A?M\s+NAME",
        r"\(?\s*For\s+Program.*?\)?",
        r"FECA\s+NUMBER",
        
        # Box labels (standalone)
        r"(?<![A-Z])CITY(?![A-Z])",
        r"(?<![A-Z])STATE(?![A-Z])",
        r"ZIP\s*CODE",
        r"(?<![A-Z])TELEPHONE(?![A-Z])",
        r"(?<![A-Z])SEX(?![A-Z])\s*[MF]?",
        r"(?<![A-Z])NPI(?![A-Z])",
        r"(?<![A-Z])DOB(?![A-Z])",
        
        # OCR garbage patterns (common misreads)
        r"N[SU]U?n?[gC]?[OD]?o?\s*",  # "NSUngOo" etc.
        r"L[AE]ET\s*N[E3]ME",  # "Laet Neme"
        r"M[I1]DD[E3]?\s*B[I1]TA",  # "Midde bita"
        r"F[I1]RST\s*N[AE]R[E3]",  # "First Nare"
    }
    
    def _clean_field_value(self, raw_text: str, field_label: str, field_id: str) -> str:
        """
        Strip pre-printed labels from OCR text to get just the filled-in value.
        
        CMS-1500 forms have pre-printed labels like "PATIENT'S NAME (Last, First, Middle)"
        that get captured by OCR along with the actual handwritten values.
        This method removes them using comprehensive pattern matching.
        """
        import re
        
        if not raw_text:
            return ""
        
        text = raw_text
        
        # Apply all CMS-1500 template label patterns
        for pattern in self.CMS1500_TEMPLATE_LABELS:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        
        # Also try to remove the specific field label
        if field_label:
            # Escape special regex chars and make flexible
            label_pattern = re.escape(field_label).replace(r"\ ", r"\s*")
            text = re.sub(label_pattern, " ", text, flags=re.IGNORECASE)
        
        # Remove field ID patterns (e.g., "2_patient_name" -> remove "patient name")
        if field_id:
            # Extract meaningful words from field_id
            id_words = re.findall(r"[a-zA-Z]+", field_id)
            for word in id_words:
                if len(word) >= 4:  # Skip short words like "id", "of"
                    text = re.sub(rf"\b{word}\b", " ", text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Remove leading/trailing punctuation and junk
        text = re.sub(r"^[\s\(\)\[\]\.,\-:;]+", "", text)
        text = re.sub(r"[\s\(\)\[\]\.,\-:;]+$", "", text)
        
        # Remove standalone single characters (OCR artifacts)
        text = re.sub(r"\s+[A-Za-z]\s+", " ", text)
        
        # Final cleanup
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    async def _match_ocr_to_zones(
        self, 
        word_boxes: List, 
        fields: List[dict], 
        width: int, 
        height: int,
        image: np.ndarray,
        word_level: bool = False,
        template_word_blacklist: Optional[set] = None,
        require_ink: bool = False,
        ink_threshold: float = 0.02,
        skip_label_cleaning: bool = False
    ) -> List[DetectedBlock]:
        """
        Match OCR word boxes to schema field zones.
        
        Strategy:
        1. For each schema field, calculate expected pixel coordinates from bbox_norm
        2. Find all OCR words that overlap with the field region
        3. Concatenate overlapping text as the field value
        
        Template word blacklist filters out pre-printed form labels AFTER OCR,
        preserving full image quality for handwriting recognition.
        """
        blocks = []

        # Precompute zones — direct bbox_norm to pixel conversion, no offsets
        zones = []
        for field_def in fields:
            field_id = field_def.get("id")
            if not field_id:
                continue
            bbox_norm = field_def.get("bbox_norm")
            if not bbox_norm or len(bbox_norm) != 4:
                continue

            x0 = int(bbox_norm[0] * width)
            y0 = int(bbox_norm[1] * height)
            x1 = int(bbox_norm[2] * width)
            y1 = int(bbox_norm[3] * height)

            # Expand matching region (ratio + px padding)
            pad_x = max(int((x1 - x0) * float(self.config.zone_padding_ratio)), int(self.config.zone_padding_px))
            pad_y = max(int((y1 - y0) * float(self.config.zone_padding_ratio)), int(self.config.zone_padding_px))
            x0_exp = max(0, x0 - pad_x)
            y0_exp = max(0, y0 - pad_y)
            x1_exp = min(width, x1 + pad_x)
            y1_exp = min(height, y1 + pad_y)

            field_type = field_def.get("field_type", "text")
            label = field_def.get("label", field_id)

            zones.append({
                "field_id": field_id,
                "label": label,
                "field_type": field_type,
                "bbox": (x0, y0, x1, y1),
                "bbox_exp": (x0_exp, y0_exp, x1_exp, y1_exp),
                "words": []
            })

        # Helper: intersection area
        def _inter_area(a, b) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            oy = max(0.0, min(ay1, by1) - max(ay0, by0))
            return ox * oy

        # Assign each OCR "word" box to the best zone (prevents duplicates when zones overlap).
        # Only safe when we truly have word-level boxes (e.g. digital PDF text layer).
        # OCR detectors often output *line-level* boxes that span multiple fields, and forcing unique
        # assignment on those will mis-route text.
        unique_ok = bool(self.config.enforce_unique_word_assignment and word_level)
        blacklist = template_word_blacklist or set()
        gray_for_ink = None
        if require_ink and image is not None and hasattr(image, "shape"):
            try:
                gray_for_ink = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            except Exception:
                gray_for_ink = None

        def _has_ink(bbox: Tuple[float, float, float, float]) -> bool:
            if gray_for_ink is None:
                return True
            try:
                gx0, gy0, gx1, gy1 = [int(round(v)) for v in bbox]
                h, w = gray_for_ink.shape[:2]
                gx0, gy0 = max(0, gx0), max(0, gy0)
                gx1, gy1 = min(w, gx1), min(h, gy1)
                if gx1 <= gx0 + 2 or gy1 <= gy0 + 2:
                    return False
                crop = gray_for_ink[gy0:gy1, gx0:gx1]
                if crop.size < 50:
                    return False
                thr = int(np.clip(np.median(crop) - 15, 90, 210))
                dark = np.count_nonzero(crop < thr)
                ink = dark / float(crop.size)
                return ink >= ink_threshold
            except Exception:
                return True
        def _is_template_word(text: str) -> bool:
            """Check if an OCR word is from the pre-printed template.
            Returns True if >50% of the tokens in the text are blacklisted."""
            if not text or not blacklist:
                return False
            t = text.strip().lower()
            # Exact match
            if t in blacklist:
                return True
            # Token-level: if majority of tokens are template words, skip
            tokens = t.replace(".", " ").replace(",", " ").replace("(", " ").replace(")", " ").split()
            if not tokens:
                return False
            bl_count = sum(1 for tok in tokens if tok in blacklist or (len(tok) >= 3 and tok.rstrip("'s") in blacklist))
            # Skip if >50% of tokens are template words
            return bl_count > len(tokens) * 0.5
        
        if unique_ok:
            for wb in word_boxes:
                if _is_template_word(wb.text):
                    continue
                if require_ink and not _has_ink(wb.bbox):
                    continue
                wx0, wy0, wx1, wy1 = wb.bbox
                word_bbox = (float(wx0), float(wy0), float(wx1), float(wy1))
                cx = (word_bbox[0] + word_bbox[2]) / 2.0
                cy = (word_bbox[1] + word_bbox[3]) / 2.0
                # Use a top-anchored point for assignment. OCR sometimes returns tall word boxes
                # that span multiple adjacent fields (e.g., patient name + address rows).
                # Using the center can mis-assign such boxes to the lower field.
                word_h = max(word_bbox[3] - word_bbox[1], 1.0)
                anchor_y = word_bbox[1] + min(2.0, word_h * 0.2)
                anchor_x = cx

                def _contains(b, x, y) -> bool:
                    x0, y0, x1, y1 = b
                    return (x0 <= x <= x1) and (y0 <= y <= y1)

                def _area(b) -> float:
                    x0, y0, x1, y1 = b
                    return max((x1 - x0) * (y1 - y0), 1.0)

                # Prefer original bbox containment (more precise), fall back to expanded bbox containment.
                candidates = []
                for i, z in enumerate(zones):
                    if _contains(z["bbox"], anchor_x, anchor_y):
                        candidates.append((i, _area(z["bbox"])))

                if not candidates:
                    for i, z in enumerate(zones):
                        if _contains(z["bbox_exp"], anchor_x, anchor_y):
                            candidates.append((i, _area(z["bbox_exp"])))

                if candidates:
                    # Choose smallest containing zone (most specific) to avoid bleeding into neighbors
                    candidates.sort(key=lambda t: t[1])
                    best_idx = candidates[0][0]
                    zones[best_idx]["words"].append({
                        "text": wb.text,
                        "confidence": wb.confidence,
                        "x": cx,
                        "y": cy
                    })
                # NOTE: In word-level mode we do NOT perform an “intersection fallback”.
                # If a word doesn't land in (bbox or bbox_exp), leaving it unassigned is safer
                # than contaminating a neighboring field.
        else:
            # Legacy behavior: each zone collects any overlapping words
            for z in zones:
                x0_exp, y0_exp, x1_exp, y1_exp = z["bbox_exp"]
                for wb in word_boxes:
                    if _is_template_word(wb.text):
                        continue
                    if require_ink and not _has_ink(wb.bbox):
                        continue
                    wx0, wy0, wx1, wy1 = wb.bbox
                    inter = _inter_area((float(wx0), float(wy0), float(wx1), float(wy1)), (x0_exp, y0_exp, x1_exp, y1_exp))
                    if inter > 0:
                        cx = (wx0 + wx1) / 2.0
                        cy = (wy0 + wy1) / 2.0
                        z["words"].append({"text": wb.text, "confidence": wb.confidence, "x": cx, "y": cy})

        # Build blocks from zones
        for z in zones:
            field_id = z["field_id"]
            label = z["label"]
            field_type = z["field_type"]
            x0, y0, x1, y1 = z["bbox"]

            # Sort words by position (top-to-bottom, left-to-right)
            words = sorted(z["words"], key=lambda w: (w["y"], w["x"]))

            raw_text = " ".join([w["text"] for w in words]).strip()
            avg_conf = sum(w["confidence"] for w in words) / len(words) if words else 0.0

            # For scan/OCR text, strip printed labels. For digital text, skip
            # (digital values are ground truth — cleaning corrupts them,
            #  e.g. "8340 Baltimore Aveune" → "altimore Aveune").
            if skip_label_cleaning:
                cleaned_text = raw_text
            else:
                cleaned_text = self._clean_field_value(raw_text, label, field_id)

            # Assign block type based on schema hints
            label_l = (label or "").lower()
            field_id_l = (field_id or "").lower()
            if field_type == "checkbox":
                block_type = BlockType.CHECKBOX
            elif field_type == "signature" or "signature" in field_id_l or "signature" in label_l:
                block_type = BlockType.SIGNATURE
            elif "table" in field_id_l or "table" in label_l:
                block_type = BlockType.TABLE
            elif "figure" in field_id_l or "figure" in label_l or "image" in label_l or "graphic" in label_l:
                block_type = BlockType.FIGURE
            elif "title" in field_id_l or "title" in label_l:
                block_type = BlockType.TITLE
            elif "header" in field_id_l or "header" in label_l:
                block_type = BlockType.HEADER
            elif "footer" in field_id_l or "footer" in label_l:
                block_type = BlockType.FOOTER
            elif ("page" in label_l and "number" in label_l) or "page_num" in field_id_l or "pagenum" in field_id_l:
                block_type = BlockType.PAGE_NUM
            elif "list" in field_id_l or "list" in label_l:
                block_type = BlockType.LIST
            else:
                block_type = BlockType.FORM_FIELD

            blocks.append(DetectedBlock(
                id=field_id,
                block_type=block_type,
                bbox=(x0, y0, x1, y1),
                text=cleaned_text,
                confidence=avg_conf,
                metadata={
                    "label": label,
                    "semantic_label": label,  # Use schema label as semantic label
                    "source": "ocr_zone_matching",
                    "field_type": field_type,
                    "skip_label_cleaning": bool(skip_label_cleaning),
                    "digital_text": bool(skip_label_cleaning),
                    "num_matched_words": len(words),
                    "raw_ocr_text": raw_text,
                    "zone_padding_px": int(self.config.zone_padding_px),
                    "zone_padding_ratio": float(self.config.zone_padding_ratio),
                    "unique_word_assignment": bool(unique_ok),
                }
            ))
        
        # Filter out empty fields (optional - keep for completeness)
        # blocks = [b for b in blocks if b.text]
        
        return blocks
    
    def _group_words_into_blocks(self, word_boxes: List, width: int, height: int) -> List[DetectedBlock]:
        """
        Group OCR word boxes into logical text blocks based on spatial proximity.
        This provides meaningful structure instead of one giant text block.
        
        Enhanced algorithm:
        1. Filter out garbage/low-confidence words
        2. Sort words by Y position (top to bottom)
        3. Group words into lines based on Y proximity
        4. Merge nearby lines into paragraphs for better structure
        """
        if not word_boxes:
            return []
        
        # STEP 1: Minimal filtering - only remove truly garbage
        min_conf = getattr(self.config, 'min_ocr_confidence', 0.20)
        filtered_words = []
        for word in word_boxes:
            # Skip very low confidence
            if word.confidence < min_conf:
                continue
            text = str(word.text or "").strip()
            # Skip empty
            if not text:
                continue
            # Skip truly garbage patterns only
            if self._is_garbage_text(text):
                continue
            filtered_words.append(word)
        
        if not filtered_words:
            # If filtering removed everything, return all words with text
            filtered_words = [w for w in word_boxes if str(w.text or "").strip()]
            if not filtered_words:
                return []
        
        print(f"[Pipeline] OCR: {len(word_boxes)} words -> {len(filtered_words)} after minimal filter")
        
        # Sort by Y position (top to bottom), then X (left to right)
        sorted_words = sorted(filtered_words, key=lambda w: (w.bbox[1], w.bbox[0]))
        
        # STEP 2: Group into lines
        lines = []
        current_line_words = []
        current_y = None
        line_threshold = max(15, height * 0.015)  # Adaptive threshold based on page height
        
        for word in sorted_words:
            word_y = (word.bbox[1] + word.bbox[3]) / 2  # Center Y
            
            if current_y is None:
                current_y = word_y
                current_line_words = [word]
            elif abs(word_y - current_y) <= line_threshold:
                # Same line
                current_line_words.append(word)
            else:
                # New line - save current line
                if current_line_words:
                    lines.append(current_line_words)
                current_line_words = [word]
                current_y = word_y
        
        # Don't forget last line
        if current_line_words:
            lines.append(current_line_words)
        
        # STEP 3: Convert lines to blocks with smart merging
        # For general forms, merge nearby lines into paragraph-like blocks
        blocks = []
        para_gap = max(30, height * 0.03)  # Gap threshold for paragraph grouping
        current_para_lines = []
        last_y = None
        
        for line_words in lines:
            line_y = min(w.bbox[1] for w in line_words)
            
            if last_y is None or (line_y - last_y) <= para_gap:
                # Same paragraph
                current_para_lines.append(line_words)
                last_y = max(w.bbox[3] for w in line_words)
            else:
                # New paragraph - save current
                if current_para_lines:
                    block = self._lines_to_block(current_para_lines, len(blocks))
                    if block.text and len(block.text.strip()) >= 3:
                        blocks.append(block)
                current_para_lines = [line_words]
                last_y = max(w.bbox[3] for w in line_words)
        
        # Save last paragraph
        if current_para_lines:
            block = self._lines_to_block(current_para_lines, len(blocks))
            if block.text and len(block.text.strip()) >= 3:
                blocks.append(block)
        
        print(f"[Pipeline] Created {len(blocks)} text blocks from {len(lines)} lines")
        return blocks
    
    def _is_garbage_text(self, text: str) -> bool:
        """Check if text is truly garbage (very minimal filtering)."""
        import re
        
        # Empty or whitespace
        if not text or not text.strip():
            return True
        
        text = text.strip()
        
        # Only filter TRULY garbage patterns - be very conservative
        # Single character
        if len(text) == 1 and not text.isalnum():
            return True
        
        # Only special characters (no letters/numbers at all)
        if re.match(r'^[\|\-\_\=\+\#\*\.\/\\]+$', text):
            return True
        
        # Repeated single character (like "----" or "====")
        if re.match(r'^(.)\1{5,}$', text):
            return True
        
        return False
    
    def _lines_to_block(self, lines: List[List], block_idx: int) -> DetectedBlock:
        """Convert multiple lines of words into a DetectedBlock (paragraph)."""
        # Flatten all words
        all_words = []
        for line in lines:
            all_words.extend(sorted(line, key=lambda w: w.bbox[0]))
        
        if not all_words:
            return DetectedBlock(
                id=f"block_{block_idx}",
                block_type=BlockType.TEXT,
                bbox=(0, 0, 1, 1),
                text="",
                confidence=0.0
            )
        
        # Compute bounding box
        x0 = min(w.bbox[0] for w in all_words)
        y0 = min(w.bbox[1] for w in all_words)
        x1 = max(w.bbox[2] for w in all_words)
        y1 = max(w.bbox[3] for w in all_words)
        
        # Build text line by line
        text_parts = []
        for line in lines:
            line_sorted = sorted(line, key=lambda w: w.bbox[0])
            line_text = " ".join(w.text for w in line_sorted)
            text_parts.append(line_text)
        text = "\n".join(text_parts)
        
        avg_conf = sum(w.confidence for w in all_words) / len(all_words)
        
        return DetectedBlock(
            id=f"block_{block_idx}",
            block_type=BlockType.TEXT,
            bbox=(x0, y0, x1, y1),
            text=text,
            confidence=avg_conf,
            metadata={
                "source": "full_page_ocr",
                "word_count": len(all_words),
                "line_count": len(lines),
                "ocr_engine": "paddleocr"
            }
        )
    
    def _words_to_block(self, words: List, block_idx: int) -> DetectedBlock:
        """Convert a list of word boxes into a DetectedBlock (single line)."""
        # Sort words left to right
        words = sorted(words, key=lambda w: w.bbox[0])
        
        # Compute bounding box
        x0 = min(w.bbox[0] for w in words)
        y0 = min(w.bbox[1] for w in words)
        x1 = max(w.bbox[2] for w in words)
        y1 = max(w.bbox[3] for w in words)
        
        # Concatenate text
        text = " ".join(w.text for w in words)
        avg_conf = sum(w.confidence for w in words) / len(words)
        
        return DetectedBlock(
            id=f"line_{block_idx}",
            block_type=BlockType.TEXT,
            bbox=(x0, y0, x1, y1),
            text=text,
            confidence=avg_conf,
            metadata={
                "source": "full_page_ocr",
                "word_count": len(words),
                "ocr_engine": "paddleocr"
            }
        )

    async def _load_schema_zones(self, image: np.ndarray, width: int, height: int, is_scan: bool = True) -> List[DetectedBlock]:
        """Load CMS-1500 schema zones with mode-aware field selection.
        
        For scanned forms (Lane C): uses composite address fields, skips digital-only sub-fields.
        For digital forms: uses individual address sub-fields, skips scan-only composites.
        """
        import json
        from pathlib import Path
        
        schema_path = Path(__file__).parent.parent.parent / "data" / "schemas" / "cms-1500.json"
        if not schema_path.exists():
            print(f"[Pipeline] Schema not found: {schema_path}")
            return []
        
        try:
            with open(schema_path) as f:
                schema = json.load(f)
            
            x_offset = getattr(self.config, "alignment_x_offset", -0.008)
            y_offset = getattr(self.config, "alignment_y_offset", 0.0)
            pad_ratio = getattr(self.config, "zone_padding_ratio", 0.20)
            pad_px = getattr(self.config, "zone_padding_px", 15)
            
            blocks = []
            skipped_mode = 0
            for field in schema.get("fields", []):
                bbox_norm = field.get("bbox_norm")
                if not bbox_norm or len(bbox_norm) != 4:
                    continue
                
                field_id = field.get("id", f"field_{len(blocks)}")
                field_type = field.get("field_type", "text")
                mode = field.get("mode", "both")
                
                # Mode filtering: skip fields not applicable to current lane
                if is_scan and mode == "digital":
                    skipped_mode += 1
                    continue
                if not is_scan and mode == "scan":
                    skipped_mode += 1
                    continue
                
                x0_norm = bbox_norm[0] + x_offset
                y0_norm = bbox_norm[1] + y_offset
                x1_norm = bbox_norm[2] + x_offset
                y1_norm = bbox_norm[3] + y_offset
                
                x0 = int(x0_norm * width)
                y0 = int(y0_norm * height)
                x1 = int(x1_norm * width)
                y1 = int(y1_norm * height)
                
                box_w = x1 - x0
                box_h = y1 - y0
                pad_x = max(int(box_w * pad_ratio), pad_px)
                pad_y = max(int(box_h * pad_ratio), pad_px)
                
                x0_padded = max(0, x0 - pad_x)
                y0_padded = max(0, y0 - pad_y)
                x1_padded = min(width, x1 + pad_x)
                y1_padded = min(height, y1 + pad_y)
                
                if field_type == "table" or "service_lines" in field_id.lower():
                    block_type = BlockType.TABLE
                elif field_type == "signature":
                    block_type = BlockType.SIGNATURE
                elif field_type == "checkbox":
                    block_type = BlockType.CHECKBOX
                else:
                    block_type = BlockType.FORM_FIELD
                
                blocks.append(DetectedBlock(
                    id=field_id,
                    block_type=block_type,
                    bbox=(x0_padded, y0_padded, x1_padded, y1_padded),
                    confidence=0.9,
                    metadata={
                        "label": block_type.value.upper(),
                        "field_name": field.get("label", ""),
                        "source": "schema_zones",
                        "field_type": field_type,
                        "class_name": block_type.value,
                        "original_bbox": (x0, y0, x1, y1),
                        "alignment_offset": (x_offset, y_offset),
                        "padding_applied": (pad_x, pad_y),
                        "mode": mode
                    }
                ))
            
            table_count = sum(1 for b in blocks if b.block_type == BlockType.TABLE)
            print(f"[Pipeline] Loaded {len(blocks)} zones (skipped {skipped_mode} mode-filtered, tables={table_count})")
            return blocks
        except Exception as e:
            print(f"[Pipeline] Failed to load schema: {e}")
            return []
    
    def _parse_address_block(self, text: str, prefix: str) -> Dict[str, str]:
        """Parse a composite address block OCR result into sub-fields.
        
        CMS-1500 address blocks have a known layout:
        Line 1: Street address (e.g., "825 Lynn Ogden Lane")
        Line 2: City  State  (e.g., "Beaumont TX")
        Line 3: ZIP  Phone  (e.g., "77701 (409) 853-3240")
        
        Returns dict with keys like {prefix}_city, {prefix}_state, etc.
        """
        import re
        result = {}
        if not text:
            return result
        
        lines = [l.strip() for l in text.replace('\n', ' | ').split('|') if l.strip()]
        if not lines:
            lines = [text.strip()]
        
        all_text = text.strip()
        
        # US state abbreviations
        states = {
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
            "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
            "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
            "TX","UT","VT","VA","WA","WV","WI","WY","DC"
        }
        
        # Extract phone number pattern
        phone_match = re.search(r'\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}', all_text)
        if phone_match:
            result[f"{prefix}_phone"] = phone_match.group().strip()
            all_text = all_text[:phone_match.start()] + all_text[phone_match.end():]
        
        # Extract ZIP code (5 digits, optionally followed by -4 digits)
        zip_match = re.search(r'\b(\d{5})(?:-\d{4})?\b', all_text)
        if zip_match:
            result[f"{prefix}_zip"] = zip_match.group().strip()
            all_text = all_text[:zip_match.start()] + all_text[zip_match.end():]
        
        # Extract state (2-letter abbreviation)
        for token in all_text.split():
            if token.upper().rstrip('.,') in states:
                result[f"{prefix}_state"] = token.upper().rstrip('.,')
                all_text = all_text.replace(token, '', 1)
                break
        
        # Remaining text: try to split into address and city
        remaining = re.sub(r'\s+', ' ', all_text).strip().rstrip(',. ')
        
        if remaining:
            # Heuristic: if there's a comma, split at last comma
            if ',' in remaining:
                parts = remaining.rsplit(',', 1)
                result[f"{prefix}_address"] = parts[0].strip()
                if len(parts) > 1 and parts[1].strip():
                    result[f"{prefix}_city"] = parts[1].strip()
            else:
                # First line is usually the street address
                words = remaining.split()
                # If we have many words, assume first part is address and last 1-2 words are city
                if len(words) > 3:
                    # Look for common address suffixes to find the split point
                    addr_suffixes = {"lane", "road", "rd", "st", "ave", "drive", "dr", "blvd", "way", "ct", "pl", "ln"}
                    split_idx = len(words)
                    for i, w in enumerate(words):
                        if w.lower().rstrip('.,') in addr_suffixes and i > 0:
                            split_idx = i + 1
                            break
                    result[f"{prefix}_address"] = ' '.join(words[:split_idx]).strip()
                    city_part = ' '.join(words[split_idx:]).strip()
                    if city_part:
                        result[f"{prefix}_city"] = city_part
                else:
                    result[f"{prefix}_address"] = remaining
        
        return result
    
    async def _extract_service_lines_ocr(self, image: np.ndarray, table_bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """Extract Box 24 service lines using cell-by-cell OCR.
        
        CMS-1500 Box 24 has exactly 6 data rows with a fixed column layout.
        Instead of VLM (which hallucinates), we divide the table into cells
        and run PaddleOCR + TrOCR on each cell individually.
        """
        from src.processing.preprocessing import remove_red_template_text
        
        h, w = image.shape[:2]
        tx0, ty0, tx1, ty1 = [int(v) for v in table_bbox]
        tx0, ty0 = max(0, tx0), max(0, ty0)
        tx1, ty1 = min(w, tx1), min(h, ty1)
        
        table_crop = image[ty0:ty1, tx0:tx1]
        if table_crop.size == 0:
            return {"type": "table", "rows": [], "extraction_method": "cell_ocr"}
        
        # Remove red template lines from table crop
        try:
            clean_crop = remove_red_template_text(table_crop)
        except Exception:
            clean_crop = table_crop
        
        th, tw = clean_crop.shape[:2]
        
        # CMS-1500 Box 24 column layout (relative to table width)
        columns = [
            ("date_from",    0.00, 0.13),
            ("date_to",      0.13, 0.21),
            ("place",        0.21, 0.26),
            ("cpt_code",     0.30, 0.48),
            ("modifier",     0.48, 0.54),
            ("dx_pointer",   0.54, 0.59),
            ("charges",      0.59, 0.73),
            ("days_units",   0.73, 0.79),
            ("provider_id",  0.80, 1.00),
        ]
        
        # Skip header row (~18% of table height), then 6 equal data rows
        header_frac = 0.18
        data_start = int(th * header_frac)
        data_height = th - data_start
        row_height = data_height // 6
        
        rows = []
        from src.ocr.paddle_ocr import PaddleOCRWrapper
        paddle = PaddleOCRWrapper()
        
        for row_idx in range(6):
            ry0 = data_start + row_idx * row_height
            ry1 = min(th, ry0 + row_height)
            
            if ry1 - ry0 < 5:
                continue
            
            row_data = {"line_number": row_idx + 1}
            has_content = False
            
            for col_name, cx0_frac, cx1_frac in columns:
                cx0 = int(tw * cx0_frac)
                cx1 = int(tw * cx1_frac)
                
                # Crop the cell with a small vertical padding
                cell_pad_y = max(2, int(row_height * 0.05))
                cell_y0 = max(0, ry0 - cell_pad_y)
                cell_y1 = min(th, ry1 + cell_pad_y)
                cell_crop = clean_crop[cell_y0:cell_y1, cx0:cx1]
                
                if cell_crop.size == 0:
                    row_data[col_name] = ""
                    continue
                
                # Run PaddleOCR on the cell
                try:
                    word_boxes = paddle.extract_text(cell_crop)
                    if word_boxes:
                        cell_text = " ".join(wb.text for wb in word_boxes if wb.confidence >= 0.2)
                        cell_conf = max(wb.confidence for wb in word_boxes)
                    else:
                        cell_text = ""
                        cell_conf = 0.0
                except Exception:
                    cell_text = ""
                    cell_conf = 0.0
                
                # TrOCR fallback for low-confidence cells
                if (not cell_text or cell_conf < 0.3) and getattr(self.config, 'enable_trocr', False):
                    try:
                        trocr_text, trocr_conf = self.ocr_agent._trocr_ocr(cell_crop)
                        if trocr_text and len(trocr_text.strip()) > 0:
                            if not self.ocr_agent._is_hallucination(trocr_text.strip()):
                                if trocr_conf > cell_conf or not cell_text:
                                    cell_text = trocr_text.strip()
                    except Exception:
                        pass
                
                cell_text = cell_text.strip()
                if cell_text:
                    has_content = True
                row_data[col_name] = cell_text
            
            if has_content:
                rows.append(row_data)
        
        # Build summary text for the block
        summary_parts = []
        for row in rows:
            parts = []
            if row.get("date_from"):
                parts.append(row["date_from"])
            if row.get("cpt_code"):
                parts.append(row["cpt_code"])
            if row.get("charges"):
                parts.append(f"${row['charges']}")
            if row.get("provider_id"):
                parts.append(row["provider_id"])
            if parts:
                summary_parts.append(" | ".join(parts))
        
        return {
            "type": "table",
            "rows": rows,
            "summary": "\n".join(summary_parts),
            "extraction_method": "cell_ocr",
            "total_rows": len(rows)
        }
    
    def _to_reducto_format(self, result: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
        """Convert pipeline result to Reducto-like JSON format with full enrichment."""
        import uuid
        
        field_details = result.get("field_details", [])
        page_w = float(width) if width > 0 else 1.0
        page_h = float(height) if height > 0 else 1.0
        
        # 1. Build Blocks (fine-grained regions)
        blocks = []
        all_text_lines = []  # Collect all text for content
        
        for field in field_details:
            x0, y0, x1, y1 = field.get("bbox", [0, 0, 0, 0])
            fid = str(field.get("id", "")).lower()
            text = str(field.get("value") or field.get("text") or "").strip()
            
            if not text:
                continue
                
            # Collect text for content
            all_text_lines.append(text)
            
            # Reducto Type Mapping
            block_type = "Text"
            source = field.get("metadata", {}).get("source", "")
            if "table" in fid or field.get("type") == "table":
                block_type = "Table"
            elif "figure" in fid or field.get("type") == "figure":
                block_type = "Figure"
            elif "title" in fid or field.get("type") == "title":
                block_type = "Title"
            elif "header" in fid or field.get("type") == "header":
                block_type = "Header"
            elif source == "full_page_ocr":
                block_type = "Text"
            elif ":" in text or "=" in text:
                block_type = "Key Value"
            
            conf_score = field.get("confidence", 0.0)
            conf_str = "high" if conf_score > 0.85 else ("medium" if conf_score > 0.6 else "low")
            
            blocks.append({
                "type": block_type,
                "bbox": {
                    "left": x0 / page_w,
                    "top": y0 / page_h,
                    "width": (x1 - x0) / page_w,
                    "height": (y1 - y0) / page_h,
                    "page": 1,
                    "original_page": 1
                },
                "content": text,
                "image_url": None,
                "chart_data": None,
                "confidence": conf_str,
                "granular_confidence": {
                    "extract_confidence": None,
                    "parse_confidence": conf_score
                }
            })

        # 2. Build the Main Content (Reducto returns the full OCR text organized by reading order)
        form_name = result.get("form_type", "Document").upper().replace("-", " ")
        
        # For full-page OCR, just return the text in reading order
        if all_text_lines:
            full_content = f"# {form_name}\n\n" + "\n".join(all_text_lines)
        else:
            full_content = f"# {form_name}\n\n(No text extracted)"

        # 3. Assemble Reducto-like structure
        return {
            "job_id": str(uuid.uuid4()),
            "duration": result.get("processing_time", 0.0),
            "pdf_url": None,
            "studio_link": None,
            "usage": {"num_pages": 1, "credits": 4},
            "result": {
                "type": "full",
                "chunks": [
                    {
                        "content": full_content,
                        "embed": full_content,
                        "enriched": full_content,
                        "enrichment_success": True,
                        "blocks": blocks
                    }
                ],
                "ocr": None,
                "custom": None
            }
        }

    async def process(self, path: str) -> Dict[str, Any]:
        """
        Process a document through the full pipeline.
        
        Returns comprehensive extraction result.
        """
        start_time = time.time()
        print(f"[Pipeline] Processing {path}")
        
        # Load image (+ optional digital text layer boxes)
        image, width, height, digital_words = self._load_image(path)
        digital_words_present = bool(digital_words) if digital_words is not None else False
        pre_meta = {}
        
        # Step 1: Form Identification
        if self.config.enable_form_detection and not self.config.form_type_override:
            form_id = await self.form_id_agent.process(image)
            
            # Fallback: check filename if detection failed
            if form_id.form_type == FormType.GENERIC:
                fname = Path(path).name.lower()
                if "cms1500" in fname or "cms-1500" in fname:
                    print(f"[Pipeline] Filename hint override: {fname} -> CMS-1500")
                    form_id.form_type = FormType.CMS1500
                    form_id.confidence = 0.8
                elif "ub04" in fname or "ub-04" in fname or "ub_04" in fname:
                    print(f"[Pipeline] Filename hint override: {fname} -> UB-04")
                    form_id.form_type = FormType.UB04
                    form_id.confidence = 0.8
        else:
            form_id = FormIdentification(
                form_type=self.config.form_type_override or FormType.GENERIC,
                confidence=1.0,
                detection_method="override"
            )
        
        print(f"[Pipeline] Detected Form Type: {form_id.form_type}")

        # ══════════════════════════════════════════════════════════════
        # LANE A: AcroForm widget extraction (fillable PDFs)
        # This is the HIGHEST accuracy path for machine-filled CMS-1500.
        # If widgets provide enough data, we skip OCR entirely.
        # ══════════════════════════════════════════════════════════════
        if form_id.form_type in (FormType.CMS1500, FormType.UB04):
            widget_info = self._extract_widgets(path)
            if widget_info is not None:
                extracted_fields, blocks = self._map_widgets_to_schema(widget_info, form_id.form_type)

                # Validate: if we got a good set of fields, return immediately
                filled_count = sum(1 for v in extracted_fields.values() if v and len(str(v).strip()) > 0)
                print(f"[Lane A] Widget extraction produced {filled_count} non-empty fields")

                min_fields = 10 if form_id.form_type == FormType.CMS1500 else 3
                if filled_count >= min_fields:  # Good enough — skip OCR entirely
                    processing_time = time.time() - start_time

                    # Business mapping
                    from src.pipelines.business_schema import map_to_business_schema, merge_business_with_ocr
                    temp_result = {
                        "extracted_fields": extracted_fields,
                        "field_details": [
                            {"id": b.id, "bbox": list(b.bbox), "confidence": b.confidence, "metadata": b.metadata}
                            for b in blocks
                        ],
                        "page_width": width, "page_height": height,
                    }
                    if form_id.form_type == FormType.CMS1500:
                        business_result = map_to_business_schema(temp_result, "cms-1500")
                    elif form_id.form_type == FormType.UB04:
                        business_result = map_to_business_schema(temp_result, "ub-04")
                    else:
                        business_result = {
                            "business_fields": {},
                            "business_field_details": [],
                            "business_coverage": 0.0,
                        }

                    # Validation
                    validation = await self.validation_agent.process(blocks, extracted_fields)

                    final_result = {
                        "success": True,
                        "form_type": form_id.form_type.value,
                        "form_confidence": form_id.confidence,
                        "form_version": form_id.version,
                        "alignment_quality": 1.0,
                        "extracted_fields": extracted_fields,
                        "field_details": [
                            {
                                "id": b.id,
                                "label": self._get_block_label(b),
                                "type": b.block_type.value if hasattr(b.block_type, 'value') else str(b.block_type),
                                "bbox": list(b.bbox),
                                "value": b.text or "",
                                "text": b.text or "",
                                "confidence": b.confidence,
                                "detected_by": "acroform_widget",
                                "metadata": b.metadata,
                            }
                            for b in blocks
                        ],
                        "page_width": width, "page_height": height,
                        "processing_time": processing_time,
                        "validation": validation,
                        "ocr_blocks": [
                            {"text": b.text, "bbox": list(b.bbox), "confidence": b.confidence}
                            for b in blocks if b.text
                        ],
                        "extraction_method": "lane_a_acroform_widgets",
                        "config": {
                            "layout_model": "none (widget extraction)",
                            "enable_trocr": False,
                            "enable_slm": False,
                            "enable_vlm": False,
                        },
                        "debug": {
                            "lane": "A",
                            "widgets_total": widget_info.get("total_widgets", 0),
                            "widgets_filled": widget_info.get("filled", 0),
                            "digital_text_used": False,
                            "alignment_used": False,
                        },
                    }
                    final_merged = merge_business_with_ocr(final_result, business_result)
                    final_merged["reducto_format"] = self._to_reducto_format(final_merged, width, height)
                    print(f"[Pipeline] ✅ Lane A complete: {filled_count} fields in {processing_time:.1f}s (no OCR needed)")
                    return final_merged
                else:
                    print(f"[Pipeline] Lane A yielded only {filled_count} fields — falling through to Lane B/C")

        # Decide whether to use digital text layer:
        # - Only available for PDFs
        # - Only trust it when it contains *real filled values*, not just the pre-printed template layer.
        use_digital_text = False
        
        # For non-CMS1500 forms (UB-04, generic, etc.):
        # If _digital_layer_matches_visual passed (meaning digital_words_present is True),
        # trust it and skip deskew - digital PDFs are already straight.
        if digital_words_present and form_id.form_type != FormType.CMS1500:
            use_digital_text = True
            print(f"[Pipeline] ✅ Digital text layer present for {form_id.form_type} - using it (skip deskew).")
        
        # For CMS-1500: do additional validation to catch template-only layers
        elif digital_words_present and form_id.form_type == FormType.CMS1500:
            try:
                schema_path = Config.PROJECT_ROOT / "data" / "schemas" / "cms-1500.json"
                if schema_path.exists():
                    import json
                    with open(schema_path) as f:
                        schema = json.load(f)
                    fields_list = schema.get("fields", [])
                    candidate_blocks = await self._match_ocr_to_zones(
                        digital_words or [], fields_list, width, height, image, word_level=True, skip_label_cleaning=True
                    )
                    by_id = {b.id: ((b.metadata or {}).get("raw_ocr_text") or b.text or "").strip() for b in candidate_blocks}
                    insured_id = by_id.get("1a_insured_id", "")
                    patient_name = by_id.get("2_patient_name", "")
                    patient_dob = by_id.get("3_patient_dob", "")

                    import re
                    def looks_like_member_id(s: str) -> bool:
                        s = (s or "").strip()
                        return bool(re.search(r"[A-Za-z]{0,4}\d{4,}", s)) and len(s) >= 5
                    def looks_like_name(s: str) -> bool:
                        s = (s or "").strip()
                        return bool(re.search(r"[A-Za-z]{2,}", s)) and ("," in s or " " in s)
                    def looks_like_dob(s: str) -> bool:
                        s = (s or "").strip()
                        nums = re.findall(r"\d{2,4}", s)
                        return len(nums) >= 3

                    score = sum([
                        1 if looks_like_member_id(insured_id) else 0,
                        1 if looks_like_name(patient_name) else 0,
                        1 if looks_like_dob(patient_dob) else 0,
                    ])
                    print(f"[Pipeline] Digital QA: insured_id='{insured_id[:40]}', patient_name='{patient_name[:40]}', dob='{patient_dob[:40]}', score={score}/3")
                    # Require at least 2/3 anchor fields to look sane; otherwise treat as scan.
                    use_digital_text = score >= 2
                    if use_digital_text:
                        print("[Pipeline] ✅ Digital text layer validated - using it (skip preprocess/alignment).")
                    else:
                        print("[Pipeline] ⚠️ Digital text layer present but looks like template-only; using scan OCR path.")
            except Exception as e:
                print(f"[Pipeline] Digital layer validation error: {e}")
                use_digital_text = False

        # Preprocess ONLY for scan/camera path.
        # For CMS-1500 scans: SKIP all preprocessing (no deskew, no contrast).
        # The alignment agent handles rotation via homography, and aggressive
        # contrast/CLAHE degrades handwritten text quality.
        # For general forms: light preprocessing only (deskew, no heavy contrast).
        if not use_digital_text:
            from src.processing.preprocessing import preprocess_image
            will_align = bool(self.config.enable_alignment and form_id.form_type == FormType.CMS1500)
            if will_align:
                # CMS-1500 scan: SKIP preprocessing entirely — alignment handles it
                pre_meta = {"skipped": True, "reason": "cms1500_will_align"}
                print("[Pipeline] Skipping preprocessing for CMS-1500 scan (alignment will handle rotation)")
            else:
                image, pre_meta = preprocess_image(
                    image,
                    deskew=True,
                    denoise=False,
                    doc_type="generic"
                )
            height, width = image.shape[:2]

        # Step 2: Template Alignment (scan path only)
        aligned_image = image
        alignment_result = None
        aligned_preview_path = None
        scan_ocr_source = None
        alignment_quality_override = None
        prod_aligned_shape = None
        if (not use_digital_text) and self.config.enable_alignment and form_id.form_type == FormType.CMS1500:
            try:
                alignment_result = await self.alignment_agent.process(image, form_id.form_type)
                if alignment_result.success and alignment_result.aligned_image is not None:
                    aligned_image = alignment_result.aligned_image
                    height, width = aligned_image.shape[:2]
                    print(f"[Pipeline] Alignment succeeded, quality: {alignment_result.alignment_quality:.2f}")
                else:
                    print("[Pipeline] Alignment failed; will NOT use schema zones on raw scan.")
            except Exception as e:
                print(f"[Pipeline] Alignment exception: {e}")

        # Write an aligned preview image for UI overlays (optional but very useful for debugging)
        try:
            import uuid
            cache_dir = Config.PROJECT_ROOT / "cache" / "previews"
            cache_dir.mkdir(parents=True, exist_ok=True)
            aligned_preview_path = cache_dir / f"aligned_{uuid.uuid4().hex}.png"
            # cv2.imwrite expects BGR; our pipeline images are RGB
            if aligned_image is not None and aligned_image.ndim == 3 and aligned_image.shape[2] == 3:
                bgr = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aligned_preview_path), bgr)
            elif aligned_image is not None:
                cv2.imwrite(str(aligned_preview_path), aligned_image)
        except Exception:
            aligned_preview_path = None

        # Step 3: Layout Detection
        blocks = []
        
        # ====== CMS-1500 PRODUCTION PIPELINE (Track 2C: Ink Extraction) ======
        # Use the production pipeline for CMS-1500 scanned forms when enabled.
        # This provides ink extraction, better alignment, and structured business fields.
        if (form_id.form_type == FormType.CMS1500 
            and not use_digital_text 
            and self.config.use_cms1500_production_pipeline
            and self.config.enable_alignment):
            try:
                from src.pipelines.cms1500_production import CMS1500ProductionPipeline
                print("[Pipeline] Using CMS-1500 Production Pipeline (with ink extraction)")
                
                cms_pipeline = CMS1500ProductionPipeline()
                prod_result = None
                use_aligned_for_prod = bool(
                    alignment_result
                    and alignment_result.success
                    and alignment_result.alignment_quality >= float(self.config.alignment_quality_threshold)
                    and alignment_result.aligned_image is not None
                )
                if use_aligned_for_prod:
                    prod_result = cms_pipeline.extract(
                        image_override=alignment_result.aligned_image,
                        already_aligned=True,
                        alignment_quality_override=alignment_result.alignment_quality,
                        source_shape=image.shape[:2],
                    )
                else:
                    # The production pipeline needs a file path, but we have an image.
                    # Save temporarily and process.
                    import tempfile
                    import uuid
                    temp_path = Path(tempfile.gettempdir()) / f"cms1500_{uuid.uuid4().hex}.png"
                    if image.ndim == 3 and image.shape[2] == 3:
                        cv2.imwrite(str(temp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(str(temp_path), image)
                    prod_result = cms_pipeline.extract(str(temp_path))
                    # Clean up temp file
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                
                if prod_result.get("success"):
                    # Convert production result to our format
                    business_fields = prod_result.get("business_fields", {})
                    field_details = prod_result.get("field_details", [])
                    ocr_blocks = prod_result.get("ocr_blocks", [])
                    alignment_quality_override = float(prod_result.get("alignment_quality", 0.0))
                    prod_page_width = prod_result.get("page_width") or width
                    prod_page_height = prod_result.get("page_height") or height
                    width, height = int(prod_page_width), int(prod_page_height)
                    prod_aligned_shape = prod_result.get("aligned_image_shape")
                    prod_preview_path = prod_result.get("aligned_preview_path")
                    if prod_preview_path:
                        try:
                            aligned_preview_path = Path(prod_preview_path)
                        except Exception:
                            aligned_preview_path = prod_preview_path
                    
                    # Build blocks from field_details
                    for fd in field_details:
                        bbox = fd.get("bbox", [0, 0, 1, 1])
                        if len(bbox) == 4:
                            # Convert normalized bbox to absolute
                            abs_bbox = (
                                int(bbox[0] * width),
                                int(bbox[1] * height),
                                int(bbox[2] * width),
                                int(bbox[3] * height)
                            )
                        else:
                            abs_bbox = (0, 0, width, height)
                        
                        blocks.append(DetectedBlock(
                            id=fd.get("id", ""),
                            block_type=BlockType.FORM_FIELD,
                            bbox=abs_bbox,
                            confidence=fd.get("confidence", 0.8),
                            text=fd.get("value", ""),
                            metadata={
                                "source": "cms1500_production",
                                "label": fd.get("label", ""),
                                "ocr_engine": fd.get("ocr_engine", ""),
                                "validated": fd.get("validated", False),
                            }
                        ))
                    
                    print(f"[Pipeline] CMS-1500 Production: extracted {len(blocks)} fields, alignment={alignment_quality_override:.2f}")
                    
                    # Skip remaining CMS-1500 processing since production pipeline handled it
                    # Store business fields for later
                    if not hasattr(self, '_production_business_fields'):
                        self._production_business_fields = {}
                    self._production_business_fields = business_fields
                    
            except Exception as e:
                print(f"[Pipeline] CMS-1500 Production pipeline failed: {e}, falling back to standard path")
                import traceback
                traceback.print_exc()
                blocks = []  # Reset and fall through to standard CMS-1500 handling
        
        if form_id.form_type == FormType.CMS1500 and not blocks:
            schema_path = Config.PROJECT_ROOT / "data" / "schemas" / "cms-1500.json"
            if use_digital_text and schema_path.exists():
                # DIGITAL CMS-1500: zone match using validated digital words (best quality)
                import json
                with open(schema_path) as f:
                    schema = json.load(f)
                fields_list = schema.get("fields", [])
                blocks = await self._match_ocr_to_zones(digital_words or [], fields_list, width, height, aligned_image, word_level=True, skip_label_cleaning=True)
                print(f"[Pipeline] CMS-1500 digital (Lane B): matched {len(blocks)} fields — label cleaning SKIPPED")
                
                # REALITY CHECK: If too few zones have meaningful text, template is wrong
                filled_zones = sum(1 for b in blocks if (b.text or '').strip() and len((b.text or '').strip()) >= 3)
                if filled_zones < 10:  # < ~20% of 48 zones
                    print(f"[Pipeline] ⚠️ Template mismatch (digital): only {filled_zones}/48 zones have text - falling back to layout model")
                    blocks = []  # Force fallback
                    use_digital_text = False  # Disable digital path
            else:
                # SCANNED CMS-1500: Per-field crop OCR
                # Load schema zones as empty blocks, then OCRAgent.process_blocks()
                # will crop each field individually and OCR it with safe red removal.
                # This avoids the full-page OCR + zone matching problem entirely.
                if alignment_result is not None and alignment_result.success:
                    blocks = await self._load_schema_zones(aligned_image, width, height)
                    for b in blocks:
                        if b.metadata is None:
                            b.metadata = {}
                        b.metadata["form_type"] = "cms-1500"
                    scan_ocr_source = aligned_image
                    print(f"[Pipeline] CMS-1500 scan (Lane C): loaded {len(blocks)} schema zones for per-field crop OCR")
                else:
                    # Alignment failed — still try per-field crop OCR on raw image
                    blocks = await self._load_schema_zones(aligned_image, width, height)
                    for b in blocks:
                        if b.metadata is None:
                            b.metadata = {}
                        b.metadata["form_type"] = "cms-1500"
                    scan_ocr_source = aligned_image
                    print(f"[Pipeline] CMS-1500 scan (no alignment): loaded {len(blocks)} schema zones for per-field crop OCR")
        elif form_id.form_type == FormType.UB04 and not blocks and use_digital_text:
            schema_path = Config.PROJECT_ROOT / "data" / "schemas" / "ub-04.json"
            if schema_path.exists():
                import json
                with open(schema_path) as f:
                    schema = json.load(f)
                fields_list = schema.get("fields", [])
                blocks = await self._match_ocr_to_zones(
                    digital_words or [],
                    fields_list,
                    width,
                    height,
                    aligned_image,
                    word_level=True,
                    skip_label_cleaning=True
                )
                filled_zones = sum(1 for b in blocks if (b.text or '').strip() and len((b.text or '').strip()) >= 2)
                print(f"[Pipeline] UB-04 digital (Lane B): matched {len(blocks)} fields, filled={filled_zones}")
                if filled_zones < 2:
                    print("[Pipeline] ⚠️ UB-04 digital match too sparse; falling back to layout model")
                    blocks = []
        
        # GENERAL FORM PATH (only if no blocks yet)
        if not blocks:
            # Try Detectron2/PaddleDetection. Drop giant blocks; fallback to OCR line grouping.
            print("[Pipeline] Running layout detection for general form...")
            try:
                blocks = await self.layout_agent.process(aligned_image, form_id.form_type)
                print(f"[Pipeline] Detected {len(blocks)} blocks")
                
                # Drop blocks that are basically "the whole page"
                page_area = float(width * height) if width > 0 and height > 0 else 1.0
                filtered = []
                for b in blocks:
                    x0, y0, x1, y1 = b.bbox
                    area = float(max(0.0, x1 - x0) * max(0.0, y1 - y0))
                    if area / page_area > 0.85:
                        continue
                    filtered.append(b)
                if len(filtered) != len(blocks):
                    print(f"[Pipeline] Dropped {len(blocks) - len(filtered)} giant blocks")
                blocks = filtered

                if len(blocks) < 3:
                    print("[Pipeline] ⚠️ Layout too coarse (<3 blocks), using OCR line grouping fallback")
                    from src.ocr.paddle_ocr import PaddleOCRWrapper
                    paddle = PaddleOCRWrapper()
                    word_boxes = paddle.extract_text(aligned_image)
                    if word_boxes:
                        blocks = self._group_words_into_blocks(word_boxes, width, height)
                        print(f"[Pipeline] Full-page OCR: {len(word_boxes)} words -> {len(blocks)} blocks")
            except Exception as e:
                print(f"[Pipeline] Layout detection failed: {e}")
                # Fallback to OCR
                from src.ocr.paddle_ocr import PaddleOCRWrapper
                paddle = PaddleOCRWrapper()
                word_boxes = paddle.extract_text(aligned_image)
                if word_boxes:
                    blocks = self._group_words_into_blocks(word_boxes, width, height)
        
        # If still no blocks, use full-page OCR with intelligent word grouping
        if not blocks:
            print("[Pipeline] No layout blocks detected, using full-page OCR with word grouping")
            from src.ocr.paddle_ocr import PaddleOCRWrapper
            paddle = PaddleOCRWrapper()
            word_boxes = paddle.extract_text(aligned_image)
            
            if word_boxes:
                # Use word grouping instead of one giant block
                blocks = self._group_words_into_blocks(word_boxes, width, height)
                if not blocks:
                    # Final fallback: concatenate all text if grouping fails
                    all_text = " ".join([wb.text for wb in word_boxes if wb.confidence >= self.config.min_ocr_confidence])
                    if all_text.strip():
                        blocks = [DetectedBlock(
                            id="full_page",
                            block_type=BlockType.TEXT,
                            bbox=(0, 0, width, height),
                            text=all_text,
                            confidence=sum(wb.confidence for wb in word_boxes) / len(word_boxes)
                        )]
        
        # Step 4: OCR
        # Attach form_type to block metadata for OCR routing (template-diff, etc.)
        for b in blocks:
            if b.metadata is None:
                b.metadata = {}
            b.metadata.setdefault("form_type", form_id.form_type.value if hasattr(form_id.form_type, "value") else str(form_id.form_type))
        
        # Per-field OCR uses the raw aligned image (no red removal).
        # For CMS-1500 scans, zone-matched blocks already have text from
        # full-page OCR and skip per-field re-OCR (line 1442).
        # This mainly affects checkboxes and signatures.
        ocr_image = aligned_image

        blocks = await self.ocr_agent.process_blocks(ocr_image, blocks)

        # Post-clean CMS-1500 zone OCR: strip printed labels from OCR text.
        # ONLY for scan/OCR path (not use_digital_text). Digital text layer values
        # are already clean ground truth. _clean_field_value corrupts them:
        #   "8340 Baltimore Aveune" → regex strips "8340 B" → "altimore Aveune"
        if form_id.form_type == FormType.CMS1500 and not use_digital_text:
            for b in blocks:
                try:
                    src = (b.metadata or {}).get("source")
                    if src not in ("schema_zones", "ocr_zone_matching"):
                        continue
                    if not b.text:
                        continue
                    field_label = (b.metadata or {}).get("field_name") or (b.metadata or {}).get("label") or b.id
                    cleaned = self._clean_field_value(str(b.text), str(field_label), str(b.id))
                    if cleaned and len(cleaned) <= len(str(b.text)) + 2:
                        b.metadata["raw_ocr_text"] = b.metadata.get("raw_ocr_text") or b.text
                        b.text = cleaned
                        b.metadata["post_cleaned"] = True
                except Exception:
                    continue
        
        # Step 5: SLM/VLM Labeling
        # For CMS-1500: 
        #   - Skip SLM for text/form_field blocks (already schema-identified)
        #   - ALWAYS process table blocks (service lines) via VLM for structured extraction
        #   - Process figure blocks via VLM if enabled
        # For general forms: full SLM labeling on all blocks.
        if form_id.form_type == FormType.CMS1500:
            # Box 24 service lines: use cell-by-cell OCR (not VLM which hallucinates)
            for block in blocks:
                if block.block_type == BlockType.TABLE and "service_lines" in block.id:
                    print(f"[Pipeline] Extracting service lines via cell-by-cell OCR: {block.id}")
                    table_data = await self._extract_service_lines_ocr(aligned_image, block.bbox)
                    block.metadata["table_data"] = table_data
                    if table_data.get("summary"):
                        block.text = table_data["summary"]
                        block.metadata["cell_ocr_extracted"] = True
                    print(f"[Pipeline] Service lines: {table_data.get('total_rows', 0)} rows extracted")
                elif block.block_type == BlockType.FIGURE and self.config.enable_vlm_figures:
                    figure_data = await self.labeling_agent.process_figure(aligned_image, block)
                    block.metadata["figure_data"] = figure_data
        elif self.config.enable_slm_labeling:
            blocks = await self.labeling_agent.process(aligned_image, blocks)
        
        # Step 6: Build extracted data
        # SORT blocks by Y position (top to bottom) for logical ordering
        blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))  # Sort by Y, then X
        
        # For CMS-1500, always use block.id as the key (schema field ID).
        # For general forms, use semantic_label if SLM provided one.
        extracted_fields = {}
        for block in blocks:
            if block.text:
                val = str(block.text).strip()
                if not val or val.lower() in ("null", "none", "n/a", "-", ""):
                    continue
                if form_id.form_type == FormType.CMS1500:
                    label = block.id
                else:
                    label = block.metadata.get("semantic_label", block.id)
                extracted_fields[label] = val
        
        # Post-process composite address blocks for CMS-1500 scans:
        # Parse full-address OCR into individual sub-fields (city, state, zip, phone)
        if form_id.form_type == FormType.CMS1500 and not use_digital_text:
            for composite_id, prefix in [("5_patient_address_full", "5_patient"),
                                          ("7_insured_address_full", "7_insured")]:
                if composite_id in extracted_fields:
                    parsed = self._parse_address_block(extracted_fields[composite_id], prefix)
                    for sub_key, sub_val in parsed.items():
                        if sub_key not in extracted_fields and sub_val:
                            extracted_fields[sub_key] = sub_val
                    # Keep composite field for reference but rename to _raw
                    extracted_fields[composite_id.replace("_full", "_raw")] = extracted_fields.pop(composite_id)
        
        # Step 7: Validation
        validation = await self.validation_agent.process(blocks, extracted_fields)
        
        # Step 8: Business Mapping (Canonical Schema)
        from src.pipelines.business_schema import map_to_business_schema, merge_business_with_ocr
        
        # Prepare intermediate result for mapping
        temp_result = {
            "extracted_fields": extracted_fields,
            "field_details": [
                {
                    "id": b.id,
                    "bbox": list(b.bbox),
                    "confidence": b.confidence,
                    "metadata": b.metadata
                }
                for b in blocks
            ],
            "page_width": width,
            "page_height": height
        }
        
        # Map to business schema
        business_result = map_to_business_schema(temp_result, form_id.form_type.value)
        
        # Build final result dict
        processing_time = time.time() - start_time
        if use_digital_text:
            alignment_quality_value = 1.0
        elif alignment_quality_override is not None:
            alignment_quality_value = float(alignment_quality_override)
        elif alignment_result:
            alignment_quality_value = float(alignment_result.alignment_quality)
        else:
            alignment_quality_value = 0.0
        if alignment_quality_override is not None:
            alignment_success_value = alignment_quality_value >= float(self.config.alignment_quality_threshold)
        else:
            alignment_success_value = bool(alignment_result.success) if alignment_result else False
        
        final_result = {
            "success": True,
            "form_type": form_id.form_type.value,
            "form_confidence": form_id.confidence,
            "form_version": form_id.version,
            "alignment_quality": alignment_quality_value,
            "extracted_fields": extracted_fields,
            "field_details": [
                {
                    "id": b.id,
                    "label": self._get_block_label(b),  # Block type label for visualization
                    "type": b.block_type.value if hasattr(b.block_type, 'value') else str(b.block_type),
                    "bbox": list(b.bbox),
                    "value": b.text or "",
                    "text": b.text or "",
                    "confidence": b.confidence,
                    "detected_by": b.metadata.get("source", "yolo"),
                    "metadata": b.metadata
                }
                for b in blocks
            ],
            "page_width": width,
            "page_height": height,
            "processing_time": processing_time,
            "validation": validation,
            "ocr_blocks": [
                {"text": b.text, "bbox": list(b.bbox), "confidence": b.confidence}
                for b in blocks if b.text
            ],
            "extraction_method": "multi_agent",
            "config": {
                "layout_model": self.config.layout_model,
                "enable_trocr": self.config.enable_trocr,
                "enable_slm": self.config.enable_slm_labeling,
                "enable_vlm": self.config.enable_vlm_figures
            },
            "debug": {
                "alignment_used": bool(self.config.enable_alignment),
                "alignment_success": alignment_success_value,
                "aligned_image_shape": prod_aligned_shape or (list(aligned_image.shape[:2]) if aligned_image is not None else None),
                "alignment_quality": alignment_quality_value,
                "alignment_method": (
                    (alignment_result.metadata or {}).get("alignment_method")
                    if alignment_result is not None
                    else None
                ),
                "alignment_profile": (
                    ((alignment_result.metadata or {}).get("registrar_debug") or {}).get("profile")
                    if alignment_result is not None
                    else None
                ),
                "aligned_preview_path": str(aligned_preview_path) if aligned_preview_path else None,
                "digital_text_used": bool(use_digital_text),
                "digital_words_count": int(len(digital_words)) if digital_words is not None else 0,
            },
        }
        
        # Merge business data
        final_merged = merge_business_with_ocr(final_result, business_result)
        
        # Add Reducto-style output
        final_merged["reducto_format"] = self._to_reducto_format(final_merged, width, height)
        
        return final_merged
    
    def process_sync(self, path: str) -> Dict[str, Any]:
        """Synchronous wrapper for process()."""
        return asyncio.run(self.process(path))


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Document Pipeline")
    parser.add_argument("input", help="Input document path")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--form-type", choices=["cms-1500", "ub-04", "generic"], help="Override form type")
    parser.add_argument("--no-alignment", action="store_true", help="Disable template alignment")
    parser.add_argument("--no-trocr", action="store_true", help="Disable TrOCR")
    parser.add_argument("--no-slm", action="store_true", help="Disable SLM labeling")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM for figures")
    args = parser.parse_args()
    
    config = PipelineConfig(
        form_type_override=FormType(args.form_type) if args.form_type else None,
        enable_alignment=not args.no_alignment,
        enable_trocr=not args.no_trocr,
        enable_slm_labeling=not args.no_slm,
        enable_vlm_figures=not args.no_vlm
    )
    
    pipeline = MultiAgentPipeline(config)
    result = pipeline.process_sync(args.input)
    
    print(f"\n✅ Processing complete in {result['processing_time']:.2f}s")
    print(f"   Form type: {result['form_type']} ({result['form_confidence']:.0%})")
    print(f"   Fields extracted: {len(result['extracted_fields'])}")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()

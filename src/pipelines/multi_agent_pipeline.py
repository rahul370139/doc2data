"""
Multi-Agent Document Processing Pipeline

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOCUMENT INGESTION                                  │
│                     (PDF/Image → 300 DPI RGB Array)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FORM IDENTIFICATION AGENT                                │
│  - OCR header for "CMS-1500", "UB-04", etc.                                  │
│  - Layout fingerprint matching (Sensible-style)                              │
│  - Returns: form_type, confidence, version                                   │
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
│                      TEMPLATE ALIGNMENT AGENT                                │
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
    enable_trocr: bool = True
    ocr_confidence_threshold: float = 0.5
    handwriting_threshold: float = 0.35
    # Schema-zone matching / OCR tuning
    zone_padding_px: int = 8
    zone_padding_ratio: float = 0.15
    # If True, each OCR word box is assigned to at most one schema zone to avoid duplicates/overlaps
    enforce_unique_word_assignment: bool = True
    
    # Template alignment - ENABLED for CMS-1500 scans (safe alignment implementation below)
    enable_alignment: bool = True
    alignment_fallback_to_ml: bool = True
    # Only trust schema zones for scans if alignment is strong
    alignment_quality_threshold: float = 0.85
    
    # SLM/VLM
    enable_slm_labeling: bool = True
    enable_vlm_figures: bool = True
    slm_model: str = "llama3.2:3b"
    vlm_model: str = "llama3.2:3b"  # or minicpm-v
    
    # Validation
    enable_validators: bool = True
    enable_llm_qa: bool = True
    
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
            "hcfa-1500", "please print or type"
        ],
        FormType.UB04: [
            "ub-04", "ub04", "uniform bill",
            "patient control no", "med rec no"
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
    STRONG_TOKENS = {
        FormType.UB04: ["ub-04", "ub04", "uniform bill", "ub 04"],
        FormType.CMS1500: ["cms-1500", "cms 1500", "cms1500", "hcfa-1500"],
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
    
    def _compute_homography(self, image: np.ndarray, template: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Compute alignment from input image -> template space.

        IMPORTANT:
        - We DO NOT use ECC refinement here. ECC is powerful but can introduce "tilt"/shear
          when the input has handwriting/noise (exactly the issue you reported).
        - We first try a stable quad-based perspective warp (outer form boundary).
        - If that fails, we fall back to feature matching (SIFT/ORB) WITHOUT ECC.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Build a "structure" image (mainly form ruling lines) to make alignment robust to handwriting.
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

        th, tw = template.shape[:2]

        # 0) Stable primary: detect outer form quad and warp directly to template corners
        quad, q_score = self._detect_form_quad(gray)
        if quad is not None and q_score >= 0.55:
            dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
            if H is not None:
                return H, float(q_score)

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
        Validate homography is sane.
        We ALLOW mild perspective (scans), but we reject wild warps that tilt/flip the page.
        """
        if H is None:
            return False
        try:
            h, w = img_shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            if not np.isfinite(warped).all():
                return False
            # Reject if warped polygon area is too small/huge
            area = float(cv2.contourArea(warped.astype(np.float32)))
            orig = float(w * h)
            if area < orig * 0.20 or area > orig * 5.0:
                return False
            # Reject flips (polygon should be convex)
            if not cv2.isContourConvex(warped.astype(np.float32)):
                return False
            return True
        except Exception:
            return False
    
    async def process(self, image: np.ndarray, form_type: FormType) -> AlignmentResult:
        """Align image to template."""
        await self.initialize()
        
        # Check if we have a template for this form type
        if form_type not in self._templates:
            return AlignmentResult(
                success=False,
                aligned_image=image,
                homography_matrix=None,
                alignment_quality=0.0,
                fallback_used=True,
                error_message="No template available"
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
            
            return AlignmentResult(
                success=True,
                aligned_image=aligned,
                homography_matrix=H,
                alignment_quality=quality,
                fallback_used=False
            )
        
        # Fallback: return original image
        return AlignmentResult(
            success=False,
            aligned_image=image,
            homography_matrix=None,
            alignment_quality=0.0,
            fallback_used=True,
            error_message="Homography validation failed"
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
        
        # Initialize Detectron2/LayoutParser
        try:
            import layoutparser as lp
            # Try Detectron2 first (better quality)
            # IMPORTANT: Use Detectron2 config, NOT PaddleDetection config
            self._detectron = lp.Detectron2LayoutModel(
                'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.config.detectron_threshold],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
            self.log("✅ Detectron2 initialized with faster_rcnn_R_50_FPN_3x")
        except Exception as e:
            self.log(f"⚠️ Detectron2 init failed: {e}")
            # Try PaddleDetection as fallback (easier install, comparable quality)
            try:
                import layoutparser as lp
                self._detectron = lp.PaddleDetectionLayoutModel(
                    config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config",
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                )
                self.log("✅ PaddleDetectionLayoutModel initialized (fallback)")
            except Exception as e2:
                self.log(f"❌ PaddleDetection fallback failed: {e2}")
        
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
            
            for i, element in enumerate(layout):
                block_type = self._map_detectron_type(element.type)
                bbox = (element.block.x_1, element.block.y_1, 
                       element.block.x_2, element.block.y_2)
                
                blocks.append(DetectedBlock(
                    id=f"det-{i}",
                    block_type=block_type,
                    bbox=bbox,
                    confidence=element.score,
                    metadata={"model": "detectron2"}
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
            self.log("Using Detectron2 for general detection")
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
        
        # Standard text - for aligned scans, use template-diff crop to suppress printed labels/lines.
        crop_for_ocr = crop
        tmpl = self._get_template_gray((block.metadata or {}).get("form_type"))
        if tmpl is not None and image.shape[0] == tmpl.shape[0] and image.shape[1] == tmpl.shape[1]:
            diff_img = self._template_diff_crop(crop, tmpl, (x0_p, y0_p, x1_p, y1_p))
            if diff_img is not None:
                crop_for_ocr = diff_img
                block.metadata["template_diff_used"] = True

        # Tiered OCR (PaddleOCR first, TrOCR fallback)
        # For small form fields, try extra sharpening/contrast
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

        # If PaddleOCR produced something but it's low-confidence / noisy, try TrOCR and pick the better result.
        if self._is_handwritten(crop_for_ocr, paddle_conf, paddle_text) or (paddle_text and _looks_noisy(paddle_text)) or (paddle_conf < 0.60):
            trocr_text, trocr_conf = self._trocr_ocr(crop_for_ocr)
            if trocr_text and len(trocr_text.strip()) > 0:
                # Pick by "looks less noisy" first, then longer text.
                pt = (paddle_text or "").strip()
                tt = (trocr_text or "").strip()
                pick_trocr = False
                if _looks_noisy(pt) and not _looks_noisy(tt):
                    pick_trocr = True
                elif _alnum_ratio(tt) >= _alnum_ratio(pt) + 0.10:
                    pick_trocr = True
                elif len(tt) > len(pt) + 3:
                    pick_trocr = True

                if pick_trocr:
                    block.text = tt
                    block.confidence = trocr_conf
                    block.metadata["ocr_engine"] = "trocr"
                    return block

        # Default: PaddleOCR result (may be empty)
        block.text = (paddle_text or "").strip()
        block.confidence = paddle_conf
        block.metadata["ocr_engine"] = "paddleocr"
        return block
    
    async def process_blocks(self, image: np.ndarray, blocks: List[DetectedBlock]) -> List[DetectedBlock]:
        """Process multiple blocks concurrently."""
        await self.initialize()
        
        # Process blocks (could be parallelized with asyncio.gather)
        results = []
        for block in blocks:
            result = await self.process(image, block)
            results.append(result)
        
        return results


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
    
    async def initialize(self):
        if self._initialized:
            return
        self._initialized = True
    
    def _call_slm(self, prompt: str) -> str:
        """Call SLM via Ollama."""
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
    
    async def label_text_block(self, block: DetectedBlock, context: str = "") -> DetectedBlock:
        """Label a text block and clean its value using SLM."""
        if not self.config.enable_slm_labeling or not block.text:
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
            clean_resp = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_resp)
            
            # Store fine-grained semantic role
            role = (data.get("role") or "text").lower()
            field_name = data.get("field_name", "Unknown")
            block.metadata["semantic_role"] = role
            block.metadata["semantic_label"] = field_name

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
            if cleaned and cleaned.lower() not in ["null", "none", ""]:
                block.metadata["original_text"] = block.text
                block.text = cleaned
        except Exception:
            pass
            
        return block
    
    async def process_table(self, image: np.ndarray, block: DetectedBlock) -> Dict[str, Any]:
        """Process table block using TATR + SLM."""
        # Extract table structure
        table_data = {
            "type": "table",
            "bbox": block.bbox,
            "rows": [],
            "raw_text": block.text
        }
        
        if self.config.enable_slm_labeling:
            prompt = f"""Extract structured data from this table text:
"{block.text}"

Return as JSON with rows and columns."""
            
            response = self._call_slm(prompt)
            try:
                # Try to parse as JSON
                table_data["structured"] = json.loads(response)
            except:
                table_data["structured"] = response
        
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
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize agents
        self.form_id_agent = FormIdentificationAgent()
        self.alignment_agent = TemplateAlignmentAgent()
        self.layout_agent = LayoutDetectionAgent(self.config)
        self.ocr_agent = OCRAgent(self.config)
        self.labeling_agent = LabelingAgent(self.config)
        self.validation_agent = ValidationAgent(self.config)

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
    
    def _clean_field_value(self, raw_text: str, field_label: str, field_id: str) -> str:
        """
        Strip pre-printed labels from OCR text to get just the filled-in value.
        
        CMS-1500 forms have pre-printed labels like "PATIENT'S NAME" that get captured
        by OCR along with the actual values. This method removes them.
        """
        import re
        
        if not raw_text:
            return ""
        
        text = raw_text
        
        # Common CMS-1500 pre-printed patterns to remove
        patterns_to_remove = [
            # Field numbers and labels (with or without dot; OCR often drops the dot)
            r"^\s*\d+\s*[a-z]?\s*\.?\s*",  # "1", "1.", "1a", "1a.", etc.
            
            # Common field labels (case-insensitive)
            # More robust: allow missing leading letters due to OCR errors (P?ATIENT, I?NSURED)
            r"P?ATIENT'?S?\s*(NAME|BIRTH|ADDRESS|SEX|PHONE|RELATIONSHIP)",
            r"I?NSURED'?S?\s*(NAME|I\.?D\.?\s*N|ADDRESS|DATE|POLICY|GROUP)",
            r"OTHER\s+INSURED'?S?\s*(NAME|POLICY)",
            r"INSURANCE\s+PLAN\s+NAME",
            r"EMPLOYER'?S?\s*NAME",
            r"REFERRING\s+PROVIDER",
            r"SIGNATURE\s+OF",
            r"BILLING\s+PROVIDER",
            r"SERVICE\s+FACILITY",
            r"DATE\s+OF\s+CURRENT",
            r"DIAGNOSIS\s+OR\s+NATURE",
            
            # Date format labels
            r"\(?\s*MM\s+DD\s+Y+\s*\)?",
            r"MM\s*DD\s*YY",
            
            # Common suffixes/noise
            r"\(Last.*?(First|Name)\)",
            r"\(No\.,?\s*S.*?\)",
            r"\(Include\s+Area\s+Code\)",
            r"OR\s+PROGR\s*AM\s+NAME",
            r"\(For\s+Program.*?\)",
            
            # Box labels
            r"CITY(?:\s|$)",
            r"STATE(?:\s|$)",
            r"ZIP\s*CODE",
            r"TELEPHONE",
            r"SEX\s*[MF]?",
        ]
        
        # Also try to remove the specific field label
        if field_label:
            # Escape special regex chars and make flexible
            label_pattern = re.escape(field_label).replace(r"\ ", r"\s*")
            patterns_to_remove.append(label_pattern)
        
        # Apply patterns
        for pattern in patterns_to_remove:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Remove leading/trailing punctuation and junk
        text = re.sub(r"^[\s\(\)\[\]\.,\-:]+", "", text)
        text = re.sub(r"[\s\(\)\[\]\.,\-:]+$", "", text)
        
        return text.strip()
    
    async def _match_ocr_to_zones(
        self, 
        word_boxes: List, 
        fields: List[dict], 
        width: int, 
        height: int,
        image: np.ndarray,
        word_level: bool = False
    ) -> List[DetectedBlock]:
        """
        Match OCR word boxes to schema field zones.
        This is more robust than cropping regions because it handles misalignment.
        
        Strategy:
        1. For each schema field, calculate expected pixel coordinates
        2. Find all OCR words that overlap with the field region
        3. Concatenate overlapping text as the field value
        """
        blocks = []

        # Precompute zones
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
        if unique_ok:
            for wb in word_boxes:
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

            # For machine-filled forms, remove printed label noise now.
            cleaned_text = self._clean_field_value(raw_text, label, field_id)

            if field_type == "checkbox":
                block_type = BlockType.CHECKBOX
            elif field_type == "signature":
                block_type = BlockType.SIGNATURE
            elif "table" in field_id.lower():
                block_type = BlockType.TABLE
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
                    "semantic_label": field_id,
                    "source": "ocr_zone_matching",
                    "field_type": field_type,
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
        
        Algorithm:
        1. Sort words by Y position (top to bottom)
        2. Group words into lines based on Y proximity
        3. Each line becomes a block
        """
        if not word_boxes:
            return []
        
        # Sort by Y position (top to bottom), then X (left to right)
        sorted_words = sorted(word_boxes, key=lambda w: (w.bbox[1], w.bbox[0]))
        
        blocks = []
        current_line_words = []
        current_y = None
        line_threshold = 15  # Pixels - words within this Y range are same line
        
        for word in sorted_words:
            word_y = (word.bbox[1] + word.bbox[3]) / 2  # Center Y
            
            if current_y is None:
                current_y = word_y
                current_line_words = [word]
            elif abs(word_y - current_y) <= line_threshold:
                # Same line
                current_line_words.append(word)
            else:
                # New line - save current line as block
                if current_line_words:
                    blocks.append(self._words_to_block(current_line_words, len(blocks)))
                current_line_words = [word]
                current_y = word_y
        
        # Don't forget last line
        if current_line_words:
            blocks.append(self._words_to_block(current_line_words, len(blocks)))
        
        return blocks
    
    def _words_to_block(self, words: List, block_idx: int) -> DetectedBlock:
        """Convert a list of word boxes into a DetectedBlock."""
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

    async def _load_schema_zones(self, image: np.ndarray, width: int, height: int) -> List[DetectedBlock]:
        """Load CMS-1500 schema zones as fallback layout detection."""
        import json
        from pathlib import Path
        
        schema_path = Path(__file__).parent.parent.parent / "data" / "schemas" / "cms-1500.json"
        if not schema_path.exists():
            print(f"[Pipeline] Schema not found: {schema_path}")
            return []
        
        try:
            with open(schema_path) as f:
                schema = json.load(f)
            
            blocks = []
            for field in schema.get("fields", []):
                bbox_norm = field.get("bbox_norm")
                if not bbox_norm or len(bbox_norm) != 4:
                    continue
                
                # Convert normalized bbox to pixels
                x0 = int(bbox_norm[0] * width)
                y0 = int(bbox_norm[1] * height)
                x1 = int(bbox_norm[2] * width)
                y1 = int(bbox_norm[3] * height)
                
                # Determine block type from field_type
                field_type = field.get("field_type", "text")
                if field_type == "checkbox":
                    block_type = BlockType.CHECKBOX
                elif field_type == "signature":
                    block_type = BlockType.SIGNATURE
                elif "table" in field.get("id", "").lower():
                    block_type = BlockType.TABLE
                else:
                    block_type = BlockType.FORM_FIELD
                
                blocks.append(DetectedBlock(
                    id=field.get("id", f"field_{len(blocks)}"),
                    block_type=block_type,
                    bbox=(x0, y0, x1, y1),
                    confidence=0.9,
                    metadata={
                        "label": block_type.value.upper(),  # Use block type as label
                        "field_name": field.get("label", ""),  # Keep original field name
                        "source": "schema_zones",
                        "field_type": field_type,
                        "class_name": block_type.value  # For consistent display
                    }
                ))
            
            print(f"[Pipeline] Loaded {len(blocks)} zones from schema")
            return blocks
        except Exception as e:
            print(f"[Pipeline] Failed to load schema: {e}")
            return []
    
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
        else:
            form_id = FormIdentification(
                form_type=self.config.form_type_override or FormType.GENERIC,
                confidence=1.0,
                detection_method="override"
            )
        
        print(f"[Pipeline] Detected Form Type: {form_id.form_type}")

        # Decide whether to use digital text layer:
        # - Only available for PDFs
        # - Only trust it when it contains *real filled values*, not just the pre-printed template layer.
        use_digital_text = False
        if digital_words_present and form_id.form_type == FormType.CMS1500:
            try:
                schema_path = Config.PROJECT_ROOT / "data" / "schemas" / "cms-1500.json"
                if schema_path.exists():
                    import json
                    with open(schema_path) as f:
                        schema = json.load(f)
                    fields_list = schema.get("fields", [])
                    candidate_blocks = await self._match_ocr_to_zones(
                        digital_words or [], fields_list, width, height, image, word_level=True
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
        # IMPORTANT for CMS-1500:
        # - If we plan to template-align, DO NOT deskew first (deskew can mis-rotate handwritten scans).
        #   Alignment will rectify rotation in a more stable way using the form boundary.
        if not use_digital_text:
            from src.processing.preprocessing import preprocess_image
            will_align = bool(self.config.enable_alignment and form_id.form_type == FormType.CMS1500)
            image, pre_meta = preprocess_image(
                image,
                deskew=(not will_align),
                doc_type="cms1500" if form_id.form_type == FormType.CMS1500 else "generic"
            )
            height, width = image.shape[:2]

        # Step 2: Template Alignment (scan path only)
        aligned_image = image
        alignment_result = None
        aligned_preview_path = None
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
        
        if form_id.form_type == FormType.CMS1500:
            schema_path = Config.PROJECT_ROOT / "data" / "schemas" / "cms-1500.json"
            if use_digital_text and schema_path.exists():
                # DIGITAL CMS-1500: zone match using validated digital words (best quality)
                import json
                with open(schema_path) as f:
                    schema = json.load(f)
                fields_list = schema.get("fields", [])
                blocks = await self._match_ocr_to_zones(digital_words or [], fields_list, width, height, aligned_image, word_level=True)
                print(f"[Pipeline] CMS-1500 digital: matched {len(blocks)} fields")
                
                # REALITY CHECK: If too few zones have meaningful text, template is wrong
                filled_zones = sum(1 for b in blocks if (b.text or '').strip() and len((b.text or '').strip()) >= 3)
                if filled_zones < 10:  # < ~20% of 48 zones
                    print(f"[Pipeline] ⚠️ Template mismatch (digital): only {filled_zones}/48 zones have text - falling back to layout model")
                    blocks = []  # Force fallback
                    use_digital_text = False  # Disable digital path
            else:
                # SCANNED CMS-1500: ONLY use schema zones if alignment succeeded (template space).
                if alignment_result is not None and alignment_result.success and float(alignment_result.alignment_quality) >= float(self.config.alignment_quality_threshold):
                    # Remove red lines AFTER alignment for OCR quality
                    try:
                        from src.processing.preprocessing import remove_form_lines
                        aligned_image = remove_form_lines(aligned_image)
                    except Exception:
                        pass
                    # Full-page OCR once, then zone-match word boxes (more accurate than per-field crop OCR).
                    if schema_path.exists():
                        import json
                        with open(schema_path) as f:
                            schema = json.load(f)
                        fields_list = schema.get("fields", [])
                        from src.ocr.paddle_ocr import PaddleOCRWrapper
                        paddle = PaddleOCRWrapper()
                        scan_words = paddle.extract_text(aligned_image)
                        # PaddleOCRWrapper returns word-level boxes -> enable unique assignment to reduce zone bleed.
                        blocks = await self._match_ocr_to_zones(scan_words or [], fields_list, width, height, aligned_image, word_level=True)
                        print(f"[Pipeline] CMS-1500 scan: aligned -> zone-matched from full-page OCR ({len(scan_words or [])} words) into {len(blocks)} fields")
                        
                        # REALITY CHECK: If too few zones have meaningful text, template is wrong
                        # (e.g., UB-04 misclassified as CMS-1500)
                        filled_zones = sum(1 for b in blocks if (b.text or '').strip() and len((b.text or '').strip()) >= 3)
                        if filled_zones < 10:  # < ~20% of 48 zones
                            print(f"[Pipeline] ⚠️ Template mismatch: only {filled_zones}/48 zones have text - falling back to layout model")
                            blocks = []  # Force fallback to general layout path
                    else:
                        blocks = await self._load_schema_zones(aligned_image, width, height)
                        print(f"[Pipeline] CMS-1500 scan: aligned -> loaded {len(blocks)} schema zones")
                else:
                    # If we can't align reliably, do full-page OCR line grouping (no fake boxes).
                    print("[Pipeline] CMS-1500 scan: alignment not reliable -> full-page OCR fallback")
                    from src.ocr.paddle_ocr import PaddleOCRWrapper
                    paddle = PaddleOCRWrapper()
                    word_boxes = paddle.extract_text(aligned_image)
                    if word_boxes:
                        blocks = self._group_words_into_blocks(word_boxes, width, height)
                        print(f"[Pipeline] Full-page OCR: {len(word_boxes)} words -> {len(blocks)} blocks")
        else:
            # GENERAL FORM PATH:
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
        
        # If still no blocks, use full-page OCR
        if not blocks:
            print("[Pipeline] No layout blocks detected, using full-page OCR")
            from src.ocr.paddle_ocr import PaddleOCRWrapper
            paddle = PaddleOCRWrapper()
            word_boxes = paddle.extract_text(aligned_image)
            
            if word_boxes:
                all_text = " ".join([wb.text for wb in word_boxes])
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
        
        # Accuracy boost for CMS-1500: remove red lines from the aligned image before OCR
        if form_id.form_type == FormType.CMS1500 and not use_digital_text:
            try:
                from src.processing.preprocessing import remove_form_lines
                ocr_image = remove_form_lines(aligned_image)
                print("[Pipeline] Applied red-line removal for superior CMS-1500 OCR")
            except Exception:
                ocr_image = aligned_image
        else:
            ocr_image = aligned_image

        blocks = await self.ocr_agent.process_blocks(ocr_image, blocks)

        # Post-clean CMS-1500 zone OCR: remove printed labels that leak into the crop.
        if form_id.form_type == FormType.CMS1500:
            for b in blocks:
                try:
                    src = (b.metadata or {}).get("source")
                    if src not in ("schema_zones", "ocr_zone_matching"):
                        continue
                    if not b.text:
                        continue
                    # Prefer human label from schema if present
                    field_label = (b.metadata or {}).get("field_name") or (b.metadata or {}).get("label") or b.id
                    cleaned = self._clean_field_value(str(b.text), str(field_label), str(b.id))
                    # Keep cleaned if it materially improves
                    if cleaned and len(cleaned) <= len(str(b.text)) + 2:
                        b.metadata["raw_ocr_text"] = b.metadata.get("raw_ocr_text") or b.text
                        b.text = cleaned
                        b.metadata["post_cleaned"] = True
                except Exception:
                    continue
        
        # Step 5: SLM/VLM Labeling
        if self.config.enable_slm_labeling:
            blocks = await self.labeling_agent.process(aligned_image, blocks)
        
        # Step 6: Build extracted data
        extracted_fields = {}
        for block in blocks:
            if block.text:
                label = block.metadata.get("semantic_label", block.id)
                extracted_fields[label] = block.text
        
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
        
        final_result = {
            "success": True,
            "form_type": form_id.form_type.value,
            "form_confidence": form_id.confidence,
            "form_version": form_id.version,
            "alignment_quality": 1.0 if use_digital_text else (alignment_result.alignment_quality if alignment_result else 0.0),
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
                "alignment_success": bool(alignment_result.success) if alignment_result else False,
                "aligned_image_shape": list(aligned_image.shape[:2]) if aligned_image is not None else None,
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

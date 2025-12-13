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
│                      LAYOUT DETECTION AGENT                                  │
│  CMS-1500:              │  General Forms:                                        │
│  - YOLOv8 (fine-tuned)  │  - LayoutLMv3 / Detectron2 (PubLayNet)            │
│  - Template zones       │  - Donut (end-to-end)                                  │
│  Returns: blocks with type (text, table, figure, form fields, checkbox)        │
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
    detectron_threshold: float = 0.5
    
    # OCR
    enable_trocr: bool = True
    ocr_confidence_threshold: float = 0.5
    handwriting_threshold: float = 0.35
    
    # Template alignment
    enable_alignment: bool = True
    alignment_fallback_to_ml: bool = True
    
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
        ]
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
            self._ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
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
        
        try:
            result = self._ocr.ocr(crop, cls=True)
            if result and result[0]:
                texts = [line[1][0] for line in result[0] if line]
                return " ".join(texts).lower()
        except:
            pass
        return ""
    
    def _match_fingerprint(self, text: str) -> Tuple[FormType, float]:
        """Match text against form fingerprints."""
        text_lower = text.lower()
        
        best_match = FormType.UNKNOWN
        best_score = 0.0
        
        for form_type, patterns in self.FINGERPRINTS.items():
            matches = sum(1 for p in patterns if p in text_lower)
            score = matches / len(patterns)
            if score > best_score:
                best_score = score
                best_match = form_type
        
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
        
        # Match fingerprint
        form_type, confidence = self._match_fingerprint(combined_text)
        
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
    
    def _compute_homography(self, image: np.ndarray, template: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Compute homography matrix using ORB feature matching."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize template to match image aspect ratio
        h, w = gray.shape[:2]
        th, tw = template.shape[:2]
        scale = min(w / tw, h / th)
        template_resized = cv2.resize(template, (int(tw * scale), int(th * scale)))
        
        # ORB detector
        orb = cv2.ORB_create(nfeatures=5000)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(gray, None)
        kp2, des2 = orb.detectAndCompute(template_resized, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, 0.0
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 10:
            return None, 0.0
        
        # Compute homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return None, 0.0
        
        # Compute alignment quality (inlier ratio)
        inliers = mask.sum() if mask is not None else 0
        quality = inliers / len(good_matches)
        
        return H, quality
    
    def _validate_homography(self, H: np.ndarray, img_shape: Tuple[int, int]) -> bool:
        """Validate that homography doesn't produce unreasonable warps."""
        if H is None:
            return False
        
        h, w = img_shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        try:
            warped_corners = cv2.perspectiveTransform(corners, H)
            
            # Check if corners are within reasonable bounds
            for corner in warped_corners:
                x, y = corner[0]
                if x < -w * 0.5 or x > w * 1.5 or y < -h * 0.5 or y > h * 1.5:
                    return False
            
            # Check if area is preserved (within 20%)
            original_area = w * h
            warped_area = cv2.contourArea(warped_corners)
            if warped_area < original_area * 0.5 or warped_area > original_area * 2.0:
                return False
            
            return True
        except:
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
        
        # Compute homography
        H, quality = self._compute_homography(image, template)
        
        # Validate homography
        if H is not None and self._validate_homography(H, image.shape):
            # Apply warp
            h, w = image.shape[:2]
            aligned = cv2.warpPerspective(image, H, (w, h))
            
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
        if Config.YOLO_MODEL_PATH:
            try:
                from src.pipelines.yolo_layout import YOLOLayoutDetector
                self._yolo = YOLOLayoutDetector(
                    Config.YOLO_MODEL_PATH,
                    conf=self.config.yolo_confidence,
                    iou=Config.YOLO_IOU
                )
                self.log("YOLO detector initialized")
            except Exception as e:
                self.log(f"YOLO init failed: {e}")
        
        # Initialize Detectron2/LayoutParser
        try:
            import layoutparser as lp
            self._detectron = lp.Detectron2LayoutModel(
                'lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.config.detectron_threshold],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
            self.log("Detectron2 initialized")
        except Exception as e:
            self.log(f"Detectron2 init failed: {e}")
        
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
    
    async def initialize(self):
        if self._initialized:
            return
        
        # PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        except Exception as e:
            self.log(f"PaddleOCR init failed: {e}")
        
        # TrOCR (lazy load on first use)
        self._initialized = True
    
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
        """Run PaddleOCR."""
        if self._paddle is None:
            return "", 0.0, []
        
        try:
            result = self._paddle.ocr(image, cls=True)
            if result and result[0]:
                texts, confs, boxes = [], [], []
                for line in result[0]:
                    if line and len(line) >= 2:
                        texts.append(line[1][0])
                        confs.append(line[1][1])
                        boxes.append({
                            "text": line[1][0],
                            "bbox": [p for point in line[0] for p in point],
                            "confidence": line[1][1]
                        })
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
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        density = np.count_nonzero(binary) / binary.size
        is_checked = density > 0.18
        confidence = min(1.0, abs(density - 0.18) / 0.15 + 0.5)
        
        return is_checked, confidence
    
    def _is_handwritten(self, image: np.ndarray, paddle_conf: float) -> bool:
        """Heuristic to detect if text is handwritten."""
        if paddle_conf < self.config.handwriting_threshold:
            return True
        
        # Additional heuristics could be added here
        return False
    
    async def process(self, image: np.ndarray, block: DetectedBlock) -> DetectedBlock:
        """OCR a single block with appropriate method."""
        await self.initialize()
        
        h, w = image.shape[:2]
        x0, y0, x1, y1 = block.bbox
        crop = image[int(y0):int(y1), int(x0):int(x1)]
        
        if crop.size == 0:
            block.text = ""
            return block
        
        # Checkbox detection
        if block.block_type == BlockType.CHECKBOX:
            is_checked, conf = self._detect_checkbox(crop)
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
        
        # Standard text - tiered OCR
        paddle_text, paddle_conf, paddle_boxes = self._paddle_ocr(crop)
        block.metadata["ocr_boxes"] = paddle_boxes
        
        # Try TrOCR if low confidence or likely handwritten
        if self._is_handwritten(crop, paddle_conf):
            trocr_text, trocr_conf = self._trocr_ocr(crop)
            if trocr_text and (trocr_conf > paddle_conf or not paddle_text):
                block.text = trocr_text
                block.confidence = trocr_conf
                block.metadata["ocr_engine"] = "trocr"
                return block
        
        block.text = paddle_text
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
        """Label a text block using SLM."""
        if not self.config.enable_slm_labeling or not block.text:
            return block
        
        prompt = f"""Identify what field this text belongs to on a medical form.
Text: "{block.text}"
Context: {context}

Respond with just the field name (e.g., "Patient Name", "Date of Birth", "Insurance ID").
If unsure, respond with "Unknown"."""
        
        label = self._call_slm(prompt)
        block.metadata["semantic_label"] = label.strip()
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
    
    def _load_image(self, path: str) -> Tuple[np.ndarray, int, int]:
        """Load document image."""
        path = Path(path)
        
        if path.suffix.lower() == ".pdf":
            import fitz
            doc = fitz.open(str(path))
            page = doc[0]
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            doc.close()
        else:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img, img.shape[1], img.shape[0]
    
    async def process(self, path: str) -> Dict[str, Any]:
        """
        Process a document through the full pipeline.
        
        Returns comprehensive extraction result.
        """
        start_time = time.time()
        
        # Load image
        image, width, height = self._load_image(path)
        
        # Step 1: Form Identification
        if self.config.enable_form_detection and not self.config.form_type_override:
            form_id = await self.form_id_agent.process(image)
        else:
            form_id = FormIdentification(
                form_type=self.config.form_type_override or FormType.GENERIC,
                confidence=1.0,
                detection_method="override"
            )
        
        # Step 2: Template Alignment (if applicable)
        aligned_image = image
        alignment_result = None
        
        if self.config.enable_alignment and form_id.form_type != FormType.GENERIC:
            alignment_result = await self.alignment_agent.process(image, form_id.form_type)
            if alignment_result.success:
                aligned_image = alignment_result.aligned_image
        
        # Step 3: Layout Detection
        blocks = await self.layout_agent.process(aligned_image, form_id.form_type)
        
        # Step 4: OCR
        blocks = await self.ocr_agent.process_blocks(aligned_image, blocks)
        
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
        
        # Build result
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "form_type": form_id.form_type.value,
            "form_confidence": form_id.confidence,
            "form_version": form_id.version,
            "alignment_quality": alignment_result.alignment_quality if alignment_result else 0.0,
            "extracted_fields": extracted_fields,
            "field_details": [
                {
                    "id": b.id,
                    "type": b.block_type.value,
                    "bbox": list(b.bbox),
                    "text": b.text,
                    "confidence": b.confidence,
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
            }
        }
    
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


"""
Form Identification Agent - identifies document form type via OCR fingerprinting.

PURPOSE: First agent in the pipeline. OCRs header region and matches keywords
(CMS-1500, UB-04, HCFA) to identify form type. Returns FormIdentification
with form_type and confidence. Used to choose processing lane (A/B/C).

USE CASE: Pipeline calls this before layout/OCR to decide template alignment
and schema. No manual use; part of MultiAgentPipeline.
"""
from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np

from src.pipelines.core import BaseAgent, FormIdentification, FormType


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

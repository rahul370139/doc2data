"""
OCR Agent - tiered OCR for text extraction from blocks.

PURPOSE: Runs PaddleOCR on printed text, TrOCR on handwriting/signatures,
checkbox fill-ratio detector on checkbox blocks. Validates and normalizes
output. Escalates to VLM when confidence is low.

USE CASE: Pipeline calls this after layout detection. Converts image
regions to text. No manual use; part of MultiAgentPipeline.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.pipelines.core import BaseAgent, BlockType, DetectedBlock, PipelineConfig

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.config import Config


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
        """Lazy load TrOCR onto GPU. Uses large-handwritten for best accuracy (~2.89% CER)."""
        if self._trocr_model is not None:
            return
        
        model_name = getattr(self.config, "trocr_model", "large")
        hf_id = "microsoft/trocr-large-handwritten" if model_name == "large" else "microsoft/trocr-base-handwritten"
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self._trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._trocr_processor = TrOCRProcessor.from_pretrained(hf_id)
            self._trocr_model = VisionEncoderDecoderModel.from_pretrained(hf_id).to(self._trocr_device)
            self.log(f"TrOCR loaded ({hf_id}) on {self._trocr_device}")
        except Exception as e:
            self._trocr_device = "cpu"
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
        """Run TrOCR on GPU with real per-token confidence from output logits."""
        if not self.config.enable_trocr:
            return "", 0.0
        
        self._load_trocr()
        
        if self._trocr_model is None:
            return "", 0.0
        
        try:
            from PIL import Image
            import torch
            
            if len(image.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image)
            
            device = getattr(self, "_trocr_device", "cpu")
            pixel_values = self._trocr_processor(
                images=pil_img, return_tensors="pt"
            ).pixel_values.to(device)
            
            with torch.no_grad():
                outputs = self._trocr_model.generate(
                    pixel_values,
                    max_length=128,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            
            text = self._trocr_processor.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )[0].strip()
            
            if hasattr(outputs, "scores") and outputs.scores:
                token_probs = []
                for score in outputs.scores:
                    prob = torch.softmax(score, dim=-1).max(dim=-1).values
                    token_probs.append(prob.item())
                avg_conf = sum(token_probs) / len(token_probs) if token_probs else 0.5
            else:
                avg_conf = 0.75
            
            return text, float(avg_conf)
        except Exception as e:
            self.log(f"TrOCR error: {e}")
        
        return "", 0.0
    
    def _vlm_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Use local Ollama VLM (llava, minicpm-v, llama3.2-vision) to transcribe when OCR confidence is low."""
        if not getattr(self.config, "enable_vlm_ocr_fallback", True):
            return "", 0.0
        try:
            import requests
            import base64
            model = self.config.vlm_ocr_model
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image)
            img_b64 = base64.b64encode(buffer).decode("utf-8")
            prompt = "Transcribe all handwritten or printed text in this image exactly as written. Output ONLY the text, nothing else. No explanation."
            resp = requests.post(
                f"http://{Config.OLLAMA_HOST}/api/generate",
                json={"model": model, "prompt": prompt, "images": [img_b64], "stream": False, "options": {"temperature": 0.0, "num_predict": 200}},
                timeout=15
            )
            if resp.ok:
                text = (resp.json().get("response") or "").strip()
                if text and len(text) > 1 and not self._is_hallucination(text):
                    return text, 0.75
        except Exception as e:
            self.log(f"VLM OCR fallback failed: {e}")
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
        # Raised from 0.020 to 0.026 to reduce false X on empty checkboxes (red lines, shadows).
        is_checked = ink_ratio > 0.026
        confidence = float(min(1.0, max(0.0, (ink_ratio - 0.015) / 0.05)))
        
        return is_checked, confidence
    
    def _has_ink(self, img: np.ndarray, threshold: float = 0.015) -> bool:
        """Check if crop has visible ink (handwritten/printed content)."""
        if img is None or img.size == 0:
            return False
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        g = cv2.GaussianBlur(g, (3, 3), 0)
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9)
        ink_ratio = float(np.count_nonzero(bw) / max(1, bw.size))
        return ink_ratio > threshold
    
    def _is_handwritten(self, image: np.ndarray, paddle_conf: float, paddle_text: str) -> bool:
        """Heuristic to detect if text is handwritten.
        
        TrOCR is prone to hallucinating on blank/noisy crops.
        Only consider TrOCR if there is visible ink AND PaddleOCR is low-quality.
        """
        if not self._has_ink(image):
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
        
        # ── Red removal / template diff ──────────────────────────────
        crop_for_ocr = crop
        form_type_meta = (block.metadata or {}).get("form_type", "")
        is_cms_scan = form_type_meta in ("cms-1500", "CMS1500", "cms1500")
        tmpl = None
        if is_cms_scan:
            try:
                from src.processing.preprocessing import remove_red_template_text
                crop_for_ocr = remove_red_template_text(crop)
                block.metadata["red_removal_used"] = True
            except Exception:
                pass
        else:
            tmpl = self._get_template_gray(form_type_meta)
        if tmpl is not None and image.shape[0] == tmpl.shape[0] and image.shape[1] == tmpl.shape[1]:
            diff_img = self._template_diff_crop(crop, tmpl, (x0_p, y0_p, x1_p, y1_p))
            if diff_img is not None:
                crop_for_ocr = diff_img
                block.metadata["template_diff_used"] = True

        # ── CMS-1500 scan: tiered field-aware OCR pipeline ───────────
        if is_cms_scan:
            field_type = (block.metadata or {}).get("field_type", "text")
            return self._process_cms_field(crop_for_ocr, crop, block, field_type)

        # ── General forms: PaddleOCR → TrOCR → VLM chain ────────────
        paddle_text, paddle_conf, paddle_boxes = self._paddle_ocr(crop_for_ocr)
        if paddle_conf < 0.5 and paddle_text:
            zoomed = cv2.resize(crop_for_ocr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            z_text, z_conf, z_boxes = self._paddle_ocr(zoomed)
            if z_conf > paddle_conf:
                paddle_text, paddle_conf, paddle_boxes = z_text, z_conf, z_boxes
        block.metadata["ocr_boxes"] = paddle_boxes

        use_trocr = getattr(self.config, "enable_trocr", False) and self.config.enable_trocr
        if use_trocr and (not paddle_text or paddle_conf < 0.30):
            if self._is_handwritten(crop_for_ocr, paddle_conf, paddle_text):
                trocr_text, trocr_conf = self._trocr_ocr(crop_for_ocr)
                if trocr_text and not self._is_hallucination(trocr_text):
                    if trocr_conf > paddle_conf:
                        block.text = trocr_text
                        block.confidence = trocr_conf
                        block.metadata["ocr_engine"] = "trocr"
                        return block

        best_text = (paddle_text or "").strip()
        best_conf = paddle_conf
        best_engine = "paddleocr"
        if best_conf < 0.40 and self._has_ink(crop_for_ocr):
            vlm_text, vlm_conf = self._vlm_ocr(crop_for_ocr)
            if vlm_text and vlm_conf > best_conf:
                best_text, best_conf, best_engine = vlm_text, vlm_conf, "vlm_ocr"

        block.text = best_text
        block.confidence = best_conf
        block.metadata["ocr_engine"] = best_engine
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

    # ── Field-type-aware decision tree helpers ─────────────────────────

    _FIELD_REGEXES: Dict[str, str] = {
        "date": r"^[\d/\-\s\.OoIl]{4,14}$",
        "phone": r"^[\d\(\)\s\-OoIl]{7,16}$",
        "tax_id": r"^[\dOoIl\-]{7,11}$",
        "npi": r"^[\dOoIl]{9,11}$",
        "money": r"^\$?[\d,OoIl]+\.?\d{0,2}$",
        "icd10": r"^[A-Za-z]\d{1,4}\.?\d{0,2}$",
    }

    def _categorize_field(self, field_type: str, field_id: str) -> str:
        """Map schema field_type to escalation category."""
        if field_type in ("checkbox",):
            return "checkbox"
        if field_type in ("signature",):
            return "signature"
        if field_type in ("table",):
            return "table"
        if field_type in ("date", "phone", "tax_id", "npi", "money"):
            return "numeric"
        long_ids = {
            "address_full", "additional_claim", "service_lines",
            "hospitalization", "claim_info", "diagnosis_all", "claim_codes",
        }
        if any(lid in field_id.lower() for lid in long_ids):
            return "long_text"
        return "short_text"

    def _validate_field_regex(self, text: str, field_type: str) -> bool:
        """Check if OCR text passes the field-type regex."""
        pattern = self._FIELD_REGEXES.get(field_type)
        if not pattern:
            return True
        clean = (text or "").strip()
        if not clean:
            return False
        return bool(re.match(pattern, clean, re.IGNORECASE))

    def _numeric_escalation(self, crop: np.ndarray, field_type: str) -> Tuple[str, float]:
        """Multi-binarization + digit extraction for numeric fields. Never uses VLM."""
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return "", 0.0

        upscaled = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY) if upscaled.ndim == 3 else upscaled

        variants: List[np.ndarray] = []
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            _, b1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(cv2.cvtColor(b1, cv2.COLOR_GRAY2RGB))
        except Exception:
            pass
        try:
            b2 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
            )
            variants.append(cv2.cvtColor(b2, cv2.COLOR_GRAY2RGB))
        except Exception:
            pass
        try:
            inv = cv2.bitwise_not(gray)
            clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced2 = clahe2.apply(inv)
            _, b3 = cv2.threshold(enhanced2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            variants.append(cv2.cvtColor(b3, cv2.COLOR_GRAY2RGB))
        except Exception:
            pass
        variants.append(upscaled)

        def _extract_allowed(t: str) -> str:
            if field_type == "date":
                return re.sub(r"[^\d/\-\s]", "", t).strip()
            if field_type == "money":
                return re.sub(r"[^\d\$\.,]", "", t).strip()
            if field_type == "phone":
                return re.sub(r"[^\d\(\)\-\s]", "", t).strip()
            return re.sub(r"[^\d\-]", "", t).strip()

        def _fix_ocr_digits(t: str) -> str:
            return (
                t.replace("O", "0").replace("o", "0")
                .replace("I", "1").replace("l", "1")
                .replace("S", "5").replace("B", "8")
            )

        regex = self._FIELD_REGEXES.get(field_type, "")
        best_text, best_conf = "", 0.0

        for variant in variants:
            text, conf = self._trocr_ocr(variant)
            cleaned = _fix_ocr_digits(_extract_allowed(text))
            if not cleaned:
                continue
            if regex and re.match(regex, cleaned, re.IGNORECASE):
                return cleaned, max(conf, 0.80)
            if len(cleaned) > len(best_text):
                best_text, best_conf = cleaned, conf

        paddle_text, paddle_conf, _ = self._paddle_ocr(upscaled)
        if paddle_text:
            cleaned = _fix_ocr_digits(_extract_allowed(paddle_text))
            if regex and re.match(regex, cleaned, re.IGNORECASE):
                return cleaned, max(paddle_conf, 0.80)
            if len(cleaned) > len(best_text) and paddle_conf > best_conf:
                best_text, best_conf = cleaned, paddle_conf

        return best_text, best_conf

    def _compute_composite_confidence(
        self, ocr_conf: float, text: str, field_type: str, regex_ok: bool
    ) -> float:
        """Multi-factor confidence: 0.4 OCR + 0.2 regex + 0.2 charclass + 0.2 context."""
        regex_score = 1.0 if regex_ok else 0.0

        t = (text or "").strip()
        char_valid = 0.5
        if t:
            if field_type in ("date", "phone", "tax_id", "npi", "money"):
                ok = sum(c.isdigit() or c in "/-.()$ " for c in t)
                char_valid = ok / max(1, len(t))
            elif field_type in ("text", "icd10"):
                ok = sum(c.isalpha() or c.isspace() or c in ",.-'" for c in t)
                char_valid = ok / max(1, len(t))

        context_score = 1.0 if t and len(t) >= 2 else 0.3

        return float(
            0.4 * ocr_conf
            + 0.2 * regex_score
            + 0.2 * char_valid
            + 0.2 * context_score
        )

    def _llm_normalize_field(self, text: str, field_type: str) -> str:
        """Light normalization for common OCR character swaps. No hallucination risk."""
        if not text or not text.strip():
            return text
        t = text.strip()
        if field_type in ("date",):
            t = t.replace("O", "0").replace("o", "0")
            t = t.replace("I", "1").replace("l", "1")
            t = re.sub(r"\s+", "/", t)
            t = re.sub(r"/+", "/", t)
        elif field_type in ("phone",):
            t = t.replace("O", "0").replace("o", "0")
            t = t.replace("I", "1").replace("l", "1")
        elif field_type in ("tax_id", "npi"):
            t = t.replace("O", "0").replace("o", "0")
            t = t.replace("I", "1").replace("l", "1")
            t = t.replace("S", "5").replace("B", "8")
        elif field_type in ("money",):
            t = t.replace("O", "0").replace("o", "0")
            t = t.replace("I", "1").replace("l", "1")
            t = re.sub(r"[^\d\.,\$]", "", t)
        elif field_type in ("text",):
            if "," in t and len(t) < 40:
                parts = t.split(",")
                t = ",".join(p.strip().title() for p in parts)
        return t

    # ── CMS-1500 tiered OCR process ───────────────────────────────────

    def _process_cms_field(
        self,
        crop_for_ocr: np.ndarray,
        crop_raw: np.ndarray,
        block: DetectedBlock,
        field_type: str,
    ) -> DetectedBlock:
        """Tiered OCR for a single CMS-1500 scan field.

        Step 1: TrOCR primary pass (fast, GPU)
        Step 2: Field-type decision tree
        Step 3: Escalation by category (numeric -> multi-binarization, text -> flag VLM)
        """
        category = self._categorize_field(field_type, block.id or "")

        # ── Step 1: Primary TrOCR pass ────────────────────────────────
        trocr_text, trocr_conf = self._trocr_ocr(crop_for_ocr)

        if trocr_text and self._is_hallucination(trocr_text):
            trocr_text, trocr_conf = "", 0.0

        paddle_text, paddle_conf, paddle_boxes = self._paddle_ocr(crop_for_ocr)
        block.metadata["ocr_boxes"] = paddle_boxes

        if paddle_conf > trocr_conf and paddle_text:
            primary_text, primary_conf, primary_engine = paddle_text, paddle_conf, "paddleocr"
        elif trocr_text:
            primary_text, primary_conf, primary_engine = trocr_text, trocr_conf, "trocr"
        else:
            primary_text, primary_conf, primary_engine = paddle_text, paddle_conf, "paddleocr"

        # ── Step 2: Field-type decision tree ──────────────────────────
        regex_ok = self._validate_field_regex(primary_text, field_type)

        if primary_conf > 0.70 and regex_ok and primary_text:
            text = self._llm_normalize_field(primary_text, field_type)
            block.text = text
            block.confidence = self._compute_composite_confidence(
                primary_conf, text, field_type, True
            )
            block.metadata["ocr_engine"] = primary_engine
            block.metadata["escalation"] = "none"
            return block

        # ── Step 3: Escalation by category ────────────────────────────
        if category == "numeric":
            esc_text, esc_conf = self._numeric_escalation(crop_for_ocr, field_type)
            if esc_conf > primary_conf or (esc_text and not primary_text):
                text = self._llm_normalize_field(esc_text, field_type)
                r_ok = self._validate_field_regex(text, field_type)
                block.text = text
                block.confidence = self._compute_composite_confidence(
                    esc_conf, text, field_type, r_ok
                )
                block.metadata["ocr_engine"] = "numeric_escalation"
            elif primary_text:
                text = self._llm_normalize_field(primary_text, field_type)
                block.text = text
                block.confidence = self._compute_composite_confidence(
                    primary_conf, text, field_type, regex_ok
                )
                block.metadata["ocr_engine"] = primary_engine
            else:
                block.text = ""
                block.confidence = 0.0
                block.metadata["ocr_engine"] = "none"
            block.metadata["escalation"] = "numeric"
            return block

        if category in ("short_text", "long_text"):
            vlm_threshold = 0.55 if category == "short_text" else 0.50
            if primary_conf < vlm_threshold and self._has_ink(crop_for_ocr):
                block.metadata["needs_vlm"] = True
                block.metadata["best_text_before_vlm"] = primary_text
                block.metadata["best_conf_before_vlm"] = primary_conf
            block.text = primary_text
            block.confidence = self._compute_composite_confidence(
                primary_conf, primary_text, field_type, regex_ok
            )
            block.metadata["ocr_engine"] = primary_engine
            block.metadata["escalation"] = category
            return block

        block.text = primary_text
        block.confidence = primary_conf
        block.metadata["ocr_engine"] = primary_engine
        return block

    # ── Phased process_blocks ─────────────────────────────────────────

    async def process_blocks(self, image: np.ndarray, blocks: List[DetectedBlock]) -> List[DetectedBlock]:
        """Phased OCR: fast primary pass, then targeted VLM escalation on <30% of fields.

        Phase 1 — Primary OCR + field-type decision tree (TrOCR/PaddleOCR per field)
        Phase 2 — Targeted VLM only on text fields flagged needs_vlm (capped at 30%)
        Phase 3 — LLM normalization on all text fields
        """
        await self.initialize()
        t0 = time.time()

        # ── Phase 1: Primary OCR on every field ──────────────────────
        results: List[DetectedBlock] = []
        for block in blocks:
            result = await self.process(image, block)
            results.append(result)
        phase1_dt = time.time() - t0

        # ── Phase 2: Targeted VLM escalation ─────────────────────────
        vlm_candidates = [
            b for b in results
            if (b.metadata or {}).get("needs_vlm")
            and b.block_type not in (BlockType.CHECKBOX, BlockType.SIGNATURE, BlockType.TABLE)
        ]
        max_vlm = max(2, int(len(results) * 0.30))
        vlm_count = 0
        for block in vlm_candidates[:max_vlm]:
            bx0, by0, bx1, by1 = [int(v) for v in block.bbox]
            ih, iw = image.shape[:2]
            bx0, by0 = max(0, bx0), max(0, by0)
            bx1, by1 = min(iw, bx1), min(ih, by1)
            crop = image[by0:by1, bx0:bx1]
            if crop.size == 0:
                continue
            is_cms = (block.metadata or {}).get("form_type", "") in ("cms-1500", "CMS1500", "cms1500")
            if is_cms:
                try:
                    from src.processing.preprocessing import remove_red_template_text
                    crop = remove_red_template_text(crop)
                except Exception:
                    pass
            ch, cw = crop.shape[:2]
            if max(ch, cw) < 256:
                scale = min(1024.0 / max(ch, cw), 3.0)
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            vlm_text, vlm_conf = self._vlm_ocr(crop)
            if vlm_text and vlm_conf > block.confidence:
                block.text = vlm_text
                block.confidence = vlm_conf
                block.metadata["ocr_engine"] = "vlm_escalation"
                block.metadata.pop("needs_vlm", None)
                vlm_count += 1

        # ── Phase 3: LLM normalization on all text fields ────────────
        for block in results:
            if block.text and block.block_type not in (BlockType.CHECKBOX, BlockType.SIGNATURE, BlockType.TABLE):
                ft = (block.metadata or {}).get("field_type", "text")
                block.text = self._llm_normalize_field(block.text, ft)

        dt = time.time() - t0
        print(
            f"[OCR] Tiered pipeline: {len(results)} fields in {dt:.1f}s "
            f"(Phase1={phase1_dt:.1f}s, VLM={vlm_count}/{len(vlm_candidates)} escalated)"
        )

        # ── Filtering ────────────────────────────────────────────────
        filtered = []
        for block in results:
            src = (block.metadata or {}).get("source", "")
            if src in ("schema_zones", "ocr_zone_matching", "cms1500_production", "full_page_ocr"):
                filtered.append(block)
                continue
            if block.block_type in (BlockType.CHECKBOX, BlockType.SIGNATURE):
                filtered.append(block)
                continue
            if block.text and len(str(block.text).strip()) > 0:
                filtered.append(block)
                continue
            if block.confidence >= 0.3:
                filtered.append(block)

        return filtered

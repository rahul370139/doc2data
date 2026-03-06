"""
OCR Agent - tiered OCR for text extraction from blocks.

PURPOSE: Runs PaddleOCR on printed text, TrOCR on handwriting/signatures,
checkbox fill-ratio detector on checkbox blocks. Validates and normalizes
output. Escalates to VLM when confidence is low.

USE CASE: Pipeline calls this after layout detection. Converts image
regions to text. No manual use; part of MultiAgentPipeline.

Contains:
- PaddleOCRWrapper: low-level PaddleOCR engine (extract_text, extract_text_lines)
- OCRAgent: high-level orchestration (TrOCR, VLM, checkbox, tiered pipeline)
"""
from __future__ import annotations

# CRITICAL: Must be set BEFORE any `from transformers import ...` to prevent
# a broken TensorFlow/ml_dtypes install from crashing the TrOCR import chain.
# The NVIDIA PyTorch containers (nvcr.io/nvidia/pytorch:*) ship TF by default
# and the bundled ml_dtypes is often incompatible.
import os as _os
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import asyncio
import json
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.pipelines.core import BaseAgent, BlockType, DetectedBlock, PipelineConfig

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.config import Config
from utils.models import WordBox


# =============================================================================
# Florence-2-large: lazy-loaded OCR rescue (merged from florence2_box24)
# =============================================================================
_florence2_model = None
_florence2_processor = None
_florence2_load_attempts = 0
_florence2_max_attempts = 2
_florence2_lock = threading.Lock()


def _load_florence2() -> bool:
    """Lazy-load Florence-2-large with multi-strategy fallback.

    Uses ``florence-community/Florence-2-large`` which ships converted weights
    compatible with native ``Florence2ForConditionalGeneration`` (no
    ``trust_remote_code`` needed).  The processor is loaded via
    ``AutoProcessor`` which correctly initialises the ``RobertaTokenizer``
    with the ``image_token`` attribute already set.
    """
    global _florence2_model, _florence2_processor, _florence2_load_attempts
    with _florence2_lock:
        if _florence2_model is not None and _florence2_processor is not None:
            return True
        if _florence2_load_attempts >= _florence2_max_attempts:
            return False
        _florence2_load_attempts += 1

        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as e:
            print(f"[Florence2] Required classes not available: {e}")
            return False

        has_cuda = torch.cuda.is_available()
        model_id = "florence-community/Florence-2-large"

        # (local_only, device)
        strategies: list = []
        if has_cuda:
            strategies.append((True, "cuda:0"))
            strategies.append((False, "cuda:0"))
        strategies.append((True, "cpu"))
        strategies.append((False, "cpu"))

        for local_only, device in strategies:
            dtype = torch.float16 if "cuda" in device else torch.float32
            tag = f"{model_id}/{device}/local={local_only}"
            try:
                print(f"[Florence2] Trying {tag} ...")

                proc = AutoProcessor.from_pretrained(
                    model_id, local_files_only=local_only,
                )

                mdl = Florence2ForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    local_files_only=local_only,
                ).eval().to(device)

                _florence2_processor = proc
                _florence2_model = mdl
                print(f"[Florence2] ✅ Loaded on {next(mdl.parameters()).device}")
                return True
            except Exception as e:
                short_err = str(e).split('\n')[0][:200]
                print(f"[Florence2] Failed {tag}: {short_err}")
                continue

        print("[Florence2] All strategies failed.")
        return False


def _florence2_ocr_run(image: np.ndarray) -> Optional[str]:
    """Run Florence-2 <OCR> on crop. Returns None if unavailable or empty."""
    if not _load_florence2() or image is None or image.size == 0:
        return None
    try:
        import torch
        from PIL import Image as PILImage
        if image.ndim == 2:
            pil_img = PILImage.fromarray(image).convert("RGB")
        elif image.ndim == 3 and image.shape[2] == 4:
            pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
        else:
            pil_img = PILImage.fromarray(image)
        device = next(_florence2_model.parameters()).device
        dtype = next(_florence2_model.parameters()).dtype
        task = "<OCR>"
        inputs = _florence2_processor(
            text=task, images=pil_img, return_tensors="pt",
        ).to(device, dtype)
        with torch.no_grad():
            gen = _florence2_model.generate(
                **inputs,
                max_new_tokens=1024, do_sample=False, num_beams=3,
            )
        raw_text = _florence2_processor.batch_decode(gen, skip_special_tokens=False)[0]
        parsed = _florence2_processor.post_process_generation(
            raw_text, task=task, image_size=(pil_img.width, pil_img.height),
        )

        result_text = None
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, str) and v.strip():
                    result_text = v.strip()
                    break
        elif isinstance(parsed, str) and parsed.strip():
            result_text = parsed.strip()

        if result_text:
            print(f"[Florence2] OCR result: '{result_text[:80]}'")
            return result_text
    except Exception as e:
        print(f"[Florence2] OCR error: {e}")
    return None


# =============================================================================
# PaddleOCRWrapper - low-level PaddleOCR engine
# =============================================================================

class PaddleOCRWrapper:
    """Wrapper for PaddleOCR with lazy initialization and optimized settings."""

    def __init__(
        self,
        lang: str = 'en',
        use_angle_cls: bool = True,
        use_gpu: Optional[bool] = None
    ):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        if use_gpu is None:
            detected = False
            try:
                import paddle
                detected = bool(getattr(paddle, "is_compiled_with_cuda", lambda: False)())
                if detected:
                    try:
                        detected = paddle.device.cuda.device_count() > 0
                    except Exception:
                        detected = True
            except Exception:
                detected = False
            self.use_gpu = bool(detected or Config.USE_GPU)
        else:
            self.use_gpu = use_gpu
        self.ocr = None
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return
        print("🔄 Initializing PaddleOCR (this may take 30-60 seconds on first run)...")
        import time
        start_time = time.time()
        try:
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(
                    lang=self.lang,
                    use_angle_cls=self.use_angle_cls,
                    use_gpu=self.use_gpu,
                    show_log=False
                )
            except Exception:
                from paddleocr import PaddleOCR
                print("⚠️ Retrying PaddleOCR init with minimal args...")
                try:
                    self.ocr = PaddleOCR(lang=self.lang, use_angle_cls=self.use_angle_cls, use_gpu=self.use_gpu)
                except Exception:
                    self.ocr = PaddleOCR(lang=self.lang, use_angle_cls=self.use_angle_cls)
            init_time = time.time() - start_time
            print(f"✅ PaddleOCR initialized in {init_time:.2f}s")
            self._initialized = True
        except Exception as e:
            print(f"❌ Error initializing PaddleOCR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def extract_text(self, image: np.ndarray) -> List[WordBox]:
        if not self._initialized:
            self._initialize()
        word_boxes = []
        try:
            if image is None or image.size == 0:
                return word_boxes
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] != 3:
                return word_boxes
            if image.shape[0] < 10 or image.shape[1] < 10:
                return word_boxes
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
            try:
                result = self.ocr.predict(image)
            except AttributeError:
                result = self.ocr.ocr(image)
            if result is None:
                return word_boxes
            if isinstance(result, np.ndarray) and result.size == 0:
                return word_boxes
            if not result:
                return word_boxes
            processed_results = []
            raw_list = result if isinstance(result, list) else [result]
            for item in raw_list:
                if item is None:
                    continue
                if "OCRResult" in str(type(item)):
                    try:
                        json_data = {}
                        if hasattr(item, 'json'):
                            j = item.json
                            json_data = j() if callable(j) else j
                        elif hasattr(item, '__getitem__'):
                            processed_results.append(item)
                            continue
                        if isinstance(json_data, str):
                            json_data = json.loads(json_data)
                        if isinstance(json_data, dict):
                            res = json_data.get('res', json_data)
                            if isinstance(res, dict):
                                json_data = {
                                    'rec_texts': res.get('rec_texts', res.get('rec_text', [])),
                                    'rec_boxes': res.get('dt_polys', res.get('rec_polys', [])),
                                    'rec_scores': res.get('rec_scores', res.get('rec_score', []))
                                }
                            processed_results.append(json_data)
                    except Exception:
                        pass
                else:
                    processed_results.append(item)
            result = processed_results if processed_results else raw_list
            for page_result in result:
                if page_result is None:
                    continue
                is_mapping = isinstance(page_result, (dict, Mapping))
                if is_mapping:
                    texts = list(page_result.get("rec_texts") or [])
                    scores_raw = page_result.get("rec_scores")
                    scores = [1.0] * len(texts) if scores_raw is None else (
                        scores_raw.tolist() if isinstance(scores_raw, np.ndarray) and scores_raw.size > 0 else list(scores_raw) if scores_raw else [1.0] * len(texts)
                    )
                    if len(scores) != len(texts):
                        scores = [1.0] * len(texts)
                    boxes = None
                    for key in ["dt_polys", "rec_polys", "rec_boxes", "text_word_boxes"]:
                        box_data = page_result.get(key)
                        if box_data is not None:
                            if isinstance(box_data, np.ndarray) and box_data.size > 0:
                                boxes = box_data.tolist()
                                break
                            elif box_data:
                                boxes = list(box_data) if not isinstance(box_data, list) else box_data
                                break
                    if boxes is None:
                        boxes = []
                    min_len = min(len(texts), len(boxes), len(scores))
                    for idx in range(min_len):
                        text = texts[idx]
                        score = scores[idx] if idx < len(scores) else 1.0
                        box = boxes[idx] if idx < len(boxes) else None
                        if box is None:
                            continue
                        raw_text = text[0] if isinstance(text, (list, tuple)) else text
                        if isinstance(raw_text, np.ndarray):
                            if raw_text.size == 0:
                                continue
                            raw_text = raw_text.item() if raw_text.size == 1 else " ".join(map(str, raw_text.flatten().tolist()))
                        text_str = str(raw_text).strip()
                        if not text_str:
                            continue
                        raw_score = score[0] if isinstance(score, (list, tuple)) else score
                        if isinstance(raw_score, np.ndarray):
                            raw_score = raw_score.item() if raw_score.size == 1 else np.mean(raw_score)
                        confidence_val = float(raw_score)
                        if isinstance(box, np.ndarray):
                            box = box.tolist()
                        if isinstance(box, (list, tuple)) and len(box) >= 4:
                            if isinstance(box[0], (list, tuple)):
                                x_coords = [pt[0] for pt in box]
                                y_coords = [pt[1] for pt in box]
                                x0, x1 = min(x_coords), max(x_coords)
                                y0, y1 = min(y_coords), max(y_coords)
                            else:
                                x0, y0, x1, y1 = box[:4]
                            word_boxes.append(WordBox(text=text_str, bbox=(float(x0), float(y0), float(x1), float(y1)), confidence=confidence_val))
                    continue
                if isinstance(page_result, (list, tuple)):
                    for line in page_result:
                        if line is None or not isinstance(line, (list, tuple)) or len(line) < 2:
                            continue
                        try:
                            bbox_coords, text_info = line[0], line[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text, confidence = text_info[0], text_info[1]
                            elif isinstance(text_info, dict):
                                text, confidence = text_info.get('text', ''), text_info.get('confidence', 1.0)
                            else:
                                continue
                            if isinstance(bbox_coords, (list, tuple)) and len(bbox_coords) >= 4:
                                if isinstance(bbox_coords[0], (list, tuple)):
                                    x_coords = [pt[0] for pt in bbox_coords]
                                    y_coords = [pt[1] for pt in bbox_coords]
                                    x0, y0, x1, y1 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                                else:
                                    x0, y0, x1, y1 = bbox_coords[:4]
                            else:
                                continue
                            raw_text = text[0] if isinstance(text, (list, tuple)) else text
                            if isinstance(raw_text, np.ndarray):
                                if raw_text.size == 0:
                                    continue
                                raw_text = raw_text.item() if raw_text.size == 1 else " ".join(map(str, raw_text.flatten().tolist()))
                            text_str = str(raw_text).strip()
                            if not text_str:
                                continue
                            raw_conf = confidence[0] if isinstance(confidence, (list, tuple)) else confidence
                            if isinstance(raw_conf, np.ndarray):
                                raw_conf = raw_conf.item() if raw_conf.size == 1 else np.mean(raw_conf)
                            confidence_val = float(raw_conf)
                            word_boxes.append(WordBox(text=text_str, bbox=(float(x0), float(y0), float(x1), float(y1)), confidence=confidence_val))
                        except (ValueError, TypeError, IndexError):
                            continue
        except Exception as e:
            import traceback
            traceback.print_exc()
        return word_boxes

    def extract_text_lines(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        word_boxes = self.extract_text(image)
        lines, current_line, current_y = [], [], None
        for word_box in sorted(word_boxes, key=lambda wb: (wb.bbox[1], wb.bbox[0])):
            y_center = (word_box.bbox[1] + word_box.bbox[3]) / 2
            if current_y is None or abs(y_center - current_y) > 10:
                if current_line:
                    line_text = " ".join([wb.text for wb in current_line])
                    line_conf = sum([wb.confidence for wb in current_line]) / len(current_line)
                    x0 = min([wb.bbox[0] for wb in current_line])
                    y0 = min([wb.bbox[1] for wb in current_line])
                    x1 = max([wb.bbox[2] for wb in current_line])
                    y1 = max([wb.bbox[3] for wb in current_line])
                    lines.append((line_text, line_conf, (x0, y0, x1, y1)))
                current_line, current_y = [word_box], y_center
            else:
                current_line.append(word_box)
        if current_line:
            line_text = " ".join([wb.text for wb in current_line])
            line_conf = sum([wb.confidence for wb in current_line]) / len(current_line)
            x0 = min([wb.bbox[0] for wb in current_line])
            y0 = min([wb.bbox[1] for wb in current_line])
            x1 = max([wb.bbox[2] for wb in current_line])
            y1 = max([wb.bbox[3] for wb in current_line])
            lines.append((line_text, line_conf, (x0, y0, x1, y1)))
        return lines


# =============================================================================
# OCRAgent - high-level orchestration
# =============================================================================

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
        self._template_rgb_cache: Dict[str, np.ndarray] = {}
        self._trocr_lock = threading.Lock()  # PyTorch/TrOCR is not thread-safe
    
    async def initialize(self):
        if self._initialized:
            return
        
        try:
            self._paddle = PaddleOCRWrapper()
            self.log("PaddleOCRWrapper initialized")
        except Exception as e:
            self.log(f"PaddleOCR init failed: {e}")
        
        # Pre-load Florence-2 so it's ready for multi-engine consensus.
        # This avoids cold-start latency on the first field.
        try:
            loaded = _load_florence2()
            if loaded:
                self.log("Florence-2 pre-loaded successfully")
            else:
                self.log("Florence-2 pre-load failed — rescue will be unavailable")
        except Exception as e:
            self.log(f"Florence-2 pre-load error: {e}")

        self._initialized = True

    def _get_template_rgb(self, form_type: Optional[str] = None) -> Optional[np.ndarray]:
        """Load and cache the CMS-1500 template in RGB via the registrar.

        The registrar already renders the PDF→RGB at 300 DPI and caches it.
        The aligned scan is warped into this same coordinate space, so a
        pixel-wise subtraction is valid.
        """
        key = form_type or "cms-1500"
        if key in self._template_rgb_cache:
            return self._template_rgb_cache[key]
        try:
            from src.pipelines.registration import get_cms1500_registrar
            registrar = get_cms1500_registrar()
            tdata = registrar.get_template_data()
            rgb = tdata.get("image_rgb")
            if rgb is not None and rgb.size > 0:
                self._template_rgb_cache[key] = rgb
                return rgb
        except Exception:
            pass
        return None

    def _get_template_gray(self, form_type: Optional[str]) -> Optional[np.ndarray]:
        """Load and cache a grayscale template image for template-diff OCR."""
        if not form_type:
            return None
        key = str(form_type)
        if key in self._templates:
            return self._templates[key]
        rgb = self._get_template_rgb(form_type)
        if rgb is not None:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            self._templates[key] = gray
            return gray
        return None

    def _template_subtract(
        self,
        scan_crop: np.ndarray,
        template_rgb: np.ndarray,
        bbox_px: Tuple[int, int, int, int],
        diff_threshold: int = 20,
    ) -> Tuple[np.ndarray, float]:
        """Template subtraction: remove template pixels, keep only ink.

        Returns ``(clean_rgb, ink_ratio)`` where *ink_ratio* is the fraction
        of pixels that differ from the template (i.e. likely contain ink).
        The caller uses *ink_ratio* together with PaddleOCR results to decide
        whether a field is genuinely blank.
        """
        x0, y0, x1, y1 = bbox_px
        th, tw = template_rgb.shape[:2]

        x0c = max(0, min(x0, tw))
        y0c = max(0, min(y0, th))
        x1c = max(0, min(x1, tw))
        y1c = max(0, min(y1, th))

        crop_h, crop_w = scan_crop.shape[:2]
        tmpl_h, tmpl_w = y1c - y0c, x1c - x0c

        if tmpl_h < 4 or tmpl_w < 4 or crop_h < 4 or crop_w < 4:
            return scan_crop, 1.0

        tmpl_crop_rgb = template_rgb[y0c:y1c, x0c:x1c]

        if tmpl_crop_rgb.shape[:2] != scan_crop.shape[:2]:
            tmpl_crop_rgb = cv2.resize(tmpl_crop_rgb, (crop_w, crop_h), interpolation=cv2.INTER_AREA)

        tmpl_blur = cv2.GaussianBlur(tmpl_crop_rgb, (5, 5), 0)

        scan_gray = cv2.cvtColor(scan_crop, cv2.COLOR_RGB2GRAY) if scan_crop.ndim == 3 else scan_crop
        tmpl_gray = cv2.cvtColor(tmpl_blur, cv2.COLOR_RGB2GRAY) if tmpl_blur.ndim == 3 else tmpl_blur

        diff = cv2.absdiff(scan_gray, tmpl_gray)

        _, ink_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        ink_ratio = float(np.count_nonzero(ink_mask)) / max(1, ink_mask.size)

        clean_gray = np.full_like(scan_gray, 255, dtype=np.uint8)
        clean_gray[ink_mask > 0] = scan_gray[ink_mask > 0]

        return cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2RGB), ink_ratio

    def _template_diff_crop(self, crop_rgb: np.ndarray, template_gray: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Legacy template-diff (grayscale). Used for non-CMS forms and checkboxes."""
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
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            _, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
            return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
        except Exception:
            return None
    
    def _load_trocr(self):
        """Lazy-load TrOCR.  Tries large model first, falls back to base."""
        if self._trocr_model is not None:
            return
        if getattr(self, "_trocr_load_failed", False):
            return

        import os
        os.environ.setdefault("USE_TF", "0")

        model_name = getattr(self.config, "trocr_model", "large")
        candidates = (
            ["microsoft/trocr-large-handwritten", "microsoft/trocr-base-handwritten"]
            if model_name == "large"
            else ["microsoft/trocr-base-handwritten"]
        )

        for hf_id in candidates:
            try:
                import torch
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel

                self._trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"🔄 Loading TrOCR ({hf_id}) on {self._trocr_device} ...")
                t0 = time.time()
                self._trocr_processor = TrOCRProcessor.from_pretrained(hf_id)
                # use_safetensors=True bypasses CVE-2025-32434 torch.load restriction
                self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    hf_id, use_safetensors=True
                )
                self._trocr_model = self._trocr_model.to(self._trocr_device).eval()
                dt = time.time() - t0
                self._trocr_load_failed = False
                print(f"✅ TrOCR loaded ({hf_id}) on {self._trocr_device} in {dt:.1f}s")
                self.log(f"TrOCR loaded ({hf_id}) on {self._trocr_device}")
                return
            except Exception as e:
                print(f"⚠️ TrOCR load failed for {hf_id}: {e}")
                import traceback
                traceback.print_exc()

        # All candidates exhausted
        self._trocr_device = "cpu"
        self._trocr_load_failed = True
        print("❌ TrOCR: all model candidates failed to load — TrOCR will be unavailable")
        self.log("TrOCR load failed for all candidates")
    
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
    
    def _preprocess_for_trocr(self, image: np.ndarray) -> np.ndarray:
        """Minimal preprocessing for TrOCR.

        TrOCR's ViT encoder handles its own normalisation and resizing to
        384x384.  Heavy processing (morphological ops, binarisation, aggressive
        sharpening) *destroys* the stroke details the model relies on.

        We only do two things:
        1. Up-scale tiny crops so the ViT encoder has enough pixels to work with.
        2. Strip residual colour (convert to grayscale → RGB) so any leftover
           red-template artefacts don't confuse the model.
        """
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]

        # Up-scale tiny crops — TrOCR's ViT input is 384x384; a 20px-high
        # crop gets stretched to mush.  2x–3x cubic resize preserves strokes.
        if h < 48:
            scale = max(2.0, 48.0 / h)
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

        # Grayscale round-trip removes residual colour artefacts
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Light CLAHE — boosts faint pencil/ballpoint strokes without clipping
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    def _trocr_ocr(self, image: np.ndarray, preprocess: bool = True) -> Tuple[str, float]:
        """Run TrOCR inference with per-token confidence.

        Uses greedy decoding (num_beams=1) so output_scores gives reliable
        per-step probabilities for confidence estimation.
        """
        if not self.config.enable_trocr:
            return "", 0.0

        self._load_trocr()

        if self._trocr_model is None:
            return "", 0.0
        if image is None or image.size == 0:
            return "", 0.0

        try:
            from PIL import Image as PILImage
            import torch

            processed = self._preprocess_for_trocr(image) if preprocess else image

            # Ensure 3-channel RGB
            if processed.ndim == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            elif processed.ndim == 3 and processed.shape[2] == 4:
                processed = cv2.cvtColor(processed, cv2.COLOR_RGBA2RGB)

            pil_img = PILImage.fromarray(processed)
            device = getattr(self, "_trocr_device", "cpu")

            with self._trocr_lock:
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

            # Per-token confidence from greedy scores
            avg_conf = 0.75
            if text and hasattr(outputs, "scores") and outputs.scores:
                probs = [
                    torch.softmax(s, dim=-1).max(dim=-1).values.item()
                    for s in outputs.scores
                ]
                avg_conf = sum(probs) / len(probs) if probs else 0.5

            # Fallback: if preprocessing blanked the crop, retry without it
            if not text and preprocess:
                return self._trocr_ocr(image, preprocess=False)

            return text, float(avg_conf)

        except Exception as e:
            self.log(f"TrOCR inference error: {e}")
            import traceback
            traceback.print_exc()

        return "", 0.0
    
    @property
    def VLM_MODEL_RESCUE(self):
        return Config.VLM_MODEL_RESCUE

    def _vlm_ocr_field(
        self, image: np.ndarray, field_name: str = "",
        field_type: str = "text", model: str = "",
        timeout: int = 60,
    ) -> Tuple[str, float]:
        """Task-routed VLM call via Ollama.

        Each task type uses the optimal model:
        - minicpm-v: VLM rescue for fields Florence-2 missed (5.5GB)
        - llava: general-purpose fallback (4.7GB)
        """
        try:
            import requests
            import base64

            if not model:
                model = self.VLM_MODEL_RESCUE

            _, buffer = cv2.imencode(
                ".jpg",
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image,
            )
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            type_hints = {
                "date": "a date (MM/DD/YYYY format)",
                "date_range": "a date range with FROM and TO dates (MM/DD/YYYY - MM/DD/YYYY)",
                "phone": "a phone number",
                "name": "a person's name",
                "address": "an address",
                "npi": "a 10-digit NPI number",
                "tax_id": "a tax ID or SSN",
                "money": "a dollar amount",
            }
            hint = type_hints.get(field_type, "text or numbers")

            prompt = f"""Extract the handwritten text from this form field.
Field: {field_name}
Expected content: {hint}

Rules:
- Output ONLY the extracted text
- If empty or blank, output: EMPTY
- No explanation needed

Text:"""

            resp = requests.post(
                f"http://{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 100},
                },
                timeout=timeout,
            )

            if resp.ok:
                text = (resp.json().get("response") or "").strip()
                if not text or text.upper() in (
                    "EMPTY", "BLANK", "N/A", "NONE", "[EMPTY]", "[BLANK]",
                ):
                    return "", 0.0
                text = text.strip('"\'`')
                if self._is_hallucination(text, field_type):
                    return "", 0.0
                self.log(f"VLM({model}) extracted: '{text[:30]}...' for {field_name}")
                return text, 0.80

        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            self.log(f"VLM({model}) OCR failed: {e}")
        return "", 0.0

    
    def _florence2_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Use Florence-2-large <OCR> on a crop. Returns (text, confidence)."""
        if image is None or image.size == 0:
            return "", 0.0
        text = _florence2_ocr_run(image)
        if not text:
            return "", 0.0
        if self._is_hallucination(text):
            self.log(f"Florence-2 filtered (hallucination): '{text[:60]}'")
            return "", 0.0
        _PLACEHOLDER_CHARS = frozenset({"-", "–", "—", ".", "_", "|", "~", "*"})
        if text.strip() in _PLACEHOLDER_CHARS:
            self.log(f"Florence-2 filtered (placeholder): '{text}'")
            return "", 0.0
        self.log(f"Florence-2 OCR: '{text[:50]}'")
        return text, 0.82
    
    def _detect_checkbox(self, image: np.ndarray) -> Tuple[bool, float]:
        """Detect if checkbox is checked. Returns (is_checked, ink_ratio).

        The ink_ratio is returned raw so the pipeline can compare paired
        checkboxes (yes/no) and only mark the higher one.
        """
        if image is None or image.size == 0:
            return False, 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9
        )

        h, w = binary.shape[:2]
        pad_x = int(w * 0.22)
        pad_y = int(h * 0.22)
        inner = binary[pad_y:max(pad_y + 1, h - pad_y), pad_x:max(pad_x + 1, w - pad_x)]
        if inner.size == 0:
            inner = binary

        ink = np.count_nonzero(inner)
        area = max(inner.size, 1)
        ink_ratio = ink / area

        is_checked = ink_ratio > 0.045
        return is_checked, ink_ratio
    
    def _has_ink(self, img: np.ndarray, threshold: float = 0.015) -> bool:
        """Check if crop has visible ink (handwritten/printed content)."""
        if img is None or img.size == 0:
            return False
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        g = cv2.GaussianBlur(g, (3, 3), 0)
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9)
        ink_ratio = float(np.count_nonzero(bw) / max(1, bw.size))
        return ink_ratio > threshold

    def _has_meaningful_content(self, crop: np.ndarray, min_components: int = 2) -> bool:
        """Detect real handwritten content via Connected Component Analysis.

        Real handwriting creates multiple large, structured stroke components
        in the interior of the crop.  Template residue after subtraction
        creates scattered tiny specks, often at borders.

        This is structurally more robust than a single ink-ratio threshold
        because it measures the *shape* of the ink, not just the *amount*.

        Args:
            crop: template-subtracted crop (RGB or gray)
            min_components: minimum number of significant interior components
                            to classify as "has content" (2=strict, 1=lenient)
        """
        if crop is None or crop.size == 0:
            return False
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 9,
        )

        # Shave margins to ignore border-line artifacts
        mx = max(3, int(w * 0.10))
        my = max(3, int(h * 0.12))
        interior = binary[my:h - my, mx:w - mx]
        if interior.size == 0:
            return False

        # Minimum component area: at least 0.1% of crop or 20px
        min_area = max(20, int(h * w * 0.001))

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            interior, connectivity=8,
        )

        significant = 0
        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            # Filter extremely elongated components (border line remnants)
            cw = max(stats[i, cv2.CC_STAT_WIDTH], 1)
            ch = max(stats[i, cv2.CC_STAT_HEIGHT], 1)
            if max(cw, ch) / min(cw, ch) > 12:
                continue
            significant += 1
            if significant >= min_components:
                return True

        return False

    def _is_truly_blank(self, crop: np.ndarray, variance_threshold: float = 150.0) -> bool:
        """Structural blank detection using Laplacian variance.

        Raw ink_ratio treats scanner noise and box borders the same as
        handwriting.  Laplacian variance measures *edge density* — real
        handwriting strokes have high, structured edge energy (>300) while
        scanner dust and smooth box lines have low variance (<100).

        Steps:
          1. Shave 8% margins to discard bounding-box border lines.
          2. Compute Laplacian variance on the interior region.
          3. Below threshold → truly blank (no structured strokes).
        """
        if crop is None or crop.size == 0:
            return True
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return True
        mx = max(3, int(w * 0.08))
        my = max(3, int(h * 0.08))
        if h <= my * 2 + 4 or w <= mx * 2 + 4:
            return True
        shaved = crop[my:h - my, mx:w - mx]
        gray = cv2.cvtColor(shaved, cv2.COLOR_RGB2GRAY) if shaved.ndim == 3 else shaved
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return lap_var < variance_threshold
    
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
    
    def _process_sync(self, image: np.ndarray, block: DetectedBlock) -> DetectedBlock:
        """Sync OCR for a single block (used by parallel process_blocks). Assumes initialize() already called."""
        return self._process_impl(image, block)

    async def process(self, image: np.ndarray, block: DetectedBlock) -> DetectedBlock:
        """OCR a single block with appropriate method."""
        await self.initialize()
        return self._process_impl(image, block)

    def _process_impl(self, image: np.ndarray, block: DetectedBlock) -> DetectedBlock:
        """Core OCR logic for a single block."""
        # TABLE blocks (Box 24 etc.) are handled by labeling_agent.process_table()
        # They must NOT go through TrOCR/PaddleOCR — that produces garbage on tables.
        if block.block_type == BlockType.TABLE:
            block.metadata["ocr_engine"] = "skipped_table_route"
            return block

        # If this block already has text from full-page OCR zone matching, do NOT re-OCR tiny crops.
        # EXCEPTION: date/date_range fields — zone matching + _clean_field_value strips
        # leading digits (e.g. "10/15/1973" → "/15/1973"). Always run VLM for dates.
        src = (block.metadata or {}).get("source")
        field_type_early = (block.metadata or {}).get("field_type", "")
        if src == "ocr_zone_matching" and block.block_type not in (BlockType.CHECKBOX, BlockType.SIGNATURE):
            if field_type_early not in ("date", "date_range"):
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
        
        # Checkbox detection — use template subtraction to isolate the check mark
        if block.block_type == BlockType.CHECKBOX:
            form_type_cb = (block.metadata or {}).get("form_type", "")
            is_cms_cb = form_type_cb in ("cms-1500", "CMS1500", "cms1500")
            diff_img = None
            if is_cms_cb:
                template_rgb = self._get_template_rgb(form_type_cb)
                if template_rgb is not None:
                    bbox_px = (int(x0_p), int(y0_p), int(x1_p), int(y1_p))
                    diff_img, _ = self._template_subtract(crop, template_rgb, bbox_px, diff_threshold=25)
            if diff_img is None:
                tmpl = self._get_template_gray(form_type_cb)
                diff_img = self._template_diff_crop(crop, tmpl, (x0_p, y0_p, x1_p, y1_p)) if tmpl is not None else None
            is_checked, ink_ratio = self._detect_checkbox(diff_img if diff_img is not None else crop)
            block.text = "X" if is_checked else ""
            block.confidence = float(min(1.0, max(0.0, (ink_ratio - 0.02) / 0.08)))
            block.metadata["ocr_engine"] = "checkbox_detector"
            block.metadata["checkbox_ink_ratio"] = round(ink_ratio, 4)
            return block
        
        # Signature: mark as signed only, no OCR (user requirement)
        if block.block_type == BlockType.SIGNATURE:
            form_type_sig = (block.metadata or {}).get("form_type", "")
            is_cms_sig = form_type_sig in ("cms-1500", "CMS1500", "cms1500")
            if is_cms_sig:
                template_rgb = self._get_template_rgb(form_type_sig)
                if template_rgb is not None:
                    bbox_px = (int(x0_p), int(y0_p), int(x1_p), int(y1_p))
                    clean_sig, sig_ink = self._template_subtract(crop, template_rgb, bbox_px)
                    has_ink = sig_ink > 0.008
                else:
                    has_ink = self._has_ink(crop, threshold=0.02)
            else:
                has_ink = self._has_ink(crop, threshold=0.02)
            if has_ink:
                block.text = "[SIGNED]"
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                density = np.count_nonzero(binary) / binary.size
                block.confidence = min(1.0, density * 5)
            else:
                block.text = ""
                block.confidence = 0.0
                block.metadata["is_blank"] = True
            block.metadata["ocr_engine"] = "signature_detector"
            block.metadata["has_ink"] = has_ink
            return block
        
        # ── Template subtraction / red removal ──────────────────────────
        crop_for_ocr = crop
        ink_ratio = 1.0  # default: assume ink present
        form_type_meta = (block.metadata or {}).get("form_type", "")
        is_cms_scan = form_type_meta in ("cms-1500", "CMS1500", "cms1500")
        tmpl = None

        if is_cms_scan:
            template_rgb = self._get_template_rgb(form_type_meta)
            if template_rgb is not None:
                bbox_px = (int(x0_p), int(y0_p), int(x1_p), int(y1_p))
                crop_for_ocr, ink_ratio = self._template_subtract(crop, template_rgb, bbox_px)
                block.metadata["template_subtract_used"] = True
                block.metadata["ink_ratio"] = round(ink_ratio, 4)
            else:
                try:
                    from src.processing.preprocessing import remove_red_template_text
                    crop_for_ocr = remove_red_template_text(crop)
                    block.metadata["red_removal_fallback"] = True
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
            block = self._process_cms_field(crop_for_ocr, crop, block, field_type, ink_ratio)
            if block.text:
                field_name = (block.metadata or {}).get("field_name", "") or block.id
                block.text = self._strip_template_bleed(block.text, field_name)
            return block

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
            vlm_text, vlm_conf = self._vlm_ocr_field(crop_for_ocr, model=self.VLM_MODEL_RESCUE, timeout=90)
            if vlm_text and vlm_conf > best_conf:
                best_text, best_conf, best_engine = vlm_text, vlm_conf, "vlm_ocr"

        block.text = best_text
        block.confidence = best_conf
        block.metadata["ocr_engine"] = best_engine
        return block
    
    def _is_hallucination(self, text: str, field_type: str = "") -> bool:
        """Structural hallucination detection — no phrase lists.

        Template text is already eliminated at the pixel level by
        ``_template_subtract``.  This function catches *structural* anomalies
        that indicate the OCR model fabricated text from noise:
          - too many words for a short form field
          - repeated characters / words
          - low alphanumeric ratio (garbage symbols)
          - excessive punctuation density
        """
        if not text or len(text.strip()) < 2:
            return False

        text = text.strip()
        words = text.split()

        # Form fields are short: names, dates, IDs, codes, short addresses.
        max_words = 14 if field_type in ("address", "long_text") else 8
        if len(words) > max_words:
            return True

        # Very short text with only 1-2 unique characters → noise
        if len(text) > 5 and len(text) <= 15:
            if len(set(text.replace(' ', '').replace('.', ''))) <= 2:
                return True

        stripped = text.replace('.', '').replace(' ', '')
        if re.search(r'(.)\1{4,}', stripped):
            return True

        dot_count = text.count('.')
        if dot_count > 3 and dot_count / max(1, len(text)) > 0.15:
            return True

        alphanum = sum(1 for c in text if c.isalnum())
        if len(text) > 20 and alphanum / len(text) < 0.45:
            return True

        if len(words) >= 4:
            unique_words = set(w.lower() for w in words)
            if len(unique_words) / len(words) < 0.4:
                return True

        return False

    _CMS_TEMPLATE_KEYWORDS = re.compile(
        r"RESERVED|NUCC|N0CC|NLICC|FOR\s*USE|CLAIM\s*(CODE|ID)"
        r"|ACCEPT\s*ASSIGN|INSURANCE\s*(PLAN|TYPE)|EMPLOYER|REFERRING"
        r"|BILLING\s*PROVIDER|SERVICE\s*FACILITY|TELEPHONE"
        r"|PATIENT'?S?\s*(NAME|ADDRESS|BIRTH|CONDITION|SIGNATURE|ACCOUNT|RELATIONSHIP)"
        r"|INSURED'?S?\s*(NAME|ID|ADDRESS|DATE|POLICY|SIGNATURE|GROUP)"
        r"|OTHER\s*INSURED|ZIP\s*CODE|CITY|STATE|SEX|NPI|DOB"
        r"|AMOUNT\s*PAID|TOTAL\s*CHARGE|OUTSIDE\s*LAB|\bCHARGES\b"
        r"|PRIOR\s*AUTH|AUTHORIZATION|RESUBMISSION|HOSPITALIZATION"
        r"|DIAGNOSIS|FEDERAL\s*TAX|SIGNATURE\s*OF"
        r"|HEALTH\s*INSURANCE|CLAIM\s*FORM|CURRENT\s*ILLNESS"
        r"|PLACE\s*OF\s*SERVICE|DATE\s*OF|PLEASE\s*PRINT"
        r"|Delaware|FECA|PROGRAM",
        re.IGNORECASE,
    )

    _TEMPLATE_BLEED_PHRASES = re.compile(
        r"\bCARRIER\b|\bPICA\b|\bPLEASE\s+PRINT\s+OR\s+TYPE\b"
        r"|\bHEALTH\s+INSURANCE\s+CLAIM\s+FORM\b"
        r"|\bAPPROVED\s+BY\b|\bNUCC\b|\b02/12\b"
        r"|\bAPPROVED\s+OMB\b|\bFORM\s+CMS[\s-]*1500\b",
        re.IGNORECASE,
    )

    def _strip_template_bleed(self, text: str, field_name: str) -> str:
        """Remove template phrases that bleed into field text from padded bounding box.

        Padding around bounding boxes can capture neighboring template labels
        (e.g. "CARRIER" above the header area).  Instead of blanking the whole
        field, we surgically remove known template fragments and keep the real
        handwritten/typed content.
        """
        if not text:
            return text
        cleaned = self._TEMPLATE_BLEED_PHRASES.sub("", text)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"(?m)^\s*\n", "", cleaned)
        cleaned = cleaned.strip()
        if not cleaned and text.strip():
            return text
        return cleaned

    def _is_vlm_template_text(self, text: str) -> bool:
        """Detect if VLM output is CMS template label text rather than actual field content.

        Uses density check: only flag as template when keywords make up
        the majority of the text.  This prevents real content like
        addresses that happen to contain a keyword fragment from being
        incorrectly blanked.
        """
        if not text or len(text.strip()) < 3:
            return False
        t = text.strip()
        if self._CMS_TEMPLATE_KEYWORDS.search(t):
            matches = list(self._CMS_TEMPLATE_KEYWORDS.finditer(t))
            kw_chars = sum(m.end() - m.start() for m in matches)
            if kw_chars / max(1, len(t)) > 0.5:
                return True
        upper_ratio = sum(1 for c in t if c.isupper()) / max(1, sum(1 for c in t if c.isalpha()))
        if len(t) > 8 and upper_ratio > 0.75 and not any(c.isdigit() for c in t):
            return True
        return False

    # ── Field-type-aware decision tree helpers ─────────────────────────

    _FIELD_REGEXES: Dict[str, str] = {
        "date": r"^[\d/\-\s\.OoIl]{4,14}$",
        "date_range": r"^[\d/\-\s\.OoIlto]+$",
        "phone": r"^[\d\(\)\s\-OoIl]{7,16}$",
        "tax_id": r"^[\dOoIl\-]{7,11}$",
        "npi": r"^[\dOoIl]{9,11}$",
        "money": r"^\$?[\d,OoIl]+\.?\d{0,2}$",
        "icd10": r"^[A-Za-z]\d{1,4}\.?\d{0,2}$",
        "zip": r"^\+?\d{3,5}(-\d{4})?$",
    }

    def _categorize_field(self, field_type: str, field_id: str) -> str:
        """Map schema field_type to escalation category."""
        if field_type in ("checkbox",):
            return "checkbox"
        if field_type in ("signature",):
            return "signature"
        if field_type in ("table",):
            return "table"
        if field_type in ("date", "date_range", "phone", "tax_id", "npi", "money", "zip", "account"):
            return "numeric"
        if field_type == "address":
            return "short_text"
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

        mode = getattr(self.config, "ocr_engine_mode", "tiered") or "tiered"
        if mode in ("tiered", "paddle_only"):
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
            if field_type in ("date", "date_range", "phone", "tax_id", "npi", "money", "zip", "account"):
                ok = sum(c.isdigit() or c in "/-.()$ " for c in t)
                char_valid = ok / max(1, len(t))
            elif field_type == "address":
                ok = sum(c.isalnum() or c.isspace() or c in ",.-'#/" for c in t)
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
        if field_type in ("date", "date_range"):
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
        elif field_type == "zip":
            t = t.replace("O", "0").replace("o", "0")
            t = t.replace("I", "1").replace("l", "1")
            t = re.sub(r"[^\d\-]", "", t)
        elif field_type in ("text",):
            if "," in t and len(t) < 40:
                parts = t.split(",")
                t = ",".join(p.strip().title() for p in parts)
        return t

    # ── CMS-1500 field-aware OCR pipeline ───────────────────────────────────
    #
    # Multi-engine consensus (v1.3):
    #   1. Template subtraction cleans the crop (removes template text/lines)
    #   2. Checkbox  → fill-ratio detector (handled in _process_impl)
    #   3. Signature → "[SIGNED]" only, no OCR (handled in _process_impl)
    #   4. Date      → all three engines + VLM fallback
    #   5. Text      → PaddleOCR + TrOCR + Florence-2 → consensus pick
    #   6. Table     → VLM direct with structured prompt (handled in labeling_agent)
    #   7. Blank     → determined ONLY when all OCR engines return nothing
    #

    def _pick_best_candidate(
        self,
        candidates: List[Tuple[str, float, float, str, bool]],
        field_type: str,
    ) -> Tuple[str, float, float, str, bool]:
        """Pick the best OCR result from multiple engines.

        Each candidate is (text, composite_conf, raw_conf, engine, regex_ok).
        Agreement bonus: if 2+ engines produce similar text, that text
        gets a confidence boost because independent agreement is strong
        evidence of correctness.
        """
        if len(candidates) == 1:
            return candidates[0]

        from difflib import SequenceMatcher

        def _similar(a: str, b: str) -> bool:
            if not a or not b:
                return False
            return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() >= 0.65

        boosted: List[Tuple[str, float, float, str, bool]] = []
        for i, (text_i, comp_i, raw_i, eng_i, regex_i) in enumerate(candidates):
            agree_count = sum(
                1 for j, (text_j, _, _, _, _) in enumerate(candidates)
                if i != j and _similar(text_i, text_j)
            )
            bonus = 0.15 * agree_count
            boosted.append((text_i, min(1.0, comp_i + bonus), raw_i, eng_i, regex_i))

        boosted.sort(key=lambda c: (c[4], c[1]), reverse=True)
        return boosted[0]

    def _florence2_raw_fallback(
        self,
        crop_raw: np.ndarray,
        field_name: str,
        field_type: str,
    ) -> Tuple[str, float]:
        """Try Florence-2 on the original (un-subtracted) crop as fallback.

        Template subtraction can damage real content (especially multi-line
        addresses, mixed text+numbers).  Running Florence-2 on the raw crop
        recovers this content.  Template labels that Florence-2 reads from
        the raw crop are caught by keyword / hallucination / box-number
        filters, so blank fields still return ("", 0.0).

        This replaces the expensive ink-ratio-gated VLM rescue (~60s) with
        a cheap Florence-2 call (~200ms) that is more reliable.
        """
        raw_result = _florence2_ocr_run(crop_raw)
        if not raw_result or not raw_result.strip():
            return "", 0.0
        raw_text = raw_result.strip()
        raw_conf = 0.82
        _PLACEHOLDER_CHARS = frozenset({"-", "–", "—", ".", "_", "|", "~", "*"})
        if raw_text in _PLACEHOLDER_CHARS:
            return "", 0.0
        if self._is_hallucination(raw_text, field_type):
            self.log(f"Raw-fallback hallucination: '{raw_text[:40]}' for {field_name}")
            return "", 0.0
        if self._is_vlm_template_text(raw_text):
            self.log(f"Raw-fallback template text: '{raw_text[:40]}' for {field_name}")
            return "", 0.0
        if self._CMS_TEMPLATE_KEYWORDS.search(raw_text):
            matches = list(self._CMS_TEMPLATE_KEYWORDS.finditer(raw_text))
            kw_chars = sum(m.end() - m.start() for m in matches)
            if kw_chars / max(1, len(raw_text.strip())) > 0.5:
                self.log(f"Raw-fallback keyword density: '{raw_text[:40]}' for {field_name}")
                return "", 0.0
        if len(raw_text.strip()) < 2:
            return "", 0.0
        if field_type == "text" and re.match(r'^\d{1,3}[a-z]?\.?$', raw_text.strip()):
            self.log(f"Raw-fallback box-number: '{raw_text}' for {field_name}")
            return "", 0.0
        return raw_text, raw_conf

    def _process_cms_field(
        self,
        crop_for_ocr: np.ndarray,
        crop_raw: np.ndarray,
        block: DetectedBlock,
        field_type: str,
        ink_ratio: float = 1.0,
    ) -> DetectedBlock:
        """Field-aware OCR for CMS-1500 — Florence-2 as sole blank detector.

        Architecture (v5 — raw-crop fallback, no VLM rescue):
        Florence-2 is the SOLE blank detector for non-date fields.
        When Florence-2 on the template-subtracted crop fails (empty or
        inconsistent), we try Florence-2 on the ORIGINAL raw crop before
        giving up.  This recovers content that template subtraction
        damaged, while template labels are caught by keyword filters.
        VLM rescue is removed from this method — it added 60s latency
        per field and caused hallucinations on blank fields.

        Flow:
          Florence-2 on subtracted crop → filters → use if good
          Florence-2 empty / inconsistent → try raw crop fallback
          Raw crop also empty → blank (no VLM rescue)
        """
        field_name = (block.metadata or {}).get("field_name", "") or block.id

        if field_type in ("date", "date_range"):
            return self._process_date_field_vlm(crop_for_ocr, crop_raw, block, field_name, ink_ratio)

        # ── Step 1: Florence-2 primary OCR ─────────────────────────────────
        florence_text, florence_conf = self._florence2_ocr(crop_for_ocr)

        self.log(
            f"[F2-DIAG] {field_name} type={field_type}: "
            f"text='{(florence_text or '')[:60]}' conf={florence_conf:.3f} "
            f"ink={ink_ratio:.4f} crop={crop_for_ocr.shape[:2]}"
        )

        # Multi-layer filter: catch template residue that survives subtraction.
        # Keyword filter uses DENSITY check: only blank when keywords make
        # up the majority of the text.
        if florence_text:
            if self._is_hallucination(florence_text, field_type):
                florence_text, florence_conf = "", 0.0
            elif self._CMS_TEMPLATE_KEYWORDS.search(florence_text):
                matches = list(self._CMS_TEMPLATE_KEYWORDS.finditer(florence_text))
                kw_chars = sum(m.end() - m.start() for m in matches)
                if kw_chars / max(1, len(florence_text.strip())) > 0.5:
                    self.log(f"Florence-2 template text filtered ({kw_chars}/{len(florence_text.strip())} kw): '{florence_text[:40]}' for {field_name}")
                    florence_text, florence_conf = "", 0.0
            elif len(florence_text.strip()) < 2:
                florence_text, florence_conf = "", 0.0
            elif field_type == "text" and re.match(r'^\d{1,3}[a-z]?\.?$', florence_text.strip()):
                self.log(f"Florence-2 box-number filtered: '{florence_text}' for {field_name}")
                florence_text, florence_conf = "", 0.0

        # ── Case A: Florence-2 empty → try recovery, then blank ────────────
        # Trust Florence-2 for blank detection.  Recovery paths:
        #   1. Upscale retry (ink > 0.008) — cheap, Florence-2 only
        #   2. Raw crop fallback (inner_ink > 0.10) — run Florence-2 on the
        #      ORIGINAL crop (before template subtraction).  Gated by
        #      INNER ink ratio (center 70% of crop, excluding padding
        #      edges) to avoid false triggers from neighboring fields
        #      bleeding into padded bbox.  Template keywords get caught
        #      by filters; real content passes through.
        if not florence_text:
            if ink_ratio > 0.008:
                h_c, w_c = crop_for_ocr.shape[:2]
                up = cv2.resize(crop_for_ocr, (w_c * 2, h_c * 2), interpolation=cv2.INTER_CUBIC)
                retry_text, retry_conf = self._florence2_ocr(up)
                if (
                    retry_text
                    and len(retry_text.strip()) >= 2
                    and not self._is_hallucination(retry_text, field_type)
                    and not self._is_vlm_template_text(retry_text)
                ):
                    text = self._llm_normalize_field(retry_text, field_type)
                    r_ok = self._validate_field_regex(text, field_type)
                    if r_ok:
                        self.log(f"Recovered via upscaled Florence-2: '{text}' for {field_name}")
                        block.text = text
                        block.confidence = self._compute_composite_confidence(retry_conf, text, field_type, r_ok)
                        block.metadata["ocr_engine"] = "florence2_upscaled"
                        block.metadata["escalation"] = "upscaled_retry"
                        return block

            h_c, w_c = crop_for_ocr.shape[:2]
            mx = max(1, int(w_c * 0.15))
            my = max(1, int(h_c * 0.15))
            interior = crop_for_ocr[my:h_c - my, mx:w_c - mx]
            inner_ink = 0.0
            if interior.size > 0:
                gi = cv2.cvtColor(interior, cv2.COLOR_RGB2GRAY) if interior.ndim == 3 else interior
                bi = cv2.adaptiveThreshold(gi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9)
                inner_ink = float(np.count_nonzero(bi)) / max(1, bi.size)

            if inner_ink > 0.10:
                self.log(f"Raw-fallback gate: ink={ink_ratio:.4f} inner_ink={inner_ink:.4f} for {field_name}")
                raw_text, raw_conf = self._florence2_raw_fallback(crop_raw, field_name, field_type)
                if raw_text:
                    text = self._llm_normalize_field(raw_text, field_type)
                    r_ok = self._validate_field_regex(text, field_type)
                    comp = self._compute_composite_confidence(raw_conf, text, field_type, r_ok)
                    self.log(f"Recovered via raw-crop Florence-2: '{text[:50]}' conf={comp:.2f} for {field_name}")
                    block.text = text
                    block.confidence = comp
                    block.metadata["ocr_engine"] = "florence2_raw_fallback"
                    block.metadata["escalation"] = "raw_crop_retry"
                    return block

            self.log(f"Blank (Florence-2 empty, ink={ink_ratio:.4f} inner={inner_ink:.4f}): {field_name}")
            block.text = ""
            block.confidence = 0.0
            block.metadata["ocr_engine"] = "blank_confirmed"
            block.metadata["is_blank"] = True
            return block

        # ── Case B: Florence-2 has text — CCA noise filter ────────────────
        # CCA is used ONLY here: to discard Florence-2 text when the crop
        # has no real stroke structure (Florence-2 read template residue).
        # For money fields, mask left 22% to ignore the pre-printed "$".
        crop_for_cca = crop_for_ocr
        if field_type == "money":
            h_m, w_m = crop_for_ocr.shape[:2]
            if w_m > 20:
                crop_for_cca = crop_for_ocr.copy()
                crop_for_cca[:, :int(w_m * 0.22)] = 255

        if not self._has_meaningful_content(crop_for_cca, min_components=1):
            self.log(f"Noise discard (text='{florence_text[:30]}' no structure): {field_name}")
            block.text = ""
            block.confidence = 0.0
            block.metadata["ocr_engine"] = "blank_confirmed"
            block.metadata["is_blank"] = True
            return block

        # ── Case C: Florence-2 has text + structure → evaluate ────────────
        original_florence = florence_text
        r_ok = self._validate_field_regex(florence_text, field_type)
        composite = self._compute_composite_confidence(florence_conf, florence_text, field_type, r_ok)

        # Digit-type regex fix: upscale retry if regex failed.
        # Cache the upscale result — reused below for consistency check.
        _upscale_text, _upscale_conf = None, None
        _DIGIT_TYPES = ("zip", "npi", "tax_id", "money", "phone")
        if not r_ok and field_type in _DIGIT_TYPES:
            h, w = crop_for_ocr.shape[:2]
            upscaled = cv2.resize(crop_for_ocr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            _upscale_text, _upscale_conf = self._florence2_ocr(upscaled)
            if _upscale_text:
                norm_retry = self._llm_normalize_field(_upscale_text, field_type)
                if self._validate_field_regex(norm_retry, field_type):
                    self.log(f"Florence-2 upscaled retry: '{norm_retry}' for {field_name}")
                    florence_text, florence_conf = norm_retry, _upscale_conf
                    r_ok = True
                    composite = self._compute_composite_confidence(florence_conf, florence_text, field_type, True)

        if composite >= 0.55:
            text = self._llm_normalize_field(florence_text, field_type)
            block.text = text
            block.confidence = composite
            block.metadata["ocr_engine"] = "florence2"
            block.metadata["escalation"] = "none"
            return block

        # ── Step 3: Self-consistency check before VLM ─────────────────────
        # KEY INSIGHT: real handwriting produces consistent Florence-2
        # output at different scales; template residue / noise does not.
        # Cost: ~200ms (one Florence-2 call) vs ~60s (VLM rescue).
        if len(florence_text.strip()) >= 3:
            if _upscale_text is None:
                h_v, w_v = crop_for_ocr.shape[:2]
                up = cv2.resize(crop_for_ocr, (w_v * 2, h_v * 2), interpolation=cv2.INTER_CUBIC)
                _upscale_text, _upscale_conf = self._florence2_ocr(up)

            text_confirmed = False
            if _upscale_text and len(_upscale_text.strip()) >= 2:
                from difflib import SequenceMatcher
                orig_lc = original_florence.lower().strip()
                up_lc = _upscale_text.lower().strip()
                char_sim = SequenceMatcher(None, orig_lc, up_lc).ratio()
                words_orig = set(orig_lc.split())
                words_up = set(up_lc.split())
                union = words_orig | words_up
                word_sim = len(words_orig & words_up) / len(union) if union else 0.0
                sim = max(char_sim, word_sim)
                if sim >= 0.4:
                    text_confirmed = True
                    if (_upscale_conf or 0) > florence_conf:
                        florence_text, florence_conf = _upscale_text, _upscale_conf
                        r_ok = self._validate_field_regex(florence_text, field_type)
                        composite = self._compute_composite_confidence(
                            florence_conf, florence_text, field_type, r_ok
                        )

            if not text_confirmed:
                h_c2, w_c2 = crop_for_ocr.shape[:2]
                mx2 = max(1, int(w_c2 * 0.15))
                my2 = max(1, int(h_c2 * 0.15))
                interior2 = crop_for_ocr[my2:h_c2 - my2, mx2:w_c2 - mx2]
                inner_ink2 = 0.0
                if interior2.size > 0:
                    gi2 = cv2.cvtColor(interior2, cv2.COLOR_RGB2GRAY) if interior2.ndim == 3 else interior2
                    bi2 = cv2.adaptiveThreshold(gi2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9)
                    inner_ink2 = float(np.count_nonzero(bi2)) / max(1, bi2.size)
                if inner_ink2 > 0.10:
                    raw_text, raw_conf = self._florence2_raw_fallback(crop_raw, field_name, field_type)
                    if raw_text:
                        text = self._llm_normalize_field(raw_text, field_type)
                        r_ok_v = self._validate_field_regex(text, field_type)
                        comp = self._compute_composite_confidence(raw_conf, text, field_type, r_ok_v)
                        self.log(f"Recovered via raw-crop (consistency-fail): '{text[:50]}' conf={comp:.2f}")
                        block.text = text
                        block.confidence = comp
                        block.metadata["ocr_engine"] = "florence2_raw_fallback"
                        block.metadata["escalation"] = "raw_crop_consistency_fail"
                        return block
                self.log(
                    f"Consistency FAIL (orig='{original_florence[:25]}' "
                    f"up='{(_upscale_text or '')[:25]}' ink={ink_ratio:.4f}) → blank: {field_name}"
                )
                block.text = ""
                block.confidence = 0.0
                block.metadata["ocr_engine"] = "blank_confirmed"
                block.metadata["is_blank"] = True
                return block

            if composite >= 0.55:
                text = self._llm_normalize_field(florence_text, field_type)
                block.text = text
                block.confidence = composite
                block.metadata["ocr_engine"] = "florence2"
                block.metadata["escalation"] = "upscale_verified"
                return block

            # ── Step 3b: Confirmed real text, low confidence → raw crop fallback
            raw_text, raw_conf = self._florence2_raw_fallback(crop_raw, field_name, field_type)
            if raw_text:
                f_r_ok = self._validate_field_regex(florence_text, field_type)
                f_comp = self._compute_composite_confidence(florence_conf, florence_text, field_type, f_r_ok)
                r_ok_v = self._validate_field_regex(raw_text, field_type)
                raw_comp = self._compute_composite_confidence(raw_conf, raw_text, field_type, r_ok_v)
                if f_comp >= raw_comp:
                    text = self._llm_normalize_field(florence_text, field_type)
                    block.text = text
                    block.confidence = f_comp
                    block.metadata["ocr_engine"] = "florence2"
                    block.metadata["escalation"] = "none"
                    return block
                text = self._llm_normalize_field(raw_text, field_type)
                block.text = text
                block.confidence = raw_comp
                block.metadata["ocr_engine"] = "florence2_raw_fallback"
                block.metadata["escalation"] = "raw_crop_low_conf"
                return block

        # ── Step 4: Use Florence-2 output even if low confidence ──────────
        text = self._llm_normalize_field(florence_text, field_type)
        r_ok = self._validate_field_regex(text, field_type)
        block.text = text
        block.confidence = self._compute_composite_confidence(florence_conf, text, field_type, r_ok)
        block.metadata["ocr_engine"] = "florence2"
        block.metadata["escalation"] = "none"
        return block

    def _process_date_field_vlm(
        self,
        crop_for_ocr: np.ndarray,
        crop_raw: np.ndarray,
        block: DetectedBlock,
        field_name: str,
        ink_ratio: float = 1.0,
    ) -> DetectedBlock:
        """Date fields: Florence-2 + ink ratio hybrid, VLM for incomplete dates."""
        field_type = (block.metadata or {}).get("field_type", "date")

        def _clean_date(raw: str) -> str:
            m = re.search(r'(\d{1,2})[/\-\s](\d{1,2})[/\-\s](\d{2,4})', raw)
            if m:
                return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
            return raw.strip()

        def _is_complete_date(t: str) -> bool:
            return bool(re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', t.strip()))

        # ── Step 1: Florence-2 + CCA hybrid ─────────────────────────────
        florence_text, florence_conf = self._florence2_ocr(crop_for_ocr)
        if florence_text and self._is_hallucination(florence_text, field_type):
            florence_text, florence_conf = "", 0.0

        has_content = self._has_meaningful_content(crop_for_ocr, min_components=2)

        if not florence_text and not has_content:
            self.log(f"Blank date (no text, no structure): {field_name}")
            block.text = ""
            block.confidence = 0.0
            block.metadata["ocr_engine"] = "blank_confirmed"
            block.metadata["is_blank"] = True
            return block

        if florence_text and not self._has_meaningful_content(crop_for_ocr, min_components=1):
            self.log(f"Date noise discard (text='{florence_text[:30]}' no structure): {field_name}")
            block.text = ""
            block.confidence = 0.0
            block.metadata["ocr_engine"] = "blank_confirmed"
            block.metadata["is_blank"] = True
            return block

        if not florence_text and has_content:
            self.log(f"Florence-2 empty + CCA content → VLM date rescue: {field_name}")
            vlm_raw, vlm_conf = self._vlm_ocr_field(
                crop_for_ocr, field_name, field_type,
                model=self.VLM_MODEL_RESCUE, timeout=60,
            )
            if vlm_raw and not self._is_hallucination(vlm_raw, field_type):
                vlm_cleaned = _clean_date(vlm_raw)
                text = self._llm_normalize_field(vlm_cleaned, field_type)
                if _is_complete_date(text):
                    block.text = text
                    block.confidence = max(self._compute_composite_confidence(vlm_conf, text, field_type, True), 0.92)
                    block.metadata["ocr_engine"] = "vlm_rescue_date"
                    block.metadata["escalation"] = "vlm_rescue"
                    return block
            block.text = ""
            block.confidence = 0.0
            block.metadata["ocr_engine"] = "blank_confirmed"
            block.metadata["is_blank"] = True
            return block

        # ── Step 2: Florence-2 has text + ink → evaluate ─────────────────
        cleaned = _clean_date(florence_text)
        text = self._llm_normalize_field(cleaned, field_type)
        r_ok = self._validate_field_regex(text, field_type)
        composite = self._compute_composite_confidence(florence_conf, text, field_type, r_ok)

        if _is_complete_date(text):
            composite = max(composite, 0.90)
            block.text = text
            block.confidence = composite
            block.metadata["ocr_engine"] = "florence2"
            block.metadata["escalation"] = "none"
            return block

        # ── Step 3: VLM rescue — incomplete date from Florence-2 ─────────
        vlm_raw, vlm_conf = self._vlm_ocr_field(
            crop_for_ocr, field_name, field_type,
            model=self.VLM_MODEL_RESCUE, timeout=60,
        )
        if vlm_raw and not self._is_hallucination(vlm_raw, field_type):
            vlm_cleaned = _clean_date(vlm_raw)
            self.log(f"VLM date cleaned: '{vlm_raw[:40]}' → '{vlm_cleaned}'")
            text = self._llm_normalize_field(vlm_cleaned, field_type)
            r_ok = self._validate_field_regex(text, field_type)
            composite = self._compute_composite_confidence(vlm_conf, text, field_type, r_ok)
            if _is_complete_date(text):
                composite = max(composite, 0.92)
            block.text = text
            block.confidence = composite
            block.metadata["ocr_engine"] = "vlm_rescue_date"
            block.metadata["escalation"] = "vlm_rescue"
            return block

        # ── Step 3: Use Florence-2 even if incomplete ─────────────────────
        if florence_text:
            text = self._llm_normalize_field(_clean_date(florence_text), field_type)
            r_ok = self._validate_field_regex(text, field_type)
            block.text = text
            block.confidence = self._compute_composite_confidence(florence_conf, text, field_type, r_ok)
            block.metadata["ocr_engine"] = "florence2"
            block.metadata["escalation"] = "none"
            return block

        block.text = ""
        block.confidence = 0.0
        block.metadata["ocr_engine"] = "blank_confirmed"
        block.metadata["is_blank"] = True
        return block

    # ── process_blocks: parallel OCR with stats + empty field filtering ────

    async def process_blocks(self, image: np.ndarray, blocks: List[DetectedBlock]) -> List[DetectedBlock]:
        """Process all blocks through the multi-engine OCR pipeline.

        Flow per field type (v1.3 — multi-engine consensus):
          - Checkbox  → fill-ratio detector
          - Signature → "[SIGNED]" only (no OCR)
          - Date      → PaddleOCR + TrOCR + Florence-2 → consensus pick → VLM rescue
          - Text      → PaddleOCR + TrOCR + Florence-2 → consensus pick → VLM rescue
          - Table     → skipped here (handled by labeling_agent)

        Empty fields (no ink) are skipped and excluded from output.
        """
        await self.initialize()
        t0 = time.time()

        # Florence-2 is used on every field now, so serialize to avoid GPU
        # contention between Florence-2 + TrOCR running in parallel threads.
        sem = asyncio.Semaphore(2)

        async def _process_one(block: DetectedBlock) -> DetectedBlock:
            async with sem:
                return await asyncio.to_thread(self._process_sync, image, block)

        results = list(await asyncio.gather(*[_process_one(b) for b in blocks]))

        # ── Paired checkbox resolution ──────────────────────────────────
        # For yes/no pairs (e.g. 11d_another_health_plan_yes / _no),
        # only mark the one with higher ink_ratio. Both being "X" is wrong.
        # Yes/No pairs: only one can be checked
        _CHECKBOX_PAIRS = [
            ("10a_employment_yes", "10a_employment_no"),
            ("10b_auto_accident_yes", "10b_auto_accident_no"),
            ("10c_other_accident_yes", "10c_other_accident_no"),
            ("11d_another_health_plan_yes", "11d_another_health_plan_no"),
            ("20_outside_lab_yes", "20_outside_lab_no"),
            ("27_accept_assignment_yes", "27_accept_assignment_no"),
        ]
        block_map = {b.id: b for b in results}
        for id_a, id_b in _CHECKBOX_PAIRS:
            ba, bb = block_map.get(id_a), block_map.get(id_b)
            if ba and bb and ba.text == "X" and bb.text == "X":
                ink_a = (ba.metadata or {}).get("checkbox_ink_ratio", 0)
                ink_b = (bb.metadata or {}).get("checkbox_ink_ratio", 0)
                if ink_a >= ink_b:
                    bb.text = ""
                    bb.confidence = 0.0
                else:
                    ba.text = ""
                    ba.confidence = 0.0

        # Mutually exclusive groups: only the highest-ink one wins
        _CHECKBOX_GROUPS = [
            [
                "1_insurance_type_medicare", "1_insurance_type_medicaid",
                "1_insurance_type_tricare", "1_insurance_type_champva",
                "1_insurance_type_group", "1_insurance_type_feca",
                "1_insurance_type_other",
            ],
            [
                "6_patient_relationship_self", "6_patient_relationship_spouse",
                "6_patient_relationship_child", "6_patient_relationship_other",
            ],
            ["3_patient_sex_m", "3_patient_sex_f"],
            ["11a_insured_sex_m", "11a_insured_sex_f"],
        ]
        for group_ids in _CHECKBOX_GROUPS:
            checked = [
                (bid, block_map[bid])
                for bid in group_ids
                if bid in block_map and block_map[bid].text == "X"
            ]
            if len(checked) > 1:
                best_id, _ = max(
                    checked,
                    key=lambda pair: (pair[1].metadata or {}).get("checkbox_ink_ratio", 0),
                )
                for bid, blk in checked:
                    if bid != best_id:
                        blk.text = ""
                        blk.confidence = 0.0

        # Gather stats for logging
        stats = {"florence2": 0, "vlm_rescue": 0,
                 "blank": 0, "signed": 0, "checkbox": 0, "other": 0}
        for b in results:
            engine = (b.metadata or {}).get("ocr_engine", "")
            is_blank = (b.metadata or {}).get("is_blank", False)
            if is_blank:
                stats["blank"] += 1
            elif "florence2" in engine:
                stats["florence2"] += 1
            elif "vlm" in engine:
                stats["vlm_rescue"] += 1
            elif "signature" in engine:
                stats["signed"] += 1
            elif "checkbox" in engine:
                stats["checkbox"] += 1
            else:
                stats["other"] += 1

        # LLM normalization pass on text fields
        for block in results:
            if block.text and block.block_type not in (
                BlockType.CHECKBOX, BlockType.SIGNATURE, BlockType.TABLE
            ):
                ft = (block.metadata or {}).get("field_type", "text")
                block.text = self._llm_normalize_field(block.text, ft)

        dt = time.time() - t0
        print(
            f"[OCR] {len(results)} fields in {dt:.1f}s — "
            f"Florence2={stats['florence2']} VLM_rescue={stats['vlm_rescue']} "
            f"Signed={stats['signed']} "
            f"Checkbox={stats['checkbox']} Blank={stats['blank']}"
        )

        # ── Filter out blank fields (no ink, no text) ────────────────────
        # Blank fields should NOT appear in output / have bounding boxes drawn.
        filtered = []
        for block in results:
            is_blank = (block.metadata or {}).get("is_blank", False)
            has_text = block.text and len(str(block.text).strip()) > 0

            if is_blank and not has_text:
                continue  # skip — no bounding box for empty fields

            filtered.append(block)

        return filtered

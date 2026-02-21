"""
CMS-1500 Direct Pipeline - Robust, Simple, End-to-End
======================================================

This is a clean, direct implementation that:
1. Ingests PDF/Image -> RGB array
2. Aligns to CMS-1500 template (ORB homography)
3. Extracts zones from schema
4. Runs tiered OCR (PaddleOCR -> TrOCR for handwriting)
5. Returns structured field_details

No complex agent orchestration. Just works.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Image processing
try:
    import cv2
except ImportError:
    cv2 = None

# PDF handling
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None


@dataclass
class FieldResult:
    """Single extracted field."""
    field_id: str
    field_name: str
    value: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    method: str = "ocr"  # ocr, icr, checkbox


@dataclass
class ExtractionResult:
    """Full extraction result."""
    form_type: str = "CMS-1500"
    extraction_method: str = "direct_pipeline"
    extracted_fields: Dict[str, str] = field(default_factory=dict)
    field_details: List[Dict[str, Any]] = field(default_factory=list)
    ocr_boxes: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0
    alignment_success: bool = False
    error: Optional[str] = None


class CMS1500DirectPipeline:
    """
    Direct, robust CMS-1500 extraction pipeline.
    
    Usage:
        pipeline = CMS1500DirectPipeline()
        result = pipeline.extract("path/to/cms1500.pdf")
    """
    
    # Template zones (normalized 0-1 coordinates for standard CMS-1500)
    # Format: field_id -> (x, y, w, h) normalized
    TEMPLATE_ZONES = {
        # Patient Information (Box 1-4)
        "1_insurance_type": (0.02, 0.08, 0.45, 0.03),
        "1a_insured_id": (0.52, 0.08, 0.46, 0.03),
        "2_patient_name": (0.02, 0.11, 0.30, 0.025),
        "3_patient_dob": (0.33, 0.11, 0.12, 0.025),
        "3_patient_sex": (0.46, 0.11, 0.06, 0.025),
        "4_insured_name": (0.52, 0.11, 0.46, 0.025),
        
        # Address fields (Box 5-7)
        "5_patient_address": (0.02, 0.135, 0.30, 0.025),
        "5_patient_city": (0.02, 0.16, 0.15, 0.02),
        "5_patient_state": (0.18, 0.16, 0.05, 0.02),
        "5_patient_zip": (0.24, 0.16, 0.08, 0.02),
        "5_patient_phone": (0.02, 0.18, 0.30, 0.02),
        "6_patient_relationship": (0.33, 0.135, 0.18, 0.025),
        "7_insured_address": (0.52, 0.135, 0.46, 0.025),
        
        # More patient/insured info (Box 8-11)
        "8_reserved": (0.02, 0.20, 0.30, 0.025),
        "9_other_insured_name": (0.02, 0.225, 0.30, 0.025),
        "9a_other_insured_policy": (0.02, 0.25, 0.30, 0.02),
        "10_condition_employment": (0.33, 0.20, 0.18, 0.02),
        "10_condition_auto": (0.33, 0.22, 0.18, 0.02),
        "10_condition_other": (0.33, 0.24, 0.18, 0.02),
        "11_insured_policy": (0.52, 0.20, 0.46, 0.025),
        "11a_insured_dob": (0.52, 0.225, 0.23, 0.02),
        "11b_other_claim_id": (0.52, 0.25, 0.46, 0.02),
        "11c_insurance_plan": (0.52, 0.275, 0.46, 0.02),
        "11d_another_plan": (0.52, 0.30, 0.46, 0.02),
        
        # Signature boxes (12-13)
        "12_patient_signature": (0.02, 0.32, 0.45, 0.03),
        "13_insured_signature": (0.52, 0.32, 0.46, 0.03),
        
        # Condition info (14-19)
        "14_date_current_illness": (0.02, 0.36, 0.15, 0.025),
        "15_other_date": (0.18, 0.36, 0.15, 0.025),
        "16_dates_unable_work": (0.34, 0.36, 0.18, 0.025),
        "17_referring_provider": (0.02, 0.385, 0.30, 0.025),
        "17a_referring_npi": (0.33, 0.385, 0.18, 0.025),
        "18_hospitalization_dates": (0.52, 0.36, 0.46, 0.025),
        "19_additional_info": (0.02, 0.41, 0.50, 0.025),
        
        # Diagnosis codes (21)
        "21_diagnosis_a": (0.02, 0.44, 0.12, 0.02),
        "21_diagnosis_b": (0.15, 0.44, 0.12, 0.02),
        "21_diagnosis_c": (0.28, 0.44, 0.12, 0.02),
        "21_diagnosis_d": (0.41, 0.44, 0.12, 0.02),
        "21_diagnosis_e": (0.02, 0.46, 0.12, 0.02),
        "21_diagnosis_f": (0.15, 0.46, 0.12, 0.02),
        "21_diagnosis_g": (0.28, 0.46, 0.12, 0.02),
        "21_diagnosis_h": (0.41, 0.46, 0.12, 0.02),
        "21_diagnosis_i": (0.02, 0.48, 0.12, 0.02),
        "21_diagnosis_j": (0.15, 0.48, 0.12, 0.02),
        "21_diagnosis_k": (0.28, 0.48, 0.12, 0.02),
        "21_diagnosis_l": (0.41, 0.48, 0.12, 0.02),
        
        # Service lines (24) - simplified, just first line
        "24_service_date_1": (0.02, 0.52, 0.10, 0.02),
        "24_place_service_1": (0.13, 0.52, 0.03, 0.02),
        "24_cpt_1": (0.17, 0.52, 0.08, 0.02),
        "24_modifier_1": (0.26, 0.52, 0.06, 0.02),
        "24_diagnosis_ptr_1": (0.33, 0.52, 0.04, 0.02),
        "24_charges_1": (0.38, 0.52, 0.08, 0.02),
        "24_units_1": (0.47, 0.52, 0.03, 0.02),
        
        # Provider info (25-33)
        "25_federal_tax_id": (0.02, 0.70, 0.20, 0.025),
        "26_patient_account": (0.23, 0.70, 0.15, 0.025),
        "27_accept_assignment": (0.39, 0.70, 0.10, 0.025),
        "28_total_charge": (0.50, 0.70, 0.12, 0.025),
        "29_amount_paid": (0.63, 0.70, 0.12, 0.025),
        "30_rsvd_nucc": (0.76, 0.70, 0.12, 0.025),
        
        "31_physician_signature": (0.02, 0.75, 0.30, 0.04),
        "32_service_facility": (0.33, 0.75, 0.30, 0.04),
        "32a_facility_npi": (0.33, 0.80, 0.15, 0.02),
        "33_billing_provider": (0.64, 0.75, 0.34, 0.04),
        "33a_billing_npi": (0.64, 0.80, 0.15, 0.02),
        "33b_billing_other_id": (0.80, 0.80, 0.18, 0.02),
    }
    
    # Field display names
    FIELD_NAMES = {
        "1_insurance_type": "Insurance Type",
        "1a_insured_id": "Insured's ID Number",
        "2_patient_name": "Patient Name",
        "3_patient_dob": "Patient DOB",
        "3_patient_sex": "Patient Sex",
        "4_insured_name": "Insured's Name",
        "5_patient_address": "Patient Address",
        "5_patient_city": "City",
        "5_patient_state": "State",
        "5_patient_zip": "ZIP Code",
        "5_patient_phone": "Phone",
        "21_diagnosis_a": "Diagnosis A",
        "21_diagnosis_b": "Diagnosis B",
        "24_cpt_1": "CPT Code",
        "24_charges_1": "Charges",
        "25_federal_tax_id": "Federal Tax ID",
        "28_total_charge": "Total Charge",
        "33a_billing_npi": "Billing NPI",
    }
    
    def __init__(self, use_icr: bool = True, use_gpu: bool = True):
        """Initialize pipeline with OCR engines."""
        self.use_icr = use_icr
        self.use_gpu = use_gpu
        self._paddle_ocr = None
        self._trocr_model = None
        self._trocr_processor = None
        self._template_img = None
        
    def _init_paddle(self):
        """Lazy init PaddleOCR."""
        if self._paddle_ocr is not None:
            return
            
        try:
            from paddleocr import PaddleOCR
            # Try different init signatures for compatibility with various PaddleOCR versions
            init_attempts = [
                {"lang": "en"},
                {"use_angle_cls": True, "lang": "en"},
                {"lang": "en", "use_gpu": True},
            ]
            
            for kwargs in init_attempts:
                try:
                    self._paddle_ocr = PaddleOCR(**kwargs)
                    print(f"✅ PaddleOCR initialized with {kwargs}")
                    return
                except TypeError as e:
                    continue
                    
            print("⚠️ PaddleOCR init failed with all parameter combinations")
            self._paddle_ocr = None
        except Exception as e:
            print(f"⚠️ PaddleOCR import/init failed: {e}")
            self._paddle_ocr = None
            
    def _init_trocr(self):
        """Lazy init TrOCR for handwriting."""
        if self._trocr_model is not None:
            return
            
        if not self.use_icr:
            return
            
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            
            model_name = "microsoft/trocr-base-handwritten"
            self._trocr_processor = TrOCRProcessor.from_pretrained(model_name)
            self._trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            if self.use_gpu and torch.cuda.is_available():
                self._trocr_model = self._trocr_model.cuda()
            self._trocr_model.eval()
            print("✅ TrOCR initialized")
        except Exception as e:
            print(f"⚠️ TrOCR init failed: {e}")
            self._trocr_model = None
            
    def _load_template(self) -> Optional[np.ndarray]:
        """Load CMS-1500 blank template for alignment."""
        if self._template_img is not None:
            return self._template_img
            
        template_paths = [
            Path(__file__).parent.parent.parent / "data" / "templates" / "cms1500_template.png",
            Path(__file__).parent.parent.parent / "data" / "sample_docs" / "cms1500_blank.pdf",
            Path("data/templates/cms1500_template.png"),
            Path("data/sample_docs/cms1500_blank.pdf"),
        ]
        
        for path in template_paths:
            if path.exists():
                try:
                    if path.suffix == '.pdf':
                        img = self._pdf_to_image(str(path))
                    else:
                        img = cv2.imread(str(path))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img is not None:
                        self._template_img = img
                        print(f"✅ Template loaded: {path}")
                        return img
                except Exception as e:
                    print(f"⚠️ Template load failed {path}: {e}")
                    
        print("⚠️ No template found, skipping alignment")
        return None
        
    def _pdf_to_image(self, path: str, dpi: int = 300) -> Optional[np.ndarray]:
        """Convert PDF to RGB image array."""
        try:
            # Try PyMuPDF first (faster)
            if fitz is not None:
                doc = fitz.open(path)
                page = doc[0]
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                doc.close()
                return img
        except Exception as e:
            print(f"⚠️ PyMuPDF failed: {e}")
            
        try:
            # Fallback to pdf2image
            if convert_from_path is not None:
                images = convert_from_path(path, dpi=dpi, first_page=1, last_page=1)
                if images:
                    return np.array(images[0])
        except Exception as e:
            print(f"⚠️ pdf2image failed: {e}")
            
        return None
        
    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """Load image from PDF or image file."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            print(f"❌ File not found: {path}")
            return None
            
        if path_obj.suffix.lower() == '.pdf':
            return self._pdf_to_image(path)
        else:
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return None
            
    def _align_to_template(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Align input image to template using ORB + homography."""
        if cv2 is None:
            return image, False
            
        template = self._load_template()
        if template is None:
            return image, False
            
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = image
                
            if len(template.shape) == 3:
                gray_tpl = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
            else:
                gray_tpl = template
                
            # ORB feature detection
            orb = cv2.ORB_create(nfeatures=5000)
            kp1, des1 = orb.detectAndCompute(gray_img, None)
            kp2, des2 = orb.detectAndCompute(gray_tpl, None)
            
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                print("⚠️ Not enough features for alignment")
                return image, False
                
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:100]
            
            if len(matches) < 10:
                print("⚠️ Not enough matches for alignment")
                return image, False
                
            # Compute homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                return image, False
                
            # Warp image
            h, w = template.shape[:2]
            aligned = cv2.warpPerspective(image, H, (w, h))
            
            print(f"✅ Alignment successful ({sum(mask.ravel())} inliers)")
            return aligned, True
            
        except Exception as e:
            print(f"⚠️ Alignment failed: {e}")
            return image, False
            
    def _run_paddle_ocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run PaddleOCR on image, return word boxes."""
        self._init_paddle()
        
        if self._paddle_ocr is None:
            return []
            
        try:
            # Try new predict() API first, fall back to ocr()
            result = None
            try:
                result = self._paddle_ocr.predict(image)
            except AttributeError:
                result = self._paddle_ocr.ocr(image)
            
            if result is None:
                print("⚠️ PaddleOCR returned None")
                return []
                
            word_boxes = []
            
            # Normalize result to list
            if not isinstance(result, list):
                result = [result]
            
            # Debug: print what we got
            if len(result) > 0:
                print(f"  DEBUG: Result type: {type(result[0])}")
                
            for page_result in result:
                if page_result is None:
                    continue
                
                # Handle PaddleX OCRResult objects (has .json property)
                result_type = str(type(page_result))
                if "OCRResult" in result_type or hasattr(page_result, 'json'):
                    try:
                        # Extract via json property
                        json_data = {}
                        if hasattr(page_result, 'json'):
                            j = page_result.json
                            json_data = j() if callable(j) else j
                        if isinstance(json_data, str):
                            json_data = json.loads(json_data)
                        
                        # PaddleX v3 wraps data in "res" key
                        res = json_data.get('res', json_data)
                        
                        # Get texts, boxes, scores from res
                        texts = res.get('rec_texts', res.get('rec_text', []))
                        boxes = res.get('dt_polys', res.get('rec_polys', []))
                        scores = res.get('rec_scores', res.get('rec_score', [0.9] * len(texts)))
                        
                        # Convert numpy arrays to lists if needed
                        if hasattr(texts, 'tolist'):
                            texts = texts.tolist()
                        if hasattr(scores, 'tolist'):
                            scores = scores.tolist()
                        
                        print(f"  DEBUG: OCRResult parsed: {len(texts)} texts")
                        
                        for text, box, score in zip(texts, boxes, scores):
                            if not text or not str(text).strip():
                                continue
                            # Convert polygon to bbox
                            pts = np.array(box)
                            x, y = pts[:, 0].min(), pts[:, 1].min()
                            w = pts[:, 0].max() - x
                            h = pts[:, 1].max() - y
                            word_boxes.append({
                                "text": str(text),
                                "confidence": float(score) if score else 0.9,
                                "bbox": [int(x), int(y), int(w), int(h)],
                                "polygon": box if isinstance(box, list) else box.tolist()
                            })
                    except Exception as e:
                        print(f"⚠️ OCRResult parse error: {e}")
                        import traceback
                        traceback.print_exc()
                        
                # Handle legacy list format [[box, (text, conf)], ...]
                elif isinstance(page_result, list):
                    for line in page_result:
                        if not isinstance(line, (list, tuple)) or len(line) < 2:
                            continue
                        box, text_conf = line[0], line[1]
                        
                        if isinstance(text_conf, (list, tuple)):
                            text, conf = text_conf[0], text_conf[1] if len(text_conf) > 1 else 0.9
                        else:
                            text, conf = str(text_conf), 0.9
                            
                        if not text.strip():
                            continue
                            
                        pts = np.array(box)
                        x, y = pts[:, 0].min(), pts[:, 1].min()
                        w = pts[:, 0].max() - x
                        h = pts[:, 1].max() - y
                        
                        word_boxes.append({
                            "text": text,
                            "confidence": float(conf),
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "polygon": box
                        })
                        
            print(f"📝 PaddleOCR: {len(word_boxes)} words detected")
            return word_boxes
            
        except Exception as e:
            print(f"⚠️ PaddleOCR error: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    def _run_trocr(self, image: np.ndarray) -> str:
        """Run TrOCR on image region for handwriting."""
        self._init_trocr()
        
        if self._trocr_model is None or self._trocr_processor is None:
            return ""
            
        try:
            from PIL import Image
            import torch
            
            # Convert to PIL
            if isinstance(image, np.ndarray):
                pil_img = Image.fromarray(image)
            else:
                pil_img = image
                
            # Process
            pixel_values = self._trocr_processor(pil_img, return_tensors="pt").pixel_values
            
            if self.use_gpu and torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
                
            with torch.no_grad():
                generated_ids = self._trocr_model.generate(pixel_values, max_length=64)
                
            text = self._trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
            
        except Exception as e:
            print(f"⚠️ TrOCR error: {e}")
            return ""
            
    def _extract_zone(self, image: np.ndarray, zone: Tuple[float, float, float, float], 
                      word_boxes: List[Dict]) -> Tuple[str, float, str]:
        """
        Extract text from a zone using word boxes or TrOCR fallback.
        
        Returns: (text, confidence, method)
        """
        h, w = image.shape[:2]
        x, y, zw, zh = zone
        
        # Convert normalized to pixel coords
        px = int(x * w)
        py = int(y * h)
        pw = int(zw * w)
        ph = int(zh * h)
        
        # Find word boxes that overlap with zone
        zone_texts = []
        zone_confs = []
        
        for wb in word_boxes:
            bx, by, bw, bh = wb["bbox"]
            # Check overlap
            if (bx < px + pw and bx + bw > px and by < py + ph and by + bh > py):
                zone_texts.append(wb["text"])
                zone_confs.append(wb["confidence"])
                
        if zone_texts:
            text = " ".join(zone_texts)
            avg_conf = sum(zone_confs) / len(zone_confs)
            return text, avg_conf, "ocr"
            
        # Fallback: TrOCR for potentially handwritten zones
        if self.use_icr:
            try:
                crop = image[py:py+ph, px:px+pw]
                if crop.size > 0:
                    text = self._run_trocr(crop)
                    if text:
                        return text, 0.7, "icr"
            except Exception as e:
                print(f"⚠️ TrOCR fallback failed: {e}")
                
        return "", 0.0, "none"
        
    def extract(self, path: str) -> Dict[str, Any]:
        """
        Main extraction method.
        
        Args:
            path: Path to PDF or image file
            
        Returns:
            Extraction result dict with extracted_fields, field_details, etc.
        """
        start_time = time.time()
        result = ExtractionResult()
        
        print(f"\n{'='*60}")
        print(f"🏥 CMS-1500 Direct Pipeline")
        print(f"📄 Input: {path}")
        print(f"{'='*60}")
        
        # Step 1: Load image
        print("\n[1/4] Loading document...")
        image = self._load_image(path)
        
        if image is None:
            result.error = f"Failed to load: {path}"
            return result.__dict__
            
        print(f"  ✅ Image loaded: {image.shape}")
        
        # Step 2: Align to template
        print("\n[2/4] Aligning to template...")
        aligned, alignment_ok = self._align_to_template(image)
        result.alignment_success = alignment_ok
        
        if alignment_ok:
            print("  ✅ Alignment successful")
        else:
            print("  ⚠️ Alignment skipped/failed, using original")
            aligned = image
            
        # Step 3: Full-page OCR
        print("\n[3/4] Running OCR...")
        word_boxes = self._run_paddle_ocr(aligned)
        result.ocr_boxes = word_boxes
        
        if not word_boxes:
            print("  ⚠️ No text detected by PaddleOCR, will use TrOCR per-zone")
            # Don't run TrOCR on full page - let per-zone extraction handle it
        else:
            print(f"  ✅ Detected {len(word_boxes)} words")
            
        # Step 4: Extract fields from zones
        print("\n[4/4] Extracting fields...")
        
        for field_id, zone in self.TEMPLATE_ZONES.items():
            text, conf, method = self._extract_zone(aligned, zone, word_boxes)
            
            h, w = aligned.shape[:2]
            x, y, zw, zh = zone
            bbox = (int(x*w), int(y*h), int(zw*w), int(zh*h))
            
            result.extracted_fields[field_id] = text
            result.field_details.append({
                "field_id": field_id,
                "field_name": self.FIELD_NAMES.get(field_id, field_id.replace("_", " ").title()),
                "value": text,
                "confidence": conf,
                "bbox": bbox,
                "method": method,
            })
            
        # Count extracted
        extracted_count = sum(1 for f in result.field_details if f["value"].strip())
        total_count = len(result.field_details)
        
        result.processing_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ Extraction complete!")
        print(f"  📊 Fields: {extracted_count}/{total_count}")
        print(f"  ⏱️  Time: {result.processing_time:.2f}s")
        print(f"{'='*60}\n")
        
        return {
            "form_type": result.form_type,
            "extraction_method": result.extraction_method,
            "extracted_fields": result.extracted_fields,
            "field_details": result.field_details,
            "ocr_boxes": result.ocr_boxes,
            "processing_time": result.processing_time,
            "alignment_success": result.alignment_success,
        }


def extract_cms1500(path: str, use_icr: bool = True, use_gpu: bool = True) -> Dict[str, Any]:
    """
    Convenience function to extract CMS-1500 data.
    
    Args:
        path: Path to PDF or image
        use_icr: Use TrOCR for handwriting
        use_gpu: Use GPU acceleration
        
    Returns:
        Extraction result dict
    """
    pipeline = CMS1500DirectPipeline(use_icr=use_icr, use_gpu=use_gpu)
    return pipeline.extract(path)


# CLI test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cms1500_direct.py <path_to_pdf>")
        sys.exit(1)
        
    result = extract_cms1500(sys.argv[1])
    
    print("\n📋 Extracted Fields:")
    for field in result.get("field_details", []):
        if field["value"].strip():
            print(f"  {field['field_name']}: {field['value']} ({field['confidence']:.0%})")


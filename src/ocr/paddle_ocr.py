"""
PaddleOCR wrapper with auto-download support.
"""
import numpy as np
from typing import List, Tuple, Optional
from collections.abc import Mapping
from paddleocr import PaddleOCR
from utils.models import WordBox
from utils.config import Config


class PaddleOCRWrapper:
    """Wrapper for PaddleOCR with lazy initialization and optimized settings."""
    
    def __init__(
        self,
        lang: str = 'en',
        use_angle_cls: bool = False,
        use_gpu: Optional[bool] = None
    ):
        """
        Initialize PaddleOCR wrapper (lazy initialization).
        
        Args:
            lang: Language code (default: 'en')
            use_angle_cls: Whether to use angle classifier (default: False for faster init)
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = Config.USE_GPU if use_gpu is None else use_gpu
        self.ocr = None
        self._initialized = False
    
    def _initialize(self):
        """Initialize PaddleOCR model lazily (only when first used)."""
        if self._initialized:
            return
        
        print("🔄 Initializing PaddleOCR (this may take 30-60 seconds on first run)...")
        import time
        start_time = time.time()
        
        try:
            # Robust initialization logic
            try:
                # Try standard kwargs
                self.ocr = PaddleOCR(
                    lang=self.lang,
                    use_angle_cls=self.use_angle_cls,
                    use_gpu=self.use_gpu,
                    show_log=False
                )
            except Exception:
                # Fallback without optional kwargs
                print("⚠️ Retrying PaddleOCR init with minimal args...")
                self.ocr = PaddleOCR(
                    lang=self.lang, 
                    use_angle_cls=self.use_angle_cls
                )
            
            init_time = time.time() - start_time
            print(f"✅ PaddleOCR initialized in {init_time:.2f}s")
            self._initialized = True
        except Exception as e:
            print(f"❌ Error initializing PaddleOCR: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_text(self, image: np.ndarray) -> List[WordBox]:
        """
        Extract text from image with word-level bounding boxes.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of WordBox objects
        """
        if not self._initialized:
            self._initialize()
        
        word_boxes = []
        
        try:
            # Ensure image is valid and properly formatted for PaddleOCR
            if image is None or image.size == 0:
                print("⚠️ Empty image passed to OCR")
                return word_boxes
            
            # Ensure image is uint8 and in correct format
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Ensure image is RGB (3 channels) or grayscale (1 channel)
            if len(image.shape) == 2:
                # Grayscale - convert to RGB
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA - convert to RGB
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] != 3:
                print(f"⚠️ Unexpected image shape: {image.shape}")
                return word_boxes
            
            # Ensure minimum size
            if image.shape[0] < 10 or image.shape[1] < 10:
                print(f"⚠️ Image too small for OCR: {image.shape}")
                return word_boxes
            
            # Make image contiguous in memory (helps with PaddleOCR)
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
                
            # Run OCR (new API doesn't use cls parameter)
            result = self.ocr.ocr(image)
            
            # Debug: print result type
            if result is None:
                print("⚠️ PaddleOCR returned None")
                return word_boxes
            
            # Handle empty array check properly
            if isinstance(result, np.ndarray):
                if result.size == 0:
                    print("⚠️ PaddleOCR returned empty array")
                    return word_boxes
            elif not result:
                print("⚠️ PaddleOCR returned empty result")
                return word_boxes
            
            # Process results - handle modern PaddleOCR (PaddleX) objects or legacy lists
            # Result can be a list of page results or a single result
            if not isinstance(result, list):
                result = [result]
            
            # Debug: Check what we got
            print(f"🔍 PaddleOCR result type: {type(result)}, length: {len(result) if isinstance(result, list) else 'N/A'}")
            
            for page_idx, page_result in enumerate(result):
                if page_result is None:
                    print(f"⚠️ Page result {page_idx} is None")
                    continue
                
                print(f"🔍 Page {page_idx} result type: {type(page_result)}")
                
                # New PaddleX-style result is a dict-like OCRResult
                if isinstance(page_result, Mapping):
                    texts = list(page_result.get("rec_texts") or [])
                    print(f"🔍 Found {len(texts)} texts in Mapping format")
                    
                    # Handle scores
                    scores_raw = page_result.get("rec_scores")
                    if scores_raw is None:
                        scores = [1.0] * len(texts)
                    else:
                        # Convert numpy array to list if needed
                        if isinstance(scores_raw, np.ndarray):
                            if scores_raw.size == 0:
                                scores = [1.0] * len(texts)
                            else:
                                scores = scores_raw.tolist() if hasattr(scores_raw, 'tolist') else list(scores_raw)
                        else:
                            scores = list(scores_raw) if scores_raw else [1.0] * len(texts)
                        
                        if len(scores) != len(texts):
                            scores = [1.0] * len(texts)
                    
                    # Handle boxes - properly check for empty numpy arrays
                    boxes = None
                    for key in ["text_word_boxes", "rec_polys", "rec_boxes"]:
                        box_data = page_result.get(key)
                        if box_data is not None:
                            # Check if it's a numpy array
                            if isinstance(box_data, np.ndarray):
                                if box_data.size > 0:
                                    boxes = box_data.tolist() if hasattr(box_data, 'tolist') else list(box_data)
                                    break
                            elif box_data:  # Non-empty list or other iterable
                                boxes = list(box_data) if not isinstance(box_data, list) else box_data
                                break
                    
                    # Default to empty list if no boxes found
                    if boxes is None:
                        boxes = []
                    
                    print(f"🔍 Found {len(boxes)} boxes")
                    
                    # Match texts with boxes and scores
                    min_len = min(len(texts), len(boxes), len(scores))
                    if min_len == 0:
                        print(f"⚠️ No matching texts/boxes/scores (texts: {len(texts)}, boxes: {len(boxes)}, scores: {len(scores)})")
                        continue
                    
                    # Process matched pairs
                    for idx in range(min_len):
                        text = texts[idx]
                        score = scores[idx] if idx < len(scores) else 1.0
                        box = boxes[idx] if idx < len(boxes) else None
                        
                        if box is None:
                            continue
                        
                        # Extract text string
                        raw_text = text[0] if isinstance(text, (list, tuple)) else text
                        if isinstance(raw_text, np.ndarray):
                            if raw_text.size == 0:
                                continue
                            raw_text = raw_text.item() if raw_text.size == 1 else " ".join(map(str, raw_text.flatten().tolist()))
                        text_str = str(raw_text).strip()
                        if not text_str:
                            continue
                        
                        # Extract confidence score
                        raw_score = score[0] if isinstance(score, (list, tuple)) else score
                        if isinstance(raw_score, np.ndarray):
                            raw_score = raw_score.item() if raw_score.size == 1 else np.mean(raw_score)
                        confidence_val = float(raw_score)
                        
                        # Parse bounding box
                        if isinstance(box, np.ndarray):
                            box = box.tolist()
                        
                        if isinstance(box, (list, tuple)) and len(box) >= 4:
                            if isinstance(box[0], (list, tuple)):
                                # Quadrilateral format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                x_coords = [pt[0] for pt in box]
                                y_coords = [pt[1] for pt in box]
                                x0, x1 = min(x_coords), max(x_coords)
                                y0, y1 = min(y_coords), max(y_coords)
                            else:
                                # Simple format: [x0, y0, x1, y1]
                                x0, y0, x1, y1 = box[:4]
                            
                            word_boxes.append(
                                WordBox(
                                    text=text_str,
                                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                                    confidence=confidence_val,
                                )
                            )
                    continue
                
                # Legacy list-of-lines format: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
                if isinstance(page_result, (list, tuple)):
                    print(f"🔍 Processing {len(page_result)} lines in legacy format")
                    for line_idx, line in enumerate(page_result):
                        if line is None or not isinstance(line, (list, tuple)) or len(line) < 2:
                            continue
                        
                        try:
                            bbox_coords = line[0]
                            text_info = line[1]
                            
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                            elif isinstance(text_info, dict):
                                text = text_info.get('text', '')
                                confidence = text_info.get('confidence', 1.0)
                            else:
                                continue
                            
                            # Parse bounding box coordinates
                            if isinstance(bbox_coords, (list, tuple)) and len(bbox_coords) >= 4:
                                if isinstance(bbox_coords[0], (list, tuple)):
                                    # Quadrilateral format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                    x_coords = [pt[0] for pt in bbox_coords]
                                    y_coords = [pt[1] for pt in bbox_coords]
                                    x0 = min(x_coords)
                                    y0 = min(y_coords)
                                    x1 = max(x_coords)
                                    y1 = max(y_coords)
                                else:
                                    # Simple format: [x0, y0, x1, y1]
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
                            
                            word_boxes.append(
                                WordBox(
                                    text=text_str,
                                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                                    confidence=confidence_val
                                )
                            )
                        except (ValueError, TypeError, IndexError) as e:
                            if line_idx < 3:  # Only print first few errors
                                print(f"⚠️ Error parsing OCR line {line_idx}: {e}")
                            continue
                    
                    if len(word_boxes) > 0:
                        print(f"✅ Extracted {len(word_boxes)} word boxes from legacy format")
                    continue
                
                # Unknown format
                print(f"⚠️ Unknown page_result format: {type(page_result)}")
                print(f"   Content preview: {str(page_result)[:200]}")
        
        except Exception as e:
            print(f"❌ Error in PaddleOCR extraction: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"📊 Total word boxes extracted: {len(word_boxes)}")
        return word_boxes
    
    def extract_text_lines(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        """
        Extract text lines with bounding boxes.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of tuples: (text, confidence, bbox)
        """
        word_boxes = self.extract_text(image)
        
        # Group words into lines (simple approach: group by y-coordinate)
        lines = []
        current_line = []
        current_y = None
        
        for word_box in sorted(word_boxes, key=lambda wb: (wb.bbox[1], wb.bbox[0])):
            y_center = (word_box.bbox[1] + word_box.bbox[3]) / 2
            
            if current_y is None or abs(y_center - current_y) > 10:  # New line threshold
                if current_line:
                    # Combine words in line
                    line_text = " ".join([wb.text for wb in current_line])
                    line_confidence = sum([wb.confidence for wb in current_line]) / len(current_line)
                    
                    # Get line bbox
                    x0 = min([wb.bbox[0] for wb in current_line])
                    y0 = min([wb.bbox[1] for wb in current_line])
                    x1 = max([wb.bbox[2] for wb in current_line])
                    y1 = max([wb.bbox[3] for wb in current_line])
                    
                    lines.append((line_text, line_confidence, (x0, y0, x1, y1)))
                
                current_line = [word_box]
                current_y = y_center
            else:
                current_line.append(word_box)
        
        # Add last line
        if current_line:
            line_text = " ".join([wb.text for wb in current_line])
            line_confidence = sum([wb.confidence for wb in current_line]) / len(current_line)
            
            x0 = min([wb.bbox[0] for wb in current_line])
            y0 = min([wb.bbox[1] for wb in current_line])
            x1 = max([wb.bbox[2] for wb in current_line])
            y1 = max([wb.bbox[3] for wb in current_line])
            
            lines.append((line_text, line_confidence, (x0, y0, x1, y1)))
        
        return lines

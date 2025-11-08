"""
Tesseract OCR fallback wrapper.
"""
import numpy as np
import pytesseract
from typing import List
from PIL import Image
from utils.models import WordBox


class TesseractOCRWrapper:
    """Wrapper for Tesseract OCR as fallback."""
    
    def __init__(self, lang: str = 'eng'):
        """
        Initialize Tesseract OCR.
        
        Args:
            lang: Language code (default: 'eng')
        """
        self.lang = lang
    
    def extract_text(self, image: np.ndarray) -> List[WordBox]:
        """
        Extract text from image with word-level bounding boxes.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of WordBox objects
        """
        word_boxes = []
        
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(
                pil_image,
                lang=self.lang,
                output_type=pytesseract.Output.DICT
            )
            
            # Process results
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = float(data['conf'][i])
                
                # Skip empty text or low confidence
                if not text or conf < 0:
                    continue
                
                # Get bounding box
                x0 = data['left'][i]
                y0 = data['top'][i]
                x1 = x0 + data['width'][i]
                y1 = y0 + data['height'][i]
                
                word_box = WordBox(
                    text=text,
                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                    confidence=conf / 100.0  # Tesseract confidence is 0-100
                )
                word_boxes.append(word_box)
        
        except Exception as e:
            print(f"Error in Tesseract OCR extraction: {e}")
        
        return word_boxes
    
    def extract_text_lines(self, image: np.ndarray) -> List[tuple]:
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


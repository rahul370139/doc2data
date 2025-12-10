"""
TrOCR (Transformer OCR) Wrapper for ICR (Intelligent Character Recognition).
Uses Microsoft's TrOCR model which excels at handwriting recognition.
"""

import torch
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
import time
import cv2

# Lazy load transformers to avoid import errors if not installed
TrOCRProcessor = None
VisionEncoderDecoderModel = None

class TrOCRWrapper:
    """
    Wrapper for Microsoft's TrOCR model.
    Best for: Handwriting, noisy text, and "ICR" tasks.
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten", use_gpu: bool = True):
        global TrOCRProcessor, VisionEncoderDecoderModel
        
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"[TrOCR] Loading model: {model_name} on {self.device}...")
        
        try:
            # Lazy import
            from transformers import TrOCRProcessor as _TrOCRProcessor
            from transformers import VisionEncoderDecoderModel as _VisionEncoderDecoderModel
            TrOCRProcessor = _TrOCRProcessor
            VisionEncoderDecoderModel = _VisionEncoderDecoderModel
            
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            self.model.eval()  # Set to evaluation mode
            print("✓ TrOCR loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load TrOCR: {e}")
            self.model = None
            self.processor = None

    def predict(self, image: np.ndarray) -> str:
        """
        Run ICR on a single image crop.
        
        Args:
            image: numpy array (BGR or RGB) or PIL Image
            
        Returns:
            Extracted text string
        """
        if self.model is None or self.processor is None:
            return ""
            
        try:
            # Convert to PIL RGB
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    # Grayscale
                    pil_image = Image.fromarray(image).convert("RGB")
                elif image.shape[2] == 4:
                    # BGRA
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
                elif image.shape[2] == 3:
                    # Assume BGR from OpenCV, convert to RGB
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            else:
                return ""

            # Prepare image for model
            pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)

            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_length=64)
            
            # Decode
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
            
        except Exception as e:
            print(f"[TrOCR] Error: {e}")
            return ""

    def process_crops(self, images: List[np.ndarray]) -> List[str]:
        """Batch process list of image crops."""
        results = []
        for img in images:
            results.append(self.predict(img))
        return results


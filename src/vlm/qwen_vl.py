"""
Qwen-VL integration for figure/table processing via Ollama.
"""
import base64
import json
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image
from io import BytesIO

from utils.config import Config


class QwenVLProcessor:
    """
    Qwen-VL processor for vision-language tasks via Ollama.
    
    Returns default/empty results when disabled.
    To enable: Set ENABLE_VLM=true in .env and ensure Ollama is running.
    """
    
    def __init__(self, enabled: Optional[bool] = None):
        """
        Initialize Qwen-VL processor.
        
        Args:
            enabled: Whether to enable VLM processing (default: from Config.ENABLE_VLM)
        """
        self.enabled = enabled if enabled is not None else Config.ENABLE_VLM
        self.client = None
        self.model = None
        
        if self.enabled:
            try:
                from src.vlm.ollama_client import get_ollama_client
                self.client = get_ollama_client()
                self.model = Config.OLLAMA_MODEL_VLM
                print("✓ VLM processing enabled (Ollama required)")
            except Exception as e:
                print(f"⚠ VLM enabled but Ollama unavailable: {e}")
                print("  Continuing with stub (no VLM processing)")
                self.enabled = False
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """
        Convert numpy image to base64 string.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Base64-encoded image string
        """
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(image).convert('RGB')
        
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def classify_figure(
        self,
        image: np.ndarray,
        caption: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify figure type using Qwen-VL - STUBBED.
        
        When disabled, returns default "other" type.
        
        Args:
            image: Figure image
            caption: Optional caption text
            
        Returns:
            Dictionary with figure_type and confidence
        """
        if not self.enabled:
            return {"figure_type": "other", "confidence": 0.0}
        
        # Convert image to base64
        image_b64 = self.image_to_base64(image)
        
        # Build prompt
        prompt = """Analyze this image and classify it into one of these categories:
- bar: Bar chart
- line: Line chart
- pie: Pie chart
- scatter: Scatter plot
- non_chart_image: Regular image (photo, diagram, etc.)
- diagram: Technical diagram or flowchart
- other: Other type

Respond with ONLY a JSON object in this exact format:
{"figure_type": "category_name", "confidence": 0.0-1.0}"""
        
        if caption:
            prompt += f"\n\nCaption: {caption}"
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                images=[image_b64],
                format="json"
            )
            
            # Parse response
            response_text = response.get("response", "")
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                return {
                    "figure_type": result.get("figure_type", "other"),
                    "confidence": float(result.get("confidence", 0.5))
                }
        
        except Exception as e:
            print(f"Error in figure classification: {e}")
        
        return {"figure_type": "other", "confidence": 0.0}
    
    def extract_chart_data(
        self,
        image: np.ndarray,
        chart_type: str
    ) -> Dict[str, Any]:
        """
        Extract data from chart using Qwen-VL - STUBBED.
        
        When disabled, returns empty data structure.
        
        Args:
            image: Chart image
            chart_type: Type of chart (bar, line, pie, scatter)
            
        Returns:
            Dictionary with extracted data series
        """
        if not self.enabled:
            return {
                "axes": {},
                "units": {},
                "series": [],
                "confidence": 0.0
            }
        
        image_b64 = self.image_to_base64(image)
        
        prompt = f"""Extract the data from this {chart_type} chart.
Return ONLY a JSON object with this structure:
{{
  "axes": {{"x": "label", "y": "label"}},
  "units": {{"x": "unit", "y": "unit"}},
  "series": [
    {{"name": "series_name", "data": [{{"x": value, "y": value}}]}}
  ],
  "confidence": 0.0-1.0
}}"""
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                images=[image_b64],
                format="json"
            )
            
            response_text = response.get("response", "")
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                return result
        
        except Exception as e:
            print(f"Error in chart data extraction: {e}")
        
        return {
            "axes": {},
            "units": {},
            "series": [],
            "confidence": 0.0
        }
    
    def process_table(
        self,
        image: np.ndarray,
        ocr_tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process table image and extract structured data using Qwen-VL - STUBBED.
        
        When disabled, returns empty table structure.
        
        Args:
            image: Table image
            ocr_tokens: Optional OCR tokens from previous processing
            
        Returns:
            Dictionary with table structure and data
        """
        if not self.enabled:
            return {
                "shape": [0, 0],
                "headers": [],
                "header_rows": 0,
                "body": [],
                "units": None,
                "confidence": 0.0
            }
        
        image_b64 = self.image_to_base64(image)
        
        prompt = """Extract the table structure and data from this image.
Return ONLY a JSON object with this structure:
{
  "shape": [rows, cols],
  "headers": [["header1", "header2", ...]],
  "header_rows": number_of_header_rows,
  "body": [["cell1", "cell2", ...], ...],
  "units": ["unit1", "unit2", ...] (optional),
  "confidence": 0.0-1.0
}"""
        
        if ocr_tokens:
            prompt += f"\n\nOCR tokens for reference: {', '.join(ocr_tokens[:20])}"
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                images=[image_b64],
                format="json"
            )
            
            response_text = response.get("response", "")
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                return result
        
        except Exception as e:
            print(f"Error in table processing: {e}")
        
        return {
            "shape": [0, 0],
            "headers": [],
            "header_rows": 0,
            "body": [],
            "units": None,
            "confidence": 0.0
        }


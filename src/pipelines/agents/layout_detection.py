"""
Layout Detection Agent - detects document layout blocks.

PURPOSE: Segments page into blocks (text, table, figure, form_field, checkbox).
For CMS-1500: uses YOLOv8 (if configured) or template zones from schema.
For general forms: uses Detectron2 (PubLayNet) or PaddleDetection.

USE CASE: Pipeline calls this after form ID and alignment. Output blocks
feed into OCR agent. No manual use; part of MultiAgentPipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from src.pipelines.core import BaseAgent, BlockType, DetectedBlock, FormType, PipelineConfig

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.config import Config


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
        self._layout_backend = None
    
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
        
        # Initialize layout detection model for general forms.
        # Prefer Detectron2 weights if available (better for some docs), otherwise use PaddleDetection.
        # PaddleDetection ppyolov2 (PubLayNet) is cached at:
        #   /root/.torch/iopath_cache/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet/
        # It produces 5 classes: Text, Title, List, Table, Figure
        from pathlib import Path
        layout_pref = (self.config.layout_model or "auto").lower().strip()

        detectron_candidates = [
            Path("/root/.detectron2/models/publaynet_faster_rcnn_R_50_FPN_3x.pth"),
            Path("/app/models/publaynet_faster_rcnn_R_50_FPN_3x.pth"),
            Config.PROJECT_ROOT / "models" / "publaynet_faster_rcnn_R_50_FPN_3x.pth",
            Config.PROJECT_ROOT / "models" / "detectron2" / "publaynet_faster_rcnn_R_50_FPN_3x.pth",
        ]
        detectron_model = next((p for p in detectron_candidates if p.exists()), None)

        def _init_detectron(model_path: Path) -> bool:
            try:
                import layoutparser as lp
                config_uri = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
                self._detectron = lp.Detectron2LayoutModel(
                    config_uri,
                    model_path=str(model_path),
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.config.detectron_threshold],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                )
                self._layout_backend = "detectron2"
                self.log(f"✅ Detectron2 initialized ({model_path})")
                return True
            except Exception as e:
                self.log(f"⚠️ Detectron2 init failed: {e}")
                return False

        def _init_paddle() -> bool:
            try:
                import layoutparser as lp
                cached_paddle = Path("/root/.torch/iopath_cache/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet/inference.pdiparams")
                if cached_paddle.exists():
                    self.log(f"Found cached PaddleDetection weights at {cached_paddle.parent}")
                self._detectron = lp.PaddleDetectionLayoutModel(
                    config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config",
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                    extra_config={"threshold": self.config.detectron_threshold},
                )
                self._layout_backend = "paddle"
                self.log("✅ PaddleDetection (PubLayNet ppyolov2) initialized for general forms")
                return True
            except Exception as e:
                self.log(f"⚠️ PaddleDetection init failed: {e}")
                return False

        prefer_detectron = layout_pref in ("detectron2", "detectron")
        prefer_paddle = layout_pref in ("paddle", "layoutparser", "publaynet")

        initialized = False
        if prefer_detectron or (layout_pref == "auto" and detectron_model is not None):
            if detectron_model is None:
                self.log("❌ No Detectron2 weights found locally, skipping Detectron2 init")
            else:
                initialized = _init_detectron(detectron_model)

        if not initialized and (prefer_paddle or layout_pref == "auto"):
            initialized = _init_paddle()

        if not initialized and not prefer_detectron and detectron_model is not None:
            initialized = _init_detectron(detectron_model)
        
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
            backend = self._layout_backend or "layoutparser"
            
            for i, element in enumerate(layout):
                block_type = self._map_detectron_type(element.type)
                bbox = (element.block.x_1, element.block.y_1, 
                       element.block.x_2, element.block.y_2)
                
                blocks.append(DetectedBlock(
                    id=f"det-{i}",
                    block_type=block_type,
                    bbox=bbox,
                    confidence=element.score,
                    metadata={"model": backend}
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
            backend = self._layout_backend or "layoutparser"
            self.log(f"Using {backend} for general detection")
            return self._detectron_detect(image)
        else:
            self.log("No layout model available, returning empty")
            return []

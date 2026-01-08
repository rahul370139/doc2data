"""
YOLOv8-based layout detector for CMS-1500.
Uses Ultralytics YOLO for detection; optional SAHI tiling if installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from utils.models import Block, BlockType


class YOLOLayoutDetector:
    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.6,
        imgsz: int = 1280,
        use_sahi: bool = False,
        tile_size: int = 640,
        tile_overlap: float = 0.15,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics not installed. pip install ultralytics") from exc

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.use_sahi = use_sahi
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        # class name mapping to block types
        self._class_map: Dict[str, BlockType] = {
            "table": BlockType.TABLE,
            "figure": BlockType.FIGURE,
            "text": BlockType.TEXT,
            "title": BlockType.TITLE,
            "form": BlockType.FORM_FIELD,
            "field": BlockType.FORM_FIELD,
            "checkbox": BlockType.CHECKBOX,
            "section": BlockType.LIST,
            "header": BlockType.HEADER,
            "signature": BlockType.SIGNATURE,
            "footer": BlockType.FOOTER,
        }

    def _predict_single(self, image: np.ndarray):
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        return results

    def _predict_sahi(self, image: np.ndarray):
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ImportError as exc:
            raise ImportError("sahi not installed. pip install sahi") from exc

        det_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(self.model.model),
            confidence_threshold=self.conf,
            image_size=self.imgsz,
            device=self.model.device,
        )
        pred = get_sliced_prediction(
            image,
            detection_model=det_model,
            slice_height=self.tile_size,
            slice_width=self.tile_size,
            overlap_height_ratio=self.tile_overlap,
            overlap_width_ratio=self.tile_overlap,
        )
        return pred.to_coco_predictions()

    def predict(self, image: np.ndarray, page_id: int = 0) -> List[Block]:
        if image is None or not hasattr(image, "shape"):
            return []
        h, w = image.shape[:2]
        blocks: List[Block] = []

        if self.use_sahi:
            detections = self._predict_sahi(image)
            for det in detections:
                x0, y0, bw, bh = det.bbox.to_xywh()
                x1 = x0 + bw
                y1 = y0 + bh
                cls_name = det.category_name.lower()
                btype = self._class_map.get(cls_name, BlockType.FORM)
                blocks.append(
                    Block(
                        id=f"yolo-{page_id}-{len(blocks)}",
                        type=btype,
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
                        page_id=page_id,
                        confidence=float(det.score.value),
                        metadata={
                            "model_backend": "yolov8-sahi",
                            "class_name": cls_name,
                        },
                    )
                )
            return blocks

        results = self._predict_single(image)
        if not results:
            return []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy().flatten().tolist()
                if len(xyxy) < 4:
                    continue
                x0, y0, x1, y1 = xyxy[:4]
                cls_idx = int(box.cls.item()) if hasattr(box, "cls") else 0
                cls_name = r.names.get(cls_idx, "form").lower() if hasattr(r, "names") else "form"
                btype = self._class_map.get(cls_name, BlockType.FORM)
                conf = float(box.conf.item()) if hasattr(box, "conf") else 0.0
                blocks.append(
                    Block(
                        id=f"yolo-{page_id}-{len(blocks)}",
                        type=btype,
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
                        page_id=page_id,
                        confidence=conf,
                        metadata={
                            "model_backend": "yolov8",
                            "class_name": cls_name,
                        },
                    )
                )
        return blocks

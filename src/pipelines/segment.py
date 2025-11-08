"""
Layout segmentation pipeline using LayoutParser with optional heuristics.
"""
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import layoutparser as lp
import numpy as np
from utils.models import Block, BlockType, TableBlock, FigureBlock
from utils.config import Config

try:
    from layoutparser.models.paddledetection.catalog import LABEL_MAP_CATALOG
except Exception:  # pragma: no cover - layoutparser minimal install
    LABEL_MAP_CATALOG = {}


class LayoutSegmenter:
    """Layout segmentation using LayoutParser."""
    
    LOCAL_MODEL_FOLDERS = {
        "lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config": "ppyolov2_r50vd_dcn_365e_publaynet",
        "lp://TableBank/ppyolov2_r50vd_dcn_365e/config": "ppyolov2_r50vd_dcn_365e_tableBank_word",
    }
    DATASET_HINTS = {
        "lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config": "PubLayNet",
        "lp://TableBank/ppyolov2_r50vd_dcn_365e/config": "TableBank",
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        score_threshold: float = 0.25,  # Lower default for more granular detection
        target_size: Optional[Tuple[int, int]] = None,
        table_threshold: float = 0.25,
        enable_table_model: bool = True,
        split_large_blocks: bool = True,  # Enable splitting large blocks
        max_block_area_ratio: float = 0.3  # Max block area as ratio of page
    ):
        """
        Initialize layout segmenter.
        
        Args:
            model_name: Model name override (defaults to Config.LAYOUT_MODEL)
            score_threshold: Confidence threshold for detection (lower = more granular)
            target_size: Optional resize target [height, width]
            split_large_blocks: Split large blocks into smaller components
            max_block_area_ratio: Maximum block area as ratio of page (default: 0.3 = 30%)
        """
        self.model_name = (model_name or Config.LAYOUT_MODEL or "publaynet").lower()
        self.model = None
        self.model_backend = None
        self.model_config = None
        self.extra_config = {
            "threshold": score_threshold,
        }
        if target_size:
            self.extra_config["target_size"] = list(target_size)
        self.table_threshold = table_threshold
        self.enable_table_model = enable_table_model
        self.table_model = None
        self.split_large_blocks = split_large_blocks
        self.max_block_area_ratio = max_block_area_ratio
        self._initialize()
    
    def _layout_to_blocks(self, layout, page_id: int) -> List[Block]:
        """Convert LayoutParser elements to internal Block objects."""
        blocks: List[Block] = []
        if layout is None:
            return blocks
        
        for i, element in enumerate(layout):
            block_type_str = element.type
            try:
                if hasattr(element, "block"):
                    bbox = element.block
                    if hasattr(bbox, "coordinates"):
                        coords = bbox.coordinates
                        if isinstance(coords, (list, tuple)) and len(coords) >= 4:
                            x0, y0, x1, y1 = map(float, coords[:4])
                        else:
                            raise ValueError("Invalid coordinate list")
                    elif hasattr(bbox, "x_1"):
                        x0 = float(bbox.x_1)
                        y0 = float(bbox.y_1)
                        x1 = float(bbox.x_2)
                        y1 = float(bbox.y_2)
                    else:
                        raise ValueError("Cannot extract bbox coordinates")
                else:
                    raise ValueError("Layout element missing block attribute")
            except Exception as bbox_error:
                print(f"  Warning: Could not parse bbox for element {i}: {bbox_error}")
                continue
            
            block_type = self._map_block_type(block_type_str)
            block_id = f"block_{page_id}_{i}"
            confidence = 1.0
            if hasattr(element, "score"):
                confidence = float(element.score)
            elif hasattr(element, "confidence"):
                confidence = float(element.confidence)
            
            metadata = {
                "model_config": self.model_config or "unknown",
                "model_backend": self.model_backend or ("heuristic" if self.use_heuristic else "unknown")
            }
            block_kwargs = {
                "id": block_id,
                "type": block_type,
                "bbox": (x0, y0, x1, y1),
                "page_id": page_id,
                "confidence": confidence,
                "metadata": metadata
            }
            if block_type == BlockType.TABLE:
                block = TableBlock(**block_kwargs)
            elif block_type == BlockType.FIGURE:
                block = FigureBlock(**block_kwargs)
            else:
                block = Block(**block_kwargs)
            blocks.append(block)
        
        return blocks
    
    def _load_local_model(self, config_uri: str, extra_config: Optional[Dict[str, Any]] = None):
        """Try to load PaddleDetection weights from local cache when offline."""
        folder = self.LOCAL_MODEL_FOLDERS.get(config_uri)
        if not folder:
            return None
        
        dataset_name = self.DATASET_HINTS.get(config_uri)
        label_map = None
        if dataset_name and LABEL_MAP_CATALOG:
            label_map = LABEL_MAP_CATALOG.get(dataset_name)
        
        search_dirs = [
            Config.MODEL_CACHE_DIR / folder,
            Path.home() / ".torch" / "iopath_cache" / "model" / "layout-parser" / folder
        ]
        
        for local_dir in search_dirs:
            if not local_dir.exists():
                continue
            pdmodel = local_dir / "inference.pdmodel"
            pdparams = local_dir / "inference.pdiparams"
            if not pdmodel.exists() or not pdparams.exists():
                print(f"  ⚠ Missing Paddle inference files under {local_dir}")
                continue
            try:
                print(f"  ↪ Loading cached weights from {local_dir}")
                return lp.PaddleDetectionLayoutModel(
                    config_path=str(pdmodel),
                    model_path=str(local_dir),
                    label_map=label_map,
                    extra_config=extra_config or {}
                )
            except Exception as err:
                print(f"  ⚠ Failed to load local weights at {local_dir}: {err}")
        
        return None
    
    def _initialize(self):
        """Initialize LayoutParser model (auto-downloads if needed)."""
        self.use_heuristic = False
        self.model_backend = None
        self.model_config = None
        
        # Check device availability
        device = "cpu"
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                print("ℹ MPS (Apple Silicon) detected, but LayoutParser will use CPU")
            elif torch.cuda.is_available():
                device = "cuda"
                print(f"ℹ CUDA detected, using device: {device}")
        except Exception:
            pass
        
        model_candidates = self._get_model_candidates(self.model_name)
        
        for backend, config_uri in model_candidates:
            try:
                if backend == "paddle":
                    print(f"Attempting PaddleDetection model: {config_uri}")
                    model = lp.PaddleDetectionLayoutModel(
                        config_uri,
                        extra_config=self.extra_config
                    )
                elif backend == "auto":
                    print(f"Attempting AutoLayoutModel: {config_uri}")
                    model = lp.AutoLayoutModel(config_uri)
                elif backend == "detectron2":
                    print(f"Attempting Detectron2 model: {config_uri}")
                    model = lp.Detectron2LayoutModel(config_uri)
                else:
                    continue
                
                if model is not None:
                    self.model = model
                    self.model_backend = backend
                    self.model_config = config_uri
                    print(f"✓ Using {backend} backend with config: {config_uri}")
                    if device != "cpu":
                        print(f"  ℹ Layout models currently execute on CPU; detected device {device}")
                    break
            except Exception as load_err:
                print(f"  {backend} loader failed ({config_uri}): {load_err}")
                local_model = self._load_local_model(config_uri, self.extra_config if backend == "paddle" else None)
                if local_model is not None:
                    self.model = local_model
                    self.model_backend = f"{backend}-local"
                    self.model_config = config_uri
                    print(f"✓ Using cached weights for {config_uri}")
                    break
        
        if self.model is None:
            print("⚠ LayoutParser models unavailable, using heuristic fallback")
            self.use_heuristic = True
        else:
            self.use_heuristic = False
        
        # Attempt to load auxiliary table detector when primary model is available
        if not self.use_heuristic and self.enable_table_model:
            try:
                print("Attempting TableBank table detector...")
                print("  (This may download 221MB model on first run - already cached if downloaded before)")
                table_uri = "lp://TableBank/ppyolov2_r50vd_dcn_365e/config"
                self.table_model = lp.PaddleDetectionLayoutModel(
                    table_uri,
                    extra_config={
                        "threshold": self.table_threshold,
                    }
                )
                print("✓ Loaded TableBank table detector")
            except Exception as table_err:
                print(f"⚠ TableBank detector unavailable: {table_err}")
                # Try cached TableBank weights
                cached = self._load_local_model(
                    "lp://TableBank/ppyolov2_r50vd_dcn_365e/config",
                    extra_config={"threshold": self.table_threshold}
                )
                if cached is not None:
                    print("✓ Loaded TableBank table detector from cache")
                    self.table_model = cached
                else:
                    self.table_model = None
    
    def _get_model_candidates(self, model_key: str) -> List[Tuple[str, str]]:
        """
        Get ordered list of (backend, config_uri) tuples to try for LayoutParser.
        
        Args:
            model_key: Name of requested model
            
        Returns:
            List of backend/config tuples
        """
        key = (model_key or "publaynet").lower()
        
        if key in ("publaynet", "default", "text"):
            return [
                ("paddle", "lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config"),
                ("paddle", "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"),
                ("auto", "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"),
            ]
        if key in ("prima", "magazine"):
            return [
                ("paddle", "lp://PrimaLayout/ppyolov2_r50vd_dcn_365e/config"),
                ("auto", "lp://PrimaLayout/ppyolov2_r50vd_dcn_365e/config"),
            ]
        if key in ("docbank",):
            return [
                ("auto", "lp://DocBank/faster_rcnn_R_101_FPN_3x/config"),
            ]
        if key in ("tablebank", "tables"):
            return [
                ("auto", "lp://TableBank/faster_rcnn_R_101_FPN_3x/config"),
                ("auto", "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"),
            ]
        if key in ("forms", "funsd"):
            # FUNSD focuses on forms; fall back to general-purpose models if unavailable
            return [
                ("auto", "lp://PrimaLayout/ppyolov2_r50vd_dcn_365e/config"),
                ("auto", "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"),
            ]
        
        # Fallback to default PubLayNet attempts
        return [
            ("paddle", "lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config"),
            ("auto", "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"),
        ]
    
    @staticmethod
    def _clip_bbox(
        bbox: Tuple[float, float, float, float],
        width: int,
        height: int
    ) -> Tuple[float, float, float, float]:
        """Ensure bounding box stays within image bounds."""
        x0, y0, x1, y1 = bbox
        x0 = max(0.0, min(float(width - 1), x0))
        y0 = max(0.0, min(float(height - 1), y0))
        x1 = max(x0 + 1.0, min(float(width), x1))
        y1 = max(y0 + 1.0, min(float(height), y1))
        return (x0, y0, x1, y1)
    
    @staticmethod
    def _calculate_iou(
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def _detect_form_candidates(
        self,
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect form-like regions using line density heuristics.
        
        Args:
            image: Page image
            
        Returns:
            List of candidate dictionaries with bbox and scores
        """
        try:
            import cv2
        except ImportError:
            print("⚠ OpenCV not available; skipping form detection heuristics")
            return []
        
        if image is None or image.size == 0:
            return []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            6
        )
        
        width = image.shape[1]
        height = image.shape[0]
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(12, width // 40), 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(12, height // 40))
        )
        
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        
        form_mask = cv2.bitwise_or(horizontal, vertical)
        form_mask = cv2.dilate(form_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        
        contours, _ = cv2.findContours(form_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(width * height)
        min_area = max(2000.0, image_area * 0.002)
        max_area = image_area * 0.75
        
        candidates: List[Dict[str, Any]] = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            
            if area < min_area or area > max_area or w < 60 or h < 60:
                continue
            
            region_mask = form_mask[y:y + h, x:x + w]
            horizontal_region = horizontal[y:y + h, x:x + w]
            vertical_region = vertical[y:y + h, x:x + w]
            
            line_density = cv2.countNonZero(region_mask) / area
            horizontal_density = cv2.countNonZero(horizontal_region) / area
            vertical_density = cv2.countNonZero(vertical_region) / area
            
            if line_density < 0.035:
                continue
            if horizontal_density < 0.01 or vertical_density < 0.01:
                continue
            
            aspect_ratio = w / float(h)
            if aspect_ratio > 9.0 or aspect_ratio < 0.18:
                continue
            
            confidence = float(min(0.95, 0.35 + line_density * 3.0))
            candidates.append({
                "bbox": (float(x), float(y), float(x + w), float(y + h)),
                "density": float(line_density),
                "horizontal_density": float(horizontal_density),
                "vertical_density": float(vertical_density),
                "confidence": confidence
            })
        
        return candidates
    
    def _apply_form_enhancement(
        self,
        image: np.ndarray,
        blocks: List[Block],
        page_id: int
    ) -> List[Block]:
        """
        Enhance block list with heuristic form detection.
        
        Args:
            image: Page image
            blocks: Existing blocks
            page_id: Page identifier
            
        Returns:
            Updated block list
        """
        candidates = self._detect_form_candidates(image)
        if not candidates:
            return blocks
        
        updated_blocks: List[Block] = list(blocks)
        table_blocks = [b for b in updated_blocks if b.type == BlockType.TABLE]
        
        for candidate in candidates:
            cand_bbox = candidate["bbox"]
            
            # Skip if heavily overlapping with table
            overlaps_table = any(
                self._calculate_iou(cand_bbox, tbl.bbox) > 0.6
                for tbl in table_blocks
            )
            if overlaps_table:
                continue
            
            best_block: Optional[Block] = None
            best_iou = 0.0
            for block in updated_blocks:
                iou = self._calculate_iou(cand_bbox, block.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_block = block
            
            if best_block and best_iou >= 0.45 and best_block.type not in (BlockType.TABLE, BlockType.FIGURE):
                prev_type = best_block.type
                best_block.type = BlockType.FORM
                best_block.confidence = max(best_block.confidence, candidate["confidence"])
                best_block.metadata = dict(best_block.metadata)
                best_block.metadata.update({
                    "detected_by": "form_heuristic",
                    "form_density": candidate["density"],
                    "form_previous_type": prev_type.value if hasattr(prev_type, "value") else str(prev_type),
                    "form_iou": best_iou,
                    "form_detector": "line_density"
                })
            else:
                clipped_bbox = self._clip_bbox(cand_bbox, image.shape[1], image.shape[0])
                new_block = Block(
                    id=f"{page_id}-form-{len(updated_blocks)}",
                    type=BlockType.FORM,
                    bbox=clipped_bbox,
                    page_id=page_id,
                    confidence=candidate["confidence"],
                    metadata={
                        "detected_by": "form_heuristic",
                        "form_density": candidate["density"],
                        "form_iou": best_iou,
                        "form_detector": "line_density",
                        "model_backend": "form_heuristic",
                        "model_config": "heuristic_line_density"
                    }
                )
                updated_blocks.append(new_block)
        
        return updated_blocks
    
    def detect_layout_heuristic(self, image: np.ndarray, page_id: int = 0) -> List[Block]:
        """
        Heuristic-based layout detection using contour analysis.
        Fallback when LayoutParser models are unavailable.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of Block objects
        """
        import cv2
        
        blocks = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to get text regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and create blocks
        min_area = (image.shape[0] * image.shape[1]) * 0.001  # 0.1% of image area
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = float(x), float(y), float(x + w), float(y + h)
            
            # Simple heuristics for block type
            aspect_ratio = w / h if h > 0 else 0
            
            if h < image.shape[0] * 0.05:  # Very short = likely header/footer
                block_type = BlockType.TEXT
            elif aspect_ratio > 2:  # Wide = likely table or text region
                block_type = BlockType.TEXT
            elif aspect_ratio < 0.5:  # Tall = likely list or sidebar
                block_type = BlockType.LIST
            else:
                block_type = BlockType.TEXT
            
            block = Block(
                id=f"block_{i}",
                type=block_type,
                bbox=(x0, y0, x1, y1),
                page_id=page_id,
                confidence=0.7  # Lower confidence for heuristic
            )
            block.metadata.update({
                "detected_by": "heuristic_contour",
                "model_backend": "heuristic",
                "model_config": "contour_detection"
            })
            blocks.append(block)
        
        return blocks
    
    def detect_layout(self, image: np.ndarray, page_id: int = 0) -> List[Block]:
        """
        Detect layout blocks in image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of Block objects
        """
        if self.model is None and not self.use_heuristic:
            self._initialize()
        
        blocks = []
        
        # Use heuristic if model unavailable
        if self.use_heuristic or self.model is None:
            return self.detect_layout_heuristic(image, page_id)
        
        try:
            layout = self.model.detect(image)
            blocks = self._layout_to_blocks(layout, page_id)
            
            # Check if model only returned one giant figure block - this is a common failure mode
            if len(blocks) == 1 and blocks[0].type == BlockType.FIGURE:
                block_area = (blocks[0].bbox[2] - blocks[0].bbox[0]) * (blocks[0].bbox[3] - blocks[0].bbox[1])
                page_area = image.shape[0] * image.shape[1]
                if block_area / page_area > 0.5:  # Block covers >50% of page
                    print(f"  ⚠️ Model returned single large figure block ({block_area/page_area:.1%} of page)")
                    print(f"  🔄 Retrying with lower threshold and heuristic augmentation...")
                    # Retry with lower threshold
                    retry_threshold = 0.05
                    current_threshold = getattr(self.model, "threshold", self.extra_config.get("threshold", 0.25))
                    new_threshold = max(retry_threshold, current_threshold * 0.4)
                    if new_threshold < current_threshold - 1e-3:
                        self.model.threshold = new_threshold
                        self.extra_config["threshold"] = new_threshold
                        layout = self.model.detect(image)
                        blocks = self._layout_to_blocks(layout, page_id)
                        print(f"  ✓ Retry with threshold {new_threshold:.2f} returned {len(blocks)} blocks")
            
            # Automatic threshold back-off if we get too few blocks
            retry_threshold = 0.05
            current_threshold = getattr(self.model, "threshold", self.extra_config.get("threshold", 0.25))
            if self._should_retry_with_lower_threshold(blocks) and current_threshold - retry_threshold > 1e-3:
                new_threshold = max(retry_threshold, current_threshold * 0.5)
                if new_threshold < current_threshold - 1e-3:
                    print(f"  ℹ Low layout recall (only {len(blocks)} blocks); lowering threshold {current_threshold:.2f}→{new_threshold:.2f}")
                    self.model.threshold = new_threshold
                    self.extra_config["threshold"] = new_threshold
                    layout = self.model.detect(image)
                    blocks = self._layout_to_blocks(layout, page_id)
        
        except Exception as e:
            print(f"Error in layout detection: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to heuristic
            print("Falling back to heuristic detection...")
            blocks = self.detect_layout_heuristic(image, page_id)
        
        return blocks
    
    def _map_block_type(self, type_str: str) -> BlockType:
        """Map LayoutParser type to BlockType."""
        if isinstance(type_str, (int, float)):
            # PaddleDetection sometimes returns numeric class ids when label map is missing
            numeric_mapping = {
                0: BlockType.TEXT,
                1: BlockType.TITLE,
                2: BlockType.LIST,
                3: BlockType.TABLE,
                4: BlockType.FIGURE,
            }
            return numeric_mapping.get(int(type_str), BlockType.UNKNOWN)
        
        key = (type_str or "").strip().lower()
        mapping = {
            "text": BlockType.TEXT,
            "paragraph": BlockType.TEXT,
            "title": BlockType.TITLE,
            "heading": BlockType.TITLE,
            "list": BlockType.LIST,
            "table": BlockType.TABLE,
            "figure": BlockType.FIGURE,
            "chart": BlockType.FIGURE,
            "form": BlockType.FORM,
            "formfield": BlockType.FORM,
            "form_field": BlockType.FORM,
            "keyvalue": BlockType.FORM,
            "kvpair": BlockType.FORM,
            "field": BlockType.FORM
        }
        return mapping.get(key, BlockType.UNKNOWN)
    
    def _should_retry_with_lower_threshold(self, blocks: List[Block]) -> bool:
        """Decide if we should rerun detection with a lower threshold."""
        if not blocks:
            return True
        counts = Counter(block.type for block in blocks)
        textual = counts.get(BlockType.TEXT, 0) + counts.get(BlockType.TITLE, 0) + counts.get(BlockType.LIST, 0)
        tables = counts.get(BlockType.TABLE, 0)
        figures = counts.get(BlockType.FIGURE, 0)
        
        # If we only have one giant figure or no textual/table detections, try again
        if textual == 0 and tables == 0 and figures <= 1:
            return True
        if len(blocks) <= 2 and textual == 0:
            return True
        return False
    
    def _filter_new_blocks(
        self,
        existing_blocks: List[Block],
        new_blocks: List[Block],
        iou_threshold: float = 0.4
    ) -> List[Block]:
        """Filter out new blocks that significantly overlap with existing ones."""
        filtered = []
        for block in new_blocks:
            overlaps = any(
                self._calculate_iou(block.bbox, existing.bbox) >= iou_threshold
                for existing in existing_blocks
            )
            if not overlaps:
                filtered.append(block)
        return filtered
    
    def _detect_tables_with_model(
        self,
        image: np.ndarray,
        page_id: int
    ) -> List[Block]:
        """Detect tables using auxiliary TableBank model."""
        if not self.table_model:
            return []
        
        table_blocks: List[Block] = []
        try:
            layout = self.table_model.detect(image)
            for idx, element in enumerate(layout):
                if str(element.type).lower() != "table":
                    continue
                
                bbox = element.block.coordinates if hasattr(element.block, "coordinates") else element.block
                if hasattr(bbox, "x_1"):
                    x0, y0, x1, y1 = float(bbox.x_1), float(bbox.y_1), float(bbox.x_2), float(bbox.y_2)
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x0, y0, x1, y1 = map(float, bbox[:4])
                else:
                    continue
                
                width = image.shape[1]
                height = image.shape[0]
                area = (x1 - x0) * (y1 - y0)
                page_area = max(float(width * height), 1.0)
                area_ratio = area / page_area
                
                # Skip near-page-sized detections to avoid swallowing everything
                if area_ratio > 0.55:
                    continue
                if (x1 - x0) < 80 or (y1 - y0) < 80:
                    continue
                
                confidence = 1.0
                if hasattr(element, "score"):
                    confidence = float(element.score)
                elif hasattr(element, "confidence"):
                    confidence = float(element.confidence)
                
                sub_tables = self._split_table_block(image, (x0, y0, x1, y1))
                if sub_tables:
                    for split_idx, (sx0, sy0, sx1, sy1) in enumerate(sub_tables):
                        block = TableBlock(
                            id=f"table_model_{page_id}_{idx}_{split_idx}",
                            type=BlockType.TABLE,
                            bbox=(sx0, sy0, sx1, sy1),
                            page_id=page_id,
                            confidence=confidence,
                            metadata={
                                "detected_by": "table_model_split",
                                "model_backend": "paddle",
                                "model_config": "tablebank_ppyolov2",
                                "parent_bbox": [x0, y0, x1, y1]
                            }
                        )
                        structure = self._estimate_table_structure(image, (sx0, sy0, sx1, sy1))
                        if structure:
                            block.shape = structure
                        table_blocks.append(block)
                else:
                    block = TableBlock(
                        id=f"table_model_{page_id}_{idx}",
                        type=BlockType.TABLE,
                        bbox=(x0, y0, x1, y1),
                        page_id=page_id,
                        confidence=confidence,
                        metadata={
                            "detected_by": "table_model",
                            "model_backend": "paddle",
                            "model_config": "tablebank_ppyolov2"
                        }
                    )
                    structure = self._estimate_table_structure(image, (x0, y0, x1, y1))
                    if structure:
                        block.shape = structure
                    table_blocks.append(block)
        except Exception as err:
            print(f"⚠ Table model detection failed: {err}")
        
        return table_blocks
    
    def _split_table_block(
        self,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        min_gap: int = 18
    ) -> List[Tuple[float, float, float, float]]:
        """Split a large table block into multiple segments based on horizontal whitespace."""
        try:
            import cv2
        except ImportError:
            return []
        
        x0, y0, x1, y1 = map(int, bbox)
        height, width = image.shape[:2]
        
        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop.copy()
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5
        )
        
        row_scores = np.mean(binary > 0, axis=1)
        blank_rows = row_scores < 0.02
        
        segments: List[Tuple[int, int]] = []
        start = None
        for idx, is_blank in enumerate(blank_rows):
            if not is_blank:
                if start is None:
                    start = idx
            else:
                if start is not None and (idx - start) > min_gap:
                    segments.append((start, idx))
                start = None
        if start is not None and (len(blank_rows) - start) > min_gap:
            segments.append((start, len(blank_rows)))
        
        # If only one segment spans nearly the whole block, return empty to avoid duplicates
        if len(segments) <= 1:
            return []
        
        sub_blocks = []
        for seg_start, seg_end in segments:
            seg_height = seg_end - seg_start
            if seg_height < min_gap:
                continue
            sy0 = y0 + seg_start
            sy1 = y0 + seg_end
            sub_blocks.append((float(x0), float(sy0), float(x1), float(sy1)))
        
        return sub_blocks
    
    def _detect_table_candidates(
        self,
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect table candidates using morphological heuristics."""
        try:
            import cv2
        except ImportError:
            print("⚠ OpenCV not available; skipping table heuristics")
            return []
        
        if image is None or image.size == 0:
            return []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        height, width = gray.shape[:2]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, width // 30), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35, height // 30)))
        
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        
        combined = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        table_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, closing_kernel, iterations=2)
        table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
        
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = width * height * 0.004  # Require at least 0.4% of page
        candidates: List[Dict[str, Any]] = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < min_area or w < 50 or h < 50:
                continue
            
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.35 or aspect_ratio > 18:
                continue
            
            line_density = cv2.countNonZero(table_mask[y:y + h, x:x + w]) / float(area)
            if line_density < 0.008:
                continue
            
            candidates.append({
                "bbox": (float(x), float(y), float(x + w), float(y + h)),
                "score": min(0.95, 0.3 + line_density * 3.0),
                "density": line_density
            })
        
        return candidates
    
    def _apply_table_enhancement(
        self,
        image: np.ndarray,
        blocks: List[Block],
        page_id: int
    ) -> List[Block]:
        """Enhance block list with additional table detections."""
        existing_tables = [b for b in blocks if b.type == BlockType.TABLE]
        candidates = self._detect_table_candidates(image)
        
        for idx, candidate in enumerate(candidates):
            if any(self._calculate_iou(candidate["bbox"], tbl.bbox) > 0.45 for tbl in existing_tables):
                continue
            
            x0, y0, x1, y1 = candidate["bbox"]
            table_block = TableBlock(
                id=f"table_heuristic_{page_id}_{idx}",
                type=BlockType.TABLE,
                bbox=(x0, y0, x1, y1),
                page_id=page_id,
                confidence=candidate["score"],
                metadata={
                    "detected_by": "table_heuristic",
                    "table_density": candidate["density"]
                }
            )
            structure = self._estimate_table_structure(image, (x0, y0, x1, y1))
            if structure:
                table_block.shape = structure
            blocks.append(table_block)
            existing_tables.append(table_block)
        
        return blocks
    
    def _augment_text_with_heuristic(
        self,
        image: np.ndarray,
        blocks: List[Block],
        page_id: int
    ) -> List[Block]:
        """Add heuristic text/list blocks when ML model misses them."""
        text_like_types = {BlockType.TEXT, BlockType.TITLE, BlockType.LIST}
        existing_text = sum(1 for b in blocks if b.type in text_like_types)
        if existing_text >= max(3, len(blocks) // 4):
            return blocks
        
        heuristic_blocks = self.detect_layout_heuristic(image, page_id)
        page_height = image.shape[0]
        for hb in heuristic_blocks:
            if hb.type not in text_like_types:
                continue
            if any(self._calculate_iou(hb.bbox, existing.bbox) > 0.25 for existing in blocks):
                continue
            # Promote top-of-page text blocks to title
            if hb.type == BlockType.TEXT and hb.bbox[1] < page_height * 0.12:
                hb.type = BlockType.TITLE
                hb.level = 0
                hb.metadata["role_hint"] = "title_candidate"
            hb.metadata.update({"detected_by": "text_heuristic"})
            blocks.append(hb)
        
        return blocks
    
    def _analyze_block_content(
        self,
        image: np.ndarray,
        block: Block
    ) -> Optional[Dict[str, float]]:
        """Compute simple texture features to decide if a block is text/table/form."""
        try:
            import cv2
        except ImportError:
            return None
        
        x0, y0, x1, y1 = map(int, block.bbox)
        height, width = image.shape[:2]
        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop.copy()
        
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            8
        )
        
        area = float((x1 - x0) * (y1 - y0))
        if area <= 0:
            return None
        
        text_pixels = cv2.countNonZero(binary)
        text_density = text_pixels / area
        
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, binary.shape[1] // 40), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, binary.shape[0] // 40)))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
        horizontal_density = cv2.countNonZero(horizontal) / area
        vertical_density = cv2.countNonZero(vertical) / area
        line_mask = cv2.bitwise_or(horizontal, vertical)
        line_density = cv2.countNonZero(line_mask) / area
        
        edges = cv2.Canny(blur, 50, 150)
        stroke_density = cv2.countNonZero(edges) / area
        
        return {
            "text_density": float(text_density),
            "horizontal_density": float(horizontal_density),
            "vertical_density": float(vertical_density),
            "line_density": float(line_density),
            "stroke_density": float(stroke_density),
            "area": area
        }
    
    def _reclassify_blocks_by_texture(
        self,
        image: np.ndarray,
        blocks: List[Block],
        page_id: int
    ) -> List[Block]:
        """Use simple texture cues to relabel misclassified figure blocks."""
        page_area = float(image.shape[0] * image.shape[1])
        updated_blocks: List[Block] = []
        
        for block in blocks:
            # Skip if already correctly classified
            if block.type in {BlockType.TEXT, BlockType.TITLE, BlockType.LIST, BlockType.TABLE, BlockType.FORM}:
                updated_blocks.append(block)
                continue
            
            # For large figure blocks, analyze content to reclassify
            block_area = (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1])
            block_area_ratio = block_area / page_area if page_area > 0 else 0
            
            # If it's a large figure (>20% of page), it's likely misclassified
            if block.type == BlockType.FIGURE and block_area_ratio > 0.20:
                features = self._analyze_block_content(image, block)
                if not features:
                    updated_blocks.append(block)
                    continue
                
                block.metadata = dict(block.metadata)
                block.metadata.setdefault("texture_features", features)
                
                line_density = features["line_density"]
                text_density = features["text_density"]
                horiz = features["horizontal_density"]
                vert = features["vertical_density"]
                stroke_density = features["stroke_density"]
                
                new_type: Optional[BlockType] = None
                
                # If it has dense grid lines, it's likely a table or form
                if line_density > 0.03 and horiz > 0.01 and vert > 0.01:
                    # Check if it has table-like structure
                    if text_density < 0.4:  # Low text, high structure = table
                        new_type = BlockType.TABLE
                    else:  # High text + structure = form
                        new_type = BlockType.FORM
                # If it has high text density, it's likely text
                elif text_density > 0.06 and line_density < 0.02:
                    # Check height to determine title vs text
                    if (block.bbox[3] - block.bbox[1]) < image.shape[0] * 0.15:
                        new_type = BlockType.TITLE
                    else:
                        new_type = BlockType.TEXT
                # If it has form-like characteristics
                elif stroke_density > 0.03 and block_area_ratio > 0.10:
                    new_type = BlockType.FORM
                
                if new_type and new_type != block.type:
                    print(f"  🔄 Reclassified block {block.id} from {block.type.value} to {new_type.value} (area: {block_area_ratio:.1%}, line_density: {line_density:.3f}, text_density: {text_density:.3f})")
                    block.type = new_type
                    block.metadata["reclassified_by"] = "texture_analysis"
            
            updated_blocks.append(block)
        
        return updated_blocks
    
    def _estimate_table_structure(
        self,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[Tuple[int, int]]:
        """Estimate table row/column counts using line detection heuristics."""
        try:
            import cv2
        except ImportError:
            return None
        
        x0, y0, x1, y1 = map(int, bbox)
        height, width = image.shape[:2]
        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop.copy()
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, binary.shape[1] // 30), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, binary.shape[0] // 30)))
        
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
        
        def count_components(mask: np.ndarray, axis: str) -> int:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if axis == "row":
                valid = sum(1 for c in contours if cv2.boundingRect(c)[2] > mask.shape[1] * 0.5)
            else:
                valid = sum(1 for c in contours if cv2.boundingRect(c)[3] > mask.shape[0] * 0.5)
            return max(valid, 0)
        
        rows = count_components(horizontal, "row")
        cols = count_components(vertical, "col")
        
        rows = rows if rows and rows > 0 else None
        cols = cols if cols and cols > 0 else None
        
        if rows is None and cols is None:
            return None
        return (rows or 1, cols or 1)
    
    def _detect_all_lines(self, image: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Detect all horizontal and vertical lines in the image.
        Returns lists of y-coordinates (horizontal) and x-coordinates (vertical).
        """
        try:
            import cv2
        except ImportError:
            return [], []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect horizontal lines
        h_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=gray.shape[1] // 4,  # At least 25% of width
            maxLineGap=20
        )
        
        # Detect vertical lines
        v_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=gray.shape[0] // 4,  # At least 25% of height
            maxLineGap=20
        )
        
        h_y_coords = []
        v_x_coords = []
        
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly horizontal
                if abs(y2 - y1) < 5:
                    h_y_coords.append((y1 + y2) // 2)
        
        if v_lines is not None:
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly vertical
                if abs(x2 - x1) < 5:
                    v_x_coords.append((x1 + x2) // 2)
        
        # Remove duplicates and sort
        h_y_coords = sorted(set(h_y_coords))
        v_x_coords = sorted(set(v_x_coords))
        
        return h_y_coords, v_x_coords
    
    def _split_block_by_lines(
        self,
        image: np.ndarray,
        block: Block,
        h_lines: List[int],
        v_lines: List[int],
        page_id: int
    ) -> List[Block]:
        """
        Split a large block using detected horizontal and vertical lines.
        """
        x0, y0, x1, y1 = map(int, block.bbox)
        height, width = image.shape[:2]
        
        # Filter lines that are within block bounds
        valid_h_lines = [line for line in h_lines if y0 < line < y1]
        valid_v_lines = [line for line in v_lines if x0 < line < x1]
        
        # If we have enough lines, split the block
        if len(valid_h_lines) >= 2 or len(valid_v_lines) >= 2:
            sub_blocks = []
            
            # Split by horizontal lines (rows)
            if len(valid_h_lines) >= 2:
                split_points_y = [y0] + valid_h_lines + [y1]
                for idx in range(len(split_points_y) - 1):
                    sub_y0 = split_points_y[idx]
                    sub_y1 = split_points_y[idx + 1]
                    
                    if sub_y1 - sub_y0 < 30:  # Skip too small segments
                        continue
                    
                    # If we also have vertical lines, split into grid
                    if len(valid_v_lines) >= 2:
                        split_points_x = [x0] + valid_v_lines + [x1]
                        for jdx in range(len(split_points_x) - 1):
                            sub_x0 = split_points_x[jdx]
                            sub_x1 = split_points_x[jdx + 1]
                            
                            if sub_x1 - sub_x0 < 30:
                                continue
                            
                            sub_block = Block(
                                id=f"{block.id}_split_l_{idx}_{jdx}",
                                type=block.type,
                                bbox=(sub_x0, sub_y0, sub_x1, sub_y1),
                                page_id=page_id,
                                confidence=block.confidence * 0.9,
                                metadata={
                                    **block.metadata,
                                    "split_from": block.id,
                                    "split_method": "line_grid"
                                }
                            )
                            sub_blocks.append(sub_block)
                    else:
                        # Only horizontal split
                        sub_block = Block(
                            id=f"{block.id}_split_l_h_{idx}",
                            type=block.type,
                            bbox=(x0, sub_y0, x1, sub_y1),
                            page_id=page_id,
                            confidence=block.confidence * 0.9,
                            metadata={
                                **block.metadata,
                                "split_from": block.id,
                                "split_method": "horizontal_lines"
                            }
                        )
                        sub_blocks.append(sub_block)
            elif len(valid_v_lines) >= 2:
                # Only vertical split
                split_points_x = [x0] + valid_v_lines + [x1]
                for idx in range(len(split_points_x) - 1):
                    sub_x0 = split_points_x[idx]
                    sub_x1 = split_points_x[idx + 1]
                    
                    if sub_x1 - sub_x0 < 30:
                        continue
                    
                    sub_block = Block(
                        id=f"{block.id}_split_l_v_{idx}",
                        type=block.type,
                        bbox=(sub_x0, y0, sub_x1, y1),
                        page_id=page_id,
                        confidence=block.confidence * 0.9,
                        metadata={
                            **block.metadata,
                            "split_from": block.id,
                            "split_method": "vertical_lines"
                        }
                    )
                    sub_blocks.append(sub_block)
            
            if len(sub_blocks) > 1:
                return sub_blocks
        
        return [block]  # Can't split meaningfully
    
    def _split_large_blocks(
        self,
        image: np.ndarray,
        blocks: List[Block],
        page_id: int
    ) -> List[Block]:
        """
        Split large blocks (especially forms) into smaller granular components.
        
        Args:
            image: Page image
            blocks: List of blocks
            page_id: Page ID
            
        Returns:
            List of blocks with large ones split
        """
        import cv2
        
        height, width = image.shape[:2]
        page_area = height * width
        max_area = page_area * self.max_block_area_ratio
        
        split_blocks = []
        split_count = 0
        
        for block in blocks:
            block_area = (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1])
            block_area_ratio = block_area / page_area if page_area > 0 else 0
            
            # More aggressive splitting: lower threshold and include more block types
            # Split if block is >15% of page (instead of 30%) OR if it's a form/text >10%
            textual_like = {BlockType.FORM, BlockType.TEXT, BlockType.UNKNOWN, BlockType.FIGURE}
            should_split = (
                block.type != BlockType.TABLE and (
                    (block_area_ratio > 0.15 and block.type in textual_like) or
                    (block_area_ratio > 0.10 and block.type == BlockType.FORM) or
                    (block_area_ratio > 0.20 and block.type == BlockType.FIGURE)  # Split large figures too
                )
            )
            
            if should_split:
                # First try line-based splitting (most accurate for forms with borders/lines)
                h_lines, v_lines = self._detect_all_lines(image)
                sub_blocks = self._split_block_by_lines(image, block, h_lines, v_lines, page_id)
                if len(sub_blocks) > 1:
                    split_blocks.extend(sub_blocks)
                    split_count += 1
                    print(f"  ✓ Split block {block.id} ({block.type.value}) into {len(sub_blocks)} sub-blocks using lines (area: {block_area_ratio:.1%})")
                    continue
                
                # Fallback to contour detection
                sub_blocks = self._split_block_by_contours(image, block, page_id)
                if len(sub_blocks) > 1:
                    split_blocks.extend(sub_blocks)
                    split_count += 1
                    print(f"  ✓ Split block {block.id} ({block.type.value}) into {len(sub_blocks)} sub-blocks using contours (area: {block_area_ratio:.1%})")
                    continue
            
            split_blocks.append(block)
        
        if split_count > 0:
            print(f"  📊 Split {split_count} large blocks into {len(split_blocks) - len(blocks) + split_count} sub-blocks")
        
        return split_blocks
    
    def _split_block_by_contours(
        self,
        image: np.ndarray,
        block: Block,
        page_id: int
    ) -> List[Block]:
        """
        Split a large block into smaller components using contour detection.
        
        Args:
            image: Page image
            block: Block to split
            page_id: Page ID
            
        Returns:
            List of smaller blocks
        """
        try:
            import cv2
        except ImportError:
            return [block]
        
        x0, y0, x1, y1 = map(int, block.bbox)
        height, width = image.shape[:2]
        
        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return [block]
        
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop.copy()
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size - more lenient threshold
        min_area = (x1 - x0) * (y1 - y0) * 0.02  # At least 2% of block area (more lenient)
        valid_contours = [
            c for c in contours
            if cv2.contourArea(c) > min_area
        ]
        
        # Also try horizontal line-based splitting if contours don't work
        if len(valid_contours) < 2:
            # Try splitting by horizontal lines (for forms with clear rows)
            h_lines = self._detect_horizontal_lines(gray)
            if len(h_lines) >= 2:
                return self._split_by_horizontal_lines(image, block, h_lines, page_id)
            return [block]  # Can't split meaningfully
        
        # Create blocks from contours
        sub_blocks = []
        for idx, contour in enumerate(valid_contours):
            bx, by, bw, bh = cv2.boundingRect(contour)
            
            # Convert back to page coordinates
            abs_x0 = x0 + bx
            abs_y0 = y0 + by
            abs_x1 = x0 + bx + bw
            abs_y1 = y0 + by + bh
            
            # Create new block
            sub_block = Block(
                id=f"{block.id}_split_{idx}",
                type=block.type,
                bbox=(abs_x0, abs_y0, abs_x1, abs_y1),
                page_id=page_id,
                confidence=block.confidence * 0.9,  # Slightly lower confidence for split blocks
                metadata={
                    **block.metadata,
                    "split_from": block.id,
                    "split_method": "contour"
                }
            )
            sub_blocks.append(sub_block)
        
        return sub_blocks if len(sub_blocks) > 1 else [block]
    
    def _detect_horizontal_lines(self, gray_image: np.ndarray) -> List[int]:
        """Detect horizontal lines in grayscale image for splitting."""
        try:
            import cv2
        except ImportError:
            return []
        
        # Detect horizontal lines using HoughLinesP
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=gray_image.shape[1] // 3,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        # Extract y-coordinates of horizontal lines
        y_coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is roughly horizontal
            if abs(y2 - y1) < 5:  # Nearly horizontal
                y_coords.append((y1 + y2) // 2)
        
        # Remove duplicates and sort
        y_coords = sorted(set(y_coords))
        return y_coords
    
    def _split_by_horizontal_lines(
        self,
        image: np.ndarray,
        block: Block,
        h_lines: List[int],
        page_id: int
    ) -> List[Block]:
        """Split block by horizontal lines."""
        x0, y0, x1, y1 = map(int, block.bbox)
        block_y0 = y0
        
        # Filter lines that are within block bounds
        valid_lines = [line for line in h_lines if block_y0 < line < y1]
        
        if len(valid_lines) < 1:
            return [block]
        
        # Create sub-blocks between lines
        sub_blocks = []
        split_points = [block_y0] + valid_lines + [y1]
        
        for idx in range(len(split_points) - 1):
            sub_y0 = split_points[idx]
            sub_y1 = split_points[idx + 1]
            
            # Skip if too small
            if sub_y1 - sub_y0 < 20:
                continue
            
            sub_block = Block(
                id=f"{block.id}_split_h_{idx}",
                type=block.type,
                bbox=(x0, sub_y0, x1, sub_y1),
                page_id=page_id,
                confidence=block.confidence * 0.9,
                metadata={
                    **block.metadata,
                    "split_from": block.id,
                    "split_method": "horizontal_lines"
                }
            )
            sub_blocks.append(sub_block)
        
        return sub_blocks if len(sub_blocks) > 1 else [block]
    
    @staticmethod
    def _merge_group(block: Block) -> str:
        """Group block types when evaluating merges."""
        if block.type in {BlockType.TEXT, BlockType.TITLE, BlockType.LIST}:
            return "textual"
        if block.type == BlockType.TABLE:
            return "table"
        if block.type == BlockType.FIGURE:
            return "figure"
        if block.type == BlockType.FORM:
            return "form"
        return "other"
    
    def merge_overlapping_boxes(self, blocks: List[Block], iou_threshold: float = 0.5) -> List[Block]:
        """
        Merge overlapping boxes.
        
        Args:
            blocks: List of blocks
            iou_threshold: IoU threshold for merging
            
        Returns:
            List of merged blocks
        """
        if len(blocks) <= 1:
            return blocks
        
        merged = []
        used = set()
        
        for i, block1 in enumerate(blocks):
            if i in used:
                continue
            
            current_blocks = [block1]
            current_bbox = list(block1.bbox)
            
            group1 = self._merge_group(block1)
            # Find overlapping blocks
            for j, block2 in enumerate(blocks[i+1:], start=i+1):
                if j in used:
                    continue
                group2 = self._merge_group(block2)
                if group1 != group2:
                    continue
                
                iou = self._calculate_iou(block1.bbox, block2.bbox)
                if iou > iou_threshold:
                    current_blocks.append(block2)
                    used.add(j)
                    
                    # Merge bboxes
                    x0 = min(current_bbox[0], block2.bbox[0])
                    y0 = min(current_bbox[1], block2.bbox[1])
                    x1 = max(current_bbox[2], block2.bbox[2])
                    y1 = max(current_bbox[3], block2.bbox[3])
                    current_bbox = [x0, y0, x1, y1]
            
            # Create merged block
            if len(current_blocks) > 1:
                # Use the block with highest confidence
                main_block = max(current_blocks, key=lambda b: b.confidence)
                merged_block = Block(
                    id=main_block.id,
                    type=main_block.type,
                    bbox=tuple(current_bbox),
                    page_id=main_block.page_id,
                    confidence=main_block.confidence
                )
                merged.append(merged_block)
            else:
                merged.append(block1)
        
        return merged
    
    def resolve_reading_order(
        self,
        blocks: List[Block],
        multi_column: bool = True
    ) -> List[Block]:
        """
        Resolve reading order: top→bottom, left→right.
        
        Args:
            blocks: List of blocks
            multi_column: Whether to handle multi-column layout
            
        Returns:
            Blocks sorted by reading order
        """
        if not blocks:
            return blocks
        
        def get_sort_key(block: Block) -> Tuple[float, float, float]:
            """Get sort key for reading order."""
            y_center = (block.bbox[1] + block.bbox[3]) / 2
            x_center = (block.bbox[0] + block.bbox[2]) / 2
            x_min = block.bbox[0]
            
            if multi_column:
                # Cluster by x-position for multi-column
                # Simple approach: group blocks by x-center position
                x_cluster_id = int(x_center / 200)  # 200px cluster width
                return (y_center, x_cluster_id, x_min)
            else:
                return (y_center, x_min)
        
        return sorted(blocks, key=get_sort_key)
    
    def segment_page(
        self,
        image: np.ndarray,
        page_id: int,
        merge_boxes: bool = True,
        resolve_order: bool = True
    ) -> List[Block]:
        """
        Segment a single page.
        
        Args:
            image: Page image
            page_id: Page ID
            merge_boxes: Whether to merge overlapping boxes
            resolve_order: Whether to resolve reading order
            
        Returns:
            List of blocks
        """
        # Detect layout
        blocks = self.detect_layout(image, page_id)
        
        # Set page IDs
        for block in blocks:
            block.page_id = page_id
        
        # Add auxiliary table detections (model based)
        if self.table_model:
            table_blocks = self._detect_tables_with_model(image, page_id)
            if table_blocks:
                blocks.extend(self._filter_new_blocks(blocks, table_blocks))
        
        # Heuristic table enhancement (line density)
        blocks = self._apply_table_enhancement(image, blocks, page_id)
        
        # Heuristic text/list augmentation if needed
        blocks = self._augment_text_with_heuristic(image, blocks, page_id)
        
        # Reclassify ambiguous figure blocks using texture cues
        blocks = self._reclassify_blocks_by_texture(image, blocks, page_id)
        
        # Split large blocks into smaller components for better granularity
        if self.split_large_blocks:
            blocks = self._split_large_blocks(image, blocks, page_id)
        
        # Merge overlapping boxes (less aggressive for granularity)
        if merge_boxes:
            blocks = self.merge_overlapping_boxes(blocks, iou_threshold=0.6)  # Higher threshold = less merging
        
        # Detect and label form regions (after reclassification & splitting)
        blocks = self._apply_form_enhancement(image, blocks, page_id)
        
        # Resolve reading order
        if resolve_order:
            blocks = self.resolve_reading_order(blocks)
        
        # Update block IDs with page prefix (maintain order)
        for i, block in enumerate(blocks):
            block.id = f"{page_id}-{i}"
            block.page_id = page_id  # Ensure page_id is set correctly
            if not block.citations:
                block.add_citation(page_id, block.bbox)
        
        return blocks

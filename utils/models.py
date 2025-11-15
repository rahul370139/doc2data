"""
Data models for document processing pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class BlockType(str, Enum):
    """Types of document blocks."""
    TEXT = "text"
    TITLE = "title"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    FORM = "form"
    UNKNOWN = "unknown"


class BlockRole(str, Enum):
    """Semantic roles for text blocks."""
    TITLE = "title"
    H1 = "h1"
    H2 = "h2"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUM = "page_num"
    LIST_ITEM = "list_item"
    KV_LABEL = "kv_label"
    KV_VALUE = "kv_value"
    CAPTION = "caption"
    PARAGRAPH = "paragraph"
    UNKNOWN = "unknown"


@dataclass
class Citation:
    """Citation to source location in document."""
    page: int
    bbox: Tuple[float, float, float, float]  # [x0, y0, x1, y1]


@dataclass
class WordBox:
    """Word-level bounding box with text and confidence."""
    text: str
    bbox: Tuple[float, float, float, float]  # [x0, y0, x1, y1]
    confidence: float = 1.0


@dataclass
class PageImage:
    """Page image with metadata and analysis layers."""
    image: Any  # numpy array or PIL Image
    page_id: int
    width: int
    height: int
    dpi: int = 300
    digital_text: bool = False  # True if PDF has digital text layer
    digital_words: List[WordBox] = field(default_factory=list)
    preprocess_metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_image: Optional[Any] = None  # High-contrast grayscale image
    binary_image: Optional[Any] = None  # Adaptive-thresholded image
    line_mask: Optional[Any] = None  # Highlighted horizontal/vertical lines
    box_mask: Optional[Any] = None  # Regions with dense boxes/grids
    orientation: float = 0.0  # Estimated orientation in degrees (0/90)
    orientation_confidence: float = 0.0
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if not self.preprocess_metadata:
            self.preprocess_metadata = {
                "deskewed": False,
                "denoised": False,
                "digital_text_extracted": False
            }

    @property
    def shape(self):
        """Compat helper so code that expects numpy-like `.shape` doesn't crash.

        Returns a tuple (H, W[, C]) taking from the backing image when possible,
        otherwise synthesized from height/width fields. The optional channel
        dimension is included as 3 to mimic RGB when unknown.
        """
        try:
            img = getattr(self, "image", None)
            if img is not None and hasattr(img, "shape"):
                return img.shape
        except Exception:
            pass
        # Fallback to height/width
        h = int(getattr(self, "height", 0) or 0)
        w = int(getattr(self, "width", 0) or 0)
        if h > 0 and w > 0:
            return (h, w, 3)
        return (0, 0, 3)


@dataclass
class Block:
    """Base block with common attributes."""
    id: str
    type: BlockType
    bbox: Tuple[float, float, float, float]  # [x0, y0, x1, y1]
    role: Optional[BlockRole] = None
    level: Optional[int] = None  # For headers (1=h1, 2=h2, etc.)
    text: Optional[str] = None
    page_id: int = 0
    word_boxes: List[WordBox] = field(default_factory=list)
    children: List[str] = field(default_factory=list)  # IDs of child blocks
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_citation(self, page: int, bbox: Tuple[float, float, float, float]):
        """Add a citation to this block."""
        self.citations.append(Citation(page=page, bbox=bbox))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "role": self.role.value if self.role else None,
            "level": self.level,
            "text": self.text,
            "bbox": list(self.bbox),
            "page": self.page_id,
            "word_boxes": [
                {
                    "text": wb.text,
                    "bbox": list(wb.bbox),
                    "confidence": wb.confidence
                }
                for wb in self.word_boxes
            ],
            "children": self.children,
            "citations": [
                {"page": c.page, "bbox": list(c.bbox)}
                for c in self.citations
            ],
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class TableBlock(Block):
    """Table-specific block with structured data."""
    shape: Optional[Tuple[int, int]] = None  # (rows, cols)
    headers: List[List[str]] = field(default_factory=list)
    body: List[List[str]] = field(default_factory=list)
    units: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize table-specific attributes."""
        if self.type != BlockType.TABLE:
            self.type = BlockType.TABLE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert table block to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "shape": list(self.shape) if self.shape else None,
            "headers": self.headers,
            "body": self.body,
            "units": self.units
        })
        return base_dict


@dataclass
class FigureBlock(Block):
    """Figure-specific block with image data."""
    figure_type: Optional[str] = None  # bar, line, pie, scatter, non_chart_image, diagram, other
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    data_or_link: Optional[str] = None  # Path to image or structured data
    series: List[Dict[str, Any]] = field(default_factory=list)  # For charts
    
    def __post_init__(self):
        """Initialize figure-specific attributes."""
        if self.type != BlockType.FIGURE:
            self.type = BlockType.FIGURE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert figure block to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "figure_type": self.figure_type,
            "caption": self.caption,
            "alt_text": self.alt_text,
            "data_or_link": self.data_or_link,
            "series": self.series
        })
        return base_dict


@dataclass
class FormFieldCandidate:
    """Candidate form field detected during geometry analysis."""
    id: str
    page_id: int
    bbox: Tuple[float, float, float, float]
    field_type: str = "text_field"  # text_field, checkbox_group, grid_cell, etc.
    rotation: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckboxCandidate:
    """Checkbox candidate with optional state metadata."""
    id: str
    page_id: int
    bbox: Tuple[float, float, float, float]
    state: Optional[str] = None  # checked / unchecked / ambiguous
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Document container with pages and blocks."""
    doc_id: str
    pages: List[PageImage] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_page(self, page_id: int) -> Optional[PageImage]:
        """Get page by ID."""
        for page in self.pages:
            if page.page_id == page_id:
                return page
        return None
    
    def get_blocks_by_page(self, page_id: int) -> List[Block]:
        """Get all blocks for a specific page."""
        return [block for block in self.blocks if block.page_id == page_id]
    
    def get_block_by_id(self, block_id: str) -> Optional[Block]:
        """Get block by ID."""
        for block in self.blocks:
            if block.id == block_id:
                return block
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for JSON serialization."""
        # Group blocks by page
        pages_dict = {}
        for page in self.pages:
            pages_dict[page.page_id] = {
                "page": page.page_id,
                "blocks": []
            }
        
        for block in self.blocks:
            if block.page_id in pages_dict:
                pages_dict[block.page_id]["blocks"].append(block.to_dict())
        
        return {
            "doc_id": self.doc_id,
            "metadata": self.metadata,
            "pages": [pages_dict[pid] for pid in sorted(pages_dict.keys())]
        }

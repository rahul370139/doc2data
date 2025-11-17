"""
Streamlit app for document processing pipeline visualization.
Professional UI with collapsible sidebar and enhanced visualization.
"""
import sys
from pathlib import Path
import json
import inspect

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from typing import Tuple, Optional, List, Dict, Any
import cv2
import time
from uuid import uuid4

from utils.models import Document, Block, BlockType, BlockRole, TableBlock, FigureBlock
from utils.config import Config
from src.pipelines.ingest import ingest_document
from src.pipelines.segment import LayoutSegmenter
from src.pipelines.ocr import OCRPipeline
from src.pipelines.slm_label import SLMLabeler
from src.pipelines.assemble import DocumentAssembler
from utils.visualization import draw_bounding_boxes, BLOCK_TYPE_COLORS, get_block_color


# Page config
st.set_page_config(
    page_title="Document-to-Data Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .block-info {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .result-card {
        border: 1px solid #E0E3EB;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        background: #fff;
        transition: all 0.15s ease;
    }
    .result-card:hover {
        border-color: #5F6FFF;
        box-shadow: 0 2px 6px rgba(95,111,255,0.15);
    }
    .result-card.active {
        border-color: #3956D1;
        box-shadow: 0 4px 10px rgba(57,86,209,0.25);
        background: #EEF2FF;
    }
    .result-card .result-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #68718A;
        margin-bottom: 0.25rem;
    }
    .result-card .result-text {
        font-size: 0.95rem;
        color: #131620;
        line-height: 1.4;
        white-space: pre-wrap;
    }
    .block-pill {
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        border: 1px solid #E0E3EB;
        margin: 0.2rem 0;
        cursor: pointer;
    }
    .block-pill.active {
        border-color: #1E88E5;
        background: #E3F2FD;
    }
</style>
""", unsafe_allow_html=True)


def _bgr_to_hex(color: Tuple[int, int, int]) -> str:
    """Convert BGR color tuple to HEX string."""
    b, g, r = color
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _format_duration(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return "—"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _render_block_text(block: Block) -> str:
    text = (block.text or "").strip()
    if text:
        return text
    if isinstance(block, TableBlock):
        if block.headers or block.body:
            lines = []
            if block.headers:
                for header_row in block.headers:
                    lines.append("| " + " | ".join(header_row) + " |")
                    lines.append("| " + " | ".join(["---"] * len(header_row)) + " |")
            if block.body:
                for row in block.body:
                    lines.append("| " + " | ".join(row) + " |")
            return "\n".join(lines) if lines else "[Table region]"
        if block.shape:
            rows, cols = block.shape
            return f"[Table region ~ {rows}x{cols}]"
        return "[Table region]"
    if isinstance(block, FigureBlock):
        return block.caption or "[Figure region]"
    return "[Empty]"


def _set_highlight(block_id: str):
    st.session_state.highlight_id = block_id


def _build_result_summary(
    blocks: List[Block],
    pages: List[Any],
    duration: Optional[float] = None
) -> Dict[str, Any]:
    if not blocks or not pages:
        return {}
    
    pages_map: Dict[int, Dict[str, Any]] = {}
    for page in pages:
        pages_map[page.page_id] = {
            "page_id": page.page_id,
            "page_number": page.page_id + 1,
            "blocks": [],
            "char_count": 0
        }
    
    sorted_blocks = sorted(blocks, key=lambda b: (b.page_id, b.bbox[1], b.bbox[0]))

    # Build page lookup to normalize coordinates
    page_dims: Dict[int, Tuple[int, int]] = {}
    for p in pages:
        try:
            if hasattr(p, "image") and getattr(p, "image", None) is not None:
                image_obj = getattr(p, "image")
                if hasattr(image_obj, "shape") and len(image_obj.shape) >= 2:
                    h, w = int(image_obj.shape[0]), int(image_obj.shape[1])
                elif hasattr(image_obj, "size"):
                    w, h = image_obj.size  # type: ignore[attr-defined]
                else:
                    h = int(getattr(p, "height", 1) or 1)
                    w = int(getattr(p, "width", 1) or 1)
            elif hasattr(p, "width") and hasattr(p, "height"):
                w, h = int(getattr(p, "width", 1) or 1), int(getattr(p, "height", 1) or 1)
            else:
                h = int(getattr(p, "height", 1) or 1)
                w = int(getattr(p, "width", 1) or 1)
        except Exception:
            w, h = 1, 1
        page_dims[getattr(p, "page_id", 0)] = (max(w, 1), max(h, 1))

    def norm_bbox(page_id: int, bbox: Tuple[float, float, float, float]) -> Dict[str, float]:
        w, h = page_dims.get(page_id, (1, 1))
        x0, y0, x1, y1 = bbox
        left = max(0.0, float(x0) / max(w, 1))
        top = max(0.0, float(y0) / max(h, 1))
        width = max(0.0, float(x1 - x0) / max(w, 1))
        height = max(0.0, float(y1 - y0) / max(h, 1))
        return {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "page": page_id + 1,
            "original_page": page_id + 1,
        }

    def map_type_for_summary(block: Block) -> str:
        if block.type == BlockType.TITLE:
            return "Section Header"
        if block.type == BlockType.FORM:
            return "Key Value"
        if block.type == BlockType.TABLE:
            return "Table"
        if block.type == BlockType.FIGURE:
            return "Figure"
        return "text"
    for block in sorted_blocks:
        page_entry = pages_map.setdefault(block.page_id, {
            "page_id": block.page_id,
            "page_number": block.page_id + 1,
            "blocks": [],
            "char_count": 0
        })
        text = _render_block_text(block)
        entry = {
            "id": block.id,
            "type": map_type_for_summary(block),
            "role": block.role.value if block.role else None,
            "type_label": block.role.value if block.role else block.type.value,
            "text": text,
            "text_markdown": text.replace("\n", "  \n"),
            "char_count": len(text),
            "confidence": block.confidence,
            "bbox": block.bbox,
            "bbox_norm": norm_bbox(block.page_id, block.bbox),
        }
        page_entry["blocks"].append(entry)
        page_entry["char_count"] += len(text)
    
    pages_list = []
    chunks = []
    total_chars = 0
    for page_id in sorted(pages_map):
        entry = pages_map[page_id]
        entry["block_count"] = len(entry["blocks"])
        total_chars += entry["char_count"]
        pages_list.append(entry)
        chunk_text = "\n\n".join([b["text"] for b in entry["blocks"] if b["text"]])
        chunks.append({
            "page_number": entry["page_number"],
            "content": chunk_text,
            "embed": chunk_text,
            "enriched": None,
            "enrichment_success": False,
            "char_count": len(chunk_text),
            "blocks": [
                {
                    "type": b["type"],
                    "bbox": b["bbox_norm"],
                    "content": b["text"],
                    "image_url": None,
                    "confidence": "high" if (b["confidence"] or 0) >= 0.8 else "low",
                    "granular_confidence": {
                        "extract_confidence": None,
                        "parse_confidence": float(b["confidence"] or 0.0),
                    },
                }
                for b in entry["blocks"]
            ]
        })
    
    summary = {
        "job_id": str(uuid4()),
        "duration": duration or 0.0,
        "usage": {
            "num_pages": len(pages),
            "credits": round(len(pages) * 0.75, 2),
            "total_characters": total_chars
        },
        "result": {
            "type": "full",
            "pages": pages_list,
            "chunks": chunks
        }
    }
    return summary


def _render_results_panel(summary: Dict[str, Any]):
    if not summary:
        return
    st.header("Results")
    usage = summary.get("usage", {})
    result_data = summary.get("result", {})
    pages = result_data.get("pages", [])
    chunks = result_data.get("chunks", [])
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Pages parsed", usage.get("num_pages", 0))
    with col_b:
        st.metric("Processing time", _format_duration(summary.get("duration")))
    with col_c:
        st.metric("Total chunks", len(chunks))
    
    st.download_button(
        "⬇️ Download Result JSON",
        json.dumps(summary, indent=2),
        file_name="doc2data_result.json",
        mime="application/json",
        use_container_width=True
    )
    
    highlight_id = st.session_state.get("highlight_id")
    all_choices: List[Tuple[str, str]] = []
    for page in pages:
        for block_entry in page.get("blocks", []):
            block_label = block_entry.get("type_label", block_entry.get("type", "text"))
            preview = (block_entry.get("text", "") or "").replace("\n", " ").strip()
            preview = preview[:80] + ("…" if len(preview) > 80 else "")
            choice_label = f"Page {page['page_number']} • {block_label} • {preview or '[Empty]'}"
            all_choices.append((choice_label, block_entry["id"]))
    
    if all_choices:
        labels = [label for label, _ in all_choices]
        id_map = {label: block_id for label, block_id in all_choices}
        default_index = 0
        if highlight_id:
            for idx, (_, block_id) in enumerate(all_choices):
                if block_id == highlight_id:
                    default_index = idx
                    break
        selected_label = st.selectbox(
            "Jump to block",
            options=labels,
            index=default_index,
            label_visibility="collapsed"
        )
        selected_id = id_map[selected_label]
        if selected_id != highlight_id:
            st.session_state.highlight_id = selected_id
            highlight_id = selected_id
    
    for page in pages:
        st.subheader(f"Page {page['page_number']} • {page.get('block_count', 0)} blocks • {page.get('char_count', 0)} characters")
        for block_entry in page.get("blocks", []):
            is_active = highlight_id == block_entry["id"]
            card_class = "result-card active" if is_active else "result-card"
            block_label = block_entry.get("type_label", block_entry.get("type", "text"))
            text_md = block_entry.get("text_markdown", "")
            text_html = text_md.replace("\n", "<br/>") if text_md else "<span style='color:#9AA0B5;'>[Empty]</span>"
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div class="result-label">{block_label} • Page {page['page_number']}</div>
                    <div class="result-text">{text_html}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Cards highlight automatically based on selectbox / page radio; no per-card buttons


def draw_annotated_image(
    image: np.ndarray,
    blocks: list[Block],
    highlight_id: Optional[str] = None,
    show_labels: bool = True,
    label_size: float = 0.6
) -> np.ndarray:
    """
    Draw bounding boxes with labels on image.
    
    Args:
        image: Image to annotate
        blocks: List of blocks
        highlight_id: ID to highlight
        show_labels: Show text labels
        label_size: Font size multiplier
        
    Returns:
        Annotated image
    """
    result = image.copy()
    
    for block in blocks:
        x0, y0, x1, y1 = [int(coord) for coord in block.bbox]
        
        # Clip to image bounds
        x0 = max(0, min(x0, image.shape[1] - 1))
        y0 = max(0, min(y0, image.shape[0] - 1))
        x1 = max(x0 + 1, min(x1, image.shape[1]))
        y1 = max(y0 + 1, min(y1, image.shape[0]))
        
        if x1 <= x0 or y1 <= y0:
            continue
        
        # Get color
        is_highlight = (block.id == highlight_id) if highlight_id else False
        color = (255, 0, 0) if is_highlight else get_block_color(block)
        color_bgr = (color[2], color[1], color[0])  # RGB to BGR for OpenCV
        
        # Draw rectangle with thicker line for highlight
        thickness = 4 if is_highlight else 2
        cv2.rectangle(result, (x0, y0), (x1, y1), color_bgr, thickness)
        
        # Draw label near box
        if show_labels:
            label = f"{block.type.value.upper()}"
            if block.role:
                label += f":{block.role.value}"
            
            # Add confidence if low
            if block.confidence < 0.9:
                label += f" ({block.confidence:.2f})"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = label_size
            thickness_text = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness_text
            )
            
            # Position label above box, or inside if no space
            label_y = max(y0 - 5, text_height + 5)
            if label_y < text_height:
                label_y = y0 + text_height + 10
            
            # Draw background for text
            padding = 4
            cv2.rectangle(
                result,
                (x0, label_y - text_height - padding),
                (x0 + text_width + padding * 2, label_y + padding),
                color_bgr,
                -1
            )
            
            # Draw text
            cv2.putText(
                result,
                label,
                (x0 + padding, label_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness_text,
                cv2.LINE_AA
            )
    
    return result


def main():
    """Main Streamlit app."""
    # Header
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.title("📄 Document-to-Data Pipeline")
        st.caption("Convert PDFs/images into structured JSON + Markdown with bounding-box citations")
    with col_header2:
        sidebar_collapsed = st.checkbox("🔀 Hide Sidebar", value=False, help="Maximize document view")
    
    # Initialize session state defaults
    if "run_ingest" not in st.session_state:
        st.session_state.run_ingest = True
    if "run_segment" not in st.session_state:
        st.session_state.run_segment = True
    if "run_ocr_assemble" not in st.session_state:
        st.session_state.run_ocr_assemble = False
    if "run_label" not in st.session_state:
        st.session_state.run_label = False
    # Cache for OCR results per page to avoid re-processing
    if "ocr_cache" not in st.session_state:
        st.session_state.ocr_cache = {}  # {page_id: {block_id: block_with_text}}
    if "show_labels" not in st.session_state:
        st.session_state.show_labels = True
    if "label_size" not in st.session_state:
        st.session_state.label_size = 0.6
    if "enable_slm" not in st.session_state:
        st.session_state.enable_slm = Config.ENABLE_SLM
    if "enable_vlm" not in st.session_state:
        st.session_state.enable_vlm = Config.ENABLE_VLM
    if "result_summary" not in st.session_state:
        st.session_state.result_summary = None
    
    # Sidebar (collapsible)
    if not sidebar_collapsed:
        with st.sidebar:
            st.header("📋 Controls")
            
            # Document selection
            st.subheader("📁 Document")
            sample_dir_input = st.text_input(
                "Sample directory",
                value=st.session_state.get("sample_dir", "data/sample_docs"),
                help="Path to sample documents"
            )
            st.session_state.sample_dir = sample_dir_input
            sample_dir = Path(sample_dir_input).expanduser()
            
            # Sample metadata with descriptions
            sample_metadata = {
                "form-cms1500.pdf": "📋 CMS-1500 Medical Claim Form (standard)",
                "ub04_clean.pdf": "🏥 UB-04 Hospital Billing Form (cleaned)",
                "ucf_form_page1.pdf": "📝 UCF Form (page 1 cropped)",
                "healthcare.pdf": "🏥 Healthcare Document (original)",
                "custom_flattened-svSzt1Bc.pdf": "📄 Custom Flattened Document"
            }
            
            # Files to exclude from sample list (originals that have been preprocessed)
            exclude_files = {"ub04_sample.pdf", "ucfforminstruct.pdf"}
            
            sample_files = []
            if sample_dir.exists():
                sample_files = sorted([
                    p for p in sample_dir.glob("*")
                    if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}
                    and p.name not in exclude_files
                ], key=lambda x: (x.name not in sample_metadata, x.name))  # Prioritize known samples
            else:
                st.warning(f"Directory not found: {sample_dir}")
            
            # Create display options with descriptions
            sample_display = []
            for p in sample_files:
                if p.name in sample_metadata:
                    sample_display.append(f"{sample_metadata[p.name]}")
                else:
                    sample_display.append(f"📄 {p.name}")
            
            sample_options = ["<Upload new file>"] + sample_display
            sample_choice_display = st.selectbox("Choose sample", sample_options, key="sample_choice")
            
            uploaded_file = st.file_uploader(
                "Or upload PDF/Image",
                type=["pdf", "png", "jpg", "jpeg"],
                help="Upload a new document"
            )
            
            document_path: Optional[Path] = None
            document_label: Optional[str] = None
            
            # Map display choice back to filename
            if sample_choice_display != "<Upload new file>":
                # Find the actual filename from the display string
                selected_index = sample_options.index(sample_choice_display) - 1  # -1 to account for "<Upload new>"
                if 0 <= selected_index < len(sample_files):
                    actual_file = sample_files[selected_index]
                    document_path = actual_file.resolve()
                    document_label = actual_file.name
            elif uploaded_file is not None:
                temp_path = Path("cache") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                document_path = temp_path.resolve()
                document_label = uploaded_file.name
            
            if document_path is None:
                st.info("👆 Select a sample document or upload your own to begin")
                
                # Show sample showcase
                st.markdown("---")
                st.subheader("📚 Available Sample Documents")
                st.markdown("Select a sample from the dropdown above to get started:")
                
                cols = st.columns(3)
                sample_info = [
                    {
                        "name": "form-cms1500.pdf",
                        "display": "CMS-1500 Medical Claim",
                        "desc": "Standard medical claim form used by healthcare providers",
                        "icon": "📋"
                    },
                    {
                        "name": "ub04_clean.pdf",
                        "display": "UB-04 Hospital Billing",
                        "desc": "Hospital billing form with complex table structures",
                        "icon": "🏥"
                    },
                    {
                        "name": "ucf_form_page1.pdf",
                        "display": "UCF Form",
                        "desc": "University form with mixed layout elements",
                        "icon": "📝"
                    }
                ]
                
                for idx, info in enumerate(sample_info):
                    with cols[idx]:
                        st.markdown(f"### {info['icon']} {info['display']}")
                        st.caption(info['desc'])
                        sample_path = sample_dir / info['name']
                        if sample_path.exists():
                            st.success(f"✓ Available")
                        else:
                            st.warning(f"⚠ Not found")
                
                st.markdown("---")
                st.markdown("""
                **Pipeline Features:**
                - 🔍 **Layout Detection**: Detects text, tables, forms, figures
                - 📝 **OCR**: Extracts text from images using PaddleOCR
                - 🎯 **Form Detection**: Identifies checkboxes and form fields
                - 📊 **Table Extraction**: Extracts structured table data
                - 📑 **JSON/Markdown Export**: Structured output with bounding boxes
                """)
                
                st.stop()
            
            # Store in session state
            st.session_state.document_path = str(document_path)
            st.session_state.document_label = document_label
            st.success(f"✓ {document_label}")
            
            # Pipeline stages
            st.subheader("⚙️ Pipeline")
            st.session_state.run_ingest = st.checkbox("1️⃣ Ingest", value=st.session_state.run_ingest, help="Load and preprocess document")
            st.session_state.run_segment = st.checkbox("2️⃣ Segment", value=st.session_state.run_segment, help="Detect layout blocks")
            st.session_state.run_label = st.checkbox("3️⃣ Label", value=st.session_state.run_label, help="Semantic labeling via Ollama SLM")
            st.session_state.run_ocr_assemble = st.checkbox("4️⃣ OCR & Assemble", value=st.session_state.run_ocr_assemble, help="Extract text from blocks and build JSON/Markdown")
            
            # Model settings
            with st.expander("🔧 Model Settings", expanded=False):
                from utils.config import Config as ConfigClass
                default_model = (ConfigClass.LAYOUT_MODEL or "publaynet").lower()
                model_options = ["publaynet", "prima", "docbank", "tablebank"]
                selected_model = st.selectbox(
                    "Layout Model",
                    options=model_options,
                    index=0 if default_model in model_options else 0,
                    help="Choose LayoutParser model. DocBank/PRIMA may provide more granular detection."
                )
                st.session_state.layout_model_choice = selected_model
                
                det_threshold = st.slider(
                    "ML Model Detection Threshold",
                    min_value=0.05,
                    max_value=0.50,
                    value=st.session_state.get("layout_threshold", 0.25),
                    step=0.05,
                    help="Lower = more blocks detected (may include noise). Higher = fewer, higher-confidence blocks. Recommended: 0.15-0.25"
                )
                st.session_state.layout_threshold = det_threshold
                
                heuristic_strictness = st.slider(
                    "Heuristic Strictness",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("heuristic_strictness", 0.7),
                    step=0.1,
                    help="Controls how strict heuristic rules are. Higher = fewer false positives (logos as FORM, # as LIST). Lower = more aggressive detection. Recommended: 0.6-0.8"
                )
                st.session_state.heuristic_strictness = heuristic_strictness

                enable_form_geom = st.checkbox(
                    "Enable Form Geometry Detection",
                    value=st.session_state.get("enable_form_geometry", True),
                    help="Detect checkboxes and form fields using geometry from the analysis masks. Disable if over-detecting on dense pages."
                )
                st.session_state.enable_form_geometry = enable_form_geom

                geometry_strictness = st.slider(
                    "Form Geometry Strictness",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("geometry_strictness", 0.7),
                    step=0.1,
                    help="Higher = fewer small boxes (checkbox noise). Lower = more candidates."
                )
                st.session_state.geometry_strictness = geometry_strictness
                
                slm_toggle = st.checkbox(
                    "Enable Semantic Labeling (SLM)",
                    value=st.session_state.enable_slm,
                    help="Uses Qwen2.5-7B-Instruct via Ollama to assign semantic roles (title, h1, h2, header, footer, etc.) to text blocks. Requires: 1) Ollama running (ollama serve), 2) Model pulled (ollama pull qwen2.5:7b-instruct)"
                )
                st.session_state.enable_slm = slm_toggle
                
                vlm_toggle = st.checkbox(
                    "Enable Qwen-VL for tables/figures (VLM)",
                    value=st.session_state.enable_vlm,
                    help="Uses Qwen-VL via Ollama for advanced table structure extraction and figure classification. Requires: 1) Ollama running (ollama serve), 2) Model pulled (ollama pull qwen-vl)"
                )
                st.session_state.enable_vlm = vlm_toggle
            
            # Visualization settings
            with st.expander("🎨 Visualization", expanded=False):
                st.session_state.show_labels = st.checkbox("Show Labels", value=st.session_state.show_labels, help="Display block type labels on image")
                st.session_state.label_size = st.slider("Label Size", 0.4, 1.0, st.session_state.label_size, 0.1)
                
                visible_type_defaults = [bt.value for bt in BlockType if bt != BlockType.UNKNOWN]
                if "block_type_filter" not in st.session_state:
                    st.session_state.block_type_filter = visible_type_defaults
                
                selected_types = st.multiselect(
                    "Visible Types",
                    options=visible_type_defaults,
                    default=st.session_state.block_type_filter,
                    help="Filter block types to display"
                )
                if selected_types:
                    st.session_state.block_type_filter = selected_types
                else:
                    st.session_state.block_type_filter = visible_type_defaults
            
            # Block legend
            with st.expander("🎨 Block Colors", expanded=False):
                for block_type in BlockType:
                    if block_type == BlockType.UNKNOWN:
                        continue
                    color = BLOCK_TYPE_COLORS.get(block_type, (128, 128, 128))
                    hex_color = _bgr_to_hex(color)
                    st.markdown(
                        f"<span style='display:inline-block;width:20px;height:20px;"
                        f"border-radius:4px;background:{hex_color};margin-right:8px;vertical-align:middle;"
                        f"border:1px solid #ccc;'></span>"
                        f"<strong>{block_type.value.upper()}</strong>",
                        unsafe_allow_html=True
                    )
    
    else:
        # Minimal sidebar when collapsed - use session state values
        with st.sidebar:
            st.write("**Sidebar collapsed**")
            if st.button("Show Sidebar"):
                sidebar_collapsed = False
                st.rerun()
        
        # Get document path from session state
        if "document_path" in st.session_state:
            document_path = Path(st.session_state.document_path)
            document_label = st.session_state.get("document_label", "Unknown")
        else:
            st.info("👆 Please expand sidebar to select a document")
            st.stop()
    
    # Get variables from session state (works for both collapsed and expanded)
    run_ingest = st.session_state.run_ingest
    run_segment = st.session_state.run_segment
    run_ocr_assemble = st.session_state.run_ocr_assemble
    run_label = st.session_state.run_label
    show_labels = st.session_state.show_labels
    label_size = st.session_state.label_size
    
    # Get document path if not already set
    if not sidebar_collapsed:
        if document_path is None:
            if "document_path" in st.session_state:
                document_path = Path(st.session_state.document_path)
                document_label = st.session_state.get("document_label", "Unknown")
            else:
                st.info("👆 Select or upload a document to begin")
                st.stop()
    else:
        document_path = Path(st.session_state.document_path)
        document_label = st.session_state.get("document_label", "Unknown")
    
    # Initialize additional session state (already initialized above)
    if "pages" not in st.session_state:
        st.session_state.pages = None
    if "blocks" not in st.session_state:
        st.session_state.blocks = []
    if "document" not in st.session_state:
        st.session_state.document = None
    if "highlight_id" not in st.session_state:
        st.session_state.highlight_id = None
    if "assembled_json_dict" not in st.session_state:
        st.session_state.assembled_json_dict = None
    if "assembled_json_str" not in st.session_state:
        st.session_state.assembled_json_str = None
    if "assembled_markdown" not in st.session_state:
        st.session_state.assembled_markdown = None
    if "form_fields" not in st.session_state:
        st.session_state.form_fields = []
    if "checkboxes" not in st.session_state:
        st.session_state.checkboxes = []
    if "layout_model_choice" not in st.session_state:
        st.session_state.layout_model_choice = "publaynet"
    if "layout_threshold" not in st.session_state:
        st.session_state.layout_threshold = 0.25  # Lower default for better granularity
    if "current_page_idx" not in st.session_state:
        st.session_state.current_page_idx = 0  # Default to first page
    
    # Run pipeline stages
    pipeline_start = time.time()
    
    if run_ingest:
        with st.spinner("🔄 Ingesting document..."):
            try:
                pages = ingest_document(str(document_path))
                st.session_state.pages = pages
                st.session_state.blocks = []
                st.session_state.document = None
                st.session_state.assembled_json_dict = None
                st.session_state.assembled_json_str = None
                st.session_state.assembled_markdown = None
                st.session_state.form_fields = []
                st.session_state.checkboxes = []
                st.session_state.result_summary = None
                st.session_state.result_summary = None
                st.session_state.result_summary = None
                if not sidebar_collapsed:
                    st.sidebar.success(f"✅ {len(pages)} pages loaded")
            except Exception as e:
                st.error(f"❌ Ingest error: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
    
    if run_segment and st.session_state.pages:
        with st.spinner("🔄 Detecting layout blocks..."):
            try:
                # Use cached segmenter for faster inference (models are loaded once)
                # But don't cache by threshold - allow threshold changes
                @st.cache_resource
                def get_segmenter_base(model_name):
                    print(f"🔄 Initializing LayoutSegmenter (this may take 1-2 minutes on first run)...")
                    print(f"   Model: {model_name}")
                    # Initialize with default threshold, we'll override it
                    segmenter = LayoutSegmenter(
                        model_name=model_name,
                        score_threshold=0.25,  # Default, will be overridden
                        enable_table_model=True
                    )
                    print(f"✅ LayoutSegmenter initialized successfully")
                    return segmenter
                
                # Get base segmenter (cached by model only)
                segmenter = get_segmenter_base(
                    st.session_state.get("layout_model_choice", "publaynet")
                )
                
                # Override threshold and heuristic strictness (not cached, so changes are respected)
                current_threshold = st.session_state.get("layout_threshold", 0.25)
                heuristic_strictness = st.session_state.get("heuristic_strictness", 0.7)
                segmenter.extra_config["threshold"] = current_threshold
                segmenter.heuristic_strictness = heuristic_strictness
                segmenter.enable_form_geometry = st.session_state.get("enable_form_geometry", True)
                segmenter.geometry_strictness = st.session_state.get("geometry_strictness", 0.7)
                if hasattr(segmenter.model, "threshold"):
                    segmenter.model.threshold = current_threshold
                print(f"   Using ML threshold: {current_threshold}, Heuristic strictness: {heuristic_strictness}, Form geometry: {segmenter.enable_form_geometry}, Form strictness: {segmenter.geometry_strictness}")
                all_blocks = []
                
                progress_bar = st.progress(0)
                segment_sig = inspect.signature(segmenter.segment_page)
                supports_digital = "digital_words" in segment_sig.parameters
                
                for idx, page in enumerate(st.session_state.pages):
                    kwargs = {
                        "merge_boxes": True,
                        "resolve_order": True
                    }
                    if supports_digital:
                        kwargs["digital_words"] = getattr(page, "digital_words", None)
                    analysis_layers = {
                        "analysis_image": getattr(page, "analysis_image", None),
                        "binary_image": getattr(page, "binary_image", None),
                        "line_mask": getattr(page, "line_mask", None),
                        "box_mask": getattr(page, "box_mask", None),
                        "orientation": getattr(page, "orientation", None),
                        "orientation_confidence": getattr(page, "orientation_confidence", None)
                    }
                    kwargs["analysis_layers"] = analysis_layers
                    kwargs["orientation"] = getattr(page, "orientation", None)
                    blocks = segmenter.segment_page(
                        page.image,
                        page.page_id,
                        **kwargs
                    )
                    all_blocks.extend(blocks)
                    progress_bar.progress((idx + 1) / len(st.session_state.pages))
                
                progress_bar.empty()
                st.session_state.blocks = all_blocks
                
                if all_blocks:
                    type_counts = Counter([block.type.value for block in all_blocks])
                    summary_str = " · ".join(f"{k}:{v}" for k, v in type_counts.most_common())
                    if not sidebar_collapsed:
                        st.sidebar.success(f"✅ {len(all_blocks)} blocks detected")
                        st.sidebar.caption(f"Types: {summary_str}")
                
                st.session_state.document = None
                st.session_state.assembled_json_dict = None
                st.session_state.assembled_json_str = None
                st.session_state.assembled_markdown = None
                st.session_state.form_fields = []
                st.session_state.checkboxes = []
            except Exception as e:
                st.error(f"❌ Segment error: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Old separate OCR step removed - now merged with OCR & Assemble below
    
    if run_label and st.session_state.blocks:
        with st.spinner("🔄 Labeling blocks..."):
            try:
                slm_labeler = SLMLabeler(enabled=st.session_state.get("enable_slm", Config.ENABLE_SLM))
                page_images = st.session_state.pages if st.session_state.pages else []
                st.session_state.blocks = slm_labeler.label_blocks(
                    st.session_state.blocks,
                    page_images
                )
                if not sidebar_collapsed:
                    enable_slm = st.session_state.get("enable_slm", Config.ENABLE_SLM)
                    if enable_slm:
                        labeled_count = len([b for b in st.session_state.blocks if b.role and b.role != BlockRole.UNKNOWN])
                        total_text_blocks = len([b for b in st.session_state.blocks if b.type in [BlockType.TEXT, BlockType.TITLE, BlockType.LIST] and b.text])
                        if labeled_count > 0:
                            st.sidebar.success(f"✅ Labeling complete: {labeled_count}/{total_text_blocks} text blocks labeled")
                        else:
                            st.sidebar.success("✅ Labeling complete (no semantic roles assigned)")
                    else:
                        st.sidebar.success("✅ Labeling skipped (SLM disabled)")
                st.session_state.document = None
                st.session_state.assembled_json_dict = None
                st.session_state.assembled_json_str = None
                st.session_state.assembled_markdown = None
            except Exception as e:
                st.error(f"❌ Label error: {e}")
    
    if run_ocr_assemble and st.session_state.blocks and st.session_state.pages:
        pipeline_start = time.time()
        with st.spinner("🔄 Running OCR & Assemble..."):
            try:
                # Lazy OCR pipeline initialization (cached to avoid re-initialization)
                @st.cache_resource
                def get_ocr_pipeline():
                    print("🔄 Initializing OCR pipeline (one-time, may take 30-60 seconds)...")
                    return OCRPipeline(max_workers=4)
                
                ocr_pipeline = get_ocr_pipeline()
                
                # Process OCR for ALL pages (but cache results per page to avoid re-processing)
                # Check which pages need OCR processing
                # Use old method: process ALL text-like blocks (don't skip if they have text - OCR is more reliable)
                pages_to_process = []
                for page_idx, page in enumerate(st.session_state.pages):
                    # Skip if already cached
                    if page_idx in st.session_state.ocr_cache:
                        continue
                    page_blocks = [b for b in st.session_state.blocks if b.page_id == page_idx]
                    # Process ALL text-like blocks (old method - always run OCR for best results)
                    blocks_needing_ocr = [
                        b for b in page_blocks
                        if b.type in {BlockType.TEXT, BlockType.TITLE, BlockType.LIST, BlockType.FORM, BlockType.TABLE}
                    ]
                    if blocks_needing_ocr:
                        pages_to_process.append(page_idx)
                
                # Process OCR for pages that need it (all pages on first run)
                if pages_to_process:
                    st.info(f"🔄 Processing OCR on {len(pages_to_process)} page(s) (cached for future use)...")
                    # Process all pages in parallel batches
                    all_blocks_to_process = []
                    all_pages_to_process = []
                    for page_idx in pages_to_process:
                        page_blocks = [b for b in st.session_state.blocks if b.page_id == page_idx]
                        if page_blocks:
                            all_blocks_to_process.extend(page_blocks)
                            all_pages_to_process.append(st.session_state.pages[page_idx])
                    
                    if all_blocks_to_process:
                        # Use old reliable method: always run OCR (don't skip digital), sequential processing
                        # This matches the old assemble step that had better results
                        processed_batch = ocr_pipeline.process_blocks(
                            all_blocks_to_process,
                            all_pages_to_process,
                            skip_ocr_for_digital=False,  # Always run OCR for better results (old method)
                            parallel=False  # Sequential for reliability (PaddleOCR is not thread-safe)
                        )
                        # Cache results per page
                        for block in processed_batch:
                            page_idx = block.page_id
                            if page_idx not in st.session_state.ocr_cache:
                                st.session_state.ocr_cache[page_idx] = {}
                            st.session_state.ocr_cache[page_idx][block.id] = block
                else:
                    st.info("✅ Using cached OCR results (no re-processing needed)")
                
                # Merge cached OCR results with current blocks
                updated_blocks = []
                for block in st.session_state.blocks:
                    page_idx = block.page_id
                    if page_idx in st.session_state.ocr_cache and block.id in st.session_state.ocr_cache[page_idx]:
                        # Use cached OCR result
                        updated_blocks.append(st.session_state.ocr_cache[page_idx][block.id])
                    else:
                        # Keep original block
                        updated_blocks.append(block)
                
                st.session_state.blocks = updated_blocks
                
                # Now run assemble
                try:
                    assembler = DocumentAssembler(
                        process_tables=True,
                        process_figures=True,
                        use_vlm=st.session_state.get("enable_vlm", Config.ENABLE_VLM)
                    )
                except TypeError:
                    assembler = DocumentAssembler()
                
                doc = Document(
                    doc_id=document_label or document_path.name,
                    pages=st.session_state.pages or [],
                    blocks=st.session_state.blocks
                )
                
                assembled_json = assembler.assemble_json(doc)
                assembled_markdown = assembler.assemble_markdown(doc)
                
                st.session_state.document = doc
                st.session_state.assembled_json_dict = assembled_json
                st.session_state.assembled_json_str = json.dumps(assembled_json, indent=2)
                st.session_state.assembled_markdown = assembled_markdown
                st.session_state.form_fields = assembled_json.get("form_fields", [])
                st.session_state.checkboxes = assembled_json.get("checkboxes", [])
                st.session_state.result_summary = _build_result_summary(
                    st.session_state.blocks,
                    st.session_state.pages or [],
                    duration=time.time() - pipeline_start
                )
                
                # Count blocks with text
                populated_blocks = [
                    b for b in st.session_state.blocks
                    if (b.text and b.text.strip()) or (b.word_boxes and len(b.word_boxes) > 0)
                ]
                total_chars = sum(
                    len(str(b.text)) if (hasattr(b, "text") and b.text) else 
                    sum(len(wb.text) for wb in b.word_boxes if hasattr(wb, "text") and wb.text)
                    for b in populated_blocks
                )
                
                if not sidebar_collapsed:
                    st.sidebar.success(
                        f"✅ OCR & Assemble complete: {len(populated_blocks)}/{len(st.session_state.blocks)} blocks with text ({total_chars:,} chars)"
                    )
            except Exception as e:
                st.error(f"❌ OCR & Assemble error: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results
    if not st.session_state.pages:
        st.info("👆 Run 'Ingest' stage to load document")
        st.stop()
    
    # Page navigation
    num_pages = len(st.session_state.pages)
    if not sidebar_collapsed:
        # Handle single-page documents
        if num_pages ==1:
            page_idx = 0
            st.sidebar.write("📄 Page: 1 of 1")
        else:
            page_idx = st.sidebar.slider("📄 Page", 0, num_pages - 1, st.session_state.get("page_slider", 0), key="page_slider")
    else:
        # Minimal page selector when sidebar collapsed
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if num_pages == 1:
                page_idx = 0
                st.write("Page: 1 of 1")
            else:
                page_idx = st.selectbox("Page", range(num_pages), index=st.session_state.get("page_select", 0), key="page_select")
    
    # Store current page index for OCR (before OCR runs)
    st.session_state.current_page_idx = page_idx
    
    # Main content area
    current_page = st.session_state.pages[page_idx]
    page_image = current_page.image
    
    # Get blocks for current page
    page_blocks = [
        b for b in st.session_state.blocks
        if b.page_id == page_idx
    ]
    
    # Filter visible blocks
    visible_types = set(st.session_state.block_type_filter or [])
    highlight_id = st.session_state.highlight_id
    visible_blocks = [
        b for b in page_blocks
        if b.type.value in visible_types or (highlight_id and b.id == highlight_id)
    ]
    
    # Create annotated image
    show_labels_setting = show_labels
    label_size_setting = label_size
    
    if visible_blocks:
        annotated_image = draw_annotated_image(
            page_image,
            visible_blocks,
            highlight_id=highlight_id,
            show_labels=show_labels_setting,
            label_size=label_size_setting
        )
    else:
        annotated_image = page_image
    
    # Professional two-column layout: Image left, Results/JSON right (like Reducto/Extend)
    st.subheader(f"📄 Page {page_idx + 1} of {num_pages}")
    
    # Main content area: Image on left, Results/JSON on right
    col_image, col_results = st.columns([1.2, 1])
    
    with col_image:
        # Display annotated image
        st.image(annotated_image, width='stretch')
        
        # Stats and metrics below image
        st.markdown("---")
        st.subheader("📊 Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Blocks", len(page_blocks))
        with col2:
            type_counts = Counter([b.type.value for b in page_blocks])
            st.metric("Block Types", len(type_counts))
        with col3:
            avg_conf = np.mean([b.confidence for b in page_blocks]) if page_blocks else 0
            st.metric("Avg Confidence", f"{avg_conf:.3f}")
        
        # Additional metrics
        col4, col5 = st.columns(2)
        with col4:
            text_blocks = [b for b in page_blocks if b.text and b.text.strip()]
            st.metric("Text Blocks", len(text_blocks))
        with col5:
            if st.session_state.assembled_json_dict:
                st.metric("Status", "✅ Complete")
            else:
                st.metric("Status", "🔄 Processing")
        
        # Processing time if available
        if st.session_state.get("result_summary"):
            summary = st.session_state.result_summary
            if "processing_time" in summary:
                time_str = summary["processing_time"]
                st.caption(f"⏱️ Processing time: {time_str}")
        
        st.markdown("---")
        
        # Block summary table
        if page_blocks:
            st.subheader("📋 Block Summary")
            summary_data = []
            for block_type, count in Counter([b.type.value for b in page_blocks]).most_common():
                blocks_of_type = [b for b in page_blocks if b.type.value == block_type]
                avg_conf = np.mean([b.confidence for b in blocks_of_type])
                text_count = sum(1 for b in blocks_of_type if b.text and b.text.strip())
                summary_data.append({
                    "Type": block_type.upper(),
                    "Count": count,
                    "With Text": text_count,
                    "Avg Confidence": f"{avg_conf:.3f}"
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            # Block list (read-only, no interaction to avoid re-runs)
            st.subheader("📝 Detected Blocks")
            for idx, block in enumerate(page_blocks, 1):
                with st.expander(f"{idx}. {block.type.value.upper()} - {block.id[:20]}...", expanded=False):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**ID:** `{block.id}`")
                        st.markdown(f"**Type:** `{block.type.value}`")
                        if block.role:
                            st.markdown(f"**Role:** `{block.role.value}`")
                        st.markdown(f"**Confidence:** `{block.confidence:.3f}`")
                    with col_b:
                        st.markdown(f"**BBox:** `{block.bbox}`")
                        st.markdown(f"**Page:** `{block.page_id + 1}`")
                    
                    if block.text and block.text.strip():
                        st.markdown("**Extracted Text:**")
                        st.text_area("Block Text", block.text or "(No text extracted)", height=150, key=f"text_{block.id}", label_visibility="collapsed", disabled=True)
                    else:
                        st.info("No text extracted yet. Run OCR to extract text from this block.")
        else:
            st.info("No blocks detected. Run 'Segment' to detect layout blocks.")
    
    with col_results:
        # Results and JSON tabs on the right
        tab_results, tab_json = st.tabs(["📊 Results", "📄 JSON"])
        
        with tab_results:
            if st.session_state.get("result_summary"):
                _render_results_panel(st.session_state.result_summary)
            else:
                st.info("Run 'Assemble' to generate the consolidated results summary.")
                # Show quick page-level summary
                if page_blocks:
                    st.subheader("Page Blocks")
                    for idx, block in enumerate(page_blocks[:10], 1):  # Show first 10
                        snippet = (block.text or "").strip()[:100]
                        st.markdown(f"**{idx}. {block.type.value.upper()}**")
                        if snippet:
                            st.caption(snippet)
                        st.markdown("---")
        
        with tab_json:
            # JSON viewer with download buttons
            if st.session_state.assembled_json_dict:
                st.json(st.session_state.assembled_json_dict)
                st.markdown("---")
                if st.session_state.assembled_json_str:
                    st.download_button(
                        "⬇️ Download JSON",
                        st.session_state.assembled_json_str,
                        file_name=f"{document_label or 'document'}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                if st.session_state.assembled_markdown:
                    st.download_button(
                        "⬇️ Download Markdown",
                        st.session_state.assembled_markdown,
                        file_name=f"{document_label or 'document'}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            elif page_blocks:
                blocks_dict = [block.to_dict() for block in page_blocks]
                st.json(blocks_dict)
            else:
                st.info("Run 'Assemble' to generate full JSON output.")

        st.markdown("---")
        form_fields_data = st.session_state.get("form_fields") or []
        checkbox_data = st.session_state.get("checkboxes") or []
        if form_fields_data:
            st.download_button(
                "⬇️ Download Form Fields",
                json.dumps(form_fields_data, indent=2),
                file_name=f"{document_label or 'document'}_form_fields.json",
                mime="application/json",
                use_container_width=True
            )
        if checkbox_data:
            st.download_button(
                "⬇️ Download Checkboxes",
                json.dumps(checkbox_data, indent=2),
                file_name=f"{document_label or 'document'}_checkboxes.json",
                mime="application/json",
                use_container_width=True
            )
        if form_fields_data:
            with st.expander(f"🧾 Form Fields ({len(form_fields_data)})", expanded=False):
                for entry in form_fields_data:
                    label = entry.get("label_text") or "(No label)"
                    value = entry.get("value") or "[Empty]"
                    validator_badge = "✅" if entry.get("validator_passed") else "⚠️"
                    st.markdown(f"**{label}** → {value} {validator_badge}")
                    cols = st.columns([3, 1])
                    with cols[0]:
                        confidence = entry.get("ocr_confidence")
                        type_text = entry.get("field_type") or "unknown"
                        conf_text = f" • OCR: {confidence:.2f}" if confidence is not None else ""
                        st.caption(f"Type: {type_text}{conf_text}")
                    with cols[1]:
                        st.button(
                            "Highlight",
                            key=f"form_field_{entry['id']}",
                            on_click=_set_highlight,
                            args=(entry['id'],),
                            use_container_width=True
                        )
                    st.divider()
        if checkbox_data:
            with st.expander(f"☑️ Checkboxes ({len(checkbox_data)})", expanded=False):
                for entry in checkbox_data:
                    label = entry.get("label_text") or "(No label)"
                    state = entry.get("state", "ambiguous")
                    st.markdown(f"**{label}** → `{state}`")
                    cols = st.columns([3, 1])
                    with cols[0]:
                        st.caption(f"Confidence: {entry.get('confidence', 0.0):.2f}")
                    with cols[1]:
                        st.button(
                            "Highlight",
                            key=f"checkbox_{entry['id']}",
                            on_click=_set_highlight,
                            args=(entry['id'],),
                            use_container_width=True
                        )
                    st.divider()


if __name__ == "__main__":
    main()

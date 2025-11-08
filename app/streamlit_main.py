"""
Streamlit app for document processing pipeline visualization.
Professional UI with collapsible sidebar and enhanced visualization.
"""
import sys
from pathlib import Path
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from typing import Tuple, Optional
import cv2

from utils.models import Document, Block, BlockType
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
</style>
""", unsafe_allow_html=True)


def _bgr_to_hex(color: Tuple[int, int, int]) -> str:
    """Convert BGR color tuple to HEX string."""
    b, g, r = color
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


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
    if "run_ocr" not in st.session_state:
        st.session_state.run_ocr = False
    if "run_label" not in st.session_state:
        st.session_state.run_label = False
    if "run_assemble" not in st.session_state:
        st.session_state.run_assemble = False
    if "show_labels" not in st.session_state:
        st.session_state.show_labels = True
    if "label_size" not in st.session_state:
        st.session_state.label_size = 0.6
    
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
            sample_files = []
            if sample_dir.exists():
                sample_files = sorted([
                    p for p in sample_dir.glob("*")
                    if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}
                ])
            else:
                st.warning(f"Directory not found: {sample_dir}")
            
            sample_options = ["<Upload new>"] + [p.name for p in sample_files]
            sample_choice = st.selectbox("Choose sample", sample_options, key="sample_choice")
            
            uploaded_file = st.file_uploader(
                "Or upload PDF/Image",
                type=["pdf", "png", "jpg", "jpeg"],
                help="Upload a new document"
            )
            
            document_path: Optional[Path] = None
            document_label: Optional[str] = None
            
            if sample_choice != "<Upload new>":
                document_path = (sample_dir / sample_choice).resolve()
                document_label = sample_choice
            elif uploaded_file is not None:
                temp_path = Path("cache") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                document_path = temp_path.resolve()
                document_label = uploaded_file.name
            
            if document_path is None:
                st.info("👆 Select or upload a document to begin")
                st.stop()
            
            # Store in session state
            st.session_state.document_path = str(document_path)
            st.session_state.document_label = document_label
            st.success(f"✓ {document_label}")
            
            # Pipeline stages
            st.subheader("⚙️ Pipeline")
            st.session_state.run_ingest = st.checkbox("1️⃣ Ingest", value=st.session_state.run_ingest, help="Load and preprocess document")
            st.session_state.run_segment = st.checkbox("2️⃣ Segment", value=st.session_state.run_segment, help="Detect layout blocks")
            st.session_state.run_ocr = st.checkbox("3️⃣ OCR", value=st.session_state.run_ocr, help="Extract text from blocks")
            st.session_state.run_label = st.checkbox("4️⃣ Label", value=st.session_state.run_label, help="Semantic labeling (stubbed)")
            st.session_state.run_assemble = st.checkbox("5️⃣ Assemble", value=st.session_state.run_assemble, help="Generate JSON/Markdown")
            
            # Model settings
            with st.expander("🔧 Model Settings", expanded=False):
                default_model = (Config.LAYOUT_MODEL or "publaynet").lower()
                model_options = ["publaynet", "prima", "docbank", "tablebank"]
                selected_model = st.selectbox(
                    "Layout Model",
                    options=model_options,
                    index=0 if default_model in model_options else 0,
                    help="Choose LayoutParser model. DocBank/PRIMA may provide more granular detection."
                )
                st.session_state.layout_model_choice = selected_model
                
                det_threshold = st.slider(
                    "Detection Threshold",
                    min_value=0.10,
                    max_value=0.80,
                    value=st.session_state.get("layout_threshold", 0.25),  # Lower default for granularity
                    step=0.05,
                    help="Lower = more granular blocks, Higher = fewer high-confidence blocks"
                )
                st.session_state.layout_threshold = det_threshold
            
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
    run_ocr = st.session_state.run_ocr
    run_label = st.session_state.run_label
    run_assemble = st.session_state.run_assemble
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
    if "layout_model_choice" not in st.session_state:
        st.session_state.layout_model_choice = "publaynet"
    if "layout_threshold" not in st.session_state:
        st.session_state.layout_threshold = 0.25  # Lower default for better granularity
    if "current_page_idx" not in st.session_state:
        st.session_state.current_page_idx = 0  # Default to first page
    
    # Run pipeline stages
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
                @st.cache_resource
                def get_segmenter(model_name, threshold):
                    return LayoutSegmenter(
                        model_name=model_name,
                        score_threshold=threshold,
                        enable_table_model=True
                    )
                
                segmenter = get_segmenter(
                    st.session_state.get("layout_model_choice", "publaynet"),
                    st.session_state.get("layout_threshold", 0.25)  # Lower default for better granularity
                )
                all_blocks = []
                
                progress_bar = st.progress(0)
                for idx, page in enumerate(st.session_state.pages):
                    blocks = segmenter.segment_page(
                        page.image,
                        page.page_id,
                        merge_boxes=True,
                        resolve_order=True
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
            except Exception as e:
                st.error(f"❌ Segment error: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    if run_ocr and st.session_state.blocks and st.session_state.pages:
        # Get current page index from session state (set by page navigation or default to 0)
        current_page_idx = st.session_state.current_page_idx
        if current_page_idx >= len(st.session_state.pages):
            current_page_idx = 0
            st.session_state.current_page_idx = 0
        try:
            # Use cached OCR pipeline for faster inference (models loaded once)
            @st.cache_resource
            def get_ocr_pipeline():
                return OCRPipeline(max_workers=4)  # Use parallel processing
            
            ocr_pipeline = get_ocr_pipeline()
            
            # OPTIMIZATION: Only process blocks from the currently displayed page
            current_page_id = current_page_idx
            current_page_image = st.session_state.pages[current_page_id].image if current_page_id < len(st.session_state.pages) else None
            
            if current_page_image is None:
                st.warning("⚠️ No page image available for OCR")
                st.stop()
            
            # Filter blocks for current page only
            valid_blocks = []
            skipped_count = 0
            page_area = current_page_image.shape[0] * current_page_image.shape[1]
            
            for block in st.session_state.blocks:
                # Only process blocks from the current page
                if block.page_id != current_page_id:
                    continue
                
                x0, y0, x1, y1 = block.bbox
                block_area = (x1 - x0) * (y1 - y0)
                
                # Skip blocks that are too small (less than 100 pixels)
                if block_area < 100:
                    skipped_count += 1
                    continue
                
                # Skip figure blocks that are too large (likely misclassified, will be slow)
                if block.type == BlockType.FIGURE and block_area > page_area * 0.5:
                    skipped_count += 1
                    continue
                
                # Only process TEXT blocks initially for speed (skip FORM and TABLE)
                if block.type in [BlockType.TEXT, BlockType.TITLE, BlockType.LIST]:
                    valid_blocks.append(block)
                else:
                    skipped_count += 1
            
            if not valid_blocks:
                st.warning(f"⚠️ No valid blocks to process with OCR (skipped {skipped_count} blocks)")
                st.stop()
            
            # Show progress
            progress_container = st.container()
            with progress_container:
                st.info(f"🔄 Processing OCR on page {current_page_id + 1}: {len(valid_blocks)} blocks (skipped {skipped_count} invalid/large blocks)...")
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Process blocks for current page only
            processed_blocks_dict = {}
            total_blocks = len(valid_blocks)
            
            if total_blocks > 0:
                # Process all blocks for current page
                page_images_list = [current_page_image]  # Only current page
                batch_result = ocr_pipeline.process_blocks(
                    valid_blocks,
                    page_images_list,
                    skip_ocr_for_digital=True,
                    parallel=True
                )
                
                # Store results
                for block in batch_result:
                    processed_blocks_dict[block.id] = block
                    progress = len(processed_blocks_dict) / total_blocks
                    progress_bar.progress(progress)
                    status_text.text(f"Processed: {len(processed_blocks_dict)}/{total_blocks} blocks ({int(progress*100)}%)")
            
            # Reconstruct full block list (preserve order, include skipped blocks)
            final_blocks = []
            for block in st.session_state.blocks:
                if block.id in processed_blocks_dict:
                    final_blocks.append(processed_blocks_dict[block.id])
                else:
                    # Keep original block if not processed (will have empty text)
                    if not hasattr(block, 'text') or block.text is None:
                        block.text = ""
                        block.word_boxes = []
                    final_blocks.append(block)
            
            st.session_state.blocks = final_blocks
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            progress_container.empty()
            
            # Count results
            text_blocks = [b for b in final_blocks if b.text and len(b.text.strip()) > 0]
            total_chars = sum(len(b.text) for b in text_blocks)
            
            if not sidebar_collapsed:
                st.sidebar.success(f"✅ OCR complete: {len(text_blocks)}/{len(final_blocks)} blocks have text ({total_chars:,} chars)")
            
            st.session_state.document = None
            st.session_state.assembled_json_dict = None
            st.session_state.assembled_json_str = None
            st.session_state.assembled_markdown = None
        except Exception as e:
            st.error(f"❌ OCR error: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    if run_label and st.session_state.blocks:
        with st.spinner("🔄 Labeling blocks..."):
            try:
                slm_labeler = SLMLabeler(enabled=False)  # Stubbed
                page_images = [page.image for page in st.session_state.pages] if st.session_state.pages else []
                st.session_state.blocks = slm_labeler.label_blocks(
                    st.session_state.blocks,
                    page_images
                )
                if not sidebar_collapsed:
                    st.sidebar.success("✅ Labeling completed (stubbed)")
                st.session_state.document = None
                st.session_state.assembled_json_dict = None
                st.session_state.assembled_json_str = None
                st.session_state.assembled_markdown = None
            except Exception as e:
                st.error(f"❌ Label error: {e}")
    
    if run_assemble and st.session_state.blocks:
        with st.spinner("🔄 Assembling document..."):
            try:
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
                if not sidebar_collapsed:
                    st.sidebar.success("✅ Assembly completed")
            except Exception as e:
                st.error(f"❌ Assemble error: {e}")
    
    # Display results
    if not st.session_state.pages:
        st.info("👆 Run 'Ingest' stage to load document")
        st.stop()
    
    # Page navigation
    num_pages = len(st.session_state.pages)
    if not sidebar_collapsed:
        page_idx = st.sidebar.slider("📄 Page", 0, num_pages - 1, st.session_state.get("page_slider", 0), key="page_slider")
    else:
        # Minimal page selector when sidebar collapsed
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
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
    
    # Layout: Image on left, info on right (or full width if sidebar collapsed)
    if sidebar_collapsed:
        # Full width layout when sidebar collapsed
        st.image(annotated_image, use_column_width=True)
        
        # Stats below image
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Blocks", len(page_blocks))
        with col2:
            type_counts = Counter([b.type.value for b in page_blocks])
            st.metric("Types", len(type_counts))
        with col3:
            avg_conf = np.mean([b.confidence for b in page_blocks]) if page_blocks else 0
            st.metric("Avg Confidence", f"{avg_conf:.2f}")
        with col4:
            if st.session_state.assembled_json_dict:
                st.metric("Status", "✅ Complete")
            else:
                st.metric("Status", "🔄 Processing")
        
        # Tabs for outputs
        tab1, tab2, tab3 = st.tabs(["📊 Blocks", "📄 JSON", "📝 Markdown"])
        
        with tab1:
            if page_blocks:
                blocks_data = []
                for block in page_blocks:
                    blocks_data.append({
                        "ID": block.id,
                        "Type": block.type.value,
                        "Confidence": f"{block.confidence:.3f}",
                        "Text": (block.text or "")[:100] + ("..." if block.text and len(block.text) > 100 else ""),
                        "BBox": f"({block.bbox[0]:.0f}, {block.bbox[1]:.0f}, {block.bbox[2]:.0f}, {block.bbox[3]:.0f})"
                    })
                st.dataframe(pd.DataFrame(blocks_data), use_container_width=True, hide_index=True)
            else:
                st.info("No blocks detected. Run 'Segment' stage.")
        
        with tab2:
            if st.session_state.assembled_json_dict:
                st.json(st.session_state.assembled_json_dict)
                st.download_button(
                    "⬇️ Download JSON",
                    st.session_state.assembled_json_str,
                    file_name=f"{document_label or 'document'}.json",
                    mime="application/json"
                )
            elif page_blocks:
                blocks_dict = [block.to_dict() for block in page_blocks]
                st.json(blocks_dict)
            else:
                st.info("Run 'Assemble' to generate JSON")
        
        with tab3:
            if st.session_state.assembled_markdown:
                st.markdown(st.session_state.assembled_markdown)
                st.download_button(
                    "⬇️ Download Markdown",
                    st.session_state.assembled_markdown,
                    file_name=f"{document_label or 'document'}.md",
                    mime="text/markdown"
                )
            else:
                st.info("Run 'Assemble' to generate Markdown")
    
    else:
        # Two-column layout when sidebar visible
        col_img, col_info = st.columns([2, 1])
        
        with col_img:
            st.subheader(f"📄 Page {page_idx + 1} of {num_pages}")
            
            # Display annotated image
            st.image(annotated_image, use_column_width=True)
            
            # Quick stats
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Blocks", len(page_blocks))
            with col_stat2:
                type_counts = Counter([b.type.value for b in page_blocks])
                st.metric("Types", len(type_counts))
            with col_stat3:
                avg_conf = np.mean([b.confidence for b in page_blocks]) if page_blocks else 0
                st.metric("Confidence", f"{avg_conf:.2f}")
            
            # Block selector
            if page_blocks:
                st.subheader("🔍 Select Block")
                block_options = {
                    f"{b.id} ({b.type.value})": b.id
                    for b in page_blocks
                }
                selected_block_key = st.selectbox(
                    "Choose block to highlight",
                    options=list(block_options.keys()),
                    index=0 if highlight_id else None,
                    key="block_select"
                )
                if selected_block_key:
                    st.session_state.highlight_id = block_options[selected_block_key]
                    if st.button("Highlight"):
                        st.rerun()
        
        with col_info:
            st.subheader("📊 Block Details")
            
            if highlight_id:
                selected_block = next((b for b in page_blocks if b.id == highlight_id), None)
                if selected_block:
                    st.markdown(f"**ID:** `{selected_block.id}`")
                    st.markdown(f"**Type:** `{selected_block.type.value}`")
                    if selected_block.role:
                        st.markdown(f"**Role:** `{selected_block.role.value}`")
                    st.markdown(f"**Confidence:** `{selected_block.confidence:.3f}`")
                    st.markdown(f"**BBox:** `{selected_block.bbox}`")
                    
                    if selected_block.text:
                        st.markdown("**Text:**")
                        st.text_area("", selected_block.text, height=200, key="block_text", label_visibility="collapsed")
            
            # Block summary table
            if page_blocks:
                st.subheader("📋 Block Summary")
                summary_data = []
                for block_type, count in Counter([b.type.value for b in page_blocks]).most_common():
                    blocks_of_type = [b for b in page_blocks if b.type.value == block_type]
                    avg_conf = np.mean([b.confidence for b in blocks_of_type])
                    summary_data.append({
                        "Type": block_type.upper(),
                        "Count": count,
                        "Avg Conf": f"{avg_conf:.2f}"
                    })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            # Outputs
            st.subheader("📥 Outputs")
            
            if st.session_state.assembled_json_dict:
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
            
            # Tabs for detailed view
            tab1, tab2 = st.tabs(["JSON", "Markdown"])
            
            with tab1:
                if st.session_state.assembled_json_dict:
                    st.json(st.session_state.assembled_json_dict)
                elif page_blocks:
                    blocks_dict = [block.to_dict() for block in page_blocks]
                    st.json(blocks_dict)
                else:
                    st.info("Run 'Assemble' for full JSON")
            
            with tab2:
                if st.session_state.assembled_markdown:
                    st.markdown(st.session_state.assembled_markdown)
                else:
                    st.info("Run 'Assemble' for Markdown")


if __name__ == "__main__":
    main()

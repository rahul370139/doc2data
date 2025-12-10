"""
Doc2Data - Intelligent Document Extraction Workstation
Professional UI for CMS-1500 and General Forms
"""
import sys
from pathlib import Path
import json
import time
import base64
from io import BytesIO

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# Core imports
from utils.config import Config
from src.pipelines.form_extractor import load_form_schema, extract_with_full_pipeline

# --- UI Configuration ---
st.set_page_config(
    page_title="Doc2Data | Intelligent Extraction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    /* Main Container */
    .main {
        background-color: #0E1117;
    }
    
    /* Card Styling */
    .css-1r6slb0 {
        background-color: #1E2329;
        border: 1px solid #30363D;
        padding: 1.5rem;
        border-radius: 0.5rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        color: #E6E6E6;
    }
    
    /* Custom Classes */
    .stat-box {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #6366F1;
        margin-bottom: 10px;
    }
    
    .stat-label {
        color: #9CA3AF;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stat-value {
        color: #F3F4F6;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .confidence-high { color: #10B981; }
    .confidence-med { color: #F59E0B; }
    .confidence-low { color: #EF4444; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

def get_image_download_link(img, filename, text):
    """Generates a link to download the image"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def draw_bounding_boxes(image_np, field_details, highlight_id=None, ocr_blocks=None, 
                        source_width=None, source_height=None):
    """
    Draw bounding boxes on the image.
    Handles ML-detected boxes and highlights.
    
    Args:
        image_np: Display image (may be different DPI than source)
        field_details: List of field dicts with bbox
        highlight_id: ID of field to highlight
        ocr_blocks: Raw OCR blocks for fallback visualization
        source_width: Width of source image (for coordinate scaling)
        source_height: Height of source image (for coordinate scaling)
    """
    img = image_np.copy()
    display_h, display_w = img.shape[:2]
    
    # Calculate scale factors if source dimensions provided
    scale_x = display_w / source_width if source_width else 1.0
    scale_y = display_h / source_height if source_height else 1.0
    
    print(f"[draw_bounding_boxes] Display: {display_w}x{display_h}, Source: {source_width}x{source_height}")
    print(f"[draw_bounding_boxes] Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
    print(f"[draw_bounding_boxes] Field details count: {len(field_details)}, OCR blocks: {len(ocr_blocks) if ocr_blocks else 0}")
    
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    def scale_bbox(bbox):
        """Scale bbox from source to display coordinates."""
        x0, y0, x1, y1 = bbox
        return [int(x0 * scale_x), int(y0 * scale_y), int(x1 * scale_x), int(y1 * scale_y)]
    
    def clamp_bbox(x0, y0, x1, y1):
        """Clamp bbox to image bounds."""
        x0 = max(0, min(x0, display_w - 1))
        y0 = max(0, min(y0, display_h - 1))
        x1 = max(0, min(x1, display_w))
        y1 = max(0, min(y1, display_h))
        return x0, y0, x1, y1
    
    boxes_drawn = 0
    fields_with_bbox = 0
    
    # Draw field boxes
    for field in field_details:
        bbox = field.get("bbox")
        if not bbox:
            continue
        
        fields_with_bbox += 1
        
        # Handle both tuple and list formats
        if len(bbox) != 4:
            continue
            
        x0, y0, x1, y1 = scale_bbox(bbox)
        
        # Clamp to image bounds instead of skipping
        x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1)
        
        # Skip only if completely invalid
        if x1 <= x0 or y1 <= y0:
            continue
            
        color = (100, 100, 100) # Default gray
        thickness = 2
        
        # Method specific colors
        method = field.get("detected_by", "unknown")
        if "slm" in method or "llm" in method:
            color = (0, 220, 80) # Bright Green
            thickness = 2
        elif "heuristic" in method or "ocr" in method:
            color = (80, 150, 255) # Blue
            thickness = 2
            
        if field.get("id") == highlight_id:
            continue # Draw highlight last
            
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        boxes_drawn += 1
    
    print(f"[draw_bounding_boxes] Fields with bbox: {fields_with_bbox}, Boxes drawn: {boxes_drawn}")

    # Draw OCR blocks as fallback if few field boxes were drawn
    if boxes_drawn < 5 and ocr_blocks:
        print(f"[draw_bounding_boxes] Few field boxes ({boxes_drawn}), drawing OCR blocks")
        for block in ocr_blocks:
            bbox = block.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            
            x0, y0, x1, y1 = scale_bbox(bbox)
            x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1)
            
            if x1 <= x0 or y1 <= y0:
                continue
                
            # Light cyan for raw OCR blocks
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 200), 1)
            boxes_drawn += 1

    # Draw highlight
    if highlight_id:
        for field in field_details:
            if field.get("id") == highlight_id:
                bbox = field.get("bbox")
                if bbox and len(bbox) == 4:
                    x0, y0, x1, y1 = scale_bbox(bbox)
                    x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1)
                    if x1 > x0 and y1 > y0:
                        # Glow effect
                        cv2.rectangle(img, (x0-2, y0-2), (x1+2, y1+2), (255, 165, 0), 4) # Orange outline
                        # Add label
                        label = field.get("label", "")[:20]
                        cv2.putText(img, label, (x0, max(y0-10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                break
    
    print(f"[draw_bounding_boxes] Drew {boxes_drawn} boxes, scale=({scale_x:.2f}, {scale_y:.2f})")
    return img

# --- Main Application ---

def main():
    # Header Area
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Doc2Data Workstation")
        st.caption("Intelligent Document Processing • CMS-1500 • General Forms")
    
    with col2:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 1rem;">
                <span style="background-color: #374151; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem;">
                    v2.1.0 (Beta)
                </span>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Controls")
        
        st.subheader("1. Document Source")
        uploaded_file = st.file_uploader("Upload PDF/Image", type=['pdf', 'png', 'jpg'])
        
        sample_dir = Path(project_root / "data" / "sample_docs")
        sample_files = sorted([f.name for f in sample_dir.glob("*") if f.suffix in ['.pdf', '.png', '.jpg']])
        selected_sample = st.selectbox("Or select sample", ["None"] + sample_files)
        
        st.subheader("2. Processing Mode")
        doc_type = st.selectbox("Document Type", ["CMS-1500", "General Form (Layout Only)"])
        
        use_vlm = st.toggle("Enable VLM/ICR", value=True, help="Use Vision Language Model for intelligent field extraction")
        
        if st.button("▶ Run Extraction", type="primary", use_container_width=True):
            st.session_state.run_extraction = True
            # Determine file path
            if uploaded_file:
                # Save temp
                temp_path = Path("cache") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.target_path = str(temp_path)
            elif selected_sample != "None":
                st.session_state.target_path = str(sample_dir / selected_sample)
            else:
                st.error("Please select a document first.")
                st.session_state.run_extraction = False

    # --- Main Content Area ---
    
    if "target_path" not in st.session_state:
        # Welcome State
        st.info("👈 Please upload a document or select a sample to begin.")
        
        # Show features
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🚀 Intelligent Extraction")
            st.write("Uses advanced Vision-Language Models (Qwen-VL) to understand forms semantically.")
        with c2:
            st.markdown("### 🎯 Precise Grounding")
            st.write("Maps extracted data back to original pixels for verification.")
        with c3:
            st.markdown("### 📝 Form Agnostic")
            st.write("Works on CMS-1500 and adapts to other structures via schema.")
        return

    # Extraction Logic
    if st.session_state.get("run_extraction", False):
        with st.spinner("Processing document... (OCR + VLM Analysis)"):
            try:
                schema = load_form_schema("cms-1500") if doc_type == "CMS-1500" else None
                result = extract_with_full_pipeline(
                    st.session_state.target_path,
                    schema=schema,
                    use_vlm=use_vlm
                )
                st.session_state.extraction_result = result
                st.session_state.run_extraction = False # Reset trigger
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Results Display
    if "extraction_result" in st.session_state:
        result = st.session_state.extraction_result
        
        # --- Statistics Bar ---
        fields = result.get("field_details", [])
        extracted_count = sum(1 for f in fields if f.get("value"))
        total_fields = len(fields)
        confidence_scores = [f.get("confidence", 0) for f in fields if f.get("value")]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        st.markdown("---")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f'<div class="stat-box"><div class="stat-label">Fields Found</div><div class="stat-value">{extracted_count}/{total_fields}</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-box"><div class="stat-label">Avg Confidence</div><div class="stat-value">{avg_confidence:.1%}</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="stat-box"><div class="stat-label">Method</div><div class="stat-value">{result.get("extraction_method", "N/A")}</div></div>', unsafe_allow_html=True)
        with s4:
            st.markdown(f'<div class="stat-box"><div class="stat-label">Blocks Detected</div><div class="stat-value">{result.get("total_blocks_detected", 0)}</div></div>', unsafe_allow_html=True)
            
        # --- Split View ---
        col_viewer, col_data = st.columns([1.2, 1])
        
        with col_viewer:
            st.subheader("📄 Document Viewer")
            
            # Render Image
            import fitz
            pdf = fitz.open(st.session_state.target_path)
            page = pdf[0]
            pix = page.get_pixmap(dpi=200) # Good balance for display
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4: # RGBA
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
            # Draw Boxes with proper coordinate scaling
            highlight_id = st.session_state.get("highlight_id")
            source_w = result.get("page_width")
            source_h = result.get("page_height")
            ocr_blocks = result.get("ocr_blocks", [])
            
            annotated_img = draw_bounding_boxes(
                img_np, fields, highlight_id,
                ocr_blocks=ocr_blocks,
                source_width=source_w,
                source_height=source_h
            )
            
            st.image(annotated_img, use_container_width=True)
            
            st.caption("Green: LLM Grounded • Blue: OCR Detected • Cyan: Raw OCR • Orange: Selected")

        with col_data:
            st.subheader("📊 Extracted Data")
            
            # Tabbed Interface
            tab_form, tab_json = st.tabs(["Form View", "Raw JSON"])
            
            with tab_form:
                # Prepare data for interactive table
                df_data = []
                for f in fields:
                    df_data.append({
                        "ID": f.get("id"),
                        "Label": f.get("label"),
                        "Value": f.get("value"),
                        "Conf": f.get("confidence", 0.0),
                        "_bbox": f.get("bbox") # Hidden column for logic
                    })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    
                    # Interactive Data Editor
                    edited_df = st.data_editor(
                        df,
                        column_config={
                            "Conf": st.column_config.ProgressColumn(
                                "Conf", format="%.2f", min_value=0, max_value=1
                            ),
                            "_bbox": None # Hide bbox column
                        },
                        use_container_width=True,
                        key="data_editor",
                        num_rows="dynamic"
                    )
                    
                    # Highlighting Logic
                    # We create a simple selectbox to pick a field to locate
                    field_options = {f"{r['Label']} ({r['Value'][:15]}...)": r['ID'] for r in df_data if r['Value']}
                    
                    selected_field_label = st.selectbox(
                        "🔍 Locate Field on Document", 
                        options=["Select a field..."] + list(field_options.keys()),
                        index=0
                    )
                    
                    if selected_field_label != "Select a field...":
                        st.session_state.highlight_id = field_options[selected_field_label]
                    else:
                        st.session_state.highlight_id = None
                        
                else:
                    st.warning("No fields extracted.")

            with tab_json:
                st.json(result.get("extracted_fields", {}))
                
                # Download Button
                json_str = json.dumps(result, indent=2)
                st.download_button(
                    label="Download Full Result JSON",
                    data=json_str,
                    file_name="extraction_result.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()

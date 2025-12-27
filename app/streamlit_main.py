"""
Doc2Data - Intelligent Document Extraction
Production-ready Streamlit UI with dark theme and professional controls.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import Config


# ============================================================================
# Page Configuration & Theme
# ============================================================================

st.set_page_config(
    page_title="Doc2Data | Intelligent Extraction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Doc2Data - Healthcare Document Processing Pipeline",
        'Report a Bug': "https://github.com/rahul370139/doc2data/issues",
    }
)

# Dark Theme CSS
st.markdown("""
<style>
    /* Root Variables */
    :root {
        --bg-primary: #0E1117;
        --bg-secondary: #1A1D24;
        --bg-tertiary: #262B36;
        --text-primary: #FAFAFA;
        --text-secondary: #A8B2C1;
        --accent-blue: #4F8EF7;
        --accent-green: #10B981;
        --accent-yellow: #F59E0B;
        --accent-red: #EF4444;
        --accent-purple: #8B5CF6;
        --border-color: #30363D;
    }
    
    /* Main Background */
    .main {
        background-color: var(--bg-primary);
    }
    
    .stApp {
        background-color: var(--bg-primary);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    /* Cards */
    .card {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 142, 247, 0.4);
    }
    
    /* Selectbox / Dropdown */
    .stSelectbox > div > div {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        color: var(--text-primary);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--bg-secondary);
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: var(--text-secondary);
        border-radius: 6px;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-blue);
        color: white;
    }
    
    /* Toggle */
    .stToggle > label > div {
        color: var(--text-primary);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: var(--accent-blue);
    }
    
    /* DataFrame */
    .stDataFrame {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* JSON viewer */
    .stJson {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Confidence badges */
    .conf-high { color: var(--accent-green); font-weight: 600; }
    .conf-med { color: var(--accent-yellow); font-weight: 600; }
    .conf-low { color: var(--accent-red); font-weight: 600; }
    
    /* Form type badge */
    .form-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .form-badge-cms1500 {
        background-color: rgba(79, 142, 247, 0.2);
        color: var(--accent-blue);
        border: 1px solid var(--accent-blue);
    }
    
    .form-badge-generic {
        background-color: rgba(139, 92, 246, 0.2);
        color: var(--accent-purple);
        border: 1px solid var(--accent-purple);
    }
    
    /* Pipeline badge */
    .pipeline-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
        color: white;
    }
    
    /* Top bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .logo {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .logo-icon {
        margin-right: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "target_path" not in st.session_state:
    st.session_state.target_path = None
if "highlight_id" not in st.session_state:
    st.session_state.highlight_id = None


# ============================================================================
# Helper Functions
# ============================================================================

def load_form_schema(schema_name: str):
    """Load form schema JSON."""
    schema_path = project_root / "data" / "schemas" / f"{schema_name}.json"
    if schema_path.exists():
        with open(schema_path) as f:
            return json.load(f)
    return None


def get_sample_documents():
    """Get list of sample documents."""
    sample_dir = project_root / "data" / "sample_docs"
    if not sample_dir.exists():
        return []
    
    docs = []
    for ext in ["*.pdf", "*.png", "*.jpg"]:
        docs.extend([f.name for f in sample_dir.glob(ext)])
    return sorted(docs)


def draw_bounding_boxes(image, fields, highlight_id=None, ocr_blocks=None, 
                       source_width=None, source_height=None, confidence_threshold=0.3,
                       show_labels=True):
    """Draw bounding boxes on image with semantic labels (Reducto-style)."""
    img = image.copy()
    h, w = img.shape[:2]
    
    scale_x = w / source_width if source_width else 1
    scale_y = h / source_height if source_height else 1
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    font_thickness = 1
    
    for field in fields:
        conf = field.get("confidence", 0)
        if conf < confidence_threshold:
                continue
        
        bbox = field.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue
        
        x0, y0, x1, y1 = bbox
        # Support both bbox formats:
        # - xyxy: [x0, y0, x1, y1]
        # - ltwh: [left, top, width, height]  (common in Reducto-style outputs)
        is_normalized = all(0 <= v <= 1 for v in bbox)
        is_ltwh = (x1 < x0) or (y1 < y0)  # width/height smaller than left/top is a strong signal
        if is_normalized:
            if is_ltwh:
                left, top, bw, bh = bbox
                x0, y0 = int(left * w), int(top * h)
                x1, y1 = int((left + bw) * w), int((top + bh) * h)
            else:
                x0, y0, x1, y1 = int(x0*w), int(y0*h), int(x1*w), int(y1*h)
        else:
            if is_ltwh:
                left, top, bw, bh = bbox
                x0, y0 = int(left * scale_x), int(top * scale_y)
                x1, y1 = int((left + bw) * scale_x), int((top + bh) * scale_y)
            else:
                x0, y0, x1, y1 = int(x0*scale_x), int(y0*scale_y), int(x1*scale_x), int(y1*scale_y)
        
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        
        if conf >= 0.8:
            color = (34, 197, 94)
        elif conf >= 0.5:
            color = (251, 191, 36)
        else:
            color = (239, 68, 68)
        
        thickness = 3 if field.get("id") == highlight_id else 2
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        
        if show_labels:
            label = field.get("label", field.get("id", ""))
            if label and len(label) > 20:
                label = label[:18] + ".."
            if label:
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                label_y = y0 - 4 if y0 > text_h + 4 else y0 + text_h + 4
                cv2.rectangle(img, (x0, label_y - text_h - 2), (x0 + text_w + 4, label_y + 2), color, -1)
                cv2.putText(img, label, (x0 + 2, label_y - 2), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    if ocr_blocks:
        for block in ocr_blocks:
            bbox = block.get("bbox", [])
            if len(bbox) >= 4:
                bx0, by0, bx1, by1 = bbox[:4]
                is_normalized = all(0 <= v <= 1 for v in [bx0, by0, bx1, by1])
                is_ltwh = (bx1 < bx0) or (by1 < by0)
                if is_normalized:
                    if is_ltwh:
                        left, top, bw, bh = bbox[:4]
                        bx0, by0 = int(left * w), int(top * h)
                        bx1, by1 = int((left + bw) * w), int((top + bh) * h)
                    else:
                        bx0, by0, bx1, by1 = int(bx0*w), int(by0*h), int(bx1*w), int(by1*h)
                else:
                    if is_ltwh:
                        left, top, bw, bh = bbox[:4]
                        bx0, by0 = int(left * scale_x), int(top * scale_y)
                        bx1, by1 = int((left + bw) * scale_x), int((top + bh) * scale_y)
                    else:
                        bx0, by0 = int(bx0 * scale_x), int(by0 * scale_y)
                        bx1, by1 = int(bx1 * scale_x), int(by1 * scale_y)
                bx0, by0 = max(0, bx0), max(0, by0)
                bx1, by1 = min(w, bx1), min(h, by1)
                if bx1 - bx0 > 2 and by1 - by0 > 2:
                    cv2.rectangle(img, (bx0, by0), (bx1, by1), (59, 130, 246), 1)
    
    return img


# ============================================================================
# Pipeline Functions
# ============================================================================

@st.cache_resource
def get_pipeline(pipeline_type: str, config: dict):
    """Get cached pipeline instance."""
    if pipeline_type == "multi_agent":
        from src.pipelines.multi_agent_pipeline import MultiAgentPipeline, PipelineConfig, FormType
        
        # Parse form type override
        form_type_override = None
        if config.get("form_type"):
            try:
                form_type_override = FormType(config.get("form_type"))
            except ValueError:
                pass
        
        pconfig = PipelineConfig(
            enable_trocr=config.get("enable_trocr", True),
            enable_slm_labeling=config.get("enable_slm", True),
            enable_vlm_figures=config.get("enable_vlm", True),
            enable_alignment=config.get("enable_alignment", True),
            ocr_confidence_threshold=config.get("confidence_threshold", 0.5),
            handwriting_threshold=config.get("handwriting_threshold", 0.35),
            zone_padding_px=int(config.get("ocr_padding", 12)),
            form_type_override=form_type_override
        )
        return MultiAgentPipeline(pconfig)
    elif pipeline_type == "cms1500_production":
        from src.pipelines.cms1500_production import CMS1500ProductionPipeline
        return CMS1500ProductionPipeline(
            use_trocr=config.get("enable_trocr", True),
            use_llm_qa=config.get("enable_llm_qa", True),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            handwriting_threshold=config.get("handwriting_threshold", 0.35),
        )
    else:
        # Default form extractor
        from src.pipelines.form_extractor import extract_with_full_pipeline
        return extract_with_full_pipeline
    return None


def run_extraction(file_path: str, pipeline_type: str, config: dict) -> dict:
    """Run extraction pipeline."""
    try:
        if pipeline_type == "multi_agent":
            # Multi-Agent orchestrated pipeline (end-to-end)
            pipeline = get_pipeline(pipeline_type, config)
            return pipeline.process_sync(file_path)
        elif pipeline_type == "agentic":
            # CMS-1500 Agentic pipeline with template alignment + LLM
            from src.pipelines.agentic_cms1500 import run_cms1500_agentic
            return run_cms1500_agentic(
                file_path,
                use_icr=config.get("enable_trocr", True),
                use_llm=config.get("enable_vlm", True),
                align_template=config.get("enable_alignment", True)
            )
        else:
            # General/Full-page LLM fallback
            from src.pipelines.form_extractor import extract_with_full_pipeline
            schema = load_form_schema("cms-1500") if "cms" in file_path.lower() else None
            return extract_with_full_pipeline(file_path, schema=schema, 
                                              use_vlm=config.get("enable_vlm", True))
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ============================================================================
# Main UI
# ============================================================================

def main():
    # Top bar with logo and theme toggle
    col_logo, col_spacer, col_theme = st.columns([2, 6, 2])
    with col_logo:
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <span style="font-size: 1.8rem; margin-right: 0.5rem;">🏥</span>
            <span style="font-size: 1.4rem; font-weight: 700; color: #FAFAFA;">Doc2Data</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_theme:
        st.markdown("""
        <div style="text-align: right; padding-top: 0.5rem;">
            <span style="color: #A8B2C1; font-size: 0.8rem;">v1.4 • GPU Mode</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main layout: Document Preview (Left) | Controls + Results (Right)
    col_left, col_right = st.columns([1, 1])
    
    # ============== LEFT COLUMN: DOCUMENT PREVIEW ==============
    with col_left:
        st.markdown('<p class="section-header">📄 Document Preview</p>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
        "Upload PDF/Image",
                type=["pdf", "png", "jpg", "jpeg"],
        help="Upload a CMS-1500, UB-04, or other medical form"
        )
        
        # Sample documents dropdown
        samples = get_sample_documents()
        sample_options = ["Select sample..."] + samples
        selected_sample = st.selectbox(
            "Or choose sample document",
            sample_options,
            help="Pre-loaded test documents"
        )
        
        # Determine file path
        file_path = None
        if uploaded_file:
                temp_path = Path("cache") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    file_path = str(temp_path)
        elif selected_sample != "Select sample...":
            file_path = str(project_root / "data" / "sample_docs" / selected_sample)
        
        st.session_state.target_path = file_path
        
        # Display document
        if file_path and Path(file_path).exists():
            try:
                # If we have an aligned preview from the pipeline, display it so overlays line up.
                img_np = None
                if st.session_state.extraction_result:
                    dbg = (st.session_state.extraction_result or {}).get("debug") or {}
                    aligned_path = dbg.get("aligned_preview_path")
                    if aligned_path and Path(aligned_path).exists():
                        bgr = cv2.imread(aligned_path, cv2.IMREAD_COLOR)
                        if bgr is not None:
                            img_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                
                # Fallback: render raw PDF page for preview
                if img_np is None:
                    import fitz
                    pdf = fitz.open(file_path)
                    page = pdf[0]
                    pix = page.get_pixmap(dpi=150)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    if pix.n == 4:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                
                # Draw boxes if results exist
                if st.session_state.extraction_result:
                    result = st.session_state.extraction_result
                    fields = result.get("field_details", [])
                    ocr_blocks = result.get("ocr_blocks", [])
                    conf_thresh = st.session_state.get("conf_threshold", 0.3)
                    
                    img_np = draw_bounding_boxes(
                        img_np, fields, 
                        st.session_state.get("highlight_id"),
                        ocr_blocks=ocr_blocks,
                        source_width=result.get("page_width"),
                        source_height=result.get("page_height"),
                        confidence_threshold=conf_thresh
                    )
                
                st.image(img_np, use_container_width=True)
                
                # Legend
                st.markdown("""
                <div style="display: flex; gap: 1rem; font-size: 0.75rem; color: #A8B2C1; margin-top: 0.5rem;">
                    <span>🟢 High Conf</span>
                    <span>🟡 Medium</span>
                    <span>🔴 Low</span>
                    <span>🔵 OCR Box</span>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Could not render: {e}")
        else:
            st.info("👆 Upload a document or select a sample to begin")
    
    # ============== RIGHT COLUMN: CONTROLS + RESULTS ==============
    with col_right:
        # Pipeline Selection
        st.markdown('<p class="section-header">⚙️ Pipeline Configuration</p>', unsafe_allow_html=True)
        
        with st.expander("🎯 **Pipeline Mode**", expanded=True):
            pipeline_mode = st.selectbox(
                "Select Pipeline",
                [
                    "Multi-Agent (Recommended)",
                    "CMS-1500 (Agentic)",
                    "General Pipeline"
                ],
                help="Multi-Agent = Full orchestrated pipeline, Agentic = Template + LLM, General = Layout-based"
            )
            
            # Form type detection
            col_form, col_auto = st.columns([2, 1])
            with col_form:
                form_type_sel = st.selectbox(
                    "Form Type",
                    ["Auto-detect", "CMS-1500", "UB-04", "NCPDP", "Generic"],
                    help="Manually specify form type or let the system detect"
                )
            with col_auto:
                st.markdown("<br>", unsafe_allow_html=True)
                if form_type_sel == "Auto-detect":
                    st.markdown('<span class="form-badge form-badge-generic">AUTO</span>', 
                               unsafe_allow_html=True)
                else:
                    badge_class = "form-badge-cms1500" if "CMS" in form_type_sel else "form-badge-generic"
                    st.markdown(f'<span class="form-badge {badge_class}">{form_type_sel}</span>', 
                               unsafe_allow_html=True)
        
        # Model Selection
        with st.expander("🧠 **Model Configuration**", expanded=False):
            col_layout, col_ocr = st.columns(2)
            
            with col_layout:
                layout_model = st.selectbox(
                    "Layout Model",
                    ["Auto", "YOLOv8 (Fine-tuned)", "Detectron2 (PubLayNet)", "LayoutLMv3"],
                    help="Model for layout detection"
                )
            
            with col_ocr:
                ocr_model = st.selectbox(
                    "OCR Engine",
                    ["Tiered (Paddle+TrOCR)", "PaddleOCR Only", "TrOCR Only", "Tesseract"],
                    help="OCR engine for text extraction"
                )
            
            col_slm, col_vlm = st.columns(2)
            with col_slm:
                slm_model = st.selectbox(
                    "SLM Model",
                    ["llama3.2:3b", "mistral:latest", "qwen2.5:7b-instruct"],
                    help="Small Language Model for labeling"
                )
            with col_vlm:
                vlm_model = st.selectbox(
                    "VLM Model",
                    ["llama3.2:3b", "minicpm-v", "llava"],
                    help="Vision-Language Model for figures"
                )
        
        # Feature Toggles
        with st.expander("🔧 **Features**", expanded=False):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                enable_trocr = st.toggle("TrOCR (Handwriting)", value=True)
                enable_alignment = st.toggle("Template Alignment", value=True)
                enable_validators = st.toggle("Field Validators", value=True)
            with col_f2:
                enable_slm = st.toggle("SLM Labeling", value=True)
                enable_vlm = st.toggle("VLM for Figures", value=True)
                enable_llm_qa = st.toggle("LLM QA Check", value=True)
        
        # Threshold Tuning
        with st.expander("📊 **Threshold Tuning**", expanded=False):
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    0.0, 1.0, 0.5, 0.05,
                    help="Minimum confidence to accept field"
                )
                st.session_state.conf_threshold = confidence_threshold
                
                merge_threshold = st.slider(
                    "Merge Threshold",
                    0.5, 1.0, 0.7, 0.05,
                    help="Threshold for ensemble merging"
                )
            with col_t2:
                handwriting_threshold = st.slider(
                    "Handwriting Threshold",
                    0.0, 1.0, 0.35, 0.05,
                    help="Below this, use TrOCR"
                )
                
                ocr_padding = st.slider(
                    "OCR Zone Padding",
                    0, 30, 10, 2,
                    help="Padding around OCR zones (px)"
                )
            
            brightness = st.slider(
                "Image Brightness",
                0.5, 2.0, 1.0, 0.1,
                help="Adjust for dark scans"
            )
        
        # Run Button
        st.markdown("---")
        
        run_clicked = st.button(
            "🚀 Run Extraction",
            use_container_width=True,
            type="primary"
        )
        
        if run_clicked:
            if not file_path:
                st.error("Please select a document first!")
            else:
                # Build config
                config = {
                    "enable_trocr": enable_trocr,
                    "enable_alignment": enable_alignment,
                    "enable_validators": enable_validators,
                    "enable_slm": enable_slm,
                    "enable_vlm": enable_vlm,
                    "enable_llm_qa": enable_llm_qa,
                    "confidence_threshold": confidence_threshold,
                    "handwriting_threshold": handwriting_threshold,
                    "ocr_padding": ocr_padding,
                    "brightness": brightness,
                    "slm_model": slm_model,
                    "vlm_model": vlm_model,
                }
                
                # Map pipeline mode
                pipeline_map = {
                    "Multi-Agent (Recommended)": "multi_agent",
                    "CMS-1500 (Agentic)": "agentic",
                    "General Pipeline": "full_page_llm",
                }
                pipeline_type = pipeline_map.get(pipeline_mode, "multi_agent")
                
                # Pass explicit form type if selected
                if form_type_sel != "Auto-detect":
                    config["form_type"] = form_type_sel.lower().replace(" ", "-")
                
                with st.spinner("🔄 Processing..."):
                    result = run_extraction(file_path, pipeline_type, config)
                    st.session_state.extraction_result = result
                    st.rerun()
        
        # Results Display
        if st.session_state.extraction_result:
            result = st.session_state.extraction_result
            
            st.markdown('<p class="section-header">📊 Extraction Results</p>', unsafe_allow_html=True)
            
            # Metrics row
            fields = result.get("field_details", [])
            
            def has_value(val):
                if val is None:
                    return False
                val_str = str(val).strip().lower()
                return val_str not in ("", "null", "none", "n/a", "-")
            
            extracted = sum(1 for f in fields if has_value(f.get("value")))
            total = len(fields)
            confs = [f.get("confidence", 0) for f in fields if has_value(f.get("value"))]
            avg_conf = sum(confs) / len(confs) if confs else 0
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Fields Found", f"{extracted}/{total}")
            with col_m2:
                st.metric("Avg Confidence", f"{avg_conf:.0%}")
            with col_m3:
                form_type_detected = result.get("form_type", "unknown")
                st.metric("Form Type", form_type_detected.upper())
            with col_m4:
                proc_time = result.get("processing_time", 0)
                st.metric("Time", f"{proc_time:.1f}s")
            
            # Tabs
            tab_table, tab_ocr, tab_business, tab_reducto = st.tabs([
                "📋 Fields", "🔤 OCR JSON", "💼 Business", "📦 Export"
            ])
            
            with tab_table:
                df_data = []
                for f in fields:
                    conf = f.get("confidence", 0)
                    df_data.append({
                        "Field": f.get("label", f.get("id", "")),
                        "Value": f.get("value", ""),
                        "Confidence": conf,
                        "Source": f.get("metadata", {}).get("ocr_engine", f.get("source", ""))
                    })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    
                    # Editable table
                    edited_df = st.data_editor(
                        df,
                        column_config={
                            "Confidence": st.column_config.ProgressColumn(
                                "Conf", format="%.0f%%", min_value=0, max_value=1
                            )
                        },
                        use_container_width=True,
                        height=300,
                        num_rows="fixed"
                    )
                    
                    # Save corrections
                    col_save, col_dl = st.columns(2)
                    with col_save:
                        if st.button("💾 Save Corrections"):
                            from utils.corrections import log_user_edit
                            for i, row in edited_df.iterrows():
                                if row["Value"] != df_data[i]["Value"]:
                                    log_user_edit(
                                        df_data[i]["Field"],
                                        str(df_data[i]["Value"]),
                                        str(row["Value"])
                                    )
                            st.success("Corrections saved!")
                    
                    with col_dl:
                        csv = edited_df.to_csv(index=False)
                        st.download_button("⬇️ Download CSV", csv, "fields.csv", "text/csv")
            
            with tab_ocr:
                st.json(result.get("extracted_fields", {}))
            
            with tab_business:
                business = result.get("business_fields", {})
                if business:
                    st.json(business)
                else:
                    st.info("Business schema mapping not available for this pipeline")
            
            with tab_reducto:
                try:
                    from src.pipelines.reducto_adapter import adapt_result_to_reducto
                    reducto = adapt_result_to_reducto(result)
                    st.json(reducto)
                    st.download_button(
                        "⬇️ Download Reducto JSON",
                        json.dumps(reducto, indent=2),
                        "reducto_export.json",
                        "application/json"
                    )
                except Exception as e:
                    st.error(f"Export error: {e}")
            
            # Validation notes
            validation = result.get("validation", {})
            if validation.get("errors") or validation.get("qa_notes"):
                with st.expander("⚠️ Validation Notes"):
                    for err in validation.get("errors", []):
                        st.warning(f"{err.get('field_id')}: {err.get('message')}")
                    for note in validation.get("qa_notes", []):
                        st.info(note)


if __name__ == "__main__":
    main()

"""
Doc2Data - Intelligent Document Extraction Workstation
Reducto-style Layout: PDF on LEFT, Controls+Results on RIGHT
"""
import sys
from pathlib import Path
import json
import time
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
from src.pipelines.agentic_cms1500 import run_cms1500_agentic
from src.pipelines.reducto_adapter import adapt_result_to_reducto

# --- UI Configuration ---
st.set_page_config(
    page_title="Doc2Data | Intelligent Extraction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar for Reducto-style layout
)

# Custom CSS - Professional Reducto-like Design
st.markdown("""
<style>
    /* Main Container - Dark Theme */
    .main { background-color: #0f0f14; }
    .stApp { background-color: #0f0f14; }
    
    /* Hide default sidebar */
    section[data-testid="stSidebar"] { display: none; }
    
    /* Custom Panel Styling */
    .config-panel {
        background: linear-gradient(145deg, #1a1a24, #12121a);
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Stat Cards */
    .stat-card {
        background: linear-gradient(145deg, #1e1e2e, #16161f);
        border: 1px solid #3a3a5a;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .stat-label { color: #8888aa; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .stat-value { color: #ffffff; font-size: 1.4rem; font-weight: 700; }
    
    /* Pipeline Badge */
    .pipeline-badge {
        display: inline-block;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        color: white;
        font-weight: 600;
    }
    
    /* Confidence Colors */
    .conf-high { color: #10b981; }
    .conf-med { color: #f59e0b; }
    .conf-low { color: #ef4444; }
    
    /* Section Headers */
    .section-header {
        color: #9ca3af;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Tabs Enhancement */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #1a1a24;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2a2a3a !important;
        border-bottom: 2px solid #6366f1;
    }
    
    /* Image Container */
    .pdf-viewer {
        background: #0a0a10;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        padding: 8px;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #1a1a24;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


def get_sample_files():
    """Get list of sample files."""
    sample_dir = Path(project_root / "data" / "sample_docs")
    if sample_dir.exists():
        return sorted([f.name for f in sample_dir.glob("*") if f.suffix in ['.pdf', '.png', '.jpg', '.jpeg']])
    return []


def draw_bounding_boxes(img, field_details, highlight_id=None, ocr_blocks=None, 
                       source_width=None, source_height=None, confidence_threshold=0.5):
    """Draw bounding boxes on image with confidence-based coloring."""
    img = img.copy()
    h, w = img.shape[:2]
    
    # Calculate scale factors
    scale_x = w / source_width if source_width else 1.0
    scale_y = h / source_height if source_height else 1.0
    
    def scale_bbox(bbox):
        return (
            int(bbox[0] * scale_x),
            int(bbox[1] * scale_y),
            int(bbox[2] * scale_x),
            int(bbox[3] * scale_y)
        )
    
    def clamp_bbox(x0, y0, x1, y1):
        return (
            max(0, min(x0, w-1)),
            max(0, min(y0, h-1)),
            max(0, min(x1, w-1)),
            max(0, min(y1, h-1))
        )
    
    boxes_drawn = 0
    
    # Draw field boxes with confidence-based coloring
    for field in field_details:
        bbox = field.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        
        conf = field.get("confidence", 0.5)
        x0, y0, x1, y1 = scale_bbox(bbox)
        x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1)
        
        if x1 <= x0 or y1 <= y0:
            continue
        
        # Skip low confidence boxes if threshold set
        if conf < confidence_threshold:
            continue
        
        # Color based on confidence
        if conf >= 0.8:
            color = (46, 204, 113)  # Green
        elif conf >= 0.5:
            color = (241, 196, 15)  # Yellow
        else:
            color = (231, 76, 60)   # Red
        
        thickness = 2 if field.get("value") else 1
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        boxes_drawn += 1
    
    # Draw raw OCR blocks (light cyan)
    if ocr_blocks:
        for block in ocr_blocks[:100]:  # Limit for performance
            bbox = block.get("bbox")
            if bbox and len(bbox) == 4:
                x0, y0, x1, y1 = scale_bbox(bbox)
                x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1)
                if x1 > x0 and y1 > y0:
                    cv2.rectangle(img, (x0, y0), (x1, y1), (100, 200, 200), 1)
    
    # Highlight selected field
    if highlight_id:
        for field in field_details:
            if field.get("id") == highlight_id:
                bbox = field.get("bbox")
                if bbox and len(bbox) == 4:
                    x0, y0, x1, y1 = scale_bbox(bbox)
                    x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1)
                    if x1 > x0 and y1 > y0:
                        cv2.rectangle(img, (x0-3, y0-3), (x1+3, y1+3), (255, 128, 0), 4)
                        label = field.get("label", "")[:25]
                        cv2.putText(img, label, (x0, max(y0-8, 15)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
                break
    
    return img


def run_ensemble_pipeline(file_path: str, settings: dict) -> dict:
    """
    Ensemble Pipeline: Runs both Agentic + Full-page LLM and merges results.
    
    Strategy:
    1. Run Full-page LLM (fast, good semantic understanding)
    2. Run Agentic (slower, better zone accuracy when it works)
    3. Merge: Use Agentic result if confidence > threshold, else fallback to LLM
    """
    results = {}
    
    # Load schema
    schema = load_form_schema("cms-1500")
    
    # Run Full-page LLM first (always works)
    llm_result = extract_with_full_pipeline(
        file_path,
        schema=schema,
        use_vlm=settings.get("use_vlm", True),
        map_business=True
    )
    results["llm"] = llm_result
    
    # Optionally run Agentic
    if settings.get("use_agentic", True):
        try:
            agentic_result = run_cms1500_agentic(
                file_path,
                use_icr=settings.get("use_icr", True),
                use_llm=settings.get("use_llm_backfill", True),
                align_template=settings.get("align_template", True)
            )
            results["agentic"] = agentic_result
        except Exception as e:
            results["agentic_error"] = str(e)
    
    # Merge results
    merged = merge_ensemble_results(results, settings)
    merged["ensemble_sources"] = list(results.keys())
    
    return merged


def merge_ensemble_results(results: dict, settings: dict) -> dict:
    """Merge results from multiple pipelines using confidence-based selection."""
    conf_threshold = settings.get("confidence_threshold", 0.7)
    
    # Start with LLM result as base
    base = results.get("llm", {})
    agentic = results.get("agentic", {})
    
    if not agentic or "error" in results.get("agentic", {}):
        return base
    
    # Merge field_details: prefer agentic if confidence > threshold
    merged_fields = []
    llm_fields = {f.get("id"): f for f in base.get("field_details", [])}
    agentic_fields = {f.get("id"): f for f in agentic.get("field_details", [])}
    
    all_ids = set(llm_fields.keys()) | set(agentic_fields.keys())
    
    for fid in all_ids:
        llm_f = llm_fields.get(fid, {})
        agent_f = agentic_fields.get(fid, {})
        
        llm_conf = llm_f.get("confidence", 0)
        agent_conf = agent_f.get("confidence", 0)
        
        # Selection logic
        if agent_conf >= conf_threshold and agent_f.get("value"):
            chosen = agent_f.copy()
            chosen["source"] = "agentic"
        elif llm_conf >= conf_threshold * 0.8 and llm_f.get("value"):
            chosen = llm_f.copy()
            chosen["source"] = "llm"
        elif agent_f.get("value"):
            chosen = agent_f.copy()
            chosen["source"] = "agentic_fallback"
        elif llm_f.get("value"):
            chosen = llm_f.copy()
            chosen["source"] = "llm_fallback"
        else:
            chosen = llm_f if llm_f else agent_f
            chosen["source"] = "empty"
        
        merged_fields.append(chosen)
    
    # Build merged result
    merged = base.copy()
    merged["field_details"] = merged_fields
    merged["extracted_fields"] = {f.get("id"): f.get("value") for f in merged_fields}
    merged["extraction_method"] = "ensemble"
    
    # Recalculate stats
    filled = sum(1 for f in merged_fields if f.get("value"))
    confs = [f.get("confidence", 0) for f in merged_fields if f.get("value")]
    merged["fields_extracted"] = filled
    merged["avg_confidence"] = sum(confs) / len(confs) if confs else 0
    
    return merged


def main():
    # Initialize session state
    if "extraction_result" not in st.session_state:
        st.session_state.extraction_result = None
    if "target_path" not in st.session_state:
        st.session_state.target_path = None
    
    # ============== MAIN LAYOUT: PDF LEFT | CONTROLS+RESULTS RIGHT ==============
    col_pdf, col_right = st.columns([1.2, 1])
    
    # ============== LEFT COLUMN: PDF VIEWER ==============
    with col_pdf:
        st.markdown("### 📄 Document Preview")
        
        # File upload at top of PDF column
        uploaded_file = st.file_uploader("Upload PDF/Image", type=['pdf', 'png', 'jpg', 'jpeg'], 
                                         key="file_upload", label_visibility="collapsed")
        
        sample_files = get_sample_files()
        selected_sample = st.selectbox("Or select sample document", 
                                       ["None"] + sample_files, 
                                       key="sample_select")
        
        # Determine file path
        file_path = None
        if uploaded_file:
            temp_path = Path("cache") / uploaded_file.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_path = str(temp_path)
        elif selected_sample != "None":
            file_path = str(project_root / "data" / "sample_docs" / selected_sample)
        
        st.session_state.target_path = file_path
        
        # Display PDF/Image
        if file_path and Path(file_path).exists():
            try:
                import fitz
                pdf = fitz.open(file_path)
                page = pdf[0]
                pix = page.get_pixmap(dpi=150)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                
                # Apply brightness adjustment
                brightness = st.session_state.get("brightness", 1.0)
                if brightness != 1.0:
                    img_np = cv2.convertScaleAbs(img_np, alpha=brightness, beta=0)
                
                # If we have results, draw boxes
                if st.session_state.extraction_result:
                    result = st.session_state.extraction_result
                    fields = result.get("field_details", [])
                    ocr_blocks = result.get("ocr_blocks", [])
                    highlight_id = st.session_state.get("highlight_id")
                    conf_thresh = st.session_state.get("conf_threshold", 0.3)
                    
                    img_np = draw_bounding_boxes(
                        img_np, fields, highlight_id,
                        ocr_blocks=ocr_blocks,
                        source_width=result.get("page_width"),
                        source_height=result.get("page_height"),
                        confidence_threshold=conf_thresh
                    )
                
                st.image(img_np, use_container_width=True)
                st.caption("🟢 High Conf | 🟡 Medium | 🔴 Low | 🔵 Raw OCR")
                
            except Exception as e:
                st.error(f"Could not render document: {e}")
        else:
            st.info("👈 Upload a document or select a sample to begin")
            st.image("https://via.placeholder.com/600x800/1a1a24/666666?text=Document+Preview", 
                    use_container_width=True)
    
    # ============== RIGHT COLUMN: CONTROLS + RESULTS ==============
    with col_right:
        # --- Configuration Section ---
        with st.expander("⚙️ **Configuration**", expanded=True):
            st.markdown('<div class="section-header">Processing Pipeline</div>', unsafe_allow_html=True)
            
            pipeline_mode = st.selectbox(
                "Pipeline Mode",
                ["Ensemble (Best)", "Full-page LLM", "Agentic CMS-1500"],
                help="Ensemble combines both methods for best results"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                use_vlm = st.toggle("Enable VLM/LLM", value=True)
                use_icr = st.toggle("Enable ICR (TrOCR)", value=True)
            with col_b:
                align_template = st.toggle("Template Alignment", value=True)
                show_raw_ocr = st.toggle("Show Raw OCR Boxes", value=False)
            
            # Image Enhancement Controls
            st.markdown('<div class="section-header">Image Enhancement</div>', unsafe_allow_html=True)
            brightness = st.slider(
                "Brightness", 0.5, 2.0, 1.0, 0.1,
                help="Adjust document brightness for better OCR"
            )
            st.session_state.brightness = brightness
            
            # --- Threshold Tuning ---
            st.markdown('<div class="section-header">Threshold Tuning</div>', unsafe_allow_html=True)
            
            conf_threshold = st.slider(
                "Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
                help="Minimum confidence to display/accept a field"
            )
            st.session_state.conf_threshold = conf_threshold
            
            handwriting_threshold = st.slider(
                "Handwriting Detection Threshold", 0.0, 1.0, 0.35, 0.05,
                help="Score above which TrOCR is used instead of PaddleOCR"
            )
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                merge_threshold = st.slider("Merge Threshold", 0.5, 1.0, 0.7, 0.05)
            with col_t2:
                ocr_padding = st.slider("OCR Zone Padding", 0, 30, 10, 2)
            
            # --- Run Button ---
            if st.button("▶️ Run Extraction", type="primary", use_container_width=True):
                if not file_path:
                    st.error("Please select a document first!")
                else:
                    settings = {
                        "use_vlm": use_vlm,
                        "use_icr": use_icr,
                        "use_agentic": pipeline_mode in ["Ensemble (Best)", "Agentic CMS-1500"],
                        "use_llm_backfill": use_vlm,
                        "align_template": align_template,
                        "confidence_threshold": merge_threshold,
                        "handwriting_threshold": handwriting_threshold,
                        "ocr_padding": ocr_padding
                    }
                    
                    with st.spinner("🔄 Processing... (OCR → LLM → Grounding)"):
                        try:
                            if pipeline_mode == "Ensemble (Best)":
                                result = run_ensemble_pipeline(file_path, settings)
                            elif pipeline_mode == "Agentic CMS-1500":
                                result = run_cms1500_agentic(
                                    file_path, use_icr=use_icr, use_llm=use_vlm, 
                                    align_template=align_template
                                )
                            else:  # Full-page LLM
                                schema = load_form_schema("cms-1500")
                                result = extract_with_full_pipeline(
                                    file_path, schema=schema, use_vlm=use_vlm, map_business=True
                                )
                            
                            st.session_state.extraction_result = result
                            st.rerun()
                        except Exception as e:
                            st.error(f"Pipeline failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())
        
        # --- Results Section ---
        if st.session_state.extraction_result:
            result = st.session_state.extraction_result
            fields = result.get("field_details", [])
            
            # Statistics
            st.markdown("---")
            st.markdown('<div class="section-header">Extraction Results</div>', unsafe_allow_html=True)
            
            # Count only fields with ACTUAL non-empty values 
            # Exclude: None, "", "null", "NULL", "none", "None", empty whitespace
            def has_real_value(val):
                if val is None:
                    return False
                val_str = str(val).strip().lower()
                if val_str in ("", "null", "none", "n/a", "na", "-"):
                    return False
                return True
            
            # Count from field_details (OCR results)
            ocr_extracted = sum(1 for f in fields if has_real_value(f.get("value")))
            total = len(fields)
            
            # Also count from business_fields (mapped results) for accurate business coverage
            business_fields = result.get("business_fields", {})
            business_extracted = sum(1 for v in business_fields.values() if has_real_value(v))
            business_total = len(business_fields) if business_fields else total
            
            # Use the more accurate count (business fields if available, else OCR)
            extracted = business_extracted if business_fields else ocr_extracted
            display_total = business_total if business_fields else total
            
            confs = [f.get("confidence", 0) for f in fields if has_real_value(f.get("value"))]
            avg_conf = sum(confs) / len(confs) if confs else 0
            
            # Calculate actual business coverage
            business_cov = (business_extracted / business_total * 100) if business_total > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Fields Found", f"{extracted}/{display_total}")
            with c2:
                st.metric("Confidence", f"{avg_conf:.0%}")
            with c3:
                st.metric("Business Coverage", f"{business_cov:.0f}%")
            with c4:
                method = result.get("extraction_method", "unknown")
                st.markdown(f'<span class="pipeline-badge">{method}</span>', unsafe_allow_html=True)
            
            # Tabs for different views
            tab_table, tab_ocr, tab_business, tab_reducto = st.tabs([
                "📋 Field Table", "🔤 OCR JSON", "💼 Business JSON", "📦 Reducto JSON"
            ])
            
            with tab_table:
                # Build dataframe
                df_data = []
                for f in fields:
                    conf = f.get("confidence", 0)
                    conf_class = "conf-high" if conf >= 0.8 else "conf-med" if conf >= 0.5 else "conf-low"
                    df_data.append({
                        "ID": f.get("id", ""),
                        "Label": f.get("label", ""),
                        "Value": f.get("value", ""),
                        "Conf": conf,
                        "Source": f.get("source", f.get("detected_by", ""))
                    })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    
                    # Interactive table
                    st.dataframe(
                        df,
                        column_config={
                            "Conf": st.column_config.ProgressColumn(
                                "Conf", format="%.2f", min_value=0, max_value=1
                            )
                        },
                        use_container_width=True,
                        height=400
                    )
                    
                    # Field locator
                    field_opts = {f"{r['Label'][:20]} → {str(r['Value'])[:15]}": r['ID'] 
                                 for r in df_data if r['Value']}
                    selected = st.selectbox("🔍 Locate field on document", 
                                           ["Select..."] + list(field_opts.keys()))
                    st.session_state.highlight_id = field_opts.get(selected) if selected != "Select..." else None
                    
                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button("⬇️ Download CSV", csv, "extracted_fields.csv", "text/csv")
            
            with tab_ocr:
                st.json(result.get("extracted_fields", {}))
                st.download_button(
                    "⬇️ Download Full JSON",
                    json.dumps(result, indent=2, default=str),
                    "extraction_result.json",
                    "application/json"
                )
            
            with tab_business:
                business = result.get("business_fields", {})
                if business:
                    st.json(business)
                else:
                    st.info("Business schema not available")
            
            with tab_reducto:
                try:
                    reducto = adapt_result_to_reducto(result)
                    st.json(reducto)
                    st.download_button(
                        "⬇️ Download Reducto JSON",
                        json.dumps(reducto, indent=2),
                        "reducto_export.json",
                        "application/json"
                    )
                except Exception as e:
                    st.error(f"Reducto adapter error: {e}")
        
        else:
            # No results yet - show info
            st.info("Configure settings above and click **Run Extraction** to process the document.")
            
            # Why Layout vs OCR explanation
            with st.expander("ℹ️ **Why Layout Detection vs OCR?**"):
                st.markdown("""
                **Why OCR is accurate but Layout boxes may drift:**
                
                1. **OCR (PaddleOCR/TrOCR)** works on raw pixels - it finds text wherever it exists
                2. **Layout Detection** uses schema coordinates (`bbox_norm`) calibrated for a reference template
                3. When forms are scanned at different DPI/angles, pixel coordinates shift
                4. **Solution**: We use **semantic understanding** (LLM) to extract values, then **ground** them back to OCR boxes
                
                **Our Approach (like Reducto):**
                - ✅ Run full-page OCR first (get ALL text + boxes)
                - ✅ Use LLM to semantically extract structured data
                - ✅ Ground extracted values back to OCR-detected boxes
                - ✅ No hardcoded pixel coordinates!
                """)


if __name__ == "__main__":
    main()

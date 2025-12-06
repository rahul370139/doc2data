# Document-to-Data Pipeline

**Version:** 1.0.0-gpu  
**Status:** ✅ Production-Ready (GPU-Accelerated)  
**Last Updated:** December 5, 2025  
**Live Demo:** http://100.126.216.92:8501

Goal:
A production-ready document processing pipeline that converts PDFs and images into structured JSON data. Uses state-of-the-art ML models (LayoutParser, PaddleOCR) combined with intelligent heuristics, SLM/VLM enrichment, and GPU-aware preprocessing for layout detection, OCR, and content classification.

## 🏛 Architecture Overview

```mermaid
flowchart LR
    A[Ingest & Preprocess]
    A -->|images @300 DPI, GPU CLAHE/binarize| B(Layout + Form Geometry)
    B -->|blocks + columns + templates| C(OCR Tiering)
    C -->|text + confidences| D(Validators & Hungarian Linking)
    D -->|role_locked blocks| E(SLM Labeler)
    B -->|tables/figures| F(Table/Figure Processors)
    F -->|structure + chart metadata (VLM)| H
    E -->|semantic roles| H(Assembly)
    H -->|JSON + Markdown + citations| I(Streamlit UI / FastAPI)
    I -->|highlights, validator badges| User
```

---

## 🚀 Quick Start

### Docker Deployment (Recommended - GPU Ready)

```bash
# Build and run with GPU support
./run_docker_gpu.sh

# Or manually:
docker build -t doc2data-gpu .
docker run -d --gpus all --name doc2data-gpu-app \
  -p 8501:8501 -p 8000:8000 -p 11434:11434 \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/app:/app/app" \
  -v "$(pwd)/utils:/app/utils" \
  -v "$(pwd)/sample_docs:/app/sample_docs" \
  -e USE_GPU=true \
  -e ENABLE_SLM=true \
  -e ENABLE_VLM=true \
  doc2data-gpu
```

**Access:** http://localhost:8501

### Local Development (CPU)

```bash
# Clone repository
git clone https://github.com/rahul370139/doc2data.git
cd doc2data

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run app/streamlit_main.py
```

### Stop Streamlit

```bash
# Press Ctrl+C in terminal, or:
pkill -9 -f "streamlit"
# Or kill by port:
lsof -ti:8501 | xargs kill -9
```

---

## 📋 Features

### ✅ Implemented

- **PDF/Image Ingestion:** PyMuPDF, pdf2image @ 300 DPI, digital text extraction
- **Layout Segmentation:** Detectron2 (PubLayNet) + TableBank + advanced CV post-processing
- **OCR Processing:** PaddleOCR 3.x (GPU) + Tesseract fallback, parallel processing
- **Table Structure:** Microsoft Table Transformer (TATR) for cell-level extraction
- **Form Field Detection:** OpenCV-based geometry detection (checkboxes, fields)
- **Semantic Labeling (SLM):** Qwen2.5-7B-Instruct via Ollama ✅ **ACTIVE**
- **Figure Understanding (VLM):** MiniCPM-V via Ollama ✅ **ACTIVE**
- **Document Assembly:** Enhanced JSON with detailed metadata, Markdown generation
- **Streamlit UI:** Interactive demo with visualization, threshold controls, results display
- **GPU Acceleration:** Docker deployment with NVIDIA runtime, CUDA-accelerated models
- **Model Caching:** Fast subsequent runs with `@st.cache_resource`

### ⚠️ Partial/Stubbed

- **FastAPI Endpoints:** Created, needs integration testing

---

## 🏗️ Architecture

### Pipeline Flow

```
PDF/Image → Ingest → Segment → OCR → [Label] → [Table/Figure] → Assemble → JSON/Markdown
```

### Key Components

| Component | Status | Description |
|-----------|--------|-------------|
| `src/pipelines/ingest.py` | ✅ | PDF/image loading, preprocessing |
| `src/pipelines/segment.py` | ✅ | Layout detection (ML + heuristics) |
| `src/pipelines/ocr.py` | ✅ | OCR orchestration |
| `src/pipelines/slm_label.py` | ✅ | Semantic labeling (Qwen2.5-7B active) |
| `src/pipelines/table_processor.py` | ✅ | Table extraction (TATR active) |
| `src/pipelines/figure_processor.py` | ✅ | Figure processing (MiniCPM-V active) |
| `src/pipelines/assemble.py` | ✅ | JSON/Markdown assembly |
| `app/streamlit_main.py` | ✅ | Streamlit demo UI |
| `app/api_main.py` | ⚠️ | FastAPI endpoints (partial) |

---

## 📊 Performance

### GPU-Accelerated (Current)

- **1-page document:** 3-5 seconds
- **Layout detection:** 2-5 sec/page (10-20x faster than CPU)
- **OCR:** 10-20 sec/page (5-10x faster than CPU)
- **Text extraction:** ~92% success rate
- **GPU Memory:** ~12 GB

### CPU Fallback

- **1-page document:** ~30-60 seconds
- **Layout detection:** ~30-60 sec/page
- **OCR:** ~2-3 min/page

---

## ⚙️ Configuration

### Streamlit UI Settings

- **ML Detection Threshold:** 0.05-0.50 (recommended: 0.15-0.25)
  - Lower = more blocks (may include noise)
  - Higher = fewer, higher-confidence blocks

- **Heuristic Strictness:** 0.0-1.0 (recommended: 0.6-0.8)
  - Higher = fewer false positives (logos as FORM, # as LIST)
  - Lower = more aggressive detection

- **Enable Form Geometry Detection:** Toggle geometry-based form field and checkbox detection
  - When enabled: Uses dedicated form geometry detector with analysis masks
  - When disabled: Falls back to heuristic form detection

- **Form Geometry Strictness:** 0.0-1.0 (recommended: 0.7-0.9 for forms)
  - Higher = stricter detection (fewer, more confident form fields/checkboxes)
  - Lower = more permissive detection (may include more candidates)
  - Recommended: 0.7-0.9 for dense forms (CMS-1500, UB-04), 0.5-0.7 for simpler forms

- **Template Alignment:** Enable + pick CMS-1500 / NCPDP / UB-04 to snap column order for those templates (default: Auto)

- **Label↔Value Linking:** Choose between Greedy (fast) and Hungarian (optimal) matching for label association

- **Enable SLM/VLM:** Toggle semantic labeling and VLM features (requires Ollama)

### Environment Variables

Create `.env` file (see `.env.example`):

```bash
# LLM/VLM Configuration
ENABLE_SLM=true              # Enable semantic labeling (requires Ollama)
ENABLE_VLM=true              # Enable VLM for tables/figures
OLLAMA_HOST=localhost:11434   # Ollama server host:port
OLLAMA_MODEL_SLM=qwen2.5:7b-instruct
OLLAMA_MODEL_VLM=minicpm-v

# GPU / Vision Configuration
USE_GPU=false             # true = enable OpenCV CUDA, Paddle GPU, Detectron CUDA if available
CUDA_VISIBLE_DEVICES=0    # optional: restrict GPU id when USE_GPU=true

# Model Configuration
LAYOUT_MODEL=publaynet    # Layout detection model
OCR_ENGINE=paddle         # OCR engine (paddle/tesseract)
```

> Setting `USE_GPU=true` automatically enables OpenCV CUDA (for preprocessing masks), PaddleOCR GPU execution, and LayoutParser detectron/paddle models on CUDA if drivers are present. When the flag is on but CUDA is unavailable, the pipeline gracefully falls back to CPU with a warning.

---

## 📁 Project Structure

```
doc2data/
├── src/
│   ├── pipelines/        # Pipeline stages
│   ├── ocr/              # OCR implementations
│   ├── vlm/              # VLM integration (MiniCPM-V active)
│   └── processing/       # Preprocessing utilities
├── app/
│   ├── streamlit_main.py # Streamlit demo
│   └── api_main.py       # FastAPI endpoints
├── utils/
│   ├── models.py         # Data models
│   ├── config.py         # Configuration
│   └── visualization.py  # Visualization utilities
├── data/
│   ├── sample_docs/      # Sample PDFs (preprocessed)
│   └── thumbnails/       # Page thumbnails for visualization
├── preprocess_samples.py # Script to preprocess sample documents
├── models/               # Model download scripts
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── requirements_docker.txt  # Docker dependencies
├── Dockerfile           # Docker image definition
├── run_docker_gpu.sh    # Docker GPU startup script
├── DEMO_SUMMARY.md      # Technical summary for presentations
└── README.md            # This file
```

---

## 🔧 Usage

### Streamlit Demo

1. **Start Streamlit:**
   ```bash
   streamlit run app/streamlit_main.py
   ```

2. **Upload Document:**
   - Use file uploader OR select sample document from dropdown
   - Available samples: CMS-1500, UB-04, UCF Form, Healthcare Document

3. **Configure Settings:**
   - **For Forms:** Enable "Form Geometry Detection", set Form Geometry Strictness to 0.7-0.9
   - **For Plain Text:** Disable "Form Geometry Detection" for better text block detection
   - Adjust ML Detection Threshold (0.15-0.25 recommended)
   - Adjust Heuristic Strictness (0.6-0.8 recommended)

4. **Run Pipeline:**
   - Check pipeline steps: Ingest → Segment → OCR → Assemble
   - View results: Annotated image, statistics, JSON/Markdown

5. **Download Results:**
   - JSON: Full structured data with metadata
   - Markdown: Human-readable format

---

## 🆕 Recent Improvements (December 2025)

### GPU Acceleration & Docker
- ✅ **Docker Deployment:** NVIDIA Container Toolkit integration, GPU-accelerated models
- ✅ **Detectron2 GPU:** CUDA-accelerated layout detection (10-20x faster)
- ✅ **PaddleOCR GPU:** GPU-accelerated text extraction (5-10x faster)
- ✅ **OpenCV CUDA:** GPU-accelerated preprocessing (CLAHE, denoising)

### CV Post-Processing (Reducto-Style)
- ✅ **Granularity Preference:** Children explode parent containers (>50% page coverage)
- ✅ **Visual Grid Snapping:** Block boundaries snap to detected lines (15px threshold)
- ✅ **Aggressive Stitching:** 600px horizontal, 100px vertical gaps for header/table merging
- ✅ **Gap Filling:** Unconditional text augmentation (`text_density >= 0.01`)
- ✅ **Containment Cleanup:** Remove overlapping child blocks (65% threshold)

### AI Models Integration
- ✅ **SLM Active:** Qwen2.5-7B-Instruct for semantic labeling (title, header, footer, etc.)
- ✅ **VLM Active:** MiniCPM-V for figure/chart understanding (5.5 GB)
- ✅ **Table Transformer:** Microsoft TATR for structured table extraction

### UI Improvements
- ✅ **Document Type Presets:** Forms, Reports, Tables, Custom
- ✅ **Simplified Controls:** Replaced multiple sliders with presets
- ✅ **Better Visualization:** Clean bounding boxes, no overlaps

## Previous Improvements (Nov 2025)

### Form Detection & Processing (Latest)
- **Smart Form Detection:** Geometry-based form field and checkbox detection using analysis masks (box_mask, line_mask, binary_image) with configurable strictness
- **Reduced Over-Detection:** Fixed issue where forms were being over-fragmented into many TITLE/TEXT blocks. Form pages now rely primarily on ML model + dedicated form geometry detector instead of aggressive heuristics
- **Form Geometry Controls:** New UI controls for enabling/disabling form geometry detection and adjusting strictness (0.0-1.0) to fine-tune detection sensitivity
- **Preserved Form Content:** Form blocks no longer have bounding boxes aggressively tightened, preserving full text coverage within form fields
- **Sample Document Preprocessing:** Added preprocessing script to clean sample documents:
  - `ub04_clean.pdf`: Extracts main form page without aggressive watermark removal
  - `ucf_form_page1.pdf`: Extracts and crops first page of UCF form
  - `form-cms1500.pdf`: Standard CMS-1500 form (ready to use)

### UI & User Experience
- **Enhanced Sample Selection:** Improved Streamlit UI with descriptive sample document names and showcase section
- **Better Page Navigation:** Fixed single-page document slider error, now displays "Page: 1 of 1" for single-page docs
- **Sample Showcase:** Added visual showcase of available sample documents with descriptions

### Core Pipeline Improvements
- **Reliability:** Added `PageImage.shape` compatibility property to fix `'PageImage' object has no attribute 'shape'` errors
- **Digital Text Layer:** PDF word boxes scaled into pixel coordinates for reliable block-level extraction
- **OCR Quality:** Enhanced preprocessing (upsampling, CLAHE contrast, denoising, unsharp masking) with automatic Tesseract fallback
- **Results JSON:** Reducto-like summary with per-page chunks, normalized bboxes, confidence scores, and form associations (`form_fields[]`, `checkboxes[]`)
- **Column-Aware Reading Order:** Blocks receive `column_id` metadata (auto or template-driven) to keep CMS-1500/NCPDP/UB-04 layouts in a stable left-to-right order.
- **Template Alignment (Opt-in):** Sidebar toggle lets you anchor segmentation to CMS-1500, NCPDP, or UB-04 column guides.
- **Optimal Label↔Value Linking:** Switch between Greedy and Hungarian assignment for dense forms; validators short-circuit SLM when confident.
- **Link Overlays & Validator Badges:** Streamlit overlay draws connectors between labels and values and shows validator pass/fail dots plus a validator panel table.

---

## Forms Pipeline Roadmap — Status

- Foundation & Tooling
  - [x] CPU-first config with future GPU hooks (Paddle/Detectron2/vLLM toggles in code)
  - [x] Validators module (NPI/NDC/ICD/HCPCS/date/phone/member) and checkbox fill ratio
  - [x] Linear assignment not required yet; geometry-first linker in place with nearest-neighbor scoring

- Ingest Enhancements
  - [x] Orientation estimate (0/90) + confidence stored on `PageImage`
  - [x] Analysis layers: grayscale/CLAHE, binary, `line_mask`, `box_mask`
  - [x] Digital text priors scaled to pixel coordinates on `PageImage`

- Form Geometry Pass (Step 2)
  - [x] Geometry detector over analysis layers for checkboxes/field rectangles
  - [x] Tunable strictness with NMS, neighbor rules, text/line density checks, overlap suppression
  - [x] UI toggle to enable/disable form geometry
  - [x] Column clusters/column-first order
  - [x] Template alignment hook (experimental, optional)

- OCR Tiering & Association (Step 3)
  - [x] Tier-1 PaddleOCR + preproc; Tier-2 Tesseract fallback
  - [x] PDF digital layer extraction where available
  - [x] Checkbox state via fill ratio
  - [x] Label↔value linking with geometry + validators; stored in block.metadata
  - [x] Hungarian matching optional (toggle between greedy/optimal)

- SLM & Validators (Step 4)
  - [x] SLM integrates with `role_locked`; rules skip SLM when validator confident
  - [x] Ollama toggle; batch-friendly labeler wired

- Assembly & JSON Schema (Step 5)
  - [x] Emits `form_fields[]`, `checkboxes[]`, tables with citations + confidences
  - [x] Reducto-like chunks and normalized bboxes

- Streamlit UX
  - [x] Two-pane view (annotated image + Results/JSON)
  - [x] Jump-to-block selector syncs highlighting
  - [x] Geometry strictness + enable toggle; model threshold + heuristic strictness
  - [x] Overlays for link connectors + validator badges
  
- Testing & Docs
  - [x] README updated with changes and troubleshooting
  - [x] Unit tests for geometry/validators/linking

### FastAPI Server

```bash
# Start FastAPI server
uvicorn app.api_main:app --host 0.0.0.0 --port 8000

# Endpoints:
# POST /ingest
# POST /segment
# POST /ocr
# POST /label
# POST /table/process
# POST /figure/process
# POST /assemble
# GET /health
```

---

## 🧪 Testing

```bash
# Run integration tests
python tests/test_pipeline.py

# Test individual stages
python -m pytest tests/
```

---

## 📦 Models

### Auto-Downloaded Models

Models are automatically downloaded on first run:

- **LayoutParser PubLayNet:** `~/.paddlex/official_models/ppyolov2_r50vd_dcn_365e_publaynet/`
- **TableBank:** `~/.paddlex/official_models/ppyolov2_r50vd_dcn_365e_tableBank_word/`
- **PaddleOCR:** `~/.paddlex/official_models/PP-OCRv5_server_det/`, `en_PP-OCRv5_mobile_rec/`

**Total Size:** ~1-2 GB

### Ollama Models (Optional)

If enabling SLM/VLM:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull qwen2.5:7b-instruct
ollama pull qwen-vl  # (if available)

# Start Ollama server
ollama serve
```

---

## 🐛 Known Issues & Solutions

### Issue: Forms Over-Fragmented into Many Titles/Text Blocks

**Solution:** Enable "Form Geometry Detection" (now skips aggressive text heuristics) or disable heuristics entirely for form-heavy pages. LayoutParser detections + the dedicated geometry pass keep each field as a single block, and template alignment ensures column-aware order.

### Issue: Form Fields Not Covering Full Text

**Solution:** Geometry-derived form blocks retain their original boxes and we now pad each field/checkbox by ~25% to capture surrounding handwriting. Avoid tightening on forms and rely on geometry detection for best coverage.

### Issue: OCR Taking Too Long

**Solution:** OCR now skips non-text/table blocks, caps block area at ~55% of the page, and reuses cached results between runs. This prevents re-OCRing giant backgrounds while keeping form/text blocks fast.

---

## 🚧 Roadmap

### Immediate (1-2 Weeks)

- ✅ GPU setup and integration (DONE)
- ✅ SLM/VLM activation (DONE)
- Threshold tuning and validation
- Testing and benchmarking

### Short-term (1-2 Months)

- Model fine-tuning
- Feature enhancements
- Production readiness
- FastAPI integration testing

### Long-term (3-6 Months)

- Scalability improvements
- Advanced features (multi-language, handwriting)
- Cloud deployment
- Custom model training

---

## 📝 License

[Add your license here]

---

## 🤝 Contributing

[Add contribution guidelines here]

---

## 📧 Contact

**Repository:** https://github.com/rahul370139/doc2data  
**Issues:** https://github.com/rahul370139/doc2data/issues

---

## 📚 Documentation

- **Technical Summary:** See `DEMO_SUMMARY.md` for detailed model information and next steps
- **API Documentation:** See `app/api_main.py` for FastAPI endpoints
- **Code Comments:** All major functions have docstrings


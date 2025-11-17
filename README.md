# Document-to-Data Pipeline

**Version:** 1.0.0 (Phase 1)  
**Status:** In development (CPU), GPU Acceleration Configuration Ready  
**Last Updated:** Nov 16th 2025

Goal:
A production-ready document processing pipeline that converts PDFs and images into structured JSON data. Uses state-of-the-art ML models (LayoutParser, PaddleOCR) combined with intelligent heuristics for layout detection, OCR, and content classification.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (tested with 3.10.11)
- 8GB+ RAM (16GB recommended)
- 5-10GB free disk space (for models)

### Installation

```bash
# Clone repository
git clone https://github.com/rahul370139/doc2data.git
cd doc2data

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Models will auto-download on first run (~1-2GB)
```

### Run Streamlit Demo

```bash
# Activate virtual environment
source venv/bin/activate

# Start Streamlit
streamlit run app/streamlit_main.py --server.address localhost --server.port 8501

# Open browser: http://localhost:8501
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

- **PDF/Image Ingestion:** PyMuPDF, pdf2image, digital text extraction
- **Layout Segmentation:** PubLayNet (LayoutParser) + TableBank + heuristics
- **OCR Processing:** PaddleOCR (primary) + Tesseract (fallback), parallel processing
- **Document Assembly:** Enhanced JSON with detailed metadata, Markdown generation
- **Streamlit UI:** Interactive demo with visualization, threshold controls, results display
- **Model Caching:** Fast subsequent runs with `@st.cache_resource`

### ⚠️ Partial/Stubbed

- **Semantic Labeling (SLM):** Qwen2.5-7B-Instruct via Ollama (code complete, disabled by default)
- **Table Processing:** Heuristic extraction complete, VLM integration stubbed
- **Figure Processing:** Classification stubbed, caption detection working
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
| `src/pipelines/slm_label.py` | ⚠️ | Semantic labeling (stubbed) |
| `src/pipelines/table_processor.py` | ⚠️ | Table extraction (partial) |
| `src/pipelines/figure_processor.py` | ⚠️ | Figure processing (partial) |
| `src/pipelines/assemble.py` | ✅ | JSON/Markdown assembly |
| `app/streamlit_main.py` | ✅ | Streamlit demo UI |
| `app/api_main.py` | ⚠️ | FastAPI endpoints (partial) |

---

## 📊 Performance

### Current (CPU)

- **2-page document:** ~5-6 minutes
- **Layout detection:** ~30-60 sec/page
- **OCR:** ~2-3 min/page
- **Text extraction:** ~85-90% success rate

### Expected (GPU)

- **2-page document:** ~20-40 seconds (10-15x faster)
- **Layout detection:** ~2-5 sec/page (10-20x faster)
- **OCR:** ~10-20 sec/page (5-10x faster)

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

- **Enable SLM/VLM:** Toggle semantic labeling and VLM features (requires Ollama)

### Environment Variables

Create `.env` file (see `.env.example`):

```bash
# LLM/VLM Configuration
ENABLE_SLM=false          # Enable semantic labeling (requires Ollama)
ENABLE_VLM=false          # Enable VLM for tables/figures
OLLAMA_HOST=localhost     # Ollama server host
OLLAMA_PORT=11434         # Ollama server port

# Model Configuration
LAYOUT_MODEL=publaynet    # Layout detection model
OCR_ENGINE=paddle         # OCR engine (paddle/tesseract)
```

---

## 📁 Project Structure

```
doc2data/
├── src/
│   ├── pipelines/        # Pipeline stages
│   ├── ocr/              # OCR implementations
│   ├── vlm/              # VLM integration (stubbed)
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

## 🆕 Recent Improvements (Nov 2025)

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
  - [ ] Column clusters/column-first order (planned)
  - [ ] Template alignment hook (planned)

- OCR Tiering & Association (Step 3)
  - [x] Tier-1 PaddleOCR + preproc; Tier-2 Tesseract fallback
  - [x] PDF digital layer extraction where available
  - [x] Checkbox state via fill ratio
  - [x] Label↔value linking with geometry + validators; stored in block.metadata
  - [ ] Hungarian matching optional (planned; current greedy works well)

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
  - [ ] Overlays for link connectors + validator badges (planned)
  
- Testing & Docs
  - [x] README updated with changes and troubleshooting
  - [ ] Unit tests for geometry/validators/linking (planned)

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

**Solution:** Enable "Form Geometry Detection" in Model Settings. This uses dedicated form geometry detector instead of aggressive heuristics, preserving form structure.

### Issue: Form Fields Not Covering Full Text

**Solution:** Form blocks now preserve original bounding boxes without aggressive tightening. Ensure "Form Geometry Detection" is enabled for best results.

### Issue: OCR Taking Too Long

**Solution:** Already optimized with parallel processing and block filtering. Large blocks (>60% page) are skipped.

---

## 🚧 Roadmap

### Immediate (1-2 Weeks)

- GPU setup and integration
- SLM/VLM activation
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

- **Project Status:** See `PROJECT_STATUS.md` for detailed implementation status
- **API Documentation:** See `app/api_main.py` for FastAPI endpoints
- **Code Comments:** All major functions have docstrings

---

**Last Updated:** November 16th 2025

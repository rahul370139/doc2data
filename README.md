# Document-to-Data Pipeline - Phase 1

**Schema-agnostic document parser** that converts PDFs/images into layout-aware JSON + Markdown with bounding-box citations.

## 🎯 Overview

This pipeline extracts structured information from documents while preserving layout, reading order, and traceability through bounding-box citations. Designed for on-prem, open-source deployment with CPU-first support (GPU optional for later phases).

## ✨ Features

- **Document Ingestion**: PDF and image support with preprocessing (de-skew, de-noise, DPI normalization)
- **Layout Segmentation**: Detect text, title, list, table, figure, and form blocks using LayoutParser + heuristic line-density enhancement
- **OCR**: PaddleOCR (primary) and Tesseract (fallback) with word-level bounding boxes
- **Semantic Labeling**: Qwen2.5-7B-Instruct via Ollama for fine-grained role classification
- **Table Processing**: Path A (heuristics) and Path B (Qwen-VL) for structured extraction
- **Figure Processing**: Qwen-VL classification and chart data extraction
- **Interactive Visualization**: Streamlit app with side-by-side document/JSON/Markdown view
- **REST API**: FastAPI endpoints for each pipeline stage
- **Caching**: SHA256-based artifact caching for pipeline efficiency

## 📁 Project Structure

```
doc2data/
├── src/                    # Source code
│   ├── pipelines/         # Pipeline stages (ingest, segment, ocr, slm_label, assemble)
│   ├── processing/        # Pre/post-processing utilities
│   ├── ocr/               # OCR modules (PaddleOCR, Tesseract)
│   └── vlm/               # Vision-language models (Ollama client, Qwen-VL)
├── app/                   # Application interfaces
│   ├── api_main.py        # FastAPI REST endpoints
│   └── streamlit_main.py # Streamlit interactive UI
├── utils/                 # Utility modules (models, config, cache, visualization)
├── data/                  # Data directory
│   └── sample_docs/       # Sample PDFs for testing
├── models/                # Model weights and download scripts
├── tests/                 # Test suite
│   └── test_pipeline.py   # Comprehensive integration test
├── eval/                  # Evaluation scripts
├── cache/                 # Artifact cache (gitignored)
└── validation/            # Validation results
```

## 🚀 Setup

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running (for SLM/VLM inference)
- **Tesseract OCR** installed (`brew install tesseract` on macOS)
- **Virtual environment** (recommended)

### Installation

1. **Clone and navigate:**
```bash
cd doc2data
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download models** (auto-downloads on first use, or run manually):
```bash
python models/download_models.py
```

5. **Pull Ollama models:**
```bash
ollama pull qwen2.5:7b-instruct
ollama pull qwen2-vl:7b
```

6. **Configure environment** (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

## 📊 Current Status

### ✅ Completed (Step 1 & 2)

#### Step 1: Ingest & Preprocess ✅ **PASS (5/5)**
- ✅ PDF/image loading with PyMuPDF/pdf2image
- ✅ DPI normalization (300 DPI default)
- ✅ De-skew using OpenCV Hough transform
- ✅ De-noise using median/Gaussian blur
- ✅ Digital text layer extraction (bypasses OCR when available)
- ✅ Word-level bounding box extraction from digital text

**Status**: Production-ready, fully functional

#### Step 2: Layout Segmentation ✅ **Working (ML Model + Heuristic Fallback)**
- ✅ LayoutParser integration with PaddleDetectionLayoutModel (PubLayNet)
- ✅ Auto-downloads PubLayNet weights (221MB, PPYOLOv2 variant)
- ✅ Works on CPU/MPS (no CUDA required)
- ✅ Box merging with IoU threshold (0.5)
- ✅ Reading order resolution (top→bottom, left→right, multi-column support)
- ✅ Form region detection via line-density heuristics layered on ML outputs
- ✅ Heuristic fallback (OpenCV contour detection) when ML model unavailable
- ⚠ Parameter tuning needed for optimal granularity

**Status**: Functional, ML model working, needs parameter tuning for fine-grained detection

### 🚧 In Progress

#### Step 3: OCR (per-block)
- ✅ PaddleOCR wrapper with auto-download
- ✅ Tesseract fallback wrapper
- ✅ OCR orchestration pipeline
- ⚠ Header/footer detection heuristics
- ⚠ Caption candidate detection
- ⚠ Text cleaning and post-processing

**Status**: Core functionality implemented, heuristics need refinement

#### Step 4: Semantic Labeling (SLM)
- ✅ Ollama client integration
- ✅ Qwen2.5-7B-Instruct prompt templates
- ⚠ Semantic role labeling (title, H1/H2, header/footer, page#, list-item, kv-pair hints)
- ⚠ JSON output parsing and validation

**Status**: Infrastructure ready, labeling logic needs implementation

#### Step 5: Table & Figure Processing
- ✅ Qwen-VL integration scaffolding
- ⚠ Table extraction (Path A: heuristics, Path B: Qwen-VL)
- ⚠ Figure classification and chart extraction
- ⚠ Caption extraction and linking

**Status**: Placeholders implemented, core logic pending

#### Step 6: Assembly
- ✅ Document data models (Block, TableBlock, FigureBlock, Document)
- ⚠ Hierarchical JSON builder
- ⚠ Markdown generator with citations
- ⚠ Bounding-box citation preservation

**Status**: Data models ready, assembly logic pending

### 📋 Planned

#### UI & API
- ⚠ Streamlit UI (file upload, page viewer, JSON/MD tabs, interactive overlays)
- ⚠ FastAPI endpoints (all pipeline stages)
- ⚠ Real-time progress updates
- ⚠ Error handling and validation

#### Testing & Validation
- ✅ Integration test (tests/test_pipeline.py)
- ⚠ Unit tests for each pipeline stage
- ⚠ Evaluation scripts (layout validation, OCR validation)
- ⚠ Performance benchmarks

#### Deployment
- ✅ Dockerfile (CPU-first)
- ✅ docker-compose.yml
- ⚠ GPU support (NVIDIA DGX integration)
- ⚠ vLLM integration for GPU-accelerated SLM inference

## 🧪 Testing

### Run Integration Test

Test Steps 1 & 2 (Ingest & Layout Segmentation):

```bash
python tests/test_pipeline.py
```

This will:
- Load sample PDFs from `data/sample_docs/`
- Run ingestion and preprocessing
- Perform layout segmentation
- Display detailed results and evaluation scores

### Expected Output

```
Step 1: 5/5 (PASS)
Step 2: 3-6/6 (Working, needs parameter tuning)
```

## 📖 Usage

### Streamlit App (Coming Soon)

```bash
streamlit run app/streamlit_main.py
```

Open http://localhost:8501 in your browser.

### FastAPI Server (Coming Soon)

```bash
python -m uvicorn app.api_main:app --reload
```

API will be available at http://localhost:8000

### API Endpoints (Planned)

- `POST /ingest` - Ingest document (PDF/image)
- `POST /segment` - Segment layout
- `POST /ocr` - Run OCR
- `POST /label` - Semantic labeling
- `POST /table/process` - Process table
- `POST /figure/process` - Process figure
- `POST /assemble` - Assemble final JSON/Markdown
- `GET /health` - Health check

## 🐳 Docker

### Build and Run

```bash
docker-compose up --build
```

This will start:
- API server on port 8000
- Streamlit app on port 8501

**Note**: Ollama should be running on the host (or add to docker-compose.yml).

## 🎯 Milestones

### M1: Ingest + Segment + OCR ✅ **In Progress**
- [x] PDF/image ingestion
- [x] Layout segmentation (ML model + heuristic fallback)
- [x] OCR with PaddleOCR/Tesseract
- [ ] OCR heuristics and validation
- [ ] Target: ≥95% text coverage

### M2: SLM Labeling + Assembly 📋 **Planned**
- [ ] Semantic role labeling
- [ ] JSON/Markdown assembly
- [ ] Table/Figure basic processing
- [ ] Target: Correct header/footer/KV pair tagging

### M3: Full Demo 📋 **Planned**
- [ ] Streamlit UI
- [ ] Docker image
- [ ] Interactive visualization
- [ ] Target: One-click demo working

## 🔧 Configuration

See `.env.example` for configuration options:

- `OLLAMA_HOST`: Ollama server address (default: http://localhost:11434)
- `SLM_MODEL`: SLM model name (default: qwen2.5:7b-instruct)
- `VLM_MODEL`: VLM model name (default: qwen-vl)
- `DPI`: Image resolution (default: 300)
- `DESKEW_ENABLED`: Enable de-skew (default: True)
- `DENOISE_ENABLED`: Enable de-noise (default: True)
- `OCR_PRIMARY`: Primary OCR engine (default: paddleocr)
- `LAYOUT_MODEL_NAME`: Layout model name (default: publaynet)

## 🏗️ Architecture

### Pipeline Flow

```
PDF/Image
  ↓
[1] Ingest & Preprocess (de-skew, de-noise, digital text extraction)
  ↓
[2] Layout Segmentation (ML model or heuristic fallback)
  ↓
[3] OCR (per-block, PaddleOCR/Tesseract)
  ↓
[4] Semantic Labeling (SLM via Ollama)
  ↓
[5] Table & Figure Processing (Qwen-VL)
  ↓
[6] Assembly (JSON + Markdown with citations)
  ↓
Document (JSON + Markdown)
```

### Key Design Decisions

1. **CPU-First**: All components work on CPU, with optional GPU acceleration later
2. **Progressive Fallbacks**: ML model → Heuristic → Basic processing
3. **Caching**: SHA256-based caching for pipeline artifacts
4. **Modular**: Each pipeline stage is independent and testable
5. **Traceability**: All extracted data includes bounding-box citations

## 🚀 Future Enhancements

### GPU Acceleration
- NVIDIA DGX integration
- vLLM for GPU-accelerated SLM inference
- CUDA-enabled PyTorch for Detectron2 (if needed)

### Model Improvements
- Fine-tune LayoutParser model for better granularity
- Custom SLM prompts for domain-specific labeling
- Advanced table extraction (TATR integration)
- Chart data extraction (Plotly/Matplotlib parsing)

### Features
- Multi-language support
- Batch processing
- Incremental processing
- Webhook notifications
- Export formats (JSON, Markdown, HTML, PDF)

## 📝 Notes

### Detectron2 Alternative
- **PaddleDetectionLayoutModel** is used instead of Detectron2
- Works on CPU/MPS without CUDA
- Auto-downloads PubLayNet weights (221MB)
- No Detectron2 installation required

### Model Downloads
- LayoutParser models auto-download on first use
- PaddleOCR models auto-download on initialization
- Ollama models must be pulled manually (`ollama pull`)

### Performance
- CPU processing: ~5-10 seconds per page
- Heuristic fallback: Faster but less accurate
- ML model: Slower but more accurate
- Caching reduces redundant processing

## 🤝 Contributing

Contributions welcome! Please:
1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure CPU-first compatibility

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- LayoutParser team for PubLayNet models
- PaddleOCR for OCR capabilities
- Ollama for SLM/VLM hosting
- Qwen team for language models

---

**Status**: 🚧 Active Development - Steps 1 & 2 Complete, Steps 3-6 In Progress

**Last Updated**: 2024

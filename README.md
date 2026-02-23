# Doc2Data — Document-to-Data Pipeline

**Version:** 1.1.0-beta (CMS-1500 Focus)  
**Status:** ✅ Production-Ready (GPU-Accelerated)  
**Live Demo:** http://100.126.216.92:8501

A production-ready document processing pipeline that converts PDFs and images into structured JSON data. Uses ML models (LayoutParser, PaddleOCR, TrOCR) combined with heuristics, SLM/VLM enrichment (Llama/Qwen), and GPU-aware preprocessing for layout detection, OCR, and content classification.

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Architecture Overview](#-architecture-overview)
- [Project Structure](#-project-structure)
- [Scripts Reference](#-scripts-reference)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Documentation](#-documentation)

---

## 🚀 Quick Start

### Docker (Recommended)

```bash
# Build and run (API + Streamlit)
docker-compose -f docker/docker-compose.yml up --build

# Or build image only
docker build -f docker/Dockerfile -t doc2data-gpu .

# Run with GPU
docker run -d --gpus all -p 8501:8501 -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -e USE_GPU=true doc2data-gpu
```

**Access:** http://localhost:8501 (Streamlit) | http://localhost:8000 (API)

### Local Development

```bash
git clone https://github.com/rahul370139/doc2data.git
cd doc2data
python3.10 -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app/streamlit_main.py
```

---

## 🏛 Architecture Overview

```
PDF/Image → Form ID → [Lane A/B/C] → Layout → OCR → Validation → JSON
```

| Stage | What Happens |
|-------|---------------|
| **Form Identification** | OCR header, match CMS-1500/UB-04 keywords, layout fingerprint |
| **Lane A** | Fillable PDF → extract AcroForm widgets directly (no OCR) |
| **Lane B** | Digital PDF → extract embedded text layer, zone matching |
| **Lane C** | Scanned → align to template, OCR (PaddleOCR + TrOCR), zone match |
| **Layout** | Detectron2/YOLOv8 → blocks (text, table, figure, form_field) |
| **OCR** | PaddleOCR (print) + TrOCR (handwriting) + checkbox detector |
| **Validation** | NPI, date, phone, ICD-10 validators + LLM QA |
| **Assembly** | Map to business schema (patient_name, diagnosis, etc.) → JSON |

See [PIPELINE_OVERVIEW.md](documents/PIPELINE_OVERVIEW.md) for detailed flowcharts.

---

## 📁 Project Structure

```
doc2data/
├── app/                      # Web interfaces
│   ├── streamlit_main.py     # Streamlit UI (upload, extract, visualize)
│   └── api_main.py          # FastAPI REST API (/extract, /health)
│
├── src/
│   ├── pipelines/           # Core extraction pipeline
│   │   ├── multi_agent_pipeline.py   # Main orchestrator (3-lane extraction)
│   │   ├── core/            # Base types (BaseAgent, FormType, PipelineConfig)
│   │   ├── agents/          # Pipeline agents
│   │   │   ├── form_identification.py   # Detects CMS-1500, UB-04, etc.
│   │   │   ├── template_alignment.py   # Aligns scanned form to template
│   │   │   ├── layout_detection.py     # YOLO/Detectron2 block detection
│   │   │   ├── ocr.py                  # PaddleOCR + TrOCR + checkbox
│   │   │   ├── labeling.py             # SLM semantic labels
│   │   │   └── validation.py           # Field validators + LLM QA
│   │   ├── registration/    # CMS-1500 alignment (cms1500_register)
│   │   ├── validators/      # NPI, date, phone, ICD, etc.
│   │   └── schemas/         # Business schema mapping
│   │
│   ├── processing/          # Image preprocessing
│   │   ├── registration.py # Template loading (delegates to cms1500_register)
│   │   └── preprocessing.py # Deskew, denoise, red removal
│   │
│   ├── ocr/                 # OCR wrappers
│   │   └── paddle_ocr.py    # PaddleOCR integration
│   │
│   └── chatbot/             # Chat/query utilities
│       └── ocr_query.py
│
├── utils/
│   ├── config.py            # Environment, paths, model config
│   ├── models.py            # Shared data models
│   ├── corrections.py       # Correction logging, threshold tuning
│   └── cache.py             # Caching utilities
│
├── scripts/                 # CLI and helper scripts
│   ├── api_client.py        # Test API from command line
│   ├── test_pipeline.py     # Full pipeline integration test
│   ├── grade_extraction.py  # Grade predictions vs gold labels
│   ├── debug_cms1500_alignment.py  # Debug alignment on a PDF
│   ├── tune_cms1500_thresholds.py  # Grid search for alignment thresholds
│   └── recompute_thresholds.py     # Recompute thresholds from corrections
│
├── docker/                  # Docker configuration
│   ├── Dockerfile           # NVIDIA PyTorch base, Detectron2, PaddleOCR
│   ├── requirements_docker.txt
│   ├── start_services.sh    # Container entrypoint
│   └── docker-compose.yml   # API + Streamlit services
│
├── data/
│   ├── sample_docs/         # Sample PDFs for testing
│   ├── schemas/             # CMS-1500, UB-04 field definitions
│   ├── gold_labels/         # Ground truth for grading
│   └── thresholds.json      # Tuned thresholds from corrections
│
├── deploy_and_run.sh        # Deploy to DGX via rsync + docker
├── requirements.txt         # Local Python dependencies
└── README.md
```

---

## 📜 Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| **api_client.py** | Call Doc2Data API from CLI | `python scripts/api_client.py document.pdf --url http://host:8000` |
| **test_pipeline.py** | Run full pipeline on sample docs | `python scripts/test_pipeline.py` |
| **grade_extraction.py** | Grade predictions vs gold JSON | `python scripts/grade_extraction.py --pred pred_dir --gold gold_dir` |
| **debug_cms1500_alignment.py** | Debug CMS-1500 alignment on a PDF | `python scripts/debug_cms1500_alignment.py --input cms1500.pdf` |
| **tune_cms1500_thresholds.py** | Grid search for alignment thresholds | `python scripts/tune_cms1500_thresholds.py --input data/sample_docs/` |
| **recompute_thresholds.py** | Recompute thresholds from corrections.jsonl | `python scripts/recompute_thresholds.py` |
| **start_api_server.sh** | Start FastAPI server (used in deployment) | `./scripts/start_api_server.sh` |

---

## ⚙️ Configuration

### Environment Variables

Create `.env` (see `.env.example`):

```bash
# LLM/VLM
ENABLE_SLM=true
ENABLE_VLM=true
OLLAMA_HOST=localhost:11434
OLLAMA_MODEL_SLM=qwen2.5:7b-instruct
OLLAMA_MODEL_VLM=minicpm-v

# GPU
USE_GPU=true
CUDA_VISIBLE_DEVICES=0

# Layout (optional)
YOLO_MODEL_PATH=runs/detect/cms1500_yolo/weights/best.pt
```

### Streamlit UI Settings

- **Form Type:** CMS-1500, UB-04, or Auto
- **Enable Alignment:** Template alignment for scanned forms
- **Enable SLM/VLM:** Semantic labeling and figure understanding (requires Ollama)
- **Confidence Threshold:** OCR confidence cutoff (0.5 default)

---

## 🔧 Usage

### Streamlit

1. Start: `streamlit run app/streamlit_main.py`
2. Upload PDF or select sample
3. Configure form type and options
4. Run extraction → view JSON, business fields, annotated image

### API

```bash
# Health check
curl http://localhost:8000/health

# Extract document
curl -X POST -F "file=@cms1500.pdf" http://localhost:8000/extract/cms1500
```

### Python

```python
from src.pipelines.multi_agent_pipeline import MultiAgentPipeline, PipelineConfig

config = PipelineConfig(enable_alignment=True)
pipeline = MultiAgentPipeline(config)
result = pipeline.process_sync("document.pdf")
print(result["extracted_fields"])
print(result["business_fields"])
```

---

## 📚 Documentation

| Document | Description |
|----------|--------------|
| [PIPELINE_OVERVIEW.md](documents/PIPELINE_OVERVIEW.md) | Detailed architecture, every script, methods, models |
| [SCRUM_REPORT.md](documents/SCRUM_REPORT.md) | Status, roadmap, component inventory |
| [docker/README.md](docker/README.md) | Docker build and run instructions |

---

## 📦 Models

- **Layout:** Detectron2 (PubLayNet), PaddleDetection, optional YOLOv8 (CMS-1500)
- **OCR:** PaddleOCR (print), TrOCR (handwriting)
- **Tables:** Microsoft Table Transformer (TATR)
- **SLM/VLM:** Qwen, MiniCPM-V via Ollama

Models auto-download on first run. Total ~1–2 GB.

---

## 📧 Contact

**Repository:** https://github.com/rahul370139/doc2data  
**Issues:** https://github.com/rahul370139/doc2data/issues

# Doc2Data — Document-to-Data Extraction Pipeline

**Implementation snapshot:** March 2026

Doc2Data converts healthcare forms (primarily **CMS-1500** and **UB-04**) from PDF or image into structured JSON:

- **extracted_fields** — schema-level field IDs and values
- **field_details** — confidence, bounding box, source metadata per field
- **business_fields** — normalized keys for downstream systems
- **validation** — format and consistency diagnostics
- **reducto_format** — optional Reducto-style export

**Runtime entrypoint:** `src/pipelines/multi_agent_pipeline.py` (`process` / `process_sync`)

**Service entrypoints:** `app/api_main.py` (FastAPI), `app/streamlit_main.py` (Streamlit)

## What the pipeline does

1. Loads page 1 from PDF/image at rendering resolution (PDF renders at 300 DPI).
2. Identifies form type (`cms-1500`, `ub-04`, `ncpdp`, `generic`).
3. Selects extraction path:
   - Lane A: AcroForm widgets
   - Lane B: digital text layer + zone matching
   - Lane C: scan alignment + schema zones + OCR
4. Runs OCR routing by block type (checkbox/signature/date/text/table).
5. Runs table extraction for service lines when applicable.
6. Validates fields (date, phone, NPI, ICD, money, etc.).
7. Applies conservative targeted VLM rescue only for clearly bad fields.
8. Maps schema fields to business schema.
9. Returns full JSON and Reducto-like format.

## Current architecture (code-accurate)

```text
Input PDF/Image
  -> FormIdentificationAgent
  -> Lane A/B/C selection
  -> (optional) TemplateAlignmentAgent
  -> Schema zones or LayoutDetectionAgent fallback
  -> OCRAgent.process_blocks
  -> LabelingAgent (tables/figures; optional SLM text labeling)
  -> ValidationAgent
  -> business_schema.map_to_business_schema
  -> Final response + reducto_format
```

### Three-lane extraction strategy

| Lane | Trigger | Method | Notes |
|---|---|---|---|
| A (widgets) | Fillable PDF with enough non-empty widgets | `_extract_widgets` + `_map_widgets_to_schema` | Fastest and most reliable. No OCR. |
| B (digital text) | Digital text layer is present and visually valid | `page.get_text("words")` + `_match_ocr_to_zones` | For CMS-1500, extra anchor QA is enforced before trust. |
| C (scan OCR) | No trustworthy text layer | Alignment + schema zones + per-field OCR | Default for scanned/handwritten forms. |

## OCR routing summary

| Field/block type | Runtime path |
|---|---|
| Checkbox | Fill-ratio detector after template subtraction/diff |
| Signature | Ink detector only, returns `[SIGNED]` or empty |
| Date/date_range (CMS) | Florence-2 first, VLM rescue for incomplete/empty-with-content |
| Text/address/numeric (CMS) | Florence-2-first field pipeline + raw-crop fallback + structural filters |
| General text (non-CMS) | PaddleOCR -> optional TrOCR -> VLM fallback |
| Table (CMS Box 24) | VLM table extraction in `LabelingAgent`, with cell-OCR fallback |

Important: blank-field suppression is active; fields marked blank are filtered from visual overlays/details.

## Schemas and field coverage

### CMS-1500 schema (`data/schemas/cms-1500.json`)

- Total fields: **86**
- Uses `bbox_norm_new` when available (preferred over `bbox_norm`)
- Includes `mode` (`both`, `digital`, `scan`) for lane-aware field activation
- Contains checkbox subfields, money fields, date fields, signatures, and Box 24 table region

### UB-04 schema (`data/schemas/ub-04.json`)

- Total fields: **98**
- Includes `block_type` and optional `business_key`
- Supports direct widget mapping for fillable UB-04 PDFs plus spatial fallback mapping

## Output contract

Top-level response (typical):

```json
{
  "success": true,
  "form_type": "cms-1500",
  "extracted_fields": {"2_patient_name": "..."},
  "field_details": [
    {
      "id": "2_patient_name",
      "value": "...",
      "confidence": 0.91,
      "bbox": [x0, y0, x1, y1],
      "metadata": {"source": "schema_zones", "ocr_engine": "florence2"}
    }
  ],
  "business_fields": {"patient_name": "..."},
  "validation": {"errors": [], "warnings": [], "qa_notes": []},
  "debug": {
    "alignment_used": true,
    "alignment_success": true,
    "alignment_quality": 0.88,
    "digital_text_used": false,
    "vlm_rescue_count": 1
  },
  "reducto_format": {"result": {"chunks": [...]}}
}
```

## Quick start

### Local development

```bash
git clone https://github.com/rahul370139/doc2data.git
cd doc2data
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_main.py
```

### Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

Services:

- Streamlit UI: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## API endpoints

Defined in `app/api_main.py`:

- `POST /extract/v2`
- `POST /extract/reducto`
- `POST /extract/cms1500`
- `POST /extract/ub04`
- `POST /extract/generic`
- `POST /chat/query`
- `GET /health`
- `GET /schemas`

## Python usage

```python
from src.pipelines.multi_agent_pipeline import MultiAgentPipeline, PipelineConfig

config = PipelineConfig(
    enable_alignment=True,
    enable_trocr=True,
    enable_vlm_ocr_fallback=True,
)

pipeline = MultiAgentPipeline(config)
result = pipeline.process_sync("data/sample_docs/cms1500.pdf")
print(result["extracted_fields"])
print(result.get("business_fields", {}))
```

## Configuration

### Environment variables (`utils/config.py`)

Core:

- `OLLAMA_HOST` (default `localhost:11434`)
- `OLLAMA_MODEL_SLM` (default `llama3.2:3b`)
- `VLM_MODEL_RESCUE` (default `minicpm-v`)
- `VLM_MODEL_TABLE` (default `openbmb/minicpm-o4.5:latest`)
- `VLM_MODEL_TABLE_FALLBACK` (default `minicpm-v`)
- `ENABLE_SLM` / `ENABLE_VLM` (both default `false`)
- `YOLO_MODEL_PATH`, `YOLO_CONFIDENCE`, `YOLO_IOU`
- `USE_GPU`, `CUDA_VISIBLE_DEVICES`

CMS-1500 registration tuning (used by registrar):

- `CMS1500_TEMPLATE_PATH`
- `CMS1500_RED_S_MIN`, `CMS1500_RED_V_MIN`, `CMS1500_RED_RATIO_SWITCH`
- `CMS1500_MATCH_RATIO_DEFAULT`, `CMS1500_MATCH_RATIO_HANDWRITTEN`
- `CMS1500_MIN_KEYPOINTS_DEFAULT`, `CMS1500_MIN_KEYPOINTS_HANDWRITTEN`
- `CMS1500_MIN_MATCHES_DEFAULT`, `CMS1500_MIN_MATCHES_HANDWRITTEN`
- `CMS1500_RANSAC_REPROJ_DEFAULT`, `CMS1500_RANSAC_REPROJ_HANDWRITTEN`
- `CMS1500_QUAD_MIN_SCORE_DEFAULT`, `CMS1500_QUAD_MIN_SCORE_HANDWRITTEN`
- `CMS1500_MIN_FEATURE_QUALITY`

### Runtime config (`PipelineConfig`)

Common toggles:

- `enable_form_detection`
- `form_type_override`
- `enable_alignment`
- `layout_model` (`auto`, `detectron2`, `paddle`, `yolo`)
- `enable_trocr`
- `ocr_engine_mode`
- `enable_vlm_ocr_fallback`
- `zone_padding_px`, `zone_padding_ratio`
- `enable_slm_labeling`
- `enable_vlm_figures`
- `enable_vlm_tables`
- `enable_validators`

## Key Components

| Component | Script | Role |
|-----------|--------|------|
| Orchestrator | `multi_agent_pipeline.py` | Lane selection, alignment, OCR, validation, assembly |
| Form ID | `form_identification.py` | Detect CMS-1500, UB-04, etc. |
| Alignment | `template_alignment.py`, `cms1500_register.py` | Align scan to template |
| OCR | `ocr.py` | Florence-2 primary, raw-crop fallback, blank detection |
| Tables | `labeling.py` | Box 24 VLM extraction |
| Validation | `validation.py` | NPI, date, phone, ICD, etc. |
| Business mapping | `business_schema.py` | Schema ID → business keys |

## Repository structure

```text
doc2data/
├── app/
│   ├── api_main.py
│   └── streamlit_main.py
├── src/
│   ├── pipelines/
│   │   ├── multi_agent_pipeline.py
│   │   ├── core/
│   │   ├── agents/
│   │   ├── validators/
│   │   ├── schemas/
│   │   └── registration/
│   ├── processing/
│   └── chatbot/
├── data/
│   ├── schemas/
│   ├── sample_docs/
│   ├── gold_labels/
│   └── templates/
├── scripts/
├── docker/
└── utils/
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_pipeline.py` | End-to-end smoke test |
| `scripts/grade_extraction.py` | Compare predictions vs gold labels |
| `scripts/debug_cms1500_alignment.py` | Single-file alignment diagnostics |
| `scripts/tune_cms1500_thresholds.py` | Registrar threshold grid search |
| `scripts/recompute_thresholds.py` | Threshold suggestions from corrections log |
| `scripts/api_client.py` | CLI client for REST endpoints |
| `deploy_and_run.sh` | DGX deployment (rsync + Docker) |

## Known limitations

- Current pipeline processes **page 1** only.
- CMS-1500 scan quality still depends heavily on registration quality and handwriting clarity.
- First run can be slow due to model initialization/warmup.
- Ollama-dependent features (SLM/VLM) require local model availability.

## Additional Documentation

| Document | Description |
|---------|-------------|
| `documents/PIPELINE_OVERVIEW.md` | Architecture, Mermaid diagrams, script-to-function mapping, OCR pipeline details |
| `documents/SCRUM_REPORT.md` | Structured process flow narrative, component status, risks, and roadmap (for manager review) |
| `documents/system_prompt.md` | Engineering principles for reliability and traceability |

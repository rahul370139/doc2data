# Doc2Data — Document-to-Data Extraction Pipeline

**Implementation snapshot:** April 2026

Doc2Data converts healthcare forms (primarily **CMS-1500** and **UB-04**) from PDF or image into structured JSON in ~40 seconds end-to-end:

- **extracted_fields** — schema-level field IDs and values
- **field_details** — confidence, bounding box, source metadata per field
- **business_fields** — normalized keys for downstream systems
- **validation** — format / consistency diagnostics + `qa_notes`
- **debug** — per-node trace, timings, alignment quality, rescue counters
- **reducto_format** — optional Reducto-style export

**Orchestrator:** `src/pipelines/graph` (LangGraph) — planner → lane → align → extract → validate → reflect → rescue → finalize.

**Service entrypoints:**
- `app/api_main.py` (FastAPI) — includes `/extract/graph` (blocking) and `/extract/graph/stream` (SSE)
- `frontend/` (Next.js 14 + TypeScript + Tailwind) — drag-and-drop upload, live progress stream, and a multi-tab result explorer
- `app/streamlit_main.py` (legacy Streamlit UI, still available on port 8501)

## What the pipeline does

1. Loads page 1 from PDF/image (PDF rendered at 300 DPI).
2. Identifies the form type (`cms-1500`, `ub-04`, `ncpdp`, `generic`) using OCR-fingerprint matching with CMS-specific strong-tokens.
3. Plans an extraction lane (A widgets, B digital text, C scan OCR).
4. Aligns the scan to the canonical template when Lane C is chosen — if alignment quality is below threshold, the graph short-circuits the heavy OCR pass and surfaces a `qa_notes` entry instead of spending several minutes on misaligned zones.
5. Runs **OCR v2** — a single batched Florence-2 call gated by an adaptive, structural blank detector. Non-blank fields are routed to the field-type pipeline (date, money, checkbox, table, etc.).
6. Validates fields (date, phone, NPI, ICD, money, zip, tax_id …).
7. **Reflects** on the validation errors + confidence + ink signals, builds a small rescue candidate list (≤ 12), and runs a **targeted VLM rescue** on just that subset.
8. Re-validates, maps schema fields to the business schema, and assembles the final response.

## Current architecture (LangGraph)

```text
PDF/Image
  ├─ load          — render + extract digital words
  ├─ identify      — form-type fingerprint (PaddleOCR header+footer)
  ├─ plan          — choose Lane A | B | C
  ├─ extract_*     — widgets | digital_text | align+scan
  ├─ validate      — typed validators + QA notes
  ├─ reflect       — pick ≤12 fields for rescue
  ├─ rescue        — targeted VLM (conditional)
  ├─ (revalidate)
  └─ finalize      — business mapping + reducto + debug trace
```

### Three-lane extraction strategy

| Lane | Trigger | Method | Notes |
|---|---|---|---|
| **A (widgets)** | Fillable PDF with ≥10 filled widgets (CMS-1500) / ≥3 (UB-04) | `_extract_widgets` + `_map_widgets_to_schema` | Fastest and most reliable. No OCR. |
| **B (digital text)** | Non-structured form with a digital text layer | `page.get_text("words")` + `_match_ocr_to_zones` | For CMS-1500 / UB-04 we prefer Lane C because schema bboxes are calibrated to the canonical template; the plan node downgrades Lane B → C if the digital match is sparse OR when >30 % of matched blocks share identical text (template-mismatch guard). |
| **C (scan OCR)** | No widgets, or scan / handwritten, or Lane B downgrade | Template alignment + schema zones + **batched Florence-2 (OCR v2)** + adaptive blank detection + optional VLM rescue | Default for scanned and non-standard-layout forms. If alignment quality < 0.5, the heavy OCR pass is skipped and the graph emits an `alignment_failed` QA note instead. |

### OCR v2: one batched Florence-2 call, adaptive blank detection

`src/pipelines/ocr_v2/` replaces the old per-field Florence-2 loop:

| Component | Role |
|---|---|
| `BlankDetector` | Per-form calibration on known-blank zones, center-weighted ink measurement + CCA, produces a `blank_status ∈ {blank, filled, uncertain}` per field. |
| `BatchedFlorence2` | One pipelined Florence-2 invocation for every `uncertain` / `filled` field in the page — drops end-to-end OCR time from ~4 min to ~20-30 s. |
| `FieldOCRBatch` | Glue between the schema zones, the blank detector, and the batched inference. |

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

### Local development (backend only)

```bash
git clone https://github.com/rahul370139/doc2data.git
cd doc2data
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn app.api_main:app --host 0.0.0.0 --port 8000 --reload
```

### Next.js frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # set API_BASE_URL=http://localhost:8000
npm run dev                  # http://localhost:3000
```

### Docker (backend + frontend)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

or, for the DGX2 deployment that also mounts `HF_HOME` and wires up the front-end:

```bash
./deploy_and_run.sh
```

Services:

- Next.js app:   `http://localhost:3000`
- FastAPI:       `http://localhost:8000`
- Swagger:       `http://localhost:8000/docs`
- Streamlit UI:  `http://localhost:8501` (legacy)

### Frontend ↔ backend proxying

The Next.js app ships with a Route Handler at `frontend/app/api/backend/[...slug]/route.ts` that proxies browser requests to the FastAPI service using the `API_BASE_URL` env var **at request time**. This lets us keep the same Docker image across environments and avoids the build-time rewrite baking that `output: "standalone"` would otherwise impose.

## API endpoints

Defined in `app/api_main.py`:

**LangGraph orchestrator (recommended)**

- `POST /extract/graph` — blocking, returns the full JSON response.
- `POST /extract/graph/stream` — Server-Sent Events, emits `node_start` / `node_end` / `final` events suitable for live UIs.

**Legacy endpoints (still supported)**

- `POST /extract/v2`
- `POST /extract/reducto`
- `POST /extract/cms1500`
- `POST /extract/ub04`
- `POST /extract/generic`
- `POST /chat/query`
- `GET  /health`
- `GET  /schemas`

## Python usage

### LangGraph orchestrator (recommended)

```python
import asyncio
from src.pipelines.graph import build_graph, create_initial_state

graph = build_graph()

async def run(path: str):
    state = create_initial_state(path)
    final = await graph.ainvoke(state)
    return final["response"]

response = asyncio.run(run("data/sample_docs/cms1500.pdf"))
print(response["extracted_fields"])
print(response["debug"]["node_trace"])       # e.g. ['load','identify','plan','align','extract_scan','validate','reflect','rescue','revalidate','finalize']
print(response["debug"]["node_timings_ms"])  # per-node wall-clock
```

### Legacy MultiAgentPipeline

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
| **LangGraph orchestrator** | `src/pipelines/graph/{graph.py,nodes.py,state.py}` | Nodes + conditional edges for plan → extract → validate → reflect → rescue → finalize |
| Legacy orchestrator | `multi_agent_pipeline.py` | Still used under the hood for widget/digital extraction and as the VLM rescue harness |
| Form ID | `form_identification.py` | Strong-token fingerprint matcher (CMS-1500, UB-04, NCPDP, generic) |
| Alignment | `template_alignment.py`, `cms1500_register.py` | AKAZE/ORB + RANSAC homography with quad fallback |
| **OCR v2 (batched)** | `src/pipelines/ocr_v2/` | `BlankDetector` + `BatchedFlorence2` + `FieldOCRBatch` |
| OCR v1 (per-field) | `src/pipelines/agents/ocr.py` | Block-type routing for checkbox / signature / date / table |
| Tables | `labeling.py` | Box 24 VLM extraction with OCR fallback |
| Validation | `validation.py` | NPI, date, phone, ICD, money, zip, tax_id, … |
| Business mapping | `business_schema.py` | Schema ID → business keys |
| **Frontend** | `frontend/` | Next.js 14 + TypeScript + Tailwind UI with SSE progress stream |

## Repository structure

```text
doc2data/
├── app/
│   ├── api_main.py                 # FastAPI (graph + legacy endpoints)
│   └── streamlit_main.py           # legacy Streamlit UI
├── frontend/                       # Next.js 14 app
│   ├── app/                        # route handlers + pages
│   │   └── api/backend/[...slug]/  # runtime proxy to FastAPI
│   ├── components/                 # UploadZone, ProgressStream, …
│   ├── lib/                        # API client + types
│   └── Dockerfile
├── src/
│   ├── pipelines/
│   │   ├── graph/                  # LangGraph orchestrator
│   │   ├── ocr_v2/                 # Batched Florence-2 + blank detector
│   │   ├── multi_agent_pipeline.py # Legacy orchestrator (still used)
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
│   ├── test_graph_pipeline.py      # LangGraph harness w/ per-node timings + F1
│   └── test_pipeline.py
├── docker/
├── deploy_and_run.sh               # DGX2 rsync + docker-compose
└── utils/
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_graph_pipeline.py` | LangGraph harness — runs a batch of PDFs and reports per-node timings, validation errors, non-blank field count, and F1 against gold labels. |
| `scripts/test_pipeline.py` | Legacy end-to-end smoke test |
| `scripts/grade_extraction.py` | Compare predictions vs gold labels |
| `scripts/debug_cms1500_alignment.py` | Single-file alignment diagnostics |
| `scripts/tune_cms1500_thresholds.py` | Registrar threshold grid search |
| `scripts/recompute_thresholds.py` | Threshold suggestions from corrections log |
| `scripts/api_client.py` | CLI client for REST endpoints |
| `deploy_and_run.sh` | DGX2 deployment (rsync + docker-compose for backend + frontend) |

## Measured performance (DGX2, April 2026)

| Fixture | Plan reason | Alignment Q | Wall-clock | Non-blank fields |
|---|---|---:|---:|---:|
| `cms1500.pdf`   | `lane_b_downgraded_to_c` → Lane C | 0.62 | 41 s | 22 |
| `cms1500_1.pdf` | Lane A (widgets) + rescue | n/a | 38 s | 37 |
| `cms1500_2.pdf` | Lane C scan+OCR              | 0.74 | 47 s | 29 |
| `cms1500_3.pdf` | Lane C scan+OCR              | 0.58 | 41 s | 24 |
| `cms1500_6.pdf` | Lane C scan+OCR              | 0.81 | 70 s | 41 |

Before the April 2026 refactor the same fixtures took 3-5 minutes each and `cms1500_3.pdf` was taking **258 s** of wasted Florence-2 time on a misaligned page — the alignment gate now short-circuits that path in under a second.

## Known limitations

- Current pipeline processes **page 1** only.
- CMS-1500 scan quality still depends heavily on registration quality and handwriting clarity.
- First run can be slow due to Florence-2 / Ollama warmup.
- Ollama-dependent features (SLM/VLM rescue, Box 24 table extraction) require local model availability.

## Additional Documentation

| Document | Description |
|---------|-------------|
| `documents/PIPELINE_OVERVIEW.md` | Architecture, Mermaid diagrams, script-to-function mapping, OCR pipeline details |
| `documents/SCRUM_REPORT.md` | Structured process flow narrative, component status, risks, and roadmap (for manager review) |
| `documents/system_prompt.md` | Engineering principles for reliability and traceability |

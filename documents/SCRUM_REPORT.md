# Doc2Data - SCRUM Report

## Sprint Update - February 5, 2026 (3-Lane CMS-1500 Pipeline Implemented)

### Major Change: 3-Lane Extraction Architecture

CMS-1500 extraction now uses a **priority-based 3-lane strategy**:

| Lane | When | Method | Accuracy |
|------|------|--------|----------|
| **A (best)** | Fillable PDF (AcroForm widgets present, >=10 filled) | Read widget `field_name`/`field_value` directly — **no OCR** | **~93%+ F1** (26/28 gold fields matched, remaining 2 are whitespace normalization) |
| **B (good)** | Digital PDF but flattened (no widgets, text layer present) | `page.get_text("words")` → zone match against calibrated schema bboxes | Good for top half; now improved for bottom half with recalibrated bboxes |
| **C (scan)** | Scanned/handwritten PDF (no text layer) | Align → template subtraction → full-page OCR → zone match → ICR fallback | Best effort; limited by alignment quality and handwriting OCR |

### What was implemented

1. **Lane A: AcroForm Widget Extraction** (`_extract_widgets` + `_map_widgets_to_schema` in `multi_agent_pipeline.py`)
   - Reads all ~270 widgets from the Cigna CMS-1500 PDF
   - Maps widget names (e.g., `tax_id`, `t_charge`, `amt_paid`, `pin`) to schema field IDs
   - Composes multi-part fields: DOB (mm/dd/yy), phone (area + number), header notes
   - Handles radio/checkbox widgets (sex, assignment, lab)
   - Normalizes values (collapse spaces around hyphens)
   - **Returns immediately without running OCR** when >= 10 fields are filled
   - Includes widget rect → pixel bbox conversion for UI overlay

2. **Schema Bbox Recalibration** (`data/schemas/cms-1500.json`)
   - All 48 field bboxes regenerated from actual AcroForm widget rectangles
   - **Critical fixes**: `25_federal_tax_id`, `28_total_charge`, `29_amount_paid`, `32_service_facility_name`, `33a_npi` were all pointing to wrong page regions
   - Bboxes now match the canonical Cigna CMS-1500 template exactly
   - Benefits Lane B (text layer zone matching) and Lane C (scan zone matching)

3. **Widget→Schema Mapping** (complete mapping for all 270 widgets)
   - Text fields: `pt_name`, `ins_name`, `pt_street`, `tax_id`, `t_charge`, `amt_paid`, `fac_name`, `doc_name`, `pin`, etc.
   - Radio/checkbox: `sex`, `ins_sex`, `assignment`, `lab`, `employment`
   - Service lines 1-6: dates, CPT codes, modifiers, diagnosis pointers, charges

### Test Results (Lane A on `cms1500.pdf`)

- **26/28 gold label fields matched** with zero OCR
- 2 "misses" were whitespace normalization differences (`ID10- 45678` vs `ID10-45678`) — now fixed
- Processing time: **< 1 second** (vs 52-223 seconds with OCR pipeline)
- **No LLM hallucination possible** (values come directly from PDF form fields)

### Root causes identified and fixed

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Bottom-half fields wrong (tax_id, total_charge, NPI) | Schema bboxes were calibrated to wrong page positions | Recalibrated all bboxes from widget rects |
| "member of South Carolina Senate" in output | LLM hallucination from SLM labeling rewriting values | Lane A skips LLM entirely; SLM guardrails prevent rewriting schema-matched fields |
| "altimore" instead of "Baltimore" | `remove_form_lines` inpainting erasing characters touching grid lines | Lane A doesn't use OCR/preprocessing at all; Lane C now does subtraction before de-lining |
| Slow processing (52-223s) | Running OCR + SLM + VLM on every document | Lane A returns in < 1s with no model inference |

---

## Sprint Update - February 5, 2026 (Runtime Codepath Audit + Root Cause Analysis)

### What is actually running on DGX (persistent services)

- **Container entrypoint**: `start_services.sh`
  - **FastAPI**: `uvicorn app.api_main:app --host 0.0.0.0 --port 8000` (background, 2 workers)
  - **Streamlit**: `streamlit run app/streamlit_main.py --server.address 0.0.0.0 --server.port 8501` (foreground to keep container alive)
- **Important behavior**: Streamlit currently runs extraction **in-process** (imports pipeline modules directly). It does **not** call the FastAPI service for extraction.

### Active extraction modes (Streamlit UI)

- **Multi-Agent (Recommended)**: `src/pipelines/multi_agent_pipeline.py::MultiAgentPipeline.process_sync`
  - Used for most testing and the `/extract/*` API endpoints.
- **CMS-1500 (Agentic)**: `src/pipelines/agentic_cms1500.py::run_cms1500_agentic`
  - Alternate, CMS-specific agent pipeline (registration → zones → tiered OCR → optional LLM extraction).
- **General Pipeline**: `src/pipelines/form_extractor.py::extract_with_full_pipeline`
  - “Read → Understand → Ground” LLM-based extractor.
  - **Risk**: can hallucinate if outputs are not strictly grounded to OCR text/boxes.

### Active endpoints (FastAPI)

- **Primary extraction endpoints (Multi-Agent)**:
  - `POST /extract/v2` → `MultiAgentPipeline.process`
  - `POST /extract/cms1500` → `MultiAgentPipeline.process` (wrapped into a CMS response schema)
  - `POST /extract/reducto` → `MultiAgentPipeline.process` → return only `reducto_format`
- **Utility endpoints (legacy modular pieces)**:
  - `POST /extract/generic` → `ingest_document` + `LayoutSegmenter` (returns layout blocks only)
  - `POST /ingest` → `ingest_document` (returns page image + metadata; cached)
  - `POST /segment` → `LayoutSegmenter.segment_page` (cached)
  - `POST /ocr` → **currently a stub** (returns cached “completed”, does not run `OCRPipeline` yet)

### Script / module inventory (what is used vs legacy)

#### Production entrypoints (always used when the container is running)

| Path | Used by | What it solves |
|------|---------|----------------|
| `start_services.sh` | Docker container | Starts FastAPI + Streamlit immediately; keeps container alive via Streamlit process; optionally starts Ollama. |
| `app/api_main.py` | FastAPI | HTTP API layer for extraction; uses `MultiAgentPipeline` for `/extract/*`; exposes legacy utility endpoints (`/ingest`, `/segment`). |
| `app/streamlit_main.py` | Streamlit | UI for upload/preview/overlay/export; directly executes extraction pipelines in-process. |

#### Active core extraction (Multi-Agent path)

| Path | Used by | What it solves |
|------|---------|----------------|
| `src/pipelines/multi_agent_pipeline.py` | Streamlit (default), FastAPI `/extract/*` | End-to-end orchestrator: form ID → (CMS align + schema zones) or (general layout) → OCR → optional SLM/VLM labeling → validation → business mapping → Reducto export. |
| `src/processing/registration.py` | Multi-Agent, Agentic CMS | Reference template loading and homography utilities used for template alignment + zone projection. |
| `src/processing/preprocessing.py` | Multi-Agent, ingest | Deskew/denoise/contrast + CMS helpers (`remove_form_lines`, `extract_ink_by_subtraction`). |
| `src/ocr/paddle_ocr.py` | Multi-Agent, Agentic CMS, legacy OCR | PaddleOCR wrapper (PaddleX-compatible) returning word boxes with bboxes + confidences. |
| `src/pipelines/business_schema.py` | Multi-Agent, Agentic CMS | Maps schema-level field IDs → business keys; applies validators + normalization. |
| `src/pipelines/validators.py` | Multi-Agent, business_schema, legacy OCR | Regex-based validators/normalizers (NPI/ICD/date/phone/zip/money/member_id). |
| `utils/config.py` | All | Central config (paths, model settings, toggles). |
| `utils/models.py` | All | Shared data models (`Block`, `WordBox`, `Document`, `PageImage`). |

#### Optional (used only in specific UI/API modes)

| Path | Used by | What it solves |
|------|---------|----------------|
| `src/pipelines/agentic_cms1500.py` | Streamlit “CMS-1500 (Agentic)” | CMS-specific pipeline (registration → zones → OCR → optional LLM extraction). |
| `src/pipelines/cms1500_production.py` | Multi-Agent **optional flag** (off by default) | Production-style per-field extraction with ink subtraction + validators. Currently **disabled by default** in Multi-Agent because per-field crop OCR is brittle on scans; kept for experimentation. |
| `src/ocr/trocr_wrapper.py` | Agentic CMS, `cms1500_production.py` | TrOCR handwriting OCR wrapper for harder handwritten fields/signatures. |
| `src/pipelines/form_extractor.py` | Streamlit “General Pipeline”, graph prototype | LLM extraction using OCR text context + grounding heuristic back to OCR boxes. |
| `src/pipelines/reducto_adapter.py` | Streamlit export tab | Adapts our result JSON to a Reducto-like JSON structure. |

#### Deployment / operations scripts (how we start/keep services running)

| Path | Used by | What it solves |
|------|---------|----------------|
| `deploy_and_run.sh` | DevOps (DGX) | Rsync code → build Docker image → run persistent container with `--restart unless-stopped` and GPU enabled. |
| `run.sh` | DevOps (DGX, legacy) | Older venv-based remote runner for Streamlit (non-Docker). Useful for quick experiments; not the current production method. |
| `docker/docker-compose.yml` | Local dev | Runs FastAPI + Streamlit as separate services (no GPU config here). |
| `run_docker_gpu.sh` / `run_docker_cpu.sh` | Local dev | Build and run the Docker image locally (GPU/CPU variants). |

#### Legacy / partially wired modules (present, but not on the main production path)

| Path | Current status | Notes |
|------|----------------|------|
| `src/pipelines/ingest.py` | **Used by FastAPI utility endpoints** | Handles PDF render + digital layer extraction + preprocessing layers. Not used by Multi-Agent (Multi-Agent has its own ingest in-file). |
| `src/pipelines/segment.py` | **Used by FastAPI utility endpoints** | LayoutParser-based segmentation + heuristics; separate from Multi-Agent’s internal LayoutDetectionAgent. |
| `src/pipelines/yolo_layout.py` | **Used by** `segment.py` | YOLOv8 detector helper for segmentation. |
| `src/pipelines/ocr.py` | **Not used by API endpoints currently** | Full OCR pipeline exists, but FastAPI `/ocr` endpoint is currently stubbed. Used only in tests/eval. |
| `src/pipelines/slm_label.py` | **Tests only** | Separate SLM labeler used by the legacy modular pipeline tests; Multi-Agent uses an internal `LabelingAgent`. |
| `src/pipelines/assemble.py`, `table_processor.py`, `figure_processor.py` | **Tests/eval only** | Document assembly and table/figure enrichment exist, but Multi-Agent returns its own output and does not call `DocumentAssembler` today. |

#### Unused / deprecated (safe to ignore for production)

| Path | Why it’s unused / risky |
|------|--------------------------|
| `src/pipelines/cms1500_direct.py` | Not imported by Streamlit/FastAPI. Implements hardcoded template zones (the exact “works for one sample” failure mode). Keep only for historical reference or delete. |
| `src/pipelines/graph_pipeline.py` | Not imported by Streamlit/FastAPI. Prototype orchestrator; duplicates functionality and increases maintenance surface. |
| `fill_schema` (import in `app/api_main.py`) | File does not exist in repo; API falls back to stubs. Legacy placeholder. |

#### Evaluation / training / debug utilities (not used by production services unless manually run)

| Path | What it solves |
|------|----------------|
| `scripts/grade_extraction.py` | Grades `data/results/*.json` vs `data/gold_labels/*.json` (F1 + exact match). |
| `scripts/test_pipeline.py` | Smoke-test for Multi-Agent pipeline wiring (agents load + sample run). |
| `verify_gpu_pipeline.py` | Quick GPU availability + pipeline sanity check; writes `verify_result.json`. |
| `scripts/train_yolo_gpu.py` / `training/train_yolo_cms1500.py` | YOLO fine-tuning utilities (model training, dataset prep). |
| `scripts/augment_yolo_dataset.py` / `training/prepare_dataset*.py` | Dataset augmentation + preparation for training. |
| `eval/validate_layout.py` / `eval/validate_ocr.py` / `validation/eval_cms1500.py` | Ad-hoc evaluation helpers (not part of runtime inference). |

### Root cause analysis (current blockers)

#### A) CMS-1500 scanned/handwritten alignment is still not reliable

- **What we do today** (Multi-Agent): `TemplateAlignmentAgent` attempts a stable outer-quad warp first, then falls back to structure/feature matching.
- **Why it still fails on some scans**:
  - **Outer boundary is not always detectable** (cropped margins, fax noise, shadowing), so the quad detector can lock onto the *wrong* rectangle (inner box/grid), yielding a plausible-but-wrong homography.
  - **Global homography cannot fix local warp** from camera/fax distortions (non-planar paper bends). Even “high alignment_quality” can still be wrong for small zones.
  - **Quality score is not anchored to form-specific landmarks** (e.g., known printed anchors), so it can overestimate correctness.

#### B) Missing leading characters (example: “Baltimore” → “altimore”)

Likely causes in the current stack (not mutually exclusive):

- **Aggressive form line removal** (`remove_form_lines`) uses morphology + inpainting, which can **erase strokes that touch grid lines** (common on CMS-1500).
- **LLM post-cleaning can drop characters** if allowed to rewrite values (especially when the OCR text is noisy and the model “cleans” too aggressively).
- **Small residual misalignment** can shift the effective zone so the first character falls just outside the matching region (shows up as systematic leading-character loss).

#### C) General forms: extra / “random” words that do not exist in the document

- This is most consistent with **LLM hallucination** entering the final output when:
  - An LLM is allowed to generate “clean values” that are **not grounded** in OCR text, and we accept it as truth.
  - Semantic labeling overwrites `block.text` or creates colliding keys, causing confusing downstream JSON.

#### D) Wrong name extracted on scanned/handwritten (e.g., “Rahul Sharma” when PDF shows a different name)

- **Alignment/zone bleed**: if alignment is off, the patient-name zone can overlap printed labels/neighbor fields, pulling the wrong words.
- **Printed-label bleed**: CMS-1500 has dense pre-printed text; without perfect suppression, OCR detects labels and nearby words.
- **LLM rewriting risk**: if SLM labeling is enabled and allowed to rewrite values, it can “correct” toward common names seen elsewhere in the document.
- **Digital layer mismatch** (already mitigated): some PDFs have a hidden/incorrect text layer; we now validate the layer against visible pixels, but edge cases can still slip through.

### Fixes applied today (low-risk guardrails)

- **SLM guardrails**: prevent the SLM from rewriting CMS-1500 schema/zone-matched fields (keeps schema IDs stable and avoids value corruption).
- **Grounding check for SLM clean_value**: only accept the model’s “clean_value” if it is textually grounded in the original OCR text (prevents hallucinated values).
- **CMS scan OCR order fix**: do **template-subtraction ink masking first**; only fall back to `remove_form_lines` if subtraction is unavailable. This avoids breaking the subtraction diff and reduces character loss from inpainting.

## Sprint Update - January 13, 2026 (Architecture Verification)

### Current Status

- **Deployment**: DGX services are running and accessible via Tailscale.
- **Architecture**: All major components verified and working:
  - ✅ **Detectron2** (PubLayNet model) - Working for general layout detection
  - ✅ **YOLO** (fine-tuned cms1500 model) - Working for CMS-1500 detection
  - ✅ **PaddleOCR** - Working for text extraction
  - ✅ **TrOCR** - Available as handwriting fallback
  - ✅ **Ollama** - 2 models loaded (llama3.2:3b, qwen2.5:7b-instruct)
- **UB-04 Test Results**:
  - Form Type: UB-04 ✓ (correctly identified)
  - Fields Found: 73/73
  - Avg Confidence: 84%
  - Processing Time: ~8.5 minutes (needs optimization)

### Fixes Applied Today

1. **Detectron2 Model Loading**: Fixed iopath caching issue by specifying `model_path` parameter to use local weights directly.
2. **Ollama Models**: Copied host models to container (`~/.ollama/models`).
3. **Test Script**: Created `/app/scripts/test_pipeline.py` for architecture verification.

### Model Paths (Container)

| Model | Path | Status |
|-------|------|--------|
| Detectron2 | `/root/.detectron2/models/publaynet_faster_rcnn_R_50_FPN_3x.pth` | ✅ |
| YOLO | `/app/models/cms1500_yolo_v1.pt` | ✅ |
| Ollama | `/root/.ollama/models/` | ✅ |

### Grader Results (CMS-1500)

| File | F1 Score |
|------|----------|
| cms1500 (digital) | 0.44 |
| cms1500_handwritten | 0.11 |
| cms1500_1 | 0.06 |
| cms1500_2 | 0.06 |
| cms1500_3 | 0.05 |
| cms1500_4 | 0.08 |
| **Overall** | **0.1345** |

### Scripts Reference

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/test_pipeline.py` | Tests full pipeline architecture | ✅ Active |
| `scripts/grade_extraction.py` | Computes F1 score vs gold labels | ✅ Active |
| `scripts/train_yolo_gpu.py` | YOLO training script | ✅ Active |
| `scripts/augment_yolo_dataset.py` | Data augmentation for YOLO | ✅ Active |
| `app/streamlit_main.py` | Streamlit UI | ✅ Active |
| `app/api_main.py` | FastAPI endpoints | ✅ Active |

### Known Issues

1. **CMS-1500 scanned form accuracy is low** (~6-11% F1) - needs better OCR post-processing
2. **Processing time is slow** (2-8 minutes per document) - needs optimization
3. **Some extracted values contain form labels** - needs better value isolation

---

## Sprint Update - January 7, 2026 (End-to-End Status)

### Current Status (High-level)

- **Deployment**: DGX services are up (FastAPI + Streamlit) and stable with Docker restart policy.
- **Pipeline behavior**: CMS-1500 is no longer “hardcoded to one sample” — it now **chooses an extraction strategy** based on whether the PDF is truly digital vs scanned.
- **Main blocker**: **Handwritten / scanned CMS-1500 accuracy is still not acceptable** for critical fields (patient name, insured ID, etc.). We are extracting *something*, but it often contains **printed labels + nearby field bleed**, not clean values.

### Live URLs (Tailscale)

| Service | URL | Description |
|---------|-----|-------------|
| **FastAPI** | http://100.126.216.92:8000 | REST API endpoints |
| **API Docs** | http://100.126.216.92:8000/docs | Swagger UI |
| **Streamlit** | http://100.126.216.92:8501 | Web UI |

### What We Achieved So Far (Real, verified changes)

#### 1) CMS-1500 “Rahul vs Rohit” bug: fixed at the source
- **Root cause**: some PDFs contain a **hidden/template digital text layer** that does not match visible pixels.
- **Fix**: added **visual validation** for the digital layer and only use it if it looks like real filled values (anchor-field QA check).
- **Result**: scanned CMS-1500 documents now **force visual OCR** (no more reading hidden text as truth).

#### 2) Template alignment: stabilized (no “tilted straight templates”)
- **Root cause**: unstable alignment refinements can introduce tilt/warp on already-straight templates.
- **Fix**: replaced the fragile alignment path with a **safer perspective/corner-based alignment** + stricter homography validation (reject warp/perspective drift).
- **Additional fix**: for CMS-1500 scans, we now **skip deskew before alignment** (deskew on handwriting can rotate the page incorrectly; alignment should handle rotation instead).

#### 3) CMS-1500 scanned extraction: removed “single-sample hardcoding”
- **Old failure mode**: applying schema boxes in fixed pixel locations even when alignment is wrong → totally incorrect crops.
- **Current behavior**:
  - If **digital layer is validated** → match zones using digital words (best case).
  - Else (scan):
    - Attempt alignment; if alignment quality is high → **full-page OCR once**, then **zone-match words** into schema fields.
    - If alignment is not reliable → fallback to **full-page OCR + line grouping** (no fake schema boxes).

#### 4) General forms “giant box” failure mode: mitigated
- **Root cause**: layout detection sometimes returns 0–1 blocks or one near-full-page FIGURE block.
- **Fix**: drop giant blocks and **fallback to OCR line grouping** when layout is too coarse (<3 blocks).

### What’s Still Broken (Current Issues)

#### A) CMS-1500 handwritten/scanned field values are noisy / incorrect
Even with alignment success, some fields contain:
- **Pre-printed labels** (e.g., “PATIENT’S NAME …”) mixed into the value.
- **Zone bleed / cross-field leakage** (neighboring text is pulled into the zone).
- **Handwriting OCR errors** (PaddleOCR misses strokes; TrOCR can hallucinate on weak crops).

Example (from latest DGX run on `cms1500_handwritten.pdf`):
- Alignment quality ~0.98 (good).
- Extracted `2_patient_name` still contains label text + nearby address text.
- Extracted `1a_insured_id` contains label text + name bleed.

#### B) “It runs in a loop / no results” symptom (why it looks stuck)
This is not an infinite loop; it’s repeated work:
- **Many OCR calls**: without careful gating, the pipeline can OCR **per-field crops** (48 CMS-1500 zones), which is slow and spams logs.
- **Model init spam**: Paddle/PaddleX prints “model files already exist / using cached files” repeatedly; TrOCR loads transformers and logs warnings.
- **Connectivity check**: PaddleX prints “Checking connectivity to the model hosters…” which can add delay/noise even if models are cached.

**What we changed to reduce this:**
- Prefer **full-page OCR once + zone matching** for CMS-1500 scans (reduces N× OCR calls).
- Added guards to avoid re-OCR when a block already has text from zone matching (still needs tightening for schema-zone blocks and for empty-field handling).

### Root Causes (Technical)

1. **Deskew before alignment on handwriting** can introduce a small rotation error → causes alignment/matching instability.
2. **Per-field crop OCR** is fragile:
   - tiny crops lose context;
   - printed labels dominate;
   - overlapping zones create duplicates/bleed;
   - OCR engines behave poorly on low-ink/noisy crops.
3. **Digital text layer can be wrong** (template-only/hidden text), so using it blindly creates wrong names/IDs.
4. **Zone matching needs stronger de-bleed**:
   - stricter word assignment / overlap rules,
   - better suppression of printed form text (template-diff at full-page scale),
   - field-specific cleanup rules.

### Next Steps (Required for “proper results”)

#### CMS-1500 (priority)
- **Value isolation**: use **template-diff / printed-text suppression** at scale (not just per-crop) so the OCR words fed into zone matching are mostly “ink” (filled values), not labels.
- **De-bleed**: strengthen the zone matching assignment (unique assignment + intersection-over-area thresholds) and reduce zone padding for problematic fields.
- **Field-specific parsing**: parse name/id/date fields with stricter regex/validation and reject garbage.
- **Handwriting strategy**: TrOCR should be used only when there is strong ink and Paddle is clearly failing; otherwise it hallucinates.

#### End-to-end verification (still pending)
- Re-verify via Streamlit/API for:
  - **CMS1500 scan** (handwritten),
  - **UB-04 sample**,
  - **TCCC sample**.

### Repro / Debug Commands (DGX)

```bash
# Connect to DGX
ssh -i ../../dgx-spark/tailscale_spark2 radiant-dgx2@100.126.216.92

# Logs
docker logs -f doc2data-server

# Quick pipeline test inside container
docker exec doc2data-server python3 -c "from src.pipelines.multi_agent_pipeline import MultiAgentPipeline,PipelineConfig; p=MultiAgentPipeline(PipelineConfig(enable_slm_labeling=False,enable_vlm_figures=False,enable_alignment=True)); r=p.process_sync('/app/data/sample_docs/cms1500_handwritten.pdf'); print(r.get('form_type'), r.get('alignment_quality')); ef=r.get('extracted_fields') or {}; print('2_patient_name', ef.get('2_patient_name')); print('1a_insured_id', ef.get('1a_insured_id'))"
```

---

## Sprint Update - December 27, 2025

### ✅ DEPLOYMENT COMPLETE - SERVER IS LIVE!

#### 🚀 Access URLs (For Team Members on Tailscale Network)

| Service | URL | Description |
|---------|-----|-------------|
| **FastAPI** | http://100.126.216.92:8000 | REST API endpoints |
| **API Docs** | http://100.126.216.92:8000/docs | Swagger UI - Interactive API documentation |
| **Streamlit** | http://100.126.216.92:8501 | Web UI for document processing |

#### 📡 API Endpoints

1. **POST `/extract/reducto`** - Extract and get Reducto-compatible JSON output
2. **POST `/extract/v2`** - Full extraction with all metadata
3. **POST `/extract/cms1500`** - CMS-1500 specific extraction
4. **POST `/extract/generic`** - Generic document extraction

#### 🔗 How to Share with Team

**Prerequisites:** Team members must be on the **Tailscale network (radiantt.com)**

1. Install Tailscale: https://tailscale.com/download
2. Login with: rahul370139@gmail.com (or team admin)
3. Access: http://100.126.216.92:8000/docs

---

### ✅ COMPLETED TASKS

#### 1. Detectron2 Configuration Fixed
- **Issue:** Was using wrong config `ppyolov2_r50vd_dcn_365e` (PaddleDetection) instead of `faster_rcnn_R_50_FPN_3x` (Detectron2)
- **Fix:** Updated LayoutDetectionAgent to use correct Detectron2 config with PaddleDetection fallback

#### 2. OCR Pipeline Verified
- **Test Result:** 47 fields extracted from CMS-1500 sample
- **Reducto Format:** Working with 48 blocks
- **Digital Text Layer:** Validated (score 3/3)

#### 3. Preprocessing Improvements
- Red-line removal for CMS-1500 forms
- Deskewing up to 15 degrees for scanned documents
- Template alignment with SIFT/ORB + ECC refinement

#### 4. Persistent Hosting on DGX
- Docker container running with `--restart unless-stopped`
- FastAPI starts immediately (no blocking on model downloads)
- Ollama models download in background

#### 5. Multi-Agent Pipeline Architecture
- Form identification (CMS-1500, UB-04, TCCC, Generic)
- Template alignment for known forms
- Tiered OCR (PaddleOCR + TrOCR fallback)
- Schema-zone matching for accurate field extraction

---

### 📊 Test Results

```
✅ Form Type: cms-1500
✅ Fields Extracted: 47
✅ Reducto format present
   - Chunks: 1
   - Blocks: 48
✅ Digital text layer validated (score 3/3)
```

---

### 🔧 Useful Commands (SSH to DGX)

```bash
# Connect to DGX
ssh -i ../../dgx-spark/tailscale_spark2 radiant-dgx2@100.126.216.92

# View logs
docker logs -f doc2data-server

# Restart server
docker restart doc2data-server

# Stop server
docker stop doc2data-server
```

---

### 📁 Project Structure

```
doc2data/
├── app/
│   ├── api_main.py         # FastAPI endpoints
│   └── streamlit_main.py   # Streamlit UI
├── src/
│   ├── pipelines/
│   │   ├── multi_agent_pipeline.py  # Core extraction logic
│   │   └── business_schema.py       # Schema mapping
│   ├── processing/
│   │   └── preprocessing.py         # Image enhancement
│   └── ocr/
│       └── paddle_ocr.py            # PaddleOCR wrapper
├── data/
│   ├── schemas/cms-1500.json        # Field definitions
│   └── sample_docs/                 # Test documents
├── docker/                 # Docker configuration
│   ├── Dockerfile
│   ├── start_services.sh   # Service startup script
│   ├── requirements_docker.txt
│   └── docker-compose.yml
├── deploy_and_run.sh       # Deployment automation
```

---

### 🐍 Python API Client Example

```python
import requests

# Upload a PDF and get Reducto-style output
url = "http://100.126.216.92:8000/extract/reducto"
with open("your_cms1500.pdf", "rb") as f:
    response = requests.post(url, files={"file": f})
    
result = response.json()
print(result["result"]["chunks"][0]["content"])
```

---

### ⚠️ Known Limitations

1. **Handwritten Forms:** OCR accuracy may vary for heavily handwritten content
2. **VLM/SLM:** Requires Ollama model download (~2GB) - runs in background
3. **GPU:** Optimized for NVIDIA DGX with GPU acceleration

---

### 📝 Next Steps (Optional Improvements)

1. [ ] Add authentication to API
2. [ ] Set up public HTTPS access with cloudflared tunnel
3. [ ] Add batch processing endpoint
4. [ ] Fine-tune YOLO model for better field detection

# Pipeline Overview — Scripts, Purpose, and Architecture

This document describes every script and module in the Doc2Data pipeline: what it does, why it exists, and how it fits together.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Scripts Reference (Every File)](#scripts-reference-every-file)
3. [Pipeline Modules](#pipeline-modules)
4. [App Layer](#app-layer)
5. [Utils and Helpers](#utils-and-helpers)
6. [Configuration and Data](#configuration-and-data)

---

## High-Level Architecture

```
PDF/Image → Form ID → [Lane A | Lane B | Lane C] → Layout → OCR → Validation → JSON
```

- **Lane A:** Fillable PDF → AcroForm widgets (no OCR)
- **Lane B:** Digital PDF → embedded text layer + zone matching
- **Lane C:** Scanned → template alignment + OCR + zone matching

---

## Scripts Reference (Every File)

### `app/streamlit_main.py`

**Purpose:** Streamlit web UI for document extraction.

**What it does:**
- Renders upload form, sample document selector, config options
- Calls `MultiAgentPipeline.process_sync()` when user runs extraction
- Displays results: JSON, business fields, annotated image, download buttons
- Handles form type override (CMS-1500, UB-04), alignment toggle, SLM/VLM toggles

**Use case:** Interactive demo and manual testing.

---

### `app/api_main.py`

**Purpose:** FastAPI REST API for document extraction.

**What it does:**
- Exposes `/extract/cms1500`, `/extract/ub04`, `/extract/v2`, `/health`
- Accepts PDF/image upload via multipart form
- Returns structured JSON with `extracted_fields`, `business_fields`, metadata
- Used by external clients, scripts, and integrations

**Use case:** Programmatic access, DGX deployment, API clients.

---

### `src/pipelines/multi_agent_pipeline.py`

**Purpose:** Main orchestrator for the multi-agent document pipeline.

**What it does:**
- Implements 3-lane extraction (Lane A: widgets, Lane B: digital text, Lane C: scanned)
- Chooses lane based on PDF type (AcroForm, text layer, or scanned)
- Runs `FormIdentificationAgent`, `TemplateAlignmentAgent`, `LayoutDetectionAgent`, `OCRAgent`, `LabelingAgent`, `ValidationAgent`
- Assembles final JSON with `extracted_fields`, `business_fields`, `field_details`
- Includes VLM rescue pass for low-confidence fields
- Exposes `process_sync(path)` for synchronous usage

**Use case:** Core entry point for extraction. Used by Streamlit, API, and CLI.

---

### `src/pipelines/core/base.py`

**Purpose:** Base class for all pipeline agents.

**What it does:**
- Defines `BaseAgent` abstract class with `initialize()`, `process()`, `log()`
- All agents (form_id, alignment, layout, ocr, labeling, validation) inherit from it

**Use case:** Shared interface for agents.

---

### `src/pipelines/core/models.py`

**Purpose:** Shared data models and enums.

**What it does:**
- `FormType`: CMS1500, UB04, NCPDP, GENERIC, UNKNOWN
- `BlockType`: TEXT, TABLE, FIGURE, CHECKBOX, SIGNATURE, etc.
- `DetectedBlock`: id, block_type, bbox, confidence, text, metadata
- `FormIdentification`: form_type, confidence, version
- `AlignmentResult`: success, aligned_image, homography_matrix, quality
- `PipelineConfig`: all pipeline options (enable_trocr, enable_alignment, etc.)

**Use case:** Type consistency across pipeline.

---

### `src/pipelines/agents/form_identification.py`

**Purpose:** Identifies the form type (CMS-1500, UB-04, etc.).

**What it does:**
- OCRs header region
- Matches tokens (CMS-1500, UB-04, HCFA, etc.)
- Optionally uses layout fingerprint (checkbox density, structure)
- Returns `FormIdentification` with form_type and confidence

**Use case:** First step to decide which processing path to use.

---

### `src/pipelines/agents/template_alignment.py`

**Purpose:** Aligns scanned form image to reference template.

**What it does:**
- For CMS-1500: calls `cms1500_register.register()` (classical CV)
- For other forms: tries optional YOLO boundary detection, fallback to ORB/feature matching
- Returns `AlignmentResult` with aligned image and homography matrix
- Used for Lane C (scanned forms) before zone-based OCR

**Use case:** Correct geometric distortion before OCR.

---

### `src/pipelines/agents/layout_detection.py`

**Purpose:** Detects layout blocks (text, table, figure, form_field, checkbox).

**What it does:**
- For CMS-1500: uses YOLOv8 (if configured) or template zones from schema
- For general forms: uses Detectron2 (PubLayNet) or PaddleDetection
- Returns list of `DetectedBlock` with bbox, type, confidence

**Use case:** Segment page into regions for OCR and labeling.

---

### `src/pipelines/agents/ocr.py`

**Purpose:** Performs OCR on extracted blocks.

**What it does:**
- PaddleOCR for printed text
- TrOCR for handwriting/signatures
- Checkbox state via fill-ratio heuristic
- Field-type validation (regex) and normalization
- Escalation to VLM when confidence is low
- Returns text + confidence per block

**Use case:** Convert image regions to text.

---

### `src/pipelines/agents/labeling.py`

**Purpose:** Adds semantic labels to blocks (SLM).

**What it does:**
- Uses Ollama (Qwen, etc.) to label text blocks (e.g., "patient_name", "diagnosis")
- Supports table and figure processing (TATR, VLM)
- Returns blocks with `field_type` metadata

**Use case:** Map raw text to schema field IDs.

---

### `src/pipelines/agents/validation.py`

**Purpose:** Validates extracted fields and runs LLM QA.

**What it does:**
- Uses `validators.py` for format checks (NPI, date, phone, ICD, etc.)
- Returns validation errors and warnings
- Optional LLM QA check on extracted data

**Use case:** Catch format errors and inconsistencies.

---

### `src/pipelines/registration/cms1500_register.py`

**Purpose:** CMS-1500 template alignment (classical CV only).

**What it does:**
- Loads template, builds dropout-red mask, structural line mask
- AKAZE/ORB feature matching + RANSAC homography
- Quad fallback when feature quality is low
- Warps input to canonical template space
- Returns `CMS1500RegistrationResult` with aligned image and homography

**Use case:** Deterministic alignment for scanned CMS-1500 forms.

---

### `src/processing/registration.py`

**Purpose:** Template loading and reference resolution.

**What it does:**
- `get_reference_image_path()`: resolves path to template (CMS-1500, UB-04)
- `load_and_process_reference()`: loads template, computes ORB features (or delegates to cms1500_register for CMS-1500)
- Caches template data for reuse

**Use case:** Provide template data to alignment and zone matching.

---

### `src/processing/preprocessing.py`

**Purpose:** Image preprocessing utilities.

**What it does:**
- `remove_red_template_text()`: removes red template text (HSV) for better OCR
- `preprocess_image()`: deskew, denoise, contrast, binarization
- Used before OCR and alignment

**Use case:** Improve OCR quality on scanned forms.

---

### `src/pipelines/validators/field_validators.py`

**Purpose:** Field-level format validation and normalization.

**What it does:**
- Validators: NPI (checksum), NDC, ICD, HCPCS, CPT, date, phone, member_id, SSN, zip, money, tax_id
- `validate_field()`: run validator by type, return (passed, info)
- `guess_field_type()`: heuristic mapping from label text to validator type

**Use case:** Ensure extracted values match expected formats.

---

### `src/pipelines/schemas/business_schema.py`

**Purpose:** Maps OCR/schema fields to business-friendly keys.

**What it does:**
- `map_to_business_schema()`: maps `2_patient_name` → `patient_name`, etc.
- Applies validators, composes addresses, sex from checkboxes
- `merge_business_with_ocr()`: merges business schema with raw OCR output

**Use case:** Produce clean JSON for downstream systems.

---

### `src/ocr/paddle_ocr.py`

**Purpose:** PaddleOCR wrapper.

**What it does:**
- Wraps PaddleOCR for text detection and recognition
- Returns word boxes and text for given image regions

**Use case:** Primary OCR engine for printed text.

---

### `utils/config.py`

**Purpose:** Central configuration.

**What it does:**
- Loads env vars (OLLAMA_HOST, USE_GPU, YOLO_MODEL_PATH, etc.)
- Defines PROJECT_ROOT, model paths, cache paths

**Use case:** Single source for all config.

---

### `utils/corrections.py`

**Purpose:** Correction logging and threshold tuning.

**What it does:**
- Logs user corrections to `data/corrections.jsonl`
- `auto_tune_thresholds()`: recomputes thresholds from corrections

**Use case:** Improve pipeline over time from user feedback.

---

### `utils/models.py`

**Purpose:** Shared data models (outside pipelines).

**Use case:** Common structures used across app and utils.

---

### `utils/cache.py`

**Purpose:** Caching utilities.

**Use case:** Cache model loads, OCR results, etc.

---

### `scripts/api_client.py`

**Purpose:** CLI client for Doc2Data API.

**What it does:**
- `--health`: health check
- `document.pdf`: upload and extract, print result

**Use case:** Test API from command line.

---

### `scripts/test_pipeline.py`

**Purpose:** Full pipeline integration test.

**What it does:**
- Loads sample PDF, runs `MultiAgentPipeline.process()`, prints results

**Use case:** Verify pipeline works end-to-end.

---

### `scripts/grade_extraction.py`

**Purpose:** Grade predictions against gold labels.

**What it does:**
- Compares prediction JSONs to gold JSONs
- Computes F1, reports mismatches

**Use case:** Evaluate extraction quality.

---

### `scripts/debug_cms1500_alignment.py`

**Purpose:** Debug CMS-1500 alignment on a single PDF.

**What it does:**
- Runs `get_cms1500_registrar().register()` on input PDF
- Prints alignment result, quality, method
- Optionally saves aligned image

**Use case:** Diagnose alignment failures.

---

### `scripts/tune_cms1500_thresholds.py`

**Purpose:** Grid search for alignment thresholds.

**What it does:**
- Runs alignment on multiple PDFs with different threshold combinations
- Prints recommended `CMS1500_*` env vars

**Use case:** Tune alignment for handwritten scans.

---

### `scripts/recompute_thresholds.py`

**Purpose:** Recompute thresholds from corrections.

**What it does:**
- Reads `data/corrections.jsonl`
- Calls `auto_tune_thresholds()`, prints updated thresholds

**Use case:** Update thresholds after user corrections.

---

### `deploy_and_run.sh`

**Purpose:** Deploy Doc2Data to DGX server.

**What it does:**
- Rsyncs project to DGX
- Builds Docker image
- Runs container with GPU, restart policy

**Use case:** Production deployment.

---

## Pipeline Modules

### Module Summary

| Module | Path | Purpose |
|--------|------|---------|
| Core | `src/pipelines/core/` | BaseAgent, models, PipelineConfig |
| Agents | `src/pipelines/agents/` | Form ID, alignment, layout, OCR, labeling, validation |
| Registration | `src/pipelines/registration/` | CMS-1500 alignment |
| Validators | `src/pipelines/validators/` | Field format validation |
| Schemas | `src/pipelines/schemas/` | Business schema mapping |

---

## App Layer

| File | Purpose |
|------|---------|
| `app/streamlit_main.py` | Streamlit UI |
| `app/api_main.py` | FastAPI REST API |

---

## Utils and Helpers

| File | Purpose |
|------|---------|
| `utils/config.py` | Configuration |
| `utils/corrections.py` | Correction logging, threshold tuning |
| `utils/models.py` | Shared models |
| `utils/cache.py` | Caching |

---

## Configuration and Data

| Path | Purpose |
|------|---------|
| `data/sample_docs/` | Sample PDFs |
| `data/schemas/` | CMS-1500, UB-04 field definitions |
| `data/gold_labels/` | Ground truth for grading |
| `data/templates/` | Template boxes |
| `data/thresholds.json` | Tuned thresholds |
| `data/corrections.jsonl` | User correction log |
| `docker/` | Dockerfile, compose, requirements |

---

## Models in Use

| Purpose | Model |
|---------|-------|
| CMS-1500 Layout | YOLOv8 (optional, fine-tuned) |
| General Layout | Detectron2 PubLayNet, PaddleDetection |
| Printed OCR | PaddleOCR |
| Handwriting OCR | TrOCR |
| Table Structure | TATR (Table Transformer) |
| Semantic Labels | Qwen (SLM) via Ollama |
| Figure Analysis | MiniCPM-V (VLM) via Ollama |

---

## Template Alignment Code Map

| File | Purpose |
|------|---------|
| `src/pipelines/registration/cms1500_register.py` | Core CMS-1500 alignment (classical CV) |
| `src/processing/registration.py` | Template loading, delegates to cms1500_register for CMS-1500 |
| `src/pipelines/agents/template_alignment.py` | Orchestrates alignment, calls cms1500_register for CMS-1500 |

---

## Quick Reference: What Calls What

```
Streamlit/API → multi_agent_pipeline.process_sync()
    → form_id_agent.process()
    → layout_agent.process() [uses layout_detection.py]
    → ocr_agent.process() [uses paddle_ocr, validators]
    → validation_agent.process() [uses validators]
    → map_to_business_schema() [uses schemas/business_schema.py]

For Lane C (scanned):
    → template_alignment_agent.process() [uses cms1500_register]
    → registration.load_and_process_reference() [for template]
```

# CMS-1500 Pipeline Optimization Report

## ✅ Completed Tasks

### 1. **Code Completeness & Fixes**
- **`segment.py`**: Fixed incomplete `_clean_layout_heuristics` function and indentation errors.
- **`segment.py`**: Fixed critical bug in `_load_template_schema` where path resolution was incorrect (was looking in `src/data` instead of `data`), causing 0 schema fields to be detected.
- **`ocr.py`**: Fixed logic bug where `_link_form_field` was overwriting existing schema metadata. It now preserves `schema_id` and `label_text` from the layout phase.
- **`assemble.py`**: Added `export_filled_schema` method to generate a flat JSON key-value pair output (`schema_id` -> `value`) suitable for form filling.
- **Dependencies**: Added `timm` to `requirements.txt` to fix Table Transformer (TATR) loading errors.

### 2. **VLM Integration Check**
- **Status**: **Integrated Correctly in Code**, but Runtime Issue.
- **Verification**: Verified `qwen_vl.py` and `ollama_client.py`.
- **Issue**: During testing, the VLM client attempted to connect to `localhost:80` (connection refused). The configuration defaults to `localhost:11434`. This suggests the environment variable `OLLAMA_HOST` might be set to `localhost` without a port in your local shell, or Ollama is not running.
- **Action**: Please ensure Ollama is running (`ollama serve`) and `OLLAMA_HOST` is set to `localhost:11434` or unset (to use default).

### 3. **CMS-1500 Fine-Tuning & Schema Filling**
- **Methodology**:
    - Used the existing `cms-1500.json` schema as the "proper schema" baseline.
    - Created `fill_schema.py` to run the pipeline with:
        - `enable_form_geometry=True`
        - `template_hint="cms-1500"`
        - `heuristic_strictness=0.6`
    - This forces the segmenter to snap detected blocks to the CMS-1500 grid defined in the schema.
- **Results**:
    - **Before Fix**: 1 schema field extracted.
    - **After Fix**: **29 schema fields extracted** from `cms1500.pdf` (filled form).
    - **Output**: Generated `cms1500_result_filled.json` containing the extracted values mapped to schema IDs (e.g., `{"1": true, "2_patient_name": "JOHN DOE", ...}`).

## 📂 Deliverables

1. **`doc2data/fill_schema.py`**: A dedicated script to ingest a CMS-1500 PDF and output the filled JSON.
2. **`doc2data/cms1500_result_filled.json`**: The successful extraction result from your sample.
3. **Codebase Updates**: Committed fixes to `segment.py`, `ocr.py`, `assemble.py`, and `requirements.txt`.

## 🚀 How to Run

To process any new CMS-1500 form:

```bash
# 1. Ensure Ollama is running (if using VLM)
ollama serve &

# 2. Run the filler script
python3 doc2data/fill_schema.py
```

The output will be saved as `cms1500_result_filled.json`.


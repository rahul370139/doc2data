# Doc2Data - SCRUM Report

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
├── Dockerfile
├── start_services.sh       # Service startup script
├── deploy_and_run.sh       # Deployment automation
└── requirements_docker.txt
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

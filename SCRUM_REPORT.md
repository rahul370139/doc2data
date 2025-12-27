# Doc2Data - SCRUM Report

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

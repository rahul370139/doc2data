# Pipeline Overview — Architecture, Scripts, and Technical Details

This document describes the Doc2Data pipeline: architecture, extraction lanes, OCR strategy, and how each component fits together.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Three-Lane Extraction](#three-lane-extraction)
3. [CMS-1500 OCR Pipeline (Lane C)](#cms-1500-ocr-pipeline-lane-c)
4. [Blank Field Detection](#blank-field-detection)
5. [Template Subtraction](#template-subtraction)
6. [Table Extraction (Box 24)](#table-extraction-box-24)
7. [Scripts Reference](#scripts-reference)
8. [Configuration and Data](#configuration-and-data)
9. [Models in Use](#models-in-use)
10. [Roadmap and Known Limitations](#roadmap-and-known-limitations)

---

## High-Level Architecture

```
PDF/Image → Form ID → [Lane A | Lane B | Lane C] → Layout → OCR → Validation → JSON
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Doc2Data Pipeline v2.0                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌───────────────────────────────────────────────────┐    │
│   │  PDF /   │───▶│               Form Identification                  │    │
│   │  Image   │    │  (CMS-1500, UB-04, Generic)                       │    │
│   └──────────┘    └───────────────────────────────────────────────────┘    │
│                                        │                                    │
│                    ┌───────────────────┼───────────────────┐               │
│                    ▼                   ▼                   ▼               │
│            ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│            │   LANE A    │     │   LANE B    │     │   LANE C    │        │
│            │  (Widgets)  │     │  (Digital)  │     │  (Scanned)  │        │
│            │  AcroForm   │     │  Text Layer │     │   OCR +     │        │
│            │  No OCR     │     │  + Matching │     │  Alignment  │        │
│            └──────┬──────┘     └──────┬──────┘     └──────┬──────┘        │
│                   │                   │                   │                │
│                   └───────────────────┴───────────────────┘                │
│                                       │                                    │
│                                       ▼                                    │
│                          ┌────────────────────────┐                        │
│                          │  Layout → OCR → Valid  │                        │
│                          │  → Business Mapping    │                        │
│                          └────────────────────────┘                        │
│                                       │                                    │
│                                       ▼                                    │
│                              ┌──────────────┐                              │
│                              │  JSON Output │                              │
│                              └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Three-Lane Extraction

### Lane A: Fillable PDF (AcroForm Widgets)
**When:** PDF contains interactive form widgets with values
**Method:** Extract widget values directly — no OCR needed
**Accuracy:** Near-perfect (digital text, no recognition errors)
**Latency:** <1 second

```python
# Lane A extracts from AcroForm widgets
if pdf_has_acroform_widgets():
    return extract_widget_values()  # No OCR
```

### Lane B: Digital PDF (Embedded Text Layer)
**When:** PDF has selectable/copyable text (printed + flattened)
**Method:** Extract text layer + zone matching to schema fields
**Accuracy:** Very high (digital text, minimal OCR)
**Latency:** 2-5 seconds

```python
# Lane B uses embedded text layer
if pdf_has_text_layer():
    words = extract_text_layer()
    return match_words_to_schema_zones(words)
```

### Lane C: Scanned Form (Image-based)
**When:** PDF is a scan, photo, or image-only document
**Method:** Align to template → Template subtraction → Per-field OCR
**Accuracy:** High with proper alignment (see OCR pipeline below)
**Latency:** 60-180 seconds (first run with model loading: 5-7 minutes)

```python
# Lane C: Full OCR pipeline
aligned_image = template_alignment(scan)
for field in schema_zones:
    crop = extract_crop(aligned_image, field.bbox)
    clean_crop, ink_ratio = template_subtract(crop)
    text = florence2_ocr(clean_crop, ink_ratio)
    if not text and ink_ratio_interior > 0.10:
        text = florence2_ocr(crop_raw)  # Raw fallback
```

---

## CMS-1500 OCR Pipeline (Lane C)

The CMS-1500 OCR pipeline is optimized for scanned healthcare claim forms with handwritten and printed content.

### Architecture (v2.0)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CMS-1500 Field OCR Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 1: Template Subtraction                                         │   │
│  │   • Subtract registered template from scan                          │   │
│  │   • Output: clean crop (content only) + ink_ratio                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 2: Florence-2 Primary OCR                                       │   │
│  │   • Run Florence-2-large <OCR> on template-subtracted crop          │   │
│  │   • Apply filters: hallucination, template keywords, box numbers    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│              ┌───────────────┴───────────────┐                             │
│              ▼                               ▼                             │
│     ┌─────────────────┐            ┌─────────────────┐                     │
│     │ F2 has text     │            │ F2 empty        │                     │
│     │ (conf > 0)      │            │ (conf = 0)      │                     │
│     └────────┬────────┘            └────────┬────────┘                     │
│              │                              │                              │
│              ▼                              ▼                              │
│  ┌───────────────────────┐    ┌───────────────────────────────────────┐   │
│  │ Case B/C: Validate    │    │ Case A: Try Recovery                   │   │
│  │ • CCA structure check │    │ 1. Upscale retry (if ink > 0.008)     │   │
│  │ • Self-consistency    │    │ 2. Raw crop fallback (if inner_ink    │   │
│  │ • Use F2 result       │    │    > 0.10 — excludes padding edges)   │   │
│  └───────────────────────┘    │ 3. Otherwise → BLANK                   │   │
│                               └───────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Field Type Routing

| Field Type | Method | Notes |
|------------|--------|-------|
| **Text** | Florence-2 on subtracted crop | Keyword/hallucination filters |
| **Address** | Florence-2 + raw fallback | Multi-line needs raw crop recovery |
| **Date** | Florence-2 + VLM for incomplete | Dates need handwriting context |
| **Phone/NPI/Zip** | Florence-2 + upscale retry | Digit fields benefit from 2x upscale |
| **Money** | Florence-2 (left 22% masked for "$") | Pre-printed "$" excluded from CCA |
| **Checkbox** | Fill-ratio detector | Template subtraction isolates check mark |
| **Signature** | Ink detection only | Returns "[SIGNED]" or blank |
| **Table (Box 24)** | VLM directly (MiniCPM-o4.5) | Structured extraction, separate route |

---

## Blank Field Detection

### The Problem
Blank fields on scanned forms contain:
- Template gridlines and box borders
- Pre-printed labels ("PATIENT NAME", "$", etc.)
- Alignment noise from registration
- Dust, scanner artifacts

Traditional ink-ratio thresholds fail because blank fields (0.03–0.11 ink) overlap with content fields (0.10–0.20 ink).

### The Solution: Florence-2 as Blank Detector

**Key Insight:** Florence-2's <OCR> task returns empty string on truly blank regions — it doesn't hallucinate text from gridlines or noise. This is more reliable than pixel-based ink detection.

```
Florence-2 empty + filters pass → BLANK (no VLM rescue, no hallucination)
Florence-2 text + filters pass  → Use text
Florence-2 text + filters fail  → Try raw crop fallback
```

### Multi-Layer Filtering (Step 2)

When Florence-2 returns text, we apply filters to catch template residue:

1. **Hallucination Filter:** Too many words, repeated chars, low alphanum ratio
2. **Template Keyword Filter (density-based):** Only blank when keywords > 50% of text
3. **Box Number Filter:** Short numeric patterns like "23" (box labels)
4. **Placeholder Filter:** Single chars like "-", ".", "_"

```python
# Density-based keyword filter (won't blank "927 Main St." just because it contains "St")
if keyword_chars / len(text) > 0.5:
    text = ""  # Template label, not content
```

### Raw Crop Fallback (Recovery for Damaged Content)

Template subtraction can damage real content (especially multi-line addresses where text overlaps template lines). When Florence-2 returns empty on the subtracted crop but the field has genuine content:

```python
# Gate: inner ink ratio (center 70% of crop, excludes padding bleed from neighbors)
h, w = crop.shape[:2]
interior = crop[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)]
inner_ink = compute_ink_ratio(interior)

if inner_ink > 0.10:
    # Try Florence-2 on RAW crop (before template subtraction)
    raw_text = florence2_ocr(crop_raw)
    if raw_text and passes_all_filters(raw_text):
        return raw_text  # Recovered!

return ""  # Truly blank
```

**Why inner ink?** Padded bounding boxes can bleed in content from neighboring fields, inflating the overall ink_ratio. The interior (center 70%) excludes edge padding, giving accurate measurement of the field's actual content.

---

## Template Subtraction

### Purpose
Remove pre-printed template elements (gridlines, labels, boxes) from scanned crops, leaving only handwritten/typed content.

### Method
1. **Register scan to template** using AKAZE/ORB features + RANSAC homography
2. **Extract corresponding regions** from both scan and template
3. **Pixel-wise subtraction** with adaptive thresholding
4. **Compute ink_ratio** = non-template pixels / total pixels

```python
def template_subtract(scan_crop, template_rgb, bbox):
    template_crop = template_rgb[y0:y1, x0:x1]
    diff = cv2.absdiff(scan_crop, template_crop)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, threshold, 255, BINARY)[1]
    ink_ratio = np.count_nonzero(binary) / binary.size
    return clean_crop, ink_ratio
```

### Ink Ratio Interpretation

| Ink Ratio | Meaning |
|-----------|---------|
| 0.00–0.03 | Empty (dust/noise only) |
| 0.03–0.09 | Likely blank (template alignment noise) |
| 0.09–0.12 | Ambiguous (may have faint content) |
| 0.12–0.20 | Content present |
| 0.20+ | Heavy content (multi-line, signatures) |

---

## Table Extraction (Box 24)

Box 24 (service lines) requires structured extraction of tabular data.

### Method
1. **VLM Direct Extraction:** MiniCPM-o4.5 (primary) or MiniCPM-V (fallback)
2. **Temperature 0.0** for deterministic output
3. **Structured Prompt** with explicit column definitions
4. **Deduplication** by (CPT code, charges) to prevent row duplication

```python
# Table extraction uses VLM directly (not the text OCR pipeline)
table_data = vlm_extract_table(
    crop=box24_crop,
    model="openbmb/minicpm-o4.5:latest",
    temperature=0.0,
    prompt=BOX24_PROMPT
)
```

### Output Format
```json
{
  "rows": [
    {
      "date_from": "09/20/12",
      "date_to": "09/20/12",
      "place_of_service": "11",
      "cpt_code": "99213",
      "modifier": "",
      "charges": "125.00",
      "units": "1",
      "rendering_npi": "1234567890"
    }
  ],
  "total_rows": 3,
  "extraction_method": "vlm_primary"
}
```

---

## Scripts Reference

### Core Pipeline

| File | Purpose |
|------|---------|
| `src/pipelines/multi_agent_pipeline.py` | Main orchestrator (3-lane extraction) |
| `src/pipelines/agents/ocr.py` | Florence-2 OCR, blank detection, field routing |
| `src/pipelines/agents/labeling.py` | Table extraction, semantic labeling |
| `src/pipelines/agents/form_identification.py` | Form type detection |
| `src/pipelines/agents/template_alignment.py` | Alignment orchestration |
| `src/pipelines/registration/cms1500_register.py` | CMS-1500 classical CV alignment |

### App Layer

| File | Purpose |
|------|---------|
| `app/api_main.py` | FastAPI REST API |
| `app/streamlit_main.py` | Streamlit web UI |

### Utilities

| File | Purpose |
|------|---------|
| `utils/config.py` | Central configuration |
| `src/pipelines/validators/validation.py` | Field format validation |
| `src/pipelines/schemas/business_schema.py` | Business-friendly field mapping |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/api_client.py` | CLI client for API |
| `scripts/test_pipeline.py` | Integration test |
| `scripts/grade_extraction.py` | Accuracy evaluation |
| `deploy_and_run.sh` | DGX deployment |

---

## Configuration and Data

### Schema Files
```
data/schemas/
├── cms-1500.json    # Field definitions with bbox_norm_new (refined coordinates)
└── ub-04.json       # UB-04 field definitions
```

Schema fields include:
- `id`: Field identifier (e.g., "33_billing_provider_address")
- `label`: Human-readable name
- `field_type`: text, address, date, phone, npi, money, checkbox, signature, table
- `bbox_norm_new`: Manually refined normalized coordinates [x0, y0, x1, y1]
- `mode`: "scan", "digital", or "both"

### Environment Variables
```bash
OLLAMA_HOST=http://localhost:11434
VLM_MODEL_TABLE=openbmb/minicpm-o4.5:latest
VLM_MODEL_RESCUE=minicpm-v
USE_GPU=true
```

---

## Models in Use

| Purpose | Model | Location |
|---------|-------|----------|
| **Primary OCR** | Florence-2-large | HuggingFace (cached on DGX) |
| **Table Extraction** | MiniCPM-o4.5 | Ollama |
| **VLM Fallback** | MiniCPM-V | Ollama |
| **Handwriting OCR** | TrOCR-large | HuggingFace (optional) |
| **Printed Text** | PaddleOCR v5 | PaddlePaddle |
| **Semantic Labels** | Qwen 2.5 3B | Ollama |
| **Layout (optional)** | YOLOv8 | Custom trained |

### Model Loading
- Florence-2 is pre-loaded on first request (~45s load time)
- Subsequent requests use cached model
- All Ollama models are pre-warmed at container startup

---

## Roadmap and Known Limitations

### Current Limitations
1. **First-run latency:** Model loading adds 5-7 minutes on first extraction
2. **Multi-line addresses:** May require raw crop fallback (adds ~200ms)
3. **Table dates:** Year sometimes truncated in service lines
4. **Handwritten cursive:** Challenging for all OCR models

### Completed (v2.0)
- [x] Florence-2 as primary OCR (replaces TrOCR+PaddleOCR consensus)
- [x] Blank field detection via Florence-2 (no pixel-based heuristics)
- [x] Template subtraction with ink_ratio
- [x] Raw crop fallback with inner-ink gating
- [x] Removed VLM rescue for text fields (latency reduction)
- [x] MiniCPM-o4.5 for table extraction
- [x] Density-based keyword filtering

### Planned
- [ ] Confidence calibration for business metrics
- [ ] Active learning from user corrections
- [ ] Multi-page document support
- [ ] Real-time streaming extraction
- [ ] Field-level uncertainty quantification

---

## Quick Reference: Processing Flow

```
1. Form Identification
   └─ OCR header region → detect form type

2. Lane Selection
   ├─ Lane A: AcroForm widgets → direct extraction
   ├─ Lane B: Text layer → zone matching
   └─ Lane C: Scan → alignment + OCR

3. Lane C OCR Pipeline (per field)
   ├─ Template subtraction → clean crop + ink_ratio
   ├─ Florence-2 primary OCR
   ├─ Multi-layer filters (hallucination, keywords, box numbers)
   ├─ Case A (F2 empty): upscale retry → raw fallback → blank
   ├─ Case B/C (F2 text): CCA check → consistency → use result
   └─ Tables: VLM direct extraction (separate route)

4. Post-Processing
   ├─ Checkbox resolution (paired yes/no, mutually exclusive groups)
   ├─ Field validation (NPI checksum, date format, etc.)
   └─ Business schema mapping

5. Output
   └─ JSON with extracted_fields, business_fields, statistics
```

---

*Last updated: March 2026*

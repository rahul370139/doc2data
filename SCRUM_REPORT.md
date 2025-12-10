# 📊 Doc2Data Scrum Report

**Date:** December 10, 2025  
**Current Sprint:** CMS-1500 Optimization (Phase 1 Finalization)  
**Sprint Goal:** 100% accuracy on CMS-1500 forms + Business Schema mapping  
**Status:** 🟡 In Progress

---

## 📅 Sprint History

### ✅ Week of Nov 2: Foundations
| Item | Status | Notes |
|------|--------|-------|
| Problem Analysis | ✅ Done | Deep dive into parsing requirements |
| Architecture Design | ✅ Done | Layout → OCR → VLM pipeline established |
| Infrastructure | ✅ Done | NVIDIA DGX Spark with GPU configured |
| MVP Deployment | ✅ Done | Streamlit demo hosted on DGX |

### ✅ Week of Nov 9: GPU & AI Integration
| Item | Status | Notes |
|------|--------|-------|
| GPU Acceleration | ✅ Done | Detectron2 + PaddleOCR on CUDA (10-20x speedup) |
| Layout Segmentation | ✅ Done | LayoutParser + Detectron2 (PubLayNet) |
| Table Extraction (TATR) | ✅ Done | Microsoft Table Transformer integrated |
| Semantic Labeling (SLM) | ✅ Done | Qwen/Llama via Ollama |
| VLM for Figures | ✅ Done | MiniCPM-V for chart understanding |
| Remote Access | ✅ Done | Secure demo on Spark DGX (100.126.216.92:8501) |

---

## 🏃 Current Sprint: Week of Dec 8

### 🎯 Sprint Goals
1. **100% accuracy on CMS-1500 forms** (main deliverable)
2. **Business Schema mapping** (semantic field names: `patient_name`, `insurance_id`)
3. **Demo-ready UI** showing: Upload → OCR → OCR JSON → Business JSON
4. **Fallback for Chinese models** (use Llama 3.2 instead of Qwen)

### 📋 Sprint Backlog

| Task | Priority | Status | Assignee |
|------|----------|--------|----------|
| Collect 5-10 filled CMS-1500 forms (hand + machine) | P0 | ⏳ Pending | - |
| Fine-tune pipeline for CMS-1500 field extraction | P0 | 🔄 In Progress | - |
| Implement Business Schema mapper (CMS-1500 specific) | P0 | 🔄 In Progress | - |
| Add Llama 3.2 as primary model (replace Qwen) | P1 | ✅ Done | - |
| Demo UI: OCR view, Raw JSON, Business JSON tabs | P1 | 🔄 In Progress | - |
| Validate field accuracy on test forms | P2 | ⏳ Pending | - |

---

## 🔍 Technical Deep Dive: The Two-Schema Architecture

### Problem Statement
The current pipeline extracts **OCR Schema** (generic), but clients need **Business Schema** (domain-specific).

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   OCR Schema    │ ──► │  Business Logic │ ──► │ Business Schema │
│   (Generic)     │     │    (Mapping)    │     │  (CMS-1500)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘

OCR Schema (same for all docs):
{
  "page": 0,
  "bbox": [100, 200, 300, 250],
  "type": "TEXT",
  "value": "JOHN DOE",
  "confidence": 0.95
}

Business Schema (CMS-1500 specific):
{
  "patient_name": "JOHN DOE",
  "patient_dob": "01/15/1985",
  "insurance_id": "XYZ123456",
  "npi": "1234567890"
}
```

### Current Implementation (`form_extractor.py`)

```
Read → Understand → Ground

1. READ: Full-page PaddleOCR → All text + bboxes
2. UNDERSTAND: LLM extracts values based on schema
3. GROUND: Match extracted values back to OCR bboxes
```

**What's Working:**
- Full-page OCR captures all text ✅
- LLM can extract values when Ollama is running ✅
- Grounding links values back to visual locations ✅

**What's Failing:**
- LLM not always available on DGX (Ollama connection issues)
- Heuristic fallback is too naive for dense forms
- No validation layer (NPI, dates, phone)

---

## 🧠 Proposed Architecture: Template-First for CMS-1500

Since CMS-1500 is a **fixed-layout form**, we don't need ML to "discover" field locations. We **know** where each field is.

### The "Template-First" Approach

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. ALIGN    │ ──► │  2. CROP     │ ──► │  3. OCR      │ ──► │  4. VALIDATE │
│  Template    │     │  Field Zones │     │  Per-Field   │     │  & Format    │
│  Registration│     │  from Schema │     │  TrOCR/Paddle│     │  Business    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

**Step 1: Template Registration**
- Use ORB/SIFT feature matching to align scanned form to reference template
- Handle rotation, skew, and scale variations
- Already implemented: `src/processing/registration.py`

**Step 2: Zone-Based Cropping**
- CMS-1500 schema defines bounding boxes for each field
- Crop each field region from the aligned image
- No ML needed - deterministic coordinates

**Step 3: Field-Specific OCR**
| Field Type | OCR Strategy |
|------------|--------------|
| Printed text | PaddleOCR (fast, accurate) |
| Handwritten | TrOCR (specialized) |
| Checkboxes | Pixel density analysis |
| Dates | OCR + regex validation |
| NPI/Phone | OCR + format validation |

**Step 4: Business Schema Assembly**
- Map each field value to its semantic name
- Apply validators (NPI checksum, date format, phone format)
- Flag low-confidence fields for review

### Why This Beats the "Agentic" Approach

| Aspect | Agentic (Multi-Agent) | Template-First |
|--------|----------------------|----------------|
| Complexity | High (multiple agents, retry loops) | Low (deterministic) |
| Speed | Slow (multiple LLM calls) | Fast (single OCR pass) |
| Accuracy for CMS-1500 | Variable (depends on LLM) | High (known field locations) |
| Debugging | Hard (agent decisions are opaque) | Easy (clear pipeline) |
| Generalization | Better for unknown forms | Perfect for fixed forms |

**Recommendation:** Use Template-First for CMS-1500 (100% accuracy goal), keep ML pipeline for unknown forms.

---

## 🛠 What's Already Implemented

### ✅ Fully Working
| Component | File | Notes |
|-----------|------|-------|
| Ingestion | `src/pipelines/ingest.py` | PDF/Image → 300 DPI, deskew, denoise |
| Layout Segmentation | `src/pipelines/segment.py` | Detectron2 + LayoutParser + heuristics |
| PaddleOCR | `src/ocr/paddle_ocr.py` | GPU-accelerated, v2.6 |
| TrOCR Wrapper | `src/ocr/trocr_wrapper.py` | Handwriting recognition |
| Tesseract Fallback | `src/ocr/tesseract_ocr.py` | CPU fallback |
| Table Extraction | `src/pipelines/table_processor.py` | TATR + VLM (MiniCPM-V) |
| Figure Processing | `src/pipelines/figure_processor.py` | Chart understanding |
| SLM Labeler | `src/pipelines/slm_label.py` | Qwen/Llama semantic roles |
| Form Extractor | `src/pipelines/form_extractor.py` | Read-Understand-Ground pipeline |
| Image Registration | `src/processing/registration.py` | ORB-based alignment |
| Streamlit UI | `app/streamlit_main.py` | Demo with split-pane view |

### ⚠️ Needs Work
| Component | Issue | Fix |
|-----------|-------|-----|
| Ollama Connection | Unreliable on DGX | Add retry logic, health checks |
| Business Schema Mapper | Not implemented | Map OCR output to CMS-1500 fields |
| Field Validators | Partial | Add NPI, date, phone, ICD-10 validators |
| Template Alignment | Basic | Improve for real-world scans |

---

## 🔜 Next Steps (Week of Dec 8)

### Must-Do (P0)
1. **Collect CMS-1500 Test Set**
   - 5-10 filled forms (mix of hand-filled and machine-filled)
   - Include varied handwriting styles
   - Store in `data/sample_docs/cms1500_test/`

2. **Implement Business Schema Mapper**
   ```python
   # New file: src/pipelines/business_schema.py
   def map_to_business_schema(ocr_result: Dict) -> Dict:
       return {
           "patient_name": extract_field("1_insured_name"),
           "patient_dob": extract_field("3_patient_dob"),
           "insurance_id": extract_field("1a_insured_id"),
           ...
       }
   ```

3. **Demo UI Enhancement**
   - Tab 1: Document View (with bounding boxes)
   - Tab 2: OCR JSON (raw extraction)
   - Tab 3: Business JSON (mapped fields)
   - Dropdown: Select pipeline (CMS-1500 / General)

4. **Model Fallback Chain**
   ```
   Llama 3.2 (primary) → Mistral → Qwen (fallback)
   ```

### Should-Do (P1)
5. **Field Validators**
   - NPI: 10 digits + Luhn checksum
   - Date: MM/DD/YYYY format
   - Phone: (XXX) XXX-XXXX
   - ICD-10: A00-Z99 format

6. **Accuracy Measurement**
   - Create ground-truth JSON for test forms
   - Script to compare extracted vs ground-truth
   - Report: Field-level accuracy %

### Nice-to-Have (P2)
7. **Confidence Visualization**
   - Green: High confidence (>0.9)
   - Yellow: Medium (0.7-0.9)
   - Red: Low (<0.7) - needs review

---

## 📊 Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| CMS-1500 Field Accuracy | 100% | ~60% (needs tuning) |
| General Form Accuracy | 80-90% | ~70% |
| Processing Time (1 page) | <5 sec | ~3 sec ✅ |
| Demo Uptime | 99% | ~95% |

---

## 🗓 Sprint Timeline

| Date | Milestone |
|------|-----------|
| Dec 8-10 | Collect test forms, fix Ollama, basic mapping |
| Dec 11-13 | Implement Business Schema mapper, validators |
| Dec 14-15 | Demo polish, accuracy testing |
| Dec 16 | **Sprint Demo** |

---

## 📝 Notes for Next Presentation

1. **Key Message:** We've built a complete IDP pipeline that works for general documents. For CMS-1500 specifically, we're implementing a template-based approach for 100% accuracy.

2. **Demo Flow:**
   - Upload CMS-1500 form
   - Show OCR extraction (bounding boxes on image)
   - Show Raw JSON (OCR Schema)
   - Show Business JSON (patient_name, insurance_id, etc.)
   - Highlight confidence scores

3. **Technical Differentiators:**
   - GPU-accelerated (10-20x faster than CPU)
   - Hybrid approach (ML for unknown, Template for known forms)
   - Grounding (every extracted value linked to visual source)
   - Self-improving (corrections feed back to training)

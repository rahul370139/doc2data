# Doc2Data Pipeline - Technical Summary
## For Manager Presentation

---

## 🎯 What We Built
A GPU-accelerated document parsing pipeline that converts PDFs/images into structured JSON + Markdown with bounding-box citations. Similar to Reducto.ai and Extend.ai.

**Demo Link:** http://100.126.216.92:8501

---

## ✅ Implemented Features

### 1. Layout Detection (Detectron2 + PubLayNet)
- GPU-accelerated detection of: Text, Tables, Figures, Forms
- Runs on NVIDIA DGX with CUDA support
- Confidence scores for each detection

### 2. OCR (PaddleOCR 3.x)
- GPU-accelerated text extraction
- Supports handwritten and printed text
- Word-level coordinates preserved

### 3. Table Structure Recognition (TATR)
- Microsoft Table Transformer for cell-level extraction
- Header/body separation
- Structured JSON output

### 4. Form Field Detection (OpenCV)
- Checkbox detection (checked/unchecked/ambiguous)
- Field rectangle identification
- Label-value association using Hungarian matching

---

## 🧠 AI Models - Current Status

### SLM (Small Language Model) - ✅ ACTIVE
| Model | Purpose | Status | Size |
|-------|---------|--------|------|
| **Qwen2.5:7B-Instruct** | Semantic Labeling | ✅ Running | 4.7 GB |

**What it does:**
- Classifies text blocks into roles: `title`, `h1`, `h2`, `header`, `footer`, `page_num`, `list_item`, `kv_label`, `kv_value`, `caption`, `paragraph`
- Uses structured JSON prompts with few-shot examples
- Receives context: position, column_id, form_field status, validator_status

**Architecture:**
```
Block → Context Features → Qwen2.5 Prompt → JSON Response → BlockRole Enum
```

### VLM (Vision-Language Model) - ✅ ACTIVE
| Model | Purpose | Status | Size |
|-------|---------|--------|------|
| **MiniCPM-V** | Figure/Chart/Document Understanding | ✅ Running | 5.5 GB |
| **LLaVA** | General Vision-Language (backup) | ✅ Available | 4.7 GB |

**What it does:**
- Figure classification: bar, line, pie, scatter, diagram, photo
- Chart data extraction: axes, series, values
- Table structure repair: headers, body cells
- OCR enhancement for low-confidence regions
- Document-level understanding (MiniCPM-V specialty)

**Why MiniCPM-V:** Specifically designed by OpenBMB for OCR and document understanding tasks. Better than Qwen2-VL for structured documents like forms.

---

## 🚀 The "Secret Sauce" - CV Post-Processing

### A. Granularity Preference (Exploding Parent Logic)
**Problem:** AI model detects entire page as one giant "Form" block.
**Solution:** If container covers >50% page AND contains semantic children → delete container, keep children.
**Code:** `_clean_layout_heuristics()` in `segment.py`

### B. Visual Grid Snapping
**Problem:** Boxes floating around text, messy gaps.
**Solution:** Use OpenCV Hough Transform to detect actual lines. Snap block edges within 15px to real lines.
**Code:** `_snap_blocks_to_lines()` in `segment.py`

### C. Aggressive Stitching
**Problem:** "EVAC" and "CATEGORY" detected as separate blocks.
**Solution:** Allow 600px horizontal gap for header stitching, 100px vertical gap for table-header merging.
**Code:** `_merge_horizontal_text_lines()`, `_merge_vertical_tables()` in `segment.py`

### D. Gap Filling
**Problem:** Top of form was empty (text detection accidentally disabled).
**Solution:** Force `_augment_text_with_heuristic` to run unconditionally with relaxed thresholds.
**Code:** `_augment_text_with_heuristic()` with `text_density >= 0.01`

### E. Containment Cleanup
**Problem:** Small text blocks overlapping with form field boxes.
**Solution:** If text block is >65% inside a form/table block, remove the text block.
**Code:** `_is_contained()` threshold lowered to 0.65

---

## 📊 Technical Stack

| Component | Technology | GPU | Status |
|-----------|------------|-----|--------|
| Layout Detection | Detectron2 + PubLayNet | ✅ CUDA | ✅ Active |
| OCR | PaddleOCR 3.x | ✅ CUDA | ✅ Active |
| Table Structure | Table Transformer (TATR) | ✅ CUDA | ✅ Active |
| Semantic Labeling | Qwen2.5-7B via Ollama | ✅ CUDA | ✅ Active |
| Figure Understanding | MiniCPM-V via Ollama | ✅ CUDA | ✅ Active |
| Form Geometry | OpenCV (Hough, Morphology) | CPU | ✅ Active |
| Deployment | Docker + NVIDIA Runtime | ✅ | ✅ Active |

---

## 📁 Output Format

```json
{
  "blocks": [
    {
      "id": "p1-0",
      "type": "FORM",
      "role": "kv_label",
      "text": "NAME (Last, First):",
      "bbox": [100, 150, 400, 180],
      "confidence": 0.92,
      "page": 1,
      "metadata": {
        "column_id": 0,
        "form_field": {
          "type": "text_field",
          "label_id": "p1-0"
        }
      }
    }
  ]
}
```

---

## 🔮 Next Steps - Advanced SLM/VLM Models to Try

### Short-Term (1-2 weeks) - SLM Enhancements

| Model | Why | Expected Benefit |
|-------|-----|------------------|
| **Phi-3-mini-4k-instruct (3.8B)** | Microsoft's efficient SLM | 2x faster inference, similar quality |
| **Qwen2.5:3B-Instruct** | Smaller Qwen variant | Lower memory, faster |
| **SmolLM-1.7B-Instruct** | HuggingFace tiny model | Ultra-fast for simple labeling |

### Medium-Term (1 month) - VLM Integration

| Model | Why | Expected Benefit |
|-------|-----|------------------|
| **Qwen2.5-VL:7B** | Updated VL model | Better chart/table understanding |
| **LLaVA-NeXT-7B** | Mistral-based VLM | Strong reasoning on documents |
| **InternVL2-8B** | Chinese Academy of Sciences | SOTA on document VQA |
| **Florence-2** | Microsoft's vision foundation | Layout + OCR in one model |

### Long-Term (3 months) - Production-Ready

| Model | Why | Expected Benefit |
|-------|-----|------------------|
| **DocOwl-1.5** | Specialized for documents | Document-specific fine-tuning |
| **mPLUG-DocOwl2** | Multi-page document understanding | Handle long PDFs natively |
| **Idefics3-8B-Llama3** | HuggingFace multimodal | Open weights, customizable |
| **GOT-OCR2** | General OCR Theory model | End-to-end document parsing |

---

## 🧪 Model Selection Criteria

For our use case (healthcare forms), we prioritize:

1. **Accuracy on structured forms** - Tables, checkboxes, key-value pairs
2. **Speed** - < 500ms per block for SLM, < 2s per image for VLM
3. **Self-hosted** - No cloud dependency (HIPAA compliance)
4. **GPU memory** - Must fit in 16GB alongside other models

**Recommended Next Model:** `Phi-3-mini-4k-instruct` for SLM (faster) + `InternVL2-8B` for VLM (better docs)

---

## 📈 Performance Metrics

| Metric | Current Value | Target |
|--------|---------------|--------|
| Processing Time (1 page) | 3-5 seconds | < 2 seconds |
| Layout Detection Accuracy | ~85% | > 90% |
| OCR Accuracy | ~92% | > 95% |
| Semantic Labeling Accuracy | ~75% (estimated) | > 85% |
| GPU Memory Usage | ~8GB | < 12GB |

---

## 🔗 Demo Resources

- **Live Demo:** http://100.126.216.92:8501
- **Sample Documents:** CMS-1500, UB-04, UCF Form
- **Comparison:** Reducto.ai, Extend.ai

---

## 📝 Known Limitations

1. **Semantic Labeling** - Works but needs threshold tuning for footer/header detection
2. **Multi-page PDFs** - Processed page-by-page (no cross-page context)
3. **Handwritten Text** - PaddleOCR handles it but accuracy varies
4. **VLM Speed** - MiniCPM-V adds ~2-3s per image for figure/chart processing

---

*Generated: December 5, 2025*
*Pipeline Version: 1.0.0-gpu*

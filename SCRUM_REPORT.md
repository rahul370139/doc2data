# 📊 Doc2Data Scrum Report

**Date:** December 10, 2025  
**Current Sprint:** Phase 1 Finalization & CMS-1500 Optimization  
**Status:** 🟢 On Track

---

## 📅 Sprint Status

### ✅ Completed Work (Previous Sprints)

**Week of Nov 2: Foundations**
- **Problem Analysis:** Deep dive into parsing requirements and business schema mapping.
- **Architecture Design:** Established the Layout -> OCR -> VLM pipeline architecture.
- **Infrastructure:** Selected and configured NVIDIA DGX environment with GPU support.
- **MVP Deployment:** Hosted initial building blocks (OCR, Layout) on Streamlit.

**Week of Nov 9: GPU & AI Integration**
- **GPU Acceleration:** Enabled CUDA for Detectron2 and PaddleOCR (10x-20x speedup).
- **Segmentation:** Implemented Detectron2 (PubLayNet) for robust layout analysis.
- **Semantic Labeling:** Integrated SLM (Qwen/Llama) for semantic role assignment.
- **VLM Parsing:** Added VLM support for extracting structured data from tables and figures.
- **Remote Access:** Setup secure demo environment on Spark DGX machine.

### 🏃 Current Sprint (Week of Dec 8)
- **Goal:** Achieve 100% accuracy on CMS-1500 forms.
- **Form-Specific Tuning:** Fine-tuned pipeline parameters for dense form layouts.
- **Business Schema Mapping:** Mapped raw OCR output to "Business Schema" (e.g., `patient_name`, `insurance_id`).
- **Smart Extraction:** Implemented "Smart Pipeline" using schema-guided extraction + VLM grounding.
- **Demo Enhancement:** "Workstation" UI in Streamlit showing OCR, Raw JSON, and Business JSON side-by-side.
- **Model Fallback:** Added support for Western models (Llama 3.2, TrOCR) to replace/supplement Qwen where appropriate.

---

## 🛠 Phase 1 Plan: Schema-Agnostic, Text-Centric MVP

This architecture is currently implemented and running in production.

### 1) Ingest & Pre-process
*   **Input:** PDF/Images → Normalized 300 DPI images.
*   **Cleanup:** GPU-accelerated de-skew, de-noise, and contrast enhancement (OpenCV CUDA).
*   **Digital Layer:** If PDF has text layer, extract word-boxes directly to skip OCR (High Accuracy).
*   **Form Mask:** Precompute "lines & boxes mask" for geometry detection.

### 2) Layout Segmentation (Vision-First)
*   **Models:** LayoutParser + Detectron2 (PubLayNet) on GPU.
*   **Form Analyzer:**
    *   **Line/Box Graph:** Detects physical lines and boxes to identify fields/tables.
    *   **Column-Aware:** Sorts blocks by column first, then row (crucial for forms).
    *   **Granularity:** "Child-Explodes-Parent" logic to prefer fine-grained text over giant generic boxes.
    *   **Grid Snapping:** Snaps detected blocks to visual grid lines for perfect alignment.

### 3) OCR (Intelligent Character Recognition)
*   **Engines:** PaddleOCR v2.6 (Primary, GPU) + TrOCR (Handwriting) + Tesseract (Fallback).
*   **Tiered Processing:**
    *   **Tier 1:** Fast pass on all blocks.
    *   **Tier 2:** Re-OCR low-confidence regions with enhanced binarization.
*   **Association:** Links "Labels" to "Values" using spatial heuristics and Hungarian matching.
*   **Checkboxes:** Detects state (Checked/Unchecked) using pixel density and connected components.

### 4) Semantic Labeling (SLM) & VLM
*   **Models:** Llama-3.2-3B / Qwen-2.5-7B (via Ollama).
*   **Role:** Assigns semantic roles (Header, Patient Name, DOB) based on content and position.
*   **Grounding:** Maps extracted semantic values back to original bounding boxes for verification.
*   **Tables/Figures:** VLM used to summarize charts or fix broken table structures.

### 5) Assembly & UI
*   **Output:** Canonical JSON with `page`, `bbox`, `confidence`, `business_key`.
*   **UI:** Streamlit "Workstation" with split-pane view (Image + Data), interactive correction, and JSON download.

---

## 🚀 Key Technical Improvements Implemented

### 1. Vision-First Layout Segmentation (The "Reducto" Look)
*   **Problem:** Standard models produced either giant blocks (hiding fields) or messy fragments.
*   **Solution:** Implemented **Hybrid Vision-Heuristic Segmenter** (`src/pipelines/segment.py`).
    *   **Strict NMS:** No overlapping blocks allowed.
    *   **Visual Grid Snapping:** Uses OpenCV Hough Transform to snap block edges to actual pixel lines.
    *   **Granularity Preference:** Discards giant containers if they contain valid children.

### 2. Aggressive Section Stitching
*   **Problem:** Headers like "EVAC CATEGORY" were split.
*   **Solution:** Added **Aggressive Merge Logic**:
    *   **Horizontal:** Stitches text with gaps up to 600px if aligned.
    *   **Vertical:** Recovers split tables (Header + Body) even with gaps.

### 3. Gap Filling (No Blank Spaces)
*   **Problem:** ML models often miss non-standard text regions.
*   **Solution:** **Unconditional Text Augmentation**. A CV pass (connected components) finds *any* text-like pixels missed by the model and sends them to OCR.

### 4. Multi-Pass OCR & Validation
*   **Implementation:**
    *   **Tiered OCR:** PaddleOCR first; if conf < 0.75, re-run with image enhancement.
    *   **Validators:** NPI, Date, ICD-10, Phone regex validators integrated.
    *   **VLM Fallback:** Ambiguous regions sent to Llama/Qwen for "reading".

---

## 🧠 Proposed Agentic "Bootstrap" Pipeline

To achieve 100% accuracy on CMS-1500 and >90% on others, we are moving towards an **Agentic Bootstrap Architecture**:

1.  **Fast Pass (The Scout):** Run optimized Layout + PaddleOCR. High speed, moderate accuracy.
2.  **Validator Gate (The Critic):**
    *   Check extracted fields against business rules (e.g., "Is DOB a date?", "Is NPI 10 digits?").
    *   Pass: ✅ Commit to JSON.
    *   Fail: ❌ Send to "Fixer".
3.  **The Fixer (Agentic Loop):**
    *   **Attempt 1 (Vision):** Apply aggressive binarization/upscaling to the specific field crop. Re-OCR.
    *   **Attempt 2 (Brain):** Send the crop to **VLM (Llama 3.2 Vision / Qwen-VL)** with prompt: *"Read the handwriting in this box exactly."*
    *   **Attempt 3 (Context):** Look at neighboring fields for context clues.
4.  **Human-in-the-Loop (Bootstrap):**
    *   If all agents fail, flag for human review in Streamlit.
    *   **Bootstrap:** Save human corrections to a dataset.
    *   **Retrain:** Periodically fine-tune the "Fast Pass" models on this hard data to make them smarter.

---

## 🔜 Next Steps (Roadmap)

1.  **Vision-First Parsing:** Implement multi-pass OCR (Vision -> Text) with self-correction.
2.  **Fine-tune Detectron2:** Train specifically on healthcare forms (CMS-1500, UB-04) using DocLayNet base.
3.  **Agentic OCR:** Build a self-correcting loop using VLM validation.
4.  **Schema-Driven Extraction:** Fully generalize the schema mapper for other form types (UB-04, Invoices).
5.  **Advanced Tables:** Integrate Table Transformer (TATR) for complex grid parsing.
6.  **Multi-Model Ensemble:** Combine results from Detectron2, Paddle, and Heuristics for robust layout.
7.  **Continuous Learning:** Implement a feedback loop where user corrections retrain the base models.


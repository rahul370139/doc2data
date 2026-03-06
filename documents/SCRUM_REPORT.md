# Doc2Data — SCRUM Report

**Report Date:** March 2026  
**Owner:** Document AI / Computer Vision Engineering  
**Audience:** Engineering and Product Management

---

## Executive Summary

Doc2Data is a production-oriented document extraction pipeline for healthcare forms, primarily CMS-1500 and UB-04. The system uses a three-lane architecture to handle fillable PDFs, digital text layers, and scanned documents. Recent work has focused on reliability, blank-field accuracy, and latency reduction in the scanned OCR path.

---

## 1. Document Processing Flow (Narrative)

### 1.1 Ingestion and Form Identification

The pipeline receives a PDF or image as input. The first page is rendered at 300 DPI to produce a high-resolution image. A form identification agent inspects the header and footer regions to determine the form type. Supported types include CMS-1500, UB-04, NCPDP, and generic forms. The identification uses OCR on key regions plus optional layout fingerprint matching. The result is a form type and confidence score that drives subsequent lane selection.

### 1.2 Lane Selection

The pipeline selects one of three extraction lanes based on document characteristics.

**Lane A (Widgets):** If the PDF contains interactive AcroForm widgets and a sufficient number of fields are filled (at least 10 for CMS-1500), the pipeline extracts values directly from the form widgets. No OCR is performed. This path is the fastest and most reliable for fillable PDFs.

**Lane B (Digital Text):** If the document has an embedded digital text layer, the pipeline extracts text and spatial coordinates from that layer. A visual trust check ensures the text layer matches what is visible in the rendered image. For CMS-1500, an additional anchor quality check validates three critical fields (insured ID, patient name, date of birth). At least two of three must pass before the digital layer is trusted. If the layer appears template-only or sparse, the pipeline falls back to the scan path.

**Lane C (Scan OCR):** When no trustworthy widget or text layer exists, the pipeline treats the document as a scan. Template alignment is applied first to correct geometric distortion. Schema-defined zones are then loaded and padded. Each zone is processed by a tiered OCR pipeline that routes by field type (checkbox, signature, date, text, table).

### 1.3 Template Alignment (Lane C Only)

For CMS-1500 scans, a deterministic registration step aligns the scanned page to a canonical template. The process uses dropout-red masking to isolate handwritten content from pre-printed template elements. Feature detection (AKAZE/ORB) and descriptor matching with RANSAC produce a homography that warps the scan into template space. When feature quality is low, an outer-quad fallback provides a coarse alignment. Thresholds are tunable via environment variables to accommodate handwritten versus machine-filled forms.

### 1.4 Zone Loading and Layout

Schema files define field boundaries using normalized coordinates. The pipeline prefers refined coordinates when available. Configurable padding expands each zone to capture edge content. The schema also specifies which fields apply to scan versus digital modes, so only relevant zones are processed per lane.

### 1.5 OCR Processing by Block Type

The OCR agent routes each block according to its type.

**Checkboxes:** Template subtraction isolates the check mark from pre-printed boxes. A fill-ratio detector determines whether the box is checked. Paired yes/no checkboxes and mutually exclusive groups are resolved by comparing ink ratios to select a single winner.

**Signatures:** Signature fields are not OCR’d. The pipeline only detects whether ink is present and returns a marker (e.g., "[SIGNED]") or blank.

**Dates:** Date fields use Florence-2 first. If the result is incomplete or the field has content but Florence-2 returns empty, a VLM rescue path attempts to recover the date.

**Text, Address, and Numeric Fields:** The primary path runs Florence-2 on a template-subtracted crop. Multi-layer filters remove hallucinated output, template labels, and box numbers. When Florence-2 returns empty, an upscale retry is attempted. If the interior of the crop (excluding padded edges) has sufficient ink, a raw-crop fallback runs Florence-2 on the original, un-subtracted crop to recover content damaged by template subtraction. Blank fields are confirmed when no reliable content remains. A final post-processing step strips known template phrases (e.g., "CARRIER", "PICA") that may bleed in from padded bounding boxes.

**Tables (Box 24):** Service line tables are sent directly to a vision-language model. The model runs with deterministic settings to reduce hallucination. Output is parsed into structured rows and deduplicated. If the VLM returns no rows, a fallback OCR path groups text from a full-table OCR.

### 1.6 Validation and Rescue

A validation agent runs typed validators (NPI checksum, date format, phone, ICD, money, etc.) on extracted fields. A targeted rescue path applies only to fields that are clearly invalid, noisy, or critical. Rescue candidates are validated against expected types before acceptance. Updates are conservative to avoid introducing new errors.

### 1.7 Business Mapping and Output

Schema field IDs are mapped to business-friendly keys (e.g., patient name, insured ID, billing NPI). Address components are composed. Sex is derived from checkbox groups. The final response includes extracted fields, field details (confidence, bounding box, source), business fields, validation diagnostics, and optional Reducto-style output.

---

## 2. Quality and Reliability Controls

The pipeline implements several controls to reduce hallucination and improve traceability:

- **Blank-field suppression:** Fields confirmed blank are not surfaced with spurious values. Florence-2 serves as the primary blank detector; pixel-based ink thresholds are supplemented by an inner-ink metric that excludes padding bleed.

- **Checkbox conflict resolution:** Paired and mutually exclusive checkbox groups are resolved by ink ratio so that only one value is emitted per logical field.

- **Template text filtering:** Density-based keyword checks prevent template labels from being returned as field content. A separate post-processing step strips known header phrases that bleed into zones.

- **Targeted rescue:** VLM rescue is applied only to clearly problematic fields, not broadly, to limit latency and hallucination risk.

- **Source metadata:** Each field carries metadata (source, OCR engine, escalation path) to support debugging and auditing.

---

## 3. Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| Form identification | Stable | OCR fingerprint with strong-token override |
| Lane A widget extraction | Stable | Early return on sufficient filled widgets |
| Lane B digital extraction | Stable with guardrails | Visual match plus CMS anchor QA |
| CMS registration | Active, tunable | Deterministic CV registrar with env-threshold tuning |
| Layout detection | Active fallback chain | YOLO, Detectron, Paddle; OCR grouping fallback |
| OCR agent | Active, multi-route | Type-specific behavior; Florence-2 primary for CMS |
| Labeling agent | Active | Table VLM route; optional SLM/VLM for text and figures |
| Validation agent | Active | Typed validators; optional QA |
| Business mapping | Active | CMS and UB mapping with normalization |

---

## 4. Known Risks and Limitations

1. **Single-page processing:** The pipeline processes only the first page of multi-page documents.

2. **Handwriting sensitivity:** Scanned handwriting quality depends on alignment accuracy and scan clarity. Poor registration or low contrast can degrade extraction.

3. **Startup latency:** First-run latency is high due to model loading. Subsequent requests benefit from cached models.

4. **Ollama dependency:** VLM and SLM paths require a reachable Ollama service with the appropriate models available.

---

## 5. Next Sprint Priorities

1. **Field-level accuracy:** Reduce zone bleed into adjacent fields; improve stability for low-ink handwritten content.

2. **Latency reduction:** Further reduce expensive rescue calls; improve model warm-up strategy.

3. **Evaluation discipline:** Run regular grading against gold labels; track error distribution by field type over time.

4. **Scale and coverage:** Multi-page support; stronger robustness for generic forms.

---

## 6. Verification and Operations

Smoke tests can be run via the test pipeline script. Extraction quality can be graded against gold labels in the data directory. Alignment issues can be diagnosed with the CMS-1500 alignment debug script. Registrar thresholds can be tuned using a grid-search script over sample documents. The API client script provides a CLI for REST endpoints. Deployment to DGX is handled by the deploy script (rsync and Docker).

---

## 7. Definition of Done (This Cycle)

- Documentation matches runtime code behavior.
- Lane, OCR, validation, and rescue decisions are traceable in documentation.
- Core scripts and operational flow are documented with executable commands.
- SCRUM report is structured for manager review.

---

*This report reflects the pipeline state as of March 2026.*

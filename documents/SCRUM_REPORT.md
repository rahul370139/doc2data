# Doc2Data — SCRUM Report

**Report Date:** April 2026  
**Owner:** Document AI / Computer Vision Engineering  
**Audience:** Engineering and Product Management

---

## Executive Summary

Doc2Data is a production-oriented document extraction pipeline for healthcare forms, primarily CMS-1500 and UB-04. The system uses a three-lane architecture to handle fillable PDFs, digital text layers, and scanned documents.

**This sprint (April 2026)** delivered four major changes:

1. **LangGraph orchestration (`src/pipelines/graph`).** The extraction flow is now an explicit state machine — `load → identify → plan → (widgets | digital | align → scan) → validate → reflect → rescue → revalidate → finalize` — with per-node timings, traceable plan decisions, and a reflect-rescue loop. Legacy `MultiAgentPipeline` still ships under the hood for widget / digital extraction and for the targeted VLM rescue.
2. **OCR v2 (`src/pipelines/ocr_v2`).** The per-field Florence-2 loop has been replaced by an adaptive blank detector plus a single batched Florence-2 forward pass. End-to-end latency for scanned CMS-1500s dropped from 3–5 minutes to ~40–70 seconds, with alignment-gated early-out on misaligned pages.
3. **Next.js 14 frontend (`frontend/`).** A new TypeScript + Tailwind UI replaces the Streamlit surface for external users. It drives the `/extract/graph/stream` SSE endpoint so users see each node complete in real time, with a multi-tab result explorer (fields, validation, debug trace, business view).
4. **DGX2 deployment (`deploy_and_run.sh`).** One script now rsyncs the repo, builds both the FastAPI and Next.js containers, wires them up on a shared Docker network, and pre-downloads Florence-2.

---

## 1. Document Processing Flow (Narrative)

### 1.1 Ingestion and Form Identification

The pipeline receives a PDF or image as input. The first page is rendered at 300 DPI to produce a high-resolution image. A form identification agent inspects the header and footer regions to determine the form type. Supported types include CMS-1500, UB-04, NCPDP, and generic forms. Identification uses OCR on key regions plus a strong-token fingerprint matcher. In April 2026 we strengthened the CMS-1500 fingerprint with the phrases "health insurance claim form", "national uniform claim committee" and "nucc" so the five CMS-1500 variants we test against are all identified with high confidence. The result is a form type and confidence score that drives subsequent lane selection.

### 1.2 Lane Selection (LangGraph `plan_node`)

The graph's `plan_node` chooses one of three extraction lanes. The decision is emitted as `plan_reason` on the state so it shows up in the final debug trace.

**Lane A (Widgets):** If the PDF contains interactive AcroForm widgets and a sufficient number of fields are filled (at least 10 for CMS-1500, 3 for UB-04), the pipeline extracts values directly from the widgets. No OCR is performed. This path is the fastest and most reliable for fillable PDFs.

**Lane B (Digital Text):** For non-structured forms (anything that is not CMS-1500 / UB-04) with a digital text layer, the pipeline extracts text and spatial coordinates from that layer. A visual trust check ensures the text layer matches what is visible in the rendered image. For CMS-1500 / UB-04 we now **prefer Lane C over Lane B** because the schema bounding boxes are tied to the canonical template and direct zone-matching on raw PDF text is fragile. `extract_digital_node` will additionally downgrade to Lane C when:

- no schema fields are loaded for the identified form, OR
- the number of filled blocks is sparse (<10 for CMS-1500, <5 for UB-04), OR
- more than 30 % of filled blocks share identical text — a strong signal that the same phrase is being matched to multiple zones (template-mismatch guard).

When either of the last two trips, the `plan_reason` becomes `lane_b_downgraded_to_c` and the validator surfaces a `digital_downgrade` QA note.

**Lane C (Scan OCR):** When no trustworthy widget or text layer exists, the pipeline treats the document as a scan. Template alignment is applied first to correct geometric distortion. Schema-defined zones are loaded with minimal padding, and every non-blank zone is routed through **OCR v2** — a single batched Florence-2 call gated by an adaptive structural blank detector. If alignment quality falls below 0.5, the graph now **short-circuits the heavy OCR pass**, emits an `alignment_failed` QA note, and lets the reflect/rescue loop decide what to do. Before this change a single misaligned CMS-1500 could spend 4+ minutes producing unusable output.

### 1.3 Template Alignment (Lane C Only)

For CMS-1500 scans, a deterministic registration step aligns the scanned page to a canonical template. The process uses dropout-red masking to isolate handwritten content from pre-printed template elements. Feature detection (AKAZE/ORB) and descriptor matching with RANSAC produce a homography that warps the scan into template space. When feature quality is low, an outer-quad fallback provides a coarse alignment. Thresholds are tunable via environment variables to accommodate handwritten versus machine-filled forms.

### 1.4 Zone Loading and Layout

Schema files define field boundaries using normalized coordinates. The pipeline prefers refined coordinates when available. Configurable padding expands each zone to capture edge content. The schema also specifies which fields apply to scan versus digital modes, so only relevant zones are processed per lane.

### 1.5 OCR Processing by Block Type

In Lane C, every schema zone is fed into the new **OCR v2** batch. Zones classified `blank` by the structural blank detector never hit Florence-2; zones classified `filled` or `uncertain` are collected into a single pipelined Florence-2 forward pass. Block-type-specific behaviour is preserved for the specialised routes:

**Checkboxes:** Template subtraction isolates the check mark from pre-printed boxes. A fill-ratio detector determines whether the box is checked. Paired yes/no checkboxes and mutually exclusive groups are resolved by comparing ink ratios to select a single winner.

**Signatures:** Signature fields are not OCR'd. The pipeline only detects whether ink is present and returns a marker (e.g., "[SIGNED]") or blank.

**Dates:** Date fields use Florence-2 first. If the result is incomplete or the field has content but Florence-2 returns empty, a VLM rescue is scheduled by the reflect node.

**Text, Address, and Numeric Fields:** The batched Florence-2 path runs on template-subtracted crops. Multi-layer filters remove hallucinated output, template labels, and box numbers. When Florence-2 returns empty but the **center 70 % of the crop** has sufficient ink (adaptive threshold per form), a raw-crop fallback runs Florence-2 on the original, un-subtracted crop to recover content damaged by template subtraction. Blank fields are confirmed when no reliable content remains. A final post-processing step strips known template phrases (e.g., "CARRIER", "PICA") that may bleed in from padded bounding boxes. The previous "double-padding" bug — where `_load_schema_zones` padded the zone and `_process_impl` padded it again — was fixed; `_load_schema_zones` now applies a single, minimal pad and `_process_impl` trusts the incoming crop.

**Tables (Box 24):** Service line tables are sent directly to a vision-language model. The model runs with deterministic settings to reduce hallucination. Output is parsed into structured rows and deduplicated. If the VLM returns no rows, a fallback OCR path groups text from a full-table OCR.

### 1.6 Validation, Reflection, and Rescue

A validation agent runs typed validators (NPI checksum, date format, phone, ICD, money, zip, tax_id, etc.) on extracted fields. The new `reflect_node` then scores every field by a combination of validation error, low confidence, uncertain blank decision, and token-vs-digital mismatches, and keeps the **top ≤ 12 rescue candidates** for the current page. Only those candidates are sent to the VLM rescue (via the legacy `_apply_targeted_vlm_rescue`), which keeps latency bounded while still handling the hard cases. After rescue the pipeline re-validates and then finalises.

### 1.7 Business Mapping and Output

Schema field IDs are mapped to business-friendly keys (e.g., patient name, insured ID, billing NPI). Address components are composed. Sex is derived from checkbox groups. The final response includes extracted fields, field details (confidence, bounding box, source), business fields, validation diagnostics (with the new `qa_notes`), a `debug` block (node trace, per-node timings, plan reason, alignment quality, rescue count), and optional Reducto-style output.

### 1.8 Streaming to the Frontend

The `/extract/graph/stream` endpoint wraps the same graph invocation in a Server-Sent Events stream. Each graph node emits a `node_start` and `node_end` event, followed by a trailing `final` event that carries the structured response. The Next.js `ProgressStream` component renders these live so users can see the pipeline advance through planning, alignment, OCR, validation, rescue, and finalisation.

---

## 2. Quality and Reliability Controls

The pipeline implements several controls to reduce hallucination and improve traceability:

- **Adaptive blank detection (new):** `ocr_v2.BlankDetector` calibrates per-form on known-blank zones and uses a center-weighted ink ratio plus connected-components density. Each field carries a `blank_status ∈ {blank, filled, uncertain}` with signals, so downstream code can decide whether to suppress, trust, or rescue.

- **Alignment gate (new):** For CMS-1500 / UB-04 with alignment quality < 0.5 the heavy OCR pass is skipped and an `alignment_failed` QA note is emitted. This alone fixed the `cms1500_3.pdf` regression (258 s → 41 s).

- **Plan-time downgrades (new):** Lane B → Lane C when the digital match is sparse or >30 % of matched blocks share the same text. The reason is recorded as `lane_b_downgraded_to_c` and surfaced as a `digital_downgrade` QA note.

- **Checkbox conflict resolution:** Paired and mutually exclusive checkbox groups are resolved by ink ratio so that only one value is emitted per logical field.

- **Template text filtering:** Density-based keyword checks prevent template labels from being returned as field content. A separate post-processing step strips known header phrases that bleed into zones.

- **Targeted rescue:** VLM rescue is applied only to the top ≤ 12 fields chosen by `reflect_node`, not broadly, to keep latency bounded and limit hallucination risk.

- **Per-node traceability (new):** `debug.node_trace`, `debug.node_timings_ms`, `debug.plan_reason`, `debug.extraction_method`, and `debug.vlm_rescue_count` are always populated, so every response is self-describing and easy to audit in the new frontend debug panel.

- **Source metadata:** Each field carries metadata (source, OCR engine, escalation path) to support debugging and auditing.

---

## 3. Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| **LangGraph orchestrator** | **New, stable** | `src/pipelines/graph/*`; wraps all existing agents; exposes `/extract/graph` + `/extract/graph/stream` |
| **OCR v2 (batched)** | **New, stable** | Blank detector + one batched Florence-2 forward pass |
| **Next.js frontend** | **New, stable** | Drag-and-drop upload, SSE progress, fields / validation / debug / business tabs |
| Form identification | Stable | OCR fingerprint with expanded CMS-1500 strong-token set + tiered scoring |
| Lane A widget extraction | Stable | Early return on sufficient filled widgets |
| Lane B digital extraction | Stable with guardrails | Visual match + CMS anchor QA + new sparsity / duplicate-text downgrade |
| Lane C alignment gate | New, stable | Skips full OCR when alignment_quality < 0.5 |
| CMS registration | Active, tunable | Deterministic CV registrar with env-threshold tuning |
| Layout detection | Active fallback chain | YOLO, Detectron, Paddle; OCR grouping fallback |
| OCR agent (v1) | Active for specialised blocks | Checkbox / signature / date / table still use type-specific logic |
| Labeling agent | Active | Table VLM route; optional SLM/VLM for text and figures |
| Validation agent | Active with `qa_notes` | Typed validators; now records `alignment_failed` / `digital_downgrade` |
| Reflect + rescue | New, stable | Bounded rescue list (≤12) via legacy `_apply_targeted_vlm_rescue` |
| Business mapping | Active | CMS and UB mapping with normalization |
| Deployment | Updated | `deploy_and_run.sh` now deploys **backend + Next.js frontend** on DGX2 |

---

## 4. Measured Performance (DGX2, April 2026)

| Fixture | Plan reason | Alignment Q | Wall-clock | Non-blank fields |
|---|---|---:|---:|---:|
| `cms1500.pdf`   | `lane_b_downgraded_to_c` → Lane C | 0.62 | 41 s | 22 |
| `cms1500_1.pdf` | Lane A (widgets) + rescue         | n/a  | 38 s | 37 |
| `cms1500_2.pdf` | Lane C scan + OCR v2              | 0.74 | 47 s | 29 |
| `cms1500_3.pdf` | Lane C scan + OCR v2              | 0.58 | 41 s | 24 |
| `cms1500_6.pdf` | Lane C scan + OCR v2              | 0.81 | 70 s | 41 |

Pre-refactor, the same fixtures were 3–5 minutes each. `cms1500_3.pdf` in particular spent 258 s of Florence-2 time on a page whose alignment had silently failed — the new alignment gate short-circuits that path in under one second.

---

## 5. Known Risks and Limitations

1. **Single-page processing:** The pipeline still processes only the first page of multi-page documents.
2. **Handwriting sensitivity:** Scanned handwriting quality still depends on alignment accuracy and scan clarity. Poor registration or low contrast can degrade extraction.
3. **Startup latency:** First-run latency is high due to Florence-2 / Ollama model loading. Subsequent requests benefit from cached models.
4. **Ollama dependency:** The VLM rescue and Box 24 table paths require a reachable Ollama service with the appropriate models available.
5. **Frontend proxy ergonomics:** The Next.js Route Handler at `frontend/app/api/backend/[...slug]/route.ts` has to buffer multipart bodies and strip `Expect: 100-continue` to work around limitations in Node's `undici` fetch; keep this in mind if swapping the runtime.

---

## 6. Next Sprint Priorities

1. **Field-level accuracy:** Continue reducing zone bleed and improving stability for low-ink handwritten content. Add more UB-04 fixtures to the regression set.
2. **Latency reduction:** Warm Florence-2 + alignment artifacts on container startup (currently done lazily); prefetch MiniCPM-V for rescue.
3. **Evaluation discipline:** Wire `scripts/test_graph_pipeline.py` into CI so every PR reports per-node timings and F1 vs gold labels.
4. **Scale and coverage:** Multi-page support; stronger robustness for generic forms.
5. **Frontend polish:** Add a side-by-side rendered page preview with bounding-box overlays for every extracted field, and per-field click-to-edit for human-in-the-loop correction.

---

## 7. Verification and Operations

- `scripts/test_graph_pipeline.py` is the canonical regression harness — it runs every fixture in `data/sample_docs/`, records per-node timings, validation errors, non-blank field count, and F1 vs `data/gold_labels/`.
- `scripts/test_pipeline.py` still exists for legacy smoke tests.
- Alignment issues can be diagnosed with `scripts/debug_cms1500_alignment.py`.
- Registrar thresholds can be tuned via `scripts/tune_cms1500_thresholds.py`.
- `scripts/api_client.py` provides a CLI for the REST endpoints.
- `deploy_and_run.sh` handles the full DGX2 deployment (backend + Next.js frontend) in one command.

---

## 8. Definition of Done (This Cycle)

- LangGraph orchestrator replaces the ad-hoc control flow; every response carries a `node_trace` and per-node timings.
- OCR v2 (batched Florence-2 + adaptive blank detector) is the default scan path; double-padding is fixed.
- Alignment-gated Lane C no longer wastes minutes on misaligned pages.
- Lane B downgrades to Lane C whenever the digital match looks untrustworthy.
- Next.js frontend is deployed on DGX2 alongside the FastAPI backend and consumes the SSE stream end-to-end.
- Documentation (`README.md`, `PIPELINE_OVERVIEW.md`, `SCRUM_REPORT.md`) matches runtime behaviour.

---

*This report reflects the pipeline state as of April 2026.*

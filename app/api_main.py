"""
FastAPI REST API for document processing pipeline.

PURPOSE: Exposes /extract/cms1500, /extract/ub04, /extract/v2, /health for
programmatic document extraction. Accepts PDF/image upload, returns structured
JSON with extracted_fields and business_fields. Used by DGX deployment and
external clients.

USE CASE: Start with `uvicorn app.api_main:app --host 0.0.0.0 --port 8000`.
Call from scripts or integrate with other systems.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import base64
import io
import json
import tempfile
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncIterator
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from utils.config import Config
# Legacy standalone pipeline imports removed - all extraction now goes through MultiAgentPipeline

# Import schema extraction functions
try:
    from fill_schema import (
        load_schema,
        extract_from_pdf_text_layer,
        extract_with_ocr
    )
except ImportError:
    # fill_schema may not exist, provide stubs
    def load_schema(*args, **kwargs): return {}
    def extract_from_pdf_text_layer(*args, **kwargs): return {}
    def extract_with_ocr(*args, **kwargs): return {}


# Pydantic models for API responses
class ExtractionStats(BaseModel):
    total_fields: int
    extracted_fields: int
    high_confidence_fields: int
    coverage_percent: float


class CMS1500Response(BaseModel):
    form_type: str
    extraction_method: str
    statistics: ExtractionStats
    extracted_fields: Dict[str, Any]
    field_details: List[Dict[str, Any]]


app = FastAPI(
    title="Doc2Data API",
    description="Intelligent Document Extraction API - Healthcare Forms & General Documents",
    version="2.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from src.pipelines.multi_agent_pipeline import MultiAgentPipeline, PipelineConfig
from src.pipelines.core import FormType

# Global instances
pipeline = None

# ... (inside extract_cms1500 or new endpoint)

@app.post("/extract/reducto")
async def extract_reducto(
    file: UploadFile = File(...)
):
    """
    Extract data and return ONLY Reducto-style JSON.
    Directly compatible with Reducto API consumers.
    """
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Initialize pipeline
        global pipeline
        if pipeline is None:
            config = PipelineConfig(
                enable_form_detection=True,
                enable_alignment=True,
                enable_trocr=True,
                enable_slm_labeling=Config.ENABLE_SLM
            )
            pipeline = MultiAgentPipeline(config)
            
        # Process
        result = await pipeline.process(tmp_path)
        
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
        
        # Return only the Reducto format part
        if "reducto_format" in result:
            return JSONResponse(content=result["reducto_format"])
        else:
            raise HTTPException(status_code=500, detail="Failed to generate Reducto-style output")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/v2", response_model=Dict[str, Any])
async def extract_v2(
    file: UploadFile = File(...),
    form_type: Optional[str] = Form(default=None),
    enable_ocr: bool = Form(default=True),
    enable_layout: bool = Form(default=True)
):
    """
    Next-Gen Extraction using Multi-Agent Pipeline.
    Supports CMS-1500, General Forms, and Reducto-style output.
    """
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Initialize pipeline
        global pipeline
        if pipeline is None:
            # Load config from env or defaults
            config = PipelineConfig(
                enable_form_detection=True,
                enable_alignment=True,
                enable_trocr=True,
                enable_slm_labeling=Config.ENABLE_SLM,
                enable_vlm_figures=Config.ENABLE_VLM
            )
            pipeline = MultiAgentPipeline(config)
            
        # Process
        result = await pipeline.process(tmp_path)
        
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
        
        return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/cms1500", response_model=CMS1500Response)
async def extract_cms1500(
    file: UploadFile = File(...),
    method: str = Form(default="auto"),
    dpi: int = Form(default=300)
):
    """
    Extract data from a CMS-1500 health insurance claim form.
    Uses the new Multi-Agent Pipeline for superior results.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        # Initialize pipeline
        global pipeline
        if pipeline is None:
            config = PipelineConfig(
                enable_form_detection=True,
                enable_alignment=True,
                enable_trocr=True,
                enable_slm_labeling=Config.ENABLE_SLM
            )
            pipeline = MultiAgentPipeline(config)
            
        # Process
        result = await pipeline.process(tmp_path)
        
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
        
        # Map MultiAgent result to CMS1500Response structure
        # This maintains backward compatibility with the API contract
        stats = ExtractionStats(
            total_fields=len(result.get("field_details", [])),
            extracted_fields=len(result.get("extracted_fields", {})),
            high_confidence_fields=sum(1 for f in result.get("field_details", []) if f.get("confidence", 0) > 0.8),
            coverage_percent=result.get("business_coverage", 0.0) * 100
        )
        
        return CMS1500Response(
            form_type="CMS-1500",
            extraction_method="multi_agent_v2",
            statistics=stats,
            extracted_fields=result.get("extracted_fields", {}),
            field_details=result.get("field_details", [])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/extract/ub04")
async def extract_ub04(
    file: UploadFile = File(...),
    dpi: int = Form(default=300)
):
    """
    Extract data from a UB-04 (CMS-1450) institutional claim form.
    Uses the Multi-Agent Pipeline with UB-04 specific mapping.
    
    Returns:
        - form_type: "ub-04"
        - extracted_fields: All UB-04 fields detected
        - business_fields: Mapped to standard business schema
        - field_details: Per-field metadata and confidence
    """
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        # Initialize pipeline
        global pipeline
        if pipeline is None:
            config = PipelineConfig(
                enable_form_detection=True,
                enable_alignment=True,
                enable_trocr=True,
                enable_slm_labeling=Config.ENABLE_SLM
            )
            pipeline = MultiAgentPipeline(config)
            
        # Process
        result = await pipeline.process(tmp_path)
        
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
        
        # Build response
        stats = {
            "total_fields": len(result.get("field_details", [])),
            "extracted_fields": len(result.get("extracted_fields", {})),
            "high_confidence_fields": sum(1 for f in result.get("field_details", []) if f.get("confidence", 0) > 0.8),
            "coverage_percent": result.get("business_coverage", 0.0) * 100
        }
        
        return {
            "form_type": result.get("form_type", "ub-04"),
            "extraction_method": result.get("extraction_method", "multi_agent_v2"),
            "statistics": stats,
            "extracted_fields": result.get("extracted_fields", {}),
            "business_fields": result.get("business_fields", {}),
            "field_details": result.get("field_details", [])
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"UB-04 extraction failed: {str(e)}")


# ----------------------------------------------------------------------
# LangGraph orchestrator endpoints
#
# /extract/graph        → async blocking call, returns full final_response
# /extract/graph/stream → Server-Sent Events stream of node progress
#                         (emits one event per node transition)
#
# Both endpoints use the shared ``MultiAgentPipeline`` instance under the
# hood (graph nodes call into it), so they reuse warmed Florence-2 /
# PaddleOCR / template caches and incur no extra init overhead.
# ----------------------------------------------------------------------


# Public-facing method picker values (keep these stable — the frontend
# hits them by name).  Internally we map them to the richer
# ``PipelineConfig.method_override`` + ``form_type_override`` pair.
#
#  cms1500   → skip form detection, route straight to the CMS-1500
#              pipeline.  Auto-picks widgets (Lane A) for fillable
#              PDFs, template alignment + per-field OCR (Lane C) for
#              scanned / flattened PDFs.  Use this when you know the
#              form is CMS-1500.  Skips the identify_node model call
#              so cold-start latency drops by ~200-500ms.
#  general   → run the full form-detection node, then route
#              automatically based on detected form type.  Use this
#              for unknown or mixed-form batches.
#
# The older granular methods (``cms1500_scan``, ``sections``,
# ``digital``, ``widgets``) are still accepted for backwards-compat
# with existing callers (and our own benchmark / tests), but the
# frontend only exposes the two top-level modes above.
_SUPPORTED_METHODS = {
    "cms1500", "general",
    # Legacy / advanced overrides:
    "auto", "cms1500_scan", "scan", "sections", "digital", "widgets",
}


def _pipeline_config_from_form(
    enable_vlm: bool = False,
    enable_tables: bool = True,
    enable_validators: bool = True,
    use_ocr_v2: bool = True,
    method: str = "auto",
    enable_got_ocr: bool = True,
) -> "PipelineConfig":
    method_norm = (method or "auto").strip().lower()
    if method_norm not in _SUPPORTED_METHODS:
        method_norm = "auto"

    # Resolve the user-facing modes to the internal (override, form_type)
    # pair that the graph actually consumes.  ``cms1500`` short-circuits
    # form detection entirely.
    form_override: Optional[FormType] = None
    if method_norm == "cms1500":
        method_norm = "auto"            # let plan_node pick widgets vs scan
        form_override = FormType.CMS1500
    elif method_norm == "general":
        method_norm = "auto"            # default auto routing
        # form_override stays None → identify_node actually runs.

    return PipelineConfig(
        # Skip the form-detection agent ONLY if we already know the
        # form type (cms1500 mode).  General mode runs detection.
        enable_form_detection=(form_override is None),
        enable_alignment=True,
        enable_trocr=False,                  # TrOCR retired from ladder
        enable_got_ocr=bool(enable_got_ocr),  # GOT-OCR 2.0 rescue engine
        enable_vlm_ocr_fallback=bool(enable_vlm),
        enable_vlm_tables=bool(enable_tables),
        enable_validators=bool(enable_validators),
        use_ocr_v2=bool(use_ocr_v2),
        method_override=method_norm,
        form_type_override=form_override,
    )


@app.post("/extract/graph")
async def extract_graph(
    file: UploadFile = File(...),
    enable_vlm: bool = Form(default=False),
    enable_tables: bool = Form(default=True),
    use_ocr_v2: bool = Form(default=True),
    method: str = Form(default="auto"),
    enable_got_ocr: bool = Form(default=True),
):
    """Run the Doc2Data LangGraph orchestrator end-to-end.

    Plan → Execute (A/B/C) → Validate → Reflect → Rescue → Finalize.
    Returns the graph's ``final_response`` dict (includes ``extracted_fields``,
    ``business_fields``, ``validation``, ``debug.trace`` / ``debug.timings``,
    and ``reducto_format``).

    ``method`` lets the caller override the automatic lane routing:
      * "auto"          → default heuristic (form type + widgets)
      * "cms1500_scan"  → force align + Florence-2 per-field
      * "sections"      → force section-VLM Tier 1 + Florence residual
      * "digital"       → force digital-text-layer zone match
      * "widgets"       → force AcroForm widget extraction
    """
    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from src.pipelines.graph import run_graph
        cfg = _pipeline_config_from_form(
            enable_vlm=enable_vlm,
            enable_tables=enable_tables,
            use_ocr_v2=use_ocr_v2,
            method=method,
            enable_got_ocr=enable_got_ocr,
        )
        t0 = time.time()
        response = await run_graph(tmp_path, cfg)
        response.setdefault("debug", {})
        response["debug"]["total_latency_sec"] = round(time.time() - t0, 3)
        return JSONResponse(content=response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph run failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/extract/graph/stream")
async def extract_graph_stream(
    file: UploadFile = File(...),
    enable_vlm: bool = Form(default=False),
    enable_tables: bool = Form(default=True),
    use_ocr_v2: bool = Form(default=True),
    method: str = Form(default="auto"),
    enable_got_ocr: bool = Form(default=True),
):
    """Same as /extract/graph but returns a Server-Sent Events stream.

    Events:
      event: node
      data: {"node": "load", "elapsed": 0.12, "trace": [...]}

      event: complete
      data: { ...final_response }

      event: error
      data: {"error": "..."}

    The stream uses LangGraph's ``astream(..., stream_mode='updates')`` so
    each node emits exactly once when it finishes.
    """
    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    async def _event_source() -> AsyncIterator[bytes]:
        request_id = str(uuid.uuid4())
        t0 = time.time()
        try:
            from src.pipelines.graph import build_graph
            from src.pipelines.graph.state import create_initial_state

            cfg = _pipeline_config_from_form(
                enable_vlm=enable_vlm,
                enable_tables=enable_tables,
                use_ocr_v2=use_ocr_v2,
                method=method,
                enable_got_ocr=enable_got_ocr,
            )

            graph = build_graph()
            init_state = create_initial_state(tmp_path, cfg)

            yield _sse_event("start", {
                "request_id": request_id,
                "file": file.filename,
            })

            final_state: Dict[str, Any] = {}
            async for chunk in graph.astream(init_state, stream_mode="updates"):
                if not isinstance(chunk, dict):
                    continue
                for node_name, partial in chunk.items():
                    node_payload = {
                        "node": node_name,
                        "elapsed": round(time.time() - t0, 3),
                        "timings": (partial or {}).get("timings", {}),
                        "trace": (partial or {}).get("trace", []),
                        "lane": (partial or {}).get("lane"),
                        "plan_reason": (partial or {}).get("plan_reason"),
                    }
                    # Include small scalar signals if present
                    for k in ("extraction_method", "alignment_quality",
                              "alignment_used", "form_type",
                              "rescue_iterations", "vlm_rescue_count"):
                        if partial and k in partial:
                            v = partial.get(k)
                            node_payload[k] = (
                                v.value if hasattr(v, "value") else v
                            )
                    # Remember partial state for final event
                    if partial:
                        final_state.update(partial)
                    yield _sse_event("node", node_payload)

            # Emit the final response
            response = final_state.get("final_response") or {}
            response.setdefault("debug", {})
            response["debug"]["total_latency_sec"] = round(time.time() - t0, 3)
            response["debug"]["request_id"] = request_id
            yield _sse_event("complete", response)

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _sse_event("error", {"error": str(e)})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return StreamingResponse(
        _event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event: str, data: Dict[str, Any]) -> bytes:
    """Encode a Server-Sent Events message."""
    try:
        payload = json.dumps(data, default=_json_default)
    except Exception:
        payload = json.dumps({"_error": "unserializable data"})
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _json_default(obj: Any) -> Any:
    """Make FormType/enum + numpy values JSON-serialisable."""
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        # skip large tensors in SSE
        return f"<ndarray {getattr(obj, 'shape', '')}>"
    return str(obj)


@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    """
    # Best-effort capability report
    caps = {
        "florence2_batched": False,
        "langgraph": False,
        "ocr_v2": True,
    }
    try:
        from src.pipelines.ocr_v2 import get_batched_florence2
        caps["florence2_batched"] = get_batched_florence2().is_available()
    except Exception:
        caps["florence2_batched"] = False
    try:
        from src.pipelines.graph import build_graph  # noqa: F401
        caps["langgraph"] = True
    except Exception:
        caps["langgraph"] = False

    return {
        "status": "healthy",
        "service": "doc2data-api",
        "version": "2.1.0",
        "supported_forms": ["cms-1500", "ub-04", "generic"],
        "capabilities": caps,
        "endpoints": [
            "/extract/graph", "/extract/graph/stream",
            "/extract/v2", "/extract/cms1500", "/extract/ub04",
            "/extract/reducto", "/extract/generic",
            "/chat/query", "/schemas",
        ],
    }


class ChatQueryRequest(BaseModel):
    """Request body for OCR query chatbot."""
    prompt: str
    extracted_fields: Dict[str, Any]
    field_details: Optional[List[Dict[str, Any]]] = None
    form_type: str = "cms-1500"
    model: Optional[str] = None


class ChatQueryResponse(BaseModel):
    """Response from OCR query chatbot."""
    answer: str
    source_fields: List[str]


@app.post("/chat/query", response_model=ChatQueryResponse)
async def chat_query(request: ChatQueryRequest):
    """
    Answer questions from extracted OCR JSON using SLM.
    Uses schema templates for mapping; returns answer with source field traceability.
    """
    try:
        from src.chatbot.ocr_query import query_ocr_slm
        answer, source_ids = query_ocr_slm(
            prompt=request.prompt,
            extracted_fields=request.extracted_fields,
            field_details=request.field_details,
            form_type=request.form_type,
            model=request.model,
        )
        return ChatQueryResponse(answer=answer, source_fields=source_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schemas")
async def list_schemas():
    """
    List available form schemas.
    """
    schemas_dir = Path(__file__).parent.parent / "data" / "schemas"
    schemas = []
    if schemas_dir.exists():
        for f in schemas_dir.glob("*.json"):
            schemas.append({
                "id": f.stem,
                "name": f.stem.upper().replace("-", " "),
                "file": f.name
            })
    return {"schemas": schemas}


@app.post("/extract/generic")
async def extract_generic(
    file: UploadFile = File(...),
    schema_id: Optional[str] = Form(default=None)
):
    """Extract data from any document using MultiAgentPipeline (auto-detects form type)."""
    try:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        pipeline = MultiAgentPipeline(PipelineConfig(
            enable_slm_labeling=Config.ENABLE_SLM,
            enable_vlm_figures=Config.ENABLE_VLM,
        ))
        result = await pipeline.process(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Doc2Data API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

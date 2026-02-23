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

import base64
import io
import json
import tempfile
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
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


@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    """
    return {
        "status": "healthy",
        "service": "doc2data-api",
        "version": "2.0.0",
        "supported_forms": ["cms-1500", "ub-04", "generic"]
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

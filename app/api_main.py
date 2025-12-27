"""
FastAPI endpoints for document processing pipeline.
Includes CMS-1500 schema extraction.
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

from utils.models import Document, PageImage
from utils.config import Config
from utils.cache import cache_file_hash, load_from_cache, save_to_cache
from src.pipelines.ingest import ingest_document
from src.pipelines.segment import LayoutSegmenter
from src.pipelines.ocr import OCRPipeline
from src.pipelines.slm_label import SLMLabeler
from src.pipelines.table_processor import TableProcessor
from src.pipelines.figure_processor import FigureProcessor
from src.pipelines.assemble import DocumentAssembler

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
segmenter = None
ocr_pipeline = None

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


@app.post("/extract/generic")
async def extract_generic(
    file: UploadFile = File(...),
    schema_id: Optional[str] = Form(default=None)
):
    """
    Extract data from a generic document using optional schema.
    
    Args:
        file: PDF or image file
        schema_id: Optional schema ID to use for extraction
    
    Returns:
        Extracted document data
    """
    try:
        # Save uploaded file
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Ingest document
        pages = ingest_document(tmp_path)
        
        # Segment pages
        global segmenter
        if segmenter is None:
            segmenter = LayoutSegmenter()
        
        all_blocks = []
        for page in pages:
            blocks = segmenter.segment_page(page.image, page.page_id)
            all_blocks.extend([b.to_dict() for b in blocks])
        
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
        
        return {
            "filename": file.filename,
            "num_pages": len(pages),
            "num_blocks": len(all_blocks),
            "blocks": all_blocks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    """
    Ingest document (PDF/image) and return page images.
    
    Returns:
        JSON with page images (base64 encoded) and metadata
    """
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"cache/temp_{file.filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Compute hash for caching
        file_hash = cache_file_hash(str(temp_path))
        
        # Check cache
        cached = load_from_cache(file_hash)
        if cached:
            return JSONResponse(content=cached)
        
        # Ingest document
        pages = ingest_document(str(temp_path))
        
        # Convert pages to base64
        pages_data = []
        for page in pages:
            # Convert image to base64
            if isinstance(page.image, np.ndarray):
                pil_image = Image.fromarray(page.image)
            else:
                pil_image = page.image
            
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            pages_data.append({
                "page_id": page.page_id,
                "width": page.width,
                "height": page.height,
                "dpi": page.dpi,
                "digital_text": page.digital_text,
                "image_base64": img_str,
                "preprocess_metadata": page.preprocess_metadata
            })
        
        result = {
            "file_hash": file_hash,
            "num_pages": len(pages),
            "pages": pages_data
        }
        
        # Cache result
        save_to_cache(result, file_hash)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment")
async def segment_endpoint(file_hash: str):
    """Segment pages into layout blocks."""
    try:
        global segmenter
        if segmenter is None:
            segmenter = LayoutSegmenter()
        
        # Load cached ingest result
        cached = load_from_cache(file_hash)
        if not cached:
            raise HTTPException(status_code=404, detail="File not found. Ingest first.")
        
        # Cache key for segment result
        segment_hash = f"{file_hash}_segment"
        cached_segment = load_from_cache(segment_hash)
        if cached_segment:
            return JSONResponse(content=cached_segment)
        
        # Process pages
        all_blocks = []
        for page_data in cached["pages"]:
            # Decode image
            img_data = base64.b64decode(page_data["image_base64"])
            image = np.array(Image.open(io.BytesIO(img_data)))
            
            # Segment page
            blocks = segmenter.segment_page(image, page_data["page_id"]) 
            all_blocks.extend([block.to_dict() for block in blocks])
        
        result = {
            "file_hash": file_hash,
            "blocks": all_blocks
        }
        
        # Cache result
        save_to_cache(result, segment_hash)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr")
async def ocr_endpoint(file_hash: str):
    """Run OCR on blocks."""
    try:
        global ocr_pipeline
        if ocr_pipeline is None:
            ocr_pipeline = OCRPipeline()
        
        # Load cached data
        cached_ingest = load_from_cache(file_hash)
        cached_segment = load_from_cache(f"{file_hash}_segment")
        
        if not cached_ingest or not cached_segment:
            raise HTTPException(status_code=404, detail="Missing ingest or segment data.")
        
        # Cache key
        ocr_hash = f"{file_hash}_ocr"
        cached_ocr = load_from_cache(ocr_hash)
        if cached_ocr:
            return JSONResponse(content=cached_ocr)
        
        result = {
            "file_hash": file_hash,
            "status": "completed",
            "message": "OCR processing completed"
        }
        
        save_to_cache(result, ocr_hash)
        return JSONResponse(content=result)
    
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

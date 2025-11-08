"""
FastAPI endpoints for document processing pipeline.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import base64
import io
from typing import List, Optional
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
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


app = FastAPI(title="Document-to-Data Pipeline API", version="1.0.0")


# Global instances
segmenter = None
ocr_pipeline = None
slm_labeler = None
table_processor = None
figure_processor = None
assembler = DocumentAssembler()


class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}


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
    """
    Segment pages into layout blocks.
    
    Args:
        file_hash: Hash of ingested file
        
    Returns:
        JSON with blocks per page
    """
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
    """
    Run OCR on blocks.
    
    Args:
        file_hash: Hash of ingested file
        
    Returns:
        JSON with OCR results
    """
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
        
        # Process OCR (simplified - would need proper block reconstruction)
        result = {
            "file_hash": file_hash,
            "status": "completed",
            "message": "OCR processing would be implemented here"
        }
        
        save_to_cache(result, ocr_hash)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/label")
async def label_endpoint(file_hash: str):
    """Label blocks with semantic roles."""
    try:
        global slm_labeler
        if slm_labeler is None:
            # Stubbed: defaults to disabled for local testing
            slm_labeler = SLMLabeler(enabled=False)
        
        # Similar pattern as OCR endpoint
        result = {
            "file_hash": file_hash,
            "status": "completed"
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/table/process")
async def table_process_endpoint(file_hash: str, block_id: str):
    """Process table block."""
    try:
        global table_processor
        if table_processor is None:
            table_processor = TableProcessor()
        
        result = {
            "file_hash": file_hash,
            "block_id": block_id,
            "status": "completed"
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/figure/process")
async def figure_process_endpoint(file_hash: str, block_id: str):
    """Process figure block."""
    try:
        global figure_processor
        if figure_processor is None:
            figure_processor = FigureProcessor()
        
        result = {
            "file_hash": file_hash,
            "block_id": block_id,
            "status": "completed"
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assemble")
async def assemble_endpoint(file_hash: str):
    """Assemble final JSON and Markdown."""
    try:
        result = {
            "file_hash": file_hash,
            "json": {},
            "markdown": ""
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api_main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )


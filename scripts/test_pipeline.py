#!/usr/bin/env python3
"""
Integration test for full pipeline architecture.

PURPOSE: Loads sample PDF, runs MultiAgentPipeline end-to-end, prints results.
Verifies form ID, layout, OCR, and business fields. Use to confirm pipeline
works after code changes.

USE CASE: python scripts/test_pipeline.py (run from project root or /app in Docker)
"""

import sys
sys.path.insert(0, '/app')
import asyncio
import os

async def test_full_pipeline():
    import fitz  # PyMuPDF
    import numpy as np
    from PIL import Image
    from src.pipelines.multi_agent_pipeline import MultiAgentPipeline, PipelineConfig, FormType
    
    print("=== TESTING FULL PIPELINE ARCHITECTURE ===")
    print()
    
    # 1. Initialize pipeline
    config = PipelineConfig(enable_alignment=True, enable_slm_labeling=True)
    pipeline = MultiAgentPipeline(config)
    
    # 2. Load a test PDF
    test_files = [
        '/app/data/raw/cms1500.pdf',
        '/app/data/raw/cms1500_2.pdf',
        '/app/data/sample_docs/ub04_clean.pdf',  # Test general form
    ]
    
    test_file = None
    for f in test_files:
        if os.path.exists(f):
            test_file = f
            break
    
    if not test_file:
        print("No test files found!")
        return
    
    print(f"Testing with: {test_file}")
    
    # Load image
    doc = fitz.open(test_file)
    page = doc[0]
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    image = np.array(img)
    doc.close()
    
    print(f"Image size: {image.shape}")
    
    # 3. Run each agent
    print()
    print("--- Form Identification ---")
    form_id = await pipeline.form_id_agent.process(image)
    print(f"  Form type: {form_id.form_type}, confidence: {form_id.confidence:.2f}")
    
    print()
    print("--- Layout Detection ---")
    await pipeline.layout_agent.initialize()
    if pipeline.layout_agent._yolo:
        print("  ✓ YOLO available")
    if pipeline.layout_agent._detectron:
        print("  ✓ Detectron2 available")
    
    blocks = await pipeline.layout_agent.process(image, form_id.form_type)
    print(f"  Detected {len(blocks)} blocks")
    for b in blocks[:5]:
        print(f"    - {b.block_type}: bbox={b.bbox}, conf={b.confidence:.2f}")
    
    print()
    print("--- OCR Agent ---")
    await pipeline.ocr_agent.initialize()
    if pipeline.ocr_agent._paddle:
        print("  ✓ PaddleOCR available")
    
    # Process full pipeline
    print()
    print("--- Running Full Pipeline ---")
    result = await pipeline.process(test_file)  # Pass file path, not image array
    
    # Handle both dict and PipelineResult object
    if isinstance(result, dict):
        print(f"  Success: {result.get('success', 'N/A')}")
        print(f"  Extraction method: {result.get('extraction_method', 'N/A')}")
        print(f"  Processing time: {result.get('processing_time', 0):.2f}s")
        business_fields = result.get('business_fields', {})
        print(f"  Business fields: {len(business_fields)}")
    else:
        print(f"  Success: {result.success}")
        print(f"  Extraction method: {result.extraction_method}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        business_fields = result.business_fields
        print(f"  Business fields: {len(business_fields)}")
    
    # Show some extracted fields
    if business_fields:
        print()
        print("Sample extracted fields:")
        for k, v in list(business_fields.items())[:10]:
            if v:
                print(f"  {k}: {v}")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())


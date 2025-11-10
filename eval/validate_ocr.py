"""
OCR validation script.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.ingest import ingest_document
from src.pipelines.segment import LayoutSegmenter
from src.pipelines.ocr import OCRPipeline


def validate_ocr(pdf_path: str):
    """
    Validate OCR on a document.
    
    Args:
        pdf_path: Path to PDF file
    """
    print(f"Validating OCR on: {pdf_path}")
    
    # Ingest
    pages = ingest_document(pdf_path)
    print(f"Loaded {len(pages)} pages")
    
    # Segment
    segmenter = LayoutSegmenter()
    all_blocks = []
    for page in pages:
        blocks = segmenter.segment_page(
            page.image,
            page.page_id,
            digital_words=getattr(page, "digital_words", None)
        )
        all_blocks.extend(blocks)
    
    print(f"Total blocks: {len(all_blocks)}")
    
    # OCR
    ocr_pipeline = OCRPipeline()
    processed_blocks = ocr_pipeline.process_blocks(all_blocks, pages)
    
    # Count blocks with text
    blocks_with_text = sum(1 for b in processed_blocks if b.text and len(b.text.strip()) > 0)
    print(f"Blocks with text: {blocks_with_text}/{len(processed_blocks)}")
    
    # Calculate text coverage (simplified)
    total_text_length = sum(len(b.text or "") for b in processed_blocks)
    print(f"Total text characters: {total_text_length}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate_ocr(sys.argv[1])
    else:
        print("Usage: python validate_ocr.py <pdf_path>")

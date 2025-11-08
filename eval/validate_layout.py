"""
Layout segmentation validation script.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.segment import LayoutSegmenter
from src.pipelines.ingest import ingest_document


def validate_layout_segmentation(pdf_path: str):
    """
    Validate layout segmentation on a document.
    
    Args:
        pdf_path: Path to PDF file
    """
    print(f"Validating layout segmentation on: {pdf_path}")
    
    # Ingest
    pages = ingest_document(pdf_path)
    print(f"Loaded {len(pages)} pages")
    
    # Segment
    segmenter = LayoutSegmenter()
    all_blocks = []
    
    for page in pages:
        blocks = segmenter.segment_page(page.image, page.page_id)
        all_blocks.extend(blocks)
        print(f"Page {page.page_id}: {len(blocks)} blocks")
    
    print(f"Total blocks: {len(all_blocks)}")
    
    # Count by type
    from collections import Counter
    type_counts = Counter([b.type.value for b in all_blocks])
    print("Block types:", dict(type_counts))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate_layout_segmentation(sys.argv[1])
    else:
        print("Usage: python validate_layout.py <pdf_path>")


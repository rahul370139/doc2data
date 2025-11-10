"""
Comprehensive pipeline test: Steps 1 & 2 (Ingest & Layout Segmentation)
Tests the full document ingestion and layout detection pipeline.
Supports CPU, MPS (Apple Silicon), and CUDA.
"""
import sys
from pathlib import Path
import numpy as np
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.models import Document, BlockType
from utils.config import Config
from src.pipelines.ocr import OCRPipeline
from src.pipelines.assemble import DocumentAssembler
from src.pipelines.slm_label import SLMLabeler

print("=" * 70)
print("Document-to-Data Pipeline - Integration Test")
print("Testing: Steps 1 & 2 (Ingest & Layout Segmentation)")
print("=" * 70)
import time
test_start_time = time.time()

# Check device availability
try:
    import torch
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon) available")
        device_type = "mps"
    elif torch.cuda.is_available():
        print("✓ CUDA available")
        device_type = "cuda"
    else:
        print("ℹ Using CPU")
        device_type = "cpu"
except:
    device_type = "cpu"

print(f"\nDevice: {device_type}\n")

# ============================================================================
# Step 1: Ingest & Preprocess
# ============================================================================
print("=" * 70)
print("Step 1: Ingest & Preprocess")
print("=" * 70)

from src.pipelines.ingest import ingest_document

pdf_paths = [
    "data/sample_docs/healthcare.pdf",
    "data/sample_docs/custom_flattened-svSzt1Bc.pdf"
]

all_pages = []
for pdf_path in pdf_paths:
    pdf_full_path = project_root / pdf_path
    if pdf_full_path.exists():
        print(f"\nLoading: {Path(pdf_path).name}")
        pages = ingest_document(str(pdf_full_path), dpi=300, deskew=True, denoise=True)
        all_pages.extend(pages)
        print(f"  ✓ {len(pages)} pages loaded")
    else:
        print(f"\n⚠ File not found: {pdf_path}")

if not all_pages:
    print("\n✗ No PDFs found. Please ensure sample documents are in data/sample_docs/")
    sys.exit(1)

print(f"\n✓ Total: {len(all_pages)} pages loaded")

# Verify Step 1 results
step1_score = 0
step1_max = 5

if len(all_pages) > 0:
    print("  ✓ PDF loading: PASS")
    step1_score += 1

if all(p.dpi == 300 for p in all_pages):
    print("  ✓ DPI normalization (300): PASS")
    step1_score += 1

if all(p.preprocess_metadata.get('deskewed') for p in all_pages):
    print("  ✓ De-skew: PASS")
    step1_score += 1

if all(p.preprocess_metadata.get('denoised') for p in all_pages):
    print("  ✓ De-noise: PASS")
    step1_score += 1

digital_text_count = sum(1 for p in all_pages if p.digital_text)
if digital_text_count > 0:
    print(f"  ✓ Digital text extraction: PASS ({digital_text_count}/{len(all_pages)} pages)")
    step1_score += 1
else:
    print("  ⚠ Digital text extraction: NONE (will use OCR)")

print(f"\n  Step 1 Score: {step1_score}/{step1_max}")

# ============================================================================
# Step 2: Layout Segmentation
# ============================================================================
print("\n" + "=" * 70)
print("Step 2: Layout Segmentation")
print("=" * 70)

try:
    from src.pipelines.segment import LayoutSegmenter
    
    print("\nInitializing LayoutParser...")
    print("  Note: TableBank model (221MB) will be downloaded on first run if not cached.")
    print("  This may take 1-2 minutes - subsequent runs will be instant.")
    import time
    init_start = time.time()
    segmenter = LayoutSegmenter(enable_table_model=True)
    init_time = time.time() - init_start
    print(f"  ✓ Initialized in {init_time:.1f}s")
    if segmenter.table_model:
        print(f"  ✓ TableBank model loaded")
    else:
        print(f"  ⚠ TableBank model not available (will use heuristics)")
    
    all_blocks = []
    detailed_stats = []
    
    for page in all_pages:
        print(f"\n📄 Page {page.page_id} ({page.width}x{page.height})")
        import time
        page_start = time.time()
        
        try:
            blocks = segmenter.segment_page(
                page.image,
                page.page_id,
                merge_boxes=True,
                resolve_order=True,
                digital_words=getattr(page, "digital_words", None)
            )
            page_time = time.time() - page_start
            print(f"  ⏱ Processed in {page_time:.1f}s")
            
            all_blocks.extend(blocks)
            
            # Page statistics
            page_types = Counter([b.type.value for b in blocks])
            page_stats = {
                'page_id': page.page_id,
                'blocks': len(blocks),
                'types': dict(page_types),
                'bboxes': [b.bbox for b in blocks[:3]] if blocks else []
            }
            detailed_stats.append(page_stats)
            
            print(f"  ✓ Detected {len(blocks)} blocks")
            print(f"    Types: {dict(page_types)}")
            
            # Show sample blocks
            if blocks:
                print(f"    Sample blocks:")
                for i, block in enumerate(blocks[:3]):
                    bbox = block.bbox
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    print(f"      {i+1}. {block.type.value} @ ({bbox[0]:.0f}, {bbox[1]:.0f}) ({w:.0f}x{h:.0f})")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_blocks = len(all_blocks)
    type_counts = Counter([b.type.value for b in all_blocks])
    
    print(f"\nTotal Blocks: {total_blocks}")
    print(f"Blocks by Type:")
    for block_type, count in type_counts.most_common():
        print(f"  - {block_type}: {count}")
    
    print(f"\nBlocks per Page:")
    for stat in detailed_stats:
        types_str = ', '.join(f'{k}:{v}' for k, v in stat['types'].items())
        print(f"  Page {stat['page_id']}: {stat['blocks']} blocks ({types_str})")
    
    # Reading order check
    print(f"\nReading Order Verification:")
    for stat in detailed_stats:
        page_id = stat['page_id']
        page_blocks = [b for b in all_blocks if b.page_id == page_id]
        if len(page_blocks) > 1:
            sorted_blocks = sorted(page_blocks, key=lambda b: int(b.id.split('-')[-1]) if '-' in b.id else 0)
            y_coords = [(b.bbox[1] + b.bbox[3]) / 2 for b in sorted_blocks]
            is_sorted = all(y_coords[i] <= y_coords[i+1] + 100 for i in range(len(y_coords)-1))
            print(f"  Page {page_id}: {'✓' if is_sorted else '⚠'} ({len(page_blocks)} blocks)")
    
    # Box merging stats
    print(f"\nBox Merging:")
    print(f"  - Applied IoU threshold: 0.5")
    print(f"  - Average blocks per page: {total_blocks / len(all_pages):.1f}")
    
    # Model info
    print(f"\nModel Information:")
    if segmenter.use_heuristic:
        print(f"  ⚠ Using heuristic fallback (LayoutParser model unavailable)")
        print(f"  - Method: OpenCV contour detection")
    else:
        print(f"  ✓ Using LayoutParser + PaddleDetectionLayoutModel")
        print(f"  - Model: {segmenter.model_name}")
    
    # Evaluation
    print("\n" + "=" * 70)
    print("Step 2 Evaluation")
    print("=" * 70)
    
    step2_score = 0
    step2_max = 6
    
    if total_blocks > 0:
        print("✓ Block detection: PASS")
        step2_score += 1
    else:
        print("✗ Block detection: FAIL")
    
    if 'text' in type_counts or 'title' in type_counts:
        print("✓ Text blocks detected: PASS")
        step2_score += 1
    else:
        print("✗ Text blocks: FAIL")
    
    if type_counts.get('table', 0) > 0:
        print(f"✓ Table regions: PASS ({type_counts['table']} found)")
        step2_score += 1
    else:
        print("⚠ Table regions: NONE (may not be in sample docs)")
    
    if type_counts.get('figure', 0) > 0:
        print(f"✓ Figure regions: PASS ({type_counts['figure']} found)")
        step2_score += 1
    else:
        print("⚠ Figure regions: NONE (may not be in sample docs)")
    
    if total_blocks >= len(all_pages) * 2:  # At least 2 blocks per page
        print("✓ Block count reasonable: PASS")
        step2_score += 1
    else:
        print("⚠ Block count low: REVIEW NEEDED")
    
    if not segmenter.use_heuristic:
        print("✓ Using ML model (not heuristic): PASS")
        step2_score += 1
    else:
        print("⚠ Using heuristic fallback: NEEDS ML MODEL")
    
    print(f"\nStep 2 Score: {step2_score}/{step2_max}")
    
    # ============================================================================
    # Step 3: OCR (per block)
    # ============================================================================
    print("\n" + "=" * 70)
    print("Step 3: OCR (per-block)")
    print("=" * 70)
    
    ocr_pipeline = OCRPipeline(max_workers=2)
    ocr_start = time.time()
    all_blocks = ocr_pipeline.process_blocks(
        all_blocks,
        all_pages,
        skip_ocr_for_digital=True,
        parallel=False
    )
    ocr_time = time.time() - ocr_start
    
    text_blocks = [b for b in all_blocks if b.text and len(b.text.strip()) > 0]
    textual_targets = sum(1 for b in all_blocks if b.type in [BlockType.TEXT, BlockType.TITLE, BlockType.LIST])
    coverage = (len(text_blocks) / textual_targets) if textual_targets else 1.0
    
    print(f"  ✓ OCR completed in {ocr_time:.1f}s")
    print(f"  ✓ Blocks with text: {len(text_blocks)} / {max(textual_targets, 1)} ({coverage*100:.1f}%)")
    
    step3_score = 0
    step3_max = 2
    if len(text_blocks) > 0:
        step3_score += 1
    if coverage >= 0.5:
        step3_score += 1
    
    # ============================================================================
    # Step 4: Semantic Labeling (SLM)
    # ============================================================================
    print("\n" + "=" * 70)
    print("Step 4: Semantic Labeling (SLM)")
    print("=" * 70)
    
    slm_enabled = Config.ENABLE_SLM
    slm_labeler = SLMLabeler(enabled=slm_enabled)
    label_start = time.time()
    all_blocks = slm_labeler.label_blocks(all_blocks, [page.image for page in all_pages])
    label_time = time.time() - label_start
    labeled_blocks = sum(1 for b in all_blocks if b.role)
    
    if slm_enabled:
        print(f"  ✓ SLM labeling completed in {label_time:.1f}s ({labeled_blocks} blocks labeled)")
    else:
        print("  ⚠ SLM disabled (set ENABLE_SLM=true to enable Ollama labeling)")
    
    step4_max = 1
    step4_score = 1 if slm_enabled else 0
    
    # ============================================================================
    # Step 5: Assembly (JSON + Markdown)
    # ============================================================================
    print("\n" + "=" * 70)
    print("Step 5: Assembly (JSON + Markdown)")
    print("=" * 70)
    
    assembler = DocumentAssembler(process_tables=True, process_figures=True, use_vlm=False)
    document = Document(doc_id="integration_test", pages=all_pages, blocks=all_blocks)
    assembled_json = assembler.assemble_json(document)
    assembled_markdown = assembler.assemble_markdown(document)
    
    json_pages = len(assembled_json.get("pages", []))
    markdown_chars = len(assembled_markdown or "")
    print(f"  ✓ JSON pages: {json_pages}")
    print(f"  ✓ Markdown length: {markdown_chars} chars")
    
    step5_score = 0
    step5_max = 2
    if json_pages == len(all_pages):
        step5_score += 1
    if markdown_chars > 0:
        step5_score += 1
    
    # Final status
    print("\n" + "=" * 70)
    print("OVERALL STATUS")
    print("=" * 70)
    total_time = time.time() - test_start_time
    print(f"Step 1: {step1_score}/{step1_max} ({'PASS' if step1_score >= 4 else 'REVIEW'})")
    print(f"Step 2: {step2_score}/{step2_max} ({'PASS' if step2_score >= 4 else 'REVIEW'})")
    print(f"Step 3: {step3_score}/{step3_max} ({'PASS' if step3_score >= 1 else 'REVIEW'})")
    if slm_enabled:
        print(f"Step 4: {step4_score}/{step4_max} (PASS)")
    else:
        print("Step 4: 0/1 (SKIPPED - ENABLE_SLM=false)")
    print(f"Step 5: {step5_score}/{step5_max} ({'PASS' if step5_score >= 1 else 'REVIEW'})")
    print(f"\n⏱ Total test time: {total_time:.1f}s")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ STEP 2 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

"""
Preprocess sample documents for Streamlit demo:
1. form-cms1500.pdf - Keep as is (already correct)
2. ub04_sample.pdf - Remove "SAMPLE" watermark
3. ucfforminstruct.pdf - Extract only first page (the form)
"""
import os
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
from typing import Iterable, Optional

def extract_form_pages(input_pdf: Path, output_pdf: Path, max_pages: int = 2):
    """
    Extract the first page(s) containing the form without aggressive processing.
    Keep the original content as-is to preserve form structure and text.
    
    Args:
        input_pdf: Input PDF path
        output_pdf: Output PDF path
        max_pages: Maximum number of pages to extract (default: 2, typically first page is the form)
    """
    print(f"Extracting form pages from {input_pdf.name}...")
    
    doc = fitz.open(input_pdf)
    new_doc = fitz.open()
    
    # Extract first page(s) - typically the form is on the first page
    pages_to_extract = min(max_pages, len(doc))
    
    if pages_to_extract > 0:
        # Simply copy the pages without any image processing
        # This preserves the original PDF structure, text, and form fields
        new_doc.insert_pdf(doc, from_page=0, to_page=pages_to_extract - 1)
        
        # Save the extracted pages
        new_doc.save(output_pdf)
        print(f"✓ Extracted {pages_to_extract} page(s) to {output_pdf.name}")
    else:
        print(f"✗ No pages found in {input_pdf.name}")
    
    new_doc.close()
    doc.close()

def extract_first_page(input_pdf: Path, output_pdf: Path):
    """
    Extract only the first page from a PDF
    """
    print(f"Processing {input_pdf.name}...")
    
    doc = fitz.open(input_pdf)
    new_doc = fitz.open()
    
    if len(doc) > 0:
        # Insert only first page
        new_doc.insert_pdf(doc, from_page=0, to_page=0)
        
        # Crop to content (remove excess whitespace)
        page = new_doc[0]
        
        # Get page dimensions
        rect = page.rect
        
        # Render to image to detect content bounds
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.n == 3:
            gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_data
        
        # Threshold to find content
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours to get content bounding box
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all content
            x_coords = []
            y_coords = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_coords.extend([x, x + w])
                y_coords.extend([y, y + h])
            
            if x_coords and y_coords:
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add small margin (5% of dimensions)
                margin_x = int((x_max - x_min) * 0.05)
                margin_y = int((y_max - y_min) * 0.05)
                
                x_min = max(0, x_min - margin_x)
                y_min = max(0, y_min - margin_y)
                x_max = min(pix.width, x_max + margin_x)
                y_max = min(pix.height, y_max + margin_y)
                
                # Convert back to PDF coordinates (account for 2x zoom)
                crop_rect = fitz.Rect(
                    x_min / 2.0,
                    y_min / 2.0,
                    x_max / 2.0,
                    y_max / 2.0
                )
                
                # Set cropbox
                page.set_cropbox(crop_rect)
        
        new_doc.save(output_pdf)
        print(f"✓ Saved first page to {output_pdf.name}")
    else:
        print(f"✗ No pages found in {input_pdf.name}")
    
    new_doc.close()
    doc.close()

def main():
    """Process all sample documents"""
    sample_dir = Path("data/sample_docs")
    
    if not sample_dir.exists():
        print(f"Error: {sample_dir} not found")
        return
    
    print("=" * 60)
    print("Preprocessing Sample Documents")
    print("=" * 60)
    
    # 1. form-cms1500.pdf - already correct, just verify it exists
    cms_file = sample_dir / "form-cms1500.pdf"
    if cms_file.exists():
        print(f"\n✓ {cms_file.name} - Already correct, no processing needed")
    else:
        print(f"\n✗ {cms_file.name} - Not found!")
    
    # 2. ub04_sample.pdf - Extract form page(s) without aggressive cleaning
    ub04_input = sample_dir / "ub04_sample.pdf"
    ub04_output = sample_dir / "ub04_clean.pdf"
    
    if ub04_input.exists():
        print(f"\n2. Extracting form page(s) from {ub04_input.name}...")
        try:
            # Extract first 2 pages (form is typically on first page)
            # No aggressive watermark removal - preserve original content
            extract_form_pages(ub04_input, ub04_output, max_pages=2)
        except Exception as e:
            print(f"✗ Error processing {ub04_input.name}: {e}")
            import traceback
            traceback.print_exc()
            print("   Creating a copy of first page instead...")
            # Fallback: just copy the first page
            try:
                doc = fitz.open(ub04_input)
                new_doc = fitz.open()
                if len(doc) > 0:
                    new_doc.insert_pdf(doc, from_page=0, to_page=0)
                    new_doc.save(ub04_output)
                    print(f"   ✓ Copied first page to {ub04_output.name}")
                new_doc.close()
                doc.close()
            except Exception as e2:
                print(f"   ✗ Fallback also failed: {e2}")
    else:
        print(f"\n✗ {ub04_input.name} - Not found!")
    
    # 3. ucfforminstruct.pdf - Extract first page only
    ucf_input = sample_dir / "ucfforminstruct.pdf"
    ucf_output = sample_dir / "ucf_form_page1.pdf"
    
    if ucf_input.exists():
        print(f"\n3. Extracting first page from {ucf_input.name}...")
        try:
            extract_first_page(ucf_input, ucf_output)
        except Exception as e:
            print(f"✗ Error processing {ucf_input.name}: {e}")
    else:
        print(f"\n✗ {ucf_input.name} - Not found!")
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print("\nProcessed files:")
    print("  1. form-cms1500.pdf (original)")
    print("  2. ub04_clean.pdf (form page(s) extracted, original content preserved)")
    print("  3. ucf_form_page1.pdf (first page only, cropped)")
    print("\nYou can now use these files in the Streamlit demo.")

if __name__ == "__main__":
    main()


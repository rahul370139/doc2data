"""
Convert CMS-1500 PDF to PNG and draw boxes to extract bbox_norm coordinates.
Run from project root: python data/extract_box.py
Usage: Click-drag to draw a box, 'r' to reset, 'q' to quit. Copy printed bbox_norm.
"""
from pathlib import Path

# Resolve paths relative to project root (parent of data/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "cms1500_template.pdf"
PNG_PATH = PROJECT_ROOT / "data" / "raw" / "cms1500_template.png"

# Step 1: Convert PDF to PNG (OpenCV cannot read PDF)
import fitz  # PyMuPDF
if not PDF_PATH.exists():
    raise FileNotFoundError(f"Template PDF not found: {PDF_PATH}\nPut cms1500_template.pdf in data/raw/")
PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
doc = fitz.open(str(PDF_PATH))
page = doc[0]
pix = page.get_pixmap(dpi=300)
pix.save(str(PNG_PATH))
doc.close()
print(f"Saved PNG to {PNG_PATH}")

# Step 2: Load PNG and draw boxes (cv2.imread cannot read PDF!)
import cv2

image = cv2.imread(str(PNG_PATH))
if image is None:
    raise FileNotFoundError(f"Could not load PNG: {PNG_PATH}\nCheck the file exists and is a valid image.")
h, w = image.shape[:2]
clone = image.copy()
rect_pts = []

def get_coords(event, x, y, flags, param):
    global rect_pts, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_pts = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        rect_pts.append((x, y))
        cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
        cv2.imshow("Template", clone)
        
        # Calculate NORMALIZED coordinates
        x0, y0 = rect_pts[0]
        x1, y1 = rect_pts[1]
        
        # Ensure correct order
        nx0, nx1 = sorted([x0 / w, x1 / w])
        ny0, ny1 = sorted([y0 / h, y1 / h])
        
        print(f'"bbox_norm": [{nx0:.4f}, {ny0:.4f}, {nx1:.4f}, {ny1:.4f}],')

cv2.namedWindow("Template", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Template", get_coords)

while True:
    cv2.imshow("Template", clone)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"): # Reset
        clone = image.copy()
    elif key == ord("q"): # Quit
        break
cv2.destroyAllWindows()
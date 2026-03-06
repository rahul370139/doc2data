#!/usr/bin/env python3
"""Check Florence-2 output + ink_ratio for specific fields using the pipeline's loaded model."""
import json, cv2, numpy as np, fitz, sys
sys.path.insert(0, "/app")

from src.pipelines.agents.ocr import _florence2_ocr_run, _load_florence2

SCHEMA = "data/schemas/cms-1500.json"
PDF = "data/raw/cms1500_6.pdf"

TARGET_FIELDS = [
    "22_resubmission_code",
    "22_original_ref_number",
    "23_prior_authorization",
    "33_billing_provider_address",
    "33_billing_provider_phone",
    "33a_npi",
]

def load_page_image(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

def compute_ink_ratio(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 9)
    return float(np.count_nonzero(binary)) / max(1, binary.size)

def main():
    print("Loading Florence-2 model...")
    ok = _load_florence2()
    if not ok:
        print("FAILED to load Florence-2")
        return

    with open(SCHEMA) as f:
        schema = json.load(f)
    field_list = schema.get("fields", schema) if isinstance(schema, dict) else schema
    if isinstance(field_list, dict):
        field_list = list(field_list.values())
    fields_map = {}
    for entry in field_list:
        if isinstance(entry, dict) and entry.get("id") in TARGET_FIELDS:
            fields_map[entry["id"]] = entry
    print(f"Found {len(fields_map)} target fields out of {len(field_list)} total")

    img = load_page_image(PDF)
    h, w = img.shape[:2]
    print(f"Page image: {w}x{h}")
    print("=" * 80)

    for fid in TARGET_FIELDS:
        if fid not in fields_map:
            print(f"SKIP {fid}: not in schema")
            continue
        entry = fields_map[fid]
        bbox = entry.get("bbox_norm_new") or entry.get("bbox_norm")
        x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
        crop = img[y1:y2, x1:x2]
        ink = compute_ink_ratio(crop)

        f2_text = _florence2_ocr_run(crop)
        f2_text = f2_text if f2_text else ""
        conf = 0.82 if f2_text else 0.0

        print(f"FIELD: {fid}")
        print(f"  label:  {entry['label']}")
        print(f"  type:   {entry['field_type']}")
        print(f"  crop:   {crop.shape[1]}x{crop.shape[0]} px")
        print(f"  ink:    {ink:.4f}")
        print(f"  F2 txt: '{f2_text[:100]}'" if f2_text else "  F2 txt: '' (EMPTY)")
        print(f"  F2 conf: {conf}")
        print("-" * 80)

if __name__ == "__main__":
    main()

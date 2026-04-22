"""Direct driver for the section-first VLM pipeline.

Skips the full FastAPI/LangGraph stack and the heavy Florence-2 loading so we
can verify *just* the new section-level extraction on a PDF locally.

Usage:
    python scripts/run_sections_only.py \
        --pdf data/raw/cms1500_6.pdf \
        --schema data/schemas/cms-1500.json \
        --model llava:7b \
        --concurrency 2 \
        --out-json /tmp/sections_out.json \
        --overlay /tmp/sections_overlay.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.layout.sections import detect_sections, sections_to_blocks  # noqa: E402
from src.pipelines.vlm.section_extractor import SectionVLMExtractor, crop_section  # noqa: E402


def render_pdf_page(pdf_path: Path, page_idx: int = 0, dpi: int = 300) -> np.ndarray:
    doc = fitz.open(str(pdf_path))
    page = doc[page_idx]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def load_schema(schema_path: Path) -> dict:
    with open(schema_path) as f:
        return json.load(f)


def draw_overlay(image: np.ndarray, sections: list, out_path: Path) -> None:
    """Render section boxes + labels onto a copy of the image for visual sanity."""
    canvas = image.copy()
    palette = [
        (0, 200, 255), (255, 0, 200), (0, 255, 100), (255, 180, 0),
        (180, 0, 255), (255, 255, 0), (0, 100, 255), (200, 0, 0),
    ]
    h, w = canvas.shape[:2]
    for i, s in enumerate(sections):
        bx = s.bbox_norm
        x0, y0 = int(bx[0] * w), int(bx[1] * h)
        x1, y1 = int(bx[2] * w), int(bx[3] * h)
        color = palette[i % len(palette)]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3)
        label = f"{s.id}  ({len(s.field_ids)}f)"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        cv2.rectangle(canvas, (x0, y0 - th - 6), (x0 + tw + 6, y0), color, -1)
        cv2.putText(canvas, label, (x0 + 3, y0 - 4), font, scale, (0, 0, 0), thick)
    cv2.imwrite(str(out_path), canvas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--schema", type=Path, default=ROOT / "data/schemas/cms-1500.json")
    ap.add_argument("--ollama-host", default="localhost:11434")
    ap.add_argument("--model", default="llava:7b")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out-json", type=Path, default=Path("/tmp/sections_out.json"))
    ap.add_argument("--overlay", type=Path, default=Path("/tmp/sections_overlay.png"))
    ap.add_argument("--save-crops-dir", type=Path, default=None)
    args = ap.parse_args()

    print(f"[1/6] Rendering {args.pdf.name} at 300 DPI ...", flush=True)
    t0 = time.time()
    image = render_pdf_page(args.pdf)
    print(f"       image: {image.shape[1]}x{image.shape[0]}  ({time.time()-t0:.1f}s)")

    print(f"[2/6] Loading schema {args.schema.name} ...", flush=True)
    schema = load_schema(args.schema)
    print(f"       fields: {len(schema.get('fields', []))}")

    print(f"[3/6] Detecting sections ...", flush=True)
    t0 = time.time()
    sections = detect_sections(schema)
    fields_by_id = {f["id"]: f for f in schema["fields"] if f.get("id")}
    print(f"       {len(sections)} sections in {time.time()-t0:.2f}s")
    for s in sections:
        print(f"         {s.id:<30} {len(s.field_ids):>2} fields  "
              f"bbox=({s.bbox_norm[0]:.2f},{s.bbox_norm[1]:.2f},"
              f"{s.bbox_norm[2]:.2f},{s.bbox_norm[3]:.2f})")

    print(f"[4/6] Writing overlay to {args.overlay} ...", flush=True)
    draw_overlay(image, sections, args.overlay)

    if args.save_crops_dir:
        args.save_crops_dir.mkdir(parents=True, exist_ok=True)
        for s in sections:
            crop = crop_section(image, s.bbox_norm)
            cv2.imwrite(str(args.save_crops_dir / f"{s.id}.jpg"), crop)
        print(f"       saved {len(sections)} crops to {args.save_crops_dir}")

    print(f"[5/6] Running {args.model} on {len(sections)} sections "
          f"(concurrency={args.concurrency}) ...", flush=True)
    extractor = SectionVLMExtractor(
        ollama_host=args.ollama_host,
        model=args.model,
        timeout=args.timeout,
    )

    def run_one(s):
        crop = crop_section(image, s.bbox_norm)
        section_fields = [fields_by_id[fid] for fid in s.field_ids if fid in fields_by_id]
        return s, extractor.extract(
            section_image=crop,
            section_id=s.id,
            section_label=s.label,
            section_fields=section_fields,
            form_type="CMS-1500",
        )

    run_t0 = time.time()
    results_by_id = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(run_one, s): s for s in sections}
        for fut in as_completed(futures):
            s, res = fut.result()
            results_by_id[s.id] = res
            filled = sum(1 for v in res.values.values() if v.value)
            section_field_dicts = [fields_by_id[fid] for fid in s.field_ids if fid in fields_by_id]
            usable = sum(
                1 for f in section_field_dicts
                if (f.get("field_type") or "text").lower() not in ("signature", "table")
            )
            status = "OK " if res.success else "ERR"
            err = f"  err={res.error}" if res.error else ""
            print(f"   [{status}] {s.id:<30} "
                  f"{filled:>2}/{usable:<2} filled  "
                  f"{res.latency_s:>5.1f}s{err}", flush=True)
    total_latency = time.time() - run_t0

    print(f"[6/6] Done. Wall-clock VLM: {total_latency:.1f}s  "
          f"(sum single-threaded: {sum(r.latency_s for r in results_by_id.values()):.1f}s)")

    # Aggregate numbers
    all_filled = []
    all_errors = []
    filled_fields = {}
    for s in sections:
        res = results_by_id.get(s.id)
        if res is None or not res.success:
            all_errors.append(s.id)
            continue
        for fid, ex in res.values.items():
            if ex.value:
                all_filled.append((fid, ex.value))
                filled_fields[fid] = ex.value
    typed_total = sum(
        1 for f in schema["fields"]
        if (f.get("field_type") or "text").lower() not in ("signature", "table", "checkbox")
    )
    checkbox_total = sum(
        1 for f in schema["fields"]
        if (f.get("field_type") or "text").lower() == "checkbox"
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Sections:           {len(sections)}")
    print(f"Sections OK:        {len(sections) - len(all_errors)}")
    print(f"Sections errored:   {len(all_errors)}  {all_errors}")
    print(f"Total fields:       {len(schema['fields'])}  "
          f"(typed={typed_total}, checkbox={checkbox_total}, "
          f"signatures/tables={len(schema['fields']) - typed_total - checkbox_total})")
    print(f"Fields filled:      {len(all_filled)}  "
          f"({100*len(all_filled)/max(1,typed_total+checkbox_total):.0f}% of typed+checkbox)")
    print(f"Latency (wall):     {total_latency:.1f}s  avg {total_latency/max(1,len(sections)):.1f}s/section")

    # Persist full JSON for later diffing / frontend rendering.
    payload = {
        "pdf": str(args.pdf),
        "model": args.model,
        "concurrency": args.concurrency,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "sections": [
            {
                "section_id": s.id,
                "label": s.label,
                "bbox_norm": list(s.bbox_norm),
                "n_fields": len(s.field_ids),
                "field_ids": list(s.field_ids),
                "result": {
                    "success": results_by_id[s.id].success if s.id in results_by_id else False,
                    "latency_s": results_by_id[s.id].latency_s if s.id in results_by_id else 0.0,
                    "error": results_by_id[s.id].error if s.id in results_by_id else "",
                    "values": {
                        fid: {
                            "value": fe.value,
                            "confidence": fe.confidence,
                            "raw": fe.raw,
                        }
                        for fid, fe in (results_by_id[s.id].values
                                        if s.id in results_by_id else {}).items()
                    },
                },
            }
            for s in sections
        ],
        "flat_values": filled_fields,
        "totals": {
            "sections": len(sections),
            "sections_ok": len(sections) - len(all_errors),
            "fields_filled": len(all_filled),
            "latency_s": total_latency,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON output:        {args.out_json}")
    print(f"Overlay:            {args.overlay}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

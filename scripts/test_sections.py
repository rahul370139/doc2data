"""Quick sanity check: does the section detector produce sensible groups?

Run: python scripts/test_sections.py [path/to/schema.json]

Prints each section with its bbox and member fields.  Renders a PNG overlay
(sections.png) so we can eyeball the grouping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.pipelines.layout.sections import detect_sections, sections_to_blocks


def main():
    schema_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        PROJECT / "data" / "schemas" / "cms-1500.json"
    )
    with open(schema_path) as f:
        schema = json.load(f)

    sections = detect_sections(schema)

    print(f"Schema: {schema_path.name}")
    print(f"Fields in schema: {len(schema.get('fields', []))}")
    print(f"Sections detected: {len(sections)}")
    print()
    for i, s in enumerate(sections, 1):
        x0, y0, x1, y1 = s.bbox_norm
        w = x1 - x0
        h = y1 - y0
        print(f"[{i:2d}] {s.id:30s}  {s.label}")
        print(f"     bbox=({x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f})  w={w:.3f} h={h:.3f}")
        print(f"     {len(s.field_ids)} fields: {s.field_ids[:6]}"
              + (f" ... +{len(s.field_ids)-6}" if len(s.field_ids) > 6 else ""))
        print(f"     types: {s.field_types}")
        print()

    # Render an overlay if PIL is available
    try:
        from PIL import Image, ImageDraw, ImageFont
        W, H = 1200, 1600
        img = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", 14,
            )
        except Exception:
            font = ImageFont.load_default()
        palette = [
            (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
            (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
            (188, 189, 34), (23, 190, 207),
            (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150),
        ]
        for i, s in enumerate(sections):
            x0, y0, x1, y1 = s.bbox_norm
            color = palette[i % len(palette)]
            dr.rectangle(
                [x0 * W, y0 * H, x1 * W, y1 * H],
                outline=color, width=3,
            )
            dr.text(
                (x0 * W + 4, y0 * H + 4),
                f"{i+1}. {s.label} ({len(s.field_ids)})",
                fill=color, font=font,
            )
        out = PROJECT / "test_outputs" / "sections_preview.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"Wrote overlay: {out.relative_to(PROJECT)}")
    except Exception as e:
        print(f"(overlay skipped: {e})")


if __name__ == "__main__":
    main()

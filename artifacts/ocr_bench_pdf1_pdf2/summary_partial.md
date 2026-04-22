# OCR engine benchmark — CMS-1500 crops

## Per-engine totals

| engine | crops | non_empty | valid | agree_ref | p50 ms |
|---|---|---|---|---|---|
| `florence2_aggressive` | 46 | 17 | 0 | 0 | 596.0 |
| `florence2_raw_upscale` | 46 | 16 | 0 | 0 | 581.9 |
| `got_ocr` | 46 | 46 | 0 | 0 | 1125.2 |
| `vlm` | 46 | 46 | 11 | 11 | 16881.2 |

## Winners by field type

| field_type | florence2_aggressive | florence2_raw_upscale | got_ocr | vlm |
|---|---|---|---|---|
| state | 0 | 0 | 0 | 3 |
| text | 0 | 0 | 0 | 5 |
| zip | 0 | 0 | 0 | 3 |

(`agree_ref` = exact-match to the live pipeline's post-rescue value; use as a proxy for ground truth when no labels are available.)

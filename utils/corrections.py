"""Simple JSONL logger for OCR/validator corrections and threshold tuning."""
from __future__ import annotations
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

CORRECTIONS_PATH = Path("data/corrections.jsonl")
CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
THRESHOLDS_PATH = Path("data/thresholds.json")


def log_correction(entry: Dict[str, Any]) -> None:
    """Append a correction or auto-fix event to the JSONL log."""
    try:
        with CORRECTIONS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Fail silently; logging is best-effort
        pass


def _load_corrections() -> List[Dict[str, Any]]:
    if not CORRECTIONS_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with CORRECTIONS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line.strip()))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def load_threshold_overrides(defaults: Dict[str, float]) -> Dict[str, float]:
    """Return persisted threshold overrides merged over defaults."""
    merged = dict(defaults)
    if THRESHOLDS_PATH.exists():
        try:
            with THRESHOLDS_PATH.open("r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    merged.update({k: float(v) for k, v in saved.items() if isinstance(v, (int, float))})
        except Exception:
            pass
    return merged


def auto_tune_thresholds(defaults: Dict[str, float]) -> Dict[str, float]:
    """
    Compute lightweight threshold suggestions from logged corrections.
    Heuristic: use the median of accepted new_confidence values to adjust the
    low-confidence cutoff, then persist to thresholds.json.
    """
    tuned = dict(defaults)
    corrected = _load_corrections()
    if not corrected:
        return load_threshold_overrides(tuned)

    confs = [row.get("new_confidence") for row in corrected if isinstance(row.get("new_confidence"), (int, float))]
    if confs:
        median_conf = statistics.median(confs)
        tuned_low = max(0.4, min(0.9, median_conf * 0.9))
        tuned["low_confidence_threshold"] = tuned_low

    try:
        with THRESHOLDS_PATH.open("w", encoding="utf-8") as f:
            json.dump(tuned, f, indent=2)
    except Exception:
        pass

    return tuned


def log_user_edit(
    block_id: str,
    old_text: str,
    new_text: str,
    old_confidence: float = 0.0,
    new_confidence: float = 0.0
) -> None:
    """
    Convenience helper to record a user edit. Call from any UI save handler.
    """
    log_correction(
        {
            "block_id": block_id,
            "source": "user_edit",
            "old_text": old_text,
            "new_text": new_text,
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
        }
    )

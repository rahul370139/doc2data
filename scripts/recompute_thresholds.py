"""
Offline helper to recompute tuned thresholds from data/corrections.jsonl.
Run: python scripts/recompute_thresholds.py
"""
from utils.corrections import auto_tune_thresholds

DEFAULTS = {"low_confidence_threshold": 0.72}


def main():
    tuned = auto_tune_thresholds(DEFAULTS)
    print("Updated thresholds:", tuned)


if __name__ == "__main__":
    main()

"""
Recompute tuned thresholds from user corrections.

PURPOSE: Reads data/corrections.jsonl (user corrections from Streamlit),
calls auto_tune_thresholds(), prints updated thresholds for data/thresholds.json.

USE CASE: Run after users correct extractions to improve pipeline over time.
"""
from utils.corrections import auto_tune_thresholds

DEFAULTS = {"low_confidence_threshold": 0.72}


def main():
    tuned = auto_tune_thresholds(DEFAULTS)
    print("Updated thresholds:", tuned)


if __name__ == "__main__":
    main()

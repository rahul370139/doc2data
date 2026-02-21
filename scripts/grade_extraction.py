import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from collections import defaultdict
import difflib

def normalize_text(text: Any) -> str:
    """Normalize text for comparison (lower, strip, remove punctuation)."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Keep only alphanumeric
    return "".join(c for c in text if c.isalnum() or c.isspace())

def compute_f1(pred: str, truth: str) -> float:
    """Compute word-level F1 score."""
    pred_toks = normalize_text(pred).split()
    truth_toks = normalize_text(truth).split()
    
    if not pred_toks and not truth_toks:
        return 1.0
    if not pred_toks or not truth_toks:
        return 0.0
        
    common = 0
    pred_copy = list(pred_toks)
    for token in truth_toks:
        if token in pred_copy:
            common += 1
            pred_copy.remove(token)
            
    precision = common / len(pred_toks)
    recall = common / len(truth_toks)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def grade_submission(pred_dir: Path, gold_dir: Path, keys_to_grade: List[str] = None):
    """
    Grade predictions against gold standard.
    
    Args:
        pred_dir: Directory containing prediction JSONs.
        gold_dir: Directory containing ground truth JSONs.
        keys_to_grade: Optional list of specific keys to grade.
    """
    scores = defaultdict(list)
    field_scores = defaultdict(list)
    
    # Map filenames (ignoring extension case)
    gold_files = {f.stem: f for f in gold_dir.glob("*.json")}
    pred_files = {f.stem: f for f in pred_dir.glob("*.json")}
    
    common_stems = set(gold_files.keys()) & set(pred_files.keys())
    
    print(f"Found {len(common_stems)} matching files.")
    
    for stem in common_stems:
        with open(gold_files[stem]) as f:
            gold = json.load(f)
        with open(pred_files[stem]) as f:
            pred = json.load(f)
            
        # Handle wrapped structure if necessary (e.g. if pred is inside "business_fields")
        if "business_fields" in pred:
            pred = pred["business_fields"]
        if "business_fields" in gold:
            gold = gold["business_fields"]
            
        # Determine keys to grade
        keys = keys_to_grade if keys_to_grade else list(gold.keys())
        
        file_f1s = []
        
        for key in keys:
            if key not in gold:
                continue
                
            truth_val = gold[key]
            pred_val = pred.get(key, "")
            
            f1 = compute_f1(pred_val, truth_val)
            exact = 1.0 if normalize_text(pred_val) == normalize_text(truth_val) else 0.0
            
            field_scores[key].append({
                "f1": f1,
                "exact": exact,
                "file": stem
            })
            file_f1s.append(f1)
            
        avg_file_f1 = np.mean(file_f1s) if file_f1s else 0.0
        scores["file_avg_f1"].append(avg_file_f1)
        print(f"File: {stem} | Avg F1: {avg_file_f1:.2f}")

    # Aggregate results
    print("\n" + "="*50)
    print("AGGREGATE RESULTS")
    print("="*50)
    print(f"Overall Average F1: {np.mean(scores['file_avg_f1']):.4f}")
    
    print("\nPer-Field Performance (Bottom 10):")
    field_avgs = []
    for key, vals in field_scores.items():
        avg_f1 = np.mean([v["f1"] for v in vals])
        field_avgs.append((key, avg_f1))
        
    field_avgs.sort(key=lambda x: x[1])
    
    for key, score in field_avgs[:10]:
        print(f"{key:30s}: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade extraction results against gold standard.")
    parser.add_argument("--pred", required=True, help="Directory containing prediction JSONs")
    parser.add_argument("--gold", required=True, help="Directory containing gold label JSONs")
    args = parser.parse_args()
    
    grade_submission(Path(args.pred), Path(args.gold))


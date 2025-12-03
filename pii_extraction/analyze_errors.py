#!/usr/bin/env python3
"""
Analyze PII extraction errors (false positives and false negatives) across datasets.

This script parses the metrics output files and extracts detailed error information
to help understand where the spaCy + regex pipeline struggles.

Usage:
    python pii_extraction/analyze_errors.py
"""

import pandas as pd
import os
from ast import literal_eval
from collections import Counter

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PII_METRICS_DIR = os.path.join(SCRIPT_DIR, "pii_metrics")
ERROR_ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "error_analysis")

# Datasets to analyze
DATASETS = [
    ("dataset_final_metrics.csv", "dataset_final"),
    ("dataset_balanced_metrics.csv", "dataset_balanced"),
    ("dataset_1500_bank_balanced_metrics.csv", "dataset_1500_bank_balanced"),
]


def parse_label_list(s: str) -> set:
    """Parse ground truth string like '[NAME, PHONE]' into a set."""
    if not isinstance(s, str):
        return set()
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s.strip():
        return set()
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return {p for p in parts if p}


def parse_predicted_labels(s: str) -> set:
    """Parse predicted labels string like "['EMAIL', 'PHONE']" into a set."""
    if not isinstance(s, str) or not s.strip():
        return set()
    try:
        return set(literal_eval(s))
    except:
        return set()


def analyze_dataset_errors(input_path: str, dataset_name: str) -> dict:
    """
    Analyze errors for a single dataset.
    Returns dict with error statistics and examples.
    """
    if not os.path.exists(input_path):
        print(f"  WARNING: File not found: {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    
    # Collect errors
    false_negatives = []  # Labels in ground truth but not predicted
    false_positives = []  # Labels predicted but not in ground truth
    fn_counter = Counter()
    fp_counter = Counter()
    
    for idx, row in df.iterrows():
        gt = parse_label_list(row["ground_truth"])
        pred = parse_predicted_labels(row["predicted_labels"])
        
        # False negatives (missed)
        fn = gt - pred
        for label in fn:
            fn_counter[label] += 1
            if len(false_negatives) < 50:  # Keep first 50 examples
                false_negatives.append({
                    "row": idx,
                    "missed_label": label,
                    "ground_truth": sorted(gt),
                    "predicted": sorted(pred),
                    "conversation": row["conversation"][:200] + "..." if len(row["conversation"]) > 200 else row["conversation"]
                })
        
        # False positives (extra)
        fp = pred - gt
        for label in fp:
            fp_counter[label] += 1
            if len(false_positives) < 50:  # Keep first 50 examples
                false_positives.append({
                    "row": idx,
                    "extra_label": label,
                    "ground_truth": sorted(gt),
                    "predicted": sorted(pred),
                    "conversation": row["conversation"][:200] + "..." if len(row["conversation"]) > 200 else row["conversation"]
                })
    
    return {
        "dataset": dataset_name,
        "total_rows": len(df),
        "rows_with_fn": len(df[df["fn"] > 0]),
        "rows_with_fp": len(df[df["fp"] > 0]),
        "fn_by_label": dict(fn_counter.most_common()),
        "fp_by_label": dict(fp_counter.most_common()),
        "fn_examples": false_negatives[:10],  # Top 10 examples
        "fp_examples": false_positives[:10],
    }


def save_error_summary(all_errors: list):
    """Save a consolidated error summary CSV."""
    rows = []
    for err in all_errors:
        if err is None:
            continue
        
        # Add FN breakdown
        for label, count in err["fn_by_label"].items():
            rows.append({
                "dataset": err["dataset"],
                "error_type": "False Negative (missed)",
                "label": label,
                "count": count,
            })
        
        # Add FP breakdown
        for label, count in err["fp_by_label"].items():
            rows.append({
                "dataset": err["dataset"],
                "error_type": "False Positive (extra)",
                "label": label,
                "count": count,
            })
    
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_path = os.path.join(ERROR_ANALYSIS_DIR, "error_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved error summary to: {summary_path}")
        return summary_df
    return None


def save_error_examples(all_errors: list):
    """Save detailed error examples for each dataset."""
    for err in all_errors:
        if err is None:
            continue
        
        dataset_name = err["dataset"]
        
        # Save FN examples
        if err["fn_examples"]:
            fn_df = pd.DataFrame(err["fn_examples"])
            fn_path = os.path.join(ERROR_ANALYSIS_DIR, f"{dataset_name}_fn_examples.csv")
            fn_df.to_csv(fn_path, index=False)
        
        # Save FP examples
        if err["fp_examples"]:
            fp_df = pd.DataFrame(err["fp_examples"])
            fp_path = os.path.join(ERROR_ANALYSIS_DIR, f"{dataset_name}_fp_examples.csv")
            fp_df.to_csv(fp_path, index=False)


def main():
    os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("PII EXTRACTION ERROR ANALYSIS")
    print("=" * 60)
    
    all_errors = []
    
    for metrics_filename, dataset_name in DATASETS:
        input_path = os.path.join(PII_METRICS_DIR, metrics_filename)
        print(f"\n--- Analyzing: {dataset_name} ---")
        
        errors = analyze_dataset_errors(input_path, dataset_name)
        if errors:
            all_errors.append(errors)
            
            print(f"  Total rows: {errors['total_rows']}")
            print(f"  Rows with false negatives: {errors['rows_with_fn']}")
            print(f"  Rows with false positives: {errors['rows_with_fp']}")
            
            if errors["fn_by_label"]:
                print(f"\n  False Negatives (missed labels):")
                for label, count in sorted(errors["fn_by_label"].items(), key=lambda x: -x[1]):
                    print(f"    {label}: {count}")
            
            if errors["fp_by_label"]:
                print(f"\n  False Positives (extra labels):")
                for label, count in sorted(errors["fp_by_label"].items(), key=lambda x: -x[1]):
                    print(f"    {label}: {count}")
    
    # Save consolidated results
    summary_df = save_error_summary(all_errors)
    save_error_examples(all_errors)
    
    # Print cross-dataset comparison
    if summary_df is not None and len(summary_df) > 0:
        print("\n" + "=" * 60)
        print("CROSS-DATASET ERROR COMPARISON")
        print("=" * 60)
        
        # Pivot for easier reading
        pivot = summary_df.pivot_table(
            index=["error_type", "label"], 
            columns="dataset", 
            values="count", 
            fill_value=0
        )
        print("\n" + pivot.to_string())
    
    print("\n" + "=" * 60)
    print(f"Error analysis complete! Output directory: {ERROR_ANALYSIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

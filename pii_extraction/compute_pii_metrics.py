import pandas as pd
import json
import os

# Use paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input/Output directories
EXTRACTED_PII_DIR = os.path.join(SCRIPT_DIR, "extracted_pii")
PII_METRICS_DIR = os.path.join(SCRIPT_DIR, "pii_metrics")

# Datasets to process: (extracted_filename, metrics_output_name)
# These correspond to the outputs from spacy_regex.py
DATASETS = [
    ("dataset_final_extracted.csv", "dataset_final_metrics"),
    ("dataset_balanced_extracted.csv", "dataset_balanced_metrics"),
    ("dataset_1500_bank_balanced_extracted.csv", "dataset_1500_bank_balanced_metrics"),
]

# Map raw spaCy/regex labels to canonical ground-truth labels.
# Ground truth labels (from dataset): NAME, PHONE, EMAIL, DATE/DOB, company, location, SSN, CREDIT_CARD, IP, age, sex
# Predicted labels (from spacy_regex.py): PERSON, EMAIL, PHONE, IP_ADDRESS, SSN, CREDIT_CARD_16, CREDIT_CARD_4,
#                                          ORG, FAC, GPE, LOC, DATE, AGE, SEX, COMPANY_DOMAIN
LABEL_MAP = {
    # spaCy NER labels
    "PERSON": "NAME",
    "ORG": "company",
    "FAC": "company",      # FAC (Facility) is normalized to ORG in spacy_regex.py, but kept here for safety
    "GPE": "location",     # Geo-Political Entity (cities, countries)
    "LOC": "location",     # Non-GPE locations (rivers, mountains, etc.)
    # Regex-detected labels
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "IP_ADDRESS": "IP",
    "SSN": "SSN",
    "CREDIT_CARD_16": "CREDIT_CARD",
    "CREDIT_CARD_4": "CREDIT_CARD",
    "DATE": "DATE/DOB",
    "AGE": "age",
    "SEX": "sex",
    "COMPANY_DOMAIN": "company",  # Domain names mentioned as "my company xyz.com"
}


def parse_label_list(s: str) -> set:
    """
    Parse a string like "[NAME, PHONE, EMAIL]" into a set {"NAME","PHONE","EMAIL"}.
    Handles various formats in ground truth column.
    """
    if not isinstance(s, str):
        return set()
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s.strip():
        return set()
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return {p for p in parts if p}


def extract_pred_labels(pii_str: str) -> set:
    """
    Parse the pii_entities JSON string and map labels to canonical labels.
    Returns a set of label strings (e.g. {"NAME","EMAIL","PHONE"}).
    
    The pii_entities column contains JSON arrays like:
    [{"text": "...", "label": "PERSON", "start": 0, "end": 10, "source": "spacy"}, ...]
    """
    if not isinstance(pii_str, str) or not pii_str.strip():
        return set()

    # Handle CSV escaping: doubled quotes become single quotes
    clean = pii_str.replace('""', '"')
    
    # Handle edge case where string starts/ends with extra quotes from CSV
    if clean.startswith('"') and clean.endswith('"'):
        clean = clean[1:-1]
    
    try:
        entities = json.loads(clean)
    except json.JSONDecodeError:
        # Try one more fix: sometimes the entire string is double-escaped
        try:
            entities = json.loads(json.loads(clean))
        except Exception:
            return set()
    except Exception:
        return set()

    if not isinstance(entities, list):
        return set()

    labels = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        raw_label = ent.get("label")
        mapped = LABEL_MAP.get(raw_label)
        if mapped:
            labels.add(mapped)
        # Handle unmapped labels - they won't contribute to metrics
        # but we could log them for debugging

    return labels


def compute_row_metrics(y_true: set, y_pred: set):
    """
    Compute TP, FP, FN, precision, recall, f1, and subset accuracy for one row.
    
    - TP (True Positives): Labels correctly predicted (in both y_true and y_pred)
    - FP (False Positives): Labels predicted but not in ground truth (over-detection)
    - FN (False Negatives): Labels in ground truth but not predicted (under-detection)
    """
    tp = len(y_true & y_pred)
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if y_true == y_pred else 0.0

    return tp, fp, fn, precision, recall, f1, exact_match


def compute_metrics_for_dataset(input_csv: str, output_csv: str, dataset_name: str) -> dict:
    """
    Compute PII extraction metrics for a single dataset.
    Returns a summary dict of the metrics.
    """
    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"  WARNING: Input file not found: {input_csv}")
        return None
    
    df = pd.read_csv(input_csv)
    
    # Validate required columns
    required_columns = ["ground_truth", "pii_entities"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"  WARNING: Missing required columns: {missing_columns}")
        return None

    # Containers for global (micro) stats
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Containers for macro-averaged stats
    row_precisions = []
    row_recalls = []
    row_f1s = []

    # Columns to fill
    pred_labels_col = []
    tp_col = []
    fp_col = []
    fn_col = []
    prec_col = []
    rec_col = []
    f1_col = []
    exact_match_col = []

    for _, row in df.iterrows():
        y_true = parse_label_list(row["ground_truth"])
        y_pred = extract_pred_labels(row["pii_entities"])

        tp, fp, fn, precision, recall, f1, exact_match = compute_row_metrics(y_true, y_pred)

        pred_labels_col.append(sorted(list(y_pred)))
        tp_col.append(tp)
        fp_col.append(fp)
        fn_col.append(fn)
        prec_col.append(precision)
        rec_col.append(recall)
        f1_col.append(f1)
        exact_match_col.append(exact_match)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # For macro-average (only count rows with ground truth)
        if len(y_true) > 0:
            row_precisions.append(precision)
            row_recalls.append(recall)
            row_f1s.append(f1)

    # Add columns to dataframe
    df["predicted_labels"] = pred_labels_col
    df["tp"] = tp_col
    df["fp"] = fp_col
    df["fn"] = fn_col
    df["precision"] = prec_col
    df["recall"] = rec_col
    df["f1"] = f1_col
    df["exact_match"] = exact_match_col

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"  Saved metrics to: {output_csv}")

    # Calculate metrics
    # Micro-averaged: aggregate all TP/FP/FN across dataset
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    
    # Macro-averaged: average per-row metrics
    macro_precision = sum(row_precisions) / len(row_precisions) if row_precisions else 0.0
    macro_recall = sum(row_recalls) / len(row_recalls) if row_recalls else 0.0
    macro_f1 = sum(row_f1s) / len(row_f1s) if row_f1s else 0.0
    
    # Exact match accuracy
    exact_match_accuracy = sum(exact_match_col) / len(exact_match_col) if exact_match_col else 0.0

    return {
        "dataset": dataset_name,
        "rows": len(df),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "exact_match_accuracy": exact_match_accuracy,
        "exact_match_count": int(sum(exact_match_col)),
    }


def main():
    """
    Process all extracted PII datasets and compute metrics.
    """
    # Create output directory if it doesn't exist
    os.makedirs(PII_METRICS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("PII METRICS COMPUTATION PIPELINE")
    print("=" * 60)
    
    all_summaries = []
    
    for extracted_filename, metrics_name in DATASETS:
        input_path = os.path.join(EXTRACTED_PII_DIR, extracted_filename)
        output_path = os.path.join(PII_METRICS_DIR, f"{metrics_name}.csv")
        
        print(f"\n--- Processing: {extracted_filename} ---")
        
        summary = compute_metrics_for_dataset(input_path, output_path, metrics_name)
        
        if summary:
            all_summaries.append(summary)
            print(f"  Rows: {summary['rows']}")
            print(f"  TP: {summary['total_tp']}, FP: {summary['total_fp']}, FN: {summary['total_fn']}")
            print(f"  Micro F1: {summary['micro_f1']:.4f}")
            print(f"  Macro F1: {summary['macro_f1']:.4f}")
            print(f"  Exact Match: {summary['exact_match_accuracy']:.4f}")
    
    # Save summary of all datasets
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(PII_METRICS_DIR, "all_datasets_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print("\n" + "=" * 60)
        print("SUMMARY ACROSS ALL DATASETS")
        print("=" * 60)
        print(f"\nSummary saved to: {summary_path}\n")
        
        for summary in all_summaries:
            print(f"\n{summary['dataset']}:")
            print(f"  Dataset size: {summary['rows']} rows")
            print(f"  Total TP: {summary['total_tp']}, FP: {summary['total_fp']}, FN: {summary['total_fn']}")
            print(f"  Micro-averaged: P={summary['micro_precision']:.4f}, R={summary['micro_recall']:.4f}, F1={summary['micro_f1']:.4f}")
            print(f"  Macro-averaged: P={summary['macro_precision']:.4f}, R={summary['macro_recall']:.4f}, F1={summary['macro_f1']:.4f}")
            print(f"  Exact Match: {summary['exact_match_accuracy']:.4f} ({summary['exact_match_count']}/{summary['rows']})")
    
    print("\n" + "=" * 60)
    print("PII metrics computation complete!")
    print(f"Output directory: {PII_METRICS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

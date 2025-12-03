"""
Analyze PII type probabilities/frequencies in the dataset for each domain.

Shows how often each PII type appears in allowed_restaurant vs allowed_bank.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from collections import Counter
from common.config import PII_TYPES


def parse_list_str(s: str) -> list:
    """Parse a list-like string into a list of strings."""
    if not isinstance(s, str) or s.strip() == '' or s.strip().lower() == 'nan':
        return []
    s = s.strip('[]').strip('"').strip()
    if not s:
        return []
    return [x.strip() for x in s.split(',') if x.strip()]


def analyze_dataset_probabilities(dataset_path: str):
    """
    Analyze PII type probabilities for each domain.
    
    Returns:
        Dictionary with probabilities for restaurant and bank domains
    """
    df = pd.read_csv(dataset_path)
    
    rest_allowed = []
    bank_allowed = []
    total_rest = 0
    total_bank = 0
    
    for _, row in df.iterrows():
        rest_str = str(row.get('allowed_restaurant', '')).strip('[]').strip('"')
        bank_str = str(row.get('allowed_bank', '')).strip('[]').strip('"')
        
        if rest_str and rest_str != 'nan':
            rest_list = parse_list_str(rest_str)
            rest_allowed.extend(rest_list)
            total_rest += 1
        
        if bank_str and bank_str != 'nan':
            bank_list = parse_list_str(bank_str)
            bank_allowed.extend(bank_list)
            total_bank += 1
    
    rest_counter = Counter(rest_allowed)
    bank_counter = Counter(bank_allowed)
    
    # Calculate probabilities (frequency / total rows)
    rest_probs = {pii: rest_counter.get(pii, 0) / total_rest if total_rest > 0 else 0 
                  for pii in PII_TYPES}
    bank_probs = {pii: bank_counter.get(pii, 0) / total_bank if total_bank > 0 else 0 
                  for pii in PII_TYPES}
    
    # Also get raw counts
    rest_counts = {pii: rest_counter.get(pii, 0) for pii in PII_TYPES}
    bank_counts = {pii: bank_counter.get(pii, 0) for pii in PII_TYPES}
    
    return {
        'restaurant': {
            'probabilities': rest_probs,
            'counts': rest_counts,
            'total_rows': total_rest
        },
        'bank': {
            'probabilities': bank_probs,
            'counts': bank_counts,
            'total_rows': total_bank
        }
    }


def print_probabilities(results: dict):
    """Print probabilities in a readable format."""
    
    print("\n" + "="*80)
    print("PII TYPE PROBABILITIES IN DATASET (by Domain)")
    print("="*80)
    
    print(f"\n{'PII Type':<20} {'Restaurant':<25} {'Bank':<25} {'Difference'}")
    print("-" * 80)
    print(f"{'':<20} {'Count':<12} {'Prob':<12} {'Count':<12} {'Prob':<12} {'(Bank-Rest)'}")
    print("-" * 80)
    
    for pii in PII_TYPES:
        rest_count = results['restaurant']['counts'][pii]
        rest_prob = results['restaurant']['probabilities'][pii]
        bank_count = results['bank']['counts'][pii]
        bank_prob = results['bank']['probabilities'][pii]
        diff = bank_prob - rest_prob
        
        rest_str = f"{rest_count:<12} {rest_prob:.3f}"
        bank_str = f"{bank_count:<12} {bank_prob:.3f}"
        diff_str = f"{diff:+.3f}"
        
        print(f"{pii:<20} {rest_str:<25} {bank_str:<25} {diff_str}")
    
    print("-" * 80)
    print(f"{'Total Rows':<20} {results['restaurant']['total_rows']:<25} {results['bank']['total_rows']:<25}")
    print("="*80)
    
    # Summary
    print("\n Summary:")
    print("-" * 80)
    
    print("\n Restaurant Domain - Most Common PII:")
    rest_sorted = sorted(results['restaurant']['probabilities'].items(), 
                        key=lambda x: x[1], reverse=True)
    for pii, prob in rest_sorted[:5]:
        if prob > 0:
            print(f"   {pii:<15} {prob:.3f} ({results['restaurant']['counts'][pii]} times)")
    
    print("\n Bank Domain - Most Common PII:")
    bank_sorted = sorted(results['bank']['probabilities'].items(), 
                        key=lambda x: x[1], reverse=True)
    for pii, prob in bank_sorted[:5]:
        if prob > 0:
            print(f"   {pii:<15} {prob:.3f} ({results['bank']['counts'][pii]} times)")
    
    print("\nDomain Differences (Bank - Restaurant):")
    print("-" * 80)
    differences = []
    for pii in PII_TYPES:
        diff = results['bank']['probabilities'][pii] - results['restaurant']['probabilities'][pii]
        if abs(diff) > 0.01:  # Only show significant differences
            differences.append((pii, diff))
    
    differences.sort(key=lambda x: x[1], reverse=True)
    for pii, diff in differences:
        print(f"  {pii:<15} {diff:+.3f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze PII probabilities in dataset')
    parser.add_argument('--dataset', type=str, default='690-Project-Dataset.csv',
                       help='Path to dataset CSV file')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = Path("final_project") / args.dataset
        if not dataset_path.exists():
            print(f" Dataset not found: {args.dataset}")
            return
    
    results = analyze_dataset_probabilities(str(dataset_path))
    print_probabilities(results)


if __name__ == "__main__":
    main()


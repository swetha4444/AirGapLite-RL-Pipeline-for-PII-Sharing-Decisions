"""
Analyze learned patterns with directive system showing utility-privacy tradeoff.

This script shows:
1. What patterns the model learns for each domain (regex)
2. How different directives (strictly, accurately, balanced) affect utility-privacy tradeoff
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from collections import Counter

from pipeline.algorithm_registry import AlgorithmRegistry
from common.config import PII_TYPES, NUM_PII, SCENARIO_NAME2ID, SCENARIOS
from common.mdp import build_state


def compute_utility_privacy(
    present_mask: List[int],
    allowed_mask: List[int],
    action_mask: List[int]
) -> Tuple[float, float, float]:
    """
    Compute utility, privacy, and privacy breach metrics.
    
    Returns:
        utility: Fraction of allowed PII that was shared
        privacy: Fraction of disallowed PII that was NOT shared
        privacy_breach: Fraction of disallowed PII that WAS shared
    """
    present = [i for i, m in enumerate(present_mask) if m == 1]
    if not present:
        return 0.0, 1.0, 0.0
    
    allowed = [i for i in present if allowed_mask[i] == 1]
    disallowed = [i for i in present if allowed_mask[i] == 0]
    shared = [i for i in present if action_mask[i] == 1]
    
    # Utility: how much allowed PII was shared
    if allowed:
        shared_allowed = [i for i in shared if i in allowed]
        utility = len(shared_allowed) / len(allowed)
    else:
        utility = 1.0 if len(shared) == 0 else 0.0
    
    # Privacy: how much disallowed PII was NOT shared
    if disallowed:
        shared_disallowed = [i for i in shared if i in disallowed]
        privacy = 1.0 - (len(shared_disallowed) / len(disallowed))
        privacy_breach = len(shared_disallowed) / len(disallowed)
    else:
        privacy = 1.0
        privacy_breach = 0.0
    
    return utility, privacy, privacy_breach


def analyze_with_directives(
    algorithm: str,
    model_path: str,
    dataset_path: str,
    num_samples: int = 200
) -> Dict:
    """
    Analyze learned patterns and utility-privacy tradeoff for different directives.
    """
    # Load model
    algo_config = AlgorithmRegistry.get(algorithm.lower())
    policy = AlgorithmRegistry.create_policy(algorithm.lower())
    
    try:
        policy.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f" Loaded {algorithm.upper()} model from: {model_path}")
    except Exception as e:
        print(f" Error loading model: {e}")
        return None
    
    policy.eval()
    
    # Load dataset
    if algorithm.lower() == 'grpo':
        from algorithms.grpo.grpo_train import load_dataset_from_excel
    elif algorithm.lower() == 'groupedppo':
        from algorithms.groupedppo.grpo_train import load_dataset_from_excel
    else:
        from algorithms.vanillarl.train import load_dataset_from_excel
    
    dataset_rows = load_dataset_from_excel(dataset_path)
    
    # Analyze patterns and tradeoffs
    results = {}
    
    for domain in ["restaurant", "bank"]:
        results[domain] = {
            "patterns": {},
            "tradeoffs": []
        }
        
        # Get learned probabilities (all PII present)
        scenario_id = SCENARIO_NAME2ID[domain]
        present_mask_all = [1] * NUM_PII
        state_all = build_state(present_mask_all, scenario_id)
        
        if algorithm.lower() == 'vanillarl':
            logits = policy(state_all)
        else:
            logits, _ = policy(state_all)
        
        probs = torch.sigmoid(logits)[0].cpu().tolist()
        results[domain]["patterns"]["probabilities"] = probs
        
        # Test different directives
        directives = ["strictly", "balanced", "accurately"]
        
        for directive in directives:
            # Evaluate on dataset samples
            utilities = []
            privacies = []
            privacy_breaches = []
            shared_counts = []
            
            for _ in range(num_samples):
                row = np.random.choice(dataset_rows)
                present_mask = row.present_mask
                allowed_mask = (
                    row.allowed_mask_restaurant if domain == "restaurant" 
                    else row.allowed_mask_bank
                )
                
                # Get actions with directive
                state = build_state(present_mask, scenario_id)
                actions = policy.act(state, deterministic=True, threshold=0.5, directive=directive)
                
                # Compute metrics
                utility, privacy, privacy_breach = compute_utility_privacy(
                    present_mask, allowed_mask, actions
                )
                
                utilities.append(utility)
                privacies.append(privacy)
                privacy_breaches.append(privacy_breach)
                shared_counts.append(sum(actions))
            
            results[domain]["tradeoffs"].append({
                "directive": directive,
                "utility": np.mean(utilities),
                "privacy": np.mean(privacies),
                "privacy_breach": np.mean(privacy_breaches),
                "avg_shared": np.mean(shared_counts),
                "utility_std": np.std(utilities),
                "privacy_std": np.std(privacies),
            })
        
        # Get actions for each directive (all PII present)
        for directive in directives:
            actions = policy.act(state_all, deterministic=True, threshold=0.5, directive=directive)
            shared_pii = [PII_TYPES[i] for i, a in enumerate(actions) if a == 1]
            results[domain]["patterns"][directive] = shared_pii
    
    return results


def print_analysis(results: Dict):
    """Print comprehensive analysis."""
    
    print("\n" + "="*80)
    print("LEARNED PATTERNS & UTILITY-PRIVACY TRADEOFF")
    print("="*80)
    
    for domain in ["restaurant", "bank"]:
        print(f"\n{'='*80}")
        print(f" {domain.upper()} Domain")
        print("="*80)
        
        # Show learned probabilities
        probs = results[domain]["patterns"]["probabilities"]
        pii_with_probs = list(zip(PII_TYPES, probs))
        pii_with_probs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n Learned Probabilities (all PII present):")
        print("-" * 80)
        print(f"{'PII Type':<20} {'Probability':<15} {'Interpretation'}")
        print("-" * 80)
        for pii_type, prob in pii_with_probs:
            if prob >= 0.7:
                interp = "🔴 High (likely share)"
            elif prob >= 0.5:
                interp = "🟡 Medium (may share)"
            elif prob >= 0.3:
                interp = "🟠 Low (unlikely share)"
            else:
                interp = "⚫ Very Low (rarely share)"
            print(f"{pii_type:<20} {prob:.3f}           {interp}")
        
        # Show patterns for each directive
        print(f"\n Learned Patterns (Regex) by Directive:")
        print("-" * 80)
        for directive in ["strictly", "balanced", "accurately"]:
            shared = results[domain]["patterns"][directive]
            pattern = " | ".join(shared) if shared else "(none)"
            print(f"  {directive.upper():<12}: {pattern}")
        
        # Show utility-privacy tradeoff
        print(f"\n⚖️  Utility-Privacy Tradeoff:")
        print("-" * 80)
        print(f"{'Directive':<12} {'Utility':<12} {'Privacy':<12} {'Breach':<12} {'Avg Shared'}")
        print("-" * 80)
        
        for tradeoff in results[domain]["tradeoffs"]:
            d = tradeoff["directive"]
            u = tradeoff["utility"]
            p = tradeoff["privacy"]
            b = tradeoff["privacy_breach"]
            s = tradeoff["avg_shared"]
            
            print(f"{d.upper():<12} {u:.3f}        {p:.3f}        {b:.3f}        {s:.1f}")
        
        # Interpretation
        print(f"\n Interpretation:")
        strictly = next(t for t in results[domain]["tradeoffs"] if t["directive"] == "strictly")
        balanced = next(t for t in results[domain]["tradeoffs"] if t["directive"] == "balanced")
        accurately = next(t for t in results[domain]["tradeoffs"] if t["directive"] == "accurately")
        
        print(f"  • STRICTLY: High privacy ({strictly['privacy']:.3f}), Low utility ({strictly['utility']:.3f})")
        print(f"  • BALANCED: Moderate privacy ({balanced['privacy']:.3f}), Moderate utility ({balanced['utility']:.3f})")
        print(f"  • ACCURATELY: Low privacy ({accurately['privacy']:.3f}), High utility ({accurately['utility']:.3f})")
    
    print("\n" + "="*80)
    print("Summary: Directives control the utility-privacy tradeoff")
    print("="*80)


def create_tradeoff_table(results: Dict):
    """Create a comparison table."""
    
    print("\n" + "="*80)
    print("DIRECTIVE COMPARISON TABLE")
    print("="*80)
    
    print(f"\n{'Domain':<15} {'Directive':<12} {'Utility':<10} {'Privacy':<10} {'Breach':<10} {'Shared'}")
    print("-" * 80)
    
    for domain in ["restaurant", "bank"]:
        for tradeoff in results[domain]["tradeoffs"]:
            d = tradeoff["directive"]
            u = tradeoff["utility"]
            p = tradeoff["privacy"]
            b = tradeoff["privacy_breach"]
            s = tradeoff["avg_shared"]
            
            print(f"{domain.capitalize():<15} {d.upper():<12} {u:.3f}      {p:.3f}      {b:.3f}      {s:.1f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze learned patterns with directives')
    parser.add_argument('--algorithm', type=str, default='grpo',
                       choices=AlgorithmRegistry.list_algorithms(),
                       help='Algorithm to analyze')
    parser.add_argument('--model', type=str, default='models/grpo_model.pt',
                       help='Path to trained model file')
    parser.add_argument('--dataset', type=str, default='690-Project-Dataset.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--samples', type=int, default=200,
                       help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    # Check paths
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = Path("final_project") / args.model
        if not model_path.exists():
            print(f" Model not found: {args.model}")
            return
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = Path("final_project") / args.dataset
        if not dataset_path.exists():
            print(f" Dataset not found: {args.dataset}")
            return
    
    # Analyze
    results = analyze_with_directives(
        args.algorithm, 
        str(model_path), 
        str(dataset_path),
        num_samples=args.samples
    )
    
    if results:
        print_analysis(results)
        create_tradeoff_table(results)
    else:
        print("\n To train a model, run:")
        print(f"   python pipeline/train.py --algorithm {args.algorithm}")


if __name__ == "__main__":
    main()


"""
Unified Testing Pipeline - Evaluate any trained algorithm.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from typing import Dict, Any
import json
import random

from pipeline.algorithm_registry import AlgorithmRegistry
from common.config import NUM_PII, SCENARIO_NAME2ID, PII_TYPES
from common.mdp import build_state


class EvaluationPipeline:
    """Unified evaluation pipeline for any algorithm."""
    
    def __init__(self, algorithm: str):
        """Initialize evaluation pipeline."""
        self.algorithm_name = algorithm.lower()
        self.algo_config = AlgorithmRegistry.get(self.algorithm_name)
        
        # Create policy
        self.policy = AlgorithmRegistry.create_policy(self.algorithm_name)
        
        # Get algorithm-specific functions
        self.load_dataset = self.algo_config['config']['load_dataset']
        self.evaluate = self.algo_config['config']['evaluate']
        
        self.dataset_rows = None
    
    def load_model(self, model_path: str):
        """Load trained model."""
        self.policy.load_state_dict(torch.load(model_path))
        self.policy.eval()
        print(f" Loaded {self.algorithm_name.upper()} model from: {model_path}")
    
    def load_data(self, dataset_path: str):
        """Load dataset."""
        self.dataset_rows = self.load_dataset(dataset_path)
        print(f" Loaded {len(self.dataset_rows)} dataset rows")
    
    def evaluate_average_reward(self, num_samples: int = 200) -> float:
        """Evaluate average reward (uses dataset for reward computation)."""
        if self.dataset_rows is None:
            print(" Warning: No dataset loaded. Cannot compute average reward.")
            return 0.0
        return self.evaluate(self.policy, self.dataset_rows, num_samples=num_samples)
    
    def evaluate_utility_privacy(self, domain: str, num_samples: int = 200, directive: str = 'balanced', threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate utility and privacy metrics.
        
        NO DATASET NEEDED - uses fixed expected patterns based on model's derived regex:
        - Restaurant: EMAIL, PHONE
        - Bank: CREDIT_CARD, DATE/DOB, EMAIL, PHONE, SSN
        
        Utility = fraction of expected pattern that model shares
        Privacy = 1 - (extra shared / total disallowed)
        """
        scenario_id = SCENARIO_NAME2ID[domain]
        
        # Fixed expected patterns (NOT from dataset - these are the ground truth patterns)
        expected_patterns = {
            'restaurant': {'EMAIL', 'PHONE'},
            'bank': {'CREDIT_CARD', 'DATE/DOB', 'EMAIL', 'PHONE', 'SSN'}
        }
        expected_pii_set = expected_patterns[domain]
        
        # Get PII type indices for expected pattern
        expected_indices = set([i for i, pii in enumerate(PII_TYPES) if pii in expected_pii_set])
        
        # All PII present (test with all PII types to get model's derived regex)
        present_mask_all = [1] * NUM_PII
        state_all = build_state(present_mask_all, scenario_id)
        
        with torch.no_grad():
            # Get model's actions (based on threshold/directive) - this is the model's derived regex
            if self.algorithm_name == 'grpo':
                logits, _ = self.policy(state_all)
                probs = torch.sigmoid(logits)[0]
                actions = self.policy.act(state_all, deterministic=True, threshold=threshold, directive=directive)
            else:
                # GroupedPPO and VanillaRL: per-PII binary actions (same as GRPO)
                if self.algorithm_name == 'groupedppo':
                    logits, _ = self.policy(state_all)
                else:
                    logits = self.policy(state_all)
                actions = self.policy.act(state_all, deterministic=True, threshold=threshold, directive=directive)
            
            # Model's shared PII indices (the derived regex pattern)
            shared_indices = set([i for i, a in enumerate(actions) if a == 1])
            
            # Utility: how much of expected pattern does model share?
            shared_expected = expected_indices & shared_indices
            if expected_indices:
                utility = len(shared_expected) / len(expected_indices)
            else:
                utility = 1.0 if len(shared_indices) == 0 else 0.0
            
            # Privacy: penalty for sharing extra (not expected) PII
            all_pii_indices = set(range(NUM_PII))
            disallowed_indices = all_pii_indices - expected_indices
            extra_shared = shared_indices - expected_indices
            
            if disallowed_indices:
                privacy = 1.0 - (len(extra_shared) / len(disallowed_indices))
            else:
                privacy = 1.0 if len(extra_shared) == 0 else 0.0
            
            privacy_breach_rate = len(extra_shared) / len(disallowed_indices) if disallowed_indices else 0.0
        
        return {
            'utility': utility,
            'privacy': privacy,
            'avg_shared': len(shared_indices),
            'privacy_breach_rate': privacy_breach_rate,
        }
    
    def run_full_evaluation(self, directive: str = 'balanced', threshold: float = 0.5) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        print(f"\n{'='*70}")
        print(f"EVALUATION: {self.algorithm_name.upper()}")
        print(f"{'='*70}\n")
        
        results = {
            'algorithm': self.algorithm_name,
            'average_reward': self.evaluate_average_reward(),
        }
        
        print(f"Average Reward: {results['average_reward']:.4f}\n")
        
        # Domain-specific evaluation (based on model's derived regex, NOT dataset)
        print(f"Directive: {directive.upper()} (threshold adjusted based on directive)")
        print()
        for domain in ['restaurant', 'bank']:
            print(f"--- {domain.upper()} Domain ---")
            metrics = self.evaluate_utility_privacy(domain, directive=directive, threshold=threshold)
            results[f'{domain}_metrics'] = metrics
            
            print(f"  Directive:      {directive.upper()}")
            print(f"  Utility:        {metrics['utility']:.4f}")
            print(f"  Privacy:        {metrics['privacy']:.4f}")
            print(f"  Privacy Breach: {metrics['privacy_breach_rate']:.4f}")
            print(f"  Avg Shared:     {metrics['avg_shared']:.2f} PII types\n")
        
        # Show summary for all directives
        print(f"\n{'='*70}")
        print("UTILITY-PRIVACY TRADEOFF SUMMARY (All Directives)")
        print(f"{'='*70}\n")
        for domain in ['restaurant', 'bank']:
            print(f"{domain.upper()} Domain:")
            for dir_name in ['strictly', 'balanced', 'accurately']:
                dir_metrics = self.evaluate_utility_privacy(domain, directive=dir_name, threshold=0.5)
                print(f"  {dir_name.upper():<10} Utility={dir_metrics['utility']:.3f}  Privacy={dir_metrics['privacy']:.3f}")
            print()
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Unified Testing Pipeline')
    parser.add_argument('--algorithm', type=str, required=True,
                       choices=AlgorithmRegistry.list_algorithms(),
                       help=f'Algorithm to evaluate: {AlgorithmRegistry.list_algorithms()}')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--dataset', type=str, default='690-Project-Dataset.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for evaluation results')
    parser.add_argument('--directive', type=str, default='balanced',
                       choices=['strictly', 'balanced', 'accurately'],
                       help='Directive for regex pattern extraction: strictly (high privacy), balanced (default), accurately (high utility)')
    parser.add_argument('--get-regex', action='store_true',
                       help='Get learned regex pattern for the specified directive and exit')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(args.algorithm)
    
    # Load model
    pipeline.load_model(args.model)
    
    # If --get-regex flag is set, just show regex patterns and exit
    if args.get_regex:
        from common.config import PII_TYPES, NUM_PII, SCENARIO_NAME2ID
        from common.mdp import build_state
        
        # Expected patterns from dataset
        expected = {
            "restaurant": {"EMAIL", "PHONE"},
            "bank": {"EMAIL", "PHONE", "DATE/DOB", "SSN", "CREDIT_CARD"}
        }
        
        print("\n" + "="*80)
        print(f"LEARNED REGEX PATTERN - Directive: {args.directive.upper()}")
        print("="*80)
        
        pipeline.policy.eval()
        with torch.no_grad():
            for domain in ["restaurant", "bank"]:
                scenario_id = SCENARIO_NAME2ID[domain]
                present_mask = [1] * NUM_PII
                state = build_state(present_mask, scenario_id)
                
                # Get actions with directive
                actions = pipeline.policy.act(state, deterministic=True, threshold=0.5, directive=args.directive)
                
                # Convert to regex pattern
                shared_pii = [PII_TYPES[i] for i, a in enumerate(actions) if a == 1]
                model_set = set(shared_pii)
                expected_set = expected[domain]
                
                if shared_pii:
                    regex_pattern = " | ".join(shared_pii)
                else:
                    regex_pattern = "(none)"
                
                print(f"\n {domain.upper()} Domain:")
                print(f"   Regex: {regex_pattern}")
                print(f"   Shared PII: {', '.join(shared_pii) if shared_pii else '(none)'}")
                print(f"   Count: {len(shared_pii)} PII types")
                print(f"   Expected (from dataset): {', '.join(sorted(expected_set))}")
                
                missing = expected_set - model_set
                if missing:
                    print(f"     Missing: {', '.join(sorted(missing))}")
                elif model_set == expected_set:
                    print(f"    Perfect match!")
        
        print("\n" + "="*80)
        return
    
    # Load data
    dataset_path = args.dataset
    if not Path(dataset_path).exists():
        dataset_path = f"final_project/{args.dataset}"
    pipeline.load_data(dataset_path)
    
    # Run evaluation
    results = pipeline.run_full_evaluation(directive=args.directive, threshold=0.5)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n Evaluation results saved to: {args.output}")


if __name__ == "__main__":
    main()


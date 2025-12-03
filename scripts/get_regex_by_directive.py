"""
Get learned regex pattern based on directive.

Usage:
    python scripts/get_regex_by_directive.py --directive balanced
    python scripts/get_regex_by_directive.py --directive strictly --domain bank
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse

from pipeline.algorithm_registry import AlgorithmRegistry
from common.config import PII_TYPES, NUM_PII, SCENARIO_NAME2ID
from common.mdp import build_state


def get_regex_pattern(
    algorithm: str,
    model_path: str,
    directive: str,
    domain: str = None
) -> dict:
    """
    Get learned regex pattern for a given directive.
    
    Args:
        algorithm: Algorithm name (grpo, groupedppo, vanillarl)
        model_path: Path to trained model
        directive: "strictly", "balanced", or "accurately"
        domain: "restaurant" or "bank" (if None, returns both)
    
    Returns:
        Dictionary with regex patterns for each domain
    """
    # Load model
    algo_config = AlgorithmRegistry.get(algorithm.lower())
    policy = AlgorithmRegistry.create_policy(algorithm.lower())
    
    try:
        policy.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        print(f" Error loading model: {e}")
        return None
    
    policy.eval()
    
    # Validate directive
    valid_directives = ["strictly", "balanced", "accurately"]
    if directive.lower() not in valid_directives:
        print(f" Invalid directive: {directive}")
        print(f"   Valid options: {', '.join(valid_directives)}")
        return None
    
    directive = directive.lower()
    
    # Get patterns for requested domain(s)
    domains = [domain] if domain else ["restaurant", "bank"]
    results = {}
    
    with torch.no_grad():
        for dom in domains:
            if dom not in ["restaurant", "bank"]:
                print(f"  Invalid domain: {dom}, skipping...")
                continue
            
            scenario_id = SCENARIO_NAME2ID[dom]
            
            # Test with all PII present
            present_mask = [1] * NUM_PII
            state = build_state(present_mask, scenario_id)
            
            # Get actions with directive
            actions = policy.act(state, deterministic=True, threshold=0.5, directive=directive)
            
            # Convert to regex pattern
            shared_pii = [PII_TYPES[i] for i, a in enumerate(actions) if a == 1]
            
            if shared_pii:
                regex_pattern = " | ".join(shared_pii)
            else:
                regex_pattern = "(none)"
            
            results[dom] = {
                "regex": regex_pattern,
                "shared_pii": shared_pii,
                "count": len(shared_pii)
            }
    
    return results


def print_regex(results: dict, directive: str, domain: str = None):
    """Print regex patterns in a readable format."""
    
    # Expected patterns from dataset
    expected = {
        "restaurant": {"EMAIL", "PHONE"},
        "bank": {"EMAIL", "PHONE", "DATE/DOB", "SSN", "CREDIT_CARD"}
    }
    
    print("\n" + "="*80)
    print(f"LEARNED REGEX PATTERN - Directive: {directive.upper()}")
    print("="*80)
    
    if domain:
        # Single domain
        if domain in results:
            r = results[domain]
            expected_set = expected[domain]
            model_set = set(r['shared_pii'])
            
            print(f"\n {domain.upper()} Domain:")
            print(f"   Regex: {r['regex']}")
            print(f"   Shared PII: {', '.join(r['shared_pii']) if r['shared_pii'] else '(none)'}")
            print(f"   Count: {r['count']} PII types")
            print(f"\n   Expected (from dataset): {', '.join(sorted(expected_set))}")
            
            missing = expected_set - model_set
            if missing:
                print(f"     Missing: {', '.join(sorted(missing))}")
            elif model_set == expected_set:
                print(f"    Perfect match!")
    else:
        # Both domains
        for dom in ["restaurant", "bank"]:
            if dom in results:
                r = results[dom]
                expected_set = expected[dom]
                model_set = set(r['shared_pii'])
                
                print(f"\n {dom.upper()} Domain:")
                print(f"   Regex: {r['regex']}")
                print(f"   Shared PII: {', '.join(r['shared_pii']) if r['shared_pii'] else '(none)'}")
                print(f"   Count: {r['count']} PII types")
                print(f"   Expected (from dataset): {', '.join(sorted(expected_set))}")
                
                missing = expected_set - model_set
                if missing:
                    print(f"     Missing: {', '.join(sorted(missing))}")
                elif model_set == expected_set:
                    print(f"    Perfect match!")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Get learned regex pattern based on directive',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get regex for balanced directive (both domains)
  python scripts/get_regex_by_directive.py --directive balanced
  
  # Get regex for strictly directive in bank domain
  python scripts/get_regex_by_directive.py --directive strictly --domain bank
  
  # Get regex for accurately directive
  python scripts/get_regex_by_directive.py --directive accurately
        """
    )
    
    parser.add_argument('--directive', type=str, required=True,
                       choices=['strictly', 'balanced', 'accurately'],
                       help='Directive: strictly (high privacy), balanced (default), accurately (high utility)')
    parser.add_argument('--domain', type=str, default=None,
                       choices=['restaurant', 'bank'],
                       help='Domain to analyze (if not specified, shows both)')
    parser.add_argument('--algorithm', type=str, default='grpo',
                       choices=AlgorithmRegistry.list_algorithms(),
                       help='Algorithm to use')
    parser.add_argument('--model', type=str, default='models/grpo_model.pt',
                       help='Path to trained model file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (optional, saves as JSON)')
    
    args = parser.parse_args()
    
    # Check model path
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = Path("final_project") / args.model
        if not model_path.exists():
            print(f" Model not found: {args.model}")
            print(" Train a model first using: python pipeline/train.py --algorithm grpo")
            return
    
    # Get regex patterns
    results = get_regex_pattern(
        args.algorithm,
        str(model_path),
        args.directive,
        args.domain
    )
    
    if results:
        print_regex(results, args.directive, args.domain)
        
        # Save to file if requested
        if args.output:
            import json
            output_data = {
                "directive": args.directive,
                "algorithm": args.algorithm,
                "patterns": results
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n Results saved to: {args.output}")
    else:
        print("\n Failed to get regex patterns")


if __name__ == "__main__":
    main()


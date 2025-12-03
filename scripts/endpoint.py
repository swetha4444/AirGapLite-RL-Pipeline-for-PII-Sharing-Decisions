#!/usr/bin/env python3
"""
API Endpoint for getting regex patterns from trained models.

Usage:
    # Command line
    python endpoint.py --algorithm grpo --directive balanced --domain bank
    
    # As a function
    from endpoint import get_regex
    result = get_regex("grpo", "balanced", "bank")
    print(result)  # ['EMAIL', 'PHONE', 'DATE/DOB', 'SSN', 'CREDIT_CARD']
"""

import sys
from pathlib import Path
import torch
import argparse
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.algorithm_registry import AlgorithmRegistry
from common.config import PII_TYPES, NUM_PII, SCENARIO_NAME2ID
from common.mdp import build_state


def get_regex(
    algorithm: str,
    directive: str,
    domain: str,
    model_path: str = None
) -> list:
    """
    Get learned regex pattern as a list of PII types.
    
    Args:
        algorithm: Algorithm name (grpo, groupedppo, vanillarl)
        directive: "strictly", "balanced", or "accurately"
        domain: "restaurant" or "bank"
        model_path: Path to trained model (default: models/{algorithm}_model.pt)
    
    Returns:
        List of PII types that should be shared (e.g., ['EMAIL', 'PHONE', 'SSN'])
        Returns empty list if error occurs.
    """
    # Validate inputs
    valid_algorithms = ["grpo", "groupedppo", "vanillarl"]
    if algorithm.lower() not in valid_algorithms:
        print(f"Error: Invalid algorithm '{algorithm}'. Valid options: {', '.join(valid_algorithms)}", file=sys.stderr)
        return []
    
    valid_directives = ["strictly", "balanced", "accurately"]
    if directive.lower() not in valid_directives:
        print(f"Error: Invalid directive '{directive}'. Valid options: {', '.join(valid_directives)}", file=sys.stderr)
        return []
    
    valid_domains = ["restaurant", "bank"]
    if domain.lower() not in valid_domains:
        print(f"Error: Invalid domain '{domain}'. Valid options: {', '.join(valid_domains)}", file=sys.stderr)
        return []
    
    algorithm = algorithm.lower()
    directive = directive.lower()
    domain = domain.lower()
    
    # Default model path
    if model_path is None:
        model_path = f"models/{algorithm}_model.pt"
    
    model_path = Path(model_path)
    if not model_path.exists():
        # Try relative to project root
        alt_path = Path(__file__).parent / model_path
        if alt_path.exists():
            model_path = alt_path
        else:
            print(f"Error: Model not found at {model_path}", file=sys.stderr)
            return []
    
    try:
        # Load model
        algo_config = AlgorithmRegistry.get(algorithm)
        policy = AlgorithmRegistry.create_policy(algorithm)
        policy.load_state_dict(torch.load(model_path, map_location='cpu'))
        policy.eval()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return []
    
    # Get regex pattern
    try:
        scenario_id = SCENARIO_NAME2ID[domain]
        
        # Test with all PII present
        present_mask = [1] * NUM_PII
        state = build_state(present_mask, scenario_id)
        
        # Get actions with directive
        with torch.no_grad():
            actions = policy.act(state, deterministic=True, threshold=0.5, directive=directive)
        
        # Convert to list of PII types
        shared_pii = [PII_TYPES[i] for i, a in enumerate(actions) if a == 1]
        
        return shared_pii
        
    except Exception as e:
        print(f"Error getting regex: {e}", file=sys.stderr)
        return []


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Get regex pattern from trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python endpoint.py --algorithm grpo --directive balanced --domain bank
  python endpoint.py --algorithm groupedppo --directive strictly --domain restaurant --model models/groupedppo_model.pt
  python endpoint.py --algorithm grpo --directive accurately --domain bank --json
        """
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        required=True,
        choices=['grpo', 'groupedppo', 'vanillarl'],
        help='Algorithm name'
    )
    
    parser.add_argument(
        '--directive',
        type=str,
        required=True,
        choices=['strictly', 'balanced', 'accurately'],
        help='Directive: strictly (high threshold), balanced (default), accurately (low threshold)'
    )
    
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=['restaurant', 'bank'],
        help='Domain: restaurant or bank'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (default: models/{algorithm}_model.pt)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    args = parser.parse_args()
    
    # Get regex
    result = get_regex(
        algorithm=args.algorithm,
        directive=args.directive,
        domain=args.domain,
        model_path=args.model
    )
    
    # Output
    if args.json:
        output = {
            "algorithm": args.algorithm,
            "directive": args.directive,
            "domain": args.domain,
            "regex": result,
            "count": len(result)
        }
        print(json.dumps(output, indent=2))
    else:
        if result:
            print(f"Regex pattern: {' | '.join(result)}")
            print(f"PII types: {result}")
            print(f"Count: {len(result)}")
        else:
            print("No PII types to share")
            sys.exit(1)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Comparison Script: Baseline LLM Minimizer vs RL-Based Integration Pipeline

Compares:
1. Utility: % of allowed PII correctly shared
2. Privacy: % of disallowed PII correctly NOT shared  
3. Quickness: Inference time (seconds)

Usage:
    python compare_baseline_vs_rl.py --test-cases test_cases.json
    python compare_baseline_vs_rl.py --num-samples 10 --domain restaurant
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import time
import pandas as pd

# Import torch only if needed for GPU baseline
try:
    import torch
except ImportError:
    torch = None

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "MLP"))
sys.path.insert(0, str(PROJECT_ROOT / "pii_extraction"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline"))

from integration_pipeline import minimize_data
from common.config import PII_TYPES

# Try to import MLX baseline minimizer (works on Apple Silicon without GPU)
try:
    from mlx_baseline_minimizer import load_model_mlx, minimize_pii_mlx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX baseline minimizer not available (install: pip install mlx mlx-lm)")

# Try to import GPU baseline minimizer as fallback
try:
    from baseline_minimizer import load_model, minimize_pii
    GPU_BASELINE_AVAILABLE = True
except ImportError:
    GPU_BASELINE_AVAILABLE = False


def extract_pii_from_text(text: str) -> List[str]:
    """Extract all PII types present in text."""
    from pii_extraction.spacy_regex import extract_pii as _extract_all_pii
    from pii_extraction.pii_extractor import SPACY_TO_GRPO
    
    entities = json.loads(_extract_all_pii(text))
    pii_types = set()
    for entity in entities:
        pii_type = SPACY_TO_GRPO.get(entity['label'], entity['label'])
        if pii_type in PII_TYPES:
            pii_types.add(pii_type)
    return list(pii_types)


def evaluate_baseline(
    model, tokenizer, prompt: str, user_data: str, 
    allowed_pii: List[str], domain: str, use_mlx: bool = False
) -> Dict[str, Any]:
    """
    Run baseline LLM minimizer and return metrics.
    
    Returns:
        {
            'minimized_pii': List[str],
            'utility': float,
            'privacy': float,
            'inference_time': float,
            'tokens': int
        }
    """
    # Get all PII present in user data
    present_pii = extract_pii_from_text(user_data)
    
    # Create task description from prompt
    task = prompt
    
    start_time = time.time()
    
    try:
        if use_mlx:
            minimized_pii, decision_text, elapsed, tokens = minimize_pii_mlx(
                model, tokenizer, user_data, task, present_pii
            )
        else:
            minimized_pii, decision_text, elapsed, tokens = minimize_pii(
                model, tokenizer, user_data, task, present_pii
            )
    except Exception as e:
        print(f"Baseline error: {e}", file=sys.stderr)
        return {
            'minimized_pii': [],
            'utility': 0.0,
            'privacy': 0.0,
            'inference_time': 0.0,
            'tokens': 0,
            'error': str(e)
        }
    
    inference_time = time.time() - start_time
    
    # Calculate metrics
    private_pii = set(present_pii) - set(allowed_pii)
    protected = private_pii - set(minimized_pii)
    
    privacy_score = len(protected) / len(private_pii) if private_pii else 1.0
    utility_score = len(set(minimized_pii) & set(allowed_pii)) / len(allowed_pii) if allowed_pii else 0.0
    
    return {
        'minimized_pii': minimized_pii,
        'utility': utility_score,
        'privacy': privacy_score,
        'inference_time': inference_time,
        'tokens': tokens,
        'present_pii': present_pii,
        'allowed_pii': allowed_pii,
        'private_pii': list(private_pii)
    }


def evaluate_rl_pipeline(
    prompt: str, user_data: str, allowed_pii: List[str],
    algorithm: str = "grpo", directive: str = "balanced"
) -> Dict[str, Any]:
    """
    Run RL-based integration pipeline and return metrics.
    
    Returns:
        {
            'minimized_pii': List[str],
            'utility': float,
            'privacy': float,
            'inference_time': float
        }
    """
    # Detect domain from prompt
    from MLP.context_agent_classifier import ContextAgentClassifier
    classifier = ContextAgentClassifier()
    classifier_path = PROJECT_ROOT / "MLP" / "context_agent_mlp.pth"
    if not classifier_path.exists():
        classifier_path = Path(__file__).parent.parent / "MLP" / "context_agent_mlp.pth"
    classifier.load_model(str(classifier_path))
    LABELS = {0: "restaurant", 1: "bank"}
    context_result = classifier.predict(prompt, LABELS)
    domain = context_result['label']
    
    start_time = time.time()
    
    try:
        result = minimize_data(
            third_party_prompt=prompt,
            user_data=user_data,
            algorithm=algorithm,
            directive=directive
        )
    except Exception as e:
        print(f"RL pipeline error: {e}", file=sys.stderr)
        return {
            'minimized_pii': [],
            'utility': 0.0,
            'privacy': 0.0,
            'inference_time': 0.0,
            'error': str(e)
        }
    
    inference_time = time.time() - start_time
    
    if 'error' in result:
        return {
            'minimized_pii': [],
            'utility': 0.0,
            'privacy': 0.0,
            'inference_time': inference_time,
            'error': result['error']
        }
    
    # Get minimized PII from result
    minimized_pii = list(result.get('minimized_pii_dict', {}).keys())
    
    # Get all PII present in user data
    present_pii = extract_pii_from_text(user_data)
    
    # Calculate metrics
    private_pii = set(present_pii) - set(allowed_pii)
    protected = private_pii - set(minimized_pii)
    
    privacy_score = len(protected) / len(private_pii) if private_pii else 1.0
    utility_score = len(set(minimized_pii) & set(allowed_pii)) / len(allowed_pii) if allowed_pii else 0.0
    
    return {
        'minimized_pii': minimized_pii,
        'utility': utility_score,
        'privacy': privacy_score,
        'inference_time': inference_time,
        'present_pii': present_pii,
        'allowed_pii': allowed_pii,
        'private_pii': list(private_pii),
        'domain': domain
    }


def compare_methods(
    test_cases: List[Dict[str, Any]],
    baseline_model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
    rl_algorithm: str = "grpo",
    rl_directive: str = "balanced",
    use_mlx: bool = None  # None = auto-detect (prefer MLX)
) -> pd.DataFrame:
    """
    Compare baseline and RL methods on test cases.
    
    Args:
        test_cases: List of dicts with 'prompt', 'user_data', 'allowed_pii', 'domain'
        baseline_model_name: Model to use for baseline
        rl_algorithm: RL algorithm to use
        rl_directive: Privacy directive
    
    Returns:
        DataFrame with comparison results
    """
    print("=" * 80)
    print("COMPARISON: Baseline LLM Minimizer vs RL-Based Integration Pipeline")
    print("=" * 80)
    
    # Load baseline model (prefer MLX for Apple Silicon, fallback to GPU)
    baseline_model = None
    tokenizer = None
    baseline_available = False
    # Always try MLX first if available (works on Apple Silicon without GPU)
    use_mlx_baseline = False
    
    # Try MLX first (works on Apple Silicon without GPU)
    if MLX_AVAILABLE:
        print(f"\nLoading MLX baseline model: {baseline_model_name}...")
        try:
            baseline_model, tokenizer = load_model_mlx(baseline_model_name)
            print("✓ MLX baseline model loaded")
            baseline_available = True
            use_mlx_baseline = True
        except Exception as e:
            print(f"⚠ MLX baseline model failed to load: {e}")
            print(f"  Error details: {type(e).__name__}: {str(e)}")
            baseline_available = False
    elif use_mlx is True:
        # User explicitly requested MLX but it's not available
        print(f"\n⚠ MLX not available but requested. Install with: pip install mlx mlx-lm")
        print("  Trying GPU baseline as fallback...")
    
    # Fallback to GPU baseline if MLX failed or not requested
    if not baseline_available and GPU_BASELINE_AVAILABLE and torch and torch.cuda.is_available():
        print(f"\nTrying GPU baseline model: {baseline_model_name}...")
        try:
            baseline_model, tokenizer = load_model(baseline_model_name)
            print("✓ GPU baseline model loaded")
            baseline_available = True
            use_mlx_baseline = False
        except Exception as e:
            print(f"⚠ GPU baseline model not available: {e}")
            baseline_available = False
    
    if not baseline_available:
        print("\n⚠ Baseline model not available - continuing with RL pipeline only")
        print("  To use baseline:")
        print("    - For Apple Silicon: pip install mlx mlx-lm")
        print("    - For GPU: pip install bitsandbytes transformers (requires CUDA)")
    
    results = []
    
    for idx, test_case in enumerate(test_cases, 1):
        prompt = test_case['prompt']
        user_data = test_case['user_data']
        allowed_pii = test_case['allowed_pii']
        domain = test_case.get('domain', 'restaurant')
        
        print(f"\n{'='*80}")
        print(f"Test Case {idx}/{len(test_cases)}")
        print(f"Prompt: {prompt[:60]}...")
        print(f"Domain: {domain}")
        print(f"Allowed PII: {allowed_pii}")
        
        # Evaluate Baseline (if available)
        if baseline_available:
            print(f"\n[1/2] Running Baseline LLM Minimizer ({'MLX' if use_mlx_baseline else 'GPU'})...")
            baseline_result = evaluate_baseline(
                baseline_model, tokenizer, prompt, user_data, allowed_pii, domain, use_mlx=use_mlx_baseline
            )
        else:
            print(f"\n[1/2] Skipping Baseline LLM Minimizer (not available)")
            baseline_result = {
                'minimized_pii': [],
                'utility': 0.0,
                'privacy': 0.0,
                'inference_time': 0.0,
                'tokens': 0,
                'present_pii': extract_pii_from_text(user_data),
                'allowed_pii': allowed_pii,
                'private_pii': [],
                'error': 'Baseline model not available'
            }
        
        # Evaluate RL Pipeline
        print(f"[2/2] Running RL Integration Pipeline ({rl_algorithm})...")
        rl_result = evaluate_rl_pipeline(
            prompt, user_data, allowed_pii, rl_algorithm, rl_directive
        )
        
        # Store comparison
        results.append({
            'test_id': idx,
            'prompt': prompt,
            'domain': domain,
            'allowed_pii': ', '.join(allowed_pii),
            'present_pii': ', '.join(baseline_result.get('present_pii', [])),
            
            # Baseline metrics
            'baseline_minimized': ', '.join(baseline_result.get('minimized_pii', [])),
            'baseline_utility': baseline_result.get('utility', 0.0),
            'baseline_privacy': baseline_result.get('privacy', 0.0),
            'baseline_time': baseline_result.get('inference_time', 0.0),
            'baseline_tokens': baseline_result.get('tokens', 0),
            
            # RL metrics
            'rl_minimized': ', '.join(rl_result.get('minimized_pii', [])),
            'rl_utility': rl_result.get('utility', 0.0),
            'rl_privacy': rl_result.get('privacy', 0.0),
            'rl_time': rl_result.get('inference_time', 0.0),
            
            # Differences
            'utility_diff': rl_result.get('utility', 0.0) - baseline_result.get('utility', 0.0),
            'privacy_diff': rl_result.get('privacy', 0.0) - baseline_result.get('privacy', 0.0),
            'time_speedup': baseline_result.get('inference_time', 0.0) / max(rl_result.get('inference_time', 0.001), 0.001),
            
            # Errors
            'baseline_error': baseline_result.get('error', ''),
            'rl_error': rl_result.get('error', '')
        })
        
        # Print summary for this test case
        print(f"\n--- Results ---")
        if baseline_available:
            print(f"Baseline: Utility={baseline_result.get('utility', 0):.1%}, "
                  f"Privacy={baseline_result.get('privacy', 0):.1%}, "
                  f"Time={baseline_result.get('inference_time', 0):.3f}s")
            print(f"RL Pipeline: Utility={rl_result.get('utility', 0):.1%}, "
                  f"Privacy={rl_result.get('privacy', 0):.1%}, "
                  f"Time={rl_result.get('inference_time', 0):.3f}s")
            if baseline_result.get('inference_time', 0) > 0:
                print(f"Speedup: {baseline_result.get('inference_time', 0) / max(rl_result.get('inference_time', 0.001), 0.001):.1f}x")
        else:
            print(f"RL Pipeline: Utility={rl_result.get('utility', 0):.1%}, "
                  f"Privacy={rl_result.get('privacy', 0):.1%}, "
                  f"Time={rl_result.get('inference_time', 0):.3f}s")
            print(f"(Baseline skipped - not available)")
    
    df = pd.DataFrame(results)
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    if len(df) > 0:
        if baseline_available:
            print(f"\nBaseline LLM Minimizer:")
            print(f"  Average Utility: {df['baseline_utility'].mean():.1%}")
            print(f"  Average Privacy: {df['baseline_privacy'].mean():.1%}")
            print(f"  Average Time: {df['baseline_time'].mean():.3f}s")
            print(f"  Total Time: {df['baseline_time'].sum():.3f}s")
            
            print(f"\nRL Integration Pipeline ({rl_algorithm}):")
            print(f"  Average Utility: {df['rl_utility'].mean():.1%}")
            print(f"  Average Privacy: {df['rl_privacy'].mean():.1%}")
            print(f"  Average Time: {df['rl_time'].mean():.3f}s")
            print(f"  Total Time: {df['rl_time'].sum():.3f}s")
            
            print(f"\nComparison:")
            print(f"  Utility Difference: {df['utility_diff'].mean():+.1%} "
                  f"({'better' if df['utility_diff'].mean() > 0 else 'worse'})")
            print(f"  Privacy Difference: {df['privacy_diff'].mean():+.1%} "
                  f"({'better' if df['privacy_diff'].mean() > 0 else 'worse'})")
            if df['baseline_time'].sum() > 0:
                print(f"  Average Speedup: {df['time_speedup'].mean():.1f}x faster")
                print(f"  Total Time Saved: {df['baseline_time'].sum() - df['rl_time'].sum():.3f}s")
        else:
            print(f"\nRL Integration Pipeline ({rl_algorithm}) - Baseline Not Available:")
            print(f"  Average Utility: {df['rl_utility'].mean():.1%}")
            print(f"  Average Privacy: {df['rl_privacy'].mean():.1%}")
            print(f"  Average Time: {df['rl_time'].mean():.3f}s")
            print(f"  Total Time: {df['rl_time'].sum():.3f}s")
            print(f"\nNote: Baseline comparison skipped (GPU/bitsandbytes not available)")
    
    return df


def create_test_cases(num_samples: int = 10, domain: str = "restaurant") -> List[Dict[str, Any]]:
    """Create sample test cases."""
    test_cases = []
    
    if domain == "restaurant":
        prompts = [
            "I need to book a table for tonight",
            "Reserve a table for 2 people",
            "Can I make a reservation?",
            "Book a table for dinner",
            "I want to reserve a table"
        ]
        allowed_pii = ["PHONE", "EMAIL"]
    else:  # bank
        prompts = [
            "I need to check my account balance",
            "Transfer money to another account",
            "What is my current balance?",
            "I want to open a new account",
            "Check my transaction history"
        ]
        allowed_pii = ["PHONE", "EMAIL", "DATE/DOB", "SSN", "CREDIT_CARD"]
    
    user_data_samples = [
        "Hi, my name is John Smith. My email is john@example.com and you can reach me at 555-1234. My SSN is 123-45-6789.",
        "Name: Jane Doe, Email: jane@example.com, Phone: 555-5678, SSN: 987-65-4321, Credit Card: 4111-1111-1111-1111",
        "Contact me at email@test.com or phone 555-9999. My SSN is 111-22-3333 and DOB is 01/15/1990.",
        "Email: user@domain.com, Phone: 555-0000, Name: Bob Johnson, SSN: 444-55-6666",
        "My information: john.doe@email.com, 555-1111, SSN 222-33-4444, Credit Card ending in 1234"
    ]
    
    for i in range(min(num_samples, len(prompts) * len(user_data_samples))):
        prompt = prompts[i % len(prompts)]
        user_data = user_data_samples[i % len(user_data_samples)]
        test_cases.append({
            'prompt': prompt,
            'user_data': user_data,
            'allowed_pii': allowed_pii,
            'domain': domain
        })
    
    return test_cases[:num_samples]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare Baseline LLM Minimizer vs RL Integration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--test-cases',
        type=str,
        default=None,
        help='JSON file with test cases (if not provided, generates samples)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of test samples to generate (default: 5)'
    )
    
    parser.add_argument(
        '--domain',
        type=str,
        default='restaurant',
        choices=['restaurant', 'bank'],
        help='Domain for test cases (default: restaurant)'
    )
    
    parser.add_argument(
        '--baseline-model',
        type=str,
        default='mlx-community/Qwen2.5-7B-Instruct-4bit',
        help='Baseline model name (default: mlx-community/Qwen2.5-7B-Instruct-4bit for MLX)'
    )
    
    parser.add_argument(
        '--use-mlx',
        action='store_true',
        help='Use MLX for baseline (works on Apple Silicon, default: True if MLX available)'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU baseline instead of MLX (requires CUDA)'
    )
    
    parser.add_argument(
        '--rl-algorithm',
        type=str,
        default='grpo',
        choices=['grpo', 'groupedppo', 'vanillarl'],
        help='RL algorithm to use (default: grpo)'
    )
    
    parser.add_argument(
        '--rl-directive',
        type=str,
        default='balanced',
        choices=['strictly', 'balanced', 'accurately'],
        help='Privacy directive (default: balanced)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='comparison_results.csv',
        help='Output CSV file (default: comparison_results.csv)'
    )
    
    args = parser.parse_args()
    
    # Load or create test cases
    if args.test_cases:
        with open(args.test_cases, 'r') as f:
            test_cases = json.load(f)
    else:
        test_cases = create_test_cases(args.num_samples, args.domain)
    
    # Run comparison
    # Always try MLX first if available (unless explicitly using GPU)
    if args.use_gpu:
        use_mlx = False
    else:
        # Default: Always try MLX if available
        use_mlx = MLX_AVAILABLE
    
    df = compare_methods(
        test_cases,
        baseline_model_name=args.baseline_model,
        rl_algorithm=args.rl_algorithm,
        rl_directive=args.rl_directive,
        use_mlx=use_mlx
    )
    
    # Save results
    if len(df) > 0:
        output_path = PROJECT_ROOT / args.output
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
    else:
        print("\n✗ No results to save")


if __name__ == '__main__':
    main()

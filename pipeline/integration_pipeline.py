#!/usr/bin/env python3
"""
Integration Pipeline: Third-party prompt + user data → minimized data

Flow:
1. Third-party prompt + user data → Context Classifier → Domain
2. Domain → GRPO → Regex (allowed PII types)
3. User data + Regex → PII Extractor → Minimized data

Usage:
    from integration_pipeline import minimize_data
    
    result = minimize_data(
        third_party_prompt="I need to book a table for tonight",
        user_data="My name is John Smith, email is john@example.com, phone is 555-1234, SSN is 123-45-6789"
    )
    print(result['minimized_data'])  # Only EMAIL and PHONE
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from pipeline/ to final_project/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "MLP"))
sys.path.insert(0, str(PROJECT_ROOT / "pii_extraction"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import components with better error handling
try:
    # We'll import ContextAgentClassifier directly when needed
    pass
except ImportError as e:
    print("=" * 60)
    print("ERROR: Failed to import context classifier")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nPlease ensure:")
    print("  1. MLP/inference.py exists")
    print("  2. sentence-transformers is installed: pip install sentence-transformers")
    print("  3. torch is installed: pip install torch")
    sys.exit(1)

try:
    from scripts.endpoint import get_regex
except ImportError as e:
    print("=" * 60)
    print("ERROR: Failed to import GRPO endpoint")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nPlease ensure scripts/endpoint.py exists")
    sys.exit(1)

try:
    from pii_extraction.pii_extractor import extract_pii, extract_pii_by_type, SPACY_TO_GRPO
    from pii_extraction.spacy_regex import extract_pii as _extract_all_pii
except ImportError as e:
    print("=" * 60)
    print("ERROR: Failed to import PII extractor")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nMissing dependencies detected. Please install:")
    print("  pip install spacy")
    print("  python -m spacy download en_core_web_sm")
    print("\nAlso ensure other dependencies are installed:")
    print("  pip install sentence-transformers torch pandas")
    sys.exit(1)


def minimize_data(
    third_party_prompt: str,
    user_data: str,
    algorithm: str = "grpo",
    directive: str = "balanced",
    classifier_model_path: Optional[str] = None,
    grpo_model_path: Optional[str] = None,
    return_full_details: bool = False
) -> Dict[str, Any]:
    """
    Main integration function: Minimize user data based on third-party prompt context.
    
    Args:
        third_party_prompt: The prompt/query from third party (e.g., "Book a table")
        user_data: User's data containing PII (e.g., "Name: John, Email: john@example.com")
        algorithm: RL algorithm to use (default: "grpo")
        directive: Privacy directive - "strictly", "balanced", or "accurately" (default: "balanced")
        classifier_model_path: Path to context classifier model (default: MLP/context_agent_mlp.pth)
        grpo_model_path: Path to GRPO model (default: models/{algorithm}_model.pt)
        return_full_details: If True, return detailed breakdown (default: False)
    
    Returns:
        Dictionary with:
            - domain: Detected domain ("restaurant" or "bank")
            - confidence: Classifier confidence
            - allowed_pii_types: List of PII types allowed for this domain
            - extracted_pii: List of extracted PII entities
            - minimized_data: User data with only allowed PII
            - removed_pii: List of PII types that were removed (optional if return_full_details=True)
    """
    
    # Step 1: Context Classification - Get domain from third-party prompt
    try:
        # Set default classifier model path if not provided
        if classifier_model_path is None:
            classifier_model_path = str(PROJECT_ROOT / "MLP" / "context_agent_mlp.pth")
        
        # Load classifier model directly
        from MLP.context_agent_classifier import ContextAgentClassifier
        classifier = ContextAgentClassifier()
        classifier.load_model(classifier_model_path)
        LABELS = {0: "restaurant", 1: "bank"}
        result = classifier.predict(third_party_prompt, LABELS)
        domain = result['label']
        confidence = result['confidence']
    except Exception as e:
        return {
            "error": f"Context classification failed: {e}",
            "domain": None,
            "minimized_data": user_data  # Return original data on error
        }
    
    # Step 2: Get allowed PII types from GRPO for this domain
    try:
        if grpo_model_path is None:
            grpo_model_path = PROJECT_ROOT / "models" / f"{algorithm}_model.pt"
        
        allowed_pii_types = get_regex(
            algorithm=algorithm,
            directive=directive,
            domain=domain,
            model_path=str(grpo_model_path)
        )
        
        if not allowed_pii_types:
            # If GRPO fails, return original data
            return {
                "error": "Failed to get PII types from GRPO",
                "domain": domain,
                "confidence": confidence,
                "minimized_data": user_data
            }
    except Exception as e:
        return {
            "error": f"GRPO regex extraction failed: {e}",
            "domain": domain,
            "confidence": confidence,
            "minimized_data": user_data
        }
    
    # Step 3: Extract PII from user data based on allowed types
    try:
        extracted_entities = extract_pii(
            text=user_data,
            domain=domain,
            algorithm=algorithm,
            directive=directive
        )
        
        # Get all PII types present in user data (for comparison)
        all_extracted = extract_pii_by_type(
            text=user_data,
            domain=domain,
            algorithm=algorithm,
            directive=directive
        )
        
        # Build minimized data - only keep allowed PII
        minimized_pii = {}
        for entity in extracted_entities:
            pii_type = entity['label']
            if pii_type in allowed_pii_types:
                if pii_type not in minimized_pii:
                    minimized_pii[pii_type] = []
                minimized_pii[pii_type].append(entity['text'])
        
        # Create structured format for convenience
        minimized_data_parts = []
        for pii_type in allowed_pii_types:
            if pii_type in minimized_pii:
                for value in minimized_pii[pii_type]:
                    minimized_data_parts.append(f"{pii_type}: {value}")
        
        minimized_data_structured = ", ".join(minimized_data_parts) if minimized_data_parts else "(No allowed PII found)"
        
        # Create minimized data: redact disallowed PII from original text
        minimized_text = user_data
        # Get all entities from raw text extraction
        try:
            all_entities_json = _extract_all_pii(user_data)
            all_entities = json.loads(all_entities_json) if isinstance(all_entities_json, str) else all_entities_json
            # Sort entities by start position (descending) to redact from end to start
            all_entities_sorted = sorted(
                all_entities,
                key=lambda x: x['start'],
                reverse=True
            )
            
            # Import SPACY_TO_GRPO mapping
            from pii_extraction.pii_extractor import SPACY_TO_GRPO
            
            # Redact disallowed PII from the text
            for entity in all_entities_sorted:
                pii_type = SPACY_TO_GRPO.get(entity['label'], entity['label'])
                if pii_type not in allowed_pii_types:
                    # Redact this PII (replace with [REDACTED])
                    start = entity['start']
                    end = entity['end']
                    minimized_text = minimized_text[:start] + "[REDACTED]" + minimized_text[end:]
        except Exception as e:
            # If redaction fails, fall back to structured format
            print(f"Warning: Could not redact text, using structured format: {e}", file=sys.stderr)
            minimized_text = minimized_data_structured
        
        # Build result
        result = {
            "domain": domain,
            "confidence": confidence,
            "allowed_pii_types": allowed_pii_types,
            "extracted_pii": extracted_entities,
            "minimized_data": minimized_text,  # Redacted text (raw format)
            "minimized_data_structured": minimized_data_structured,  # Structured format
            "minimized_pii_dict": minimized_pii
        }
        
        # Add detailed breakdown if requested
        if return_full_details:
            # Find removed PII types
            all_pii_in_data = set(all_extracted.keys())
            removed_pii = all_pii_in_data - set(allowed_pii_types)
            
            result.update({
                "all_pii_found": all_extracted,
                "removed_pii_types": list(removed_pii),
                "original_data": user_data,
                "third_party_prompt": third_party_prompt
            })
        
        return result
        
    except Exception as e:
        return {
            "error": f"PII extraction failed: {e}",
            "domain": domain,
            "confidence": confidence,
            "allowed_pii_types": allowed_pii_types,
            "minimized_data": user_data
        }


def main():
    """Command-line interface for the integration pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integration Pipeline: Minimize user data based on third-party prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python integration_pipeline.py \\
    --prompt "I need to book a table" \\
    --data "Name: John, Email: john@example.com, Phone: 555-1234, SSN: 123-45-6789"
  
  # With custom directive
  python integration_pipeline.py \\
    --prompt "Check my account balance" \\
    --data "Email: user@bank.com, SSN: 123-45-6789" \\
    --directive strictly
  
  # JSON output with full details
  python integration_pipeline.py \\
    --prompt "Reserve a table" \\
    --data "Name: Jane, Email: jane@example.com" \\
    --json --full-details
        """
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Third-party prompt/query'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='User data containing PII'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        default='grpo',
        choices=['grpo', 'groupedppo', 'vanillarl'],
        help='RL algorithm to use (default: grpo)'
    )
    
    parser.add_argument(
        '--directive',
        type=str,
        default='balanced',
        choices=['strictly', 'balanced', 'accurately'],
        help='Privacy directive (default: balanced)'
    )
    
    parser.add_argument(
        '--classifier-model',
        type=str,
        default=None,
        help='Path to context classifier model (default: MLP/context_agent_mlp.pth)'
    )
    
    parser.add_argument(
        '--grpo-model',
        type=str,
        default=None,
        help='Path to GRPO model (default: models/{algorithm}_model.pt)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    parser.add_argument(
        '--full-details',
        action='store_true',
        help='Return full detailed breakdown'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    result = minimize_data(
        third_party_prompt=args.prompt,
        user_data=args.data,
        algorithm=args.algorithm,
        directive=args.directive,
        classifier_model_path=args.classifier_model,
        grpo_model_path=args.grpo_model,
        return_full_details=args.full_details
    )
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            sys.exit(1)
        
        print("=" * 60)
        print("INTEGRATION PIPELINE RESULTS")
        print("=" * 60)
        print(f"\nThird-party Prompt: {args.prompt}")
        print(f"Detected Domain: {result['domain'].upper()} (confidence: {result['confidence']:.2%})")
        print(f"\nAllowed PII Types: {', '.join(result['allowed_pii_types'])}")
        print(f"\nMinimized Data (Redacted Text):")
        print(f"  {result['minimized_data']}")
        print(f"\nMinimized Data (Structured):")
        print(f"  {result.get('minimized_data_structured', 'N/A')}")
        
        if args.full_details:
            print(f"\nExtracted PII Entities: {len(result['extracted_pii'])}")
            if result.get('removed_pii_types'):
                print(f"Removed PII Types: {', '.join(result['removed_pii_types'])}")
        
        print("=" * 60)


if __name__ == '__main__':
    main()
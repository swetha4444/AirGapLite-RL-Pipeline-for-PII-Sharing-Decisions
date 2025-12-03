#!/usr/bin/env python3
"""
PII Extractor - Extract PII from text based on GRPO-learned patterns.

The classifier provides the domain, and this module extracts the relevant PII types.

Usage:
    from pii_extractor import extract_pii
    
    # Extract PII for a domain (classifier provides domain)
    entities = extract_pii("My email is test@example.com", domain="bank")
    # Returns: [{"text": "test@example.com", "label": "EMAIL", "start": 12, "end": 28}]
    
    # Get just the PII values grouped by type
    from pii_extractor import extract_pii_by_type
    result = extract_pii_by_type("SSN: 123-45-6789, email: test@example.com", domain="bank")
    # Returns: {"SSN": ["123-45-6789"], "EMAIL": ["test@example.com"]}
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # Also add pii_extraction folder for local imports

from spacy_regex import extract_pii as _extract_all_pii
from scripts.endpoint import get_regex as _get_regex


def get_regex(algorithm: str, directive: str, domain: str) -> list:
    """Wrapper that passes absolute model path to endpoint.get_regex."""
    model_path = PROJECT_ROOT / "models" / f"{algorithm}_model.pt"
    return _get_regex(algorithm, directive, domain, model_path=str(model_path))


# Label mappings: GRPO format <-> spacy_regex format
GRPO_TO_SPACY = {
    "NAME": ["PERSON"], "PHONE": ["PHONE"], "EMAIL": ["EMAIL"], "DATE/DOB": ["DATE"],
    "company": ["ORG"], "location": ["GPE", "LOC"], "IP": ["IP_ADDRESS"],
    "SSN": ["SSN"], "CREDIT_CARD": ["CREDIT_CARD_16", "CREDIT_CARD_4"],
    "age": ["AGE"], "sex": ["SEX"],
}
SPACY_TO_GRPO = {s: g for g, labels in GRPO_TO_SPACY.items() for s in labels}


def extract_pii(text: str, domain: str, algorithm: str = "grpo", 
                directive: str = "balanced") -> List[Dict[str, Any]]:
    """
    Extract PII from text based on domain-specific GRPO patterns.
    
    Args:
        text: Raw text to extract PII from
        domain: Domain from classifier ("restaurant" or "bank")
        algorithm: RL algorithm (default: "grpo")
        directive: Privacy level - "strictly", "balanced", or "accurately"
    
    Returns:
        List of entities: [{"text": "...", "label": "EMAIL", "start": 0, "end": 10}, ...]
    """
    if not text:
        return []
    
    # Get allowed PII types from GRPO for this domain
    pii_types = get_regex(algorithm, directive, domain)
    allowed_labels = {s for t in pii_types for s in GRPO_TO_SPACY.get(t, [])}
    
    # Extract and filter entities, normalize labels to GRPO format
    entities = []
    for e in json.loads(_extract_all_pii(text)):
        if e["label"] in allowed_labels:
            e["label"] = SPACY_TO_GRPO.get(e["label"], e["label"])
            entities.append(e)
    
    return entities


def extract_pii_by_type(text: str, domain: str, algorithm: str = "grpo",
                        directive: str = "balanced") -> Dict[str, List[str]]:
    """
    Extract PII from text, grouped by type (simplified output).
    
    Args:
        text: Raw text to extract PII from
        domain: Domain from classifier ("restaurant" or "bank")
    
    Returns:
        Dict mapping PII type to list of values: {"EMAIL": ["test@example.com"], "SSN": ["123-45-6789"]}
    """
    result = {}
    for e in extract_pii(text, domain, algorithm, directive):
        result.setdefault(e["label"], []).append(e["text"])
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract PII using GRPO-learned patterns")
    parser.add_argument('--domain', required=True, choices=['restaurant', 'bank'], help='Domain from classifier')
    parser.add_argument('--text', type=str, help='Text to extract PII from')
    parser.add_argument('--directive', default='balanced', choices=['strictly', 'balanced', 'accurately'])
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    # Show what PII types are extracted for this domain
    pii_types = get_regex("grpo", args.directive, args.domain)
    print(f"Domain: {args.domain} | Directive: {args.directive} | PII Types: {pii_types}\n")
    
    texts = [args.text] if args.text else [
        "Hello, my name is John Smith and my email is john.smith@example.com",
        "Please call me at 555-123-4567. My SSN is 123-45-6789.",
        "My credit card number is 4111-1111-1111-1111 and DOB is 01/15/1990.",
    ]
    
    for text in texts:
        entities = extract_pii(text, args.domain, directive=args.directive)
        if args.json:
            print(json.dumps(entities))
        else:
            pii_found = ", ".join(f"{e['label']}: '{e['text']}'" for e in entities) or "(No PII)"
            print(f"Text: {text}\n  → {pii_found}\n")

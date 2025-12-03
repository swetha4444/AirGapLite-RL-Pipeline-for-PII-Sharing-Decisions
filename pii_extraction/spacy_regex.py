import re
import json
import pandas as pd
import spacy

# -------------------------
# Load spaCy model
# -------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Regex patterns
# -------------------------

# Email
EMAIL_RE = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
)

# Phone number (very simple pattern, US-style-ish)
PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)

# IPv4 address
IP_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

# US SSN: 123-45-6789
SSN_RE = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

# Credit card: 16 digits (e.g. 1234-5678-9012-3456 or 1234 5678 9012 3456 or 1234567890123456)
CREDIT_CARD_16_RE = re.compile(
    r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
)

# Credit card last 4: e.g. "xxxx1234", "XXXX1234", "**** 1234", "last 4: 1234", "ending 1234", "card ending 1234", "ends in 1234"
CREDIT_CARD_LAST4_RE = re.compile(
    r"(?:(?:xxxx|XXXX|\*{4}|last\s+4(?:\s+digits)?\s*[:\-]?|(?:card\s+)?end(?:s|ing)(?:\s+in)?)\s*)(\d{4})",
    re.IGNORECASE
)

# Age pattern: "25 years old", "age 25", "age is 33", "aged 25", "25 yo", "25 y/o"
AGE_RE = re.compile(
    r"\b(?:age[d]?\s*(?:is\s*)?:?\s*(\d{1,3})|(?<!\d)(\d{1,3})\s*(?:years?\s*old|y/?o|yrs?\s*old))\b",
    re.IGNORECASE
)

# Sex/Gender pattern: explicit gender mentions
SEX_RE = re.compile(
    r"\b(male|female)\b",
    re.IGNORECASE
)

# Date/DOB pattern: MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, YYYY/MM/DD
# Also matches dates with context like "DOB 01/01/1990" or "date of birth 01/01/1990"
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
)

# Company domain pattern: matches "my company domain.com" style mentions
# Captures domain names that appear after "company" context
# Examples: "my company apexanalytics.com", "company bluecloud.io"
COMPANY_DOMAIN_RE = re.compile(
    r"(?:my\s+)?company\s+([a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)*\.(?:com|io|ai|org|net|dev|co))\b",
    re.IGNORECASE
)

# Explicit name pattern: matches "my name is First Last" or "name is First Last"
# Captures names that spaCy's NER misses (e.g., South Asian names like "Amit Sharma", "Ivy Singh")
# Examples: "my name is Amit Sharma", "name is Ivy Desai", "and my name is Ananya Kumar"
NAME_EXPLICIT_RE = re.compile(
    r"(?:my\s+)?name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    re.IGNORECASE
)

# -------------------------
# False positive filters
# -------------------------
# SpaCy labels PII-related keywords as ORG incorrectly
FALSE_POSITIVE_ORG_TEXTS = {"SSN", "DOB", "IP", "ssn", "dob", "ip"}

# SpaCy extracts area codes or partial numbers as CARDINAL - filter short numbers
# that are substrings of phone/SSN matches
MIN_CARDINAL_LENGTH = 4  # Filter out 3-digit area codes

# Credit card brand names that spaCy incorrectly labels as ORG or GPE
# These are not PII themselves, just context
CREDIT_CARD_BRANDS = {"visa", "mastercard", "amex", "discover", "american express"}

# Street suffixes to detect addresses incorrectly labeled as PERSON
STREET_SUFFIXES = {"st", "st.", "street", "ave", "ave.", "avenue", "rd", "rd.", "road", 
                   "blvd", "blvd.", "boulevard", "ln", "ln.", "lane", "dr", "dr.", "drive",
                   "ct", "ct.", "court", "pl", "pl.", "place", "way", "cir", "circle"}

# Regex to detect name context - phrases that indicate the following text is a person's name
# Matches: "name", "my name is", "I'm", "I am", "this is", "called", "named"
NAME_CONTEXT_RE = re.compile(
    r"(?:(?:my\s+)?name\s+(?:is\s+)?|I'm\s+|I\s+am\s+|this\s+is\s+|called\s+|named\s+)$",
    re.IGNORECASE
)

# Pattern to check if text looks like a person's name (First Last, each capitalized)
PERSON_NAME_PATTERN = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$")

# -------------------------
# File names and directories
# -------------------------
import os

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Output subdirectory
EXTRACTED_PII_DIR = os.path.join(SCRIPT_DIR, "extracted_pii")

# Datasets to process: (input_filename, output_name)
DATASETS = [
    ("690-Project-Dataset-final.csv", "dataset_final"),
    ("690-Project-Dataset-balanced.csv", "dataset_balanced"),
    ("690-Project-Dataset-1500-bank-balanced-55-50-57.csv", "dataset_1500_bank_balanced"),
]


def extract_regex_pii(text: str):
    """
    Use regex to find specific PII-like patterns in the text.
    Returns a list of dicts.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    results = []

    # First, find credit card 16-digit matches to exclude from phone detection
    credit_card_spans = []
    for m in CREDIT_CARD_16_RE.finditer(text):
        credit_card_spans.append((m.start(), m.end()))

    # Email
    for m in EMAIL_RE.finditer(text):
        results.append(
            {
                "text": m.group(),
                "label": "EMAIL",
                "start": m.start(),
                "end": m.end(),
                "source": "regex",
            }
        )

    # Phone - but skip matches that overlap with credit card spans
    for m in PHONE_RE.finditer(text):
        phone_start, phone_end = m.start(), m.end()
        overlaps_cc = any(
            cc_start <= phone_start < cc_end or cc_start < phone_end <= cc_end
            for cc_start, cc_end in credit_card_spans
        )
        if not overlaps_cc:
            results.append(
                {
                    "text": m.group(),
                    "label": "PHONE",
                    "start": m.start(),
                    "end": m.end(),
                    "source": "regex",
                }
            )

    # IP address
    for m in IP_RE.finditer(text):
        results.append(
            {
                "text": m.group(),
                "label": "IP_ADDRESS",
                "start": m.start(),
                "end": m.end(),
                "source": "regex",
            }
        )

    # SSN
    for m in SSN_RE.finditer(text):
        results.append(
            {
                "text": m.group(),
                "label": "SSN",
                "start": m.start(),
                "end": m.end(),
                "source": "regex",
            }
        )

    # Credit card 16 digits
    for m in CREDIT_CARD_16_RE.finditer(text):
        results.append(
            {
                "text": m.group(),
                "label": "CREDIT_CARD_16",
                "start": m.start(),
                "end": m.end(),
                "source": "regex",
            }
        )

    # Credit card last 4
    for m in CREDIT_CARD_LAST4_RE.finditer(text):
        # group(1) is just the 4 digits
        results.append(
            {
                "text": m.group(1),
                "label": "CREDIT_CARD_4",
                "start": m.start(1),
                "end": m.end(1),
                "source": "regex",
            }
        )

    # Age
    for m in AGE_RE.finditer(text):
        # group(1) is from "age 25" pattern, group(2) is from "25 years old" pattern
        age_val = m.group(1) if m.group(1) else m.group(2)
        if age_val:
            results.append(
                {
                    "text": age_val,
                    "label": "AGE",
                    "start": m.start(1) if m.group(1) else m.start(2),
                    "end": m.end(1) if m.group(1) else m.end(2),
                    "source": "regex",
                }
            )

    # Sex/Gender
    for m in SEX_RE.finditer(text):
        results.append(
            {
                "text": m.group(),
                "label": "SEX",
                "start": m.start(),
                "end": m.end(),
                "source": "regex",
            }
        )

    # Date/DOB - but skip if it overlaps with IP address (IP addresses look like dates)
    ip_spans = [(m.start(), m.end()) for m in IP_RE.finditer(text)]
    for m in DATE_RE.finditer(text):
        date_start, date_end = m.start(), m.end()
        # Check if this date overlaps with an IP address
        overlaps_ip = any(
            ip_start <= date_start < ip_end or ip_start < date_end <= ip_end
            for ip_start, ip_end in ip_spans
        )
        if not overlaps_ip:
            results.append(
                {
                    "text": m.group(),
                    "label": "DATE",
                    "start": m.start(),
                    "end": m.end(),
                    "source": "regex",
                }
            )

    # Company domain names (e.g., "my company apexanalytics.com")
    for m in COMPANY_DOMAIN_RE.finditer(text):
        # group(1) captures just the domain name
        results.append(
            {
                "text": m.group(1),
                "label": "COMPANY_DOMAIN",
                "start": m.start(1),
                "end": m.end(1),
                "source": "regex",
            }
        )

    # Explicit name pattern (e.g., "my name is Amit Sharma")
    # This catches names that spaCy's NER misses
    for m in NAME_EXPLICIT_RE.finditer(text):
        # group(1) captures just the name (First Last)
        name_text = m.group(1)
        # Normalize capitalization: "amit sharma" -> "Amit Sharma"
        name_text = " ".join(word.capitalize() for word in name_text.split())
        results.append(
            {
                "text": name_text,
                "label": "PERSON",
                "start": m.start(1),
                "end": m.end(1),
                "source": "regex",
            }
        )

    return results


def filter_spacy_false_positives(entities: list, regex_entities: list, original_text: str = "") -> list:
    """
    Filter out known spaCy false positives:
    1. SSN/DOB/IP labels detected as ORG
    2. Area codes detected as CARDINAL (3-digit numbers that are part of phone matches)
    3. IP addresses detected as DATE/CARDINAL (prefer regex IP_ADDRESS)
    4. Emails detected as GPE/PERSON/ORG (prefer regex EMAIL)
    5. Credit card last 4 detected as DATE (prefer regex CREDIT_CARD_4)
    6. Age phrases like "age 29" detected as DATE (prefer regex AGE)
    7. Dates incorrectly labeled as PERSON (prefer regex DATE)
    8. Credit card brands (Visa, Mastercard) labeled as ORG/GPE - not PII
    9. Street names incorrectly labeled as PERSON (e.g., "Broad St")
    10. Entities labeled as ORG/PRODUCT that are preceded by "name" context → relabel as PERSON
    """
    # Build a set of regex spans for overlap checking
    regex_spans = {(e["start"], e["end"], e["label"]) for e in regex_entities}
    regex_span_ranges = [(e["start"], e["end"]) for e in regex_entities]

    filtered = []
    for ent in entities:
        text = ent["text"]
        label = ent["label"]
        start, end = ent["start"], ent["end"]

        # 1. Filter SSN/DOB/IP labels as ORG
        if label == "ORG" and text.upper() in {t.upper() for t in FALSE_POSITIVE_ORG_TEXTS}:
            continue

        # 2. Filter ORG that includes "IP " prefix (e.g., "IP 192.168.1.77")
        if label == "ORG" and text.upper().startswith("IP "):
            continue

        # 3. Filter short CARDINAL values (likely area codes or partial matches)
        if label == "CARDINAL" and len(text) < MIN_CARDINAL_LENGTH:
            # Check if this CARDINAL is contained within a regex match
            is_substring = any(
                rs <= start and end <= re for rs, re in regex_span_ranges
            )
            if is_substring or text.isdigit():
                continue

        # 4. Filter PERSON labels that look like dates (e.g., "07/04/1991" detected as PERSON)
        if label == "PERSON" and DATE_RE.fullmatch(text):
            continue

        # 5. Filter spaCy entities that overlap with regex matches (prefer regex)
        # This handles: IP as DATE, emails as GPE/PERSON, etc.
        overlaps_regex = False
        for rs, re, rlabel in regex_spans:
            # Check if spans overlap significantly
            if start < re and end > rs:  # overlapping
                # Prefer regex for specific types
                if rlabel in {"IP_ADDRESS", "EMAIL", "PHONE", "SSN", "CREDIT_CARD_16", "CREDIT_CARD_4", "AGE", "DATE", "COMPANY_DOMAIN"}:
                    overlaps_regex = True
                    break
        if overlaps_regex:
            continue

        # 6. Filter "age X" detected as DATE by spaCy (regex AGE is better)
        if label == "DATE" and text.lower().startswith("age "):
            continue

        # 7. Filter temporal phrases that aren't actual PII dates
        if label == "DATE" and text.lower() in {"last week", "yesterday", "today", "tomorrow", "last month", "last year"}:
            continue
        
        # 7b. Filter spaCy DATE that is just 4 digits (likely credit card last 4, year, etc.)
        # Real dates should have separators like "/" or "-"
        if label == "DATE" and text.isdigit() and len(text) == 4:
            continue

        # 8. Filter credit card brands labeled as ORG or GPE (not PII, just context)
        if label in {"ORG", "GPE"} and text.lower() in CREDIT_CARD_BRANDS:
            continue

        # 9. Filter street names incorrectly labeled as PERSON
        # Check if text ends with a street suffix (e.g., "Broad St", "Main Ave")
        if label == "PERSON":
            text_lower = text.lower()
            words = text_lower.split()
            if len(words) >= 1 and words[-1] in STREET_SUFFIXES:
                continue

        # 10. Relabel ORG/PRODUCT/FAC as PERSON if preceded by name context
        # e.g., "and name Avery Campbell" → Avery Campbell should be PERSON
        # Note: spaCy sometimes labels names as FAC (Facility) or ORG
        if label in {"ORG", "PRODUCT", "FAC"} and original_text and PERSON_NAME_PATTERN.match(text):
            # Check the text before this entity for name context
            prefix_text = original_text[:start]
            if NAME_CONTEXT_RE.search(prefix_text):
                ent = ent.copy()
                ent["label"] = "PERSON"
                label = "PERSON"  # Update local variable to avoid FAC→ORG conversion below

        # 11. Normalize FAC (Facility) labels to ORG for company names
        if label == "FAC":
            ent = ent.copy()  # Don't modify the original
            ent["label"] = "ORG"

        filtered.append(ent)

    return filtered


def filter_redundant_credit_card_last4(entities: list) -> list:
    """
    Remove CREDIT_CARD_4 entities when a CREDIT_CARD_16 already contains those digits.
    This avoids duplicate detection like "5555 4444 3333 2222" + "2222".
    """
    # Find all CREDIT_CARD_16 spans and their last 4 digits
    cc16_entities = [e for e in entities if e["label"] == "CREDIT_CARD_16"]
    cc16_last4_digits = set()
    for cc in cc16_entities:
        # Extract last 4 digits from the credit card number
        digits_only = ''.join(c for c in cc["text"] if c.isdigit())
        if len(digits_only) >= 4:
            cc16_last4_digits.add(digits_only[-4:])

    # Filter out CREDIT_CARD_4 if its digits match any CREDIT_CARD_16's last 4
    filtered = []
    for ent in entities:
        if ent["label"] == "CREDIT_CARD_4" and ent["text"] in cc16_last4_digits:
            continue
        filtered.append(ent)

    return filtered


def deduplicate_entities(entities: list) -> list:
    """
    Remove duplicate entities that have the same span but different sources.
    Prefer regex labels over spaCy for overlapping spans.
    """
    # Group by (start, end) span
    span_to_entities = {}
    for ent in entities:
        span = (ent["start"], ent["end"])
        if span not in span_to_entities:
            span_to_entities[span] = []
        span_to_entities[span].append(ent)

    deduplicated = []
    for span, ents in span_to_entities.items():
        if len(ents) == 1:
            deduplicated.append(ents[0])
        else:
            # Multiple entities for same span - prefer regex source
            regex_ents = [e for e in ents if e["source"] == "regex"]
            spacy_ents = [e for e in ents if e["source"] == "spacy"]

            if regex_ents:
                # Take the first regex match (they should be the same)
                deduplicated.append(regex_ents[0])
            else:
                # No regex match, take first spaCy
                deduplicated.append(spacy_ents[0])

    # Sort by start position for consistent output
    deduplicated.sort(key=lambda x: x["start"])
    return deduplicated


def extract_pii(text: str) -> str:
    """
    Combine spaCy entities + regex matches.
    Return as a JSON string to be stored in the CSV.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    spacy_entities = []
    regex_entities = []

    # 1) spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        spacy_entities.append(
            {
                "text": ent.text,
                "label": ent.label_,        # e.g. PERSON, ORG, GPE
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy",
            }
        )

    # 2) Regex-based PII
    regex_entities = extract_regex_pii(text)

    # 3) Filter spaCy false positives (pass original text for context-aware filtering)
    filtered_spacy = filter_spacy_false_positives(spacy_entities, regex_entities, text)

    # 4) Filter redundant CREDIT_CARD_4 when CREDIT_CARD_16 is present
    filtered_regex = filter_redundant_credit_card_last4(regex_entities)

    # 5) Combine and deduplicate
    all_entities = filtered_spacy + filtered_regex
    deduplicated = deduplicate_entities(all_entities)

    # 6) Return as JSON string
    return json.dumps(deduplicated, ensure_ascii=False)


def main():
    """
    Process all configured datasets and save extracted PII to organized subdirectories.
    """
    # Create output directory if it doesn't exist
    os.makedirs(EXTRACTED_PII_DIR, exist_ok=True)
    
    print("=" * 60)
    print("PII EXTRACTION PIPELINE")
    print("=" * 60)
    
    for input_filename, output_name in DATASETS:
        input_path = os.path.join(PROJECT_DIR, input_filename)
        output_path = os.path.join(EXTRACTED_PII_DIR, f"{output_name}_extracted.csv")
        
        print(f"\n--- Processing: {input_filename} ---")
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"  WARNING: Input file not found: {input_path}")
            print(f"  Skipping this dataset.")
            continue
        
        # Read the input CSV (must have column "conversation")
        df = pd.read_csv(input_path)
        
        if "conversation" not in df.columns:
            print(f"  WARNING: 'conversation' column not found in {input_filename}")
            print(f"  Skipping this dataset.")
            continue
        
        print(f"  Rows: {len(df)}")
        
        # Apply PII extraction
        df["pii_entities"] = df["conversation"].apply(extract_pii)
        
        # Save to new CSV
        df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("PII extraction complete!")
    print(f"Output directory: {EXTRACTED_PII_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
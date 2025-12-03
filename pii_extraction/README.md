# How to Extract PII

Extract PII from text based on domain-specific patterns learned by GRPO.

## Quick Start

```python
from pii_extraction.pii_extractor import extract_pii

# Classifier provides domain → extract relevant PII
entities = extract_pii("My SSN is 123-45-6789", domain="bank")
# Returns: [{"text": "123-45-6789", "label": "SSN", "start": 10, "end": 21}]
```

> **Note:** All paths (model files, imports) are resolved automatically relative to the project structure. Just ensure you're running from the `final_project/` directory or importing with the full module path `pii_extraction.pii_extractor`.

## How It Works

1. **GRPO determines what to extract** — Based on domain, returns allowed PII types
2. **spacy_regex extracts PII** — Uses spaCy NER + regex patterns to find all PII in text
3. **pii_extractor filters results** — Only returns PII types allowed for the domain

### Domain → PII Types Mapping

| Domain | PII Types Extracted |
|--------|---------------------|
| `bank` | `PHONE`, `EMAIL`, `DATE/DOB`, `SSN`, `CREDIT_CARD` |
| `restaurant` | `PHONE`, `EMAIL` |

## API Reference

### `extract_pii(text, domain, algorithm, directive)`

Extract PII from text, filtered by domain-specific GRPO patterns.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | str | Yes | - | Raw text to extract PII from |
| `domain` | str | Yes | - | `"bank"` or `"restaurant"` |
| `algorithm` | str | No | `"grpo"` | `"grpo"`, `"groupedppo"`, or `"vanillarl"` |
| `directive` | str | No | `"balanced"` | `"strictly"`, `"balanced"`, or `"accurately"` |

**Returns:** List of entity dicts with `text`, `label`, `start`, `end`, `source`

```python
from pii_extraction.pii_extractor import extract_pii

# Bank domain - extracts financial PII
extract_pii("SSN: 123-45-6789, email: a@b.com", domain="bank")
# [{"text": "123-45-6789", "label": "SSN", ...}, {"text": "a@b.com", "label": "EMAIL", ...}]

# Restaurant domain - only contact PII  
extract_pii("SSN: 123-45-6789, email: a@b.com", domain="restaurant")
# [{"text": "a@b.com", "label": "EMAIL", ...}]  ← SSN filtered out

# With directive for privacy control
extract_pii("My email is a@b.com", domain="bank", directive="strictly")
```

### `extract_pii_by_type(text, domain, algorithm, directive)`

Same parameters as `extract_pii`, but returns values grouped by PII type.

**Returns:** Dict mapping PII type to list of extracted values

```python
from pii_extraction.pii_extractor import extract_pii_by_type

extract_pii_by_type("Call 555-1234, SSN 123-45-6789", domain="bank")
# {"PHONE": ["555-1234"], "SSN": ["123-45-6789"]}
```

## Command Line

```bash
# Run from the final_project directory
cd final_project

# Extract from text
python pii_extraction/pii_extractor.py --domain bank --text "My SSN is 123-45-6789"

# With directive
python pii_extraction/pii_extractor.py --domain bank --directive strictly --text "My SSN is 123-45-6789"

# JSON output
python pii_extraction/pii_extractor.py --domain bank --text "My SSN is 123-45-6789" --json

# Demo with sample texts
python pii_extraction/pii_extractor.py --domain bank
```

**CLI Arguments:**
| Argument | Required | Choices | Description |
|----------|----------|---------|-------------|
| `--domain` | Yes | `bank`, `restaurant` | Domain from classifier |
| `--text` | No | - | Text to extract PII from (demo texts if omitted) |
| `--directive` | No | `strictly`, `balanced`, `accurately` | Privacy/utility tradeoff |
| `--json` | No | - | Output as JSON |

## Supported PII Types

All 11 PII types from `common/config.py`:

| PII Type | Label | Example Patterns |
|----------|-------|------------------|
| `NAME` | Person names | "John Smith", "Jane Doe" |
| `PHONE` | Phone numbers | "555-123-4567", "(555) 123-4567" |
| `EMAIL` | Email addresses | "user@example.com" |
| `DATE/DOB` | Dates/birthdays | "01/15/1990", "1990-01-15" |
| `company` | Organizations | "Acme Corp", "Google" |
| `location` | Places | "New York", "California" |
| `IP` | IP addresses | "192.168.1.1" |
| `SSN` | Social Security | "123-45-6789" |
| `CREDIT_CARD` | Credit cards | "4111-1111-1111-1111" |
| `age` | Age mentions | "25" (from "age 25", "25 years old") |
| `sex` | Gender | "male", "female" |

## Directive Options

Controls the privacy/utility tradeoff threshold:

| Directive | Threshold | Description |
|-----------|-----------|-------------|
| `strictly` | ≥0.7 | Higher privacy, lower utility |
| `balanced` | ≥0.5 | Default balanced tradeoff |
| `accurately` | ≤0.3 | Higher utility, lower privacy |

## Integration with GRPO

The extractor uses `get_regex()` from `scripts/endpoint.py`:

```python
from scripts.endpoint import get_regex

# Get allowed PII types for domain
pii_types = get_regex("grpo", "balanced", "bank")
# ['PHONE', 'EMAIL', 'DATE/DOB', 'SSN', 'CREDIT_CARD']

pii_types = get_regex("grpo", "balanced", "restaurant")
# ['PHONE', 'EMAIL']
```

---

## Running the Extraction Pipeline

To process datasets and evaluate PII extraction performance:

```bash
# Step 1: Extract PII from datasets
python pii_extraction/spacy_regex.py

# Step 2: Compute metrics
python pii_extraction/compute_pii_metrics.py

# Step 3: Analyze errors
python pii_extraction/analyze_errors.py
```

The pipeline processes CSV datasets located in `final_project/` (the project root) and outputs results to organized subdirectories within `pii_extraction/`:
- `extracted_pii/` - PII extraction results
- `pii_metrics/` - Performance metrics
- `error_analysis/` - Error breakdowns

All paths are resolved automatically relative to script locations.

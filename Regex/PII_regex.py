import re
from typing import Dict, List

PII_NAME = "NAME"
PII_EMAIL = "EMAIL"
PII_PHONE = "PHONE"
PII_SSN = "SSN"
PII_DATE = "DATE/DOB"

def detect_pii_with_regex(text: str) -> Dict[str, List[str]]:
    """
    Very simple regex-based PII detector.
    Returns a mapping PII_TYPE -> list of matched values in the text.
    """
    pii: Dict[str, List[str]] = {}

    # EMAIL
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, text)
    if emails:
        pii[PII_EMAIL] = emails

    # PHONE (very rough)
    phone_pattern = r"\b(?:\+?\d{1,2}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
    phones = re.findall(phone_pattern, text)
    if phones:
        pii[PII_PHONE] = [p.strip() for p in phones]

    # SSN (US style)
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    ssns = re.findall(ssn_pattern, text)
    if ssns:
        pii[PII_SSN] = ssns

    # DATE (very simple – you can improve)
    date_pattern = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
    dates = re.findall(date_pattern, text)
    if dates:
        pii[PII_DATE] = dates

    # NAME is the only one tricky: you probably already have it from another system
    # For now you can manually annotate or skip it in regex.
    # Example placeholder: if you know "my name is X"
    name_pattern = r"\bmy name is\s+([A-Z][a-zA-Z]+)\b"
    names = re.findall(name_pattern, text)
    if names:
        pii[PII_NAME] = names

    return pii

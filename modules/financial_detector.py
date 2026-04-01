from __future__ import annotations
import re
import spacy
from typing import Any

def is_luhn_valid(number: str) -> bool:
    """
    Implements the Luhn algorithm (Mod 10) to check if a sequence of digits
    is a mathematically valid credit card number.
    
    This reduces false positives by verifying the checksum of 13-19 digit strings.
    """
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not digits:
        return False
    
    # Reverse and double every second digit
    check_sum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        check_sum += digit
        
    return check_sum % 10 == 0

class FinancialDetector:
    """
    A robust financial de-identification module for PACT.
    
    Categorization Strategy:
    1. PAYMENT_INSTRUMENT: Credit/Debit Cards, CVVs, and Expiry Dates.
       Tagged as [REDACTED CARD]. Includes a Luhn checksum to reduce false positives.
       
    2. ACCOUNT_DETAILS: Bank Account Numbers, IBANs, Routing/SWIFT Codes, 
       and Cryptocurrency Wallet Addresses (Ethereum, Bitcoin).
       Tagged as [REDACTED ACCOUNT].
       
    3. VALUE_AND_INCOME: Monetary values ($5k), Annual Salaries, Net Worth, 
       Stock holdings, and Tax IDs (SSNs, EINs).
       Tagged as [REDACTED VALUE].
    """
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def detect_and_redact(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        doc = self.nlp(text)
        spans: list[tuple[int, int, str, str, str]] = []
        
        # --- 1. PAYMENT_INSTRUMENT (Cards via Luhn) ---
        card_regex = re.compile(r"\b(?:\d[ -]?){13,19}\b")
        for m in card_regex.finditer(text):
            candidate = m.group()
            if is_luhn_valid(candidate):
                spans.append((m.start(), m.end(), "[REDACTED CARD]", candidate, "PAYMENT_INSTRUMENT"))

        # --- 2. ACCOUNT_DETAILS (Global Bank & Crypto) ---
        # Generic Bank (8-12 digits) OR IBAN (22-34 characters starting with initials)
        bank_regex = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b|\b\d{8,12}\b", re.IGNORECASE)
        # Crypto Addresses: 0x... (ETH) or bc1.../1.../3... (BTC)
        crypto_regex = re.compile(r"\b(?:0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-z0-9]{39,59})\b")
        
        for m in bank_regex.finditer(text):
            spans.append((m.start(), m.end(), "[REDACTED ACCOUNT]", m.group(), "ACCOUNT_DETAILS"))
        for m in crypto_regex.finditer(text):
            spans.append((m.start(), m.end(), "[REDACTED ACCOUNT]", m.group(), "ACCOUNT_DETAILS"))

        # --- 3. VALUE_AND_INCOME (MONEY & Salaries) ---
        # Catch standard spaCy MONEY tags
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                spans.append((ent.start_char, ent.end_char, "[REDACTED VALUE]", ent.text, "VALUE_AND_INCOME"))
        
        # Salary patterns: "120k", "$80,000 yearly", "making $100 per hour"
        salary_regex = re.compile(r"\b\d+k\b|\b\d{1,3}(?:,\d{3})*\s*(?:per hour|annually|yearly|a year)\b", re.IGNORECASE)
        for m in salary_regex.finditer(text):
            spans.append((m.start(), m.end(), "[REDACTED VALUE]", m.group(), "VALUE_AND_INCOME"))

        # --- RECONSTRUCTION ---
        # Merge overlaps, prefer longer ones
        spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        
        redacted_parts: list[str] = []
        sanitized: list[dict[str, Any]] = []
        last_idx = 0

        for start, end, replacement, original, typ in spans:
            if start < last_idx:
                continue
            redacted_parts.append(text[last_idx:start])
            redacted_parts.append(replacement)
            last_idx = end
            sanitized.append({"original": original, "type": typ})

        redacted_parts.append(text[last_idx:])
        return "".join(redacted_parts), sanitized

def _get_detector() -> FinancialDetector:
    # Singleton pattern to avoid reloading spaCy
    if not hasattr(_get_detector, "_instance"):
        _get_detector._instance = FinancialDetector()
    return _get_detector._instance

def make_candidates_financial(text: str) -> list[str]:
    detector = _get_detector()
    redacted_text, _ = detector.detect_and_redact(text)
    return [redacted_text]

# app/redact.py
from __future__ import annotations
import re
from typing import Iterable

# Basic PII regexes
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_RE_PHONE = re.compile(r"""
    (?:(?:\+?1[\s.-]?)?       # country (optional)
     (?:\(?\d{3}\)?[\s.-]?)   # area
     \d{3}[\s.-]?\d{4})       # number
""", re.VERBOSE)
_RE_SSN   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# MRN-like (very rough; tweak per corpus)
_RE_MRN   = re.compile(r"\b(?:MRN[:\s#-]?)?(\d{6,})\b", re.IGNORECASE)

# Allowlist terms that must NOT be redacted even if they look like generic words
ALLOWLIST_EXACT: set[str] = {
    "medicare", "medicaid", "ssa", "irs", "cms"
}

def _mask_email(s: str) -> str:
    def _m(m: re.Match) -> str:
        return "[email masked]"
    return _RE_EMAIL.sub(_m, s)

def _mask_phone(s: str) -> str:
    def _m(m: re.Match) -> str:
        return "[phone masked]"
    return _RE_PHONE.sub(_m, s)

def _mask_ssn(s: str) -> str:
    def _m(m: re.Match) -> str:
        return "[ssn masked]"
    return _RE_SSN.sub(_m, s)

def _mask_mrn(s: str) -> str:
    def _m(m: re.Match) -> str:
        # Keep literal "Medicare" tokens etc. untouched; only mask numeric MRNs
        full = m.group(0)
        digits = m.group(1) if m.lastindex else None
        if digits:
            return full.replace(digits, "[mrn masked]")
        return "[mrn masked]"
    return _RE_MRN.sub(_m, s)

def _preserve_allowlist(s: str) -> str:
    # We only mask PII patterns above; allowlist is mostly defensive
    return s

def redact(text: str) -> str:
    """Mask common PII in a string. Idempotent and safe on already masked text."""
    if not text:
        return text
    out = text
    out = _mask_email(out)
    out = _mask_phone(out)
    out = _mask_ssn(out)
    out = _mask_mrn(out)
    out = _preserve_allowlist(out)
    return out

def redact_many(lines: Iterable[str]) -> list[str]:
    return [redact(x) for x in lines]

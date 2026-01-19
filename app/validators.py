# app/validators.py
from __future__ import annotations
import re
from typing import List, Tuple, Set

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+", re.M)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def _split_sentences(text: str) -> List[str]:
    # very lightweight sentence splitter
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]

def _extract_years(text: str) -> Set[str]:
    return set(_YEAR_RE.findall(text))  # returns matches of the first group? -> fix below

def extract_years(text: str) -> Set[str]:
    return set(m.group(0) for m in _YEAR_RE.finditer(text))

def assert_has_citations(answer: str) -> bool:
    # Require at least one [n] marker in bullets or paragraphs
    return bool(re.search(r"\[\d+\]", answer))

def add_citations_round_robin(answer: str, n_cites: int) -> str:
    """
    Append [i] markers to each bullet/sentence in round-robin fashion.
    We don't alter content; just add markers at line ends if missing.
    """
    if n_cites <= 0:
        return answer

    lines = answer.splitlines()
    cite_idx = 1
    out = []
    for ln in lines:
        if not ln.strip():
            out.append(ln)
            continue
        needs_marker = not re.search(r"\[\d+\]\s*$", ln)
        if needs_marker:
            # Prefer to tag bullets and summary lines
            if _BULLET_RE.match(ln) or ln.strip() and not ln.strip().startswith("**Source"):
                ln = ln.rstrip() + f" [{cite_idx}]"
                cite_idx = (cite_idx % n_cites) + 1
        out.append(ln)
    return "\n".join(out)

def assert_no_external_claims(answer: str, context: str) -> Tuple[bool, str]:
    """
    Heuristic: the answer shouldn't introduce years not present in context.
    Returns (ok, reason).
    """
    ans_years = extract_years(answer)
    ctx_years = extract_years(context)
    extra = ans_years - ctx_years
    if extra:
        return False, f"Answer references unseen years: {sorted(extra)}"
    return True, ""

def repair_external_claims(answer: str, allowed_years: Set[str]) -> str:
    """
    Drop sentences that contain disallowed years. If a bullet line holds only
    disallowed info, remove that line.
    """
    lines = answer.splitlines()
    fixed = []
    for ln in lines:
        sents = _split_sentences(ln) if ln.strip() else []
        if not sents:
            fixed.append(ln)
            continue
        kept_sents = []
        for s in sents:
            yrs = extract_years(s)
            if not yrs or yrs.issubset(allowed_years):
                kept_sents.append(s)
        if kept_sents:
            fixed.append(" ".join(kept_sents))
        # else drop the line entirely
    return "\n".join(fixed).strip()

def validate_and_repair(answer: str, context: str, citations: List[str]) -> Tuple[bool, str, str]:
    """
    Returns (ok, final_answer, reason_if_not_ok)
    """
    # 1) Enforce no-external-claims (years)
    ok, reason = assert_no_external_claims(answer, context)
    if not ok:
        allowed = extract_years(context)
        repaired = repair_external_claims(answer, allowed)
        # Re-check after repair
        ok2, reason2 = assert_no_external_claims(repaired, context)
        if not ok2 or not repaired.strip():
            return False, "", reason or reason2
        answer = repaired

    # 2) Enforce citation markers
    if not assert_has_citations(answer):
        answer = add_citations_round_robin(answer, len(citations))

    # 3) Ensure we didn't produce empty output
    if not answer.strip():
        return False, "", "Empty output after repair"

    return True, answer, ""

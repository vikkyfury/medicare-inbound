# app/query_rewrite.py
from __future__ import annotations
import re
from typing import Optional

# ------- Heuristic normalizer -------
_SYNONYMS = {
    r"\b(costs?|price|rate|fees?)\b": "premium",
    r"\bhiked?|increase(d|s)?|went up|raise(d|s)?\b": "premium change",
    r"\bdecrease(d|s)?|went down|lower(ed)?\b": "premium change",
    r"\badvantage\b": "part c",
    r"\borginal\b": "original",
}
_ABBREV = {
    r"\b'24\b": "2024",
    r"\b'25\b": "2025",
    r"\bpart\s*b\b": "part b",
    r"\bpart\s*a\b": "part a",
    r"\bpart\s*c\b": "part c",
    r"\bpart\s*d\b": "part d",
    r"\bfor b\b": "for part b",
}

_NOISE = r"(?:\bpls\b|\bplease\b|\bkinda\b|\bsorta\b|\bum\b|\bah\b|\buh\b)"

def _basic_clean(q: str) -> str:
    q = q.lower().strip()
    q = q.replace("’", "'").replace("“", '"').replace("”", '"')
    q = re.sub(_NOISE, " ", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()

def _apply_maps(q: str) -> str:
    for pat, repl in _ABBREV.items():
        q = re.sub(pat, repl, q)
    for pat, repl in _SYNONYMS.items():
        q = re.sub(pat, repl, q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()

def heuristic_rewrite(q: str) -> str:
    q = _basic_clean(q)
    q = _apply_maps(q)
    # Make year forms explicit when vague
    q = re.sub(r"\bthis year\b", "2025", q)
    q = re.sub(r"\blast year\b", "2024", q)
    return q

# ------- Optional short LLM rewriter -------
def llm_rewrite(q: str, *, region: str, model_id: str, max_tokens: int = 80) -> str:
    try:
        import boto3
        from botocore.config import Config
        from langchain_aws import ChatBedrock
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except Exception:
        # If libs not available, fall back silently
        return heuristic_rewrite(q)

    sys_prompt = (
        "You normalize Medicare questions for retrieval. "
        "Return a concise, literal paraphrase that expands abbreviations "
        "(e.g., '25->2025, B->Part B), replaces vague terms with precise ones, "
        "keeps key entities (Part, year), and removes chit-chat. "
        "Do NOT answer the question; only rewrite it."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "User query:\n{q}\n\nRewritten (one line):")
    ])
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=30, connect_timeout=8),
    )
    llm = ChatBedrock(model_id=model_id, client=client, model_kwargs={"max_tokens": max_tokens, "temperature": 0})
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"q": heuristic_rewrite(q)}).strip()
    except Exception:
        return heuristic_rewrite(q)

def rewrite(q: str, mode: str = "heuristic", *, region: str, model_id: str) -> str:
    """
    mode: 'off' | 'heuristic' | 'llm'
    """
    mode = (mode or "heuristic").lower()
    if mode == "off":
        return q
    if mode == "llm":
        return llm_rewrite(q, region=region, model_id=model_id)
    return heuristic_rewrite(q)

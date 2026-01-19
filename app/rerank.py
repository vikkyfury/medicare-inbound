# app/rerank.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import math
import re

def _normalize_text(t: str) -> str:
    # lower, strip punctuation-ish, collapse spaces
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _token_set(s: str) -> set:
    return set(_normalize_text(s).split())

def _jaccard(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

@dataclass
class RerankConfig:
    # Weight the original KB similarity (if provided) and lexical Jaccard
    w_kb: float = 0.6
    w_lex: float = 0.4
    # If the KB score is "distance" (smaller is better), set invert_kb=True
    invert_kb: bool = False
    # Take top_k_prime after rerank
    top_k_prime: int = 5

class Reranker:
    def __init__(self, config: RerankConfig | None = None):
        self.cfg = config or RerankConfig()

    def rerank(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Each chunk is expected like:
        {
          "text": "...",
          "metadata": {...},
          # optional numeric score from the retriever:
          "score": 0.83   # higher-is-better OR a distance (see invert_kb)
        }
        """
        if not chunks:
            return []

        lex_scores = [_jaccard(query, c.get("text", "")) for c in chunks]

        # Normalize KB scores to [0,1] (min-max) so we can blend sensibly.
        kb_raw = [c.get("score", None) for c in chunks]
        kb_present = [s for s in kb_raw if isinstance(s, (int, float))]
        kb_norm = [0.0] * len(chunks)
        if kb_present:
            smin, smax = min(kb_present), max(kb_present)
            span = (smax - smin) if (smax - smin) != 0 else 1.0
            for i, s in enumerate(kb_raw):
                if isinstance(s, (int, float)):
                    val = (s - smin) / span
                    # If score is actually a distance (smaller is better), invert
                    if self.cfg.invert_kb:
                        val = 1.0 - val
                    kb_norm[i] = max(0.0, min(1.0, val))

        blended: List[Tuple[float, int]] = []
        for i in range(len(chunks)):
            score = self.cfg.w_kb * kb_norm[i] + self.cfg.w_lex * lex_scores[i]
            blended.append((score, i))

        blended.sort(key=lambda x: x[0], reverse=True)
        keep = blended[: max(1, self.cfg.top_k_prime)]
        indices = [i for _, i in keep]
        return [chunks[i] for i in indices]

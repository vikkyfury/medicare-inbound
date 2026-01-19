# app/handler.py
from __future__ import annotations
import os
from typing import Any, Dict
# Allow runtime style override from caller
def _maybe_set_style(event: Dict[str, Any]) -> None:
    style = (event.get("style") or event.get("answer_style") or "").strip()
    if style:
        os.environ["ANSWER_STYLE"] = style  # handled by bedrock_chain config

# Import after potential env updates
from .bedrock_chain import generate_answer  # noqa: E402

def lambda_handler(event, context=None):
    """
    Input:
      {
        "text": "question here",
        "filters": {"year": "2025", "part": "B", "topic": "costs", "source": "md"},
        "style": "csr_short"   // optional: default|csr_short|medicare_compliance
      }

    Output:
      { "answer": str, "citations": [..], "chunk_count": int }
    """
    if not isinstance(event, dict):
        return {"answer": "Invalid request.", "citations": [], "chunk_count": 0}

    _maybe_set_style(event)

    q = (event.get("text") or "").strip()
    flt = (event.get("filters") or {}) if isinstance(event.get("filters"), dict) else {}
    out = generate_answer(
        q,
        year=flt.get("year"),
        topic=flt.get("topic"),
        source=flt.get("source"),
        part=flt.get("part"),
    )
    # keep contract compact for Lex/Connect
    return {
        "answer": out.get("answer", ""),
        "citations": out.get("citations", []),
        "chunk_count": out.get("chunk_count", 0),
    }

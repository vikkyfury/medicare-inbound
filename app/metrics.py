# app/metrics.py
from __future__ import annotations
import csv, os, time, hashlib
from typing import Dict, Any

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
DEFAULT_FILE = os.path.join(DEFAULT_DIR, "metrics.csv")

_HEADERS = [
    "ts_iso",               # ISO-8601 timestamp (local)
    "ts_unix",              # epoch seconds (float)
    "qid",                  # short hash of question
    "question_chars",       # len(question)
    "rewritten_chars",      # len(rewritten)
    "rewrite_mode",         # off | heuristic | llm
    "answer_style",         # default | csr_short | medicare_compliance
    "model_id",             # LLM model id
    "retrieve_ms",          # retrieval latency in ms
    "llm_ms",               # LLM latency in ms
    "total_ms",             # end-to-end in ms
    "chunk_count",          # number of chunks used
    "no_hit",               # 1 if no chunks / fallback, else 0
    "throttle_retries",     # how many retry loops hit throttling
    "region",               # AWS region
]

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _ensure_header(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(_HEADERS)

def _short_qid(text: str) -> str:
    if not text:
        return ""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return h[:10]

def log_metrics(row: Dict[str, Any], *, path: str = DEFAULT_FILE) -> None:
    """
    Append a row to metrics.csv. Any missing columns default to empty.
    Extra keys are ignored.
    """
    _ensure_dir(path)
    _ensure_header(path)

    # Fixed columns ordering
    out = [row.get(k, "") for k in _HEADERS]
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(out)

def build_row(
    *,
    question: str,
    rewritten: str,
    rewrite_mode: str,
    answer_style: str,
    model_id: str,
    retrieve_ms: float,
    llm_ms: float,
    total_ms: float,
    chunk_count: int,
    no_hit: bool,
    throttle_retries: int,
    region: str,
) -> Dict[str, Any]:
    # Timestamps
    ts_unix = time.time()
    ts_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_unix))

    return {
        "ts_iso": ts_iso,
        "ts_unix": f"{ts_unix:.3f}",
        "qid": _short_qid(question),
        "question_chars": len(question or ""),
        "rewritten_chars": len(rewritten or ""),
        "rewrite_mode": rewrite_mode or "",
        "answer_style": answer_style or "",
        "model_id": model_id or "",
        "retrieve_ms": f"{retrieve_ms:.1f}",
        "llm_ms": f"{llm_ms:.1f}",
        "total_ms": f"{total_ms:.1f}",
        "chunk_count": int(chunk_count or 0),
        "no_hit": 1 if no_hit else 0,
        "throttle_retries": int(throttle_retries or 0),
        "region": region or "",
    }

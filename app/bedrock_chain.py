# app/bedrock_chain.py
import os, time, random
from typing import List, Tuple, Optional, Dict, Any
import re as _re
import pathlib
import hashlib

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from dotenv import load_dotenv

from .metrics import log_metrics, build_row               # NEW (3.7)
from .cache import make_key, get_cache, set_cache         # NEW (3.8)
from .redact import redact

load_dotenv()

# -------------------- Optional external helpers (with safe fallbacks) --------------------
try:
    from .query_rewrite import rewrite as rewrite_query
except Exception:
    def rewrite_query(q: str, mode: str = "heuristic", *, region: str, model_id: str) -> str:
        q = (q or "").lower().strip()
        q = q.replace("’", "'").replace("“", '"').replace("”", '"')
        q = _re.sub(r"\b'24\b", "2024", q)
        q = _re.sub(r"\b'25\b", "2025", q)
        q = _re.sub(r"\bfor b\b", "for part b", q)
        q = _re.sub(r"\s+", " ", q)
        return q

try:
    from .validators import validate_and_repair
except Exception:
    def validate_and_repair(answer: str, context: str, citations: List[str]):
        return True, answer, ""

try:
    from app.filters import build_vector_filter  # noqa: F401
except Exception:
    def build_vector_filter(**kwargs) -> None:  # type: ignore
        return None

# -------------------- Simple reranker --------------------
def _normalize_text__(_t: str) -> str:
    _t = _t.lower()
    _t = _re.sub(r"[^\w\s]", " ", _t)
    _t = _re.sub(r"\s+", " ", _t).strip()
    return _t

def _tokset__(_s: str) -> set:
    return set(_normalize_text__(_s).split()) if _s else set()

def _jaccard__(_a: str, _b: str) -> float:
    A, B = _tokset__(_a), _tokset__(_b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def _blend_and_top__(query: str, items: List[Dict[str, Any]], top_k_prime: int = 5, invert_kb: bool = False):
    if not items:
        return [], 0.0, 0.0

    kb_vals = []
    for it in items:
        s = it.get("score", it.get("retrievalScore"))
        kb_vals.append(s if isinstance(s, (int, float)) else None)

    present = [s for s in kb_vals if isinstance(s, (int, float))]
    kb_norm = [0.0] * len(items)
    if present:
        smin, smax = min(present), max(present)
        span = (smax - smin) if (smax - smin) else 1.0
        for i, s in enumerate(kb_vals):
            if isinstance(s, (int, float)):
                v = (s - smin) / span
                if invert_kb:
                    v = 1.0 - v
                kb_norm[i] = max(0.0, min(1.0, v))

    lex = [_jaccard__(query, it.get("text") or "") for it in items]

    w_kb, w_lex = 0.6, 0.4
    blended = [(w_kb * kb_norm[i] + w_lex * lex[i], lex[i], i) for i in range(len(items))]
    blended.sort(key=lambda x: x[0], reverse=True)

    keep = []
    for score, lex_s, i in blended[:max(1, top_k_prime)]:
        it = dict(items[i])
        it["_blend"] = score
        it["_lex"] = lex_s
        keep.append(it)

    top_blend = blended[0][0]
    top_lex = blended[0][1]
    return keep, top_blend, top_lex

# -------------------- Config --------------------
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

def _load_cfg() -> dict:
    cfg = {
        "region": os.getenv("AWS_REGION", "us-east-1"),
        "knowledge_base_id": os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", ""),
        "top_k": int(os.getenv("TOP_K", "10")),
        "model_id": os.getenv("MODEL_ID", "anthropic.claude-3-5-sonnet-20240620"),
        "temperature": float(os.getenv("TEMPERATURE", "0.2")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "800")),
        "rewrite_mode": os.getenv("REWRITE_MODE", "heuristic"),      # off | heuristic | llm
        "answer_style": os.getenv("ANSWER_STYLE", "default"),        # default | csr_short | medicare_compliance
        # Caching knobs
        "cache_retrieval_ttl_s": float(os.getenv("CACHE_RETRIEVAL_TTL_S", "86400")),  # 24h
        "cache_llm_ttl_s": float(os.getenv("CACHE_LLM_TTL_S", "172800")),            # 48h
        "cache_enabled": os.getenv("CACHE_ENABLED", "1") not in ("0", "false", "False"),
    }
    p = os.path.join(os.path.dirname(__file__), "..", "configs", "bedrock.yaml")
    if yaml and os.path.exists(p):
        with open(p, "r") as f:
            y = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in y.items() if v not in (None, "")})
    return cfg

CFG = _load_cfg()
BEDROCK_REGION = CFG["region"]
KNOWLEDGE_BASE_ID = CFG["knowledge_base_id"]

# -------------------- AWS clients --------------------
agent_rt = boto3.client("bedrock-agent-runtime", region_name=BEDROCK_REGION)
bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=Config(
        retries={"max_attempts": 12, "mode": "adaptive"},
        read_timeout=60,
        connect_timeout=10,
    ),
)

# -------------------- Prompt packs --------------------
def _load_system_prompt(style: str) -> str:
    style = (style or "default").strip().lower()
    base = pathlib.Path(__file__).parent / "prompts"
    fname = {
        "default": "default.md",
        "csr_short": "csr_short.md",
        "medicare_compliance": "medicare_compliance.md",
    }.get(style, "default.md")
    p = base / fname
    if p.exists():
        return p.read_text(encoding="utf-8")
    return """You are a Medicare benefits assistant that answers ONLY using the provided context (retrieved KB chunks).
Keep answers concise. If info is missing, say so. Do not use facts outside the context."""

SYSTEM_PROMPT = _load_system_prompt(CFG.get("answer_style", "default"))

OUTPUT_TEMPLATE = """Follow this output structure:

<short answer line>

**Summary**
• ...

**Edge cases**
• ...

(Optional) brief paragraph

Every statement must be supported by the provided context. If insufficient, say so.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context (from KB):\n{context}\n\nUser question:\n{question}\n\n" + OUTPUT_TEMPLATE)
])

llm = ChatBedrock(
    model_id=CFG["model_id"],
    client=bedrock_runtime,
    model_kwargs={"temperature": CFG["temperature"], "max_tokens": CFG["max_tokens"]},
)

chain = prompt | llm | StrOutputParser()

# -------------------- Client-side filter helpers --------------------
def _match_any(s: str, needles: List[str]) -> bool:
    s = (s or "").lower()
    return any(n and n in s for n in needles)

def _client_passes_filters(
    text: str,
    path: str,
    *,
    year: Optional[int | str],
    topic: Optional[str],
    source: Optional[str],
    part: Optional[str],
) -> bool:
    t = (text or "").lower()
    p = (path or "").lower()

    if year:
        y = str(year).strip()
        if not (y and (y in t or y in p)):
            return False

    if part:
        pv = str(part).strip().lower()
        candidates = {pv, f"part {pv}", f"part-{pv}"}
        if len(pv) == 1 and pv in "abcd":
            candidates |= {f"part {pv}", f"part-{pv}"}
        if not _match_any(t, list(candidates)) and not _match_any(p, list(candidates)):
            return False

    if topic:
        tok = str(topic).strip().lower()
        if not _match_any(t, [tok]) and not _match_any(p, [tok]):
            return False

    if source:
        src = str(source).strip().lower()
        if not _match_any(t, [src]) and not _match_any(p, [src]):
            return False

    return True

# -------------------- Retrieval --------------------
def _context_digest(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

def retrieve_chunks(
    query: str,
    top_k: Optional[int] = None,
    *,
    year: Optional[int | str] = None,
    topic: Optional[str] = None,
    source: Optional[str] = None,
    part: Optional[str] = None,
    debug_filter: bool = False,
) -> Tuple[str, List[str]]:
    """
    Fetch K (default 10) from KB (no server-side filters), apply client-side filters,
    rerank to K'=5, and return (context, citations). Empty => allow no-hit fallback.

    Applies a retrieval cache keyed on (KB id, rewritten query, filters, top_k).
    """
    top_k = int(top_k or 10)

    # ------------- Retrieval cache (3.8) -------------
    if CFG["cache_enabled"]:
        ret_key = make_key("retrieval", {
            "kb": KNOWLEDGE_BASE_ID,
            "q": query,
            "year": str(year) if year is not None else None,
            "topic": topic,
            "source": source,
            "part": part,
            "k": top_k,
        })
        cached = get_cache(ret_key, ttl_sec=CFG["cache_retrieval_ttl_s"])
        if cached and "context" in cached and "citations" in cached:
            return cached["context"], cached["citations"]

    def _call():
        params: Dict[str, Any] = {
            "knowledgeBaseId": KNOWLEDGE_BASE_ID,
            "retrievalQuery": {"text": query},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                }
            },
        }
        return agent_rt.retrieve(**params)

    resp = _call()
    raw_items = resp.get("retrievalResults", []) or []

    items: List[Dict[str, Any]] = []
    for it in raw_items:
        content = it.get("content") or {}
        text = content.get("text") or ""
        meta = it.get("metadata") or {}

        loc = it.get("location") or {}
        s3 = (loc.get("s3Location") or {}) if isinstance(loc, dict) else {}
        bucket = s3.get("bucket")
        key = s3.get("key")
        path = (
            f"s3://{bucket}/{key}"
            if bucket and key
            else meta.get("s3Uri")
            or meta.get("s3_url")
            or meta.get("source")
            or meta.get("path")
            or "s3://(unknown)"
        )
        page = meta.get("page") or meta.get("pageNumber") or meta.get("PageNumber") or ""

        items.append({
            "text": text,
            "metadata": meta,
            "score": it.get("score", it.get("retrievalScore")),
            "path": path,
            "page": page,
        })

    # Client-side filters
    filtered_items = []
    for it in items:
        if _client_passes_filters(
            text=it.get("text") or "",
            path=it.get("path") or "",
            year=year, topic=topic, source=source, part=part
        ):
            filtered_items.append(it)

    items_for_rerank = filtered_items if filtered_items else items

    kept, top_blend, top_lex = _blend_and_top__(query, items_for_rerank, top_k_prime=5, invert_kb=False)

    # Relevance gate (slightly relaxed for year mention)
    years_in_q = set(m.group(0) for m in _re.finditer(r"\b(19|20)\d{2}\b", query))
    ctx_has_asked_year = False
    for it in kept:
        t = it.get("text") or ""
        if any(y in t for y in years_in_q):
            ctx_has_asked_year = True
            break

    blend_thresh = 0.12 if ctx_has_asked_year else 0.15
    lex_thresh = 0.08 if ctx_has_asked_year else 0.10
    if not kept or (top_blend < blend_thresh and top_lex < lex_thresh):
        return "", []

    # Build context & citations
    chunks: List[str] = []
    citations: List[str] = []
    for i, it in enumerate(kept, 1):
        t = it.get("text") or ""
        if not t:
            continue
        label = f"Source Chunk {i}: {it.get('path')}"
        if it.get("page"):
            label += f" (page {it.get('page')})"
        chunks.append(t)
        citations.append(label)

    context = "\n\n---\n\n".join(chunks) if chunks else ""

    # Store retrieval cache
    if CFG["cache_enabled"]:
        set_cache(ret_key, {"context": context, "citations": citations})

    return context, citations

# -------------------- Generation (with caching + metrics + guardrails) --------------------
def generate_answer(
    question: str,
    *,
    year: Optional[int | str] = None,
    topic: Optional[str] = None,
    source: Optional[str] = None,
    part: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end: rewrite -> retrieve (client-side filters) -> rerank -> prompt LLM
    Caching:
      - Retrieval cache on (KB id, rewritten, filters, top_k)
      - LLM cache on (context hash, original question, style, model)
    Logs metrics to logs/metrics.csv.
    """
    t0 = time.time()
    throttle_retries = 0

    # Query rewrite
    rewritten = rewrite_query(
        question,
        mode=CFG.get("rewrite_mode", "heuristic"),
        region=BEDROCK_REGION,
        model_id=CFG["model_id"],
    )

    # Retrieval
    tr0 = time.time()
    context, citations = retrieve_chunks(
        rewritten, top_k=CFG["top_k"], year=year, topic=topic, source=source, part=part
    )
    retrieve_ms = (time.time() - tr0) * 1000.0

    # No-hit fallback
    if not citations or not context.strip():
        total_ms = (time.time() - t0) * 1000.0
        log_metrics(build_row(
            question=question,
            rewritten=rewritten,
            rewrite_mode=CFG.get("rewrite_mode", "heuristic"),
            answer_style=CFG.get("answer_style", "default"),
            model_id=CFG["model_id"],
            retrieve_ms=retrieve_ms,
            llm_ms=0.0,
            total_ms=total_ms,
            chunk_count=0,
            no_hit=True,
            throttle_retries=0,
            region=BEDROCK_REGION,
        ))
        return {
            "answer": "KB has no info on this. Try rephrasing or check year/topic.",
            "citations": [],
            "chunk_count": 0,
            "rewritten": rewritten,
        }

    # ---------------- LLM cache (3.8) ----------------
    ctx_hash = _context_digest(context)
    if CFG["cache_enabled"]:
        llm_key = make_key("llm", {
            "model": CFG["model_id"],
            "style": CFG["answer_style"],
            "question": question,           # keep original user wording
            "ctx": ctx_hash,                # ensure different contexts produce different answers
            "template_v": 1,                # bump if you change OUTPUT_TEMPLATE materially
        })
        hit = get_cache(llm_key, ttl_sec=CFG["cache_llm_ttl_s"])
        if hit and "answer" in hit:
            # Cache hit: log metrics quickly and return
            total_ms = (time.time() - t0) * 1000.0
            log_metrics(build_row(
                question=question,
                rewritten=rewritten,
                rewrite_mode=CFG.get("rewrite_mode", "heuristic"),
                answer_style=CFG.get("answer_style", "default"),
                model_id=CFG["model_id"],
                retrieve_ms=retrieve_ms,
                llm_ms=0.0,
                total_ms=total_ms,
                chunk_count=len(citations),
                no_hit=False,
                throttle_retries=0,
                region=BEDROCK_REGION,
            ))
            return {
                "answer": hit["answer"],
                "citations": citations,
                "chunk_count": len(citations),
                "rewritten": rewritten,
            }

    question_red = redact(question)
    context_red = redact(context)

    payload = {"question": question_red, "context": context_red}

    # LLM invoke with retry on throttling
    tl0 = time.time()
    base = 1.8
    for attempt in range(6):
        try:
            if attempt == 0:
                time.sleep(random.uniform(0.05, 0.2))
            answer = chain.invoke(payload)
            answer = redact(answer)  # mask any PII the model might echo
            break
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("ThrottlingException", "TooManyRequestsException"):
                throttle_retries += 1
                sleep_s = min(45.0, base * (2 ** attempt))
                time.sleep(sleep_s + random.uniform(0, sleep_s * 0.3))
                continue
            raise
    else:
        # Total failure: log + safe no-hit
        llm_ms = (time.time() - tl0) * 1000.0
        total_ms = (time.time() - t0) * 1000.0
        log_metrics(build_row(
            question=question,
            rewritten=rewritten,
            rewrite_mode=CFG.get("rewrite_mode", "heuristic"),
            answer_style=CFG.get("answer_style", "default"),
            model_id=CFG["model_id"],
            retrieve_ms=retrieve_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            chunk_count=0,
            no_hit=True,
            throttle_retries=throttle_retries,
            region=BEDROCK_REGION,
        ))
        return {
            "answer": "KB has no info on this. Try rephrasing or check year/topic.",
            "citations": [],
            "chunk_count": 0,
            "rewritten": rewritten,
        }

    llm_ms = (time.time() - tl0) * 1000.0

    # Guardrails
    ok, fixed, reason = validate_and_repair(answer, context, citations)
    if not ok:
        total_ms = (time.time() - t0) * 1000.0
        log_metrics(build_row(
            question=question,
            rewritten=rewritten,
            rewrite_mode=CFG.get("rewrite_mode", "heuristic"),
            answer_style=CFG.get("answer_style", "default"),
            model_id=CFG["model_id"],
            retrieve_ms=retrieve_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            chunk_count=0,
            no_hit=True,
            throttle_retries=throttle_retries,
            region=BEDROCK_REGION,
        ))
        return {
            "answer": "KB has no info on this. Try rephrasing or check year/topic.",
            "citations": [],
            "chunk_count": 0,
            "rewritten": rewritten,
        }
    answer = fixed

    # Store LLM cache
    if CFG["cache_enabled"]:
        set_cache(llm_key, {"answer": answer})

    # Success: log metrics
    total_ms = (time.time() - t0) * 1000.0
    log_metrics(build_row(
        question=question,
        rewritten=rewritten,
        rewrite_mode=CFG.get("rewrite_mode", "heuristic"),
        answer_style=CFG.get("answer_style", "default"),
        model_id=CFG["model_id"],
        retrieve_ms=retrieve_ms,
        llm_ms=llm_ms,
        total_ms=total_ms,
        chunk_count=len(citations),
        no_hit=False,
        throttle_retries=throttle_retries,
        region=BEDROCK_REGION,
    ))

    return {
        "answer": answer,
        "citations": citations,
        "chunk_count": len(citations),
        "rewritten": rewritten,
    }

__all__ = ["generate_answer", "retrieve_chunks", "SYSTEM_PROMPT", "OUTPUT_TEMPLATE"]

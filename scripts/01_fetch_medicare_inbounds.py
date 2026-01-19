#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 — Dataset Preparation (Medicare inbound only)

What this script does
---------------------
1) Downloads a local snapshot of the Hugging Face dataset
   "AIxBlock/92k-real-world-call-center-scripts-english" into ./data/_hf_snapshot

2) Locates *only* the ZIP named like "medicare_inbounds.zip" (case-insensitive).
   - If multiple matches exist, processes all of them (rare).
   - Uses a safe ZIP extraction (prevents ZipSlip).

3) Normalizes every JSON/JSONL conversation file inside the ZIP into a uniform schema:
   {
     "source_path": "<path in zip>",
     "conversation_id": "call_<stable_hash>",
     "utterances": [
       {"speaker": "agent"|"customer"|"unknown", "text": "..."},
       ...
     ],
     "meta": {
       "asr_confidence_overall": <float|null>,
       "duration_sec": <int|null>
     }
   }

4) Writes:
   - ./data/medicare_inbounds/medicare_conversations.jsonl
   - ./data/medicare_inbounds/_sample_preview.json  (first ~5)
   - ./data/medicare_inbounds/_file_index.csv       (zip member → staged copy)

Run
---
$ cd medicare-inbound
$ python3 scripts/01_fetch_medicare_inbounds.py

Requires
--------
pip install huggingface_hub tqdm orjson
"""

import csv
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from huggingface_hub import snapshot_download
from tqdm import tqdm

# Optional faster JSON
try:
    import orjson as _json
    def jloads(b): return _json.loads(b)
    def jdumps(o): return _json.dumps(o)
except Exception:
    def jloads(b): return json.loads(b)
    def jdumps(o): return json.dumps(o, ensure_ascii=False).encode("utf-8")


# ----------------------------- Config ---------------------------------

DATASET_ID = "AIxBlock/92k-real-world-call-center-scripts-english"

ROOT              = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR      = ROOT / "data" / "_hf_snapshot"
OUT_DIR           = ROOT / "data" / "medicare_inbounds"
STAGED_DIR        = OUT_DIR / "_raw_medicare_json"

NORMALIZED_JSONL  = OUT_DIR / "medicare_conversations.jsonl"
PREVIEW_JSON      = OUT_DIR / "_sample_preview.json"
INDEX_CSV         = OUT_DIR / "_file_index.csv"
ERROR_LOG         = OUT_DIR / "_normalization_errors.log"

# Target ZIP name pattern: "medicare_inbounds.zip" (be forgiving on underscores/dashes/plurals)
ZIP_NAME_RE = re.compile(r"medicare[_\- ]?inbounds?\.zip$", re.IGNORECASE)

# Only parse JSON/JSONL
ALLOWED_EXTS = {".json", ".jsonl"}


# ------------------------ Helpers: FS + ZIP safety --------------------

def is_json_like(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXTS

def iter_snapshot_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p

def safe_extract_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, target_root: Path) -> Optional[Path]:
    """
    Safely extract a single member to target_root (prevents ZipSlip).
    Returns the extracted path or None if skipped.
    """
    if member.is_dir():
        return None

    if not is_json_like(member.filename):
        return None

    # Normalize and prevent traversal
    member_path = Path(member.filename)
    normalized = Path(*member_path.parts)
    dest_path = (target_root / normalized).resolve()
    if not str(dest_path).startswith(str(target_root.resolve())):
        return None

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(dest_path, "wb") as dst:
        dst.write(src.read())
    return dest_path


# ---------------------- Heuristic transcript parsing ------------------

AGENT_HINTS = re.compile(r"\b(agent|rep|advocate|csr)\b", re.I)
CUST_HINTS  = re.compile(r"\b(cust|customer|caller|member|patient|user)\b", re.I)

def _guess_role_from_key(k: str) -> Optional[str]:
    k_lower = k.lower()
    if "agent" in k_lower or AGENT_HINTS.search(k_lower):
        return "agent"
    if "customer" in k_lower or "caller" in k_lower or CUST_HINTS.search(k_lower):
        return "customer"
    return None

def _normalize_text(x: Any) -> Optional[str]:
    if isinstance(x, str):
        # Remove simple [mm:ss] timestamps etc.
        return re.sub(r"\s*\[\d{1,2}:\d{2}(:\d{2})?\]\s*", " ", x).strip()
    return None

def extract_utterances_generic(obj: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Defensive extraction across inconsistent schemas.
    Tries common patterns:
      - obj["transcript"] / "dialogue" / "conversation" / "utterances" / "turns"
        list of {speaker, text} or {role, content}
      - fall back to obj["text"] / "full_text"
    """
    utterances: List[Dict[str, str]] = []

    def add_turn(role: Optional[str], text: str):
        role_norm = role if role in {"agent", "customer"} else "unknown"
        text = text.strip()
        if text:
            utterances.append({"speaker": role_norm, "text": text})

    # Container list
    container = None
    for key in ("transcript", "dialogue", "conversation", "utterances", "turns"):
        if key in obj and isinstance(obj[key], list):
            container = obj[key]
            break

    if container is not None:
        for item in container:
            if isinstance(item, dict):
                role = None
                text = None
                for k, v in item.items():
                    if text is None and isinstance(v, str) and k.lower() in {"text", "content", "utterance", "value"}:
                        text = _normalize_text(v)
                    if role is None and isinstance(v, str) and k.lower() in {"speaker", "role", "from", "who"}:
                        role = v.lower()
                if role is None:
                    for k in item.keys():
                        role = _guess_role_from_key(k)
                        if role:
                            break
                if text:
                    add_turn(role, text)
            elif isinstance(item, str):
                add_turn(None, _normalize_text(item) or "")
    else:
        # Fallback: single block text
        for k in ("text", "full_text", "conversation_text"):
            if k in obj and isinstance(obj[k], str):
                txt = _normalize_text(obj[k])
                if txt:
                    add_turn(None, txt)
                    break

    # Meta fields (best-effort)
    meta: Dict[str, Any] = {}
    for k in ("overall_confidence", "confidence", "asr_confidence", "overall_confidence_score"):
        v = obj.get(k)
        if isinstance(v, (int, float)):
            meta["asr_confidence_overall"] = float(v); break
    if "asr_confidence_overall" not in meta:
        meta["asr_confidence_overall"] = None

    for k in ("duration", "duration_sec", "audio_duration", "duration_seconds"):
        v = obj.get(k)
        if isinstance(v, (int, float)):
            meta["duration_sec"] = int(v); break
    if "duration_sec" not in meta:
        meta["duration_sec"] = None

    return utterances, meta

def derive_conversation_id(source_rel_path: str) -> str:
    # Stable ID from relative path
    h = hashlib.sha1(source_rel_path.encode("utf-8")).hexdigest()
    return f"call_{h[:16]}"


# ----------------------------- Main -----------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Downloading dataset snapshot: {DATASET_ID}")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(SNAPSHOT_DIR),
        local_dir_use_symlinks=False,
    )
    snapshot_root = Path(local_path)

    print(f"[2/4] Searching for medicare_inbounds.zip in: {snapshot_root}")
    zip_files: List[Path] = []
    for p in iter_snapshot_files(snapshot_root):
        name = p.name
        if ZIP_NAME_RE.search(name):
            zip_files.append(p)

    if not zip_files:
        raise FileNotFoundError(
            "Could not find 'medicare_inbounds.zip' in the dataset snapshot. "
            "Check the dataset layout or update ZIP_NAME_RE."
        )

    # Extract JSON members from each matching ZIP
    index_rows: List[Tuple[str, str]] = []
    total_extracted = 0
    for zpath in zip_files:
        rel_zip = zpath.relative_to(snapshot_root)
        print(f"  - Found: {rel_zip}")
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                for member in zf.infolist():
                    out_path = safe_extract_member(zf, member, STAGED_DIR)
                    if out_path:
                        staged_rel = out_path.relative_to(STAGED_DIR)
                        index_rows.append((f"{rel_zip}::{member.filename}", str(staged_rel)))
                        total_extracted += 1
        except zipfile.BadZipFile:
            print(f"[WARN] Bad zip skipped: {rel_zip}")

    print(f"[2/4] Extracted {total_extracted} JSON files from {len(zip_files)} ZIP(s).")

    # Normalize staged files → JSONL
    print(f"[3/4] Normalizing to: {NORMALIZED_JSONL}")
    n_ok, n_bad = 0, 0
    preview: List[Dict[str, Any]] = []
    ERROR_LOG.unlink(missing_ok=True)

    with open(NORMALIZED_JSONL, "wb") as out_f:
        for staged in tqdm(list(STAGED_DIR.rglob("*")), desc="Normalize", unit="file"):
            if staged.suffix.lower() not in ALLOWED_EXTS or not staged.is_file():
                continue
            rel_in = staged.relative_to(STAGED_DIR)
            try:
                raw = staged.read_bytes()
                obj = jloads(raw)
                if isinstance(obj, list):
                    # Sometimes a list of turns directly.
                    obj = {"transcript": obj}

                utterances, meta = extract_utterances_generic(obj)
                if not utterances:
                    raise ValueError("No utterances extracted")

                record = {
                    "source_path": str(rel_in),
                    "conversation_id": derive_conversation_id(str(rel_in)),
                    "utterances": utterances,
                    "meta": meta,
                }

                out_f.write(jdumps(record))
                out_f.write(b"\n")
                n_ok += 1
                if len(preview) < 5:
                    preview.append(record)

            except Exception as e:
                n_bad += 1
                with open(ERROR_LOG, "a", encoding="utf-8") as ef:
                    ef.write(f"{rel_in} :: {type(e).__name__}: {e}\n")

    # Index + preview
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zip_member", "staged_relative"])
        w.writerows(index_rows)

    with open(PREVIEW_JSON, "wb") as f:
        f.write(jdumps(preview))

    print(f"[4/4] DONE. Normalized={n_ok}, Failed={n_bad}")
    print(f"Outputs:")
    print(f" - {NORMALIZED_JSONL}")
    print(f" - {PREVIEW_JSON}")
    print(f" - {INDEX_CSV}")
    if n_bad:
        print(f"[NOTE] Some files failed normalization; see {ERROR_LOG}")

if __name__ == "__main__":
    main()

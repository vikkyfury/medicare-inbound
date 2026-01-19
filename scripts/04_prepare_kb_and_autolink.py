#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 — Prepare local Medicare KB and auto-link to intents.

What it does
------------
1) Reads input files:
   - configs/dataset.yaml
   - configs/intent_policy.yaml (kb_answerable vs handoff_required)
   - reports/medicare_cluster_summary.csv (from your 03 script)
   - reports/medicare_only_intent_counts.csv (from 02b)
   - data/intents/medicare_clusters.csv (utterance → cluster_id; optional but used if present)

2) Converts docs in data/kb/raw/ (PDF, HTML, DOCX, TXT) into Markdown → data/kb/clean/
   - Minimal dependencies; uses pdfminer.six, html2text, python-docx if available
   - Gracefully skips formats without extractor installed

3) Chunks Markdown into sections using headings (#, ##, ###) and creates:
   - reports/kb_index.csv  (section_id, file_relpath, heading, heading_level, char_count, snippet)

4) Builds per-intent query text from:
   - intent name,
   - top cluster phrases (from cluster_summary if present),
   - top seed keywords (from intent name heuristics),
   - frequent words from customer utterances in that cluster (optional if medicaid_clusters present)

5) TF-IDF ranks top K sections per intent and writes:
   - reports/kb_links.csv (intent, section_id, score, file, heading, snippet_start)

6) Auto-drafts a stub answer for KB-answerable intents:
   - eval/qna/gold_autodraft.csv with columns:
     [intent, question_stub, draft_answer_text, source_sections]
   - Handoff intents get a row with draft_answer_text="", and a note "(handoff)"

Install (one-time)
------------------
pip install pdfminer.six html2text python-docx scikit-learn

Run
---
$ cd medicare-inbound
$ python3 scripts/04_prepare_kb_and_autolink.py --top-sections 5

Then open:
- reports/kb_links.csv
- eval/qna/gold_autodraft.csv

Notes
-----
- Put official Medicare docs (FAQs, coverage pages, plan documents) into data/kb/raw/ first.
- This script does no web fetching; it only processes local files.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import io
import os
import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import yaml

# Optional extractors
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

try:
    import html2text as _html2text
    _HTML2MD = _html2text.HTML2Text()
    _HTML2MD.ignore_links = False
    _HTML2MD.ignore_images = True
    _HTML2MD.body_width = 0
except Exception:
    _html2text = None
    _HTML2MD = None

try:
    import docx
except Exception:
    docx = None

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DATASET = ROOT / "configs" / "dataset.yaml"
CONFIG_POLICY  = ROOT / "configs" / "intent_policy.yaml"

KB_RAW_DIR   = ROOT / "data" / "kb" / "raw"
KB_CLEAN_DIR = ROOT / "data" / "kb" / "clean"
REPORTS_DIR  = ROOT / "reports"
EVAL_QNA_DIR = ROOT / "eval" / "qna"
INTENTS_DIR  = ROOT / "data" / "intents"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_QNA_DIR.mkdir(parents=True, exist_ok=True)
KB_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_SUMMARY_CSV     = REPORTS_DIR / "medicare_cluster_summary.csv"
INTENT_COUNTS_CSV       = REPORTS_DIR / "medicare_only_intent_counts.csv"
MEDICARE_CLUSTERS_CSV   = INTENTS_DIR / "medicare_clusters.csv"  # optional (utterance->cluster)
KB_INDEX_CSV            = REPORTS_DIR / "kb_index.csv"
KB_LINKS_CSV            = REPORTS_DIR / "kb_links.csv"
GOLD_AUTODRAFT_CSV      = EVAL_QNA_DIR / "gold_autodraft.csv"


# ------------ Utilities ------------

def read_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def head_snippet(s: str, n: int = 220) -> str:
    s = norm_space(s)
    return s[:n]

def read_csv_kv(p: Path, key: str, val: str) -> Dict[str, str]:
    out = {}
    if not p.exists(): return out
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row[key]] = row[val]
    return out

def read_cluster_summary(p: Path) -> List[Dict[str, str]]:
    rows = []
    if not p.exists(): return rows
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def tokenize_heading_blocks(md_text: str) -> List[Tuple[str, int, str]]:
    """
    Splits markdown into sections on headings (#, ##, ###, ####).
    Returns list of (heading_text, level, section_body).
    If no headings, returns a single pseudo-section ("", 0, whole_text).
    """
    lines = md_text.splitlines()
    sections = []
    curr_head = ""
    curr_level = 0
    buf = []

    def flush():
        if buf or curr_head:
            sections.append((curr_head, curr_level, "\n".join(buf).strip()))
        buf.clear()

    for ln in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", ln.strip())
        if m:
            # New heading
            flush()
            curr_level = len(m.group(1))
            curr_head = m.group(2).strip()
        else:
            buf.append(ln)
    flush()

    if not sections:
        return [("", 0, md_text)]
    return sections

# ------------ Extractors ------------

def extract_text_from_file(p: Path) -> Optional[str]:
    suf = p.suffix.lower()
    try:
        if suf == ".txt":
            return p.read_text(encoding="utf-8", errors="ignore")
        if suf in (".md", ".markdown"):
            return p.read_text(encoding="utf-8", errors="ignore")
        if suf == ".pdf":
            if pdf_extract_text is None:
                print(f"[WARN] pdfminer.six not installed, skipping: {p}")
                return None
            return pdf_extract_text(str(p))
        if suf in (".html", ".htm"):
            if _HTML2MD is None:
                print(f"[WARN] html2text not installed, skipping: {p}")
                return None
            html = p.read_text(encoding="utf-8", errors="ignore")
            return _HTML2MD.handle(html)
        if suf in (".docx",):
            if docx is None:
                print(f"[WARN] python-docx not installed, skipping: {p}")
                return None
            d = docx.Document(str(p))
            return "\n".join([para.text for para in d.paragraphs])
    except Exception as e:
        print(f"[WARN] Failed to extract {p}: {e}")
        return None
    # Unknown type
    print(f"[WARN] Unsupported file type: {p.name}")
    return None

def convert_kb_to_clean_markdown():
    converted = 0
    for p in KB_RAW_DIR.rglob("*"):
        if not p.is_file(): continue
        txt = extract_text_from_file(p)
        if txt is None:
            continue
        # Normalize to md
        md = txt.replace("\r\n", "\n").strip()
        rel = p.relative_to(KB_RAW_DIR)
        outp = KB_CLEAN_DIR / rel.with_suffix(".md")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(md, encoding="utf-8")
        converted += 1
    print(f"[1/5] Converted KB files → Markdown: {converted} file(s)")

# ------------ Build section index ------------

@dataclass
class KBSection:
    section_id: str
    file_relpath: str
    heading: str
    level: int
    body: str

def build_kb_index() -> List[KBSection]:
    sections: List[KBSection] = []
    for p in KB_CLEAN_DIR.rglob("*.md"):
        md = p.read_text(encoding="utf-8", errors="ignore")
        blocks = tokenize_heading_blocks(md)
        rel = str(p.relative_to(KB_CLEAN_DIR))
        for head, lvl, body in blocks:
            content = (head + "\n" + body).strip()
            sid = f"kb_{sha1(rel + '|' + head + '|' + head_snippet(body, 64))}"
            sections.append(KBSection(
                section_id=sid,
                file_relpath=rel,
                heading=head,
                level=lvl,
                body=content
            ))
    # write index csv
    with open(KB_INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section_id", "file_relpath", "heading", "heading_level", "char_count", "snippet"])
        for s in sections:
            w.writerow([s.section_id, s.file_relpath, s.heading, s.level, len(s.body), head_snippet(s.body, 220)])
    print(f"[2/5] KB index → {KB_INDEX_CSV} ({len(sections)} sections)")
    return sections

# ------------ Intent query builder ------------

INTENT_KEYWORD_EXPAND = {
    "rx_part_d": ["Part D", "drug coverage", "formulary", "pharmacy", "prescription", "prior auth", "tier", "exceptions"],
    "benefits_coverage": ["what's covered", "coverage rules", "preventive services", "benefits"],
    "premiums_billing": ["premium", "late payment", "Part B premium", "IRMAA", "bill", "autopay"],
    "claims_status": ["claim", "EOB", "denied", "appeal", "reimbursement"],
    "copay_coins_deduct": ["copay", "coinsurance", "deductible", "out-of-pocket maximum"],
    "pcp_network": ["in network", "out of network", "primary care", "referrals"],
    "prior_auth": ["prior authorization", "PA required", "medical necessity"],
    "eligibility": ["who can enroll", "eligibility", "age 65", "disability", "ESRD"],
    "enrollment": ["Open Enrollment", "AEP", "OEP", "SEP", "switch plan", "disenroll"],
    "appeals_grievances": ["appeal", "grievance", "complaint", "coverage decision"],
    "id_card": ["member ID card", "replacement card"],
    "vision_dental_hearing": ["dental", "vision", "hearing", "extra benefits"],
    "otc_benefit": ["OTC", "over the counter"],
    "address_contact_update": ["address change", "contact update", "mailing address"],
    "cancel_disenroll": ["cancel plan", "disenroll"],
    "appointment_referral": ["referral", "authorization", "appointment"],
}

def collect_intent_queries(cluster_summary_rows: List[Dict[str, str]], intent_counts: Dict[str, int]) -> Dict[str, str]:
    """
    Build a query for each intent by combining:
    - the intent name
    - top terms/phrases from cluster_summary (if the row maps to that intent)
    - the keyword expansion table above
    """
    queries: Dict[str, List[str]] = defaultdict(list)

    # 1) Seed with the intent name
    for intent in intent_counts.keys():
        queries[intent].append(intent.replace("_", " "))

    # 2) Add expansions
    for intent, toks in INTENT_KEYWORD_EXPAND.items():
        if intent in queries:
            queries[intent].extend(toks)

    # 3) From cluster summary (optional)
    # Expected columns from your 03 script: cluster_id, intent_guess, top_terms (comma/space separated), examples_count
    for row in cluster_summary_rows:
        ig = (row.get("intent_guess") or "").strip()
        if not ig:
            continue
        intent = ig
        if intent in queries:
            raw_terms = (row.get("top_terms") or "").replace(",", " ")
            toks = [t for t in raw_terms.split() if len(t) > 2]
            queries[intent].extend(toks)

    # Join
    return {k: norm_space(" ".join(v)) for k, v in queries.items()}

# ------------ Ranking KB sections per intent ------------

def tfidf_rank(intents_to_queries: Dict[str, str], sections: List[KBSection], top_k: int) -> List[Dict[str, str]]:
    # Build corpus = [intent_query texts ...] + [section bodies ...]
    intent_ids = list(intents_to_queries.keys())
    intent_texts = [intents_to_queries[i] for i in intent_ids]
    section_texts = [s.body for s in sections]

    vect = TfidfVectorizer(stop_words="english", max_df=0.9)
    mat = vect.fit_transform(intent_texts + section_texts)

    iq = len(intent_texts)
    intent_mat = mat[:iq]
    section_mat = mat[iq:]

    sims = cosine_similarity(intent_mat, section_mat)

    rows = []
    for i, intent in enumerate(intent_ids):
        # pick top_k sections for this intent
        sim_row = sims[i]
        top_idx = sim_row.argsort()[::-1][:top_k]
        for j in top_idx:
            s = sections[j]
            rows.append({
                "intent": intent,
                "section_id": s.section_id,
                "score": f"{float(sim_row[j]):.4f}",
                "file_relpath": s.file_relpath,
                "heading": s.heading,
                "snippet": head_snippet(s.body, 180),
            })
    return rows

# ------------ Gold autodraft builder ------------

def autodraft_gold(policy: dict,
                   kb_links_rows: List[Dict[str, str]],
                   sections_by_id: Dict[str, KBSection],
                   intents_to_queries: Dict[str, str]) -> List[Dict[str, str]]:
    kb_answerable = set(policy.get("kb_answerable", []))
    handoff_required = set(policy.get("handoff_required", []))

    rows_out = []
    by_intent: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in kb_links_rows:
        by_intent[r["intent"]].append(r)

    for intent, links in by_intent.items():
        answerable = intent in kb_answerable
        handoff = intent in handoff_required

        question_stub = intent.replace("_", " ").capitalize()
        if not links:
            rows_out.append({
                "intent": intent,
                "question_stub": question_stub,
                "draft_answer_text": "" if handoff else "[No KB sections found — add content or mark handoff]",
                "source_sections": "",
                "notes": "handoff" if handoff else "",
            })
            continue

        # Take the best section for the initial draft; include others as sources
        best = links[0]
        best_sec = sections_by_id.get(best["section_id"])
        best_text = best_sec.body if best_sec else ""

        # Build a short, factual stub — you'll polish wording later
        draft = norm_space(best_text)
        if len(draft) > 1200:
            draft = draft[:1200] + " …"

        source_ids = [l["section_id"] for l in links[:3]]
        note = "handoff" if handoff else ""
        rows_out.append({
            "intent": intent,
            "question_stub": question_stub,
            "draft_answer_text": "" if handoff else draft,
            "source_sections": ",".join(source_ids),
            "notes": note,
        })

    # Also include intents that had no links at all (if any)
    for intent in intents_to_queries.keys():
        if not any(r["intent"] == intent for r in rows_out):
            question_stub = intent.replace("_", " ").capitalize()
            note = "handoff" if intent in handoff_required else ""
            rows_out.append({
                "intent": intent,
                "question_stub": question_stub,
                "draft_answer_text": "" if note == "handoff" else "[No KB sections found — add content or mark handoff]",
                "source_sections": "",
                "notes": note,
            })
    return rows_out


# ------------ Main ------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-sections", type=int, default=5, help="Top N KB sections per intent")
    args = ap.parse_args()

    cfg = read_yaml(CONFIG_DATASET)
    policy = read_yaml(CONFIG_POLICY)

    # 1) Convert KB → Markdown
    convert_kb_to_clean_markdown()

    # 2) Build KB index
    sections = build_kb_index()
    if not sections:
        print("[ERR] No sections found. Put docs into data/kb/raw/ then re-run.", file=sys.stderr)
        sys.exit(1)
    sections_by_id = {s.section_id: s for s in sections}

    # 3) Load cluster summary + intent counts
    cluster_rows = read_cluster_summary(CLUSTER_SUMMARY_CSV)
    intent_counts = {}
    if INTENT_COUNTS_CSV.exists():
        with open(INTENT_COUNTS_CSV, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                intent = (row.get("intent") or "").strip()
                try:
                    cnt = int(row.get("count") or "0")
                except Exception:
                    cnt = 0
                if intent:
                    intent_counts[intent] = cnt
    if not intent_counts:
        print("[WARN] No intent counts — linking will proceed but coverage may be odd.")

    # 4) Build intent queries
    intent_queries = collect_intent_queries(cluster_rows, intent_counts)
    if not intent_queries:
        print("[ERR] No intents detected to link.", file=sys.stderr)
        sys.exit(1)

    # 5) Rank KB sections per intent
    kb_links = tfidf_rank(intent_queries, sections, top_k=args.top_sections)
    with open(KB_LINKS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["intent","section_id","score","file_relpath","heading","snippet"])
        w.writeheader()
        for r in kb_links:
            w.writerow(r)
    print(f"[3/5] KB links → {KB_LINKS_CSV} ({len(kb_links)} rows)")

    # 6) Autodraft gold answers (stubs)
    drafts = autodraft_gold(policy, kb_links, sections_by_id, intent_queries)
    with open(GOLD_AUTODRAFT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["intent","question_stub","draft_answer_text","source_sections","notes"])
        w.writeheader()
        for r in drafts:
            w.writerow(r)
    print(f"[4/5] Gold autodraft → {GOLD_AUTODRAFT_CSV} ({len(drafts)} intents)")

    print("[5/5] DONE. Next: review links and drafts, then we’ll finalize polished gold answers.")

if __name__ == "__main__":
    main()

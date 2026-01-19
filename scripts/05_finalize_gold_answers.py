#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5 — Convert autodrafts into polished 'gold' answers.

Inputs:
- reports/kb_links.csv            (intent, section_id, score, file_relpath, heading, snippet)
- eval/qna/gold_autodraft.csv     (intent, question_stub, draft_answer_text, source_sections, notes)
- configs/intent_policy.yaml      (kb_answerable / handoff_required)

Output:
- eval/qna/gold_answers.csv       (intent, question, answer, answer_type[gold|handoff], sources)

Run:
  $ cd medicare-inbound
  $ python3 scripts/05_finalize_gold_answers.py --min-words 90 --max-words 180
"""

from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
EVAL_QNA_DIR = ROOT / "eval" / "qna"
CONFIGS_DIR = ROOT / "configs"

KB_LINKS_CSV = REPORTS_DIR / "kb_links.csv"
AUTODRAFT_CSV = EVAL_QNA_DIR / "gold_autodraft.csv"
POLICY_YAML = CONFIGS_DIR / "intent_policy.yaml"
OUT_GOLD = EVAL_QNA_DIR / "gold_answers.csv"

# --- Helpers ---
def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}

def load_csv(p: Path) -> List[Dict[str, str]]:
    if not p.exists(): return []
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]

def detect_autodraft(rows: List[Dict[str, str]]) -> bool:
    if not rows: return False
    cols = set(rows[0].keys())
    return {"question_stub","draft_answer_text","source_sections"}.issubset(cols)

def detect_kblinks(rows: List[Dict[str, str]]) -> bool:
    if not rows: return False
    cols = set(rows[0].keys())
    return {"section_id","file_relpath","heading","snippet"}.issubset(cols)

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def trim_to_bounds(text: str, min_words: int, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    words = words[:max_words]
    t = " ".join(words)
    # end gracefully
    if not t.endswith((".", "!", "?")):
        t += "…"
    return t

PLAN_SPECIFIC_INTENTS = {"vision_dental_hearing","otc_benefit","pcp_network","prior_auth","rx_part_d","benefits_coverage","claims_status","premiums_billing"}
PLAN_SPECIFIC_NOTE = "Details can vary by plan. Check your plan’s Evidence of Coverage (EOC) or call your plan for specifics."

def polish_answer(text: str, min_words: int, max_words: int, add_plan_note: bool) -> str:
    t = norm_space(text)
    # Drop obvious boilerplate / nav detritus that sometimes leaks from pages
    t = re.sub(r"(?i)(learn more|see also|for more information).*$", "", t)
    t = trim_to_bounds(t, min_words, max_words)
    if add_plan_note:
        if not t.endswith((".", "!", "?","…")): t += "."
        t += " " + PLAN_SPECIFIC_NOTE
    return t

HANDOFF_SCRIPT = (
    "This request is best handled by a live agent. "
    "I can connect you to a specialist who can review your account and complete this request. "
    "Please have your Medicare number and any relevant documents available."
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-words", type=int, default=90)
    ap.add_argument("--max-words", type=int, default=180)
    args = ap.parse_args()

    policy = read_yaml(POLICY_YAML)
    kb_answerable = set(policy.get("kb_answerable", []))
    handoff_required = set(policy.get("handoff_required", []))

    # Load both CSVs (the user might have swapped names)
    a_rows = load_csv(AUTODRAFT_CSV)
    k_rows = load_csv(KB_LINKS_CSV)

    # Auto-detect which is which by columns
    if detect_autodraft(a_rows) and detect_kblinks(k_rows):
        autodraft_rows = a_rows
        kblinks_rows = k_rows
    elif detect_autodraft(k_rows) and detect_kblinks(a_rows):
        autodraft_rows = k_rows
        kblinks_rows = a_rows
    else:
        raise SystemExit("[ERR] Could not detect autodraft vs kb_links schema. Check the CSVs.")

    # Build a quick index of top links per intent (fallback for missing source_sections)
    links_by_intent: Dict[str, List[Dict[str, str]]] = {}
    for r in kblinks_rows:
        intent = r.get("intent","").strip()
        links_by_intent.setdefault(intent, []).append(r)
    # sort links by score desc if present
    for intent, lst in links_by_intent.items():
        try:
            lst.sort(key=lambda x: float(x.get("score","0")), reverse=True)
        except Exception:
            pass

    # Build gold answers
    out_rows = []
    seen_intents = set()
    for row in autodraft_rows:
        intent = (row.get("intent") or "").strip()
        if not intent: continue
        seen_intents.add(intent)

        question = (row.get("question_stub") or intent.replace("_"," ").capitalize()).strip()
        draft = row.get("draft_answer_text") or ""
        notes = (row.get("notes") or "").strip().lower()
        src_ids = (row.get("source_sections") or "").strip()

        is_handoff = (intent in handoff_required) or (notes == "handoff") or (not draft.strip())
        if is_handoff:
            out_rows.append({
                "intent": intent,
                "question": question,
                "answer": HANDOFF_SCRIPT,
                "answer_type": "handoff",
                "sources": src_ids,
            })
            continue

        add_plan_note = intent in PLAN_SPECIFIC_INTENTS
        polished = polish_answer(draft, args.min_words, args.max_words, add_plan_note)

        # If no sources were provided in autodraft, backfill from kb_links
        if not src_ids:
            top = links_by_intent.get(intent, [])[:3]
            src_ids = ",".join([t.get("section_id","") for t in top if t.get("section_id")])

        out_rows.append({
            "intent": intent,
            "question": question,
            "answer": polished,
            "answer_type": "gold",
            "sources": src_ids,
        })

    # Also add any intents we know about from the policy that weren’t in autodraft at all
    for intent in sorted(kb_answerable | handoff_required):
        if intent in seen_intents: continue
        question = intent.replace("_"," ").capitalize()
        answer_type = "handoff" if intent in handoff_required else "gold"
        ans = HANDOFF_SCRIPT if answer_type == "handoff" else "[Add content from KB]"
        # Backfill sources from links
        top = links_by_intent.get(intent, [])[:3]
        src_ids = ",".join([t.get("section_id","") for t in top if t.get("section_id")])
        out_rows.append({
            "intent": intent,
            "question": question,
            "answer": ans,
            "answer_type": answer_type,
            "sources": src_ids,
        })

    # Write
    OUT_GOLD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_GOLD, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["intent","question","answer","answer_type","sources"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"✅ Finalized gold answers → {OUT_GOLD} ({len(out_rows)} rows)")

if __name__ == "__main__":
    main()

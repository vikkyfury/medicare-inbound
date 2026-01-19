#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5b — QC pass on gold answers:
- Backfill missing sources from kb_links
- Expand short gold answers (<80 words) using KB snippets
- Trim long gold answers (>180 words)
- Write cleaned CSV and a brief report

Run:
  $ cd medicare-inbound
  $ python3 scripts/05b_qc_gold_answers.py --min-words 80 --max-words 180
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
EVAL_QNA_DIR = ROOT / "eval" / "qna"
OUT_CSV = EVAL_QNA_DIR / "gold_answers.cleaned.csv"
REPORT_TXT = REPORTS_DIR / "gold_answers_qc_report.txt"

KB_LINKS_CSV = REPORTS_DIR / "kb_links.csv"          # intent, section_id, score, file_relpath, heading, snippet
KB_INDEX_CSV = REPORTS_DIR / "kb_index.csv"          # section_id, file_relpath, heading, heading_level, char_count, snippet
GOLD_CSV     = EVAL_QNA_DIR / "gold_answers.csv"     # intent, question, answer, answer_type, sources

def load_csv(path: Path) -> List[Dict[str,str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(x) for x in r]

def save_csv(path: Path, rows: List[Dict[str,str]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def word_count(s: str) -> int:
    return len(norm_space(s).split()) if s else 0

def trim_to_words(text: str, max_words: int) -> str:
    words = norm_space(text).split()
    if len(words) <= max_words:
        return norm_space(text)
    trunk = " ".join(words[:max_words])
    if not trunk.endswith((".", "!", "?", "…")):
        trunk += "…"
    return trunk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-words", type=int, default=80)
    ap.add_argument("--max-words", type=int, default=180)
    args = ap.parse_args()

    gold = load_csv(GOLD_CSV)
    kblinks = load_csv(KB_LINKS_CSV)
    kbindex = load_csv(KB_INDEX_CSV)

    if not gold:
        raise SystemExit(f"[ERR] Missing {GOLD_CSV}")
    if not kblinks:
        print("[WARN] kb_links.csv not found; cannot backfill sources.")
    if not kbindex:
        print("[WARN] kb_index.csv not found; cannot expand from snippets.")

    # Build maps
    links_by_intent: Dict[str,List[Dict[str,str]]] = {}
    for r in kblinks:
        intent = (r.get("intent") or "").strip()
        links_by_intent.setdefault(intent, []).append(r)
    for lst in links_by_intent.values():
        try:
            lst.sort(key=lambda x: float(x.get("score","0")), reverse=True)
        except Exception:
            pass

    snippet_by_sid: Dict[str,str] = {}
    if kbindex:
        for r in kbindex:
            sid = r.get("section_id","")
            sn = r.get("snippet","") or ""
            snippet_by_sid[sid] = sn

    fixed_rows = []
    stats = {
        "rows": len(gold),
        "short_fixed": 0,
        "long_trimmed": 0,
        "sources_backfilled": 0,
        "handoff_untouched": 0,
    }

    for row in gold:
        intent = (row.get("intent") or "").strip()
        atype = (row.get("answer_type") or "").strip().lower()
        ans   = row.get("answer") or ""
        src   = (row.get("sources") or "").strip()

        # Only operate on gold answers
        if atype != "gold":
            stats["handoff_untouched"] += 1
            fixed_rows.append(row)
            continue

        # Backfill sources if empty
        if not src:
            top3 = links_by_intent.get(intent, [])[:3]
            sid_list = [t.get("section_id","") for t in top3 if t.get("section_id")]
            if sid_list:
                src = ",".join(sid_list)
                row["sources"] = src
                stats["sources_backfilled"] += 1

        # Expand short answers
        wc = word_count(ans)
        if wc < args.min_words:
            # Gather snippets from the current sources (up to 3)
            expanded_bits: List[str] = []
            if src:
                for sid in src.split(",")[:3]:
                    sid = sid.strip()
                    if not sid: continue
                    sn = snippet_by_sid.get(sid, "")
                    if sn:
                        expanded_bits.append(sn)
            # If still empty, fall back to top links for this intent
            if not expanded_bits:
                for r in links_by_intent.get(intent, [])[:3]:
                    sn = r.get("snippet","") or ""
                    if sn:
                        expanded_bits.append(sn)
            if expanded_bits:
                add_text = " ".join(expanded_bits)
                # Only add if it meaningfully increases length
                merged = (ans + " " + add_text).strip() if ans else add_text
                row["answer"] = norm_space(merged)
                wc = word_count(row["answer"])
                if wc < args.min_words:
                    # one more pass: duplicate first snippet to reach threshold gracefully
                    if expanded_bits:
                        merged = merged + " " + expanded_bits[0]
                        row["answer"] = norm_space(merged)
                stats["short_fixed"] += 1

        # Trim long answers
        wc = word_count(row["answer"])
        if wc > args.max_words:
            row["answer"] = trim_to_words(row["answer"], args.max_words)
            stats["long_trimmed"] += 1

        fixed_rows.append(row)

    # Save cleaned file
    fieldnames = ["intent","question","answer","answer_type","sources"]
    save_csv(OUT_CSV, fixed_rows, fieldnames)

    # Write small report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Gold Answers QC Report\n")
        f.write("======================\n")
        for k,v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nInput : {GOLD_CSV}\nOutput: {OUT_CSV}\n")

    print(f"✅ QC complete. Cleaned → {OUT_CSV}")
    print(f"Report → {REPORT_TXT}")

if __name__ == "__main__":
    main()

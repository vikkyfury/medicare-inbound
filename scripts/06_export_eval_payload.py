#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6 — Export evaluation payload.

Creates:
- eval/qna/evalset.jsonl   (records for automated testing)
- eval/qna/evalset.csv     (tabular view)
- reports/eval_payload_report.txt (stats & warnings)

If available, uses data/intents/medicare_clusters.csv to add sample utterances per intent.

Run:
  $ cd medicare-inbound
  $ python3 scripts/06_export_eval_payload.py --max-utterances 5
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
EVAL_QNA_DIR = ROOT / "eval" / "qna"
DATA_INTENTS = ROOT / "data" / "intents"
REPORTS_DIR  = ROOT / "reports"

GOLD_CLEAN   = EVAL_QNA_DIR / "gold_answers.cleaned.csv"
CLUSTERS_CSV = DATA_INTENTS / "medicare_clusters.csv"

OUT_JSONL    = EVAL_QNA_DIR / "evalset.jsonl"
OUT_CSV      = EVAL_QNA_DIR / "evalset.csv"
OUT_REPORT   = REPORTS_DIR  / "eval_payload_report.txt"

def load_csv(path: Path) -> List[Dict[str,str]]:
    if not path.exists(): return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utterances", type=int, default=5)
    args = ap.parse_args()

    gold = load_csv(GOLD_CLEAN)
    if not gold:
        raise SystemExit(f"[ERR] Missing {GOLD_CLEAN}")

    # Optional utterances from clusters: expect columns: intent, utterance (or text), cluster_id (optional)
    utt_map: Dict[str, List[str]] = {}
    cl_rows = load_csv(CLUSTERS_CSV)
    if cl_rows:
        # try to sniff column names for utterance text
        utt_col = None
        if cl_rows:
            cols = {c.lower(): c for c in cl_rows[0].keys()}
            for cand in ["utterance", "text", "customer_utterance", "customer_text"]:
                if cand in cols:
                    utt_col = cols[cand]
                    break
        if utt_col:
            for r in cl_rows:
                intent = (r.get("intent") or r.get("intent_guess") or "").strip()
                if not intent:
                    continue
                t = norm_space(r.get(utt_col) or "")
                if not t or len(t) < 8:
                    continue
                lst = utt_map.setdefault(intent, [])
                if len(lst) < args.max_utterances:
                    lst.append(t)

    # Build eval items
    items: List[Dict] = []
    stats = {
        "total": 0,
        "handoff": 0,
        "gold": 0,
        "missing_sources": 0,
        "no_utterances": 0
    }

    for r in gold:
        intent = (r.get("intent") or "").strip()
        q = (r.get("question") or intent.replace("_"," ").capitalize()).strip()
        ans = norm_space(r.get("answer") or "")
        a_type = (r.get("answer_type") or "").strip().lower()
        sources = (r.get("sources") or "").strip()

        handoff = (a_type == "handoff")
        utters = utt_map.get(intent, [])[:args.max_utterances]
        if not utters:
            stats["no_utterances"] += 1

        items.append({
            "intent": intent,
            "handoff": handoff,
            "question": q,
            "answer": ans,
            "sources": [s for s in sources.split(",") if s.strip()],
            "utterances": utters
        })

        stats["total"] += 1
        stats["handoff"] += int(handoff)
        stats["gold"] += int(not handoff)
        if (not sources) and (not handoff):
            stats["missing_sources"] += 1

    # Write JSONL
    EVAL_QNA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Write CSV mirror
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["intent","handoff","question","answer","sources","utterances"])
        w.writeheader()
        for it in items:
            w.writerow({
                "intent": it["intent"],
                "handoff": "true" if it["handoff"] else "false",
                "question": it["question"],
                "answer": it["answer"],
                "sources": "|".join(it["sources"]),
                "utterances": " || ".join(it["utterances"])
            })

    # Report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("Eval Payload Report\n")
        f.write("===================\n")
        for k,v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nJSONL : {OUT_JSONL}\nCSV   : {OUT_CSV}\n")

    print(f"✅ Wrote eval payload → {OUT_JSONL}")
    print(f"CSV view → {OUT_CSV}")
    print(f"Report → {OUT_REPORT}")

if __name__ == "__main__":
    main()

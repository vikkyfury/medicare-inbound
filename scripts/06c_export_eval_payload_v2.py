#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export eval payload with utterances + source backfill.

Inputs:
  - eval/qna/gold_answers.cleaned.csv
  - data/intents/intent_utterances.csv   (preferred) OR data/intents/medicare_clusters.csv
  - reports/kb_links.csv                 (for source backfill)
Outputs:
  - eval/qna/evalset.jsonl
  - eval/qna/evalset.csv
  - reports/eval_payload_report.txt
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
UTTS_CSV     = DATA_INTENTS / "intent_utterances.csv"
CLUSTERS_CSV = DATA_INTENTS / "medicare_clusters.csv"
KB_LINKS_CSV = REPORTS_DIR / "kb_links.csv"

OUT_JSONL    = EVAL_QNA_DIR / "evalset.jsonl"
OUT_CSV      = EVAL_QNA_DIR / "evalset.csv"
OUT_REPORT   = REPORTS_DIR  / "eval_payload_report.txt"

def load_csv(path: Path):
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

    # Build utterances map
    utt_map: Dict[str, List[str]] = {}
    rows = load_csv(UTTS_CSV)
    if rows:
        for r in rows:
            intent = (r.get("intent") or "").strip()
            u = norm_space(r.get("utterance") or "")
            if not intent or not u: continue
            utt_map.setdefault(intent, [])
            if len(utt_map[intent]) < args.max_utterances:
                utt_map[intent].append(u)
    else:
        # Fallback to clusters if present
        cl_rows = load_csv(CLUSTERS_CSV)
        if cl_rows:
            # sniff utterance column
            cols = {c.lower(): c for c in cl_rows[0].keys()}
            cand = None
            for k in ["utterance","text","customer_utterance","customer_text"]:
                if k in cols:
                    cand = cols[k]; break
            if cand:
                for r in cl_rows:
                    intent = (r.get("intent") or r.get("intent_guess") or "").strip()
                    u = norm_space(r.get(cand) or "")
                    if not intent or not u: continue
                    utt_map.setdefault(intent, [])
                    if len(utt_map[intent]) < args.max_utterances:
                        utt_map[intent].append(u)

    # Source backfill map
    links = load_csv(KB_LINKS_CSV)
    top_by_intent: Dict[str, List[str]] = {}
    for r in links:
        intent = (r.get("intent") or "").strip()
        sid = (r.get("section_id") or "").strip()
        if not intent or not sid: continue
        top_by_intent.setdefault(intent, [])
        top_by_intent[intent].append((float(r.get("score","0") or 0.0), sid))
    for k,v in top_by_intent.items():
        v.sort(key=lambda x: x[0], reverse=True)
        top_by_intent[k] = [sid for _,sid in v[:3]]

    # Build items
    items, stats = [], {"total":0, "handoff":0, "gold":0, "missing_sources":0, "no_utterances":0}
    for r in gold:
        intent = (r.get("intent") or "").strip()
        question = (r.get("question") or intent.replace("_"," ").capitalize()).strip()
        answer = norm_space(r.get("answer") or "")
        a_type = (r.get("answer_type") or "").strip().lower()
        src = (r.get("sources") or "").strip()

        handoff = (a_type == "handoff")
        src_list = [s for s in src.split(",") if s.strip()]
        if not src_list and not handoff:
            src_list = top_by_intent.get(intent, [])
            if not src_list:
                stats["missing_sources"] += 1

        utters = utt_map.get(intent, [])[:args.max_utterances]
        if not utters:
            stats["no_utterances"] += 1

        items.append({
            "intent": intent,
            "handoff": handoff,
            "question": question,
            "answer": answer,
            "sources": src_list,
            "utterances": utters
        })
        stats["total"] += 1
        stats["handoff"] += int(handoff)
        stats["gold"] += int(not handoff)

    # Write JSONL + CSV
    EVAL_QNA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
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

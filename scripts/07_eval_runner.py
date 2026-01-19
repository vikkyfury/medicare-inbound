#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate model/agent answers against gold set.

Inputs
- eval/qna/evalset.jsonl            # one record per intent: {intent, handoff, question, answer, sources[], utterances[]}
- eval/qna/predictions.jsonl        # your model outputs: {intent, predicted_answer, predicted_handoff?}

Outputs
- reports/eval_scores.txt
- reports/eval_detailed.csv         # row per intent with metrics & text
- reports/eval_confusion_by_intent.csv
- reports/eval_top_errors.csv       # hardest misses by ROUGE-L

Metrics
- Handoff accuracy (for handoff intents)
- Token F1 (lexical) vs gold answers
- ROUGE-L (LCS-based) vs gold answers
- “Pass” heuristic (F1 >= 0.55 OR ROUGE-L >= 0.55), configurable

Usage
  $ cd medicare-inbound
  $ python3 scripts/07_eval_runner.py \
        --pred eval/qna/predictions.jsonl \
        --f1-thresh 0.55 \
        --rougeL-thresh 0.55
"""
from __future__ import annotations
import argparse, json, re, csv
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
EVAL_QNA = ROOT / "eval" / "qna"
REPORTS  = ROOT / "reports"

EVALSET = EVAL_QNA / "evalset.jsonl"

def load_jsonl(p: Path) -> List[dict]:
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

_ws = re.compile(r"\s+")
_punc = re.compile(r"[^\w\s]")

def norm(s: str) -> str:
    return _ws.sub(" ", (s or "").strip()).lower()

def tokenize(s: str) -> List[str]:
    s = norm(s)
    s = _punc.sub(" ", s)
    return [t for t in s.split() if t]

def lcs(a: List[str], b: List[str]) -> int:
    # ROUGE-L core: LCS length
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        ai = a[i]
        for j in range(n):
            if ai == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def rouge_l(hyp: List[str], ref: List[str]) -> float:
    if not hyp or not ref: return 0.0
    l = lcs(hyp, ref)
    prec = l / max(1, len(hyp))
    rec  = l / max(1, len(ref))
    if prec + rec == 0: return 0.0
    beta2 = (1.2**2)  # standard ROUGE-L beta ~1.2
    score = ((1 + beta2) * prec * rec) / (rec + beta2 * prec + 1e-12)
    return float(score)

def token_f1(hyp: List[str], ref: List[str]) -> float:
    if not hyp and not ref: return 1.0
    if not hyp or not ref: return 0.0
    ref_set = {}
    for t in ref:
        ref_set[t] = ref_set.get(t,0) + 1
    overlap = 0
    used = {}
    for t in hyp:
        if ref_set.get(t,0) > used.get(t,0):
            overlap += 1
            used[t] = used.get(t,0) + 1
    prec = overlap / max(1, len(hyp))
    rec  = overlap / max(1, len(ref))
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, default=str(EVAL_QNA / "predictions.jsonl"))
    ap.add_argument("--f1-thresh", type=float, default=0.55)
    ap.add_argument("--rougeL-thresh", type=float, default=0.55)
    args = ap.parse_args()

    gold = {r["intent"]: r for r in load_jsonl(EVALSET)}
    preds = {r["intent"]: r for r in load_jsonl(Path(args.pred))}

    records = []
    handoff_gold = handoff_correct = 0
    passed = 0

    for intent, g in gold.items():
        g_ans = g.get("answer","")
        g_hand = bool(g.get("handoff", False))
        p = preds.get(intent, {})
        p_ans = p.get("predicted_answer","")
        p_hand = p.get("predicted_handoff", None)

        # handoff accuracy (only where gold=handoff)
        if g_hand:
            handoff_gold += 1
            if p_hand is True:
                handoff_correct += 1

        # textual metrics (only for gold answers)
        hyp_toks = tokenize(p_ans)
        ref_toks = tokenize(g_ans)
        f1 = token_f1(hyp_toks, ref_toks) if not g_hand else None
        rl = rouge_l(hyp_toks, ref_toks) if not g_hand else None

        ok = None
        if not g_hand:
            ok = (f1 is not None and f1 >= args.f1_thresh) or (rl is not None and rl >= args.rougeL_thresh)
            if ok: passed += 1

        records.append({
            "intent": intent,
            "gold_handoff": str(g_hand).lower(),
            "pred_handoff": "true" if p_hand is True else "false" if p_hand is False else "",
            "token_f1": "" if f1 is None else f"{f1:.3f}",
            "rougeL": "" if rl is None else f"{rl:.3f}",
            "passed": "" if ok is None else ("true" if ok else "false"),
            "gold_answer": g_ans,
            "pred_answer": p_ans
        })

    # Summaries
    gold_count = sum(1 for r in gold.values() if not r.get("handoff", False))
    pass_rate = (passed / max(1, gold_count)) * 100.0
    handoff_acc = (handoff_correct / max(1, handoff_gold)) * 100.0

    REPORTS.mkdir(parents=True, exist_ok=True)
    # Detailed CSV
    det_csv = REPORTS / "eval_detailed.csv"
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        for r in records:
            w.writerow(r)

    # Confusion-by-intent (very small: whether passed, and if handoff labeled correctly)
    conf_csv = REPORTS / "eval_confusion_by_intent.csv"
    with open(conf_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["intent","is_handoff_gold","handoff_pred_correct","gold_passed"])
        for r in records:
            g_hand = (r["gold_handoff"] == "true")
            h_ok = (r["pred_handoff"] == "true") if g_hand else ""
            w.writerow([r["intent"], r["gold_handoff"], h_ok, r["passed"]])

    # Top errors (lowest ROUGE-L among golds)
    errs = [r for r in records if r["gold_handoff"] == "false" and r["passed"] == "false"]
    errs.sort(key=lambda x: float(x["rougeL"] or 0.0))
    top_errs = REPORTS / "eval_top_errors.csv"
    with open(top_errs, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["intent","token_f1","rougeL","gold_answer","pred_answer"])
        w.writeheader()
        for r in errs[:10]:
            w.writerow({k:r[k] for k in ["intent","token_f1","rougeL","gold_answer","pred_answer"]})

    # Scores
    scores_txt = REPORTS / "eval_scores.txt"
    with open(scores_txt, "w", encoding="utf-8") as f:
        f.write("Eval Scores\n===========\n")
        f.write(f"Gold intents (answerable): {gold_count}\n")
        f.write(f"Pass rate (F1>={args.f1_thresh} or ROUGE-L>={args.rougeL_thresh}): {pass_rate:.1f}%\n")
        f.write(f"Handoff intents: {handoff_gold}\n")
        f.write(f"Handoff accuracy: {handoff_acc:.1f}%\n")
        f.write(f"\nDetailed: {det_csv}\nConfusion: {conf_csv}\nTop errors: {top_errs}\n")

    print(f"✅ Wrote reports to {REPORTS}")
    print(f"- scores: {scores_txt}")
    print(f"- details: {det_csv}")
    print(f"- confusion: {conf_csv}")
    print(f"- top errors: {top_errs}")
    print(f"Gold count={gold_count}, Pass%={pass_rate:.1f}, Handoff%={handoff_acc:.1f}")
if __name__ == "__main__":
    main()

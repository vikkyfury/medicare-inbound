#!/usr/bin/env python3
# Make a baseline predictions.jsonl that mirrors gold answers.
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
EVAL_QNA = ROOT / "eval" / "qna"

evalset = EVAL_QNA / "evalset.jsonl"
pred = EVAL_QNA / "predictions.jsonl"

rows = []
with open(evalset, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        rows.append({
            "intent": r["intent"],
            "predicted_handoff": bool(r.get("handoff", False)),
            "predicted_answer": r.get("answer","") if not r.get("handoff", False) else ""
        })

with open(pred, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Dummy predictions → {pred} ({len(rows)} rows)")

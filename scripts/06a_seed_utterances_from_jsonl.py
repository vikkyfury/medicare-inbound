#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mine customer utterances per intent from normalized JSONL using keyword rules.

Input:
  - configs/dataset.yaml (points to normalized_jsonl)
  - reports/medicare_only_intent_counts.csv (intents present)
  - (optional) configs/intent_policy.yaml (to include intents)
Output:
  - data/intents/intent_utterances.csv  (intent, utterance)
Run:
  $ cd medicare-inbound
  $ python3 scripts/06a_seed_utterances_from_jsonl.py --per-intent 8
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from typing import Dict, List, Set

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "dataset.yaml"
POLICY = ROOT / "configs" / "intent_policy.yaml"
INTENT_COUNTS = ROOT / "reports" / "medicare_only_intent_counts.csv"
NORMALIZED = None
OUT_CSV = ROOT / "data" / "intents" / "intent_utterances.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}

def load_intents() -> List[str]:
    intents: Set[str] = set()
    if INTENT_COUNTS.exists():
        with open(INTENT_COUNTS, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("intent"):
                    intents.add(row["intent"].strip())
    pol = read_yaml(POLICY)
    intents |= set(pol.get("kb_answerable", []) or [])
    intents |= set(pol.get("handoff_required", []) or [])
    return sorted(intents)

# Simple keyword rules (lowercase). Edit/extend as you prefer.
SEEDS = {
    "enrollment": ["enroll", "open enrollment", "oep", "sep", "switch plan", "join plan", "sign up"],
    "eligibility": ["eligible", "eligibility", "turning 65", "disability", "esrd"],
    "premiums_billing": ["premium", "irmaa", "bill", "billing", "late payment", "autopay"],
    "benefits_coverage": ["covered", "coverage", "preventive", "benefits"],
    "claims_status": ["claim", "eob", "denied", "reimbursement", "appeal"],
    "appeals_grievances": ["appeal", "grievance", "complaint", "coverage decision"],
    "rx_part_d": ["part d", "drug", "formulary", "pharmacy", "prescription", "tier", "prior auth"],
    "copay_coins_deduct": ["copay", "coinsurance", "deductible", "out-of-pocket"],
    "pcp_network": ["pcp", "primary care", "in network", "out of network", "referral"],
    "prior_auth": ["prior authorization", "pre authorization", "pa required", "medical necessity"],
    "vision_dental_hearing": ["dental", "vision", "hearing", "eyeglasses"],
    "otc_benefit": ["otc", "over the counter"],
    "id_card": ["id card", "member card", "replacement card"],
    "address_contact_update": ["address change", "update address", "change phone", "contact update"],
    "cancel_disenroll": ["disenroll", "cancel plan", "drop plan"],
    "appointment_referral": ["referral", "authorization", "appointment"],
}

AGENT_PATTERNS = re.compile(
    r"(?i)\b(thank you for calling|how can i help|please hold|one moment|let me pull|this is .* speaking)\b"
)

def is_likely_customer(text: str) -> bool:
    # Avoid obvious agent boilerplate
    return not bool(AGENT_PATTERNS.search(text))

def matches_intent(intent: str, text: str) -> bool:
    toks = SEEDS.get(intent, [])
    t = text.lower()
    return any(k in t for k in toks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-intent", type=int, default=8)
    args = ap.parse_args()

    cfg = read_yaml(CONFIG)
    global NORMALIZED
    NORMALIZED = ROOT / cfg.get("normalized_jsonl", "data/medicare_inbounds/medicare_conversations.jsonl")
    if not NORMALIZED.exists():
        raise SystemExit(f"[ERR] Missing normalized JSONL at {NORMALIZED}")

    intents = load_intents()
    # Precompile pattern for each intent
    out: Dict[str, List[str]] = {i: [] for i in intents}

    with open(NORMALIZED, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for utt in obj.get("utterances", []):
                text = (utt.get("text") or "").strip()
                if not text or len(text) < 12:
                    continue
                if not is_likely_customer(text):
                    continue
                for intent in intents:
                    if len(out[intent]) >= args.per_intent:
                        continue
                    if matches_intent(intent, text):
                        out[intent].append(text)

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["intent","utterance"])
        for intent in intents:
            for u in out[intent][:args.per_intent]:
                w.writerow([intent, u])

    total = sum(len(v) for v in out.values())
    print(f"✅ Seeded utterances → {OUT_CSV} (total {total})")

if __name__ == "__main__":
    main()

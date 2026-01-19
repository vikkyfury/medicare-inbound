#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 → Part 2 (revised):
Extract customer utterances + seed intents with robust starting-speaker inference.

Run:
  $ cd medicare-inbound
  $ python3 scripts/02_build_intent_dataset.py
"""

from pathlib import Path
import csv
import re
import json
from collections import defaultdict, Counter
import hashlib
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]

# --- Config paths ---
CONFIG_YAML = ROOT / "configs" / "dataset.yaml"
with open(CONFIG_YAML, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

NORMALIZED_JSONL = ROOT / cfg["normalized_jsonl"]

# --- Outputs ---
INTENTS_DIR = ROOT / "data" / "intents"
INTENTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_QNA_DIR = ROOT / "eval" / "qna"
EVAL_QNA_DIR.mkdir(parents=True, exist_ok=True)

CUSTOMER_CSV = INTENTS_DIR / "customer_utterances.csv"
SEEDED_CSV   = INTENTS_DIR / "seed_labeled.csv"
COUNTS_CSV   = REPORTS_DIR / "intent_counts.csv"
SEEDS_JSONL  = EVAL_QNA_DIR / "intent_seeds.jsonl"
STATS_LOG    = REPORTS_DIR / "intent_extraction_stats.txt"
INTENT_POLICY_YAML = ROOT / "configs" / "intent_policy.yaml"

# --- Seed intent rules (unchanged) ---
SEED_RULES = [
    ("enrollment", "enroll_1", r"\b(enroll|sign ?up|switch to medicare|start medicare|join a (plan|medicare))\b"),
    ("plan_change", "planchg_1", r"\b(change|switch|move)\s+(my|our)?\s*(plan|coverage)\b"),
    ("benefits_coverage", "cov_1", r"\b(cover(ed|age)?|benefit(s)?|included)\b"),
    ("premiums_billing", "bill_1", r"\b(premium|bill(ing)?|payment|auto[- ]?pay|due date|past due)\b"),
    ("claims_status", "claim_1", r"\b(claim(s)?|reimburs(e|ement)|filed|denied|processed|status)\b"),
    ("id_card", "id_1", r"\b(id|identification|member|insurance)\s*(card)\b|\b(new card|replacement card)\b"),
    ("pcp_network", "pcp_1", r"\b(primary care|pcp|in[- ]?network|out[- ]?of[- ]?network|provider directory)\b"),
    ("rx_part_d", "rx_1", r"\b(prescription|rx|pharmacy|formulary|part\s*d|drug(s)? (cover(ed|age)?))\b"),
    ("prior_auth", "pa_1", r"\b(prior authorization|pre[- ]?auth|authorization required)\b"),
    ("eligibility", "elig_1", r"\b(eligib(le|ility)|qualif(y|ication)|who can enroll)\b"),
    ("appeals_grievances", "appeal_1", r"\b(appeal(s)?|grievance(s)?|complaint(s)?)\b"),
    ("copay_coins_deduct", "costshare_1", r"\b(co[- ]?pay|co[- ]?insurance|coinsurance|deductible)\b"),
    ("vision_dental_hearing", "vdh_1", r"\b(dental|vision|hearing|eyewear)\b"),
    ("otc_benefit", "otc_1", r"\b(otc|over[- ]?the[- ]?counter)\b"),
    ("address_contact_update", "addr_1", r"\b(address|mailing|contact)\s*(change|update)\b"),
    ("cancel_disenroll", "disenroll_1", r"\b(cancel|terminate|disenroll)\b"),
    ("fraud_waste_abuse", "fwa_1", r"\b(fraud|waste|abuse|scam)\b"),
    ("compliment_complaint", "fbk_1", r"\b(compliment|feedback|complain(t)?)\b"),
    ("appointment_referral", "appt_1", r"\b(appointment|referral|schedule|book)\b"),
    ("idk_other", "other_1", r"\b(not sure|don.?t know|confused|help)\b"),
]
COMPILED_RULES = [(i, r, re.compile(rx, re.IGNORECASE)) for i, r, rx in SEED_RULES]

# --- Heuristics for starting speaker ---
AGENT_PATTERNS = [
    r"\b(thank you|thanks) for calling\b",
    r"\bwelcome to\b",
    r"\byou(?:'| )?ve reached\b",
    r"\bthis is\b.*\b(customer service|support|claims|member services|medicare)\b",
    r"\bhow (?:can|may) i (?:help|assist) (?:you|today)\b",
    r"\bfor security (?:purposes|reasons).*(?:verify|confirm)\b",
    r"\bmay i have your (?:member|policy) (?:id|number)\b",
    r"\bplease (?:provide|confirm) your (?:date of birth|dob|zip code)\b",
]

CUSTOMER_PATTERNS = [
    r"\bi[' ]?m calling\b|\bi am calling\b",
    r"\bi (?:need|want|would like)\b",
    r"\bmy (?:claim|coverage|card|plan|premium|pharmacy)\b",
    r"\b(can|could) you\b|\bwhat\b|\bwhy\b|\bhow\b|\bwhere\b|\bwhen\b",  # question-led
    r"\bi was (?:charged|billed|denied)\b",
    r"\bquestion about\b",
]

AGENT_RX = [re.compile(p, re.IGNORECASE) for p in AGENT_PATTERNS]
CUST_RX  = [re.compile(p, re.IGNORECASE) for p in CUSTOMER_PATTERNS]

def score_text(txt: str, patterns):
    s = 0
    for rx in patterns:
        if rx.search(txt):
            s += 1
    return s

def guess_starting_speaker(turns):
    """
    Look at first 1-3 turns' text to guess who starts.
    Returns "agent" or "customer" plus a bool 'certain'.
    """
    first_k = [ (t.get("text") or "").strip() for t in turns[:3] ]
    a = sum(score_text(t, AGENT_RX) for t in first_k if t)
    c = sum(score_text(t, CUST_RX)  for t in first_k if t)

    if a > c:
        return "agent", True
    if c > a:
        return "customer", True
    return "agent", False  # tie/uncertain → sensible inbound default

# --- Misc helpers ---
def hash_id(*parts: str) -> str:
    m = hashlib.sha1()
    for p in parts:
        m.update(p.encode("utf-8"))
        m.update(b"|")
    return m.hexdigest()[:16]

def ensure_policy_yaml():
    if INTENT_POLICY_YAML.exists():
        return
    policy = {
        "kb_answerable": [
            "enrollment", "plan_change", "benefits_coverage", "premiums_billing",
            "claims_status", "id_card", "pcp_network", "rx_part_d", "prior_auth",
            "eligibility", "appeals_grievances", "copay_coins_deduct",
            "vision_dental_hearing", "otc_benefit", "address_contact_update",
            "cancel_disenroll"
        ],
        "handoff_required": [
            "fraud_waste_abuse", "compliment_complaint", "appointment_referral", "idk_other"
        ],
        "notes": "Edit as your KB grows; promote intents when authoritative content is added."
    }
    with open(INTENT_POLICY_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(policy, f, sort_keys=False, allow_unicode=True)

def main():
    if not NORMALIZED_JSONL.exists():
        print(f"[ERR] Missing normalized JSONL: {NORMALIZED_JSONL}", file=sys.stderr)
        sys.exit(1)

    ensure_policy_yaml()

    total_convs = 0
    explicit_role_convs = 0
    inferred_convs = 0
    inferred_certain = 0
    inferred_uncertain = 0
    total_customer_utts = 0

    cust_rows = []  # (utterance_id, conversation_id, turn_idx, text)

    print("[1/4] Extracting customer utterances (heuristic start + alternation)…")
    with open(NORMALIZED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            total_convs += 1

            conv_id = obj.get("conversation_id") or hash_id(obj.get("source_path",""))
            turns = obj.get("utterances", [])

            # If any explicit roles, trust them
            if any((t.get("speaker","").lower() in {"customer","agent"}) for t in turns):
                explicit_role_convs += 1
                for i, t in enumerate(turns):
                    if (t.get("speaker") or "").lower() == "customer":
                        txt = (t.get("text") or "").strip()
                        if txt:
                            uid = hash_id(conv_id, str(i), txt[:64])
                            cust_rows.append((uid, conv_id, i, txt))
                            total_customer_utts += 1
                continue

            # All unknown → guess starting speaker, then alternate
            inferred_convs += 1
            start_role, certain = guess_starting_speaker(turns)
            if certain:
                inferred_certain += 1
            else:
                inferred_uncertain += 1

            # Build alternation pattern from starting role
            # agent, customer, agent, customer... or customer, agent, ...
            for i, t in enumerate(turns):
                role = "customer" if ((start_role == "agent" and i % 2 == 1) or (start_role == "customer" and i % 2 == 0)) else "agent"
                if role == "customer":
                    txt = (t.get("text") or "").strip()
                    if txt:
                        uid = hash_id(conv_id, str(i), txt[:64])
                        cust_rows.append((uid, conv_id, i, txt))
                        total_customer_utts += 1

    # Write customer utterances
    with open(CUSTOMER_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utterance_id", "conversation_id", "turn_idx", "text"])
        w.writerows(cust_rows)

    # 2) Seed rules
    print("[2/4] Applying seed keyword rules…")
    labeled = []
    for uid, conv, idx, txt in cust_rows:
        for intent, rule_id, rx in COMPILED_RULES:
            if rx.search(txt):
                labeled.append((uid, intent, rule_id))
    with open(SEEDED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utterance_id", "intent", "rule_id"])
        w.writerows(labeled)

    # 3) Counts
    print("[3/4] Aggregating counts…")
    counter = Counter([i for _, i, _ in labeled])
    with open(COUNTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["intent", "count"])
        for intent, cnt in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([intent, cnt])

    # 4) Seed examples
    print("[4/4] Selecting representative examples…")
    by_intent = defaultdict(list)
    for uid, intent, rule in labeled:
        by_intent[intent].append(uid)
    text_map = {uid: txt for uid, _, _, txt in cust_rows}
    K = 50
    with open(SEEDS_JSONL, "w", encoding="utf-8") as out:
        for intent, uids in sorted(by_intent.items(), key=lambda x: x[0]):
            picked = uids[:K]
            out.write(json.dumps({
                "intent": intent,
                "examples": [{"utterance_id": u, "text": text_map.get(u, "")} for u in picked]
            }, ensure_ascii=False) + "\n")

    # Stats
    with open(STATS_LOG, "w", encoding="utf-8") as f:
        f.write(
            "Intent Extraction Stats (heuristic start)\n"
            f"Total conversations: {total_convs}\n"
            f"Convs w/ explicit roles: {explicit_role_convs}\n"
            f"Inferred conversations: {inferred_convs}\n"
            f"  - certain start guess: {inferred_certain}\n"
            f"  - uncertain start guess: {inferred_uncertain}\n"
            f"Total customer utterances captured: {total_customer_utts}\n"
            f"Seed-labeled utterances: {len(labeled)}\n"
        )

    print("✅ DONE.")
    print(f"  - Customer utterances → {CUSTOMER_CSV}")
    print(f"  - Seed labels → {SEEDED_CSV}")
    print(f"  - Intent counts → {COUNTS_CSV}")
    print(f"  - Seed examples → {SEEDS_JSONL}")
    print(f"  - Stats → {STATS_LOG}")

if __name__ == "__main__":
    main()

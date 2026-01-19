#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Medicare-only slice from normalized conversations, then label intents.

Run:
  $ cd medicare-inbound
  $ python3 scripts/02b_filter_medicare_slice.py
"""

from pathlib import Path
import csv, json, re, sys, yaml, hashlib
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]

# ---- Config ----
CONFIG_YAML = ROOT / "configs" / "dataset.yaml"
with open(CONFIG_YAML, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

NORMALIZED_JSONL = ROOT / cfg["normalized_jsonl"]

INTENTS_DIR = ROOT / "data" / "intents"
INTENTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_QNA_DIR = ROOT / "eval" / "qna"
EVAL_QNA_DIR.mkdir(parents=True, exist_ok=True)

OUT_CUST = INTENTS_DIR / "medicare_only_customer_utterances.csv"
OUT_SEEDED = INTENTS_DIR / "medicare_only_seed_labeled.csv"
OUT_COUNTS = REPORTS_DIR / "medicare_only_intent_counts.csv"
OUT_SEEDS = EVAL_QNA_DIR / "medicare_only_intent_seeds.jsonl"
OUT_STATS = REPORTS_DIR / "medicare_slice_stats.txt"

# ---- Medicare domain filters (conservative but broad) ----
MEDICARE_PATTERNS = [
    r"\bmedicare\b",
    r"\bpart\s*[abd]\b", r"\bpart\s*c\b", r"\bpart\s*d\b", r"\bma(plan)?\b", r"\bmedicare advantage\b",
    r"\bsupplement(al)?\b", r"\bmedigap\b",
    r"\bcms\b", r"\bcenter(s)? for medicare\b",
    r"\benrollment\b|\bopen enrollment\b|\bane\b|\boep\b|\bsep\b",
    r"\bpremium(s)?\b|\bco[- ]?pay\b|\bcoinsurance\b|\bdeductible\b|\bout[- ]of[- ]pocket\b",
    r"\bformulary\b|\bpart\s*d\b|\brx\b|\bprior authorization\b|\bpre[- ]?auth\b",
    r"\bpcp\b|\bprimary care\b|\bin[- ]?network\b|\bout[- ]?of[- ]?network\b",
    r"\bclaims?\b|\bappeal(s)?\b|\bgrievance(s)?\b",
    r"\bbenefit(s)?\b|\bcoverage\b",
]
MEDICARE_RX = [re.compile(p, re.IGNORECASE) for p in MEDICARE_PATTERNS]

# ---- Intent seed rules (same set as 02 script) ----
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

# ---- Helpers ----
def hash_id(*parts: str) -> str:
    m = hashlib.sha1()
    for p in parts:
        m.update(p.encode("utf-8")); m.update(b"|")
    return m.hexdigest()[:16]

def speaker_known(t): return (t.get("speaker") or "").lower() in {"customer","agent"}

AGENT_HINTS = [
    r"\b(thank you|thanks) for calling\b",
    r"\bwelcome to\b",
    r"\byou(?:'| )?ve reached\b",
    r"\bthis is\b.*\b(customer service|support|claims|member services|medicare)\b",
    r"\bhow (?:can|may) i (?:help|assist)\b",
    r"\bfor security (?:purposes|reasons).*(?:verify|confirm)\b",
    r"\bmay i have your (?:member|policy) (?:id|number)\b",
]
CUSTOMER_HINTS = [
    r"\bi[' ]?m calling\b|\bi am calling\b",
    r"\bi (?:need|want|would like)\b",
    r"\bmy (?:claim|coverage|card|plan|premium|pharmacy)\b",
    r"\b(can|could) you\b|\bwhat\b|\bwhy\b|\bhow\b|\bwhere\b|\bwhen\b",
    r"\bi was (?:charged|billed|denied)\b",
    r"\bquestion about\b",
]
AGENT_RX = [re.compile(p, re.IGNORECASE) for p in AGENT_HINTS]
CUST_RX  = [re.compile(p, re.IGNORECASE) for p in CUSTOMER_HINTS]

def score(txt, pat_list): return sum(1 for rx in pat_list if rx.search(txt or ""))

def guess_start(turns):
    first = [(t.get("text") or "").strip() for t in turns[:3]]
    a = sum(score(t, AGENT_RX) for t in first)
    c = sum(score(t, CUST_RX) for t in first)
    if a > c: return "agent"
    if c > a: return "customer"
    return "agent"  # fallback for inbound

def any_medicare_text(turns):
    for t in turns:
        txt = (t.get("text") or "")
        for rx in MEDICARE_RX:
            if rx.search(txt):
                return True
    return False

def main():
    if not NORMALIZED_JSONL.exists():
        print(f"[ERR] Missing {NORMALIZED_JSONL}"); sys.exit(1)

    total_convs = 0
    medicare_convs = 0
    explicit_role_convs = 0
    cust_utts = 0
    labeled = 0

    cust_rows = []  # (utterance_id, conversation_id, turn_idx, text)
    with open(NORMALIZED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line); total_convs += 1
            cid = obj.get("conversation_id") or hash_id(obj.get("source_path",""))
            turns = obj.get("utterances", [])

            # Keep only Medicare-relevant conversations
            if not any_medicare_text(turns):
                continue
            medicare_convs += 1

            # Pull customer turns (explicit roles if present; else infer start then alternate)
            if any(speaker_known(t) for t in turns):
                explicit_role_convs += 1
                for i, t in enumerate(turns):
                    if (t.get("speaker") or "").lower() == "customer":
                        txt = (t.get("text") or "").strip()
                        if txt:
                            uid = hash_id(cid, str(i), txt[:64])
                            cust_rows.append((uid, cid, i, txt)); cust_utts += 1
            else:
                start = guess_start(turns)
                for i, t in enumerate(turns):
                    role = "customer" if ((start == "agent" and i % 2 == 1) or (start == "customer" and i % 2 == 0)) else "agent"
                    if role == "customer":
                        txt = (t.get("text") or "").strip()
                        if txt:
                            uid = hash_id(cid, str(i), txt[:64])
                            cust_rows.append((uid, cid, i, txt)); cust_utts += 1

    # Write customers
    with open(OUT_CUST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["utterance_id","conversation_id","turn_idx","text"]); w.writerows(cust_rows)

    # Seed label on the Medicare slice
    labels = []
    compiled = COMPILED_RULES
    for uid, cid, i, txt in cust_rows:
        for intent, rule_id, rx in compiled:
            if rx.search(txt):
                labels.append((uid, intent, rule_id))
    with open(OUT_SEEDED, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["utterance_id","intent","rule_id"]); w.writerows(labels)
    labeled = len(labels)

    # Counts
    counter = Counter([i for _, i, _ in labels])
    with open(OUT_COUNTS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["intent","count"])
        for intent, cnt in sorted(counter.items(), key=lambda x:(-x[1], x[0])):
            w.writerow([intent, cnt])

    # Seeds per intent
    text_map = {uid: txt for uid, _, _, txt in cust_rows}
    by_intent = defaultdict(list)
    for uid, intent, _ in labels:
        by_intent[intent].append(uid)
    with open(OUT_SEEDS, "w", encoding="utf-8") as out:
        K = 50
        for intent, uids in sorted(by_intent.items()):
            out.write(json.dumps({
                "intent": intent,
                "examples": [{"utterance_id": u, "text": text_map.get(u,"")} for u in uids[:K]]
            }, ensure_ascii=False) + "\n")

    # Stats
    with open(OUT_STATS, "w", encoding="utf-8") as s:
        s.write(
            "Medicare Slice Stats\n"
            f"Total conversations scanned: {total_convs}\n"
            f"Medicare-flagged conversations: {medicare_convs}\n"
            f"Conversations w/ explicit roles: {explicit_role_convs}\n"
            f"Total customer utterances (Medicare slice): {cust_utts}\n"
            f"Seed-labeled utterances (Medicare slice): {labeled}\n"
        )

    print("✅ Medicare slice built.")
    print(f"  - Customers → {OUT_CUST}")
    print(f"  - Seed labels → {OUT_SEEDED}")
    print(f"  - Intent counts → {OUT_COUNTS}")
    print(f"  - Seeds → {OUT_SEEDS}")
    print(f"  - Stats → {OUT_STATS}")

if __name__ == "__main__":
    main()

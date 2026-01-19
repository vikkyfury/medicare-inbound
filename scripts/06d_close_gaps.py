#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Close remaining gaps in eval payload:
- Backfill missing sources for gold answers
- Top up utterances per intent (mine JSONL again; synthesize if needed)
- Re-export evalset.jsonl / evalset.csv

Run:
  $ cd medicare-inbound
  $ python3 scripts/06d_close_gaps.py --per-intent 5
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML   = ROOT / "configs" / "dataset.yaml"
GOLD_CLEAN    = ROOT / "eval" / "qna" / "gold_answers.cleaned.csv"
KB_LINKS_CSV  = ROOT / "reports" / "kb_links.csv"
KB_INDEX_CSV  = ROOT / "reports" / "kb_index.csv"
UTTS_CSV      = ROOT / "data" / "intents" / "intent_utterances.csv"
NORMALIZED    = None

OUT_UTTS_CSV  = UTTS_CSV
OUT_JSONL     = ROOT / "eval" / "qna" / "evalset.jsonl"
OUT_CSV       = ROOT / "eval" / "qna" / "evalset.csv"
OUT_REPORT    = ROOT / "reports" / "eval_payload_report.txt"

def load_csv(path: Path):
    if not path.exists(): return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_csv(path: Path, rows: List[Dict[str,str]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# Keywords per intent (used for mining + synthesis)
SEEDS = {
    "enrollment": ["enroll","open enrollment","oep","sep","switch plan","join plan","sign up"],
    "eligibility": ["eligible","eligibility","turning 65","disability","esrd"],
    "premiums_billing": ["premium","irmaa","bill","billing","late payment","autopay"],
    "benefits_coverage": ["covered","coverage","preventive","benefits"],
    "claims_status": ["claim","eob","denied","reimbursement","appeal"],
    "appeals_grievances": ["appeal","grievance","complaint","coverage decision"],
    "rx_part_d": ["part d","drug","formulary","pharmacy","prescription","tier","prior auth"],
    "copay_coins_deduct": ["copay","coinsurance","deductible","out-of-pocket"],
    "pcp_network": ["pcp","primary care","in network","out of network","referral"],
    "prior_auth": ["prior authorization","pre authorization","pa required","medical necessity"],
    "vision_dental_hearing": ["dental","vision","hearing","eyeglasses"],
    "otc_benefit": ["otc","over the counter"],
    "id_card": ["id card","member card","replacement card"],
    "address_contact_update": ["address change","update address","change phone","contact update"],
    "cancel_disenroll": ["disenroll","cancel plan","drop plan"],
    "appointment_referral": ["referral","authorization","appointment"],
    # safety net for unknown buckets (e.g., idk_other)
    "idk_other": ["question","help","not sure","don’t know","other"]
}

AGENT_PAT = re.compile(r"(?i)\b(thank you for calling|how can i help|please hold|one moment|let me pull|this is .* speaking)\b")

def is_customer_like(text: str) -> bool:
    return not bool(AGENT_PAT.search(text))

def text_matches(intent: str, text: str) -> bool:
    t = text.lower()
    seeds = SEEDS.get(intent, SEEDS["idk_other"])
    return any(k in t for k in seeds)

def mine_more_utterances(intent: str, need: int, normalized_jsonl: Path) -> List[str]:
    if not normalized_jsonl.exists() or need <= 0:
        return []
    out = []
    with open(normalized_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if len(out) >= need: break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for utt in obj.get("utterances", []):
                text = (utt.get("text") or "").strip()
                if len(text) < 8:
                    continue
                if not is_customer_like(text):
                    continue
                if text_matches(intent, text):
                    out.append(norm_space(text))
                    if len(out) >= need: break
    return out

def synthesize_from_question(intent: str, question: str, want: int) -> List[str]:
    q = norm_space(question)
    seeds = SEEDS.get(intent, SEEDS["idk_other"])[:3]
    templates = [
        "Can you help me with {}?",
        "I need info about {}.",
        "What are my options for {}?",
        "How do I {}?",
        "I want to {}. What should I do?",
    ]
    outs = []
    slot = seeds[0] if seeds else q.lower()
    for t in templates:
        if len(outs) >= want: break
        outs.append(norm_space(t.format(slot)))
    # uniqueness + minimum length
    uniq, seen = [], set()
    for u in outs:
        lu = u.lower()
        if lu in seen or len(u) < 8:
            continue
        uniq.append(u)
        seen.add(lu)
        if len(uniq) >= want: break
    return uniq

def main():
    import yaml, json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-intent", type=int, default=5, help="Target utterances per intent")
    args = ap.parse_args()

    # dataset.yaml -> normalized jsonl
    cfg = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8")) if CONFIG_YAML.exists() else {}
    global NORMALIZED
    NORMALIZED = ROOT / cfg.get("normalized_jsonl", "data/medicare_inbounds/medicare_conversations.jsonl")

    gold = load_csv(GOLD_CLEAN)
    kblinks = load_csv(KB_LINKS_CSV)
    kbindex = load_csv(KB_INDEX_CSV)
    utts = load_csv(UTTS_CSV)

    # Build utterance map
    utt_map: Dict[str, List[str]] = {}
    for r in utts:
        intent = (r.get("intent") or "").strip()
        u = norm_space(r.get("utterance") or "")
        if not intent or not u:
            continue
        utt_map.setdefault(intent, []).append(u)

    # Build top sources map
    top_by_intent: Dict[str, List[str]] = {}
    for r in kblinks:
        intent = (r.get("intent") or "").strip()
        sid = (r.get("section_id") or "").strip()
        if not intent or not sid:
            continue
        top_by_intent.setdefault(intent, []).append((float(r.get("score","0") or 0.0), sid))
    for k, v in top_by_intent.items():
        v.sort(key=lambda x: x[0], reverse=True)
        top_by_intent[k] = [sid for _, sid in v[:3]]

    # Generic fallback section_id (broad docs)
    generic_sid = ""
    if kbindex:
        for r in kbindex:
            h = (r.get("heading") or "").lower()
            fp = (r.get("file_relpath") or "").lower()
            if "coverage" in h or "coverage" in fp or "parts_of_medicare" in fp or "medicare_and_you" in fp:
                generic_sid = r.get("section_id","") or generic_sid
                if generic_sid: break
        if not generic_sid:
            generic_sid = kbindex[0].get("section_id","") or ""

    # Patch gold + build full utterance set
    patched_gold = []
    for r in gold:
        intent = (r.get("intent") or "").strip()
        a_type = (r.get("answer_type") or "").strip().lower()
        question = (r.get("question") or intent.replace("_"," ").capitalize()).strip()
        # ensure key exists to avoid KeyError
        utt_map.setdefault(intent, [])

        # backfill sources if needed
        src_list = [s for s in (r.get("sources") or "").split(",") if s.strip()]
        if not src_list and a_type == "gold":
            src_list = top_by_intent.get(intent, [])
            if not src_list and generic_sid:
                src_list = [generic_sid]
            r["sources"] = ",".join(src_list)

        # ensure target number of utterances
        have = len(utt_map[intent])
        need = max(0, args.per-intent if False else args.per_intent - have)  # clarity: avoid hyphen parse
        need = max(0, args.per_intent - have)
        if need > 0:
            mined = mine_more_utterances(intent, need, NORMALIZED)
            for m in mined:
                if len(utt_map[intent]) >= args.per_intent: break
                utt_map[intent].append(m)
        have = len(utt_map[intent])
        need = max(0, args.per_intent - have)
        if need > 0:
            syn = synthesize_from_question(intent, question, need)
            for s in syn:
                if len(utt_map[intent]) >= args.per_intent: break
                utt_map[intent].append(s)

        patched_gold.append(r)

    # Write updated utterances
    out_rows = []
    for intent, lst in sorted(utt_map.items()):
        for u in lst[:args.per_intent]:
            out_rows.append({"intent": intent, "utterance": u})
    save_csv(OUT_UTTS_CSV, out_rows, fieldnames=["intent","utterance"])

    # Save patched gold (with updated sources if any)
    save_csv(GOLD_CLEAN, patched_gold, fieldnames=["intent","question","answer","answer_type","sources"])

    # Build eval items + stats
    items, stats = [], {"total":0,"handoff":0,"gold":0,"missing_sources":0,"no_utterances":0}
    for r in patched_gold:
        intent = (r.get("intent") or "").strip()
        a_type = (r.get("answer_type") or "").strip().lower()
        question = (r.get("question") or intent.replace("_"," ").capitalize()).strip()
        answer = norm_space(r.get("answer") or "")
        src_list = [s for s in (r.get("sources") or "").split(",") if s.strip()]
        if not src_list and a_type == "gold":
            stats["missing_sources"] += 1
        utters = (utt_map.get(intent, []) or [])[:5]
        if not utters: stats["no_utterances"] += 1
        handoff = (a_type == "handoff")

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

    # Write JSONL & CSV
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
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
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("Eval Payload Report\n")
        f.write("===================\n")
        f.write(f"total: {stats['total']}\n")
        f.write(f"handoff: {stats['handoff']}\n")
        f.write(f"gold: {stats['gold']}\n")
        f.write(f"missing_sources: {stats['missing_sources']}\n")
        f.write(f"no_utterances: {stats['no_utterances']}\n")
        f.write(f"\nJSONL : {OUT_JSONL}\nCSV   : {OUT_CSV}\n")

    print("✅ Gaps closed and eval payload re-written.")
    print(f"Report → {OUT_REPORT}")
    print(f"JSONL  → {OUT_JSONL}")
    print(f"CSV    → {OUT_CSV}")

if __name__ == "__main__":
    main()

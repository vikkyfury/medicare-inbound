#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 → Part 3:
Cluster Medicare customer utterances and scaffold a gold-answers template.

Inputs
- data/intents/medicare_only_customer_utterances.csv
- data/intents/medicare_only_seed_labeled.csv
- configs/intent_policy.yaml
- (optional) --k <int> number of clusters (default: 25)

Outputs
- data/intents/medicare_clusters.csv
  (utterance_id, conversation_id, turn_idx, text, cluster_id)
- reports/medicare_cluster_summary.csv
  (cluster_id, size, labeled_frac, top_terms, suggested_intent, support, example_ids, example_texts)
- eval/qna/gold_template.csv
  (intent, example_question, gold_answer, references, handoff, notes)

Run
$ cd medicare-inbound
$ python3 scripts/03_cluster_and_gold_template.py --k 25
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]

# ---------- paths ----------
INTENTS_DIR = ROOT / "data" / "intents"
REPORTS_DIR = ROOT / "reports"
EVAL_QNA_DIR = ROOT / "eval" / "qna"
POLICY_YAML = ROOT / "configs" / "intent_policy.yaml"

CUST_CSV   = INTENTS_DIR / "medicare_only_customer_utterances.csv"
SEEDED_CSV = INTENTS_DIR / "medicare_only_seed_labeled.csv"

CLUSTERS_CSV = INTENTS_DIR / "medicare_clusters.csv"
CLUSTER_SUMMARY_CSV = REPORTS_DIR / "medicare_cluster_summary.csv"
GOLD_TEMPLATE_CSV = EVAL_QNA_DIR / "gold_template.csv"

for p in [INTENTS_DIR, REPORTS_DIR, EVAL_QNA_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def load_policy():
    if POLICY_YAML.exists():
        with open(POLICY_YAML, "r", encoding="utf-8") as f:
            pol = yaml.safe_load(f) or {}
    else:
        pol = {}
    kb = set(pol.get("kb_answerable", []))
    ho = set(pol.get("handoff_required", []))
    return kb, ho

def pick_k_auto(n_rows: int) -> int:
    # heuristic if user didn't pass --k
    # sqrt(n/50) bounded [10, 60]
    k = max(10, min(60, int(round(math.sqrt(max(1, n_rows) / 50.0)))))
    return k

def top_terms_for_cluster(tfidf, feature_names, labels, cluster_id, topn=12):
    # Get mean vector for items in the cluster, rank features
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    centroid = tfidf[idx].mean(axis=0).A1  # average tf-idf across cluster
    top_idx = np.argsort(centroid)[::-1][:topn]
    return [feature_names[i] for i in top_idx if centroid[i] > 0]

def sanitize_text(x: str) -> str:
    x = (x or "").strip()
    # collapse spaces and remove bracketed placeholders like [PERSON_NAME]
    x = re.sub(r"\s+", " ", x)
    return x

def majority_vote_intent(utter_ids, seed_df):
    # seed_df: columns [utterance_id, intent, rule_id]
    sub = seed_df[seed_df["utterance_id"].isin(utter_ids)]
    if sub.empty:
        return None, 0
    counts = Counter(sub["intent"].tolist())
    intent, cnt = counts.most_common(1)[0]
    support = cnt / max(1, len(utter_ids))
    return intent, support

def pick_examples(df_cluster, k=3):
    # choose first k examples deterministically
    ex = df_cluster.head(k)
    return ex["utterance_id"].tolist(), ex["text"].tolist()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="number of clusters (default: auto or 25)")
    args = ap.parse_args()

    # Load data
    if not CUST_CSV.exists():
        raise FileNotFoundError(f"Missing {CUST_CSV}")
    cust_df = pd.read_csv(CUST_CSV)  # utterance_id, conversation_id, turn_idx, text
    cust_df["text"] = cust_df["text"].astype(str).map(sanitize_text)

    if not SEEDED_CSV.exists():
        seed_df = pd.DataFrame(columns=["utterance_id", "intent", "rule_id"])
    else:
        seed_df = pd.read_csv(SEEDED_CSV)

    if cust_df.empty:
        raise SystemExit("No customer utterances in Medicare slice.")

    # Vectorize
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        strip_accents="unicode",
    )
    X = vec.fit_transform(cust_df["text"].tolist())
    feature_names = np.array(vec.get_feature_names_out())

    # Choose k
    k = args.k if args.k else 25
    if args.k is None:
        k = pick_k_auto(cust_df.shape[0])

    # Cluster
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init="auto")
    labels = km.fit_predict(X)

    # Save utterance-level assignments
    out_df = cust_df.copy()
    out_df["cluster_id"] = labels
    out_df.to_csv(CLUSTERS_CSV, index=False)

    # Build cluster summary
    rows = []
    for c in range(k):
        c_df = out_df[out_df["cluster_id"] == c]
        size = int(c_df.shape[0])
        labeled_frac = float(
            c_df["utterance_id"].isin(seed_df["utterance_id"] if not seed_df.empty else []).mean()
        )

        terms = top_terms_for_cluster(X, feature_names, labels, c, topn=12)
        ex_ids, ex_texts = pick_examples(c_df, k=3)
        sugg_intent, support = majority_vote_intent(ex_ids + c_df["utterance_id"].tolist(), seed_df)

        rows.append({
            "cluster_id": c,
            "size": size,
            "labeled_frac": round(labeled_frac, 4),
            "top_terms": ", ".join(terms),
            "suggested_intent": sugg_intent if sugg_intent else "",
            "support": round(float(support), 4) if sugg_intent else 0.0,
            "example_ids": "|".join(ex_ids),
            "example_texts": " || ".join(ex_texts),
        })

    summary_df = pd.DataFrame(rows).sort_values(["size"], ascending=False)
    summary_df.to_csv(CLUSTER_SUMMARY_CSV, index=False)

    # Gold template: one starter row per suggested intent (top by size/support),
    # and also add rows for seed intents that never appeared as suggestions.
    kb, ho = load_policy()

    # pick best cluster example as a sample question
    gold_rows = []
    seen_intents = set()
    for _, r in summary_df.iterrows():
        intent = (r["suggested_intent"] or "").strip()
        if not intent:
            continue
        if intent in seen_intents:
            continue
        # choose the first example text from that cluster
        example_q = ""
        if r["example_texts"]:
            example_q = r["example_texts"].split(" || ")[0].strip()
        handoff = "FALSE" if intent in kb else ("TRUE" if intent in ho else "FALSE")
        gold_rows.append({
            "intent": intent,
            "example_question": example_q,
            "gold_answer": "",      # TODO: author this manually based on KB
            "references": "",       # TODO: link policy URLs / doc titles
            "handoff": handoff,     # from policy
            "notes": f"suggested from cluster {int(r['cluster_id'])} (support={r['support']})",
        })
        seen_intents.add(intent)

    # Add any seed intents not already suggested (ensure coverage)
    if not seed_df.empty:
        for intent in sorted(seed_df["intent"].unique().tolist()):
            if intent in seen_intents:
                continue
            handoff = "FALSE" if intent in kb else ("TRUE" if intent in ho else "FALSE")
            gold_rows.append({
                "intent": intent,
                "example_question": "",
                "gold_answer": "",
                "references": "",
                "handoff": handoff,
                "notes": "seed intent (no dominant cluster suggestion)",
            })

    gold_df = pd.DataFrame(gold_rows, columns=[
        "intent", "example_question", "gold_answer", "references", "handoff", "notes"
    ])
    gold_df.to_csv(GOLD_TEMPLATE_CSV, index=False)

    print("✅ Clustering + gold template complete.")
    print(f" - Utterance assignments → {CLUSTERS_CSV}")
    print(f" - Cluster summary → {CLUSTER_SUMMARY_CSV}")
    print(f" - Gold template → {GOLD_TEMPLATE_CSV}")
    print(f"Used k = {k} clusters.")

if __name__ == "__main__":
    main()

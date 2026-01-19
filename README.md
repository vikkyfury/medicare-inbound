# Medicare Inbound — Call Center FAQ Prep

This repo extracts a **Medicare-inbound** slice from a large, real-world call-center corpus, mines customer **intents**, drafts and curates **gold Q&A**, and exports an **evaluation set** you can score any model/agent against (token-F1 / ROUGE-L + handoff accuracy).

---

## Table of Contents

- [What You Get](#what-you-get)  
- [Repo Layout](#repo-layout)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Quickstart](#quickstart)  
- [Pipeline (End-to-End)](#pipeline-end-to-end)  
- [Evaluation](#evaluation)  
- [Makefile (Optional)](#makefile-optional)  
- [Tips, QA & Troubleshooting](#tips-qa--troubleshooting)  
- [Ethics & Compliance](#ethics--compliance)  

---

## What You Get

- Cleaned **Medicare-inbound transcripts** (`jsonl`)
- **Intent mining** (seed rules + optional clustering)
- Auto-drafted Q&A from authoritative CMS/SSA sources, then **curated gold answers**
- **Utterance seeds** per intent for realistic evaluation
- **Eval payload** (`evalset.jsonl`) + offline **evaluator** and reports

## Data & Artifacts (Not in Git)

Large datasets and generated artifacts live under `data/`, `reports/`, `outputs/`, and `package/` and are intentionally **ignored by Git**. Use the scripts to regenerate them locally.

## Key Scripts (01–07)

- `scripts/01_fetch_medicare_inbounds.py` — download & normalize Medicare inbound slice
- `scripts/02_build_intent_dataset.py` — extract customer utterances + seed intents
- `scripts/03_cluster_and_gold_template.py` — optional clustering + gold template
- `scripts/04_prepare_kb_and_autolink.py` — fetch CMS/SSA docs and auto-link intents
- `scripts/05_finalize_gold_answers.py` — curate gold answers
- `scripts/06*_export_eval_payload*.py` — build eval payloads
- `scripts/07*_eval*.py` — run evals / diagnostics

---

## Repo Layout

```
medicare-inbound/
├─ configs/
│  ├─ dataset.yaml              # paths, filters
│  └─ intent_policy.yaml        # (optional) kb_answerable vs handoff
├─ data/
│  ├─ _hf_snapshot/             # HF snapshot (large)
│  ├─ medicare_inbounds/
│  │  ├─ _raw_medicare_json/    # staged json prior to normalization
│  │  └─ medicare_conversations.jsonl
│  ├─ kb/
│  │  ├─ raw/                   # downloaded CMS/SSA (pdf/html/txt)
│  │  └─ md/                    # normalized to Markdown
│  └─ intents/
│     ├─ customer_utterances.csv
│     ├─ medicare_clusters.csv  # (from clustering step)
│     └─ intent_utterances.csv  # mined/synthesized examples
├─ eval/
│  └─ qna/
│     ├─ gold_template.csv
│     ├─ gold_autodraft.csv
│     ├─ gold_answers.csv
│     ├─ gold_answers.cleaned.csv
│     ├─ evalset.jsonl
│     └─ predictions.jsonl      # your model predictions (for eval)
├─ reports/
│  ├─ kb_index.csv
│  ├─ kb_links.csv
│  ├─ gold_answers_qc_report.txt
│  ├─ eval_payload_report.txt
│  ├─ eval_scores.txt
│  ├─ eval_detailed.csv
│  ├─ eval_confusion_by_intent.csv
│  └─ eval_top_errors.csv
├─ scripts/                      # 01..07 pipeline scripts
└─ README.md
```

---

## Prerequisites

- Python **3.10+** (3.11 recommended)
- A virtualenv (e.g., `.venv`) activated
- Sufficient disk space (HF snapshot + ZIP extraction are large)

---

## Installation

```bash
pip install -r requirements.txt
# If no requirements file is provided, minimally:
pip install datasets huggingface_hub pyyaml pandas numpy scikit-learn rouge-score
```

---

## Configuration

Create/edit `configs/dataset.yaml`:

```yaml
dataset_id: "AIxBlock/92k-real-world-call-center-scripts-english"
snapshot_dir: "data/_hf_snapshot"
staged_medicare_raw: "data/medicare_inbounds/_raw_medicare_json"
normalized_jsonl: "data/medicare_inbounds/medicare_conversations.jsonl"
preview_json: "data/medicare_inbounds/_sample_preview.json"
index_csv: "data/medicare_inbounds/_file_index.csv"
domain_filter_regex: "(?i)medicare[_\- ]?inbound"
```

(Optional) declare handoff policies in `configs/intent_policy.yaml`:

```yaml
# Example:
handoff_intents:
  - billing_dispute_complex
  - identity_verification_failed
kb_answerable_intents:
  - part_b_eligibility
  - enrollment_periods
```

---

## Quickstart

If you added the Makefile below:

```bash
cd medicare-inbound
make all
```

Without Makefile, run each script in order (see [Pipeline](#pipeline-end-to-end)).

---

## Pipeline (End-to-End)

> All commands assume repo root: `medicare-inbound/`

### 1) Fetch & Normalize Medicare-Inbound

```bash
python3 scripts/01_fetch_medicare_inbounds.py
```

- Downloads HF snapshot  
- Scans ZIP(s) safely (schema quirks handled)  
- Filters Medicare-inbound files  
- **Writes:** `data/medicare_inbounds/medicare_conversations.jsonl`

---

### 2) Build Intent Dataset (customer utterances + seed labeling)

```bash
python3 scripts/02_build_intent_dataset.py
```

- Extracts likely customer utterances  
- Applies seed rules (keywords) to count intents  
- **Outputs:**  
  - `data/intents/customer_utterances.csv`  
  - `reports/medicare_only_intent_counts.csv`  
  - `eval/qna/intent_seeds.jsonl`

---

### 3) Clustering & Gold Template

```bash
python3 scripts/03_cluster_and_gold_template.py --k 25
```

- Optional clustering to group utterances  
- Drafts gold template skeleton  
- **Outputs:**  
  - `data/intents/medicare_clusters.csv`  
  - `reports/medicare_cluster_summary.csv`  
  - `eval/qna/gold_template.csv`

---

### 4) Knowledge Base Ingest & Auto-Linking

```bash
python3 scripts/04_prepare_kb_and_autolink.py --top-sections 5
```

- Downloads CMS/SSA docs → `data/kb/raw/` and normalizes → `data/kb/md/`  
- Builds KB index and links intents to top sections  
- **Outputs:**  
  - `reports/kb_index.csv`  
  - `reports/kb_links.csv`  
  - `eval/qna/gold_autodraft.csv`

---

### 5) Finalize Gold Answers (human pass)

```bash
python3 scripts/05_finalize_gold_answers.py --min-words 90 --max-words 180
```

- Turn the autodraft into concise, correct answers (cite sections)  
- **Output:** `eval/qna/gold_answers.csv`

---

### 6) QC Gold Answers (length + sources)

```bash
python3 scripts/05b_qc_gold_answers.py --min-words 80 --max-words 180
```

- Expands short, trims long, backfills missing sources; writes summary report  
- **Outputs:**  
  - `eval/qna/gold_answers.cleaned.csv`  
  - `reports/gold_answers_qc_report.txt`

---

### 7) Seed Utterances per Intent (from normalized JSONL)

```bash
python3 scripts/06a_seed_utterances_from_jsonl.py --per-intent 8
```

- Mines plausible customer utterances (keyword filters; avoids agent boilerplate)  
- **Output:** `data/intents/intent_utterances.csv`

---

### 8) Export Evaluation Payload

```bash
python3 scripts/06c_export_eval_payload_v2.py --max-utterances 5
```

- Builds `eval/qna/evalset.jsonl` (+ CSV) with:
  - `intent`, `handoff`, `question`, `answer`, `sources[]`, `utterances[]`
- **Report:** `reports/eval_payload_report.txt`

---

### 9) Close Gaps (ensure utterances/sources = 0)

```bash
python3 scripts/06d_close_gaps.py --per-intent 5
```

- Tops up utterances (mine + synth) and backfills missing sources  
- Rewrites evalset; **report should show:**  
  - `missing_sources: 0`  
  - `no_utterances: 0`

---

### 10) Evaluate a Model (or Dummy Perfect Baseline)

Create a dummy perfect baseline:

```bash
python3 scripts/07a_make_dummy_predictions.py
```

Score predictions:

```bash
python3 scripts/07_eval_runner.py --pred eval/qna/predictions.jsonl
```

**Reports written:**

- `reports/eval_scores.txt` (pass rate + handoff accuracy)  
- `reports/eval_detailed.csv`  
- `reports/eval_confusion_by_intent.csv`  
- `reports/eval_top_errors.csv`

---

## Evaluation

- **Generative Q&A scoring:** token-F1 and ROUGE-L vs `gold_answers.cleaned.csv`  
- **Handoff tracking:** intents marked as *handoff* are excluded from textual scoring; evaluator records **handoff accuracy** separately  
- Use `eval/qna/predictions.jsonl` (your model outputs) with the evaluation runner

---

## Makefile (Optional)

Create a `Makefile` at the repo root:

```makefile
PY ?= python3
EVAL_QNA := eval/qna
REPORTS  := reports
SCRIPTS  := scripts

.PHONY: all fetch intents clusters kb gold qc seed_utts export_eval close_gaps eval_dummy eval clean

all: fetch intents clusters kb gold qc seed_utts export_eval close_gaps eval_dummy

fetch:        ; $(PY) $(SCRIPTS)/01_fetch_medicare_inbounds.py
intents:      ; $(PY) $(SCRIPTS)/02_build_intent_dataset.py
clusters:     ; $(PY) $(SCRIPTS)/03_cluster_and_gold_template.py --k 25
kb:           ; $(PY) $(SCRIPTS)/04_prepare_kb_and_autolink.py --top-sections 5
gold:         ; $(PY) $(SCRIPTS)/05_finalize_gold_answers.py --min-words 90 --max-words 180
qc:           ; $(PY) $(SCRIPTS)/05b_qc_gold_answers.py --min-words 80 --max-words 180
seed_utts:    ; $(PY) $(SCRIPTS)/06a_seed_utterances_from_jsonl.py --per-intent 8
export_eval:  ; $(PY) $(SCRIPTS)/06c_export_eval_payload_v2.py --max-utterances 5
close_gaps:   ; $(PY) $(SCRIPTS)/06d_close_gaps.py --per-intent 5
eval_dummy:   ; $(PY) $(SCRIPTS)/07a_make_dummy_predictions.py && $(PY) $(SCRIPTS)/07_eval_runner.py --pred $(EVAL_QNA)/predictions.jsonl
eval:         ; $(PY) $(SCRIPTS)/07_eval_runner.py --pred $(EVAL_QNA)/predictions.jsonl

clean:
	@echo "Cleaning generated artifacts…"
	@rm -f $(EVAL_QNA)/gold_answers.cleaned.csv
	@rm -f $(EVAL_QNA)/evalset.jsonl $(EVAL_QNA)/evalset.csv
	@rm -f $(EVAL_QNA)/predictions.jsonl
	@rm -f $(REPORTS)/kb_index.csv $(REPORTS)/kb_links.csv
	@rm -f $(REPORTS)/gold_answers_qc_report.txt
	@rm -f $(REPORTS)/eval_payload_report.txt
	@rm -f $(REPORTS)/eval_scores.txt $(REPORTS)/eval_detailed.csv $(REPORTS)/eval_confusion_by_intent.csv $(REPORTS)/eval_top_errors.csv
```
---

## Tips, QA & Troubleshooting

- **Zero utterances?**  
  Confirm `configs/dataset.yaml.normalized_jsonl` points to your actual JSONL and re-run **Step 7**.

- **Sources missing in golds?**  
  Run **Step 9** (`06d_close_gaps.py`) to backfill from `reports/kb_links.csv` or broad KB sections.

- **Over/under long answers?**  
  Re-run **Step 6** with tuned `--min-words/--max-words`.

- **Handoff vs Gold:**  
  Keep **handoff** intents out of textual scoring; the evaluator tracks **handoff accuracy** separately.

- **Repro builds:**  
  Prefer `make all` to regenerate from scratch.

---

## Ethics & Compliance

- The source dataset contains **redacted PII**; still treat transcripts prudently.  
- **Cite** CMS/SSA sections for factual answers; golds should reflect policy-accurate guidance and avoid legal/medical advice.

---

*Happy building and evaluating!*

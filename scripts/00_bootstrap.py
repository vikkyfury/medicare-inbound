#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstraps the medicare-inbound project directory structure.

Run from the project root (the folder named `medicare-inbound`):
    $ python3 scripts/00_bootstrap.py
"""

from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]

DIRS = [
    "configs",
    "data/_hf_snapshot",                     # HF snapshot cache (read-only mirror)
    "data/medicare_inbounds/_raw_medicare_json",  # staged Medicare_inbound JSON before normalization
    "data/kb/raw",                           # raw KB docs (pdf/html/faq)
    "data/kb/clean",                         # cleaned KB docs (txt/markdown)
    "data/gold_answers",                     # curated Q&A / “gold” answers
    "docs",                                  # living docs / notes
    "eval/qna",                              # evaluation assets for Q&A
    "eval/reports",                          # evaluation reports/metrics
    "logs",                                  # pipeline logs
    "notebooks",                             # optional EDA notebooks
    "outputs",                               # intermediate outputs (embedding dumps, etc.)
    "reports",                               # human-readable reports/exports
    "scripts",                               # pipeline scripts
    "src",                                   # python package code
    "src/ingestion",                         # data fetching / normalization code
    "src/intent_mining",                     # keyword/cluster intent discovery
    "src/gold",                              # gold-answer generation/validation helpers
    "src/utils",                             # shared utilities
]

GITIGNORE = """\
# Byte-compiled / cache
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.DS_Store

# Virtual envs
.venv/
venv/

# IDE
.vscode/
.idea/

# Data caches / heavy artifacts
data/_hf_snapshot/
outputs/
logs/
reports/
eval/reports/

# Checkpoints & temp
*.tmp
*.bak
*.log
"""

README = """\
# Medicare Inbound — Call Center FAQ Prep

This repo prepares a Medicare-inbound call center subset, mines intents, and
curates gold Q&A for evaluation.

## Directory Layout (created by `scripts/00_bootstrap.py`)

- `configs/` — YAML configs (paths, thresholds).
- `data/_hf_snapshot/` — Hugging Face dataset snapshot (local mirror; large).
- `data/medicare_inbounds/_raw_medicare_json/` — staged Medicare JSON before normalization.
- `data/kb/` — knowledge base raw → clean docs.
- `data/gold_answers/` — curated Q&A pairs used for evaluation.
- `docs/` — running notes.
- `eval/` — evaluation datasets/outputs.
- `logs/` — pipeline logs.
- `notebooks/` — ad-hoc EDA.
- `outputs/` — intermediate artifacts (embeddings, clusters…).
- `reports/` — human-readable summaries/exports.
- `scripts/` — CLI scripts (bootstrap, fetch, normalize, etc.).
- `src/` — Python package modules (ingestion, intent mining, gold, utils).

## Next Steps

1. `python3 scripts/01_fetch_medicare_inbounds.py`
   - Downloads dataset snapshot
   - Safely scans ZIPs/folders
   - Filters only Medicare_inbound
   - Normalizes to JSONL for intent mining

2. Intent mining (keywords + embeddings), then gold Q&A curation.
"""

CONFIG_DATASET_YAML = """\
# configs/dataset.yaml
dataset_id: "AIxBlock/92k-real-world-call-center-scripts-english"
snapshot_dir: "data/_hf_snapshot"
staged_medicare_raw: "data/medicare_inbounds/_raw_medicare_json"
normalized_jsonl: "data/medicare_inbounds/medicare_conversations.jsonl"
preview_json: "data/medicare_inbounds/_sample_preview.json"
index_csv: "data/medicare_inbounds/_file_index.csv"
domain_filter_regex: "(?i)medicare[_\\- ]?inbound"
"""

PKG_INIT = """\
# Makes `src` a package so we can `python -m src.<module>`
"""

UTILS_INIT = """\
# Shared helpers (logging, IO, text cleaning, etc.) will live here.
"""

INGESTION_INIT = """\
# Ingestion & normalization modules (used by 01_fetch_medicare_inbounds.py)
"""

INTENT_INIT = """\
# Intent mining modules (keyword rules, embedding clustering, labeling)
"""

GOLD_INIT = """\
# Gold answer generation, validation, and export helpers
"""

def touch_gitkeep(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / ".gitkeep").write_text("")

def main():
    # Ensure root exists
    ROOT.mkdir(parents=True, exist_ok=True)

    # Create dirs + .gitkeep
    for rel in DIRS:
        d = ROOT / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").write_text("")

    # Root README
    (ROOT / "README.md").write_text(textwrap.dedent(README).strip() + "\n")

    # .gitignore
    (ROOT / ".gitignore").write_text(textwrap.dedent(GITIGNORE).strip() + "\n")

    # Starter config
    (ROOT / "configs" / "dataset.yaml").write_text(textwrap.dedent(CONFIG_DATASET_YAML))

    # Package inits
    (ROOT / "src" / "__init__.py").write_text(PKG_INIT)
    (ROOT / "src" / "utils" / "__init__.py").write_text(UTILS_INIT)
    (ROOT / "src" / "ingestion" / "__init__.py").write_text(INGESTION_INIT)
    (ROOT / "src" / "intent_mining" / "__init__.py").write_text(INTENT_INIT)
    (ROOT / "src" / "gold" / "__init__.py").write_text(GOLD_INIT)

    print("✅ Project scaffold created.")
    print(f"Root: {ROOT}")
    print("Next: run `python3 scripts/01_fetch_medicare_inbounds.py` (we'll add that next).")

if __name__ == "__main__":
    main()

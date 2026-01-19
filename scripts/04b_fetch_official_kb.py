#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4b — Fetch a starter Medicare KB locally.

What it does
------------
- Creates data/kb/raw/ if needed
- Downloads a curated list of official Medicare/SSA resources (PDF/HTML/TXT)
- (Optional) reads additional sources from configs/kb_sources.yaml
- Writes a manifest to reports/kb_download_manifest.csv

Run
---
$ cd medicare-inbound
$ python3 scripts/04b_fetch_official_kb.py
# (then)
$ python3 scripts/04_prepare_kb_and_autolink.py --top-sections 5

Customize
---------
- Add/edit sources in configs/kb_sources.yaml (see format below).
- Re-run this script; it will only re-download if you pass --force.

configs/kb_sources.yaml format (optional)
-----------------------------------------
# Example:
# sources:
#   - url: "https://www.medicare.gov/Pubs/pdf/10050-medicare-and-you.pdf"
#     filename: "medicare_and_you_2026.pdf"
#     notes: "Core handbook"
#   - url: "https://www.medicare.gov/basics/costs/medicare-costs"
#     filename: "medicare_costs_2025.html"
#     notes: "Costs overview"
"""
from __future__ import annotations
import csv
import hashlib
import mimetypes
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

# Requests with retries
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception as e:
    print("[ERR] This script requires 'requests'. Install it via `pip install requests`.", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "kb" / "raw"
CONF_YAML = ROOT / "configs" / "kb_sources.yaml"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_CSV = REPORTS_DIR / "kb_download_manifest.csv"

# ---------- Default authoritative sources (best-effort) ----------
# Note: These URLs are stable patterns used by Medicare/SSA, but may change.
# If any fail, add/replace via configs/kb_sources.yaml and re-run.
DEFAULT_SOURCES = [
    # Core handbook (CMS Pub. 10050 – latest edition)
    {"url": "https://www.medicare.gov/publications/10050-medicare-and-you.pdf", "filename": "medicare_and_you_2026.pdf", "notes": "Core handbook"},  # :contentReference[oaicite:0]{index=0}

    # Costs & premiums (Medicare + SSA/IRMAA context)
    {"url": "https://www.medicare.gov/basics/costs/medicare-costs", "filename": "medicare_costs_2025.html", "notes": "Costs overview"},  # :contentReference[oaicite:1]{index=1}
    {"url": "https://www.medicare.gov/publications/11579-medicare-costs.pdf", "filename": "medicare_costs_2025.pdf", "notes": "Official costs PDF"},  # :contentReference[oaicite:2]{index=2}
    {"url": "https://www.ssa.gov/benefits/medicare/medicare-premiums.html", "filename": "ssa_medicare_premiums_2025.html", "notes": "SSA premiums & IRMAA brackets"},  # :contentReference[oaicite:3]{index=3}

    # Parts & coverage
    {"url": "https://www.medicare.gov/basics/get-started-with-medicare/medicare-basics/parts-of-medicare", "filename": "parts_of_medicare_overview.html", "notes": "Parts A/B/C/D overview"},  # :contentReference[oaicite:4]{index=4}
    {"url": "https://www.medicare.gov/coverage/find-alphabetically", "filename": "coverage_index.html", "notes": "What’s covered – A–Z index"},  # :contentReference[oaicite:5]{index=5}
    {"url": "https://www.medicare.gov/health-drug-plans/part-d/what-drug-plans-cover", "filename": "part_d_drug_coverage.html", "notes": "Part D basics & formularies"},  # :contentReference[oaicite:6]{index=6}

    # Enrollment windows & plan changes
    {"url": "https://www.medicare.gov/basics/get-started-with-medicare/get-more-coverage/joining-a-plan", "filename": "joining_a_plan.html", "notes": "When/how to join; OEP/IEP"},  # :contentReference[oaicite:7]{index=7}
    {"url": "https://www.medicare.gov/basics/get-started-with-medicare/get-more-coverage/joining-a-plan/special-enrollment-periods", "filename": "special_enrollment_periods.html", "notes": "SEPs & qualifying events"},  # :contentReference[oaicite:8]{index=8}
    {"url": "https://www.medicare.gov/basics/get-started-with-medicare/sign-up/when-does-medicare-coverage-start", "filename": "coverage_start_dates.html", "notes": "GEP/coverage start timing"},  # :contentReference[oaicite:9]{index=9}

    # Appeals & grievances
    {"url": "https://www.medicare.gov/claims-appeals/file-an-appeal", "filename": "appeals_overview.html", "notes": "How to file an appeal"},  # :contentReference[oaicite:10]{index=10}
    {"url": "https://www.medicare.gov/claims-appeals/your-medicare-rights/get-help-with-appeals", "filename": "appeals_get_help.html", "notes": "Get help with appeals"},  # :contentReference[oaicite:11]{index=11}

    # Prior auth / networks / extra benefits (Medicare Advantage)
    {"url": "https://www.medicare.gov/basics/get-started-with-medicare/get-more-coverage/your-coverage-options/compare-original-medicare-medicare-advantage", "filename": "compare_original_vs_ma.html", "notes": "OM vs MA (incl. prior auth)"},
    # :contentReference[oaicite:12]{index=12}
    {"url": "https://www.medicare.gov/health-drug-plans/health-plans", "filename": "ma_health_plans_overview.html", "notes": "MA & other plan types"},  # :contentReference[oaicite:13]{index=13}
    {"url": "https://www.medicare.gov/health-drug-plans/health-plans/your-health-plan-options/compare", "filename": "ma_plan_types_compare.html", "notes": "Compare HMO/PPO/SNP/MSA/PFFS"},  # :contentReference[oaicite:14]{index=14}

    # Member logistics (ID card, address updates via SSA)
    {"url": "https://www.ssa.gov/faqs/en/questions/KA-01735.html", "filename": "ssa_replace_medicare_card.html", "notes": "Replace Medicare card (how-to)"},  # :contentReference[oaicite:15]{index=15}
    {"url": "https://www.ssa.gov/myaccount/", "filename": "ssa_my_account_portal.html", "notes": "my Social Security portal (address, etc.)"},  # :contentReference[oaicite:16]{index=16}
    {"url": "https://www.ssa.gov/personal-record/update-contact-information", "filename": "ssa_update_contact.html", "notes": "Update contact info (beneficiaries)"},  # :contentReference[oaicite:17]{index=17}

    # Extras / “not covered” quick ref
    {"url": "https://www.medicare.gov/basics/get-started-with-medicare/medicare-basics/how-does-medicare-work", "filename": "how_medicare_works.html", "notes": "OM vs MA basics; where extras fit"},  # :contentReference[oaicite:18]{index=18}
]
# ---------- Helpers ----------
def make_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=16)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({
        "User-Agent": "medicare-inbound-kb-fetch/1.0 (+https://example.local)",
        "Accept": "*/*",
        "Connection": "keep-alive",
    })
    return sess

def safe_ext_from_ct(ct: Optional[str], url: str, fallback: str = ".html") -> str:
    if not ct:
        # Guess from URL
        guessed = mimetypes.guess_extension(url.split("?")[0])
        return guessed or fallback
    ct = ct.split(";")[0].strip().lower()
    mapping = {
        "application/pdf": ".pdf",
        "text/html": ".html",
        "application/xhtml+xml": ".html",
        "text/plain": ".txt",
    }
    if ct in mapping:
        return mapping[ct]
    # Last resort: URL-based
    guessed = mimetypes.guess_extension(url.split("?")[0])
    return guessed or fallback

def sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

@dataclass
class SourceItem:
    url: str
    filename: Optional[str] = None
    notes: str = ""

def load_sources() -> List[SourceItem]:
    items: List[SourceItem] = []
    # YAML first (if present)
    if CONF_YAML.exists():
        try:
            data = yaml.safe_load(CONF_YAML.read_text(encoding="utf-8")) or {}
            for row in data.get("sources", []):
                items.append(SourceItem(
                    url=row.get("url", "").strip(),
                    filename=(row.get("filename") or "").strip() or None,
                    notes=row.get("notes") or ""
                ))
        except Exception as e:
            print(f"[WARN] Could not parse {CONF_YAML}: {e}")

    # Add defaults (skip duplicates)
    existing_urls = {s.url for s in items}
    for d in DEFAULT_SOURCES:
        if d["url"] not in existing_urls:
            items.append(SourceItem(url=d["url"], filename=d.get("filename"), notes=d.get("notes","")))
    return items

def download_one(sess: requests.Session, src: SourceItem) -> dict:
    url = src.url
    out_dir = RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        r = sess.get(url, stream=True, timeout=20)
    except Exception as e:
        return {"url": url, "filename": src.filename or "", "status": "error", "detail": f"request-failed: {e}"}

    if r.status_code >= 400:
        return {"url": url, "filename": src.filename or "", "status": "error", "detail": f"HTTP {r.status_code}"}

    ct = r.headers.get("Content-Type", "")
    # Decide filename
    if src.filename:
        fname = src.filename
    else:
        # derive from URL path
        path_part = url.split("?")[0].rstrip("/").split("/")[-1] or f"download-{sha1(url)}"
        # ensure extension
        if "." not in path_part.rsplit("/", 1)[-1]:
            path_part += safe_ext_from_ct(ct, url)
        fname = path_part

    outp = out_dir / fname

    # Stream-save
    try:
        with open(outp, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        return {"url": url, "filename": fname, "status": "error", "detail": f"write-failed: {e}"}

    size = outp.stat().st_size
    status = "ok" if size > 0 else "empty"
    return {"url": url, "filename": fname, "status": status, "detail": f"{ct}; {size} bytes", "notes": src.notes}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-download even if file already exists")
    ap.add_argument("--sleep", type=float, default=0.8, help="Seconds to sleep between downloads (politeness)")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_sources()
    if not sources:
        print("[ERR] No sources found. Add URLs to configs/kb_sources.yaml.", file=sys.stderr)
        sys.exit(1)

    sess = make_session()
    rows = []
    for src in sources:
        # skip if exists (unless --force)
        target = src.filename
        if target:
            outp = RAW_DIR / target
            if outp.exists() and not args.force:
                rows.append({"url": src.url, "filename": target, "status": "skipped", "detail": "exists", "notes": src.notes})
                continue
        res = download_one(sess, src)
        rows.append(res)
        time.sleep(args.sleep)

    # write manifest
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url","filename","status","detail","notes"])
        w.writeheader()
        for r in rows:
            if "notes" not in r:
                r["notes"] = ""
            w.writerow(r)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"✅ Done. Downloaded OK: {ok}/{len(rows)}")
    print(f"Manifest → {MANIFEST_CSV}")
    print(f"Files → {RAW_DIR}")

if __name__ == "__main__":
    main()

# eval/run_eval.py
from __future__ import annotations
import csv, json, time, os, argparse, datetime as dt
from typing import Dict, Any, List
from .bedrock_chain import generate_answer
# Import your chain
# from .bedrock_chain import generate_answer

def load_gold(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # expected columns: question, expected_span, doc_id, year, topic
            rows.append({
                "question": row.get("question","").strip(),
                "expected_span": row.get("expected_span","").strip(),
                "doc_id": row.get("doc_id","").strip(),
                "year": row.get("year","").strip(),
                "topic": row.get("topic","").strip(),
                "part": row.get("part","").strip() if "part" in row else "",
            })
    return rows

def contains_span(text: str, span: str) -> bool:
    if not span: return False
    return span.lower() in (text or "").lower()

def citations_include(citations: List[str], doc_id: str) -> bool:
    if not doc_id: return False
    d = doc_id.lower()
    return any(d in (c or "").lower() for c in citations)

def run_eval(file: str, sleep_s: float = 0.0) -> Dict[str, Any]:
    gold = load_gold(file)
    results: List[Dict[str, Any]] = []
    t_begin = time.time()

    for i, row in enumerate(gold, 1):
        q = row["question"]
        flt = {
            "year": row.get("year") or None,
            "topic": row.get("topic") or None,
            "part": row.get("part") or None,
            "source": None
        }

        t0 = time.time()
        out = generate_answer(q, **flt)
        dt_ms = (time.time() - t0) * 1000.0

        ans = out.get("answer", "")
        cites = out.get("citations", [])
        span_ok = contains_span(ans, row["expected_span"])
        cite_ok = citations_include(cites, row["doc_id"])
        ok = (span_ok and cite_ok)

        results.append({
            "id": i,
            "question": q,
            "answer": ans,
            "chunk_count": out.get("chunk_count", 0),
            "latency_ms": round(dt_ms, 1),
            "span_expected": row["expected_span"],
            "span_ok": bool(span_ok),
            "doc_id": row["doc_id"],
            "cite_ok": bool(cite_ok),
            "ok": bool(ok),
            "citations": cites,
        })

        print(f"[{i:03d}] ok={int(ok)} span={int(span_ok)} cite={int(cite_ok)} "
              f"lat={dt_ms:.1f}ms chunks={out.get('chunk_count',0)}")
        if sleep_s > 0:
            time.sleep(sleep_s)

    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    span_acc = sum(1 for r in results if r["span_ok"]) / max(1, total)
    cite_acc = sum(1 for r in results if r["cite_ok"]) / max(1, total)
    overall  = passed / max(1, total)
    latencies = [r["latency_ms"] for r in results]
    lat_avg = round(sum(latencies)/max(1,len(latencies)), 1)

    summary = {
        "total": total,
        "passed": passed,
        "overall_acc": round(overall, 3),
        "span_acc": round(span_acc, 3),
        "cite_acc": round(cite_acc, 3),
        "avg_latency_ms": lat_avg,
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    # Save JSON results
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_path}")
    return {"summary": summary, "results": results, "path": out_path}

def main():
    ap = argparse.ArgumentParser(description="Run gold Q&A evaluation.")
    ap.add_argument("--file", required=True, help="Path to eval/gold_qa.csv")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep between queries (s)")
    args = ap.parse_args()
    run_eval(args.file, sleep_s=args.sleep)

if __name__ == "__main__":
    main()

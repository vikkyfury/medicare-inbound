# app/cli.py
from time import sleep
import argparse
import os
# Use absolute import since we run with: python -m app.cli
from .bedrock_chain import generate_answer


DEFAULT_QUERIES = [
    "Did Medicare remove all premiums in 2025?",
    "How much did the Part B premium increase from 2024 to 2025?",
]

def run_query(q: str, delay: float, **flt) -> None:
    print("\n========================")
    print("Q:", q)
    out = generate_answer(q, **flt)  # returns dict
    # Pretty print
    print("\nANSWER:\n", out.get("answer", ""))

    # Show rewritten query if provided (useful when testing 3.4)
    if "rewritten" in out and out["rewritten"] and out["rewritten"] != q:
        print("\n(rewritten →)", out["rewritten"])

    citations = out.get("citations", [])
    if citations:
        print("\nSOURCE CHUNKS:")
        for i, c in enumerate(citations, 1):
            print(f"{i}. {c}")
    print(f"\nchunk_count={out.get('chunk_count', 0)}")
    sleep(max(0.0, delay))

def main():
    parser = argparse.ArgumentParser(description="RAG CLI (client-side filters + rerank + rewrite).")
    parser.add_argument("--q", action="append", help="Question. Repeat --q to ask multiple.")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between questions.")
    parser.add_argument("--year", help="Filter: year (e.g., 2025).")
    parser.add_argument("--topic", help="Filter: topic/category (e.g., costs, enrollment).")
    parser.add_argument("--source", help="Filter: source tag (e.g., md, pdf, url).")
    parser.add_argument("--part", help="Filter: Medicare part (A/B/C/D).")
    parser.add_argument("--rewrite", default=None, choices=["off", "heuristic", "llm"],
                        help="Query rewrite mode (default uses REWRITE_MODE env).")
    parser.add_argument("--debug-filter", action="store_true",
                        help="If set, enables filter debugging in retrieve() (env DEBUG_FILTER=1).")
    parser.add_argument("--style", default=None,
                        choices=["default", "csr_short", "medicare_compliance"],
                        help="Answer style prompt pack (default uses ANSWER_STYLE env).")
    parser.add_argument("--show-metrics-path", action="store_true",
                        help="Print the path to logs/metrics.csv after running.")
    parser.add_argument("--eval", dest="eval_file", default=None,
                        help="Run evaluation against a gold CSV instead of single queries.")
    args = parser.parse_args()

    # Quick override for rewrite mode (3.4)
    if args.rewrite:
        os.environ["REWRITE_MODE"] = args.rewrite

    # Optional debug flag consumed inside bedrock_chain.retrieve_chunks
    if args.debug_filter:
        os.environ["DEBUG_FILTER"] = "1"

    if args.style:
        os.environ["ANSWER_STYLE"] = args.style
    if args.show_metrics_path:
        from .metrics import DEFAULT_FILE
        print(f"\n[metrics] {DEFAULT_FILE}")
    if args.eval_file:
        from .run_eval import run_eval
        run_eval(args.eval_file, sleep_s=args.sleep)
        return  # exit after eval

    flt = {"year": args.year, "topic": args.topic, "source": args.source, "part": args.part}
    questions = args.q if args.q else DEFAULT_QUERIES
    for q in questions:
        run_query(q, args.sleep, **flt)

if __name__ == "__main__":
    main()

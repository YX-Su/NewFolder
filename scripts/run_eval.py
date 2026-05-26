"""Run evaluation across variants and dump results."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from mathrag.config import paths
from mathrag.eval import run_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=None, help="JSONL eval set")
    ap.add_argument("--index", type=Path, default=None, help="Index dir")
    ap.add_argument(
        "--variants",
        type=str,
        default="baseline,rag,agent",
        help="comma-separated: baseline,rag,agent",
    )
    ap.add_argument("--no-judge", action="store_true", help="Skip LLM judge (faster, cheaper)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    p = paths()
    eval_path = args.eval or (p.eval / "seed.jsonl")
    index_dir = args.index or p.index
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    out = args.out or (p.eval / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    run_eval(
        eval_path=eval_path,
        index_dir=index_dir,
        variants=variants,
        out_path=out,
        judge=not args.no_judge,
        limit=args.limit,
    )
    print(f"[done] results -> {out}")


if __name__ == "__main__":
    main()

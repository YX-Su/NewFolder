"""Parse a PDF into Markdown (uses MinerU if installed)."""
from __future__ import annotations

import argparse
from pathlib import Path

from mathrag.config import paths
from mathrag.parsing import parse_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = args.out or paths().parsed
    md = parse_pdf(args.pdf, out, force=args.force)
    print(f"[done] markdown at: {md}")


if __name__ == "__main__":
    main()

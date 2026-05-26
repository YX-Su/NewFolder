"""Chunk parsed Markdown(s) and build hybrid index."""
from __future__ import annotations

import argparse
from pathlib import Path

from mathrag.chunking import chunk_markdown
from mathrag.config import paths
from mathrag.indexing import build_index


def collect_markdowns(src: Path) -> list[tuple[str, Path]]:
    """Returns list of (book_name, md_path)."""
    if src.is_file() and src.suffix == ".md":
        return [(src.stem, src)]
    pairs: list[tuple[str, Path]] = []
    for md in src.rglob("*.md"):
        # MinerU layout: <out>/<stem>/auto/<stem>.md → book = stem
        if md.parent.name == "auto":
            book = md.parent.parent.name
        else:
            book = md.stem
        pairs.append((book, md))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None, help="Dir or file of parsed markdown")
    ap.add_argument("--out", type=Path, default=None, help="Index output dir")
    ap.add_argument("--no-expand-nl", action="store_true", help="Skip formula→NL expansion")
    args = ap.parse_args()

    p = paths()
    src = args.src or p.parsed
    out = args.out or p.index

    md_files = collect_markdowns(src)
    if not md_files:
        raise SystemExit(f"No markdown found under {src}")

    all_chunks = []
    for book, md_path in md_files:
        print(f"[chunk] {book}  ({md_path})")
        md = md_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(md, book=book)
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"[index] building over {len(all_chunks)} chunks -> {out}")
    build_index(all_chunks, out, expand_nl=not args.no_expand_nl)
    print("[done]")


if __name__ == "__main__":
    main()

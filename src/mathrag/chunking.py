"""Semantic chunker for Chinese math textbooks.

Splits MinerU-style markdown by:
  - Markdown headers (#, ##, ###) — chapter/section structure
  - Math unit markers within sections — 定义/定理/命题/推论/引理/例/证明/解/注

Each chunk preserves metadata so retrieval can be filtered by chapter/type and
proofs can be linked back to their theorems.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Iterator, Sequence


# ----- Type markers -----
# Order matters: longer/more specific first to avoid 命题 matching 题.
UNIT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("definition", re.compile(r"^\s*(?:【)?\s*定义\s*([\d.]+)?")),
    ("theorem",    re.compile(r"^\s*(?:【)?\s*定理\s*([\d.]+)?")),
    ("proposition", re.compile(r"^\s*(?:【)?\s*命题\s*([\d.]+)?")),
    ("corollary", re.compile(r"^\s*(?:【)?\s*推论\s*([\d.]+)?")),
    ("lemma",     re.compile(r"^\s*(?:【)?\s*引理\s*([\d.]+)?")),
    ("example",   re.compile(r"^\s*(?:【)?\s*例\s*([\d.]+)?")),
    ("proof",     re.compile(r"^\s*证\s*明[\s.：:]")),
    ("solution",  re.compile(r"^\s*解[\s.：:]")),
    ("remark",    re.compile(r"^\s*注[\s.：:\d]")),
]

HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")


@dataclass
class Chunk:
    id: str
    text: str
    natural_language: str = ""  # populated later by LaTeX→NL pass
    type: str = "text"
    book: str = ""
    chapter: str = ""
    section: str = ""
    number: str = ""
    page: int | None = None
    linked_id: str | None = None  # proof/solution → theorem/example id

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "natural_language": self.natural_language,
            "metadata": {
                "type": self.type,
                "book": self.book,
                "chapter": self.chapter,
                "section": self.section,
                "number": self.number,
                "page": self.page,
                "linked_id": self.linked_id,
            },
        }


def _classify_line(line: str) -> tuple[str, str] | None:
    for kind, pat in UNIT_PATTERNS:
        m = pat.match(line)
        if m:
            number = m.group(1) if m.lastindex else ""
            return kind, number or ""
    return None


def _make_id(book: str, kind: str, number: str, salt: str) -> str:
    base = f"{book}_{kind}_{number}".replace(" ", "").replace("/", "_")
    if not number:
        base = f"{base}_{salt}"
    return base or f"chunk_{salt}"


def chunk_markdown(
    md: str,
    *,
    book: str,
    max_chars: int = 1500,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Split a single book's markdown into semantic chunks.

    Headers reset chapter/section. Unit markers (定义/定理/...) start a new chunk.
    Long units are split by max_chars with overlap, sharing the same linked_id.
    """
    lines = md.splitlines()
    chunks: list[Chunk] = []

    chapter = ""
    section = ""
    cur_buf: list[str] = []
    cur_kind = "text"
    cur_number = ""
    cur_link: str | None = None
    last_theorem_like_id: str | None = None  # to link 证明 → 定理

    def flush() -> None:
        nonlocal cur_buf, cur_kind, cur_number, cur_link
        text = "\n".join(cur_buf).strip()
        if not text:
            cur_buf = []
            return

        # Long unit → split with overlap
        pieces = _split_long(text, max_chars, overlap_chars)
        base_id = _make_id(book, cur_kind, cur_number, uuid.uuid4().hex[:6])
        for i, piece in enumerate(pieces):
            cid = base_id if len(pieces) == 1 else f"{base_id}_p{i}"
            chunks.append(
                Chunk(
                    id=cid,
                    text=piece,
                    type=cur_kind,
                    book=book,
                    chapter=chapter,
                    section=section,
                    number=cur_number,
                    linked_id=cur_link if cur_kind in {"proof", "solution"} else None,
                )
            )
        cur_buf = []

    for raw in lines:
        line = raw.rstrip()

        # Header
        h = HEADER_RE.match(line)
        if h:
            flush()
            level, title = len(h.group(1)), h.group(2)
            if level == 1:
                chapter = title
                section = ""
            else:
                section = title
            cur_kind, cur_number, cur_link = "text", "", None
            continue

        # Unit marker
        cls = _classify_line(line)
        if cls:
            flush()
            kind, number = cls
            # Track the most recent theorem-like for linking
            if kind in {"theorem", "proposition", "corollary", "lemma", "example", "definition"}:
                last_theorem_like_id = _make_id(book, kind, number, "next")
                cur_link = None
            elif kind in {"proof", "solution"}:
                cur_link = last_theorem_like_id
            cur_kind, cur_number = kind, number
            cur_buf.append(line)
            continue

        cur_buf.append(line)

    flush()
    return chunks


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    step = max(1, max_chars - overlap)
    for start in range(0, len(text), step):
        end = min(len(text), start + max_chars)
        pieces.append(text[start:end])
        if end == len(text):
            break
    return pieces


def chunks_to_jsonl(chunks: Sequence[Chunk]) -> Iterator[str]:
    import json

    for c in chunks:
        yield json.dumps(c.to_dict(), ensure_ascii=False)

"""Build & load hybrid index: Chroma (dense) + BM25 (sparse over jieba tokens).

Both are persisted to disk so the index step is one-shot.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jieba
from rank_bm25 import BM25Okapi
from tqdm import tqdm

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None  # type: ignore

from .chunking import Chunk
from .embeddings import EmbeddingClient
from .formula_nl import expand_formulas


# Math-aware tokens we want jieba to treat as single words
_MATH_TOKENS = [
    # LaTeX commands
    "\\int", "\\sum", "\\prod", "\\lim", "\\partial", "\\nabla",
    "\\sin", "\\cos", "\\tan", "\\log", "\\ln", "\\exp", "\\sqrt", "\\frac",
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\theta",
    "\\lambda", "\\mu", "\\pi", "\\sigma", "\\infty",
    # Chinese math concepts
    "夹逼定理", "洛必达法则", "中值定理", "罗尔定理", "拉格朗日中值定理",
    "柯西中值定理", "泰勒公式", "麦克劳林公式", "牛顿-莱布尼茨公式",
    "等价无穷小", "数学归纳法", "ε-N", "ε-δ",
    "实对称矩阵", "正定矩阵", "特征值", "特征向量", "线性相关", "线性无关",
    "极大无关组", "矩阵的秩", "初等变换", "可逆矩阵", "正交矩阵",
    "二次型", "标准形", "若尔当标准形", "克莱姆法则",
]


def _init_jieba() -> None:
    for tok in _MATH_TOKENS:
        jieba.add_word(tok, freq=1000)


_init_jieba()


def tokenize_math(text: str) -> list[str]:
    """jieba tokenizer that keeps LaTeX commands as atoms."""
    return [t for t in jieba.lcut(text) if t.strip()]


@dataclass
class HybridIndex:
    chunks: list[Chunk]
    bm25: BM25Okapi
    tokenized: list[list[str]]
    chroma_collection: Any | None  # chromadb Collection

    def bm25_search(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        toks = tokenize_math(query)
        scores = self.bm25.get_scores(toks)
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, float(scores[i])) for i in idx if scores[i] > 0]

    def dense_search(self, query_embedding: list[float], k: int = 30) -> list[tuple[int, float]]:
        if self.chroma_collection is None:
            return []
        res = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        ids = res["ids"][0]
        distances = res["distances"][0]
        id_to_idx = {c.id: i for i, c in enumerate(self.chunks)}
        out: list[tuple[int, float]] = []
        for cid, dist in zip(ids, distances):
            if cid in id_to_idx:
                # Chroma returns distance; convert to similarity ~ 1 - dist (cosine)
                out.append((id_to_idx[cid], float(1.0 - dist)))
        return out


def build_index(
    chunks: list[Chunk],
    out_dir: Path,
    *,
    embedder: EmbeddingClient | None = None,
    expand_nl: bool = True,
) -> HybridIndex:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Optionally expand inline formulas → natural language (DECISIONS §7)
    if expand_nl:
        for c in chunks:
            if "$" in c.text and not c.natural_language:
                c.natural_language = expand_formulas(c.text)

    # 2. Persist chunks
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    # 3. BM25 (over text + NL if available)
    corpus_texts = [
        (c.text + " " + c.natural_language) if c.natural_language else c.text
        for c in chunks
    ]
    tokenized = [tokenize_math(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)

    # 4. Dense (Chroma)
    collection = None
    if chromadb is not None:
        if embedder is None:
            embedder = EmbeddingClient()
        client = chromadb.PersistentClient(
            path=str(out_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            client.delete_collection("mathrag")
        except Exception:
            pass
        collection = client.create_collection(
            name="mathrag",
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 32
        for i in tqdm(range(0, len(chunks), batch_size), desc="embedding"):
            batch = chunks[i : i + batch_size]
            texts = [
                (c.text + "\n" + c.natural_language) if c.natural_language else c.text
                for c in batch
            ]
            embeddings = embedder.embed(texts)
            collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {k: ("" if v is None else v) for k, v in c.to_dict()["metadata"].items()}
                    for c in batch
                ],
            )

    return HybridIndex(chunks=chunks, bm25=bm25, tokenized=tokenized, chroma_collection=collection)


def load_index(index_dir: Path) -> HybridIndex:
    chunks: list[Chunk] = []
    with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta = d.get("metadata", {})
            chunks.append(
                Chunk(
                    id=d["id"],
                    text=d["text"],
                    natural_language=d.get("natural_language", ""),
                    type=meta.get("type", "text"),
                    book=meta.get("book", ""),
                    chapter=meta.get("chapter", ""),
                    section=meta.get("section", ""),
                    number=meta.get("number", ""),
                    page=meta.get("page"),
                    linked_id=meta.get("linked_id"),
                )
            )

    with (index_dir / "bm25.pkl").open("rb") as f:
        bm25_data = pickle.load(f)

    collection = None
    if chromadb is not None and (index_dir / "chroma").exists():
        client = chromadb.PersistentClient(
            path=str(index_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection("mathrag")

    return HybridIndex(
        chunks=chunks,
        bm25=bm25_data["bm25"],
        tokenized=bm25_data["tokenized"],
        chroma_collection=collection,
    )

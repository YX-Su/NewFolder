"""Retrieval pipeline: query rewrite → hybrid (BM25 + dense) → rerank → top-k."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

from .chunking import Chunk
from .embeddings import EmbeddingClient, RerankClient
from .indexing import HybridIndex
from .llm import LLMClient, default_llm
from .prompts import QUERY_REWRITE_SYSTEM


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    source: str  # "bm25" | "dense" | "rerank"


@dataclass
class RetrievalConfig:
    bm25_k: int = 30
    dense_k: int = 30
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    final_k: int = 5
    use_rewrite: bool = True
    use_rerank: bool = True


@dataclass
class Retriever:
    index: HybridIndex
    embedder: EmbeddingClient = field(default_factory=EmbeddingClient)
    reranker: RerankClient | None = field(default_factory=RerankClient)
    llm: LLMClient = field(default_factory=default_llm)
    config: RetrievalConfig = field(default_factory=RetrievalConfig)

    # ---------- public ----------

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        queries = self._rewrite(query) if self.config.use_rewrite else [query]
        fused = self._fuse(queries)
        if self.config.use_rerank and self.reranker is not None and fused:
            fused = self._rerank(query, fused)
        return fused[: self.config.final_k]

    # ---------- internals ----------

    def _rewrite(self, query: str) -> list[str]:
        try:
            res = self.llm.chat(
                [
                    {"role": "system", "content": QUERY_REWRITE_SYSTEM},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=300,
            )
            data = json.loads(res.content)
            queries = list(data.get("queries", []))
            concepts = list(data.get("concepts", []))
            out = [query] + [q for q in queries if isinstance(q, str) and q.strip()]
            # Concepts go in too — BM25 loves these
            out += [c for c in concepts if isinstance(c, str) and c.strip()]
            # dedup, keep order
            seen: set[str] = set()
            uniq: list[str] = []
            for q in out:
                if q not in seen:
                    seen.add(q)
                    uniq.append(q)
            return uniq[:4]
        except Exception:
            return [query]

    def _fuse(self, queries: Sequence[str]) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion over BM25 and dense, across multiple queries."""
        k = self.config.rrf_k
        scores: dict[int, float] = {}
        sources: dict[int, str] = {}

        for q in queries:
            # BM25
            for rank, (idx, _) in enumerate(self.index.bm25_search(q, k=self.config.bm25_k)):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
                sources.setdefault(idx, "bm25")

            # Dense
            if self.index.chroma_collection is not None:
                emb = self.embedder.embed_one(q)
                for rank, (idx, _) in enumerate(
                    self.index.dense_search(emb, k=self.config.dense_k)
                ):
                    scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
                    sources[idx] = "hybrid" if sources.get(idx) == "bm25" else "dense"

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [
            RetrievedChunk(chunk=self.index.chunks[idx], score=s, source=sources[idx])
            for idx, s in ranked
        ]

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        # Cap input to reranker — bge-reranker is slow over too many docs
        head = candidates[:30]
        docs = [c.chunk.text[:1500] for c in head]
        try:
            ranks = self.reranker.rerank(query, docs, top_n=len(docs))
        except Exception:
            return candidates
        reranked: list[RetrievedChunk] = []
        for orig_idx, score in ranks:
            rc = head[orig_idx]
            reranked.append(RetrievedChunk(chunk=rc.chunk, score=float(score), source="rerank"))
        return reranked + candidates[30:]

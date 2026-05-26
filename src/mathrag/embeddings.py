"""Embedding + Rerank clients. Both go through SiliconFlow OpenAI-compatible API."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests
from openai import OpenAI

from .config import EmbeddingConfig, embedding_config


@dataclass
class EmbeddingClient:
    cfg: EmbeddingConfig = field(default_factory=embedding_config)
    _client: OpenAI = field(init=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)

    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for attempt in range(3):
                try:
                    resp = self._client.embeddings.create(
                        model=self.cfg.embedding_model,
                        input=chunk,
                    )
                    out.extend([d.embedding for d in resp.data])
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@dataclass
class RerankClient:
    """SiliconFlow exposes rerank via /v1/rerank (not strictly OpenAI-compatible)."""

    cfg: EmbeddingConfig = field(default_factory=embedding_config)

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        """Returns list of (original_index, score) sorted by score desc."""
        url = f"{self.cfg.base_url.rstrip('/')}/rerank"
        payload: dict[str, Any] = {
            "model": self.cfg.rerank_model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        for attempt in range(3):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                return [(r["index"], r["relevance_score"]) for r in results]
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return []

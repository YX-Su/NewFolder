"""Central config. All env vars resolved here; rest of the code imports from here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    reasoner_model: str | None = None


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str
    base_url: str
    embedding_model: str
    rerank_model: str


@dataclass(frozen=True)
class Paths:
    raw: Path
    parsed: Path
    index: Path
    eval: Path


def _env(key: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val or ""


def llm_config() -> LLMConfig:
    return LLMConfig(
        api_key=_env("DEEPSEEK_API_KEY", required=True),
        base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=_env("DEEPSEEK_MODEL", "deepseek-chat"),
        reasoner_model=_env("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
    )


def embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        api_key=_env("SILICONFLOW_API_KEY", required=True),
        base_url=_env("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-m3"),
        rerank_model=_env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
    )


def paths() -> Paths:
    return Paths(
        raw=ROOT / _env("RAW_DIR", "data/raw"),
        parsed=ROOT / _env("PARSED_DIR", "data/parsed"),
        index=ROOT / _env("INDEX_DIR", "data/index"),
        eval=ROOT / _env("EVAL_DIR", "data/eval"),
    )

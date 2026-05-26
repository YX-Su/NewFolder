"""LLM client. OpenAI-compatible — works for DeepSeek / Qwen / Kimi / SiliconFlow."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from .config import LLMConfig, llm_config


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Any = None


@dataclass
class LLMClient:
    cfg: LLMConfig = field(default_factory=llm_config)
    _client: OpenAI = field(init=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict | None = None,
        max_retries: int = 3,
    ) -> ChatResult:
        model = model or self.cfg.model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                usage = resp.usage
                return ChatResult(
                    content=resp.choices[0].message.content or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                    raw=resp,
                )
            except (APITimeoutError, RateLimitError, APIError) as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")

    def reasoner_chat(self, messages: list[dict[str, str]], **kw: Any) -> ChatResult:
        return self.chat(messages, model=self.cfg.reasoner_model, **kw)


_default: LLMClient | None = None


def default_llm() -> LLMClient:
    global _default
    if _default is None:
        _default = LLMClient()
    return _default

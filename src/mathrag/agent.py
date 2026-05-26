"""Solver agent. Hand-rolled ReAct loop — single agent, multiple tools.

Single-agent by design (see DECISIONS §5). Loop:
  1. LLM plans next step in JSON: {thought, action, args}
  2. We dispatch action; if final_answer → return; else feed result back as observation
  3. Repeat up to MAX_STEPS
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import LLMClient, default_llm
from .prompts import AGENT_SYSTEM, BASELINE, SOLVE
from .retrieval import Retriever
from .tools import ToolResult, make_retrieve_tool, verify_with_sympy


# ---------- Trace types ----------

@dataclass
class Step:
    thought: str
    action: str
    args: dict
    observation: dict | None = None

    def to_dict(self) -> dict:
        return {
            "thought": self.thought,
            "action": self.action,
            "args": self.args,
            "observation": self.observation,
        }


@dataclass
class SolveResult:
    answer: str
    trace: list[Step]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    variant: str = "agent"
    retrieved_chunk_ids: list[str] = field(default_factory=list)


# ---------- Variants ----------

def solve_baseline(problem: str, llm: LLMClient | None = None) -> SolveResult:
    llm = llm or default_llm()
    res = llm.chat(BASELINE.render(problem=problem), temperature=0.0, max_tokens=2048)
    return SolveResult(
        answer=res.content,
        trace=[],
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        variant="baseline",
    )


def solve_rag(problem: str, retriever: Retriever, llm: LLMClient | None = None) -> SolveResult:
    llm = llm or default_llm()
    retrieved = retriever.retrieve(problem)
    context = _format_context(retrieved)
    res = llm.chat(
        SOLVE.render(problem=problem, context=context),
        temperature=0.0,
        max_tokens=2048,
    )
    return SolveResult(
        answer=res.content,
        trace=[],
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        variant="rag",
        retrieved_chunk_ids=[r.chunk.id for r in retrieved],
    )


def _format_context(retrieved) -> str:
    if not retrieved:
        return "（无检索到的教材片段）"
    parts = []
    for i, r in enumerate(retrieved, 1):
        c = r.chunk
        header = f"[{i}] {c.book} · {c.chapter} · {c.section} · {c.number} ({c.type})".strip()
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


# ---------- Agent ----------

@dataclass
class SolverAgent:
    retriever: Retriever
    llm: LLMClient = field(default_factory=default_llm)
    max_steps: int = 6

    def solve(self, problem: str) -> SolveResult:
        tools: dict[str, Callable[..., ToolResult]] = {
            "retrieve_textbook": make_retrieve_tool(self.retriever),
            "verify_with_sympy": verify_with_sympy,
        }

        trace: list[Step] = []
        retrieved_ids: list[str] = []
        history: list[dict[str, str]] = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": f"题目：{problem}"},
        ]
        total_prompt = 0
        total_completion = 0

        for step_i in range(self.max_steps):
            res = self.llm.chat(
                history,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1024,
            )
            total_prompt += res.prompt_tokens
            total_completion += res.completion_tokens

            try:
                decision = json.loads(res.content)
            except json.JSONDecodeError:
                # Recover: ask LLM to wrap up
                history.append({"role": "assistant", "content": res.content})
                history.append({
                    "role": "user",
                    "content": "你的上一条输出不是合法 JSON。请直接输出 final_answer。",
                })
                continue

            action = decision.get("action", "")
            args = decision.get("args", {}) or {}
            thought = decision.get("thought", "")
            step = Step(thought=thought, action=action, args=args)

            if action == "final_answer":
                step.observation = {"answer": args.get("answer", "")}
                trace.append(step)
                return SolveResult(
                    answer=args.get("answer", "") + "\n\n" + args.get("reasoning", ""),
                    trace=trace,
                    prompt_tokens=total_prompt,
                    completion_tokens=total_completion,
                    variant="agent",
                    retrieved_chunk_ids=retrieved_ids,
                )

            if action not in tools:
                history.append({"role": "assistant", "content": res.content})
                history.append({
                    "role": "user",
                    "content": f"未知动作 {action}。可用：retrieve_textbook, verify_with_sympy, final_answer。",
                })
                trace.append(step)
                continue

            try:
                tool_result = tools[action](**args)
            except TypeError as e:
                tool_result = ToolResult(ok=False, payload=None, error=f"参数错误: {e}")

            step.observation = tool_result.to_dict()
            trace.append(step)

            if action == "retrieve_textbook" and tool_result.ok:
                retrieved_ids.extend(item["id"] for item in tool_result.payload or [])

            history.append({"role": "assistant", "content": res.content})
            history.append({
                "role": "user",
                "content": "工具返回：\n" + json.dumps(tool_result.to_dict(), ensure_ascii=False)[:3000],
            })

        # Out of steps — force a final answer
        forced = self.llm.chat(
            history + [{
                "role": "user",
                "content": "已达最大步数。请立即输出 final_answer JSON。",
            }],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1024,
        )
        total_prompt += forced.prompt_tokens
        total_completion += forced.completion_tokens
        try:
            d = json.loads(forced.content)
            ans = d.get("args", {}).get("answer", forced.content)
        except Exception:
            ans = forced.content
        return SolveResult(
            answer=ans,
            trace=trace,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            variant="agent",
            retrieved_chunk_ids=retrieved_ids,
        )

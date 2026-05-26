"""Evaluation framework. Three variants (baseline / rag / agent) × multi-metric.

Metrics (DECISIONS §6):
  - retrieval_recall@k     : ground-truth chunk ids ⊂ retrieved set?
  - final_answer_match     : sympy.simplify(answer - ground_truth) == 0
  - faithfulness@judge     : LLM judge 0-5
  - tokens_per_problem
  - latency
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import sympy as sp

from .agent import SolveResult, SolverAgent, solve_baseline, solve_rag
from .indexing import HybridIndex, load_index
from .llm import LLMClient, default_llm
from .prompts import JUDGE
from .retrieval import Retriever


# ---------- Eval set ----------

@dataclass
class EvalItem:
    id: str
    problem: str
    topic: str = ""
    ground_truth_answer: str = ""
    ground_truth_chunks: list[str] = field(default_factory=list)
    rubric: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "EvalItem":
        return cls(
            id=d["id"],
            problem=d["problem"],
            topic=d.get("topic", ""),
            ground_truth_answer=d.get("ground_truth_answer", ""),
            ground_truth_chunks=list(d.get("ground_truth_chunks", [])),
            rubric=d.get("rubric", ""),
        )


def load_eval_set(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            items.append(EvalItem.from_dict(json.loads(line)))
    return items


# ---------- Metrics ----------

_BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text: str) -> str | None:
    m = _BOX_RE.search(text)
    return m.group(1).strip() if m else None


def answer_match(predicted: str, ground_truth: str) -> bool:
    """Try numeric equality, then sympy symbolic equality, then string equality."""
    if not ground_truth:
        return False
    pred = extract_boxed(predicted) or predicted.strip()
    gt = ground_truth.strip()

    if pred == gt:
        return True
    # Numeric
    try:
        if abs(float(pred) - float(gt)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    # Sympy
    try:
        diff = sp.simplify(sp.sympify(pred, evaluate=True) - sp.sympify(gt, evaluate=True))
        if diff == 0:
            return True
    except Exception:
        pass
    return False


def retrieval_recall(retrieved_ids: list[str], gt_ids: list[str]) -> float:
    if not gt_ids:
        return float("nan")
    hits = sum(1 for g in gt_ids if g in retrieved_ids)
    return hits / len(gt_ids)


# ---------- LLM judge ----------

def judge_solution(
    item: EvalItem, solution: str, llm: LLMClient | None = None
) -> dict[str, Any]:
    llm = llm or default_llm()
    res = llm.chat(
        JUDGE.render(problem=item.problem, rubric=item.rubric, solution=solution),
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=500,
    )
    try:
        return json.loads(res.content)
    except json.JSONDecodeError:
        return {
            "final_answer_correct": False,
            "reasoning_score": 0,
            "faithfulness_score": 0,
            "errors": ["judge_parse_failed"],
            "comment": res.content[:200],
        }


# ---------- Runner ----------

@dataclass
class ItemResult:
    item_id: str
    variant: str
    answer_match: bool
    retrieval_recall: float
    judge: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    answer: str
    trace: list[dict] = field(default_factory=list)


VARIANTS: dict[str, Callable] = {}  # populated below


def _make_solver(variant: str, index: HybridIndex | None, llm: LLMClient):
    if variant == "baseline":
        return lambda problem: solve_baseline(problem, llm=llm)
    if variant == "rag":
        retriever = Retriever(index=index)
        return lambda problem: solve_rag(problem, retriever=retriever, llm=llm)
    if variant == "agent":
        retriever = Retriever(index=index)
        agent = SolverAgent(retriever=retriever, llm=llm)
        return lambda problem: agent.solve(problem)
    raise ValueError(f"Unknown variant: {variant}")


def run_eval(
    eval_path: Path,
    index_dir: Path | None,
    variants: list[str],
    *,
    out_path: Path | None = None,
    judge: bool = True,
    limit: int | None = None,
) -> list[ItemResult]:
    items = load_eval_set(eval_path)
    if limit:
        items = items[:limit]
    index = load_index(index_dir) if (index_dir and any(v != "baseline" for v in variants)) else None
    llm = default_llm()

    results: list[ItemResult] = []
    for variant in variants:
        solver = _make_solver(variant, index, llm)
        for item in items:
            t0 = time.time()
            sr: SolveResult = solver(item.problem)
            dt = time.time() - t0

            am = answer_match(sr.answer, item.ground_truth_answer)
            rr = (
                retrieval_recall(sr.retrieved_chunk_ids, item.ground_truth_chunks)
                if variant != "baseline"
                else float("nan")
            )
            judge_data: dict[str, Any] = {}
            if judge:
                judge_data = judge_solution(item, sr.answer, llm=llm)

            results.append(
                ItemResult(
                    item_id=item.id,
                    variant=variant,
                    answer_match=am,
                    retrieval_recall=rr,
                    judge=judge_data,
                    prompt_tokens=sr.prompt_tokens,
                    completion_tokens=sr.completion_tokens,
                    latency_s=dt,
                    answer=sr.answer,
                    trace=[s.to_dict() for s in sr.trace],
                )
            )
            print(
                f"[{variant}] {item.id}: match={am} recall={rr:.2f if rr==rr else float('nan')} "
                f"latency={dt:.1f}s tokens={sr.prompt_tokens + sr.completion_tokens}"
            )

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print_summary(results)
    return results


def print_summary(results: list[ItemResult]) -> None:
    from collections import defaultdict

    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r.variant]["answer_match"].append(1.0 if r.answer_match else 0.0)
        if r.retrieval_recall == r.retrieval_recall:  # not nan
            agg[r.variant]["retrieval_recall"].append(r.retrieval_recall)
        if r.judge:
            agg[r.variant]["reasoning_score"].append(float(r.judge.get("reasoning_score", 0)))
            agg[r.variant]["faithfulness_score"].append(float(r.judge.get("faithfulness_score", 0)))
        agg[r.variant]["latency_s"].append(r.latency_s)
        agg[r.variant]["tokens"].append(float(r.prompt_tokens + r.completion_tokens))

    print("\n=== Summary ===")
    print(f"{'variant':<10}{'match':>8}{'recall':>10}{'reason':>8}{'faith':>8}{'lat_s':>8}{'tokens':>10}")
    for v, m in agg.items():
        def avg(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")
        print(
            f"{v:<10}"
            f"{avg(m['answer_match']):>8.2f}"
            f"{avg(m['retrieval_recall']):>10.2f}"
            f"{avg(m['reasoning_score']):>8.2f}"
            f"{avg(m['faithfulness_score']):>8.2f}"
            f"{avg(m['latency_s']):>8.1f}"
            f"{avg(m['tokens']):>10.0f}"
        )

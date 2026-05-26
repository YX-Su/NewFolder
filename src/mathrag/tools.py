"""Agent tools. Each tool is a plain function that takes/returns JSON-able dicts."""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Callable

import sympy as sp

from .retrieval import Retriever


@dataclass
class ToolResult:
    ok: bool
    payload: Any
    error: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "payload": self.payload, "error": self.error}


def make_retrieve_tool(retriever: Retriever) -> Callable[..., ToolResult]:
    def retrieve_textbook(query: str, k: int = 5) -> ToolResult:
        try:
            results = retriever.retrieve(query)
            payload = [
                {
                    "id": r.chunk.id,
                    "type": r.chunk.type,
                    "book": r.chunk.book,
                    "chapter": r.chunk.chapter,
                    "section": r.chunk.section,
                    "number": r.chunk.number,
                    "text": r.chunk.text[:1200],
                    "score": round(r.score, 4),
                }
                for r in results[:k]
            ]
            return ToolResult(ok=True, payload=payload)
        except Exception as e:
            return ToolResult(ok=False, payload=None, error=str(e))

    return retrieve_textbook


def verify_with_sympy(expression: str, claim: str = "") -> ToolResult:
    """Evaluate a SymPy expression and compare to a claim.

    The agent passes raw Python/SymPy code in `expression`. We exec it in a
    restricted namespace and return the result of the last expression statement.

    `claim` is optional natural-language description for logging.
    """
    # Pre-declare common symbols so `limit(sin(x)/x, x, 0)` works without
    # the agent having to remember to declare x first.
    _common_syms = sp.symbols("x y z t u v w n k m a b c p q r s")
    safe_globals: dict[str, Any] = {
        "__builtins__": {"abs": abs, "len": len, "range": range, "min": min, "max": max},
        "sp": sp,
        "Symbol": sp.Symbol,
        "symbols": sp.symbols,
        "Matrix": sp.Matrix,
        "I": sp.I,
        "pi": sp.pi,
        "E": sp.E,
        "oo": sp.oo,
        "Rational": sp.Rational,
        **{s.name: s for s in _common_syms},
        "sqrt": sp.sqrt,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "log": sp.log, "ln": sp.log, "exp": sp.exp,
        "limit": sp.limit, "Limit": sp.Limit,
        "diff": sp.diff, "Derivative": sp.Derivative,
        "integrate": sp.integrate, "Integral": sp.Integral,
        "Sum": sp.Sum, "summation": sp.summation,
        "solve": sp.solve, "simplify": sp.simplify,
        "factor": sp.factor, "expand": sp.expand,
        "series": sp.series, "Poly": sp.Poly,
        "eigenvals": lambda m: sp.Matrix(m).eigenvals(),
        "eigenvects": lambda m: sp.Matrix(m).eigenvects(),
        "det": lambda m: sp.Matrix(m).det(),
        "rank": lambda m: sp.Matrix(m).rank(),
    }
    safe_locals: dict[str, Any] = {}

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            # Try eval first (single expression), fall back to exec
            try:
                value = eval(expression, safe_globals, safe_locals)
            except SyntaxError:
                exec(expression, safe_globals, safe_locals)
                value = safe_locals.get("result", None)
        return ToolResult(
            ok=True,
            payload={
                "result": str(value),
                "claim": claim,
                "stdout": buf.getvalue(),
            },
        )
    except Exception as e:
        return ToolResult(ok=False, payload={"stdout": buf.getvalue()}, error=f"{type(e).__name__}: {e}")

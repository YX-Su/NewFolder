"""LaTeX → Chinese natural language. Regex covers ~80%, LLM fallback for the rest.

This is the dual-indexing trick from DECISIONS §7. Chunks that contain formulas
get a `natural_language` field that we embed alongside the original text, so
queries phrased in words can still hit formula-heavy chunks.
"""
from __future__ import annotations

import re

# (pattern, replacement) — applied in order. Replacements may be functions.
_SIMPLE_RULES: list[tuple[re.Pattern[str], str]] = [
    # Limits
    (re.compile(r"\\lim_\{([^}]+)\\to\s*([^}]+)\}"), r"\1 趋于 \2 时的极限 "),
    (re.compile(r"\\lim"), "极限 "),
    # Sums / integrals
    (re.compile(r"\\int_\{?([^}\s]+)\}?\^\{?([^}\s]+)\}?"), r"从 \1 到 \2 的定积分 "),
    (re.compile(r"\\int"), "积分 "),
    (re.compile(r"\\sum_\{?([^}\s]+)\}?\^\{?([^}\s]+)\}?"), r"从 \1 到 \2 求和 "),
    (re.compile(r"\\sum"), "求和 "),
    (re.compile(r"\\prod"), "求积 "),
    # Derivatives
    (re.compile(r"\\frac\{d([^}]+)\}\{d([^}]+)\}"), r"\1 对 \2 的导数 "),
    (re.compile(r"\\partial"), "偏 "),
    (re.compile(r"\\nabla"), "梯度 "),
    # Common functions
    (re.compile(r"\\sin"), "sin "),
    (re.compile(r"\\cos"), "cos "),
    (re.compile(r"\\tan"), "tan "),
    (re.compile(r"\\log"), "log "),
    (re.compile(r"\\ln"), "ln "),
    (re.compile(r"\\exp"), "exp "),
    (re.compile(r"\\sqrt\{([^}]+)\}"), r"根号 \1 "),
    # Greek (subset)
    (re.compile(r"\\alpha"), "α"),
    (re.compile(r"\\beta"), "β"),
    (re.compile(r"\\gamma"), "γ"),
    (re.compile(r"\\delta"), "δ"),
    (re.compile(r"\\epsilon|\\varepsilon"), "ε"),
    (re.compile(r"\\theta"), "θ"),
    (re.compile(r"\\lambda"), "λ"),
    (re.compile(r"\\mu"), "μ"),
    (re.compile(r"\\pi"), "π"),
    (re.compile(r"\\sigma"), "σ"),
    (re.compile(r"\\infty"), "无穷 "),
    # Relations
    (re.compile(r"\\leq|\\le"), "≤"),
    (re.compile(r"\\geq|\\ge"), "≥"),
    (re.compile(r"\\neq|\\ne"), "≠"),
    (re.compile(r"\\to"), "趋于 "),
    (re.compile(r"\\Rightarrow"), "推出 "),
    (re.compile(r"\\Leftrightarrow"), "等价于 "),
    # Frac (after derivative form)
    (re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}"), r"\1 比 \2 "),
    # Cleanup
    (re.compile(r"\\,|\\;|\\!|\\:"), " "),
    (re.compile(r"\\left|\\right"), ""),
    (re.compile(r"\\(?:cdot|times)"), "乘 "),
    (re.compile(r"\\(?:displaystyle|textstyle)"), ""),
    (re.compile(r"[{}]"), ""),
    (re.compile(r"\s+"), " "),
]

_INLINE_LATEX_RE = re.compile(r"\$\$?(.+?)\$\$?", re.DOTALL)


def latex_to_nl(s: str) -> str:
    """Convert one LaTeX fragment (no surrounding $) to Chinese natural language."""
    out = s
    for pat, rep in _SIMPLE_RULES:
        out = pat.sub(rep, out)
    return out.strip()


def expand_formulas(text: str) -> str:
    """For each $...$ in text, append a natural-language version inline.

    Original: `求 $\\lim_{x\\to 0} \\frac{\\sin x}{x}$ 的值`
    Output:   `求 $\\lim_{x\\to 0} \\frac{\\sin x}{x}$ (x 趋于 0 时的极限 sin x 比 x) 的值`
    """
    def repl(m: re.Match[str]) -> str:
        latex = m.group(1)
        nl = latex_to_nl(latex)
        return f"{m.group(0)} ({nl})"

    return _INLINE_LATEX_RE.sub(repl, text)

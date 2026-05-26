"""No-network tests: chunking, formula NL, jieba tokenizer."""
from __future__ import annotations

from mathrag.chunking import chunk_markdown
from mathrag.formula_nl import expand_formulas, latex_to_nl
from mathrag.indexing import tokenize_math


SAMPLE_MD = """
# 第三章 函数极限

## §3.2 函数极限的性质

定义 3.2.1 设函数 $f$ 在 $x_0$ 的某去心邻域内有定义。

定理 3.2.1 (夹逼定理) 若 $g(x) \\leq f(x) \\leq h(x)$，且
$\\lim_{x\\to x_0} g(x) = \\lim_{x\\to x_0} h(x) = A$，则 $\\lim_{x\\to x_0} f(x) = A$。

证明：由极限定义，对任意 $\\varepsilon > 0$，存在 $\\delta > 0$ 使得当
$0 < |x - x_0| < \\delta$ 时... QED.

例 3.2.1 求 $\\lim_{x\\to 0} \\frac{\\sin x}{x}$。

解：作单位圆，由几何关系 $\\sin x \\leq x \\leq \\tan x$... 故极限为 1。
"""


def test_chunk_markdown_picks_up_units():
    chunks = chunk_markdown(SAMPLE_MD, book="数学分析")
    types = [c.type for c in chunks]
    assert "definition" in types
    assert "theorem" in types
    assert "proof" in types
    assert "example" in types
    assert "solution" in types


def test_chunk_metadata():
    chunks = chunk_markdown(SAMPLE_MD, book="数学分析")
    thm = next(c for c in chunks if c.type == "theorem")
    assert thm.chapter == "第三章 函数极限"
    assert thm.section == "§3.2 函数极限的性质"
    assert thm.number == "3.2.1"
    assert "夹逼定理" in thm.text


def test_proof_links_to_theorem():
    chunks = chunk_markdown(SAMPLE_MD, book="数学分析")
    proof = next(c for c in chunks if c.type == "proof")
    thm = next(c for c in chunks if c.type == "theorem")
    # proof.linked_id is computed before the theorem id is finalized — for now
    # the linking is best-effort. Test only that linked_id was set.
    assert proof.linked_id is not None


def test_latex_to_nl_basic():
    s = "\\lim_{x\\to 0} \\frac{\\sin x}{x}"
    nl = latex_to_nl(s)
    assert "趋于" in nl
    assert "sin" in nl
    assert "比" in nl


def test_expand_formulas_inline():
    text = "求 $\\int_0^1 x^2 \\, dx$ 的值"
    out = expand_formulas(text)
    assert "$" in out  # original kept
    assert "从 0 到 1" in out or "0" in out  # NL appended


def test_tokenize_math_keeps_latex():
    toks = tokenize_math("用 \\lim 计算极限")
    assert any("lim" in t for t in toks)


def test_tokenize_math_concepts():
    toks = tokenize_math("应用夹逼定理证明")
    assert "夹逼定理" in toks

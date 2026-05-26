"""Metrics tests — no network."""
from __future__ import annotations

from mathrag.eval import answer_match, extract_boxed, retrieval_recall


def test_extract_boxed():
    assert extract_boxed(r"答案 $\boxed{42}$") == "42"
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed("没有 boxed") is None


def test_answer_match_string():
    assert answer_match(r"\boxed{3}", "3") is True


def test_answer_match_numeric():
    assert answer_match(r"\boxed{0.5}", "1/2") is True


def test_answer_match_symbolic():
    assert answer_match(r"\boxed{(x-1)*exp(x)}", "(x-1)*exp(x)") is True
    # x*exp(x) - exp(x) is the same thing
    assert answer_match(r"\boxed{x*exp(x) - exp(x)}", "(x-1)*exp(x)") is True


def test_answer_match_negative():
    assert answer_match(r"\boxed{2}", "3") is False


def test_retrieval_recall():
    assert retrieval_recall(["a", "b", "c"], ["a", "b"]) == 1.0
    assert retrieval_recall(["a", "b", "c"], ["a", "z"]) == 0.5
    assert retrieval_recall(["a", "b"], []) != retrieval_recall(["a", "b"], [])  # nan != nan

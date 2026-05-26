"""SymPy tool tests — no network."""
from __future__ import annotations

from mathrag.tools import verify_with_sympy


def test_sympy_limit():
    r = verify_with_sympy("limit(sin(3*x)/x, x, 0)", claim="lim sin(3x)/x at 0 = 3")
    assert r.ok
    assert "3" in r.payload["result"]


def test_sympy_integral():
    r = verify_with_sympy("integrate(x**2, (x, 0, 1))", claim="∫_0^1 x^2 dx = 1/3")
    assert r.ok
    assert "1/3" in r.payload["result"]


def test_sympy_eigen():
    r = verify_with_sympy("eigenvals(Matrix([[2,1],[1,2]]))")
    assert r.ok
    # eigenvalues are 1 and 3
    assert "1" in r.payload["result"] and "3" in r.payload["result"]


def test_sympy_blocks_builtins():
    r = verify_with_sympy("open('/etc/passwd')")
    assert not r.ok

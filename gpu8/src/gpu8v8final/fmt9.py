"""Number formatting of the SARA manuscript.

Negative numbers use U+2212; a value that rounds to zero is never printed with a sign; an interval whose
bounds round to zero gets one more decimal.  Every formatted number is recorded in TRACE (text -> value),
which verify9 uses to check that each number in the manuscript comes from the results.
"""
from __future__ import annotations

import math

TRACE: dict[str, list[float]] = {}
MINUS = "−"


def _rec(s: str, v) -> str:
    try:
        TRACE.setdefault(s, []).append(float(v))
    except (TypeError, ValueError):
        pass
    return s


def _raw(x: float, d: int) -> str:
    s = f"{x:.{d}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return s.replace("-", MINUS)


def num(x, d: int = 3) -> str:
    return _rec(_raw(float(x), d), x)


def sgn(x, d: int = 3) -> str:
    x = float(x)
    s = _raw(x, d)
    if float(f"{x:.{d}f}") > 0:
        s = "+" + s
    return _rec(s, x)


def _needs_more(vals, d):
    return any(v != 0 and float(f"{v:.{d}f}") == 0.0 for v in vals)


def ci(e: dict, d: int = 3, signed: bool = True, label: bool = True) -> str:
    """'+0.045（95% CI 0.029 至 0.061）' -- one extra decimal when a bound rounds to zero."""
    a, lo, hi = float(e["delta"]), float(e["ci_low"]), float(e["ci_high"])
    while _needs_more((a, lo, hi), d) and d < 5:
        d += 1
    f = sgn if signed else num
    head = f(a, d)
    return f"{head}（{'95% CI ' if label else ''}{f(lo, d)} 至 {f(hi, d)}）"


def cin(e: dict, d: int = 3, signed: bool = True, label: bool = True) -> str:
    """'+0.045，95% CI +0.029 至 +0.061' -- the form used inside parentheses (no nested brackets)."""
    a, lo, hi = float(e["delta"]), float(e["ci_low"]), float(e["ci_high"])
    while _needs_more((a, lo, hi), d) and d < 5:
        d += 1
    f = sgn if signed else num
    return f"{f(a, d)}，{'95% CI ' if label else ''}{f(lo, d)} 至 {f(hi, d)}"


def rng(lo: float, hi: float, d: int = 3, signed: bool = True) -> str:
    while _needs_more((lo, hi), d) and d < 5:
        d += 1
    f = sgn if signed else num
    return f"{f(lo, d)} 至 {f(hi, d)}"


def pc(x, d: int = 1) -> str:
    x = float(x)
    s = f"{100 * x:.{d}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return _rec(s.replace("-", MINUS) + "%", x)


def i0(x) -> str:
    return _rec(f"{int(round(float(x))):,}", x)


def sci(x: float, d: int = 1) -> str:
    if x == 0:
        return _rec("0", 0)
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    sup = str(e).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return _rec(f"{m:.{d}f} × 10{sup}", x)


def fold(a: float, b: float, d: int = 0) -> str:
    """ratio a/b as '约 25 倍'."""
    return _rec(f"{a / b:.{d}f}", a / b)

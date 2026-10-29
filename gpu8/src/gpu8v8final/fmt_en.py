"""English number formatting of the SARA manuscript, built on fmt9 (every formatted value is recorded in fmt9.TRACE,
so the number-tracing check of the Chinese package applies unchanged).  Negative numbers use U+2212; a value that
rounds to zero is printed without a sign; an interval whose bounds round to zero gets one more decimal."""
from __future__ import annotations

from .fmt9 import TRACE, _needs_more, i0, num, pc, sci, sgn  # noqa: F401  (re-exported)


def _digits(vals, d: int) -> int:
    while _needs_more(vals, d) and d < 5:
        d += 1
    return d


def ci(e: dict, d: int = 3, signed: bool = True, label: bool = True) -> str:
    """'+0.045 (95% CI +0.029 to +0.061)'."""
    a, lo, hi = float(e["delta"]), float(e["ci_low"]), float(e["ci_high"])
    d = _digits((a, lo, hi), d)
    f = sgn if signed else num
    return f"{f(a, d)} ({'95% CI ' if label else ''}{f(lo, d)} to {f(hi, d)})"


def cin(e: dict, d: int = 3, signed: bool = True, label: bool = True) -> str:
    """'+0.045, 95% CI +0.029 to +0.061' -- the form used inside parentheses."""
    a, lo, hi = float(e["delta"]), float(e["ci_low"]), float(e["ci_high"])
    d = _digits((a, lo, hi), d)
    f = sgn if signed else num
    return f"{f(a, d)}, {'95% CI ' if label else ''}{f(lo, d)} to {f(hi, d)}"


def rng(lo: float, hi: float, d: int = 3, signed: bool = True) -> str:
    d = _digits((lo, hi), d)
    f = sgn if signed else num
    return f"{f(lo, d)} to {f(hi, d)}"

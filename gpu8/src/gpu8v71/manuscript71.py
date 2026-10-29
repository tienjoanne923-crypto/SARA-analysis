"""Compose the V7.1 manuscript (Markdown -> DOCX with OMML equations).

The V6 manuscript remains the base: its methods 3.1-3.10 and results 4.1-4.7
are carried over verbatim (they are still the V6 zero-support retrieval,
in-list scoring and support line on which V7.1 builds).  V7.1 adds methods
3.11-3.16 (``mstext71``) and results 4.8-4.17, and rewrites the title, abstract,
introduction, discussion, limitations and conclusion from the V7.1 numbers.
Every result number is read through ``N`` from
``reports_v71/manuscript_numbers_v71.json`` and the keys used are recorded.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from gpu8 import paths as p5
from gpu8v6 import paths6

from . import paths71
from .common71 import dump_json, now_utc
from .mstext71 import METHODS_V71

OUT_DIR = p5.PROJECT_ROOT / "manuscript_v71"
V6_MD = next(p for p in (p5.PROJECT_ROOT / "manuscript_v6").glob("*_V6_*.md"))
DATE = "20260923"


class N:
    def __init__(self, v: dict):
        self.v = v
        self.used: set = set()

    def has(self, k) -> bool:
        return k in self.v

    def raw(self, k):
        self.used.add(k)
        return self.v[k]

    def f(self, k, d=3):
        return f"{self.raw(k):.{d}f}"

    def i(self, k):
        return f"{int(round(self.raw(k))):,}"

    def pc(self, k, d=1):
        return f"{100 * self.raw(k):.{d}f}%"

    def sd(self, k, d=3):
        return f"{self.raw(k):+.{d}f}"

    def ci(self, pre, d=3):
        """'+0.012（95% CI +0.004 至 +0.020）' from pre_delta / pre_ci_low / pre_ci_high."""
        dk = pre + ("_delta" if self.has(pre + "_delta") else "_delta_recall10")
        return (f"{self.raw(dk):+.{d}f}（95% CI {self.raw(pre + '_ci_low'):+.{d}f} 至 "
                f"{self.raw(pre + '_ci_high'):+.{d}f}）")


def v6_sections() -> list[tuple[str, str]]:
    """[(heading line, body)] in document order, split at '## ' and '### '."""
    text = V6_MD.read_text(encoding="utf-8").lstrip("﻿")
    out, head, buf = [], "", []
    for line in text.split("\n"):
        if re.match(r"^#{1,3}\s", line):
            out.append((head, "\n".join(buf).strip("\n")))
            head, buf = line, []
        else:
            buf.append(line)
    out.append((head, "\n".join(buf).strip("\n")))
    return out


def pick(secs, prefix: str) -> str:
    for h, b in secs:
        if h.startswith(prefix):
            return f"{h}\n\n{b}\n"
    raise KeyError(prefix)


def v6_block(secs, first: str, last_exclusive: str) -> str:
    """Concatenate V6 sections from ``first`` up to (not incl.) ``last_exclusive``."""
    on, parts = False, []
    for h, b in secs:
        if h.startswith(first):
            on = True
        if on and h.startswith(last_exclusive):
            break
        if on:
            parts.append(f"{h}\n\n{b}\n")
    return "\n".join(parts)


def copy_figures() -> Path:
    fig = OUT_DIR / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    for src in list(paths6.FIGURES.glob("*.png")) + list(paths71.FIGURES.glob("*.png")):
        shutil.copy2(src, fig / src.name)
    return fig


def build_markdown(parts: list[str]) -> str:
    return "\n\n".join(p.strip("\n") for p in parts if p) + "\n"


def write(md: str, used: set, title: str) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / f"{title}_V71_{DATE}.md"
    md_path.write_text(md, encoding="utf-8")
    fig = copy_figures()
    from gpu8v6.build_docx6 import build as build_docx
    st = build_docx(md_path, md_path.with_suffix(".docx"), fig)
    dump_json(paths71.REPORTS / "docx_build_report_v71.json", st)
    dump_json(paths71.REPORTS / "manuscript_numbers_used_v71.json", sorted(used))
    return {"markdown": str(md_path), "docx": st.get("path"), "omml": st.get("verified_omml_objects"),
            "tables": st.get("verified_tables"), "images": st.get("images"),
            "equation_failures": len(st.get("equation_failures", [])),
            "missing_figures": st.get("missing_figures"), "numbers_used": len(used)}


if __name__ == "__main__":
    from .manuscript71_body import compose
    v = json.loads((paths71.REPORTS / "manuscript_numbers_v71.json").read_text(encoding="utf-8"))["values"]
    n = N(v)
    title, md = compose(n, v6_sections(), METHODS_V71)
    print(json.dumps(write(md, n.used, title), indent=2, ensure_ascii=False))

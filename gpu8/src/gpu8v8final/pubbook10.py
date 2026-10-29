"""Figure book of the second-revision SARA package: cover page with the figure titles of manuscript10/supp10 and the
18 vector figures (the figures themselves are the unchanged figures of the 2026-09-29 package)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .manuscript10 import DATE, FIGS10, TITLE
from .numbers9 import FIG, ROOT
from .pubstyle8 import FS, FS_L, MM, setup
from .supp10 import SUPP_STEM
from .supp9 import SFIGS


def supp_titles() -> dict:
    md = (ROOT / f"{SUPP_STEM}.md").read_text(encoding="utf-8")
    return dict(re.findall(r"\*\*图 (S\d+)　([^。*]+)。\*\*", md))


def build(out: Path | None = None) -> Path:
    sys.path.append(str(Path(r"D:\信\gpu8\pub_libs")))       # pypdf only (appended: the interpreter's packages win)
    from pypdf import PdfReader, PdfWriter
    import matplotlib.pyplot as plt
    setup()
    st = supp_titles()
    out = out or ROOT / f"SARA_图集_{DATE}.pdf"
    order = [(stem, f"图 {n}", t) for n, stem, t in FIGS10] + [(stem, f"图 {n}", st[n]) for n, stem, _, _ in SFIGS]
    cover = ROOT / "build" / "_cover10.pdf"
    cover.parent.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(183 * MM, 247 * MM))
    fig.text(0.07, 0.94, TITLE, fontsize=10.5, fontweight="bold")
    fig.text(0.07, 0.915, f"图集：主图 {len(FIGS10)} 张，补充图 {len(SFIGS)} 张", fontsize=FS_L)
    y = 0.87
    for stem, n, t in order:
        fig.text(0.07, y, n, fontsize=FS + 0.5, fontweight="bold")
        fig.text(0.17, y, t, fontsize=FS + 0.5)
        y -= 0.036 if not n.startswith("图 S") or n == "图 S1" else 0.034
        if n == "图 10":
            y -= 0.012
    fig.text(0.07, 0.07, "每个面板的作图数据以 CSV 提供（figures/source_data/）；图件同时提供矢量 PDF/SVG 与 600 dpi PNG。\n"
                         "分类配色经色觉缺陷模拟校验，同一方法在全部图中使用同一颜色。", fontsize=FS - 0.5, color="#52514e",
             linespacing=1.5)
    fig.savefig(cover)
    plt.close(fig)
    w = PdfWriter()
    for p in [cover] + [FIG / f"{s}.pdf" for s, _, _ in order]:
        for page in PdfReader(str(p)).pages:
            w.add_page(page)
    w.add_metadata({"/Title": "SARA 图集", "/Subject": TITLE})
    with open(out, "wb") as fh:
        w.write(fh)
    cover.unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    print(build())

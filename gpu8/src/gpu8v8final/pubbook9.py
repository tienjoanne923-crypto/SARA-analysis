"""Figure book of the SARA submission: one PDF with a cover page and the 18 vector figures."""
from __future__ import annotations

import sys
from pathlib import Path

from .manuscript9 import DATE, FIGS, TITLE
from .pubstyle8 import FS, FS_L, MM, setup
from .numbers9 import FIG, ROOT
from .supp9 import SFIGS


def build(out: Path | None = None, cover_png: Path | None = None) -> Path:
    sys.path.append(str(Path(r"D:\信\gpu8\pub_libs")))       # pypdf only (appended: the interpreter's packages win)
    from pypdf import PdfReader, PdfWriter
    import matplotlib.pyplot as plt
    setup()
    out = out or ROOT / f"SARA_图集_{DATE}.pdf"
    order = [(stem, f"图 {n}", t) for n, stem, t in FIGS] + [(stem, f"图 {n}", t) for n, stem, t, _ in SFIGS]
    cover = ROOT / "build" / "_cover9.pdf"
    cover.parent.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(183 * MM, 247 * MM))
    fig.text(0.07, 0.94, TITLE, fontsize=10.5, fontweight="bold")
    fig.text(0.07, 0.915, f"图集：主图 {len(FIGS)} 张，补充图 {len(SFIGS)} 张", fontsize=FS_L)
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
    if cover_png:
        fig.savefig(cover_png, dpi=110)
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

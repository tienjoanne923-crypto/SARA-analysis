"""English figure book: cover page with the English figure titles and the 18 English vector figures."""
from __future__ import annotations

import sys
from pathlib import Path

from .manuscript10 import FIGS10
from .manuscript_en import DATE, EN, FIG_TITLES, TITLE
from .pubstyle8 import FS, FS_L, MM, setup
from .supp_en import SFIGS

FIG_EN = EN / "figures"


def build(out: Path | None = None) -> Path:
    sys.path.append(str(Path(r"D:\信\gpu8\pub_libs")))       # pypdf only (appended: the interpreter's packages win)
    from pypdf import PdfReader, PdfWriter
    import matplotlib.pyplot as plt
    from .pubstyle_en import en_fonts
    setup()
    en_fonts()
    out = out or EN / f"SARA_figures_{DATE}.pdf"
    order = [(stem, f"Figure {n}", FIG_TITLES[n]) for n, stem, _ in FIGS10] + \
            [(stem, f"Figure {n}", t) for n, stem, t, _ in SFIGS]
    cover = EN.parent / "build" / "en" / "_cover_en.pdf"
    cover.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(183 * MM, 247 * MM))
    fig.text(0.07, 0.945, "SARA: availability-aware semantic retrieval of disease genes", fontsize=10.5, fontweight="bold")
    fig.text(0.07, 0.925, "under uneven knowledge coverage", fontsize=10.5, fontweight="bold")
    fig.text(0.07, 0.9, f"Figure book: {len(FIGS10)} main figures and {len(SFIGS)} supplementary figures", fontsize=FS_L)
    y = 0.86
    for stem, n, t in order:
        fig.text(0.07, y, n, fontsize=FS + 0.5, fontweight="bold")
        fig.text(0.19, y, t, fontsize=FS + 0.5)
        y -= 0.036 if not n.startswith("Figure S") or n == "Figure S1" else 0.034
        if n == "Figure 10":
            y -= 0.012
    fig.text(0.07, 0.07, "The plotting data of every panel are provided as CSV files (figures/source_data/); every figure is "
                         "provided as vector PDF/SVG and 600 dpi PNG.\nCategorical palettes were checked by simulating "
                         "color-vision deficiency, and each method keeps the same color in all figures.",
             fontsize=FS - 0.5, color="#52514e", linespacing=1.5)
    fig.savefig(cover)
    plt.close(fig)
    w = PdfWriter()
    for p in [cover] + [FIG_EN / f"{s}.pdf" for s, _, _ in order]:
        for page in PdfReader(str(p)).pages:
            w.add_page(page)
    w.add_metadata({"/Title": "SARA figure book", "/Subject": TITLE})
    with open(out, "wb") as fh:
        w.write(fh)
    cover.unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    print(build())

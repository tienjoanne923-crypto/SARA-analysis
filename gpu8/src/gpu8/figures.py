"""Manuscript figures: 6.75 inch, 600 dpi, greyscale, SimSun, live-text SVG.

The contract matches the previous rounds of this project so the new figures
drop into the same manuscript pipeline: minimum 7.5 pt type, pure greyscale,
no colour carrying meaning, and a hard validation pass that fails the build
rather than emitting an out-of-spec panel.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import paths

FIG_WIDTH_IN = 6.75
DPI = 600
MIN_FONT_PT = 7.5
GREY = ["#000000", "#3c3c3c", "#707070", "#a6a6a6", "#d4d4d4", "#eeeeee", "#ffffff"]
SIMSUN = Path(r"C:\Windows\Fonts\simsun.ttc")


def setup_style() -> None:
    if SIMSUN.exists():
        fm.fontManager.addfont(str(SIMSUN))
        try:
            name = fm.FontProperties(fname=str(SIMSUN)).get_name()
        except Exception:
            name = "SimSun"
    else:
        name = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": name, "font.size": 8.0,
        "axes.labelsize": 8.0, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "figure.titlesize": 9.0,
        "axes.edgecolor": "#000000", "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "text.color": "#000000", "axes.labelcolor": "#000000",
        "xtick.color": "#000000", "ytick.color": "#000000",
        # SimSun has no U+2212; use the ASCII hyphen so negative numbers render
        "axes.unicode_minus": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "axes.grid": True, "grid.color": "#d4d4d4", "grid.linewidth": 0.4,
        "grid.linestyle": "-", "axes.axisbelow": True,
    })


def panel_label(ax, letter: str, title: str = "") -> None:
    ax.text(-0.08, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", va="bottom", ha="left")
    if title:
        ax.set_title(title, loc="left", pad=6, x=0.02)


def save(fig, stem: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    png, svg, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.svg", out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(svg)
    fig.savefig(pdf)
    plt.close(fig)
    from PIL import Image
    with Image.open(png) as im:
        w, h = im.size
        arr = np.asarray(im.convert("RGB"))
        greyscale = bool(np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 1] == arr[..., 2]))
    return {"stem": stem, "png": str(png), "svg": str(svg), "pdf": str(pdf),
            "width_px": w, "height_px": h, "dpi": DPI, "greyscale": greyscale}


def validate(fig, records: list[dict]) -> list[str]:
    problems = []
    for r in records:
        expect = int(round(FIG_WIDTH_IN * DPI))
        if abs(r["width_px"] - expect) > expect * 0.08:
            problems.append(f"{r['stem']}: width {r['width_px']} px, expected ~{expect}")
        if not r["greyscale"]:
            problems.append(f"{r['stem']}: not greyscale")
    return problems


# --------------------------------------------------------------------------
# data access helpers
# --------------------------------------------------------------------------
def read_table(name: str) -> pd.DataFrame | None:
    for root in (paths.RESULT_TABLE_DIR, paths.RESULTS_DIR, paths.MANIFEST_DIR):
        p = root / name
        if p.exists():
            return pd.read_csv(p)
    return None


def read_json_any(name: str) -> dict:
    for root in (paths.REPORT_DIR, paths.MANIFEST_DIR, paths.RESULTS_DIR,
                 paths.TABLES_DIR, paths.RESOURCE_LOG_DIR):
        p = root / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def bar(ax, labels, values, errs=None, hatches=None, ylabel=""):
    x = np.arange(len(labels))
    colors = [GREY[min(3, i % 4) + 1] for i in range(len(labels))]
    b = ax.bar(x, values, yerr=errs, color=colors, edgecolor="#000000",
               linewidth=0.6, capsize=2.5, error_kw={"elinewidth": 0.6})
    if hatches:
        for rect, h in zip(b, hatches):
            rect.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    return b

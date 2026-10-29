"""Publication style and drawing helpers for the V8.1 figures.

Conventions (bioinformatics journal practice + the data-viz method):
* double-column width 183 mm, single 89 mm; Arial for Latin, DengXian for CJK
  (glyph fallback); 6.5-7 pt text; bold lowercase panel letters;
* thin marks, hairline solid grid behind the data, recessive axes; text in ink,
  never in the series colour; a legend for >= 2 series plus selective direct labels;
* every panel writes its source data (CSV) next to the figure; vector PDF/SVG
  (TrueType/CID fonts, no Type 3) and 600-dpi PNG.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

from .pubcolor8 import AXIS, BLUE, GRID, INK, INK2, MUTED, SURFACE, ink_on  # noqa: E402

MM = 1 / 25.4
W2, W15, W1 = 183 * MM, 120 * MM, 89 * MM
FIG_DIR = Path(r"D:\信\manuscript_v8\figures_v2")
SRC_DIR = FIG_DIR / "source_data"
CJK = re.compile(r"[\u2460-\u24ff\u3000-\u303f\u3400-\u9fff\uff00-\uffef]+")
# The original compact export used 5.8 pt small annotations, which became
# difficult to read after rasterisation in Word.  These values keep the
# journal-sized 183-mm canvas while giving labels and legend text a reliable
# print scale in the full-width manuscript figures.
FS = 7.5          # base font size (pt)
FS_S = 7.2        # small annotation
FS_L = 8.7        # panel titles
FS_P = 10.0       # panel letters


def setup() -> None:
    plt.rcParams.update({
        "font.family": ["Arial", "DengXian"], "font.size": FS, "axes.titlesize": FS_L,
        "axes.labelsize": FS, "xtick.labelsize": FS - 0.3, "ytick.labelsize": FS - 0.3,
        "legend.fontsize": FS - 0.3, "legend.title_fontsize": FS,
        "mathtext.fontset": "custom", "mathtext.rm": "DengXian", "mathtext.it": "Arial:italic",
        "mathtext.bf": "DengXian:bold", "mathtext.sf": "Arial", "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlepad": 4.0, "axes.titlelocation": "left",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.minor.size": 1.3, "ytick.minor.size": 1.3,
        "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.4, "grid.linestyle": "-",
        "lines.linewidth": 1.1, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 3.2, "patch.linewidth": 0.0,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.2, "legend.columnspacing": 1.0, "legend.labelspacing": 0.35,
        "figure.dpi": 150, "savefig.dpi": 600, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK, "hatch.linewidth": 0.4,
    })


def mt(s: str, bold: bool = False) -> str:
    """Mixed CJK + mathtext: wrap CJK runs outside $...$ in \\mathrm (or \\mathbf for bold titles),
    so they are rendered by the CJK face that mathtext is configured with."""
    if "$" not in s:
        return s
    cmd = "\\mathbf" if bold else "\\mathrm"
    parts = s.split("$")
    for i in range(0, len(parts), 2):
        parts[i] = CJK.sub(lambda m: "$" + cmd + "{" + m.group(0) + "}$", parts[i])
    return "$".join(parts)


def fig_mm(w_mm: float, h_mm: float):
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def axmm(fig, x: float, y: float, w: float, h: float, **kw):
    """Axes placed in millimetres from the figure's top-left corner."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def label_mm(fig, x: float, y: float, letter: str, title: str | None = None):
    """Panel letter (and title) at millimetre coordinates from the top-left corner (baseline at y)."""
    W, H = fig.get_size_inches() / MM
    fig.text(x / W, 1 - y / H, letter, fontsize=FS_P, fontweight="bold", color=INK, ha="left", va="baseline")
    if title:
        fig.text((x + 4.4) / W, 1 - y / H, mt(title, bold=True), fontsize=FS_L, fontweight="bold", color=INK,
                 ha="left", va="baseline")


def panel(fig, ax, letter: str, dx_mm: float = -7.0, dy_mm: float = 1.5, title: str | None = None,
          x_ax=None):
    """Bold lowercase panel letter at the top-left of the axes (offset in mm); optional title.
    ``x_ax``: take the left edge from another axes (e.g. the left-most axes of a composite panel)."""
    bb = ax.get_position()
    x0 = (x_ax.get_position().x0 if x_ax is not None else bb.x0)
    W, H = fig.get_size_inches() / MM
    fig.text(x0 + dx_mm / W, bb.y1 + dy_mm / H, letter, fontsize=FS_P, fontweight="bold", color=INK,
             ha="left", va="bottom")
    if title:
        fig.text(x0 + (dx_mm + 4.2) / W, bb.y1 + dy_mm / H + 0.25 / H, mt(title, bold=True), fontsize=FS_L,
                 fontweight="bold", color=INK, ha="left", va="bottom")


def shrink(ax, fx: float = 1.0, fy: float = 1.0, anchor: str = "W"):
    """Shrink an axes inside its slot (e.g. to leave room for a colour bar)."""
    p = ax.get_position()
    w, h = p.width * fx, p.height * fy
    x0 = p.x0 if anchor in ("W", "SW", "NW") else p.x0 + (p.width - w)
    y0 = p.y0 if anchor in ("S", "SW", "W") else p.y0 + (p.height - h)
    ax.set_position([x0, y0, w, h])
    return ax


def style(ax, grid: str | None = "y", left: bool = True, bottom: bool = True):
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    if not left:
        ax.tick_params(axis="y", length=0)
    if not bottom:
        ax.tick_params(axis="x", length=0)
    return ax


def pct(ax, axis: str = "y", digits: int = 0):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{100 * v:.{digits}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def save(fig, stem: str, sources: dict[str, pd.DataFrame] | None = None, meta: dict | None = None) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    out = {"stem": stem, "files": []}
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, **kw)
        out["files"].append(p.name)
    w, h = fig.get_size_inches() / MM
    out["size_mm"] = [round(w, 1), round(h, 1)]
    for k, df in (sources or {}).items():
        p = SRC_DIR / f"{stem}_{k}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        out["files"].append(f"source_data/{p.name}")
    if meta:
        out.update(meta)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ statistics
def fam_draws(values: np.ndarray, fam: np.ndarray, draws: int = 10000, seed: int = 20260923) -> np.ndarray:
    """Family (cluster) bootstrap draws of the disease mean -- same algorithm and seed as the registered CIs."""
    uniq, inv = np.unique(fam, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    cnts = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    n = len(uniq)
    out = np.empty(draws)
    for b0 in range(0, draws, 1000):
        Wt = rng.multinomial(n, np.full(n, 1.0 / n), size=min(1000, draws - b0)).astype(np.float64)
        out[b0:b0 + len(Wt)] = (Wt @ sums) / np.maximum(Wt @ cnts, 1.0)
    return out


def fam_ci(values, fam, draws: int = 2000, seed: int = 20260923) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    fam = np.asarray(fam)
    d = fam_draws(values, fam, draws, seed)
    return float(values.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fam_ratio_ci(num, den, fam, draws: int = 2000, seed: int = 20260923) -> tuple[float, float, float]:
    """Micro rate sum(num)/sum(den) with a family bootstrap CI (clusters = disease families)."""
    num, den, fam = np.asarray(num, float), np.asarray(den, float), np.asarray(fam)
    uniq, inv = np.unique(fam, return_inverse=True)
    sn, sd = np.bincount(inv, weights=num), np.bincount(inv, weights=den)
    rng = np.random.default_rng(seed)
    n = len(uniq)
    Wt = rng.multinomial(n, np.full(n, 1.0 / n), size=draws).astype(np.float64)
    r = (Wt @ sn) / np.maximum(Wt @ sd, 1e-12)
    est = sn.sum() / max(sd.sum(), 1e-12)
    return float(est), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def fam_curve_ci(M: np.ndarray, fam: np.ndarray, draws: int = 1000, seed: int = 20260923):
    """[n_disease, K] matrix -> mean curve and pointwise 95% family-bootstrap band."""
    uniq, inv = np.unique(fam, return_inverse=True)
    S = np.zeros((len(uniq), M.shape[1]))
    np.add.at(S, inv, M)
    cnt = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    n = len(uniq)
    lo, hi = np.empty(M.shape[1]), np.empty(M.shape[1])
    Wt = rng.multinomial(n, np.full(n, 1.0 / n), size=draws).astype(np.float64)
    B = (Wt @ S) / np.maximum(Wt @ cnt, 1.0)[:, None]
    lo[:] = np.percentile(B, 2.5, axis=0)
    hi[:] = np.percentile(B, 97.5, axis=0)
    return M.mean(axis=0), lo, hi


# ------------------------------------------------------------------ marks
def ci_h(ax, y, est, lo, hi, color, marker="o", size=3.4, lw=0.9, zorder=3, ring=True, cap=0.0):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=zorder)
    if cap:
        ax.plot([lo, lo], [y - cap, y + cap], color=color, lw=lw * 0.8, zorder=zorder)
        ax.plot([hi, hi], [y - cap, y + cap], color=color, lw=lw * 0.8, zorder=zorder)
    ax.plot([est], [y], marker=marker, ms=size, color=color, mec=SURFACE if ring else color,
            mew=0.6 if ring else 0, zorder=zorder + 1, linestyle="none")


def ci_v(ax, x, est, lo, hi, color, marker="o", size=3.4, lw=0.9, zorder=3, ring=True):
    ax.plot([x, x], [lo, hi], color=color, lw=lw, solid_capstyle="round", zorder=zorder)
    ax.plot([x], [est], marker=marker, ms=size, color=color, mec=SURFACE if ring else color,
            mew=0.6 if ring else 0, zorder=zorder + 1, linestyle="none")


def half_violin(ax, data, y, color, height=0.36, side=1, alpha=0.18, bw=None, lw=0.6, grid=256):
    """Horizontal half-violin (kernel density) of a 1-D sample centred on row y."""
    data = np.asarray(data, float)
    lo, hi = np.percentile(data, [0.1, 99.9])
    xs = np.linspace(lo, hi, grid)
    sd = data.std() or 1e-9
    bw = bw or 1.06 * sd * len(data) ** (-1 / 5)
    dens = np.exp(-0.5 * ((xs[:, None] - data[None, ::max(1, len(data) // 4000)]) / bw) ** 2).sum(axis=1)
    dens = dens / dens.max() * height
    ax.fill_between(xs, y, y - side * dens, color=color, alpha=alpha, lw=0, zorder=1)
    ax.plot(xs, y - side * dens, color=color, lw=lw, alpha=0.9, zorder=2)


def rounded_box(ax, x, y, w, h, fc, ec, lw=0.7, r=0.012, zorder=2, **kw):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=zorder, **kw)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.7, head=3.2, style="-|>", connection="arc3,rad=0", zorder=3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=zorder,
                arrowprops=dict(arrowstyle=f"{style},head_length={head / 10:.2f},head_width={head / 20:.2f}",
                                color=color, lw=lw, shrinkA=0, shrinkB=0, connectionstyle=connection))


def heat_table(ax, M: np.ndarray, rows: list[str], cols: list[str], cmap, norm_by: str = "col",
               fmt: str = "{:.3f}", vmin=None, vmax=None, text_size=FS - 0.6, gap=0.06, mask=None,
               col_label_rot=0, ha_cols="center"):
    """Annotated heat table: each cell a rounded tile with a surface gap; label ink by luminance."""
    R, C = M.shape
    V = np.array(M, float)
    if norm_by == "col":
        N = np.full_like(V, np.nan)
        for j in range(C):
            col = V[:, j]
            fin = np.isfinite(col)
            if fin.sum():
                a, b = np.nanmin(col), np.nanmax(col)
                N[:, j] = (col - a) / (b - a) if b > a else 0.5
    else:
        a = np.nanmin(V) if vmin is None else vmin
        b = np.nanmax(V) if vmax is None else vmax
        N = (V - a) / (b - a) if b > a else np.full_like(V, 0.5)
    from matplotlib.colors import to_hex
    for i in range(R):
        for j in range(C):
            if mask is not None and mask[i, j]:
                continue
            if not np.isfinite(V[i, j]):
                ax.add_patch(Rectangle((j + gap / 2, i + gap / 2), 1 - gap, 1 - gap, fc="#f7f7f5", ec="none"))
                ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center", fontsize=text_size, color=MUTED)
                continue
            fc = to_hex(cmap(0.08 + 0.84 * float(np.clip(N[i, j], 0, 1))))
            ax.add_patch(FancyBboxPatch((j + gap / 2, i + gap / 2), 1 - gap, 1 - gap,
                                        boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec="none"))
            ax.text(j + 0.5, i + 0.5, fmt.format(V[i, j]), ha="center", va="center", fontsize=text_size,
                    color=ink_on(fc))
    ax.set_xlim(0, C)
    ax.set_ylim(R, 0)
    ax.set_xticks(np.arange(C) + 0.5, [mt(c) for c in cols], rotation=col_label_rot, ha=ha_cols)
    ax.set_yticks(np.arange(R) + 0.5, [mt(r) for r in rows])
    ax.tick_params(length=0, pad=2)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)


def sankey_band(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.45, zorder=1):
    """A smooth band from [y0a, y0b] at x0 to [y1a, y1b] at x1 (cubic Bezier edges)."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4,
             MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", alpha=alpha, zorder=zorder))


def text_ink(ax, x, y, s, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("fontsize", FS)
    return ax.text(x, y, mt(s), **kw)


def legend_handles(items: list[tuple[str, str, str]]):
    """[(label, color, kind)] -> handles for fig/ax.legend (kind: line | dot | box)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = []
    for lab, col, kind in items:
        if kind == "line":
            hs.append(Line2D([0], [0], color=col, lw=1.3, label=mt(lab)))
        elif kind == "dot":
            hs.append(Line2D([0], [0], color=col, marker="o", ms=4, mec=SURFACE, mew=0.6, lw=0, label=mt(lab)))
        else:
            hs.append(Patch(fc=col, ec="none", label=mt(lab)))
    return hs


def polygon(ax, pts, fc, ec="none", lw=0.0, alpha=1.0, zorder=2):
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder))


BLUE_STEPS = [BLUE[k] for k in (250, 350, 450, 550, 650)]

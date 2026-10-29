"""Advanced chart primitives for the top-tier figure set (matplotlib only).

raincloud, ridgeline, chord (circos-style), clustered heatmap with dendrograms and annotation tracks,
radar, SHAP beeswarm, 2-D density with marginals, network, slopegraph.
All primitives draw into given axes and never pick colours on their own: colours come from pubcolor8.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch, PathPatch, Wedge
from matplotlib.path import Path as MPath

from .pubcolor8 import AXIS, GRID, INK, INK2, MUTED, SURFACE, ink_on
from .pubstyle8 import FS, FS_S, mt


# ------------------------------------------------------------------ density helpers
def kde(x: np.ndarray, grid: np.ndarray, bw: float | None = None, max_n: int = 6000) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.zeros_like(grid)
    if len(x) > max_n:
        x = np.random.default_rng(0).choice(x, max_n, replace=False)
    sd = x.std() or 1e-6
    iqr = np.subtract(*np.percentile(x, [75, 25])) or sd
    bw = bw or 0.9 * min(sd, iqr / 1.34) * len(x) ** (-0.2)
    bw = max(bw, 1e-3)
    d = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(axis=1)
    return d / (d.max() or 1)


# ------------------------------------------------------------------ raincloud (horizontal)
def raincloud(ax, data: list[np.ndarray], colors: list[str], ypos=None, height=0.38, grid=None, bw=None,
              box=True, strip=True, strip_n=900, strip_size=1.2, seed=0):
    rng = np.random.default_rng(seed)
    ypos = np.arange(len(data)) if ypos is None else ypos
    allx = np.concatenate([np.asarray(d, float) for d in data])
    grid = np.linspace(np.nanmin(allx), np.nanmax(allx), 300) if grid is None else grid
    for y, x, c in zip(ypos, data, colors):
        x = np.asarray(x, float)
        dens = kde(x, grid, bw)
        # cloud (density) above the baseline, box on the baseline, rain (points) below
        ax.fill_between(grid, y + 0.02, y + 0.02 + dens * height, color=c, alpha=0.22, lw=0, zorder=2)
        ax.plot(grid, y + 0.02 + dens * height, color=c, lw=0.8, zorder=3)
        if box:
            q1, q2, q3 = np.percentile(x, [25, 50, 75])
            lo, hi = np.percentile(x, [5, 95])
            ax.plot([lo, hi], [y - 0.06, y - 0.06], color=INK2, lw=0.6, zorder=4)
            ax.add_patch(FancyBboxPatch((q1, y - 0.11), max(q3 - q1, 1e-9), 0.1,
                                        boxstyle="round,pad=0,rounding_size=0.01", fc=c, ec=INK2, lw=0.5, zorder=5))
            ax.plot([q2, q2], [y - 0.13, y + 0.01], color=SURFACE, lw=1.2, zorder=6)
        if strip:
            xs = x if len(x) <= strip_n else rng.choice(x, strip_n, replace=False)
            ax.scatter(xs, y - 0.2 - rng.uniform(0, 0.18, len(xs)), s=strip_size, color=c, alpha=0.5, lw=0,
                       zorder=3, rasterized=True)


# ------------------------------------------------------------------ ridgeline
def ridgeline(ax, series: list[tuple[str, list[tuple[np.ndarray, str]]]], grid: np.ndarray, overlap=1.6, bw=None,
              label_x=None, fill_alpha=0.28, lw=0.8):
    """series: [(row label, [(values, colour), ...])] -> stacked density ridges (first row on top)."""
    n = len(series)
    for i, (lab, parts) in enumerate(series):
        base = n - 1 - i
        for vals, c in parts:
            d = kde(vals, grid, bw) * overlap
            ax.fill_between(grid, base, base + d, color=c, alpha=fill_alpha, lw=0, zorder=2 * i + 1)
            ax.plot(grid, base + d, color=c, lw=lw, zorder=2 * i + 2)
        ax.plot([grid[0], grid[-1]], [base, base], color=GRID, lw=0.5, zorder=0)
        if label_x is not None:
            ax.text(label_x, base + 0.18, lab, ha="right", va="bottom", fontsize=FS_S - 0.9, color=INK)
    ax.set_ylim(-0.2, n - 1 + overlap + 0.2)
    ax.set_yticks([])


# ------------------------------------------------------------------ chord / circos
def _arc_pts(a0, a1, r, n=40):
    t = np.linspace(a0, a1, n)
    return np.c_[r * np.cos(t), r * np.sin(t)]


def chord(ax, M: np.ndarray, labels: list[str], arc_colors: list[str], ribbon_colors=None, gap=0.035, r=1.0,
          width=0.07, ribbon_alpha=0.5, min_ribbon=0.0, label_r=1.12, fs=FS_S - 0.6, sizes=None, emphasis=None,
          label_fmt=None):
    """Symmetric M (diag ignored). Arc length ∝ sizes (default: row sums). Ribbons ∝ M[i, j] at both ends."""
    n = len(labels)
    M = np.array(M, float)
    np.fill_diagonal(M, 0)
    tot = M.sum(axis=1) if sizes is None else np.asarray(sizes, float)
    full = 2 * np.pi - gap * n
    spans = tot / tot.sum() * full
    starts = np.pi / 2 - np.r_[0, np.cumsum(spans + gap)][:-1]
    # arcs go clockwise from the top
    ends = starts - spans
    for i in range(n):
        th = np.linspace(starts[i], ends[i], 60)
        outer = np.c_[(r + width) * np.cos(th), (r + width) * np.sin(th)]
        inner = np.c_[r * np.cos(th[::-1]), r * np.sin(th[::-1])]
        verts = np.vstack([outer, inner, outer[:1]])
        ax.add_patch(PathPatch(MPath(verts), fc=arc_colors[i], ec=SURFACE, lw=0.6, zorder=3))
        mid = (starts[i] + ends[i]) / 2
        rot = np.degrees(mid)
        ha = "left" if np.cos(mid) >= 0 else "right"
        rot = rot if np.cos(mid) >= 0 else rot + 180
        txt = labels[i] if label_fmt is None else label_fmt(i)
        ax.text(label_r * np.cos(mid), label_r * np.sin(mid), txt, rotation=rot, rotation_mode="anchor", ha=ha,
                va="center", fontsize=fs, color=INK)
    # ribbon slots: within each arc, allocate proportional sub-spans to partners in order
    scale = spans / np.maximum(M.sum(axis=1) if sizes is None else np.maximum(M.sum(axis=1), 1e-12), 1e-12)
    if sizes is not None:
        scale = np.minimum(scale, spans / np.maximum(M.sum(axis=1), 1e-12))
    cursor = starts.copy()
    slot = {}
    order = np.argsort(-M, axis=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = M[i, j] * scale[i]
            slot[(i, j)] = (cursor[i], cursor[i] - w)
            cursor[i] -= w
    drawn = []
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] <= min_ribbon:
                continue
            a0, a1 = slot[(i, j)]
            b0, b1 = slot[(j, i)]
            p = [(r * np.cos(a0), r * np.sin(a0))]
            p += [tuple(x) for x in _arc_pts(a0, a1, r, 12)[1:]]
            p += [(0, 0), (0, 0), (r * np.cos(b0), r * np.sin(b0))]
            p += [tuple(x) for x in _arc_pts(b0, b1, r, 12)[1:]]
            p += [(0, 0), (0, 0), (r * np.cos(a0), r * np.sin(a0))]
            codes = ([MPath.MOVETO] + [MPath.LINETO] * 11 + [MPath.CURVE4, MPath.CURVE4, MPath.CURVE4] +
                     [MPath.LINETO] * 11 + [MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
            c = (ribbon_colors[i][j] if isinstance(ribbon_colors, (list, np.ndarray)) and
                 isinstance(ribbon_colors[0], (list, np.ndarray)) else (ribbon_colors or arc_colors)[i])
            al = ribbon_alpha
            if emphasis is not None:
                al = ribbon_alpha if (i, j) in emphasis or (j, i) in emphasis else ribbon_alpha * 0.35
            ax.add_patch(PathPatch(MPath(p, codes), fc=c, ec="none", alpha=al, zorder=2))
            drawn.append((i, j, M[i, j]))
    lim = label_r + 0.55
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    return drawn


# ------------------------------------------------------------------ clustered heatmap with dendrograms
def dendro_heatmap(fig, axes: dict, M: np.ndarray, row_labels, col_labels, cmap, norm=None, cluster_rows=True,
                   cluster_cols=True, fmt=None, fs=FS_S - 1.2, metric="euclidean", method="average",
                   annotate_min=None, gap=0.05):
    """axes: {'heat': ax, 'rden': ax or None, 'cden': ax or None}; returns (row_order, col_order)."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from matplotlib.colors import to_hex
    R, C = M.shape
    ro, co = np.arange(R), np.arange(C)
    Mf = np.nan_to_num(M)
    if cluster_rows and R > 2:
        Zr = linkage(Mf, method=method, metric=metric)
        dr = dendrogram(Zr, orientation="left", no_plot=True)
        ro = np.array(dr["leaves"])[::-1]
        if axes.get("rden") is not None:
            _draw_dendro(axes["rden"], Zr, orientation="left", n=R)
    if cluster_cols and C > 2:
        Zc = linkage(Mf.T, method=method, metric=metric)
        dc = dendrogram(Zc, orientation="top", no_plot=True)
        co = np.array(dc["leaves"])
        if axes.get("cden") is not None:
            _draw_dendro(axes["cden"], Zc, orientation="top", n=C)
    ax = axes["heat"]
    A = M[np.ix_(ro, co)]
    for i in range(R):
        for j in range(C):
            v = A[i, j]
            if not np.isfinite(v):
                continue
            fc = to_hex(cmap(norm(v)) if norm is not None else cmap(v))
            ax.add_patch(FancyBboxPatch((j + gap / 2, i + gap / 2), 1 - gap, 1 - gap,
                                        boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec="none"))
            if fmt and (annotate_min is None or abs(v) >= annotate_min):
                ax.text(j + 0.5, i + 0.5, fmt.format(v).replace("-", "\u2212"), ha="center", va="center", fontsize=fs,
                        color=ink_on(fc))
    ax.set_xlim(0, C)
    ax.set_ylim(R, 0)
    ax.set_xticks(np.arange(C) + 0.5, [mt(col_labels[k]) for k in co])
    ax.set_yticks(np.arange(R) + 0.5, [mt(row_labels[k]) for k in ro])
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    return ro, co


def _draw_dendro(ax, Z, orientation: str, n: int, color=INK2, lw=0.6):
    from scipy.cluster.hierarchy import dendrogram
    d = dendrogram(Z, orientation=orientation, no_plot=True)
    ic, dc = np.array(d["icoord"]), np.array(d["dcoord"])
    for xs, ys in zip(ic, dc):
        if orientation == "left":
            ax.plot(ys, (n * 10 - np.array(xs)) / 10.0, color=color, lw=lw, solid_capstyle="butt")
        else:
            ax.plot(np.array(xs) / 10.0, ys, color=color, lw=lw, solid_capstyle="butt")
    if orientation == "left":
        ax.set_ylim(n, 0)
        ax.invert_xaxis()
    else:
        ax.set_xlim(0, n)
    ax.axis("off")


# ------------------------------------------------------------------ radar
def radar(ax, axes_labels: list[str], series: list[tuple[str, np.ndarray, str]], rmax=1.0, rings=(0.25, 0.5, 0.75, 1.0),
          fs=FS_S - 0.8):
    k = len(axes_labels)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False) + np.pi / 2
    for rr in rings:
        a = np.r_[ang, ang[:1]]
        ax.plot(rr * np.cos(a), rr * np.sin(a), color=GRID, lw=0.5, zorder=0)
    for a in ang:
        ax.plot([0, rmax * np.cos(a)], [0, rmax * np.sin(a)], color=GRID, lw=0.5, zorder=0)
    for a, lab in zip(ang, axes_labels):
        ca = np.cos(a)
        ax.text(1.1 * ca, 1.1 * np.sin(a), mt(lab), ha="center" if abs(ca) < 0.3 else ("left" if ca > 0 else "right"),
                va="center", fontsize=fs, color=INK)
    for name, vals, c in series:
        v = np.r_[vals, vals[:1]] / rmax
        a = np.r_[ang, ang[:1]]
        ax.fill(v * np.cos(a), v * np.sin(a), color=c, alpha=0.1, lw=0, zorder=2)
        ax.plot(v * np.cos(a), v * np.sin(a), color=c, lw=1.1, zorder=3)
        ax.scatter(v[:-1] * np.cos(ang), v[:-1] * np.sin(ang), s=9, color=c, edgecolors=SURFACE, linewidths=0.5,
                   zorder=4)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------------------------------------ SHAP beeswarm
def beeswarm(ax, shap: np.ndarray, vals: np.ndarray, y: float, cmap, width=0.36, nbins=120, size=1.4, seed=0,
             max_pts=2500):
    rng = np.random.default_rng(seed)
    s = np.asarray(shap, float)
    v = np.asarray(vals, float)
    if len(s) > max_pts:
        k = rng.choice(len(s), max_pts, replace=False)
        s, v = s[k], v[k]
    fin = np.isfinite(v)
    lo, hi = (np.nanpercentile(v[fin], 2), np.nanpercentile(v[fin], 98)) if fin.any() else (0, 1)
    col = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
    edges = np.linspace(s.min(), s.max() + 1e-12, nbins + 1)
    b = np.clip(np.digitize(s, edges) - 1, 0, nbins - 1)
    off = np.zeros(len(s))
    for k in np.unique(b):
        idx = np.flatnonzero(b == k)
        m = len(idx)
        order = rng.permutation(m)
        steps = (np.arange(m) + 1) // 2 * np.where(np.arange(m) % 2 == 0, 1, -1)
        off[idx[order]] = steps
    mx = np.abs(off).max() or 1
    off = off / mx * width
    ax.scatter(s[fin], y + off[fin], s=size, c=col[fin], cmap=cmap, vmin=0, vmax=1, lw=0, zorder=3, rasterized=True)
    ax.scatter(s[~fin], y + off[~fin], s=size, color="#b9b8b2", lw=0, zorder=3, rasterized=True)


# ------------------------------------------------------------------ network
def network(ax, pos: dict, nodes: list[dict], edges: list[dict], node_color, node_size, label_key="symbol",
            edge_style=None, fs=FS_S - 1.3, ring=None):
    es = edge_style or {}
    for e in edges:
        if e["a"] not in pos or e["b"] not in pos:
            continue
        st = es.get(e["kind"], {"color": AXIS, "lw": 0.5, "ls": "-"})
        (x0, y0), (x1, y1) = pos[e["a"]], pos[e["b"]]
        ax.plot([x0, x1], [y0, y1], color=st["color"], lw=st["lw"], linestyle=st.get("ls", "-"),
                alpha=st.get("alpha", 0.9), zorder=1, solid_capstyle="round")
    for n in nodes:
        k = n["gene_row"] if "gene_row" in n else n["id"]
        if k not in pos:
            continue
        x, y = pos[k]
        c = node_color(n)
        sz = node_size(n)
        ec = ring(n) if ring else SURFACE
        ax.scatter([x], [y], s=sz, color=c, edgecolors=ec, linewidths=0.9 if ec != SURFACE else 0.6, zorder=3)
        ax.text(x, y - 0.075 - np.sqrt(sz) / 400, n.get(label_key, ""), ha="center", va="top", fontsize=fs, color=INK,
                zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------------------------------------ slopegraph
def slopegraph(ax, a: np.ndarray, b: np.ndarray, c_up, c_down, c_same=MUTED, x=(0, 1), lw=0.6, alpha=0.55,
               jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    for u, v in zip(a, b):
        c = c_up if v > u + 1e-12 else (c_down if v < u - 1e-12 else c_same)
        j = rng.uniform(-jitter, jitter) if jitter else 0.0
        ax.plot(x, [u + j, v + j], color=c, lw=lw, alpha=alpha if c != c_same else 0.35, zorder=2)
    ax.scatter(np.full(len(a), x[0]), a, s=5, color=INK2, zorder=3, lw=0)
    ax.scatter(np.full(len(b), x[1]), b, s=5, color=INK2, zorder=3, lw=0)

"""Publication figures 1-5 (landscape, method, main results, budget, knowledge strata)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths8
from .pubcolor8 import AXIS, BLUE, CAT, ENTITY, GRID, INK, INK2, MUTED, NEUTRAL_MID, SURFACE, div_cmap_rev, \
    ink_on, seq_cmap
from .pubstyle8 import (FS, FS_L, FS_S, arrow, ci_h, ci_v, fam_ci, fam_curve_ci, fam_draws, fam_ratio_ci,
                        fig_mm, half_violin, heat_table, legend_handles, mt, panel, pct, rounded_box, sankey_band,
                        save, shrink, style)

PUB = paths8.RESULTS / "pub"
NUMS = paths8.RESULTS / "numbers_v8_results.json"

MLAB = {"deployed": "V8.1 部署模型", "channel_lgb_v6": "仅检索通道 LGB（V6 500）",
        "v71_direct_strict": "V7.1 直接融合（严格 500）", "v71_direct_v6": "V7.1 直接融合（V6 500）",
        "B": "基础排序器 B", "F7_v6": "F7 固定融合（V6 500）", "F7_strict": "F7 固定融合（严格 500）",
        "standalone_sF": "教师 $s_F$ 单独（全背景）", "standalone_r": "V7.1 核心 $r$ 单独（全背景）",
        "standalone_sF0": "无学习 SapBERT（全背景）", "standalone_sS": "教师序列分数（全背景）",
        "C_b0": "门控纠偏 β=0", "C_b1": "门控纠偏 β=1", "C_b2": "门控纠偏 β=2",
        "kdgene": "KDGene", "sign": "SIGN 式"}
MSHORT = {"deployed": "V8.1 部署", "channel_lgb_v6": "仅通道 LGB", "v71_direct_strict": "V7.1 直接融合",
          "B": "基础排序器 B", "standalone_sF": "教师 $s_F$ 单独", "v71_direct_v6": "V7.1（V6 500）",
          "F7_v6": "F7", "kdgene": "KDGene", "sign": "SIGN 式"}


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PUB / name)


def nums() -> dict:
    return json.loads(NUMS.read_text(encoding="utf-8"))


def fam_of() -> dict:
    d = load("diseases.parquet")
    return dict(zip(d["disease_id"], d["family_group"]))


# ------------------------------------------------------------------ shared derived tables
_CACHE: dict = {}


def per_disease_recall(K: int = 10, methods=None) -> pd.DataFrame:
    """Seed-averaged per-disease Recall@K from the query-level ranks (never a union of seeds)."""
    key = ("pdr", K, tuple(methods or ()))
    if key in _CACHE:
        return _CACHE[key]
    qr = load("query_ranks.parquet")
    if methods:
        qr = qr[qr["method"].isin(methods)]
    hit = (qr["rank"] > 0) & (qr["rank"] <= K)
    g = qr.assign(hit=hit).groupby(["method", "seed", "disease_id"], observed=True).agg(h=("hit", "sum"),
                                                                                          n=("hit", "size"))
    g["r"] = g["h"] / g["n"]
    out = g.groupby(["method", "disease_id"], observed=True).agg(recall=("r", "mean"), n=("n", "first"),
                                                                 seeds=("r", "size")).reset_index()
    _CACHE[key] = out
    return out


def recall_curve(method: str, kmax: int = 100) -> tuple[np.ndarray, list[str]]:
    """[n_disease, kmax] seed-averaged Recall@1..kmax for one method."""
    qr = load("query_ranks.parquet")
    q = qr[qr["method"] == method]
    grp = q.groupby(["seed", "disease_id"], observed=True).ngroup().to_numpy()
    n_g = grp.max() + 1
    M = np.zeros((n_g, kmax + 1))
    r = q["rank"].to_numpy().astype(int)
    ok = (r > 0) & (r <= kmax)
    np.add.at(M, (grp[ok], r[ok]), 1.0)
    M = np.cumsum(M, axis=1)[:, 1:]
    n = q.groupby(["seed", "disease_id"], observed=True).size().to_numpy()
    M = M / n[:, None]
    keys = q.groupby(["seed", "disease_id"], observed=True).size().reset_index()["disease_id"].to_numpy()
    df = pd.DataFrame(M)
    df["disease_id"] = keys
    avg = df.groupby("disease_id").mean()
    return avg.to_numpy(), avg.index.tolist()


# ================================================================== Figure 1
def fig1() -> dict:
    genes = load("genes.parquet")
    dis = load("diseases.parquet")
    H1 = 220
    fig = fig_mm(183, H1)
    gs = fig.add_gridspec(2, 3, left=0.062, right=0.975, top=1 - 8.3 / H1, bottom=1 - 118.4 / H1, hspace=0.46,
                          wspace=0.42, height_ratios=[1.0, 1.12], width_ratios=[1, 1, 1.08])
    src = {}

    # ---- a: Lorenz curves
    ax = fig.add_subplot(gs[0, 0])

    def lorenz(x):
        x = np.sort(np.asarray(x, float))
        c = np.cumsum(x)
        return np.r_[0, np.arange(1, len(x) + 1) / len(x)], np.r_[0, c / c[-1]]

    def gini(x):
        x = np.sort(np.asarray(x, float))
        n = len(x)
        return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)

    gx, gy = lorenz(genes["n_rel"])
    drel = dis.loc[dis["n_rel"] > 0, "n_rel"]
    dx, dy = lorenz(drel)
    gg, gd = gini(genes["n_rel"]), gini(drel)
    ax.fill_between(gx, gy, gx, color=CAT[0], alpha=0.07, lw=0)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.6)
    ax.plot(gx, gy, color=CAT[0], lw=1.3)
    ax.plot(dx, dy, color=CAT[1], lw=1.3)
    top10 = 1 - np.interp(0.9, gx, gy)
    ax.plot([0.9], [np.interp(0.9, gx, gy)], "o", ms=3.4, color=CAT[0], mec=SURFACE, mew=0.6, zorder=5)
    ax.annotate(mt(f"前 10% 的基因\n占 {top10:.0%} 的已知关联"), xy=(0.9, np.interp(0.9, gx, gy)),
                xytext=(0.42, 0.50), fontsize=FS_S, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=2))
    ax.set_xlabel("按关联数升序的累计比例")
    ax.set_ylabel("已知疾病–基因关联的累计比例")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(handles=legend_handles([(f"基因  Gini {gg:.2f}（n = {len(genes):,}）", CAT[0], "line"),
                                      (f"疾病  Gini {gd:.2f}（n = {len(drel):,}）", CAT[1], "line")]),
              loc="upper left", fontsize=FS_S)
    style(ax, grid="both")
    panel(fig, ax, "a", title="关联知识高度集中")
    src["a"] = pd.DataFrame({"entity": ["gene"] * len(gx) + ["disease"] * len(dx),
                             "cum_share_entities": np.r_[gx, dx], "cum_share_relations": np.r_[gy, dy]}).iloc[::20]

    # ---- b: complementary CDFs (log-log)
    ax = fig.add_subplot(gs[0, 1])
    series = [("基因：已知关联", genes.loc[genes["n_rel"] > 0, "n_rel"].to_numpy(), CAT[0]),
              ("疾病：已知关联", drel.to_numpy(), CAT[1]),
              ("基因：GO 基因–术语对", genes.loc[genes["n_go_pairs"] > 0, "n_go_pairs"].to_numpy(), CAT[2])]
    rows_b = []
    for lab, x, c in series:
        v = np.sort(x)
        u = np.unique(v)
        cc = 1.0 - np.searchsorted(v, u, side="left") / len(v)
        ax.plot(u, cc, color=c, lw=1.2, label=lab)
        rows_b.append(pd.DataFrame({"series": lab, "k": u, "P(X>=k)": cc}))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(mt("数量 $k$"))
    ax.set_ylabel(mt("$P(X \\geq k)$"))
    ax.legend(loc="lower left", fontsize=FS_S, handlelength=1.0)
    style(ax, grid="both")
    panel(fig, ax, "b", title="重尾分布")
    src["b"] = pd.concat(rows_b)

    # ---- c: conditional distribution of known relations given GO richness
    ax = fig.add_subplot(gs[0, 2])
    from scipy.stats import spearmanr
    rho = spearmanr(genes["n_go_pairs"], genes["n_rel"]).statistic
    xb = [(-1, 0, "0"), (0, 4, "1–4"), (4, 9, "5–9"), (9, 19, "10–19"), (19, 49, "20–49"), (49, 99, "50–99"),
          (99, 10 ** 9, "≥100")]
    yb = [(-1, 0, "0"), (0, 1, "1"), (1, 3, "2–3"), (3, 7, "4–7"), (7, 15, "8–15"), (15, 10 ** 9, "≥16")]
    H = np.zeros((len(yb), len(xb)))
    ncol = []
    for jx, (a0, a1, _) in enumerate(xb):
        colg = genes[(genes["n_go_pairs"] > a0) & (genes["n_go_pairs"] <= a1)]
        ncol.append(len(colg))
        for iy, (b0, b1, _) in enumerate(yb):
            H[iy, jx] = ((colg["n_rel"] > b0) & (colg["n_rel"] <= b1)).mean() if len(colg) else np.nan
    shrink(ax, fx=0.84)
    im = ax.imshow(H, cmap=seq_cmap(), vmin=0, vmax=1, aspect="auto", origin="lower", interpolation="nearest")
    from matplotlib.colors import to_hex
    for jx in range(len(xb)):
        for iy in range(len(yb)):
            v = H[iy, jx]
            if v >= 0.1:
                ax.text(jx, iy, f"{v:.0%}", ha="center", va="center", fontsize=FS_S - 1.0,
                        color=ink_on(to_hex(seq_cmap()(v))))
    for jx, n in enumerate(ncol):
        ax.text(jx, len(yb) - 0.38, f"{n:,}", ha="center", va="bottom", fontsize=FS_S - 1.1, color=INK2)
    ax.text(-0.62, len(yb) - 0.38, "n", ha="right", va="bottom", fontsize=FS_S - 1.1, color=INK2, style="italic")
    ax.set_xticks(range(len(xb)), [b[2] for b in xb], fontsize=FS_S - 0.2, rotation=22, ha="right")
    ax.set_yticks(range(len(yb)), [b[2] for b in yb])
    ax.set_xlabel("GO 注释数（基因–术语对）")
    ax.set_ylabel("已知疾病关联数")
    ax.set_ylim(-0.5, len(yb) + 0.05)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = ax.inset_axes([1.04, 0.0, 0.05, len(yb) / (len(yb) + 0.55)])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("列内基因比例", fontsize=FS_S - 0.3, labelpad=2)
    cb.ax.tick_params(labelsize=FS_S - 0.7, length=1.5)
    cb.ax.yaxis.set_major_formatter(__import__("matplotlib").ticker.PercentFormatter(1.0, decimals=0))
    cb.outline.set_visible(False)
    panel(fig, ax, "c", dx_mm=-8.5, title=f"注释越丰富，已知关联越多（ρ = {rho:.2f}）")
    src["c"] = pd.DataFrame(H, index=[b[2] for b in yb], columns=[b[2] for b in xb]).reset_index().rename(
        columns={"index": "known_relations"})

    # ---- d: UpSet of information layers
    sub = gs[1, :2].subgridspec(2, 3, width_ratios=[0.2, 0.3, 1], height_ratios=[1.0, 0.95], wspace=0.0,
                                hspace=0.07)
    ax_bar = fig.add_subplot(sub[0, 2])
    ax_mat = fig.add_subplot(sub[1, 2], sharex=ax_bar)
    ax_set = fig.add_subplot(sub[1, 0], sharey=ax_mat)
    ax_lab = fig.add_subplot(sub[1, 1], sharey=ax_mat)
    layers = [("has_go", "GO 注释"), ("has_seq", "蛋白序列"), ("has_expr", "组织表达（HPA）"),
              ("has_gene_hpo", "基因–表型（HPO）"), ("has_ppi", "物理互作（STRING）"), ("has_rel", "已知疾病关联")]
    keys = [k for k, _ in layers]
    combo = genes[keys].astype(int).astype(str).agg("".join, axis=1)
    cnt = genes.assign(combo=combo, prot=genes["gtype"] == "protein").groupby("combo").agg(
        n=("gene_row", "size"), prot=("prot", "sum")).sort_values("n", ascending=False)
    top = cnt.head(14)
    xs = np.arange(len(top))
    ax_bar.bar(xs, top["prot"], width=0.62, color=INK2, zorder=2)
    ax_bar.bar(xs, top["n"] - top["prot"], bottom=top["prot"], width=0.62, color=MUTED, alpha=0.55, zorder=2,
               edgecolor=SURFACE, linewidth=0.4)
    for xi, n in zip(xs, top["n"]):
        ax_bar.text(xi, n + top["n"].max() * 0.02, f"{n:,}", ha="center", va="bottom", fontsize=FS_S - 0.9,
                    color=INK2, rotation=90 if n >= 1000 else 90)
    ax_bar.set_ylim(0, top["n"].max() * 1.28)
    ax_bar.set_ylabel("交集基因数")
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.legend(handles=legend_handles([("蛋白编码", INK2, "box"), ("非蛋白编码", "#c4c3be", "box")]),
                  loc="upper right", fontsize=FS_S)
    ax_bar.yaxis.set_major_formatter(__import__("matplotlib").ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    style(ax_bar, grid="y")
    ax_bar.spines["bottom"].set_visible(False)
    for i, (k, lab) in enumerate(layers):
        yy = len(layers) - 1 - i
        ax_mat.axhspan(yy - 0.5, yy + 0.5, color="#f6f6f3" if i % 2 == 0 else SURFACE, zorder=0, lw=0)
        for xi, c in enumerate(top.index):
            on = c[i] == "1"
            ax_mat.plot(xi, yy, "o", ms=3.6, color=INK2 if on else "#dcdbd5", zorder=3, mec=SURFACE, mew=0.4)
    for xi, c in enumerate(top.index):
        ys = [len(layers) - 1 - i for i in range(len(layers)) if c[i] == "1"]
        if len(ys) > 1:
            ax_mat.plot([xi, xi], [min(ys), max(ys)], color=INK2, lw=1.0, zorder=2)
    ax_mat.set_ylim(-0.6, len(layers) - 0.4)
    ax_mat.set_yticks(range(len(layers)), [lab for _, lab in layers][::-1])
    ax_mat.tick_params(axis="y", length=0, labelleft=False)
    ax_mat.tick_params(axis="x", bottom=False, labelbottom=False)
    for s in ax_mat.spines.values():
        s.set_visible(False)
    sizes = [genes[k].sum() for k, _ in layers][::-1]
    ax_set.barh(range(len(layers)), sizes, height=0.55, color=MUTED, zorder=2)
    ax_set.set_xlim(max(sizes) * 1.05, 0)
    ax_set.set_xlabel("集合大小")
    ax_set.xaxis.set_major_formatter(__import__("matplotlib").ticker.FuncFormatter(
        lambda v, _: "0" if v == 0 else f"{v / 1000:.0f}k"))
    ax_set.set_xticks([20000, 10000, 0])
    ax_set.tick_params(axis="y", left=False, labelleft=False)
    style(ax_set, grid="x", left=False)
    ax_lab.set_xlim(0, 1)
    ax_lab.axis("off")
    for yy, (_, lab) in enumerate(layers[::-1]):
        ax_lab.text(0.95, yy, lab, ha="right", va="center", fontsize=FS_S, color=INK)
    panel(fig, ax_bar, "d", dx_mm=-6.0, x_ax=ax_set, title="基因信息层的可用性（UpSet，前 14 个交集）")
    src["d"] = cnt.reset_index().assign(layers="|".join(lab for _, lab in layers))

    # ---- e: disease information by known-relation burden
    ax = fig.add_subplot(gs[1, 2])
    bins = [(-1, 0, "0"), (0, 1, "1"), (1, 5, "2–5"), (5, 20, "6–20"), (20, 10 ** 9, ">20")]
    cats = [("定义文本 + HPO", BLUE[650]), ("仅定义文本", BLUE[450]), ("仅 HPO", BLUE[250]), ("仅名称/同义词", "#e2e1dc")]
    rows_e, ylabs = [], []
    for yi, (lo, hi, lab) in enumerate(bins):
        sub_d = dis[(dis["n_rel"] > lo) & (dis["n_rel"] <= hi)]
        n = len(sub_d)
        both = (sub_d["has_def"] & (sub_d["n_hpo"] > 0)).mean()
        dfo = (sub_d["has_def"] & (sub_d["n_hpo"] == 0)).mean()
        hpo = (~sub_d["has_def"] & (sub_d["n_hpo"] > 0)).mean()
        none = 1 - both - dfo - hpo
        left = 0.0
        for (cl, colr), v in zip(cats, (both, dfo, hpo, none)):
            ax.barh(yi, v, left=left, height=0.6, color=colr, edgecolor=SURFACE, linewidth=0.7, zorder=2)
            if v > 0.09:
                ax.text(left + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=FS_S - 0.6,
                        color=ink_on(colr))
            left += v
        ylabs.append(f"{lab}（{n:,}）")
        rows_e.append({"known_relations": lab, "n": n, "def_and_hpo": both, "def_only": dfo, "hpo_only": hpo,
                       "name_only": none})
    ax.set_yticks(range(len(bins)), ylabs)
    ax.set_ylim(len(bins) - 0.45, -1.35)
    ax.set_ylabel("已知关联数（疾病数）")
    ax.set_xlim(0, 1)
    pct(ax, "x")
    ax.set_xlabel("疾病比例")
    ax.legend(handles=legend_handles([(c, col, "box") for c, col in cats]), loc="upper left",
              bbox_to_anchor=(-0.02, 1.02), ncol=2, fontsize=FS_S - 0.3, columnspacing=0.8, handlelength=1.0)
    style(ax, grid=None)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel(fig, ax, "e", dx_mm=-17, title="疾病描述的完整度")
    src["e"] = pd.DataFrame(rows_e)
    from .pubfigs8x import p_category_heatmap, p_chord_categories
    from .pubstyle8 import label_mm
    label_mm(fig, 2, 136, "f", "12 个最大疾病大类之间共享的关联基因")
    src["f"] = p_chord_categories(fig, 6, 138, 74, 66)
    label_mm(fig, 86, 136, "g", "各疾病大类的知识覆盖指标（列内 z 分数着色，格内为原值）")
    src["g"] = p_category_heatmap(fig, 88, 140, 93, 76)
    meta = {"gini_genes": gg, "gini_diseases": gd, "top10_gene_share": top10, "spearman_go_rel": rho,
            "n_genes": int(len(genes)), "n_diseases_with_relations": int(len(drel))}
    return save(fig, "Fig1_knowledge_landscape", src, {"numbers": meta})

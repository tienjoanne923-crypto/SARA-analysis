"""Figures 4-5: candidate budget / availability-aware fusion / hit flow; knowledge-coverage strata."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from . import paths8
from .pubcolor8 import AXIS, BLUE, CAT, ENTITY, INK, INK2, MUTED, SURFACE, div_cmap_rev, ink_on, seq_cmap
from .pubstyle8 import (FS, FS_S, axmm, fam_ci, fam_curve_ci, fam_ratio_ci, fig_mm, label_mm, legend_handles, mt,
                        pct, sankey_band, save, style)

EVAL = paths8.RESULTS / "eval"
PUB = paths8.RESULTS / "pub"
PLAIN_GRAY = "#6f6e69"


def _ev(tag: str) -> pd.DataFrame:
    return pd.read_parquet(EVAL / f"eval_test_{tag}.parquet")


def _per(ev: pd.DataFrame, method: str, col: str = "recall10", seed: int | None = None) -> pd.Series:
    sub = ev[ev["method"] == method]
    if seed is not None:
        sub = sub[sub["seed"] == seed]
    return sub.groupby("disease_id")[col].mean()


# ====================================================================== Figure 4
def fig4() -> dict:
    from .pubfigs8a import fam_of
    fam = fam_of()
    H4 = 236
    fig = fig_mm(183, H4)
    src = {}

    # ---------------------------------------------------- a: candidate coverage vs budget K
    label_mm(fig, 2, 6, "a", "候选覆盖随预算 K 的变化（种子 42，完整背景重算）")
    ax = axmm(fig, 13, 11, 60, 46)
    br = pd.read_parquet(PUB / "budget_ranks.parquet")
    Ks = np.unique(np.round(np.logspace(1, np.log10(19861), 70)).astype(int))
    series = [("rB", "V6 检索顺序 $r_B$", CAT[1]), ("rS", "语义顺序 $r_S$（仅语义可用基因）", CAT[6]),
              ("r_plain", "普通 RRF（同一 $w_j$）", PLAIN_GRAY), ("r_aware", "缺失感知 RRF（$w_j$）", CAT[0])]
    rows = []
    ids = br["disease_id"].unique()
    pos = {d: i for i, d in enumerate(ids)}
    di = br["disease_id"].map(pos).to_numpy()
    nq = np.bincount(di)
    f = np.array([fam.get(x, x) for x in ids])
    at500 = {}
    for col, lab, colr in series:
        rk = br[col].to_numpy().astype(float)
        if col == "rS":
            rk = np.where(br["m"].to_numpy(), rk, np.inf)
        M = np.zeros((len(ids), len(Ks)))
        for k, K in enumerate(Ks):
            M[:, k] = np.bincount(di, weights=(rk <= K).astype(float), minlength=len(ids)) / nq
        mean, lo, hi = fam_curve_ci(M, f, 500)
        ax.fill_between(Ks, lo, hi, color=colr, alpha=0.13, lw=0)
        ax.plot(Ks, mean, color=colr, lw=1.6 if col == "r_plain" else 1.1, label=mt(lab),
                zorder=4 if col == "r_aware" else 3)
        c500 = (np.bincount(di, weights=(rk <= 500).astype(float), minlength=len(ids)) / nq)
        at500[col] = fam_ci(c500, f, 2000)
        rows.append(pd.DataFrame({"series": col, "K": Ks, "coverage": mean, "lo": lo, "hi": hi}))
    ax.set_xscale("log")
    ax.set_xlim(10, 19861)
    ax.set_ylim(0, 1.0)
    for K, lab in ((500, "K = 500"), (950, "≈950")):
        ax.axvline(K, color=AXIS, lw=0.7, zorder=0)
        ax.text(K * (0.93 if K == 500 else 1.07), 1.005, lab, fontsize=FS_S - 1.2, color=INK2, va="bottom",
                ha="right" if K == 500 else "left")
    for col, lab, colr in series:
        v = at500[col][0]
        ax.plot([500], [v], "o", ms=3.4, color=colr, mec=SURFACE, mew=0.6, zorder=5)
    txt = "\n".join(f"{s[1].split('（')[0].replace('$r_B$', '').replace('$r_S$', '').strip()}：{at500[s[0]][0]:.1%}"
                    for s in series)
    ax.text(0.03, 0.97, mt("K = 500 处的覆盖\n" + txt), transform=ax.transAxes, fontsize=FS_S - 1.1, color=INK,
            va="top", linespacing=1.3)
    ax.text(0.97, 0.30, "缺失感知与普通 RRF 的总体曲线几乎重合；\n差别集中在缺少语义的阳性（见 c）", transform=ax.transAxes,
            fontsize=FS_S - 1.4, color=INK2, ha="right", va="bottom", linespacing=1.25)
    ax.set_xlabel("候选预算 K（对数轴）")
    ax.set_ylabel("测试阳性进入前 K 的比例")
    pct(ax, "y")
    ax.legend(loc="lower right", fontsize=FS_S - 1.0, handlelength=1.2)
    style(ax, grid="both")
    src["a"] = pd.concat(rows)

    # ---------------------------------------------------- b: budget slope chart (E7)
    label_mm(fig, 93, 6, "b", "预算归因：同一方法在三种名单上的 Recall@10（种子 42）")
    ax = axmm(fig, 104, 11, 62, 46)
    main, v6p, br9 = _ev("main"), _ev("v6pool"), _ev("bridge950")
    pts = {
        "A": [(0, _per(v6p, "deployed")), (1, _per(main, "deployed", seed=42)), (2, _per(br9, "deployed"))],
        "ch": [(0, _per(main, "channel_lgb_v6", seed=42)), (1, _per(main, "B", seed=42)), (2, _per(br9, "B"))],
        "v71": [(0, _per(main, "v71_direct_v6", seed=42)), (1, _per(main, "v71_direct_strict", seed=42))],
    }
    sty = {"A": ("候选 A（部署规则）", CAT[0], "o"), "v71": ("V7.1 直接融合", CAT[2], "D"),
           "ch": ("仅检索通道排序器", CAT[1], "s")}
    rows_b = []
    offs = {"A": -0.05, "v71": 0.05, "ch": 0.0}
    last = {}
    for key in ("ch", "v71", "A"):
        lab, colr, mk = sty[key]
        xs, ys = [], []
        for x, s_ in pts[key]:
            e = fam_ci(s_.to_numpy(), np.array([fam.get(d, d) for d in s_.index]), 2000)
            xo = x + offs[key]
            ax.plot([xo, xo], [e[1], e[2]], color=colr, lw=0.9)
            ax.plot([xo], [e[0]], marker=mk, ms=3.8, color=colr, mec=SURFACE, mew=0.6, zorder=4)
            xs.append(xo)
            ys.append(e[0])
            rows_b.append({"series": lab, "list": ["V6 500", "严格 500", "约 950"][x], "recall10": e[0], "lo": e[1],
                           "hi": e[2]})
        ax.plot(xs, ys, color=colr, lw=1.0, zorder=3)
        last[key] = (xs[-1], ys[-1])
    ax.text(last["A"][0] + 0.1, last["A"][1], sty["A"][0], fontsize=FS_S - 0.9, color=INK, va="center")
    ax.text(last["ch"][0] + 0.1, last["ch"][1], sty["ch"][0], fontsize=FS_S - 0.9, color=INK, va="center")
    ax.annotate(sty["v71"][0], xy=last["v71"], xytext=(last["v71"][0] + 0.22, last["v71"][1] - 0.021),
                fontsize=FS_S - 0.9, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=CAT[2], lw=0.6, shrinkA=0, shrinkB=2))
    bud = __import__("json").loads((paths8.RESULTS / "numbers_v8_results.json").read_text(encoding="utf-8"))["budget"]
    lines = ["预算归因（家族 bootstrap 95% CI）"]
    for c, t in ((bud["v6pool_A_vs_channel"], "V6 500 上，A 减仅通道 LGB"),
                 (bud["strict_A_vs_v6pool_A"], "A：V6 500 → 严格 500"),
                 (bud["bridge_A_vs_strict_A"], "A：严格 500 → 约 950")):
        lines.append(f"{t}：{c['delta']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]")
    ax.text(0.02, 0.03, "\n".join(lines), transform=ax.transAxes, fontsize=FS_S - 1.3, color=INK2, va="bottom",
            linespacing=1.35)
    ax.text(0.98, 0.985, "V6 500 上通道排序为仅通道 LGB，严格 500 与 950 上为 B", transform=ax.transAxes,
            fontsize=FS_S - 1.7, color=MUTED, ha="right", va="top")
    ax.set_xticks([0, 1, 2], ["固定 V6 500", "共享严格 500", "约 950 桥接"])
    ax.set_xlim(-0.3, 2.75)
    ax.set_ylim(0.07, 0.195)
    ax.set_ylabel("Recall@10（疾病宏平均）")
    style(ax, grid="y")
    src["b"] = pd.DataFrame(rows_b)

    # ---------------------------------------------------- c: missing-information positives under three lists
    label_mm(fig, 2, 73, "c", "缺失信息阳性的候选覆盖")
    ax = axmm(fig, 25, 88, 36, 55)
    pr = _ev("plain_rrf")
    m42 = main[main["seed"] == 42]
    lists = [("V6 500（仅通道）", m42[m42["method"] == "channel_lgb_v6"], CAT[1]),
             ("缺失感知严格 500", m42[m42["method"] == "deployed"], CAT[0]),
             ("普通 RRF 严格 500", pr[pr["method"] == "deployed"], PLAIN_GRAY)]
    groups = [("ncrna", "非编码 RNA"), ("sem_unavail", "语义不可用"), ("nogo_protein", "无 GO 的蛋白编码")]
    rows_c = []
    for gi_, (g, glab) in enumerate(groups):
        for li, (llab, df, colr) in enumerate(lists):
            num, den = df[f"pool_{g}"].to_numpy(float), df[f"n_query_{g}"].to_numpy(float)
            fm = np.array([fam.get(d, d) for d in df["disease_id"]])
            ok = den > 0
            est, lo, hi = fam_ratio_ci(num[ok], den[ok], fm[ok], 2000)
            y = gi_ * 4 + li
            ax.barh(y, est, height=0.72, color=colr, zorder=2)
            ax.plot([lo, hi], [y, y], color=INK2, lw=0.55, zorder=3)
            for xx in (lo, hi):
                ax.plot([xx, xx], [y - 0.18, y + 0.18], color=INK2, lw=0.55, zorder=3)
            ax.text(hi + 0.008, y, f"{est:.1%}", va="center", fontsize=FS_S - 1.2, color=INK)
            rows_c.append({"group": glab, "list": llab, "pool_share": est, "lo": lo, "hi": hi, "queries": int(den.sum())})
        ax.text(-0.03, gi_ * 4 + 1, f"{glab}\n（{int(den.sum()):,} 条）", ha="right", va="center", fontsize=FS_S - 0.9,
                color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_ylim(len(groups) * 4 - 0.6, -0.8)
    ax.set_xlim(0, 0.36)
    pct(ax, "x")
    ax.set_xlabel("测试阳性进入 500 的比例（关系级，种子 42）")
    fig.legend(handles=legend_handles([(l_, c_, "box") for l_, _, c_ in lists]), loc="upper left",
               bbox_to_anchor=(4 / 183, 1 - 76.0 / H4), ncol=1, fontsize=FS_S - 1.0, handlelength=0.9)
    style(ax, grid="x", left=False)
    src["c"] = pd.DataFrame(rows_c)

    # ---------------------------------------------------- d: alluvial of rank bins
    label_mm(fig, 70, 73, "d", "逐关系名次流量（种子 42）")
    ax = axmm(fig, 72, 80, 60, 64)
    qr = pd.read_parquet(PUB / "query_ranks.parquet")
    q42 = qr[qr["seed"] == 42]
    a_ = q42[q42["method"] == "channel_lgb_v6"].set_index(["disease_id", "gene_row"])["rank"]
    b_ = q42[q42["method"] == "deployed"].set_index(["disease_id", "gene_row"])["rank"]
    qc = pd.read_parquet(PUB / "query_context.parquet").set_index(["disease_id", "gene_row"])["cold"]
    j = pd.concat([a_, b_, qc], axis=1, keys=["a", "b", "cold"]).dropna()

    def binr(r):
        r = np.asarray(r)
        return np.select([(r >= 1) & (r <= 10), (r >= 11) & (r <= 50), (r >= 51) & (r <= 500)], [0, 1, 2], 3)

    ba, bb = binr(j["a"].to_numpy()), binr(j["b"].to_numpy())
    names = ["前 10", "11–50", "51–500", "未入选"]
    T = np.zeros((4, 4), int)
    Tc = np.zeros((4, 4), int)
    np.add.at(T, (ba, bb), 1)
    np.add.at(Tc, (ba, bb), j["cold"].to_numpy().astype(int))
    total = T.sum()
    gap = total * 0.035
    xl, xr, wn = 0.0, 1.0, 0.075
    ax.set_xlim(-0.36, 1.36)
    yl = {}
    y = 0.0
    for i in range(4):
        yl[i] = y
        y += T[i].sum() + gap
    yr = {}
    y = 0.0
    for k in range(4):
        yr[k] = y
        y += T[:, k].sum() + gap
    H = max(sum(T[i].sum() for i in range(4)), sum(T[:, k].sum() for k in range(4))) + 3 * gap
    ax.set_ylim(H + gap, -gap * 1.2)
    ax.axis("off")
    node_c = [BLUE[650], BLUE[450], BLUE[250], "#d6d5cf"]
    for i in range(4):
        ax.add_patch(FancyBboxPatch((xl - wn, yl[i]), wn, T[i].sum(), boxstyle="round,pad=0,rounding_size=0.01",
                                    fc=node_c[i], ec="none"))
        ax.text(xl - wn - 0.03, yl[i] + T[i].sum() / 2, f"{names[i]}\n{T[i].sum():,}", ha="right", va="center",
                fontsize=FS_S - 1.0, color=INK, linespacing=1.2)
    for k in range(4):
        ax.add_patch(FancyBboxPatch((xr, yr[k]), wn, T[:, k].sum(), boxstyle="round,pad=0,rounding_size=0.01",
                                    fc=node_c[k], ec="none"))
        ax.text(xr + wn + 0.03, yr[k] + T[:, k].sum() / 2, f"{names[k]}\n{T[:, k].sum():,}", ha="left", va="center",
                fontsize=FS_S - 1.0, color=INK, linespacing=1.2)
    offl = dict(yl)
    offr = dict(yr)
    for i in range(4):
        for k in range(4):
            n = T[i, k]
            if n == 0:
                continue
            colr = CAT[0] if k < i else (CAT[1] if k > i else "#c9c8c2")
            alpha = 0.55 if k != i else 0.45
            sankey_band(ax, xl, offl[i], offl[i] + n, xr, offr[k], offr[k] + n, colr, alpha=alpha)
            offl[i] += n
            offr[k] += n
    ax.text(xl - wn / 2, -gap * 0.6, "仅通道 LGB", ha="center", va="bottom", fontsize=FS_S - 0.6, color=INK,
            fontweight="bold")
    ax.text(xr + wn / 2, -gap * 0.6, "V8.1 部署", ha="center", va="bottom", fontsize=FS_S - 0.6, color=INK,
            fontweight="bold")
    up = int(np.triu(T, 1).sum())
    down = int(np.tril(T, -1).sum())
    upc = int(np.tril(Tc, -1).sum())
    ax.legend(handles=legend_handles([(f"名次上升 {down:,} 条（其中冷 {upc:,}）", CAT[0], "box"),
                                      (f"名次下降 {up:,} 条", CAT[1], "box"), ("同一区间", "#c9c8c2", "box")]),
              loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=FS_S - 1.1, handlelength=0.9)
    src["d"] = pd.DataFrame(T, index=[f"channel_{n}" for n in names], columns=[f"deployed_{n}" for n in names]
                            ).reset_index()

    # ---------------------------------------------------- e: rank shift density
    label_mm(fig, 138, 73, "e", "阳性名次的逐对变化")
    ax = axmm(fig, 146, 80, 33, 33)
    edges = [(1, 1, "1"), (2, 3, "2–3"), (4, 10, "4–10"), (11, 20, "11–20"), (21, 50, "21–50"), (51, 100, "51–100"),
             (101, 200, "101–200"), (201, 500, "201–500"), (0, 0, "未入选")]

    def eb(r):
        r = np.asarray(r)
        out = np.full(r.shape, 8)
        for i_, (lo_, hi_, _) in enumerate(edges[:-1]):
            out[(r >= lo_) & (r <= hi_)] = i_
        return out

    ea, eb_ = eb(j["a"].to_numpy()), eb(j["b"].to_numpy())
    Mh = np.zeros((9, 9))
    np.add.at(Mh, (eb_, ea), 1)
    from matplotlib.colors import LogNorm
    im = ax.imshow(np.where(Mh > 0, Mh, np.nan), cmap=seq_cmap(), norm=LogNorm(vmin=1, vmax=Mh.max()),
                   origin="lower", aspect="auto")
    for i_ in range(9):
        ax.add_patch(FancyBboxPatch((i_ - 0.5, i_ - 0.5), 1, 1, boxstyle="square,pad=0", fc="none", ec=INK, lw=0.5))
    ax.set_xticks(range(9), [e_[2] for e_ in edges], rotation=90, fontsize=FS_S - 1.4)
    ax.set_yticks(range(9), [e_[2] for e_ in edges], fontsize=FS_S - 1.4)
    ax.set_xlabel("仅通道 LGB 中的名次", fontsize=FS_S - 0.5)
    ax.set_ylabel("V8.1 部署中的名次", fontsize=FS_S - 0.5)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cax = axmm(fig, 146, 124, 33, 1.6)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("阳性关系数（对数色标）", fontsize=FS_S - 1.2, labelpad=1)
    cb.ax.tick_params(labelsize=FS_S - 1.4, length=1.2)
    cb.outline.set_visible(False)
    better = float(np.triu(Mh, 1).sum())
    src["e"] = pd.DataFrame(Mh, index=[f"deployed_{e_[2]}" for e_ in edges],
                            columns=[f"channel_{e_[2]}" for e_ in edges]).reset_index()
    from .pubfigs8x import p_rank_density, p_rescue
    label_mm(fig, 2, 160, "f", "测试阳性在检索名次与语义名次上的联合分布（种子 42）")
    src["f"] = p_rank_density(fig, 12, 165, 70, 54)
    label_mm(fig, 92, 160, "g", "由 V6 500 到严格 500：语义救回与挤出")
    src["g"] = p_rescue(fig, 94, 168, 86, 46)
    meta = {"coverage_at_500": {k: v for k, v in at500.items()}, "flow_up": down, "flow_down": up,
            "flow_up_cold": upc, "flow_total": int(total), "better_cells": float(better)}
    return save(fig, "Fig4_budget_fusion_flow", src, {"numbers": meta})


# ====================================================================== Figure 5
def fig5() -> dict:
    from .pubfigs8a import MLAB, fam_of, load
    fam = fam_of()
    H5 = 250
    fig = fig_mm(183, H5)
    src = {}
    qr = load("query_ranks.parquet")
    qc = load("query_context.parquet")
    genes = load("genes.parquet").set_index("gene_row")
    dis = load("diseases.parquet").set_index("disease_id")
    meths = ["deployed", "v71_direct_strict", "channel_lgb_v6", "standalone_sF"]
    q = qr[qr["method"].isin(meths)].merge(qc[["disease_id", "gene_row", "ref_degree", "cold", "sem_src"]],
                                             on=["disease_id", "gene_row"], how="left")
    q["hit"] = ((q["rank"] > 0) & (q["rank"] <= 10)).astype(float)
    q["n_go"] = q["gene_row"].map(genes["n_go_pairs"])
    # registered cold (no reference edge and no gene HPO) = -1; degree 0 with gene HPO = 0; else the degree
    q["deg_c"] = np.where(q["cold"].astype(bool), -1, q["ref_degree"])

    def strata_panel(ax, col, bins, xlabel, key):
        rows = []
        cnt = []
        for bi, (lo, hi, lab) in enumerate(bins):
            sub = q[(q[col] > lo) & (q[col] <= hi)]
            cnt.append(sub[sub["method"] == "deployed"].groupby("seed").size().mean())
            for mi, m in enumerate(meths):
                s_ = sub[sub["method"] == m]
                per = s_.groupby(["disease_id", "seed"])["hit"].agg(["sum", "size"]).reset_index()
                per = per.groupby("disease_id")[["sum", "size"]].mean()
                fm = np.array([fam.get(d, d) for d in per.index])
                est, lo_, hi_ = fam_ratio_ci(per["sum"].to_numpy(), per["size"].to_numpy(), fm, 1000)
                rows.append({"stratum": lab, "method": m, "hit10": est, "lo": lo_, "hi": hi_,
                             "queries_per_seed": cnt[-1]})
        df = pd.DataFrame(rows)
        x = np.arange(len(bins))
        for mi, m in enumerate(meths):
            d_ = df[df["method"] == m]
            off = (mi - 1.5) * 0.11
            colr = ENTITY[m]
            ax.plot(x + off, d_["hit10"], color=colr, lw=1.0, marker="o", ms=3.0, mec=SURFACE, mew=0.5, zorder=3)
            for xi, (lo_, hi_) in enumerate(zip(d_["lo"], d_["hi"])):
                ax.plot([xi + off, xi + off], [lo_, hi_], color=colr, lw=0.7, zorder=2)
        ax.set_xticks(x, [b[2] for b in bins], fontsize=FS_S - 0.9)
        ax.set_xlabel(xlabel, fontsize=FS_S - 0.3)
        ax.set_ylabel("关系级前 10 恢复率", fontsize=FS_S - 0.3)
        pct(ax, "y")
        style(ax, grid="y")
        for xi, c in enumerate(cnt):
            ax.text(xi, 1.0, f"{c:,.0f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=FS_S - 1.6, color=INK2)
        return df

    # ---------------------------------------------------- a: by gene reference degree
    label_mm(fig, 2, 6, "a", "按基因在参考图中的度分层")
    ax = axmm(fig, 13, 14, 48, 40)
    bins_a = [(-2, -1, "冷"), (-1, 0, "0*"), (0, 1, "1"), (1, 3, "2–3"), (3, 7, "4–7"), (7, 15, "8–15"),
              (15, 31, "16–31"), (31, 10 ** 9, "≥32")]
    src["a"] = strata_panel(ax, "deg_c", bins_a, "基因的参考关系数（评价上下文）", "a")
    ax.text(-0.02, 1.045, "n", transform=ax.transAxes, fontsize=FS_S - 1.6, color=INK2, ha="right", style="italic")
    ax.text(1.0, -0.23, "冷：无参考边且无基因 HPO；0*：无参考边但有基因 HPO", transform=ax.transAxes, ha="right",
            fontsize=FS_S - 1.7, color=INK2)

    # ---------------------------------------------------- b: by GO richness
    label_mm(fig, 66, 6, "b", "按基因 GO 注释数分层")
    ax = axmm(fig, 77, 14, 48, 40)
    bins_b = [(-1, 0, "0"), (0, 4, "1–4"), (4, 9, "5–9"), (9, 19, "10–19"), (19, 49, "20–49"), (49, 99, "50–99"),
              (99, 10 ** 9, "≥100")]
    src["b"] = strata_panel(ax, "n_go", bins_b, "基因的 GO 基因–术语对数", "b")
    ax.text(-0.02, 1.045, "n", transform=ax.transAxes, fontsize=FS_S - 1.6, color=INK2, ha="right", style="italic")
    fig.legend(handles=legend_handles([(MLAB[m].replace("（全背景）", ""), ENTITY[m], "line") for m in meths]),
               loc="upper center", bbox_to_anchor=(69 / 183, 1 - 67.0 / H5), ncol=4, fontsize=FS_S - 0.8,
               columnspacing=1.2)

    # ---------------------------------------------------- c: disease information richness (delta heatmap)
    label_mm(fig, 131, 6, "c", "按疾病文本长度与 HPO 分层")
    ax = axmm(fig, 143, 16, 36, 24)
    ev = _ev("main")
    a_ = _per(ev, "deployed")
    b_ = _per(ev, "channel_lgb_v6")
    dd = pd.DataFrame({"d": a_ - b_})
    dd["tok"] = dis.reindex(dd.index)["n_tokens"].to_numpy()
    dd["hpo"] = dis.reindex(dd.index)["n_hpo"].to_numpy() > 0
    qs = np.quantile(dd["tok"], [0.25, 0.5, 0.75])
    dd["tq"] = np.digitize(dd["tok"], qs)
    Mx = np.full((2, 4), np.nan)
    Nx = np.zeros((2, 4), int)
    rows_c = []
    for r_, hv in enumerate((True, False)):
        for c_ in range(4):
            s_ = dd[(dd["hpo"] == hv) & (dd["tq"] == c_)]
            Nx[r_, c_] = len(s_)
            if len(s_) >= 20:
                e = fam_ci(s_["d"].to_numpy(), np.array([fam.get(x, x) for x in s_.index]), 1000)
                Mx[r_, c_] = e[0]
                rows_c.append({"hpo": hv, "token_quartile": c_ + 1, "n": len(s_), "delta": e[0], "lo": e[1], "hi": e[2]})
    lim = np.nanmax(np.abs(Mx))
    ax.imshow(Mx, cmap=div_cmap_rev(), vmin=-lim, vmax=lim, aspect="auto")
    from matplotlib.colors import Normalize, to_hex
    nrm = Normalize(-lim, lim)
    for r_ in range(2):
        for c_ in range(4):
            if np.isfinite(Mx[r_, c_]):
                fc = to_hex(div_cmap_rev()(nrm(Mx[r_, c_])))
                ax.text(c_, r_ - 0.1, f"{Mx[r_, c_]:+.3f}", ha="center", va="center", fontsize=FS_S - 0.8,
                        color=ink_on(fc))
                ax.text(c_, r_ + 0.25, f"n={Nx[r_, c_]:,}", ha="center", va="center", fontsize=FS_S - 1.7,
                        color=ink_on(fc))
    ax.set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"], fontsize=FS_S - 0.8)
    ax.set_yticks([0, 1], ["有 HPO", "无 HPO"], fontsize=FS_S - 0.6)
    ax.set_xlabel("疾病文本长度四分位（词片）", fontsize=FS_S - 0.5)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.text(0.5, -0.62, mt("格内：部署 − 通道 LGB 的 ΔR@10"), transform=ax.transAxes, ha="center",
            fontsize=FS_S - 1.3, color=INK2)
    src["c"] = pd.DataFrame(rows_c)

    # ---------------------------------------------------- d: disease categories
    label_mm(fig, 2, 74, "d", "按 MeSH 疾病大类（测试疾病可属于多个大类；三种子平均）")
    per = {m: _per(ev, m) for m in meths}
    d_cat = dis.reindex(per["deployed"].index)["cats"].fillna("")
    from .pubdata8 import MESH_CAT
    rows_d = []
    for code, name in MESH_CAT.items():
        idx = d_cat.index[d_cat.str.contains(code)]
        if len(idx) < 60:
            continue
        fm = np.array([fam.get(x, x) for x in idx])
        rec = {"code": code, "name": name, "n": len(idx)}
        for m in meths:
            rec[m] = float(per[m].reindex(idx).mean())
        dl = (per["deployed"].reindex(idx) - per["channel_lgb_v6"].reindex(idx)).to_numpy()
        e = fam_ci(dl, fm, 2000)
        rec.update({"delta": e[0], "lo": e[1], "hi": e[2]})
        dv = (per["deployed"].reindex(idx) - per["v71_direct_strict"].reindex(idx)).to_numpy()
        e2 = fam_ci(dv, fm, 2000)
        rec.update({"delta_v71": e2[0], "lo_v71": e2[1], "hi_v71": e2[2]})
        rows_d.append(rec)
    D = pd.DataFrame(rows_d).sort_values("delta", ascending=False).reset_index(drop=True)
    nR = len(D)
    top = 82
    rowh = min(3.9, 80 / max(nR, 1))
    axh = axmm(fig, 50, top, 60, rowh * nR)
    from .pubstyle8 import heat_table
    heat_table(axh, D[meths].to_numpy(), [f"{r.name}（{r.code}）" for r in D.itertuples()],
               [MLAB[m].replace("（全背景）", "").replace("（严格 500）", "").replace("（V6 500）", "") for m in meths],
               seq_cmap(), norm_by="all", fmt="{:.3f}", text_size=FS_S - 1.4, col_label_rot=0)
    axh.tick_params(axis="x", labelsize=FS_S - 1.1)
    axh.tick_params(axis="y", labelsize=FS_S - 0.9)
    axf = axmm(fig, 115, top, 36, rowh * nR)
    for i, r in enumerate(D.itertuples()):
        y = i + 0.5
        axf.plot([r.lo, r.hi], [y, y], color=CAT[0], lw=0.9)
        axf.plot([r.delta], [y], "o", ms=3.4, color=CAT[0], mec=SURFACE, mew=0.5, zorder=3)
        axf.plot([r.lo_v71, r.hi_v71], [y + 0.28, y + 0.28], color=CAT[2], lw=0.7, alpha=0.9)
        axf.plot([r.delta_v71], [y + 0.28], "D", ms=2.6, color=CAT[2], mec=SURFACE, mew=0.4, zorder=3)
    axf.axvline(0, color=AXIS, lw=0.7)
    axf.set_ylim(nR, 0)
    axf.set_yticks([])
    axf.set_xlabel("部署模型减对照的 Recall@10 差值", fontsize=FS_S - 0.6)
    axf.xaxis.set_label_position("bottom")
    style(axf, grid="x", left=False)
    axf.legend(handles=[Line2D([0], [0], color=CAT[0], marker="o", ms=3.4, lw=0.9, mec=SURFACE, label="对仅通道 LGB"),
                        Line2D([0], [0], color=CAT[2], marker="D", ms=2.6, lw=0.7, mec=SURFACE, label="对 V7.1 直接融合")],
               loc="lower right", fontsize=FS_S - 1.2, handlelength=1.3)
    axn = axmm(fig, 156, top, 24, rowh * nR)
    axn.barh(np.arange(nR) + 0.5, D["n"], height=0.62, color="#c9c8c2")
    for i, n in enumerate(D["n"]):
        axn.text(n + 8, i + 0.5, f"{n:,}", va="center", fontsize=FS_S - 1.5, color=INK2)
    axn.set_ylim(nR, 0)
    axn.set_yticks([])
    axn.set_xlabel("测试疾病数", fontsize=FS_S - 0.6)
    axn.set_xlim(0, D["n"].max() * 1.35)
    style(axn, grid="x", left=False)
    src["d"] = D
    from .pubfigs8x import p_ridge_categories
    label_mm(fig, 2, 172, "e", "各 MeSH 大类中逐病最佳阳性名次的提升：V8.1 部署相对仅通道 LGB（山脊图）")
    src["e"] = p_ridge_categories(fig, 4, 178, 176, 60)
    return save(fig, "Fig5_knowledge_strata", src)

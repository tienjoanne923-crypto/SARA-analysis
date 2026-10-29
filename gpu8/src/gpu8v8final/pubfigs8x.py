"""Additional high-complexity panels (added below the existing panels of Figures 1 and 3-10).

Each function draws one panel into a millimetre box of an existing figure and returns its source data.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from . import paths8
from .pubcolor8 import AXIS, BLUE, CAT, ENTITY, GRID, INK, INK2, MUTED, SURFACE, div_cmap, div_cmap_rev, seq_cmap
from .pubstyle8 import FS, FS_S, axmm, fam_ci, label_mm, legend_handles, mt, pct, style
from .pubviz8 import beeswarm, chord, dendro_heatmap, kde, network, radar, raincloud, ridgeline, slopegraph

PUB = paths8.RESULTS / "pub"
EVAL = paths8.RESULTS / "eval"
SHORT = {"C01": "感染", "C04": "肿瘤", "C05": "肌骨", "C06": "消化", "C07": "口腔", "C08": "呼吸", "C09": "耳鼻喉",
         "C10": "神经", "C11": "眼", "C12": "泌尿生殖", "C13": "女性生殖/妊娠", "C14": "心血管", "C15": "血液/淋巴",
         "C16": "先天/遗传", "C17": "皮肤/结缔", "C18": "营养/代谢", "C19": "内分泌", "C20": "免疫", "C23": "病理状态",
         "C25": "化学诱发", "C26": "创伤", "F03": "精神"}
LOGT = [np.log10(v) for v in (1, 10, 100, 500)]


def _log_ticks(ax, axis="x", vals=(1, 10, 100, 500)):
    t = [np.log10(v) for v in vals]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(t, [str(v) for v in vals])


# ====================================================================== Fig 1 f/g
def p_chord_categories(fig, x, y, w, h):
    sg = json.loads((PUB / "category_shared_genes.json").read_text(encoding="utf-8"))
    ind = pd.read_parquet(PUB / "category_indicators.parquet").set_index("code")
    codes = sg["codes"][:12]
    M = np.array(sg["shared"])[:12, :12]
    sizes = ind.loc[codes, "genes"].to_numpy(float)
    go_med = ind.loc[codes, "gene_go_median"].to_numpy(float)
    from matplotlib.colors import Normalize, to_hex
    nrm = Normalize(go_med.min(), go_med.max())
    arc_c = [to_hex(seq_cmap()(0.35 + 0.6 * nrm(v))) for v in go_med]
    union = sizes[:, None] + sizes[None, :] - M
    jac = np.where(union > 0, M / union, 0)
    np.fill_diagonal(jac, 0)
    iu = np.dstack(np.triu_indices(len(codes), 1))[0]
    top = sorted(((jac[i, j], i, j) for i, j in iu), reverse=True)[:6]
    emph = {(i, j) for _, i, j in top}
    rib = [[BLUE[450] if (i, j) in emph or (j, i) in emph else "#b9b8b2" for j in range(len(codes))]
           for i in range(len(codes))]
    ax = axmm(fig, x, y, w, h)
    chord(ax, M, [f"{SHORT.get(c, c)}" for c in codes], arc_c, ribbon_colors=rib, sizes=sizes, gap=0.05,
          ribbon_alpha=0.55, emphasis=emph, fs=FS_S - 1.2, label_r=1.13,
          label_fmt=lambda i: f"{SHORT.get(codes[i], codes[i])} {int(sizes[i]):,}")
    import matplotlib as mpl
    cax = axmm(fig, x + 4, y + h + 1.5, 30, 1.6)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=nrm, cmap=seq_cmap()), cax=cax, orientation="horizontal")
    cb.set_label("弧色：该类基因 GO 基因–术语对的中位数", fontsize=FS_S - 1.7, labelpad=1)
    cb.ax.tick_params(labelsize=FS_S - 1.8, length=1)
    cb.outline.set_visible(False)
    ax.text(0.99, -0.03,"弧长 ∝ 类内基因数；带宽 ∝ 共享基因数\n蓝色带：Jaccard 最高的 6 对", transform=ax.transAxes,
            ha="right", va="top", fontsize=FS_S - 1.7, color=INK2, linespacing=1.3)
    rows = [{"a": codes[i], "b": codes[j], "shared_genes": int(M[i, j]), "jaccard": float(jac[i, j])}
            for i, j in iu]
    return pd.DataFrame(rows)


def p_category_heatmap(fig, x, y, w, h):
    ind = pd.read_parquet(PUB / "category_indicators.parquet")
    cols = [("diseases", "疾病数", "{:.0f}"), ("rel_per_disease", "每病关联", "{:.0f}"), ("def_share", "有定义", "{:.0%}"),
            ("hpo_share", "有 HPO", "{:.0%}"), ("tokens_median", "文本词片", "{:.0f}"), ("gene_go_median", "基因 GO 对", "{:.0f}"),
            ("gene_rel_median", "基因关联数", "{:.0f}"), ("gene_ppi_share", "基因有互作", "{:.0%}"),
            ("gene_nogo_share", "基因无 GO", "{:.1%}")]
    V = ind[[c for c, _, _ in cols]].to_numpy(float)
    Z = (V - V.mean(axis=0)) / (V.std(axis=0) + 1e-12)
    from matplotlib.colors import Normalize
    ax_r = axmm(fig, x, y + 8, 8, h - 22)
    ax_h = axmm(fig, x + 9, y + 8, w - 31, h - 22)
    ax_c = axmm(fig, x + 9, y, w - 31, 7.5)
    ro, co = dendro_heatmap(fig, {"heat": ax_h, "rden": ax_r, "cden": ax_c}, Z,
                            [f"{SHORT.get(c, c)}（{c}）" for c in ind["code"]], [c[1] for c in cols], div_cmap_rev(),
                            norm=Normalize(-2.5, 2.5), fmt=None)
    for i, r in enumerate(ro):
        for j, c in enumerate(co):
            v = V[r, c]
            ax_h.text(j + 0.5, i + 0.5, cols[c][2].format(v), ha="center", va="center", fontsize=FS_S - 2.2,
                      color=INK if abs(Z[r, c]) < 1.4 else "#ffffff")
    ax_h.yaxis.tick_right()
    ax_h.tick_params(axis="y", labelsize=FS_S - 1.3)
    ax_h.xaxis.tick_bottom()
    ax_h.tick_params(axis="x", labelsize=FS_S - 1.3, rotation=90)
    return ind


# ====================================================================== Fig 2 e (UpSet)
SRC_ZH = [("a_pheno", "表型"), ("a_hier", "层级"), ("a_onto", "本体"), ("a_text", "文本"), ("a_gene_hpo", "基因 HPO"),
          ("a_rwr_pheno", "表型 RWR"), ("a_compat", "兼容")]


def upset_data() -> pd.DataFrame:
    dis = pd.read_parquet(PUB / "diseases.parquet")
    parts = []
    for j in range(5):
        di = pd.read_parquet(paths8.V6PASS / "main" / f"f{j}_outer" / "dinfo.parquet")
        ids = dis[(dis["role"] == "test_fold") & (dis["fold"] == j)]["disease_id"]
        parts.append(di[di["disease_id"].isin(ids)][["disease_id"] + [c for c, _ in SRC_ZH]].assign(fold=j))
    av = pd.concat(parts, ignore_index=True)
    ev = pd.read_parquet(EVAL / "eval_test_main.parquet")
    r = ev[ev["method"].isin(["deployed", "channel_lgb_v6"])].groupby(["method", "disease_id"])["recall10"].mean()
    av["r_dep"] = av["disease_id"].map(r["deployed"])
    av["r_ch"] = av["disease_id"].map(r["channel_lgb_v6"])
    av["family"] = av["disease_id"].map(dis.set_index("disease_id")["family_group"])
    return av


def p_upset(fig, x, y, w, h, top_n: int = 11):
    av = upset_data()
    cols = [c for c, _ in SRC_ZH]
    av["combo"] = av[cols].astype(int).astype(str).agg("".join, axis=1)
    cnt = av["combo"].value_counts()
    top_n = min(top_n, cnt.size)
    combos = cnt.index[:top_n].tolist()
    n_other = int(cnt.iloc[top_n:].sum())
    nC, nS = len(combos), len(cols)
    L = 34                                               # set-size bars + labels (mm)
    mw = w - L - 30                                      # matrix width
    ax_top = axmm(fig, x + L, y, mw, h * 0.34)
    ax_m = axmm(fig, x + L, y + h * 0.36, mw, h * 0.34)
    ax_d = axmm(fig, x + L, y + h * 0.73, mw, h * 0.27)
    ax_s = axmm(fig, x, y + h * 0.36, L - 14, h * 0.34)
    rows = []
    for k, cb in enumerate(combos):
        m = av["combo"] == cb
        sub = av[m]
        ax_top.bar(k, int(m.sum()), width=0.66, color=INK2, zorder=2)
        ax_top.text(k, int(m.sum()) + cnt.iloc[0] * 0.02, f"{int(m.sum()):,}", ha="center", va="bottom",
                    fontsize=FS_S - 2.0, color=INK2)
        on = [i for i, ch in enumerate(cb) if ch == "1"]
        for i in range(nS):
            ax_m.plot(k, i, "o", ms=3.4, color=INK2 if i in on else "#e2e1dc", mec="none", zorder=3)
        if len(on) > 1:
            ax_m.plot([k, k], [min(on), max(on)], color=INK2, lw=1.1, zorder=2)
        d = (sub["r_dep"] - sub["r_ch"]).to_numpy(float)
        e = fam_ci(d, sub["family"].fillna(sub["disease_id"]).to_numpy(), 2000)
        col = CAT[0] if e[1] > 0 else (CAT[1] if e[2] < 0 else MUTED)
        ax_d.plot([k, k], [e[1], e[2]], color=col, lw=1.0, solid_capstyle="round", zorder=2)
        ax_d.plot([k], [e[0]], "o", ms=3.2, color=col, mec=SURFACE, mew=0.5, zorder=3)
        rows.append({"combo": "".join(cb), "sources_available": "+".join(SRC_ZH[i][1] for i in on) or "无",
                     "n_diseases": int(m.sum()), "recall10_deployed": float(sub["r_dep"].mean()),
                     "recall10_channel": float(sub["r_ch"].mean()), "delta": e[0], "ci_low": e[1], "ci_high": e[2]})
    for i in range(nS):
        if i % 2 == 0:
            ax_m.axhspan(i - 0.5, i + 0.5, color="#f4f4f1", lw=0, zorder=0)
    ax_top.set_xlim(-0.6, nC - 0.4)
    ax_top.set_ylim(0, cnt.iloc[0] * 1.18)
    ax_top.set_xticks([])
    ax_top.set_ylabel("疾病数", fontsize=FS_S - 1.0)
    style(ax_top, grid="y", bottom=False)
    ax_top.spines["bottom"].set_visible(False)
    ax_m.set_xlim(-0.6, nC - 0.4)
    ax_m.set_ylim(nS - 0.5, -0.5)
    ax_m.set_yticks(range(nS), [z for _, z in SRC_ZH], fontsize=FS_S - 1.2)
    ax_m.set_xticks([])
    ax_m.tick_params(length=0, pad=1.5)
    for s in ax_m.spines.values():
        s.set_visible(False)
    tot = av[cols].sum().to_numpy()
    ax_s.barh(range(nS), tot, height=0.6, color="#b9b8b2", zorder=2)
    for i, v in enumerate(tot):
        ax_s.text(v + tot.max() * 0.06, i, f"{int(v):,}", va="center", ha="right", fontsize=FS_S - 2.1, color=INK2)
    ax_s.set_ylim(nS - 0.5, -0.5)
    ax_s.set_xlim(tot.max() * 1.45, 0)
    ax_s.set_yticks([])
    ax_s.set_xticks([0, 2000, 4000], ["0", "2k", "4k"], fontsize=FS_S - 1.8)
    ax_s.set_xlabel("来源可用的疾病数", fontsize=FS_S - 1.6, labelpad=1)
    style(ax_s, grid="x", left=False)
    ax_d.axhline(0, color=AXIS, lw=0.7, zorder=1)
    ax_d.set_xlim(-0.6, nC - 0.4)
    ax_d.set_xticks([])
    ax_d.set_ylabel(mt("ΔR@10"), fontsize=FS_S - 1.0)
    style(ax_d, grid="y", bottom=False)
    ax_d.spines["bottom"].set_visible(False)
    ax_d.text(nC - 0.4, ax_d.get_ylim()[1], "部署 − 仅通道 LGB（三种子均值；家族 bootstrap 95% CI）",
              ha="right", va="top", fontsize=FS_S - 2.0, color=INK2)
    # right-hand summary
    axr = axmm(fig, x + L + mw + 3, y, 27, h)
    axr.axis("off")
    n = len(av)
    full = int((av[cols].sum(axis=1) == nS).sum())
    none_ = int((av[cols].sum(axis=1) == 0).sum())
    lines = [f"测试疾病 {n:,} 个（5 折）", f"七个来源全部可用 {full:,}（{full / n:.0%}）",
             f"出现的可用性组合共 {cnt.size} 种" + (f"（图示前 {top_n} 种）" if n_other else ""),
             f"七个来源全不可用 {none_}"]
    for i, s in enumerate(lines):
        axr.text(0, 0.98 - i * 0.075, s, fontsize=FS_S - 1.7, color=INK2, va="top", transform=axr.transAxes)
    axr.text(0, 0.62, "区间颜色", fontsize=FS_S - 1.6, color=INK, va="top", transform=axr.transAxes, fontweight="bold")
    for i, (c, s) in enumerate(((CAT[0], "区间 > 0"), (CAT[1], "区间 < 0"), (MUTED, "区间含 0"))):
        yy = 0.55 - i * 0.07
        axr.plot([0.02, 0.1], [yy, yy], color=c, lw=1.2, transform=axr.transAxes)
        axr.plot([0.06], [yy], "o", ms=3.0, color=c, mec=SURFACE, mew=0.5, transform=axr.transAxes)
        axr.text(0.14, yy, s, fontsize=FS_S - 1.7, color=INK2, va="center", transform=axr.transAxes)
    return pd.DataFrame(rows)

# ====================================================================== Fig 3 g/h/i
def p_raincloud_best(fig, x, y, w, h):
    br = pd.read_parquet(PUB / "best_ranks.parquet")
    b42 = br[br["seed"] == 42]
    meths = ["deployed", "v71_direct_strict", "standalone_sF", "B", "channel_lgb_v6"]
    labs = {"deployed": "V8.1 部署", "v71_direct_strict": "V7.1 直接融合", "standalone_sF": "教师 $s_F$ 单独",
            "B": "基础排序器 B", "channel_lgb_v6": "仅通道 LGB"}
    ax = axmm(fig, x + 18, y, w - 30, h)
    data, cols, rows = [], [], []
    for m in meths:
        s_ = b42[b42["method"] == m]
        listed = s_[s_["best"] <= 500]["best"].to_numpy(float)
        data.append(np.log10(listed))
        cols.append(ENTITY.get(m, MUTED))
        rows.append({"method": m, "n": len(s_), "listed": len(listed), "unlisted_share": 1 - len(listed) / len(s_),
                     "median_best_rank_listed": float(np.median(listed))})
    ypos = np.arange(len(meths))[::-1] * 1.0
    raincloud(ax, data, cols, ypos=ypos, height=0.42, grid=np.linspace(0, np.log10(500), 300), strip_n=700)
    for yy, r in zip(ypos, rows):
        ax.text(np.log10(500) + 0.08, yy - 0.08, f"{r['unlisted_share']:.0%}", va="center", fontsize=FS_S - 1.1,
                color=INK2)
    ax.text(np.log10(500) + 0.08, ypos[0] + 0.45, "未入选", fontsize=FS_S - 1.3, color=MUTED)
    ax.set_yticks(ypos, [mt(labs[m]) for m in meths], fontsize=FS_S - 0.8)
    ax.set_ylim(-0.55, len(meths) - 0.4)
    _log_ticks(ax, "x")
    ax.set_xlim(-0.05, np.log10(500) + 0.02)
    ax.set_xlabel("每个疾病最靠前的阳性名次（种子 42，对数轴）", fontsize=FS_S - 0.5)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    return pd.DataFrame(rows)


def p_chord_hits(fig, x, y, w, h):
    qr = pd.read_parquet(PUB / "query_ranks.parquet")
    q = qr[(qr["seed"] == 42)]
    order = ["deployed", "v71_direct_strict", "channel_lgb_v6", "standalone_sF", "B"]
    labs = {"deployed": "V8.1 部署", "v71_direct_strict": "V7.1 直接融合", "channel_lgb_v6": "仅通道 LGB",
            "standalone_sF": "教师 $s_F$", "B": "基础排序器 B"}
    hit = {m: set(map(tuple, q[(q["method"] == m) & (q["rank"] > 0) & (q["rank"] <= 10)][["disease_id", "gene_row"]]
                      .itertuples(index=False, name=None))) for m in order}
    n = len(order)
    M = np.array([[len(hit[a] & hit[b]) if a != b else 0 for b in order] for a in order], float)
    sizes = np.array([len(hit[m]) for m in order], float)
    ax = axmm(fig, x, y, w, h)
    cols = [ENTITY.get(m, MUTED) for m in order]
    chord(ax, M, [mt(labs[m]) for m in order], cols, sizes=sizes, gap=0.06, ribbon_alpha=0.42, fs=FS_S - 1.0,
          label_fmt=lambda i: mt(f"{labs[order[i]]}\n{int(sizes[i]):,}"))
    uniq = {m: len(hit[m] - set().union(*[hit[o] for o in order if o != m])) for m in order}
    rows = [{"a": order[i], "b": order[j], "shared_top10_hits": int(M[i, j])} for i in range(n) for j in range(i + 1, n)]
    rows += [{"a": m, "b": "(unique)", "shared_top10_hits": uniq[m]} for m in order]
    ax.text(0.5, -0.02, f"V8.1 部署独有 {uniq['deployed']:,} 个；仅通道 LGB 独有 {uniq['channel_lgb_v6']:,} 个",
            transform=ax.transAxes, ha="center", fontsize=FS_S - 1.3, color=INK2)
    return pd.DataFrame(rows)


def p_radar(fig, x, y, w, h, R):
    tab = {r["method"]: r for r in R["main"]["table"]}
    br = pd.read_parquet(PUB / "best_ranks.parquet")
    mrr = br.groupby(["method", "disease_id"])["mrr"].mean().groupby("method").mean()
    meths = ["deployed", "v71_direct_strict", "channel_lgb_v6", "standalone_sF"]
    labs = {"deployed": "V8.1 部署", "v71_direct_strict": "V7.1 直接融合", "channel_lgb_v6": "仅通道 LGB",
            "standalone_sF": "教师 $s_F$ 单独"}
    keys = [("recall10", "R@10"), ("recall20", "R@20"), ("recall50", "R@50"), ("hit10", "Hit@10"),
            ("coverage500", "覆盖@500"), ("cold_recall10", "$R^c$@10"), ("mrr", "MRR")]
    vals = {m: np.array([mrr.get(m, np.nan) if k == "mrr" else tab[m][k] for k, _ in keys], float) for m in meths}
    mx = np.nanmax(np.vstack(list(vals.values())), axis=0)
    ax = axmm(fig, x, y, w, h)
    radar(ax, [k[1] for k in keys], [(labs[m], vals[m] / mx, ENTITY[m]) for m in meths], fs=FS_S - 1.2)
    ax.legend(handles=legend_handles([(labs[m], ENTITY[m], "line") for m in meths]), loc="upper center",
              bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=FS_S - 1.4, handlelength=1.0, columnspacing=0.8)
    ax.text(0.5, 1.0, "各轴除以四种方法中的最大值", transform=ax.transAxes, ha="center", fontsize=FS_S - 1.6, color=INK2)
    rows = [{"method": m, **{k: float(v) for (k, _), v in zip(keys, vals[m])}} for m in meths]
    return pd.DataFrame(rows)


# ====================================================================== Fig 4 f/g
def p_rank_density(fig, x, y, w, h):
    br = pd.read_parquet(PUB / "budget_ranks.parquet")
    qc = pd.read_parquet(PUB / "query_context.parquet")[["fold", "disease_id", "gene_row", "cold"]]
    br = br.merge(qc, on=["fold", "disease_id", "gene_row"], how="left")
    thr = pd.read_parquet(PUB / "budget_thresholds.parquet")
    s_ = br[br["m"]]
    # integer ranks are spread uniformly inside [r, r+1) for display only, so the log axes show no lattice gaps
    jit = np.random.default_rng(7)
    lx = np.log10(s_["rB"].to_numpy(float) + jit.random(len(s_)))
    ly = np.log10(s_["rS"].to_numpy(float) + jit.random(len(s_)))
    ax = axmm(fig, x, y + 9, w - 11, h - 9)
    axt = axmm(fig, x, y, w - 11, 8.5)
    axr = axmm(fig, x + w - 10.5, y + 9, 10, h - 9)
    from matplotlib.colors import LogNorm
    import matplotlib.patheffects as pe
    ext = np.log10(19862)
    hb = ax.hexbin(lx, ly, gridsize=40, extent=(0, ext, 0, ext), mincnt=1, cmap=seq_cmap(),
                   norm=LogNorm(), linewidths=0.15, edgecolors=SURFACE, zorder=1)
    cold = s_["cold"].fillna(False).to_numpy(bool)
    from scipy.stats import gaussian_kde
    halo = [pe.Stroke(linewidth=2.2, foreground=SURFACE), pe.Normal()]
    pts = np.vstack([lx[cold], ly[cold]])
    if pts.shape[1] > 50:
        k = gaussian_kde(pts[:, np.random.default_rng(0).choice(pts.shape[1], min(4000, pts.shape[1]), False)])
        gx, gy = np.meshgrid(np.linspace(0, ext, 90), np.linspace(0, ext, 90))
        zz = k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        lv = np.quantile(zz[zz > zz.max() * 0.02], [0.5, 0.8, 0.95])
        cs = ax.contour(gx, gy, zz, levels=lv, colors=[INK], linewidths=[0.6, 0.8, 1.0], linestyles="--", zorder=4)
        cs.set_path_effects(halo)
    t5 = thr[thr["w"] == 0.5]
    th = t5["R500_aware"].median()
    r = np.logspace(0, np.log10(19861), 300)
    rs_curve = 1 / ((th - 0.5 / (60 + r)) / 0.5) - 60
    ok = rs_curve > 0
    ax.plot(np.log10(r[ok]), np.log10(np.clip(rs_curve[ok], 1, None)), color=INK, lw=1.1, zorder=5,
            path_effects=halo)
    for v in (np.log10(500),):
        ax.axvline(v, color=CAT[1], lw=0.8, zorder=3, path_effects=halo)
        ax.axhline(v, color=CAT[6], lw=0.8, zorder=3, path_effects=halo)
    n = len(s_)
    q = {"均 ≤500": ((s_["rB"] <= 500) & (s_["rS"] <= 500)).mean(), "仅检索 ≤500": ((s_["rB"] <= 500) & (s_["rS"] > 500)).mean(),
         "仅语义 ≤500": ((s_["rB"] > 500) & (s_["rS"] <= 500)).mean(), "均 >500": ((s_["rB"] > 500) & (s_["rS"] > 500)).mean()}
    for (lab, v), (tx, ty) in zip(q.items(), ((0.3, 0.3), (1.2, 3.9), (3.6, 0.3), (3.05, 4.0))):
        ax.text(tx, ty, f"{lab}\n{v:.0%}", fontsize=FS_S - 1.3, color=INK, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.8))
    _log_ticks(ax, "x", (1, 10, 100, 1000, 10000))
    _log_ticks(ax, "y", (1, 10, 100, 1000, 10000))
    ax.set_xlim(0, ext)
    ax.set_ylim(0, ext)
    ax.set_xlabel(mt("V6 检索名次 $r_B$"), fontsize=FS_S - 0.4)
    ax.set_ylabel(mt("语义名次 $r_S$"), fontsize=FS_S - 0.4)
    bins = np.linspace(0, ext, 56)
    axt.hist(lx, bins=bins, color="#d6d5cf", lw=0)
    axt.hist(lx[cold], bins=bins, color=INK2, lw=0)
    axr.hist(ly, bins=bins, orientation="horizontal", color="#d6d5cf", lw=0)
    axr.hist(ly[cold], bins=bins, orientation="horizontal", color=INK2, lw=0)
    for a_ in (axt, axr):
        a_.axis("off")
    axt.set_xlim(ax.get_xlim())
    axr.set_ylim(ax.get_ylim())
    import matplotlib as mpl
    cax = axmm(fig, x + w - 10.5, y + 0.6, 10, 1.3)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=hb.norm, cmap=seq_cmap()), cax=cax, orientation="horizontal")
    cb.set_label("每格阳性数", fontsize=FS_S - 2.0, labelpad=0.5)
    cb.ax.tick_params(labelsize=FS_S - 2.2, length=1, pad=0.8)
    cb.outline.set_visible(False)
    W_, H_ = fig.get_size_inches() * 25.4
    fig.legend(handles=[Line2D([0], [0], color=INK, lw=0.9, ls="--", label="参考路径冷阳性（密度等高线）"),
                        Line2D([0], [0], color=INK, lw=1.1, label="缺失感知 RRF 中位入选边界"),
                        Line2D([0], [0], color=CAT[1], lw=0.8, label=mt("$r_B$ = 500")),
                        Line2D([0], [0], color=CAT[6], lw=0.8, label=mt("$r_S$ = 500"))],
               loc="upper left", bbox_to_anchor=(x / W_, 1 - (y + h + 7.5) / H_), ncol=4, fontsize=FS_S - 1.7,
               handlelength=1.4, columnspacing=0.7)
    return pd.DataFrame([{"region": k, "share": v, "n_semantic_available_positives": n} for k, v in q.items()])


def p_rescue(fig, x, y, w, h):
    br = pd.read_parquet(PUB / "budget_ranks.parquet")
    qc = pd.read_parquet(PUB / "query_context.parquet")[["fold", "disease_id", "gene_row", "cold"]]
    genes = pd.read_parquet(PUB / "genes.parquet").set_index("gene_row")
    br = br.merge(qc, on=["fold", "disease_id", "gene_row"], how="left")
    br["gtype"] = br["gene_row"].map(genes["gtype"])
    br["has_go"] = br["gene_row"].map(genes["has_go"])
    groups = [("全部阳性", br.index == br.index), ("参考路径冷", br["cold"].fillna(False)),
              ("非冷", ~br["cold"].fillna(False)), ("蛋白编码有 GO", (br["gtype"] == "protein") & br["has_go"]),
              ("蛋白编码无 GO", (br["gtype"] == "protein") & ~br["has_go"]), ("非编码 RNA", br["gtype"] == "ncrna")]
    cats = [("两者都入选", INK2), ("语义救回", CAT[0]), ("被挤出", CAT[1]), ("两者都未入选", "#e2e1dc")]
    ax = axmm(fig, x + 20, y, w - 20, h)
    rows = []
    for i, (lab, m) in enumerate(groups):
        s_ = br[m]
        v6, st = s_["rB"] <= 500, s_["r_aware"] <= 500
        shares = [(v6 & st).mean(), (~v6 & st).mean(), (v6 & ~st).mean(), (~v6 & ~st).mean()]
        left = 0
        for (cl, c), v in zip(cats, shares):
            ax.barh(i, v, left=left, height=0.62, color=c, edgecolor=SURFACE, linewidth=0.6)
            if v > 0.06:
                from .pubcolor8 import ink_on
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=FS_S - 1.6, color=ink_on(c))
            left += v
        rows.append({"group": lab, "n": int(m.sum()), **{cl: v for (cl, _), v in zip(cats, shares)}})
    ax.set_yticks(range(len(groups)), [f"{g[0]}（{int(g[1].sum()):,}）" for g in groups], fontsize=FS_S - 1.0)
    ax.set_ylim(len(groups) - 0.4, -0.6)
    ax.set_xlim(0, 1)
    pct(ax, "x")
    ax.set_xlabel("测试阳性比例（V6 500 → 严格 500；种子 42）", fontsize=FS_S - 0.5)
    ax.legend(handles=legend_handles([(c, col, "box") for c, col in cats]), loc="upper center",
              bbox_to_anchor=(0.45, -0.22), ncol=4, fontsize=FS_S - 1.3, handlelength=0.9, columnspacing=0.8)
    style(ax, grid=None, left=False)
    ax.tick_params(axis="y", length=0)
    return pd.DataFrame(rows)


# ====================================================================== Fig 5 e
def p_ridge_categories(fig, x, y, w, h):
    br = pd.read_parquet(PUB / "best_ranks.parquet")
    b42 = br[br["seed"] == 42]
    dis = pd.read_parquet(PUB / "diseases.parquet").set_index("disease_id")
    d5 = pd.read_csv(paths8.MANUSCRIPT / "figures_v2" / "source_data" / "Fig5_knowledge_strata_d.csv", encoding="utf-8-sig")
    codes = d5.sort_values("delta", ascending=False)["code"].tolist()
    piv = b42.pivot_table(index="disease_id", columns="method", values="best")
    piv = piv[(piv["deployed"] <= 500) | (piv["channel_lgb_v6"] <= 500)]
    lift = np.log10(piv["channel_lgb_v6"]) - np.log10(piv["deployed"])
    grid = np.linspace(-2.75, 2.75, 400)
    from matplotlib.colors import Normalize, to_hex
    means = {}
    for c in codes:
        ids = dis.index[dis["cats"].fillna("").str.contains(c)]
        means[c] = float(lift.reindex(ids).dropna().mean())
    lim = max(abs(v) for v in means.values())
    nrm = Normalize(-lim, lim)
    series, rows = [], []
    for c in codes:
        ids = dis.index[dis["cats"].fillna("").str.contains(c)]
        v = lift.reindex(ids).dropna().to_numpy(float)
        col = to_hex(div_cmap_rev()(nrm(means[c])))
        series.append((f"{SHORT.get(c, c)}", [(v, col)]))
        rows.append({"code": c, "n": len(v), "mean_log10_lift": means[c], "better": float((v > 0).mean()),
                     "worse": float((v < 0).mean()), "tie": float((v == 0).mean())})
    ax = axmm(fig, x + 14, y, w - 40, h)
    ridgeline(ax, series, grid, overlap=1.5, label_x=-2.85, fill_alpha=0.55, bw=0.16, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.6, zorder=0)
    n = len(series)
    for i, r in enumerate(rows):
        ax.text(2.85, n - 1 - i + 0.12, f"↑{r['better']:.0%}  ↓{r['worse']:.0%}  n={r['n']:,}", fontsize=FS_S - 1.7,
                color=INK2, va="bottom", ha="left")
    ax.set_xlim(-2.75, 2.75)
    ax.set_xticks([-2, -1, 0, 1, 2], ["−2", "−1", "0", "+1", "+2"])
    ax.set_xlabel(mt("名次提升 $\\log_{10}$(仅通道最佳名次) − $\\log_{10}$(部署最佳名次)（未入选记为 501；种子 42）"),
                  fontsize=FS_S - 0.6)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.text(0.03, 1.0, "← 仅通道 LGB 更靠前", transform=ax.transAxes, fontsize=FS_S - 1.2, color=INK2, va="bottom")
    ax.text(0.97, 1.0, "V8.1 部署更靠前 →", transform=ax.transAxes, fontsize=FS_S - 1.2, color=INK2, va="bottom",
            ha="right")
    import matplotlib as mpl
    cax = axmm(fig, x + w - 24, y + h + 4.2, 20, 1.3)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=nrm, cmap=div_cmap_rev()), cax=cax, orientation="horizontal")
    cb.set_label("填色：该类平均提升", fontsize=FS_S - 1.9, labelpad=0.8)
    cb.ax.tick_params(labelsize=FS_S - 2.1, length=1, pad=0.8)
    cb.outline.set_visible(False)
    return pd.DataFrame(rows)


# ====================================================================== Fig 6 e/f/g
FEAT_SHORT = None


def _feat_label(f):
    from .pubdata8 import FEAT_ZH
    return FEAT_ZH.get(f, f)


def p_shap_beeswarm(fig, x, y, w, h, top_k=14):
    sh = pd.read_parquet(PUB / "shap_f0_s42.parquet")
    feats = [c[5:] for c in sh.columns if c.startswith("shap_") and c != "shap_base"]
    imp = {f: np.abs(sh[f"shap_{f}"]).mean() for f in feats}
    top = sorted(imp, key=lambda k: -imp[k])[:top_k]
    ax = axmm(fig, x + 30, y, w - 34, h)
    cm = div_cmap()
    for i, f in enumerate(top):
        beeswarm(ax, sh[f"shap_{f}"].to_numpy(), sh[f"x_{f}"].to_numpy(), top_k - 1 - i, cm, width=0.38, size=1.1,
                 max_pts=2200, seed=i)
    ax.axvline(0, color=AXIS, lw=0.7, zorder=1)
    ax.set_yticks(range(top_k), [mt(_feat_label(f)) for f in top][::-1], fontsize=FS_S - 1.1)
    ax.set_ylim(-0.7, top_k - 0.3)
    ax.set_xlabel("SHAP 值（对排序分数的贡献）", fontsize=FS_S - 0.5)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    import matplotlib as mpl
    cax = axmm(fig, x + w - 3, y + 4, 1.6, h - 8)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cm), cax=cax)
    cb.set_ticks([0, 1], labels=["低", "高"])
    cb.set_label("特征值", fontsize=FS_S - 1.3, labelpad=1)
    cb.ax.tick_params(labelsize=FS_S - 1.5, length=1)
    cb.outline.set_visible(False)
    return pd.DataFrame({"feature": top, "mean_abs_shap": [imp[f] for f in top],
                         "label": [_feat_label(f) for f in top]})


RB_TICKS = (1, 10, 100, 500, 5000)


def _rb_norm():
    """Colour scale of the V6 retrieval rank feature ret_rank = ln(500 / r_B), centred on r_B = 500."""
    from matplotlib.colors import TwoSlopeNorm
    return TwoSlopeNorm(0.0, np.log(500 / 19861), np.log(500.0))


def p_shap_dependence(fig, x, y, w, h, feat: str, xlab: str, xform=None, logx: bool = False, xticks=None,
                      color_feat: str = "ret_rank", ylab: str | None = None, cbar_box=None, legend=None):
    """SHAP dependence of one feature, points coloured by an interacting feature (default: V6 rank r_B,
    diverging around r_B = 500).  ``xform`` maps the stored feature to the plotted quantity (e.g. rank)."""
    sh = pd.read_parquet(PUB / "shap_f0_s42.parquet")
    s_ = sh[np.isfinite(sh[f"x_{feat}"])]
    xv = s_[f"x_{feat}"].to_numpy(float)
    yv = s_[f"shap_{feat}"].to_numpy(float)
    cv = s_[f"x_{color_feat}"].to_numpy(float)
    xp = xform(xv) if xform else xv
    ax = axmm(fig, x, y, w, h)
    rng = np.random.default_rng(1)
    k = rng.choice(len(xv), min(12000, len(xv)), replace=False)
    k = k[np.argsort(np.abs(cv[k]))]          # extreme interaction values drawn last (on top)
    nrm = _rb_norm()
    ax.scatter(xp[k], yv[k], s=1.3, c=cv[k], cmap=div_cmap(), norm=nrm, lw=0, alpha=0.75, rasterized=True, zorder=2)
    pos = s_["positive"].to_numpy(bool)
    kp = np.flatnonzero(pos)
    kp = rng.choice(kp, min(700, len(kp)), replace=False)
    ax.scatter(xp[kp], yv[kp], s=2.4, facecolors="none", edgecolors=INK, linewidths=0.18, alpha=0.5, zorder=3,
               rasterized=True)
    qs = np.quantile(xv, np.linspace(0, 1, 31))
    mids, means, lo_, hi_ = [], [], [], []
    for a_, b_ in zip(qs[:-1], qs[1:]):
        m = (xv >= a_) & (xv <= b_)
        if m.sum() > 30:
            mids.append(np.median(xv[m]))
            means.append(np.mean(yv[m]))
            lo_.append(np.quantile(yv[m], 0.1))
            hi_.append(np.quantile(yv[m], 0.9))
    mp = xform(np.array(mids)) if xform else np.array(mids)
    import matplotlib.patheffects as pe
    ax.fill_between(mp, lo_, hi_, color=INK, alpha=0.08, lw=0, zorder=1)
    ax.plot(mp, means, color=INK, lw=1.1, zorder=6, path_effects=[pe.Stroke(linewidth=2.3, foreground=SURFACE),
                                                                  pe.Normal()])
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    if logx:
        ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks, [str(t) for t in xticks])
        ax.minorticks_off()
    ax.set_xlabel(mt(xlab), fontsize=FS_S - 0.6)
    ax.set_ylabel(mt(ylab or f"SHAP：{_feat_label(feat)}"), fontsize=FS_S - 0.7)
    style(ax, grid="both")
    if legend:
        ax.legend(handles=[Line2D([0], [0], marker="o", color="none", mec=INK, mew=0.4, ms=2.6, label="留出阳性（抽样 700）"),
                           Line2D([0], [0], color=INK, lw=1.1, label="分箱均值（阴影 P10–P90）")],
                  loc=legend, fontsize=FS_S - 2.0, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2,
                  framealpha=0.85, edgecolor="none")
    if cbar_box is not None:
        import matplotlib as mpl
        cax = axmm(fig, *cbar_box)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=nrm, cmap=div_cmap()), cax=cax, orientation="horizontal")
        cb.set_ticks([np.log(500 / t) for t in RB_TICKS][::-1], labels=[str(t) for t in RB_TICKS][::-1])
        cb.set_label(mt("点色：V6 检索名次 $r_B$（以 500 为中点）"), fontsize=FS_S - 1.8, labelpad=0.8)
        cb.ax.tick_params(labelsize=FS_S - 2.0, length=1, pad=0.8)
        cb.outline.set_visible(False)
    return pd.DataFrame({"x_bin_median": np.round(mp, 6), "mean_shap": means, "shap_p10": lo_, "shap_p90": hi_})


def p_shap_family_strata(fig, x, y, w, h):
    """Signed mean SHAP per feature family, by candidate class (fold 0, seed 42)."""
    from matplotlib.colors import TwoSlopeNorm, to_hex
    from .pubcolor8 import ink_on
    from .pubdata8 import feat_family
    sh = pd.read_parquet(PUB / "shap_f0_s42.parquet")
    feats = [c[5:] for c in sh.columns if c.startswith("shap_") and c != "shap_base"]
    fam = {f: feat_family(f) for f in feats}
    fams = ["V6 检索通道", "候选融合", "语义", "表达", "疾病描述", "基因可用性"]
    F = pd.DataFrame({fm: sh[[f"shap_{f}" for f in feats if fam[f] == fm]].sum(axis=1) for fm in fams})
    pos, cold, nosem = sh["positive"].astype(bool), sh["cold"].astype(bool), sh["src"] == 0
    strata = [("未标注候选", ~pos), ("未标注·参考路径冷", ~pos & cold), ("留出阳性", pos),
              ("阳性·非冷", pos & ~cold), ("阳性·参考路径冷", pos & cold), ("阳性·语义缺失", pos & nosem)]
    M = np.array([F[m].mean().to_numpy() for _, m in strata])
    lim = np.abs(M).max()
    nrm = TwoSlopeNorm(0, -lim, lim)
    ax = axmm(fig, x + 25, y + 6, w - 25, h - 6)
    import matplotlib.patches as mpa
    cm = div_cmap_rev()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            fc = to_hex(cm(nrm(M[i, j])))
            ax.add_patch(mpa.FancyBboxPatch((j + 0.04, i + 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.08",
                                            fc=fc, ec="none"))
            ax.text(j + 0.5, i + 0.5, f"{M[i, j]:+.2f}".replace("-", "\u2212"), ha="center", va="center", fontsize=FS_S - 1.9,
                    color=ink_on(fc))
    ax.set_xlim(0, M.shape[1])
    ax.set_ylim(M.shape[0], 0)
    ax.set_xticks(np.arange(len(fams)) + 0.5, fams, fontsize=FS_S - 1.5)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(strata)) + 0.5, [f"{s}（{int(m.sum()):,}）" for s, m in strata], fontsize=FS_S - 1.3)
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    rows = [{"stratum": s, "n": int(m.sum()), **{fm: float(v) for fm, v in zip(fams, M[i])}}
            for i, (s, m) in enumerate(strata)]
    return pd.DataFrame(rows)


# ====================================================================== Fig 7 f
def p_e5_ridges(fig, x, y, w, h):
    er = pd.read_parquet(PUB / "e5_ranks.parquet")
    arms = [("main", "主模型（GO 未删）", INK2), ("e5_none", "无补位", MUTED), ("e5_mlp", "两层学生", CAT[6]),
            ("e5_linear", "线性学生", CAT[6]), ("e5_nn", "序列近邻转移", CAT[6]), ("e5_teacher_sS", "教师序列投影", CAT[6]),
            ("e5_mlp_shuffled", "两层学生（打乱）", "#a8a7a1"), ("e5_protst", "ProtST 输入*", "#a8a7a1"),
            ("e6_cold", "E6 撤出监督", CAT[0]), ("e6_cold_gomask", "E6 + 删 GO", CAT[0])]
    grid = np.linspace(0, np.log10(500), 240)
    series, rows = [], []
    for a, lab, c in arms:
        s_ = er[er["arm"] == a]
        v = s_[s_["rank"] > 0]["rank"].to_numpy(float)
        lv = np.log10(v)
        hi = np.log10(500)
        series.append((lab, [(np.concatenate([lv, -lv, 2 * hi - lv]), c)]))   # reflection at both bounds
        rows.append({"arm": a, "label": lab, "positives": len(s_), "listed": len(v),
                     "listed_share": len(v) / max(len(s_), 1), "top10_share": float((s_["rank"].between(1, 10)).mean())})
    ax = axmm(fig, x + 22, y, w - 44, h)
    ridgeline(ax, series, grid, overlap=1.3, label_x=-0.05, fill_alpha=0.3, bw=0.12, lw=0.7)
    n = len(arms)
    for i, ((a, lab, c), r) in enumerate(zip(arms, rows)):
        s_ = er[(er["arm"] == a) & (er["rank"] > 0)]["rank"].to_numpy(float)
        if len(s_):
            md = np.log10(np.median(s_))
            ax.plot([md, md], [n - 1 - i, n - 1 - i + 0.42], color=INK, lw=0.8, zorder=40)
        ax.text(np.log10(500) + 0.06, n - 1 - i + 0.12, f"{r['listed_share']:.0%}", fontsize=FS_S - 1.5,
                color=INK2, va="bottom", ha="left")
        ax.text(np.log10(500) + 0.36, n - 1 - i + 0.12, f"{r['top10_share']:.1%}", fontsize=FS_S - 1.5,
                color=INK, va="bottom", ha="left")
    ax.text(np.log10(500) + 0.06, n + 0.35, "入选", fontsize=FS_S - 1.6, color=MUTED, va="bottom")
    ax.text(np.log10(500) + 0.36, n + 0.35, "前 10", fontsize=FS_S - 1.6, color=MUTED, va="bottom")
    _log_ticks(ax, "x")
    ax.set_xlim(-0.02, np.log10(500))
    ax.set_ylim(-0.25, n + 0.9)
    ax.set_xlabel("屏蔽基因阳性在严格 500 中的名次（入选者，对数轴；竖线为中位数）", fontsize=FS_S - 0.6)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    return pd.DataFrame(rows)


E5_CLASSES = [("前 10", 1, 10), ("11–50", 11, 50), ("51–500", 51, 500), ("未入选", 0, 0)]


def _rank_class(r: np.ndarray) -> np.ndarray:
    return np.select([(r >= 1) & (r <= 10), (r >= 11) & (r <= 50), (r >= 51) & (r <= 500)], [0, 1, 2], 3)


def p_e5_alluvial(fig, x, y, w, h, cols=(("main", "主模型"), ("e5_none", "删 GO·无补位"), ("e5_mlp", "删 GO·两层学生"))):
    """Paired rank-class flow of the same masked-gene test positives across arms (seed 42, all outer folds)."""
    er = pd.read_parquet(PUB / "e5_ranks.parquet")
    piv = er.pivot_table(index=["disease_id", "gene_row"], columns="arm", values="rank")
    C = [_rank_class(piv[a].to_numpy()) for a, _ in cols]
    ccol = [BLUE[650], BLUE[450], BLUE[250], "#d6d5cf"]
    N = len(piv)
    gap, bw_ = 0.035, 0.07
    ax = axmm(fig, x, y, w, h)
    tops = []
    for k, c in enumerate(C):
        cnt = np.bincount(c, minlength=4)
        hts = cnt / N * (1 - 3 * gap)
        top, t = [], 1.0
        for i in range(4):
            top.append(t)
            ax.add_patch(FancyBboxPatch((k - bw_ / 2, t - hts[i]), bw_, hts[i], boxstyle="square,pad=0",
                                        fc=ccol[i], ec="none", zorder=3))
            if k == len(C) - 1:
                ax.text(k + bw_ / 2 + 0.04, t - hts[i] / 2, f"{E5_CLASSES[i][0]}  {cnt[i]:,}（{cnt[i] / N:.1%}）",
                        va="center", ha="left", fontsize=FS_S - 1.7, color=INK)
            else:
                ax.text(k - bw_ / 2 - 0.03, t - hts[i] / 2, f"{cnt[i]:,}", va="center", ha="right",
                        fontsize=FS_S - 2.0, color=INK2, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85))
            t -= hts[i] + gap
        tops.append((top, cnt))
        ax.text(k, 1.04, cols[k][1], ha="center", va="bottom", fontsize=FS_S - 1.3, color=INK)
    from .pubstyle8 import sankey_band
    rows = []
    for k in range(len(C) - 1):
        a, b = C[k], C[k + 1]
        M = np.zeros((4, 4), int)
        np.add.at(M, (a, b), 1)
        (topA, _), (topB, _) = tops[k], tops[k + 1]
        offA, offB = list(topA), list(topB)
        for i in range(4):
            for j in range(4):
                if M[i, j] == 0:
                    continue
                hh = M[i, j] / N * (1 - 3 * gap)
                sankey_band(ax, k + bw_ / 2, offA[i], offA[i] - hh, k + 1 - bw_ / 2, offB[j], offB[j] - hh,
                            ccol[i], alpha=0.42 if i != 3 else 0.3, zorder=1)
                offA[i] -= hh
                offB[j] -= hh
                rows.append({"from_arm": cols[k][0], "to_arm": cols[k + 1][0], "from_class": E5_CLASSES[i][0],
                             "to_class": E5_CLASSES[j][0], "n": int(M[i, j])})
    ax.set_xlim(-0.35, len(C) - 1 + 0.95)
    ax.set_ylim(-0.02, 1.1)
    ax.axis("off")
    return pd.DataFrame(rows)


def p_e5_paired(fig, x, y, w, h):
    """Per-positive rank change of each imputation arm against the no-imputation arm (same positives)."""
    er = pd.read_parquet(PUB / "e5_ranks.parquet")
    piv = er.pivot_table(index=["disease_id", "gene_row"], columns="arm", values="rank").replace(0, 501)
    arms = [("e5_mlp", "两层学生"), ("e5_linear", "线性学生"), ("e5_nn", "序列近邻转移"), ("e5_teacher_sS", "教师序列投影"),
            ("e5_protst", "ProtST 输入*"), ("e5_mlp_shuffled", "两层学生（打乱）")]
    base = piv["e5_none"]
    ax = axmm(fig, x + 22, y, w - 23, h)
    rows = []
    for i, (a, lab) in enumerate(arms):
        v = piv[a]
        up, dn = int((v < base).sum()), int((v > base).sum())
        res, lost = int(((base == 501) & (v <= 500)).sum()), int(((base <= 500) & (v == 501)).sum())
        ax.barh(i, up, height=0.62, color=CAT[0], zorder=2)
        ax.barh(i, -dn, height=0.62, color=CAT[1], zorder=2)
        ax.barh(i, res, height=0.2, color=INK2, zorder=3)
        ax.barh(i, -lost, height=0.2, color=INK2, zorder=3)
        ax.text(up + 25, i, f"{up:,}", va="center", fontsize=FS_S - 1.9, color=INK2)
        ax.text(-dn - 25, i, f"{dn:,}", va="center", ha="right", fontsize=FS_S - 1.9, color=INK2)
        rows.append({"arm": a, "label": lab, "better": up, "worse": dn, "unchanged": int(len(v) - up - dn),
                     "rescued_unlisted_to_listed": res, "lost_listed_to_unlisted": lost, "n": int(len(v))})
    ax.axvline(0, color=INK2, lw=0.6, zorder=4)
    ax.set_yticks(range(len(arms)), [a[1] for a in arms], fontsize=FS_S - 1.2)
    ax.set_ylim(len(arms) - 0.45, -0.6)
    lim = max(max(r["better"], r["worse"]) for r in rows) * 1.35
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(__import__("matplotlib").ticker.FuncFormatter(lambda v_, _: f"{abs(v_):,.0f}"))
    ax.set_xlabel(f"与“无补位”相比名次变化的阳性数（共 {len(base):,}）", fontsize=FS_S - 0.8)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.02, 1.01, "← 更靠后", transform=ax.transAxes, fontsize=FS_S - 1.7, color=INK2, va="bottom")
    ax.text(0.98, 1.01, "更靠前 →", transform=ax.transAxes, fontsize=FS_S - 1.7, color=INK2, va="bottom", ha="right")
    ax.legend(handles=legend_handles([("名次上升", CAT[0], "box"), ("名次下降", CAT[1], "box"),
                                      ("其中跨越入选边界", INK2, "box")]),
              loc="upper center", bbox_to_anchor=(0.45, -0.62), ncol=3, fontsize=FS_S - 1.8, handlelength=0.8,
              borderaxespad=0, columnspacing=0.9)
    return pd.DataFrame(rows)


# ====================================================================== Fig 8 e
def p_e4_heatmap(fig, x, y, w, h):
    """Difference of differences per MeSH category (reduced - full of [deployed - channel-only], Recall@10 x 100),
    columns clustered, with a disease-count annotation track and the all-disease column."""
    import matplotlib as mpl
    from matplotlib.colors import TwoSlopeNorm, to_hex
    from .pubcolor8 import ink_on
    ec = pd.read_parquet(PUB / "e4_category.parquet")
    arms = [("e4_layer1", "训练可见来源排除"), ("e4_layer1_random", "  匹配随机"), ("e4_layer2", "评价条件化删除*"),
            ("e4_layer2_random", "  匹配随机*"), ("e4_blocks", "证据块隔离"), ("e4_blocks_random", "  匹配随机"),
            ("e4_experimental", "仅实验类证据"), ("e4_no_tas_nas", "去 TAS/NAS"), ("e4_no_mutant", "去突变/遗传"),
            ("e4_rescore_no_ic", "去 IC（重打分）"), ("e4_rescore_iea_only", "仅 IEA（重打分）")]
    codes = [c for c in ec["code"].unique() if c != "ALL"]
    M = np.array([[ec[(ec["arm"] == a) & (ec["code"] == c)]["dod"].mean() for c in codes] for a, _ in arms]) * 100
    allc = np.array([ec[(ec["arm"] == a) & (ec["code"] == "ALL")]["dod"].mean() for a, _ in arms]) * 100
    ncat = ec[ec["arm"] == arms[0][0]].set_index("code")["n"]
    lim = float(np.ceil(np.nanpercentile(np.abs(M), 97) * 2) / 2)
    nrm = TwoSlopeNorm(0, -lim, lim)
    cm = div_cmap_rev()
    L, Rw = 32, 12                                   # left label band, right "all" column band (mm)
    ax_c = axmm(fig, x + L, y, w - L - Rw, 6.5)
    ax_n = axmm(fig, x + L, y + 7.2, w - L - Rw, 5.5)
    ax_h = axmm(fig, x + L, y + 13.5, w - L - Rw, h - 13.5)
    ro, co = dendro_heatmap(fig, {"heat": ax_h, "rden": None, "cden": ax_c}, M, [a[1] for a in arms],
                            [f"{SHORT.get(c, c)}" for c in codes], cm, norm=nrm, cluster_rows=False,
                            cluster_cols=True, fmt="{:+.1f}", fs=FS_S - 2.4)
    ax_h.xaxis.tick_bottom()
    ax_h.tick_params(axis="x", labelsize=FS_S - 1.4, rotation=90)
    ax_h.tick_params(axis="y", labelsize=FS_S - 1.1)
    for k in (2, 4, 6, 9):                           # separators between the registered layers / code / rescoring groups
        ax_h.axhline(k, color=SURFACE, lw=1.6, zorder=5)
    nv = np.array([ncat.get(codes[j], 0) for j in co], float)
    ax_n.bar(np.arange(len(co)) + 0.5, nv, width=0.72, color="#c9c8c2", lw=0)
    for j, v in enumerate(nv):
        ax_n.text(j + 0.5, v + nv.max() * 0.04, f"{int(v)}", ha="center", va="bottom", fontsize=FS_S - 2.6, color=INK2)
    ax_n.set_xlim(0, len(co))
    ax_n.set_ylim(0, nv.max() * 1.55)
    ax_n.axis("off")
    ax_n.text(-0.2, nv.max() * 0.35, "疾病数", ha="right", va="center", fontsize=FS_S - 1.9, color=INK2)
    ax_a = axmm(fig, x + w - Rw + 2.5, y + 13.5, 8, h - 13.5)
    import matplotlib.patches as mpa
    for i, v in enumerate(allc[ro]):
        fc = to_hex(cm(nrm(v)))
        ax_a.add_patch(mpa.FancyBboxPatch((0.04, i + 0.025), 0.92, 0.95, boxstyle="round,pad=0,rounding_size=0.06",
                                          fc=fc, ec="none"))
        ax_a.text(0.5, i + 0.5, f"{v:+.1f}".replace("-", "\u2212"), ha="center", va="center", fontsize=FS_S - 2.1,
                  color=ink_on(fc))
    for k in (2, 4, 6, 9):
        ax_a.axhline(k, color=SURFACE, lw=1.6, zorder=5)
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(len(arms), 0)
    ax_a.axis("off")
    ax_a.text(0.5, -0.35, "全部疾病", ha="center", va="bottom", fontsize=FS_S - 1.6, color=INK)
    cax = axmm(fig, x + 2, y + h + 3.5, 26, 1.3)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=nrm, cmap=cm), cax=cax, orientation="horizontal")
    cb.set_ticks([-lim, 0, lim], labels=[f"\u2212{lim:.1f}", "0", f"+{lim:.1f}"])
    cb.set_label("差值变化 ×100（蓝：新方法优势扩大）", fontsize=FS_S - 1.9, labelpad=0.8)
    cb.ax.tick_params(labelsize=FS_S - 2.0, length=1, pad=0.8)
    cb.outline.set_visible(False)
    rows = [{"arm": a, "code": c, "n_diseases": int(ncat.get(c, 0)), "dod_x100": float(M[i, j])}
            for i, (a, _) in enumerate(arms) for j, c in enumerate(codes)]
    rows += [{"arm": a, "code": "ALL", "n_diseases": None, "dod_x100": float(allc[i])} for i, (a, _) in enumerate(arms)]
    return pd.DataFrame(rows)

# ====================================================================== Fig 9 f/g
def p_hist_pairs(fig, x, y, w, h):
    """Every X2 / X1 test pair: rank under the channel-only LightGBM (its own V6 500) vs the deployed model
    (strict 500); unlisted pairs sit in a separate strip.  Small log-space jitter for display only."""
    hr = pd.read_parquet(PUB / "historical_ranks.parquet")
    piv = hr.pivot_table(index=["task", "disease_id", "gene_row"], columns="method", values="rank").reset_index()
    UN = np.log10(500) + 0.42
    tr = lambda r: np.where(r > 0, np.log10(np.maximum(r, 1)), UN)
    rng = np.random.default_rng(3)
    ax = axmm(fig, x, y, w, h)
    lim = UN + 0.22
    ax.fill_between([-0.1, lim], [-0.1, lim], [-0.1, -0.1], color=CAT[0], alpha=0.07, lw=0, zorder=0)
    ax.fill_between([-0.1, lim], [-0.1, lim], [lim, lim], color=CAT[1], alpha=0.07, lw=0, zorder=0)
    ax.plot([-0.1, lim], [-0.1, lim], color=AXIS, lw=0.7, zorder=1)
    for v in (np.log10(500) + 0.21,):
        ax.axvline(v, color=SURFACE, lw=3.0, zorder=2)
        ax.axhline(v, color=SURFACE, lw=3.0, zorder=2)
    rows = []
    for task, mk, fc, lab in (("x2", "o", INK2, "X2：已有关系疾病的新关系"), ("x1", "^", SURFACE, "X1：截止时无监督的疾病")):
        s_ = piv[piv["task"] == task]
        a, b = s_["channel_lgb_v6"].to_numpy(float), s_["deployed"].to_numpy(float)
        xa = tr(a) + rng.uniform(-0.035, 0.035, len(a))
        yb = tr(b) + rng.uniform(-0.035, 0.035, len(b))
        ax.scatter(xa, yb, s=7 if task == "x2" else 10, marker=mk, facecolors=fc, edgecolors=INK, linewidths=0.35,
                   alpha=0.8, zorder=4, label=f"{lab}（{len(s_)} 对）")
        ra, rb = np.where(a > 0, a, 501), np.where(b > 0, b, 501)
        rows.append({"task": task, "pairs": int(len(s_)), "deployed_better": int((rb < ra).sum()),
                     "channel_better": int((rb > ra).sum()), "tie": int((rb == ra).sum()),
                     "both_unlisted": int(((a == 0) & (b == 0)).sum()),
                     "listed_deployed": int((b > 0).sum()), "listed_channel": int((a > 0).sum()),
                     "top10_deployed": int(((b > 0) & (b <= 10)).sum()), "top10_channel": int(((a > 0) & (a <= 10)).sum())})
    ticks = [0, 1, 2, np.log10(500), UN]
    lab_t = ["1", "10", "100", "500", "未入选"]
    ax.set_xticks(ticks, lab_t, fontsize=FS_S - 1.2)
    ax.set_yticks(ticks, lab_t, fontsize=FS_S - 1.2)
    ax.set_xlim(-0.1, lim)
    ax.set_ylim(-0.1, lim)
    ax.set_xlabel("仅通道 LGB 名次（自身 V6 500）", fontsize=FS_S - 0.6)
    ax.set_ylabel(mt("V8.1 部署名次（严格 500）"), fontsize=FS_S - 0.6)
    tot = {k: sum(r[k] for r in rows) for k in ("deployed_better", "channel_better", "tie")}
    ax.text(0.97, 0.05, f"部署更靠前 {tot['deployed_better']}", transform=ax.transAxes, ha="right", fontsize=FS_S - 1.4,
            color=INK2)
    ax.text(0.03, 0.93, f"仅通道更靠前 {tot['channel_better']}", transform=ax.transAxes, ha="left", fontsize=FS_S - 1.4,
            color=INK2, va="top")
    bu = sum(r["both_unlisted"] for r in rows)
    ax.annotate(f"相同 {tot['tie']}\n其中两者\n均未入选 {bu}", xy=(UN, UN), xytext=(1.05, 0.9), textcoords="axes fraction",
                ha="left", va="top", fontsize=FS_S - 1.9, color=INK2, linespacing=1.2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=3))
    style(ax, grid="both")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=FS_S - 1.7, handletextpad=0.3,
              borderaxespad=0, labelspacing=0.3)
    return pd.DataFrame(rows)


def p_support_scatter(fig, x, y, w, h, k: int, ylabel: bool = True):
    """Per-disease Recall@10 (mean over draws) of V8.1 semantic + RWR vs V6 semantic + RWR, with k revealed genes."""
    ep = pd.read_parquet(paths8.RESULTS / "external" / "support" / "support533_test_episodes.parquet")
    per = ep[ep["k"] == k].groupby(["method", "disease_id"])["recall10"].mean().unstack(0)
    a, b = per["joint_v6"].to_numpy(float), per["joint_v8_w0.25"].to_numpy(float)
    ax = axmm(fig, x, y, w, h)
    tf = np.sqrt
    from scipy.stats import gaussian_kde
    pts = np.vstack([tf(a), tf(b)])
    kd = gaussian_kde(pts, bw_method=0.3)
    gx, gy = np.meshgrid(np.linspace(0, 1, 90), np.linspace(0, 1, 90))
    zz = kd(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, zz, levels=np.quantile(zz, [0.7, 0.85, 0.95, 1.0]), colors=["#f1f0ec", "#e4e3de", "#d6d5cf"],
                zorder=0)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.7, zorder=1)
    up, dn = b > a, b < a
    rng = np.random.default_rng(k)
    jx, jy = rng.uniform(-0.008, 0.008, len(a)), rng.uniform(-0.008, 0.008, len(a))
    for m, c, s in ((~up & ~dn, "#a8a7a1", 4), (dn, CAT[1], 6), (up, CAT[0], 6)):
        ax.scatter(tf(a[m]) + jx[m], tf(b[m]) + jy[m], s=s, color=c, edgecolors=SURFACE, linewidths=0.25, alpha=0.85,
                   zorder=3)
    tk_ = [0, 0.05, 0.2, 0.5, 1.0]
    ax.set_xticks(tf(np.array(tk_)), [f"{v:g}" for v in tk_], fontsize=FS_S - 1.3)
    ax.set_yticks(tf(np.array(tk_)), [f"{v:g}" for v in tk_], fontsize=FS_S - 1.3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("V6 语义 + RWR", fontsize=FS_S - 0.7)
    if ylabel:
        ax.set_ylabel("V8.1 语义 + RWR", fontsize=FS_S - 0.7)
    ax.set_title(f"k = {k}：↑{int(up.sum())}  ↓{int(dn.sum())}  ={int((~up & ~dn).sum())}", fontsize=FS_S - 1.0,
                 pad=2, fontweight="normal", loc="center")
    style(ax, grid="both")
    return pd.DataFrame({"disease_id": per.index, "joint_v6": a, "joint_v8": b, "k": k})

# ====================================================================== Fig 10 networks
def p_network(fig, x, y, w, h, disease_id: str, title: str):
    """Gene network of the union of both top-10 lists: deployed-only genes on the left arc, channel-only genes on
    the right arc, shared genes in the middle column.  Node = gene (area ~ GO gene-term pairs; number = rank in
    that method's top 10), black ring = held-out positive, ° = reference-path cold positive.  Edges: STRING
    physical / functional links and GO Jaccard >= 0.2 (curved)."""
    from matplotlib.patches import FancyArrowPatch
    net = json.loads((PUB / "case_networks.json").read_text(encoding="utf-8"))[disease_id]
    nodes = net["nodes"]
    keep = [n for n in nodes if n["top10_deployed"] or n["top10_channel"]]
    ids = {n["gene_row"] for n in keep}
    edges = [e for e in net["edges"] if e["a"] in ids and e["b"] in ids and (e["kind"] != "go" or e["score"] >= 0.2)]
    grp = {n["gene_row"]: ("both" if n["top10_deployed"] and n["top10_channel"] else
                           "dep" if n["top10_deployed"] else "ch") for n in keep}
    pos, lab_side = {}, {}
    for gname, (a0, a1) in (("dep", (np.pi * 0.62, np.pi * 1.38)), ("ch", (np.pi * 0.38, -np.pi * 0.38))):
        mem = sorted([n for n in keep if grp[n["gene_row"]] == gname],
                     key=lambda n: n["top10_deployed"] if gname == "dep" else n["top10_channel"])
        k = len(mem)
        for i, n in enumerate(mem):
            tt = a0 + (a1 - a0) * (i + 0.5) / max(k, 1)
            pos[n["gene_row"]] = (np.cos(tt), np.sin(tt))
            lab_side[n["gene_row"]] = tt
    mem = sorted([n for n in keep if grp[n["gene_row"]] == "both"], key=lambda n: n["top10_deployed"])
    for i, n in enumerate(mem):
        yy = 0.55 - 1.1 * (i + 0.5) / max(len(mem), 1) if len(mem) > 1 else 0.0
        pos[n["gene_row"]] = (0.0, yy)
        lab_side[n["gene_row"]] = None
    ax = axmm(fig, x, y, w, h)
    est = {"physical": dict(color=INK2, lw=0.9, alpha=0.85, ls="-"),
           "functional": dict(color="#c9c8c2", lw=0.6, alpha=0.9, ls="-"),
           "go": dict(color=CAT[6], lw=0.6, alpha=0.7, ls=(0, (2.2, 1.4)))}
    order = {"functional": 0, "go": 1, "physical": 2}
    for e in sorted(edges, key=lambda e: order.get(e["kind"], 0)):
        (x0, y0), (x1, y1) = pos[e["a"]], pos[e["b"]]
        st = est.get(e["kind"], est["functional"])
        rad = 0.25 if (grp[e["a"]] == grp[e["b"]] and grp[e["a"]] != "both") else 0.12
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                                     color=st["color"], lw=st["lw"], alpha=st["alpha"], linestyle=st["ls"], zorder=1,
                                     shrinkA=2.5, shrinkB=2.5))
    for n in keep:
        g = n["gene_row"]
        xx, yy = pos[g]
        c = {"dep": CAT[0], "ch": CAT[1], "both": "#6f6e69"}[grp[g]]
        sz = 30 + 16 * np.log1p(n["n_go"])
        ax.scatter([xx], [yy], s=sz, color=c, edgecolors=INK if n["positive"] else SURFACE,
                   linewidths=1.3 if n["positive"] else 0.7, zorder=4)
        rk = (str(n["top10_deployed"]) if grp[g] == "dep" else str(n["top10_channel"]) if grp[g] == "ch"
              else f"{n['top10_deployed']}/{n['top10_channel']}")
        ax.text(xx, yy, rk, ha="center", va="center", fontsize=FS_S - 2.9 if "/" in rk else FS_S - 2.4,
                color="#ffffff", zorder=5, fontweight="bold")
        name = n["symbol"] + ("°" if (n["positive"] and n["cold"]) else "")
        tt = lab_side[g]
        if tt is None:
            ax.text(xx + 0.13, yy, name, ha="left", va="center", fontsize=FS_S - 1.9, zorder=5,
                    color=INK, fontweight="bold" if n["positive"] else "normal",
                    bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85))
        else:
            r_ = 1.17
            ax.text(r_ * np.cos(tt), r_ * np.sin(tt), name, ha="left" if np.cos(tt) > 0 else "right", va="center",
                    fontsize=FS_S - 1.9, color=INK, zorder=5, fontweight="bold" if n["positive"] else "normal")
    nd = sum(1 for n in keep if n["top10_deployed"] and n["positive"])
    nc = sum(1 for n in keep if n["top10_channel"] and n["positive"])
    ax.set_title(title, fontsize=FS_S - 0.9, pad=8.0, fontweight="bold", loc="center")
    ax.text(0.5, 1.01, f"前 10 命中：部署 {nd} · 仅通道 {nc}", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FS_S - 1.8, color=INK2)
    ax.text(-1.0, -1.28, "仅部署前 10", ha="center", fontsize=FS_S - 1.8, color=INK2)
    ax.text(1.0, -1.28, "仅通道前 10", ha="center", fontsize=FS_S - 1.8, color=INK2)
    if mem:
        ax.text(0.0, -0.78, "共有", ha="center", fontsize=FS_S - 1.8, color=INK2)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.38, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    rows = [{**n, "group": grp[n["gene_row"]], "x": pos[n["gene_row"]][0], "y": pos[n["gene_row"]][1]} for n in keep]
    return pd.DataFrame(rows), pd.DataFrame(edges)
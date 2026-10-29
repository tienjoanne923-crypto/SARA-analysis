"""Figures 2-3 (method schematic; main E1 results), laid out in millimetres."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from . import paths8
from .pubcolor8 import AXIS, BLUE, CAT, ENTITY, INK, INK2, MUTED, SURFACE, seq_cmap
from .pubstyle8 import (FS, FS_L, FS_S, arrow, axmm, ci_h, fam_ci, fam_curve_ci, fam_draws, fig_mm, half_violin,
                        heat_table, label_mm, legend_handles, mt, rounded_box, save, style)

TINT = {"input": ("#f5f5f2", AXIS), "v6": ("#fdeee7", CAT[1]), "sem": ("#ecebf6", CAT[6]),
        "rrf": ("#eef0ee", INK2), "B": ("#fdf3dd", "#c98500"), "C": ("#fbe9f0", "#d55181"),
        "A": ("#e7f0fb", CAT[0]), "deploy": ("#ffffff", INK), "student": ("#f7f7f5", MUTED)}


def _box(ax, x, y, w, h, kind, lines, fs=FS_S - 0.4, title=None, dashed=False, lw=0.7, align="center",
         title_gap=3.3, pad_top=1.5):
    fc, ec = TINT[kind]
    p = rounded_box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=1.3)
    if dashed:
        p.set_linestyle((0, (2.2, 1.6)))
    xt = x + w / 2 if align == "center" else x + 1.5
    ha = "center" if align == "center" else "left"
    yy = y + h - pad_top
    if title:
        ax.text(xt, yy, mt(title), ha=ha, va="top", fontsize=fs + 0.6, fontweight="bold", color=INK)
        yy -= title_gap
    if lines:
        body = "\n".join(lines)
        ax.text(xt, yy, mt(body) if "$" in body else body, ha=ha, va="top", fontsize=fs, color=INK2,
                linespacing=1.32)
    return p


def _poly_arrow(ax, pts, color=MUTED, lw=0.5, dashed=False):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw,
            linestyle=(0, (2.0, 1.4)) if dashed else "-", solid_capstyle="butt")
    arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1], color=color, lw=lw, head=2.4)


# ====================================================================== Figure 2
def fig2() -> dict:
    H2 = 238
    fig = fig_mm(183, H2)
    src = {}
    # ------------------------------------------------------------ a: pipeline (mm canvas, y upwards)
    ax = axmm(fig, 0, 0, 183, 94)
    ax.set_xlim(0, 183)
    ax.set_ylim(0, 94)
    ax.axis("off")
    label_mm(fig, 3, 6.5, "a", "V8.1 定稿流程：语义检索 → 缺失感知严格 500 → 二级排序与部署规则")
    for x, t in ((3, "输入与信息边界"), (45, "检索与语义通道"), (98, "候选生成"), (134, "二级排序"), (163, "部署")):
        ax.text(x, 84.2, t, fontsize=FS_S - 0.2, color=MUTED, va="bottom", fontweight="bold")
    inputs = [("疾病文本", ["名称 / 定义 / 同义词，一级符号掩蔽", "句界优先分段（≤126 词片），SapBERT"], 12.4),
              ("疾病表型", ["HPO 注释（否定项去除）"], 9.4),
              ("基因 GO", ["17 列 GAF；去 NOT、ND", "证据代码变体与三层删除"], 12.4),
              ("蛋白序列", ["ESM-2 35M 表示（冻结）"], 9.4),
              ("组织表达", ["HPA 51 组织（二级特征，可关闭）"], 9.4),
              ("参考图与基因 HPO", ["按角色排除：$F_j$、$F_{j+1}$、", "开发、外部与内块 $I_{jb}$"], 12.4)]
    y = 82.5
    ymid = {}
    for t, ls, hh in inputs:
        y -= hh
        _box(ax, 3, y, 36, hh - 1.0, "input", ls, title=t, align="left", fs=FS_S - 0.9)
        ymid[t] = y + (hh - 1.0) / 2
    bus_ys = [v for k, v in ymid.items() if k != "组织表达"]
    ax.plot([41.3, 41.3], [min(bus_ys), max(bus_ys)], color=MUTED, lw=0.8, solid_capstyle="round")
    for k, v in ymid.items():
        if k == "组织表达":
            continue
        ax.plot([39.0, 41.3], [v, v], color=MUTED, lw=0.6)
        ax.plot([41.3], [v], "o", ms=1.6, color=MUTED)
    # V6 retrieval
    _box(ax, 45, 56, 47, 26, "v6", [], title="V6 检索通道（7 个来源）", fs=FS_S - 0.6)
    chips = ["表型", "层级", "本体", "文本", "基因 HPO", "表型 RWR", "兼容"]
    for i, c in enumerate(chips):
        cx = 47.0 + (i % 4) * 11.1
        cy = 70.3 - (i // 4) * 5.9
        rounded_box(ax, cx, cy, 9.9, 4.5, fc=SURFACE, ec=CAT[1], lw=0.5, r=1.0)
        ax.text(cx + 4.95, cy + 2.25, c, ha="center", va="center", fontsize=FS_S - 1.0, color=INK)
    ax.text(68.5, 58.0, mt("融合 → 完整背景顺序 $r_B$（19,861 个基因）"), ha="center", va="bottom", fontsize=FS_S - 0.7,
            color=INK)
    # semantic teacher
    _box(ax, 45, 22, 47, 31, "sem", ["输入：疾病文本、HPO；基因 GO"], title="语义教师（V7.1 核心，秩 128）",
         fs=FS_S - 0.8)
    eqs = [r"$z_d=\mathrm{norm}\{P_T(x^{\mathrm{text}}_d,\;x^{\mathrm{HPO}}_d)\}$",
           r"$z^F_g=\mathrm{norm}\{P_T\,x^{\mathrm{GO}}_g\}$", r"$s_F(d,g)=z_d^{\top}z^F_g$"]
    for i, e in enumerate(eqs):
        ax.text(68.5, 40.4 - i * 4.6, e, ha="center", va="center", fontsize=FS_S + 0.2, color=INK)
    ax.text(68.5, 25.6, mt("159,749 个参数；检查点：同上下文 $F_{j+1}$ 上 $s_F$ 的\n全背景 Recall@10；未标注对照：全背景（5/5）"),
            ha="center", va="center", fontsize=FS_S - 1.1, color=INK2, linespacing=1.3)
    # student (negative)
    _box(ax, 45, 7.5, 47, 11.5, "student", ["输入：蛋白序列；480→256（GELU）→128",
                                            "仅在 GO 缺失时补位；开发验收 0/5 → 关闭"], title="序列学生（阴性结果）",
         dashed=True, fs=FS_S - 1.1, title_gap=3.0)
    for yy in (69.0, 37.5, 17.0):
        arrow(ax, 41.3, yy, 44.8, yy, color=MUTED, lw=0.6, head=2.2)
    ax.plot([41.3, 41.3], [17.0, min(bus_ys)], color=MUTED, lw=0.8)
    # RRF and strict 500
    _box(ax, 98, 44, 30, 38, "rrf", [], title="缺失感知 RRF", fs=FS_S - 0.6)
    for i, e in enumerate((r"$b=\dfrac{1}{60+r_B}$", r"$e=\dfrac{m}{60+r_S}$",
                           r"$R=\dfrac{w\,b+(1-w)\,m\,e}{w+(1-w)\,m}$")):
        ax.text(113, 71.2 - i * 7.2, e, ha="center", va="center", fontsize=FS_S + 0.5, color=INK)
    ax.text(113, 49.6, mt("语义缺失（$m=0$）时 $R=b$，\n不因少一个来源被额外降分；\n$w_j$ 在开发集上选定"), ha="center",
            va="center", fontsize=FS_S - 1.1, color=INK2, linespacing=1.3)
    _box(ax, 98, 18, 30, 22, "rrf", ["恰好 500 个不同基因", "GeneID 破并列，不补阳性", "开发阳性覆盖",
                                      "43.7% → 57.2%"], title="严格 500 候选", fs=FS_S - 0.9)
    # second stage
    _box(ax, 134, 62, 25, 20, "B", ["30 通道 lambdarank", "袋外 $B^{-b}$", "固定平移 / 尺度"], title="基础排序器 B",
         fs=FS_S - 1.0)
    _box(ax, 134, 40, 25, 19.5, "C", [r"$s=B+a\,r_\theta(x)$", r"$\beta\in\{0,1,2\}$，$\lambda\in\{0.03,0.1\}$",
                                      "15 叶，≤200 轮"], title="门控纠偏器", fs=FS_S - 1.0)
    _box(ax, 134, 18, 25, 19.5, "A", ["同一严格 500", "直接 LightGBM", "NaN 明示缺失；6 配置"], title="候选 A",
         fs=FS_S - 1.0)
    # deployment
    _box(ax, 163, 18, 18.5, 64, "deploy", ["每个外层上下文", "独立判定：", "", "纠偏器须在开发", "R@10 胜过 A，", "或差距 ≤0.005",
                                          "且 $R^c$ 更高", "", "结果：", "5/5 部署 A"], title="A/B 规则", fs=FS_S - 1.2)
    rounded_box(ax, 164.3, 20.0, 15.9, 7.0, fc=CAT[0], ec="none", r=1.0)
    ax.text(172.25, 23.5, "前 10 输出", ha="center", va="center", fontsize=FS_S - 0.5, color="#ffffff",
            fontweight="bold")
    # connections
    arrow(ax, 92.2, 69.0, 97.8, 69.0, color=CAT[1], lw=0.9)
    ax.text(94.9, 69.9, mt("$r_B$"), fontsize=FS_S - 0.3, ha="center", va="bottom", color=INK)
    ax.plot([92.2, 95.0, 95.0], [37.5, 37.5, 57.0], color=CAT[6], lw=0.9, solid_joinstyle="miter")
    arrow(ax, 95.0, 57.0, 97.8, 57.0, color=CAT[6], lw=0.9)
    ax.text(93.9, 46.5, mt("$r_S,\\,m$"), fontsize=FS_S - 0.3, color=INK, rotation=90, va="center", ha="center")
    _poly_arrow(ax, [(92.2, 17.0), (96.4, 17.0), (96.4, 50.5), (97.8, 50.5)], color=MUTED, lw=0.5, dashed=True)
    arrow(ax, 113, 43.8, 113, 40.2, color=INK2, lw=0.9)
    for yy in (72.0, 49.8, 27.8):
        _poly_arrow(ax, [(128.2, 29.0), (131.0, 29.0), (131.0, yy), (133.8, yy)], color=INK2, lw=0.7)
    ax.plot([146.5, 146.5], [61.8, 59.7], color="#c98500", lw=0.7, linestyle=(0, (1.2, 1.0)))
    ax.text(147.3, 60.75, "以 B 为基准", fontsize=FS_S - 1.4, color=INK2, va="center")
    for yy in (49.8, 27.8):
        arrow(ax, 159.2, yy, 162.8, yy, color=INK2, lw=0.8)
    _poly_arrow(ax, [(39.0, ymid["组织表达"]), (42.8, ymid["组织表达"]), (42.8, 3.0), (146.5, 3.0), (146.5, 17.8)],
                color=MUTED, lw=0.5, dashed=True)
    ax.text(120, 3.6, mt("表达分数 $s_E$：二级特征（可关闭）"), fontsize=FS_S - 1.2, color=INK2, ha="center", va="bottom")

    # ------------------------------------------------------------ b: folds and contexts (mm canvas)
    axb = axmm(fig, 0, 98, 80, 66)
    axb.set_xlim(0, 80)
    axb.set_ylim(0, 66)
    axb.axis("off")
    label_mm(fig, 3, 104.0, "b", "外层五折、内块与独立集合")
    cw, ch_, g = 8.6, 5.6, 0.6
    x0, ytop = 16.5, 52.0
    for k in range(5):
        axb.text(x0 + k * (cw + g) + cw / 2, ytop + 2.2, mt(f"$F_{k}$"), ha="center", va="bottom", fontsize=FS_S,
                 color=INK)
    for j in range(5):
        yy = ytop - (j + 1) * (ch_ + g)
        axb.text(x0 - 1.2, yy + ch_ / 2, f"外层 {j}", ha="right", va="center", fontsize=FS_S - 0.2, color=INK)
        tr = [k for k in range(5) if k not in (j, (j + 1) % 5)]
        for k in range(5):
            if k == j:
                fc, lab, tc = INK2, "测试", "#ffffff"
            elif k == (j + 1) % 5:
                fc, lab, tc = "#b9b8b2", "验证", INK
            else:
                fc, lab, tc = "#eceae4", mt(f"$I_{{{j}{tr.index(k) + 1}}}$"), INK2
            axb.add_patch(FancyBboxPatch((x0 + k * (cw + g), yy), cw, ch_, boxstyle="round,pad=0,rounding_size=0.8",
                                         fc=fc, ec="none"))
            axb.text(x0 + k * (cw + g) + cw / 2, yy + ch_ / 2, lab, ha="center", va="center", fontsize=FS_S - 1.0,
                     color=tc)
    for i, (lab, sub_, fc) in enumerate((("开发家族", "864 个疾病", BLUE[150]), ("外部再分析", "618 个疾病", "#e2f1ea"),
                                          ("历史任务", "X2 136 / X1 39 个疾病", "#eeeaf6"))):
        yy = ytop - 7.6 - i * 10.3
        axb.add_patch(FancyBboxPatch((66.0, yy), 13.0, 8.4, boxstyle="round,pad=0,rounding_size=0.8", fc=fc,
                                     ec="none"))
        axb.text(72.5, yy + 5.6, lab, ha="center", va="center", fontsize=FS_S - 0.6, color=INK, fontweight="bold")
        axb.text(72.5, yy + 2.4, sub_, ha="center", va="center", fontsize=FS_S - 1.4, color=INK2)
    notes = [mt("内上下文 $b$：以 $T_j\\setminus I_{jb}$ 训练，为 $I_{jb}$ 产生袋外特征"),
             mt("外上下文：以 $T_j$ 训练，为 $F_j$、$F_{j+1}$ 与开发集打分"),
             mt("每折 3 + 1 个上下文，共 20 个；检查点取同上下文的 $F_{j+1}$")]
    for i, t in enumerate(notes):
        axb.text(4.5, 13.0 - i * 4.2, t, ha="left", va="center", fontsize=FS_S - 1.0, color=INK2)

    # ------------------------------------------------------------ c: aware RRF decision surface
    thr = pd.read_parquet(paths8.RESULTS / "pub" / "budget_thresholds.parquet")
    t5 = thr[thr["w"] == 0.5]
    th_a, th_p = t5["R500_aware"].median(), t5["R500_plain"].median()
    q1, q3 = t5["R500_aware"].quantile([0.25, 0.75])
    label_mm(fig, 84, 104.0, "c", "缺失感知 RRF 的入选边界（w = 0.5）")
    axc = axmm(fig, 97, 109, 36, 34)
    r = np.logspace(0, np.log10(19861), 240)
    RB, RS = np.meshgrid(r, r)
    w = 0.5
    Rm = w / (60 + RB) + (1 - w) / (60 + RS)
    from matplotlib.colors import LogNorm
    nrm = LogNorm(vmin=Rm.min(), vmax=Rm.max())
    im = axc.pcolormesh(r, r, Rm, cmap=seq_cmap(), norm=nrm, shading="auto", rasterized=True)
    axc.contour(r, r, Rm, levels=[q1, q3], colors=["#ffffff"], linewidths=0.4, alpha=0.8)
    axc.contour(r, r, Rm, levels=[th_a], colors=[INK], linewidths=1.0)
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.tick_params(labelbottom=False)
    axc.set_ylabel(mt("语义名次 $r_S$（$m=1$）"))
    axc.text(1.7, 60, "进入严格 500", fontsize=FS_S - 0.6, color="#ffffff", fontweight="bold")
    axc.text(900, 5000, "未入选", fontsize=FS_S - 0.6, color=INK)
    axc.text(900, 2500, "（中位边界）", fontsize=FS_S - 1.3, color=INK2)
    axs = axmm(fig, 97, 145, 36, 3.6, sharex=axc)
    b = 1 / (60 + r)
    axs.pcolormesh(r, [0, 1], np.vstack([b, b]), cmap=seq_cmap(), norm=nrm, shading="auto", rasterized=True)
    cut_a, cut_p = 1 / th_a - 60, w / th_p - 60
    for cut in (cut_a, cut_p):
        axs.axvline(cut, color=INK, lw=0.9)
    axs.set_yticks([0.5], [mt("$m=0$")])
    axs.tick_params(axis="y", length=0)
    axs.set_xscale("log")
    axs.set_xlabel(mt("V6 检索名次 $r_B$"), labelpad=1.5)
    for s_ in ("left", "right", "top"):
        axs.spines[s_].set_visible(False)
    axs.text(cut_p / 1.12, 0.5, mt(f"普通 RRF $\\leq${cut_p:,.0f}"), fontsize=FS_S - 1.4, ha="right", va="center",
             color="#ffffff")
    axs.text(cut_a * 1.12, 0.5, mt(f"缺失感知 $\\leq${cut_a:,.0f}"), fontsize=FS_S - 1.4, ha="left", va="center",
             color=INK)
    axs.set_ylim(0, 1)
    cax = axmm(fig, 134.5, 109, 1.8, 34)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title("R", fontsize=FS_S - 0.5, pad=2)
    cb.ax.tick_params(labelsize=FS_S - 1.2, length=1.3)
    cb.outline.set_visible(False)
    src["c"] = pd.DataFrame({"w": [w], "median_R500_aware": [th_a], "q1": [q1], "q3": [q3],
                             "median_R500_plain": [th_p], "rB_cut_semantic_missing_aware": [cut_a],
                             "rB_cut_semantic_missing_plain": [cut_p], "n_diseases": [len(t5)]})

    # ------------------------------------------------------------ d: three kinds of cold
    label_mm(fig, 143, 104.0, "d", "三类“冷”分开定义")
    axd = axmm(fig, 158.5, 116, 23, 28)
    cols = ["参考\n边", "基因\nHPO", "GO\n注释", "序列\n表达", "训练\n监督"]
    rows = [("参考路径冷", ["○", "○", "●", "●", "○"]), ("GO 缺失（E5）", ["●", "●", "○", "●", "●"]),
            ("监督冷（E6）", ["○", "○", "●", "●", "○"]), ("E6 + 删 GO", ["○", "○", "○", "●", "○"])]
    axd.set_xlim(-0.5, len(cols) - 0.5)
    axd.set_ylim(len(rows) - 0.5, -0.5)
    for i, (rl, vals) in enumerate(rows):
        axd.axhspan(i - 0.5, i + 0.5, color="#f4f4f1" if i % 2 == 0 else SURFACE, lw=0, zorder=0)
        for jx, v in enumerate(vals):
            on = v == "●"
            axd.plot(jx, i, "o", ms=4.8, color=INK2 if on else SURFACE, mec=INK2, mew=0.8, zorder=3)
    axd.set_xticks(range(len(cols)), cols, fontsize=FS_S - 1.1, linespacing=1.1)
    axd.xaxis.tick_top()
    axd.set_yticks(range(len(rows)), [r_[0] for r_ in rows], fontsize=FS_S - 0.7)
    axd.tick_params(length=0, pad=1.5)
    for s_ in axd.spines.values():
        s_.set_visible(False)
    axd.legend(handles=[Line2D([0], [0], marker="o", color=INK2, lw=0, ms=4.2, label="保留"),
                        Line2D([0], [0], marker="o", color=SURFACE, mec=INK2, mew=0.8, lw=0, ms=4.2,
                               label="撤出/缺失")],
               loc="upper center", bbox_to_anchor=(0.2, -0.04), ncol=2, fontsize=FS_S - 0.8, columnspacing=0.6,
               handletextpad=0.2)
    src["d"] = pd.DataFrame([[r_[0]] + r_[1] for r_ in rows], columns=["condition"] + [c.replace("\n", "") for c in cols])

    # ------------------------------------------------------------ e: source availability of the test diseases (UpSet)
    from .pubfigs8x import p_upset
    label_mm(fig, 3, 172.0, "e", "测试疾病的 V6 来源可用性组合（UpSet）与各组合的端到端差值")
    src["e"] = p_upset(fig, 3, 178, 178, 56)
    return save(fig, "Fig2_method_overview", src)


# ====================================================================== Figure 3
def _eval_main() -> pd.DataFrame:
    return pd.read_parquet(paths8.RESULTS / "eval" / "eval_test_main.parquet")


def _seed_mean(ev: pd.DataFrame, method: str, col: str) -> pd.Series:
    sub = ev[ev["method"] == method]
    return sub.groupby("disease_id")[col].mean()


def fig3() -> dict:
    from .pubfigs8a import MLAB, fam_of, nums, recall_curve
    R = nums()
    fam = fam_of()
    ev = _eval_main()
    H3 = 252
    fig = fig_mm(183, H3)
    src = {}

    # ---------------------------------------------------- a: registered hierarchy with bootstrap densities
    label_mm(fig, 2, 6, "a", "事先登记的判定层级（10,000 次家族 bootstrap）")
    ax = axmm(fig, 6, 10, 70, 40)
    specs = [("①", "channel_lgb_v6", "recall10", "h1_vs_channel_lgb_v6", "① 对仅检索通道 LGB（其自身 V6 500 名单）"),
             ("②", "v71_direct_strict", "recall10", "h2_vs_v71_direct_strict", "② 对 V7.1 直接融合（同一严格 500 重训）"),
             ("③", "v71_direct_strict", "cold_recall10", "h3_cold_vs_v71_direct_strict", "③ 冷恢复 $R^c$@10，对 V7.1（描述性）")]
    rows = []
    for k, (sym, other, col, key, lab) in enumerate(specs):
        a = _seed_mean(ev, "deployed", col)
        b_ = _seed_mean(ev, other, col)
        m = pd.concat([a, b_], axis=1, keys=["a", "b"]).dropna()
        d = (m["a"] - m["b"]).to_numpy()
        f = np.array([fam.get(x, x) for x in m.index])
        draws = fam_draws(d, f, 10000)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        reg = R["main"]["hierarchy"][key]
        assert abs(lo - reg["ci_low"]) < 1e-9 and abs(hi - reg["ci_high"]) < 1e-9, (key, lo, reg["ci_low"])
        colr = CAT[0] if k < 2 else MUTED
        half_violin(ax, draws, k + 0.08, colr, height=0.36, side=1, alpha=0.16)
        ci_h(ax, k + 0.2, d.mean(), lo, hi, colr, size=4.2, lw=1.2)
        ax.text(-0.0155, k - 0.42, mt(lab), fontsize=FS_S - 0.3, color=INK, ha="left", va="center")
        rows.append({"test": sym, "contrast": lab, "n_diseases": len(d), "delta": d.mean(), "ci_low": lo,
                     "ci_high": hi, "families": len(np.unique(f))})
    ax.axvspan(-0.0165, -0.005, color="#f5eeeb", lw=0, zorder=0)
    ax.axvline(-0.005, color="#c9b5ae", lw=0.6, zorder=0)
    ax.axvline(0, color=AXIS, lw=0.7, zorder=0)
    ax.text(-0.0108, 2.72, "非劣界以外", ha="center", va="center", fontsize=FS_S - 1.2, color=INK2)
    ax.set_xlim(-0.0165, 0.072)
    ax.set_ylim(2.95, -0.75)
    ax.set_yticks([])
    ax.set_xlabel("部署模型减对照的差值")
    verdict = ["下界 > 0：成立", "下界 < −0.005：非劣未确立", "②未成立：只作描述"]
    for k, r in enumerate(rows):
        ax.text(0.0745, k + 0.2, f"{r['delta']:+.4f}\n[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]\n{verdict[k]}",
                ha="left", va="center", fontsize=FS_S - 1.0, color=INK, linespacing=1.25, clip_on=False)
    style(ax, grid="x", left=False)
    src["a"] = pd.DataFrame(rows)

    # ---------------------------------------------------- b: cold recovery
    label_mm(fig, 107, 6, "b", "参考路径冷目标几乎只由语义恢复")
    ax = axmm(fig, 141, 10, 40, 40)
    meths = ["deployed", "v71_direct_strict", "standalone_sF", "B", "channel_lgb_v6"]
    rows_b = []
    for i, mth in enumerate(meths):
        c = _seed_mean(ev, mth, "cold_recall10").dropna()
        ca = _seed_mean(ev, mth, "cold_to_all10")
        e1 = fam_ci(c.to_numpy(), np.array([fam.get(x, x) for x in c.index]), 2000)
        e2 = fam_ci(ca.to_numpy(), np.array([fam.get(x, x) for x in ca.index]), 2000)
        colr = ENTITY.get(mth, MUTED)
        ax.plot([e1[1], e1[2]], [i - 0.15, i - 0.15], color=colr, lw=1.0)
        ax.plot([e1[0]], [i - 0.15], "o", ms=4.0, color=colr, mec=SURFACE, mew=0.6, zorder=4)
        ax.plot([e2[1], e2[2]], [i + 0.17, i + 0.17], color=colr, lw=1.0, alpha=0.75)
        ax.plot([e2[0]], [i + 0.17], "o", ms=3.6, color=SURFACE, mec=colr, mew=1.0, zorder=4)
        rows_b.append({"method": mth, "Rc10": e1[0], "Rc10_lo": e1[1], "Rc10_hi": e1[2], "n_cold_diseases": len(c),
                       "Rc2all10": e2[0], "Rc2all10_lo": e2[1], "Rc2all10_hi": e2[2]})
    ax.set_yticks(range(len(meths)), [mt(MLAB[m].replace("（全背景）", "")) for m in meths], fontsize=FS_S - 0.3)
    ax.set_ylim(len(meths) - 0.4, -0.6)
    ax.set_xlim(-0.002, 0.086)
    ax.set_xlabel("恢复率")
    ax.legend(handles=[Line2D([0], [0], marker="o", color=INK2, lw=1.0, ms=4, mec=SURFACE,
                              label=mt("$R^c$@10（冷查询内）")),
                       Line2D([0], [0], marker="o", color=INK2, lw=1.0, ms=3.6, mfc=SURFACE,
                              label=mt("$R^{c\\to all}$@10（占全部）"))],
              loc="lower right", fontsize=FS_S - 0.9, handlelength=1.2)
    ax.text(0.99, 1.01, f"含冷查询的疾病 n = {rows_b[0]['n_cold_diseases']:,}", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=FS_S - 1.1, color=INK2)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    src["b"] = pd.DataFrame(rows_b)

    # ---------------------------------------------------- c: Recall@K curves
    label_mm(fig, 2, 64, "c", "Recall@K 曲线（三种子按疾病平均，95% 置信带）")
    ax = axmm(fig, 13, 68, 62, 44)
    rows_c = []
    order = ["deployed", "v71_direct_strict", "channel_lgb_v6", "standalone_sF"]
    ends = {}
    K = np.arange(1, 101)
    for mth in order:
        M, ids = recall_curve(mth, 100)
        f = np.array([fam.get(x, x) for x in ids])
        mean, lo, hi = fam_curve_ci(M, f, 1000)
        colr = ENTITY[mth]
        ax.fill_between(K, lo, hi, color=colr, alpha=0.13, lw=0)
        ax.plot(K, mean, color=colr, lw=1.25)
        ends[mth] = mean[-1]
        rows_c.append(pd.DataFrame({"method": mth, "K": K, "recall": mean, "lo": lo, "hi": hi}))
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.axvline(10, color=AXIS, lw=0.6, zorder=0)
    ax.text(10.8, 0.012, "K = 10", fontsize=FS_S - 1.0, color=INK2)
    lab_y = {}
    items = sorted(ends.items(), key=lambda kv: -kv[1])
    placed = []
    for mth, v in items:
        yv = v
        while any(abs(yv - p_) < 0.034 for p_ in placed):
            yv -= 0.006
        placed.append(yv)
        lab_y[mth] = yv
        ax.annotate(mt(MLAB[mth].split("（")[0]), xy=(100, v), xytext=(118, yv), fontsize=FS_S - 0.8, color=INK,
                    va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=ENTITY[mth], lw=0.7, shrinkA=0, shrinkB=0))
    ax.set_xlabel("截断位置 K（对数轴）")
    ax.set_ylabel("Recall@K（疾病宏平均）")
    ax.set_ylim(0, 0.46)
    style(ax, grid="both")
    src["c"] = pd.concat(rows_c)

    # ---------------------------------------------------- d: every fold x seed
    label_mm(fig, 100, 64, "d", "15 个外层 × 种子组合中的差值")
    axl, axr = axmm(fig, 118, 72, 30, 40), axmm(fig, 151, 72, 30, 40)
    rows_d, ypos, labels = [], [], []
    marks = {42: "o", 52: "s", 62: "^"}
    yy = 0.0
    for j in range(5):
        for seed in (42, 52, 62):
            e = ev[(ev["fold"] == j) & (ev["seed"] == seed)]
            a = e[e["method"] == "deployed"].set_index("disease_id")["recall10"]
            for axx, other in ((axl, "channel_lgb_v6"), (axr, "v71_direct_strict")):
                b_ = e[e["method"] == other].set_index("disease_id")["recall10"]
                m = pd.concat([a, b_], axis=1, keys=["a", "b"]).dropna()
                d = (m["a"] - m["b"]).to_numpy()
                f = np.array([fam.get(x, x) for x in m.index])
                est, lo, hi = fam_ci(d, f, 2000)
                axx.plot([lo, hi], [yy, yy], color=CAT[0], lw=0.75, alpha=0.75)
                axx.plot([est], [yy], marker=marks[seed], ms=3.0, color=CAT[0], mec=SURFACE, mew=0.4, lw=0)
                rows_d.append({"fold": j, "seed": seed, "vs": other, "delta": est, "lo": lo, "hi": hi, "n": len(d)})
            ypos.append(yy)
            labels.append(f"{j}·{seed}")
            yy += 1.0
        yy += 0.7
    hier = R["main"]["hierarchy"]
    yd = yy + 0.4
    for axx, key, xl in ((axl, "h1_vs_channel_lgb_v6", "对仅通道 LGB"), (axr, "h2_vs_v71_direct_strict", "对 V7.1 直接融合")):
        h = hier[key]
        axx.fill([h["ci_low"], h["delta"], h["ci_high"], h["delta"]], [yd, yd - 0.55, yd, yd + 0.55], color=INK, lw=0)
        axx.axvline(0, color=AXIS, lw=0.7, zorder=0)
        axx.set_ylim(yd + 1.2, -1.0)
        axx.set_xlabel(xl, fontsize=FS_S - 0.3)
        style(axx, grid="x", left=False)
        axx.tick_params(axis="y", length=0)
    axr.axvline(-0.005, color="#c9b5ae", lw=0.6, zorder=0)
    axl.set_yticks(ypos + [yd], labels + ["合并"], fontsize=FS_S - 1.4)
    axr.set_yticks([])
    fig.text(117.0 / 183, 1 - 71.2 / H3, "外层·种子", fontsize=FS_S - 1.4, color=INK2, ha="right", va="bottom")
    fig.legend(handles=[Line2D([0], [0], marker=mk, color=CAT[0], lw=0, ms=3.4, mec=SURFACE, label=f"种子 {s}")
                        for s, mk in marks.items()], loc="lower center", bbox_to_anchor=(149.5 / 183, 1 - 124.0 / H3),
               ncol=3, fontsize=FS_S - 0.9, handletextpad=0.2, columnspacing=0.8)
    src["d"] = pd.DataFrame(rows_d)

    # ---------------------------------------------------- e: metric heat table
    label_mm(fig, 2, 131, "e", "同资源主比较（4,238 个测试疾病，列内着色）")
    ax = axmm(fig, 44, 139, 66, 37)
    tab = pd.DataFrame(R["main"]["table"]).sort_values("recall10", ascending=False).reset_index(drop=True)
    cols = [("recall10", "R@10"), ("recall20", "R@20"), ("recall50", "R@50"), ("hit10", "Hit@10"),
            ("coverage500", "覆盖@500"), ("cold_recall10", "$R^c$@10"), ("cold_to_all10", "$R^{c\\to all}$@10")]
    M = tab[[c for c, _ in cols]].to_numpy(float)
    names = [MLAB.get(m, lab) for m, lab in zip(tab["method"], tab["label"])]
    heat_table(ax, M, names, [c for _, c in cols], seq_cmap(), norm_by="col", fmt="{:.3f}", text_size=FS_S - 1.2)
    ax.tick_params(axis="y", labelsize=FS_S - 0.7, pad=2)
    ax.tick_params(axis="x", labelsize=FS_S - 0.8)
    keyed = ("deployed", "channel_lgb_v6", "v71_direct_strict", "standalone_sF")   # all-pairs validated set
    for i, m in enumerate(tab["method"]):
        if m in keyed:
            ax.plot(len(cols) + 0.28, i + 0.5, "s", ms=3.0, color=ENTITY[m], clip_on=False, mec="none")
    src["e"] = tab

    # ---------------------------------------------------- f: win / tie / loss
    label_mm(fig, 116, 131, "f", "逐疾病胜负（层级①的两种方法）")
    ax = axmm(fig, 138, 139, 43, 31)
    a = _seed_mean(ev, "deployed", "recall10")
    b_ = _seed_mean(ev, "channel_lgb_v6", "recall10")
    base = ev[ev["method"] == "deployed"].groupby("disease_id")
    dfw = pd.DataFrame({"d": a - b_, "nq": base["n_query"].first(), "nqc": base["n_query_cold"].first()}).dropna()
    strata = [("全部", dfw.index == dfw.index), ("阳性数 1", dfw["nq"] == 1), ("阳性数 2–5", dfw["nq"].between(2, 5)),
              ("阳性数 >5", dfw["nq"] > 5), ("含冷查询", dfw["nqc"] > 0), ("无冷查询", dfw["nqc"] == 0)]
    rows_f = []
    for i, (lab, msk) in enumerate(strata):
        s_ = dfw.loc[msk, "d"]
        win, loss = (s_ > 1e-12).mean(), (s_ < -1e-12).mean()
        tie = 1 - win - loss
        ax.barh(i, win, height=0.56, color=CAT[0], zorder=2)
        ax.barh(i, -loss, height=0.56, color=CAT[1], zorder=2)
        ax.text(win + 0.01, i, f"{win:.0%}", va="center", ha="left", fontsize=FS_S - 1.0, color=INK)
        ax.text(-loss - 0.01, i, f"{loss:.0%}", va="center", ha="right", fontsize=FS_S - 1.0, color=INK)
        ax.text(0.66, i, f"{tie:.0%}", va="center", ha="right", fontsize=FS_S - 1.0, color=INK2)
        rows_f.append({"stratum": lab, "n": int(msk.sum()), "deployed_better": win, "channel_better": loss, "tie": tie})
    ax.text(0.66, -0.95, "持平", va="center", ha="right", fontsize=FS_S - 1.1, color=INK2)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(range(len(strata)), [f"{s[0]}（{int(s[1].sum()):,}）" for s in strata], fontsize=FS_S - 0.8)
    ax.set_ylim(len(strata) - 0.45, -1.3)
    ax.set_xlim(-0.32, 0.67)
    ax.set_xticks([-0.2, 0, 0.2, 0.4], ["20%", "0", "20%", "40%"])
    ax.set_xlabel("疾病比例（逐病比较 Recall@10）", fontsize=FS_S - 0.3)
    fig.legend(handles=legend_handles([("V8.1 部署更好", CAT[0], "box"), ("仅通道 LGB 更好", CAT[1], "box")]),
               loc="upper center", bbox_to_anchor=(152.0 / 183, 1 - 133.2 / H3), ncol=2, fontsize=FS_S - 0.9,
               columnspacing=0.8)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    src["f"] = pd.DataFrame(rows_f)
    from .pubfigs8x import p_chord_hits, p_radar, p_raincloud_best
    label_mm(fig, 2, 190, "g", "每个疾病最靠前阳性的名次分布")
    src["g"] = p_raincloud_best(fig, 2, 195, 80, 45)
    label_mm(fig, 86, 190, "h", "前 10 命中的共享（弦图）")
    src["h"] = p_chord_hits(fig, 86, 194, 50, 52)
    label_mm(fig, 138, 190, "i", "指标轮廓")
    src["i"] = p_radar(fig, 138, 196, 44, 44, R)
    return save(fig, "Fig3_main_results", src)

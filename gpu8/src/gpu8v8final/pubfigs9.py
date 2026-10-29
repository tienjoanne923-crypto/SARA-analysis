"""Restructured figures of the SARA submission: Fig 2 (SARA framework), Fig 6 (ranker explanation),
Fig 7 (functional annotation and association supervision), Fig S2 (development selection) and the new
Fig S8 (examined alternative designs).  All other figures are the V8.1 figure functions saved through
``pubstyle9.save9``.  Every number drawn here is read from the saved results.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from . import paths8
from .pubcolor8 import AXIS, BLUE, CAT, INK, INK2, MUTED, SURFACE, div_cmap_rev, ink_on, seq_cmap
from .pubfigs8a_part2 import _box, _poly_arrow
from .pubstyle8 import (FS, FS_L, FS_S, arrow, axmm, fam_ratio_ci, fig_mm, label_mm, legend_handles, mt, pct,
                        rounded_box, style)
from .pubstyle9 import save9 as save

PUB = paths8.RESULTS / "pub"
EVAL = paths8.RESULTS / "eval"
V2SRC = paths8.MANUSCRIPT / "figures_v2" / "source_data"
MAG, YEL = CAT[4], CAT[3]


def _nums() -> dict:
    return json.loads((paths8.RESULTS / "numbers_v8_results.json").read_text(encoding="utf-8"))


def _fam() -> dict:
    d = pd.read_parquet(PUB / "diseases.parquet")
    return dict(zip(d["disease_id"], d["family_group"]))


def _dev_coverage() -> tuple[float, float]:
    sels = [json.loads((paths8.DEV / f"fold{j}" / "selection.json").read_text(encoding="utf-8")) for j in range(5)]
    v6 = float(np.mean([s["w"]["coverage"]["v6_500"] for s in sels]))
    aw = float(np.mean([s["w"]["coverage"][f"aware_w{s['w']['w']}"] for s in sels]))
    return v6, aw


def _set_counts() -> dict:
    R = _nums()
    dis = pd.read_parquet(PUB / "diseases.parquet")
    hp = pd.read_parquet(paths8.EXT_RES / "historical" / "s42" / "historical_per_disease.parquet")
    h = hp[hp["method"] == "deployed"].groupby("task")["disease_id"].nunique()
    return {"dev": int((dis["role"] == "dev").sum()), "external": int(R["external"]["fields"]["diseases"]),
            "x2": int(h.get("x2", 0)), "x1": int(h.get("x1", 0))}


# ====================================================================== Figure 2
def _panel_a_pipeline(fig):
    ax = axmm(fig, 0, 0, 183, 94)
    ax.set_xlim(0, 183)
    ax.set_ylim(0, 94)
    ax.axis("off")
    label_mm(fig, 3, 6.5, "a", "SARA：多源检索与语义检索 → 可用性感知 RRF 候选 → 明示缺失的排序器")
    for x, t in ((3, "输入与信息边界"), (45, "检索与语义"), (98, "候选生成"), (134, "排序"), (163, "输出")):
        ax.text(x, 84.2, t, fontsize=FS_S - 0.2, color=MUTED, va="bottom", fontweight="bold")
    inputs = [("疾病文本", ["名称 / 定义 / 同义词，基因符号掩蔽", "句界优先分段，SapBERT 编码"], 12.4),
              ("疾病表型", ["HPO 注释（去除否定项）"], 9.4),
              ("基因功能注释", ["GO（GAF 2.2）；去 NOT、ND", "证据代码与文献可追溯"], 12.4),
              ("组织表达", ["HPA 51 种组织（排序特征）"], 9.4),
              ("参考关联图与基因 HPO", ["按角色排除留出疾病：", "测试、验证、开发与外部集合"], 12.4)]
    y = 82.5
    ymid = {}
    for t, ls, hh in inputs:
        y -= hh
        _box(ax, 3, y, 36, hh - 1.0, "input", ls, title=t, align="left", fs=FS_S - 0.9)
        ymid[t] = y + (hh - 1.0) / 2
    bus = [v for k, v in ymid.items() if k != "组织表达"]
    ax.plot([41.3, 41.3], [min(bus), max(bus)], color=MUTED, lw=0.8, solid_capstyle="round")
    for k, v in ymid.items():
        if k == "组织表达":
            continue
        ax.plot([39.0, 41.3], [v, v], color=MUTED, lw=0.6)
        ax.plot([41.3], [v], "o", ms=1.6, color=MUTED)
    # multi-source retrieval
    _box(ax, 45, 56, 47, 26, "v6", [], title="多源检索（7 个来源）", fs=FS_S - 0.6)
    for i, c in enumerate(["表型", "层级", "本体", "文本", "基因 HPO", "表型 RWR", "兼容"]):
        cx, cy = 47.0 + (i % 4) * 11.1, 70.3 - (i // 4) * 5.9
        rounded_box(ax, cx, cy, 9.9, 4.5, fc=SURFACE, ec=CAT[1], lw=0.5, r=1.0)
        ax.text(cx + 4.95, cy + 2.25, c, ha="center", va="center", fontsize=FS_S - 1.0, color=INK)
    ax.text(68.5, 58.0, mt("融合 → 全背景检索顺序 $r_B$（19,861 个基因）"), ha="center", va="bottom", fontsize=FS_S - 0.7,
            color=INK)
    # semantic model
    _box(ax, 45, 12, 47, 40, "sem", ["输入：疾病文本与 HPO；基因 GO"], title="疾病—功能语义模型（秩 128）", fs=FS_S - 0.8)
    eqs = [r"$z_d=\mathrm{norm}\{P_T(x^{\mathrm{text}}_d,\;x^{\mathrm{HPO}}_d)\}$",
           r"$z^F_g=\mathrm{norm}\{P_T\,x^{\mathrm{GO}}_g\}$", r"$s_F(d,g)=z_d^{\top}z^F_g$"]
    for i, e in enumerate(eqs):
        ax.text(68.5, 38.6 - i * 5.0, e, ha="center", va="center", fontsize=FS_S + 0.3, color=INK)
    ax.text(68.5, 18.6, mt("共享投影与功能锚对齐；袋外训练\n检查点：同一上下文 $F_{j+1}$ 上 $s_F$ 的全背景 Recall@10"),
            ha="center", va="center", fontsize=FS_S - 1.1, color=INK2, linespacing=1.35)
    for yy in (69.0, 37.5):
        arrow(ax, 41.3, yy, 44.8, yy, color=MUTED, lw=0.6, head=2.2)
    # availability-aware RRF and candidates
    _box(ax, 98, 44, 30, 38, "rrf", [], title="可用性感知 RRF", fs=FS_S - 0.6)
    for i, e in enumerate((r"$b=\dfrac{1}{60+r_B}$", r"$e=\dfrac{m}{60+r_S}$",
                           r"$R=\dfrac{w\,b+(1-w)\,m\,e}{w+(1-w)\,m}$")):
        ax.text(113, 71.2 - i * 7.2, e, ha="center", va="center", fontsize=FS_S + 0.5, color=INK)
    ax.text(113, 49.6, mt("语义不可用（$m=0$）时 $R=b$，\n不因缺少一种模态被降分；\n$w_j$ 在开发集上选定"), ha="center",
            va="center", fontsize=FS_S - 1.1, color=INK2, linespacing=1.3)
    v6c, awc = _dev_coverage()
    _box(ax, 98, 12, 30, 28, "rrf", ["恰好 500 个不同基因", "GeneID 打破并列", "", "开发集阳性覆盖",
                                      f"{v6c:.1%} → {awc:.1%}"], title="SARA 候选", fs=FS_S - 0.9)
    # ranker
    from .pubdata8 import feat_family
    fi = pd.read_parquet(PUB / "feature_importance.parquet")
    fam = fi.assign(f=fi["feature"].map(feat_family)).groupby(["seed", "fold", "f"])["gain_share"].sum().groupby("f").mean()
    fz = {"V6 检索通道": "检索通道", "候选融合": "融合名次", "语义": "语义", "表达": "表达"}
    fam_lines = [f"{fz[k]} {fam.get(k, 0):.0%}" for k in ("V6 检索通道", "候选融合", "语义", "表达")]
    _box(ax, 134, 12, 25, 70, "A", ["LambdaRank", "（LightGBM）", "", "30 个检索特征", "+ 19 个语义与", "可用性特征", "",
                                    "缺失值以 NaN", "明示输入", "", "外层上下文内", "袋外训练；", "开发集选择配置", "",
                                    "特征增益占比："] + fam_lines,
         title="SARA 排序器", fs=FS_S - 0.9)
    # output
    _box(ax, 163, 12, 18.5, 70, "deploy", ["每个疾病", "对 500 个候选", "重新排序", "", "逐基因给出", "排序分数与", "可解释的", "特征贡献"],
         title="输出", fs=FS_S - 1.1)
    rounded_box(ax, 164.3, 14.0, 15.9, 7.0, fc=CAT[0], ec="none", r=1.0)
    ax.text(172.25, 17.5, "前 10 基因", ha="center", va="center", fontsize=FS_S - 0.5, color="#ffffff", fontweight="bold")
    # connections
    arrow(ax, 92.2, 69.0, 97.8, 69.0, color=CAT[1], lw=0.9)
    ax.text(94.9, 69.9, mt("$r_B$"), fontsize=FS_S - 0.3, ha="center", va="bottom", color=INK)
    ax.plot([92.2, 95.0, 95.0], [33.0, 33.0, 57.0], color=CAT[6], lw=0.9, solid_joinstyle="miter")
    arrow(ax, 95.0, 57.0, 97.8, 57.0, color=CAT[6], lw=0.9)
    ax.text(93.9, 45.0, mt("$r_S,\\,m$"), fontsize=FS_S - 0.3, color=INK, rotation=90, va="center", ha="center")
    arrow(ax, 113, 43.8, 113, 40.2, color=INK2, lw=0.9)
    arrow(ax, 128.2, 26.0, 133.8, 26.0, color=INK2, lw=0.9)
    ax.text(131.0, 27.0, "500", fontsize=FS_S - 1.4, color=INK2, ha="center", va="bottom")
    _poly_arrow(ax, [(80.0, 82.2), (80.0, 83.4), (140.0, 83.4), (140.0, 82.3)], color=CAT[1], lw=0.5, dashed=True)
    _poly_arrow(ax, [(80.0, 11.8), (80.0, 9.2), (140.0, 9.2), (140.0, 11.8)], color=CAT[6], lw=0.5, dashed=True)
    ax.text(112.0, 9.8, mt("语义特征（$s_F$、$r_S$ 与可用性）"), fontsize=FS_S - 1.5, color=INK2, ha="center", va="bottom")
    _poly_arrow(ax, [(39.0, ymid["组织表达"]), (42.8, ymid["组织表达"]), (42.8, 6.0), (146.5, 6.0), (146.5, 11.8)],
                color=MUTED, lw=0.5, dashed=True)
    ax.text(112.0, 5.4, mt("表达分数 $s_E$（排序特征）"), fontsize=FS_S - 1.5, color=INK2, ha="center", va="top")
    ax.text(181.5, 3.2, "虚线：直接进入排序器的特征", fontsize=FS_S - 1.6, color=INK2, ha="right", va="center")
    arrow(ax, 159.2, 47.0, 162.8, 47.0, color=INK2, lw=0.8)


def p_upset9(fig, x, y, w, h):
    """Source availability of the test diseases (UpSet), drawn from the saved panel data."""
    src = SRC_V2 = V2SRC / "Fig2_method_overview_e.csv"
    u = pd.read_csv(SRC_V2, encoding="utf-8-sig", dtype={"combo": str})
    u["combo"] = u["combo"].str.zfill(7)
    names = ["表型", "层级", "本体", "文本", "基因 HPO", "表型 RWR", "兼容"]
    u = u.sort_values("n_diseases", ascending=False).reset_index(drop=True)
    nC, nS = len(u), len(names)
    L = 34
    mw = w - L - 30
    ax_top = axmm(fig, x + L, y, mw, h * 0.34)
    ax_m = axmm(fig, x + L, y + h * 0.36, mw, h * 0.34)
    ax_d = axmm(fig, x + L, y + h * 0.73, mw, h * 0.27)
    ax_s = axmm(fig, x, y + h * 0.36, L - 14, h * 0.34)
    top = int(u["n_diseases"].max())
    for k, r in u.iterrows():
        n = int(r["n_diseases"])
        ax_top.bar(k, n, width=0.66, color=INK2, zorder=2)
        ax_top.text(k, n + top * 0.02, f"{n:,}", ha="center", va="bottom", fontsize=FS_S - 2.0, color=INK2)
        on = [i for i, ch in enumerate(r["combo"]) if ch == "1"]
        for i in range(nS):
            ax_m.plot(k, i, "o", ms=3.4, color=INK2 if i in on else "#e2e1dc", mec="none", zorder=3)
        if len(on) > 1:
            ax_m.plot([k, k], [min(on), max(on)], color=INK2, lw=1.1, zorder=2)
        col = CAT[0] if r["ci_low"] > 0 else (CAT[1] if r["ci_high"] < 0 else MUTED)
        ax_d.plot([k, k], [r["ci_low"], r["ci_high"]], color=col, lw=1.0, solid_capstyle="round", zorder=2)
        ax_d.plot([k], [r["delta"]], "o", ms=3.2, color=col, mec=SURFACE, mew=0.5, zorder=3)
    for i in range(nS):
        if i % 2 == 0:
            ax_m.axhspan(i - 0.5, i + 0.5, color="#f4f4f1", lw=0, zorder=0)
    ax_top.set_xlim(-0.6, nC - 0.4)
    ax_top.set_ylim(0, top * 1.18)
    ax_top.set_xticks([])
    ax_top.set_ylabel("疾病数", fontsize=FS_S - 1.0)
    style(ax_top, grid="y", bottom=False)
    ax_top.spines["bottom"].set_visible(False)
    ax_m.set_xlim(-0.6, nC - 0.4)
    ax_m.set_ylim(nS - 0.5, -0.5)
    ax_m.set_yticks(range(nS), names, fontsize=FS_S - 1.2)
    ax_m.set_xticks([])
    ax_m.tick_params(length=0, pad=1.5)
    for s in ax_m.spines.values():
        s.set_visible(False)
    tot = np.array([sum(int(r["n_diseases"]) for _, r in u.iterrows() if r["combo"][i] == "1") for i in range(nS)])
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
    ax_d.text(nC - 0.4, ax_d.get_ylim()[1], "SARA − 检索通道 LGB（三种子均值；家族 bootstrap 95% CI）",
              ha="right", va="top", fontsize=FS_S - 2.0, color=INK2)
    axr = axmm(fig, x + L + mw + 3, y, 27, h)
    axr.axis("off")
    n = int(u["n_diseases"].sum())
    full = int(u.loc[u["combo"] == "1111111", "n_diseases"].sum())
    lines = [f"测试疾病 {n:,} 个（5 折）", f"七个来源全部可用 {full:,}（{full / n:.0%}）",
             f"出现的可用性组合共 {nC} 种", f"区间下界 > 0 的组合 {int((u['ci_low'] > 0).sum())} 种"]
    for i, s in enumerate(lines):
        axr.text(0, 0.98 - i * 0.075, s, fontsize=FS_S - 1.7, color=INK2, va="top", transform=axr.transAxes)
    axr.text(0, 0.62, "区间颜色", fontsize=FS_S - 1.6, color=INK, va="top", transform=axr.transAxes, fontweight="bold")
    for i, (c, s) in enumerate(((CAT[0], "区间 > 0"), (MUTED, "区间含 0"))):
        yy = 0.55 - i * 0.07
        axr.plot([0.02, 0.1], [yy, yy], color=c, lw=1.2, transform=axr.transAxes)
        axr.plot([0.06], [yy], "o", ms=3.0, color=c, mec=SURFACE, mew=0.5, transform=axr.transAxes)
        axr.text(0.14, yy, s, fontsize=FS_S - 1.7, color=INK2, va="center", transform=axr.transAxes)
    return u


def fig2() -> dict:
    H2 = 238
    fig = fig_mm(183, H2)
    src = {}
    _panel_a_pipeline(fig)
    sc = _set_counts()
    # ---- b: folds and contexts (as in the V8.1 figure, public wording)
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
    for i, (lab, sub_, fc) in enumerate((("开发家族", f"{sc['dev']} 个疾病", BLUE[150]),
                                          ("外部疾病", f"{sc['external']} 个疾病", "#e2f1ea"),
                                          ("时间外推", f"{sc['x2']} + {sc['x1']} 个疾病", "#eeeaf6"))):
        yy = ytop - 7.6 - i * 10.3
        axb.add_patch(FancyBboxPatch((66.0, yy), 13.0, 8.4, boxstyle="round,pad=0,rounding_size=0.8", fc=fc,
                                     ec="none"))
        axb.text(72.5, yy + 5.6, lab, ha="center", va="center", fontsize=FS_S - 0.6, color=INK, fontweight="bold")
        axb.text(72.5, yy + 2.4, sub_, ha="center", va="center", fontsize=FS_S - 1.4, color=INK2)
    notes = [mt("内上下文 $b$：以 $T_j\\setminus I_{jb}$ 训练，为 $I_{jb}$ 产生袋外特征"),
             mt("外上下文：以 $T_j$ 训练，为 $F_j$、$F_{j+1}$ 与开发集打分"),
             mt("每折 3 + 1 个上下文，共 20 个；检查点取同一上下文的 $F_{j+1}$")]
    for i, t in enumerate(notes):
        axb.text(4.5, 13.0 - i * 4.2, t, ha="left", va="center", fontsize=FS_S - 1.0, color=INK2)
    # ---- c: RRF decision surface (identical to V8.1 panel c)
    thr = pd.read_parquet(PUB / "budget_thresholds.parquet")
    t5 = thr[thr["w"] == 0.5]
    th_a, th_p = t5["R500_aware"].median(), t5["R500_plain"].median()
    q1, q3 = t5["R500_aware"].quantile([0.25, 0.75])
    label_mm(fig, 84, 104.0, "c", "可用性感知 RRF 的入选边界（w = 0.5）")
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
    axc.text(1.7, 60, "进入 SARA 候选", fontsize=FS_S - 0.6, color="#ffffff", fontweight="bold")
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
    axs.set_xlabel(mt("检索名次 $r_B$"), labelpad=1.5)
    for s_ in ("left", "right", "top"):
        axs.spines[s_].set_visible(False)
    axs.text(cut_p / 1.12, 0.5, mt(f"普通 RRF $\\leq${cut_p:,.0f}"), fontsize=FS_S - 1.4, ha="right", va="center",
             color="#ffffff")
    axs.text(cut_a * 1.12, 0.5, mt(f"可用性感知 $\\leq${cut_a:,.0f}"), fontsize=FS_S - 1.4, ha="left", va="center",
             color=INK)
    axs.set_ylim(0, 1)
    cax = axmm(fig, 134.5, 109, 1.8, 34)
    import matplotlib.pyplot as plt  # noqa: F401
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title("R", fontsize=FS_S - 0.5, pad=2)
    cb.ax.tick_params(labelsize=FS_S - 1.2, length=1.3)
    cb.outline.set_visible(False)
    src["c"] = pd.DataFrame({"w": [w], "median_R500_aware": [th_a], "q1": [q1], "q3": [q3],
                             "median_R500_plain": [th_p], "rB_cut_semantic_missing_aware": [cut_a],
                             "rB_cut_semantic_missing_plain": [cut_p], "n_diseases": [len(t5)]})
    # ---- d: three kinds of cold
    label_mm(fig, 143, 104.0, "d", "三类“冷”分开定义")
    axd = axmm(fig, 158.5, 116, 23, 28)
    cols = ["参考\n边", "基因\nHPO", "GO\n注释", "序列\n表达", "训练\n监督"]
    rows = [("参考路径冷", ["○", "○", "●", "●", "○"]), ("GO 缺失（受控）", ["●", "●", "○", "●", "●"]),
            ("关联监督冷（受控）", ["○", "○", "●", "●", "○"]), ("撤出监督 + 删 GO", ["○", "○", "○", "●", "○"])]
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
    # ---- e: UpSet of source availability
    label_mm(fig, 3, 172.0, "e", "测试疾病的检索来源可用性组合（UpSet）与各组合的性能差值")
    src["e"] = p_upset9(fig, 3, 178, 178, 56)
    return save(fig, "Fig2_method_overview", src)


# ====================================================================== Figure 6
FAM_COL = {"候选融合": CAT[0], "语义": CAT[6], "V6 检索通道": CAT[1], "表达": "#6f6e69", "疾病描述": "#a8a7a1",
           "基因可用性": "#c9c8c2"}


def fig6() -> dict:
    from .pubdata8 import FEAT_ZH, feat_family
    from .pubfigs8x import p_shap_beeswarm, p_shap_dependence, p_shap_family_strata
    R = _nums()
    H6 = 154
    fig = fig_mm(183, H6)
    src = {}
    # ---- a: feature gain of the SARA ranker (15 models)
    label_mm(fig, 2, 6, "a", "SARA 排序器的特征增益占比（5 个外层 × 3 个种子）")
    fi = pd.read_parquet(PUB / "feature_importance.parquet")
    agg = fi.groupby("feature")["gain_share"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    top = agg.head(14)
    ax = axmm(fig, 36, 13, 38, 48)
    for i, (f, r) in enumerate(top.iterrows()):
        c = FAM_COL[feat_family(f)]
        ax.barh(i, r["mean"], xerr=r["std"], height=0.62, color=c, error_kw=dict(ecolor=INK2, lw=0.5, capsize=1.2),
                zorder=2)
    ax.set_yticks(range(len(top)), [mt(FEAT_ZH.get(f, f)) for f in top.index], fontsize=FS_S - 1.0)
    ax.set_ylim(len(top) - 0.4, -0.6)
    pct(ax, "x")
    ax.set_xlabel("增益占比（均值 ± 标准差）", fontsize=FS_S - 0.5)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    fam_tot = fi.assign(fam=fi["feature"].map(feat_family)).groupby(["seed", "fold", "fam"])["gain_share"].sum()
    fam_tot = fam_tot.groupby("fam").mean().sort_values(ascending=False)
    axf = axmm(fig, 79, 13, 13, 48)
    bottom = 0.0
    for fam_, v in fam_tot.items():
        axf.bar(0, v, bottom=bottom, width=0.7, color=FAM_COL[fam_], edgecolor=SURFACE, linewidth=0.6)
        if v > 0.06:
            axf.text(0, bottom + v / 2, f"{fam_}\n{v:.0%}", ha="center", va="center", fontsize=FS_S - 2.0,
                     color="#ffffff" if fam_ in ("候选融合", "语义", "V6 检索通道", "表达") else INK, linespacing=1.1)
        bottom += v
    axf.set_xlim(-0.5, 0.5)
    axf.set_ylim(0, 1)
    axf.axis("off")
    axf.text(0, 1.02, "按特征族", ha="center", va="bottom", fontsize=FS_S - 1.2, color=INK2)
    src["a"] = agg.reset_index().assign(family=lambda d_: d_["feature"].map(feat_family),
                                        label=lambda d_: d_["feature"].map(FEAT_ZH))
    # ---- b: signed SHAP per feature family and candidate class
    label_mm(fig, 98, 6, "b", "各特征族的平均 SHAP（按候选类别，带符号）")
    src["b"] = p_shap_family_strata(fig, 98, 10, 83, 44)
    # ---- c: beeswarm
    label_mm(fig, 2, 72, "c", "SARA 排序器的 TreeSHAP 蜂群图（外层 0，种子 42）")
    src["c"] = p_shap_beeswarm(fig, 2, 80, 90, 62)
    # ---- d/e: dependence
    label_mm(fig, 100, 72, "d", "SHAP 依赖：融合名次")
    src["d"] = p_shap_dependence(fig, 109, 80, 30, 22, "lr_R", "可用性感知 RRF 名次",
                                 xform=lambda v: 500.0 * np.exp(-v), logx=True, xticks=(1, 10, 100, 500),
                                 ylab="SHAP：融合名次", legend="lower left")
    label_mm(fig, 143, 72, "e", "SHAP 依赖：语义分数 $s_F$")
    src["e"] = p_shap_dependence(fig, 152, 80, 29, 22, "s_sem", "语义分数 $s_F$", ylab="SHAP：$s_F$",
                                 cbar_box=(118, 111.3, 52, 1.3))
    # ---- f: expression ablation (robustness)
    label_mm(fig, 100, 122, "f", "表达特征的消融：对 SARA 的 Recall@10 影响")
    ax = axmm(fig, 128, 128, 52, 14)
    rows = []
    for i, (tag, lab) in enumerate((("expr_off", "关闭表达特征"), ("expr_random", "随机组织表达"))):
        d = R["variants"][tag]["deployed"]
        e = d["delta_vs_main"]
        ax.plot([e["ci_low"], e["ci_high"]], [i, i], color=INK2, lw=1.0, solid_capstyle="round")
        ax.plot([e["delta"]], [i], "o", ms=3.6, color=INK2, mec=SURFACE, mew=0.5, zorder=3)
        ax.text(0.0205, i, f"R@10 = {d['recall10']:.3f}", va="center", ha="right", fontsize=FS_S - 1.7, color=INK2)
        rows.append({"variant": tag, "label": lab, "recall10": d["recall10"], "reference": e["b"], **e})
    ax.axvline(0, color=AXIS, lw=0.7)
    ax.set_yticks([0, 1], ["关闭表达特征", "随机组织表达"], fontsize=FS_S - 1.1)
    ax.set_ylim(1.6, -0.6)
    ax.set_xlim(-0.012, 0.021)
    ax.set_xlabel(f"ΔRecall@10（对完整 SARA，{rows[0]['reference']:.3f}；种子 42；95% CI）", fontsize=FS_S - 1.2)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    src["f"] = pd.DataFrame(rows)
    return save(fig, "Fig6_ranker_explanation", src)


# ====================================================================== Figure 7
ARMS7 = [("main", "完整模型"), ("e5_none", "删 GO"), ("e6_cold", "撤出关联监督"), ("e6_cold_gomask", "撤出监督 + 删 GO")]


def p_ridges9(fig, x, y, w, h, arms):
    from .pubviz8 import ridgeline
    from .pubfigs8x import _log_ticks
    er = pd.read_parquet(PUB / "e5_ranks.parquet")
    hi = np.log10(500)
    grid = np.linspace(0, hi, 240)
    series, rows = [], []
    for a, lab, c in arms:
        s_ = er[er["arm"] == a]
        v = s_[s_["rank"] > 0]["rank"].to_numpy(float)
        lv = np.log10(v)
        series.append((lab, [(np.concatenate([lv, -lv, 2 * hi - lv]), c)]))
        rows.append({"arm": a, "label": lab, "positives": len(s_), "listed": len(v),
                     "listed_share": len(v) / max(len(s_), 1), "top10_share": float((s_["rank"].between(1, 10)).mean()),
                     "median_rank_listed": float(np.median(v)) if len(v) else np.nan})
    ax = axmm(fig, x + 26, y, w - 48, h)
    ridgeline(ax, series, grid, overlap=1.3, label_x=-0.05, fill_alpha=0.3, bw=0.12, lw=0.7)
    n = len(arms)
    for i, r in enumerate(rows):
        if np.isfinite(r["median_rank_listed"]):
            md = np.log10(r["median_rank_listed"])
            ax.plot([md, md], [n - 1 - i, n - 1 - i + 0.42], color=INK, lw=0.8, zorder=40)
        ax.text(hi + 0.06, n - 1 - i + 0.12, f"{r['listed_share']:.0%}", fontsize=FS_S - 1.5, color=INK2, va="bottom")
        ax.text(hi + 0.36, n - 1 - i + 0.12, f"{r['top10_share']:.1%}", fontsize=FS_S - 1.5, color=INK, va="bottom")
    ax.text(hi + 0.06, n + 0.35, "入选", fontsize=FS_S - 1.6, color=MUTED, va="bottom")
    ax.text(hi + 0.36, n + 0.35, "前 10", fontsize=FS_S - 1.6, color=MUTED, va="bottom")
    _log_ticks(ax, "x")
    ax.set_xlim(-0.02, hi)
    ax.set_ylim(-0.25, n + 0.9)
    ax.set_xlabel("屏蔽基因阳性在 SARA 候选中的名次（入选者，对数轴；竖线为中位数）", fontsize=FS_S - 0.6)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    return pd.DataFrame(rows)


def fig7() -> dict:
    from .pubfigs8x import p_e5_alluvial
    R = _nums()
    fam = _fam()
    H7 = 134
    fig = fig_mm(183, H7)
    src = {}
    # ---- a: masked sets matched on GO richness and PPI degree
    label_mm(fig, 2, 6, "a", "受控 GO 屏蔽集合与其余基因的分布")
    mg = pd.read_parquet(PUB / "mask_grid.parquet")
    groups = [("none", "未屏蔽", "#c9c8c2", 1.6), ("dev", "开发屏蔽 10%", MUTED, 1.0), ("test", "测试屏蔽 20%", INK, 0.9)]
    rows_a = []
    for k, (col, xl) in enumerate((("n_go_terms", "GO 术语数"), ("ppi_degree", "物理 PPI 度"))):
        ax = axmm(fig, 12 + k * 27, 12, 22, 30)
        for g, lab, c, lw in groups:
            v = np.sort(mg.loc[mg["mask_set"] == g, col].to_numpy(float))
            ax.plot(np.maximum(v, 0.5), np.arange(1, len(v) + 1) / len(v), color=c, lw=lw, label=f"{lab}（{len(v):,}）")
            rows_a.append(pd.DataFrame({"set": g, "variable": col, "value": v[::10],
                                        "cdf": (np.arange(1, len(v) + 1) / len(v))[::10]}))
        ax.set_xscale("log")
        ax.set_xlabel(xl, fontsize=FS_S - 0.4)
        if k == 0:
            ax.set_ylabel("累计比例", fontsize=FS_S - 0.3)
        else:
            ax.tick_params(labelleft=False)
        style(ax, grid="both")
        if k == 1:
            ax.legend(loc="lower right", fontsize=FS_S - 1.7, handlelength=1.0, borderaxespad=0.1)
    fig.text(12 / 183, 1 - 53.6 / H7, "三条累计分布几乎重合：屏蔽在 GO 术语数 × PPI 度十分位层内抽取", fontsize=FS_S - 1.5,
             color=INK2)
    src["a"] = pd.concat(rows_a)
    # ---- b: masked-gene recovery and overall effect under the controlled removals
    label_mm(fig, 66, 6, "b", "删除 GO 与撤出关联监督对屏蔽基因的影响（种子 42）")
    axb = axmm(fig, 96, 12, 42, 36)
    axo = axmm(fig, 146, 12, 35, 36)
    rows_b = []
    for i, (tag, lab) in enumerate(ARMS7):
        ev = pd.read_parquet(EVAL / f"eval_test_{tag}.parquet")
        ev = ev[(ev["method"] == "deployed") & (ev["seed"] == 42)]
        fm = np.array([fam.get(d, d) for d in ev["disease_id"]])
        ok = ev["n_query_mask_test"].to_numpy() > 0
        cv = fam_ratio_ci(ev["pool_mask_test"].to_numpy()[ok], ev["n_query_mask_test"].to_numpy()[ok], fm[ok], 2000)
        ht = fam_ratio_ci(ev["hits10_mask_test"].to_numpy()[ok], ev["n_query_mask_test"].to_numpy()[ok], fm[ok], 2000)
        axb.barh(i - 0.19, cv[0], height=0.34, color="#c9c8c2", zorder=2)
        axb.barh(i + 0.19, ht[0], height=0.34, color=INK2, zorder=2)
        axb.plot([cv[1], cv[2]], [i - 0.19, i - 0.19], color=INK, lw=0.6, zorder=3)
        axb.plot([ht[1], ht[2]], [i + 0.19, i + 0.19], color=INK, lw=0.6, zorder=3)
        axb.text(cv[2] + 0.012, i - 0.19, f"{cv[0]:.1%}", va="center", fontsize=FS_S - 1.6, color=INK2)
        axb.text(ht[2] + 0.012, i + 0.19, f"{ht[0]:.1%}", va="center", fontsize=FS_S - 1.6, color=INK)
        if tag != "main":
            dm = R["variants"][tag]["deployed"]["delta_vs_main"]
            axo.plot([dm["ci_low"], dm["ci_high"]], [i, i], color=INK2, lw=1.0, solid_capstyle="round")
            axo.plot([dm["delta"]], [i], "D", ms=3.2, color=INK2, mec=SURFACE, mew=0.5, zorder=3)
        else:
            dm = {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        rows_b.append({"arm": tag, "label": lab, "mask_pool": cv[0], "mask_pool_lo": cv[1], "mask_pool_hi": cv[2],
                       "mask_hit10": ht[0], "mask_hit10_lo": ht[1], "mask_hit10_hi": ht[2],
                       "overall_delta": dm["delta"], "overall_lo": dm["ci_low"], "overall_hi": dm["ci_high"]})
    axb.set_yticks(range(len(ARMS7)), [a[1] for a in ARMS7], fontsize=FS_S - 1.0)
    axb.set_ylim(len(ARMS7) - 0.45, -0.6)
    axb.set_xlim(0, 0.62)
    pct(axb, "x")
    axb.set_xlabel("屏蔽基因阳性比例（95% CI）", fontsize=FS_S - 0.7)
    axb.legend(handles=legend_handles([("进入候选 500", "#c9c8c2", "box"), ("进入前 10", INK2, "box")]),
               loc="lower right", fontsize=FS_S - 1.4, handlelength=0.8)
    style(axb, grid="x", left=False)
    axb.tick_params(axis="y", length=0)
    axo.axvline(0, color=AXIS, lw=0.7)
    axo.set_yticks([])
    axo.set_ylim(len(ARMS7) - 0.45, -0.6)
    axo.set_xlabel("全部疾病 ΔR@10（对完整模型）", fontsize=FS_S - 0.9)
    style(axo, grid="x", left=False)
    axo.text(0.0, -0.45, "参照", fontsize=FS_S - 1.6, color=INK2, ha="center", va="center")
    src["b"] = pd.DataFrame(rows_b)
    # ---- c: rank distributions
    label_mm(fig, 2, 62, "c", "屏蔽基因阳性的名次分布（种子 42，全部外层）")
    src["c"] = p_ridges9(fig, 2, 69, 100, 52, [("main", "完整模型", INK2), ("e5_none", "删 GO", MUTED),
                                                ("e6_cold", "撤出关联监督", CAT[0]),
                                                ("e6_cold_gomask", "撤出监督 + 删 GO", "#a8a7a1")])
    # ---- d: paired flows of the same positives
    label_mm(fig, 108, 62, "d", "同一批阳性的名次类别流转")
    src["d"] = p_e5_alluvial(fig, 108, 68, 74, 50, cols=(("main", "完整模型"), ("e5_none", "删 GO"),
                                                         ("e6_cold_gomask", "撤出监督 + 删 GO")))
    return save(fig, "Fig7_annotation_supervision", src)


# ====================================================================== Figure S2
def figS2() -> dict:
    dw = pd.read_parquet(PUB / "dev_w.parquet")
    dA = pd.read_parquet(PUB / "dev_A.parquet")
    fig = fig_mm(183, 92)
    label_mm(fig, 2, 6, "a", "RRF 权重 w 的开发集选择（覆盖@500）")
    ax = axmm(fig, 14, 12, 58, 70)
    for j in range(5):
        s_ = dw[dw["fold"] == j]
        aw = s_[s_["setting"].str.startswith("aware")]
        pl = s_[s_["setting"].str.startswith("plain")]
        ws = [float(x.split("_w")[1]) for x in aw["setting"]]
        ax.plot(ws, aw["coverage"], color=CAT[0], lw=1.0, alpha=0.4 + 0.12 * j, marker="o", ms=2.6, mec=SURFACE,
                mew=0.4)
        ax.plot([float(x.split("_w")[1]) for x in pl["setting"]], pl["coverage"], color="#6f6e69", lw=0.7,
                alpha=0.4 + 0.12 * j, linestyle=(0, (1.5, 1.2)))
        v6 = s_[s_["setting"] == "v6_500"]["coverage"].iloc[0]
        ax.plot([0.25, 0.95], [v6, v6], color=CAT[1], lw=0.6, alpha=0.4 + 0.12 * j)
        cw = s_["chosen_w"].iloc[0]
        cv = aw[[float(x.split("_w")[1]) == cw for x in aw["setting"]]]["coverage"].iloc[0]
        ax.plot([cw], [cv], "o", ms=4.4, color=CAT[0], mec=INK, mew=0.8, zorder=5)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.set_xlabel("w", fontsize=FS_S - 0.3)
    ax.set_ylabel("开发阳性覆盖@500", fontsize=FS_S - 0.3)
    pct(ax, "y")
    ax.legend(handles=[Line2D([0], [0], color=CAT[0], marker="o", ms=2.6, lw=1.0, label="可用性感知 RRF（5 个外层）"),
                       Line2D([0], [0], color="#6f6e69", lw=0.7, linestyle=(0, (1.5, 1.2)), label="普通 RRF"),
                       Line2D([0], [0], color=CAT[1], lw=0.6, label="检索前 500"),
                       Line2D([0], [0], marker="o", color=CAT[0], mec=INK, mew=0.8, lw=0, ms=4.4,
                              label=mt("选定的 $w_j$"))],
              loc="lower left", fontsize=FS_S - 1.2)
    style(ax, grid="both")
    label_mm(fig, 82, 6, "b", "SARA 排序器的六个候选配置：开发 Recall@10（黑框为选定配置）")
    nrm = Normalize(dA["dev_recall10"].min(), dA["dev_recall10"].max())
    for j in range(5):
        axh = axmm(fig, 90 + j * 18.2, 16, 15.5, 40)
        s_ = dA[dA["fold"] == j]
        M = np.zeros((3, 2))
        dep = None
        for r in s_.itertuples():
            ii = [15, 31, 63].index(r.num_leaves)
            jj = [50, 200].index(r.min_child_samples)
            M[ii, jj] = r.dev_recall10
            if r.deployed:
                dep = (ii, jj)
        for ii in range(3):
            for jj in range(2):
                fc = to_hex(seq_cmap()(0.1 + 0.85 * nrm(M[ii, jj])))
                axh.add_patch(FancyBboxPatch((jj + 0.05, ii + 0.05), 0.9, 0.9, boxstyle="round,pad=0,rounding_size=0.08",
                                            fc=fc, ec=INK if dep == (ii, jj) else "none", lw=1.0))
                axh.text(jj + 0.5, ii + 0.5, f"{M[ii, jj]:.3f}", ha="center", va="center", fontsize=FS_S - 1.6,
                         color=ink_on(fc))
        axh.set_xlim(0, 2)
        axh.set_ylim(3, 0)
        axh.set_xticks([0.5, 1.5], ["50", "200"], fontsize=FS_S - 1.4)
        axh.set_yticks([0.5, 1.5, 2.5], ["15", "31", "63"] if j == 0 else ["", "", ""], fontsize=FS_S - 1.4)
        axh.tick_params(length=0)
        for s2 in axh.spines.values():
            s2.set_visible(False)
        axh.set_title(f"外层 {j}", fontsize=FS_S - 0.8, pad=2, fontweight="normal", loc="center")
        if j == 0:
            axh.set_ylabel("叶数", fontsize=FS_S - 1.0)
        axh.set_xlabel("最小样本", fontsize=FS_S - 1.2)
    return save(fig, "FigS2_development_selection", {"a": dw, "b": dA})


# ====================================================================== Figure S8 (examined alternatives)
def figS8() -> dict:
    from .pubfigs8x import p_e5_paired
    R = _nums()
    fam = _fam()
    H = 263
    fig = fig_mm(183, H)
    src = {}
    cur = pd.read_parquet(PUB / "dev_C_curves.parquet")
    dA = pd.read_parquet(PUB / "dev_A.parquet")
    dC = pd.read_parquet(PUB / "dev_C.parquet")
    dB = pd.read_parquet(PUB / "dev_B.parquet")
    alpha_b = {0.0: 0.38, 1.0: 0.68, 2.0: 1.0}
    x0, w, gap = 13.0, 30.0, 3.8
    # ---- a: validation curves of the gated corrector
    label_mm(fig, 2, 6, "a", "门控纠偏器在各外层 $F_{j+1}$ 上的验证曲线（开发阶段，种子 42）")
    ymax = cur["val_recall10"].max() * 1.04
    ymin = cur["val_recall10"].min() * 0.96
    for j in range(5):
        ax = axmm(fig, x0 + j * (w + gap), 11, w, 28)
        sub = cur[cur["fold"] == j]
        for (b, lam), s_ in sub.groupby(["beta", "lam"]):
            ax.plot(s_["round"], s_["val_recall10"], color=MAG, alpha=alpha_b[b], lw=0.9 if lam == 0.1 else 0.7,
                    linestyle="-" if lam == 0.1 else (0, (1.2, 1.0)))
            k = int(s_["val_recall10"].to_numpy().argmax())
            ax.plot(s_["round"].iloc[k], s_["val_recall10"].iloc[k], "o", ms=2.6, color=MAG, alpha=alpha_b[b],
                    mec=SURFACE, mew=0.4, zorder=4)
        bi = dC[dC["fold"] == j]["best_iteration"]
        ax.set_title(f"外层 {j}（最佳轮次 {bi.min()}–{bi.max()}）", fontsize=FS_S - 0.5, pad=2, loc="center",
                     fontweight="normal")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 200)
        ax.set_xticks([0, 100, 200])
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(mt("$F_{j+1}$ Recall@10"), fontsize=FS_S - 0.3)
        ax.set_xlabel("提升轮次", fontsize=FS_S - 0.6)
        style(ax, grid="y")
    fig.legend(handles=[Line2D([0], [0], color=MAG, alpha=a_, lw=1.0, label=f"β = {b:g}") for b, a_ in alpha_b.items()]
               + [Line2D([0], [0], color=INK2, lw=0.9, label="λ = 0.1"),
                  Line2D([0], [0], color=INK2, lw=0.7, linestyle=(0, (1.2, 1.0)), label="λ = 0.03")],
               loc="upper right", bbox_to_anchor=(181 / 183, 1 - 2.2 / H), ncol=5, fontsize=FS_S - 1.0,
               handlelength=1.6, columnspacing=0.9)
    src["a"] = cur
    # ---- b: development selection space
    label_mm(fig, 2, 52, "b", "开发集上的模型选择：纠偏器未进入“最佳排序器 − 0.005”区间")
    mk_b = {0.0: "o", 1.0: "s", 2.0: "^"}
    for j in range(5):
        ax = axmm(fig, x0 + j * (w + gap), 56, w, 27)
        a_ = dA[dA["fold"] == j]
        c_ = dC[dC["fold"] == j]
        b_ = dB[dB["fold"] == j]
        best = a_["dev_recall10"].max()
        ax.axvspan(best - 0.005, best, color=BLUE[100], lw=0, zorder=0)
        ax.axvline(best, color=CAT[0], lw=0.5, zorder=1)
        for r in a_.itertuples():
            ax.plot(r.dev_recall10, r.dev_cold_recall10, "o", ms=3.4 if not r.deployed else 4.4, color=CAT[0],
                    mec=INK if r.deployed else SURFACE, mew=0.9 if r.deployed else 0.5, zorder=3)
        for r in c_.itertuples():
            filled = r.lam == 0.1
            ax.plot(r.dev_recall10, r.dev_cold_recall10, mk_b[r.beta], ms=3.3, color=MAG if filled else SURFACE,
                    mec=MAG, mew=0.8, zorder=3)
        ax.plot(b_["dev_recall10"], b_["dev_cold_recall10"], "D", ms=3.2, color=YEL, mec=SURFACE, mew=0.5, zorder=3)
        ax.set_xlim(0.09, 0.19)
        ax.set_ylim(-0.004, 0.075)
        ax.set_xticks([0.10, 0.14, 0.18])
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(mt("开发 $R^c$@10"), fontsize=FS_S - 0.3)
        ax.set_xlabel("开发 Recall@10", fontsize=FS_S - 0.6)
        ax.set_title(f"外层 {j}", fontsize=FS_S, pad=2, loc="center", fontweight="normal")
        style(ax, grid="both")
    fig.legend(handles=[Line2D([0], [0], marker="o", color=CAT[0], lw=0, ms=3.6, mec=SURFACE, label="SARA 排序器配置"),
                        Line2D([0], [0], marker="o", color=CAT[0], lw=0, ms=4.4, mec=INK, mew=0.9, label="选定配置"),
                        Line2D([0], [0], marker="o", color=MAG, lw=0, ms=3.4, label="纠偏 β=0（实心 λ=0.1）"),
                        Line2D([0], [0], marker="s", color=MAG, lw=0, ms=3.2, label="β=1"),
                        Line2D([0], [0], marker="^", color=SURFACE, mec=MAG, mew=0.8, lw=0, ms=3.4,
                               label="β=2（空心 λ=0.03）"),
                        Line2D([0], [0], marker="D", color=YEL, lw=0, ms=3.2, mec=SURFACE, label="基础排序器"),
                        Patch(fc=BLUE[100], ec="none", label="最佳排序器 − 0.005 以内")],
               loc="upper center", bbox_to_anchor=(0.5, 1 - 94.0 / H), ncol=7, fontsize=FS_S - 1.2,
               handletextpad=0.3, columnspacing=0.8)
    src["b"] = pd.concat([dA.assign(model="A"), dC.assign(model="corrector"), dB.assign(model="B")])
    # ---- c: test-set contrasts of the corrector
    label_mm(fig, 2, 106, "c", "测试集：三个 β 的纠偏器相对 SARA（三种子，开发集选定 λ）")
    con = R["e2"]["contrasts"]
    rows = [("β = 0", con["corrector_beta0"]), ("β = 1", con["corrector_beta1"]), ("β = 2", con["corrector_beta2"])]
    axl = axmm(fig, 20, 114, 30, 30)
    axr = axmm(fig, 56, 114, 30, 30)
    rows_c = []
    for i, (lab, c) in enumerate(rows):
        for axx, key in ((axl, "recall10_vs_deployed"), (axr, "cold_vs_deployed")):
            e = c[key]
            axx.plot([e["ci_low"], e["ci_high"]], [i, i], color=MAG, lw=1.0)
            axx.plot([e["delta"]], [i], "o", ms=3.8, color=MAG, mec=SURFACE, mew=0.5, zorder=3)
            rows_c.append({"beta": lab, "metric": key, **e})
    for i, (lab, key) in enumerate((("β2 − β0（冷）", "corrector_beta2_vs_beta0_cold"),
                                    ("β1 − β0（冷）", "corrector_beta1_vs_beta0_cold"))):
        e = con[key]
        yv = 3.4 + i
        axr.plot([e["ci_low"], e["ci_high"]], [yv, yv], color=INK2, lw=1.0)
        axr.plot([e["delta"]], [yv], "D", ms=3.2, color=INK2, mec=SURFACE, mew=0.5, zorder=3)
        rows_c.append({"beta": lab, "metric": key, **e})
    for axx, xl in ((axl, "ΔRecall@10"), (axr, mt("Δ$R^c$@10"))):
        axx.axvline(0, color=AXIS, lw=0.7)
        axx.set_xlabel(xl, fontsize=FS_S - 0.3)
        style(axx, grid="x", left=False)
        axx.tick_params(axis="y", length=0)
    axl.set_ylim(4.9, -0.7)
    axr.set_ylim(4.9, -0.7)
    axl.set_yticks([0, 1, 2], [r[0] for r in rows], fontsize=FS_S - 0.5)
    axr.set_yticks([0, 1, 2, 3.4, 4.4], ["", "", "", "β2 − β0", "β1 − β0"], fontsize=FS_S - 1.0)
    axl.set_xlim(-0.045, 0.004)
    axr.set_xlim(-0.014, 0.014)
    src["c"] = pd.DataFrame(rows_c)
    # ---- d: development check of the sequence-imputation models
    label_mm(fig, 96, 106, "d", "序列补位模型在开发屏蔽臂上的效果（需 > 0）")
    st = json.loads((PUB / "dev_misc.json").read_text(encoding="utf-8"))["student"]["per_context"]
    cfgs = ["mlp_wd0.0001", "mlp_wd0.01", "linear_wd0.0001", "linear_wd0.01"]
    cfg_lab = {"mlp_wd0.0001": "两层 wd 1e-4", "mlp_wd0.01": "两层 wd 1e-2", "linear_wd0.0001": "线性 wd 1e-4",
               "linear_wd0.01": "线性 wd 1e-2"}
    ax = axmm(fig, 116, 114, 40, 26)
    rows_d = []
    for i, c in enumerate(cfgs):
        for ctx, v in sorted(st.items()):
            r = next(x for x in v["rows"] if x["config"] == c)
            ax.plot(r["gain"], i - 0.12, "o", ms=3.2, color=CAT[6], mec=SURFACE, mew=0.4, zorder=3)
            ax.plot(r["shuffled_gain"], i + 0.16, "o", ms=3.0, color=SURFACE, mec=CAT[6], mew=0.8, zorder=3)
            rows_d.append({"context": ctx, "config": c, "gain": r["gain"], "shuffled_gain": r["shuffled_gain"],
                           "coverage_delta": r["coverage_delta"], "top10_self": r["gene_level_masked"]["top10"],
                           "median_rank": r["gene_level_masked"]["median_rank"],
                           "cos_true": r["gene_level_masked"]["mean_cos_true"],
                           "cos_pairwise": r["gene_level_masked"]["mean_pairwise_cos"]})
    ax.axvline(0, color=INK2, lw=0.7)
    ax.axvspan(0, 0.02, color=BLUE[100], lw=0, zorder=0)
    ax.set_xlim(-0.12, 0.02)
    ax.set_yticks(range(len(cfgs)), [cfg_lab[c] for c in cfgs], fontsize=FS_S - 0.9)
    ax.set_ylim(len(cfgs) - 0.5, -0.6)
    ax.set_xlabel("屏蔽基因阳性入 500 比例的变化", fontsize=FS_S - 0.6)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=CAT[6], lw=0, ms=3.2, mec=SURFACE, label="补位模型（5 个外层）"),
                       Line2D([0], [0], marker="o", color=SURFACE, mec=CAT[6], mew=0.8, lw=0, ms=3.0,
                              label="长度层内打乱配对")], loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=2,
              fontsize=FS_S - 1.4, handlelength=0.8, columnspacing=0.8)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    src["d"] = pd.DataFrame(rows_d)
    # ---- e: gene-level quality of the imputed vectors
    label_mm(fig, 2, 160, "e", "屏蔽基因的补位输出质量")
    B_ = pd.DataFrame(rows_d)
    ax = axmm(fig, 22, 168, 18, 28)
    ax2 = axmm(fig, 44, 168, 20, 28)
    for i, c in enumerate(cfgs):
        s_ = B_[B_["config"] == c]
        ax.plot(s_["top10_self"], np.full(len(s_), i), "o", ms=3.0, color=CAT[6], mec=SURFACE, mew=0.4)
        for r in s_.itertuples():
            ax2.plot([r.cos_pairwise, r.cos_true], [i, i], color=AXIS, lw=0.6, zorder=1)
        ax2.plot(s_["cos_pairwise"], np.full(len(s_), i), "o", ms=2.8, color=SURFACE, mec=INK2, mew=0.7, zorder=2)
        ax2.plot(s_["cos_true"], np.full(len(s_), i), "o", ms=3.0, color=CAT[6], mec=SURFACE, mew=0.4, zorder=3)
    ax.set_yticks(range(len(cfgs)), [cfg_lab[c] for c in cfgs], fontsize=FS_S - 1.1)
    ax2.set_yticks([])
    for a_ in (ax, ax2):
        a_.set_ylim(len(cfgs) - 0.5, -0.6)
        style(a_, grid="x", left=False)
        a_.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.2)
    pct(ax, "x")
    ax.set_xlabel("自身 GO 向量\n前 10 命中率", fontsize=FS_S - 1.0)
    ax2.set_xlabel("余弦：真实 vs 平均两两", fontsize=FS_S - 1.0)
    ax2.set_xlim(0.1, 0.65)
    src["e"] = B_[["context", "config", "top10_self", "median_rank", "cos_true", "cos_pairwise"]]
    # ---- f: test mask arms with sequence imputation
    label_mm(fig, 70, 160, "f", "测试屏蔽基因的 GO 被删除后，各序列补位方式的效果（种子 42）")
    arms = [("main", "完整模型（GO 保留）"), ("e5_none", "删 GO·不补位"), ("e5_mlp", "两层补位"), ("e5_linear", "线性补位"),
            ("e5_nn", "序列近邻转移"), ("e5_teacher_sS", "语义模型序列投影"), ("e5_mlp_shuffled", "两层补位（配对打乱）"),
            ("e5_protst", "ProtST 输入补位*")]
    axc = axmm(fig, 100, 168, 26, 42)
    axh = axmm(fig, 130, 168, 24, 42)
    axo = axmm(fig, 158, 168, 23, 42)
    rows_f = []
    for i, (tag, lab) in enumerate(arms):
        ev = pd.read_parquet(EVAL / f"eval_test_{tag}.parquet")
        ev = ev[(ev["method"] == "deployed") & (ev["seed"] == 42)]
        fm = np.array([fam.get(d, d) for d in ev["disease_id"]])
        ok = ev["n_query_mask_test"].to_numpy() > 0
        cv = fam_ratio_ci(ev["pool_mask_test"].to_numpy()[ok], ev["n_query_mask_test"].to_numpy()[ok], fm[ok], 2000)
        ht = fam_ratio_ci(ev["hits10_mask_test"].to_numpy()[ok], ev["n_query_mask_test"].to_numpy()[ok], fm[ok], 2000)
        colr = INK2 if tag == "main" else (MUTED if tag == "e5_none" else CAT[6])
        if tag in ("e5_mlp_shuffled", "e5_protst"):
            colr = "#a8a7a1"
        for axx, e in ((axc, cv), (axh, ht)):
            axx.plot([e[1], e[2]], [i, i], color=colr, lw=1.0)
            axx.plot([e[0]], [i], "o", ms=3.6, color=colr, mec=SURFACE, mew=0.5, zorder=3)
        if tag != "main":
            dm = R["variants"][tag]["deployed"]["delta_vs_main"]
            axo.plot([dm["ci_low"], dm["ci_high"]], [i, i], color=colr, lw=1.0)
            axo.plot([dm["delta"]], [i], "D", ms=3.0, color=colr, mec=SURFACE, mew=0.5, zorder=3)
        else:
            dm = {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        rows_f.append({"arm": tag, "label": lab, "mask_pool": cv[0], "mask_pool_lo": cv[1], "mask_pool_hi": cv[2],
                       "mask_hit10": ht[0], "mask_hit10_lo": ht[1], "mask_hit10_hi": ht[2],
                       "overall_delta": dm["delta"], "overall_lo": dm["ci_low"], "overall_hi": dm["ci_high"]})
    axc.axvline(rows_f[1]["mask_pool"], color=MUTED, lw=0.6, zorder=0)
    axh.axvline(rows_f[1]["mask_hit10"], color=MUTED, lw=0.6, zorder=0)
    axo.axvline(0, color=AXIS, lw=0.7)
    for axx, xl in ((axc, "入候选 500"), (axh, "进入前 10"), (axo, "总体 ΔR@10")):
        axx.set_ylim(len(arms) - 0.4, -0.6)
        axx.set_xlabel(xl, fontsize=FS_S - 0.8)
        style(axx, grid="x", left=False)
        axx.tick_params(axis="y", length=0)
    axc.set_yticks(range(len(arms)), [a[1] for a in arms], fontsize=FS_S - 1.1)
    axh.set_yticks([])
    axo.set_yticks([])
    pct(axc, "x")
    pct(axh, "x")
    axc.axhline(0.5, color=AXIS, lw=0.6)
    src["f"] = pd.DataFrame(rows_f)
    # ---- g: per-positive paired comparison against no imputation
    label_mm(fig, 2, 222, "g", "逐对比较：各序列补位方式 vs 不补位")
    src["g"] = p_e5_paired(fig, 2, 227, 110, 16)
    fig.text(2 / 183, 1 - (H - 2.5) / H, "* ProtST 编码器以 Swiss-Prot 功能描述训练，属于开放知识对照。",
             fontsize=FS_S - 1.3, color=INK2)
    return save(fig, "FigS8_alternative_designs", src)

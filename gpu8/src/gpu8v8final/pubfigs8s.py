"""Supplementary figures S1-S7."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from . import paths8
from .pubcolor8 import AXIS, BLUE, CAT, ENTITY, INK, INK2, MUTED, SURFACE, ink_on, seq_cmap
from .pubstyle8 import (FS, FS_S, axmm, fam_ci, fig_mm, heat_table, label_mm, legend_handles, mt, pct, save, style)

PUB = paths8.RESULTS / "pub"
EVAL = paths8.RESULTS / "eval"


def _nums() -> dict:
    return json.loads((paths8.RESULTS / "numbers_v8_results.json").read_text(encoding="utf-8"))


def _fam() -> dict:
    d = pd.read_parquet(PUB / "diseases.parquet")
    return dict(zip(d["disease_id"], d["family_group"]))


# ====================================================================== S1 teacher training
def figS1() -> dict:
    th = pd.read_parquet(PUB / "teacher_histories.parquet")
    misc = json.loads((PUB / "dev_misc.json").read_text(encoding="utf-8"))["teacher_controls"]["per_context"]
    fig = fig_mm(183, 104)
    label_mm(fig, 2, 6, "a", "语义教师的验证曲线：同一上下文 $F_{j+1}$ 上 $s_F$ 的全背景 Recall@10")
    seed_style = {42: 1.0, 52: 0.62, 62: 0.35}
    rows = []
    for j in range(5):
        ax = axmm(fig, 13 + j * 34.5, 12, 27, 34)
        sub = th[th["ctx"] == f"f{j}_outer"]
        for (controls, seed), s_ in sub.groupby(["controls", "seed"]):
            colr = CAT[6] if controls == "background" else "#a8a7a1"
            ax.plot(s_["epoch"], s_["sF_recall10"], color=colr, alpha=seed_style.get(seed, 1.0), lw=1.0)
            b = s_["best_epoch"].iloc[0]
            v = s_.loc[s_["epoch"] == b, "sF_recall10"]
            if len(v):
                ax.plot([b], [v.iloc[0]], "o", ms=2.8, color=colr, alpha=seed_style.get(seed, 1.0), mec=SURFACE,
                        mew=0.4, zorder=4)
            rows.append(s_.assign(ctx=f"f{j}_outer"))
        ax.set_title(f"外层 {j}", fontsize=FS_S, pad=2, loc="center", fontweight="normal")
        ax.set_xlabel("轮次", fontsize=FS_S - 0.6)
        if j == 0:
            ax.set_ylabel(mt("$s_F$ Recall@10"), fontsize=FS_S - 0.3)
        style(ax, grid="y")
    fig.legend(handles=[Line2D([0], [0], color=CAT[6], lw=1.0, alpha=a, label=f"全背景对照 · 种子 {s}")
                        for s, a in seed_style.items()] +
               [Line2D([0], [0], color="#a8a7a1", lw=1.0, label="参考关系基因对照 · 种子 42")],
               loc="upper center", bbox_to_anchor=(0.5, 1 - 54 / 104), ncol=4, fontsize=FS_S - 1.2)
    label_mm(fig, 2, 64, "b", "未标注对照集的开发集选择与教师的最佳轮次（种子 42）")
    ax = axmm(fig, 22, 68, 70, 24)
    ctxs = sorted(misc)
    for i, c in enumerate(ctxs):
        v = misc[c]["dev_sF_recall10"]
        ax.plot([v["reference"], v["background"]], [i, i], color=AXIS, lw=1.0, zorder=1)
        ax.plot(v["reference"], i, "o", ms=3.6, color="#a8a7a1", mec=SURFACE, mew=0.5, zorder=3)
        ax.plot(v["background"], i, "o", ms=3.8, color=CAT[6], mec=SURFACE, mew=0.5, zorder=3)
        ax.text(v["background"] + 0.0008, i, f"+{v['background'] - v['reference']:.3f}", va="center",
                fontsize=FS_S - 1.4, color=INK2)
    ax.set_yticks(range(len(ctxs)), [f"外层 {c[1]}" for c in ctxs], fontsize=FS_S - 0.8)
    ax.set_ylim(len(ctxs) - 0.5, -0.6)
    ax.set_xlabel(mt("开发集 $s_F$ Recall@10"), fontsize=FS_S - 0.5)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    ax2 = axmm(fig, 112, 68, 68, 24)
    ls = th.groupby(["ctx", "controls", "seed"]).agg(best=("best_epoch", "first"), n=("epoch", "max")).reset_index()
    ls = ls[ls["controls"] == "background"]
    for i, c in enumerate(sorted(ls["ctx"].unique())):
        s_ = ls[ls["ctx"] == c]
        ax2.scatter(s_["best"], np.full(len(s_), i), s=14, color=CAT[6], edgecolors=SURFACE, linewidths=0.4, zorder=3)
        ax2.scatter(s_["n"], np.full(len(s_), i), s=12, color=SURFACE, edgecolors=INK2, linewidths=0.7, zorder=3)
    ax2.set_yticks(range(5), [f"外层 {k}" for k in range(5)], fontsize=FS_S - 0.8)
    ax2.set_ylim(4.5, -0.6)
    ax2.set_xlabel("轮次（实心：最佳；空心：早停时）", fontsize=FS_S - 0.5)
    style(ax2, grid="x", left=False)
    ax2.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="#a8a7a1", lw=0, ms=3.6, label="参考关系基因对照"),
                       Line2D([0], [0], marker="o", color=CAT[6], lw=0, ms=3.8, label="全背景对照（5/5 选中）")],
              loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, fontsize=FS_S - 1.4)
    return save(fig, "FigS1_teacher_training", {"a": pd.concat(rows) if rows else th})


# ====================================================================== S2 development selection
def figS2() -> dict:
    dw = pd.read_parquet(PUB / "dev_w.parquet")
    dA = pd.read_parquet(PUB / "dev_A.parquet")
    fig = fig_mm(183, 100)
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
    ax.legend(handles=[Line2D([0], [0], color=CAT[0], marker="o", ms=2.6, lw=1.0, label="缺失感知 RRF（5 个外层）"),
                       Line2D([0], [0], color="#6f6e69", lw=0.7, linestyle=(0, (1.5, 1.2)), label="普通 RRF"),
                       Line2D([0], [0], color=CAT[1], lw=0.6, label="V6 500"),
                       Line2D([0], [0], marker="o", color=CAT[0], mec=INK, mew=0.8, lw=0, ms=4.4,
                              label=mt("选定的 $w_j$"))],
              loc="lower left", fontsize=FS_S - 1.2)
    style(ax, grid="both")
    label_mm(fig, 82, 6, "b", "候选 A 的六个配置：开发 Recall@10（黑框为部署）")
    for j in range(5):
        axh = axmm(fig, 90 + j * 18.2, 16, 15.5, 22)
        s_ = dA[dA["fold"] == j]
        M = np.zeros((3, 2))
        dep = None
        for r in s_.itertuples():
            ii = [15, 31, 63].index(r.num_leaves)
            jj = [50, 200].index(r.min_child_samples)
            M[ii, jj] = r.dev_recall10
            if r.deployed:
                dep = (ii, jj)
        from matplotlib.colors import Normalize, to_hex
        nrm = Normalize(dA["dev_recall10"].min(), dA["dev_recall10"].max())
        for ii in range(3):
            for jj in range(2):
                fc = to_hex(seq_cmap()(0.1 + 0.85 * nrm(M[ii, jj])))
                axh.add_patch(FancyBboxPatch((jj + 0.05, ii + 0.05), 0.9, 0.9, boxstyle="round,pad=0,rounding_size=0.08",
                                            fc=fc, ec=INK if dep == (ii, jj) else "none", lw=1.0))
                axh.text(jj + 0.5, ii + 0.5, f"{M[ii, jj]:.3f}", ha="center", va="center", fontsize=FS_S - 1.8,
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
    label_mm(fig, 82, 52, "c", "每个外层：B、最佳纠偏器与最佳 A 的开发 Recall@10")
    dC = pd.read_parquet(PUB / "dev_C.parquet")
    dB = pd.read_parquet(PUB / "dev_B.parquet")
    ax = axmm(fig, 96, 57, 76, 27)
    for j in range(5):
        b = float(dB[dB["fold"] == j]["dev_recall10"].iloc[0])
        c = dC[dC["fold"] == j]["dev_recall10"].max()
        a = dA[dA["fold"] == j]["dev_recall10"].max()
        ax.plot([b, a], [j, j], color=AXIS, lw=1.0, zorder=1)
        ax.plot(b, j, "D", ms=3.4, color=CAT[3], mec=SURFACE, mew=0.4, zorder=3)
        ax.plot(c, j, "o", ms=3.8, color=CAT[4], mec=SURFACE, mew=0.4, zorder=3)
        ax.plot(a, j, "o", ms=4.0, color=CAT[0], mec=SURFACE, mew=0.4, zorder=3)
        ax.axvspan(a - 0.005, a, ymin=(4.6 - j - 0.3) / 5.1, ymax=(4.6 - j + 0.3) / 5.1, color=BLUE[100], lw=0,
                   zorder=0)
        ax.text(a + 0.002, j, f"差 {a - c:.3f}", va="center", fontsize=FS_S - 1.5, color=INK2)
    ax.set_yticks(range(5), [f"外层 {j}" for j in range(5)], fontsize=FS_S - 0.8)
    ax.set_ylim(4.6, -0.5)
    ax.set_xlabel("开发 Recall@10", fontsize=FS_S - 0.5)
    ax.legend(handles=[Line2D([0], [0], marker="D", color=CAT[3], lw=0, ms=3.4, label="B"),
                       Line2D([0], [0], marker="o", color=CAT[4], lw=0, ms=3.8, label="最佳纠偏器"),
                       Line2D([0], [0], marker="o", color=CAT[0], lw=0, ms=4.0, label="最佳 A"),
                       __import__("matplotlib").patches.Patch(fc=BLUE[100], ec="none", label="最佳 A − 0.005 以内")],
              loc="upper center", bbox_to_anchor=(0.45, -0.28), fontsize=FS_S - 1.3, ncol=4)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    return save(fig, "FigS2_development_selection", {"a": dw, "b": dA, "c": dC})


# ====================================================================== S3 per fold x seed
def figS3() -> dict:
    from .pubfigs8a import MLAB
    ev = pd.read_parquet(EVAL / "eval_test_main.parquet")
    meths = ["deployed", "v71_direct_strict", "v71_direct_v6", "channel_lgb_v6", "B", "standalone_r", "standalone_sF",
             "F7_v6", "F7_strict", "standalone_sF0", "standalone_sS"]
    cols = [(j, s) for j in range(5) for s in (42, 52, 62)]
    M = np.zeros((len(meths), len(cols)))
    for i, m in enumerate(meths):
        for k, (j, s) in enumerate(cols):
            sub = ev[(ev["method"] == m) & (ev["fold"] == j) & (ev["seed"] == s)]
            M[i, k] = sub["recall10"].mean()
    H = 164
    fig = fig_mm(183, H)
    label_mm(fig, 2, 6, "a", "各外层 × 种子的端到端 Recall@10（行：方法；列：外层·种子；最右为行均值）")
    ax = axmm(fig, 52, 16, 118, 62)
    Mx = np.c_[M, M.mean(axis=1)]
    heat_table(ax, Mx, [MLAB[m] for m in meths], [f"{j}·{s}" for j, s in cols] + ["均值"], seq_cmap(), norm_by="all",
               fmt="{:.3f}", text_size=FS_S - 2.0, col_label_rot=90)
    ax.tick_params(axis="x", labelsize=FS_S - 1.3)
    ax.tick_params(axis="y", labelsize=FS_S - 0.8)
    df = pd.DataFrame(Mx, index=meths, columns=[f"f{j}_s{s}" for j, s in cols] + ["mean"]).reset_index()
    src = {"a": df}

    # ---------------------------------------------------- b: per-fold win / tie / loss and family-bootstrap intervals
    fam = _fam()
    label_mm(fig, 2, 90, "b", "逐外层的逐病比较（三种子均值）：胜 / 平 / 负与家族 bootstrap 区间")
    per = ev[ev["method"].isin(["deployed", "channel_lgb_v6", "v71_direct_strict"])].groupby(
        ["method", "fold", "disease_id"])["recall10"].mean().unstack(0).reset_index()
    comps = [("channel_lgb_v6", "对仅通道 LGB", ENTITY["channel_lgb_v6"]),
             ("v71_direct_strict", "对 V7.1 直接融合", ENTITY["v71_direct_strict"])]
    axb = axmm(fig, 30, 98, 62, 48)
    axf = axmm(fig, 98, 98, 34, 48)
    rows_b, ylab, yt = [], [], []
    y = 0.0
    for j in list(range(5)) + ["全部"]:
        sub = per if j == "全部" else per[per["fold"] == j]
        for ci_, (m, lab, col) in enumerate(comps):
            d = (sub["deployed"] - sub[m]).to_numpy(float)
            sh = [(d > 0).mean(), (d == 0).mean(), (d < 0).mean()]
            left = 0.0
            for v, c in zip(sh, (CAT[0], "#e2e1dc", CAT[1])):
                axb.barh(y, v, left=left, height=0.8, color=c, edgecolor=SURFACE, linewidth=0.5, zorder=2)
                if v > 0.09:
                    axb.text(left + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=FS_S - 2.2,
                             color=ink_on(c))
                left += v
            e = fam_ci(d, np.array([fam.get(x, x) for x in sub["disease_id"]]), 2000)
            axf.plot([e[1], e[2]], [y, y], color=INK2, lw=1.0, solid_capstyle="round")
            axf.plot([e[0]], [y], "o" if ci_ == 0 else "s", ms=3.2, color=INK2 if ci_ == 0 else SURFACE, mec=INK2,
                     mew=0.8, zorder=3)
            rows_b.append({"fold": j, "comparator": m, "n": int(len(d)), "better": sh[0], "tie": sh[1], "worse": sh[2],
                           "delta": e[0], "ci_low": e[1], "ci_high": e[2]})
            yt.append(y)
            ylab.append(f"外层 {j} · {lab}" if j != "全部" else f"全部 · {lab}")
            y += 1.0
        y += 0.55
    for a_ in (axb, axf):
        a_.set_ylim(y - 0.3, -0.7)
    axb.set_yticks(yt, ylab, fontsize=FS_S - 1.6)
    axb.set_xlim(0, 1)
    pct(axb, "x")
    axb.set_xlabel("疾病比例", fontsize=FS_S - 0.8)
    style(axb, grid=None, left=False)
    axb.tick_params(axis="y", length=0)
    axb.legend(handles=legend_handles([("部署更高", CAT[0], "box"), ("相同", "#e2e1dc", "box"), ("部署更低", CAT[1], "box")]),
               loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=FS_S - 1.6, handlelength=0.8)
    axf.legend(handles=[Line2D([0], [0], marker="o", color=INK2, mec=INK2, lw=1.0, ms=3.2, label="对仅通道 LGB"),
                        Line2D([0], [0], marker="s", color=INK2, mfc=SURFACE, mec=INK2, mew=0.8, lw=1.0, ms=3.2,
                               label="对 V7.1 直接融合")],
               loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=FS_S - 1.7, handlelength=1.4)
    axf.axvline(0, color=AXIS, lw=0.7)
    axf.set_yticks([])
    axf.set_xlabel("ΔRecall@10（95% CI）", fontsize=FS_S - 0.8)
    style(axf, grid="x", left=False)
    src["b"] = pd.DataFrame(rows_b)

    # ---------------------------------------------------- c: seed-to-seed agreement of per-disease Recall@10
    label_mm(fig, 138, 90, "c", "种子间一致性（逐病 R@10）")
    dep = ev[ev["method"] == "deployed"].pivot_table(index="disease_id", columns="seed", values="recall10")
    rows_c = []
    from matplotlib.colors import LogNorm
    for k, (s1, s2) in enumerate(((42, 52), (42, 62))):
        axc = axmm(fig, 146, 99 + k * 27, 22, 20)
        a_, b_ = dep[s1].to_numpy(float), dep[s2].to_numpy(float)
        hb = axc.hexbin(a_, b_, gridsize=14, extent=(0, 1, 0, 1), mincnt=1, cmap=seq_cmap(), norm=LogNorm(),
                        linewidths=0.2, edgecolors=SURFACE)
        axc.plot([0, 1], [0, 1], color=INK2, lw=0.5, ls=(0, (2, 2)))
        r = float(np.corrcoef(a_, b_)[0, 1])
        same = float((a_ == b_).mean())
        axc.set_title(f"r = {r:.3f}；逐病值相同 {same:.0%}", fontsize=FS_S - 1.9, pad=1.5, fontweight="normal",
                      loc="center", color=INK2)
        axc.set_xlabel(f"种子 {s1}", fontsize=FS_S - 1.4, labelpad=1)
        axc.set_ylabel(f"种子 {s2}", fontsize=FS_S - 1.4, labelpad=1)
        axc.tick_params(labelsize=FS_S - 2.0)
        axc.set_xticks([0, 0.5, 1])
        axc.set_yticks([0, 0.5, 1])
        style(axc, grid=None)
        rows_c.append({"seed_a": s1, "seed_b": s2, "pearson_r": r, "identical_share": same, "n": int(len(a_))})
    import matplotlib as mpl
    cax = axmm(fig, 171, 110, 1.4, 30)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=hb.norm, cmap=seq_cmap()), cax=cax)
    cb.set_label("每格疾病数", fontsize=FS_S - 1.9, labelpad=1)
    cb.ax.tick_params(labelsize=FS_S - 2.1, length=1)
    cb.outline.set_visible(False)
    src["c"] = pd.DataFrame(rows_c)
    return save(fig, "FigS3_fold_seed_stability", src)


# ====================================================================== S4 semantic standalone variants
def figS4() -> dict:
    R = _nums()
    t = pd.DataFrame(R["semantic"]["table"])
    fig = fig_mm(183, 70)
    label_mm(fig, 2, 6, "a", "全背景单独语义排名（种子 42）")
    ax = axmm(fig, 50, 12, 62, 50)
    t = t.sort_values("recall10", ascending=True).reset_index(drop=True)
    for i, r in t.iterrows():
        colr = CAT[6] if "s_F" in r["method"] else "#a8a7a1"
        ax.barh(i, r["recall10"], height=0.6, color=colr, zorder=2)
        ax.plot(r["cold_recall10"], i, "o", ms=3.6, color=SURFACE, mec=INK2, mew=0.9, zorder=3)
        ax.text(r["recall10"] + 0.002, i, f"{r['recall10']:.3f}", va="center", fontsize=FS_S - 1.4, color=INK)
    ax.set_yticks(range(len(t)), [mt(m.replace("s_F", "$s_F$")) for m in t["method"]], fontsize=FS_S - 0.9)
    ax.set_ylim(len(t) - 0.4, -0.6)
    ax.set_xlabel(mt("Recall@10（条）与 $R^c$@10（圈）"), fontsize=FS_S - 0.5)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    label_mm(fig, 118, 6, "b", "相对教师 $s_F$（分段文本）的差值")
    ax2 = axmm(fig, 124, 12, 56, 50)
    rows = []
    for i, r in t.iterrows():
        v = r.get("vs_sF")
        if not isinstance(v, dict):
            ax2.text(0, i, "参照", ha="center", va="center", fontsize=FS_S - 1.3, color=INK2)
            continue
        ax2.plot([v["ci_low"], v["ci_high"]], [i, i], color=INK2, lw=1.0)
        ax2.plot([v["delta"]], [i], "o", ms=3.6, color=INK2, mec=SURFACE, mew=0.5, zorder=3)
        rows.append({"method": r["method"], **v})
    ax2.axvline(0, color=AXIS, lw=0.7)
    ax2.set_yticks([])
    ax2.set_ylim(len(t) - 0.4, -0.6)
    ax2.set_xlabel("ΔRecall@10（与左图同行；95% CI）", fontsize=FS_S - 0.5)
    style(ax2, grid="x", left=False)
    return save(fig, "FigS4_semantic_variants", {"a": t.drop(columns=["vs_sF"], errors="ignore"), "b": pd.DataFrame(rows)})


# ====================================================================== S5 disease embedding
def figS5() -> dict:
    from .pubdata8 import MESH_CAT
    ts = pd.read_parquet(PUB / "tsne_diseases.parquet")
    cs = pd.read_parquet(PUB / "cosine_known_random.parquet")
    meta = json.loads((PUB / "tsne_meta.json").read_text(encoding="utf-8"))
    cats = ["C04", "C10", "C16", "C14", "C18", "C17", "C06", "F03"]
    fig = fig_mm(183, 108)
    fig.text(2 / 183, 1 - 6 / 108, mt(f"a  语义教师的疾病向量 $z_d$ 的 t-SNE（外层 0，种子 42；{meta['n']:,} 个有文本或 HPO 的疾病）"),
             fontsize=7.2, fontweight="bold", va="baseline")
    for k, c in enumerate(cats):
        r_, q_ = divmod(k, 4)
        ax = axmm(fig, 2 + q_ * 33.5, 10 + r_ * 45, 32, 40)
        on = ts["cats"].fillna("").str.contains(c)
        ax.scatter(ts.loc[~on, "x"], ts.loc[~on, "y"], s=0.6, color="#dcdbd5", linewidths=0, rasterized=True)
        ax.scatter(ts.loc[on, "x"], ts.loc[on, "y"], s=1.1, color=CAT[6], linewidths=0, rasterized=True)
        ax.set_title(f"{MESH_CAT[c]}（{int(on.sum()):,}）", fontsize=FS_S - 1.2, pad=1.5, fontweight="normal",
                     loc="center")
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
    fig.text(138 / 183, 1 - 6 / 108, "b  已知关联 vs 随机基因的余弦", fontsize=7.2, fontweight="bold", va="baseline")
    ax = axmm(fig, 142, 14, 38, 76)
    bins = np.linspace(-0.4, 1.0, 71)
    ax.hist(cs["cos_random"], bins=bins, color="#c9c8c2", density=True, label="随机有 GO 基因")
    ax.hist(cs["cos_known"], bins=bins, color=CAT[6], alpha=0.75, density=True, label="已知关联基因")
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(cs["cos_known"], cs["cos_random"])
    auc = u.statistic / (len(cs) * len(cs))
    ax.text(0.03, 0.97, f"{len(cs):,} 对\nAUC = {auc:.3f}", transform=ax.transAxes, va="top", fontsize=FS_S - 1.0,
            color=INK)
    ax.set_xlabel(mt("$\\cos(z_d, z^F_g)$"), fontsize=FS_S - 0.4)
    ax.set_ylabel("密度", fontsize=FS_S - 0.4)
    ax.legend(loc="upper right", fontsize=FS_S - 1.3)
    style(ax, grid="y")
    fig.text(0.5, 0.012, "说明：训练数据包含这些关联，余弦分离只说明语义空间把疾病与其已知基因拉近，不是测试集性能证据。", ha="center",
             fontsize=FS_S - 1.5, color=INK2)
    return save(fig, "FigS5_embedding", {"a": ts, "b": pd.DataFrame({"auc": [auc], "pairs": [len(cs)]})})


# ====================================================================== S6 compute resources
def figS6() -> dict:
    led = pd.read_parquet(PUB / "ledger.parquet")
    led = led[led["status"] == "done"].copy()
    zh = {"teacher": "语义教师", "student": "序列学生", "base_ranker": "基础排序器 B", "corrector": "门控纠偏器",
          "candidate_A": "候选 A", "fold": "折级汇总", "kdgene": "KDGene", "sign": "SIGN 式"}
    led = led[led["component"].isin(zh)]
    led["comp"] = led["component"].map(zh)
    order = [zh[k] for k in ("teacher", "student", "corrector", "candidate_A", "base_ranker", "kdgene", "sign", "fold")]
    fig = fig_mm(183, 92)
    label_mm(fig, 2, 6, "a", "运行时间线（账本完成时间；点大小 ∝ 计时秒数）")
    ax = axmm(fig, 26, 12, 96, 70)
    t0 = led["finished"].min()
    for i, c in enumerate(order):
        s_ = led[led["comp"] == c]
        x = (s_["finished"] - t0).dt.total_seconds() / 3600
        sz = 3 + 25 * np.sqrt(s_["seconds"].fillna(0) / max(led["seconds"].max(), 1))
        ax.scatter(x, np.full(len(s_), i) + np.random.default_rng(i).uniform(-0.25, 0.25, len(s_)), s=sz,
                   color=CAT[0] if c in ("候选 A", "语义教师") else "#6f6e69", alpha=0.55, linewidths=0, rasterized=True)
    ax.set_yticks(range(len(order)), order, fontsize=FS_S - 0.8)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlabel("自第一次完成起的小时数", fontsize=FS_S - 0.4)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    label_mm(fig, 128, 6, "b", "各组件运行数与计时机时")
    ax2 = axmm(fig, 148, 12, 25, 70)
    g = led.groupby("comp").agg(n=("run_id", "count"), sec=("seconds", "sum"))
    g = g.reindex(order).fillna(0)
    ax2.barh(range(len(order)), g["sec"] / 3600, height=0.6, color="#a8a7a1", zorder=2)
    for i, (n, s) in enumerate(zip(g["n"], g["sec"])):
        ax2.text(s / 3600 + 0.05, i, f"{int(n)} 次", va="center", fontsize=FS_S - 1.4, color=INK2)
    ax2.set_yticks(range(len(order)), order, fontsize=FS_S - 0.9)
    ax2.set_ylim(len(order) - 0.4, -0.6)
    ax2.set_xlabel("小时（有计时的运行）", fontsize=FS_S - 0.5)
    style(ax2, grid="x", left=False)
    ax2.tick_params(axis="y", length=0)
    return save(fig, "FigS6_compute_resources", {"a": led[["run_id", "comp", "role", "finished", "seconds",
                                                             "peak_gpu_allocated_mib"]], "b": g.reset_index()})


# ====================================================================== S7 E4 block relations
def figS7() -> dict:
    R = _nums()
    e4 = R["e4"]
    arms = [("e4_layer1", "训练可见来源排除"), ("e4_layer1_random", "  匹配随机"), ("e4_layer2", "评价条件化删除*"),
            ("e4_layer2_random", "  匹配随机*"), ("e4_blocks", "证据块隔离"), ("e4_blocks_random", "  匹配随机"),
            ("e4_experimental", "仅实验类证据"), ("e4_no_tas_nas", "去 TAS/NAS"), ("e4_no_mutant", "去突变/遗传相互作用"),
            ("e4_rescore_no_ic", "去 IC（重打分）"), ("e4_rescore_iea_only", "仅 IEA（重打分）")]
    fig = fig_mm(183, 92)
    label_mm(fig, 2, 6, "a", "被隔离证据块中 2,497 条关系的前 10 恢复：完整数据 → 删减数据")
    ax = axmm(fig, 40, 14, 60, 64)
    rows = []
    vals = []
    for i, (k, lab) in enumerate(arms):
        r = e4[k]
        for off, (m, colr) in ((-0.15, ("deployed", CAT[0])), (0.15, ("channel_lgb_v6", CAT[1]))):
            full = r.get(f"block_hit10_full_{m}")
            arm = r.get(f"block_hit10_{m}")
            if full is None or arm is None:
                continue
            ax.annotate("", xy=(arm, i + off), xytext=(full, i + off),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", color=colr, lw=0.8))
            ax.plot([full], [i + off], "o", ms=2.8, color=SURFACE, mec=colr, mew=0.8, zorder=3)
            rows.append({"arm": k, "method": m, "full": full, "arm_value": arm})
            vals += [full, arm]
    ax.set_yticks(range(len(arms)), [a[1] for a in arms], fontsize=FS_S - 0.9)
    ax.set_ylim(len(arms) - 0.4, -0.6)
    ax.set_xlim(min(vals) - 0.004, max(vals) + 0.004)
    pct(ax, "x", digits=1)
    ax.set_xlabel("证据块关系的前 10 恢复率（空心：完整数据；箭头：删减数据）", fontsize=FS_S - 0.6)
    fig.legend(handles=legend_handles([("V8.1 部署", CAT[0], "line"), ("仅通道 LGB", CAT[1], "line")]),
               loc="upper right", bbox_to_anchor=(100 / 183, 1 - 8.5 / 92), ncol=2, fontsize=FS_S - 1.2)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    label_mm(fig, 108, 6, "b", "删减数据中的新旧方法差")
    ax2 = axmm(fig, 112, 14, 68, 64)
    full = e4["e4_layer1"]["new_minus_old_full"]
    ax2.axvspan(full["ci_low"], full["ci_high"], color="#eef0ee", lw=0)
    ax2.axvline(full["delta"], color=INK2, lw=0.7)
    for i, (k, lab) in enumerate(arms):
        v = e4[k]["new_minus_old_reduced"]
        ax2.plot([v["ci_low"], v["ci_high"]], [i, i], color=CAT[0], lw=1.0)
        ax2.plot([v["delta"]], [i], "o", ms=3.4, color=CAT[0], mec=SURFACE, mew=0.5, zorder=3)
    ax2.set_yticks([])
    ax2.set_ylim(len(arms) - 0.4, -0.6)
    ax2.set_xlabel("部署 − 仅通道 LGB（Recall@10）", fontsize=FS_S - 0.6)
    ax2.text(full["delta"], -0.9, "完整数据（带：95% CI）", fontsize=FS_S - 1.5, color=INK2, ha="center")
    style(ax2, grid="x", left=False)
    return save(fig, "FigS7_evidence_blocks", {"a": pd.DataFrame(rows)})

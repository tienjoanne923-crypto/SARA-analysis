"""Figures 6-8: second-stage diagnostics (E2), GO masking and sequence imputation (E5/E6), evidence sources (E4)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import paths8
from .pubcolor8 import AXIS, BLUE, CAT, ENTITY, INK, INK2, MUTED, SURFACE, seq_cmap
from .pubstyle8 import (FS, FS_S, axmm, fam_ratio_ci, fig_mm, heat_table, label_mm, legend_handles, mt, pct, save,
                        style)

PUB = paths8.RESULTS / "pub"
EVAL = paths8.RESULTS / "eval"
MAG, YEL = CAT[4], CAT[3]


def _nums() -> dict:
    return json.loads((paths8.RESULTS / "numbers_v8_results.json").read_text(encoding="utf-8"))


def _fam() -> dict:
    d = pd.read_parquet(PUB / "diseases.parquet")
    return dict(zip(d["disease_id"], d["family_group"]))


# ====================================================================== Figure 6
def fig6() -> dict:
    R = _nums()
    H6 = 250
    fig = fig_mm(183, H6)
    src = {}
    cur = pd.read_parquet(PUB / "dev_C_curves.parquet")
    dA = pd.read_parquet(PUB / "dev_A.parquet")
    dC = pd.read_parquet(PUB / "dev_C.parquet")
    dB = pd.read_parquet(PUB / "dev_B.parquet")
    alpha_b = {0.0: 0.38, 1.0: 0.68, 2.0: 1.0}
    x0, w, gap = 13.0, 30.0, 3.8
    # ---------------------------------------------------- a: validation curves
    label_mm(fig, 2, 6, "a", "门控纠偏器在各外层 $F_{j+1}$ 上的验证曲线（开发阶段，种子 42）")
    axes_a = []
    ymax = cur["val_recall10"].max() * 1.04
    ymin = cur["val_recall10"].min() * 0.96
    for j in range(5):
        ax = axmm(fig, x0 + j * (w + gap), 11, w, 30)
        axes_a.append(ax)
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
               loc="upper right", bbox_to_anchor=(181 / 183, 1 - 2.2 / H6), ncol=5, fontsize=FS_S - 1.0,
               handlelength=1.6, columnspacing=0.9)
    src["a"] = cur

    # ---------------------------------------------------- b: development A/B decision space
    label_mm(fig, 2, 55, "b", "开发集上的 A/B 部署规则：纠偏器没有进入“最佳 A − 0.005”区间")
    mk_b = {0.0: "o", 1.0: "s", 2.0: "^"}
    rows_b = []
    for j in range(5):
        ax = axmm(fig, x0 + j * (w + gap), 59, w, 29)
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
        ax.set_title(f"外层 {j}：部署 A", fontsize=FS_S, pad=2, loc="center", fontweight="normal")
        style(ax, grid="both")
        rows_b.append({"fold": j, "best_A": best, "best_corrector": c_["dev_recall10"].max(),
                       "B": float(b_["dev_recall10"].iloc[0]), "margin_to_best_A": best - c_["dev_recall10"].max()})
    fig.legend(handles=[Line2D([0], [0], marker="o", color=CAT[0], lw=0, ms=3.6, mec=SURFACE, label="候选 A 配置"),
                        Line2D([0], [0], marker="o", color=CAT[0], lw=0, ms=4.4, mec=INK, mew=0.9, label="部署的 A"),
                        Line2D([0], [0], marker="o", color=MAG, lw=0, ms=3.4, label="纠偏 β=0（实心 λ=0.1）"),
                        Line2D([0], [0], marker="s", color=MAG, lw=0, ms=3.2, label="β=1"),
                        Line2D([0], [0], marker="^", color=SURFACE, mec=MAG, mew=0.8, lw=0, ms=3.4,
                               label="β=2（空心 λ=0.03）"),
                        Line2D([0], [0], marker="D", color=YEL, lw=0, ms=3.2, mec=SURFACE, label="基础排序器 B"),
                        __import__("matplotlib").patches.Patch(fc=BLUE[100], ec="none", label="最佳 A − 0.005 以内")],
               loc="upper center", bbox_to_anchor=(0.5, 1 - 98.5 / H6), ncol=7, fontsize=FS_S - 1.2,
               handletextpad=0.3, columnspacing=0.8)
    src["b"] = pd.concat([dA.assign(model="A"), dC.assign(model="corrector"), dB.assign(model="B")])

    # ---------------------------------------------------- c: test-set E2 contrasts
    label_mm(fig, 2, 110, "c", "测试集：三个 β 的纠偏器相对部署模型（三种子，登记 λ）")
    con = R["e2"]["contrasts"]
    rows = [("β = 0", con["corrector_beta0"]), ("β = 1", con["corrector_beta1"]), ("β = 2", con["corrector_beta2"])]
    axl = axmm(fig, 20, 118, 30, 30)
    axr = axmm(fig, 56, 118, 30, 30)
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
        y = 3.4 + i
        axr.plot([e["ci_low"], e["ci_high"]], [y, y], color=INK2, lw=1.0)
        axr.plot([e["delta"]], [y], "D", ms=3.2, color=INK2, mec=SURFACE, mew=0.5, zorder=3)
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
    axr.text(0.0, -0.62, "冷恢复无差别", fontsize=FS_S - 1.2, color=INK2, ha="center")
    src["c"] = pd.DataFrame(rows_c)

    # ---------------------------------------------------- d: feature importance of the deployed A models
    label_mm(fig, 96, 110, "d", "部署的候选 A：特征增益占比（5 外层 × 3 种子）")
    fi = pd.read_parquet(PUB / "feature_importance.parquet")
    from .pubdata8 import FEAT_ZH, feat_family
    agg = fi.groupby("feature")["gain_share"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    top = agg.head(14)
    fam_col = {"候选融合": CAT[0], "语义": CAT[6], "V6 检索通道": CAT[1], "表达": "#6f6e69", "疾病描述": "#a8a7a1",
               "基因可用性": "#c9c8c2"}
    ax = axmm(fig, 128, 118, 34, 40)
    for i, (f, r) in enumerate(top.iterrows()):
        c = fam_col[feat_family(f)]
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
    axf = axmm(fig, 166, 118, 15, 40)
    bottom = 0.0
    for fam_, v in fam_tot.items():
        axf.bar(0, v, bottom=bottom, width=0.7, color=fam_col[fam_], edgecolor=SURFACE, linewidth=0.6)
        if v > 0.06:
            axf.text(0, bottom + v / 2, f"{fam_}\n{v:.0%}", ha="center", va="center", fontsize=FS_S - 1.9,
                     color="#ffffff" if fam_ in ("候选融合", "语义", "V6 检索通道", "表达") else INK, linespacing=1.1)
        bottom += v
    axf.set_xlim(-0.5, 0.5)
    axf.set_ylim(0, 1)
    axf.axis("off")
    axf.text(0, 1.02, "按特征族", ha="center", va="bottom", fontsize=FS_S - 1.2, color=INK2)
    src["d"] = agg.reset_index().assign(family=lambda d_: d_["feature"].map(feat_family),
                                        label=lambda d_: d_["feature"].map(FEAT_ZH))

    # ---------------------------------------------------- e-h: TreeSHAP of the deployed model (post hoc, descriptive)
    from .pubfigs8x import p_shap_beeswarm, p_shap_dependence, p_shap_family_strata
    label_mm(fig, 2, 170, "e", "部署模型的 TreeSHAP（外层 0，种子 42；全部留出阳性 + 4 万未标注候选）")
    src["e"] = p_shap_beeswarm(fig, 2, 178, 90, 62)
    label_mm(fig, 100, 170, "f", "SHAP 依赖：融合名次")
    src["f"] = p_shap_dependence(fig, 109, 178, 30, 22, "lr_R", "缺失感知 RRF 名次",
                                 xform=lambda v: 500.0 * np.exp(-v), logx=True, xticks=(1, 10, 100, 500),
                                 ylab="SHAP：融合名次", legend="lower left")
    label_mm(fig, 143, 170, "g", "SHAP 依赖：语义分数 $s_F$")
    src["g"] = p_shap_dependence(fig, 152, 178, 29, 22, "s_sem", "语义分数 $s_F$", ylab="SHAP：$s_F$",
                                 cbar_box=(118, 209.3, 52, 1.3))
    label_mm(fig, 100, 219.5, "h", "各特征族的平均 SHAP（按候选类别，带符号）")
    src["h"] = p_shap_family_strata(fig, 100, 222.5, 81, 26)
    return save(fig, "Fig6_second_stage_diagnostics", src)


# ====================================================================== Figure 7
def fig7() -> dict:
    R = _nums()
    fam = _fam()
    H7 = 228
    fig = fig_mm(183, H7)
    src = {}
    # ---------------------------------------------------- a: masked sets are matched on GO richness and PPI degree
    label_mm(fig, 2, 6, "a", "屏蔽集合与其余基因的分布")
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

    # ---------------------------------------------------- b: development acceptance of the student
    label_mm(fig, 66, 6, "b", "学生在开发屏蔽臂上的验收（需 > 0）")
    st = json.loads((PUB / "dev_misc.json").read_text(encoding="utf-8"))["student"]["per_context"]
    cfgs = ["mlp_wd0.0001", "mlp_wd0.01", "linear_wd0.0001", "linear_wd0.01"]
    cfg_lab = {"mlp_wd0.0001": "两层 wd 1e-4", "mlp_wd0.01": "两层 wd 1e-2", "linear_wd0.0001": "线性 wd 1e-4",
               "linear_wd0.01": "线性 wd 1e-2"}
    ax = axmm(fig, 84, 12, 36, 30)
    rows_b = []
    for i, c in enumerate(cfgs):
        for j, (ctx, v) in enumerate(sorted(st.items())):
            r = next(x for x in v["rows"] if x["config"] == c)
            ax.plot(r["gain"], i - 0.12, "o", ms=3.2, color=CAT[6], mec=SURFACE, mew=0.4, zorder=3)
            ax.plot(r["shuffled_gain"], i + 0.16, "o", ms=3.0, color=SURFACE, mec=CAT[6], mew=0.8, zorder=3)
            rows_b.append({"context": ctx, "config": c, "gain": r["gain"], "shuffled_gain": r["shuffled_gain"],
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
    fig.legend(handles=[Line2D([0], [0], marker="o", color=CAT[6], lw=0, ms=3.2, mec=SURFACE, label="学生（5 个外层）"),
                        Line2D([0], [0], marker="o", color=SURFACE, mec=CAT[6], mew=0.8, lw=0, ms=3.0,
                               label="长度层内打乱配对")], loc="upper center", bbox_to_anchor=(102 / 183, 1 - 48.5 / H7),
               ncol=2, fontsize=FS_S - 1.4, handlelength=0.8, columnspacing=0.8)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    src["b"] = pd.DataFrame(rows_b)

    # ---------------------------------------------------- c: gene-level quality of the student output
    label_mm(fig, 128, 6, "c", "屏蔽基因的学生输出质量")
    B_ = pd.DataFrame(rows_b)
    ax = axmm(fig, 134, 12, 20, 30)
    ax2 = axmm(fig, 159, 12, 22, 30)
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
    src["c"] = B_[["context", "config", "top10_self", "median_rank", "cos_true", "cos_pairwise"]]

    # ---------------------------------------------------- d: test mask arms (E5)
    label_mm(fig, 2, 58, "d", "E5：测试屏蔽基因的 GO 被删除后，各补位方式的效果（种子 42）")
    arms = [("main", "主模型（GO 未删）"), ("e5_none", "无补位"), ("e5_mlp", "两层学生"), ("e5_linear", "线性学生"),
            ("e5_nn", "序列近邻转移"), ("e5_teacher_sS", "教师序列投影"), ("e5_mlp_shuffled", "两层学生（配对打乱）"),
            ("e5_protst", "ProtST 输入学生*")]
    axc = axmm(fig, 40, 66, 34, 60)
    axh = axmm(fig, 79, 66, 34, 60)
    axo = axmm(fig, 118, 66, 26, 60)
    rows_d = []
    for i, (tag, lab) in enumerate(arms):
        ev = pd.read_parquet(EVAL / f"eval_test_{tag}.parquet")
        ev = ev[(ev["method"] == "deployed") & (ev["seed"] == 42)]
        fm = np.array([fam.get(d, d) for d in ev["disease_id"]])
        ok = ev["n_query_mask_test"].to_numpy() > 0
        cv = fam_ratio_ci(ev["pool_mask_test"].to_numpy()[ok], ev["n_query_mask_test"].to_numpy()[ok], fm[ok], 2000)
        ht = fam_ratio_ci(ev["hits10_mask_test"].to_numpy()[ok], ev["n_query_mask_test"].to_numpy()[ok], fm[ok], 2000)
        colr = INK2 if tag == "main" else (MUTED if tag in ("e5_none",) else CAT[6])
        if tag in ("e5_mlp_shuffled", "e5_protst"):
            colr = "#a8a7a1"
        for axx, e in ((axc, cv), (axh, ht)):
            axx.plot([e[1], e[2]], [i, i], color=colr, lw=1.0)
            axx.plot([e[0]], [i], "o", ms=3.8, color=colr, mec=SURFACE, mew=0.5, zorder=3)
        if tag != "main":
            dm = R["variants"][tag]["deployed"]["delta_vs_main"]
            axo.plot([dm["ci_low"], dm["ci_high"]], [i, i], color=colr, lw=1.0)
            axo.plot([dm["delta"]], [i], "D", ms=3.2, color=colr, mec=SURFACE, mew=0.5, zorder=3)
        else:
            dm = {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        rows_d.append({"arm": tag, "label": lab, "mask_pool": cv[0], "mask_pool_lo": cv[1], "mask_pool_hi": cv[2],
                       "mask_hit10": ht[0], "mask_hit10_lo": ht[1], "mask_hit10_hi": ht[2],
                       "overall_delta": dm["delta"], "overall_lo": dm["ci_low"], "overall_hi": dm["ci_high"]})
    none_cov = rows_d[1]["mask_pool"]
    axc.axvline(none_cov, color=MUTED, lw=0.6, zorder=0)
    axh.axvline(rows_d[1]["mask_hit10"], color=MUTED, lw=0.6, zorder=0)
    axo.axvline(0, color=AXIS, lw=0.7)
    for axx, xl in ((axc, "屏蔽基因阳性入 500"), (axh, "屏蔽基因阳性前 10"), (axo, "总体 ΔR@10（对主模型）")):
        axx.set_ylim(len(arms) - 0.4, -0.6)
        axx.set_xlabel(xl, fontsize=FS_S - 0.5)
        style(axx, grid="x", left=False)
        axx.tick_params(axis="y", length=0)
    axc.set_yticks(range(len(arms)), [a[1] for a in arms], fontsize=FS_S - 0.8)
    axh.set_yticks([])
    axo.set_yticks([])
    pct(axc, "x")
    pct(axh, "x")
    axc.axhline(0.5, color=AXIS, lw=0.6)
    fig.text(2 / 183, 1 - (H7 - 3.0) / H7, "* ProtST 编码器以 Swiss-Prot 功能描述训练，属于开放知识对照，不用于“序列迁移”结论。",
             fontsize=FS_S - 1.3, color=INK2)
    src["d"] = pd.DataFrame(rows_d)

    # ---------------------------------------------------- e: association-supervision cold (E6)
    label_mm(fig, 150, 58, "e", "E6：关联监督冷")
    arms6 = [("main", "主模型"), ("e5_none", "删 GO"), ("e6_cold", "撤出监督"), ("e6_cold_gomask", "撤出监督\n+ 删 GO")]
    ax = axmm(fig, 156, 66, 25, 60)
    rows_e = []
    for i, (tag, lab) in enumerate(arms6):
        ev = pd.read_parquet(EVAL / f"eval_test_{tag}.parquet")
        ev = ev[(ev["method"] == "deployed") & (ev["seed"] == 42)]
        fm = np.array([fam.get(d, d) for d in ev["disease_id"]])
        ok = ev["n_query_mask_test"].to_numpy() > 0
        cv = fam_ratio_ci(ev["pool_mask_test"].to_numpy()[ok], ev["n_query_mask_test"].to_numpy()[ok], fm[ok], 2000)
        ht = fam_ratio_ci(ev["hits10_mask_test"].to_numpy()[ok], ev["n_query_mask_test"].to_numpy()[ok], fm[ok], 2000)
        ax.barh(i - 0.19, cv[0], height=0.34, color="#c9c8c2", zorder=2)
        ax.barh(i + 0.19, ht[0], height=0.34, color=INK2, zorder=2)
        ax.text(cv[0] + 0.01, i - 0.19, f"{cv[0]:.1%}", va="center", fontsize=FS_S - 1.5, color=INK2)
        ax.text(ht[0] + 0.01, i + 0.19, f"{ht[0]:.1%}", va="center", fontsize=FS_S - 1.5, color=INK)
        rows_e.append({"arm": tag, "label": lab.replace("\n", ""), "mask_pool": cv[0], "mask_hit10": ht[0]})
    ax.set_yticks(range(len(arms6)), [a[1] for a in arms6], fontsize=FS_S - 1.0)
    ax.set_ylim(len(arms6) - 0.45, -0.6)
    ax.set_xlim(0, 0.62)
    pct(ax, "x")
    ax.set_xlabel("屏蔽基因阳性比例", fontsize=FS_S - 0.6)
    ax.legend(handles=legend_handles([("入 500", "#c9c8c2", "box"), ("前 10", INK2, "box")]), loc="lower right",
              fontsize=FS_S - 1.2, handlelength=0.8)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    src["e"] = pd.DataFrame(rows_e)

    # ---------------------------------------------------- f-h: rank-level view of the same masked-gene positives
    from .pubfigs8x import p_e5_alluvial, p_e5_paired, p_e5_ridges
    label_mm(fig, 2, 143, "f", "屏蔽基因阳性的名次分布（E5/E6 各臂，种子 42，全部外层）")
    src["f"] = p_e5_ridges(fig, 2, 150, 100, 60)
    label_mm(fig, 106, 143, "g", "同一批阳性的名次类别流转")
    src["g"] = p_e5_alluvial(fig, 106, 149, 76, 33)
    label_mm(fig, 106, 190, "h", "逐对比较：各补位方式 vs 无补位")
    src["h"] = p_e5_paired(fig, 106, 195, 76, 16)
    return save(fig, "Fig7_go_mask_sequence", src)


# ====================================================================== Figure 8
def fig8() -> dict:
    R = _nums()
    H8 = 240
    fig = fig_mm(183, H8)
    src = {}
    # ---------------------------------------------------- a: evidence codes and PMID availability
    label_mm(fig, 2, 6, "a", "GO 生产记录的证据代码与 PMID 可得性（845,871 条）")
    codes = pd.read_parquet(PUB / "go_evidence_codes.parquet").fillna(0)
    codes["total"] = codes["with_pmid"] + codes["no_pmid"]
    codes = codes.sort_values("total", ascending=False).reset_index(drop=True)
    cat_of = {"EXP": "实验", "IDA": "实验", "IPI": "实验", "IMP": "实验", "IGI": "实验", "IEP": "实验",
              "HTP": "高通量", "HDA": "高通量", "HMP": "高通量", "HGI": "高通量", "HEP": "高通量",
              "IBA": "系统发育", "IBD": "系统发育", "IKR": "系统发育", "IRD": "系统发育",
              "ISS": "计算", "ISO": "计算", "ISA": "计算", "ISM": "计算", "IGC": "计算", "RCA": "计算",
              "TAS": "作者声明", "NAS": "作者声明", "IC": "管理员", "ND": "管理员", "IEA": "电子"}
    ax = axmm(fig, 16, 12, 64, 50)
    for i, r in codes.iterrows():
        ax.barh(i, r["with_pmid"], height=0.66, color=INK2, zorder=2)
        ax.barh(i, r["no_pmid"], left=r["with_pmid"], height=0.66, color="#d6d5cf", zorder=2, edgecolor=SURFACE,
                linewidth=0.4)
        share = r["no_pmid"] / r["total"] if r["total"] else 0
        ax.text(r["total"] + 4000, i, f"{r['total']:,.0f}  ({share:.0%} 无 PMID)" if share > 0.01 else
                f"{r['total']:,.0f}", va="center", fontsize=FS_S - 1.6, color=INK2)
    ax.set_yticks(range(len(codes)), [f"{c}（{cat_of.get(c, '其他')}）" for c in codes["evidence_code"]],
                  fontsize=FS_S - 1.2)
    ax.set_ylim(len(codes) - 0.4, -0.6)
    ax.set_xlim(0, codes["total"].max() * 1.45)
    ax.xaxis.set_major_formatter(__import__("matplotlib").ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("注释记录数", fontsize=FS_S - 0.4)
    ax.legend(handles=legend_handles([("有 PMID", INK2, "box"), ("无 PMID（未知出处）", "#d6d5cf", "box")]),
              loc="lower right", fontsize=FS_S - 1.1)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    src["a"] = codes

    # ---------------------------------------------------- b: PMID-relation block sizes
    label_mm(fig, 96, 6, "b", "PMID–关系二部图的连通块")
    bs = pd.read_parquet(PUB / "evidence_block_sizes.parquet")
    au = R.get("acceptance")  # noqa: F841
    audit = json.loads((paths8.AUDIT / "evidence_policy_audit_v8.json").read_text(encoding="utf-8"))
    l3 = audit["layer3_evidence_blocks"]
    ax = axmm(fig, 108, 12, 36, 42)
    ax.scatter(bs["relations_per_block"], bs["n_blocks"], s=9, color=INK2, edgecolors=SURFACE, linewidths=0.4, zorder=3)
    gi = bs["relations_per_block"].max()
    ax.scatter([gi], [1], s=26, color=CAT[7], edgecolors=SURFACE, linewidths=0.5, zorder=4)
    ax.annotate(f"巨块：{l3['giant_component_relations']:,} 条关系\n（{l3['giant_share']:.1%}，不拆分、不抽取）",
                xy=(gi, 1), xytext=(gi / 60, 6), fontsize=FS_S - 1.2, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("每块包含的关系数", fontsize=FS_S - 0.4)
    ax.set_ylabel("块数", fontsize=FS_S - 0.4)
    style(ax, grid="both")
    ax2 = axmm(fig, 150, 12, 31, 42)
    s_sorted = np.sort(np.repeat(bs["relations_per_block"].to_numpy(), bs["n_blocks"].to_numpy()))
    cum = np.cumsum(s_sorted[::-1]) / s_sorted.sum()
    ax2.plot(np.arange(1, len(cum) + 1), cum, color=INK2, lw=1.1)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.02)
    pct(ax2, "y")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.spines["right"].set_visible(True)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("按大小降序累计的块数", fontsize=FS_S - 0.6)
    ax2.set_ylabel("覆盖的有 PMID 关系比例", fontsize=FS_S - 0.6)
    ax2.text(0.97, 0.06, f"连通块 {l3['n_components']:,} 个\n中位大小 {l3['component_size_quantiles']['0.5']:.0f}，"
                         f"P99 = {l3['component_size_quantiles']['0.99']:.0f}\n抽取 {l3['blocks_chosen']:,} 块 · "
                         f"{l3['relations_held']:,} 条关系", transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=FS_S - 1.5, color=INK2, linespacing=1.3)
    ax2.xaxis.grid(True, color="#e1e0d9", lw=0.4)
    ax2.yaxis.grid(True, color="#e1e0d9", lw=0.4)
    ax2.set_axisbelow(True)
    src["b"] = bs

    # ---------------------------------------------------- c: deletion magnitudes (five denominators)
    label_mm(fig, 2, 74, "c", "每种删除的五种分母（占完整数据的比例）")
    v = audit["variants"]
    l1 = audit["layer1_training_visible_outer"]

    def shares(d):
        return [d["citations"]["deleted"] / d["citations"]["total"], d["annotation_rows"]["share"],
                d["unique_go"]["gene_term_pairs_lost"] / d["unique_go"]["gene_term_pairs_total"],
                d["genes"]["affected"] / d["genes"]["total_with_go"],
                d["query_relations"]["gene_affected"] / d["query_relations"]["total"]]

    l1m = np.mean([shares(x) for x in l1.values()], axis=0)
    arms = [("训练可见来源排除（5 外层均值）", l1m), ("评价条件化删除*", shares(audit["layer2_evaluation_conditioned"])),
            ("  其匹配随机*", shares(audit["layer2_matched_random"])), ("证据块隔离（GO 部分）", shares(l3["go_denominators"])),
            ("仅实验类证据", shares(v["experimental"])), ("去 TAS/NAS", shares(v["no_tas_nas"])),
            ("去突变/遗传相互作用", shares(v["no_mutant_genetic"])), ("去 IC", shares(v["no_ic"])),
            ("仅 IEA", shares(v["iea_only"]))]
    M = np.array([a[1] for a in arms])
    ax = axmm(fig, 40, 88, 44, 58)
    from matplotlib.colors import LogNorm, to_hex
    cm = seq_cmap()
    nrm = LogNorm(vmin=1e-4, vmax=1.0)
    for i in range(M.shape[0]):
        for jx in range(M.shape[1]):
            val = M[i, jx]
            fc = to_hex(cm(nrm(max(val, 1e-4)))) if val > 0 else "#f4f4f1"
            ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch(
                (jx + 0.04, i + 0.04), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec="none"))
            txt = f"{val:.1%}" if val >= 0.001 else (f"{val:.2%}" if val > 0 else "0")
            from .pubcolor8 import ink_on
            ax.text(jx + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=FS_S - 1.6, color=ink_on(fc))
    ax.set_xlim(0, M.shape[1])
    ax.set_ylim(M.shape[0], 0)
    ax.set_xticks(np.arange(5) + 0.5, ["引文", "注释行", "基因–\n术语对", "受影响\n基因", "受影响\n关系"], fontsize=FS_S - 1.1,
                  linespacing=1.1)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(arms)) + 0.5, [a[0] for a in arms], fontsize=FS_S - 1.0)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.text(2 / 183, 1 - (H8 - 3.0) / H8, "* 使用了评价关系的 PMID：自动标记 evaluation_conditioned_audit，只作审计、不回流选择。",
             fontsize=FS_S - 1.4, color=INK2)
    src["c"] = pd.DataFrame(M, index=[a[0] for a in arms], columns=["citations", "annotation_rows", "gene_term_pairs",
                                                                     "genes", "query_relations"]).reset_index()

    # ---------------------------------------------------- d: difference of differences
    label_mm(fig, 92, 74, "d", "“新旧方法差”的变化：删减数据 − 完整数据（种子 42）")
    e4 = R["e4"]
    order = [("e4_layer1", "训练可见来源排除", "L1"), ("e4_layer1_random", "  匹配随机", "L1"),
             ("e4_layer2", "评价条件化删除*", "L2"), ("e4_layer2_random", "  匹配随机*", "L2"),
             ("e4_blocks", "证据块隔离", "L3"), ("e4_blocks_random", "  匹配随机", "L3"),
             ("e4_experimental", "仅实验类证据", "code"), ("e4_no_tas_nas", "去 TAS/NAS", "code"),
             ("e4_no_mutant", "去突变/遗传相互作用", "code"),
             ("e4_rescore_no_ic", "去 IC（重打分）", "rescore"), ("e4_rescore_iea_only", "仅 IEA（重打分）", "rescore")]
    ax = axmm(fig, 124, 84, 57, 64)
    rows_d = []
    y = 0.0
    ypos, labels = [], []
    prev = None
    for key, lab, grp in order:
        if prev is not None and grp != prev:
            y += 0.6
        prev = grp
        d_ = e4[key]["difference"]
        rnd = lab.strip().startswith("匹配随机")
        colr = "#a8a7a1" if rnd else (CAT[0] if grp != "rescore" else INK2)
        ax.plot([d_["ci_low"], d_["ci_high"]], [y, y], color=colr, lw=1.0)
        ax.plot([d_["delta"]], [y], "o" if not rnd else "s", ms=3.6 if not rnd else 3.0, color=colr, mec=SURFACE,
                mew=0.5, zorder=3)
        if d_["ci_high"] < 0:
            ax.text(d_["ci_high"] + 0.0012, y, "区间 < 0", fontsize=FS_S - 1.5, color=INK, va="center")
        ypos.append(y)
        labels.append(lab)
        rows_d.append({"arm": key, "label": lab.strip(), "group": grp, **d_,
                       "full_gap": e4[key]["new_minus_old_full"]["delta"],
                       "reduced_gap": e4[key]["new_minus_old_reduced"]["delta"]})
        y += 1.0
    ax.axvline(0, color=AXIS, lw=0.7)
    ax.set_yticks(ypos, labels, fontsize=FS_S - 0.9)
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlabel("差值的变化（Recall@10）", fontsize=FS_S - 0.5)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    full = e4["e4_layer1"]["new_minus_old_full"]
    ax.text(0.99, 1.0, f"完整数据中新旧差 {full['delta']:+.3f} [{full['ci_low']:+.3f}, {full['ci_high']:+.3f}]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_S - 1.3, color=INK2)
    src["d"] = pd.DataFrame(rows_d)

    # ---------------------------------------------------- e: category-resolved difference of differences
    from .pubfigs8x import p_e4_heatmap
    label_mm(fig, 2, 162, "e", "E4 各臂在各疾病大类上的“新旧方法差”变化（删减 − 完整，Recall@10 × 100，种子 42；列按相似性聚类）")
    src["e"] = p_e4_heatmap(fig, 2, 168, 179, 50)
    return save(fig, "Fig8_evidence_sources", src)

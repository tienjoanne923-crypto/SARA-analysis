"""V6 manuscript figures (same contract as V5: 6.75 in, 600 dpi, greyscale, SimSun).

Every panel reads a result file written by the V6 modules; nothing is typed in
by hand.  Figure order follows section 17.3: data and reachable bound, zero-
support coverage, in-list scoring, support dose, evidence mechanism,
permutations, external / temporal / comparator budget.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from gpu8.figures import FIG_WIDTH_IN, GREY, bar, panel_label, save, setup_style, validate

from . import paths6
from .common import dump_json, load_manifest

T = paths6.TABLES_OUT
OUT = paths6.FIGURES


def jload(p):
    return json.loads(p.read_text(encoding="utf-8"))


def fig1_framework():
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.3))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    # full-width invisible frame so the tight bounding box keeps the 6.75 in width
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.0, 0.0), 10.0, 5.0, fc="none", ec="#ffffff", lw=0.1,
                           clip_on=False))
    boxes = [
        (0.1, 3.5, 2.2, 1.2, "可见信息审计\nHPO / 文本 / 本体 / 支持\n（只用可观测量分流）"),
        (2.7, 3.5, 2.3, 1.2, "零支持候选恢复\n表型·层级·本体·文本\n来源排除基因HPO·传播"),
        (5.4, 3.5, 2.1, 1.2, "固定500候选\n回退原因标记\n同预算比较"),
        (7.9, 3.5, 2.0, 1.2, "候选内评分\n调优融合 F\n/ LightGBM"),
        (2.7, 1.2, 2.3, 1.2, "支持驱动传播\nRWR（残差停止）\nPPI + 功能桥接"),
        (5.4, 1.2, 2.1, 1.2, "来源组种子\n等权 / 组权 / 打乱\n四层诊断"),
        (7.9, 1.2, 2.0, 1.2, "评价\nR = U·η 分解\n家族 bootstrap+Holm"),
        (0.1, 1.2, 2.2, 1.2, "置换检验\nP1 分支 / P2 评分\nP3 全管线"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc="#eeeeee",
                                    ec="#000000", lw=0.7))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)
    arr = dict(arrowstyle="->", lw=0.7, color="#000000")
    for (x0, y0), (x1, y1) in [((2.3, 4.1), (2.7, 4.1)), ((5.0, 4.1), (5.4, 4.1)),
                               ((7.5, 4.1), (7.9, 4.1)), ((6.45, 3.5), (3.85, 2.4)),
                               ((5.0, 1.8), (5.4, 1.8)), ((7.5, 1.8), (7.9, 1.8)),
                               ((8.9, 3.5), (8.9, 2.4))]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arr)
    ax.text(5.0, 0.35, "零支持任务 Z 与支持任务 S 分开评价；外部 / 时间任务 X 单列，不与 Z 合并",
            ha="center", fontsize=7.5)
    return save(fig, "fig1_v6_framework", OUT)


def fig2_bound():
    m = pd.read_csv(T / "z_main_endpoints.csv").set_index("method")
    cross = pd.read_csv(paths6.AUDIT / "crosstab_hpo_class_support.csv", index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 2.5))
    ax = axes[0]
    cols = [c for c in ["1", "2-5", ">=6"] if c in cross.columns]
    rows = [r for r in cross.index if r != "All"]
    names = {"valid_hpo_v5": "V5已有HPO", "mapping_failure_repaired": "严格映射修复",
             "no_resource_record": "资源无记录", "record_present_no_valid_symptom": "过滤后无症状项"}
    bottom = np.zeros(len(cols))
    for i, r in enumerate(rows):
        v = cross.loc[r, cols].to_numpy(dtype=float) / cross.loc["All", cols].to_numpy(dtype=float)
        ax.bar(cols, v, bottom=bottom, color=GREY[i + 1], edgecolor="#000000", lw=0.5,
               label=names.get(r, r))
        bottom += v
    ax.set_xlabel("已知阳性数分层")
    ax.set_ylabel("疾病比例")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    panel_label(ax, "a", "HPO 可用性")
    ax = axes[1]
    ms = ["v5_b3_cpu_fusion", "v6_F7", "v6_lightgbm"]
    lab = ["V5 融合", "V6 F7", "V6 LightGBM"]
    x = np.arange(3)
    for j, (col, nm) in enumerate([("oracle_recall10", "上界 U@10"), ("recall10_end", "Recall@10"),
                                   ("eta10", "条件效率 η")]):
        ax.bar(x + (j - 1) * 0.26, [m.loc[k, col] for k in ms], 0.26, color=GREY[j + 1],
               edgecolor="#000000", lw=0.5, label=nm)
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    panel_label(ax, "b", "R = U·η 分解")
    ax = axes[2]
    parts = [("share_not_retrieved", "未入池"), ("share_in_pool_missed", "入池未进前十"),
             ("share_recovered", "恢复")]
    bottom = np.zeros(3)
    for i, (c, nm) in enumerate(parts):
        v = np.array([m.loc[k, c] for k in ms])
        ax.bar(lab, v, bottom=bottom, color=GREY[i + 2], edgecolor="#000000", lw=0.5, label=nm)
        bottom += v
    ax.set_ylabel("疾病比例（非宏平均召回）")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    panel_label(ax, "c", "疾病结局")
    fig.tight_layout()
    return save(fig, "fig2_availability_bound", OUT)


def fig3_retrieval():
    fr = load_manifest("retriever_v6_frozen.json")
    lad = fr["ladder"]
    order = ["L0_v5_components", "L1_repaired_hpo", "L2_plus_onto", "L3_plus_text",
             "L4_plus_gene_hpo", "L5_plus_compat"]
    names = ["V5组件", "+HPO修复", "+本体", "+文本", "+基因HPO", "+兼容"]
    vals = [lad[k] for k in order] + [fr["dev_metrics"]]
    names = names + ["调优融合"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 2.6))
    ax = axes[0]
    d = np.array([v["coverage_direct_evidence"] for v in vals])
    s = np.array([v["coverage_any_signal"] for v in vals]) - d
    f = np.array([v["coverage_prior_fill"] for v in vals])
    x = np.arange(len(vals))
    ax.bar(x, d, color=GREY[1], edgecolor="#000000", lw=0.5, label="直接证据")
    ax.bar(x, s, bottom=d, color=GREY[3], edgecolor="#000000", lw=0.5, label="仅传播")
    ax.bar(x, f, bottom=d + s, color=GREY[5], edgecolor="#000000", lw=0.5, label="先验回退")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("开发集覆盖率（500）")
    ax.legend(fontsize=7.5, frameon=False)
    panel_label(ax, "a", "逐项接入（开发集）")
    ax = axes[1]
    rep = jload(paths6.RETRIEVAL / "source_report_test.json") if \
        (paths6.RETRIEVAL / "source_report_test.json").exists() else None
    if rep:
        src = [s_ for s_ in rep["sources"] if "coverage_contribution" in rep["sources"][s_]]
        single = [rep["sources"][s_]["single_source_coverage"] for s_ in src]
        contrib = [rep["sources"][s_]["coverage_contribution"] for s_ in src]
        xx = np.arange(len(src))
        ax.bar(xx - 0.2, single, 0.4, color=GREY[2], edgecolor="#000000", lw=0.5, label="单源覆盖")
        ax.bar(xx + 0.2, contrib, 0.4, color=GREY[4], edgecolor="#000000", lw=0.5, label="留一贡献")
        ax.set_xticks(xx)
        ax.set_xticklabels(src, rotation=25, ha="right")
        ax.legend(fontsize=7.5, frameon=False)
    panel_label(ax, "b", "测试折来源报告")
    fig.tight_layout()
    return save(fig, "fig3_retrieval", OUT)


def fig4_scoring():
    m = pd.read_csv(T / "z_main_endpoints.csv").set_index("method")
    st = pd.read_csv(T / "z_strata.csv")
    order = ["v5_b4_lightgbm", "v5_b8_meta_ridge", "v5_b9_meta_adapt", "v5_b3_cpu_fusion",
             "v6_F_hand", "v6_F4", "v6_lr", "v6_retrieval_order", "v6_F7", "v6_attention",
             "v6_bilinear", "v6_lightgbm"]
    lab = ["V5 LGB", "V5 元岭", "V5 适配", "V5 融合", "V6 手设", "V6 F4", "V6 LR", "V6 检索序",
           "V6 F7", "V6 注意力", "V6 双线性", "V6 LGB"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    bar(ax, lab, [m.loc[k, "recall10_end"] for k in order], ylabel="Recall@10（端到端）")
    panel_label(ax, "a", "五折 4,238 个疾病")
    ax = axes[1]
    sub = st[(st["stratum_type"] == "route") & st["stratum"].isin(["Z1_hpo", "Z2_no_hpo_text_or_ontology"])]
    ms = ["v5_b3_cpu_fusion", "v6_F7", "v6_lightgbm"]
    x = np.arange(2)
    for j, k in enumerate(ms):
        v = [sub[(sub["method"] == k) & (sub["stratum"] == s)]["recall10_end"].iloc[0]
             for s in ["Z1_hpo", "Z2_no_hpo_text_or_ontology"]]
        ax.bar(x + (j - 1) * 0.26, v, 0.26, color=GREY[j + 1], edgecolor="#000000", lw=0.5,
               label=["V5 融合", "V6 F7", "V6 LGB"][j])
    ax.set_xticks(x)
    ax.set_xticklabels(["有HPO", "无HPO"])
    ax.legend(fontsize=7.5, frameon=False)
    panel_label(ax, "b", "按可见信息分层")
    fig.tight_layout()
    return save(fig, "fig4_scoring", OUT)


def fig5_dose():
    c = pd.read_csv(T / "s_dose_curves.csv")
    s = jload(T / "s_summary.json")
    om = f"joint_w{s['selected']['omega']:g}"
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 2.6))
    ax = axes[0]
    sem = c[(c["method"] == "sem") & (c["regime"] == "S_fixed")]["recall10"].iloc[0]
    ks = [1, 3, 5]
    for j, (meth, nm, mk) in enumerate([("rwr_equal", "等权RWR", "o"), (om, "联合", "s"),
                                        ("rwr_group_eta1", "组权RWR", "^")]):
        v = [c[(c["method"] == meth) & (c["regime"] == "S_fixed") & (c["k"] == k)]["recall10"].iloc[0]
             for k in ks]
        ax.plot(ks, v, marker=mk, color=GREY[j], lw=0.9, ms=4, label=nm)
    ax.axhline(sem, color="#000000", ls="--", lw=0.7, label="零支持语义")
    ax.set_xlabel("显露支持数 k")
    ax.set_ylabel("Recall@10（固定候选）")
    ax.set_xticks(ks)
    ax.legend(fontsize=7.5, frameon=False)
    panel_label(ax, "a", "S-fixed 剂量曲线")
    ax = axes[1]
    for j, (reg, nm, mk) in enumerate([("S_fixed", "固定池 Recall@10", "o"),
                                       ("S_adaptive", "自适应池 Recall@10（=全背景）", "s")]):
        v = [c[(c["method"] == om) & (c["regime"] == reg) & (c["k"] == k)]["recall10"].iloc[0]
             for k in ks]
        ax.plot(ks, v, marker=mk, color=GREY[j * 2], lw=0.9, ms=4, label=nm)
    ax2 = ax.twinx()
    cv = [c[(c["method"] == om) & (c["regime"] == "S_adaptive") & (c["k"] == k)]["coverage"].iloc[0]
          for k in ks]
    ax2.plot(ks, cv, marker="^", color=GREY[3], lw=0.9, ms=4, ls="--", label="自适应池覆盖率")
    ax2.set_ylabel("覆盖率")
    ax2.grid(False)
    ax.set_xlabel("显露支持数 k")
    ax.set_xticks(ks)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper left")
    panel_label(ax, "b", "联合评分的候选口径")
    fig.tight_layout()
    return save(fig, "fig5_support_dose", OUT)


def fig6_mechanism():
    s = jload(T / "s_summary.json")
    lay = s["four_layer"]
    ks = ["k1", "k3", "k5"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 2.5))
    ax = axes[0]
    series = [("layer1_share_seed_changed", "种子改变"), ("layer3_share_top10_changed", "前十改变"),
              ("layer4_share_metric_changed", "指标改变")]
    x = np.arange(3)
    for j, (k_, nm) in enumerate(series):
        ax.bar(x + (j - 1) * 0.26, [lay[k][k_] for k in ks], 0.26, color=GREY[j + 1],
               edgecolor="#000000", lw=0.5, label=nm)
    ax.set_xticks(x)
    ax.set_xticklabels(["k=1", "k=3", "k=5"])
    ax.set_ylabel("episode 比例")
    ax.legend(fontsize=7.5, frameon=False)
    panel_label(ax, "a", "四层诊断")
    ax = axes[1]
    for j, (key, nm) in enumerate([("Delta_seed", "Δseed 组权-等权"), ("Gamma_grp", "Γgrp 真组-打乱")]):
        v = [s["secondary"][f"{key}_{k}"] for k in ks]
        y = [a["delta"] for a in v]
        lo = [a["delta"] - a["ci_low"] for a in v]
        hi = [a["ci_high"] - a["delta"] for a in v]
        ax.errorbar(np.arange(3) + (j - 0.5) * 0.2, y, yerr=[lo, hi], fmt="os"[j], color=GREY[j * 2],
                    capsize=2.5, lw=0.8, ms=4, label=nm)
    ax.axhline(0, color="#000000", lw=0.6)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["k=1", "k=3", "k=5"])
    ax.set_ylabel("Recall@10 差（95% CI）")
    ax.legend(fontsize=7.5, frameon=False)
    panel_label(ax, "b", "来源组权重的效应")
    fig.tight_layout()
    return save(fig, "fig6_evidence_mechanism", OUT)


def fig7_permutation():
    v5 = jload(paths6.PERMUTATION / "v5_b9_permutations_f0.json")["summary"]
    v6p = paths6.PERMUTATION / "v6_permutations_summary.csv"
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 2.5))
    ax = axes[0]
    tests = ["P1", "P2", "P3"]
    for j, sch in enumerate(["distant", "close"]):
        v = [next(r["delta"] for r in v5 if r["test"] == t and r["scheme"] == sch) for t in tests]
        ax.bar(np.arange(3) + (j - 0.5) * 0.35, v, 0.35, color=GREY[j * 2 + 1], edgecolor="#000000",
               lw=0.5, label="远供体" if sch == "distant" else "近供体")
    ax.axhline(0, color="#000000", lw=0.6)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(tests)
    ax.set_ylabel("ΔRecall@10（置换-原始）")
    ax.legend(fontsize=7.5, frameon=False)
    panel_label(ax, "a", "V5 b9，k=0，第0折")
    ax = axes[1]
    if v6p.exists():
        d = pd.read_csv(v6p)
        d = d[(d["scheme"] == "distant") & (d["subset"] == "all")]
        sc = [s_ for s_ in ["F7", "lightgbm", "attention_residual"] if s_ in set(d["scorer"])]
        for j, t in enumerate(["P1", "P2", "P3"]):
            v = [d[(d["scorer"] == s_) & (d["test"] == t)]["delta_perm_minus_orig"].iloc[0]
                 if len(d[(d["scorer"] == s_) & (d["test"] == t)]) else np.nan for s_ in sc]
            ax.bar(np.arange(len(sc)) + (j - 1) * 0.27, v, 0.27, color=GREY[j * 2 + 1],
                   edgecolor="#000000", lw=0.5, label=t + ("（仅注意力残差有疾病分支）" if t == "P1" else ""))
        ax.set_xticks(np.arange(len(sc)))
        ax.set_xticklabels(["F7", "LightGBM", "注意力残差"][:len(sc)])
        ax.axhline(0, color="#000000", lw=0.6)
        ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    panel_label(ax, "b", "V6 评分器，五折")
    fig.tight_layout()
    return save(fig, "fig7_permutations", OUT)


def fig8_external():
    ext = pd.read_csv(paths6.EXTERNAL_RES / "external_v6_layers.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 2.5))
    ax = axes[0]
    e = ext[ext["scorer"] == "F_main"].reset_index(drop=True)
    names = {"gencc_definitive_strong": "GenCC 确定/强", "gencc_moderate_sensitivity": "GenCC 中等",
             "clinvar_2star": "ClinVar ≥2星", "clinvar_citation_verified_independent": "ClinVar 引文独立"}
    y = e["recall10_end"].to_numpy()
    ax.errorbar(np.arange(len(e)), y, yerr=[y - e["ci_low"], e["ci_high"] - y], fmt="o",
                color="#000000", capsize=2.5, ms=4, lw=0.8)
    ax.set_xticks(np.arange(len(e)))
    ax.set_xticklabels([f"{names.get(l, l)}\n(n={n})" for l, n in zip(e["layer"], e["n_diseases"])],
                       fontsize=7.5)
    ax.set_ylabel("Recall@10（95% CI）")
    panel_label(ax, "a", "外部标签层")
    ax = axes[1]
    tp = paths6.EXTERNAL_RES / "temporal_x2_support_summary.json"
    if tp.exists():
        t = jload(tp)
        meths = pd.DataFrame(t["methods"]).set_index("method")
        order = [m_ for m_ in ["degree_prior_2023", "rwr_equal_ppi", f"rwr_equal_mix{t['nu']:g}"]
                 if m_ in meths.index]
        bar(ax, ["2023度先验", "RWR PPI", "RWR 桥接"][:len(order)],
            [meths.loc[m_, "recall50"] for m_ in order], ylabel="Recall@50")
    panel_label(ax, "b", "时间任务 X2")
    ax = axes[2]
    kp = paths6.SCORING / "kdgene_coldval_f0.json"
    if kp.exists():
        k = jload(kp)
        h = pd.DataFrame(k["history"])
        ax.plot(h["epoch"], h["val_recall10"], color="#000000", lw=0.9, label="冷验证 Recall@10")
        ax2 = ax.twinx()
        ax2.plot(h["epoch"], h["loss"], color=GREY[3], lw=0.9, ls="--", label="训练损失")
        ax2.set_ylabel("训练损失")
        ax.axvline(k["best_epoch"], color=GREY[2], lw=0.6, ls=":")
        ax.set_xlabel("epoch")
        ax.set_ylabel("冷验证 Recall@10")
    panel_label(ax, "c", "KDGene 学习曲线")
    fig.tight_layout()
    return save(fig, "fig8_external_temporal_kdgene", OUT)


def main() -> int:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for fn in (fig1_framework, fig2_bound, fig3_retrieval, fig4_scoring, fig5_dose,
               fig6_mechanism, fig7_permutation, fig8_external):
        try:
            recs.append(fn())
        except Exception as exc:
            recs.append({"stem": fn.__name__, "error": f"{type(exc).__name__}: {exc}"})
    ok = [r for r in recs if "error" not in r]
    problems = validate(None, ok)
    dump_json(paths6.REPORTS / "figures_v6_report.json", {"figures": recs, "problems": problems})
    print(json.dumps({"figures": [r.get("stem") for r in recs],
                      "errors": [r for r in recs if "error" in r], "problems": problems},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

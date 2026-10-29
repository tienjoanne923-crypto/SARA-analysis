"""V7.1 manuscript figures (same contract as V5/V6: 6.75 in, 600 dpi, greyscale, SimSun).

Every panel reads a V7.1 result file; nothing is typed in by hand.

Fig. 9   V7.1 framework: typed multimodal core, OOF chain, late fusion, checks
Fig. 10  modality coverage and development selection
Fig. 11  E1 main comparison (Recall@10 / Hit@10 / final-500 coverage)
Fig. 12  E3 answer-visibility strata and E6 hit flow
Fig. 13  E5 behaviour: shutdown and paired permutations (core alone and late fusion)
Fig. 14  E7 new confirmation set, E2/E4 same-data and resource x method
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from gpu8.figures import FIG_WIDTH_IN, GREY, panel_label, save, setup_style

from . import paths71
from .common71 import dump_json, load_manifest

T = paths71.TABLES_OUT
OUT = paths71.FIGURES

SHORT = {"F7 (V6 frozen fusion)": "F7",
         "V6 LightGBM (archived V6 main)": "V6 LGB",
         "Channel-only LightGBM (strong control)": "通道LGB",
         "Same-resource fixed fusion": "同资源固定",
         "Multimodal core alone": "核心单独",
         "Late fusion, fixed": "后融合-固定",
         "Late fusion, LightGBM": "后融合-LGB"}


def jload(p):
    return json.loads(p.read_text(encoding="utf-8"))


def fig9_framework():
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.4))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.add_patch(Rectangle((0.0, 0.0), 10.0, 5.0, fc="none", ec="#ffffff", lw=0.1, clip_on=False))
    boxes = [
        (0.1, 3.55, 2.25, 1.25, "疾病侧（冻结 SapBERT）\n掩蔽文本 / HPO 词条\n组织别名表 → q_d"),
        (0.1, 1.95, 2.25, 1.25, "基因侧\nESM-2 35M 序列\nGO 功能均值 / HPA 表达"),
        (2.75, 2.75, 2.3, 1.3, "有类型核心\nP_T 共享语义投影\nP_S 序列适配 · s_E 组织匹配\n门控 w_m（仅可用性）"),
        (2.75, 0.95, 2.3, 1.2, "训练\n功能锚 L_A（100步）\n多正例排序 L_DG\nF_{j+1}/开发集早停"),
        (5.45, 2.75, 2.1, 1.3, "独立检索\n全背景 ~2万基因\n核心前500\n有效位 valid"),
        (5.45, 0.95, 2.1, 1.2, "袋外整链\n3 内块 + 1 外层\n×5折×3种子\n= 60 次拟合"),
        (7.95, 2.75, 1.95, 1.3, "后融合\nV6 500 ∪ 核心500\n固定融合 / LGB\n一次排序"),
        (7.95, 0.95, 1.95, 1.2, "验收五层\n数据·计算·行为\n配对置换·预测\nE1–E8"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc="#eeeeee",
                                    ec="#000000", lw=0.7))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.2)
    arr = dict(arrowstyle="->", lw=0.7, color="#000000")
    for (x0, y0), (x1, y1) in [((2.35, 4.15), (2.75, 3.6)), ((2.35, 2.55), (2.75, 3.1)),
                               ((3.9, 2.75), (3.9, 2.15)), ((5.05, 3.4), (5.45, 3.4)),
                               ((6.5, 2.75), (6.5, 2.15)), ((7.55, 3.4), (7.95, 3.4)),
                               ((8.9, 2.75), (8.9, 2.15))]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arr)
    ax.text(5.0, 0.35, "核心先独立产生候选，再以袋外方式与 V6 后融合；图像与小鼠表型为有条件扩展（E8）",
            ha="center", fontsize=7.5)
    return save(fig, "fig9_v71_framework", OUT)


def fig10_coverage_dev():
    cov = jload(paths71.AUDIT / "featurepack_coverage_v71.json")
    sel = load_manifest("core_dev_selection_v71.json")
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 2.5))
    ax = axes[0]
    names = {"sequence": "序列", "function": "功能", "expression": "表达", "image": "图像"}
    ks = [k for k in names if k in cov["genes"]]
    x = np.arange(len(ks))
    ax.bar(x - 0.18, [cov["genes"][k]["share_all"] for k in ks], 0.36, color=GREY[1],
           edgecolor="#000000", lw=0.5, label="全部基因")
    ax.bar(x + 0.18, [cov["genes"][k]["share_positive_genes"] for k in ks], 0.36, color=GREY[3],
           edgecolor="#000000", lw=0.5, label="阳性基因")
    ax.set_xticks(x, [names[k] for k in ks])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("覆盖比例")
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    panel_label(ax, "A")
    ax = axes[1]
    dn = {"d_text": "文本", "d_hpo": "HPO", "d_tissue": "组织"}
    v = [cov["diseases"][k]["share_labelled"] for k in dn]
    ax.bar(list(dn.values()), v, color=GREY[2], edgecolor="#000000", lw=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("有标注疾病覆盖")
    panel_label(ax, "B")
    ax = axes[2]
    g = pd.DataFrame(sel["grid"])
    lab = [f"{int(r.rank)}/{r.lambda_A:g}/{'门' if r.gate == 'small' else '全'}" for r in g.itertuples()]
    col = [GREY[4] if (r.rank == sel["selected"]["rank"] and r.lambda_A == sel["selected"]["lambda_A"]
                       and r.gate == sel["selected"]["gate"]) else GREY[1] for r in g.itertuples()]
    ax.barh(np.arange(len(g)), g["dev_recall10"], color=col, edgecolor="#000000", lw=0.5)
    ax.set_yticks(np.arange(len(g)), lab, fontsize=6)
    ax.set_xlabel("开发集 Recall@10")
    ax.invert_yaxis()
    panel_label(ax, "C")
    fig.tight_layout()
    return save(fig, "fig10_v71_coverage_dev", OUT)


def fig11_main():
    t = pd.read_csv(T / "e1_main.csv")
    t = t[t["method"].isin(SHORT)]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 2.6))
    for ax, col, lab, pl in zip(axes, ["recall10_end", "hit10", "coverage_final500"],
                                ["Recall@10", "Hit@10", "最终500覆盖"], "ABC"):
        v = t[col].to_numpy(dtype=float)
        ax.bar(np.arange(len(t)), v, color=[GREY[i % 5] for i in range(len(t))],
               edgecolor="#000000", lw=0.5)
        ax.set_xticks(np.arange(len(t)), [SHORT[m] for m in t["method"]], rotation=60, fontsize=6,
                      ha="right")
        ax.set_ylabel(lab)
        for i, x in enumerate(v):
            ax.text(i, x, f"{x:.3f}", ha="center", va="bottom", fontsize=5)
        panel_label(ax, pl)
    fig.tight_layout()
    return save(fig, "fig11_v71_e1_main", OUT)


def fig12_cold_flow():
    t = pd.read_csv(T / "e3_visibility.csv")
    flow = jload(T / "e6_hit_flow.json")
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 2.6))
    ax = axes[0]
    cls = ["visible_ctd", "visible_hpo_only", "ppi_only", "unseen"]
    cname = {"visible_ctd": "参考可见", "visible_hpo_only": "仅HPO", "ppi_only": "仅PPI", "unseen": "无路径"}
    meths = [m for m in ["V6 LightGBM", "Multimodal core alone", "Late fusion LightGBM"] if m in set(t["method"])]
    mname = {"V6 LightGBM": "V6 LGB", "Multimodal core alone": "核心单独", "Late fusion LightGBM": "后融合"}
    w = 0.8 / max(len(meths), 1)
    for k, m in enumerate(meths):
        sub = t[t["method"] == m].set_index("class").reindex(cls)
        ax.bar(np.arange(len(cls)) + k * w - 0.4 + w / 2, sub["top10"].to_numpy(dtype=float), w,
               color=GREY[k + 1], edgecolor="#000000", lw=0.5, label=mname[m])
    ax.set_xticks(np.arange(len(cls)), [cname[c] for c in cls], fontsize=7)
    ax.set_ylabel("阳性进入前10的比例")
    ax.legend(fontsize=6, frameon=False)
    panel_label(ax, "A")
    ax = axes[1]
    keys = [("lf_vs_v6_lightgbm", "对 V6 LGB"), ("lf_vs_channel_lgb", "对通道LGB")]
    x = np.arange(len(keys))
    new = [flow[k].get("new_hits_total_per_seed", 0) for k, _ in keys]
    lost = [flow[k].get("lost_hits_total_per_seed", 0) for k, _ in keys]
    ax.bar(x - 0.18, new, 0.36, color=GREY[3], edgecolor="#000000", lw=0.5, label="新增命中")
    ax.bar(x + 0.18, lost, 0.36, color=GREY[1], edgecolor="#000000", lw=0.5, label="被挤出命中")
    ax.set_xticks(x, [n for _, n in keys], fontsize=7)
    ax.set_ylabel("每种子命中数（4,238 病）")
    ax.set_ylim(0, max(new + lost) * 1.3)
    ax.legend(fontsize=6, frameon=False, loc="upper center", ncol=2)
    panel_label(ax, "B")
    fig.tight_layout()
    return save(fig, "fig12_v71_cold_flow", OUT)


def fig13_behaviour():
    e = pd.read_csv(T / "e5_behaviour.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 2.5))
    mods = ["S", "F", "E"]
    mn = {"S": "序列", "F": "功能", "E": "组织表达"}
    for ax, col, lab, pl in zip(axes, ["core_drop", "lf_drop"], ["核心单独 Recall@10 下降",
                                                                  "后融合 Recall@10 下降"], "AB"):
        x = np.arange(len(mods))
        for k, (pert, name) in enumerate([("shutdown", "关闭"), ("paired_permutation", "配对置换")]):
            sub = e[e["perturbation"] == pert].set_index("modality").reindex(mods)
            ax.bar(x + (k - 0.5) * 0.36, sub[col].to_numpy(dtype=float), 0.36, color=GREY[2 * k + 1],
                   edgecolor="#000000", lw=0.5, label=name)
        ax.axhline(0, color="#000000", lw=0.5)
        ax.set_xticks(x, [mn[m] for m in mods])
        ax.set_ylabel(lab)
        ax.legend(fontsize=6, frameon=False)
        panel_label(ax, pl)
    fig.tight_layout()
    return save(fig, "fig13_v71_behaviour", OUT)


def fig14_confirm_e2e4():
    c = pd.read_csv(T / "e7_confirmation.csv")
    e2 = pd.read_csv(T / "e2_same_data.csv")
    e4 = pd.read_csv(T / "e4_resource_method.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 2.6))
    ax = axes[0]
    c = c[c["method"].isin(SHORT)]
    ax.bar(np.arange(len(c)), c["recall10"], color=GREY[2], edgecolor="#000000", lw=0.5)
    ax.set_xticks(np.arange(len(c)), [SHORT[m] for m in c["method"]], rotation=60, fontsize=6, ha="right")
    ax.set_ylabel("新确认集 Recall@10（过滤）")
    panel_label(ax, "A")
    ax = axes[1]
    e2 = e2.dropna(subset=["recall10_end"]) if "recall10_end" in e2.columns else e2
    nm = {"core (full alignment) alone": "核心", "core, lambda_A flipped, alone": "锚翻转",
          "linear two-tower alone": "双塔", "same-resource fixed fusion": "同资源",
          "late fusion LGB, full core": "LF核心", "late fusion LGB, lambda_A flipped": "LF锚翻转",
          "late fusion LGB, two-tower": "LF双塔", "raw concatenation LGB (same pool)": "原始拼接"}
    ax.bar(np.arange(len(e2)), e2["recall10_end"], color=GREY[3], edgecolor="#000000", lw=0.5)
    ax.set_xticks(np.arange(len(e2)), [nm.get(m, m) for m in e2["method"]], rotation=60, fontsize=6,
                  ha="right")
    ax.set_ylabel("Recall@10（种子42）")
    panel_label(ax, "B")
    ax = axes[2]
    if "recall10_end" in e4.columns:
        M = np.full((2, 2), np.nan)
        for r in e4.itertuples():
            res, meth = [s.strip() for s in r.method.split("/")]
            M[0 if res.startswith("old") else 1, 0 if meth.startswith("old") else 1] = r.recall10_end
        im = ax.imshow(M, cmap="Greys", vmin=np.nanmin(M) * 0.9, vmax=np.nanmax(M) * 1.05)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if M[i, j] > np.nanmean(M) else "#000000")
        ax.set_xticks([0, 1], ["旧方法", "新方法"], fontsize=7)
        ax.set_yticks([0, 1], ["旧资源", "新资源"], fontsize=7)
    panel_label(ax, "C")
    fig.tight_layout()
    return save(fig, "fig14_v71_confirm_e2e4", OUT)


def main() -> int:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    made = {}
    for fn in (fig9_framework, fig10_coverage_dev, fig11_main, fig12_cold_flow, fig13_behaviour,
               fig14_confirm_e2e4):
        try:
            made[fn.__name__] = str(fn())
        except Exception as exc:  # recorded, never silently hidden
            made[fn.__name__] = f"FAILED: {type(exc).__name__}: {exc}"
    dump_json(OUT / "figures_v71_index.json", made)
    print(json.dumps(made, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

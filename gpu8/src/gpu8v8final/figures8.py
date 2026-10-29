"""V8.1 figures (six main figures + supplementary), drawn from the saved result files only."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from . import paths8

FIG = paths8.FIGURES
plt.rcParams.update({"font.size": 9, "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 150})
C = {"new": "#1b6ca8", "old": "#9aa5b1", "cold": "#d1495b", "acc": "#edae49", "grey": "#5c6770", "green": "#2e933c"}


def _box(ax, x, y, w, h, text, fc="#eef3f8", ec="#1b6ca8", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1))


def fig1_problem():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.text(0.1, 4.7, "A  三类“冷”与信息边界", fontsize=10, weight="bold")
    rows = [("参考路径冷", "无参考边、无过滤后直接基因 HPO", "可能仍有 GO、序列、表达、PPI"),
            ("GO 输入缺失", "预定基因的 GO 及其派生量", "仍可参与关联训练、可有 HPO"),
            ("真正关联监督冷", "测试基因的疾病关联监督", "序列、正常表达等明示属性")]
    ax.text(2.2, 4.2, "删除的内容", ha="center", weight="bold")
    ax.text(6.4, 4.2, "允许保留", ha="center", weight="bold")
    for i, (a, b, c) in enumerate(rows):
        y = 3.4 - i * 0.9
        _box(ax, 0.1, y, 1.4, 0.65, a, fc="#fbeaec", ec=C["cold"])
        _box(ax, 1.7, y, 2.9, 0.65, b)
        _box(ax, 4.8, y, 3.3, 0.65, c, fc="#eef6ee", ec=C["green"])
    ax.text(8.3, 3.7, "生产输入\n不读查询标签", fontsize=8, ha="left", color=C["grey"])
    ax.text(8.3, 2.6, "角色名单排除\n($F_j$, $F_{j+1}$,\n开发, 外部, $I_{jb}$)", fontsize=8, ha="left", color=C["grey"])
    ax.text(0.1, 0.35, "三类冷分表报告，不合并；GO、文本、HPO 的不同文件来源不计为多个独立原始模态。", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_problem_information_layers.png", dpi=200)
    plt.close(fig)


def fig2_algorithm():
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    _box(ax, 0.1, 4.6, 2.2, 0.9, "V6 七个检索通道\n完整背景名次 $r_B$")
    _box(ax, 0.1, 3.3, 2.2, 0.9, "语义教师 " + r"$s_F=z_d^{\top}z^F_g$" + "\n(真实 GO 优先; 学生补位)")
    _box(ax, 2.9, 3.9, 2.4, 1.1, "缺失感知 RRF\n" + r"$R=\frac{wb+(1-w)me}{w+(1-w)m}$" + "\n恰好 500 个候选")
    _box(ax, 5.9, 4.7, 1.9, 0.8, "基础排序器 $B$\n(30 通道, OOF $B^{-b}$)")
    _box(ax, 5.9, 3.6, 1.9, 0.8, "门控纠偏\n" + r"$s=B+a\,r_\theta(x)$")
    _box(ax, 5.9, 2.5, 1.9, 0.8, "候选 A\n直接 LightGBM (NaN)")
    _box(ax, 8.2, 3.6, 1.7, 0.8, "每上下文\nA/B 部署规则", fc="#fdf3e1", ec=C["acc"])
    for (x0, y0, x1, y1) in ((2.3, 5.0, 2.9, 4.6), (2.3, 3.7, 2.9, 4.2), (5.3, 4.5, 5.9, 5.0),
                             (5.3, 4.4, 5.9, 4.0), (5.3, 4.2, 5.9, 2.9), (7.8, 4.0, 8.2, 4.0), (7.8, 2.9, 8.2, 3.8)):
        _arrow(ax, x0, y0, x1, y1)
    ax.text(0.1, 1.9, "整链 OOF：每折 3 个内上下文 + 1 个外上下文", weight="bold", fontsize=9)
    for b in range(3):
        _box(ax, 0.1 + b * 1.7, 0.7, 1.5, 0.9, f"内上下文 {b}\n训练 $T_j\\setminus I_{{jb}}$\n给 $I_{{jb}}$ 出特征", fc="#f4f4f4", ec=C["grey"],
             fs=7)
    _box(ax, 5.3, 0.7, 1.9, 0.9, "外上下文\n训练 $T_j$\n给 $F_j$ 打分", fc="#f4f4f4", ec=C["grey"], fs=7)
    _box(ax, 7.5, 0.7, 2.4, 0.9, "检查点：同上下文 $F_{j+1}$\n配置：开发集（每外层独立）", fc="#fdf3e1", ec=C["acc"], fs=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_algorithm_oof.png", dpi=200)
    plt.close(fig)


def fig3_main(res: dict):
    t = pd.DataFrame(res["main"]["table"])
    order = ["deployed", "v71_direct_strict", "B", "channel_lgb_v6", "v71_direct_v6", "F7_v6", "standalone_r",
             "standalone_sF", "standalone_sF0"]
    t = t[t["method"].isin(order)].set_index("method").reindex(order).dropna(how="all").reset_index()
    labels = {"deployed": "V8.1 部署(严格500)", "v71_direct_strict": "V7.1 直接融合(严格500)", "B": "基础排序器 B",
              "channel_lgb_v6": "仅通道 LGB(V6 500)", "v71_direct_v6": "V7.1 直接融合(V6 500)", "F7_v6": "F7(V6 500)",
              "standalone_r": "V7.1 核心 $r$ 单独", "standalone_sF": "教师 $s_F$ 单独", "standalone_sF0": "无学习 SapBERT"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    y = np.arange(len(t))
    axes[0].barh(y, t["recall10"], color=[C["new"] if m in ("deployed",) else C["old"] for m in t["method"]])
    axes[0].set_yticks(y, [labels.get(m, m) for m in t["method"]])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("端到端 Recall@10（疾病宏平均）")
    for i, v in enumerate(t["recall10"]):
        axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7)
    axes[1].barh(y, t["cold_recall10"], color=C["cold"])
    axes[1].set_yticks(y, [""] * len(t))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("冷恢复 $R^c$@10")
    for i, v in enumerate(t["cold_recall10"]):
        axes[1].text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_strict500_overall_cold.png", dpi=200)
    plt.close(fig)


def fig4_missing(res: dict):
    v = res.get("variants", {})
    arms = [("e5_none", "无补位"), ("e5_mlp", "两层学生"), ("e5_linear", "线性学生"), ("e5_nn", "序列近邻转移"),
            ("e5_teacher_sS", "教师 $s_S$"), ("e5_mlp_shuffled", "配对打乱"), ("e5_protst", "ProtST 输入(开放知识)")]
    rows = [(lab, v[a]["deployed"].get("mask_test_pool_micro"), v[a]["deployed"].get("mask_test_hit10_micro"))
            for a, lab in arms if a in v and "deployed" in v[a]]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    if rows:
        lab = [r[0] for r in rows]
        axes[0].bar(range(len(rows)), [r[1] or 0 for r in rows], color=C["new"])
        axes[0].set_xticks(range(len(rows)), lab, rotation=40, ha="right")
        axes[0].set_ylabel("屏蔽基因阳性进入 500 的比例")
        axes[1].bar(range(len(rows)), [r[2] or 0 for r in rows], color=C["acc"])
        axes[1].set_xticks(range(len(rows)), lab, rotation=40, ha="right")
        axes[1].set_ylabel("屏蔽基因阳性进入前 10 的比例")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_go_mask_sequence.png", dpi=200)
    plt.close(fig)


def fig5_sources(res: dict):
    v = res.get("variants", {})
    arms = [("e4_layer1", "训练可见来源排除"), ("e4_layer1_random", "  匹配随机"), ("e4_layer2", "评价条件化删除*"),
            ("e4_layer2_random", "  匹配随机*"), ("e4_experimental", "仅实验 GO"), ("e4_no_tas_nas", "去 TAS/NAS"),
            ("e4_no_mutant", "去突变/遗传相互作用"), ("e4_blocks", "证据块隔离"), ("e4_blocks_random", "  匹配随机")]
    rows = []
    e4 = res.get("e4", {})
    arms = arms + [("e4_rescore_no_ic", "去 IC（重打分）"), ("e4_rescore_iea_only", "仅 IEA（重打分）")]
    for a, lab in arms:
        if a in e4:
            dd = e4[a]["difference"]
            rows.append((lab, dd.get("delta"), dd.get("ci_low"), dd.get("ci_high")))
        elif a in v and "deployed" in v[a] and isinstance(v[a]["deployed"].get("delta_vs_main"), dict):
            dd = v[a]["deployed"]["delta_vs_main"]
            rows.append((lab, dd.get("delta"), dd.get("ci_low"), dd.get("ci_high")))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    if rows:
        y = np.arange(len(rows))
        axes[0].errorbar([r[1] for r in rows], y, xerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]],
                         fmt="o", color=C["new"], ecolor=C["grey"], capsize=2)
        axes[0].axvline(0, color="k", lw=0.6)
        axes[0].set_yticks(y, [r[0] for r in rows])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("新旧方法差的变化（删减数据 − 完整数据）")
    hist = res.get("historical", {}).get("results", {})
    ext = res.get("external", {})
    bars = []
    for k, lab in (("x2__deployed", "X2 部署"), ("x2__channel_lgb_v6", "X2 仅通道"), ("x1__deployed", "X1 部署"),
                   ("x1__channel_lgb_v6", "X1 仅通道")):
        if k in hist and hist[k].get("recall10") is not None:
            bars.append((lab, hist[k]["recall10"]))
    for r in ext.get("table", []):
        if r["subset"] == "union" and r["method"] in ("deployed", "channel_lgb_v6"):
            bars.append((f"618 {'部署' if r['method'] == 'deployed' else '仅通道'}", r["recall10"]))
    if bars:
        axes[1].barh(range(len(bars)), [b[1] for b in bars], color=[C["new"] if "部署" in b[0] else C["old"] for b in bars])
        axes[1].set_yticks(range(len(bars)), [b[0] for b in bars])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Recall@10")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_sources_time_external.png", dpi=200)
    plt.close(fig)


def fig6_flow_resources(res: dict, flow: dict | None, ledger: pd.DataFrame | None):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    if flow:
        keys = ["kept", "lost", "gained"]
        lab = ["保留", "失去", "新增"]
        axes[0].bar(lab, [flow.get(k, 0) for k in keys], color=[C["old"], C["cold"], C["new"]])
        axes[0].bar(lab, [flow.get(k + "_cold", 0) for k in keys], color="k", alpha=0.35, label="其中冷基因")
        axes[0].set_ylabel("前 10 命中数（相对仅通道 LGB，种子 42）")
        axes[0].legend(fontsize=7)
    if ledger is not None and not ledger.empty:
        names = {"teacher": "语义教师", "student": "序列学生", "base_ranker": "基础排序器 B", "corrector": "门控纠偏",
                 "candidate_A": "候选 A", "kdgene": "KDGene", "sign": "SIGN 式", "fold": "折级汇总"}
        g = ledger.groupby("component").agg(n=("run_id", "count"), sec=("seconds", "sum"))
        g = g[g.index.isin(names)].sort_values("sec", ascending=False)
        axes[1].barh([names[c] for c in g.index], g["sec"] / 3600.0, color=C["acc"])
        for i, (c, n) in enumerate(zip(g["sec"], g["n"])):
            axes[1].text(c / 3600.0, i, f" {n} 次", va="center", fontsize=7)
        axes[1].set_xlabel("有计时的机时（小时）")
        axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "fig6_hitflow_resources.png", dpi=200)
    plt.close(fig)


def supp_dev():
    rows = []
    for j in range(5):
        s = json.loads((paths8.DEV / f"fold{j}" / "selection.json").read_text(encoding="utf-8"))
        for k, v in s["w"]["coverage"].items():
            rows.append({"fold": j, "setting": k, "coverage": v})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for j, sub in df.groupby("fold"):
        aw = sub[sub["setting"].str.startswith("aware")]
        ax.plot([float(s.split("_w")[1]) for s in aw["setting"]], aw["coverage"], marker="o", label=f"外层 {j}")
    v6 = df[df["setting"] == "v6_500"]["coverage"].mean()
    ax.axhline(v6, ls="--", color=C["grey"], label="V6 500（均值）")
    ax.set_xlabel("RRF 权重 w")
    ax.set_ylabel("开发集覆盖@500")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "figS1_dev_rrf_w.png", dpi=200)
    plt.close(fig)


def build(res: dict, flow: dict | None = None, ledger: pd.DataFrame | None = None) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_problem()
    fig2_algorithm()
    if res.get("main"):
        fig3_main(res)
    fig4_missing(res)
    fig5_sources(res)
    fig6_flow_resources(res, flow, ledger)
    supp_dev()
    return sorted(p.name for p in FIG.glob("*.png"))

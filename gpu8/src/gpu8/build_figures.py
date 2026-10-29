"""Build the eight main figures and the supplementary panels.

Figure 1 data sources, evidence de-duplication, splits and the model path
Figure 2 five-fold main results and disease-level paired differences
Figure 3 fixed support dose, Gamma, benefit / harm proportions
Figure 4 natural source dependence, replacement support, update magnitude
Figure 5 candidate coverage, list size, review burden
Figure 6 external and temporal results with their denominators
Figure 7 family, low coverage, network and negative controls
Figure 8 single-machine resources and the independent biological support
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import paths
from .config import load_config
from .figures import (DPI, FIG_WIDTH_IN, GREY, bar, panel_label, read_json_any,
                      read_table, save, setup_style, validate)
from .hashing import atomic_write_json, read_json


def _grid(nrows, ncols, height):
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH_IN, height))
    return fig, np.atleast_1d(axes).ravel()


# --------------------------------------------------------------------------
def figure01(out_dir: Path) -> dict:
    build = read_json_any("build_report.json")
    proto = read_json_any("protocol_report.json")
    mapping = read_json_any("disease_mapping_report.json")
    feas = read_table("support_feasibility.csv")

    fig, ax = _grid(2, 2, 5.4)
    counts = build.get("counts", {})
    order = ["diseases", "genes", "relations", "evidence", "phenotypes", "hierarchy", "ppi"]
    labels = ["疾病", "基因", "唯一关系", "证据行", "表型注释", "层级边", "PPI 边"]
    vals = [counts.get(k, 0) for k in order]
    bar(ax[0], labels, vals, ylabel="记录数")
    ax[0].set_yscale("log")
    panel_label(ax[0], "a", "九张规范表的规模")

    sizes = proto.get("fold_sizes", {})
    keys = ["开发集", "外部保留"] + list(sizes)
    vals = [proto.get("n_dev", 0), proto.get("n_external", 0)] + list(sizes.values())
    bar(ax[1], keys, vals, ylabel="疾病数")
    panel_label(ax[1], "b", "疾病家族级划分")

    if feas is not None and not feas.empty:
        vc = feas["n_unique_positives"].clip(upper=11).value_counts().sort_index()
        xs = [str(int(i)) if i < 11 else "≥11" for i in vc.index]
        bar(ax[2], xs, vc.values, ylabel="疾病数")
        ax[2].set_yscale("log")
        ax[2].set_xlabel("已知阳性基因数")
        panel_label(ax[2], "c", "阳性负担分布")

        se = proto.get("support_eligibility", {})
        ks = ["zero-shot", "1-shot", "3-shot", "5-shot"]
        vs = [se.get("diseases", 0), se.get("eligible_1shot", 0),
              se.get("eligible_3shot", 0), se.get("eligible_5shot", 0)]
        bar(ax[3], ks, vs, ylabel="合格疾病数")
        panel_label(ax[3], "d", "支持剂量资格盘点")
    fig.tight_layout()
    return save(fig, "main_figure_01_data_and_protocol", out_dir)


def figure02(out_dir: Path) -> dict:
    t3 = read_table("table3_main_endpoints.csv")
    comps = read_table("primary_comparisons.csv")
    per = read_table("per_disease_all.csv")

    fig, ax = _grid(2, 2, 5.4)
    if t3 is not None and not t3.empty:
        d = t3.sort_values("recall10_end")
        bar(ax[0], d["label"].tolist(), d["recall10_end"].tolist(),
            errs=d.get("recall10_end_sd_fold"), ylabel="端到端 Recall@10")
        panel_label(ax[0], "a", "五折主终点")

        bar(ax[1], d["label"].tolist(), d["recall10_cond"].tolist(),
            ylabel="条件 Recall@10")
        panel_label(ax[1], "b", "候选内条件指标")
    if comps is not None and not comps.empty:
        y = np.arange(len(comps))
        ax[2].errorbar(comps["delta"], y,
                       xerr=[comps["delta"] - comps["ci_low"],
                             comps["ci_high"] - comps["delta"]],
                       fmt="o", color="#000000", ecolor="#3c3c3c",
                       capsize=2.5, markersize=3.5, elinewidth=0.7)
        ax[2].axvline(0, color="#707070", linewidth=0.7, linestyle="--")
        ax[2].set_yticks(y)
        ax[2].set_yticklabels([f"{a}\nvs {b}" for a, b in
                               zip(comps["method_a"], comps["method_b"])])
        ax[2].set_xlabel("绝对差值（家族分组 bootstrap 95% 区间）")
        panel_label(ax[2], "c", "预设主要比较")

        w = 0.25
        x = np.arange(len(comps))
        for i, (col, lab, h) in enumerate([("frac_benefit", "受益", ""),
                                           ("frac_tie", "持平", "//"),
                                           ("frac_harm", "受损", "xx")]):
            b = ax[3].bar(x + (i - 1) * w, comps[col], width=w, label=lab,
                          color=GREY[i + 1], edgecolor="#000000", linewidth=0.5)
            for rect in b:
                rect.set_hatch(h)
        ax[3].set_xticks(x)
        ax[3].set_xticklabels([a.replace("b9_ours", "本文") for a in comps["method_a"]],
                              rotation=15, ha="right")
        ax[3].set_ylabel("疾病比例")
        ax[3].legend(frameon=False, ncol=3, loc="upper center")
        panel_label(ax[3], "d", "疾病级结局构成")
    fig.tight_layout()
    return save(fig, "main_figure_02_main_results", out_dir)


def figure03(out_dir: Path) -> dict:
    dose = read_table("dose_summary.csv")
    gamma = read_table("gamma_summary.csv")
    fig, ax = _grid(1, 3, 2.6)
    if dose is not None and not dose.empty:
        for i, (m, sub) in enumerate(dose.groupby("method")):
            sub = sub.sort_values("k")
            ax[0].plot(sub["k"], sub["recall10_end"], marker="o", markersize=3.5,
                       linewidth=1.0, color=GREY[min(i, 3)],
                       linestyle=["-", "--", "-.", ":"][i % 4], label=str(m))
        ax[0].set_xlabel("暴露支持数 k")
        ax[0].set_ylabel("端到端 Recall@10")
        ax[0].legend(frameon=False, fontsize=7.5)
        panel_label(ax[0], "a", "固定查询剂量曲线")
    if gamma is not None and not gamma.empty:
        ax[1].errorbar(gamma["k"], gamma["gamma"],
                       yerr=[gamma["gamma"] - gamma["ci_low"],
                             gamma["ci_high"] - gamma["gamma"]],
                       fmt="s", color="#000000", ecolor="#3c3c3c", capsize=2.5,
                       markersize=3.5, elinewidth=0.7)
        ax[1].axhline(0, color="#707070", linewidth=0.7, linestyle="--")
        ax[1].set_xlabel("k")
        ax[1].set_ylabel(r"$\Gamma$（本文 − 普通元岭）")
        panel_label(ax[1], "b", "支持机制的增量")
        bar(ax[2], [str(int(k)) for k in gamma["k"]],
            gamma["frac_gamma_positive"].tolist(), ylabel=r"$\Gamma>0$ 的疾病比例")
        ax[2].axhline(0.5, color="#707070", linewidth=0.7, linestyle="--")
        ax[2].set_xlabel("k")
        panel_label(ax[2], "c", "增量方向的一致性")
    fig.tight_layout()
    return save(fig, "main_figure_03_support_dose", out_dir)


def figure04(out_dir: Path) -> dict:
    ev = read_table("evidence_summary.csv")
    fig, ax = _grid(1, 3, 2.6)
    if ev is not None and not ev.empty:
        bar(ax[0], ev["condition"].tolist(), ev["recall10_end"].tolist(),
            ylabel="端到端 Recall@10")
        panel_label(ax[0], "a", "证据处理与替换支持")
        bar(ax[1], ev["condition"].tolist(), ev["mean_a_norm"].tolist(),
            ylabel=r"平均更新幅度 $\|a_d\|$")
        panel_label(ax[1], "b", "适配幅度")
        if "frac_harmed" in ev.columns:
            bar(ax[2], ev["condition"].tolist(), ev["frac_harmed"].tolist(),
                ylabel="受损疾病比例")
            panel_label(ax[2], "c", "受损比例")
    fig.tight_layout()
    return save(fig, "main_figure_04_evidence_dependence", out_dir)


def figure05(out_dir: Path) -> dict:
    cov = read_table("coverage_summary.csv")
    fig, ax = _grid(1, 3, 2.6)
    if cov is not None and not cov.empty:
        for i, (src, sub) in enumerate(cov.groupby("source")):
            sub = sub.sort_values("n_candidates_requested")
            ax[0].plot(sub["n_candidates_requested"], sub["coverage"], marker="o",
                       markersize=3.5, linewidth=1.0, color=GREY[min(i, 3)],
                       linestyle=["-", "--", "-."][i % 3], label=str(src))
            ax[1].plot(sub["n_candidates_requested"], sub["recall10_end"], marker="s",
                       markersize=3.5, linewidth=1.0, color=GREY[min(i, 3)],
                       linestyle=["-", "--", "-."][i % 3], label=str(src))
            ax[2].plot(sub["n_candidates_requested"], sub["zero_coverage"], marker="^",
                       markersize=3.5, linewidth=1.0, color=GREY[min(i, 3)],
                       linestyle=["-", "--", "-."][i % 3], label=str(src))
        for a, lab, t in [(ax[0], "覆盖率", "检索覆盖"),
                          (ax[1], "端到端 Recall@10", "排序表现"),
                          (ax[2], "零覆盖疾病比例", "完全漏检")]:
            a.set_xscale("log")
            a.set_xlabel("候选名单规模")
            a.set_ylabel(lab)
            a.legend(frameon=False, fontsize=7.5)
        panel_label(ax[0], "a", "检索覆盖")
        panel_label(ax[1], "b", "排序表现")
        panel_label(ax[2], "c", "完全漏检比例")
    fig.tight_layout()
    return save(fig, "main_figure_05_candidate_coverage", out_dir)


def figure06(out_dir: Path) -> dict:
    ext = read_json_any("external_labels_audit.json")
    ev = read_table("external_eval_summary.csv")
    temp = read_table("temporal_summary.csv")
    taudit = read_json_any("temporal_split_audit.json")
    fig, ax = _grid(2, 2, 5.4)
    layers = ext.get("exclusion_layers", {})
    if layers:
        keys = ["input_pairs", "layer1_excluded", "layer3_pmid_overlap",
                "layer3_unverifiable", "evaluable_pairs", "strict_independent_pairs"]
        labs = ["输入关系对", "第一层排除", "第三层 PMID 重叠", "出处不可核实",
                "可评价", "严格独立"]
        bar(ax[0], labs, [layers.get(k, 0) for k in keys], ylabel="关系对数")
        panel_label(ax[0], "a", "外部独立性逐级分母")
    if ev is not None and not ev.empty:
        piv = ev.pivot_table(index="method", columns="source",
                             values="recall10_end", aggfunc="mean")
        x = np.arange(len(piv))
        w = 0.8 / max(1, piv.shape[1])
        for i, c in enumerate(piv.columns):
            ax[1].bar(x + i * w, piv[c], width=w, label=str(c), color=GREY[i + 1],
                      edgecolor="#000000", linewidth=0.5)
        ax[1].set_xticks(x + w * (piv.shape[1] - 1) / 2)
        ax[1].set_xticklabels(piv.index, rotation=20, ha="right")
        ax[1].set_ylabel("端到端 Recall@10")
        ax[1].legend(frameon=False, fontsize=7.5)
        panel_label(ax[1], "b", "外部确认集合")
    if taudit:
        keys = ["historical_positives", "current_positives",
                "future_positives_before_node_filter",
                "future_positives_after_node_filter", "train_test_overlap_edges"]
        labs = ["历史阳性", "当前阳性", "新增（过滤前）", "新增（过滤后）", "训练测试重叠"]
        bar(ax[2], labs, [taudit.get(k, 0) for k in keys], ylabel="关系数")
        ax[2].set_yscale("symlog")
        panel_label(ax[2], "c", "时间外划分审计")
    if temp is not None and not temp.empty:
        bar(ax[3], temp["method"].tolist(), temp["recall10_end"].tolist(),
            ylabel="端到端 Recall@10")
        panel_label(ax[3], "d", "时间外表现")
    fig.tight_layout()
    return save(fig, "main_figure_06_external_temporal", out_dir)


def figure07(out_dir: Path) -> dict:
    neg = read_table("negative_summary.csv")
    net = read_table("network_summary.csv")
    per = read_table("per_disease_all.csv")
    fig, ax = _grid(2, 2, 5.4)
    if neg is not None and not neg.empty:
        bar(ax[0], neg["condition"].tolist(), neg["recall10_end"].tolist(),
            ylabel="端到端 Recall@10")
        panel_label(ax[0], "a", "疾病与支持置换负对照")
    if net is not None and not net.empty:
        for i, (m, sub) in enumerate(net.groupby("method")):
            sub = sub.sort_values("k")
            ax[1].plot(sub["k"], sub["recall10_end"], marker="o", markersize=3.5,
                       linewidth=1.0, color=GREY[min(i, 3)],
                       linestyle=["-", "--", "-.", ":"][i % 4], label=str(m))
        ax[1].set_xlabel("支持数 k")
        ax[1].set_ylabel("端到端 Recall@10")
        ax[1].legend(frameon=False, fontsize=7.5)
        panel_label(ax[1], "b", "种子驱动网络方法")
    if per is not None and not per.empty and "n_query" in per.columns:
        sub = per[per["method"] == "b9_ours"] if "method" in per.columns else per
        if not sub.empty:
            bins = [0, 1, 2, 5, 10, 10 ** 6]
            labs = ["1", "2", "3-5", "6-10", ">10"]
            sub = sub.copy()
            sub["bin"] = pd.cut(sub["n_query"], bins=bins, labels=labs)
            g = sub.groupby("bin", observed=True)["recall10_end"].mean()
            bar(ax[2], [str(i) for i in g.index], g.values, ylabel="端到端 Recall@10")
            ax[2].set_xlabel("查询阳性数")
            panel_label(ax[2], "c", "阳性负担分层")
            g2 = sub.groupby("bin", observed=True)["zero_coverage"].mean()
            bar(ax[3], [str(i) for i in g2.index], g2.values, ylabel="零覆盖比例")
            ax[3].set_xlabel("查询阳性数")
            panel_label(ax[3], "d", "分层漏检")
    fig.tight_layout()
    return save(fig, "main_figure_07_failure_modes", out_dir)


def figure08(out_dir: Path) -> dict:
    t6 = read_table("table6_resources.csv")
    probe = read_json_any("probe_cuda_fp32.json")
    probe_cpu = read_json_any("probe_cpu_fp32.json")
    expr = read_json_any("expression_report.json")
    fig, ax = _grid(2, 2, 5.4)
    if t6 is not None and not t6.empty:
        d = t6.sort_values("median_minutes")
        bar(ax[0], d["label"].tolist(), d["median_minutes"].tolist(),
            ylabel="中位耗时（分钟）")
        panel_label(ax[0], "a", "各方法实际训练耗时")
        bar(ax[1], d["label"].tolist(), d["median_peak_rss_gib"].tolist(),
            ylabel="峰值主进程内存（GiB）")
        panel_label(ax[1], "b", "主机内存占用")
    labs, vals = [], []
    for name, blob in (("GPU FP32", probe), ("CPU FP32", probe_cpu)):
        if blob:
            labs.append(name)
            vals.append(blob.get("train", {}).get("seconds_per_step", np.nan))
    if labs:
        bar(ax[2], labs, vals, ylabel="每步耗时（秒）")
        panel_label(ax[2], "c", "同模型 GPU 与 CPU 对比")
    series = (expr or {}).get("series", [])
    if series:
        labs = [s.get("accession", "") for s in series]
        vals = [s.get("n_pairs", 0) for s in series]
        bar(ax[3], labs, vals, ylabel="可用配对数")
        panel_label(ax[3], "d", "表达支持队列")
    fig.tight_layout()
    return save(fig, "main_figure_08_resources_biology", out_dir)


BUILDERS = [figure01, figure02, figure03, figure04,
            figure05, figure06, figure07, figure08]


def build_all(out_dir: Path | None = None) -> dict:
    setup_style()
    out_dir = Path(out_dir) if out_dir else paths.FIGURE_DIR
    records, problems = [], []
    for fn in BUILDERS:
        try:
            rec = fn(out_dir)
            records.append(rec)
            print(f"  {rec['stem']}  {rec['width_px']}x{rec['height_px']} px "
                  f"greyscale={rec['greyscale']}", flush=True)
        except Exception as exc:  # a failing panel must not silently vanish
            problems.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
            print(f"  {fn.__name__} FAILED: {exc}", flush=True)
    problems += validate(None, records)
    payload = {"n_figures": len(records), "figures": records,
               "problems": problems,
               "contract": {"width_in": FIG_WIDTH_IN, "dpi": DPI,
                            "min_font_pt": 7.5, "palette": "greyscale",
                            "formats": ["png", "svg", "pdf"]}}
    atomic_write_json(out_dir / "figure_index.json", payload)
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the manuscript figures.")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args(argv)
    payload = build_all(Path(args.out_dir) if args.out_dir else None)
    print(json.dumps({"n_figures": payload["n_figures"],
                      "problems": payload["problems"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

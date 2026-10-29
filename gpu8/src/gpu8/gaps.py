"""Generate the record of what was not done, from the artefacts themselves.

The research plan is explicit that an unfinished item must never be written as
a result.  This module inspects the ledger, the manifests and the result files,
decides which planned items are missing or downgraded, and writes both a JSON
record and the limitations text the manuscript uses.  Nothing here is a
hand-maintained list that can drift out of date.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import paths
from .config import Gpu8Config, load_config
from .hashing import atomic_write_json, atomic_write_text, read_json
from .registry import RunRegistry


def collect(cfg: Gpu8Config | None = None) -> dict:
    cfg = cfg or load_config()
    items: list[dict] = []

    # ---- ledger completion ------------------------------------------------
    reg = RunRegistry(paths.MANIFEST_DIR / "run_registry_166.csv")
    counts = reg.status_counts()
    planned = sum(counts.values())
    done = counts.get("done", 0)
    if done < planned:
        by_group: dict[str, list[str]] = {}
        for r in reg.rows:
            if r.get("status") != "done":
                by_group.setdefault(r.get("experiment", "?"), []).append(r["run_id"])
        missing = {k: len(v) for k, v in by_group.items()}
        items.append({
            "code": "ledger",
            "title": "正式拟合账本未全部完成",
            "detail": (f"计划 {planned} 次正式拟合，已完成 {done} 次，"
                       f"未完成按实验分组为 {missing}。未完成的运行在账本中保留 "
                       f"planned 或 failed 状态，其对应结论在本文中不作声明；"
                       f"本文只报告已完成运行支撑的结果。"),
            "severity": "major" if done < 0.5 * planned else "moderate"})

    # ---- temporal freeze downgrade ---------------------------------------
    hist = read_json(paths.GPU8_ROOT / "tables_historical"
                     / "historical_build_report.json", {})
    if hist:
        items.append({
            "code": "temporal",
            "title": "时间外实验为边级冻结，而非完整时点冻结",
            "detail": ("本机不再保留历史 CTD 关联原始转储，因此历史图由 2023Q4 "
                       "处理后疾病–基因边表重建，表型与本体使用同期版本，"
                       "而蛋白互作网络与基因背景为当前版本。该条件对所有方法相同，"
                       "但结论只能表述为疾病–基因边层面的严格时间划分，"
                       "不能表述为所有辅助资源都按同一历史时点冻结。"),
            "severity": "moderate"})

    # ---- manual literature review ----------------------------------------
    blind = paths.RESULTS_DIR / "manual_review" / "blind_review_sheet.csv"
    items.append({
        "code": "manual_review",
        "title": "人工盲态文献核查未执行",
        "detail": ("按协议应由两名核查者对约 40 个疾病、共约 160 个疾病–基因对"
                   "进行去标识盲态核查。本轮只交付了去标识抽样表"
                   + (f"（{paths.rel(blind)}）" if blind.exists() else "（尚未生成）")
                   + "与自动文献共现预筛结果，人工核查本身没有执行，"
                   "因此文中不报告任何核查一致性或证据类型比例。"),
        "severity": "moderate"})

    # ---- external independence -------------------------------------------
    ext = read_json(paths.RESULTS_DIR / "external" / "external_labels_audit.json", {})
    layers = ext.get("exclusion_layers", {})
    if layers.get("layer3_unverifiable", 0) > 0:
        items.append({
            "code": "external_provenance",
            "title": "部分外部记录的出处无法核实",
            "detail": (f"共 {layers.get('layer3_unverifiable', 0):,} 条外部关系对"
                       f"无法核实原始 PMID，其中包括全部 ClinVar 记录，"
                       f"因为变异摘要文件不提供文献字段。这些记录被单列为"
                       f"“出处不可核实”，不计入通过最严格独立性检查的计数；"
                       f"这是保守判定，不等于已经证明它们与训练证据无关。"),
            "severity": "moderate"})

    # ---- candidate coverage ceiling --------------------------------------
    t3 = paths.RESULT_TABLE_DIR / "table3_main_endpoints.csv"
    if t3.exists():
        df = pd.read_csv(t3)
        if not df.empty and "coverage" in df.columns:
            cov = float(df["coverage"].iloc[0])
            zero = float(df["zero_coverage"].iloc[0]) if "zero_coverage" in df.columns else float("nan")
            items.append({
                "code": "coverage",
                "title": "候选覆盖是主要瓶颈，端到端指标受其上限压缩",
                "detail": (f"在完整蛋白编码基因背景与不读取标签的检索下，"
                           f"候选覆盖率约为 {cov:.3f}，零覆盖疾病比例约为 {zero:.3f}。"
                           f"这意味着多数低资源疾病的正确基因根本没有进入候选名单，"
                           f"端到端 Recall@10 的可达上限因此很低。"
                           f"条件指标只描述已入池疾病，不能单独用来声称实用能力。"),
                "severity": "major"})

    # ---- development-set scope -------------------------------------------
    dev = read_json(paths.MANIFEST_DIR / "frozen_hyperparameters.json", {})
    if dev:
        grid = dev.get("development_grid", [])
        items.append({
            "code": "tuning_budget",
            "title": "调参预算有限且各方法不等",
            "detail": (f"本文方法使用 {len([g for g in grid if 'b9' in str(g.get('run_id'))])} 组开发配置，"
                       f"普通元岭与 MLP 分别使用更少的配置，"
                       f"树模型与图学习对照按预算表登记。"
                       f"预算差异按方法族预先分配，用于防止新方法无限搜索而基线仅默认运行，"
                       f"但它不等于每个方法都已达到其最优配置。"),
            "severity": "minor"})

    # ---- community baseline training budget ------------------------------
    kd_logs = sorted((paths.LOG_DIR / "runs").glob("*kdgene*.log")) \
        if (paths.LOG_DIR / "runs").exists() else []
    if kd_logs:
        last_losses = []
        for p in kd_logs[:6]:
            try:
                lines = [l for l in p.read_text(encoding="utf-8", errors="replace")
                         .splitlines() if "kdgene epoch" in l]
            except Exception:
                continue
            if len(lines) >= 2:
                def val(s):
                    try:
                        return float(s.split("loss=")[1].split()[0])
                    except Exception:
                        return float("nan")
                last_losses.append((val(lines[-2]), val(lines[-1])))
        still_falling = sum(1 for a, b in last_losses if b == b and a == a and b < a)
        if still_falling:
            items.append({
                "code": "kdgene_budget",
                "title": "知识图谱对照的训练预算受固定上限限制",
                "detail": (f"KDGene 按固定 30 个 epoch 上限训练，未启用基于冷验证集的早停。"
                           f"在检查的 {len(last_losses)} 个运行中，有 {still_falling} 个"
                           f"在最后一个 epoch 时训练损失仍在下降，说明该对照可能并未收敛。"
                           f"因此本文对 KDGene 的结果应理解为在此预算下的表现，"
                           f"而不是该方法能力的上界；所有 KDGene 运行使用相同预算，"
                           f"彼此之间仍可比较。"),
                "severity": "moderate"})

    # ---- second network / family, if absent ------------------------------
    alt = paths.RESULT_TABLE_DIR / "alt_runs_all.csv"
    modes_done = set()
    if alt.exists():
        modes_done = set(pd.read_csv(alt)["mode"].astype(str))
    for mode, label in (("family", "疾病家族隔离实验"),
                        ("external", "外部确认集合评价"),
                        ("temporal", "时间外评价"),
                        ("ppi2", "第二蛋白互作网络重拟合")):
        if mode not in modes_done:
            items.append({
                "code": f"missing_{mode}",
                "title": f"{label}未完成",
                "detail": f"该实验已实现并登记，但在本轮交付时未产出结果，因此文中不作相应声明。",
                "severity": "moderate"})

    # ---- always-stated boundaries ----------------------------------------
    items.append({
        "code": "scope",
        "title": "研究边界",
        "detail": ("本文不研究患者表型到疾病的诊断、变异致病性解释或从原始测序得到候选变异；"
                   "不开展 FASTQ 比对、原始单细胞聚类、大规模变异调用或新临床样本采集；"
                   "不使用大语言模型。全基因背景评价用于检验压力与适用范围，"
                   "不构成全基因组发现能力的证据。"),
        "severity": "scope"})
    items.append({
        "code": "audit_limits",
        "title": "自动隔离检查的能力边界",
        "detail": ("通过自动交集检查不等于已经排除全部数据库同源证据。"
                   "外部来源与原始文献的重叠仍需单独核查；"
                   "本文的数值与不变性测试只验证实现的局部性质。"),
        "severity": "scope"})

    payload = {"n_items": len(items), "items": items,
               "ledger_status": counts,
               "generated_from": ["manifests/run_registry_166.csv",
                                  "results/external/external_labels_audit.json",
                                  "results/tables/table3_main_endpoints.csv",
                                  "tables_historical/historical_build_report.json"]}
    atomic_write_json(paths.REPORT_DIR / "what_was_not_done.json", payload)

    lines = ["# 未完成与受限项记录", "",
             "本文件由产物自动生成，用于保证稿件中的局限性与实际执行状态一致。", ""]
    for it in items:
        lines.append(f"## {it['title']}（{it['code']}，{it['severity']}）")
        lines.append("")
        lines.append(it["detail"])
        lines.append("")
    lines.append(f"账本状态：{counts}")
    atomic_write_text(paths.REPORT_DIR / "what_was_not_done.md", "\n".join(lines))
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Record what was not done.")
    args = ap.parse_args(argv)
    p = collect()
    print(json.dumps({"n_items": p["n_items"],
                      "codes": [i["code"] for i in p["items"]],
                      "ledger": p["ledger_status"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

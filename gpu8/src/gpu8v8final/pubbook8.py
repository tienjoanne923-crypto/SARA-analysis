"""A single review PDF of all publication figures (vector pages merged; cover with the list of figures)."""
from __future__ import annotations

import sys
from pathlib import Path

from .pubstyle8 import FIG_DIR, FS, FS_L, MM, setup

ORDER = [("Fig1_knowledge_landscape", "图 1", "关联知识与信息层可用性的不均衡"),
         ("Fig2_method_overview", "图 2", "V8.1 定稿方法与评价框架"),
         ("Fig3_main_results", "图 3", "严格 500 候选预算下的总体效应与稳定性"),
         ("Fig4_budget_fusion_flow", "图 4", "候选预算、缺失感知融合与名次流量"),
         ("Fig5_knowledge_strata", "图 5", "按基因与疾病的知识覆盖分层"),
         ("Fig6_second_stage_diagnostics", "图 6", "二级模型、门控纠偏器的诊断与 SHAP 解释"),
         ("Fig7_go_mask_sequence", "图 7", "受控 GO 缺失、序列补位与关联监督冷"),
         ("Fig8_evidence_sources", "图 8", "GO 证据来源、证据块与删除敏感性"),
         ("Fig9_external_temporal_applications", "图 9", "外部再分析、历史截止与应用任务"),
         ("Fig10_case_studies", "图 10", "六个测试疾病的前 10 名、留出关联与基因网络"),
         ("FigS1_teacher_training", "图 S1", "语义教师的训练与对照集选择"),
         ("FigS2_development_selection", "图 S2", "开发集上的配置选择"),
         ("FigS3_fold_seed_stability", "图 S3", "各外层 × 种子的稳定性"),
         ("FigS4_semantic_variants", "图 S4", "全背景单独语义排名与文本变体"),
         ("FigS5_embedding", "图 S5", "语义空间的结构"),
         ("FigS6_compute_resources", "图 S6", "计算资源"),
         ("FigS7_evidence_blocks", "图 S7", "证据块关系的恢复与删减数据中的新旧方法差")]


def build(out: Path | None = None) -> Path:
    sys.path.insert(0, str(Path(r"D:\信\gpu8\pub_libs")))
    from pypdf import PdfReader, PdfWriter
    import matplotlib.pyplot as plt
    setup()
    out = out or FIG_DIR.parent / "V81_图集_20260926.pdf"
    cover = FIG_DIR / "_cover.pdf"
    fig = plt.figure(figsize=(183 * MM, 247 * MM))
    fig.text(0.07, 0.93, "面向知识覆盖不均衡的疾病基因语义检索与可用性感知排序", fontsize=11, fontweight="bold")
    fig.text(0.07, 0.905, "V8.1 修订稿图集（2026-09-26）：10 张主图 + 7 张补充图", fontsize=FS_L)
    y = 0.86
    for stem, n, t in ORDER:
        fig.text(0.07, y, n, fontsize=FS + 0.5, fontweight="bold")
        fig.text(0.17, y, t, fontsize=FS + 0.5)
        y -= 0.034
    fig.text(0.07, 0.09, "全部数字来自冻结的运行结果；每个面板的作图数据见 figures_v2/source_data/。\n"
                         "配色：经 Machado 2009 色觉缺陷模拟校验的固定分类色，颜色始终对应同一方法。", fontsize=FS - 0.5,
             color="#52514e", linespacing=1.5)
    fig.savefig(cover)
    plt.close(fig)
    w = PdfWriter()
    for p in [cover] + [FIG_DIR / f"{s}.pdf" for s, _, _ in ORDER]:
        for page in PdfReader(str(p)).pages:
            w.add_page(page)
    w.add_metadata({"/Title": "V8.1 修订稿图集", "/Subject": "疾病基因语义检索与可用性感知排序"})
    with open(out, "wb") as fh:
        w.write(fh)
    cover.unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    print(build())

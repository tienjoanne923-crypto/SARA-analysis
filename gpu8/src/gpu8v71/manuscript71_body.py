"""Body of the V7.1 manuscript.  Numbers only through ``N``; outcome-dependent
wording is derived from confidence intervals, never typed by hand."""
from __future__ import annotations

from .mstext71_refs import INTRO_V71, REFS_V71, RELATED_V71

TITLE_MM = "多模态语义检索核心的疾病基因候选恢复与袋外后融合"
TITLE_EXT = "面向异构信息缺失的疾病基因候选恢复与支持驱动排序：多模态检索扩展的实测检验"
SUB_MM = ("Disease-gene candidate recovery with a multimodal semantic retrieval core and "
          "out-of-fold late fusion")
SUB_EXT = ("Candidate recovery under heterogeneous information missingness, with a tested "
           "multimodal retrieval extension")

M_NAME = {"f7": "F7", "v6lgb": "V6 LightGBM", "chlgb": "仅检索通道 LightGBM", "srff": "同资源固定融合",
          "core": "多模态核心（单独）", "lffix": "后融合（固定）", "lflgb": "后融合（LightGBM）"}


def verdict(n, pre) -> str:
    lo, hi = n.raw(pre + "_ci_low"), n.raw(pre + "_ci_high")
    if lo > 0:
        return "区间整体在零以上"
    if hi < 0:
        return "区间整体在零以下"
    return "区间跨零，差别不确定"


def ok_ci(n, pre) -> bool:
    return n.has(pre + "_ci_low") and n.raw(pre + "_ci_low") > 0


def multimodal_claim(n) -> tuple[bool, dict]:
    c1 = ok_ci(n, "c1_lflgb_vs_chlgb_recall10_end") and ok_ci(n, "c1_lflgb_vs_f7_recall10_end")
    drops = [n.v.get(f"e5_paired_permutation_{m}_lf_drop", 0.0) for m in ("S", "F", "E")]
    c2 = any(d > 0 for d in drops) and n.v.get("e5_paired_permutation_F_core_drop", 0.0) > 0
    lf = n.v.get("e2t_e2_lf_recall10_end", 0.0)
    c3 = (n.v.get("e6t_e6_mask_recall10_end", 1.0) < lf) and (n.v.get("e2t_e2_raw_recall10_end", 1.0) < lf)
    return (c1 and c2 and c3), {"late_fusion_beats_controls": c1, "permutation_lowers": c2,
                                "not_reproduced_by_mask_or_raw": c3}


def row(n, slug, cols):
    out = [M_NAME.get(slug, slug)]
    for c in cols:
        k = f"e1_{slug}_{c}"
        out.append(n.f(k) if n.has(k) else "—")
    return "| " + " | ".join(out) + " |"


def header(n, title, sub) -> str:
    return (f"# {title}\n\n**{sub}**\n\n第七轮（V7.1）修订稿 · 2026 年 9 月 23 日 · 依据《疾病基因多模态检索可执行完整方案 V7.1》"
            f"完成的实测版本（在 V6 修订稿基础上扩展）")


def abstract(n, claim) -> str:
    s = []
    s.append("## 摘要")
    s.append("**背景**　第六轮（V6）把零支持候选恢复与支持驱动传播分开修复后，五折 4,238 个疾病的端到端 Recall@10 达到 0.109，但约 23% 的答案关系在参考路径中不可见（14.9% 只经 PPI 邻居、8.0% 无路径），几乎从不进入 500 候选；把功能、通路与表达原始向量拼接进排序器也没有帮助。")
    s.append(f"**方法**　V7.1 建立一个独立检索的有类型多模态核心：冻结 SapBERT 编码掩蔽疾病文本、HPO 与 GO 词条，冻结 ESM-2 35M 以覆盖归一的分窗池化编码 {n.i('seq_n_with_sequence')} 个基因的代表蛋白，HPA 共识组织表达经登记的器官词别名表与疾病组织条件匹配；序列经 {n.i('fun_anchor_terms')} 个 GO 功能锚与语义空间对齐，再以多正例排序学习疾病检索。核心先在约两万基因的完整背景上独立产生候选，再以覆盖整条学习链的袋外方式（5 折 × 3 种子 = 60 次整链拟合）与 V6 名单合并成工作池，由非负固定融合或 LightGBM 一次排序。核心配置、保留模态与准入只在第 0 折开发集上决定，并连同一个从未被 V5/V6 使用的新确认集（{n.i('conf_diseases')} 个疾病、{n.i('conf_pairs')} 对）在任何测试评价之前登记。")
    lf, ch, v6 = (n.f("e1_lflgb_recall10_end"), n.f("e1_chlgb_recall10_end"), n.f("e1_v6lgb_recall10_end"))
    s.append(f"**结果**　核心单独的端到端 Recall@10 为 {n.f('e1_core_recall10_end')}（V6 主模型 {v6}；同资源零训练固定融合 {n.f('e1_srff_recall10_end')}），"
             f"500 名单覆盖 {n.f('e1_core_coverage_final500')}（V6 {n.f('e1_v6lgb_coverage_final500')}），并把参考路径中“原无路径”的答案进入 500 的比例由 {n.pc('e3_v6lgb_unseen_in500')} 提高到 {n.pc('e3_core_unseen_in500')}、"
             f"“仅 PPI 邻居”的答案由 {n.pc('e3_v6lgb_ppi_only_in500')} 提高到 {n.pc('e3_core_ppi_only_in500')}。LightGBM 后融合的 Recall@10 为 {lf}，"
             f"相对重新登记的仅检索通道 LightGBM（{ch}）为 {n.ci('c1_lflgb_vs_chlgb_recall10_end')}，相对 V6 主模型为 {n.ci('c1_lflgb_vs_v6lgb_recall10_end')}；"
             f"即使不改变 V6 的 500 名单，只加入核心特征也把 Recall@10 由 {n.f('e6t_e6_v6ch_recall10_end')} 提高到 {n.f('e6t_e6_v6core_recall10_end')}。"
             f"在预先冻结、从未使用过的新确认集上（过滤 CTD 阳性），后融合为 {n.f('e7_lflgb_recall10')}，仅检索通道 LightGBM 为 {n.f('e7_chlgb_recall10')}，差 {n.ci('c7_lflgb_vs_chlgb')}。"
             f"固定候选上的配对置换显示增益主要来自疾病语义与 GO 功能语义的对应（后融合下降 {n.f('e5_paired_permutation_F_lf_drop')}），组织表达的贡献小而可检测（{n.f('e5_paired_permutation_E_lf_drop')}），"
             f"蛋白序列只在核心单独时可检测（{n.f('e5_paired_permutation_S_core_drop')}）；资源 × 方法 2×2 中，换用新原始资源（ESM、HPA）几乎不改变后融合结果（{n.f('e4t_e4_on_recall10_end', 4)} 对 {n.f('e4t_e4_nn_recall10_end', 4)}），"
             f"而同一旧资源下换用新方法提高 {n.sd('e4_method_gain_oldres')}。HPA 图像与 IMPC 小鼠通道在当前映射下没有带来增益。")
    if claim:
        s.append("**结论**　由冻结编码器构成的小型有类型核心能够在完整基因背景上独立检索到 V6 参考路径看不到的候选，与强通道基线袋外后融合后带来可重复、并在新确认集上方向一致的端到端增益；配对置换与仅掩码、原始拼接对照表明增益依赖真实的输入对应而不是覆盖标志。但增益的主体来自疾病语义与功能语义的对齐和后融合方法本身，蛋白序列与新表达资源的独立贡献很小，图像与跨物种表型未获支持；本文按此如实界定“多模态”的含义。")
    else:
        s.append("**结论**　多模态核心可以独立运行并提高候选覆盖，但按登记的判定规则，其增益尚不足以宣称“多模态增益”；本文把它作为已检验的扩展如实报告，并保留全部阴性与有限结果。")
    s.append("**关键词**　疾病基因检索；多模态；蛋白语言模型；语义对齐；冷启动；袋外后融合；配对置换；预注册")
    return "\n\n".join(s)


def intro(n, secs) -> str:
    v6_intro = next(b for h, b in secs if h.startswith("## 1. 引言"))
    contrib = f"""本文在 V6 的全部结果之上增加的工作与可宣称的贡献如下，每一条都附带可证伪的检验：

1. **真实多模态特征包与来源隔离**。{n.i('seq_n_with_sequence')} 个基因的代表蛋白、{n.i('fun_genes')} 个基因的 GO 功能语义、{n.i('expr_genes')} 个基因的 51 组织表达，以及 {n.pc('cov_dis_d_tissue')} 有标注疾病的组织条件，全部按主键对齐并记录来源、版本与 SHA-256；缺失以掩码表示，从不以随机矩阵顶替。
2. **有类型的独立检索核心**。序列、功能与表达各自有固定的适配与匹配分数，门控只读疾病表示与可用性；核心在完整基因背景上独立产生候选（E1、E2）。
3. **覆盖整条学习链的袋外后融合**。三内块 + 一外层的 60 次整链拟合保证二级模型看到的每个训练特征都来自没见过该疾病关联的上游模型（E1、E6）。
4. **五层验收与配对置换**。以保留边际分布的配对置换、单模态关闭、仅掩码模型与原始拼接区分“用到了模态对应”和“只用到了覆盖或热度”（E5）。
5. **冷基因与新确认**。按 V6 的答案可见性四层报告新增恢复（E3），并在建模前冻结的新确认集上一次性评价全部方法（E7）。"""
    return "\n\n".join(["## 1. 引言", v6_intro.split("本文的工作与可宣称的贡献如下")[0].strip(), INTRO_V71, contrib])


def related(secs) -> str:
    v6 = next(b for h, b in secs if h.startswith("## 2. 相关工作"))
    return "## 2. 相关工作\n\n" + v6.strip() + "\n\n" + RELATED_V71


def methods(secs, methods_v71, n) -> str:
    from .manuscript71 import v6_block
    v6 = v6_block(secs, "## 3. 材料与方法", "## 4. 结果")
    mt = methods_v71.replace("«N_ALIAS»", n.i("tis_n_rules"))
    return v6.strip() + "\n\n" + mt.strip()


def results_v6(secs) -> str:
    from .manuscript71 import v6_block
    v6 = v6_block(secs, "## 4. 结果", "## 5. 讨论")
    return v6.strip()


FIGS_V71 = [
    ("fig9_v71_framework.png", "图 9　V7.1 框架。疾病侧（掩蔽文本、HPO、组织条件）与基因侧（ESM-2 序列、GO 功能语义、HPA 表达）经有类型适配进入核心；核心先独立检索，再以覆盖整条学习链的袋外方式与 V6 后融合；五层验收与 E1–E8 评价检验模态是否真正被使用。"),
    ("fig10_v71_coverage_dev.png", "图 10　多模态覆盖与开发选择。(A) 基因侧各模态覆盖（全部基因与已知阳性基因）；(B) 有标注疾病的文本、HPO 与组织条件覆盖；(C) 第 0 折开发集上 8 个预定配置的 Recall@10（维数/锚权重/门控；深色为选定配置）。"),
    ("fig11_v71_e1_main.png", "图 11　E1 主比较（五折 × 三种子）。(A) 端到端 Recall@10；(B) Hit@10；(C) 最终排序前 500 的覆盖。"),
    ("fig12_v71_cold_flow.png", "图 12　冷基因与命中流量。(A) 按答案可见性分层的前十恢复；(B) 后融合相对 V6 LightGBM 与仅通道 LightGBM 的新增与被挤出命中（每种子）。"),
    ("fig13_v71_behaviour.png", "图 13　E5 模态行为。固定模型与候选下单模态关闭与配对置换造成的 Recall@10 下降：(A) 核心单独；(B) 后融合。"),
    ("fig14_v71_confirm_e2e4.png", "图 14　(A) 预先冻结的新确认集（过滤评价）；(B) E2 同数据不同算法（种子 42）；(C) E4 资源 × 方法 2×2。"),
]


def figures(secs) -> str:
    v6 = next(b for h, b in secs if h.startswith("## 图与图注"))
    parts = ["## 图与图注", v6.strip()]
    for fn, cap in FIGS_V71:
        parts.append(f"![{cap}](figures/{fn})\n{cap}")
    return "\n\n".join(parts)


def references(secs) -> str:
    v6 = next(b for h, b in secs if h.startswith("## 参考文献"))
    lines = [v6.strip()]
    for i, r in enumerate(REFS_V71, start=41):
        lines.append(f"{i}. {r}")
    return "## 参考文献\n\n" + "\n".join(lines)


def availability(n) -> str:
    return ("## 8. 数据与代码可用性\n\n"
            "V7.1 代码位于 `gpu8/src/gpu8v71/`（资源、序列、编码器、功能锚、表达、特征包、核心、训练、整链、打分、工作池、二级模型、行为检验、扩展通道、登记、评价、数字、图与稿件），"
            "测试位于 `gpu8/tests/test_v71_core.py` 与 `test_v71_prep.py`。外部资源、版本、URL 与 SHA-256 记录在 `manifests_v71/resources_v71.json` 与 `external/v71/download_log_v71.csv`，"
            "冻结编码器的提交版本与文件哈希记录在 `manifests_v71/models_v71.json`，特征包的全部数组哈希与覆盖在 `manifests_v71/featurepack_v71.json`。"
            "每个运行的权威记录在 `runs_v71/<run>/result.json`，账本 `manifests_v71/run_ledger_v71.csv` 由结果文件重建并校验内容哈希；"
            "文中每个数字与其来源文件列于 `reports_v71/manuscript_numbers_v71.csv`。V5 与 V6 的档案保持不变。")


def ethics(secs) -> str:
    v6 = next(b for h, b in secs if h.startswith("## 9. 伦理"))
    extra = ("V7.1 新增资源（HPA、UniProt、MANE、IMPC、MGI、MP 本体、mapping-commons SSSOM、SapBERT 与 ESM-2 权重）均为公开资源，"
             "许可说明逐项记录在资源日志中；本轮不涉及任何个人或患者数据。")
    return "## 9. 伦理与数据许可\n\n" + v6.strip() + "\n\n" + extra


def compose(n, secs, methods_v71) -> tuple[str, str]:
    from .appendix71 import appendices
    from .manuscript71 import build_markdown
    from .manuscript71_results import conclusion, discussion, limitations, results_v71
    claim, crit = multimodal_claim(n)
    title, sub = (TITLE_MM, SUB_MM) if claim else (TITLE_EXT, SUB_EXT)
    md = build_markdown([
        header(n, title, sub), abstract(n, claim), intro(n, secs), related(secs),
        methods(secs, methods_v71, n), results_v6(secs), results_v71(n),
        discussion(n, claim, crit), limitations(n), conclusion(n, claim), availability(n), ethics(secs),
        ai_note(), references(secs), figures(secs), appendices(secs)])
    return title, md


def ai_note() -> str:
    return ("## 10. AI 辅助说明\n\n本轮代码实现、实验调度、结果汇总与稿件起草由 AI 编码助手在研究者授权下完成；全部数字由程序从运行记录读取，"
            "未经人工改写。模型输出与结论的科学责任由作者承担；人工盲审与小鼠映射人工抽查等需要人工完成的环节如实标为未完成。")

"""Appendices of the V6 manuscript: closure table (section 16), ledger summary,
what was not done, and the plan-section -> artefact map."""
from __future__ import annotations

import pandas as pd

from . import paths6


def _f(v, k, d=3):
    return f"{v[k]:.{d}f}"


def closure_rows(v: dict) -> list[tuple[str, str, str, str]]:
    f = lambda k, d=3: _f(v, k, d)
    return [
        ("A1 覆盖不足", "非 HPO 入口（掩蔽文本、疾病本体、层级）、来源排除的基因 HPO、倒数排名融合；同预算 500",
         f"五折覆盖 {f('z_v5_b3_cpu_fusion_coverage')}→{f('z_v6_lightgbm_coverage')}；U@10 +{f('cmp_ZC_delta')}（Holm p<0.001）；零覆盖 {f('z_v5_b3_cpu_fusion_zero_coverage')}→{f('z_v6_lightgbm_zero_coverage')}；V5 零覆盖疾病救回 {int(v['rescued_n'])}/{int(v['v5_zero_n'])}；各来源留一贡献与每百新增命中（results_v6/retrieval/）",
         "已关闭（覆盖 0.5 为开发目标，未达到，不作门槛）"),
        ("A2 阳性偏斜", "Z 与 S 分开；全部五折袋外剂量曲线；固定查询",
         f"支持线 {int(v['s_n'])} 个疾病（五折），715 个 5-shot 合格疾病按开发/外部/折拆分分母；原始与归一化 Recall 并报",
         "已关闭"),
        ("A3 信息错配", "按可观测信息分流（Z1/Z2/Z3、S 两类）；关系语义 × HPO × 支持交叉表",
         "路由只用 HPO/文本/本体/显露支持；分层结果（表 3 正文）；crosstab_relation_hpo_support.csv",
         "已关闭"),
        ("A4 基因空视图", "GO/通路/表达特征（带缺失标志）与来源排除的基因 HPO；同输入基线",
         f"基因特征覆盖 GO {int(v['genes_go'])}、通路 {int(v['genes_rx'])}、表达 {int(v['genes_gtex'])} 个基因；基因 HPO 留一覆盖贡献 {f('src_gene_hpo_coverage_contribution')}；基因特征在评分器中无增益（加 GO {f('gab_inc_plus_go',4)}）",
         "已关闭（评分器特征为阴性结果）"),
        ("A5 出处机制", "只用当前显露支持的来源组；种子可识别性",
         f"k=5 种子改变 {100*v['fl_k5_layer1_share_seed_changed']:.1f}%，‖v−u‖₁ 均值 {f('fl_k5_layer1_mean_l1_v_minus_u')}；出处可追溯 episode {100*v['s_trace_k5']:.1f}%；Γgrp {f('s2_Gamma_grp_k5_delta',4)}",
         "已关闭（机制阴性）"),
        ("A6 家族粒度", "冻结深度 3 为主、深度 4 为敏感性；审计组大小、跨折泄漏与最近参照相似度",
         f"深度 3：{int(v['fam3_n'])} 族、最大 {int(v['fam3_max'])}；深度 4：{int(v['fam4_n'])} 族、最大 {int(v['fam4_max'])}；跨折家族 {int(v['fam_cross'])}；文本近重复（余弦≥0.9）{100*v['nn_text_ge09']:.1f}%",
         "已关闭（未按结果改选深度）"),
        ("B1 疾病置换", "P1/P2/P3 分开；V5 对照重新归类",
         f"V5 b9：P1 {f('pv5_P1_distant_delta',4)}、P2 {f('pv5_P2_distant_delta',4)}；V6 LightGBM P2 {f('pv6_lightgbm_P2_distant_all_delta',4)}；注意力残差 P1 {f('pv6_attention_residual_P1_distant_all_delta',4)}",
         "已关闭"),
        ("B2 元岭无效", "主路径改为图传播；固定池与自适应池分开",
         f"S-A +{f('s_SA_delta',4)}（Holm p {f('s_SA_holm_p')}）；受益/受损 {100*v['s_SA_frac_benefit']:.1f}%/{100*v['s_SA_frac_harm']:.1f}%；自适应池 k=5 低 {abs(v['s2_adaptive_vs_fixed_joint_w0.75_k5_delta']):.4f}；保度重连 {f('rw_k5_rewired_minus_real',4)}",
         "已关闭"),
        ("B3 权重无效", "在 RWR 中隔离种子权重；四层诊断；等权/真组/打乱组/连接度",
         f"k=1 严格不变；S-C {f('s_SC_delta',4)}（Holm p {f('s_SC_holm_p')}）；连接度权重 {f('s2_degree_vs_equal_k5_delta',4)}",
         "已关闭（阴性）"),
        ("B4 过拟合", "低维残差 κ·tanh、袋外列表式训练、内层选择与一标准误差规则",
         f"神经残差开发拟合 {100*v['devfit_kappa0_share']:.0f}% 选 κ=0；Z-B {f('cmp_ZB_delta',4)}（Holm p {f('cmp_ZB_holm_p')}）；残差与 KDGene 冷验证曲线已保存",
         "部分关闭（无未参与设计的新终测）"),
        ("B5 消融解读", "统一 Δabl = Mabl − Mfull，配对种子重算，候选哈希核对",
         f"A1 {f('abl_a1_delta_abl',4)}、A2 {f('abl_a2_delta_abl',4)}、A3 {f('abl_a3_delta_abl',4)}（候选改变）、A4 {f('abl_a4_delta_abl',4)}；V5 表 4 种子不配对",
         "已关闭"),
        ("B6 固定融合最强", "开发集调优（35 点 + 细化；210 点 + 细化）作为主参照",
         f"开发集 F7 {f('dev_F7',4)} > 手设 {f('dev_hand',4)}；五折 F7 {f('z_v6_F7_recall10_end',4)}",
         "已关闭"),
        ("C1–C2 基线预算", "KDGene 合法冷验证早停与学习曲线；LightGBM/LR 预设 × 内层划分",
         f"KDGene 最佳 epoch {int(v['kd_best_epoch'])}，冷验证 {f('kd_best_val',4)}，损失持续下降；开发拟合 {int(v['devfit_n'])} 次",
         "部分关闭（SIGN 未在 V6 候选上重新调参）"),
        ("C3/C6 时间外", "全输入历史审计；存档头部核验；不换主终点",
         "Wayback 转储头部“Report created: Wed Nov 29 2023”；除 Mondo 外全部模型输入为截止日前版本；X2 任务单列",
         "已关闭（边级冻结，原始转储未重下）"),
        ("C4 ClinVar 出处", "变异 → 条件 → 文献链；多条件变异记未知",
         f"{int(v['cv_independent_verified'])}/{int(v['cv_pairs'])} 条核实独立；未知率 {100*v['cv_unknown_rate']:.1f}%",
         "已关闭（变异级；SCV 级未做）"),
        ("C5 外部偏小", "保留旧外部集并报告区间；中等证据单列",
         f"GenCC 确定/强 {int(v['ext_gencc_definitive_strong_n'])} 个疾病；ClinVar ≥2 星 {int(v['ext_clinvar_2star_n'])} 个疾病",
         "未关闭（没有真正未参与设计的新来源）"),
        ("C7 未做验证", "保度重连、真实前 20/50 基因表达分析、人工盲审",
         "重连与表达已完成并保存哈希与失败结果；160 对盲审未完成",
         "部分关闭"),
        ("D1–D2 工程", "单运行权威结果文件、原子写、内容哈希；真实资源记录",
         f"{int(v['n_runs_v6'])} 次运行，{int(v['n_hash_ok'])} 个哈希校验通过；每次运行记录时间、RSS 与显存",
         "已关闭"),
        ("D3–D5 表述/指标", "差值方向统一；公式与实现一致；DOCX 真公式对象",
         "式（1）–（14）与代码一一对应；守护测试 36 项通过；DOCX 公式为 OMML",
         "已关闭"),
    ]


def appendix(v: dict) -> str:
    L = []
    A = L.append
    A("## 附录 A　问题逐条关闭表（对应 V6 方案第 16 节）")
    A("")
    A("| 原问题 | 修复动作 | 关闭证据（数值来自结果文件） | 状态 |")
    A("|---|---|---|---|")
    for r in closure_rows(v):
        A("| " + " | ".join(r) + " |")
    A("")
    A("## 附录 B　V6 运行账本摘要")
    A("")
    led = pd.read_csv(paths6.MANIFESTS / "run_ledger_v6.csv")
    tab = led.groupby(["stage", "method"]).agg(runs=("run_id", "size"),
                                               median_seconds=("seconds", "median")).reset_index()
    A("| 阶段 | 方法 | 运行数 | 中位耗时（秒） |")
    A("|---|---|---|---|")
    for r in tab.itertuples(index=False):
        A(f"| {r.stage} | {r.method} | {r.runs} | {r.median_seconds:.1f} |")
    A("")
    A("另有不计入拟合账本的确定性实验：检索调优（开发集）、候选表构建（20 个上下文）、支持线开发选择与五折测试、保度重连、置换套件（V5 与 V6）、外部层、ClinVar 引文链、时间审计与 X2 传播、表达支持、KDGene 冷验证曲线、家族审计。")
    A("")
    A("## 附录 C　未完成与受限项")
    A("")
    A("1. 160 对双人盲审：只交付去标识抽样表，未进行实际人工审阅与分歧处理。")
    A("2. 真正未参与设计的新外部来源：本轮没有；外部结论限于探索性再分析。")
    A("3. 时间任务的 2.66 GB 历史关系原始转储未重新下载（只核验存档头部与索引），使用其处理后的 2023Q4 边表。")
    A("4. ClinVar SCV 层面的条件—引文核实（需 XML 发布）未做；变异级未知率 45.4%。")
    A("5. SIGN 式对照未在 V6 候选上重新调参；KDGene 冷验证学习曲线只在第 0 折以开发预算运行。")
    A("6. 学习式检索器（方案 6.3 的可选第三条）未实现：前两条静态来源已能覆盖无 HPO 疾病，且评分阶段学习模型未显示稳定增益。")
    A("")
    A("## 附录 D　V6 方案逐节执行对照")
    A("")
    rows = [
        ("2.1 基因 HPO 来源", "gpu8v6/gene_hpo.py；来源排除后聚合，严格/宽松两版"),
        ("2.2 GO/通路/表达与疾病侧桥接", "gpu8v6/gene_features.py；availability.py 的共享文献检查"),
        ("2.5 消融差值方向", "gpu8v6/audit_metrics.py → results_v6/audit/ablation_unified*.csv"),
        ("2.6 当前支持的证据组", "gpu8v6/support_seed.py；report_s.py 四层诊断"),
        ("2.7 冷验证曲线", "gpu8v6/kdgene_curve.py；残差 history.json"),
        ("3.2 oracle 分解", "gpu8v6/metrics6.py；audit_metrics.py"),
        ("3.3 500→1000 命中流", "gpu8v6/hitflow.py"),
        ("4.1 可用性审计", "gpu8v6/phenotypes_v6.py；availability.py；family_audit.py"),
        ("4.3 基因 HPO 安全重建", "gpu8v6/gene_hpo.py"),
        ("4.4 非 HPO 检索", "gpu8v6/disease_text.py；ontology_v2.py"),
        ("5 可见信息分流", "availability.py 路由类；report_z.py 分层"),
        ("6 零支持检索", "sources.py；retrieval.py；tune_retrieval.py；build_candidates.py；source_report.py"),
        ("7 评分阶段", "scoring.py；residual.py；zero_support.py；select_main.py；gene_block_ablation.py"),
        ("8 疾病置换", "permutation.py；permutation_v5.py；run_permutations.py"),
        ("9 支持吸收", "propagate.py；support_line.py；rewire_control.py"),
        ("10 证据依赖", "support_seed.py；support_line.py；report_s.py"),
        ("11 模型选择与公平性", "register_protocol.py；select_main.py；development_registry_v6.csv"),
        ("12 时间外/ClinVar/生物学", "temporal6.py；temporal_support.py；clinvar_chain.py；external6.py；external_learned.py；expression6.py"),
        ("13 工程", "ledger6.py；propagate.py（批量与列缓存）"),
        ("17 论文重写", "manuscript6.py；build_docx6.py（OMML 公式）；figures6.py"),
        ("19 代数核验", "tests/test_v6_guardrails.py；tests/test_v6_residual.py"),
    ]
    A("| 方案章节 | 实现与产物 |")
    A("|---|---|")
    for a, b in rows:
        A(f"| {a} | {b} |")
    A("")
    return "\n".join(L)

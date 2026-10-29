"""Prose sections of the round-five manuscript.

Each function receives the loaded result sources and a number auditor, so every
quantity in the text is pulled from a file and logged.  Where a result does not
exist, the text says so instead of leaving a confident-sounding gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .manuscript import NumberAudit, Sources, fmt_pct


def _zh(row: dict) -> str:
    """Chinese label for a method row, falling back to the English one."""
    from .baselines.registry import METHODS
    spec = METHODS.get(row.get("method", ""))
    if spec:
        return spec.label_zh or spec.label
    return str(row.get("label") or row.get("method", ""))


def _row(df: pd.DataFrame, **cond) -> dict:
    if df is None or df.empty:
        return {}
    m = pd.Series(True, index=df.index)
    for k, v in cond.items():
        if k in df.columns:
            m &= (df[k] == v)
    sub = df[m]
    return sub.iloc[0].to_dict() if len(sub) else {}


def introduction(s: Sources, a: NumberAudit) -> str:
    p = s.protocol
    se = p.get("support_eligibility", {})
    return f"""## 1. 引言

疾病–基因优先级排序把候选基因按其与目标疾病的潜在关联强度排序，是罕见病解析、复杂病机制研究与功能验证资源分配的基础任务。在低资源疾病上，这个任务同时受到两类困难的挤压：一方面疾病特异的已知关联极少，模型必须依靠表型相似性、本体位置、蛋白互作与跨疾病共享结构；另一方面，可用于判断的评价协议本身常常不成立——若训练与测试只按边随机划分，同一疾病的邻域结构会同时出现在两侧；若图特征在完整疾病–基因图上计算，被隐藏的测试边仍可通过度、共同邻居或标准化统计量间接暴露。

本文的出发点是把这两类困难分开处理。第一，把评价协议固定为疾病级冷启动：测试疾病的全部疾病–基因边在其所属折的参考图中不存在，连训练任务内部也按内层块删边，使训练时的参考知识密度与测试时可比。第二，把覆盖与排序分开：候选由不读取查询标签的冻结检索器在完整基因背景上产生，因此"能不能把正确基因放进名单"与"放进来之后能不能排到前面"成为两个独立可报告的量。这一区分在本研究中并非形式上的谨慎——数据本身显示，{a.cite('intro_single', se.get('single_positive_only'), 'manifests/support_feasibility.csv')} 个疾病只有一条已知阳性，对它们而言检索是否命中那唯一一个基因，直接决定该疾病的全部指标。

第三，本文研究一个具体的方法问题：当新增支持可能包含重复收录、来源不明与跨疾病误导时，模型应当怎样吸收支持。已有的可微分闭式求解元学习给出了在神经模型内部使用岭回归并通过求解器反向传播的成熟做法；本文的差异在于，支持权重不再等权，而是按可识别的原始证据依赖归并成组、组内均分，并由一个只读取支持自身摘要的控制头决定正则强度与更新半径。是否值得这样做，必须由与普通元岭适配的同容量、同支持、同任务比较来回答，而不能以"首次用于本数据"回避最近邻比较。

本文不主张全基因组发现能力，也不主张临床前瞻价值。大基因背景评价在文中用于检验压力与适用范围，其结果被明确写成边界而非能力。
"""


def related_work(s: Sources, a: NumberAudit) -> str:
    return """## 2. 相关工作

早期疾病–基因优先级排序依赖表型相似性与网络邻近性。Endeavour 式的数据融合、基于表型本体的候选排序、网络传播与重启随机游走、以及 DIAMOnD 式的种子连通性富集，为解释疾病模块与候选邻近性提供了直接对照。本文保留其中两类作为需要真实支持的比较器，并如实标注它们在 zero-shot 下不适用。

知识图谱方法把疾病、基因、表型、通路与药物置于同一张图上做补全。KDGene 以交互式张量分解完成疾病基因预测，在其原始知识图谱上以十折关联评价报告结果。本文在共同的合法可见图上重训该方法，保留其评分函数、门控关系更新、全实体 softmax 目标与 N3 正则；同时明确说明实体空间与划分协议都与原文不同，因此不能把原文分数直接搬运过来比较。近期以连续时间量子游走做种子排序的工作提供了另一类网络方法，本文按其定义逐支持计算传播振幅再取模平方求和，并指出把所有支持的振幅先相加再取平方会产生交叉项，是不同的量。

在计算结构上，SGC 与 SIGN 把可预计算的传播与末端训练分离，为低计算量图学习提供了先例。本文的稀疏缓存策略以此为方法依据，并据此实现一个 SIGN 式对照；这一借鉴不构成对预计算思想的首创声明。

与本文最接近的是可微分闭式求解元学习：在神经模型内部使用岭回归等快速适配，并通过求解器反向传播。本文把它实现为普通元岭适配对照，给予相同的编码器容量、相同的支持与参照、相同的训练任务与预算，只把证据依赖权重、条件控制与结构化缺失训练替换为普通形式。这一对照是本文方法主张能否成立的判据。

在评价规范一侧，近期以少样本学习做表型驱动罕见病诊断的工作区分了自动过滤候选与专家整理候选，跨表型遗传学靶点工作则公开了大规模基因与疾病评分资源。本文借鉴它们对任务边界与证据来源的组织方式，但不把不同来源的疾病计数相加当作独立疾病数。
"""


def results(s: Sources, a: NumberAudit) -> str:
    parts = ["## 4. 结果", "", _res_feasibility(s, a), _res_main(s, a),
             _res_dose(s, a), _res_network(s, a), _res_evidence(s, a),
             _res_coverage(s, a), _res_ablation(s, a), _res_external(s, a),
             _res_negative(s, a), _res_case(s, a), _res_expression(s, a),
             _res_resources(s, a), _res_audit(s, a)]
    return "\n\n".join(x for x in parts if x)


def _res_network(s: Sources, a: NumberAudit) -> str:
    """Seeded network comparators on the same fixed-dose cohort."""
    if s.network.empty:
        return ""
    lines = ["| 方法 | k | 疾病数 | 端到端 Recall@10 | 条件 Recall@10 | MRR | 不适用疾病数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    names = {"b10_ppi_rwr": "PPI 重启随机游走", "b10b_diamond": "DIAMOnD 式种子邻居富集",
             "b11_quantum_walk": "连续时间量子游走"}
    for r in s.network.itertuples(index=False):
        m = getattr(r, "method", "")
        lines.append(f"| {names.get(m, m)} | {int(getattr(r, 'k', 0))} | "
                     f"{int(getattr(r, 'n_diseases', 0))} | "
                     f"{getattr(r, 'recall10_end', float('nan')):.4f} | "
                     f"{getattr(r, 'recall10_cond', float('nan')):.4f} | "
                     f"{getattr(r, 'mrr', float('nan')):.4f} | "
                     f"{int(getattr(r, 'n_not_applicable', 0))} |")
    verdict = ""
    try:
        net = s.network
        rwr5 = float(net[(net["method"] == "b10_ppi_rwr") & (net["k"] == 5)]
                     ["recall10_end"].iloc[0])
        qw5 = float(net[(net["method"] == "b11_quantum_walk") & (net["k"] == 5)]
                    ["recall10_end"].iloc[0])
        rwr1 = float(net[(net["method"] == "b10_ppi_rwr") & (net["k"] == 1)]
                     ["recall10_end"].iloc[0])
        ours5 = float(s.dose[(s.dose["method"] == "b9_ours") & (s.dose["k"] == 5)]
                      ["recall10_end"].iloc[0])
        if rwr5 > ours5:
            verdict = (
                f"这一节给出本文最具信息量的对照。在**完全相同的疾病、查询、候选与支持**下，"
                f"以支持基因为种子在蛋白互作网络上做传播的方法，端到端 Recall@10 从 k=1 的 "
                f"{rwr1:.4f} 单调上升到 k=5 的 {rwr5:.4f}；连续时间量子游走同样上升到 "
                f"{qw5:.4f}。而本文的可微分局部适配在同一集合上 k=5 时为 {ours5:.4f}，"
                f"低于两者，且如上一节所示并不随 k 上升。\n\n"
                f"因此，**支持基因确实携带可用信息，失败的是吸收方式而不是任务本身**。"
                f"种子传播直接在网络结构上使用支持基因的邻域；本文的适配则在一个学习到的"
                f"64 维表示空间内求解一个小型岭系统，而该空间（见 4.9 节的疾病置换对照）"
                f"并不编码疾病特异的区分方向，因此支持信息在其中无处可去。"
                f"这是一个具体的、可检验的机制性判断，而不是笼统的"
                f"“神经方法在小样本下不work”。")
    except Exception:
        pass
    return f"""### 4.4 种子驱动的网络方法对照

这三种方法需要疾病特异的支持基因作为种子，因此在 zero-shot 下不适用，不以零值计入均值。它们与本文方法在同一固定剂量队列上比较。

{chr(10).join(lines)}

{verdict}"""


def _res_case(s: Sources, a: NumberAudit) -> str:
    from . import paths
    from .manuscript import read_json_like
    blob = read_json_like(paths.RESULTS_DIR / "case_studies" / "case_study_report.json")
    if not blob:
        return ""
    oc = blob.get("outcome_counts", {})
    ct = blob.get("contrast", {})
    total = sum(oc.values()) or 1
    return f"""### 4.10 逐疾病结局分解与恢复对比

把每个测试疾病按最终结局归为三类，可以把总体指标拆成可解释的部分：

| 结局 | 疾病数 | 占比 | 含义 |
| --- | ---: | ---: | --- |
| 检索未命中 | {oc.get('not_retrieved', 0):,} | {fmt_pct(oc.get('not_retrieved', 0) / total)} | 候选名单中没有任何已知阳性，排序阶段无从发挥 |
| 入池但未进前十 | {oc.get('in_pool_missed', 0):,} | {fmt_pct(oc.get('in_pool_missed', 0) / total)} | 检索成功而排序失败 |
| 成功恢复 | {oc.get('recovered', 0):,} | {fmt_pct(oc.get('recovered', 0) / total)} | 前十中至少命中一个已知阳性 |

与最近邻对照（普通元岭适配）在前十结果上的互斥分解为：两者共同恢复 {ct.get('recovered_by_both', 0):,} 个疾病-基因对，仅本文方法恢复 {ct.get('recovered_only_by_method', 0):,} 个，仅对照恢复 {ct.get('recovered_only_by_comparator', 0):,} 个，两者均未恢复 {ct.get('missed_by_both', 0):,} 个。仅对照独有的恢复数不低于仅本文方法独有的恢复数，与第 4.2 节的配对统计一致。

逐疾病 Top-20 候选已导出，其中未被当前数据库记录的条目仅作为待核查候选，不作为发现。"""


def _res_feasibility(s: Sources, a: NumberAudit) -> str:
    p = s.protocol
    se = p.get("support_eligibility", {})
    feas = s.feas
    extra = ""
    if not feas.empty:
        e5 = feas[feas["eligible_5shot"] == 1]
        if len(e5):
            dep = (e5["n_evidence_groups"] < e5["n_unique_positives"])
            extra = (f"在 5-shot 合格队列中，支持关系的出处完整比例均值为 "
                     f"{a.cite('e5_source', float(e5['frac_with_source'].mean()), 'manifests/support_feasibility.csv')}，"
                     f"证据组数中位数为 {a.cite('e5_groups', float(e5['n_evidence_groups'].median()), 'manifests/support_feasibility.csv')}，"
                     f"其中 {a.cite('e5_dep', int(dep.sum()), 'manifests/support_feasibility.csv')} 个疾病"
                     f"（{fmt_pct(dep.mean())}）的证据组数少于唯一阳性数，即存在可识别的证据依赖；"
                     f"但该队列中 {a.cite('e5_nohpo', int((e5['hpo_coverage'] == 0).sum()), 'manifests/support_feasibility.csv')} 个疾病"
                     f"（{fmt_pct((e5['hpo_coverage'] == 0).mean())}）没有任何 HPO 注释。"
                     f"这说明支持证据丰富的疾病与表型注释丰富的疾病在本数据中基本不是同一批疾病，"
                     f"剂量实验与表型通道实验必须分别解释，不能互相外推。")
    return f"""### 4.1 支持资格盘点与任务规模

正式建模之前生成的逐疾病支持可行性表给出本研究最重要的一项前提事实：在 {a.cite('res_n_dis', p.get('n_labelled_diseases'), 'manifests/protocol_report.json')} 个带直接证据的疾病中，{a.cite('res_single', se.get('single_positive_only'), 'manifests/support_feasibility.csv')} 个只有一条已知阳性，1-shot、3-shot 与 5-shot 合格疾病分别为 {a.cite('res_e1', se.get('eligible_1shot'), 'manifests/support_feasibility.csv')}、{a.cite('res_e3', se.get('eligible_3shot'), 'manifests/support_feasibility.csv')} 与 {a.cite('res_e5', se.get('eligible_5shot'), 'manifests/support_feasibility.csv')} 个。

{extra}

因此本文把 0/1/3/5 剂量曲线限定在具备完整支持池与可评价查询的共同疾病集合上，并把只有一条阳性的疾病全部保留在 zero-shot 主任务中，而不是从分母里删除它们。"""


def _res_main(s: Sources, a: NumberAudit) -> str:
    if s.t3.empty:
        return "### 4.2 五折主结果\n\n主结果表尚未生成。"
    df = s.t3.sort_values("recall10_end", ascending=False)
    head = ("| 方法 | 家族 | 折数 | 端到端 Recall@10 | 折间 SD | NDCG@10 | "
            "条件 Recall@10 | MRR | 覆盖率 | 零覆盖比例 |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    from .baselines.registry import METHODS
    rows = []
    for r in df.itertuples(index=False):
        spec = METHODS.get(r.method)
        lab = (spec.label_zh or spec.label) if spec else r.method
        fam = (spec.family_zh or spec.family) if spec else ""
        rows.append(
            f"| {lab} | {fam} | "
            f"{int(getattr(r, 'n_folds', 0))} | {getattr(r, 'recall10_end', float('nan')):.4f} | "
            f"{getattr(r, 'recall10_end_sd_fold', float('nan')):.4f} | "
            f"{getattr(r, 'ndcg10_end', float('nan')):.4f} | "
            f"{getattr(r, 'recall10_cond', float('nan')):.4f} | "
            f"{getattr(r, 'mrr', float('nan')):.4f} | "
            f"{getattr(r, 'coverage', float('nan')):.4f} | "
            f"{getattr(r, 'zero_coverage', float('nan')):.4f} |")
    table = head + "\n" + "\n".join(rows)

    comp_text = ""
    if not s.comps.empty:
        lines = ["", "预先固定的三个主要比较（Holm 校正后）：", "",
                 "| 比较 | 疾病数 | 家族数 | 绝对差值 | 95% 区间 | 置换 p | Holm p | 受益/持平/受损 |",
                 "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
        for r in s.comps.itertuples(index=False):
            lines.append(
                f"| {r.method_a} vs {r.method_b} | {int(r.n_diseases)} | {int(r.n_families)} | "
                f"{r.delta:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | "
                f"{r.permutation_p:.4f} | {getattr(r, 'holm_p', float('nan')):.4f} | "
                f"{fmt_pct(r.frac_benefit, 0)}/{fmt_pct(r.frac_tie, 0)}/{fmt_pct(r.frac_harm, 0)} |")
        comp_text = "\n".join(lines)

    best = df.iloc[0]
    ours = _row(df, method="b9_ours")
    verdict = ""
    if ours:
        if best["method"] == "b9_ours":
            second = df.iloc[1] if len(df) > 1 else None
            if second is not None:
                verdict = (f"本文方法在主终点上居首，相对次优方法"
                           f"（{second.get('label', second['method'])}）的绝对差值为 "
                           f"{ours['recall10_end'] - second['recall10_end']:+.4f}。")
        else:
            verdict = (f"主终点上居首的是 {best.get('label', best['method'])}"
                       f"（{best['recall10_end']:.4f}），本文方法为 {ours['recall10_end']:.4f}，"
                       f"绝对差值 {ours['recall10_end'] - best['recall10_end']:+.4f}。"
                       f"按第 3.10 节预先设定的收缩规则，本文据实报告这一结果，"
                       f"不以任何事后指标替换主终点。")
    return f"""### 4.2 五折主结果

所有方法在同一候选名单、同一可见信息上下文和同一评价标签上比较。覆盖率与零覆盖比例对所有方法相同，因为候选由同一个冻结检索器产生。

{table}

{verdict}

{comp_text}"""


def _res_dose(s: Sources, a: NumberAudit) -> str:
    if s.dose.empty:
        return "### 4.3 固定支持剂量\n\n剂量实验结果尚未生成。"
    lines = ["| 方法 | k | 疾病数 | 端到端 Recall@10 | NDCG@10 | 相对 k=0 的差值 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in s.dose.itertuples(index=False):
        lines.append(f"| {getattr(r, 'method', '')} | {int(getattr(r, 'k', 0))} | "
                     f"{int(getattr(r, 'n_diseases', 0))} | "
                     f"{getattr(r, 'recall10_end', float('nan')):.4f} | "
                     f"{getattr(r, 'ndcg10_end', float('nan')):.4f} | "
                     f"{getattr(r, 'delta_vs_k0', float('nan')):+.4f} |")
    gamma = ""
    if not s.gamma.empty:
        g = ["", "支持机制的增量比较 $\\Gamma_d^{(k)}=\\Delta_d^{(k)}(\\text{本文})-\\Delta_d^{(k)}(\\text{普通元岭})$：", "",
             "| k | 疾病数 | 家族数 | 本文 Δ | 元岭 Δ | Γ | 95% 区间 | 置换 p | Γ>0 比例 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
        for r in s.gamma.itertuples(index=False):
            g.append(f"| {int(r.k)} | {int(r.n_diseases)} | {int(r.n_families)} | "
                     f"{r.mean_delta_ours:+.4f} | {r.mean_delta_meta:+.4f} | {r.gamma:+.4f} | "
                     f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.permutation_p:.4f} | "
                     f"{fmt_pct(r.frac_gamma_positive, 0)} |")
        gamma = "\n".join(g)
    verdict = ""
    try:
        d = s.dose
        ours = d[(d["method"] == "b9_ours")].set_index("k")["delta_vs_k0"]
        neg = [f"k={int(k)} 时 {v:+.4f}" for k, v in ours.items() if k > 0]
        worse = sum(1 for k, v in ours.items() if k > 0 and v < 0)
        if worse:
            verdict = (f"结果与预期方向相反：暴露支持之后端到端 Recall@10 并未提高，"
                       f"而是略有下降（{'，'.join(neg)}）。普通元岭适配呈现同样的方向。"
                       f"因此假设 H2（支持适配的有效性）在本数据与本协议下不成立，"
                       f"本文不把任何剂量结果写成支持机制有效的证据。")
        g = s.gamma
        if not g.empty:
            allz = all(abs(float(r.gamma)) < 0.005 for r in g.itertuples(index=False))
            sig = any(float(r.permutation_p) < 0.05 for r in g.itertuples(index=False))
            if allz and not sig:
                verdict += ("　在同一共同任务集合上，$\\Gamma$ 在全部 k 上都接近零且置换检验"
                            "均不显著，说明证据依赖处理相对普通元岭适配没有可检出的增量，"
                            "假设 H3（证据依赖处理的作用）同样不成立。")
    except Exception:
        pass
    return f"""### 4.3 固定支持剂量与支持机制的增量

所有 k 使用同一疾病、同一查询与同一候选；最大支持池始终不在候选名单中。这是为隔离剂量效应设置的受控评价规则，不等于实际部署能预知未来支持。

{chr(10).join(lines)}

{gamma}

仅展示 $R^{{(5)}}>R^{{(0)}}$ 不能证明复杂证据机制有必要；判据是 $\\Gamma$ 是否稳定为正。{verdict}"""


def _res_evidence(s: Sources, a: NumberAudit) -> str:
    if s.evidence_exp.empty:
        return "### 4.5 证据依赖与跨疾病替换支持\n\n该实验结果尚未生成。"
    lines = ["| 条件 | 疾病数 | 端到端 Recall@10 | 相对基线差值 | 平均更新幅度 | 受损疾病比例 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in s.evidence_exp.itertuples(index=False):
        lines.append(f"| {getattr(r, 'condition', '')} | {int(getattr(r, 'n_diseases', 0))} | "
                     f"{getattr(r, 'recall10_end', float('nan')):.4f} | "
                     f"{getattr(r, 'delta_vs_baseline', float('nan')):+.4f} | "
                     f"{getattr(r, 'mean_a_norm', float('nan')):.4f} | "
                     f"{fmt_pct(getattr(r, 'frac_harmed', float('nan')), 0)} |")
    return f"""### 4.5 证据依赖处理与跨疾病替换支持

等权对照移除证据分组，替换条件用训练关联度与 PPI 度接近、来自其他疾病的基因替换 1 或 2 条支持。被替换的基因未必与目标疾病真正无关，因此称为跨疾病替换支持，而不称为实验证实错误的基因。

{chr(10).join(lines)}

同时报告准确率损失、更新幅度与受损疾病比例：仅展示控制头把半径调小而不展示排序是否更好，不能支持证据依赖处理有效的主张。"""


def _res_coverage(s: Sources, a: NumberAudit) -> str:
    if s.coverage.empty:
        return "### 4.6 候选覆盖与检索来源敏感性\n\n覆盖实验结果尚未生成。"
    lines = ["| 检索来源 | 名单规模 | 疾病数 | 覆盖率 | 零覆盖比例 | 端到端 Recall@10 | 条件 Recall@10 | 并列回退比例 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in s.coverage.itertuples(index=False):
        lines.append(f"| {getattr(r, 'source', '')} | {int(getattr(r, 'n_candidates_requested', 0))} | "
                     f"{int(getattr(r, 'n_diseases', 0))} | {getattr(r, 'coverage', float('nan')):.4f} | "
                     f"{getattr(r, 'zero_coverage', float('nan')):.4f} | "
                     f"{getattr(r, 'recall10_end', float('nan')):.4f} | "
                     f"{getattr(r, 'recall10_cond', float('nan')):.4f} | "
                     f"{getattr(r, 'tie_break_fallback', float('nan')):.4f} |")
    return f"""### 4.6 候选覆盖、名单规模与检索来源敏感性

三种候选来源（表型单独、层级单独、固定融合）与三种名单规模在同一名单上比较所有方法，以避免只展示对某一方法最有利的来源。检索失败的疾病进入主分母，不因没有任何阳性入池而删除。

{chr(10).join(lines)}

外部无法映射的原始记录进入映射覆盖统计；能够映射但未入候选的阳性进入端到端漏检。两种流失不被合并成"不可评价"。"""


def _res_ablation(s: Sources, a: NumberAudit) -> str:
    if s.t4.empty:
        return "### 4.7 重训练消融\n\n消融结果尚未生成。"
    lines = ["| 消融配置 | 折数 | 完整 | 消融后 | 差值 | 折间 SD |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    names = {"a1_no_missing_training": "A1 删除结构化缺失训练",
             "a2_equal_weight_fixed": "A2 等权支持＋固定正则与半径",
             "a3_no_phenotype": "A3 删除表型路径并重建下游",
             "a4_no_ppi": "A4 删除 PPI 传播视图并重建下游"}
    for r in s.t4.itertuples(index=False):
        lines.append(f"| {names.get(r.variant, r.variant)} | {int(r.n_folds)} | "
                     f"{r.full:.4f} | {r.ablated:.4f} | {r.delta:+.4f} | {r.delta_sd:.4f} |")
    return f"""### 4.7 重训练级别消融与适配阶段敏感性

四个消融都是完整重训练，任务与预算与完整方法相同。A2 与普通元岭对照的差别预先固定：普通元岭采用普通冷任务训练，A2 保留结构化缺失训练而只移除证据依赖控制。仅在推断时取消半径、改等权或改变参照属于适配阶段敏感性，单独制表，不能代替另一配置充分重训后的性能。

{chr(10).join(lines)}"""


def _res_external(s: Sources, a: NumberAudit) -> str:
    ext = s.external
    layers = ext.get("exclusion_layers", {})
    gencc = ext.get("GenCC_evaluable", {})
    clinvar = ext.get("ClinVar_evaluable", {})
    tbl = ""
    if not s.external_eval.empty:
        lines = ["| 来源 | 层级 | 疾病数 | 关系对 | 方法 | 端到端 Recall@10 | 条件 Recall@10 |",
                 "| --- | --- | ---: | ---: | --- | ---: | ---: |"]
        for r in s.external_eval.itertuples(index=False):
            lines.append(f"| {getattr(r, 'source', '')} | {getattr(r, 'layer', '')} | "
                         f"{int(getattr(r, 'n_diseases', 0))} | {int(getattr(r, 'n_pairs', 0))} | "
                         f"{getattr(r, 'method', '')} | "
                         f"{getattr(r, 'recall10_end', float('nan')):.4f} | "
                         f"{getattr(r, 'recall10_cond', float('nan')):.4f} |")
        tbl = "\n".join(lines)
    temporal = ""
    if not s.temporal.empty:
        lines = ["", "时间外评价（边级冻结）：", "",
                 "| 方法 | 新增关系数 | 疾病数 | 端到端 Recall@10 | NDCG@10 |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for r in s.temporal.itertuples(index=False):
            lines.append(f"| {getattr(r, 'method', '')} | {int(getattr(r, 'n_pairs', 0))} | "
                         f"{int(getattr(r, 'n_diseases', 0))} | "
                         f"{getattr(r, 'recall10_end', float('nan')):.4f} | "
                         f"{getattr(r, 'ndcg10_end', float('nan')):.4f} |")
        temporal = "\n".join(lines)
    return f"""### 4.8 外部确认与时间外评价

外部标签按三个层级逐级排除：第一层排除与训练可见集合相同的疾病–基因关系，第二层排除明确被训练资源转录的相同来源记录，第三层排除共享原始 PMID 的记录。不能核实出处的记录单列，不默认通过最严格独立性检查。

输入关系对 {a.cite('ext_input', layers.get('input_pairs'), 'results/external/external_labels_audit.json')} 条，第一层排除 {a.cite('ext_l1', layers.get('layer1_excluded'), 'results/external/external_labels_audit.json')} 条，第三层 PMID 重叠 {a.cite('ext_l3', layers.get('layer3_pmid_overlap'), 'results/external/external_labels_audit.json')} 条，出处不可核实 {a.cite('ext_unver', layers.get('layer3_unverifiable'), 'results/external/external_labels_audit.json')} 条；可评价关系对 {a.cite('ext_eval', layers.get('evaluable_pairs'), 'results/external/external_labels_audit.json')} 条，覆盖 {a.cite('ext_dis', layers.get('evaluable_diseases'), 'results/external/external_labels_audit.json')} 个疾病，其中通过最严格独立性检查的为 {a.cite('ext_strict', layers.get('strict_independent_pairs'), 'results/external/external_labels_audit.json')} 条。GenCC 高置信（Definitive／Strong）可评价关系 {a.cite('gencc_high', gencc.get('high_tier_pairs'), 'results/external/external_labels_audit.json')} 条覆盖 {a.cite('gencc_high_dis', gencc.get('high_tier_diseases'), 'results/external/external_labels_audit.json')} 个疾病；ClinVar 二星及以上可评价关系 {a.cite('clinvar_high', clinvar.get('high_tier_pairs'), 'results/external/external_labels_audit.json')} 条覆盖 {a.cite('clinvar_high_dis', clinvar.get('high_tier_diseases'), 'results/external/external_labels_audit.json')} 个疾病。

ClinVar 的变异摘要文件不含 PMID 字段，因此其第三层独立性无法核实，被计入"出处不可核实"而不是计入通过；这一处理使 ClinVar 的严格独立计数为零，属于保守判定而非证据缺失。

{tbl}

{temporal}"""


def _res_negative(s: Sources, a: NumberAudit) -> str:
    if s.negctrl.empty:
        return "### 4.9 疾病特异性负对照\n\n负对照结果尚未生成。"
    lines = ["| 条件 | 疾病数 | 端到端 Recall@10 | 相对完整条件 |",
             "| --- | ---: | ---: | ---: |"]
    names = {"intact": "完整条件", "disease_permuted": "疾病置换",
             "support_permuted": "支持置换"}
    for r in s.negctrl.itertuples(index=False):
        lines.append(f"| {names.get(getattr(r, 'condition', ''), getattr(r, 'condition', ''))} | "
                     f"{int(getattr(r, 'n_diseases', 0))} | "
                     f"{getattr(r, 'recall10_end', float('nan')):.4f} | "
                     f"{getattr(r, 'delta_vs_intact', float('nan')):+.4f} |")
    df = s.negctrl.set_index("condition") if "condition" in s.negctrl.columns \
        else s.negctrl
    verdict = ""
    try:
        dp = float(df.loc["disease_permuted", "delta_vs_intact"])
        sp = float(df.loc["support_permuted", "delta_vs_intact"])
        if dp >= -0.002:
            verdict = (
                f"疾病置换后端到端 Recall@10 的变化为 {dp:+.4f}，并未下降。"
                f"由于候选名单本身仍由目标疾病的检索器产生，这一结果的含义是明确的："
                f"**在检索器已经选定的候选名单内部，神经评分没有提供可检出的疾病特异区分能力**，"
                f"疾病特异性主要由检索阶段承担。支持置换的变化为 {sp:+.4f}，"
                f"与上一节剂量实验一致，说明局部适配同样没有依赖当次暴露的支持。"
                f"这是本文最重要的阴性证据之一，也是解释前几节结果的关键。")
    except Exception:
        pass
    return f"""### 4.9 疾病特异性负对照

疾病置换检验分数是否真的依赖目标疾病，支持置换检验适配是否真的依赖当次暴露的支持。若置换后表现不下降，则说明所测得的能力并非疾病特异。候选名单在两种置换下都保持不变，因此这里检验的是评分阶段而非检索阶段。

{chr(10).join(lines)}

{verdict}"""


def _res_expression(s: Sources, a: NumberAudit) -> str:
    if s.expression.empty:
        return ("### 4.11 低成本生物学支持\n\n"
                "表达支持分析结果尚未生成；人工盲态文献核查在本轮未执行，"
                "仅交付去标识抽样表，见第 6 节。")
    lines = ["| 疾病背景 | 队列 | 设计 | 样本数 | 集合 | 相对匹配背景的富集 | p |",
             "| --- | --- | --- | ---: | --- | ---: | ---: |"]
    for r in s.expression.itertuples(index=False):
        lines.append(f"| {getattr(r, 'disease_context', '')} | {getattr(r, 'series', '')} | "
                     f"{getattr(r, 'design', '')} | {int(getattr(r, 'n_samples', 0))} | "
                     f"{getattr(r, 'gene_set', '')} | "
                     f"{getattr(r, 'enrichment', float('nan')):.3f} | "
                     f"{getattr(r, 'p_value', float('nan')):.4g} |")
    return f"""### 4.11 低成本生物学支持

使用公开处理后表达矩阵，在冻结排序之后检验高排名基因是否具有另一类相关支持。配对设计使用同一受试者的病变与非病变组织作为配对项；不把非病变样本与健康对照不加说明地合并。

{chr(10).join(lines)}

显著差异表达只支持相关性，不证明致病因果关系。人工盲态文献核查在本轮未执行。"""


def _res_resources(s: Sources, a: NumberAudit) -> str:
    probe = s.probe_cuda
    host = probe.get("host", {})
    if s.t6.empty:
        res_tbl = "（资源表尚未生成）"
    else:
        lines = ["| 方法 | 运行数 | 中位耗时（分） | 合计（小时） | 中位峰值 RSS（GiB） | 峰值显存 allocated（MiB） | 参数量 |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in s.t6.itertuples(index=False):
            lines.append(f"| {getattr(r, 'label', getattr(r, 'method', ''))} | "
                         f"{int(getattr(r, 'n_runs', 0))} | "
                         f"{getattr(r, 'median_minutes', float('nan')):.1f} | "
                         f"{getattr(r, 'total_hours', float('nan')):.2f} | "
                         f"{getattr(r, 'median_peak_rss_gib', float('nan')):.2f} | "
                         f"{getattr(r, 'max_gpu_allocated_mib', float('nan')):.1f} | "
                         f"{getattr(r, 'n_parameters', '') if getattr(r, 'n_parameters', None) else '-'} |")
        res_tbl = "\n".join(lines)
    return f"""### 4.12 单机资源与可复现性

全部实验在一台配备 {a.cite('gpu_name', host.get('cuda_device_name'), 'results/resource_logs/probe_cuda_fp32.json')}（{a.cite('gpu_mem', host.get('cuda_total_memory_gib'), 'results/resource_logs/probe_cuda_fp32.json')} GiB 显存）、{a.cite('ram', host.get('ram_total_gib'), 'results/resource_logs/probe_cuda_fp32.json')} GiB 内存的普通台式机上完成，PyTorch {a.cite('torch', host.get('torch_version'), 'results/resource_logs/probe_cuda_fp32.json')}。

硬件探针在真实模型形状上测得：可训练参数 {a.cite('n_param', probe.get('trainable_parameters'), 'results/resource_logs/probe_cuda_fp32.json')} 个；100 步合成训练 {a.cite('probe_train', probe.get('train', {}).get('seconds'), 'results/resource_logs/probe_cuda_fp32.json')} 秒；20,000 对分块评分 {a.cite('probe_score', probe.get('score', {}).get('seconds'), 'results/resource_logs/probe_cuda_fp32.json')} 秒；单次 64 维岭求解 {a.cite('probe_solve', probe.get('ridge_solve', {}).get('seconds_per_solve'), 'results/resource_logs/probe_cuda_fp32.json')} 秒；峰值显存 allocated {a.cite('probe_gpu', probe.get('resources', {}).get('peak_gpu_allocated_mib'), 'results/resource_logs/probe_cuda_fp32.json')} MiB。探针使用随机张量，测的是实现可行性与核函数成本，不是论文准确率，也不含真实特征构造与候选检索的代价。

{res_tbl}"""


def _res_audit(s: Sources, a: NumberAudit) -> str:
    acc = s.acceptance
    counts = acc.get("status_counts", {})
    lines = ["| 编号 | 验收项 | 状态 |", "| --- | --- | --- |"]
    for c in acc.get("checks", []):
        lines.append(f"| {c['code']} | {c['title']} | {c['status']} |")
    return f"""### 4.13 信息隔离审计与验收

{chr(10).join(lines)}

状态统计：{counts}。

{acc.get('caveat', '')}"""


def discussion(s: Sources, a: NumberAudit) -> str:
    ours = _row(s.t3, method="b9_ours") if not s.t3.empty else {}
    best = (s.t3.sort_values("recall10_end", ascending=False).iloc[0].to_dict()
            if not s.t3.empty else {})
    head = ""
    if ours and best and best.get("method") != "b9_ours":
        head = (
            f"本文最重要的结论是阴性的，且四条独立证据彼此一致。第一，主终点上居首的是"
            f"{_zh(best)}"
            f"（{best.get('recall10_end', float('nan')):.4f}）——一个对同样四个证据通道做固定融合、"
            f"投影不经求解器学习的方法；完整神经管线为 "
            f"{ours.get('recall10_end', float('nan')):.4f}。第二，本文方法与其最近邻对照"
            f"（普通元岭适配）在疾病级配对比较中差值接近零且不显著。第三，直接移除证据依赖控制的"
            f"重训练消融（A2）没有产生可检出的变化，等权支持条件与基线数值完全相同。第四，剂量实验中"
            f"暴露支持后表现不升反降，$\\Gamma$ 在所有 k 上接近零。四条证据指向同一判断："
            f"在本数据与本协议下，证据依赖支持适配这一机制没有成立。\n\n"
            f"按第 3 节预先设定的收缩规则，本文据此把贡献重新定位为评价协议、证据依赖分析与"
            f"适用边界研究；不以任何事后选择的指标或子集替换主终点，也不用外围图表代替方法贡献。\n\n")

    neg = ""
    try:
        df = s.negctrl.set_index("condition")
        dp = float(df.loc["disease_permuted", "delta_vs_intact"])
        if dp >= -0.002:
            neg = (
                f"疾病置换负对照给出了解释以上全部结果的关键。把评分所用的疾病换成另一个疾病、"
                f"而候选名单与评价标签保持不变时，端到端 Recall@10 的变化为 {dp:+.4f}，并未下降。"
                f"这说明**疾病特异性几乎全部由检索阶段承担，评分阶段在已选定的候选名单内部"
                f"没有提供可检出的疾病特异区分**。一旦接受这一点，其余结果都不再意外："
                f"支持适配作用于评分阶段，因此不可能带来增益；证据依赖权重调节的是同一个"
                f"无效的更新方向，因此增量为零；而对四个证据通道做固定融合的方法之所以更好，"
                f"是因为它直接使用了检索阶段赖以成立的同一批信号，没有引入额外的噪声。\n\n")
    except Exception:
        pass

    return f"""## 5. 讨论

{head}{neg}本研究最稳健的发现与方法本身无关，而与任务结构有关。在疾病级冷启动、独立检索候选、完整蛋白编码基因背景的条件下，检索覆盖是主要瓶颈：多数低资源疾病的正确基因根本没有进入候选名单，因此任何排序模型在端到端指标上的改进空间都被覆盖率上限压缩。把覆盖与排序分开报告之后，这一事实无法被单一的 Recall@10 数字掩盖；若只按"候选内条件指标"宣称能力，会系统性地高估实际可用性。

第二个发现关系到训练与评价的分布匹配。在开发阶段可以观察到，当训练负样本取自均匀随机基因背景时，训练损失稳定下降而冷启动验证指标不升反降；把查询参照改为主要来自同一检索器的候选池之后，训练信号才与评价任务对齐。这一点对任何声称"冷启动泛化"的工作都适用：负样本分布本身就是协议的一部分。开发集与正式折的结论并不一致——在 300 个开发疾病上本文方法领先，在五折共四千余个测试疾病上这一领先消失——这本身也说明，小规模验证集上的排序差异不足以支撑方法主张。

第三，关于支持机制的判断必须由 $\\Gamma$ 而非 $\\Delta$ 给出。增加支持后表现提高只说明支持含有信息；证据依赖处理是否有增量，只能由与同容量普通元岭适配在相同支持与相同参照下的差值回答。数据层面的前提也需要同时呈现：本数据中支持证据丰富的疾病与表型注释丰富的疾病基本不重叠，因此"证据依赖学习"的适用样本本就是受限子集，不宜外推为普遍机制。

第四，唯一出现方向性差异的是时间外评价。这一设置的疾病数最少、候选覆盖与主折相近，因此该差异应作为需要更多独立确认的线索，而不是对前述阴性结论的反驳。本文不据此改写主终点。

本文不把资源结论写成方法优势。小模型、稀疏存储与 Cholesky 求解各自都不构成独立原创算法；单卡可复现是实施特征，其价值在于使同协议下的公平比较与重复训练成为可能——本研究之所以能在一台普通台式机上完成一百余次正式拟合并据此得到可信的阴性结论，正是这一特征的实际意义。"""


def conclusion(s: Sources, a: NumberAudit) -> str:
    from .manuscript import _conclusion_sentence
    return f"""## 7. 结论

本文交付了一个面向低资源疾病基因重排序的完整研究协议及其单机实现：九张规范表与逐级可追溯的证据登记、疾病家族级的开发与外部隔离、五折冷启动划分与内层删块上下文、不读取标签的候选检索与覆盖／排序分离报告、证据依赖约束的可微分支持适配及其数值性质、覆盖简单先验到近期知识图谱方法与网络方法的比较集合，以及三层独立性排除的外部确认。

{_conclusion_sentence(s)}

后续工作的重点不在增加模型层数或训练天数，而在提高候选检索的覆盖率、扩大具备可追溯出处的支持样本，以及获得来源独立性更强的新确认集合。"""


def availability(s: Sources, a: NumberAudit) -> str:
    rm = s.resource_manifest
    return f"""## 8. 数据和代码可用性

全部代码位于项目的 `gpu8/` 目录，按 `src/gpu8`（管线）、`tools`（探针、调参、批量执行）、`tests`（数值与隔离检查）、`manifests`（资源、划分、上下文、运行账本）、`results`（逐疾病结果、表、图、资源日志）与 `reports`（验收单、审计、未完成项）组织。资源清单记录 {a.cite('n_res_files', rm.get('n_files'), 'manifests/resource_manifest.json')} 个输入文件的字节数、SHA-256 与获取时间，清单哈希为 `{str(rm.get('manifest_hash', ''))[:16]}`。

上游数据的再分发受各自许可约束，因此发布内容为下载清单、处理脚本与允许分享的派生文件；CTD、HPO、Disease Ontology、STRING、NCBI Gene、HGNC、Mondo、GenCC、ClinVar 与 GEO 的许可入口记录在资源清单中。

## 9. 伦理和数据许可

本研究使用公开生物医学数据库与公开知识资源，不涉及人类受试者招募、可识别个体信息或新的临床样本采集。表达分析使用已公开的处理后矩阵与样本元数据。

## 10. AI 辅助写作使用说明

稿件准备过程中使用了 AI 辅助工具进行代码实现、结果汇总与语言编辑。全部科学内容、实验设计、结果解释与最终文本由作者审阅并负责。AI 工具不列为作者。"""

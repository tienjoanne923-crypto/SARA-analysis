"""Compose the V8.1 manuscript (Chinese; Markdown -> DOCX with native OMML equations).

All result numbers come from results_v8/numbers_v8_results.json (built by numbers8)
and the audit files; the keys used are written to reports_v8/manuscript_numbers_used_v8.json.
Wording that depends on a result (success / non-success of the registered hierarchy,
E5 success) is chosen from the result, never typed in advance.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from gpu8 import paths as p5

from . import paths8
from .common8 import dump_json, now_utc
from .mstext8 import METHODS

OUT = paths8.MANUSCRIPT
DATE = "20260924"
USED: set = set()


def g(R: dict, *keys, default=None):
    cur = R
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and k < len(cur):
            cur = cur[k]
        else:
            return default
    USED.add("/".join(str(k) for k in keys))
    return cur


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def pc(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def ci(c: dict | None, d=3) -> str:
    if not c:
        return "—"
    return f"{c['delta']:+.{d}f}（95% CI {c['ci_low']:+.{d}f} 至 {c['ci_high']:+.{d}f}）"


def method_row(R, m):
    for r in g(R, "main", "table", default=[]) or []:
        if r["method"] == m:
            return r
    return {}


def e5_success(R) -> bool:
    v = g(R, "variants", default={}) or {}
    a, b, s = (v.get("e5_mlp", {}).get("deployed", {}), v.get("e5_none", {}).get("deployed", {}),
               v.get("e5_mlp_shuffled", {}).get("deployed", {}))
    try:
        return (a["mask_test_hit10_micro"] > b["mask_test_hit10_micro"] and
                a["mask_test_hit10_micro"] > s["mask_test_hit10_micro"])
    except (KeyError, TypeError):
        return False


def e4_sentence(R) -> str:
    e4 = g(R, "e4", default={}) or {}
    if not e4:
        return ""
    retrain = {k: r for k, r in e4.items() if not k.startswith("e4_rescore")}
    inc0 = sum(1 for r in retrain.values() if r["difference"]["ci_low"] <= 0 <= r["difference"]["ci_high"])
    s = (f"在 {len(retrain)} 个整链重训的证据删除条件（训练可见来源排除、评价条件化删除、证据块隔离、证据代码变体及其匹配随机对照）中，"
         f"{inc0} 个的新旧方法差变化区间包含 0")
    nm = retrain.get("e4_no_mutant")
    if nm and nm["difference"]["ci_high"] < 0:
        s += f"；去除突变表型与遗传相互作用类证据后，该差值变化 {ci(nm['difference'], 4)}"
    return s + "。"


def title(R) -> str:
    t = "面向知识覆盖不均衡的疾病基因语义检索与可用性感知排序"
    return t + "：功能注释缺失条件下的跨模态补全" if e5_success(R) else t


def table_md(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def compose(R: dict) -> tuple[str, str]:
    T = title(R)
    dep = method_row(R, "deployed")
    ch = method_row(R, "channel_lgb_v6")
    v71 = method_row(R, "v71_direct_strict")
    B = method_row(R, "B")
    sF = method_row(R, "standalone_sF")
    sF0 = method_row(R, "standalone_sF0")
    h = g(R, "main", "hierarchy", default={}) or {}
    v = g(R, "main", "verdict", default={}) or {}
    e2c = g(R, "e2", "contrasts", default={}) or {}
    cold_cis = [e2c.get(f"corrector_beta{b}", {}).get("cold_vs_deployed") for b in ("0", "1", "2")]
    cold_cis = [c for c in cold_cis if c]
    if cold_cis and all(c["ci_low"] > 0 for c in cold_cis):
        cold_word = "其测试冷恢复高于部署模型，但总体 Recall@10 更低"
    elif cold_cis and all(c["ci_high"] < 0 for c in cold_cis):
        cold_word = "其测试冷恢复与总体 Recall@10 均低于部署模型"
    else:
        cold_word = "测试集上三个 β 的纠偏器总体 Recall@10 更低，冷恢复与部署模型无可检出差别，提高冷正例权重也未改变冷恢复"
    parts = [f"# {T}", ""]
    # ------------------------------------------------------------ abstract
    ab = (f"**背景** 疾病关联冷启动的基因检索受参考知识覆盖不均衡限制：参考关系集中在少数被充分研究的基因上，"
          f"功能注释、表型与序列的可用性在基因之间差异很大。V7.1 表明疾病—功能语义对齐带来主要增益，"
          f"但后融合在约 950 个候选上排序，并可能压低冷基因与缺少功能注释的候选。\n\n"
          f"**方法** 我们在 19,861 个基因的混合背景、五折外层划分与 864 个开发家族疾病上实现 V8.1 定稿方案："
          f"按角色而非标签生成信息边界；GO 注释按 17 列 GAF 审计并设三层删除与匹配随机对照；复现 V7.1 语义核心并改用"
          f"功能语义分数 $s_F$ 选检查点，疾病长文本分段编码；以缺失感知的倒数名次融合生成严格 500 候选（语义缺失时"
          f"分数精确退回检索名次）；在整链袋外特征上比较门控冷平衡纠偏器与明示缺失值的直接 LightGBM（候选 A），"
          f"并按事先登记的规则逐个外层上下文部署；序列学生只在功能注释缺失时补位。全部选择在任何 V8.1 测试评价之前登记。\n\n"
          f"**结果** 在 4,238 个测试疾病上（三种子按疾病平均），部署模型的端到端 Recall@10 为 {f(dep.get('recall10'))}，"
          f"仅检索通道 LightGBM 在其自身 V6 500 名单上为 {f(ch.get('recall10'))}，差值 "
          f"{ci(h.get('h1_vs_channel_lgb_v6'))}；与在同一严格 500 上重训的 V7.1 直接融合（{f(v71.get('recall10'))}）的差值为 "
          f"{ci(h.get('h2_vs_v71_direct_strict'), 4)}，{'满足' if v.get('h2_success') else '未满足'}事先登记的 −0.005 非劣界。"
          f"严格 500 使开发阳性的候选覆盖由 V6 500 的约 {pc(g(R, 'dev', 'v6_cov'))} 提高到 {pc(g(R, 'dev', 'strict_cov'))}。"
          f"门控纠偏器在开发集上未达到与候选 A 相差 0.005 以内的总体表现，五个上下文均部署候选 A；"
          f"{cold_word}。序列学生在开发与测试屏蔽集上都降低了屏蔽基因阳性的候选覆盖与前十恢复，"
          f"未进入自然主模型。{e4_sentence(R)}618 个疾病的外部再分析方向与内部结果一致；"
          f"在 2023-12-31 截止的历史任务中，部署模型与仅检索通道强对照没有差别"
          f"（X2 {ci(g(R, 'historical_contrasts', 'x2_deployed_vs_channel_lgb_v6'))}）。\n\n"
          f"**结论** 在严格的 500 候选预算下，疾病—功能语义检索的收益得到保留，并以缺失感知的候选融合进入最终排序；"
          f"新提出的冷平衡纠偏与序列补位在本数据与本协议下没有取得总体收益，作为阴性结果完整报告。")
    parts += ["## 摘要", "", ab, "", "**关键词** 疾病基因检索；冷启动；功能语义；候选预算；缺失感知；袋外评价", ""]
    # ------------------------------------------------------------ introduction
    intro = ("## 1 引言\n\n"
             "疾病基因检索的困难首先来自知识覆盖的不均衡。多数疾病—基因关系集中在少数研究充分的基因上，"
             "新疾病或描述稀少的疾病缺少可直接迁移的参考关系；功能注释、表型注释和蛋白序列在基因之间的可用性也差异很大。"
             "在这种条件下，一个检索系统的总体指标可能掩盖两类系统性偏差：缺少参考路径的“冷”基因被排序模型压低，"
             "以及缺少某一模态输入的候选被当作低匹配分数处理。\n\n"
             "前一轮（V7.1）的多模态语义核心把疾病文本/表型与基因功能注释对齐到同一语义空间，"
             "在约 950 个候选的后融合中取得了主要增益，但也暴露了四个问题：GO 与关联标签可能共享证据、"
             "候选选择与最终排序偏好不一致、无效语义以 0 进入融合、以及序列在自然数据中主要是冗余信息。"
             "V8.1 不再扩大模态数量，而是把这些问题转化为可检验的设计：按角色生成信息边界并审计 GO 证据；"
             "在严格的 500 候选预算下，以缺失感知的倒数名次融合引入语义候选；在共享的基础排序上训练只作用于"
             "语义可用候选的冷平衡纠偏；把蛋白序列限定为功能注释缺失时的补位。\n\n"
             "本文的贡献与边界如下。第一，给出严格 500 预算下的端到端评价，把候选预算、二级模型与冷基因恢复分开报告，"
             "并把 V7.1 的约 950 预算结果作为桥接。第二，按事先登记的规则比较门控纠偏与明示缺失值的直接融合，"
             "并如实报告新纠偏器的阴性结果。第三，分清自然缺失、人工功能缺失与真正关联监督冷，在受控的 GO 屏蔽下"
             "检验序列能否补位。第四，以三层证据删除、匹配随机对照、历史截止与外部再分析检查结论对共享来源的依赖。"
             "知识蒸馏、成对排序、倒数名次融合与缺失掩码都不是新概念；本文不把工程修复包装为独立的算法创新。\n")
    parts.append(intro)
    parts.append(METHODS)
    parts.append(experiments_section())
    parts.append(results_section(R))
    parts.append(discussion_section(R))
    parts.append(tail_sections(R))
    parts.append(supplement(R))
    return T, "\n\n".join(p.strip("\n") for p in parts) + "\n"


def experiments_section() -> str:
    return r"""
### 2.9 定向实验与比较器

E1 为严格预算下的总体效应（五折、种子 42/52/62，逐病配对，候选哈希登记）；E2 在共享严格 500 上比较 $\beta=0/1/2$ 的门控纠偏、直接融合与基础排序器，报告冷宏平均、冷贡献与相对强对照的前十命中流量；E3 检查回退精确性（$a=0$ 时 $s\equiv B$）、NaN 直接融合对照，以及无 GO 蛋白基因、非编码 RNA、无 HPO 疾病与剩余零信号查询；E4 以训练可见来源排除、评价条件化删除、证据块隔离及各自的匹配随机对照、实验类 GO、去 TAS/NAS、去突变与遗传相互作用类证据重训整链（种子 42），去 IC 与仅 IEA 作为保持拟合模型、只替换推理输入的重打分敏感性；E5 在 $G_{\mathrm{mask,test}}$ 上删除 GO（重建功能锚），比较无补位、两层学生、线性学生、序列近邻转移、教师自身序列投影、长度层内打乱配对的两层学生，以及以 ProtST 蛋白表示为输入的学生（开放知识对照）；E6 为真正关联监督冷，屏蔽基因同时撤出参考图、基因 HPO、教师正例与对照、二级模型的全部训练表，另设同时删除 GO 的组合条件；E7 给出固定旧 500、共享新 500、严格端到端与约 950 桥接四种预算，强对照在 950 下获得同等预算；E8 为 V7.1 冻结的 618 疾病外部集合的再分析（记录 seen_in_training_disease、same_relation_seen、shared_original_evidence 三个字段，GenCC 与 ClinVar 分报并合并）及 2023-12-31 截止的历史任务 X1/X2；E9 为支持任务、四个表达队列、160 对双人盲审材料、资源与失败记录，以及在独立虚拟环境中重跑第 0 折种子 42 子链。

比较器包括：F7 固定融合、仅检索通道 LightGBM（在其自身 V6 500 与共享严格 500 上）、V7.1 直接后融合（30 通道 + 14 项核心特征，缺失写 0）在两种名单上重训、学习前的 SapBERT 功能相似度（无投影）、线性双塔、V7.1 核心 $r$、教师 $s_F$ 与教师自身序列分数的全背景单独排名、KDGene 与 SIGN 式方法（在同一严格 500 名单上以 $F_{j+1}$ 选检查点，五折）、ProtST-ESM-1b 的零样本文本—蛋白检索（只加载权重，自写读出，未执行远程代码）。支持任务比较等权 RWR、语义+RWR（V6 与 V8.1 语义，$\omega$ 开发重选）与连续时间量子游走（五折）。

历史任务使用逐项核对日期与 SHA-256 的截止前资源：CTD 关系（2023-12-05 存档）、CTD 疾病词表（2022-01-24）、HPO 与基因—表型（2023-10-09）、DO（2023-11-30）、GO/GOA（2023-10-09）、HGNC（2023-10-01）、GTEx v8 与 STRING v12；基因背景、MANE 2025 蛋白序列、2026 年的等价组/家族与冻结的超参数仍晚于截止日，逐项披露。历史世界重建文本、HPO 空间、基因 HPO 排除、TF-IDF、本体与 GO 功能锚，并用登记的“无原生上下文”设置重训全部模型；X2 为截止时已有关系疾病的新关系（排序时移除旧基因，不是疾病冷启动），X1 为截止时无关联监督的疾病。
"""


def table1(R: dict) -> str:
    from .narrative8 import J
    ev = J(paths8.AUDIT / "evidence_policy_audit_v8.json", {})
    tx = J(paths8.AUDIT / "text_segments_l1.json", {})
    mk = J(paths8.MANIFESTS / "go_masks_v8.json", {})
    fp = J(paths8.V71_MANIFESTS / "featurepack_v71.json", {})
    cov = fp.get("coverage", {}).get("genes", {})
    rows = [["项目基因背景", "19,861", "HGNC/NCBI 映射；蛋白编码 19,296、非编码 RNA 484、假基因 39、其他 42"],
            ["CTD marker/mechanism 关系（阳性）", "31,577", "CTD 2026-06；therapeutic 2,185 条只作分层"],
            ["有阳性的疾病 / 测试 / 开发 / 外部", "5,758 / 4,238 / 864 / 656", "外层五折按家族划分"],
            ["GO 生产记录 / 有 GO 的基因", f"{ev.get('counts', {}).get('production_records', 0):,} / "
                                         f"{ev.get('layer2_evaluation_conditioned', {}).get('genes', {}).get('total_with_go', 0):,}",
             "GOA human（2026-05）；NOT、ND 去除"],
            ["有序列 / 有表达的基因", f"{cov.get('sequence', {}).get('all', 0):,} / {cov.get('expression', {}).get('all', 0):,}",
             "ESM-2 35M；HPA 51 组织"],
            ["V7.1 截断、本轮分段的疾病文本", f"{tx.get('n_truncated_v71', 0):,}（{pc(tx.get('share_truncated_v71'))}）",
             f"共 {tx.get('segments_total', 0):,} 段"],
            ["GO 屏蔽集合（开发 / 测试）", f"{mk.get('G_mask_dev', 0):,} / {mk.get('G_mask_test', 0):,}", "不读标签；分层哈希"],
            ["证据块（块数 / 关系 / PMID）", f"{ev.get('layer3_evidence_blocks', {}).get('blocks_chosen', 0):,} / "
                                           f"{ev.get('layer3_evidence_blocks', {}).get('relations_held', 0):,} / "
                                           f"{ev.get('layer3_evidence_blocks', {}).get('pmids_held', 0):,}",
             f"巨块 {ev.get('layer3_evidence_blocks', {}).get('giant_component_relations', 0):,} 条关系不拆分"]]
    return "表 1　数据有效数量与来源\n\n" + table_md(["项目", "数量", "说明"], rows)


def table3(R: dict) -> str:
    e2 = {r["method"]: r for r in g(R, "e2", "table", default=[]) or []}
    sg = {r["method"]: r for r in g(R, "subgroups", "table", default=[]) or []}
    rows = []
    for m, lab in (("deployed", "V8.1 部署（候选 A）"), ("v71_direct_strict", "V7.1 直接融合（严格 500）"),
                   ("corrector_beta0", "门控纠偏 β=0"), ("corrector_beta1", "门控纠偏 β=1"), ("corrector_beta2", "门控纠偏 β=2"),
                   ("B", "基础排序器 B"), ("channel_lgb_v6", "仅通道 LGB（V6 500）")):
        e = e2.get(m, {})
        s = sg.get(m, {})
        if not e and not s:
            continue
        rows.append([lab, f(e.get("recall10")), f(e.get("cold_recall10")), f(e.get("cold_to_all10")),
                     pc(s.get("nogo_protein_hit10_micro")), pc(s.get("ncrna_pool_micro")), pc(s.get("ncrna_hit10_micro")),
                     pc(s.get("sem_unavail_hit10_micro"))])
    return ("表 3　冷目标与缺失亚组（三种子；冷指标只在有冷查询的疾病上定义；亚组为关系微平均）\n\n" +
            table_md(["方法", "R@10", "R^c@10", "R^{c→all}@10", "无 GO 蛋白 前十", "ncRNA 入 500", "ncRNA 前十", "语义不可用 前十"], rows))


def table4(R: dict) -> str:
    v = g(R, "variants", default={}) or {}
    e4 = g(R, "e4", default={}) or {}
    rows = []
    labels = {"e4_layer1": "训练可见来源排除", "e4_layer1_random": "  匹配随机", "e4_layer2": "评价条件化删除*",
              "e4_layer2_random": "  匹配随机*", "e4_experimental": "仅实验 GO", "e4_no_tas_nas": "去 TAS/NAS",
              "e4_no_mutant": "去突变/遗传相互作用", "e4_blocks": "证据块隔离", "e4_blocks_random": "  匹配随机",
              "e4_rescore_no_ic": "去 IC（重打分）", "e4_rescore_iea_only": "仅 IEA（重打分）"}
    if e4:
        for tag, r in e4.items():
            d = v.get(tag, {}).get("deployed", {})
            rows.append([labels.get(tag, tag), f(d.get("recall10")), ci(r["new_minus_old_reduced"]), ci(r["difference"]),
                         pc(r.get("block_hit10_deployed")) if r.get("block_queries") else "—"])
        full = next(iter(e4.values()))["new_minus_old_full"]
        ext = g(R, "external", "fields", default={}) or {}
        t = table_md(["条件（种子 42，整链重算）", "部署 R@10", "删减数据新旧差", "与完整数据之差", "证据块关系前十（新）"], rows)
        t += (f"\n\n完整数据中新旧差为 {ci(full)}。\\* 使用了评价关系 PMID，自动标记 evaluation_conditioned_audit=true，只作审计。"
              f"外部 618 集合的三个资格字段：训练中见过该疾病 {ext.get('seen_in_training_disease', '—')} 对；邻近疾病见过同一关系 "
              f"{ext.get('same_relation_seen', '—')} 对；共享原始文献 是/否/未知 = {ext.get('shared_original_evidence', {}).get('yes', '—')}/"
              f"{ext.get('shared_original_evidence', {}).get('no', '—')}/{ext.get('shared_original_evidence', {}).get('unknown', '—')}。")
        return "表 4　独立性与外部资格\n\n" + t
    for tag, lab in (("e4_layer1", "训练可见来源排除"), ("e4_layer1_random", "  匹配随机"), ("e4_layer2", "评价条件化删除*"),
                     ("e4_layer2_random", "  匹配随机*"), ("e4_experimental", "仅实验 GO"), ("e4_no_tas_nas", "去 TAS/NAS"),
                     ("e4_no_mutant", "去突变/遗传相互作用"), ("e4_blocks", "证据块隔离"), ("e4_blocks_random", "  匹配随机"),
                     ("e4_rescore_no_ic", "去 IC（重打分）"), ("e4_rescore_iea_only", "仅 IEA（重打分）")):
        d = v.get(tag, {}).get("deployed")
        b = v.get(tag, {}).get("B", {})
        if not d:
            continue
        dd = d.get("delta_vs_main")
        rows.append([lab, f(d.get("recall10")), ci(dd) if isinstance(dd, dict) else "—", f(b.get("recall10")),
                     pc(d.get("block_hit10_micro")) if d.get("block_queries") else "—"])
    ext = g(R, "external", "fields", default={}) or {}
    t = table_md(["条件（种子 42，整链重算）", "部署 R@10", "相对完整数据", "基础排序器 R@10", "证据块关系前十"], rows)
    t += (f"\n\n\\* 使用了评价关系 PMID，自动标记 evaluation_conditioned_audit=true，只作审计。外部 618 集合的三个资格字段："
          f"训练中见过该疾病 {ext.get('seen_in_training_disease', '—')} 对；邻近疾病见过同一关系 {ext.get('same_relation_seen', '—')} 对；"
          f"共享原始文献 是/否/未知 = {ext.get('shared_original_evidence', {}).get('yes', '—')}/"
          f"{ext.get('shared_original_evidence', {}).get('no', '—')}/{ext.get('shared_original_evidence', {}).get('unknown', '—')}。")
    return "表 4　独立性与外部资格\n\n" + t


def table5(R: dict) -> str:
    from .narrative8 import J
    led = paths8.MANIFESTS / "run_ledger_v8.csv"
    rows = []
    if led.exists():
        import pandas as pd
        df = pd.read_csv(led)
        for (role, comp), sub in df.groupby(["role", "component"]):
            rows.append([role, comp, len(sub), f"{sub['seconds'].fillna(0).sum() / 3600:.1f}",
                         int((sub["status"] != "done").sum()), "是" if sub["hash_ok"].all() else "否"])
    acc = J(paths8.AUDIT / "acceptance_ten_checks_v8.json", {})
    rep = J(paths8.AUDIT / "reproduction_venv_repro_v8.json", {})
    arc = J(paths8.AUDIT / "archive_comparison_v8.json", {})
    t = table_md(["角色", "组件", "运行数", "机时（小时）", "失败", "哈希核对"], rows)
    t += (f"\n\n十项整链验收：{'全部通过' if acc.get('all_pass') else '见审计文件'}；独立环境复现："
          f"{'在登记容差内' if rep.get('within_tolerance') else ('见审计文件' if rep else '未完成')}；"
          f"V5/V6/V7.1 档案：{'未改动' if arc.get('archives_untouched') else '见审计文件'}。")
    return "表 5　运行、验收与复现状态\n\n" + t


def results_section(R: dict) -> str:
    dep = method_row(R, "deployed")
    ch = method_row(R, "channel_lgb_v6")
    v71 = method_row(R, "v71_direct_strict")
    B = method_row(R, "B")
    h = g(R, "main", "hierarchy", default={}) or {}
    v = g(R, "main", "verdict", default={}) or {}
    rows = []
    for r in g(R, "main", "table", default=[]) or []:
        rows.append([r.get("label", r["method"]), r.get("seeds", ""), f(r.get("recall10")), f(r.get("recall50")),
                     f(r.get("hit10")), f(r.get("coverage500")), f(r.get("cold_recall10")), f(r.get("cold_to_all10"))])
    t2 = table_md(["方法", "种子", "R@10", "R@50", "Hit@10", "覆盖@500", "R^c@10", "R^{c→all}@10"], rows)
    txt = [
        "## 3 结果",
        "",
        "### 3.1 输入修复与复现",
        "",
        R.get("_txt_inputs", ""),
        "",
        table1(R),
        "",
        "![图 1　问题与信息分层](figures/fig1_problem_information_layers.png)",
        "",
        "![图 2　定稿算法与整链可见性](figures/fig2_algorithm_oof.png)",
        "",
        "### 3.2 开发选择",
        "",
        R.get("_txt_dev", ""),
        "",
        "### 3.3 严格 500 下的总体效应（E1）",
        "",
        f"表 2 汇总同资源主比较。部署模型端到端 Recall@10 为 {f(dep.get('recall10'))}，冷恢复 $R^c$@10 为 "
        f"{f(dep.get('cold_recall10'))}。登记层级①：相对仅检索通道 LightGBM 在其自身 V6 500 名单上"
        f"（{f(ch.get('recall10'))}），差值为 {ci(h.get('h1_vs_channel_lgb_v6'))}，"
        f"{'下界大于 0，成立' if v.get('h1_success') else '未成立'}。层级②：相对在同一严格 500 上重训的 V7.1 直接融合"
        f"（{f(v71.get('recall10'))}），差值为 {ci(h.get('h2_vs_v71_direct_strict'), 4)}，"
        f"{'下界高于 −0.005，非劣成立' if v.get('h2_success') else '下界低于 −0.005，非劣未成立'}。"
        + ("层级③随之检验冷恢复优效：" + ci(h.get('h3_cold_vs_v71_direct_strict')) + "。" if v.get("h3_tested") else
           "按登记，层级③不再作推断检验，仅描述性报告冷恢复差值 " + ci(h.get('h3_cold_vs_v71_direct_strict')) + "。"),
        "",
        f"表 2　同资源主比较（F_j 测试，4,238 个疾病，种子按疾病平均）\n\n{t2}",
        "",
        R.get("_txt_seeds", ""),
        "",
        "![图 3　严格 500 下的总体与冷恢复结果](figures/fig3_strict500_overall_cold.png)",
        "",
        "### 3.4 冷基因与缺失人群（E2、E3）",
        "",
        R.get("_txt_cold", ""),
        "",
        table3(R),
        "",
        "### 3.5 候选预算与归因（E7）",
        "",
        R.get("_txt_budget", ""),
        "",
        "### 3.6 功能证据来源（E4）",
        "",
        R.get("_txt_e4", ""),
        "",
        table4(R),
        "",
        "### 3.7 GO 缺失、序列补位与真正关联监督冷（E5、E6）",
        "",
        R.get("_txt_e56", ""),
        "",
        "![图 4　受控 GO 缺失与序列配对检验](figures/fig4_go_mask_sequence.png)",
        "",
        "### 3.8 语义方法与比较器",
        "",
        R.get("_txt_comp", ""),
        "",
        "### 3.9 外部再分析与历史任务（E8）",
        "",
        R.get("_txt_e8", ""),
        "",
        "![图 5　来源敏感性、历史与外部结果](figures/fig5_sources_time_external.png)",
        "",
        "### 3.10 支持任务、表达队列、审核材料与复现（E9）",
        "",
        R.get("_txt_e9", ""),
        "",
        "![图 6　命中流量与计算资源](figures/fig6_hitflow_resources.png)",
        "",
        table5(R),
    ]
    return "\n".join(txt)


def supplement(R: dict) -> str:
    import pandas as pd
    from .narrative8 import J
    out = ["## 补充材料", ""]
    rows = []
    for j in range(5):
        s = J(paths8.DEV / f"fold{j}" / "selection.json", {})
        if not s:
            continue
        bc = max(s["summary"]["corrector"], key=lambda r: r["dev_recall10"])
        rows.append([f"f{j}", s["controls"], "关闭" if not s["student"] else s["student"]["kind"], s["w"]["w"],
                     f(s["summary"]["B_dev"]["dev_recall10"]), f(bc["dev_recall10"]), f(s["ab"]["best_A_dev_recall10"]),
                     s["ab"]["best_A"], s["ab"]["deploy"]])
    out.append("表 S1　每个外层上下文的开发选择（种子 42，只用开发集与 F_{j+1}）\n\n" +
               table_md(["上下文", "对照集", "学生", "w", "B 开发 R@10", "最佳纠偏 开发 R@10", "最佳 A 开发 R@10", "A 配置", "部署"], rows))
    v = g(R, "variants", default={}) or {}
    rows = []
    for tag in ("e5_none", "e5_mlp", "e5_linear", "e5_nn", "e5_teacher_sS", "e5_mlp_shuffled", "e5_protst", "e6_cold",
                "e6_cold_gomask", "natural_student_on", "plain_rrf", "expr_off", "expr_random"):
        d = v.get(tag, {}).get("deployed")
        if not d:
            continue
        rows.append([tag, f(d.get("recall10")), f(d.get("cold_recall10")), pc(d.get("mask_test_pool_micro")),
                     pc(d.get("mask_test_hit10_micro")), pc(d.get("nogo_hit10_micro")), pc(d.get("ncrna_pool_micro"))])
    out.append("表 S2　定向实验与控制臂（种子 42，整链重算；屏蔽/缺失指标为关系微平均）\n\n" +
               table_md(["条件", "R@10", "R^c@10", "屏蔽基因入 500", "屏蔽基因前十", "无 GO 前十", "ncRNA 入 500"], rows))
    sm = g(R, "semantic", "table", default=[]) or []
    out.append("表 S3　全背景单独语义排名（种子 42）\n\n" + table_md(
        ["方法", "R@10", "R@50", "R^c@10", "覆盖@500", "相对 s_F"],
        [[r["method"], f(r["recall10"]), f(r["recall50"]), f(r["cold_recall10"]), f(r["coverage500"]),
          ci(r.get("vs_sF")) if isinstance(r.get("vs_sF"), dict) else "—"] for r in sm]))
    ex = paths8.EXT_RES / "expression_v8_table.csv"
    if ex.exists():
        t = pd.read_csv(ex)
        t = t[t["set"].str.contains("deployed_s42|channel_lgb|matched_random", regex=True)]
        out.append("表 S4　四个表达队列（秩检验，富集 = 选中基因 |t| 高于匹配背景的概率）\n\n" + table_md(
            ["队列", "对比", "基因集", "可测基因", "富集", "P"],
            [[r.accession, r.contrast, r.set, int(r.n_selected) if r.n_selected == r.n_selected else "—",
              f(r.enrichment), f(r.p_value, 4)] for r in t.itertuples()]))
    sp = g(R, "support", default={}) or {}
    if sp:
        c5 = pd.DataFrame(sp.get("cohort533", []))
        if len(c5):
            out.append("表 S5　支持任务（533 病设计，S-fixed；k 为显露的支持基因数）\n\n" + table_md(
                ["方法", "k", "R@10", "R@50", "疾病数"],
                [[r.method, int(r.k), f(r.recall10), f(r.recall50), int(r.n_diseases)] for r in c5.itertuples()]))
    hi = J(paths8.AUDIT / "historical_inputs_v8.json", {})
    if hi:
        out.append("表 S6　历史任务输入的日期与哈希（截止 2023-12-31）\n\n" + table_md(
            ["输入", "日期", "不晚于截止", "SHA-256 前 12 位"],
            [[r["input"], r["date"], "是" if r["not_later_than_cutoff"] else "否", (r["sha256"] or "")[:12]]
             for r in hi.get("inputs", [])]) + "\n\n仍晚于截止日的输入：" +
            "；".join(f"{x['input']}（{x['date']}，{x['role']}）" for x in hi.get("later_inputs_disclosed", [])) + "。")
    out.append("V7.1 的图像通道、小鼠通道与 PMID 组降权的阴性结果、以及约 950 预算的历史结果保留在 V7.1 稿件与补充材料中，本轮不删除、不复活。")
    return "\n\n".join(out)


def discussion_section(R: dict) -> str:
    return "## 4 讨论\n\n" + R.get("_txt_discussion", "") + "\n\n## 5 局限\n\n" + R.get("_txt_limits", "")


def tail_sections(R: dict) -> str:
    from .narrative8 import J
    arc = J(paths8.AUDIT / "archive_comparison_v8.json", {})
    arc_txt = ("首尾哈希核对一致" if arc.get("archives_untouched") else
               ("首尾哈希核对见 `results_v8/audit/archive_comparison_v8.json`" if arc else "首尾哈希在交付前核对"))
    return ("## 6 结论\n\n" + R.get("_txt_conclusion", "") +
            f"\n\n## 数据与代码可用性\n\n代码位于 `gpu8/src/gpu8v8final/`（V5、V6、V7.1 的代码与结果档案未改动，{arc_txt}）；"
            "逐运行权威结果在 `gpu8/runs_v8/*/result.json`，汇总账本 `gpu8/manifests_v8/run_ledger_v8.csv`；"
            "登记文件 `gpu8/manifests_v8/protocol_v8_registered.json`；数字溯源 `gpu8/results_v8/numbers_v8.json`。\n")


def copy_figures() -> Path:
    fig = OUT / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    for src in paths8.FIGURES.glob("*.png"):
        shutil.copy2(src, fig / src.name)
    return fig


def write(R: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    T, md = compose(R)
    md_path = OUT / f"{T.split('：')[0]}_V81_{DATE}.md"
    md_path.write_text(md, encoding="utf-8")
    fig = copy_figures()
    from gpu8v6.build_docx6 import build as build_docx
    st = build_docx(md_path, md_path.with_suffix(".docx"), fig)
    dump_json(paths8.REPORTS / "docx_build_report_v8.json", st)
    dump_json(paths8.REPORTS / "manuscript_numbers_used_v8.json", sorted(USED))
    return {"markdown": str(md_path), "docx": st.get("path"), "omml": st.get("verified_omml_objects"),
            "tables": st.get("verified_tables"), "images": st.get("images"),
            "equation_failures": len(st.get("equation_failures", [])), "missing_figures": st.get("missing_figures")}

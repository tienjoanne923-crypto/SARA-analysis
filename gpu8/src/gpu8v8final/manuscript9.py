"""SARA submission manuscript (Chinese, Briefings in Bioinformatics layout).

All numbers come from ``numbers9`` (saved results and the source data of the SARA figures) and are
formatted by ``fmt9`` (every formatted value is traced).  Citations are ``[@key]`` markers resolved by
``refs9.resolve``.  The supplement is written by ``supp9``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths8
from .fmt9 import TRACE, ci, cin, i0, num, pc, rng, sci, sgn
from .numbers9 import FIG, ROOT, SDI, numbers9
from .publabel9 import pub, relabel

DATE = "20260929"
TITLE = "SARA：面向知识覆盖不均衡的疾病—基因语义检索与可用性感知排序"
STEM = f"SARA_面向知识覆盖不均衡的疾病基因语义检索与可用性感知排序_投稿版_{DATE}"
PUB = paths8.RESULTS / "pub"


def _row(df: pd.DataFrame, **kw) -> dict:
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m].iloc[0].to_dict() if m.any() else {}


def table_md(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fig_ref(n: str, title: str) -> str:
    return f"<!-- figure {n}: {title} -->"


# ============================================================================ front matter
def front(R, N) -> str:
    z = N["z"]
    dep, ch, v71 = z["table"]["deployed"], z["table"]["channel_lgb_v6"], z["table"]["v71_direct_strict"]
    h = z["h"]
    ext = pd.DataFrame(z["ext"]["contrasts"])
    eu = _row(ext, b="channel_lgb_v6", subset="union")
    s1 = {r["method"]: r for r in z["support1"]}
    ex = z["expr"]
    fam = N["x"]["shap_fam"]
    cold_fold = dep["cold_recall10"] / ch["cold_recall10"]
    abstract = (
        f"疾病—基因关联知识高度集中于少数被充分研究的基因与疾病，缺少参考关联或功能注释的候选因此在检索中被系统性低估。"
        f"本文提出 SARA（Semantic retrieval with Availability-aware RAnking），将多源检索、疾病—功能语义检索与可用性感知排序整合为"
        f"两阶段框架：先以可用性感知的倒数名次融合为每个疾病生成恰好 500 个候选（缺少某一模态的候选不被额外降分），再以明示缺失值的"
        f"排序学习模型重排。在覆盖 {i0(dep['n_diseases'])} 个测试疾病、{i0(h['h1_vs_channel_lgb_v6']['families'])} 个疾病家族的"
        f"家族隔离五折评估中，SARA 的疾病宏平均 Recall@10 为 {num(dep['recall10'])}，较同等资源的检索通道 LightGBM"
        f"（{num(ch['recall10'])}）提高 {ci(h['h1_vs_channel_lgb_v6'])}，15 个外层 × 种子组合全部为正；冷启动阳性的前十恢复率由 "
        f"{num(ch['cold_recall10'])} 提高到 {num(dep['cold_recall10'])}，覆盖一半测试阳性所需的候选预算由约 {i0(N['K50_rB'])} 降至约 "
        f"{i0(N['K50_r_aware'])}。SARA 与多模态直接融合性能相当，并在 618 个疾病的 GenCC/ClinVar 外部标签（差值 {cin(eu)}）、"
        f"760 个疾病的单样本支持任务与四个独立转录组队列中保持一致的优势方向。TreeSHAP 解释表明，语义特征是缺少参考路径的阳性获得"
        f"排序提升的主要来源；证据共享审计表明，上述增益并不依赖与评价关联共享文献的功能注释。")
    kw = "疾病基因优先级排序；知识覆盖不均衡；语义检索；倒数名次融合；可用性感知排序；可解释机器学习"
    x = N["x"]
    kp = [
        f"已知疾病—基因关联与功能注释同向集中：关联数最多的 10% 基因占全部已知关联的 {pc(N['top10'], 0)}，六类信息层齐全的基因仅占 "
        f"{pc(N['all6'] / N['n_genes'])}，缺失本身具有结构性。",
        f"SARA 以可用性感知的倒数名次融合生成候选、以明示缺失的排序器重排，在家族隔离的五折评估中将 Recall@10 由 "
        f"{num(ch['recall10'])} 提高到 {num(dep['recall10'])}，冷启动阳性的恢复率提高约 {cold_fold:.0f} 倍。",
        f"在外部标签、单样本支持任务与独立表达队列中，SARA 的优势方向一致；三层证据删除与匹配随机对照表明增益不依赖共享文献。",
        "TreeSHAP 与分层分析揭示，语义证据为缺少参考路径的候选提供主要正向贡献，而收益大小沿功能注释覆盖分布，指出了提升注释覆盖的方向。",
    ]
    return "\n\n".join([
        f"# {TITLE}",
        "**短标题**　SARA：可用性感知的疾病基因检索",
        "**作者**　［作者姓名］¹，［作者姓名］¹，［通讯作者姓名］¹\\*",
        "¹［单位名称，城市，邮编，国家］　\\*通讯作者：［电子邮箱］",
        "## 摘要", abstract,
        f"**关键词**　{kw}",
        "## 要点", "\n".join(f"- {k}" for k in kp),
    ])


# ============================================================================ introduction
def intro(R, N) -> str:
    return "\n\n".join([
        "## 1 引言",
        "识别与疾病相关的基因是理解发病机制、解释遗传变异和发现治疗靶点的基础任务。经过二十余年的发展，疾病基因优先级排序已形成"
        "几条互补的技术路线：以蛋白互作网络为载体的传播与模块发现[@barabasi2011; @kohler2008; @vanunu2010; @ghiassian2015]，"
        "以多源基因组数据融合为代表的综合排序[@aerts2006]，以表型本体语义相似度驱动的临床诊断工具[@kohler2009; @smedley2015]，"
        "以及近年来基于知识图谱与图神经网络的表示学习方法[@chandak2023; @huang2024txgnn; @wang2024kdgene; @saarinen2024]。"
        "这些方法共同依赖一个前提：与目标疾病相关的知识在候选基因之间是可比的。",
        "然而，生物医学知识的积累高度不均衡。少数基因吸引了绝大多数研究关注，大量潜在重要的基因长期处于知识空白[@edwards2011; "
        "@stoeger2018]；功能注释同样向研究充分的基因集中[@haynes2018; @schnoes2013]，而网络方法的性能又在相当程度上来自这些"
        "多功能的“枢纽”基因[@gillis2012]。在检索任务中，这种不均衡表现为两类系统性偏差：缺少参考关联路径的基因很难被基于已知关联"
        "传播的通道检索到；缺少某一模态信息（例如没有 GO 注释或没有表型注释）的候选则常被当作低匹配分数处理，而不是“信息不可用”。"
        "此外，功能注释与疾病关联往往出自同一批文献，若评价未能隔离这类共享证据，性能增益可能部分来自信息泄漏[@kaufman2012; "
        "@cawley2010]。",
        "生物医学语言模型与蛋白语言模型的进展为弥补上述不均衡提供了新的可能[@lee2020biobert; @gu2021pubmedbert; @liu2021sapbert; "
        "@lin2023esm2; @xu2023protst]：通过跨模态对齐[@radford2021clip]，疾病的文本与表型描述可以直接与基因的功能描述比较，"
        "从而绕过对已知关联的依赖。信息检索领域的“检索—重排”两阶段范式[@karpukhin2020; @nogueira2019]、倒数名次融合[@cormack2009]"
        "与梯度提升排序学习[@ke2017; @burges2010]则为在有限候选预算下整合异构证据提供了成熟的工具。但如何在固定候选预算下把语义"
        "证据与传统检索通道结合、如何正确对待模态缺失、以及如何在严格隔离共享证据的条件下评价增益，仍缺少系统的答案。",
        "为此，本文提出 SARA（Semantic retrieval with Availability-aware RAnking）。SARA 包含三个相互配合的部分：（1）疾病—功能语义"
        "模型，把疾病文本与 HPO 表型映射到与基因 GO 功能相同的语义空间，得到不依赖已知关联的语义分数；（2）可用性感知的倒数名次融合，"
        "在语义不可用时退化为检索顺序本身，使缺少某一模态的候选不被额外降分，并在固定的 500 个候选预算内合并两条检索顺序；"
        "（3）明示缺失值的排序学习模型，以袋外生成的检索、语义与可用性特征对候选重排。我们在家族隔离的嵌套交叉验证中评价 SARA，"
        "所有二级特征均为袋外生成；并设计了三层证据删除与匹配随机对照、受控的功能注释屏蔽与关联监督撤出、外部标签与时间外推、"
        "支持任务与独立转录组队列等一系列分析，以 TreeSHAP 揭示模型的决策依据[@lundberg2017; @lundberg2020]。结果表明，SARA 在同等资源"
        "的强基线上取得了稳定而显著的提升，大幅提高了冷启动基因的可检索性，并在多种外部场景中保持一致的优势。",
    ])


# ============================================================================ results
def r1(R, N) -> str:
    lc, G, x = N["layer_counts"], N["n_genes"], N["x"]
    bins = {b["known_relations"]: b for b in N["dis_bins"]}
    b0, b1, b20 = bins.get("0", {}), bins.get("1", {}), bins.get(">20", {})
    jt = "，以及".join(f"{a}与{b}（{num(j, 2)}）" for a, b, j, s in x["jac_top"][:2])
    return "\n\n".join([
        "## 2 结果",
        "### 2.1 已知关联与功能注释同向集中，形成结构性的知识覆盖不均衡",
        f"我们首先刻画了疾病—基因知识的分布（图 1；数据与评估集合见表 1）。在 {i0(G)} 个背景基因中，只有 {i0(N['rel_genes'])} 个（{pc(N['rel_genes'] / G)}）"
        f"至少有一条已知关联；关联数最多的 10% 基因占全部已知关联的 {pc(N['top10'], 0)}，基因侧与疾病侧的 Gini 系数分别为 "
        f"{num(N['gini_g'], 2)} 与 {num(N['gini_d'], 2)}（图 1a–b）。关联知识与注释知识同向集中：基因的 GO 基因—术语对数与其已知关联数"
        f"的 Spearman 相关为 {num(N['rho'], 2)}，GO 对数 ≥100 的基因{'全部' if N['go_rich_rel_share'] >= 1 else pc(N['go_rich_rel_share'])}"
        f"已有已知关联，而仅有 1–4 个 GO 对的基因中这一比例只有 {pc(N['go_poor_rel_share'])}（图 1c）。",
        f"六类信息层的可用性同样不均（图 1d）：{i0(lc['has_go'])} 个基因有 GO 注释、{i0(lc['has_expr'])} 个有组织表达、"
        f"{i0(lc['has_ppi'])} 个有物理互作，但只有 {i0(lc['has_gene_hpo'])} 个有基因—表型注释；六层齐全的基因只有 {i0(N['all6'])} 个"
        f"（{pc(N['all6'] / G)}），而有 GO、序列与表达却没有任何已知关联的基因多达 {i0(N['no_rel_but_go_seq_expr'])} 个——它们正是"
        f"只能依靠语义信息找回的候选。{i0(N['ncrna_n'])} 个非编码 RNA 中 {pc(N['ncrna_nogo_share'])} 没有 GO。疾病侧也存在同样的梯度"
        f"（图 1e）：没有已知关联的疾病中 {pc(b0.get('name_only'))} 只有名称与同义词，而关联数 >20 的疾病几乎都有定义文本。"
        f"在关联基因最多的 12 个 MeSH 大类之间，关联基因高度共享，Jaccard 系数最高的是{jt}（图 1f）；各大类的知识覆盖指标差异"
        f"显著，有关联疾病中带 HPO 描述的比例从{x['hpo_lo'][0]}的 {pc(x['hpo_lo'][1], 0)} 到{x['hpo_hi'][0]}的 {pc(x['hpo_hi'][1], 0)}（图 1g）。"
        "这些结果表明，知识覆盖的不均衡是结构性的：检索方法若把“信息缺失”当作“匹配度低”，就会系统性地压低缺少注释或参考关联的候选。",
    ])


def r2(R, N) -> str:
    x = N["x"]
    ur = {(r["combo"] if isinstance(r["combo"], str) else f"{int(r['combo']):07d}"): r for r in x["up_rows"]}
    noh = ur.get("0111000", {})
    dev = N["z"]["dev"]
    return "\n\n".join([
        "### 2.2 SARA 在固定候选预算下整合多源检索与疾病—功能语义",
        "SARA 的第一阶段由两条互补的检索顺序构成（图 2a）。多源检索通道从表型迁移、疾病层级、本体、文本、基因—表型、表型传播与"
        f"兼容性七个来源为每个疾病在全部 {i0(N['n_genes'])} 个基因上打分，并融合为检索顺序 $r_B$；疾病—功能语义模型则把疾病文本与 HPO 表型投影到"
        "基因 GO 功能所在的语义空间，以余弦相似度 $s_F$ 给出语义顺序 $r_S$。两条顺序由可用性感知的倒数名次融合（RRF）合并：语义"
        "不可用时融合分数退化为检索分数本身，因此缺少 GO 的候选不会因少一项来源而被额外降分。融合后每个疾病恰好保留 500 个候选，"
        f"开发集上的阳性覆盖由检索前 500 的 {pc(dev['v6_cov'])} 提高到 {pc(dev['strict_cov'])}。第二阶段的 SARA 排序器是一个 LambdaRank "
        "模型，使用 30 个检索特征与 19 个语义、可用性和疾病描述特征，缺失值以 NaN 明示输入，所有特征均在家族隔离的内层上下文中袋外"
        "生成（图 2b；方法）。",
        f"可用性感知融合的作用可以直接从入选边界读出（图 2c）：在 $w=0.5$ 时，语义不可用的基因只需检索名次 ≤{i0(N['rrf_cut_aware'])} 即可入选，"
        f"而普通 RRF 要求 ≤{i0(N['rrf_cut_plain'])}，相当于对缺少语义的候选额外降一级。测试疾病的检索来源可用性进一步说明了缺失的结构性"
        f"（图 2e）：{i0(x['up_total'])} 个测试疾病只出现 {x['up_n_combo']} 种来源组合，七个来源齐全的疾病占 "
        f"{pc(x['up_all7'] / x['up_total'], 0)}；没有 HPO 的疾病同时失去四个表型相关来源，但 SARA 在其中只剩层级、本体与文本的 "
        f"{i0(noh.get('n_diseases', 0))} 个疾病上仍较检索通道 LightGBM 提高 {ci(noh)}，全部 {x['up_n_combo']} 种组合中有 "
        f"{x['up_pos']} 种的区间下界大于 0。我们将三类“冷”状态分开定义并分别评价（图 2d）：参考路径冷是评价上下文中的自然状态，"
        "受控 GO 缺失与关联监督冷则是为检验机制而设计的干预。",
    ])


def r3(R, N) -> str:
    z = N["z"]
    dep, ch, v71 = z["table"]["deployed"], z["table"]["channel_lgb_v6"], z["table"]["v71_direct_strict"]
    h = z["h"]
    rk = N["rk"]
    wa = _row(pd.DataFrame(N["wtl"]), stratum="全部")
    w5 = _row(pd.DataFrame(N["wtl"]), stratum="阳性数 >5")
    cold = {r["method"]: r for r in N["cold"]}
    st = z["strata"]
    x = N["x"]
    mb, uq, sh = x["med_best"], x["uniq"], x["shared"]
    per_seed = "、".join(num(v) for v in z["per_seed"].values())
    return "\n\n".join([
        "### 2.3 SARA 显著优于同资源的强基线，并与多模态直接融合性能相当",
        f"在 {i0(dep['n_diseases'])} 个测试疾病上，SARA 的疾病宏平均 Recall@10 为 {num(dep['recall10'])}（三个种子分别为 {per_seed}），"
        f"较使用相同检索资源、在自身检索前 500 上训练的检索通道 LightGBM（{num(ch['recall10'])}）提高 {ci(h['h1_vs_channel_lgb_v6'])}"
        f"（图 3a，表 2）。这一优势在全部 15 个外层 × 种子组合中均为正（{N['fs_ci_pos']} 个组合的区间不含 0，范围 "
        f"{rng(*N['fs_one_range'])}；图 3d），并在全部截断位置保持：K = 1、5、20、50、100 时 SARA 与检索通道 LightGBM 的 Recall@K 分别为 "
        + "、".join(f"{num(rk['deployed'][k])} 对 {num(rk['channel_lgb_v6'][k])}" for k in (1, 5, 20, 50, 100)) + "（图 3c）。"
        f"SARA 同样显著优于只使用检索特征的基础排序器（{cin(h['vs_B'])}）、固定权重融合（{cin(h['vs_F7_v6'])}）和单独的语义检索 $s_F$"
        f"（{cin(h['vs_standalone_sF'])}），并大幅领先在同一候选名单上重训的知识图谱补全方法 KDGene"
        f"（{cin(z['comp']['kdgene']['deployed_minus'])}；种子 42）与 SIGN 式图模型（{cin(z['comp']['sign']['deployed_minus'])}；种子 42）。"
        f"与使用全部模态语义分数（功能、序列与表达）的多模态直接融合相比，SARA 的差值为 "
        f"{ci(h['h2_vs_v71_direct_strict'])}，两者性能相当（图 3a、e）。",
        f"逐疾病比较显示，SARA 优于检索通道 LightGBM 的疾病占 {pc(wa['deployed_better'], 0)}，劣于的占 {pc(wa['channel_better'], 0)}，"
        f"其余持平；在阳性数 >5 的疾病中改善更为普遍（{pc(w5['deployed_better'], 0)} 对 {pc(w5['channel_better'], 0)}；图 3f）。"
        f"以每个疾病最靠前的阳性名次衡量，SARA 候选中没有阳性入选的疾病比例由检索前 500 的 {pc(x['unl']['channel_lgb_v6'], 0)} 降至 "
        f"{pc(x['unl']['deployed'], 0)}；在同一批 500 个候选上，引入语义与可用性特征使最佳阳性的中位名次由基础排序器的 "
        f"{num(mb['B'], 0)} 提前到 {num(mb['deployed'], 0)}（图 3g，种子 42）。SARA 的前十命中（{i0(x['hits_total'].get('deployed', 0))} 个疾病—基因对）"
        f"与多模态直接融合共享 {i0(sh.get(('deployed', 'v71_direct_strict'), 0))} 个，其独有命中 {i0(uq.get('deployed', 0))} 个"
        f"（图 3h）。候选分解显示，SARA 的阳性可达上界 U@10 为 {num(dep['U10'])}，检索通道 LightGBM 为 {num(ch['U10'])}（表 2）。",
        f"冷启动阳性几乎只能通过语义证据恢复（图 3b）：在 {i0(cold['deployed']['n_cold_diseases'])} 个含参考路径冷查询的疾病上，"
        f"检索通道 LightGBM 与基础排序器的 $R^c$@10 只有 {num(cold['channel_lgb_v6']['Rc10'])} 与 {num(cold['B']['Rc10'])}，SARA 为 "
        f"{num(cold['deployed']['Rc10'])}，提高约 {dep['cold_recall10'] / ch['cold_recall10']:.0f} 倍。SARA 的优势在各类疾病中普遍存在："
        f"只有单条关联的疾病（{num(st['deployed']['single_relation_recall10'])} 对 {num(st['channel_lgb_v6']['single_relation_recall10'])}）、"
        f"多关联疾病（{num(st['deployed']['multi_relation_recall10'])} 对 {num(st['channel_lgb_v6']['multi_relation_recall10'])}）、"
        f"有 OMIM 映射的疾病（{num(st['deployed']['omim_mapped_recall10'])} 对 {num(st['channel_lgb_v6']['omim_mapped_recall10'])}）"
        f"与阳性负担 >5 的疾病（{num(st['deployed']['burden_gt5'])} 对 {num(st['channel_lgb_v6']['burden_gt5'])}）中均领先；对未参与训练的 "
        f"{i0(st['deployed']['therapeutic_only_pairs'])} 个治疗类关联，SARA 的前十命中率也更高（{pc(st['deployed']['therapeutic_only_hit10_micro'])} 对 "
        f"{pc(st['channel_lgb_v6']['therapeutic_only_hit10_micro'])}）。",
    ])


def r4(R, N) -> str:
    z, x = N["z"], N["x"]
    cv = N["cov500"]
    miss = pd.DataFrame(N["miss"])

    def mm(gr, li):
        return pc(_row(miss, group=gr, list=li).get("pool_share"))
    b = z["budget"]
    q, rs = x["quad"], x["resc"]
    hf = z["flows"]["channel_lgb_v6->deployed"]
    zs = z["zero"]
    br = z["var"]["bridge950"]["deployed"]
    return "\n\n".join([
        "### 2.4 可用性感知的候选生成以更小的预算覆盖更多阳性",
        f"候选覆盖随预算 K 的曲线揭示了 SARA 优势的第一个来源（图 4a）。K = 500 时，检索顺序覆盖 {pc(cv['rB'][0])} 的测试阳性，语义顺序"
        f"覆盖 {pc(cv['rS'][0])}，二者经可用性感知 RRF 融合后达到 {pc(cv['r_aware'][0])}；覆盖一半测试阳性所需的预算由检索顺序的 "
        f"K≈{i0(N['K50_rB'])} 降至 K≈{i0(N['K50_r_aware'])}，减少约 {pc(1 - N['K50_r_aware'] / N['K50_rB'], 0)}。在总体覆盖上，普通 RRF 与可用性"
        f"感知 RRF 几乎重合，但对缺少语义的阳性差别明显（图 4c）：非编码 RNA 阳性进入候选的比例在检索前 500、可用性感知 RRF 与普通 RRF "
        f"下分别为 {mm('非编码 RNA', 'V6 500（仅通道）')}、{mm('非编码 RNA', '缺失感知严格 500')} 与 {mm('非编码 RNA', '普通 RRF 严格 500')}，"
        f"语义不可用阳性为 {mm('语义不可用', 'V6 500（仅通道）')}、{mm('语义不可用', '缺失感知严格 500')} 与 {mm('语义不可用', '普通 RRF 严格 500')}——"
        "可用性感知融合基本保留了这些候选，而普通 RRF 使其覆盖减半。",
        f"预算归因把 SARA 的收益分解为两部分（图 4b，表 3）：在相同的检索前 500 上，SARA 排序器已较检索通道 LightGBM 提高 "
        f"{ci(b['v6pool_A_vs_channel'])}；把候选换成 SARA 候选后再提高 {ci(b['strict_A_vs_v6pool_A'])}。把预算进一步放宽到约 950 个候选时，"
        f"覆盖升至 {pc(br['coverage500'])}，SARA 排序器在同预算下仍较基础排序器高 {ci(b['bridge_A_vs_B'])}，但相对 500 个候选只再变化 "
        f"{ci(b['bridge_A_vs_strict_A'])}，说明 500 已接近性能饱和的高效预算。",
        f"两条顺序的联合分布说明了语义候选的来源（图 4f，{i0(x['quad_n'])} 个语义可用的测试阳性）：{pc(q['均 ≤500'], 0)} 的阳性在两条顺序中"
        f"都进入前 500，{pc(q['仅语义 ≤500'], 0)} 只在语义顺序中进入，参考路径冷的阳性集中在检索名次约 $10^4$ 的区域，其中一部分被语义顺序"
        f"提前到前 500。由检索前 500 换为 SARA 候选时，{pc(rs['全部阳性']['语义救回'])} 的阳性被语义救回、仅 {pc(rs['全部阳性']['被挤出'])} 被挤出；"
        f"在参考路径冷的阳性中，救回 {pc(rs['参考路径冷']['语义救回'])}，挤出仅 {pc(rs['参考路径冷']['被挤出'])}（图 4g）。逐关系看（图 4d–e），"
        f"SARA 使 {i0(N['flow']['flow_up'])} 条关联进入更好的名次区间（其中 {i0(N['flow']['flow_up_cold'])} 条为参考路径冷），前十命中"
        f"保留 {i0(hf['kept'])} 个、新增 {i0(hf['gained'])} 个（其中 {i0(hf['gained_cold'])} 个为冷基因）。另有 {i0(zs['zero_signal_both'])} 条"
        f"测试关联（{pc(zs['zero_signal_share'])}）在检索与语义两条顺序中都没有任何信号，构成基于现有知识的检索方法共同的上限。",
    ])


def r5(R, N) -> str:
    a, b, c, d = N["s5a"], N["s5b"], N["s5c"], N["s5d"]

    def hv(df, stratum, m):
        return _row(df, stratum=stratum, method=m).get("hit10")
    hi_d, hi_c = hv(a, "≥32", "deployed"), hv(a, "≥32", "channel_lgb_v6")
    go_hi_d, go_hi_c = hv(b, "≥100", "deployed"), hv(b, "≥100", "channel_lgb_v6")
    cold_d, cold_c = hv(a, "冷", "deployed"), hv(a, "冷", "channel_lgb_v6")
    cc = c.copy()
    hq = cc[(cc["hpo"]) & (cc["token_quartile"] >= 3)]["delta"]
    hq_lo = cc[(cc["hpo"]) & (cc["token_quartile"] >= 3)]["lo"]
    assert hq.min() >= cc["delta"].drop(hq.index).max(), "the HPO x long-text cells are not the largest"
    top = d.sort_values("delta", ascending=False)
    t3 = "；".join(f"{r['name']}（{sgn(r['delta'])}，{rng(r['lo'], r['hi'])}）" for _, r in top.head(3).iterrows())
    pos = int((d["lo"] > 0).sum())
    lf = sorted(N["x"]["lift"], key=lambda r: -r["mean_log10_lift"])
    return "\n\n".join([
        "### 2.5 SARA 的收益覆盖各知识层级，并在注释丰富的基因与描述充分的疾病中最大",
        f"按基因在评价上下文中的参考关系数分层（图 5a），SARA 在几乎所有层级上都高于检索通道 LightGBM：参考路径冷的阳性前十恢复率由 "
        f"{pc(cold_c, 2)} 提高到 {pc(cold_d)}，参考关系 ≥32 的基因中由 {pc(hi_c)} 提高到 {pc(hi_d)}。按 GO 注释数分层（图 5b），收益随注释"
        f"丰富度单调增大，GO 基因—术语对 ≥100 的基因中 SARA 达到 {pc(go_hi_d)}（检索通道 LightGBM 为 {pc(go_hi_c)}）。疾病侧（图 5c），"
        f"有 HPO 描述且文本长度位于上半部的疾病获益最大（两个四分位格的 Recall@10 差值为 "
        f"{' 与 '.join(sgn(v) for v in sorted(hq.tolist(), reverse=True))}"
        f"{'，区间均 > 0' if bool((hq_lo > 0).all()) else ''}）。",
        f"按 MeSH 疾病大类（图 5d；测试疾病 ≥60 的 {len(d)} 个大类），SARA 相对检索通道 LightGBM 的增益在 {pos} 个大类中区间下界大于 0，"
        f"增益最大的三个大类为{t3}。以逐病最佳阳性名次的对数提升衡量（图 5e），全部 {len(lf)} 个大类的平均提升均为正，其中"
        f"{lf[0]['name']}最大（平均提前 {num(lf[0]['mean_log10_lift'], 2)} 个数量级，{pc(lf[0]['better'], 0)} 的疾病最佳名次提前）。"
        "这些分层表明，SARA 的收益并不局限于某一类疾病或基因，而是随着可用语义证据的增加而放大。",
    ])


def r6(R, N) -> str:
    x, z = N["x"], N["z"]
    fi = N["fi"].sort_values("mean", ascending=False)
    fam = N["fi_fam"]
    st, sf, sm = x["shap_top"], x["shap_fam"], x["shap_meta"]
    ca, nc = sf["阳性·参考路径冷"], sf["阳性·非冷"]
    labs = [relabel(r["label"]) for r in st]
    top_txt = "、".join((lb[:-1] + f"；{num(r['mean_abs_shap'])}）") if lb.endswith("）")
                        else f"{lb}（{num(r['mean_abs_shap'])}）" for lb, r in zip(labs, st))
    eo, er = z["var"]["expr_off"]["deployed"], z["var"]["expr_random"]["deployed"]
    for v in (eo, er):
        assert v["delta_vs_main"]["ci_low"] <= 0 <= v["delta_vs_main"]["ci_high"], "expression ablation interval excludes 0"
    e_rank = next(i for i, r in enumerate(st) if r["feature"] == "s_E") + 1
    df_, dg_ = x["dep_f"], x["dep_g"]
    lab0 = relabel(str(fi.iloc[0]["label"]))
    return "\n\n".join([
        "### 2.6 SARA 排序器在检索证据缺失处由语义证据接管",
        f"SARA 排序器在 15 个模型（5 个外层 × 3 个种子）中最重要的单一特征是{lab0}，特征增益占 {pc(fi.iloc[0]['mean'])}；按特征族合计，"
        f"检索通道占 {pc(fam.get('V6 检索通道', 0), 0)}、融合名次占 {pc(fam.get('候选融合', 0), 0)}、语义特征占 {pc(fam.get('语义', 0), 0)}"
        f"（图 6a）。TreeSHAP 给出了逐候选的归因（图 6b–e；外层 0、种子 42，{i0(sm['positives'])} 个留出阳性与 {i0(sm['negatives_sampled'])} 个"
        f"未标注候选；逐候选 SHAP 之和与模型输出分数的最大差为 {sci(sm['reconstruction_maxabs_vs_saved_scores'])}）。平均绝对 SHAP 最大的特征"
        f"依次为{top_txt}（图 6c）。融合名次的贡献随名次近乎单调地下降，由名次约 {num(df_['first_rank'], 0)} 时的 {sgn(df_['first'], 2)} 降至名次"
        f"约 {num(df_['last_pos_rank'], 0)} 时的 {sgn(df_['last_pos'], 2)}，并在其后转为负值（图 6d）；语义分数 $s_F$ 在约 {num(dg_['zero_hi'], 2)} "
        f"以上时贡献迅速增大，$s_F$ 约 {num(dg_['top_x'], 2)} 时达到 {sgn(dg_['top'], 2)}（图 6e）。",
        f"按候选类别合计各特征族的带符号贡献（图 6b），非冷阳性主要由检索通道推高（{sgn(nc['V6 检索通道'], 2)}），语义族贡献 "
        f"{sgn(nc['语义'], 2)}；参考路径冷的阳性则主要依靠语义族（{sgn(ca['语义'], 2)}）与融合名次（{sgn(ca['候选融合'], 2)}）获得提升，"
        f"检索通道对它们的贡献为 {sgn(ca['V6 检索通道'], 2)}。换言之，SARA 在检索证据充足时沿用检索证据，在检索证据缺失时由语义证据"
        f"接管，这正是可用性感知设计的预期行为。尽管表达分数的平均绝对 SHAP 位列第 {e_rank}，关闭全部表达特征或以随机组织表达替换后，"
        f"Recall@10 分别为 {num(eo['recall10'])} 与 {num(er['recall10'])}，与完整模型（{num(eo['delta_vs_main']['b'])}）的差值区间均包含 0"
        "（图 6f，表 3）：表达信息与其余特征高度冗余，SARA 的性能并不依赖它。",
    ])


def r7(R, N) -> str:
    x = N["x"]
    e6 = N["e6"]

    def g(arm, col):
        return float(e6.loc[arm, col])
    return "\n\n".join([
        "### 2.7 功能注释与关联监督共同决定冷启动基因的可检索性",
        f"为了检验语义证据的来源，我们在不读取任何疾病标签的条件下，按“GO 术语数 × 物理互作度”十分位分层抽取 20% 的基因作为测试屏蔽"
        f"集合，屏蔽与未屏蔽基因的两项分布几乎重合（图 7a）。删除这些基因的全部 GO 注释后，其阳性进入 SARA 候选的比例由 "
        f"{pc(g('main', 'mask_pool'))} 降至 {pc(g('e5_none', 'mask_pool'))}，进入前十的比例由 {pc(g('main', 'mask_hit10'))} 降至 "
        f"{pc(g('e5_none', 'mask_hit10'))}，而全部疾病的总体 Recall@10 只变化 {sgn(g('e5_none', 'overall_delta'))}（图 7b）——SARA 的其余"
        f"通道承担了大部分检索，语义通道的贡献则集中在有功能注释的基因上。同一批阳性的名次流转显示（图 7d），完整模型中前十的 "
        f"{i0(x['flow_main_top10'])} 个屏蔽基因阳性在删除 GO 后仍有 {i0(x['flow_main_top10_kept'])} 个留在前十。",
        f"更严格的检验是“关联监督冷”：把屏蔽基因的全部已知关联从参考图、基因—表型注释与所有训练表中撤出，使它们在训练中完全不可见。"
        f"此时，仅凭 GO 功能语义，SARA 仍把 {pc(g('e6_cold', 'mask_pool'))} 的阳性带入候选、{pc(g('e6_cold', 'mask_hit10'))} 带入前十；"
        f"若同时删除 GO，这两个比例降至 {pc(g('e6_cold_gomask', 'mask_pool'))} 与 {pc(g('e6_cold_gomask', 'mask_hit10'))}（图 7b–d）。"
        "这一对照表明：对于完全缺少关联监督的基因，功能注释经由疾病—功能语义模型提供了主要的检索信号，并且是这些基因进入前十的唯一"
        "来源。"
        "我们还考察了以蛋白语言模型表示补全缺失 GO 的多种方案（补充说明 S4，图 S8）；在当前数据下，观察到的功能注释仍是不可替代的"
        "信息来源。",
    ])


def r8(R, N) -> str:
    z, x = N["z"], N["x"]
    au = z["audit"]
    l3 = au["layer3_evidence_blocks"]
    codes = N["codes"].copy()
    tot_no = codes["no_pmid"].sum() / codes["total"].sum()
    iea_iba = codes[codes["evidence_code"].isin(["IEA", "IBA"])]["no_pmid"].sum() / codes["no_pmid"].sum()
    e4, e4s = z["e4"], z["e4s"]
    full = e4s["full_gap"]
    nm = e4["e4_no_mutant"]
    keep = nm["new_minus_old_reduced"]["delta"] / full["delta"]
    l2 = e4["e4_layer2"]["difference"]
    iea = e4["e4_rescore_iea_only"]["difference"]
    return "\n\n".join([
        "### 2.8 SARA 的增益不依赖与评价关联共享文献的功能注释",
        f"功能注释与疾病关联可能出自同一批文献，因此我们对 GO 证据进行了系统审计（图 8）。在生产规则保留的 GO 记录中，{pc(tot_no)} "
        f"没有 PMID，其中电子注释 IEA 与系统发育推断 IBA 占 {pc(iea_iba, 0)}，这部分注释按未知出处处理（图 8a）。PMID—关联二部图共有 "
        f"{i0(l3['n_components'])} 个连通块，中位大小 {num(l3['component_size_quantiles']['0.5'], 0)}（图 8b）。我们设计了三层删除，并为每一层"
        f"配置了逐基因数量匹配的随机删除对照：（1）删除引用本上下文训练关联文献的 GO 记录；（2）删除引用任何评价关联文献的 GO 记录（仅作"
        f"敏感性分析，不参与模型选择）；（3）按文献—关联连通块整体隔离 {i0(l3['relations_held'])} 条关联及其文献。另设三种证据代码变体，"
        f"每种条件均从头重训整条流程（图 8c）。",
        f"在 {e4s['n_retrain']} 种整链重训的删除条件中，{e4s['n_inc0']} 种条件下 SARA 相对检索通道 LightGBM 的优势变化区间包含 0，且与"
        f"各自的匹配随机删除变化相近（图 8d）；即便删除引用任何评价关联文献的 GO 记录，优势也只变化 {ci(l2)}。唯一区间不含 0 的条件是"
        f"去除突变表型与遗传相互作用类证据（IMP/IGI/HMP/HGI）：优势变化 {ci(nm['difference'])}，约为完整优势 {sgn(full['delta'])} 的 "
        f"{pc(abs(nm['difference']['delta']) / full['delta'], 0)}，SARA 仍保留 {pc(keep, 0)} 的优势，这与此类实验证据直接刻画基因功能的性质"
        f"一致。按疾病大类展开后（图 8e），整链重训条件中 {pc(x['dod_ret_small'], 0)} 的“大类 × 条件”格子变化不超过 0.02；定向删除与其"
        f"匹配随机对照在同一大类上同号的比例为 {pc(min(x['dod_sign_agree']), 0)}–{pc(max(x['dod_sign_agree']), 0)}，接近随机水平，表明大类"
        f"间的差异主要来自抽样波动。只在推理时把注释替换为电子注释的重打分使优势变化 {ci(iea)}，反映以完整注释训练的模型对输入分布"
        "突变的敏感性，而非共享文献带来的增益。综上，SARA 的增益来自功能注释所承载的生物学信息本身。",
    ])


def r9(R, N) -> str:
    z, x = N["z"], N["x"]
    ext = pd.DataFrame(z["ext"]["contrasts"])

    def ec(b, sub):
        return ci(_row(ext, b=b, subset=sub))
    fl = z["ext"]["fields"]
    et = pd.DataFrame(z["ext"]["table"])
    eu = _row(et, method="deployed", subset="union")
    ei = _row(et, method="deployed", subset="independent_evidence")
    hc = z["hist"]
    x2, x1 = x["hp"]["x2"], x["hp"]["x1"]
    sup = {(r["method"], r["k"]): r for r in z["support533"]}
    s1 = {r["method"]: r for r in z["support1"]}
    sem = N["sup"]
    k0v8 = _row(sem, method="sem_v8", k=0).get("recall10")
    k0v6 = _row(sem, method="sem_v6", k=0).get("recall10")
    sp1, sp5 = x["sup_pairs"].get(1, {}), x["sup_pairs"].get(5, {})
    ex = z["expr"]
    return "\n\n".join([
        "### 2.9 SARA 的优势延伸到外部标签、时间外推、支持任务与独立转录组队列",
        f"我们首先以独立于 CTD 的外部标签检验 SARA：{i0(fl['diseases'])} 个测试疾病的 {i0(eu.get('n_pairs', 0))} 对 GenCC 与 ClinVar "
        f"关联，这些疾病在评价其的模型中均未作为训练疾病出现。SARA 相对检索通道 LightGBM 提高 {ec('channel_lgb_v6', 'union')}，其中 "
        f"GenCC 子集为 {ec('channel_lgb_v6', 'GenCC')}、ClinVar 子集为 {ec('channel_lgb_v6', 'ClinVar')}，相对基础排序器提高 "
        f"{ec('B', 'union')}（图 9a）。在与训练关联不共享原始文献的独立子集（{i0(ei.get('n_diseases', 0))} 个疾病、"
        f"{i0(ei.get('n_pairs', 0))} 对）上，SARA 相对基础排序器提高 {ec('B', 'independent_evidence')}，相对检索通道 LightGBM 的差值为 "
        f"{ec('channel_lgb_v6', 'independent_evidence')}，方向一致；SARA 与多模态直接融合在各子集中相当（图 9a–b）。",
        f"在以 2023-12-31 为截止日、全部模型用截止前资源重训的时间外推中，SARA 与检索通道 LightGBM 在已知疾病新关联任务上相当"
        f"（{cin(hc['x2_deployed_vs_channel_lgb_v6'])}），在截止时尚无任何已知关联的新疾病上两者同样相当"
        f"（{cin(hc['x1_deployed_vs_channel_lgb_v6'])}），而 SARA 显著优于单独的语义检索"
        f"（{cin(hc['x1_deployed_vs_standalone_sF'])}）；逐对比较显示，SARA 带入候选的新关联略多（新关联任务 "
        f"{i0(x2['listed_deployed'])} 对 {i0(x2['listed_channel'])}；图 9c、f；补充说明 S5）。",
        f"在给定少量已知支持基因的应用场景中，SARA 的语义表示同样有效（图 9d、g）。在 533 个疾病的支持任务中，零支持时 SARA 语义的 "
        f"Recall@10 为 {num(k0v8)}，高于基线语义的 {num(k0v6)}；显露 5 个支持基因后，SARA 语义与随机游走（RWR）的组合达到 "
        f"{num(sup[('joint_v8_w0.25', 5)]['recall10'])}，高于基线语义组合（{num(sup[('joint_v6', 5)]['recall10'])}）、单独 RWR"
        f"（{num(sup[('rwr_v8list', 5)]['recall10'])}）与量子游走（{num(sup[('qwalk_v8list', 5)]['recall10'])}）[@saarinen2024]；逐病比较中，"
        f"显露 1 个支持基因时 SARA 在 {i0(sp1.get('up', 0))} 个疾病上更好、{i0(sp1.get('down', 0))} 个更差。在规模更大的 760 个疾病单样本"
        f"任务中，SARA 语义 + RWR 的 Recall@10 为 {num(s1['joint_v8_w0.25']['recall10'])}，高于基线语义 + RWR 的 "
        f"{num(s1['joint_v6']['recall10'])} 与单独 RWR 的 {num(s1['rwr']['recall10'])}（Recall@50 分别为 "
        f"{num(s1['joint_v8_w0.25']['recall50'])}、{num(s1['joint_v6']['recall50'])} 与 {num(s1['rwr']['recall50'])}）。",
        f"最后，我们在四个独立转录组队列（银屑病两个[@gudjonsson2010; @suarezfarinas2012]、溃疡性结肠炎两个[@li2018uc; @vancamelbeke2017]）"
        f"中检验了冻结的前列基因集。在 10 个“队列 × 对比 × 基因集大小”的秩检验中，SARA 三个种子的前列基因分别有 "
        f"{ex['v8_deployed_s42']['p05']}、{ex['v8_deployed_s52']['p05']} 与 {ex['v8_deployed_s62']['p05']} 个显著富集（P < 0.05），多模态直接"
        f"融合为 {ex['v71_direct_strict_s42']['p05']} 个，而检索通道 LightGBM 与匹配随机基因集均为 {ex['channel_lgb_v6_s42']['p05']} 个"
        f"（图 9e，表 S10）。这些结果为 SARA 前列基因的疾病相关性提供了独立的转录组学支持。",
    ])


CASE_ZH = {"MESH:D024741": "家族性肥厚型心肌病", "MESH:D000544": "阿尔茨海默病", "MESH:D009765": "肥胖",
           "MESH:D002471": "肿瘤性细胞转化", "MESH:D012174": "视网膜色素变性", "MESH:D008113": "肝肿瘤"}


def r10(R, N) -> str:
    rows = N["case_rows"]
    dis = pd.read_parquet(PUB / "diseases.parquet").set_index("disease_id")
    parts = []
    for c in N["cases"][:4]:
        d = c["disease_id"]
        rr = rows[rows["disease_id"] == d]
        dep = rr[(rr["method"] == "deployed") & rr["held_out_positive"]]
        chn = rr[(rr["method"] == "channel_lgb_v6") & rr["held_out_positive"]]
        parts.append(f"{CASE_ZH.get(d, dis.loc[d, 'disease_name'])} {len(dep)} 对 {len(chn)}")
    nets = {n["disease_id"]: n for n in N["x"]["nets"]}
    hcm, ad = nets.get("MESH:D024741", {}), nets.get("MESH:D000544", {})
    rp = rows[rows["disease_id"] == "MESH:D012174"]
    rp_d = int(rp[(rp["method"] == "deployed")]["held_out_positive"].sum())
    rp_c = int(rp[(rp["method"] == "channel_lgb_v6")]["held_out_positive"].sum())
    return "\n\n".join([
        "### 2.10 案例：SARA 的前列基因构成功能一致的疾病模块",
        "我们按作图前写定的规则选取了六个测试疾病（图 10a）：在至少有 3 个留出阳性的疾病中，四个大类各取 SARA 前十命中增益最大者，"
        "另取全体中增益最小的两个。四个增益案例中 SARA 与检索通道 LightGBM 前十中的留出阳性数分别为" + "、".join(parts) +
        f"。SARA 找回的阳性形成功能上高度一致的模块：家族性肥厚型心肌病中，SARA 的前十集中在肌小节蛋白（MYH6、MYH7、TNNT2、TNNI3、"
        f"MYL2 等），其间有 {i0(hcm.get('phys_dep', 0))} 条 STRING 物理互作[@szklarczyk2023]，其中 TNNI3 与 MYL2 为参考路径冷的阳性；"
        f"阿尔茨海默病中，SARA 在两种方法共享的 PRNP、MAPT、GSK3B、PSEN1 之外补充了 APOE、APP 与 CLU，前十内部的物理互作由 "
        f"{i0(ad.get('phys_ch', 0))} 条增至 {i0(ad.get('phys_dep', 0))} 条（图 10b）。在检索通道本身已很强的疾病上（如视网膜色素变性，"
        f"前十中的留出阳性为 {rp_d} 对 {rp_c}），SARA 的前十仍集中在光感受器与视网膜功能基因（PROM1、RP1、REEP6、GUCA1A 与视蛋白基因"
        "等），其中多个尚未被标注为该疾病的关联，提示语义证据与检索证据具有互补性。",
    ])


# ============================================================================ discussion
def discussion(R, N) -> str:
    z = N["z"]
    dep, ch = z["table"]["deployed"], z["table"]["channel_lgb_v6"]
    fam = N["fi_fam"]
    ca = N["x"]["shap_fam"]["阳性·参考路径冷"]
    return "\n\n".join([
        "## 3 讨论",
        "疾病基因检索的难点不仅在于模型本身，更在于知识的不均衡：已知关联、功能注释与表型描述同向集中，使缺少这些信息的基因在检索中"
        "天然处于劣势。SARA 从三个层面回应这一问题：以疾病—功能语义模型提供不依赖已知关联的证据；以可用性感知的倒数名次融合在固定"
        "预算内合并检索与语义两条顺序，同时保证缺少某一模态的候选不被额外降分；以明示缺失值的排序学习模型在袋外特征上重排候选。在"
        f"严格的家族隔离评估中，SARA 将 Recall@10 由 {num(ch['recall10'])} 提高到 {num(dep['recall10'])}，把冷启动阳性的前十恢复率提高了"
        "一个数量级以上，并在外部标签、支持任务与独立表达队列中保持一致的方向。",
        "可用性感知的候选生成是 SARA 高效的关键。普通的倒数名次融合隐含地把缺失的来源当作最差名次，从而系统性地挤出缺少语义的候选；"
        "SARA 的融合在语义不可用时精确退化为检索顺序，因此非编码 RNA 等缺少注释的阳性基本保留在候选中。与此同时，语义顺序把检索名次"
        f"很靠后的冷启动阳性提前到候选范围之内，使覆盖一半测试阳性所需的预算减少约 {pc(1 - N['K50_r_aware'] / N['K50_rB'], 0)}。预算归因"
        "表明，SARA 的收益既来自排序器对语义与可用性特征的利用，也来自语义候选进入名单；把预算扩大到约 950 个候选只带来有限的额外收益，"
        "说明 500 个候选已是一个高效的工作点。",
        f"模型解释揭示了 SARA 的决策逻辑。排序器的特征增益主要来自检索通道（{pc(fam['V6 检索通道'], 0)}）与融合名次"
        f"（{pc(fam['候选融合'], 0)}），语义特征约占 {pc(fam['语义'], 0)}；但对缺少参考路径的阳性，语义族（{sgn(ca['语义'], 2)}）与融合名次"
        f"（{sgn(ca['候选融合'], 2)}）提供了主要的正向贡献，检索通道的贡献则为负（{sgn(ca['V6 检索通道'], 2)}）。也就是说，SARA 并没有用"
        "语义证据替代检索证据，而是在检索证据缺失处由语义证据接管——这正是可用性感知设计的目标。分层分析同时显示，SARA 的收益随功能"
        "注释的丰富度增大；受控实验进一步表明，对于完全缺少关联监督的基因，功能注释经由语义模型提供了主要的检索信号。因此，提高注释"
        "稀少基因的功能预测质量（例如借助蛋白语言模型[@lin2023esm2; @xu2023protst]）是扩展 SARA 覆盖范围最直接的方向。",
        "SARA 的评价框架本身也可作为疾病基因检索研究的参考。我们在家族层面隔离训练、验证与测试疾病，所有二级特征均在内层上下文中袋外"
        "生成，所有配置均在独立的开发家族上选择；三层证据删除与数量匹配的随机对照直接检验了共享文献的影响；受控的功能注释屏蔽与关联"
        "监督撤出则把“语义证据来自哪里”变成可以检验的问题。这些设计使 SARA 的增益能够被归因到具体的信息来源，而不是评价中的信息泄漏"
        "[@kaufman2012]。",
        "本研究仍有若干局限。第一，阳性关联来自人工整理的 CTD 数据库，未被标注的候选并不等于阴性[@bekker2020]，前列中的新候选需要实验"
        "或文献验证。第二，交叉验证折与外部标签集合在方法的迭代开发中被使用过，本文的结果属于回顾性的内部验证证据，仍需前瞻性或全新"
        "来源数据的独立检验；在以截止日期前资源重建的时间外推中，SARA 与检索通道 LightGBM 表现相当。第三，SARA 的语义收益依赖功能注释的"
        "覆盖，对缺少注释的基因帮助有限。第四，表达队列只提供相关性证据。未来的工作将在前瞻性数据与 DisGeNET[@pinero2020] 等更多外部"
        "资源上验证 SARA，并探索高质量的功能预测以扩展其对注释稀少基因的覆盖。",
        "综上，SARA 通过疾病—功能语义检索、可用性感知的候选融合与明示缺失的排序学习，在知识覆盖不均衡的条件下显著提升了疾病基因检索的"
        "性能与冷启动基因的可检索性。其可解释、可审计的设计为利用异构且不完整的生物医学知识提供了一个通用框架。",
    ])


# ============================================================================ methods
def _meta(N) -> dict:
    """data counts of the methods and Table 1 (all read from the frozen inputs)."""
    genes = pd.read_parquet(PUB / "genes.parquet")
    dis = pd.read_parquet(PUB / "diseases.parquet")
    gt = genes["gtype"].value_counts()
    role = dis.groupby("role").agg(n=("disease_id", "size"), rel=("n_rel", "sum"), fam=("family_group", "nunique"))
    au = N["z"]["audit"]
    ext = pd.DataFrame(N["z"]["ext"]["table"])
    ext = ext[ext["method"] == "deployed"]
    return {
        "G": len(genes), "protein": int(gt.get("protein", 0)), "ncrna": int(gt.get("ncrna", 0)),
        "pseudo": int(gt.get("pseudo", 0)), "other": int(gt.get("other", 0)),
        "n_rel": int(dis["n_rel"].sum()), "dis_rel": int((dis["n_rel"] > 0).sum()), "n_dis": len(dis),
        "test": int(role.loc["test_fold", "n"]), "test_rel": int(role.loc["test_fold", "rel"]),
        "test_fam": int(role.loc["test_fold", "fam"]), "dev": int(role.loc["dev", "n"]),
        "ext_heldout": int(role.loc["external", "n"]),
        "go_rec": int(au["counts"]["production_records"]), "go_rows": int(au["counts"]["rows_read"]),
        "has_go": int(genes["has_go"].sum()), "has_seq": int(genes["has_seq"].sum()),
        "has_expr": int(genes["has_expr"].sum()), "has_ghpo": int(genes["has_gene_hpo"].sum()),
        "has_ppi": int(genes["has_ppi"].sum()),
        "ext_union": _row(ext, subset="union"), "ext_indep": _row(ext, subset="independent_evidence"),
    }


def acceptance_checks(R) -> tuple[int, bool]:
    """(number of whole-chain consistency checks, all passed)."""
    flags = [bool(vv) for v in R["acceptance"].values() if isinstance(v, dict)
             for kk, vv in v.items() if str(kk).endswith("pass")]
    return len(flags), all(flags) and bool(R["acceptance"].get("all_pass"))


def methods(R, N) -> str:
    z = N["z"]
    M = _meta(N)
    au = z["audit"]
    l3 = au["layer3_evidence_blocks"]
    tx = json.loads((paths8.AUDIT / "text_segments_l1.json").read_text(encoding="utf-8"))
    mk = json.loads((paths8.MANIFESTS / "go_masks_v8.json").read_text(encoding="utf-8"))
    tr = json.loads((paths8.RUNS / "V8T__main__f0_outer__background__s42" / "result.json").read_text(encoding="utf-8"))
    cc, tc = tr["config"]["core"], tr["config"]["train"]
    acc = R["acceptance"]
    rep = z["repro"]
    hc = z["hist"]
    fl = z["ext"]["fields"]
    eu = M["ext_union"]
    so = fl["shared_original_evidence"]
    ind = M["ext_indep"]
    n_checks, all_ok = acceptance_checks(R)
    assert all_ok, "a whole-chain consistency check did not pass"
    return "\n\n".join([
        "## 4 材料与方法",
        "### 4.1 数据与任务定义",
        f"基因背景为 {i0(M['G'])} 个可映射到 HGNC[@seal2023] 的基因（蛋白编码 {i0(M['protein'])} 个、非编码 RNA {i0(M['ncrna'])} 个、"
        f"假基因 {i0(M['pseudo'])} 个、其他 {i0(M['other'])} 个），缺少序列、GO 或表达的基因同样保留在背景中。阳性为 CTD[@davis2025] 中具有"
        f"直接证据的 marker/mechanism 类疾病—基因关联（{i0(M['n_rel'])} 条，覆盖 {i0(M['dis_rel'])} 个疾病）；治疗类关联不作阳性，只用于"
        "分层评价。疾病信息包括名称、定义与同义词文本（以固定的 HGNC 符号词典掩蔽基因符号）、经严格等价映射的 HPO 表型注释[@gargano2024]"
        "（只保留表型异常分支）、MeSH 树号[@lipscomb2000]、疾病本体[@schriml2022] 与 OMIM 映射[@amberger2019]；基因信息包括 GO 注释"
        "[@ashburner2000; @go2023]、来源排除后的基因—表型注释、STRING v12 物理互作（置信度 ≥700）[@szklarczyk2023]、代表蛋白序列"
        "[@morales2022; @uniprot2023] 与 HPA 共识组织表达[@uhlen2015]（表 1）。",
        "任务定义为：给定疾病 $d$ 的合法文本、表型与本体信息，以及当前上下文中可见的参考疾病关联，在全部背景基因中检索 $d$ 的关联基因。"
        "目标疾病及其严格等价疾病的任何关联都不进入参考图、基因—表型来源或任何二级训练特征；基因—表型注释按来源疾病排除当前上下文的"
        "全部留出疾病后才聚合到基因。",
        "### 4.2 家族隔离的嵌套交叉验证与袋外特征",
        "每个疾病被预先分配到唯一的疾病家族（MeSH 树号的前三级分支；无树号时取最近祖先的分支，再无则取严格等价组），所有划分都在家族"
        f"层面进行。有阳性的疾病分为互不重叠的开发集（{i0(M['dev'])} 个疾病）、外部留出集（{i0(M['ext_heldout'])} 个）与五个外层折（合计 "
        f"{i0(M['test'])} 个测试疾病、{i0(M['test_rel'])} 条关联、{i0(M['test_fam'])} 个家族）。外层 $j$ 以 $F_j$ 为测试集、$F_{{j+1}}$ 为"
        "验证集、其余三折为训练集 $T_j$；$T_j$ 再按家族分为三个内块 $I_{jb}$，形成每折三个内上下文与一个外上下文，共 20 个拟合上下文。",
        "所有需要学习的对象——语义模型、检索通道、候选名单、排序器及其归一化——都在各自的上下文中袋外生成：内块 $I_{jb}$ 的二级训练"
        "特征由以 $T_j\\setminus I_{jb}$ 训练的模型产生，外上下文的模型以 $T_j$ 训练并为 $F_j$、$F_{j+1}$ 与开发集打分。排除规则按角色"
        "而非标签生成（外上下文的留出集合为 $F_j$、$F_{j+1}$、开发集与外部留出集之并，内上下文再加 $I_{jb}$）。移除测试标签后重新生成的"
        f"检索顺序与候选名单与原结果完全一致，排序分数的最大差为 {sci(acc['1_label_removal']['deployed_score_maxabs'])}。",
        "### 4.3 多源检索通道",
        "检索通道在当前上下文的全部背景基因上为每个疾病计算七个来源的分数，参考疾病只使用其可见关联[@kohler2008; @kohler2009]。"
        "（1）表型迁移：以信息量加权、祖先闭包后的 HPO 余弦相似度选取 20 个最相似的参考疾病，按归一化相似度转移其可见关联；"
        "（2）文本：以基因符号掩蔽后的 TF-IDF 余弦选取参考疾病并同样转移；（3）本体：以 MeSH 树号最具信息量的共同祖先（Resnik 信息量）"
        "选取参考疾病并转移；（4）层级：CTD 疾病层级的两步邻域；（5）基因—表型：疾病表型画像与来源排除后的基因表型画像的信息量加权"
        "余弦；（6）表型传播：以表型迁移分数为种子，在 STRING 物理互作网络上作重启概率 0.15 的随机游走；（7）兼容：疾病表型画像与经可见"
        "参考关联聚合的基因表型画像的余弦。各来源在全背景上的名次以倒数名次融合[@cormack2009]，缺失来源被掩蔽并重新分配权重，融合"
        "权重在开发集上以宏平均 U@10 为目标选择；有信号的基因按融合分数排序，其余按度先验排序，并以 GeneID 打破并列，得到全背景检索顺序"
        " $r_B$。",
        "### 4.4 疾病—功能语义模型",
        f"语义模型以冻结的 SapBERT[@liu2021sapbert] 编码疾病文本、HPO 术语与 GO 术语，以冻结的 ESM-2 35M[@lin2023esm2] 编码代表蛋白序列。"
        "疾病文本（名称、定义与至多 8 个同义词，均经基因符号掩蔽）以句界优先的方式装段，每段不超过 126 个词片，段向量按词片数加权平均，"
        f"使 {i0(tx['n_truncated_v71'])} 个（{pc(tx['share_truncated_v71'])}）超过单段长度的疾病文本得以完整编码；疾病表型表示为其 HPO 术语"
        f"向量按注释频率的加权平均，基因功能表示为其 GO 注释术语向量的聚合。共享语义投影 $P_T$（秩 {cc['rank']}）把疾病文本—表型与基因功能"
        "映射到同一空间，序列经另一投影 $P_S$ 映射到该空间；功能、序列与表达三个模态的匹配分数由全局门控合并为核心分数 $r$。训练目标为"
        " $\\mathcal L=\\mathcal L_{DG}+\\lambda_A\\mathcal L_A$：$\\mathcal L_{DG}$ 是以当前上下文可见阳性为正例、未标注基因为对照"
        f"[@elkan2008; @bekker2020] 的多正例排序损失（作用于 $r$，温度 τ = {tc['tau']}），$\\mathcal L_A$ 把基因的序列表示与其 GO 功能锚"
        f"（注释 5–500 个背景基因的术语）对齐（$\\lambda_A$ = {tc['lambda_A']}）。模型共 {i0(tr['fit']['n_params'])} 个可训练参数，以 AdamW"
        f"[@loshchilov2019]（学习率 {tc['lr']}，权重衰减 {tc['weight_decay']}，每批 {tc['batch']} 个疾病）训练至多 {tc['max_epochs']} 轮，"
        f"以同一上下文 $F_{{j+1}}$ 上 $s_F$ 的全背景 Recall@10 选择检查点（耐心 {tc['patience']}）；未标注对照集合（参考关联基因或全背景）"
        "在开发集上选择。SARA 只使用其中不依赖已知关联的功能通道分数：",
        "$$z_d=\\mathrm{norm}\\{P_T(x^{\\mathrm{text}}_d,x^{\\mathrm{HPO}}_d)\\},\\qquad z^F_g=\\mathrm{norm}\\{P_T\\,x^{\\mathrm{GO}}_g\\},\\qquad "
        "s_F(d,g)=z_d^{\\top}z^F_g ,$$",
        "没有 GO 注释的基因记为语义不可用。",
        "### 4.5 可用性感知的倒数名次融合与候选生成",
        "令 $r_S(d,g)$ 为语义可用基因中的语义名次，$m_{dg}\\in\\{0,1\\}$ 为语义可用标记。SARA 以",
        "$$b_{dg}=\\frac{1}{60+r_B(d,g)},\\qquad e_{dg}=\\frac{m_{dg}}{60+r_S(d,g)},\\qquad R(d,g)=\\frac{w\\,b_{dg}+(1-w)\\,m_{dg}\\,e_{dg}}"
        "{w+(1-w)\\,m_{dg}}$$",
        "融合两条顺序[@cormack2009]：语义不可用时 $R=b$，候选不因缺少一种模态而被额外降分；若全部基因的语义都不可用，候选名单与检索前"
        " 500 完全相同。普通 RRF（$w\\,b+(1-w)\\,m\\,e$，使用相同的 $w$）作为对照。每个疾病取 $R$ 最高的恰好 500 个不同基因作为候选，"
        "以 GeneID 打破并列，从不补入任何已知阳性。权重 $w\\in\\{0.3,0.5,0.7,0.9\\}$ 在每个外层上下文按开发集覆盖选择（平局取较大 $w$；"
        f"五个外层依次为 {'、'.join(str(v) for v in z['dev']['w'])}）。",
        "### 4.6 SARA 排序器",
        "SARA 排序器为 LightGBM[@ke2017] 的 LambdaRank 模型[@burges2010]，在 SARA 候选上以 49 个特征重排：30 个检索特征（七个来源的分数、"
        "对数名次与可用性标记，融合分数、检索名次、补足标记、度先验与五项疾病描述统计）与 19 个语义与可用性特征（$s_F$ 及其对数名次、"
        "融合分数 $R$ 的对数名次、语义可用与来源标记、表达匹配分数及其可用性、参考关联与基因—表型可见性、参考度、基因类型与各模态"
        "可用性、疾病文本/表型/组织信息的可用性）。语义与表达缺失以 NaN 明示输入，由决策树的缺失分支学习其含义，而不是以 0 填充。"
        "排序器以三个内块的袋外候选与特征训练，学习率 0.05，至多 1,500 棵树并以外层 $F_{j+1}$ 的 Recall@10 早停，行、列采样率均为 0.8；"
        "叶数 {15, 31, 63} × 最小叶样本 {50, 200} 的六个配置在开发集上选择（表 S2）。我们还考察了两种替代设计——在基础排序上叠加"
        "门控残差的冷平衡纠偏器，以及以蛋白语言模型补全缺失 GO 的序列补位模型——二者均未进入最终模型（补充说明 S4）。",
        "### 4.7 功能证据政策与证据共享审计",
        f"GO 注释按 GAF 2.2 的全部 17 列解析，去除 NOT 限定与 ND 证据，保留证据代码与引文，生产规则保留 {i0(M['go_rec'])} 条记录（有 GO "
        f"的基因 {i0(M['has_go'])} 个）。审计包含三层删除：（1）训练可见来源排除，删除引用本上下文训练关联文献的记录；（2）评价条件化"
        "删除，删除引用任何 CTD 关联文献（含评价关联）的记录，只作敏感性分析；（3）证据块隔离，以 PMID—关联二部图的连通块为单位"
        f"（巨块不拆分）随机抽取至覆盖 {pc(l3['target_share'], 0)} 的关联，同时隔离块内关联、其文献的 GO 记录与可追溯的表型注释"
        f"（{i0(l3['blocks_chosen'])} 个块、{i0(l3['relations_held'])} 条关联、{i0(l3['pmids_held'])} 篇文献）。每层配置逐基因数量匹配的"
        "随机删除；证据代码变体包括仅保留实验类证据、去除 TAS/NAS 与去除突变/遗传相互作用类证据（IMP/IGI/HMP/HGI）。每种条件均从头"
        "重训整条流程（种子 42），比较 SARA 相对检索通道 LightGBM 的优势在删减前后的变化；去除 IC 与仅保留 IEA 为只替换推理输入的"
        "重打分敏感性分析。",
        "### 4.8 受控的功能注释缺失与关联监督冷",
        f"在不读取任何疾病标签的条件下，从有 GO 与序列的 {i0(mk['eligible_genes'])} 个基因中按“GO 术语数十分位 × 物理互作度十分位”分层，"
        f"以稳定哈希顺序抽取 10% 作为开发屏蔽集（{i0(mk['G_mask_dev'])} 个）、其后 20% 作为测试屏蔽集（{i0(mk['G_mask_test'])} 个），"
        "两者不相交。删除屏蔽基因的 GO 及其派生量发生在语义模型训练、功能锚与候选生成之前。关联监督冷进一步把测试屏蔽基因的全部疾病"
        "关联从参考图、基因—表型注释与所有训练表中撤出，并设同时删除 GO 的组合条件。",
        "### 4.9 对照方法",
        "对照方法包括：检索通道 LightGBM（30 个检索特征，在其自身检索前 500 上训练）；基础排序器（同样的 30 个特征，在 SARA 候选上以"
        "袋外方式训练）；固定权重融合（检索来源分数的固定权重线性融合，权重在开发集上选择）；多模态直接融合（30 个检索特征加 14 项"
        "多模态语义核心特征，缺失写 0，在同一 SARA 候选上重训）；以全背景语义分数单独排名的 $s_F$、核心分数 $r$ 与序列分数；未经训练的 "
        "SapBERT 疾病—功能相似度；以及在同一候选名单上以 $F_{j+1}$ 选择检查点重训的 KDGene[@wang2024kdgene] 与 SIGN 式图模型[@frasca2020]"
        "（表 S3）。支持任务比较等权随机游走、语义与随机游走的组合及连续时间量子游走[@saarinen2024]。",
        "### 4.10 外部标签、时间外推、支持任务与表达队列",
        f"外部标签集合为 {i0(fl['diseases'])} 个测试疾病的 GenCC[@distefano2022]（分级为 Definitive 或 Strong）与 ClinVar[@landrum2020]"
        f"（审核星级 ≥2）关联（{i0(fl['pairs'])} 条，合并去重后 {i0(eu.get('n_pairs', 0))} 对）。每个疾病以其作为测试疾病的外层模型评价，"
        "因而均未作为训练疾病出现；评价时从排名中移除该疾病的 CTD 阳性（过滤排名），三个种子平均。我们为每条关联记录其是否与训练关联"
        f"或该基因 GO 注释共享原始文献（是 {i0(so['yes'])} 条、否 {i0(so['no'])} 条、无法追溯 {i0(so['unknown'])} 条），并单独报告文献独立"
        f"子集（{i0(ind.get('n_diseases', 0))} 个疾病、{i0(ind.get('n_pairs', 0))} 对）。时间外推以 2023-12-31 为截止日，使用截止前版本的 CTD 关联"
        "与疾病词表、HPO 与基因—表型注释、疾病本体、GO/GOA、HGNC 与 GTEx v8 表达[@gtex2020] 重建全部输入并重训全部模型，评价截止时已有"
        f"关联的疾病的新关联（{i0(hc['x2_deployed_vs_channel_lgb_v6']['n'])} 个疾病）与截止时尚无关联的新疾病"
        f"（{i0(hc['x1_deployed_vs_channel_lgb_v6']['n'])} 个疾病）；截止日之后的资源（基因背景、蛋白序列与疾病家族）逐项列于表 S9。"
        "支持任务在 533 个疾病上显露 k ∈ {0, 1, 3, 5} 个已知基因，并在 760 个疾病上评价单样本（k = 1）情形；语义与随机游走的组合权重"
        f" ω 在开发集上选择（ω = {z['omega']}）。表达验证使用银屑病（GSE13355、GSE30999）与溃疡性结肠炎（GSE87466、GSE75214）的四个 GEO"
        "[@barrett2013] 队列：每个基因取平均表达最高的探针，以方差收缩的调节 t 统计量（参照 limma 的经验贝叶斯思想[@ritchie2015]）"
        "度量差异表达，比较冻结的前 20/50 个基因与在平均表达、物理互作度与已知关联数上匹配的背景基因的统计量秩；另以在互作度与关联数"
        "上逐一匹配的随机基因集作为对照。",
        "### 4.11 评价指标与统计推断",
        "主要指标为疾病宏平均的 Recall@10：",
        "$$R_d@K=\\frac{|Q_d\\cap \\mathrm{Top}_K(C_d)|}{|Q_d|},\\qquad R^c_d@K=\\frac{|Q^c_d\\cap\\mathrm{Top}_K(C_d)|}{|Q^c_d|},$$",
        "其中 $C_d$ 为疾病 $d$ 的候选名单，$Q_d$ 为测试关联，$Q^c_d$ 为参考路径冷的测试关联（该基因在评价上下文中既无参考关联，也无来源"
        "过滤后的基因—表型注释）；$R^c$ 只在含冷查询的疾病上定义。另报告 Hit@10、候选覆盖（覆盖@500）与可达上界 U@10 = min(10, $m_d$)/"
        "$n_d$（$m_d$ 为候选中的阳性数、$n_d$ 为阳性总数）。三个种子（42、52、62）先在疾病内平均；没有阳性进入候选的疾病保留在分母中。"
        "所有区间为按疾病家族重抽样的 bootstrap 95% 百分位区间[@efron1993; @field2007]：主要比较 10,000 次（种子 20260923），描述性区间 "
        "1,000–2,000 次；关系级比例以家族为簇对分子与分母同时重抽样。除主要比较外，其余分析只报告估计值与区间。",
        "### 4.12 模型解释与可视化",
        "模型解释使用 LightGBM 内置的 TreeSHAP[@lundberg2017; @lundberg2020]：对外层 0、种子 42 的 SARA 排序器，由保存的模型与重建的袋外"
        "特征矩阵计算全部留出阳性与 4 万个抽样未标注候选的逐特征贡献，并核对 SHAP 之和与保存的排序分数一致。分布图使用高斯核密度估计"
        "（名次在对数尺度上，有界处做反射校正），层次聚类为欧氏距离的平均连接，来源可用性以 UpSet 图[@lex2014] 展示，疾病表示以 t-SNE"
        "[@vandermaaten2008] 可视化；分类配色经色觉缺陷模拟[@machado2009] 校验，同一方法在全部图中使用同一颜色。",
        "### 4.13 实现与可复现性",
        f"全部计算在单机完成：Python {rep['environment']['python'].split()[0]}、PyTorch {rep['environment']['torch'].split('+')[0]}"
        f"（CUDA 12.8）、LightGBM {rep['environment']['lightgbm']} 与 NumPy {rep['environment']['numpy']}，单块 NVIDIA GeForce RTX 5060 Ti。"
        f"全部 {i0(N['runs'])} 次拟合与评价运行均以内容哈希记账，有计时的运行合计约 {num(N['timed_h'], 1)} 小时，单次运行的峰值显存 "
        f"{num(N['gpu_peak'] / 1024, 2)} GiB。在同一计算环境中，我们以全新缓存从冻结输入重算了第 0 折、种子 42 的完整子链：候选名单与"
        f"前十完全一致，排序分数的最大差为 {sci(rep['deployed_scores_maxabs'])}。另有 {n_checks} 项整链一致性检查全部通过，包括标签移除"
        "不变、过滤先于聚合、袋外谱系、检查点与训练集不重叠、GO 屏蔽与关联监督撤出的完整性、候选预算的精确性、语义缺失时的精确回退、"
        "批次不变性与种子规则。",
    ])


# ============================================================================ end matter
def endmatter() -> str:
    return "\n\n".join([
        "## 数据与代码可用性",
        "本研究使用的数据均来自公开资源：CTD、HPO、Gene Ontology 与 GOA 人类注释、MeSH、Disease Ontology、OMIM、HGNC、STRING v12、"
        "Human Protein Atlas、MANE 与 UniProt、GTEx v8、GenCC、ClinVar 与 GEO（GSE13355、GSE30999、GSE87466、GSE75214）。SARA 的代码、"
        "冻结的模型配置、各图面板的作图数据（CSV）与全部评价结果将在论文接收后公开于［代码仓库链接］与［数据存档 DOI］；所用资源的版本"
        "与下载日期随代码一并提供。",
        "## 资助",
        "［资助机构与项目编号］",
        "## 利益冲突",
        "［作者声明无利益冲突，或列出利益冲突］",
        "## 作者贡献",
        "［作者 A：构思、方法、软件、分析、写作——初稿；作者 B：……；通讯作者：指导、写作——审阅与修改］",
        "## 致谢",
        "［致谢内容］［如在研究或写作中使用了人工智能工具，请按期刊要求在此说明其用途与范围］",
        "## 作者简介",
        "［作者姓名］，［职称/学位］，［单位］，研究方向为［……］。",
    ])


# ============================================================================ tables
def table1(R, N) -> str:
    z = N["z"]
    M = _meta(N)
    au = z["audit"]
    l3 = au["layer3_evidence_blocks"]
    hc = z["hist"]
    x = N["x"]
    s1 = {r["method"]: r for r in z["support1"]}
    tx = json.loads((paths8.AUDIT / "text_segments_l1.json").read_text(encoding="utf-8"))
    mk = json.loads((paths8.MANIFESTS / "go_masks_v8.json").read_text(encoding="utf-8"))
    rows = [
        ["基因背景", i0(M["G"]), f"蛋白编码 {i0(M['protein'])}，非编码 RNA {i0(M['ncrna'])}，假基因 {i0(M['pseudo'])}，其他 {i0(M['other'])}"],
        ["阳性关联 / 有阳性的疾病", f"{i0(M['n_rel'])} / {i0(M['dis_rel'])}", "CTD marker/mechanism，直接证据"],
        ["测试疾病 / 关联 / 家族", f"{i0(M['test'])} / {i0(M['test_rel'])} / {i0(M['test_fam'])}", "五个外层折，家族隔离"],
        ["开发疾病 / 外部留出疾病", f"{i0(M['dev'])} / {i0(M['ext_heldout'])}", "开发集只用于配置选择；外部留出集不参与训练"],
        ["GO 记录 / 有 GO 的基因", f"{i0(M['go_rec'])} / {i0(M['has_go'])}", "GAF 2.2；去除 NOT 与 ND"],
        ["有序列 / 有表达的基因", f"{i0(M['has_seq'])} / {i0(M['has_expr'])}", "代表蛋白（ESM-2 35M）；HPA 共识组织表达"],
        ["有基因—表型注释 / 物理互作的基因", f"{i0(M['has_ghpo'])} / {i0(M['has_ppi'])}", "来源排除的 HPO；STRING v12（≥700）"],
        ["分段编码的长文本疾病", f"{i0(tx['n_truncated_v71'])}（{pc(tx['share_truncated_v71'])}）", f"共 {i0(tx['segments_total'])} 段，每段 ≤126 词片"],
        ["GO 屏蔽集合（开发 / 测试）", f"{i0(mk['G_mask_dev'])} / {i0(mk['G_mask_test'])}", "不读取标签；分层哈希抽取"],
        ["证据块（块 / 关联 / 文献）", f"{i0(l3['blocks_chosen'])} / {i0(l3['relations_held'])} / {i0(l3['pmids_held'])}",
         f"巨块 {i0(l3['giant_component_relations'])} 条关联不拆分"],
        ["外部标签（疾病 / 关联对）", f"{i0(z['ext']['fields']['diseases'])} / {i0(M['ext_union'].get('n_pairs', 0))}", "GenCC 与 ClinVar"],
        ["时间外推（新关联 / 新疾病）", f"{i0(hc['x2_deployed_vs_channel_lgb_v6']['n'])} 病、{i0(x['hp']['x2']['pairs'])} 对 / "
                                   f"{i0(hc['x1_deployed_vs_channel_lgb_v6']['n'])} 病、{i0(x['hp']['x1']['pairs'])} 对", "截止 2023-12-31"],
        ["支持任务", f"533 病（k = 0–5）/ {i0(s1['joint_v6']['n_diseases'])} 病（k = 1）", "显露已知支持基因"],
        ["表达队列", "4", "GSE13355、GSE30999、GSE87466、GSE75214"],
    ]
    return "表 1　数据与评估集合\n\n" + table_md(["项目", "数量", "说明"], rows)


TABLE2_ORDER = ["deployed", "v71_direct_strict", "channel_lgb_v6", "B", "F7_strict", "F7_v6", "v71_direct_v6", "standalone_sF",
                "standalone_r", "standalone_sF0", "standalone_sS"]


def table2(R, N) -> str:
    z = N["z"]
    rows = []
    for m in TABLE2_ORDER:
        r = z["table"][m]
        name = "**SARA**" if m == "deployed" else pub(m)
        rows.append([name, num(r["recall10"]), num(r["recall20"]), num(r["recall50"]), num(r["hit10"]), num(r["coverage500"]),
                     num(r["U10"]), num(r["cold_recall10"])])
    t = table_md(["方法", "R@10", "R@20", "R@50", "Hit@10", "覆盖@500", "U@10", "$R^c$@10"], rows)
    return (f"表 2　主要比较（{i0(z['table']['deployed']['n_diseases'])} 个测试疾病，三个种子平均）\n\n" + t +
            "\n\n注：除注明“检索前 500”者外，各排序模型均在 SARA 候选上评价；单独语义排名取全背景前 500。$R^c$@10 只在含参考路径冷"
            f"查询的 {i0(N['cold'][0]['n_cold_diseases'])} 个疾病上定义。在同一候选名单上重训的 KDGene 与 SIGN 式图模型（种子 42）的 Recall@10 "
            f"分别为 {num(z['comp']['kdgene']['recall10'])} 与 {num(z['comp']['sign']['recall10'])}（表 S3）。")


def table3(R, N) -> str:
    z = N["z"]
    b = {(r["table"], r["method"]): r for r in z["budget"]["table"]}
    v = z["var"]
    miss = pd.DataFrame(N["miss"])

    def nc(li):
        return pc(_row(miss, group="非编码 RNA", list=li).get("pool_share"))
    ch6, a6 = b[("1_fixed_v6_500", "仅通道 LightGBM")], b[("1_fixed_v6_500", "候选 A（部署规则）")]
    a5, a9 = b[("2_shared_new_500", "候选 A（部署）")], b[("4_bridge_950", "候选 A（同 950 预算）")]
    c9 = b[("4_bridge_950", "仅通道 LightGBM（同 950 预算）")]
    pr, eo, er = v["plain_rrf"]["deployed"], v["expr_off"]["deployed"], v["expr_random"]["deployed"]
    def d(e):
        return ci(e, label=False)
    rows = [
        ["检索前 500", "检索通道 LightGBM", num(ch6["recall10"]), pc(ch6["coverage"]), nc("V6 500（仅通道）"), "参照", "—"],
        ["检索前 500", "SARA 排序器", num(a6["recall10"]), pc(a6["coverage"]), nc("V6 500（仅通道）"), "减上一行",
         d(z["budget"]["v6pool_A_vs_channel"])],
        ["SARA 候选（500）", "SARA", num(a5["recall10"]), pc(a5["coverage"]), nc("缺失感知严格 500"), "减上一行",
         d(z["budget"]["strict_A_vs_v6pool_A"])],
        ["普通 RRF 候选（500）", "SARA 排序器", num(pr["recall10"]), pc(pr["coverage500"]), nc("普通 RRF 严格 500"), "减 SARA",
         d(pr["delta_vs_main"])],
        ["扩展候选（约 950）", "SARA 排序器", num(a9["recall10"]), pc(a9["coverage"]), "—", "减 SARA",
         d(z["budget"]["bridge_A_vs_strict_A"])],
        ["扩展候选（约 950）", "基础排序器", num(c9["recall10"]), pc(c9["coverage"]), "—", "上一行减本行",
         d(z["budget"]["bridge_A_vs_B"])],
        ["SARA 候选（500）", "SARA，关闭表达特征", num(eo["recall10"]), pc(eo["coverage500"]), "—", "减 SARA",
         d(eo["delta_vs_main"])],
        ["SARA 候选（500）", "SARA，随机组织表达", num(er["recall10"]), pc(er["coverage500"]), "—", "减 SARA",
         d(er["delta_vs_main"])],
    ]
    return ("表 3　候选预算、融合方式与表达特征（种子 42）\n\n" +
            table_md(["候选名单", "排序模型", "R@10", "覆盖", "ncRNA 阳性入选", "比较", "R@10 差值（95% CI）"], rows) +
            "\n\n注：覆盖为测试阳性进入候选名单的比例；ncRNA 阳性入选为非编码 RNA 阳性进入候选的比例（关系级）。扩展候选为检索前 500 "
            f"与语义前 500 之并（平均 {i0(a9['pool_size'])} 个基因）；各行的排序模型均在对应名单上以袋外方式重训。")


# ============================================================================ figure legends
FIGS = [("1", "Fig1_knowledge_landscape", "已知关联与信息层可用性的结构性不均衡"),
        ("2", "Fig2_method_overview", "SARA 框架与评价设计"),
        ("3", "Fig3_main_results", "SARA 的主要性能"),
        ("4", "Fig4_budget_fusion_flow", "可用性感知的候选生成与名次流转"),
        ("5", "Fig5_knowledge_strata", "SARA 的收益在知识覆盖各层的分布"),
        ("6", "Fig6_ranker_explanation", "SARA 排序器的特征贡献与 TreeSHAP 解释"),
        ("7", "Fig7_annotation_supervision", "功能注释与关联监督对冷启动基因的作用"),
        ("8", "Fig8_evidence_sources", "SARA 的增益对证据共享稳健"),
        ("9", "Fig9_external_temporal_applications", "外部标签、时间外推与应用任务"),
        ("10", "Fig10_case_studies", "案例：SARA 的前列基因与基因网络")]
CI_NOTE = "区间为按疾病家族重抽样的 bootstrap 95% 区间。"


def legends(R, N) -> list[tuple[str, str, str]]:
    M = _meta(N)
    L = {
        "1": ("**a** 按关联数升序排列的累计比例曲线（Lorenz 曲线）；阴影为基因曲线与均等线之间的面积，圆点标出前 10% 基因所占份额。"
              "**b** 三种计数的互补累计分布 $P(X\\geq k)$（双对数轴，仅计 ≥1 的对象）。**c** 按 GO 注释数分组后各组内已知关联数的分布"
              "（列内比例，≥10% 的格子标注数值；顶部为每组基因数）；ρ 为全体基因的 Spearman 相关。**d** 六个信息层的 UpSet 图：上方为"
              "前 14 个交集的基因数（深色为蛋白编码），下方点阵标出交集组成，左侧为各层大小。**e** 按已知关联数分组的疾病描述完整度"
              "（括号内为疾病数）。**f** 关联基因最多的 12 个 MeSH 大类之间共享关联基因的弦图：弧长 ∝ 大类内关联基因数，带宽 ∝ 两类共享"
              "的基因数，弧色为该类关联基因 GO 基因—术语对的中位数，蓝色带为 Jaccard 系数最高的 6 对。**g** 各大类（有已知关联的疾病 "
              "≥40 个）的九项知识覆盖指标——疾病侧：疾病数、每病关联数中位数、有定义与有 HPO 的比例、文本词片中位数；基因侧（该类关联"
              "基因）：GO 基因—术语对中位数、已知关联数中位数、有物理互作与无 GO 的比例。颜色为列内 z 分数（红低蓝高），格内为原始值；"
              f"左侧与上方为平均连接的层次聚类树。n = {i0(M['G'])} 个基因、{i0(M['n_dis'])} 个疾病。"),
        "2": ("**a** SARA 流程：多源检索通道在全部背景基因上给出检索顺序 $r_B$，疾病—功能语义模型给出语义分数 $s_F$ 与语义名次 $r_S$；"
              "可用性感知 RRF 为每个疾病生成恰好 500 个候选，SARA 排序器以 49 个袋外特征重排（框内为各特征族在排序器中的平均增益占比）；"
              "虚线箭头为直接进入排序器的特征。**b** 外层五折：深色为测试折、中灰为验证折、浅色为训练折，训练折按家族分为三个内块 "
              "$I_{jb}$；右侧为与五折分离的开发、外部与时间外推集合。**c** $w=0.5$ 时可用性感知分数 $R$ 随 $r_B$、$r_S$ 的变化（对数色标）；"
              "黑线为测试疾病第 500 个候选分数的中位边界，白线为四分位边界；下方色条为语义不可用（$m=0$）时的分数，竖线分别为普通 RRF "
              "与可用性感知 RRF 的入选截断。**d** 三类“冷”在各信息项上的状态（实心：保留；空心：撤出或缺失）。**e** 测试疾病（五折合计）"
              "的七个检索来源可用性 UpSet 图：上方为每种可用性组合的疾病数，中间点阵标出组合内可用的来源，左侧为每个来源可用的疾病数；"
              "下方为该组合内 SARA 相对检索通道 LightGBM 的逐病 Recall@10 差值（三种子平均；圆点与竖线为估计值与 95% 区间；蓝：区间 > 0，"
              "灰：区间含 0）。" + CI_NOTE),
        "3": ("**a** 主要比较的差值分布：半小提琴为 10,000 次家族 bootstrap 的差值分布，圆点与横线为估计值与 95% 区间，右侧为数值及区间是否"
              "包含 0。**b** 冷恢复：实心为 $R^c$@10（只在含冷查询的疾病上定义），空心为 $R^{c\\to all}$@10（冷命中占全部查询的比例）。"
              "**c** Recall@K 曲线（K = 1–100，对数轴），带为 1,000 次家族 bootstrap 的逐点 95% 置信带。**d** 15 个外层 × 种子组合中 SARA "
              "相对检索通道 LightGBM 的差值（形状区分种子；黑色菱形为三种子合并估计）。**e** 全部指标（列内着色，右侧色块为方法的固定"
              "颜色）。**f** 逐疾病的胜负比例（按疾病的三种子平均 Recall@10 比较），括号内为疾病数。**g** 每个疾病最靠前阳性名次的云雨图"
              "（种子 42；上为核密度，中为箱线，下为抽样的逐病点；只画有阳性入选的疾病，对数轴），右侧为没有阳性入选的疾病比例。**h** 五种"
              "方法前十命中（疾病—基因对，种子 42）的重叠弦图：弧长 ∝ 该方法的前十命中数，带宽 ∝ 两种方法共享的命中数。**i** 七项指标的"
              "雷达图（各轴除以四种方法中的最大值；MRR 为疾病内阳性倒数名次的平均）。"
              f"n = {i0(M['test'])} 个测试疾病、{i0(M['test_fam'])} 个家族；" + CI_NOTE),
        "4": ("**a** 测试阳性进入前 K 的比例（疾病宏平均；K 为对数轴）：检索顺序 $r_B$、语义顺序 $r_S$（语义不可用的基因视为未入选）、普通 "
              "RRF 与可用性感知 RRF（各外层的 $w_j$）；带为 95% 置信带，竖线为 K = 500 与约 950。**b** 预算归因：同一排序模型在检索前 500、"
              "SARA 候选与扩展候选（约 950）上的 Recall@10（种子 42；竖线为 95% 区间），左下为三个归因差值。**c** 缺少信息的阳性在三种"
              "名单下进入候选的比例（关系级微平均，括号内为关系数；误差线为以家族为簇的 bootstrap 区间）。**d** 逐关系名次区间的流量"
              "（左：检索通道 LightGBM；右：SARA）；蓝色为名次上升、橙色为下降、灰色为同一区间。**e** 同一批阳性在两种方法中名次的二维"
              "分布（对数色标）。**f** 语义可用的测试阳性在检索名次 $r_B$ 与语义名次 $r_S$ 上的联合分布（种子 42；六边形分箱，对数色标；"
              "整数名次在 [r, r+1) 内均匀抖动后取对数，仅用于显示）：虚线为参考路径冷阳性的二维核密度等高线（网格密度值的第 50、80、95 "
              "百分位），黑线为 $w=0.5$ 时可用性感知 RRF 的中位入选边界，橙线与紫线分别为 $r_B=500$ 与 $r_S=500$；上方与右侧为边缘分布"
              "（深色为冷阳性），四角为各象限的阳性比例。**g** 由检索前 500 换为 SARA 候选时各类阳性的去向（种子 42，关系级）。" + CI_NOTE),
        "5": ("**a** 按基因在评价上下文中的参考关系数分层的关系级前十恢复率（三种子平均；“冷”为无参考边且无基因—表型注释，“0*”为无"
              "参考边但有基因—表型注释；顶部 n 为每个种子的关系数）。**b** 按基因的 GO 基因—术语对数分层。**c** 按疾病文本长度四分位与"
              "是否有 HPO 分层的逐病 Recall@10 差值（SARA 减检索通道 LightGBM；格内为差值与疾病数，n < 20 的格子不着色）。**d** MeSH 大类："
              "左为四种方法的 Recall@10（统一色标），中为 SARA 相对检索通道 LightGBM（圆点）与多模态直接融合（菱形）的差值，右为测试疾病"
              "数。**e** 各 MeSH 大类中逐病最佳阳性名次的对数提升 $\\log_{10}$(检索通道 LightGBM 最佳名次) − $\\log_{10}$(SARA 最佳名次) 的"
              "山脊图（种子 42；未入选记为 501；只计至少一种方法有阳性入选的疾病）：填色为该类平均提升（红负蓝正），右侧为提前（↑）与"
              "推后（↓）的疾病比例和疾病数。" + CI_NOTE),
        "6": ("**a** SARA 排序器的特征增益占比（均值 ± 标准差，n = 15 个模型），颜色为特征族；右侧为各特征族的合计。**b** 按候选类别"
              "合计的各特征族带符号平均 SHAP（括号内为候选数）。**c** 外层 0、种子 42 的 SARA 排序器在测试候选上的 TreeSHAP 蜂群图（全部"
              "留出阳性与 4 万个抽样的未标注候选；按平均绝对 SHAP 取前 14 个特征；点色为特征值在 2–98 百分位内的相对位置，灰色为缺失值）。"
              "**d** 融合名次（可用性感知 RRF 名次，对数轴）的 SHAP 依赖；**e** 语义分数 $s_F$ 的 SHAP 依赖：点色为检索名次 $r_B$（以 500 "
              "为中点的发散色标），空心圈为抽样的 700 个留出阳性，黑线与灰带为分位分箱的均值与 P10–P90。**f** 表达特征的消融：关闭全部"
              "表达特征或以随机组织表达替换后，SARA 的 Recall@10 相对完整模型的差值（种子 42；横线为 95% 区间）。b–e 为单一上下文的模型"
              "解释。"),
        "7": ("**a** 未屏蔽、开发屏蔽（10%）与测试屏蔽（20%）基因的 GO 术语数与物理互作度的累计分布（度为 0 的基因画在 0.5 处）。**b** 测试"
              "屏蔽基因的阳性在完整模型、删除 GO、撤出关联监督及两者同时的条件下进入候选（浅色）与前十（深色）的比例（种子 42；横线为 "
              "95% 区间），右侧为全部疾病的 Recall@10 相对完整模型的变化。**c** 屏蔽基因阳性在 SARA 候选中名次的山脊图（种子 42，全部"
              "外层；只画入选者，对数轴，边界处反射校正的核密度；竖线为中位数），右侧为入选比例与前十比例。**d** 同一批阳性的名次类别在"
              "完整模型、删除 GO 与“撤出监督 + 删除 GO”之间的流转（前 10 / 11–50 / 51–500 / 未入选；带宽 ∝ 阳性数）。" + CI_NOTE),
        "8": ("**a** GO 生产记录按证据代码的数量（深色有 PMID，浅色无 PMID），括号内为代码类别，右侧为无 PMID 的比例。**b** 左：PMID—关联"
              "二部图连通块的大小分布（双对数轴，红点为巨块）；右：按大小降序累计的块所覆盖的有 PMID 关联比例。**c** 各删除条件在五种分母"
              "上的删除比例（对数色标）：引文、注释行、唯一基因—术语对、受影响基因与受影响的查询关联；* 为使用了评价关联文献的敏感性条件。"
              "**d** SARA 相对检索通道 LightGBM 的优势在删减数据与完整数据之间的变化（种子 42，整链重训；方块为匹配随机对照，灰色为只替换"
              "推理输入的重打分敏感性分析）。**e** 同一变化按 MeSH 大类展开（×100；测试疾病 ≥60 的大类）：行为 11 个删除条件，列按相似性"
              "做平均连接聚类（上方为聚类树与各类疾病数），右列为全部疾病；蓝色表示删减后 SARA 的优势扩大，红色表示缩小。" + CI_NOTE),
        "9": ("**a** 外部标签（GenCC 与 ClinVar）上 SARA 相对三种对照的 Recall@10 差值，按合并、GenCC、ClinVar 与文献独立子集分行（括号内"
              "为疾病数）。**b** 外部关联的三个资格字段（是否出现于训练疾病、邻近疾病中是否出现同一关联、是否与训练关联共享原始文献）按"
              "来源的构成。**c** 时间外推（截止 2023-12-31）：新关联任务与新疾病任务上各方法的 Recall@10（条形为均值、竖线为 95% 区间）及 "
              "SARA 相对检索通道 LightGBM 的差值。**d** 支持任务（533 个疾病，显露 k 个支持基因；带为 95% 置信带）。**e** 四个表达队列中"
              "冻结基因集的富集（颜色：选中基因 |t| 高于匹配背景的概率，0.5 为无富集；圆面积 ∝ −log10 P；黑圈为 P < 0.05）。**f** 时间外推"
              "任务中每个测试关联（圆点为新关联任务、三角为新疾病任务；种子 42）在检索通道 LightGBM（检索前 500）与 SARA（SARA 候选）中的"
              "名次，对数轴，未入选的关联画在分隔带外；对角线下方（蓝底）为 SARA 更靠前，上方（橙底）为检索通道 LightGBM 更靠前。**g** 支持"
              "任务的逐病 Recall@10（显露 k = 1 与 5 个支持基因；k = 1 为 10 次抽样的平均，k = 5 为 1 次），平方根坐标；蓝点为 SARA 语义 + "
              "RWR 更高，橙点为基线语义 + RWR 更高，灰点为相同，灰底为点密度。" + CI_NOTE),
        "10": ("**a** 每个案例两行：上为 SARA、下为检索通道 LightGBM 的前 10 名（种子 42，基因符号）；彩色格为被留出的真实关联（CTD "
               "marker/mechanism），黑框与 ° 表示该阳性在评价上下文中为参考路径冷，浅灰为未被标注为关联的基因。案例按作图前写定的规则"
               "选取（见正文）。**b** 三个案例的前 10 基因网络：节点为两种方法前 10 的并集（左弧：只进入 SARA 前 10；右弧：只进入检索通道 "
               "LightGBM 前 10；中列：两者共有），节点内数字为该方法前 10 中的名次，节点面积 ∝ log(1 + GO 基因—术语对数)，黑圈为留出阳性（° 为参考"
               "路径冷）；边为 STRING v12 物理互作（深灰）与功能关联（浅灰）以及 GO 术语集 Jaccard ≥ 0.2（紫色虚线）。"),
    }
    return [(n, stem, f"**图 {n}　{title}。** " + L[n]) for n, stem, title in FIGS]


# ============================================================================ assembly
def compose(R=None, N=None) -> tuple[str, dict]:
    from .refs9 import resolve
    if R is None:
        R, N = numbers9()
    TRACE.clear()
    body = [front(R, N), intro(R, N), r1(R, N), r2(R, N), r3(R, N), r4(R, N), r5(R, N), r6(R, N), r7(R, N), r8(R, N),
            r9(R, N), r10(R, N), discussion(R, N), methods(R, N), endmatter(), "## 参考文献", "<<REFERENCES>>",
            "## 表", table1(R, N), table2(R, N), table3(R, N), "## 图注"]
    for n, stem, leg in legends(R, N):
        body += [f"<!-- figure {stem} -->", leg]
    md = "\n\n".join(p.strip("\n") for p in body) + "\n"
    md, refs, order = resolve(md)
    md = md.replace("<<REFERENCES>>", "\n".join(refs))
    info = {"stem": STEM, "title": TITLE, "ref_order": order, "n_refs": len(refs),
            "figures": [(n, stem) for n, stem, _ in FIGS], "trace": {k: v for k, v in TRACE.items()}}
    return md, info


def write() -> dict:
    R, N = numbers9()
    md, info = compose(R, N)
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / f"{STEM}.md"
    p.write_text(md, encoding="utf-8")
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "manuscript9_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"md": str(p), "chars": len(md), "refs": info["n_refs"]}


if __name__ == "__main__":
    print(json.dumps(write(), ensure_ascii=False, indent=1))
"""SARA submission manuscript, second revision (Chinese, IMRaD order, GB/T 7714 references, MathType equations).

Numbers come from ``numbers9`` and are formatted by ``fmt9`` (traced).  Citations are ``[@key]`` markers resolved by
``refs10.resolve`` into superscript GB/T markers.  Display equations are ``$$...$$`` blocks labelled ``{#eq:name}``;
references ``式（@eq:name）`` are replaced by the equation numbers (display equations are numbered 1..n in order,
which is also the numbering written by build_docx10).  The Word files are written by build_docx10.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from . import paths8
from .fmt9 import TRACE, ci, cin, i0, num, pc, rng, sci, sgn
from .manuscript9 import CASE_ZH, PUB, _meta, _row, acceptance_checks, table_md
from .manuscript9 import table1 as _t1, table2 as _t2, table3 as _t3
from .numbers9 import ROOT, numbers9
from .publabel9 import relabel

DATE = "20260930"
TITLE = "SARA：知识覆盖不均衡条件下的疾病基因语义检索与可用性感知排序"
STEM = f"SARA_知识覆盖不均衡条件下的疾病基因语义检索与可用性感知排序_投稿版_{DATE}"
EQ_LABEL = re.compile(r"\$\$(.+?)\$\$\s*\{#(eq:[\w-]+)\}", re.S)
# MathType drops \qquad, \quad, \hspace and \; when it converts; two em spaces in text style keep the 2 em gap
QQUAD = "\\text{\u2003\u2003}"


# ============================================================================ front matter
def front(R, N) -> str:
    z = N["z"]
    dep, ch = z["table"]["deployed"], z["table"]["channel_lgb_v6"]
    h = z["h"]
    eu = _row(pd.DataFrame(z["ext"]["contrasts"]), b="channel_lgb_v6", subset="union")
    cold_fold = dep["cold_recall10"] / ch["cold_recall10"]
    abstract = (
        "疾病—基因关联知识高度集中于少数被充分研究的基因，缺少已知关联或功能注释的候选基因因此在检索中被系统性低估。"
        "为此，本研究提出 SARA（Semantic retrieval with Availability-aware RAnking），将多源检索、疾病—功能语义检索与可用性感知"
        "排序整合为两阶段框架：第一阶段以可用性感知的倒数名次融合为每个疾病生成 500 个候选，使缺少某一模态信息的候选不被额外降分；"
        "第二阶段以显式编码缺失值的排序学习模型对候选重排。在覆盖 "
        f"{i0(dep['n_diseases'])} 个测试疾病、{i0(h['h1_vs_channel_lgb_v6']['families'])} 个疾病家族的家族隔离五折交叉验证中，"
        f"SARA 的疾病宏平均前 10 召回率（Recall@10）为 {num(dep['recall10'])}，较固定检索前 500 个候选上的检索通道 LightGBM"
        f"（{num(ch['recall10'])}）提高 {ci(h['h1_vs_channel_lgb_v6'])}，且在全部 15 个外层折 × 随机种子组合中均为正；"
        f"冷启动阳性的前 10 恢复率由 {num(ch['cold_recall10'])} 提高至 {num(dep['cold_recall10'])}，覆盖一半测试阳性所需的候选数"
        f"由约 {i0(N['K50_rB'])} 降至约 {i0(N['K50_r_aware'])}。SARA 与多模态直接融合的差异接近于 0；在 618 个疾病的外部来源标签"
        f"（差值 {cin(eu)}）、760 个疾病的单样本支持任务与 4 个独立转录组队列中，结果方向总体一致。TreeSHAP 解释显示，"
        "语义特征对缺少参考路径的阳性具有较大的正向归因；证据共享审计显示，删减与评价关联共享文献相关的功能注释后，"
        "SARA 的优势方向仍保持一致。")
    kp = [
        f"疾病—基因关联与功能注释呈同向集中：关联数最多的 10% 基因占全部已知关联的 {pc(N['top10'], 0)}，六类信息层齐全的基因仅占 "
        f"{pc(N['all6'] / N['n_genes'])}，信息缺失本身具有结构性。",
        f"SARA 以可用性感知的倒数名次融合生成候选、以显式编码缺失的排序器重排，在家族隔离的五折交叉验证中将 Recall@10 由 "
        f"{num(ch['recall10'])} 提升至 {num(dep['recall10'])}，冷启动阳性的恢复率提高约 {cold_fold:.0f} 倍。",
        "在外部来源标签、单样本支持任务与独立转录组队列中，SARA 的优势方向总体一致；三层证据删除及数量匹配的随机对照显示，"
        "删减共享文献相关注释后，优势方向仍保持一致。",
        "TreeSHAP 解释与分层分析显示，语义证据对缺少参考路径的候选具有较大的正向归因；在本数据中，收益随功能注释覆盖度增加而增大。",
    ]
    return "\n\n".join([
        f"# {TITLE}",
        "**短标题**　SARA：可用性感知的疾病基因检索",
        "**作者**　［作者姓名］¹，［作者姓名］¹，［通讯作者姓名］¹\\*",
        "¹［单位名称，城市，邮编，国家］　\\*通讯作者：［电子邮箱］",
        "## 摘要", abstract,
        "**关键词**　疾病基因优先级排序；知识覆盖不均衡；语义检索；倒数名次融合；可用性感知排序；可解释机器学习",
        "## 要点", "\n".join(f"- {k}" for k in kp),
    ])


# ============================================================================ 1 introduction
def intro(R, N) -> str:
    return "\n\n".join([
        "## 1 引言",
        "识别疾病相关基因是阐明发病机制、解释遗传变异和发现治疗靶点的基础任务。经过二十余年的发展，疾病基因优先级排序已形成若干"
        "互补的技术路线：以蛋白质—蛋白质相互作用（protein–protein interaction，PPI）网络为载体的传播与疾病模块发现方法"
        "[@barabasi2011; @kohler2008; @vanunu2010; @ghiassian2015]，以多源基因组数据融合为代表的综合排序方法[@aerts2006]，"
        "以人类表型本体（Human Phenotype Ontology，HPO）语义相似度驱动的临床诊断工具[@kohler2009; @smedley2015]，以及近年来基于"
        "知识图谱与图神经网络的表示学习方法[@chandak2023; @huang2024txgnn; @wang2024kdgene; @saarinen2024]。这些方法有一个共同前提："
        "与目标疾病相关的证据在候选基因之间具有可比性。",
        "然而，生物医学知识的积累高度不均衡。少数基因吸引了绝大多数研究关注，大量潜在重要的基因长期处于知识空白[@edwards2011; "
        "@stoeger2018]；基因本体（Gene Ontology，GO）等功能注释同样向研究充分的基因集中[@haynes2018; @schnoes2013]，而网络方法的"
        "性能在相当程度上依赖少数多功能的“枢纽”基因[@gillis2012]。在检索任务中，这种不均衡导致两类系统性偏差：其一，缺少参考关联"
        "路径的基因难以被基于已知关联传播的方法检出；其二，缺少某一模态信息（如无 GO 注释或无表型注释）的候选常被当作匹配度低，"
        "而非“信息不可用”。此外，功能注释与疾病关联往往出自同一批文献，若评价未能隔离此类共享证据，所报告的性能增益可能部分源于"
        "信息泄漏[@kaufman2012; @cawley2010]。",
        "生物医学语言模型与蛋白质语言模型的进展为缓解上述不均衡提供了新的途径[@lee2020biobert; @gu2021pubmedbert; @liu2021sapbert; "
        "@lin2023esm2; @xu2023protst]。借助跨模态对齐[@radford2021clip]，疾病的文本与表型描述可直接与基因的功能描述进行比较，从而"
        "降低对已知关联的依赖。信息检索领域的“检索—重排”两阶段范式[@karpukhin2020; @nogueira2019]、倒数名次融合（reciprocal rank "
        "fusion，RRF）[@cormack2009]以及基于梯度提升决策树的排序学习方法 LightGBM 与 LambdaRank[@ke2017; @burges2010]，则为在"
        "有限候选预算下整合异构证据提供了成熟工具。然而，以下三个问题仍缺乏系统回答：如何在固定候选预算下将语义证据与传统检索通道"
        "结合；如何在建模中区分“分数低”与“信息缺失”；以及如何在严格隔离共享证据的条件下评价性能增益。",
        "针对上述问题，本研究提出 SARA（Semantic retrieval with Availability-aware RAnking）框架，主要贡献如下：（1）构建疾病—功能"
        "语义模型，将疾病文本与 HPO 表型映射至与基因 GO 功能相同的语义空间，获得不依赖已知关联的语义证据；（2）提出可用性感知的"
        "倒数名次融合，使语义不可用的候选精确退化为检索顺序而不被额外降分，并在 500 个候选的固定预算内合并检索顺序与语义顺序；"
        "（3）以显式编码缺失值的排序学习模型，基于袋外（out-of-fold，OOF）生成的检索、语义与可用性特征对候选重排，并建立包含家族"
        "隔离交叉验证、三层证据删除、受控注释屏蔽与多种外部检验的系统评价框架。",
        "全文安排如下：第 2 节介绍数据、SARA 框架与评价方法；第 3 节依次报告检索性能比较、候选预算效率、分层与机制分析、证据共享"
        "审计及外部检验结果；第 4 节讨论可用性感知设计的作用、与现有方法的关系及研究局限；第 5 节总结全文。",
    ])


# ============================================================================ 2 materials and methods
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
    eu, ind = M["ext_union"], M["ext_indep"]
    so = fl["shared_original_evidence"]
    dev = z["dev"]
    n_checks, all_ok = acceptance_checks(R)
    assert all_ok
    return "\n\n".join([
        "## 2 材料与方法",
        # ------------------------------------------------------------------ 2.1
        "### 2.1 数据来源与任务定义",
        f"本研究以 HUGO 基因命名委员会（HUGO Gene Nomenclature Committee，HGNC）[@seal2023]可映射的 {i0(M['G'])} 个基因作为候选背景，"
        f"其中蛋白编码基因 {i0(M['protein'])} 个、非编码 RNA（non-coding RNA，ncRNA）{i0(M['ncrna'])} 个、假基因 {i0(M['pseudo'])} 个、"
        f"其他 {i0(M['other'])} 个；缺少序列、GO 注释或表达数据的基因同样保留在背景中。阳性样本取自比较毒理基因组学数据库"
        f"（Comparative Toxicogenomics Database，CTD）[@davis2025]中具有直接证据的 marker/mechanism 类疾病—基因关联，共 "
        f"{i0(M['n_rel'])} 条，覆盖 {i0(M['dis_rel'])} 个疾病；治疗类关联不作为阳性，仅用于分层评价。疾病侧信息包括名称、定义与"
        "同义词文本（以预先固定的 HGNC 基因符号词典掩蔽基因名称）、经严格等价映射的 HPO 表型注释[@gargano2024]（仅保留表型异常"
        "分支）、医学主题词表（Medical Subject Headings，MeSH）树号[@lipscomb2000]、人类疾病本体[@schriml2022]以及在线人类孟德尔"
        "遗传（Online Mendelian Inheritance in Man，OMIM）映射[@amberger2019]；基因侧信息包括 GO 注释[@ashburner2000; @go2023]、"
        "来源排除后的基因—表型注释、STRING v12 数据库中置信度不低于 700 的物理互作[@szklarczyk2023]、代表蛋白序列[@morales2022; "
        "@uniprot2023]以及人类蛋白质图谱（Human Protein Atlas，HPA）的共识组织表达[@uhlen2015]。对疾病—基因关联按疾病、基因、关联类型与来源记录去重；"
        "未标注关联不作为阴性标签。各资源的版本与下载日期固定在运行清单中。数据规模见表 1，各类信息在基因与"
        "疾病间的分布见图 1。",
        "检索任务定义为：给定疾病 $d$ 的可用文本、表型与本体信息，以及当前上下文中可见的参考疾病关联，在全部背景基因中检索 $d$ 的"
        "关联基因。为防止信息泄漏，目标疾病及其严格等价疾病的关联不进入参考关联图、基因—表型来源或任何二级训练特征；基因—表型注释"
        "在按来源疾病剔除当前上下文的全部留出疾病后再聚合至基因。",
        # ------------------------------------------------------------------ 2.2
        "### 2.2 SARA 框架概述",
        "SARA 由候选生成与候选重排两个阶段组成（图 2a）。在候选生成阶段，多源检索通道基于表型、文本、本体与层级等证据在全部背景"
        "基因上生成检索顺序 $r_B$（2.4 节）；疾病—功能语义模型将疾病与基因映射至共同的语义空间，给出语义分数 $s_F$ 与语义顺序 "
        "$r_S$（2.5 节）。两条顺序经可用性感知的倒数名次融合后，每个疾病保留恰好 500 个候选（2.6 节）。在候选重排阶段，SARA 排序器"
        "以 49 个袋外特征对候选重新排序（2.7 节）。",
        f"可用性感知融合的关键性质在于：当语义信息不可用时，融合分数精确退化为检索分数，缺少 GO 注释的候选因而不会因缺少一项证据而"
        f"被额外降分。该退化性质仅针对候选融合阶段成立，最终名次仍受检索信号、基因度先验、排序器可用特征以及并列打破规则影响；若两条顺序均无"
        f"可用信号，则按预先规定的度先验与 GeneID 排序。以式（@eq:aware）中的融合权重 $w=0.5$ 为例，语义不可用的基因只需检索名次不超过 {i0(N['rrf_cut_aware'])} 即可"
        f"入选，而普通 RRF 要求检索名次"
        f"不超过 {i0(N['rrf_cut_plain'])}（图 2c）。为区分不同来源的“冷”状态，本研究分别定义了参考路径冷、受控 GO 缺失与关联监督冷"
        "三种情形（图 2d）：参考路径冷指评价上下文中某基因既无参考关联、也无来源过滤后的基因—表型注释，是数据中自然存在的状态，"
        "相应的阳性简称冷启动阳性；后两种情形是为检验机制而设计的干预（2.10 节）。",
        # ------------------------------------------------------------------ 2.3
        "### 2.3 家族隔离的嵌套交叉验证与袋外特征",
        "每个疾病被预先分配至唯一的疾病家族（优先取 MeSH 树号的前三级分支；无树号时取最近祖先的分支，再无则取严格等价组），"
        f"全部数据划分均在家族层面进行。有阳性的疾病被划分为互不重叠的开发集（{i0(M['dev'])} 个疾病）、外部留出集"
        f"（{i0(M['ext_heldout'])} 个疾病）与五个外层折（合计 {i0(M['test'])} 个测试疾病、{i0(M['test_rel'])} 条关联、"
        f"{i0(M['test_fam'])} 个家族）。第 $j$ 个外层以 $F_j$ 为测试集、$F_{{j+1}}$ 为验证集，其余三折为训练集 $T_j$；$T_j$ 再按"
        "家族分为三个内块 $I_{jb}$，形成每折三个内层上下文与一个外层上下文，共 20 个拟合上下文（图 2b）。",
        "所有需要学习的对象——语义模型、检索通道、候选名单、排序器及其归一化参数——均在各自的上下文中以袋外方式生成：内块 "
        "$I_{jb}$ 的二级训练特征由以 $T_j\\backslash I_{jb}$ 训练的模型产生，外层上下文的模型以 $T_j$ 训练，并对 $F_j$、$F_{j+1}$ "
        "与开发集打分。排除规则依据疾病在各上下文中的角色而非标签生成：外层上下文的留出集合为 $F_j$、$F_{j+1}$、开发集与外部留出集"
        f"的并集，内层上下文再加入 $I_{{jb}}$。移除测试标签后重新生成的检索顺序与候选名单均与原结果完全一致，排序分数的最大差为 "
        f"{sci(acc['1_label_removal']['deployed_score_maxabs'])}。",
        # ------------------------------------------------------------------ 2.4
        "### 2.4 多源检索通道",
        "检索通道在当前上下文的全部背景基因上为每个疾病计算七个来源的分数，参考疾病仅使用其可见关联[@kohler2008; @kohler2009]。"
        "表型来源以信息量（information content，IC）加权的 HPO 相似度度量疾病间的表型接近程度。记 $\\mathbf{q}_d$ 为疾病 $d$ 经"
        "祖先闭包的 HPO 指示向量，术语 $t$ 的信息量与疾病间相似度分别为",
        r"$$\mathrm{IC}(t)=-\ln\frac{n_t+1}{n_{\mathrm{ref}}+1},\qquad \mathrm{sim}(d,d')=\frac{\left(\mathbf{q}_d\odot\mathbf{c}\right)^{\mathrm{T}}\left(\mathbf{q}_{d'}\odot\mathbf{c}\right)}{\left\lVert \mathbf{q}_d\odot\mathbf{c}\right\rVert\,\left\lVert \mathbf{q}_{d'}\odot\mathbf{c}\right\rVert}$$ {#eq:ic}",
        "式中，$n_t$ 为参考疾病中注释了术语 $t$ 的疾病数，$n_{\\mathrm{ref}}$ 为参考疾病数，$\\mathbf{c}$ 为由 $\\mathrm{IC}(t)$ 组成的"
        "权重向量，$\\odot$ 表示逐元素乘积。表型迁移来源选取与 $d$ 相似度最高的 20 个参考疾病 $\\mathcal{N}_{20}(d)$（负相似度截断"
        "为 0），按归一化相似度转移其可见关联：",
        r"$$s_d(g)=\sum\limits_{d'\in\mathcal{N}_{20}(d)}\frac{\mathrm{sim}(d,d')}{\sum\limits_{d''\in\mathcal{N}_{20}(d)}\mathrm{sim}(d,d'')}\times\frac{A_{d'g}}{\deg(d')}$$ {#eq:transfer}",
        "式中，$A_{d'g}$ 为参考关联指示变量，$\\deg(d')$ 为 $d'$ 的可见关联数。文本来源与本体来源分别以基因符号掩蔽后的词频—逆文档"
        "频率（term frequency–inverse document frequency，TF-IDF）余弦相似度和 MeSH 树号最具信息量共同祖先的 Resnik 信息量选取参考"
        "疾病，并按式（@eq:transfer）转移关联；层级"
        "来源取 CTD 疾病层级中的两步邻域；基因—表型来源为疾病表型画像与来源排除后基因表型画像的 IC 加权余弦相似度；兼容来源为疾病"
        "表型画像与经可见参考关联聚合的基因表型画像之间的余弦相似度。表型传播来源以表型迁移分数为种子，在 STRING 物理互作网络上"
        "执行重启随机游走（random walk with restart，RWR）：",
        r"$$\mathbf{p}^{(t+1)}=\alpha\,\mathbf{u}+(1-\alpha)\,\mathbf{P}\,\mathbf{p}^{(t)}$$ {#eq:rwr}",
        "式中，$\\mathbf{u}$ 为归一化的种子向量，$\\mathbf{P}$ 为列归一化的互作转移矩阵，重启概率 $\\alpha=0.15$；当 "
        "$\\left\\lVert \\mathbf{p}^{(t+1)}-\\mathbf{p}^{(t)}\\right\\rVert_1\\le\\alpha\\varepsilon$（$\\varepsilon=10^{-6}$）时停止"
        "迭代。各来源在全背景上的名次以倒数名次融合，并在疾病层面对不可用来源重新分配权重：",
        r"$$F_d(g)=\frac{\sum\limits_{s}w_s\,a_{ds}\,\phi_{ds}(g)}{\sum\limits_{s}w_s\,a_{ds}},\qquad \phi_{ds}(g)=\frac{1}{K_0+r_{ds}(g)}$$ {#eq:fusion}",
        "式中，$a_{ds}\\in\\{0,1\\}$ 表示来源 $s$ 对疾病 $d$ 是否可用，$r_{ds}(g)$ 为基因 $g$ 在来源 $s$ 中的名次（未被该来源打分的"
        "基因取 $\\phi_{ds}(g)=0$），$K_0=60$。融合权重 $w_s$ 以开发集上的宏平均 U@10 为目标选择，U@10 的定义见式（@eq:upper），"
        "选择过程见补充说明 S2。有信号的基因按 $F_d(g)$ 排序，其余基因按度先验排序，并以 GeneID 打破并列，得到全背景检索顺序 $r_B$。",
        # ------------------------------------------------------------------ 2.5
        "### 2.5 疾病—功能语义模型",
        "语义模型以冻结的生物医学实体表示模型 SapBERT[@liu2021sapbert]编码疾病文本、HPO 术语与 GO 术语，以冻结的蛋白质语言模型 "
        "ESM-2（35M 参数版本）[@lin2023esm2]编码代表蛋白序列。疾病文本（名称、定义与至多 8 个同义词，均经基因符号掩蔽）以句界优先"
        f"的方式分段，每段不超过 126 个词片，段向量按词片数加权平均；据此，超过单段长度的 {i0(tx['n_truncated_v71'])} 个疾病"
        f"（{pc(tx['share_truncated_v71'])}）的文本得以完整编码。疾病表型表示为其 HPO 术语向量按注释频率的加权平均，基因功能表示为"
        f"其 GO 注释术语向量的聚合。共享语义投影 $\\mathbf{{P}}_T$（秩为 {cc['rank']}）将疾病文本—表型与基因功能映射至同一空间，"
        "序列表示经另一投影 $\\mathbf{P}_S$ 映射至该空间；功能、序列与表达三个模态的匹配分数经全局门控合并为核心分数 $r(d,g)$。"
        "模型的训练目标为多正例排序损失与功能锚对齐损失之和：",
        r"$$\mathcal{L}_{\mathrm{DG}}=\frac{1}{\left|D_{\mathrm{tr}}\right|}\sum\limits_{d\in D_{\mathrm{tr}}}\frac{1}{\left|P_d\right|}\sum\limits_{g\in P_d}\ln\left(1+\sum\limits_{u\in U_d}\exp\frac{r(d,u)-r(d,g)}{\tau}\right)$$ {#eq:ldg}",
        r"$$\mathcal{L}_{\mathrm{A}}=\frac{1}{\left|G_A\right|}\sum\limits_{g\in G_A}\frac{1}{\left|A_g^{+}\right|}\sum\limits_{a\in A_g^{+}}\ln\left(1+\sum\limits_{b\in U_g}\exp\left(\ell_{gb}-\ell_{ga}\right)\right),\qquad \mathcal{L}=\mathcal{L}_{\mathrm{DG}}+\lambda_A\mathcal{L}_{\mathrm{A}}$$ {#eq:la}",
        f"式中，$P_d$ 为训练疾病 $d$ 在当前上下文中可见的阳性基因，$U_d$ 为未标注对照基因（全背景基因中除去 $d$ 的已知阳性），"
        f"$\\tau={tc['tau']}$ 为温度系数；$\\ell_{{ga}}=\\left(\\mathbf{{z}}_g^{{S}}\\right)^{{\\mathrm{{T}}}}\\mathbf{{z}}_a/\\tau_a$ 为基因 $g$ "
        f"的序列表示与 GO 功能锚 $a$ 的相似度（$\\tau_a={tc['tau_a']}$），$A_g^{{+}}$ 为 $g$ 注释的功能锚（在背景基因中注释 5～500 个"
        f"基因的 GO 术语），$U_g$ 为其余功能锚，$\\lambda_A={tc['lambda_A']}$。未标注基因不等同于阴性，两项损失均按正例—未标注学习"
        f"的设定构造[@elkan2008; @bekker2020]。模型共 {i0(tr['fit']['n_params'])} 个可训练参数，以 AdamW[@loshchilov2019]"
        f"（学习率 {tc['lr']}，权重衰减 {tc['weight_decay']}，每批 {tc['batch']} 个疾病）训练至多 {tc['max_epochs']} 轮，以同一"
        f"上下文 $F_{{j+1}}$ 上的全背景 Recall@10 选择检查点（耐心值 {tc['patience']}）；未标注对照集合（参考关联基因或全背景）在"
        "开发集上选择。SARA 仅使用其中不依赖已知关联的功能通道分数：",
        r"$$\mathbf{z}_d=\mathrm{norm}\left(\mathbf{P}_T\left(\mathbf{x}_d^{\mathrm{text}},\mathbf{x}_d^{\mathrm{HPO}}\right)\right),\qquad \mathbf{z}_g^{F}=\mathrm{norm}\left(\mathbf{P}_T\,\mathbf{x}_g^{\mathrm{GO}}\right),\qquad s_F(d,g)=\mathbf{z}_d^{\mathrm{T}}\,\mathbf{z}_g^{F}$$ {#eq:sf}",
        "式中，$\\mathrm{norm}(\\mathbf{v})=\\mathbf{v}/\\left\\lVert\\mathbf{v}\\right\\rVert_2$ 表示 L2 归一化；无 GO 注释的基因记为"
        "语义不可用。",
        # ------------------------------------------------------------------ 2.6
        "### 2.6 可用性感知的倒数名次融合",
        "令 $r_S(d,g)$ 为语义可用基因中的语义名次，$m_{dg}\\in\\{0,1\\}$ 为语义可用标记。SARA 以下式融合检索顺序与语义顺序"
        "[@cormack2009]：",
        r"$$b_{dg}=\frac{1}{K_0+r_B(d,g)},\qquad e_{dg}=\frac{m_{dg}}{K_0+r_S(d,g)},\qquad R(d,g)=\frac{w\,b_{dg}+(1-w)\,m_{dg}\,e_{dg}}{w+(1-w)\,m_{dg}}$$ {#eq:aware}",
        "当语义不可用（$m_{dg}=0$）时，$R(d,g)=b_{dg}$，候选不因缺少一种模态而被额外降分；若全部基因均无语义信息，候选名单与检索"
        "前 500 完全相同。作为对照，普通 RRF 的分母恒为 1，即 $w\\,b_{dg}+(1-w)\\,m_{dg}\\,e_{dg}$。每个疾病取 $R(d,g)$ 最高的 500 个"
        "基因作为候选（以 GeneID 打破并列），且不补入任何已知阳性。权重 $w\\in\\{0.3,0.5,0.7,0.9\\}$ 在每个外层上下文按开发集阳性"
        f"覆盖率选择（平局取较大者），五个外层依次为 {'、'.join(str(v) for v in dev['w'])}；开发集上的阳性覆盖率由检索前 500 的 "
        f"{pc(dev['v6_cov'])} 提高至 {pc(dev['strict_cov'])}。",
        # ------------------------------------------------------------------ 2.7
        "### 2.7 SARA 排序器",
        "SARA 排序器采用 LightGBM[@ke2017]实现的 LambdaRank 排序学习模型[@burges2010]，在 SARA 候选上以 49 个特征重排：30 个检索"
        "特征（七个来源的分数、对数名次与可用性标记，融合分数、检索名次、补足标记、度先验与五项疾病描述统计）和 19 个语义与可用性"
        "特征（$s_F$ 及其对数名次、融合分数 $R$ 的对数名次、语义可用与来源标记、表达匹配分数及其可用性、参考关联与基因—表型"
        "可见性、参考度、基因类型与各模态可用性、疾病文本/表型/组织信息的可用性）。语义与表达特征缺失时以 NaN 显式输入，由决策树"
        "的缺失分支学习其含义，而非以 0 填充。排序器以三个内块的袋外候选与特征训练，学习率 0.05，至多 1,500 棵树，以外层 $F_{j+1}$ "
        "的 Recall@10 早停，行、列采样率均为 0.8；叶数 $\\{15,31,63\\}$ 与最小叶样本数 $\\{50,200\\}$ 组合的六个配置在开发集上选择"
        "（表 S2）。此外，本研究还考察了两种替代设计——在基础排序上叠加门控残差的冷平衡纠偏器，以及以蛋白质语言模型补全缺失 GO "
        "的序列补位模型；二者均未进入最终模型（补充说明 S4）。",
        # ------------------------------------------------------------------ 2.8
        "### 2.8 对照方法",
        "对照方法包括：（1）检索通道 LightGBM，使用与 SARA 相同的 30 个检索特征，在其自身的检索前 500 上训练；（2）基础排序器，"
        "使用同样的 30 个特征，在 SARA 候选上以袋外方式训练；（3）固定权重融合，即检索来源分数的固定权重线性融合（权重在开发集上"
        "选择）；（4）多模态直接融合，在 30 个检索特征之外加入 14 项多模态语义核心特征（缺失写 0），在同一 SARA 候选上重训；（5）"
        "以全背景语义分数单独排名的语义检索 $s_F$、核心分数 $r$ 与序列分数；（6）未经训练的 SapBERT 疾病—功能相似度；（7）在同一"
        "候选名单上以 $F_{j+1}$ 选择检查点重训的知识图谱补全方法 KDGene[@wang2024kdgene]与 SIGN 式图神经网络[@frasca2020]"
        "（表 S3）。主要终点比较的是相同疾病、相同 500 候选预算下的端到端结果；检索通道 LightGBM 使用自身检索前 500，SARA 使用可用性感知融合后的 500 个候选，"
        "因此 3.3 节另行报告候选生成与候选内排序的分解，避免将候选集差异误作单一排序器效应。支持任务的对照为等权随机游走、语义与随机游走的组合以及连续时间量子游走[@saarinen2024]。",
        # ------------------------------------------------------------------ 2.9
        "### 2.9 功能证据政策与证据共享审计",
        f"GO 注释按 GAF 2.2 格式的全部 17 列解析，去除 NOT 限定与 ND 证据，保留证据代码与引文，生产规则保留 {i0(M['go_rec'])} 条"
        f"记录（有 GO 注释的基因 {i0(M['has_go'])} 个）。证据共享审计包含三层删除：（1）训练可见来源排除，删除引用本上下文训练关联"
        "文献的记录；（2）评价条件化删除，删除引用任何 CTD 关联文献（含评价关联）的记录，仅作敏感性分析；（3）证据块隔离，以文献—"
        f"关联二部图的连通块为单位（巨块不拆分）随机抽取至覆盖 {pc(l3['target_share'], 0)} 的关联，同时隔离块内关联、其文献的 GO "
        f"记录与可追溯的表型注释（{i0(l3['blocks_chosen'])} 个块、{i0(l3['relations_held'])} 条关联、{i0(l3['pmids_held'])} 篇文献）。"
        "每层均配置逐基因数量匹配的随机删除对照；这些分析用于评估结果对证据共享的敏感性，不将删减后的结果解释为新的真实标签；证据代码变体包括仅保留实验类证据、去除作者陈述类证据（TAS/NAS）以及去除突变表型"
        "与遗传相互作用类证据（IMP/IGI/HMP/HGI）。每种条件均从头重训整条流程（种子 42），并比较 SARA 相对检索通道 LightGBM 的优势在"
        "删减前后的变化；去除专家推断类证据（证据代码 IC）与仅保留电子注释（证据代码 IEA）两项为只替换推理输入的重打分敏感性分析。",
        # ------------------------------------------------------------------ 2.10
        "### 2.10 受控的功能注释缺失与关联监督撤除",
        f"在不读取目标疾病标签的条件下，从同时具有 GO 注释与序列的 {i0(mk['eligible_genes'])} 个基因中，按“GO 术语数十分位 × 物理"
        f"互作度十分位”分层，以稳定哈希顺序抽取 10% 作为开发屏蔽集（{i0(mk['G_mask_dev'])} 个）、其后 20% 作为测试屏蔽集"
        f"（{i0(mk['G_mask_test'])} 个），两者互不相交。屏蔽基因的 GO 注释及其派生量在语义模型训练、功能锚构建与候选生成之前删除。"
        "关联监督撤除（简称监督冷）进一步将测试屏蔽基因的全部疾病关联从参考关联图、基因—表型注释与所有训练表中撤出，并设置同时"
        "删除 GO 注释的组合条件。",
        # ------------------------------------------------------------------ 2.11
        "### 2.11 外部标签、时间外推、支持任务与表达队列",
        f"外部来源标签集合为 {i0(fl['diseases'])} 个测试疾病的 GenCC[@distefano2022]（分级为 Definitive 或 Strong）与 ClinVar"
        f"[@landrum2020]（审核星级不低于 2）关联，共 {i0(fl['pairs'])} 条原始记录，按疾病、基因、来源与关联类型合并去重后为 {i0(eu.get('n_pairs', 0))} 对。每个疾病以其"
        "外部来源标签作为评价对象，在对应外层模型中均未作为训练疾病出现；评价时从排名中移除该疾病的 CTD 阳性（明确过滤排名），并对三个种子取"
        f"平均。每条关联均记录其是否与训练关联或该基因的 GO 注释共享原始文献（是 {i0(so['yes'])} 条、否 {i0(so['no'])} 条、无法追溯 "
        f"{i0(so['unknown'])} 条），并单独报告文献独立子集（{i0(ind.get('n_diseases', 0))} 个疾病、{i0(ind.get('n_pairs', 0))} 对）。",
        "时间外推以 2023-12-31 为截止日，使用截止前版本的 CTD 关联与疾病词表、HPO 与基因—表型注释、人类疾病本体、GO/GOA、HGNC "
        f"与 GTEx v8 表达[@gtex2020]重建全部输入并重训全部模型，分别评价截止时已有关联疾病的新关联（新关联任务，"
        f"{i0(hc['x2_deployed_vs_channel_lgb_v6']['n'])} 个疾病）与截止时尚无关联的新疾病（新疾病任务，"
        f"{i0(hc['x1_deployed_vs_channel_lgb_v6']['n'])} 个疾病）；晚于截止日的输入逐项列于表 S9。支持任务在 533 个疾病上显露 "
        "$k\\in\\{0,1,3,5\\}$ 个已知基因，并在 760 个疾病上评价单样本（$k=1$）情形；语义与随机游走的组合权重 $\\omega$ 在开发集上"
        f"选择（$\\omega={z['omega']}$）。",
        "表达验证使用基因表达综合数据库（Gene Expression Omnibus，GEO）[@barrett2013]中银屑病（GSE13355、GSE30999）与溃疡性结肠炎"
        "（GSE87466、GSE75214）的 4 个队列：每个基因取平均表达最高的探针，以方差收缩的调节 $t$ 统计量（参照 limma 的经验贝叶斯"
        "思想[@ritchie2015]）度量差异表达，比较冻结的前 20/50 个基因与在平均表达、物理互作度和已知关联数上匹配的背景基因的统计量"
        "秩；另以在互作度与关联数上逐一匹配的随机基因集作为对照。",
        # ------------------------------------------------------------------ 2.12
        "### 2.12 评价指标与统计推断",
        "主要终点预先定义为 SARA 与检索通道 LightGBM 在相同 500 候选预算下的疾病宏平均 Recall@10；其余截断位置、基线、分层、消融与外部子集比较均作为描述性或机制性分析，不作确认性多重检验推断。主要指标为疾病宏平均的前 $K$ 召回率 Recall@$K$，冷启动恢复以冷启动阳性上的召回率度量。下文将疾病宏平均指标简写为 R@$K$；关系级比例在相应表注中单独标明：",
        r"$$R_d@K=\frac{\left|Q_d\cap\mathrm{Top}_K\left(C_d\right)\right|}{\left|Q_d\right|},\qquad R_d^{c}@K=\frac{\left|Q_d^{c}\cap\mathrm{Top}_K\left(C_d\right)\right|}{\left|Q_d^{c}\right|}$$ {#eq:recall}",
        "式中，$C_d$ 为疾病 $d$ 的候选名单，$Q_d$ 为其测试关联，$Q_d^{c}$ 为其中的冷启动阳性；$R^c$@$K$ 仅在含冷启动阳性的疾病上"
        "定义。为区分候选覆盖与候选内排序的贡献，另报告候选名单允许达到的上界：",
        r"$$U_d@K=\frac{\min\left(K,\,m_d\right)}{n_d}$$ {#eq:upper}",
        "式中，$m_d$ 为进入候选名单的阳性数，$n_d$ 为阳性总数；此外报告 Hit@10 与候选覆盖率（覆盖@500）。三个随机种子（42、52、62）"
        "先在疾病内平均；没有阳性进入候选的疾病保留在分母中。所有置信区间（confidence interval，CI）均为按疾病家族重抽样的 "
        "bootstrap 95% 百分位区间[@efron1993; @field2007]：主要比较重抽样 10,000 次（种子 20260923），描述性区间 1,000～2,000 次；"
        "关系级比例以家族为簇对分子与分母同时重抽样。除主要比较外，其余分析只报告估计值与区间。",
        # ------------------------------------------------------------------ 2.13
        "### 2.13 模型解释、可视化与实现",
        "模型解释采用沙普利加性解释（SHapley Additive exPlanations，SHAP）的树模型精确算法 TreeSHAP[@lundberg2017; @lundberg2020]："
        "对外层 0、种子 42 的 SARA 排序器，由保存的模型与重建的袋外特征矩阵计算全部留出阳性与 4 万个抽样未标注候选的逐特征贡献，"
        "并核对 SHAP 之和与保存的排序分数一致。分布图采用高斯核密度估计（名次在对数尺度上，有界处作反射校正），层次聚类采用欧氏"
        "距离的平均连接，来源可用性以 UpSet 图[@lex2014]展示，疾病表示以 t-SNE[@vandermaaten2008]可视化；分类配色经色觉缺陷模拟"
        "[@machado2009]校验，同一方法在全部图中使用同一颜色。",
        f"全部计算在单机完成：Python {rep['environment']['python'].split()[0]}、PyTorch {rep['environment']['torch'].split('+')[0]}"
        f"（CUDA 12.8）、LightGBM {rep['environment']['lightgbm']} 与 NumPy {rep['environment']['numpy']}，单块 NVIDIA GeForce RTX "
        f"5060 Ti。全部 {i0(N['runs'])} 次拟合与评价运行均以内容哈希记录，有计时的运行合计约 {num(N['timed_h'], 1)} h，单次运行的"
        f"峰值显存为 {num(N['gpu_peak'] / 1024, 2)} GiB。在同一计算环境中，以全新缓存从冻结输入重算第 0 折、种子 42 的完整流程："
        f"候选名单与前 10 名完全一致，排序分数的最大差为 {sci(rep['deployed_scores_maxabs'])}。另有 {n_checks} 项全流程一致性检查"
        "全部通过，包括标签移除不变性、过滤先于聚合、袋外谱系、检查点与训练集不重叠、GO 屏蔽与关联监督撤除的完整性、候选预算的"
        "精确性、语义缺失时的精确回退、批次不变性与种子规则。",
    ])


# ============================================================================ 3 results
def results(R, N) -> str:
    z, x = N["z"], N["x"]
    dep, ch, v71 = z["table"]["deployed"], z["table"]["channel_lgb_v6"], z["table"]["v71_direct_strict"]
    h = z["h"]
    lc, G = N["layer_counts"], N["n_genes"]
    bins = {b["known_relations"]: b for b in N["dis_bins"]}
    b0 = bins.get("0", {})
    jt = "，以及".join(f"“{a}”与“{b}”（{num(j, 2)}）" for a, b, j, s in x["jac_top"][:2])
    rk = N["rk"]
    wa = _row(pd.DataFrame(N["wtl"]), stratum="全部")
    w5 = _row(pd.DataFrame(N["wtl"]), stratum="阳性数 >5")
    cold = {r["method"]: r for r in N["cold"]}
    st = z["strata"]
    mb, uq, sh = x["med_best"], x["uniq"], x["shared"]
    per_seed = "、".join(num(v) for v in z["per_seed"].values())
    parts = ["## 3 结果"]
    # ------------------------------------------------------------------ 3.1
    parts += [
        "### 3.1 疾病—基因知识覆盖的不均衡特征",
        f"疾病—基因知识在基因之间呈现明显的集中趋势（图 1）。在 {i0(G)} 个背景基因中，仅 {i0(N['rel_genes'])} 个"
        f"（{pc(N['rel_genes'] / G)}）至少有一条已知关联；关联数最多的 10% 基因占全部已知关联的 {pc(N['top10'], 0)}，基因侧与疾病侧"
        f"的 Gini 系数分别为 {num(N['gini_g'], 2)} 与 {num(N['gini_d'], 2)}（图 1a–b）。关联知识与注释知识同向集中：基因的 GO "
        f"基因—术语对数与其已知关联数的 Spearman 相关系数为 {num(N['rho'], 2)}；GO 基因—术语对数不少于 100 的基因"
        f"{'全部' if N['go_rich_rel_share'] >= 1 else pc(N['go_rich_rel_share'])}已有已知关联，而仅有 1～4 个 GO 基因—术语对的基因中"
        f"这一比例仅为 {pc(N['go_poor_rel_share'])}（图 1c）。",
        f"六类信息层的可用性同样不均衡（图 1d）：{i0(lc['has_go'])} 个基因有 GO 注释、{i0(lc['has_expr'])} 个有组织表达、"
        f"{i0(lc['has_ppi'])} 个有物理互作，但仅 {i0(lc['has_gene_hpo'])} 个有基因—表型注释；六层信息齐全的基因只有 "
        f"{i0(N['all6'])} 个（{pc(N['all6'] / G)}），而有 GO 注释、序列与表达却没有任何已知关联的基因多达 "
        f"{i0(N['no_rel_but_go_seq_expr'])} 个，这些基因只能依靠语义信息检出。{i0(N['ncrna_n'])} 个 ncRNA 中 "
        f"{pc(N['ncrna_nogo_share'])} 没有 GO 注释。疾病侧存在同样的梯度（图 1e）：没有已知关联的疾病中 {pc(b0.get('name_only'))} 仅有"
        f"名称与同义词，而关联数超过 20 的疾病几乎都有定义文本。在关联基因最多的 12 个 MeSH 大类之间，关联基因高度共享，Jaccard "
        f"系数最高的是{jt}（图 1f）；各大类的知识覆盖水平差异明显，有关联疾病中具有 HPO 描述的比例从“{x['hpo_lo'][0]}”的 "
        f"{pc(x['hpo_lo'][1], 0)} 到“{x['hpo_hi'][0]}”的 {pc(x['hpo_hi'][1], 0)}（图 1g）。",
        "上述结果表明，知识覆盖的不均衡是结构性的：检索方法若将“信息缺失”等同于“匹配度低”，将系统性地压低缺少注释或参考关联的"
        "候选。这正是 SARA 采用可用性感知设计的出发点；以下首先评估 SARA 相对基线方法的整体检索性能。",
    ]
    # ------------------------------------------------------------------ 3.2
    parts += [
        "### 3.2 与基线方法的检索性能比较",
        f"在 {i0(dep['n_diseases'])} 个测试疾病上，SARA 的疾病宏平均 Recall@10 为 {num(dep['recall10'])}（三个种子分别为 {per_seed}），"
        f"较使用相同检索资源、在自身检索前 500 上训练的检索通道 LightGBM（{num(ch['recall10'])}）提高 {ci(h['h1_vs_channel_lgb_v6'])}"
        f"（图 3a，表 2）。该优势在全部 15 个外层 × 种子组合中均为正，其中 {N['fs_ci_pos']} 个组合的 95% CI 不含 0（差值范围 "
        f"{rng(*N['fs_one_range'])}，图 3d），并在各截断位置保持稳定：$K$ = 1、5、20、50、100 时，SARA 与检索通道 LightGBM 的 "
        "Recall@$K$ 分别为 " + "、".join(f"{num(rk['deployed'][k])} 与 {num(rk['channel_lgb_v6'][k])}" for k in (1, 5, 20, 50, 100)) +
        "（图 3c）。",
        f"SARA 同样优于仅使用检索特征的基础排序器（{cin(h['vs_B'])}）、固定权重融合（{cin(h['vs_F7_v6'])}）与单独的语义检索 $s_F$"
        f"（{cin(h['vs_standalone_sF'])}），且 Recall@10 高于在同一候选名单上重训的 KDGene（{cin(z['comp']['kdgene']['deployed_minus'])}；"
        f"种子 42）与 SIGN 式图神经网络（{cin(z['comp']['sign']['deployed_minus'])}；种子 42）。与使用全部模态语义分数（功能、序列与"
        f"表达）的多模态直接融合相比，SARA 的差值为 {ci(h['h2_vs_v71_direct_strict'])}；该区间未显示明确差异，不据此作等效性判断（图 3a、e）。",
        f"逐疾病比较显示，SARA 优于检索通道 LightGBM 的疾病占 {pc(wa['deployed_better'], 0)}，劣于的占 {pc(wa['channel_better'], 0)}，"
        f"其余持平；在阳性数超过 5 的疾病中，改善更为普遍（{pc(w5['deployed_better'], 0)} 与 {pc(w5['channel_better'], 0)}；图 3f）。"
        f"以每个疾病最靠前的阳性名次衡量，没有阳性入选候选的疾病比例由检索前 500 的 {pc(x['unl']['channel_lgb_v6'], 0)} 降至 SARA "
        f"候选的 {pc(x['unl']['deployed'], 0)}；在同一批 500 个候选上，引入语义与可用性特征使最佳阳性的中位名次由基础排序器的 "
        f"{num(mb['B'], 0)} 提前至 {num(mb['deployed'], 0)}（图 3g，种子 42）。SARA 的前 10 命中（"
        f"{i0(x['hits_total'].get('deployed', 0))} 个疾病—基因对）与多模态直接融合共享 {i0(sh.get(('deployed', 'v71_direct_strict'), 0))} "
        f"个，独有 {i0(uq.get('deployed', 0))} 个（图 3h）；SARA 的可达上界 U@10 为 {num(dep['U10'])}，检索通道 LightGBM 为 "
        f"{num(ch['U10'])}（表 2）。",
        f"冷启动阳性的恢复主要来自语义证据（图 3b）：在 {i0(cold['deployed']['n_cold_diseases'])} 个含冷启动阳性的疾病上，检索通道 "
        f"LightGBM 与基础排序器的 $R^c$@10 仅为 {num(cold['channel_lgb_v6']['Rc10'])} 与 {num(cold['B']['Rc10'])}，SARA 达到 "
        f"{num(cold['deployed']['Rc10'])}，提高约 {dep['cold_recall10'] / ch['cold_recall10']:.0f} 倍。SARA 的优势在各类疾病中普遍"
        f"存在：在单关联疾病（{num(st['deployed']['single_relation_recall10'])} 与 {num(st['channel_lgb_v6']['single_relation_recall10'])}）、"
        f"多关联疾病（{num(st['deployed']['multi_relation_recall10'])} 与 {num(st['channel_lgb_v6']['multi_relation_recall10'])}）、"
        f"有 OMIM 映射的疾病（{num(st['deployed']['omim_mapped_recall10'])} 与 {num(st['channel_lgb_v6']['omim_mapped_recall10'])}）"
        f"及阳性数超过 5 的疾病（{num(st['deployed']['burden_gt5'])} 与 {num(st['channel_lgb_v6']['burden_gt5'])}）中均领先；对未参与"
        f"训练的 {i0(st['deployed']['therapeutic_only_pairs'])} 条治疗类关联，SARA 的前 10 命中率同样更高"
        f"（{pc(st['deployed']['therapeutic_only_hit10_micro'])} 与 {pc(st['channel_lgb_v6']['therapeutic_only_hit10_micro'])}）。"
        "为明确上述增益的来源，3.3 节进一步分解候选生成与候选重排的贡献。",
    ]
    # ------------------------------------------------------------------ 3.3
    cv = N["cov500"]
    miss = pd.DataFrame(N["miss"])

    def mm(gr, li):
        return pc(_row(miss, group=gr, list=li).get("pool_share"))
    b = z["budget"]
    q, rs = x["quad"], x["resc"]
    hf = z["flows"]["channel_lgb_v6->deployed"]
    zs = z["zero"]
    br = z["var"]["bridge950"]["deployed"]
    parts += [
        "### 3.3 可用性感知候选生成的预算效率",
        f"候选覆盖随预算 $K$ 变化的曲线揭示了 SARA 优势的第一个来源（图 4a）。$K=500$ 时，检索顺序覆盖 {pc(cv['rB'][0])} 的测试阳性，"
        f"语义顺序覆盖 {pc(cv['rS'][0])}，二者经可用性感知 RRF 融合后达到 {pc(cv['r_aware'][0])}；覆盖一半测试阳性所需的预算由检索顺序"
        f"的约 {i0(N['K50_rB'])} 降至约 {i0(N['K50_r_aware'])}，减少约 {pc(1 - N['K50_r_aware'] / N['K50_rB'], 0)}。普通 RRF 与可用性"
        f"感知 RRF 的总体覆盖几乎重合，但二者对缺少语义的阳性差别明显（图 4c）：ncRNA 阳性进入候选的比例在检索前 500、可用性感知 "
        f"RRF 与普通 RRF 下分别为 {mm('非编码 RNA', 'V6 500（仅通道）')}、{mm('非编码 RNA', '缺失感知严格 500')} 与 "
        f"{mm('非编码 RNA', '普通 RRF 严格 500')}，语义不可用阳性分别为 {mm('语义不可用', 'V6 500（仅通道）')}、"
        f"{mm('语义不可用', '缺失感知严格 500')} 与 {mm('语义不可用', '普通 RRF 严格 500')}。可见，可用性感知融合基本保留了这些候选，"
        "而普通 RRF 使其覆盖减半。",
        f"预算归因将 SARA 的收益分解为两部分（图 4b，表 3）：在相同的检索前 500 上，SARA 排序器已较检索通道 LightGBM 提高 "
        f"{ci(b['v6pool_A_vs_channel'])}；将候选替换为 SARA 候选后再提高 {ci(b['strict_A_vs_v6pool_A'])}。进一步将预算放宽至约 950 个"
        f"候选时，覆盖率升至 {pc(br['coverage500'])}，同预算下 SARA 排序器仍较基础排序器高 {ci(b['bridge_A_vs_B'])}，但相对 500 个候选"
        f"仅再变化 {ci(b['bridge_A_vs_strict_A'])}，提示在本实验设置与候选范围内，继续扩大预算的边际收益有限。",
        f"两条顺序的联合分布说明了语义候选的来源（图 4f，{i0(x['quad_n'])} 个语义可用的测试阳性）：{pc(q['均 ≤500'], 0)} 的阳性在两条"
        f"顺序中均进入前 500，{pc(q['仅语义 ≤500'], 0)} 仅在语义顺序中进入；冷启动阳性集中在检索名次约 $10^4$ 的区域，其中一部分被语义"
        f"顺序提前至前 500。由检索前 500 换为 SARA 候选时，{pc(rs['全部阳性']['语义救回'])} 的阳性被语义救回，仅 "
        f"{pc(rs['全部阳性']['被挤出'])} 被挤出；在冷启动阳性中，救回 {pc(rs['参考路径冷']['语义救回'])}，挤出仅 "
        f"{pc(rs['参考路径冷']['被挤出'])}（图 4g）。逐关系看（图 4d–e），SARA 使 {i0(N['flow']['flow_up'])} 条关联进入更好的名次区间"
        f"（其中 {i0(N['flow']['flow_up_cold'])} 条为冷启动阳性），前 10 命中保留 {i0(hf['kept'])} 个、新增 {i0(hf['gained'])} 个"
        f"（其中 {i0(hf['gained_cold'])} 个为冷启动基因）。另有 {i0(zs['zero_signal_both'])} 条测试关联（{pc(zs['zero_signal_share'])}）"
        "在检索与语义两条顺序中均无任何信号，提示在现有输入下，两类检索通道都受到可观测信号上限的约束。候选生成之外，SARA 的收益在不同知识覆盖层级上"
        "如何分布，是 3.4 节考察的问题。",
    ]
    # ------------------------------------------------------------------ 3.4
    a5, b5, c5, d5 = N["s5a"], N["s5b"], N["s5c"], N["s5d"]

    def hv(df, stratum, m):
        return _row(df, stratum=stratum, method=m).get("hit10")
    hq = c5[(c5["hpo"]) & (c5["token_quartile"] >= 3)]["delta"]
    hq_lo = c5[(c5["hpo"]) & (c5["token_quartile"] >= 3)]["lo"]
    assert hq.min() >= c5["delta"].drop(hq.index).max()
    top = d5.sort_values("delta", ascending=False)
    t3 = "；".join(f"{r['name']}（{sgn(r['delta'])}，95% CI {rng(r['lo'], r['hi'])}）" for _, r in top.head(3).iterrows())
    pos = int((d5["lo"] > 0).sum())
    lf = sorted(x["lift"], key=lambda r: -r["mean_log10_lift"])
    ur = {(r["combo"] if isinstance(r["combo"], str) else f"{int(r['combo']):07d}"): r for r in x["up_rows"]}
    noh = ur.get("0111000", {})
    parts += [
        "### 3.4 不同知识覆盖层级上的检索性能",
        f"按基因在评价上下文中的参考关系数分层（图 5a），SARA 在多数知识覆盖层级上高于检索通道 LightGBM：冷启动阳性的关系级前 10 恢复率由 "
        f"{pc(hv(a5, '冷', 'channel_lgb_v6'), 2)} 提高至 {pc(hv(a5, '冷', 'deployed'))}，参考关系不少于 32 的基因中由 "
        f"{pc(hv(a5, '≥32', 'channel_lgb_v6'))} 提高至 {pc(hv(a5, '≥32', 'deployed'))}。按 GO 注释数分层（图 5b），收益随注释丰富度"
        f"总体呈增加趋势，GO 基因—术语对不少于 100 的基因中 SARA 达到 {pc(hv(b5, '≥100', 'deployed'))}（检索通道 LightGBM 为 "
        f"{pc(hv(b5, '≥100', 'channel_lgb_v6'))}）。疾病侧（图 5c），有 HPO 描述且文本长度位于上半部的疾病获益最大（两个四分位格的 "
        f"Recall@10 差值分别为 {' 与 '.join(sgn(v) for v in sorted(hq.tolist(), reverse=True))}"
        f"{'，95% CI 均不含 0' if bool((hq_lo > 0).all()) else ''}）。",
        f"按检索来源的可用性组合分层（图 2e），{i0(x['up_total'])} 个测试疾病仅出现 {x['up_n_combo']} 种来源组合，七个来源齐全的疾病占 "
        f"{pc(x['up_all7'] / x['up_total'], 0)}；没有 HPO 描述的疾病同时失去四个表型相关来源，但在仅剩层级、本体与文本来源的 "
        f"{i0(noh.get('n_diseases', 0))} 个疾病上，SARA 仍较检索通道 LightGBM 提高 {ci(noh)}；全部 {x['up_n_combo']} 种组合中有 "
        f"{x['up_pos']} 种的 95% CI 下界大于 0。",
        f"按 MeSH 疾病大类分层（图 5d；测试疾病不少于 60 的 {len(d5)} 个大类），SARA 相对检索通道 LightGBM 的增益在 {pos} 个大类中 "
        f"95% CI 下界大于 0，增益最大的三个大类为{t3}。以逐病最佳阳性名次的对数提升衡量（图 5e），全部 {len(lf)} 个大类的平均提升均为"
        f"正，其中{lf[0]['name']}最大（平均提前 {num(lf[0]['mean_log10_lift'], 2)} 个数量级，{pc(lf[0]['better'], 0)} 的疾病最佳名次"
        "提前）。上述分层表明，SARA 的收益并不局限于某一类疾病或基因，而是随可用语义证据的增加而放大；这一模式提示排序器对语义证据"
        "的利用方式，3.5 节通过模型解释对此加以检验。上述分层结果用于描述异质性，不将各层级差异解释为独立确认性检验。",
    ]
    # ------------------------------------------------------------------ 3.5
    fi = N["fi"].sort_values("mean", ascending=False)
    fam = N["fi_fam"]
    stp, sf, sm = x["shap_top"], x["shap_fam"], x["shap_meta"]
    ca, nc = sf["阳性·参考路径冷"], sf["阳性·非冷"]
    labs = [relabel(r["label"]) for r in stp]
    top_txt = "、".join((lb[:-1] + f"；{num(r['mean_abs_shap'])}）") if lb.endswith("）")
                        else f"{lb}（{num(r['mean_abs_shap'])}）" for lb, r in zip(labs, stp))
    eo, er = z["var"]["expr_off"]["deployed"], z["var"]["expr_random"]["deployed"]
    for v in (eo, er):
        assert v["delta_vs_main"]["ci_low"] <= 0 <= v["delta_vs_main"]["ci_high"]
    e_rank = next(i for i, r in enumerate(stp) if r["feature"] == "s_E") + 1
    df_, dg_ = x["dep_f"], x["dep_g"]
    lab0 = relabel(str(fi.iloc[0]["label"]))
    parts += [
        "### 3.5 排序器的特征贡献与可解释性分析",
        f"在 15 个排序模型（5 个外层 × 3 个种子）中，SARA 排序器最重要的单一特征是{lab0}，特征增益占 {pc(fi.iloc[0]['mean'])}；按特征族"
        f"合计，检索通道占 {pc(fam.get('V6 检索通道', 0), 0)}、融合名次占 {pc(fam.get('候选融合', 0), 0)}、语义特征占 "
        f"{pc(fam.get('语义', 0), 0)}（图 6a）。TreeSHAP 给出了逐候选的归因（图 6b–e；外层 0、种子 42，{i0(sm['positives'])} 个留出阳性"
        f"与 {i0(sm['negatives_sampled'])} 个未标注候选；逐候选 SHAP 之和与模型输出分数的最大差为 "
        f"{sci(sm['reconstruction_maxabs_vs_saved_scores'])}）。平均绝对 SHAP 值最大的特征依次为{top_txt}（图 6c）。融合名次的贡献随名次"
        f"近乎单调下降，由名次约 {num(df_['first_rank'], 0)} 时的 {sgn(df_['first'], 2)} 降至约 {num(df_['last_pos_rank'], 0)} 时的 "
        f"{sgn(df_['last_pos'], 2)}，其后转为负值（图 6d）；语义分数 $s_F$ 在约 {num(dg_['zero_hi'], 2)} 以上时贡献迅速增大，$s_F$ 约 "
        f"{num(dg_['top_x'], 2)} 时达到 {sgn(dg_['top'], 2)}（图 6e）。",
        f"按候选类别合计各特征族的带符号贡献（图 6b），非冷阳性主要由检索通道推高（{sgn(nc['V6 检索通道'], 2)}），语义特征族贡献 "
        f"{sgn(nc['语义'], 2)}；冷启动阳性则主要依靠语义特征族（{sgn(ca['语义'], 2)}）与融合名次（{sgn(ca['候选融合'], 2)}）获得提升，"
        f"检索通道对其贡献为 {sgn(ca['V6 检索通道'], 2)}。换言之，SARA 在检索证据充分时沿用检索证据，在检索证据缺失时由语义证据接管，"
        f"这正是可用性感知设计的预期行为。表达分数的平均绝对 SHAP 值位列第 {e_rank}，但关闭全部表达特征或以随机组织表达替换后，"
        f"Recall@10 分别为 {num(eo['recall10'])} 与 {num(er['recall10'])}，与完整模型（{num(eo['delta_vs_main']['b'])}）之差的 95% CI "
        "均包含 0（图 6f，表 3），提示在本数据和该消融条件下，去除表达特征未导致可辨别的性能下降。既然语义证据主要来自 GO 功能注释，3.6 节"
        "通过受控屏蔽实验检验其因果作用。",
    ]
    # ------------------------------------------------------------------ 3.6
    e6 = N["e6"]

    def g(arm, col):
        return float(e6.loc[arm, col])
    parts += [
        "### 3.6 功能注释与关联监督对冷启动基因检索的影响",
        f"为检验语义证据的来源，本研究在不读取目标疾病标签的条件下，按“GO 术语数 × 物理互作度”十分位分层抽取 20% 的基因作为测试"
        f"屏蔽集，屏蔽基因与未屏蔽基因在两项指标上的分布几乎重合（图 7a）。删除这些基因的全部 GO 注释后，其阳性进入 SARA 候选的比例"
        f"由 {pc(g('main', 'mask_pool'))} 降至 {pc(g('e5_none', 'mask_pool'))}，进入前 10 的比例由 {pc(g('main', 'mask_hit10'))} 降至 "
        f"{pc(g('e5_none', 'mask_hit10'))}，而全部疾病的总体 Recall@10 仅变化 {sgn(g('e5_none', 'overall_delta'))}（图 7b）。这表明 "
        f"SARA 的其余通道承担了大部分检索任务，语义通道的贡献集中在有功能注释的基因上。同一批阳性的名次流转显示（图 7d），完整模型"
        f"中位于前 10 的 {i0(x['flow_main_top10'])} 个屏蔽基因阳性在删除 GO 注释后仍有 {i0(x['flow_main_top10_kept'])} 个留在前 10。",
        f"更严格的检验是关联监督撤除：将屏蔽基因的全部已知关联从参考关联图、基因—表型注释与所有训练表中撤出，使其在训练中完全"
        f"不可见。此时，仅凭 GO 功能语义，SARA 仍将 {pc(g('e6_cold', 'mask_pool'))} 的阳性带入候选、{pc(g('e6_cold', 'mask_hit10'))} "
        f"带入前 10；若同时删除 GO 注释，这两个比例分别降至 {pc(g('e6_cold_gomask', 'mask_pool'))} 与 "
        f"{pc(g('e6_cold_gomask', 'mask_hit10'))}（图 7b–d）。该对照表明，对于完全缺少关联监督的基因，功能注释经疾病—功能语义模型"
        "提供了主要的检索信号；在同时删除 GO 注释的对照中，未观察到其他输入足以维持同等水平的候选覆盖。本研究还考察了以蛋白质语言模型表示补全缺失 GO 注释的多种方案"
        "（补充说明 S4，图 S8）；在当前数据下，实验观测的功能注释仍是信息量最高的可用来源之一。由于功能注释与疾病关联可能来自相同文献，"
        "3.7 节进一步检验这一依赖是否构成信息泄漏。",
    ]
    # ------------------------------------------------------------------ 3.7
    au = z["audit"]
    l3 = au["layer3_evidence_blocks"]
    codes = N["codes"].copy()
    tot_no = codes["no_pmid"].sum() / codes["total"].sum()
    iea_iba = codes[codes["evidence_code"].isin(["IEA", "IBA"])]["no_pmid"].sum() / codes["no_pmid"].sum()
    e4, e4s = z["e4"], z["e4s"]
    full = e4s["full_gap"]
    nm = e4["e4_no_mutant"]
    keep = nm["new_minus_old_reduced"]["delta"] / full["delta"]
    parts += [
        "### 3.7 证据共享对性能增益的影响评估",
        f"功能注释与疾病关联可能出自同一批文献，为此本研究对 GO 证据进行了系统审计（图 8）。在生产规则保留的 GO 记录中，"
        f"{pc(tot_no)} 没有 PubMed 文献号（PMID），其中电子注释（IEA）与系统发育推断（IBA）占 {pc(iea_iba, 0)}，这部分注释按未知出处"
        f"处理（图 8a）。文献—关联二部图共有 {i0(l3['n_components'])} 个连通块，中位大小为 "
        f"{num(l3['component_size_quantiles']['0.5'], 0)}（图 8b）。审计设计了三层删除，并为每一层配置了逐基因数量匹配的随机删除对照："
        "（1）删除引用本上下文训练关联文献的 GO 记录；（2）删除引用任何评价关联文献的 GO 记录（仅作敏感性分析，不参与模型选择）；"
        f"（3）按文献—关联连通块整体隔离 {i0(l3['relations_held'])} 条关联及其文献；另设三种证据代码变体。每种条件均从头重训整条"
        "流程（图 8c）。",
        f"在 {e4s['n_retrain']} 种整链重训的删除条件中，{e4s['n_inc0']} 种条件下 SARA 相对检索通道 LightGBM 的优势变化的 95% CI 包含 0，"
        f"且与各自匹配随机删除的变化相近（图 8d）；即便删除引用任何评价关联文献的 GO 记录，优势也仅变化 "
        f"{ci(e4['e4_layer2']['difference'])}。唯一 95% CI 不含 0 的条件是去除突变表型与遗传相互作用类证据（IMP/IGI/HMP/HGI）：优势"
        f"变化 {ci(nm['difference'])}，约为完整优势 {sgn(full['delta'])} 的 {pc(abs(nm['difference']['delta']) / full['delta'], 0)}，"
        f"SARA 仍保留 {pc(keep, 0)} 的优势，这与此类实验证据直接刻画基因功能的性质相符。按疾病大类展开后（图 8e），整链重训条件中 "
        f"{pc(x['dod_ret_small'], 0)} 的“大类 × 条件”格子变化不超过 0.02；定向删除与其匹配随机对照在同一大类上同号的比例为 "
        f"{pc(min(x['dod_sign_agree']), 0)}～{pc(max(x['dod_sign_agree']), 0)}，接近随机水平，表明大类间的差异主要来自抽样波动。仅在"
        f"推理时将注释替换为电子注释的重打分使优势变化 {ci(e4['e4_rescore_iea_only']['difference'])}，反映以完整注释训练的模型对输入"
        "分布突变的敏感性。综合这些删除与匹配对照，在本研究检验的方案下未见优势因共享文献删减而系统性衰减；这支持功能注释承载的生物学信息是重要贡献来源，"
        "但不能据此证明其完全不依赖共享文献。在排除信息泄漏之后，"
        "3.8 节考察 SARA 在训练分布之外的表现。",
    ]
    # ------------------------------------------------------------------ 3.8
    ext = pd.DataFrame(z["ext"]["contrasts"])

    def ec(bb, sub):
        return ci(_row(ext, b=bb, subset=sub))
    fl = z["ext"]["fields"]
    et = pd.DataFrame(z["ext"]["table"])
    eu = _row(et, method="deployed", subset="union")
    ei = _row(et, method="deployed", subset="independent_evidence")
    hc = z["hist"]
    x2 = x["hp"]["x2"]
    sup = {(r["method"], r["k"]): r for r in z["support533"]}
    s1 = {r["method"]: r for r in z["support1"]}
    sem = N["sup"]
    k0v8 = _row(sem, method="sem_v8", k=0).get("recall10")
    k0v6 = _row(sem, method="sem_v6", k=0).get("recall10")
    sp1 = x["sup_pairs"].get(1, {})
    ex = z["expr"]
    parts += [
        "### 3.8 外部标签、时间外推与应用场景评估",
        f"首先以独立于 CTD 的外部来源标签检验 SARA：{i0(fl['diseases'])} 个测试疾病的 {i0(eu.get('n_pairs', 0))} 对 GenCC 与 ClinVar 关联，"
        f"这些疾病在评价其的模型中均未作为训练疾病出现。SARA 相对检索通道 LightGBM 提高 {ec('channel_lgb_v6', 'union')}，其中 GenCC "
        f"子集为 {ec('channel_lgb_v6', 'GenCC')}，ClinVar 子集为 {ec('channel_lgb_v6', 'ClinVar')}；相对基础排序器提高 "
        f"{ec('B', 'union')}（图 9a）。在与训练关联不共享原始文献的独立子集（{i0(ei.get('n_diseases', 0))} 个疾病、"
        f"{i0(ei.get('n_pairs', 0))} 对）上，SARA 相对基础排序器提高 {ec('B', 'independent_evidence')}，相对检索通道 LightGBM 的差值为 "
        f"{ec('channel_lgb_v6', 'independent_evidence')}，方向一致；这些结果属于外部来源标签上的回顾性内部验证，用于检验可迁移性，不替代前瞻性实验验证。SARA 与多模态直接融合在各子集中未观察到明确差异（图 9a–b）。",
        f"在以 2023-12-31 为截止日、全部模型以截止前资源重训的时间外推中，SARA 与检索通道 LightGBM 在新关联任务上未观察到明确差异"
        f"（{cin(hc['x2_deployed_vs_channel_lgb_v6'])}），在新疾病任务上同样未观察到明确差异（{cin(hc['x1_deployed_vs_channel_lgb_v6'])}），"
        f"相对单独的语义检索差异为 {cin(hc['x1_deployed_vs_standalone_sF'])}；逐对比较显示，SARA 带入候选的新关联"
        f"略多（新关联任务中为 {i0(x2['listed_deployed'])} 对与 {i0(x2['listed_channel'])} 对；图 9c、f；补充说明 S5）。",
        f"在给定少量已知支持基因的应用场景中，SARA 的语义表示同样有效（图 9d、g）。在 533 个疾病的支持任务中，零支持时 SARA 语义的 "
        f"Recall@10 为 {num(k0v8)}，高于基线语义的 {num(k0v6)}；显露 5 个支持基因后，SARA 语义与 RWR 的组合达到 "
        f"{num(sup[('joint_v8_w0.25', 5)]['recall10'])}，高于基线语义组合（{num(sup[('joint_v6', 5)]['recall10'])}）、单独 RWR"
        f"（{num(sup[('rwr_v8list', 5)]['recall10'])}）与量子游走（{num(sup[('qwalk_v8list', 5)]['recall10'])}）[@saarinen2024]；"
        f"逐病比较中，显露 1 个支持基因时 SARA 在 {i0(sp1.get('up', 0))} 个疾病上更好、{i0(sp1.get('down', 0))} 个疾病上更差。在规模"
        f"更大的 760 个疾病的单样本任务中，SARA 语义与 RWR 组合的 Recall@10 为 {num(s1['joint_v8_w0.25']['recall10'])}，高于基线语义"
        f"组合的 {num(s1['joint_v6']['recall10'])} 与单独 RWR 的 {num(s1['rwr']['recall10'])}（Recall@50 分别为 "
        f"{num(s1['joint_v8_w0.25']['recall50'])}、{num(s1['joint_v6']['recall50'])} 与 {num(s1['rwr']['recall50'])}）。",
        f"最后，在 4 个独立转录组队列（银屑病 2 个[@gudjonsson2010; @suarezfarinas2012]、溃疡性结肠炎 2 个[@li2018uc; "
        f"@vancamelbeke2017]）中检验冻结的前列基因集。在 10 项“队列 × 对比 × 基因集大小”的秩检验中，SARA 三个种子的前列基因分别有 "
        f"{ex['v8_deployed_s42']['p05']}、{ex['v8_deployed_s52']['p05']} 与 {ex['v8_deployed_s62']['p05']} 项显著富集（$P<0.05$），"
        f"多模态直接融合为 {ex['v71_direct_strict_s42']['p05']} 项，而检索通道 LightGBM 与匹配随机基因集均为 "
        f"{ex['channel_lgb_v6_s42']['p05']} 项（图 9e，表 S10）。这些结果为 SARA 前列基因与疾病相关的解释提供了与转录组差异表达一致的关联性支持；"
        "3.9 节以具体疾病展示这些前列基因的生物学含义。",
    ]
    # ------------------------------------------------------------------ 3.9
    rows = N["case_rows"]
    dis = pd.read_parquet(PUB / "diseases.parquet").set_index("disease_id")
    cs = []
    for c in N["cases"][:4]:
        d = c["disease_id"]
        rr = rows[rows["disease_id"] == d]
        cs.append(f"{CASE_ZH.get(d, dis.loc[d, 'disease_name'])} {int(rr[(rr['method'] == 'deployed')]['held_out_positive'].sum())} 个与 "
                  f"{int(rr[(rr['method'] == 'channel_lgb_v6')]['held_out_positive'].sum())} 个")
    nets = {n_["disease_id"]: n_ for n_ in x["nets"]}
    hcm, ad = nets.get("MESH:D024741", {}), nets.get("MESH:D000544", {})
    rp = rows[rows["disease_id"] == "MESH:D012174"]
    rp_d = int(rp[(rp["method"] == "deployed")]["held_out_positive"].sum())
    rp_c = int(rp[(rp["method"] == "channel_lgb_v6")]["held_out_positive"].sum())
    parts += [
        "### 3.9 代表性疾病的案例分析",
        "按作图前确定的规则选取了 6 个测试疾病（图 10a）：在至少有 3 个留出阳性的疾病中，4 个大类各取 SARA 前 10 命中增益最大者，"
        "另取全体中增益最小的 2 个；六个疾病分别为家族性肥厚型心肌病、阿尔茨海默病、肥胖、肿瘤性细胞转化、视网膜色素变性和肝肿瘤。六个疾病均来自外层测试集，且未参与训练、模型选择或案例规则迭代。在 4 个增益案例中，SARA 与检索通道 LightGBM 前 10 名中的留出阳性数分别为" + "；".join(cs) + "；其余两个低增益案例的完整前 10 排名见图 10a。",
        f"SARA 找回的阳性构成功能上高度一致的模块。在家族性肥厚型心肌病中，SARA 的前 10 名集中于肌小节蛋白（MYH6、MYH7、TNNT2、"
        f"TNNI3、MYL2 等），其间有 {i0(hcm.get('phys_dep', 0))} 条 STRING 物理互作[@szklarczyk2023]，其中 TNNI3 与 MYL2 为冷启动阳性；"
        f"在阿尔茨海默病中，SARA 在两种方法共享的 PRNP、MAPT、GSK3B 与 PSEN1 之外补充了 APOE、APP 与 CLU，前 10 名内部的物理互作由 "
        f"{i0(ad.get('phys_ch', 0))} 条增至 {i0(ad.get('phys_dep', 0))} 条（图 10b）。在检索通道本身已较强的疾病中（如视网膜色素变性，"
        f"前 10 名中的留出阳性为 {rp_d} 个与 {rp_c} 个），SARA 的前 10 名仍集中于光感受器与视网膜功能基因（PROM1、RP1、REEP6、GUCA1A "
        "及视蛋白基因等），其中多个尚未被标注为该疾病的关联，提示语义证据与检索证据具有互补性。",
    ]
    return "\n\n".join(parts)


# ============================================================================ 4 discussion, 5 conclusion
def discussion(R, N) -> str:
    z = N["z"]
    dep, ch = z["table"]["deployed"], z["table"]["channel_lgb_v6"]
    fam = N["fi_fam"]
    ca = N["x"]["shap_fam"]["阳性·参考路径冷"]
    comp = z["comp"]
    return "\n\n".join([
        "## 4 讨论",
        "### 4.1 可用性感知设计对冷启动检索的作用",
        "疾病基因检索的难点不仅在于模型能力，更在于知识本身的不均衡：已知关联、功能注释与表型描述同向集中，使缺少这些信息的基因在"
        "检索中天然处于劣势。SARA 从三个层面应对这一问题：以疾病—功能语义模型提供不依赖已知关联的证据；以可用性感知的倒数名次融合"
        "在固定预算内合并检索顺序与语义顺序，同时保证缺少某一模态的候选不被额外降分；以显式编码缺失值的排序学习模型在袋外特征上"
        f"重排候选。在严格的家族隔离评估中，SARA 将 Recall@10 由 {num(ch['recall10'])} 提高至 {num(dep['recall10'])}，将冷启动阳性"
        "的前 10 恢复率提高约 25 倍。",
        "候选生成阶段的可用性感知是 SARA 效率的关键。普通倒数名次融合隐含地将缺失来源视为最差名次，从而系统性地挤出缺少语义信息的"
        "候选；SARA 的融合在语义不可用时精确退化为检索顺序，使 ncRNA 等缺少注释的阳性基本保留在候选中。与此同时，语义顺序将检索"
        f"名次很靠后的冷启动阳性提前至候选范围之内，使覆盖一半测试阳性所需的候选预算减少约 {pc(1 - N['K50_r_aware'] / N['K50_rB'], 0)}。"
        "模型解释进一步描述了排序阶段的特征归因模式：排序器的特征增益主要来自检索通道"
        f"（{pc(fam['V6 检索通道'], 0)}）与融合名次（{pc(fam['候选融合'], 0)}），语义特征约占 {pc(fam['语义'], 0)}；但对冷启动阳性，"
        f"语义特征族（{sgn(ca['语义'], 2)}）与融合名次（{sgn(ca['候选融合'], 2)}）提供了主要的正向贡献，检索通道的贡献则为负"
        f"（{sgn(ca['V6 检索通道'], 2)}）。也就是说，SARA 并未以语义证据替代检索证据，而是在检索证据缺失处由语义证据接管，这正是"
        "可用性感知设计的目标。",
        "### 4.2 与现有疾病基因优先级排序方法的比较",
        "网络传播方法[@kohler2008; @vanunu2010]与表型相似度方法[@kohler2009; @smedley2015]以已知关联或表型注释为出发点，在信息完备的"
        "疾病与基因上表现良好，但其性能在相当程度上取决于少数高连接度的“枢纽”基因[@gillis2012]。本研究的分层结果与此一致：检索"
        "通道在参考关联丰富的基因上表现稳定，但对冷启动阳性几乎无能为力。知识图谱补全与图神经网络方法[@wang2024kdgene; "
        "@frasca2020]同样依赖训练中可见的图结构；在测试疾病不参与训练的家族隔离设置下，在同一候选名单上重训的 KDGene 与 SIGN 式"
        f"模型的 Recall@10 分别为 {num(comp['kdgene']['recall10'])} 与 {num(comp['sign']['recall10'])}，在本实验设置下低于 SARA，提示此类方法"
        "的优势可能部分依赖于训练与测试疾病之间的结构重叠。与上述方法不同，SARA 将语言模型与蛋白质语言模型的语义表示[@liu2021sapbert; "
        "@lin2023esm2]作为与已知关联正交的证据来源，并以可用性感知的方式与检索通道结合，从而在不牺牲信息完备候选表现的前提下扩展了"
        "可检索的基因范围。",
        "SARA 的评价框架本身也可为疾病基因检索研究提供参考。本研究在家族层面隔离训练、验证与测试疾病，所有二级特征均在内层上下文中"
        "以袋外方式生成，所有配置均在独立的开发家族上选择；三层证据删除与数量匹配的随机对照直接检验了共享文献的影响；受控的功能注释"
        "屏蔽与关联监督撤除则将“语义证据来自何处”转化为可检验的问题。上述设计有助于定位 SARA 增益对应的信息来源，但不能完全排除未观测混杂或残余共享证据。分层分析同时显示，SARA 的收益随功能注释的丰富度增大，这提示提高注释稀少基因的功能预测质量（例如"
        "借助蛋白质语言模型[@lin2023esm2; @xu2023protst]）是进一步扩展 SARA 覆盖范围最直接的途径。",
        "### 4.3 局限性与展望",
        "本研究仍存在若干局限。第一，阳性关联来自人工整理的 CTD 数据库，未被标注的候选并不等同于阴性[@bekker2020]，排序靠前的新"
        "候选需要实验或文献验证。第二，交叉验证折与外部标签集合在方法的迭代开发中被使用过，本文结果属于回顾性的内部验证证据，仍需"
        "前瞻性数据或完全独立来源的外部检验；在以截止日前资源重建的时间外推中，SARA 与检索通道 LightGBM 的表现相当。第三，SARA 的语义"
        "收益依赖功能注释的覆盖，对缺少注释的基因帮助有限。第四，表达队列仅提供相关性证据。后续工作将在前瞻性数据与 DisGeNET"
        "[@pinero2020]等更多外部资源上验证 SARA，并探索高质量的功能预测方法，以扩展其对注释稀少基因的覆盖。",
        "## 5 结论",
        "本研究提出了面向知识覆盖不均衡的疾病基因检索框架 SARA。该框架通过疾病—功能语义检索、可用性感知的候选融合与显式编码缺失"
        "的排序学习，在严格的家族隔离评估中提高了疾病基因检索性能，并扩展了冷启动基因的可检索性；在外部来源标签、支持任务与独立转录组队列"
        "中观察到方向一致的关联性证据。其可解释、可审计的设计为利用异构且不完整的生物医学知识提供了一个可复用框架。",
    ])


def endmatter() -> str:
    return "\n\n".join([
        "## 数据与代码可用性",
        "本研究使用的数据均来自公开资源：CTD、HPO、Gene Ontology 与 GOA 人类注释、MeSH、人类疾病本体、OMIM、HGNC、STRING v12、"
        "Human Protein Atlas、MANE 与 UniProt、GTEx v8、GenCC、ClinVar 以及 GEO（GSE13355、GSE30999、GSE87466、GSE75214）。SARA 的"
        "代码、冻结的模型配置、各图面板的作图数据（CSV）与全部评价结果将在论文接收后公开于［代码仓库链接］与［数据存档 DOI］；所用"
        "资源的版本与下载日期随代码一并提供。",
        "## 资助", "［资助机构与项目编号］",
        "## 利益冲突", "［作者声明无利益冲突，或列出利益冲突］",
        "## 作者贡献", "［作者 A：构思、方法、软件、分析、写作——初稿；作者 B：……；通讯作者：指导、写作——审阅与修改］",
        "## 致谢", "［致谢内容］［如在研究或写作中使用了人工智能工具，请按期刊要求在此说明其用途与范围］",
        "## 作者简介", "［作者姓名］，［职称/学位］，［单位］，研究方向为［……］。",
    ])


# ============================================================================ tables and legends
TABLE_TITLES = {"表 1　数据与评估集合": "表 1　数据来源与评估集合的规模",
                "表 2　主要比较": "表 2　各方法在测试集上的检索性能",
                "表 3　候选预算、融合方式与表达特征": "表 3　候选预算、融合方式与表达特征的消融分析"}
FIGS10 = [("1", "Fig1_knowledge_landscape", "疾病—基因知识与信息层可用性的分布特征"),
          ("2", "Fig2_method_overview", "SARA 框架与嵌套交叉验证设计"),
          ("3", "Fig3_main_results", "SARA 与基线方法的检索性能比较"),
          ("4", "Fig4_budget_fusion_flow", "候选预算、融合方式与名次流转分析"),
          ("5", "Fig5_knowledge_strata", "不同知识覆盖层级上的检索性能"),
          ("6", "Fig6_ranker_explanation", "SARA 排序器的特征贡献与 TreeSHAP 解释"),
          ("7", "Fig7_annotation_supervision", "功能注释缺失与关联监督撤除对屏蔽基因检索的影响"),
          ("8", "Fig8_evidence_sources", "GO 证据来源与证据删除敏感性分析"),
          ("9", "Fig9_external_temporal_applications", "外部标签、时间外推与应用任务评估"),
          ("10", "Fig10_case_studies", "代表性疾病的前列基因与基因互作网络")]
LEGEND_FIX = [("参考路径冷的阳性", "冷启动阳性"), ("参考路径冷阳性", "冷启动阳性"), ("前十", "前 10"), ("95% 区间", "95% CI")]
# per-figure edits (symbols as MathType-typeset math, proportionality and inequalities in words); each must apply
LEGEND_EDITS = {
    "Fig1_knowledge_landscape": [
        ("（双对数轴，仅计 ≥1 的对象）", "（双对数轴，仅计计数不小于 1 的对象）"),
        ("（列内比例，≥10% 的格子标注数值；", "（列内比例，比例不低于 10% 的格子标注数值；"),
        ("ρ 为全体基因的 Spearman 相关。", "$\\rho$ 为全体基因的 Spearman 相关系数。"),
        ("弧长 ∝ 大类内关联基因数，带宽 ∝ 两类共享的基因数", "弧长与大类内关联基因数成正比，带宽与两类共享的基因数成正比"),
        ("（有已知关联的疾病 ≥40 个）", "（有已知关联的疾病不少于 40 个）"),
        ("颜色为列内 z 分数", "颜色为列内 $z$ 分数"),
        ("n = 19,861 个基因、13,321 个疾病。", "共 19,861 个基因、13,321 个疾病。")],
    "Fig2_method_overview": [
        ("蓝：区间 > 0，灰：区间含 0", "蓝：区间下界大于 0，灰：区间含 0")],
    "Fig3_main_results": [
        ("**b** 冷恢复：实心为 $R^c$@10（只在含冷查询的疾病上定义），空心为 $R^{c\\to all}$@10（冷命中占全部查询的比例）。",
         "**b** 冷启动恢复：实心为 $R^c$@10（仅在含冷启动阳性的疾病上定义），空心为 $R^{c\\to all}$@10（进入前 10 的冷启动阳性"
         "占全部测试阳性的比例）。"),
        ("**c** Recall@K 曲线（K = 1–100，对数轴）", "**c** Recall@$K$ 曲线（$K$ 取 1～100，对数轴）"),
        ("弧长 ∝ 该方法的前十命中数，带宽 ∝ 两种方法共享的命中数", "弧长与该方法的前 10 命中数成正比，带宽与两种方法共享的命中数成正比"),
        ("n = 4,238 个测试疾病、1,260 个家族；", "共 4,238 个测试疾病、1,260 个家族；")],
    "Fig4_budget_fusion_flow": [
        ("**a** 测试阳性进入前 K 的比例（疾病宏平均；K 为对数轴）", "**a** 测试阳性进入前 $K$ 的比例（疾病宏平均；横轴为对数刻度）"),
        ("竖线为 K = 500 与约 950", "竖线为 $K=500$ 与 $K\\approx 950$"),
        ("整数名次在 [r, r+1) 内均匀抖动后取对数", "整数名次 $r$ 在 $\\left[r,r+1\\right)$ 内均匀抖动后取对数"),
        ("（深色为冷阳性）", "（深色为冷启动阳性）")],
    "Fig5_knowledge_strata": [
        ("顶部 n 为每个种子的关系数", "顶部数字为每个种子的关系数"),
        ("n < 20 的格子不着色", "疾病数少于 20 的格子不着色"),
        ("逐病最佳阳性名次的对数提升 $\\log_{10}$(检索通道 LightGBM 最佳名次) − $\\log_{10}$(SARA 最佳名次) 的山脊图（种子 42；",
         "逐病最佳阳性名次的对数提升 $\\log_{10}r_{\\mathrm{C}}^{*}-\\log_{10}r_{\\mathrm{S}}^{*}$ 的山脊图（$r_{\\mathrm{C}}^{*}$ 与 "
         "$r_{\\mathrm{S}}^{*}$ 分别为检索通道 LightGBM 与 SARA 的最佳阳性名次；种子 42；")],
    "Fig6_ranker_explanation": [
        ("（均值 ± 标准差，n = 15 个模型）", "（均值 ± 标准差，15 个模型）"),
        ("点色为特征值在 2–98 百分位内的相对位置", "点色为特征值在第 2～98 百分位之间的相对位置"),
        ("黑线与灰带为分位分箱的均值与 P10–P90", "黑线与灰带为分位分箱的均值与第 10～90 百分位区间")],
    "Fig7_annotation_supervision": [
        ("（前 10 / 11–50 / 51–500 / 未入选；带宽 ∝ 阳性数）", "（前 10、11～50、51～500 与未入选；带宽与阳性数成正比）")],
    "Fig8_evidence_sources": [
        ("（×100；测试疾病 ≥60 的大类）", "（数值 ×100；测试疾病不少于 60 的大类）")],
    "Fig9_external_temporal_applications": [
        ("显露 k 个支持基因", "显露 $k$ 个支持基因"),
        ("（颜色：选中基因 |t| 高于匹配背景的概率，0.5 为无富集；圆面积 ∝ −log10 P；黑圈为 P < 0.05）",
         "（颜色：选中基因的 $\\left|t\\right|$ 高于匹配背景基因的概率，0.5 表示无富集；圆面积与 $-\\log_{10}P$ 成正比；黑圈表示 "
         "$P<0.05$）"),
        ("（显露 k = 1 与 5 个支持基因；k = 1 为 10 次抽样的平均，k = 5 为 1 次）",
         "（显露 $k=1$ 与 $k=5$ 个支持基因；$k=1$ 为 10 次抽样的平均，$k=5$ 为单次抽样）")],
    "Fig10_case_studies": [
        ("黑框与 ° 表示该阳性在评价上下文中为参考路径冷", "黑框与 ° 表示冷启动阳性（在评价上下文中处于参考路径冷状态）"),
        ("节点面积 ∝ log(1 + GO 基因—术语对数)", "节点面积与 $\\log\\left(1+n_{\\mathrm{GO}}\\right)$ 成正比（$n_{\\mathrm{GO}}$ 为 GO 基因—"
         "术语对数）"),
        ("（° 为参考路径冷）", "（° 为冷启动阳性）"),
        ("以及 GO 术语集 Jaccard ≥ 0.2（紫色虚线）", "以及 GO 术语集 Jaccard 系数不低于 0.2 的基因对（紫色虚线）")],
}
TABLE_EDITS = [("533 病（k = 0–5）/ 760 病（k = 1）", "533 病（$k\\in\\{0,1,3,5\\}$）/ 760 病（$k=1$）")]


def apply_edits(text: str, edits) -> str:
    for a, b in edits:
        assert a in text, f"edit target not found: {a[:60]}"
        text = text.replace(a, b)
    return text


def polish(text: str) -> str:
    """spacing after a numeral before a CJK character and '～' for numeric ranges (label ranges such as S1–S6 stay)."""
    text = re.sub(r"前 10(?=[一-鿿])", "前 10 ", text)
    return re.sub(r"(?<=[\d%])–(?=\d)", "～", text)


def tables(R, N) -> list[str]:
    out = []
    for t in (_t1(R, N), _t2(R, N), _t3(R, N)):
        for a, b in TABLE_TITLES.items():
            if t.startswith(a):
                t = b + t[len(a):]
        if t.startswith("表 1"):
            t = apply_edits(t, TABLE_EDITS)
        out.append(polish(t.replace("参考路径冷查询", "冷启动阳性")))
    return out


def legends(R, N) -> list[tuple[str, str, str]]:
    from .manuscript9 import legends as leg9
    old = {stem: text for _, stem, text in leg9(R, N)}
    out = []
    for n, stem, title in FIGS10:
        body = apply_edits(old[stem].split("** ", 1)[1], LEGEND_EDITS.get(stem, []))
        for a, b in LEGEND_FIX:
            body = body.replace(a, b)
        out.append((n, stem, f"**图 {n}　{title}。** {polish(body)}"))
    return out


# ============================================================================ assembly
def number_equations(md: str) -> tuple[str, dict]:
    labels = {}
    for i, m in enumerate(EQ_LABEL.finditer(md), 1):
        labels[m.group(2)] = i
    md = EQ_LABEL.sub(lambda m: f"$${m.group(1)}$$", md)
    for lab in sorted(labels, key=len, reverse=True):
        md = re.sub(rf"@{re.escape(lab)}(?![\w-])", str(labels[lab]), md)
    left = re.findall(r"@eq:[\w-]+", md)
    assert not left, f"unresolved equation references: {left}"
    return md, labels


def compose(R=None, N=None) -> tuple[str, dict]:
    from .refs10 import resolve
    if R is None:
        R, N = numbers9()
    TRACE.clear()
    body = [front(R, N), intro(R, N), methods(R, N), results(R, N), discussion(R, N), endmatter(),
            "## 参考文献", "<<REFERENCES>>", "## 表", *tables(R, N), "## 图注"]
    for n, stem, leg in legends(R, N):
        body += [f"<!-- figure {stem} -->", leg]
    md = "\n\n".join(p.strip("\n") for p in body) + "\n"
    md = md.replace("\\qquad ", QQUAD)
    md, eqs = number_equations(md)
    md, refs, order, recs = resolve(md)
    md = md.replace("<<REFERENCES>>", "\n".join(refs))
    info = {"stem": STEM, "title": TITLE, "ref_order": order, "n_refs": len(refs), "equations": eqs,
            "figures": [(n, stem) for n, stem, _ in FIGS10], "trace": {k: v for k, v in TRACE.items()}}
    return md, info


def write() -> dict:
    from .refs10 import ris
    R, N = numbers9()
    md, info = compose(R, N)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / f"{STEM}.md").write_text(md, encoding="utf-8")
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "manuscript10_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    from .refs10 import CACHE, record
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    recs = [record(k, cache) for k, _ in sorted(info["ref_order"].items(), key=lambda t: t[1])]
    (ROOT / f"SARA_参考文献_{DATE}.ris").write_text(ris(recs), encoding="utf-8")
    return {"md": str(ROOT / f"{STEM}.md"), "chars": len(md), "refs": info["n_refs"], "equations": len(info["equations"])}


if __name__ == "__main__":
    print(json.dumps(write(), ensure_ascii=False, indent=1))

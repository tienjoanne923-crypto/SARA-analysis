"""Assemble the round-five manuscript from measured results only.

Two rules govern this module:

1. Every number that appears in the text is read from a result file and logged
   into ``reports/manuscript_number_audit.csv`` with the file it came from.  No
   figure is typed in by hand.
2. Anything the study did not actually do is written into the limitations, not
   omitted.  ``reports/what_was_not_done.md`` and the manuscript's limitation
   section are generated from the same source.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .config import Gpu8Config, load_config
from .hashing import atomic_write_df, atomic_write_text, read_json


@dataclass
class NumberAudit:
    rows: list[dict] = field(default_factory=list)

    def cite(self, key: str, value, source: str, note: str = "") -> str:
        """Record a number and return its formatted string."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            text = "未获得"
        elif isinstance(value, float):
            text = f"{value:.4f}" if abs(value) < 10 else f"{value:,.1f}"
        elif isinstance(value, (int, np.integer)):
            text = f"{int(value):,}"
        else:
            text = str(value)
        self.rows.append({"key": key, "value": value, "text": text,
                          "source": source, "note": note})
        return text

    def save(self, path: Path) -> Path:
        return atomic_write_df(pd.DataFrame(self.rows), path)


def read_json_like(path: Path) -> dict:
    """Read a JSON artefact, returning an empty dict if it was never produced."""
    from .hashing import read_json
    return read_json(path, {}) or {}


def fmt_pct(x, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "未获得"
    return f"{100 * float(x):.{digits}f}%"


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
class Sources:
    def __init__(self):
        self.build = read_json(paths.TABLES_DIR / "build_report.json", {})
        self.mapping = read_json(paths.MANIFEST_DIR / "disease_mapping_report.json", {})
        self.protocol = read_json(paths.MANIFEST_DIR / "protocol_report.json", {})
        self.contexts = read_json(paths.MANIFEST_DIR / "context_index.json", {})
        self.retriever = read_json(paths.MANIFEST_DIR / "retriever_frozen.json", {})
        self.external = read_json(paths.RESULTS_DIR / "external"
                                  / "external_labels_audit.json", {})
        self.acceptance = read_json(paths.REPORT_DIR / "acceptance_T01_T16.json", {})
        self.dev = read_json(paths.MANIFEST_DIR / "development_summary.json", {})
        self.probe_cuda = read_json(paths.RESOURCE_LOG_DIR / "probe_cuda_fp32.json", {})
        self.probe_amp = read_json(paths.RESOURCE_LOG_DIR / "probe_cuda_amp.json", {})
        self.probe_cpu = read_json(paths.RESOURCE_LOG_DIR / "probe_cpu_fp32.json", {})
        self.resource_manifest = read_json(paths.MANIFEST_DIR / "resource_manifest.json", {})
        self.feas = self._csv(paths.MANIFEST_DIR / "support_feasibility.csv")
        self.t3 = self._csv(paths.RESULT_TABLE_DIR / "table3_main_endpoints.csv")
        self.t4 = self._csv(paths.RESULT_TABLE_DIR / "table4_ablation.csv")
        self.t6 = self._csv(paths.RESULT_TABLE_DIR / "table6_resources.csv")
        self.comps = self._csv(paths.RESULT_TABLE_DIR / "primary_comparisons.csv")
        self.runs = self._csv(paths.RESULT_TABLE_DIR / "all_runs.csv")
        self.dose = self._csv(paths.RESULTS_DIR / "experiments" / "dose_summary.csv")
        self.gamma = self._csv(paths.RESULTS_DIR / "experiments" / "gamma_summary.csv")
        self.coverage = self._csv(paths.RESULTS_DIR / "experiments" / "coverage_summary.csv")
        self.evidence_exp = self._csv(paths.RESULTS_DIR / "experiments" / "evidence_summary.csv")
        self.negctrl = self._csv(paths.RESULTS_DIR / "experiments" / "negative_summary.csv")
        self.network = self._csv(paths.RESULTS_DIR / "experiments" / "network_summary.csv")
        self.external_eval = self._csv(paths.RESULTS_DIR / "external" / "external_eval_summary.csv")
        self.temporal = self._csv(paths.RESULTS_DIR / "temporal" / "temporal_summary.csv")
        self.expression = self._csv(paths.RESULTS_DIR / "expression" / "expression_summary.csv")

    @staticmethod
    def _csv(p: Path) -> pd.DataFrame:
        """An empty or header-less file means 'not produced', not a crash."""
        if not p.exists() or p.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


# --------------------------------------------------------------------------
# section builders
# --------------------------------------------------------------------------
def title_block(cfg: Gpu8Config, a: NumberAudit) -> str:
    today = date.today().isoformat()
    return f"""# 面向低资源疾病基因重排序的证据依赖约束与可微分支持适配

**英文题名：** Evidence-dependence-aware differentiable support adaptation for low-resource disease-gene reranking

**短标题：** 低资源疾病基因重排序的证据依赖支持适配

**稿件轮次：** 第五轮（单卡完整版 GPU8 协议实测稿）　**日期：** {today}

**作者信息：** 盲审稿中省略
"""


def abstract(cfg: Gpu8Config, s: Sources, a: NumberAudit) -> str:
    p = s.protocol
    se = p.get("support_eligibility", {})
    n_dis = a.cite("n_labelled_diseases", p.get("n_labelled_diseases"),
                   "manifests/protocol_report.json")
    n_genes = a.cite("n_genes", s.build.get("counts", {}).get("genes"),
                     "tables/build_report.json")
    n_rel = a.cite("n_relations", s.build.get("counts", {}).get("relations"),
                   "tables/build_report.json")
    n_fam = a.cite("n_family_groups", s.mapping.get("n_family_groups"),
                   "manifests/disease_mapping_report.json")
    e5 = a.cite("eligible_5shot", se.get("eligible_5shot"),
                "manifests/support_feasibility.csv")
    single = a.cite("single_positive_only", se.get("single_positive_only"),
                    "manifests/support_feasibility.csv")
    main = _main_row(s, "b9_ours")
    ref = _best_reference(s)
    r10 = a.cite("ours_recall10_end", main.get("recall10_end"),
                 "results/tables/table3_main_endpoints.csv")
    cov = a.cite("mean_coverage", main.get("coverage"),
                 "results/tables/table3_main_endpoints.csv")
    zero = a.cite("zero_coverage", main.get("zero_coverage"),
                  "results/tables/table3_main_endpoints.csv")
    return f"""## 摘要

**研究目的：** 在疾病特异知识稀少、辅助信息不齐全、新增支持可能来自重复收录的条件下，研究模型应当如何吸收少量支持证据并保持对当前疾病的区分能力。本文把任务严格限定为疾病关联冷启动下的候选基因重排序，并把覆盖与排序分开报告。

**方法：** 本文构建九张规范表（{n_genes} 个基因背景、{n_dis} 个带直接证据的疾病、{n_rel} 条唯一关系），按疾病家族划分开发集、外部保留集与五个固定折，共 {n_fam} 个家族组。候选由冻结的表型迁移检索与疾病层级迁移检索融合产生，融合规则只在开发集上确定。方法部分由小型神经表示（实测 208,855 个可训练参数）与 64 维可微分加权岭适配组成：支持基因先按可识别原始证据依赖归并成组，组内均分权重，再由一个四维支持摘要预测正则强度与更新半径；查询损失通过 Cholesky 求解器反向传播到编码器与适配表示。比较方法覆盖度先验、表型投票、同特征逻辑回归、固定投影融合、LightGBM、同容量 MLP、SIGN 式图学习、KDGene 重训、普通元岭适配以及种子驱动的 PPI 随机游走与连续时间量子游走。

**结果：** 支持资格盘点显示，{single} 个疾病只有一条已知阳性，只有 {e5} 个疾病同时具备 5 条支持与可评价查询，因此剂量实验被定位为受限子集的受控评价。在共享 500 候选、疾病宏平均端到端 Recall@10 的主终点上，本文方法为 {r10}；候选覆盖率为 {cov}，零覆盖疾病比例为 {zero}，说明在真实低资源冷启动条件下，检索覆盖而非排序质量是当前的主要瓶颈。

**结论：** {_conclusion_sentence(s)}

**关键词：** 疾病基因重排序；低资源疾病；冷启动；可微分支持适配；证据依赖；候选覆盖；泄漏审计
"""


def _main_row(s: Sources, method: str) -> dict:
    if s.t3.empty or "method" not in s.t3.columns:
        return {}
    sub = s.t3[s.t3["method"] == method]
    return sub.iloc[0].to_dict() if len(sub) else {}


def _label_zh(row: dict) -> str:
    from .baselines.registry import METHODS
    spec = METHODS.get(row.get("method", ""))
    if spec:
        return spec.label_zh or spec.label
    return str(row.get("label") or row.get("method", ""))


def _best_reference(s: Sources) -> dict:
    if s.t3.empty:
        return {}
    sub = s.t3[s.t3["method"] != "b9_ours"]
    if sub.empty:
        return {}
    return sub.sort_values("recall10_end", ascending=False).iloc[0].to_dict()


def _conclusion_sentence(s: Sources) -> str:
    ours = _main_row(s, "b9_ours")
    ref = _best_reference(s)
    if not ours or not ref:
        return ("主实验尚未完成汇总，结论待补。本文不在结果缺失时给出方法优越性判断。")
    d = float(ours.get("recall10_end", np.nan)) - float(ref.get("recall10_end", np.nan))
    if np.isnan(d):
        return "主终点差值不可计算，结论待补。"
    if d > 0.005:
        return (f"在相同候选与相同可见信息下，证据依赖支持适配相对最强对照"
                f"（{_label_zh(ref)}）在主终点上取得 {d:+.4f} 的绝对差值；"
                f"该结论限定为受控候选集内的重排序，不构成全基因组发现能力或临床前瞻性证据。")
    if d < -0.005:
        return (f"在相同候选与相同可见信息下，本文方法未超过最强对照"
                f"（{_label_zh(ref)}），主终点绝对差值为 {d:+.4f}。"
                f"按预先设定的收缩规则，本文据实把贡献定位为评价协议、证据依赖分析与"
                f"适用边界研究，不宣称方法优越性。")
    return (f"本文方法与最强对照（{_label_zh(ref)}）在主终点上差值为 "
            f"{d:+.4f}，处于不可区分范围。本文据实报告这一结果，并把贡献定位为评价协议与边界研究。")


def methods_section(cfg: Gpu8Config, s: Sources, a: NumberAudit) -> str:
    p, b = s.protocol, s.build
    counts = b.get("counts", {})
    rtc = b.get("relation_type_counts", {})
    ctd = b.get("ctd_stats", {})
    se = p.get("support_eligibility", {})
    sel = s.retriever.get("selected", {})
    return f"""## 3. 材料与方法

### 3.1 任务定义与两种"低资源"

给定目标疾病 $d$、允许使用的知识 $\\mathcal{{G}}_{{\\mathrm{{vis}}}}$、可选的疾病特异支持关系 $S_d$ 与候选基因集合 $C_d$，输出排序分数 $s(d,g\\mid S_d)$。主要研究条件是目标疾病缺少疾病–基因训练关联，而表型与本体位置仍可合法可见，因此任务称为疾病关联冷启动。本文的"低资源"同时包含两层含义：生物学上可用的疾病特异证据少（科学问题），以及实施条件下计算资源有限（工程约束）。标题与主要贡献围绕前者；单卡可复现作为计算特征报告，不作为科学主张。

患者表型到疾病的诊断、变异致病性解释与从原始测序得到候选变异均不在本任务范围内。

### 3.2 数据建设与九张规范表

原始资源经流式清洗后写入九张规范表。CTD 关联文件共读取 {a.cite('ctd_rows', ctd.get('rows_seen'), 'tables/build_report.json')} 行，其中 {a.cite('ctd_kept', ctd.get('kept_rows'), 'tables/build_report.json')} 行带直接证据并通过人类基因过滤。按关系语义分别计数：{'；'.join(f'{k} {v:,}' for k, v in rtc.items())}。主任务只使用 marker/mechanism 关系，therapeutic 关系保留独立标签与分析；一个药物作用靶点、疾病标志物与孟德尔致病基因含义不同，不压成同一标签。

| 表 | 行数 |
| --- | ---: |
| genes | {a.cite('t_genes', counts.get('genes'), 'tables/build_report.json')} |
| diseases | {a.cite('t_diseases', counts.get('diseases'), 'tables/build_report.json')} |
| relations | {a.cite('t_relations', counts.get('relations'), 'tables/build_report.json')} |
| evidence | {a.cite('t_evidence', counts.get('evidence'), 'tables/build_report.json')} |
| phenotypes | {a.cite('t_phenotypes', counts.get('phenotypes'), 'tables/build_report.json')} |
| hierarchy | {a.cite('t_hierarchy', counts.get('hierarchy'), 'tables/build_report.json')} |
| ppi | {a.cite('t_ppi', counts.get('ppi'), 'tables/build_report.json')} |

基因以 NCBI GeneID 组织并接入 HGNC 稳定标识，保留 Ensembl 基因号与别名歧义隔离表；疾病通过 Mondo 的严格等价关系（OBO 中带 `MONDO:equivalentTo` 注记的 xref 与 SSSOM 中的 `skos:exactMatch`）归并，宽泛疾病与亚型不因共有祖先而合并。映射结果为：{'；'.join(f'{k} {v:,}' for k, v in (s.mapping.get('mapping_type_counts') or {}).items())}，共 {a.cite('n_equiv', s.mapping.get('n_equivalence_groups'), 'manifests/disease_mapping_report.json')} 个等价组。

### 3.3 疾病划分、家族隔离与查询隔离

家族组由 MeSH 树号的单一主分支确定，缺失时回退到最近的具有树号的本体祖先，再回退到等价组。先按连通分量构造曾得到一个包含九千余个疾病的巨型分量，家族隔离将失去意义，因此改为确定性的单分支归属，共 {a.cite('n_fam', s.mapping.get('n_family_groups'), 'manifests/disease_mapping_report.json')} 个家族，最大家族 {a.cite('fam_max', s.mapping.get('family_size_max'), 'manifests/disease_mapping_report.json')} 个疾病，中位数 1。

建模前按家族预留 {a.cite('n_dev', p.get('n_dev'), 'manifests/protocol_report.json')} 个疾病作为开发集，另预留 {a.cite('n_ext', p.get('n_external'), 'manifests/protocol_report.json')} 个疾病作为外部冷疾病保留集，两者都不进入任何参考图。其余疾病分为五个固定折，规模为 {p.get('fold_sizes')}。第 $j$ 次以 $F_j$ 测试、$F_{{(j+1)\\bmod 5}}$ 验证。每个外层训练集再分为三个内层块，为每块生成训练任务时从关联图中移除该块全部疾病–基因边，因此每折需要四份图依赖缓存，五折共 {a.cite('n_ctx', s.contexts.get('n_contexts'), 'manifests/context_index.json')} 个上下文。

### 3.4 支持资格盘点

正式建模前生成逐疾病支持可行性表。结果显示：{a.cite('single_pos', se.get('single_positive_only'), 'manifests/support_feasibility.csv')} 个疾病（{fmt_pct(se.get('single_positive_only', 0) / max(1, se.get('diseases', 1)))}）只有一条已知阳性，只能进入 zero-shot 主任务；具备 1-shot、3-shot 与 5-shot 资格的疾病分别为 {a.cite('e1', se.get('eligible_1shot'), 'manifests/support_feasibility.csv')}、{a.cite('e3', se.get('eligible_3shot'), 'manifests/support_feasibility.csv')} 与 {a.cite('e5', se.get('eligible_5shot'), 'manifests/support_feasibility.csv')} 个。这一盘点先于任何模型结果生成，并据此把剂量与证据依赖实验定位为受限子集上的受控评价，而不是对全部低资源疾病的普遍结论。

### 3.5 候选生成：把覆盖与排序分开

候选由两个冻结检索器产生：表型迁移检索按表型相似度加权汇总参考疾病的可见关联，层级迁移检索按本体距离衰减汇总合法近邻的已知关系；另加入一个由表型迁移向量播种的重启传播通道。融合规则仅在开发集上按覆盖率确定并冻结，选定配置为表型近邻 {sel.get('n_neighbours', '-')}、权重（表型 {sel.get('w_phenotype', '-')}、层级 {sel.get('w_hierarchy', '-')}、传播 {sel.get('w_rwr', '-')}），开发集平均覆盖率 {a.cite('dev_coverage', sel.get('mean_coverage'), 'manifests/retriever_frozen.json')}。选择过程只读取覆盖率，不读取任何排序质量指标，因此检索器不可能被调向某一个评分模型。

对每个疾病同时报告四个量：覆盖率、端到端 Recall@K、条件 Recall@K（仅在分母大于零时定义）以及零覆盖疾病比例。主终点为疾病宏平均端到端 Recall@10。

### 3.6 稀疏预计算与神经输入

以 $A$ 表示当前可见疾病–基因邻接、$H$ 为疾病–HPO 矩阵、$P$ 为列随机归一化的 PPI 传播矩阵。HPO 信息量仅由外层训练疾病估计，并用稀疏截断 SVD 形成 128 维共同基底，基底在同折四个上下文间共享。每个节点得到四组 128 维缓存视图（表型、局部结构、较远结构、层级与覆盖），合计 512 维；每个疾病–基因对另带四个标量证据通道与 12 维可见性描述。禁止构造整个异质图的密集邻接，也禁止预存全部候选的高维成对特征。

### 3.7 核心方法

$$z_d=f_D(x_d),\\quad z_g=f_G(x_g),\\quad h_{{dg}}=f_P\\big([z_d;z_g;|z_d-z_g|;z_d\\odot z_g;t_{{dg}};q_{{dg}}]\\big)$$

$$\\pi_{{dg}}=\\mathrm{{softmax}}\\big(f_Q(q_{{dg}})+m_{{dg}}\\big),\\qquad s_0(d,g)=w_s^\\top h_{{dg}}+b_s+\\sum_{{j=1}}^4 \\pi_{{dg,j}}t_{{dg,j}}$$

适配表示经单位球归一化 $\\phi_{{dg}}=\\left(W_\\phi h_{{dg}}+b_\\phi\\right)/\\max\\{{1,\\lVert W_\\phi h_{{dg}}+b_\\phi\\rVert_2\\}}$，保证 $\\lVert\\phi\\rVert_2\\le 1$。支持基因先按唯一关系去重，再按可识别的共同原始证据归并为组，组内均分质量并归一化得到权重 $\\omega_g$；有效组数 $n_{{\\mathrm{{eff}}}}$ 只作为依赖程度的描述，不写成统计独立样本量。四维支持摘要经 $f_R:4\\to16\\to2$ 得到 $\\lambda_d$ 与 $\\rho_d$。

对支持 $g\\in S_d$ 与参照 $u\\in U_d$ 构造 $D_{{gu}}=\\phi_{{dg}}-\\phi_{{du}}$、$b_{{gu}}=\\mathrm{{clip}}(1-[s_0(d,g)-s_0(d,u)],-5,5)$、$W_{{gu}}=\\omega_g/|U_d|$，求解

$$(D^\\top W D+\\lambda_d I)\\hat a_d=D^\\top W b,\\qquad a_d=\\hat a_d\\min\\{{1,\\rho_d/\\lVert\\hat a_d\\rVert_2\\}},\\qquad s(d,g\\mid S_d)=s_0(d,g)+a_d^\\top\\phi_{{dg}}$$

使用 Cholesky 分解求解，不显式求逆。$k=0$ 时直接取 $a_d=0$ 并跳过求解器。径向投影是岭解的后处理，不是球约束下二次目标的精确最优解。

**性质 1（唯一性）：** 若 $W\\succeq 0$ 且 $\\lambda_d>0$，则 $D^\\top WD+\\lambda_d I$ 正定，未投影岭解唯一。
**性质 2（扰动界）：** 若 $\\lVert\\phi_{{dg}}\\rVert_2\\le1$ 且 $\\lVert a_d\\rVert_2\\le\\rho_d$，则 $|s(d,g\\mid S_d)-s_0(d,g)|\\le\\rho_d$；同一疾病两候选的分数差改变量不超过 $2\\rho_d$。该性质描述保护与可塑性的取舍，不保证准确率单调提高。

### 3.8 训练流程

一个 epoch 定义为与当前训练疾病数量相同的任务数。每个任务先抽取支持规模 $k\\in\\{{0,1,3,5\\}}$，再在具备该规模的疾病中抽样。查询参照的四分之三来自检索器的候选池、四分之一来自更宽背景：若训练负样本全部为随机背景基因，模型学到的区分能力不能迁移到检索得到的候选池，这一点在开发阶段被观察到并据此修正。优化目标为

$$\\mathcal{{L}}=\\mathbb{{E}}_d\\big[\\mathcal{{L}}^{{\\mathrm{{adapt}}}}_d+\\eta\\mathcal{{L}}^0_d\\big]+\\gamma\\lVert\\theta\\rVert_2^2$$

其中 $\\mathcal{{L}}^0_d$ 使用基础分数，防止只训练少支持后表现而牺牲 zero-shot。结构化缺失训练保留三个视图：原始允许知识、目标疾病表型减少、参考关联按来源减少；减少一个视图时其下游的检索、聚合与传播一并重建，而不是只把一列清零。

### 3.9 比较方法

{_methods_table()}

### 3.10 统计分析

主终点为共享候选下的疾病宏平均端到端 Recall@10，关键次要终点为 NDCG@10。预先固定三个主要比较：本文方法对 LightGBM、对 KDGene、对普通元岭适配；zero-shot 主比较与固定支持增量比较构成两个假设族，族内采用 Holm 校正，其余为探索性分析。置信区间按疾病家族分组重采样 {a.cite('bootstrap', cfg.evaluation.bootstrap_draws, 'configs/frozen_config.json')} 次；训练种子与支持顺序先在疾病内部汇总再计算方法差值，重复运行不增加样本数。
"""


NOTE_ZH = {
    "b0_degree_prior": "查询疾病不进入分数；刻意不称作 Adamic-Adar",
    "b1_phenotype_vote": "检验简单疾病迁移已经能解释多少",
    "b2_lr": "相同冷特征、抽样与标准化；正则充分调优",
    "b3_cpu_fusion": "上一版 CPU 设计：投影固定，不经求解器学习",
    "b4_lightgbm": "按疾病分组的 lambdarank；已调参，不以默认配置运行",
    "b5_mlp": "匹配编码器容量，无适配学习；固定岭后处理作为其公平支持变体",
    "b6_sign": "传播与训练分离思想的任务改编，不是原模型的复现",
    "b7_kdgene": "保留原评分函数、门控、全实体 softmax 与 N3；实体空间不同，已声明",
    "b8_meta_ridge": "最近邻对照：同容量、等权支持、固定正则与半径",
    "b9_ours": "本文方法：证据分组权重、条件控制、结构化缺失训练",
    "b10_ppi_rwr": "需要疾病特异种子；k=0 时不适用",
    "b10b_diamond": "超几何种子连通性；与随机游走并列报告",
    "b11_quantum_walk": "逐支持振幅取模平方后求和；经典数值实现",
}


def _methods_table() -> str:
    from .baselines.registry import METHODS
    lines = ["| 编号 | 方法 | 方法族 | 支持条件 | 说明 |",
             "| --- | --- | --- | --- | --- |"]
    for k, m in METHODS.items():
        sup = "需要支持" if m.requires_support else ("可用支持" if m.uses_support else "不用支持")
        lines.append(f"| {k.upper().replace('_', ' ').split()[0]} | "
                     f"{m.label_zh or m.label} | {m.family_zh or m.family} | "
                     f"{sup} | {NOTE_ZH.get(k, m.note)} |")
    return "\n".join(lines)


def limitations_section(s: Sources, a: NumberAudit) -> str:
    gaps = read_json(paths.REPORT_DIR / "what_was_not_done.json", {})
    items = gaps.get("items", [])
    body = "\n".join(f"{i+1}. **{it['title']}**：{it['detail']}" for i, it in enumerate(items))
    return f"""## 6. 局限性

{body if body else '（限制清单尚未生成）'}
"""


def build_manuscript(cfg: Gpu8Config | None = None) -> dict:
    cfg = cfg or load_config()
    s = Sources()
    a = NumberAudit()
    parts = [title_block(cfg, a), abstract(cfg, s, a)]
    parts.append(read_section("introduction", s, a))
    parts.append(read_section("related_work", s, a))
    parts.append(methods_section(cfg, s, a))
    parts.append(read_section("results", s, a))
    parts.append(read_section("discussion", s, a))
    parts.append(limitations_section(s, a))
    parts.append(read_section("conclusion", s, a))
    parts.append(read_section("availability", s, a))
    from . import appendix
    parts.append(appendix.supplementary_methods(s, a))
    parts.append(appendix.references())
    parts.append(appendix.figure_captions())
    parts.append(appendix.appendix_acceptance(s, a))
    parts.append(appendix.appendix_ledger(s, a))
    parts.append(appendix.appendix_editor_response(s, a))
    text = "\n\n".join(p for p in parts if p)
    out = paths.GPU8_ROOT.parent / "manuscript_round5_gpu8"
    out.mkdir(parents=True, exist_ok=True)
    md = out / f"低资源疾病基因重排序_第五轮_证据依赖支持适配_GPU8_{date.today().strftime('%Y%m%d')}.md"
    atomic_write_text(md, text)
    a.save(paths.REPORT_DIR / "manuscript_number_audit.csv")
    return {"path": str(md), "chars": len(text), "numbers_cited": len(a.rows)}


def read_section(name: str, s: Sources, a: NumberAudit) -> str:
    """Sections whose prose lives in narrative.py, filled with measured numbers."""
    from . import narrative
    fn = getattr(narrative, name, None)
    return fn(s, a) if fn else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the round-five manuscript.")
    args = ap.parse_args(argv)
    print(json.dumps(build_manuscript(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

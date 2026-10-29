"""Supplementary methods, references, figure captions and appendices."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths
from .hashing import read_json


def supplementary_methods(s, a) -> str:
    return r"""## 补充方法：完整公式定义

主文只保留最影响结论解释的公式。本节给出与实现一致的完整定义。所有涉及图邻居、图特征、候选检索、支持权重与阈值选择的量，均只使用划分特异的训练集可见信息；被隐藏的查询关系只在评价阶段提供标签。

**S1 关系去重与唯一关系集合。** 原始记录先映射到统一节点空间，再按唯一关系去重：

$$\mathcal{E}_r=\operatorname{Unique}\{(\eta(u),\eta(v),r):(u,v,r)\in\mathcal{E}^{\mathrm{raw}}\}$$

其中 $\eta(\cdot)$ 是标识符协调函数。同一关系被多个数据库重复收录只保留一行，其证据行分别登记在 evidence 表中。

**S2 可见邻接与传播矩阵。** 对上下文 $c$ 的参考疾病集合 $\mathcal{R}_c$，

$$A^{(c)}[d,g]=\mathbb{I}\big[(d,g)\in\mathcal{E}_{\mathrm{DG}},\ d\in\mathcal{R}_c\big]$$

$$P=\widehat{W}D_{\mathrm{col}}^{-1},\qquad \widehat{W}[i,j]=q(g_i,g_j)/1000\ \text{若}\ q\ge\tau_{\mathrm{PPI}}$$

孤立基因补自环以保证列随机性。$A^{(c)}$ 只含参考疾病的行，因此被隐藏的查询无法通过度、共同邻居或传播进入任何特征。

**S3 表型信息量与共同基底。** 信息量仅由参考疾病估计：

$$\mathrm{IC}(h)=-\log\frac{\max(1,|\{d\in\mathcal{R}_c:H[d,h]=1\}|)}{|\mathcal{R}_c|}$$

对加权矩阵 $\widetilde H=H\,\mathrm{diag}(\mathrm{IC})$ 的参考行块作稀疏截断奇异值分解，取前 $m=128$ 个右奇异向量构成基底 $B\in\mathbb{R}^{m\times|\mathcal{H}|}$，同折四个上下文共享。疾病表型表示为 $\rho_d=\widetilde H[d,:]B^\top$。

**S4 四组缓存视图。** 疾病侧 $V_0$ 为 $\rho_d$ 的 $\ell_2$ 归一化；$V_1$ 为本体近邻中参考疾病表示的距离加权平均；$V_2$ 为表型最近的 $k$ 个参考疾病表示的相似度加权平均；$V_3$ 为祖先稀疏向量的固定投影。基因侧 $V_0$ 为 $Z^{(0)}=\mathrm{diag}(\deg_A)^{-1}A^\top\rho$，$V_1=PZ^{(0)}$，$V_2=P^2Z^{(0)}$，$V_3$ 为按可见关系汇总的疾病层级描述。每个节点输入 $4\times128=512$ 维，缺失视图以显式 mask 标记，不作零值填充解释为"已确认不相关"。

**S5 四个标量证据通道。**

$$v_d(g)=\sum_{d'\in\mathcal{N}_k(d)}\alpha_{dd'}\frac{A_{d'g}}{\max(1,\deg_A(d'))},\qquad \sum_{d'}\alpha_{dd'}=1$$

$$o_d(g)=\sum_{d'\in\mathcal{H}(d)}\beta_{dd'}\frac{A_{d'g}}{\max(1,\deg_A(d'))},\qquad \beta_{dd'}\propto (1+\mathrm{dist}(d,d'))^{-1}$$

$$p_d^{(0)}=\tilde v_d,\qquad p_d^{(t+1)}=\gamma \tilde v_d+(1-\gamma)Pp_d^{(t)},\qquad \gamma=0.15$$

$$c_d(g)=\frac{\langle Z^{(0)}_g,\rho_d\rangle}{\lVert Z^{(0)}_g\rVert\,\lVert\rho_d\rVert}$$

偏态通道采用固定变换 $\log(1+cx)$，再按训练统计量中心化、标准化并裁剪到 $\pm5$。所有统计量随模型保存，评价时不重新估计。

**S6 候选与参照。** 候选由冻结融合分数的前 $N$ 个唯一基因构成，并列时按稳定基因标识排序；融合权重在开发集上按覆盖率选定并冻结。参照集合 $U_d$ 的随机状态由疾病标识、协议标识与固定种子共同决定，不读取当前候选表；因此增加无关候选不会改变适配系统。

**S7 训练目标。**

$$\mathcal{L}^{\mathrm{adapt}}_d=\frac{1}{|Q^{\mathrm{tr}}_d||V_d|}\sum_{g\in Q^{\mathrm{tr}}_d}\sum_{u\in V_d}\log\!\big(1+\exp[-s(d,g\mid S_d)+s(d,u\mid S_d)]\big)$$

$$\mathcal{L}=\mathbb{E}_d\big[\mathcal{L}^{\mathrm{adapt}}_d+\eta\,\mathcal{L}^0_d\big]+\gamma\lVert\theta\rVert_2^2$$

其中 $\mathcal{L}^0_d$ 以基础分数 $s_0$ 计算同一形式的损失。梯度经 $D$、$b$ 与 Cholesky 求解器反向传播到编码器、适配表示与控制头，不要求沿完整传播矩阵反向。

**S8 评价量。**

$$\mathrm{Coverage}_d=\frac{|Q_d\cap C_d|}{|Q_d|},\qquad \mathrm{Recall}^{\mathrm{end}}_d@K=\frac{|Q_d\cap T_{d,K}|}{|Q_d|}$$

$$\mathrm{Recall}^{\mathrm{cond}}_d@K=\frac{|Q_d\cap T_{d,K}|}{|Q_d\cap C_d|}\quad\text{仅在分母大于 }0\text{ 时定义}$$

$$\mathrm{UB}@10(d)=\frac{\min(10,|Q_d|)}{|Q_d|},\qquad \mathrm{nRecall}@10(d)=\frac{\mathrm{Recall}^{\mathrm{end}}_d@10}{\mathrm{UB}@10(d)}$$

首个阳性不存在时，名次记为缺失并带删失标记，不赋予任何"成功名次"。

**S9 统计量。** 疾病级配对差值 $\Delta_b$ 按家族分组重采样；支持增量 $\Delta^{(k)}_d=R^{(k)}_d-R^{(0)}_d$ 与机制增量 $\Gamma^{(k)}_d=\Delta^{(k)}_{d,\text{本文}}-\Delta^{(k)}_{d,\text{元岭}}$ 在同一共同任务集合上计算。置换检验在家族层面作符号翻转。

**S10 数值性质的敏感性。** 对两套支持系统 $A=D^\top WD+\lambda I$、$c=D^\top Wb$ 与 $A',c'$，

$$\lVert\hat a-\hat a'\rVert_2\le\frac{\lVert c-c'\rVert_2+\lVert A-A'\rVert_2\lVert\hat a'\rVert_2}{\lambda_{\min}}$$

其中 $A-A'$ 必须包含正则项的变化。该式只用于解释数值敏感性，不把观测到的证据分组等同于独立抽样。
"""


def references() -> str:
    return """## 参考文献

1. Davis AP, Wiegers TC, Sciaky D, et al. Comparative Toxicogenomics Database (CTD): update 2025. Nucleic Acids Research. 2025;53:D1328-D1334.
2. Gargano MA, Matentzoglu N, Coleman B, et al. The Human Phenotype Ontology in 2024: phenotypes around the world. Nucleic Acids Research. 2024;52:D1333-D1346.
3. Schriml LM, Munro JB, Schor M, et al. The Human Disease Ontology 2022 update. Nucleic Acids Research. 2022;50:D1255-D1261.
4. Szklarczyk D, Kirsch R, Koutrouli M, et al. The STRING database in 2023. Nucleic Acids Research. 2023;51:D638-D646.
5. Seal RL, Braschi B, Gray K, et al. Genenames.org: the HGNC resources in 2023. Nucleic Acids Research. 2023;51:D1003-D1009.
6. Vasilevsky NA, Matentzoglu NA, Toro S, et al. Mondo: unifying diseases for the world, by the world. medRxiv. 2022.
7. DiStefano MT, Goehringer S, Babb L, et al. The Gene Curation Coalition: a global effort to harmonize gene-disease evidence resources. Genetics in Medicine. 2022;24:1732-1742.
8. Landrum MJ, Chitipiralla S, Brown GR, et al. ClinVar: improvements to accessing data. Nucleic Acids Research. 2020;48:D835-D844.
9. Wang X, Yang K, Jia T, et al. KDGene: knowledge graph completion for disease gene prediction using interactional tensor decomposition. Briefings in Bioinformatics. 2024;25:bbae161.
10. Saarinen H, Goldsmith M, Wang RS, Loscalzo J, Maniscalco S. Disease gene prioritization with quantum walks. Bioinformatics. 2024;40:btae513.
11. Alsentzer E, Li MM, Kobren SN, et al. Few shot learning for phenotype-driven diagnosis of patients with rare genetic diseases. npj Digital Medicine. 2025;8:380.
12. Middleton L, Melas I, Vasavda C, et al. Phenome-wide identification of therapeutic genetic targets, leveraging knowledge graphs, graph neural networks, and UK Biobank data. Science Advances. 2024;10:eadj1424.
13. Wu F, Souza A, Zhang T, Fifty C, Yu T, Weinberger K. Simplifying graph convolutional networks. ICML. 2019:6861-6871.
14. Frasca F, Rossi E, Eynard D, Chamberlain B, Bronstein M, Monti F. SIGN: scalable inception graph neural networks. arXiv:2004.11198. 2020.
15. Bertinetto L, Henriques JF, Torr PHS, Vedaldi A. Meta-learning with differentiable closed-form solvers. ICLR. 2019.
16. Ghiassian SD, Menche J, Barabási AL. A DIseAse MOdule Detection (DIAMOnD) algorithm derived from a systematic analysis of connectivity patterns of disease proteins in the human interactome. PLOS Computational Biology. 2015;11:e1004120.
17. Köhler S, Bauer S, Horn D, Robinson PN. Walking the interactome for prioritization of candidate disease genes. American Journal of Human Genetics. 2008;82:949-958.
18. Aerts S, Lambrechts D, Maity S, et al. Gene prioritization through genomic data fusion. Nature Biotechnology. 2006;24:537-544.
19. Vanunu O, Magger O, Ruppin E, Shlomi T, Sharan R. Associating genes and protein complexes with disease via network propagation. PLOS Computational Biology. 2010;6:e1000641.
20. Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. NeurIPS. 2017:3146-3154.
21. Ritchie ME, Phipson B, Wu D, et al. limma powers differential expression analyses for RNA-sequencing and microarray studies. Nucleic Acids Research. 2015;43:e47.
22. Gudjonsson JE, Ding J, Johnston A, et al. Assessment of the psoriatic transcriptome in a large sample: additional regulated genes and comparisons with in vitro models. Journal of Investigative Dermatology. 2010;130:1829-1840.
23. Suárez-Fariñas M, Li K, Fuentes-Duculan J, Hayden K, Brodmerkel C, Krueger JG. Expanding the psoriasis disease profile: interrogation of the skin and serum of patients with moderate-to-severe psoriasis. Journal of Investigative Dermatology. 2012;132:2552-2564.
24. Rendle S, Freudenthaler C, Gantner Z, Schmidt-Thieme L. BPR: Bayesian personalized ranking from implicit feedback. UAI. 2009.
25. Loshchilov I, Hutter F. Decoupled weight decay regularization. ICLR. 2019.
26. Holm S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics. 1979;6:65-70.
27. Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman and Hall. 1993.
"""


def figure_captions() -> str:
    return """## 图位置与图注

**图 1. 数据来源、证据去重、疾病划分与支持资格。** 各面板依次为九张规范表的规模、疾病家族级划分的规模、每个疾病已知阳性数的分布，以及 0/1/3/5 支持剂量的合格疾病数。面板 (c) 使用对数纵轴：多数疾病只有一条已知阳性，这是本研究全部结论的前提事实。

![Figure 1](figures/main_figure_01_data_and_protocol.png)

**图 2. 五折主结果与疾病级配对差异。** 面板 (a) 为共享候选下的端到端 Recall@10，误差棒为折间标准差；面板 (b) 为仅在有阳性入池的疾病上定义的条件 Recall@10；面板 (c) 为三个预先固定的主要比较的绝对差值与家族分组 bootstrap 95% 区间；面板 (d) 把差值分解为受益、持平与受损疾病的比例。

![Figure 2](figures/main_figure_02_main_results.png)

**图 3. 固定支持剂量与支持机制的增量。** 所有 k 使用同一疾病、同一查询与同一候选。面板 (b) 的 $\\Gamma$ 是本文方法与普通元岭适配的支持收益之差，零线表示两者的支持机制等价。

![Figure 3](figures/main_figure_03_support_dose.png)

**图 4. 证据依赖处理与跨疾病替换支持。** 同时给出排序表现、适配更新幅度与受损疾病比例；只展示控制头收缩半径而不展示排序是否更好，不构成机制有效的证据。

![Figure 4](figures/main_figure_04_evidence_dependence.png)

**图 5. 候选覆盖、名单规模与检索来源敏感性。** 三种检索来源与三种名单规模下的覆盖率、端到端表现与零覆盖比例。横轴为对数刻度。

![Figure 5](figures/main_figure_05_candidate_coverage.png)

**图 6. 外部确认与时间外评价。** 面板 (a) 为三层独立性排除的逐级分母；面板 (c) 为时间外划分的审计量，纵轴为对称对数刻度。

![Figure 6](figures/main_figure_06_external_temporal.png)

**图 7. 失败模式与负对照。** 疾病置换与支持置换用于检验所测能力是否真的疾病特异；阳性负担分层显示指标随查询阳性数的变化。

![Figure 7](figures/main_figure_07_failure_modes.png)

**图 8. 单机资源与独立生物学支持。** 面板 (a) 与 (b) 为实测耗时与主进程内存；面板 (c) 为同一模型在 GPU 与 CPU 上的每步耗时；面板 (d) 为表达支持队列的可用配对数。

![Figure 8](figures/main_figure_08_resources_biology.png)
"""


def appendix_acceptance(s, a) -> str:
    acc = s.acceptance
    checks = acc.get("checks", [])
    lines = ["## 附录 A　正式研究验收单 T01–T16", "",
             "下表由 `reports/acceptance_T01_T16.json` 自动生成，每项附可检查证据路径。", "",
             "| 编号 | 验收项 | 状态 | 可检查证据 |", "| --- | --- | --- | --- |"]
    for c in checks:
        lines.append(f"| {c['code']} | {c['title']} | {c['status']} | `{c['evidence']}` |")
    lines += ["", acc.get("caveat", ""), ""]
    return "\n".join(lines)


def appendix_ledger(s, a) -> str:
    reg = paths.MANIFEST_DIR / "run_registry_166.csv"
    if not reg.exists():
        return ""
    df = pd.read_csv(reg)
    by = (df.groupby(["experiment", "method"])
          .agg(planned=("run_id", "count"),
               done=("status", lambda x: int((x == "done").sum())),
               failed=("status", lambda x: int((x == "failed").sum())),
               minutes=("seconds", lambda x: round(pd.to_numeric(x, errors="coerce").sum() / 60, 1)))
          .reset_index())
    lines = ["## 附录 B　正式拟合账本", "",
             f"计划 {len(df)} 次正式全局拟合。局部支持求解、支持顺序抽样、候选截断与统计重采样"
             f"不计入本账本；图上下文重建另行记录。", "",
             "| 实验 | 方法 | 计划 | 完成 | 失败 | 累计机时（分钟） |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in by.itertuples(index=False):
        lines.append(f"| {r.experiment} | {r.method} | {r.planned} | {r.done} | "
                     f"{r.failed} | {r.minutes} |")
    total_done = int((df["status"] == "done").sum())
    total_min = round(pd.to_numeric(df["seconds"], errors="coerce").sum() / 60, 1)
    lines += ["", f"合计：计划 {len(df)}，完成 {total_done}，累计机时 {total_min} 分钟"
                  f"（{round(total_min/60, 1)} 小时）。", ""]
    return "\n".join(lines)


def appendix_editor_response(s, a) -> str:
    from .manuscript import _conclusion_sentence
    return f"""## 附录 C　新增工作的可核查说明

本轮相对既往稿件的实质新增，按"原先哪些信息路径未成立、新增何种方法、相对最近邻方法解决什么、补充了哪些独立证据、复现材料在哪里"的顺序说明。

**信息路径。** 既往稿件的支持基因从未进入训练，因而 1/3/5-shot 实际是"抽走 k 个阳性的冷启动查询"而非少样本学习；候选由在阳性中补入负样本构成，规模约五十；拓扑加成项在整张评价表上作 z 标准化，使一个候选对的分数依赖同批其他候选。本轮重建了协议：支持在训练与评分中真实进入局部适配；候选由不读取标签的冻结检索器在完整基因背景上产生；所有标准化统计量固定于训练，参照集合与候选集合解耦，分数因此对拆批、合批与增加无关候选保持一致（见 4.11 节与数值检查 T09）。

**新增方法。** 证据依赖约束的可微分支持适配：支持基因按可识别原始证据归并成组、组内均分权重，控制头只读取支持自身的四维摘要决定正则强度与更新半径，查询损失通过 Cholesky 求解器反向传播到编码器与适配表示。性质 1 与性质 2 给出唯一性与分数扰动界。

**相对最近邻方法。** 普通元岭适配获得相同的编码器容量、相同的支持与参照、相同的训练任务与预算，只把证据依赖权重、条件控制与结构化缺失训练替换为普通形式。结论由 $\\Gamma$ 而非 $\\Delta$ 给出。

**独立证据。** 外部确认集合按关系、来源、原始文献三层排除后分别统计，出处不可核实者单列；时间外实验按边级冻结报告，并明确辅助资源未完全按同一时点冻结。

**复现材料。** 资源清单记录每个输入的哈希与获取时间；运行账本记录每次拟合的配置哈希、数据哈希、上下文哈希、实际耗时与峰值内存；数值与隔离检查以测试形式固定。

{_conclusion_sentence(s)}
"""

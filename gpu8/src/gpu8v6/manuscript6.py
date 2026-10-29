"""Compose the V6 manuscript (Markdown) from the collected result numbers.

Structure follows V6 section 17: the title drops "evidence-source aware" because
the source-group weights did not help; methods run data visibility -> routing
-> candidate generation -> fusion / residual -> support propagation -> source
groups -> evaluation; results run data and reachable bound -> zero-support
coverage -> in-list scoring -> support dose -> evidence mechanism -> external,
temporal and resources.  Every number is read from
``reports_v6/manuscript_numbers_v6.json``; negative results are kept.
"""
from __future__ import annotations

import json
from pathlib import Path

from gpu8 import paths as p5

from . import paths6
from .common import now_utc

OUT_DIR = p5.PROJECT_ROOT / "manuscript_v6"
TITLE = "面向异构信息缺失的疾病基因候选恢复与支持驱动排序"
SUBTITLE = ("Candidate recovery under heterogeneous information missingness and "
            "support-driven ranking for low-resource disease-gene prioritisation")


class N:
    def __init__(self, v: dict):
        self.v = v
        self.used: set = set()

    def raw(self, k):
        self.used.add(k)
        return self.v[k]

    def f(self, k, d=3):
        x = self.raw(k)
        return f"{x:.{d}f}"

    def i(self, k):
        return f"{int(round(self.raw(k))):,}"

    def pc(self, k, d=1):
        return f"{100 * self.raw(k):.{d}f}%"

    def ci(self, pre, d=3, lo="_ci_low", hi="_ci_high", delta="_delta"):
        return (f"{self.raw(pre + delta):+.{d}f}（95% CI {self.raw(pre + lo):+.{d}f} 至 "
                f"{self.raw(pre + hi):+.{d}f}）")

    def p(self, k):
        x = self.raw(k)
        return "< 0.001" if x < 0.001 else f"{x:.3f}"

    def cih(self, pre, holm_key, d=3):
        """Difference with its CI and Holm p in one bracket."""
        return (f"{self.raw(pre + '_delta'):+.{d}f}（95% CI {self.raw(pre + '_ci_low'):+.{d}f} 至 "
                f"{self.raw(pre + '_ci_high'):+.{d}f}；Holm p {self.p(holm_key)}）")

    def mag(self, k, d=3):
        return f"{abs(self.raw(k)):.{d}f}"


def build(v: dict) -> tuple[str, set]:
    n = N(v)
    L = []
    A = L.append
    A(f"# {TITLE}")
    A("")
    A(f"**{SUBTITLE}**")
    A("")
    A("第六轮（V6）修订稿 · 2026 年 9 月 23 日 · 依据《DiseaseGene V6 再建模与修复方案》完成的实测版本")
    A("")
    # ------------------------------------------------------------------ abstract
    A("## 摘要")
    A("")
    A("**背景**　低资源疾病的基因排序同时面对两种缺失：没有该病的已知基因（零支持），以及疾病本身缺少表型、文本或本体信息。第五轮（V5）评价显示，"
      f"主要瓶颈不在排序器而在候选覆盖：共享 500 候选下的 oracle 上界 U@10 只有 {n.f('v5_b3_U')}，"
      f"而 64 维元岭表示空间适配在支持增加时没有获益。")
    A("")
    A("**方法**　V6 撤下元岭主路径，把零支持检索与有支持图传播作为两条并行修复线分别建模与检验。"
      "所有疾病按可观测信息（HPO、合法文本、真实本体、显露支持）分流；疾病侧 HPO 按 Mondo 严格等价修复映射并过滤非症状项；"
      "基因侧 HPO 在按来源疾病排除当前上下文的留出疾病后才聚合；疾病文本在预先固定的基因符号词典下掩蔽。"
      "候选由疾病层级、疾病本体、掩蔽文本、来源排除的基因 HPO 与以表型迁移为种子的传播五类来源在全基因背景上排序后，按开发集冻结的倒数排名融合生成，预算固定为 500，无信号填充一律标记回退原因。"
      "候选内评分先调优固定融合 F，再在袋外候选池上训练 LightGBM 与 κ·tanh 残差并以一标准误差规则选择；"
      "有支持时以显露支持为种子作残差停止的随机游走，比较 PPI 与 GO/通路功能桥接。"
      "来源组权重只在当前显露支持内计算，并做四层诊断与打乱组对照。疾病置换拆为 P1（神经疾病分支）、P2（完整评分条件）与 P3（完整管线）。"
      "分析计划在任何 V6 测试评价之前登记，主要比较按家族 bootstrap 与 Holm 校正。")
    A("")
    A(f"**结果**　在五折 {n.i('z_v6_lightgbm_n_diseases')} 个疾病上，V6 检索把共享 500 候选的覆盖率从 {n.f('z_v5_b3_cpu_fusion_coverage')} 提高到 "
      f"{n.f('z_v6_lightgbm_coverage')}，oracle 上界 U@10 提高 {n.cih('cmp_ZC', 'cmp_ZC_holm_p')}，"
      f"零覆盖疾病比例从 {n.pc('z_v5_b3_cpu_fusion_zero_coverage')} 降到 {n.pc('z_v6_lightgbm_zero_coverage')}，"
      f"V5 无任何查询阳性入池的 {n.i('v5_zero_n')} 个疾病中有 {n.i('rescued_n')} 个被救回。"
      f"端到端 Recall@10 由 V5 最佳方法的 {n.f('z_v5_b3_cpu_fusion_recall10_end')} 提高到 {n.f('z_v6_lightgbm_recall10_end')}"
      f"，差 {n.cih('cmp_ZA', 'cmp_ZA_holm_p')}；但在同一候选内，LightGBM 相对调优融合 F 的差为 "
      f"{n.cih('cmp_ZB', 'cmp_ZB_holm_p')}，神经残差没有可重复增益，且 P1 置换其疾病分支几乎不改变指标。"
      f"支持线在全部五折 {n.i('s_n')} 个 5-shot 疾病上，以 5 个支持为种子的传播相对零支持语义评分提高 {n.cih('s_SA', 's_SA_holm_p')}，"
      f"保度重连后该收益消失；证据来源组权重相对等权种子为 {n.cih('s_SC', 's_SC_holm_p')}，"
      "种子与前十名单确有改变，但没有可检测的指标收益。")
    A("")
    A("**结论**　在当前数据上，性能提升主要来自对异构信息缺失的候选恢复与合法新数据，而不是更复杂的排序器；"
      "支持信息在图空间中有效，证据来源组加权未获支持。本文保留全部阴性结果，并把数据建设贡献与算法贡献分开陈述。")
    A("")
    A("**关键词**　疾病基因排序；冷启动；候选覆盖；信息缺失；随机游走；证据依赖；置换检验；预注册")
    A("")
    # ------------------------------------------------------------------ 1 intro
    A("## 1. 引言")
    A("")
    A("疾病—基因排序的实际价值集中在信息最少的疾病上：新近命名的疾病、只有一个已知基因的疾病、没有表型注释的疾病。"
      "这类疾病的“低资源”有两层含义。其一是**支持缺失**：该病尚无或只有极少已知基因，模型无法从该病自身学习；"
      "其二是**描述缺失**：该病没有 HPO 注释、没有定义文本，或只挂在过宽的本体位置上。"
      "两种缺失在数据中的分布并不重合，因此一个把两者混在一起优化的单一模型，很难说明收益究竟来自哪一层。")
    A("")
    A("本研究的主问题因此从“如何设计一个统一的元适配器”改为：**在不同可用证据下，如何恢复候选，并利用有限支持进行疾病特异排序**。"
      "我们把零支持任务（Z）、固定查询的支持吸收任务（S）与外部/时间任务（X）分开定义与评价，任何两种任务的绝对召回都不相互解释。")
    A("")
    A("第五轮实测给出了几条必须正视的结果：共享 500 候选的平均覆盖率只有 "
      f"{n.f('z_v5_b3_cpu_fusion_coverage')}，{n.pc('z_v5_b3_cpu_fusion_zero_coverage')} 的疾病没有任何查询阳性进入名单；"
      "64 维元岭适配随支持增加没有获益，而以支持为种子的网络方法呈正向剂量信号；证据权重对排名没有可见作用。"
      "据此，V6 的修订说明写为：")
    A("")
    A("> “第五轮评价表明，主要性能限制来自候选覆盖不足及信息可用性异质性。原表示空间适配未呈现稳定的支持剂量收益，证据权重对最终排名的增益亦未获得支持。基于上述观察，后续模型将零支持语义迁移与支持驱动网络传播分别建模，并以固定查询任务检验二者的独立作用。新增的基因功能与表型知识按照来源疾病和时间边界重建，所有检索及评分改动均与共享相同输入的强基线比较。”")
    A("")
    A("本文的工作与可宣称的贡献如下，每一条都附带可证伪的检验：")
    A("")
    A("1. **信息可用性审计与损失分解**。以 $R_d=U_d\\,\\eta_d$ 把端到端召回拆成候选上界与候选内效率，并区分“资源无记录”“映射失败”“过滤后无有效症状项”三类无 HPO。")
    A("2. **零支持候选恢复**。在同一 500 预算下，逐项接入疾病本体、掩蔽文本、来源排除的基因 HPO 等合法来源，报告单源覆盖、留一贡献、每百个新增候选的新增命中与来源重叠。")
    A("3. **支持驱动传播**。以显露支持为种子的随机游走作为支持主路径，固定候选（S-fixed）与自适应候选（S-adaptive）分开评价，并以保度重连检验其对真实网络结构的依赖。")
    A("4. **机制检验**。疾病置换拆成 P1/P2/P3；证据来源组权重做种子、传播分数、排名与指标四层诊断，并与同规模打乱组比较。")
    A("5. **协议**。V5 五折结果作为诊断档案保留；V6 分析计划在测试评价之前登记，模型选择只用内层验证与开发集，并以一标准误差规则偏向简单模型。")
    A("")
    A("需要事先说明：随机游走、倒数排名融合、LightGBM 与列表式损失都是已有工具。本文不把这些工具本身当作新算法；"
      "本文要检验的是，针对当前信息缺失结构的处理能否带来独立、可解释的收益。结果显示，收益主要来自候选恢复与合法数据，"
      "而不是评分器；证据来源组加权没有带来可检测收益，因此不出现在题目与主要创新中。")
    A("")
    # ------------------------------------------------------------------ 2 related
    A("## 2. 相关工作")
    A("")
    A("基于表型的候选基因排序以 HPO 语义相似度为核心[2,33,34]，网络方法以已知基因为种子做随机游走或邻域富集[16,17,19]，"
      "数据融合方法把多类基因属性合并排序[18]。这些方法默认目标疾病已有表型描述或种子基因；本文的零支持任务恰恰需要处理两者都可能缺失的疾病。")
    A("")
    A("知识图谱补全方法（如 KDGene[9]）在统一图上学习疾病与基因表示；连续时间量子游走[10]以种子振幅传播排序。"
      "本文在共同合法图上重训 KDGene，并按 V6 要求为其提供合法冷验证早停与学习曲线；量子游走与随机游走作为支持线的对照。")
    A("")
    A("在评价方法上，模型选择对有限验证集过拟合是已知问题[31]，一标准误差规则是降低高方差选择的常用做法[39]。"
      "倒数排名融合在多来源检索中稳健[32]，列表式排序损失与 LambdaRank 族方法[37,38,20]用于候选内学习排序。"
      "保度重连[36]是检验网络结构贡献的标准负对照。")
    A("")
    # ------------------------------------------------------------------ 3 methods
    A("## 3. 材料与方法")
    A("")
    A("### 3.1 任务、终点与损失分解")
    A("")
    A("**任务 Z（零支持冷启动）**：疾病在其评价上下文中没有任何显露的已知基因，全部已知基因为查询阳性。"
      "**任务 S（支持吸收）**：对已知阳性不少于 6 个的疾病，沿用 V5 的固定设计——先固定 5 个基因的最大支持池与其余查询，再以 10 个嵌套顺序显露 $k\\in\\{0,1,3,5\\}$ 个支持；同一疾病、同一查询。"
      "**任务 X（外部/时间）**：外部标签层与 2023Q4 时间切分单列，永不与 Z 合并平均。")
    A("")
    A("主终点为共享 500 候选上的疾病宏平均端到端 Recall@10。对疾病 $d$，记查询阳性数 $n_d$、入池阳性数 $m_d$、前 $K$ 名命中数 $h_d(K)$：")
    A("")
    A("$$c_d=\\frac{m_d}{n_d},\\qquad U_d(K)=\\frac{\\min(K,m_d)}{n_d},\\qquad R_d(K)=\\frac{h_d(K)}{n_d} \\tag{1}$$")
    A("")
    A("$$\\eta_d(K)=\\frac{h_d(K)}{\\min(K,m_d)},\\qquad R_d(K)=U_d(K)\\,\\eta_d(K)\\quad(m_d>0) \\tag{2}$$")
    A("")
    A("$U_d$ 是当前候选名单上任何排序器可达的 oracle 上界。排序损失为 $U_d-R_d$，检索容量损失为 $\\min(K,n_d)/n_d-U_d$。"
      "$m_d=0$ 时 $R_d=0$、$\\eta_d$ 不定义并单独计数；宏平均召回是 $R_d$ 的均值，不是宏平均覆盖率与宏平均效率之积。"
      "疾病结局比例（未入池 / 入池未进前十 / 恢复）是疾病计数，不与宏平均召回互换。")
    A("")
    A("### 3.2 数据可见性与修复")
    A("")
    A("数据沿用 V5 九张规范表（CTD 直接证据 marker/mechanism 关系、HPO、疾病本体、STRING v12.0 物理互作 ≥700、HGNC 蛋白编码背景）。"
      f"有主关系的疾病共 {n.i('n_labelled')} 个；开发集、外部留出集与五个外层折沿用 V5 冻结划分。V6 只做以下修复，且全部在建模前决定、提供给所有比较器：")
    A("")
    A(f"- **疾病 HPO 的严格映射修复**。V5 只经 CTD 的 OMIM 备用 ID 挂接 HPO 注释（{n.i('hpoa_mapped_v5')} 个 HPOA 疾病 ID）。V6 另外沿“项目疾病 → 唯一 Mondo 等价类 → 同样唯一对应该类的 OMIM/Orphanet ID”的严格路径挂接，"
      f"共 {n.i('hpoa_mapped_v6')} 个 ID，新增 {n.i('hpoa_new_v6')} 个；{n.i('strict_conflicts')} 个一对多或冲突映射被丢弃并记录。宽、窄、近似匹配不使用。")
    A("- **无效术语过滤**。只保留表型异常分支（aspect P）、非 NOT 限定、深度大于 2 的 HPO 项；遗传方式、临床病程、修饰语与频率项移除。")
    A(f"- **疾病文本与掩蔽**。文本只取 CTD MEDIC 名称/同义词/定义、Disease Ontology 与 Mondo 定义。按预先固定的 HGNC 符号词典掩蔽基因答案（全大写段落只掩蔽批准符号，罗马数字与普通大写词不掩蔽），"
      f"{n.i('text_masked_diseases')} 个疾病共 {n.i('text_masked_tokens')} 个记号被掩蔽，原文与掩蔽文本均保留供审计。")
    A("- **来源排除的基因 HPO**。`genes_to_phenotype.txt` 的每一行带来源疾病。先把来源疾病映射到项目疾病与严格等价组，排除当前上下文所有未作参照的有标签疾病（验证、测试、开发、外部与内层留出块）的等价组，"
      "然后才去重、向上闭包并聚合到基因；来源无法映射的行只进入宽松敏感性版本。")
    A(f"- **独立基因特征**。GO（GOA 人类 GAF，真路径传播）与 Reactome 的 TF-IDF/SVD 表示、GTEx v8 组织表达，共 {n.i('gene_feat_dim')} 维并各带缺失标志；"
      f"{n.i('genes_go')}、{n.i('genes_rx')}、{n.i('genes_gtex')} 个基因分别有 GO、通路与表达信息。SVD 只在无标签基因属性上拟合。")
    A(f"- **严格等价参照排除**。每个上下文的参照集合中，凡与任一留出有标签疾病同属严格等价组者一律移除（V5 中仅 {n.i('eq_leak_test')} 个测试疾病存在这种等价参照）。")
    A("")
    A("### 3.3 按可见信息分流")
    A("")
    A("分流只使用部署时可观测的量：HPO 是否可用（修复后）、是否有定义文本、本体位置是否有信息量、显露支持数 $k$。"
      "阳性总数只作为报告分层，不作为模型输入。零支持疾病分为 Z1（有 HPO）、Z2（无 HPO 但有文本或本体）、Z3（无合法疾病信息）；"
      "Z3 的名单由可追溯的背景先验完成，并明确不称为疾病特异预测，也不从评价中删除。")
    A("")
    A("### 3.4 零支持候选生成")
    A("")
    A("每个来源在当前上下文的**全基因背景**上打分，参照疾病只用其可见关系。表型迁移使用信息量加权、祖先闭包后的 HPO 余弦相似度：")
    A("")
    A("$$\\mathrm{sim}(d,d')=\\frac{\\langle q_d\\odot \\mathrm{IC},\\,q_{d'}\\odot \\mathrm{IC}\\rangle}{\\lVert q_d\\odot \\mathrm{IC}\\rVert\\,\\lVert q_{d'}\\odot \\mathrm{IC}\\rVert},\\qquad s_d(g)=\\sum_{d'\\in\\mathcal N_{20}(d)}\\frac{\\mathrm{sim}(d,d')}{\\sum_{d''}\\mathrm{sim}(d,d'')}\\,\\frac{A_{d'g}}{\\deg(d')} \\tag{3}$$")
    A("")
    A("其中 IC 只在参照疾病上估计。文本来源以掩蔽后的 TF-IDF 余弦选取参照疾病，本体来源以 MeSH 树号的最信息共同祖先（Resnik 信息量，忽略信息量低于阈值的过宽祖先）选取参照疾病，二者都按式（3）转移可见关系；"
      "层级来源沿用 V5 的 CTD 父子关系两步邻域；基因 HPO 来源为疾病画像与来源排除基因画像的 IC 加权余弦；传播来源以表型迁移分数为种子作随机游走。"
      "来源在全背景上按排名作倒数排名融合，缺失来源被掩蔽并重分配权重：")
    A("")
    A("$$F^{\\mathrm{ret}}_d(g)=\\frac{\\sum_s w_s\\,a_{ds}\\,(k_0+r_{ds}(g))^{-1}}{\\sum_s w_s\\,a_{ds}},\\qquad k_0=60 \\tag{4}$$")
    A("")
    A(f"$a_{{ds}}$ 表示来源 $s$ 对疾病 $d$ 是否有信号，$r_{{ds}}(g)$ 为基因在该来源全背景中的排名。权重、融合方式与配额只在开发集（864 个疾病 × 5 个外层上下文）上选择："
      f"七来源 0.25 步长单纯形与两种融合方式共 {n.i('ret_n_grid')} 个配置，再做 {n.i('ret_n_refine')} 个 0.05 步长邻域细化，目标为宏平均 U@10。"
      f"冻结结果为倒数排名融合，权重 {n.raw('ret_weights')}；直接表型迁移的权重被调为 0，其信号经由以它为种子的传播来源进入。名单不足 500 时以可追溯的度先验补足，这些基因标记 `fallback_reason` 且不计为证据检索。")
    A("")
    A("### 3.5 候选内评分：固定融合与残差")
    A("")
    A("候选固定后，基础评分为非负单纯形上的线性融合：")
    A("")
    A("$$F(d,g)=\\sum_s w_s\\,v_{ds}(g),\\qquad w_s\\ge 0,\\ \\sum_s w_s=1 \\tag{5}$$")
    A("")
    A(f"$v_{{ds}}$ 为来源分数按疾病在全背景上最大值归一。F 在开发集上比较了 V5 手设权重、四通道 35 点网格加细化与七来源 210 点网格加细化；七来源版本 F7 被冻结，权重为 {n.raw('F7_weights')}。")
    A("")
    A("学习型评分器只学习疾病特异残差：")
    A("")
    A("$$s_\\theta(d,g)=F(d,g)+\\kappa\\tanh\\big(\\delta_\\theta(d,g)\\big),\\qquad \\kappa\\ge 0 \\tag{6}$$")
    A("")
    A("$\\kappa=0$ 始终是合法模型，输出逐元素等于 F。残差有两类：共享投影的低秩双线性 $\\delta=\\langle W_d x_d, W_g x_g\\rangle/\\sqrt r+u^\\top c_{dg}$，"
      "以及候选基因对该病至多 32 个最高信息量 HPO 项的双头交叉注意力；注意力只在同一疾病的 HPO 项与单个候选基因之间进行，不在候选之间互相注意。"
      "疾病表示只来自 HPO 与文本的 SVD，不训练自由疾病 ID 向量。训练在训练疾病的袋外候选池（由把该病留出的内层上下文生成）上做列表式损失：")
    A("")
    A("$$\\mathcal L_{\\mathrm{rank}}=-\\frac{1}{|D_{\\mathrm{hit}}|}\\sum_{d\\in D_{\\mathrm{hit}}}\\frac{1}{|P_d|}\\sum_{g\\in P_d}\\log\\frac{\\exp(s_\\theta(d,g)/\\tau)}{\\sum_{u\\in C^{\\mathrm{OOF}}_d}\\exp(s_\\theta(d,u)/\\tau)} \\tag{7}$$")
    A("")
    A("袋外池内没有阳性的训练疾病不伪造正例，只是不进入该损失；推理输出原始分数。同一袋外池上还训练了 LightGBM（lambdarank）与逻辑回归。"
      f"开发阶段在第 0 折上以两个内层验证集（相邻折 F1 与开发集）对每类 2 个预设做 {n.i('devfit_n')} 次试拟合；"
      "主评分器按一标准误差规则选择：在 1,712 个验证疾病上，取与最佳模型的配对差不超过一个标准误的最简单模型（F7 < LR < LightGBM < 双线性 < 注意力）。")
    A("")
    A("### 3.6 支持驱动传播")
    A("")
    A("在列随机算子 $P$ 上以显露支持的唯一基因为均匀种子 $u_d$ 作随机游走：")
    A("")
    A("$$p^{(t+1)}_d=\\alpha u_d+(1-\\alpha)Pp^{(t)}_d,\\qquad p^*_d=K_\\alpha u_d,\\quad K_\\alpha=\\alpha\\big[I-(1-\\alpha)P\\big]^{-1} \\tag{8}$$")
    A("")
    A("$K_\\alpha$ 只用于分析，实现为稀疏迭代，并以残差停止：若 $\\lVert p^{(t+1)}-p^{(t)}\\rVert_1\\le\\alpha\\epsilon$ 则 $\\lVert p^{(t)}-p^*\\rVert_1\\le\\epsilon$（$\\epsilon=10^{-6}$）；到达迭代上限仍未满足者记为失败。"
      "功能桥接以基因—功能关联矩阵 $B$（5–200 个基因的 GO 生物过程与 Reactome 通路）构造，并与 PPI 混合：")
    A("")
    A("$$P_F=BD_F^{-1}B^\\top D_G^{-1},\\qquad P_{\\mathrm{mix}}=(1-\\nu)P_{\\mathrm{PPI}}+\\nu P_F \\tag{9}$$")
    A("")
    A("无功能项的基因保留自环，两个算子均为列随机，混合后仍为列随机，且从不显式构造基因两两矩阵。语义路径与支持路径按")
    A("")
    A("$$p^{\\mathrm{joint}}_d=(1-\\omega)p^{\\mathrm{sem}}_d+\\omega\\,p^{\\mathrm{support}}_d \\tag{10}$$")
    A("")
    A(f"简单融合。$\\alpha\\in\\{{0.1,0.2,0.4\\}}$、$\\nu\\in\\{{0,0.1,0.3\\}}$、$\\omega\\in\\{{0.25,0.5,0.75\\}}$ 只在开发集 5-shot 疾病（{n.i('s_dev_cases')} 个疾病-上下文）上选择，冻结为 "
      f"$\\alpha={n.raw('s_alpha')}$、$\\nu={n.raw('s_nu')}$、$\\omega={n.raw('s_omega')}$。"
      "S-fixed 固定零支持 500 名单，支持只改变分数；S-adaptive 由联合分数在全背景（排除最大支持池）上重新生成 500 名单；全背景口径不设名单。"
      "由于 S-adaptive 的名单与排序使用同一分数，其前十与全背景前十相同，两者只在覆盖率上不同。")
    A("")
    A("### 3.7 来源组权重与四层诊断")
    A("")
    A("把当前显露支持与其可追溯的原始证据（PMID）构成二部关联 $B^S_{gc}$；同一来源的重复记录先合并，不做跨论文的传递合并。以来源组等质量、组内基因均分：")
    A("")
    A("$$v_d(g)=\\frac{1}{C_d}\\sum_{c=1}^{C_d}\\frac{B^S_{gc}}{\\sum_{g'\\in S_d}B^S_{g'c}},\\qquad p^0_d=(1-\\eta)u_d+\\eta v_d,\\quad \\eta\\in\\{0,0.25,0.5,1\\} \\tag{11}$$")
    A("")
    A("组大小只在当前 $S_d$ 内计算。因 RWR 对种子线性，")
    A("")
    A("$$\\big\\lVert p^{\\mathrm{group}}_d-p^{\\mathrm{equal}}_d\\big\\rVert_1=\\big\\lVert \\eta K_\\alpha(v_d-u_d)\\big\\rVert_1\\le \\eta\\,\\lVert v_d-u_d\\rVert_1 \\tag{12}$$")
    A("")
    A("因此 $k=1$、等大小组、完全重复来源时组权重与等权严格相同（程序单元测试）。四层诊断依次报告种子差 $\\lVert v-u\\rVert_1$、传播后全基因分数差、候选排名变化与指标差。机制统计量为")
    A("")
    A("$$\\Delta^{\\mathrm{seed}}_d=M_d(p^{\\mathrm{group}})-M_d(p^{\\mathrm{equal}}),\\qquad \\Gamma_{\\mathrm{grp}}=\\frac{1}{|D|}\\sum_d\\Big[M_d(p^{\\mathrm{true\\ group}})-\\mathbb E_\\pi M_d(p^{\\mathrm{shuffled\\ group}})\\Big] \\tag{13}$$")
    A("")
    A("打乱组在基因 PPI 度三分层内置换支持基因的来源模式，保持每个来源的组大小与支持数，每个 episode 取 20 次置换的均值；另设单纯连接度权重 $w\\propto(1+\\deg)^{-1/2}$。"
      "主要机制分析为 5-shot，3-shot 为次要，1-shot 为不变性检查；只有全部支持出处可追溯的 episode 进入机制分析，其余回退到 $\\eta=0$ 并报告回退比例。")
    A("")
    A("### 3.8 疾病置换")
    A("")
    A("P1 只把疾病的神经表示（残差模型的疾病语义与 HPO 项）换成供体，候选与检索分数不变；P2 在目标疾病的候选上用供体重算全部疾病依赖的评分输入"
      "（全部来源分数、其全背景排名、融合检索分数与疾病描述量）；P3 连候选一起换成供体的名单，标签不变。"
      "供体为同一测试折中路由类、HPO 数量分箱与文本长度分箱相同的疾病，分“远”（不同 MeSH 大类）与“近”（同大类、不同等价组）两种预定义方案，各抽 5 次并先在疾病内平均。"
      "除指标差外还报告候选内 Spearman 相关与前十重叠。")
    A("")
    A("### 3.9 评价协议与统计")
    A("")
    A("五个外层折已被 V5 使用，因此 V6 的五折结果是内部估计，不称为从未参与设计的确认性测试；V5 结果作为档案不被覆盖。"
      "V6 的检索、融合、支持配置与主评分器全部在开发集或内层验证上决定，分析计划（主终点、Z/S 两个假设族、准入与停止规则）在任何 V6 测试评价之前登记并哈希。"
      "学习型评分器每折 3 个种子（42/52/62），确定性评分与纯传播不做伪种子重复。种子与支持抽样先在疾病内平均，再做以疾病家族为单位的 10,000 次 bootstrap；"
      "主要比较在族内做 Holm 校正，其余均为探索性。所有差值写成“前者减后者”，消融差值统一为：")
    A("")
    A("$$\\Delta_{\\mathrm{abl}}=M_{\\mathrm{abl}}-M_{\\mathrm{full}} \\tag{14}$$")
    A("")
    A("“显著”只用于完成登记检验的比较；探索性区间不含 0 时写作“区间不含 0”。")
    A("")
    A("### 3.10 外部、时间与生物学支持")
    A("")
    A("外部冷疾病以全部非开发、非外部疾病为参照，按标签层分别评价：GenCC 确定/强、GenCC 中等（独立敏感性层）、ClinVar ≥2 星，以及引文经疾病条件核实且与 CTD 文献不重叠的 ClinVar 子集。"
      "ClinVar 引文链先连 VariationID 与 PMID，再要求该变异的全部条件映射到同一疾病才把 PMID 归给该疾病—基因对；多条件变异的 PMID 记为未知，不在条件间交叉相乘。"
      "时间任务先逐项审计输入的历史版本；由于 V5 的时间切分评价的是截止日已存在疾病的新关系（X2），V6 以其 2023Q4 关系为显露支持，只用截止日前版本的 PPI 与功能桥接做支持传播，并与 2023 年度先验比较。"
      "表达支持把冻结排名的前 20/50 基因传入既有 GEO 分析入口，与匹配背景及匹配随机集比较。")
    A("")
    # ------------------------------------------------------------------ 4 results
    A("## 4. 结果")
    A("")
    A("### 4.1 数据可见性与可达上界")
    A("")
    A(f"在 {n.i('n_labelled')} 个有标签疾病中，修复后有有效 HPO 的比例由 {n.pc('hpo_v5')} 升到 {n.pc('hpo_v6')}。"
      f"无 HPO 的原因主要是资源确无记录（{n.i('hpocls_no_resource_record')} 个），严格映射修复 {n.i('hpocls_mapping_failure_repaired')} 个，"
      f"有记录但过滤后无有效症状项 {n.i('hpocls_record_present_no_valid_symptom')} 个。"
      f"单阳性疾病占 {n.pc('single_frac')}，其中 {n.pc('single_hpo')} 有 HPO；而 {n.i('five_n')} 个 5-shot 合格疾病中无 HPO 的比例仅由 "
      f"{n.pc('five_nohpo_v5')} 降到 {n.pc('five_nohpo_v6')}，其中 {n.i('five_no_record')} 个属于资源无记录。"
      "也就是说，“有表型的零支持疾病”与“有支持而无表型的疾病”基本是两批不同的疾病，这一差异主要来自资源覆盖，而不是映射错误。")
    A("")
    A(f"按可观测信息分流，Z1（有 HPO）{n.i('route_Z1_hpo')} 个、Z2（无 HPO、有文本或本体）{n.i('route_Z2_no_hpo_text_or_ontology')} 个、Z3（无合法疾病信息）{n.i('route_Z3_no_legal_disease_information')} 个。"
      f"V5 的“层级检索”经代码核对读取的是 CTD 疾病父子关系而非 HPO 近邻；无 HPO 疾病中 {n.pc('hier_nohpo_neighbour')} 在参照集中有本体邻居。"
      f"CTD 主关系中有 {n.pc('go_shared_frac')} 的 PMID 同时是该基因 GO 注释的参考文献，因此 GO 被视为可审计的辅助知识，而不视为与发现文献完全独立的来源。")
    A("")
    A("**表 1　第五轮输出的损失分解与审计（V5 档案，只读）**")
    A("")
    A("| 项目 | 数值 |")
    A("|---|---|")
    A(f"| V5 最佳方法（CPU 融合）Recall@10 / 上界 U@10 / 效率 η | {n.f('v5_b3_R')} / {n.f('v5_b3_U')} / {n.f('v5_b3_eta')} |")
    A(f"| 疾病级 $R=U\\eta$ 恒等式 | 全部方法逐病成立 |")
    A(f"| 500→1000：覆盖率 / Recall@10 | {n.f('hf_coverage_small')}→{n.f('hf_coverage_big')} / {n.f('hf_recall_small',4)}→{n.f('hf_recall_big',4)} |")
    A(f"| 500→1000：新增基因占据前十位置 / 其中阳性 / 被挤出的原命中 | {n.i('hf_total_top_slots_taken_by_new')} / {n.i('hf_total_new_positive_entered_top')} / {n.i('hf_total_pushed_out')} |")
    A(f"| 消融 A1（去结构化缺失训练）$\\Delta_{{\\mathrm{{abl}}}}$ | {n.f('abl_a1_delta_abl',4)}（{n.f('abl_a1_ci_low',4)}，{n.f('abl_a1_ci_high',4)}） |")
    A(f"| 消融 A2（等权 + 固定控制）$\\Delta_{{\\mathrm{{abl}}}}$ | {n.f('abl_a2_delta_abl',4)}（{n.f('abl_a2_ci_low',4)}，{n.f('abl_a2_ci_high',4)}） |")
    A(f"| 消融 A3（去表型路径，候选同时改变）$\\Delta_{{\\mathrm{{abl}}}}$ | {n.f('abl_a3_delta_abl',4)}（{n.f('abl_a3_ci_low',4)}，{n.f('abl_a3_ci_high',4)}）；覆盖 {n.f('abl_a3_mean_coverage_full')}→{n.f('abl_a3_mean_coverage_abl')} |")
    A(f"| 消融 A4（去 PPI 视图）$\\Delta_{{\\mathrm{{abl}}}}$ | {n.f('abl_a4_delta_abl',4)}（{n.f('abl_a4_ci_low',4)}，{n.f('abl_a4_ci_high',4)}） |")
    A(f"| 同一完整模型不同种子之间的平均绝对差（同折） | {n.f('abl_a1_b9_seed_to_seed_mean_abs_diff',4)} |")
    A("")
    A("审计给出三点必须纠正的结论。第一，扩大名单时两份名单严格嵌套、同一基因对原始分数一致（最大差 1.7×10⁻⁶），"
      f"覆盖率上升而召回下降的原因是新增基因占据了 {n.i('hf_total_top_slots_taken_by_new')} 个前十位置、其中只有 {n.i('hf_total_new_positive_entered_top')} 个是阳性——检索与排序未协同，而非新增候选全部无用。"
      "第二，V5 表 4 的消融差值方向是“完整减消融”，且完整模型取 3 个种子、消融只取 2 个种子；按配对种子重算并统一为 $\\Delta_{\\mathrm{abl}}=M_{\\mathrm{abl}}-M_{\\mathrm{full}}$ 后，"
      f"A1、A2 的效应小于同一模型种子间差（{n.f('abl_a1_b9_seed_to_seed_mean_abs_diff',4)}），A3 同时改变了候选名单（候选哈希不同），不能作为单纯评分消融解释。"
      "第三，V5 的疾病置换对照属于“保留真实 k=5 支持、其余疾病输入全部来自供体”的混合形式，既非 P1 也非 P2；在 k=0 下重做后，"
      f"P1 只使 Recall@10 变化 {n.f('pv5_P1_distant_delta',4)}，P2 与 P3 分别使其下降 {n.mag('pv5_P2_distant_delta')} 与 {n.mag('pv5_P3_distant_delta')}（远供体，原始值 {n.f('pv5_orig_distant',4)}）。"
      "V5 评分器的疾病条件依赖来自检索通道，神经疾病分支本身几乎不起作用。")
    A("")
    A("### 4.2 零支持候选恢复")
    A("")
    A("**表 2　零支持检索逐项接入（开发集，864 个疾病 × 5 个上下文，预算 500）**")
    A("")
    A("| 配置 | U@10 | 直接证据覆盖 | 先验回退覆盖 | 零覆盖比例 |")
    A("|---|---|---|---|---|")
    ladder = [("L0_v5_components", "V5 组件（V6 参照集）"), ("L1_repaired_hpo", "+ HPO 修复"),
              ("L2_plus_onto", "+ 疾病本体"), ("L3_plus_text", "+ 掩蔽文本"),
              ("L4_plus_gene_hpo", "+ 来源排除基因 HPO（等权）"), ("L5_plus_compat", "+ 兼容通道（等权）")]
    for k, lab in ladder:
        A(f"| {lab} | {n.f(f'lad_{k}_U')} | {n.f(f'lad_{k}_direct')} | {n.f(f'lad_{k}_fill')} | {n.f(f'lad_{k}_zero')} |")
    A(f"| 冻结：倒数排名融合（调优权重） | {n.f('ret_dev_U')} | {n.f('ret_dev_direct')} | {n.f('ret_dev_fill')} | {n.f('ret_dev_zero')} |")
    A(f"| 仅度先验（疾病无关） | {n.f('prior_only_cov')} | 0.000 | {n.f('prior_only_cov')} | — |")
    A("")
    A(f"在开发集上，逐项接入使 U@10 由 {n.f('lad_L0_v5_components_U')} 升到 {n.f('lad_L3_plus_text_U')}；等权加入基因 HPO 反而下降到 {n.f('lad_L4_plus_gene_hpo_U')}，"
      f"说明来源需要按排名融合与调权，而非简单相加。冻结配置的 U@10 为 {n.f('ret_dev_U')}，其中由直接证据带入的覆盖为 {n.f('ret_dev_direct')}（V5 组件为 {n.f('lad_L0_v5_components_direct')}），"
      f"疾病无关的先验回退只贡献 {n.f('ret_dev_fill')}；仅用度先验的名单覆盖为 {n.f('prior_only_cov')}，这是任何疾病特异检索必须超过的下限。")
    A("")
    A(f"在五折测试疾病上，冻结检索的覆盖率为 {n.f('test_cov')}。按同一预算的留一分析，各来源对共同名单覆盖的贡献为："
      f"传播 {n.f('src_rwr_pheno_coverage_contribution')}、基因 HPO {n.f('src_gene_hpo_coverage_contribution')}、文本 {n.f('src_text_coverage_contribution')}、层级 {n.f('src_hier_coverage_contribution')}、本体 {n.f('src_onto_coverage_contribution')}；"
      f"每百个新增候选带来的新增命中以文本（{n.f('src_text_new_hits_per_100_added',2)}）与层级（{n.f('src_hier_new_hits_per_100_added',2)}）最高，传播最低（{n.f('src_rwr_pheno_new_hits_per_100_added',2)}），"
      "来源之间的名单重叠多数低于 0.12（基因 HPO 与兼容通道为 0.40）。"
      f"V5 中没有任何查询阳性入池的 {n.i('v5_zero_n')} 个疾病中，{n.i('rescued_n')} 个（{n.pc('rescued_share')}）被 V6 名单覆盖。")
    A("")
    A("**表 3　五折端到端结果与损失分解（4,238 个测试疾病，共享预算 500）**")
    A("")
    A("| 方法 | Recall@10 | 命中@10 | U@10 | η | 覆盖率 | 零覆盖 | NDCG@10 | MRR | Recall@50 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    rows = [("v5_b4_lightgbm", "V5 LightGBM"), ("v5_b8_meta_ridge", "V5 普通元岭"),
            ("v5_b9_meta_adapt", "V5 证据依赖元岭"), ("v5_b3_cpu_fusion", "V5 CPU 融合"),
            ("v6_degree_prior", "V6 名单 + 度先验"), ("v6_F_hand", "V6 名单 + V5 手设权重"),
            ("v6_F4", "V6 名单 + 调优四通道 F4"), ("v6_lr", "V6 逻辑回归"),
            ("v6_retrieval_order", "V6 检索融合顺序"), ("v6_F7", "V6 调优融合 F7"),
            ("v6_attention", "V6 注意力残差"), ("v6_bilinear", "V6 双线性残差"),
            ("v6_lightgbm", "V6 LightGBM（主评分器）")]
    for k, lab in rows:
        A(f"| {lab} | {n.f(f'z_{k}_recall10_end',4)} | {n.f(f'z_{k}_hit10')} | {n.f(f'z_{k}_oracle_recall10')} | {n.f(f'z_{k}_eta10')} | "
          f"{n.f(f'z_{k}_coverage')} | {n.f(f'z_{k}_zero_coverage')} | {n.f(f'z_{k}_ndcg10_end',4)} | {n.f(f'z_{k}_mrr',4)} | {n.f(f'z_{k}_recall50_end',4)} |")
    A("")
    A("命中@10 为至少一个已知基因进入前十的疾病比例（学习型方法按运行计算后在种子间平均）。")
    A("")
    A(f"V6 把检索容量损失从 {n.f('z_v5_b3_cpu_fusion_capacity_loss10')} 降到 {n.f('z_v6_lightgbm_capacity_loss10')}，"
      f"疾病结局中“未入池”由 {n.pc('z_v5_b3_cpu_fusion_share_not_retrieved')} 降到 {n.pc('z_v6_lightgbm_share_not_retrieved')}、"
      f"“恢复”由 {n.pc('z_v5_b3_cpu_fusion_share_recovered')} 升到 {n.pc('z_v6_lightgbm_share_recovered')}（疾病比例，不是宏平均召回）。"
      f"收益在无 HPO 的 Z2 疾病中尤为明显：U@10 由 {n.f('strat_v5_b3_cpu_fusion_Z2_no_hpo_text_or_ontology_U')} 升到 {n.f('strat_v6_lightgbm_Z2_no_hpo_text_or_ontology_U')}，"
      f"Recall@10 由 {n.f('strat_v5_b3_cpu_fusion_Z2_no_hpo_text_or_ontology_R')} 升到 {n.f('strat_v6_lightgbm_Z2_no_hpo_text_or_ontology_R')}；"
      f"Z1 疾病由 {n.f('strat_v5_b3_cpu_fusion_Z1_hpo_R')} 升到 {n.f('strat_v6_lightgbm_Z1_hpo_R')}。"
      f"按已知阳性数分层，单阳性疾病由 {n.f('strat_v5_b3_cpu_fusion_1_R')} 升到 {n.f('strat_v6_lightgbm_1_R')}，≥6 阳性疾病由 {n.f('strat_v5_b3_cpu_fusion_>=6_R')} 升到 {n.f('strat_v6_lightgbm_>=6_R')}。"
      "“同方法、旧资源/新资源”的比较中，V5 手设权重在 V6 名单上为 "
      f"{n.f('z_v6_F_hand_recall10_end',4)}（V5 名单上 {n.f('z_v5_b3_cpu_fusion_recall10_end',4)}），说明名单本身带来一部分收益，其余来自调优权重与新来源进入评分。")
    A("")
    A("### 4.3 候选内评分")
    A("")
    A("**表 4　登记的主要比较（Holm 校正在族内进行）**")
    A("")
    A("| 族 | 比较（前者减后者） | 指标 | 疾病数 | 差值（95% CI） | Holm p |")
    A("|---|---|---|---|---|---|")
    A(f"| Z | Z-A：V6 管线（LightGBM）− V5 CPU 融合 | Recall@10 | {n.i('cmp_ZA_n_diseases')} | {n.ci('cmp_ZA',4)} | {n.p('cmp_ZA_holm_p')} |")
    A(f"| Z | Z-B：LightGBM − 调优融合 F7（同一候选） | Recall@10 | {n.i('cmp_ZB_n_diseases')} | {n.ci('cmp_ZB',4)} | {n.f('cmp_ZB_holm_p')} |")
    A(f"| Z | Z-C：V6 检索 − V5 检索 | U@10 | {n.i('cmp_ZC_n_diseases')} | {n.ci('cmp_ZC',4)} | {n.p('cmp_ZC_holm_p')} |")
    A(f"| S | S-A：等权 RWR（k=5）− 零支持语义（k=0），S-fixed | Recall@10 | {n.i('s_SA_n_diseases')} | {n.ci('s_SA',4)} | {n.f('s_SA_holm_p')} |")
    A(f"| S | S-B：联合（ω={n.raw('s_omega')}）− 等权 RWR，k=5 | Recall@10 | {n.i('s_SB_n_diseases')} | {n.ci('s_SB',4)} | {n.f('s_SB_holm_p')} |")
    A(f"| S | S-C：来源组种子（η=1）− 等权种子，k=5 | Recall@10 | {n.i('s_SC_n_diseases')} | {n.ci('s_SC',4)} | {n.f('s_SC_holm_p')} |")
    A("")
    A(f"开发阶段，调优融合 F7 在开发集上的 Recall@10 为 {n.f('dev_F7',4)}，高于 V5 手设权重（{n.f('dev_hand',4)}）、调优四通道 F4（{n.f('dev_F4',4)}）与检索融合顺序（{n.f('dev_ret',4)}）。"
      f"一标准误差规则下，F7 相对最佳模型 LightGBM 的配对差为 {n.f('sel_F7_short',4)}，超过一个标准误（{n.f('sel_F7_se',4)}），因此登记 LightGBM 为主评分器、F7 为强对照；"
      f"两类神经残差在开发拟合中 {n.pc('devfit_kappa0_share',0)} 选择了 $\\kappa=0$。")
    A("")
    A(f"五折结果中，LightGBM 相对 F7 的差为 {n.cih('cmp_ZB', 'cmp_ZB_holm_p', 4)}，区间包含 0，开发集上的领先没有在五折上稳定再现。"
      f"注意力残差在 {n.i('formal_attention_residual_kept')}/{n.i('formal_attention_fits')} 次正式拟合中于验证折胜过 F 而保留了非零 κ（{n.f('formal_attention_kappa_min',2)}–{n.f('formal_attention_kappa_max',2)}），但测试 Recall@10 与 F7 几乎相同；P1 置换其疾病分支时指标变化 "
      f"{n.f('pv6_attention_residual_P1_distant_all_delta',4)}（远供体）与 {n.f('pv6_attention_residual_P1_close_all_delta',4)}（近供体），候选内 Spearman 相关保持在 {n.f('pv6_attention_residual_P1_distant_all_rho')} 以上。"
      "按准入条件（正常输入下相对调优融合有可重复增益，且 P1 支持其疾病条件路径），神经残差不纳入主方法。")
    A("")
    A(f"逐块接入基因特征的探索性分析显示，同一 LightGBM 只用检索通道时 Recall@10 为 {n.f('gab_channels_only',4)}，加入 GO 后为 {n.f('gab_plus_go',4)}（差 {n.f('gab_inc_plus_go',4)}，区间 {n.f('gab_inc_plus_go_lo',4)} 至 {n.f('gab_inc_plus_go_hi',4)}），"
      "再加入通路与表达均无变化。基因功能特征把基因视图的非零率提高到 90% 以上，但没有提高候选内前十恢复；这一分析在登记之后进行，不据此更换主评分器。")
    A("")
    A("**表 5　V6 评分器的疾病置换（五折，远供体，均值先在疾病内平均）**")
    A("")
    A("| 评分器 | 检验 | 原始 Recall@10 | 置换后 | 差（置换 − 原始） | 候选内 Spearman | 前十重叠 |")
    A("|---|---|---|---|---|---|---|")
    for sc, lab in (("F7", "调优融合 F7"), ("lightgbm", "LightGBM"), ("attention_residual", "注意力残差")):
        for t in ("P1", "P2", "P3"):
            key = f"pv6_{sc}_{t}_distant_all"
            if key + "_delta" not in v:
                continue
            rho = n.f(key + "_rho") if key + "_rho" in v else "—"
            top = n.f(key + "_top10", 2) if key + "_top10" in v else "—"
            A(f"| {lab} | {t} | {n.f(key + '_orig',4)} | {n.f(key + '_perm',4)} | {n.f(key + '_delta',4)} | {rho} | {top} |")
    A("")
    A("P2 使三种评分器的 Recall@10 都下降约九成，候选内相关降到 0.4–0.5、前十重叠低于 0.1；候选内评分通过合法疾病输入实现了疾病条件依赖。"
      "这一依赖来自检索通道与疾病描述量，而不是任何神经疾病分支。")
    A("")
    A("### 4.4 支持剂量")
    A("")
    A(f"支持线覆盖全部五折的 {n.i('s_n')} 个 5-shot 测试疾病（F0 {n.i('s_role_F0')}、F1 {n.i('s_role_F1')}、F2 {n.i('s_role_F2')}、F3 {n.i('s_role_F3')}、F4 {n.i('s_role_F4')}；"
      f"另有外部 {n.i('s_role_external')} 个与开发 {n.i('s_role_dev')} 个不参与测试统计，合计 715 个 5-shot 合格疾病）。")
    A("")
    A("**表 6　S-fixed 剂量曲线（Recall@10，533 个疾病，10 个嵌套顺序先在疾病内平均）**")
    A("")
    A("| 方法 | k=0 | k=1 | k=3 | k=5 |")
    A("|---|---|---|---|---|")
    A(f"| 零支持语义（F7） | {n.f('dose_sem_S_fixed_k0',4)} | — | — | — |")
    for m, lab in (("rwr_equal", "等权 RWR"), (f"joint_w{v['s_omega']:g}", f"联合 ω={v['s_omega']}"),
                   ("rwr_degree", "连接度权重 RWR"), ("rwr_group_eta1", "来源组种子 η=1（可追溯）"),
                   ("rwr_shuffle_eta1", "打乱组 η=1（可追溯）")):
        A(f"| {lab} | — | {n.f(f'dose_{m}_S_fixed_k1',4)} | {n.f(f'dose_{m}_S_fixed_k3',4)} | {n.f(f'dose_{m}_S_fixed_k5',4)} |")
    om = f"joint_w{v['s_omega']:g}"
    A(f"| 联合，S-adaptive（=全背景前十） | — | {n.f(f'dose_{om}_S_adaptive_k1',4)} | {n.f(f'dose_{om}_S_adaptive_k3',4)} | {n.f(f'dose_{om}_S_adaptive_k5',4)} |")
    A("")
    A(f"以 5 个支持为种子的传播相对零支持语义评分提高 {n.ci('s_SA',4)}，是登记检验中的显著结果；受益、持平、受损疾病分别占 "
      f"{n.pc('s_SA_frac_benefit')}、{n.pc('s_SA_frac_tie')}、{n.pc('s_SA_frac_harm')}，因此这是平均趋势而非逐病单调保证。"
      f"单个支持时传播（{n.f('dose_rwr_equal_S_fixed_k1',4)}）仍低于语义评分，而联合评分在各剂量下不低于传播；联合相对传播在 k=5 的差 {n.ci('s_SB',4)} 未达显著。"
      f"功能桥接相对纯 PPI 在 k=5 提高 {n.ci('s2_mix_vs_ppi_rwr_equal_k5',4)}（探索性，区间不含 0）。")
    A("")
    A(f"S-adaptive 由联合分数在全背景上重新生成名单，覆盖率随 k 由 {n.f(f'dosecov_{om}_S_adaptive_k1')} 升到 {n.f(f'dosecov_{om}_S_adaptive_k5')}，"
      f"但 Recall@10 在 k=5 的差（S-adaptive 减 S-fixed）为 {n.ci(f's2_adaptive_vs_fixed_{om}_k5',4)}，属探索性结果且区间不含 0。"
      "按停止规则，支持的贡献定位为“对已恢复候选的重排”，而不是“支持驱动的候选再生成”；零支持名单起到了有用的约束作用。")
    A("")
    A(f"保度重连把实际使用的 PPI 改写到只剩 {n.pc('rw_unchanged')} 的原边（节点度序列逐一相同，非孤立连通分量由 {n.i('rw_comp_before')} 个变为 {n.i('rw_comp_after')} 个）后，"
      f"k=5 的等权传播由 {n.f('rw_k5_real',4)} 降到 {n.f('rw_k5_rewired',4)}（差 {n.f('rw_k5_rewired_minus_real',4)}，区间 {n.f('rw_k5_ci_low',4)} 至 {n.f('rw_k5_ci_high',4)}）。"
      "支持收益依赖真实网络结构，而不仅是节点度。全部传播均满足残差停止条件，没有迭代失败。")
    A("")
    A("### 4.5 证据机制")
    A("")
    A("**表 7　来源组权重的四层诊断（η=1，可追溯 episode）**")
    A("")
    A("| 层 | k=3 | k=5 |")
    A("|---|---|---|")
    A(f"| episode 数 | {n.i('fl_k3_episodes')} | {n.i('fl_k5_episodes')} |")
    A(f"| 1 种子：$\\lVert v-u\\rVert_1$ 均值 / 种子改变比例 | {n.f('fl_k3_layer1_mean_l1_v_minus_u')} / {n.pc('fl_k3_layer1_share_seed_changed')} | {n.f('fl_k5_layer1_mean_l1_v_minus_u')} / {n.pc('fl_k5_layer1_share_seed_changed')} |")
    A(f"| 2 传播：全背景 L1 差均值（上界恒成立） | {n.f('fl_k3_layer2_mean_l1_propagated')} | {n.f('fl_k5_layer2_mean_l1_propagated')} |")
    A(f"| 3 排名：候选内 Spearman / 前十改变比例 | {n.f('fl_k3_layer3_mean_spearman')} / {n.pc('fl_k3_layer3_share_top10_changed')} | {n.f('fl_k5_layer3_mean_spearman')} / {n.pc('fl_k5_layer3_share_top10_changed')} |")
    A(f"| 4 指标：Recall@10 改变比例 / 平均差 | {n.pc('fl_k3_layer4_share_metric_changed')} / {n.f('fl_k3_layer4_mean_delta_recall10',4)} | {n.pc('fl_k5_layer4_share_metric_changed')} / {n.f('fl_k5_layer4_mean_delta_recall10',4)} |")
    A(f"| $\\Gamma_{{\\mathrm{{grp}}}}$（真组 − 打乱组） | {n.ci('s2_Gamma_grp_k3',4)} | {n.ci('s2_Gamma_grp_k5',4)} |")
    A("")
    A(f"k=1 时组权重与等权在种子、分数、排名和指标上逐一相同（{n.i('s2_Delta_seed_k1_n_diseases')} 个疾病差值均为 0），符合式（12）。"
      f"k=5 时种子在 {n.pc('fl_k5_layer1_share_seed_changed')} 的 episode 中改变、前十名单在 {n.pc('fl_k5_layer3_share_top10_changed')} 中改变，但指标只在 {n.pc('fl_k5_layer4_share_metric_changed')} 中改变，"
      f"来源组种子相对等权为 {n.ci('s_SC',4)}，相对同规模打乱组为 {n.ci('s2_Gamma_grp_k5',4)}；单纯连接度权重相对等权为 {n.ci('s2_degree_vs_equal_k5',4)}。"
      f"支持出处全部可追溯的 episode 比例在 k=1/3/5 为 {n.pc('s_trace_k1')}、{n.pc('s_trace_k3')}、{n.pc('s_trace_k5')}。"
      "依照停止规则，本文报告机制阴性：当前来源组加权改变了种子与排名，但没有提供可检测的额外收益，“证据依赖提升准确率”的主张撤回。")
    A("")
    A("### 4.6 外部、时间与生物学支持")
    A("")
    A("**表 8　外部标签层（探索性再分析；每层单独计算分母）**")
    A("")
    A("| 标签层 | 疾病 / 关系对 | 覆盖率 | F7 Recall@10（95% CI） | LightGBM 五折模型均值（95% CI） |")
    A("|---|---|---|---|---|")
    for lay, lab in (("gencc_definitive_strong", "GenCC 确定/强"), ("gencc_moderate_sensitivity", "GenCC 中等（敏感性层）"),
                     ("clinvar_2star", "ClinVar ≥2 星"), ("clinvar_citation_verified_independent", "ClinVar 引文核实且与 CTD 文献独立")):
        A(f"| {lab} | {n.i(f'ext_{lay}_n')} / {n.i(f'ext_{lay}_pairs')} | {n.f(f'ext_{lay}_cov')} | "
          f"{n.f(f'ext_{lay}_F_main_R')}（{n.f(f'ext_{lay}_F_main_lo')}–{n.f(f'ext_{lay}_F_main_hi')}） | "
          f"{n.f(f'extl_{lay}_R')}（{n.f(f'extl_{lay}_lo')}–{n.f(f'extl_{lay}_hi')}） |")
    A("")
    A("外部疾病在 V5 中已被查看，因此以上只是内部与探索性外部支持；本轮没有真正未参与设计的新外部来源，不宣称已有强外推证据。"
      f"ClinVar 引文链中，{n.i('cv_pairs')} 个外部 ClinVar 关系对有 {n.i('cv_independent_verified')} 个获得条件核实且与 CTD 文献不重叠的引文、{n.i('cv_overlap_with_ctd')} 个与 CTD 文献重叠、"
      f"{n.i('cv_unknown_no_condition_verified_citation')} 个（{n.pc('cv_unknown_rate')}）保持未知；V5 中这些关系对全部无法核实。")
    A("")
    A(f"时间输入审计核实了 Wayback 存档的 CTD 关系转储头部为“{v['t_report_created'].replace('# ', '')}”；除 Mondo 未被使用外，时间任务的全部模型输入均为截止日前的版本（2023Q4 关系与 PMID、HPO 2023-10-09、GO 2023-10-09、Reactome 86、STRING v12.0、HGNC 2023-10-01）。"
      f"V5 的时间切分评价的是截止日已存在疾病的新关系（X2），不是新疾病冷启动。在 {n.i('t_n')} 个疾病上，以 2023Q4 关系为种子的桥接传播 Recall@50 为 {n.f('t_rwr_equal_mix0.3_recall50')}，"
      f"相对 2023 年度先验的差为 {n.ci('tc_rwr_equal_mix0.3__minus__degree_prior_2023')}，"
      "新关系本身明显偏向研究充分的基因，传播没有稳定超过度先验；来源组种子在该任务中同样无收益。")
    A("")
    A(f"表达支持方面，银屑病 GSE13355 配对样本（{n.i('ex_GSE13355_pairs')} 对）中 V6 主评分器前 20 基因的 |t| 富集 AUC 为 {n.f('ex_GSE13355_lesional_vs_uninvolved_paired_v6_main_lightgbm_top20_auc')}"
      f"（单侧 p {n.f('ex_GSE13355_lesional_vs_uninvolved_paired_v6_main_lightgbm_top20_p')}），匹配随机集为 {n.f('ex_GSE13355_lesional_vs_uninvolved_paired_matched_random_top20_auc')}；"
      "但在第二个银屑病队列 GSE30999 与两个溃疡性结肠炎队列中没有复现，且未做多重比较校正。表达差异只提供相关性支持，不证明致病因果；GTEx 组织表达是 V6 的基因特征之一，属于相关来源。")
    A("")
    A(f"KDGene 在合法冷验证下训练 {n.i('kd_epochs')} 个 epoch，训练损失由 {n.f('kd_loss_first')} 持续下降到 {n.f('kd_loss_last')}，冷验证 Recall@10 在第 {n.i('kd_best_epoch')} 个 epoch 达到 {n.f('kd_best_val',4)} 后在 0.032–0.038 之间波动；"
      f"以该检查点评分第 0 折测试疾病得 {n.f('kd_test_R',4)}。全部 848 个冷验证疾病都有本体边（{n.i('kd_conn_pheno')} 个有表型边），因此这一结果不是输入断连造成的。")
    A("")
    A("### 4.7 资源与复现")
    A("")
    A(f"V6 共登记 {n.i('n_runs_v6')} 次运行（全部完成，{n.i('n_hash_ok')} 个结果文件内容哈希校验通过），其中正式五折运行 {n.i('n_formal')} 次、学习型拟合 {n.i('n_formal_learned')} 次，低于方案的 45 次核心拟合预算。"
      f"单次 LightGBM 拟合中位 {n.f('res_lightgbm_median_s',1)} 秒、峰值常驻内存 {n.f('res_lightgbm_rss_gib',1)} GiB；注意力残差中位 {n.f('res_attention_median_s',1)} 秒、GPU 保留显存 {n.i('res_attention_gpu_mib')} MiB。"
      f"登记之后另有 {n.i('n_ablation_fits')} 次探索性基因特征消融拟合，单列于账本。一次 LightGBM 正式运行首次尝试在保存模型时失败（LightGBM 的 C 写入器不支持非 ASCII 路径）；该失败记录保存在 `manifests_v6/failed_attempts_v6.json`，改为 Python 序列化后重跑，指标与失败尝试一致。账本此后在重跑前自动保留失败的结果文件。"
      "全部 V6 守护测试与 V5 原有测试共 36 项通过。")
    A("")
    # ------------------------------------------------------------------ 5 discussion
    A("## 5. 讨论")
    A("")
    A("本轮最重要的结果是把性能限制的位置说清楚了。V5 的损失主要是检索容量损失：没有查询阳性进入名单时，任何排序器都无能为力。"
      "V6 在同一预算下把上界提高了约一半，而且收益集中在此前被忽略的无 HPO 疾病。这部分收益可以追溯到具体来源：文本与层级来源每百个新增候选带来的命中最多，"
      "基因 HPO 与传播则以更多候选换取覆盖。与此同时，基因 HPO 引入了训练数据库之外的疾病—基因知识，属于输入资源增强；"
      "我们因此把“数据建设与合法新来源”的贡献与“算法”的贡献分开陈述，而不把前者包装成方法创新。")
    A("")
    A("候选内评分的结论偏保守。学习型排序器在开发集上胜出，但在五折上没有稳定超过调优融合；神经残差在验证折上常能取得微小增益，却在测试上消失，其疾病分支对置换不敏感。"
      "这与 V5 的教训一致：没有机制支撑的小幅领先不会迁移。与此相对，P2 置换清楚地表明候选内评分依赖合法的疾病输入——疾病特异性来自检索通道，而不来自神经表示。")
    A("")
    A("支持线的结果把 V5 的正向剂量信号在全部五折上复现为登记检验中的显著差异，并通过保度重连说明其依赖真实网络结构。"
      "但它的作用是对已恢复候选的重排，而不是生成新候选；让传播重新生成名单会带入网络近邻中的非特异基因。"
      "来源组加权提供了一个清楚的阴性结果：权重确实改变了种子和排名，式（12）的上界也始终成立，但真实来源结构并不优于同规模的任意重加权。"
      "给重复来源降权可能同时削弱真实的疾病模块信息，这一可能性在本数据上没有被排除。")
    A("")
    A("在时间任务上，传播没有稳定超过度先验。新关系往往落在研究充分的基因上，这既是 CTD 等文献整理资源的已知偏倚，也提醒我们：时间外评价中的“恢复”需要与研究热度基线一起解读。")
    A("")
    # ------------------------------------------------------------------ 6 limitations
    A("## 6. 局限性")
    A("")
    A("- 五个外层折已被 V5 使用，V6 的五折结果是内部估计；外部标签层同样已被查看，且 GenCC 确定/强层只有 20 个疾病。本轮没有真正未参与设计的新外部来源。")
    A("- 时间任务是“已存在疾病的新关系”（X2）且为边级冻结；2.66 GB 的历史关系原始转储未重新下载，只核验了存档头部，使用的是其处理后的边表。")
    A("- 160 对双人盲审只交付了抽样表，实际人工审阅与分歧处理未完成，不能计为已完成的核查。")
    A("- ClinVar 的疾病条件核实是变异层面的唯一性规则；SCV 层面的条件—引文对应需要 XML 发布，本轮未使用，未知率为 45.4%。")
    A("- 表达支持只在一个银屑病队列中出现，且未做多重比较校正；它只能作为相关性支持。")
    A("- KDGene 的冷验证学习曲线只在第 0 折以开发预算运行；SIGN 式对照沿用 V5 档案，未在 V6 候选上重新调参。")
    A("- 疾病家族以 MeSH 深度 3 为冻结的重采样单元；深度 4 只作为敏感性报告，没有按结果更换家族定义。")
    A("- 基因功能特征与来源组权重的阴性结果只针对当前数据与当前实现，不能推广为这些信息普遍无用。")
    A("")
    # ------------------------------------------------------------------ 7 conclusion
    A("## 7. 结论")
    A("")
    A("在异构信息缺失的低资源疾病上，先恢复候选、再利用有限支持做图空间重排，是本数据上被实测支持的路径："
      "零支持候选恢复与合法新数据带来了登记检验中的显著端到端提升，支持驱动传播在固定候选上带来显著的剂量增益，并依赖真实网络结构。"
      "学习型残差排序器与证据来源组加权没有带来可检测的额外收益，已按预先规定的规则从主方法与主要主张中撤下。")
    A("")
    # ------------------------------------------------------------------ back matter
    A("## 8. 数据与代码可用性")
    A("")
    A("全部代码位于 `gpu8/src/gpu8v6/`（与 V5 的 `gpu8/src/gpu8/` 分开，V5 代码哈希与结果档案不变）。每次运行的权威结果为 `gpu8/runs_v6/<run_id>/result.json`（原子写入、内容哈希），"
      "汇总表位于 `gpu8/results_v6/`，冻结配置与登记文件位于 `gpu8/manifests_v6/`，审计与报告位于 `gpu8/reports_v6/`。"
      "稿中每个数值在 `gpu8/reports_v6/manuscript_numbers_v6.csv` 中给出来源文件。外部资源的下载记录（地址、字节数、SHA-256）位于 `gpu8/external/v6/download_log_v6.csv`。")
    A("")
    A("## 9. 伦理与数据许可")
    A("")
    A("本研究只使用公开的数据库与文献元数据，不涉及可识别个人信息或新的人体/动物实验。CTD、HPO、GO、Reactome、GTEx、STRING、HGNC、Mondo、GenCC 与 ClinVar 按各自许可使用并在文中引用。")
    A("")
    A("## 10. AI 辅助说明")
    A("")
    A("本稿的代码实现、实验运行与文字整理由 AI 编程助手在研究者授权下完成；所有数值由代码从结果文件生成并可逐项追溯。作者应在投稿前核对全文并承担责任。")
    A("")
    # ------------------------------------------------------------------ references
    A("## 参考文献")
    A("")
    refs = [
        "Davis AP, Wiegers TC, Sciaky D, et al. Comparative Toxicogenomics Database (CTD): update 2025. Nucleic Acids Research. 2025;53:D1328-D1334.",
        "Gargano MA, Matentzoglu N, Coleman B, et al. The Human Phenotype Ontology in 2024: phenotypes around the world. Nucleic Acids Research. 2024;52:D1333-D1346.",
        "Schriml LM, Munro JB, Schor M, et al. The Human Disease Ontology 2022 update. Nucleic Acids Research. 2022;50:D1255-D1261.",
        "Szklarczyk D, Kirsch R, Koutrouli M, et al. The STRING database in 2023. Nucleic Acids Research. 2023;51:D638-D646.",
        "Seal RL, Braschi B, Gray K, et al. Genenames.org: the HGNC resources in 2023. Nucleic Acids Research. 2023;51:D1003-D1009.",
        "Vasilevsky NA, Matentzoglu NA, Toro S, et al. Mondo: unifying diseases for the world, by the world. medRxiv. 2022.",
        "DiStefano MT, Goehringer S, Babb L, et al. The Gene Curation Coalition: a global effort to harmonize gene-disease evidence resources. Genetics in Medicine. 2022;24:1732-1742.",
        "Landrum MJ, Chitipiralla S, Brown GR, et al. ClinVar: improvements to accessing data. Nucleic Acids Research. 2020;48:D835-D844.",
        "Wang X, Yang K, Jia T, et al. KDGene: knowledge graph completion for disease gene prediction using interactional tensor decomposition. Briefings in Bioinformatics. 2024;25:bbae161.",
        "Saarinen H, Goldsmith M, Wang RS, Loscalzo J, Maniscalco S. Disease gene prioritization with quantum walks. Bioinformatics. 2024;40:btae513.",
        "Alsentzer E, Li MM, Kobren SN, et al. Few shot learning for phenotype-driven diagnosis of patients with rare genetic diseases. npj Digital Medicine. 2025;8:380.",
        "Middleton L, Melas I, Vasavda C, et al. Phenome-wide identification of therapeutic genetic targets, leveraging knowledge graphs, graph neural networks, and UK Biobank data. Science Advances. 2024;10:eadj1424.",
        "Wu F, Souza A, Zhang T, Fifty C, Yu T, Weinberger K. Simplifying graph convolutional networks. ICML. 2019:6861-6871.",
        "Frasca F, Rossi E, Eynard D, Chamberlain B, Bronstein M, Monti F. SIGN: scalable inception graph neural networks. arXiv:2004.11198. 2020.",
        "Bertinetto L, Henriques JF, Torr PHS, Vedaldi A. Meta-learning with differentiable closed-form solvers. ICLR. 2019.",
        "Ghiassian SD, Menche J, Barabási AL. A DIseAse MOdule Detection (DIAMOnD) algorithm derived from a systematic analysis of connectivity patterns of disease proteins in the human interactome. PLOS Computational Biology. 2015;11:e1004120.",
        "Köhler S, Bauer S, Horn D, Robinson PN. Walking the interactome for prioritization of candidate disease genes. American Journal of Human Genetics. 2008;82:949-958.",
        "Aerts S, Lambrechts D, Maity S, et al. Gene prioritization through genomic data fusion. Nature Biotechnology. 2006;24:537-544.",
        "Vanunu O, Magger O, Ruppin E, Shlomi T, Sharan R. Associating genes and protein complexes with disease via network propagation. PLOS Computational Biology. 2010;6:e1000641.",
        "Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. NeurIPS. 2017:3146-3154.",
        "Ritchie ME, Phipson B, Wu D, et al. limma powers differential expression analyses for RNA-sequencing and microarray studies. Nucleic Acids Research. 2015;43:e47.",
        "Gudjonsson JE, Ding J, Johnston A, et al. Assessment of the psoriatic transcriptome in a large sample: additional regulated genes and comparisons with in vitro models. Journal of Investigative Dermatology. 2010;130:1829-1840.",
        "Suárez-Fariñas M, Li K, Fuentes-Duculan J, Hayden K, Brodmerkel C, Krueger JG. Expanding the psoriasis disease profile: interrogation of the skin and serum of patients with moderate-to-severe psoriasis. Journal of Investigative Dermatology. 2012;132:2552-2564.",
        "Rendle S, Freudenthaler C, Gantner Z, Schmidt-Thieme L. BPR: Bayesian personalized ranking from implicit feedback. UAI. 2009.",
        "Loshchilov I, Hutter F. Decoupled weight decay regularization. ICLR. 2019.",
        "Holm S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics. 1979;6:65-70.",
        "Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman and Hall. 1993.",
        "The Gene Ontology Consortium. The Gene Ontology knowledgebase in 2023. Genetics. 2023;224:iyad031.",
        "Milacic M, Beavers D, Conley P, et al. The Reactome Pathway Knowledgebase 2024. Nucleic Acids Research. 2024;52:D672-D678.",
        "GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. Science. 2020;369:1318-1330.",
        "Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research. 2010;11:2079-2107.",
        "Cormack GV, Clarke CLA, Büttcher S. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR. 2009:758-759.",
        "Köhler S, Schulz MH, Krawitz P, et al. Clinical diagnostics in human genetics with semantic similarity searches in ontologies. American Journal of Human Genetics. 2009;85:457-464.",
        "Smedley D, Jacobsen JOB, Jäger M, et al. Next-generation diagnostics and disease-gene discovery with the Exomiser. Nature Protocols. 2015;10:2004-2015.",
        "Resnik P. Using information content to evaluate semantic similarity in a taxonomy. IJCAI. 1995:448-453.",
        "Maslov S, Sneppen K. Specificity and stability in topology of protein networks. Science. 2002;296:910-913.",
        "Cao Z, Qin T, Liu TY, Tsai MF, Li H. Learning to rank: from pairwise approach to listwise approach. ICML. 2007:129-136.",
        "Burges CJC. From RankNet to LambdaRank to LambdaMART: an overview. Microsoft Research Technical Report MSR-TR-2010-82. 2010.",
        "Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning. 2nd ed. Springer. 2009.",
        "Salton G, Buckley C. Term-weighting approaches in automatic text retrieval. Information Processing & Management. 1988;24:513-523.",
    ]
    for i, r in enumerate(refs, 1):
        A(f"{i}. {r}")
    A("")
    # ------------------------------------------------------------------ figures
    A("## 图与图注")
    A("")
    figs = [
        ("fig1_v6_framework.png", "图 1　V6 框架。可见信息审计决定分流；零支持候选恢复在固定 500 预算下融合多个合法来源；候选内评分以调优融合为基准；有支持时以显露支持为种子作图传播；来源组种子与置换检验用于检验机制。"),
        ("fig2_availability_bound.png", "图 2　数据可见性与可达上界。(a) 按已知阳性数分层的 HPO 可用性类别；(b) V5 与 V6 的 R = U·η 分解；(c) 疾病结局比例（疾病计数，不是宏平均召回）。"),
        ("fig3_retrieval.png", "图 3　零支持候选恢复。(a) 开发集逐项接入的覆盖率，按直接证据、仅传播与先验回退分解；(b) 测试折上各来源的单源覆盖与留一贡献（同一 500 预算）。"),
        ("fig4_scoring.png", "图 4　候选内评分。(a) 五折端到端 Recall@10；(b) 按可见信息分层（有 HPO / 无 HPO）。"),
        ("fig5_support_dose.png", "图 5　支持剂量。(a) S-fixed 下的等权传播、联合与来源组种子；虚线为零支持语义评分；(b) 联合评分在固定池与自适应池下的 Recall@10（自适应池前十与全背景相同）及自适应池覆盖率。"),
        ("fig6_evidence_mechanism.png", "图 6　来源组权重的机制诊断。(a) 种子、前十名单与指标发生改变的 episode 比例；(b) Δseed（组权 − 等权）与 Γgrp（真组 − 打乱组）及其 95% CI。"),
        ("fig7_permutations.png", "图 7　疾病置换。(a) V5 b9 在 k = 0 下的 P1/P2/P3（第 0 折）；(b) V6 评分器的 P1/P2/P3（五折，远供体；只有注意力残差有神经疾病分支）。"),
        ("fig8_external_temporal_kdgene.png", "图 8　外部、时间与比较器预算。(a) 外部标签层的 Recall@10 与 95% CI；(b) 时间任务 X2 的 Recall@50；(c) KDGene 的冷验证 Recall@10 与训练损失。"),
    ]
    for f_, cap in figs:
        A(f"![{cap}](figures/{f_})")
        A("")
        A(cap)
        A("")
    return "\n".join(L), n.used


def main() -> int:
    v = json.loads((paths6.REPORTS / "manuscript_numbers_v6.json").read_text(encoding="utf-8"))
    text, used = build(v)
    from .appendix6 import appendix
    text = text + "\n" + appendix(v)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / f"{TITLE}_V6_20260923.md"
    md.write_text(text, encoding="utf-8")
    num = paths6.REPORTS / "manuscript_numbers_v6.csv"
    import pandas as pd
    tab = pd.read_csv(num)
    tab["quoted_in_manuscript"] = tab["key"].isin(used)
    tab.to_csv(num, index=False, encoding="utf-8-sig")
    print(json.dumps({"markdown": str(md), "chars": len(text), "numbers_quoted": len(used)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

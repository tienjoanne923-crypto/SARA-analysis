"""SARA supplementary material (Chinese): Notes S1-S6, Tables S1-S10 and the legends of Figures S1-S8.

Every number is read from the saved results and formatted by ``fmt9`` (traced).  Citations use ``[@key]`` and
are numbered separately for the supplement.  Public wording only (``publabel9``); the two designs that did not
enter SARA are described in Note S4 and Table S8.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths8
from .fmt9 import TRACE, ci, cin, i0, num, pc, sci, sgn
from .manuscript9 import CASE_ZH, DATE, PUB, TITLE, _meta, _row, acceptance_checks, table_md
from .numbers9 import FIG, ROOT, SDI, numbers9
from .publabel9 import pub, relabel

SUPP_STEM = f"SARA_补充材料_{DATE}"
SDIR = FIG / "source_data_internal"

FAM_PUB = {"V6 检索通道": "检索通道", "候选融合": "融合名次", "语义": "语义", "表达": "表达", "基因可用性": "基因可用性",
           "疾病描述": "疾病描述"}
SEM_PUB = {"教师 s_F（分段文本）": "语义检索 $s_F$（分段文本）", "V7.1 核心 r": "多模态语义核心 $r$",
           "教师自身序列分数": "语义模型序列分数", "无学习 SapBERT 功能相似": "SapBERT 零训练相似度", "线性双塔": "线性双塔",
           "ProtST 零样本": "ProtST 零样本文本—蛋白检索", "教师 s_F（V7.1 短文本）": "语义检索 $s_F$（截断短文本）",
           "教师 s_F（二级掩蔽文本）": "语义检索 $s_F$（二级掩蔽文本）"}
E4_PUB = [("e4_layer1", "训练可见来源排除", "训练可见来源排除（5 外层均值）"),
          ("e4_layer1_random", "其匹配随机", None),
          ("e4_layer2", "评价条件化删除*", "评价条件化删除*"),
          ("e4_layer2_random", "其匹配随机*", "其匹配随机*"),
          ("e4_blocks", "证据块隔离", "证据块隔离（GO 部分）"),
          ("e4_blocks_random", "其匹配随机", None),
          ("e4_experimental", "仅实验类证据", "仅实验类证据"),
          ("e4_no_tas_nas", "去除 TAS/NAS", "去 TAS/NAS"),
          ("e4_no_mutant", "去除突变/遗传相互作用类", "去突变/遗传相互作用"),
          ("e4_rescore_no_ic", "去除 IC（重打分）", "去 IC"),
          ("e4_rescore_iea_only", "仅 IEA（重打分）", "仅 IEA")]
E5_PUB = [("e5_none", "删除 GO，不补位"), ("e5_mlp", "两层补位模型"), ("e5_linear", "线性补位模型"),
          ("e5_nn", "序列近邻转移"), ("e5_teacher_sS", "语义模型序列投影"), ("e5_mlp_shuffled", "两层补位（长度层内打乱配对）"),
          ("e5_protst", "ProtST 输入补位（开放知识对照）"), ("natural_student_on", "自然输入中开启补位（GO 未删除）")]
SUP_PUB = {"joint_v8_w0.25": "SARA 语义 + RWR", "joint_v6": "基线语义 + RWR", "qwalk_v6list": "量子游走（检索前 500）",
           "qwalk_v8list": "量子游走（SARA 候选）", "rwr_v6list": "RWR（检索前 500）", "rwr_v8list": "RWR（SARA 候选）",
           "sem_v6": "基线语义单独", "sem_v8": "SARA 语义单独", "rwr": "RWR"}
EXPR_SET = [("v8_deployed_s42", "SARA（种子 42）"), ("v8_deployed_s52", "SARA（种子 52）"), ("v8_deployed_s62", "SARA（种子 62）"),
            ("v71_direct_strict_s42", "多模态直接融合"), ("channel_lgb_v6_s42", "检索通道 LightGBM"),
            ("F7_v6", "固定权重融合（检索前 500）"), ("matched_random", "匹配随机基因集")]
CONTRAST_PUB = {"lesional_vs_uninvolved_paired": "病灶/非病灶（配对）", "lesional_vs_control": "病灶/对照"}
HIST_PUB = {"ctd_relations": "CTD 疾病—基因关联", "ctd_evidence": "CTD 关联的文献证据", "ctd_vocabulary": "CTD 疾病词表",
            "ctd_hierarchy": "CTD 疾病层级", "hpo_obo": "HPO 本体", "hpoa": "HPO 疾病注释",
            "hpo_phenotypes_mapped": "HPO 注释的疾病映射", "genes_to_phenotype": "基因—表型注释", "doid": "疾病本体",
            "go_obo": "GO 本体", "goa_human": "GOA 人类注释", "hgnc": "HGNC 基因命名", "gtex_v8": "GTEx v8 组织表达"}
LATER_PUB = {"gene background (19,861) and HGNC-based gene index": "基因背景（19,861）与基因索引",
             "protein sequences (MANE 2025)": "MANE 代表蛋白序列", "STRING v12.0 physical network": "STRING v12.0 物理互作网络",
             "disease equivalence groups and families": "疾病等价组与家族划分",
             "V8.1 hyper-parameters and development choices": "SARA 的超参数与开发集选择",
             "SapBERT / ESM-2 checkpoints": "SapBERT / ESM-2 预训练检查点"}
ROLE_PUB = {"entity index": "实体索引", "static gene attribute": "静态基因属性", "PPI (not later than the cut-off)": "互作网络",
            "split units only": "仅用于划分", "frozen settings": "冻结设置", "frozen encoders": "冻结编码器"}


def J(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sd(stem: str, panel: str) -> pd.DataFrame:
    return pd.read_csv(SDIR / f"{stem}_{panel}.csv", encoding="utf-8-sig")


def _sel(j: int) -> dict:
    return J(paths8.DEV / f"fold{j}" / "selection.json")


def _teacher(j: int) -> dict:
    return J(paths8.RUNS / f"V8T__main__f{j}_outer__background__s42" / "result.json")


def _cfg(name: str) -> str:
    a, b = name.replace("A_nl", "").split("_mc")
    return f"{a} 叶 / 最小样本 {b}"


# ============================================================================ notes
def note_s1(R, N) -> str:
    tx = J(paths8.AUDIT / "text_segments_l1.json")
    lr = R["acceptance"]["1_label_removal"]
    return "\n\n".join([
        "## 补充说明 S1　数据处理与信息边界",
        "基因背景、疾病与阳性关联的来源见正文方法 4.1 与表 1，以下补充三点处理细节。",
        f"**疾病文本。**疾病文本取自名称、定义与至多 8 个同义词，编码前以预先固定的 HGNC 批准符号词典掩蔽基因符号。按句界优先装段后，"
        f"{i0(tx['n_diseases'])} 个疾病共得到 {i0(tx['segments_total'])} 段，单个疾病最多 {tx['max_segments']} 段；完整文本的词片数中位数为 "
        f"{num(tx['tokens_full_median'], 0)}，第 95 百分位数为 {num(tx['tokens_full_p95'], 0)}，装段过程中没有片段被分词器截断"
        f"（{i0(tx['segment_tokenizer_truncations'])} 次）。只有一段的疾病，其向量与不分段编码逐元素相同。",
        "**表型注释。**疾病 HPO 注释经严格等价映射挂接，只保留表型异常分支、非 NOT 限定且深度大于 2 的术语；基因—表型注释的每一行都"
        "带有来源疾病，在按来源疾病排除当前上下文的全部留出疾病及其严格等价组之后才聚合到基因，来源无法映射的行不进入主分析。",
        "**信息边界。**每个上下文的参考集合剔除任一留出疾病的严格等价组；检索通道、语义模型、候选名单与排序器的全部输入均按疾病的"
        f"角色（训练、验证、测试、开发、外部留出）生成，而不读取测试标签。在第 {lr['fold']} 折的 {i0(lr['n_checked_diseases'])} 个测试疾病"
        f"上移除测试标签后重新生成输入，检索顺序逐元素一致、候选名单完全相同，SARA 分数的最大差为 {sci(lr['deployed_score_maxabs'])}。",
    ])


def note_s2(R, N) -> str:
    rv = J(paths8.GPU8_ROOT / "manifests_v6" / "retriever_v6_frozen.json")
    sv = J(paths8.GPU8_ROOT / "manifests_v6" / "scorer_v6_frozen.json")
    w, fw = rv["weights"], sv["weights"]
    tr = _teacher(0)
    tc = tr["config"]["train"]
    ctl = J(paths8.MANIFESTS / "dev_teacher_controls_v8.json")["per_context"]
    n_bg = sum(1 for v in ctl.values() if v["controls"] == "background")
    gap = max(abs(_sel(j)["w"]["coverage"][f"aware_w{x}"] - _sel(j)["w"]["coverage"][f"plain_w{x}"])
              for j in range(5) for x in (0.3, 0.5, 0.7, 0.9))
    return "\n\n".join([
        "## 补充说明 S2　检索通道、语义模型与候选生成的实现细节",
        f"**检索通道。**七个来源在全背景上按名次作倒数名次融合（$K_0$ = 60），缺失来源被掩蔽并重新分配权重。融合方式与权重只在开发集"
        f"（{i0(sv['n_dev_disease_contexts'])} 个“疾病 × 外层上下文”）上选择：先在 0.25 步长的单纯形网格与两种融合方式上评价 "
        f"{i0(rv['n_grid_configs'])} 个配置，再做 {i0(rv['n_refine_configs'])} 个 0.05 步长的邻域细化，目标为候选预算 500 下的宏平均 U@10。"
        f"冻结的权重为：层级 {num(w['hier'], 2)}、本体 {num(w['onto'], 2)}、文本 {num(w['text'], 2)}、基因—表型 {num(w['gene_hpo'], 2)}、"
        f"表型传播 {num(w['rwr_pheno'], 2)}；表型迁移与兼容来源的融合权重为 0，前者的信号经由以它为种子的表型传播进入，二者的分数、名次"
        f"与可用性仍作为排序特征输入。固定权重融合对照的权重同样在开发集上选择（层级 {num(fw['hier'], 2)}、本体 {num(fw['onto'], 2)}、"
        f"文本 {num(fw['text'], 2)}、基因—表型 {num(fw['gene_hpo'], 2)}）。",
        "**语义模型。**疾病 HPO 表示为 HPO 术语 SapBERT 向量按注释频率的加权平均（缺失频率取 1），没有 HPO 的疾病不生成任何虚构表型；"
        "GO 功能锚为在背景基因中注释 5–500 个基因的 GO 术语（is_a/part_of 传递闭包，去除三个根术语），同一基因的多个正锚互不作为负例。"
        f"训练先做 {tc['anchor_steps']} 步锚更新，再联合训练：AdamW（学习率 {tc['lr']}、权重衰减 {tc['weight_decay']}），每批 {tc['batch']} 个"
        f"疾病、锚批 {tc['anchor_batch']} 个基因，梯度范数裁剪 {tc['grad_clip']}，至多 {tc['max_epochs']} 轮，耐心 {tc['patience']}。未标注对照"
        f"集合在每个外层上下文按开发集 $s_F$ Recall@10 在“参考关联基因”与“全背景”之间选择，{n_bg} 个外层均选择全背景（表 S1，图 S1）。",
        "**候选生成。**每个外层上下文的 RRF 权重 $w$ 按开发集的阳性覆盖@500 选择（图 S2a）。在四个候选权重上，普通 RRF 与可用性感知 RRF "
        f"的开发覆盖之差不超过 {num(gap, 4)}，二者的差别集中在缺少语义的阳性上（正文图 4c）。",
    ])


def note_s3(R, N) -> str:
    cf = [_cfg(_sel(j)["ab"]["best_A"]) for j in range(5)]
    fi = pd.read_parquet(PUB / "feature_importance.parquet")
    nmod = fi.groupby(["seed", "fold"]).ngroups
    return "\n\n".join([
        "## 补充说明 S3　SARA 排序器的特征与配置选择",
        f"表 S2 列出 SARA 排序器的 49 个特征及其在 {nmod} 个模型（5 个外层 × 3 个种子）中的平均增益占比。各外层选定的配置依次为 "
        + "、".join(cf) + "（表 S1，图 S2b）。排序器的训练数据是三个内块的袋外候选与特征，验证数据是外层 $F_{j+1}$ 的候选；"
        "语义与表达特征缺失时以 NaN 输入，基因类型与各模态可用性以显式的指示特征输入，因此模型可以区分“分数低”与“信息不可用”。",
    ])


def note_s4(R, N) -> str:
    con = R["e2"]["contrasts"]
    sels = [_sel(j) for j in range(5)]
    bc = [max(s["summary"]["corrector"], key=lambda r: r["dev_recall10"])["dev_recall10"] for s in sels]
    ba = [s["ab"]["best_A_dev_recall10"] for s in sels]
    gaps = [a - c for a, c in zip(ba, bc)]
    assert all(g > 0.005 for g in gaps), "a corrector fell within the development margin"
    st = J(PUB / "dev_misc.json")["student"]["per_context"]
    gains, cos_t, cos_p = [], [], []
    for v in st.values():
        for r in v["rows"]:
            if r["config"] == "none":
                continue
            gains.append(r["gain"])
            cos_t.append(r["gene_level_masked"]["mean_cos_true"])
            cos_p.append(r["gene_level_masked"]["mean_pairwise_cos"])
    assert max(gains) < 0, "an imputation configuration passed the development check"
    npar = {}
    for p in sorted(paths8.RUNS.glob("V8S__devmask__f0_outer__*__s42")):
        kind = p.name.split("__")[3].split("_")[0]
        npar[kind] = J(p / "result.json")["fit"]["n_params"]
    v = R["variants"]
    none_pool = v["e5_none"]["deployed"]["mask_test_pool_micro"]
    others = [v[t]["deployed"]["mask_test_pool_micro"] for t, _ in E5_PUB[1:7]]
    assert max(others) <= none_pool + 1e-12
    return "\n\n".join([
        "## 补充说明 S4　未进入最终模型的两种设计",
        "在确定 SARA 之前，我们还考察了两种以提高冷启动基因恢复为目标的设计。二者都按与 SARA 相同的袋外规则在同一候选名单上训练与"
        "选择，并在开发集上与 SARA 排序器比较；由于未能带来增益，它们没有进入最终模型。这里完整报告其设计与结果（图 S8，表 S8）。",
        "**门控冷平衡纠偏器。**该设计在基础排序器 $B$ 之上叠加一个受语义可用性门控的残差：",
        "$$s(d,g)=B(d,g)+a(d,g)\\,r_\\theta(x_{dg}),$$",
        "其中 $a$ 为语义可用门，$x_{dg}$ 为与 SARA 排序器相同的 49 个特征。残差以正例—未标注成对损失训练[@elkan2008; @bekker2020]，参考"
        "路径冷正例的权重为 $1+\\beta c_g$（$c_g$ 为冷标记），并对残差施加系数为 $\\lambda$ 的二次惩罚；网格为 $\\beta\\in\\{0,1,2\\}$ × "
        "$\\lambda\\in\\{0.03,0.1\\}$，以自定义目标的 LightGBM 实现（15 叶，至多 200 轮，以 $F_{j+1}$ 的 Recall@10 选择轮次）。$a=0$ 时 "
        "$s\\equiv B$，因此该设计在语义不可用时精确回退到基础排序器。选择规则事先固定：只有开发 Recall@10 不低于最佳 SARA 排序器配置减 "
        f"0.005 的纠偏器才会被考虑。在全部五个外层上，最佳纠偏器的开发 Recall@10（{num(min(bc))}–{num(max(bc))}）都比最佳 SARA 排序器配置"
        f"（{num(min(ba))}–{num(max(ba))}）低 {num(min(gaps))} 以上（图 S8a–b，表 S1）。测试集上的比较与此一致：β = 0、1、2 的纠偏器相对 SARA "
        f"的 Recall@10 差值分别为 {ci(con['corrector_beta0']['recall10_vs_deployed'])}、{ci(con['corrector_beta1']['recall10_vs_deployed'])}与 "
        f"{ci(con['corrector_beta2']['recall10_vs_deployed'])}，而冷恢复 $R^c$@10 与 SARA 相当"
        f"（{cin(con['corrector_beta0']['cold_vs_deployed'])}；β = 0；图 S8c）。",
        f"**序列补位模型。**该设计以冻结的 ESM-2 35M 序列表示为输入，经两层感知机（480→256→128，GELU；{i0(npar['mlp'])} 个参数）或线性"
        f"映射（{i0(npar['linear'])} 个参数）回归同一上下文语义模型的功能表示 $z^F_g$（余弦蒸馏[@hinton2015]），只在基因没有观察到的 GO "
        "时补位。采用前须"
        "在开发屏蔽集上满足三个条件：屏蔽基因阳性进入候选的比例提高、长度层内打乱序列配对后提高消失、总体开发覆盖下降不超过 0.005。"
        f"四种配置在五个外层上的变化全部为负（{sgn(min(gains))} 至 {sgn(max(gains))}；图 S8d），尽管补位向量与真实功能向量的平均余弦"
        f"（{num(min(cos_t), 2)}–{num(max(cos_t), 2)}）明显高于两两平均余弦（{num(min(cos_p), 2)}–{num(max(cos_p), 2)}；图 S8e），因此补位"
        f"模型未被采用。在测试屏蔽集上，六种补位方式都没有使屏蔽基因阳性进入候选的比例超过不补位的 {pc(none_pool)}"
        f"（{pc(min(others))}–{pc(max(others))}；图 S8f–g，表 S8）。",
        "这两组结果界定了 SARA 的设计：在当前数据下，序列表示还不能替代观察到的功能注释，而对冷启动正例加权的残差也未能在不损失总体"
        "性能的前提下提高冷恢复。因此 SARA 在语义不可用时直接回退到检索证据，并把缺失以 NaN 明示给排序器，而不以预测值填补。",
    ])


def note_s5(R, N) -> str:
    hc = R["historical_contrasts"]
    hp = N["x"]["hp"]
    x2, x1 = hp["x2"], hp["x1"]
    t = "新关联任务（{n} 个疾病、{p} 对）上 SARA、检索通道 LightGBM、基础排序器与单独语义检索的 Recall@10 分别为 {a}、{b}、{c} 与 {d}"
    return "\n\n".join([
        "## 补充说明 S5　时间外推的设计与结果",
        "时间外推以 2023-12-31 为截止日。截止前可得的输入及其日期与哈希见表 S9；晚于截止日而无法回溯的输入（基因背景与基因索引、代表"
        "蛋白序列、疾病等价组与家族划分、SARA 的超参数与冻结编码器）逐项列于表 S9 下方。截止前的输入重建了疾病文本、HPO 空间、来源排除"
        "的基因—表型注释、TF-IDF、本体与 GO 功能锚，并以不依赖外层上下文的统一设置（取五个外层最常见的选择）重训全部模型。新关联任务"
        "评价截止时已有关联的疾病在截止后新增的关联（排序时移除截止前已知的基因），新疾病任务评价截止时尚无任何关联的疾病。",
        t.format(n=i0(hc["x2_deployed_vs_channel_lgb_v6"]["n"]), p=i0(x2["pairs"]),
                 a=num(hc["x2_deployed_vs_channel_lgb_v6"]["a"]), b=num(hc["x2_deployed_vs_channel_lgb_v6"]["b"]),
                 c=num(hc["x2_deployed_vs_B"]["b"]), d=num(hc["x2_deployed_vs_standalone_sF"]["b"])) +
        f"；新疾病任务（{i0(hc['x1_deployed_vs_channel_lgb_v6']['n'])} 个疾病、{i0(x1['pairs'])} 对）上分别为 "
        f"{num(hc['x1_deployed_vs_channel_lgb_v6']['a'])}、{num(hc['x1_deployed_vs_channel_lgb_v6']['b'])}、"
        f"{num(hc['x1_deployed_vs_B']['b'])} 与 {num(hc['x1_deployed_vs_standalone_sF']['b'])}。SARA 与检索通道 LightGBM 在两个任务上相当"
        f"（{cin(hc['x2_deployed_vs_channel_lgb_v6'])}；{cin(hc['x1_deployed_vs_channel_lgb_v6'])}），在新疾病任务上显著优于单独语义检索"
        f"（{cin(hc['x1_deployed_vs_standalone_sF'])}）。逐对比较中，新关联任务的 {i0(x2['pairs'])} 对里 SARA 名次更靠前 "
        f"{i0(x2['deployed_better'])} 对、检索通道 LightGBM 更靠前 {i0(x2['channel_better'])} 对、持平 {i0(x2['tie'])} 对（其中 "
        f"{i0(x2['both_unlisted'])} 对在两者中均未入选）；SARA 带入候选 {i0(x2['listed_deployed'])} 对，检索通道 LightGBM "
        f"{i0(x2['listed_channel'])} 对（正文图 9c、f）。",
    ])


def note_s6(R, N) -> str:
    h = R["main"]["hierarchy"]
    return "\n\n".join([
        "## 补充说明 S6　统计分析与计算资源",
        "主要比较——SARA 相对检索通道 LightGBM 的端到端 Recall@10 差值——在任何测试评价之前确定，其 95% 区间下界大于 0"
        f"（{cin(h['h1_vs_channel_lgb_v6'])}）。与多模态直接融合的比较预先设定了 −0.005 的非劣界：观察到的差值为 "
        f"{cin(h['h2_vs_v71_direct_strict'], d=4)}，区间下界略低于该界，因此在这一界值下非劣性未能确立；正文据点估计接近 0 且区间包含 0，"
        f"将二者描述为“性能相当”。冷恢复相对多模态直接融合的差值（{cin(h['h3_cold_vs_v71_direct_strict'], d=4)}）只作描述。其余比较与"
        "全部分层、解释分析均只报告估计值与区间，不作显著性判定，因此不做多重比较校正。开发集上 5 × 864 次评价是相关的重复，只用于"
        "选择，不当作独立样本。",
        "所有区间为按疾病家族重抽样的 bootstrap 百分位区间：主要比较 10,000 次（种子 20260923），描述性区间 1,000–2,000 次；关系级比例以"
        "家族为簇对分子与分母同时重抽样。逐疾病的指标先在三个种子间平均，再在疾病间宏平均，从不取“任一种子命中”的并集。",
        f"**计算资源。**全部 {i0(N['runs'])} 次拟合与评价运行均以内容哈希记账，有计时的运行合计约 {num(N['timed_h'], 1)} 小时；单次运行的"
        f"峰值显存 {num(N['gpu_peak'] / 1024, 2)} GiB、峰值内存 {num(N['rss_peak'] / 1024, 1)} GiB（图 S6）。",
    ])


# ============================================================================ tables
def table_s1(R, N) -> str:
    dis = pd.read_parquet(PUB / "diseases.parquet")
    t = dis[dis["role"] == "test_fold"]
    ctl = J(paths8.MANIFESTS / "dev_teacher_controls_v8.json")["per_context"]
    rows_a, rows_b = [], []
    for j in range(5):
        s = _sel(j)
        tt = t[t["fold"] == j]
        tr = _teacher(j)
        bc = max(s["summary"]["corrector"], key=lambda r: r["dev_recall10"])["dev_recall10"]
        rows_a.append([f"外层 {j}", f"$F_{j}$ / $F_{(j + 1) % 5}$", i0(len(tt)), i0(tt["n_rel"].sum()),
                       i0(tt["family_group"].nunique()), i0(tr["fit"]["n_reference"])])
        rows_b.append([f"外层 {j}", "全背景" if ctl[f"f{j}_outer"]["controls"] == "background" else "参考关联基因",
                       num(s["w"]["w"], 1), str(tr["fit"]["best_epoch"]), num(s["summary"]["B_dev"]["dev_recall10"]), num(bc),
                       num(s["ab"]["best_A_dev_recall10"]), _cfg(s["ab"]["best_A"])])
    return ("表 S1　外层上下文的规模与开发集选择（种子 42；选择只使用开发集与同一上下文的 $F_{j+1}$）\n\n（a）规模\n\n" +
            table_md(["上下文", "测试 / 验证折", "测试疾病", "测试关联", "测试家族", "参考疾病"], rows_a) +
            "\n\n（b）开发集选择（R@10 为开发集上的值）\n\n" +
            table_md(["上下文", "未标注对照", "$w_j$", "语义模型最佳轮次", "基础排序器", "最佳纠偏器", "SARA 排序器",
                      "选定配置"], rows_b))


def table_s2(R, N) -> str:
    from .pubdata8 import FEAT_ZH, feat_family
    fi = pd.read_parquet(PUB / "feature_importance.parquet")
    agg = fi.groupby("feature")["gain_share"].agg(["mean", "std"])
    order = list(fi["feature"].drop_duplicates())
    fam_order = ["V6 检索通道", "疾病描述", "候选融合", "语义", "表达", "基因可用性"]
    rows = []
    for fam in fam_order:
        for f in [x for x in order if feat_family(x) == fam]:
            rows.append([FAM_PUB[fam], f"`{f}`", relabel(FEAT_ZH.get(f, f)), f"{pc(agg.loc[f, 'mean'], 2)} ± {pc(agg.loc[f, 'std'], 2)}"])
    return ("表 S2　SARA 排序器的 49 个特征（增益占比为 15 个模型的均值 ± 标准差）\n\n" +
            table_md(["特征族", "特征", "说明", "增益占比"], rows))


def table_s3(R, N) -> str:
    comp = R["comparators"]
    rows_a = [
        ["检索通道 LightGBM", "30 个检索特征", "自身检索前 500", "LambdaRank，$F_{j+1}$ 早停", "主要对照"],
        ["基础排序器", "30 个检索特征（袋外）", "SARA 候选", "同上", "同候选名单的检索基线"],
        ["固定权重融合", "检索来源分数的固定权重", "检索前 500；SARA 候选", "无训练（权重在开发集选择）", "规则基线"],
        ["多模态直接融合", "30 个检索特征 + 14 项多模态语义核心特征（缺失写 0）", "检索前 500；SARA 候选", "LambdaRank，$F_{j+1}$ 早停",
         "多模态对照"],
        ["单独语义排名", "$s_F$、核心分数 $r$、序列分数", "全背景（取前 500）", "语义模型，$s_F$ 选择检查点", "语义消融"],
        ["SapBERT 零训练相似度", "未投影的疾病—功能 SapBERT 相似度", "全背景", "无训练", "语义消融"],
        ["线性双塔", "与语义模型相同的输入，线性投影", "全背景", "对比学习，$F_{j+1}$ 选择轮次", "语义消融"],
        ["KDGene[@wang2024kdgene]", "知识图谱嵌入（实体与关系 1,000 维）", "SARA 候选", "至多 100 轮，耐心 10，$F_{j+1}$ 冷验证选择轮次",
         f"外部方法；R@10 {num(comp['kdgene']['recall10'])}"],
        ["SIGN 式图模型[@frasca2020]", "上下文视图 0–2 阶传播 + 两层解码器", "SARA 候选", "AdamW（1e-3，wd 1e-4），30 轮，$F_{j+1}$ 选择轮次",
         f"外部方法；R@10 {num(comp['sign']['recall10'])}"],
        ["ProtST 零样本[@xu2023protst]", "文本—蛋白零样本检索（只加载权重）", "全背景（蛋白编码）", "无训练", "开放知识对照"],
    ]
    a = table_md(["方法", "输入", "候选名单", "训练与检查点", "角色"], rows_a)
    rows_b = []
    for r in R["semantic"]["table"]:
        rows_b.append([SEM_PUB.get(r["method"], r["method"]), num(r["recall10"]), num(r["recall50"]), num(r["cold_recall10"]),
                       num(r["coverage500"]), ci(r["vs_sF"]) if isinstance(r.get("vs_sF"), dict) else "—"])
    b = table_md(["方法", "R@10", "R@50", "$R^c$@10", "覆盖@500", "相对 $s_F$（分段文本）"], rows_b)
    k = comp["kdgene"]["deployed_minus"]
    s = comp["sign"]["deployed_minus"]
    return ("表 S3　对照方法与全背景单独语义排名\n\n" + a + "\n\n" +
            f"注：KDGene 与 SIGN 式图模型在同一候选名单上以袋外规则重训（种子 42）；SARA 相对二者的 Recall@10 差值分别为 {ci(k)}与 {ci(s)}。"
            "\n\n（b）全背景单独语义排名（种子 42；各方法在全部背景基因上排序后取前 500 评价）\n\n" + b)


def table_s4(R, N) -> str:
    st = R["strata"]["strata"]
    zs = R["strata"]["zero_signal_s42"]
    ms = ["deployed", "channel_lgb_v6", "v71_direct_strict"]
    items = [("single_relation_recall10", "单关联疾病 R@10", num), ("multi_relation_recall10", "多关联疾病 R@10", num),
             ("omim_mapped_recall10", "有 OMIM 映射 R@10", num), ("not_omim_recall10", "无 OMIM 映射 R@10", num),
             ("burden_1", "阳性负担 1 R@10", num), ("burden_2_5", "阳性负担 2–5 R@10", num), ("burden_gt5", "阳性负担 >5 R@10", num),
             ("therapeutic_only_hit10_micro", f"治疗类关联前十命中率（{i0(st['deployed']['therapeutic_only_pairs'])} 对）", pc)]
    rows = [[lab] + [fn(st[m][k]) for m in ms] for k, lab, fn in items]
    a = table_md(["分层", "SARA", "检索通道 LightGBM", "多模态直接融合"], rows)
    sg = {r["method"]: r for r in R["subgroups"]["table"]}
    ms2 = [m for m in ("deployed", "channel_lgb_v6", "B", "v71_direct_strict") if m in sg]
    it2 = [("ncrna", "非编码 RNA"), ("nogo_protein", "无 GO 的蛋白编码"), ("sem_unavail", "语义不可用"), ("mask_test", "测试屏蔽基因")]
    rows2 = []
    for k, lab in it2:
        q = sg["deployed"][f"{k}_queries"]
        rows2.append([f"{lab}（{i0(q)} 条）"] + [f"{pc(sg[m][f'{k}_pool_micro'])} / {pc(sg[m][f'{k}_hit10_micro'])}" for m in ms2])
    b = table_md(["亚组（查询关联数）"] + ["SARA" if m == "deployed" else pub(m) for m in ms2], rows2)
    return ("表 S4　分层、缺失亚组与零信号查询（三种子平均）\n\n（a）按疾病特征分层\n\n" + a +
            "\n\n（b）缺失亚组：进入候选的比例 / 进入前十的比例（关系级微平均）\n\n" + b +
            f"\n\n注：种子 42 的 {i0(zs['queries'])} 条测试关联中，{i0(zs['no_v6_signal'])} 条没有任何检索来源信号，{i0(zs['no_semantic'])} 条"
            f"语义不可用，{i0(zs['zero_signal_both'])} 条（{pc(zs['zero_signal_share'])}）两者皆无；后者无一进入候选。")


def table_s5(R, N) -> str:
    d3 = sd("Fig3_main_results", "d")
    rows = []
    for (f, s), g in d3.groupby(["fold", "seed"]):
        a = g[g["vs"] == "channel_lgb_v6"].iloc[0]
        b = g[g["vs"] == "v71_direct_strict"].iloc[0]
        rows.append([f"外层 {f}", str(s), i0(a["n"]), ci({"delta": a["delta"], "ci_low": a["lo"], "ci_high": a["hi"]}, label=False),
                     ci({"delta": b["delta"], "ci_low": b["lo"], "ci_high": b["hi"]}, label=False)])
    a_ = table_md(["外层", "种子", "测试疾病", "SARA − 检索通道 LightGBM", "SARA − 多模态直接融合"], rows)
    v = R["variants"]
    vr = [("v6pool", "SARA 排序器用于检索前 500"), ("plain_rrf", "普通 RRF 候选（500）"), ("bridge950", "扩展候选（约 950）"),
          ("expr_off", "关闭表达特征"), ("expr_random", "随机组织表达")]
    rows_b = []
    for k, lab in vr:
        d = v[k]["deployed"]
        rows_b.append([lab, num(d["recall10"]), num(d["cold_recall10"]), pc(d["coverage500"]), pc(d["ncrna_pool_micro"]),
                       ci(d["delta_vs_main"], label=False)])
    b_ = table_md(["变体", "R@10", "$R^c$@10", "覆盖", "ncRNA 阳性入选", "相对 SARA 的 R@10 差值（95% CI）"], rows_b)
    ref = v["plain_rrf"]["deployed"]["delta_vs_main"]["b"]
    return ("表 S5　稳健性\n\n（a）15 个外层 × 种子组合的 Recall@10 差值（95% CI）\n\n" + a_ +
            f"\n\n（b）候选名单与表达特征的变体（种子 42；各变体整链重训；SARA 的 R@10 为 {num(ref)}）\n\n" + b_)


def table_s6(R, N) -> str:
    d = N["s5d"]
    rows = [[f"{r['name']}（{r['code']}）", i0(r["n"]), num(r["deployed"]), num(r["v71_direct_strict"]), num(r["channel_lgb_v6"]),
             num(r["standalone_sF"]), ci({"delta": r["delta"], "ci_low": r["lo"], "ci_high": r["hi"]}, label=False),
             ci({"delta": r["delta_v71"], "ci_low": r["lo_v71"], "ci_high": r["hi_v71"]}, label=False)] for _, r in d.iterrows()]
    return ("表 S6　按 MeSH 疾病大类的 Recall@10（测试疾病 ≥60 的大类；三种子平均；差值为 95% CI）\n\n" +
            table_md(["大类", "疾病数", "SARA", "多模态直接融合", "检索通道 LightGBM", "语义检索 $s_F$", "SARA − 检索通道 LightGBM",
                      "SARA − 多模态直接融合"], rows))


def table_s7(R, N) -> str:
    e4 = R["e4"]
    c8 = sd("Fig8_evidence_sources", "c").set_index("index")
    rows = []
    for k, lab, den in E4_PUB:
        a = e4[k]
        red, dif = a["new_minus_old_reduced"], a["difference"]
        if den is not None and den in c8.index:
            dd = c8.loc[den]
            dtxt = " / ".join(pc(float(dd[c]), 2 if float(dd[c]) < 0.01 else 1)
                              for c in ("annotation_rows", "genes", "query_relations"))
        else:
            dtxt = "同上一行（数量匹配）"
        rows.append([lab, dtxt, num(red["a"]), num(red["b"]), sgn(red["delta"]), ci(dif, label=False)])
    full = e4["e4_layer1"]["new_minus_old_full"]
    return ("表 S7　证据删除条件（种子 42，整链重训；重打分条件只替换推理输入）\n\n" +
            table_md(["条件", "删除比例（注释行 / 受影响基因 / 受影响查询关联）", "SARA R@10", "检索通道 LightGBM R@10",
                      "SARA 优势（删减后）", "优势变化（删减 − 完整，95% CI）"], rows) +
            f"\n\n注：完整数据中 SARA 的优势为 {ci(full)}。* 使用了评价关联的文献，只作敏感性分析，不参与任何模型选择。")


def table_s8(R, N) -> str:
    e2 = {r["method"]: r for r in R["e2"]["table"]}
    con = R["e2"]["contrasts"]
    rows = []
    for b in (0, 1, 2):
        m = f"corrector_beta{b}"
        rows.append([f"门控纠偏器 β = {b}", num(e2[m]["recall10"]), num(e2[m]["cold_recall10"]), num(e2[m]["cold_to_all10"]),
                     ci(con[m]["recall10_vs_deployed"], label=False), ci(con[m]["cold_vs_deployed"], label=False)])
    d = e2["deployed"]
    rows.insert(0, ["SARA", num(d["recall10"]), num(d["cold_recall10"]), num(d["cold_to_all10"]), "参照", "参照"])
    a = table_md(["模型", "R@10", "$R^c$@10", "$R^{c\\to all}$@10", "ΔR@10（95% CI）", "Δ$R^c$@10（95% CI）"], rows)
    v = R["variants"]
    rows_b = []
    for k, lab in E5_PUB:
        dd = v[k]["deployed"]
        rows_b.append([lab, num(dd["recall10"]), num(dd["cold_recall10"]), pc(dd["mask_test_pool_micro"]),
                       pc(dd["mask_test_hit10_micro"]), ci(dd["delta_vs_main"], label=False)])
    b = table_md(["条件（测试屏蔽基因）", "R@10", "$R^c$@10", "屏蔽基因阳性入选", "屏蔽基因阳性前十", "相对完整模型的 R@10 差值"], rows_b)
    return ("表 S8　未进入最终模型的两种设计\n\n（a）门控冷平衡纠偏器（三种子平均；在 SARA 候选上重训，开发集选定 λ）\n\n" + a +
            "\n\n（b）序列补位模型（种子 42；删除测试屏蔽基因的 GO 后整链重训；最后一行为在未删除 GO 的输入上开启补位）\n\n" + b)


def table_s9(R, N) -> str:
    et = pd.DataFrame(R["external"]["table"])
    subs = [("union", "合并"), ("GenCC", "GenCC"), ("ClinVar", "ClinVar"), ("independent_evidence", "文献独立子集"),
            ("no_neighbour_relation", "邻近疾病无同一关联")]
    ms = [m for m in ("deployed", "v71_direct_strict", "channel_lgb_v6", "B", "F7_v6") if m in set(et["method"])]
    rows = []
    for s, lab in subs:
        for m in ms:
            r = _row(et, method=m, subset=s)
            if not r:
                continue
            rows.append([f"{lab}（{i0(r['n_diseases'])} 病 / {i0(r['n_pairs'])} 对）", "SARA" if m == "deployed" else pub(m),
                         num(r["recall10"]), num(r["hit10"]), num(r["recall50"]), pc(r["in_list"])])
    a = table_md(["子集", "方法", "R@10", "Hit@10", "R@50", "进入候选"], rows)
    hi = J(paths8.AUDIT / "historical_inputs_v8.json")
    rows_b = [[HIST_PUB.get(r["input"], r["input"]), r["date"], "是" if r["not_later_than_cutoff"] else "否", (r["sha256"] or "")[:12]]
              for r in hi["inputs"]]
    b = table_md(["输入", "日期", "不晚于截止日", "SHA-256 前 12 位"], rows_b)
    G = i0(N["n_genes"])
    later = "；".join(f"{LATER_PUB.get(x['input'], x['input']).replace('19,861', G)}（{x['date']}，"
                     f"{ROLE_PUB.get(x['role'], x['role'])}）" for x in hi.get("later_inputs_disclosed", []))
    return ("表 S9　外部标签与时间外推\n\n（a）外部标签上的性能（过滤排名，三个种子平均；进入候选为外部关联进入候选名单的比例；"
            "“邻近疾病无同一关联”为父、子或同家族参考疾病中均未出现同一关联的子集）\n\n" + a +
            f"\n\n（b）时间外推的输入（截止 {hi['cutoff']}）\n\n" + b + f"\n\n晚于截止日、逐项披露的输入：{later}。")


def table_s10(R, N) -> str:
    c5 = pd.DataFrame(R["support"]["cohort533"])
    rows = [[SUP_PUB.get(r.method, r.method), str(int(r.k)), num(r.recall10), num(r.recall50), i0(r.n_diseases)]
            for r in c5.itertuples()]
    a = table_md(["方法", "k", "R@10", "R@50", "疾病数"], rows)
    c1 = pd.DataFrame(R["support"]["cohort1shot"])
    rows1 = [[SUP_PUB.get(r.method, r.method), num(r.recall10), num(r.recall50), i0(r.n_diseases)] for r in c1.itertuples()]
    b = table_md(["方法", "R@10", "R@50", "疾病数"], rows1)
    t = pd.read_csv(paths8.EXT_RES / "expression_v8_table.csv")
    t["base"] = t["set"].str.rsplit("_top", n=1).str[0]
    t["K"] = t["set"].str.rsplit("_top", n=1).str[1]
    cols = [(acc, c) for acc, c in t[["accession", "contrast"]].drop_duplicates().itertuples(index=False)]
    rows_c = []
    for base, lab in EXPR_SET:
        for K in ("20", "50"):
            s = t[(t["base"] == base) & (t["K"] == K)]
            if s.empty:
                continue
            cells = []
            for acc, c in cols:
                r = s[(s["accession"] == acc) & (s["contrast"] == c)]
                if r.empty:
                    cells.append("—")
                    continue
                r = r.iloc[0]
                p = float(r["p_value"])
                cells.append(f"{num(r['enrichment'])}（{'**' if p < 0.05 else ''}P = {num(p, 3) if p >= 0.001 else sci(p)}"
                             f"{'**' if p < 0.05 else ''}）")
            rows_c.append([f"{lab}{'' if lab.endswith('）') else ' '}前 {K}"] + cells)
    c_ = table_md(["基因集"] + [f"{acc} {CONTRAST_PUB.get(c, c)}" for acc, c in cols], rows_c)
    cr = N["case_rows"]
    rows_d = []
    for c in N["cases"]:
        for m, lab in (("deployed", "SARA"), ("channel_lgb_v6", "检索通道 LightGBM")):
            s_ = cr[(cr["disease_id"] == c["disease_id"]) & (cr["method"] == m)].sort_values("rank")
            rows_d.append([f"{CASE_ZH.get(c['disease_id'], c['disease_id'])}（{c['disease_id']}）", lab,
                           "、".join((f"**{x}**" if h else x) + ("°" if cold else "")
                                    for x, h, cold in zip(s_["symbol"], s_["held_out_positive"], s_["reference_path_cold"]))])
    d_ = table_md(["疾病", "方法", "前 10 名"], rows_d)
    return ("表 S10　应用任务与案例\n\n（a）支持任务（533 个疾病；k 为显露的支持基因数；语义与 RWR 的组合权重 ω = 0.25）\n\n" + a +
            "\n\n（b）单样本支持任务（760 个疾病，k = 1）\n\n" + b +
            "\n\n（c）四个表达队列中冻结基因集的秩检验（富集 = 选中基因 |t| 高于匹配背景的概率；加粗为 P < 0.05）\n\n" + c_ +
            "\n\n（d）案例疾病的前 10 名（种子 42；粗体为留出的真实关联，° 为参考路径冷）\n\n" + d_)


# ============================================================================ figure legends
SFIGS = [
    ("S1", "FigS1_teacher_training", "语义模型的训练与未标注对照集的选择",
     "**a** 五个外层上下文中语义模型在同一上下文 $F_{j+1}$ 上的 $s_F$ 全背景 Recall@10 随训练轮次的变化（颜色深浅为种子；灰线为参考"
     "关联基因对照，种子 42），圆点为最佳轮次。**b** 左：开发集上两种未标注对照集的 $s_F$ Recall@10；右：各外层语义模型的最佳轮次"
     "（实心）与早停轮次（空心）。"),
    ("S2", "FigS2_development_selection", "开发集上的配置选择",
     "**a** 各外层的开发阳性覆盖@500 随 RRF 权重 $w$ 的变化（实线为可用性感知 RRF，虚线为普通 RRF，橙线为检索前 500；黑圈为选定的 "
     "$w_j$）。**b** SARA 排序器六个候选配置（叶数 × 最小叶样本）的开发 Recall@10，黑框为各外层选定的配置。"),
    ("S3", "FigS3_fold_seed_stability", "各外层 × 种子的稳定性",
     "**a** 各外层 × 种子的端到端 Recall@10：行为方法，列为外层 · 种子，最右为行均值（统一色标）。**b** 逐外层的逐病比较（三种子"
     "平均）：左为 SARA 相对检索通道 LightGBM 与多模态直接融合的胜 / 平 / 负比例，右为差值与 95% 区间（实心圆：对检索通道 LightGBM；"
     "空心方块：对多模态直接融合）。**c** SARA 逐病 Recall@10 在种子之间的一致性（六边形分箱，对数色标；r 为 Pearson 相关）。"
     "区间为按疾病家族重抽样的 bootstrap 95% 区间。"),
    ("S4", "FigS4_semantic_variants", "全背景单独语义排名与文本变体",
     "**a** 各语义方法的 Recall@10（条）与 $R^c$@10（圈），紫色为语义检索 $s_F$ 的三种文本版本。**b** 相对语义检索 $s_F$（分段文本）的 "
     "Recall@10 差值与 95% 区间（与左图同行）。"),
    ("S5", "FigS5_embedding", "语义空间的结构",
     "**a** 语义模型疾病向量 $z_d$ 的 t-SNE（外层 0，种子 42），每个小图突出一个 MeSH 大类（括号内为疾病数）。**b** 已知关联的疾病—基因对"
     "与随机有 GO 基因的余弦分布；这些关联在训练中可见，只用于说明空间结构。"),
    ("S6", "FigS6_compute_resources", "计算资源",
     "**a** 各组件运行的完成时间线（点大小 ∝ 计时秒数）。**b** 各组件的运行数与计时机时。"),
    ("S7", "FigS7_evidence_blocks", "证据块关联的恢复与删减数据中的 SARA 优势",
     "**a** 被隔离证据块中测试关联的前十恢复率：空心为完整数据，箭头指向删减数据（蓝：SARA；橙：检索通道 LightGBM，只在证据块条件中"
     "重训）。**b** 各删除条件下 SARA 相对检索通道 LightGBM 的 Recall@10 差值；灰带与竖线为完整数据的估计值与 95% 区间。"),
    ("S8", "FigS8_alternative_designs", "未进入最终模型的两种设计",
     "**a** 门控冷平衡纠偏器在各外层 $F_{j+1}$ 上的验证 Recall@10 随提升轮次的变化（开发阶段，种子 42；颜色深浅为 β，实线/点线为 λ），"
     "圆点为最佳轮次。**b** 开发集上的模型选择：蓝色为 SARA 排序器的六个配置（黑圈为选定配置），品红为纠偏器（形状为 β，实心/空心为 "
     "λ），黄色为基础排序器；浅蓝带为“最佳排序器 − 0.005”以内的区间。**c** 测试集上三个 β 的纠偏器相对 SARA 的差值（三种子；开发集"
     "选定的 λ）：左为 Recall@10，右为 $R^c$@10 及 β 之间的冷恢复对比。**d** 序列补位模型在开发屏蔽集上的效果：屏蔽基因阳性进入候选的"
     "比例相对不补位的变化（实心，5 个外层上下文）与长度层内打乱配对的对照（空心）；浅蓝区为采用所需的正值区。**e** 屏蔽基因补位输出"
     "的质量：左为在约 1.9 万个真实功能向量中找回自身的前十命中率，右为与真实向量的余弦（实心）和两两平均余弦（空心）。**f** 删除测试"
     "屏蔽基因的 GO 后，各补位方式下屏蔽基因阳性进入候选与前十的比例，以及总体 Recall@10 相对完整模型的差值（种子 42）；竖线为不补位"
     "的水平。**g** 各补位方式相对不补位的逐对名次变化：向右为名次上升、向左为下降，深色窄条为其中跨越入选边界的阳性。区间为按疾病"
     "家族重抽样的 bootstrap 95% 区间。"),
]


# ============================================================================ assembly
def compose(R=None, N=None) -> tuple[str, dict]:
    from .refs9 import resolve
    if R is None:
        R, N = numbers9()
    TRACE.clear()
    head = [f"# 补充材料", f"**{TITLE}**",
            "本补充材料包括补充说明 S1–S6、补充表 S1–S10 与补充图 S1–S8。正文中的图、表与方法编号均指正文。"]
    notes = [note_s1(R, N), note_s2(R, N), note_s3(R, N), note_s4(R, N), note_s5(R, N), note_s6(R, N)]
    tabs = ["## 补充表", table_s1(R, N), table_s2(R, N), table_s3(R, N), table_s4(R, N), table_s5(R, N), table_s6(R, N),
            table_s7(R, N), table_s8(R, N), table_s9(R, N), table_s10(R, N)]
    figs = ["## 补充图"]
    for n, stem, title, leg in SFIGS:
        figs += [f"<!-- figure {stem} -->", f"**图 {n}　{title}。** {leg}"]
    md = "\n\n".join(p.strip("\n") for p in head + notes + tabs + figs + ["## 补充参考文献", "<<REFERENCES>>"]) + "\n"
    md, refs, order = resolve(md)
    md = md.replace("<<REFERENCES>>", "\n".join(refs))
    return md, {"stem": SUPP_STEM, "ref_order": order, "n_refs": len(refs), "figures": [(n, s) for n, s, _, _ in SFIGS],
                "trace": {k: v for k, v in TRACE.items()}}


def write() -> dict:
    R, N = numbers9()
    md, info = compose(R, N)
    p = ROOT / f"{SUPP_STEM}.md"
    p.write_text(md, encoding="utf-8")
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "supp9_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"md": str(p), "chars": len(md), "refs": info["n_refs"]}


if __name__ == "__main__":
    print(json.dumps(write(), ensure_ascii=False, indent=1))

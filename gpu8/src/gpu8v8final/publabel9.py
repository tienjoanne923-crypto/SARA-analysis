"""Single naming source of the SARA submission (figures and text).

Internal development names (version tags, experiment codes, "候选 A", "仅通道", ...) are mapped to the
public names used in the paper.  ``relabel(s)`` works on the plain text of a figure string (mathtext
CJK wrappers removed); ``FORBIDDEN`` lists the internal terms that must not survive anywhere in the
main documents, ``SUPP_ALLOW`` the few that are legitimate in the supplement.
"""
from __future__ import annotations

import re

METHOD = "SARA"

# ------------------------------------------------------------------ public method names (full, short)
PUBLIC = {
    "deployed": ("SARA", "SARA"),
    "channel_lgb_v6": ("检索通道 LightGBM", "检索通道 LGB"),
    "v71_direct_strict": ("多模态直接融合", "直接融合"),
    "v71_direct_v6": ("多模态直接融合（检索前 500）", "直接融合（检索前 500）"),
    "B": ("基础排序器", "基础排序器"),
    "F7_v6": ("固定权重融合（检索前 500）", "固定融合（检索前 500）"),
    "F7_strict": ("固定权重融合", "固定融合"),
    "standalone_sF": ("语义检索 $s_F$", "语义检索 $s_F$"),
    "standalone_r": ("多模态语义核心 $r$", "语义核心 $r$"),
    "standalone_sF0": ("SapBERT 零训练相似度", "SapBERT 相似度"),
    "standalone_sS": ("语义模型序列分数", "序列分数"),
    "kdgene": ("KDGene", "KDGene"),
    "sign": ("SIGN 式图模型", "SIGN"),
    "C_b0": ("门控纠偏器 β=0", "纠偏 β=0"),
    "C_b1": ("门控纠偏器 β=1", "纠偏 β=1"),
    "C_b2": ("门控纠偏器 β=2", "纠偏 β=2"),
}


def pub(key: str, short: bool = False) -> str:
    v = PUBLIC.get(key, (key, key))
    return v[1] if short else v[0]


# ------------------------------------------------------------------ exact-string replacements (plain text)
FULL = {
    "非劣界以外": "",
    "仅通道前 10": "仅检索 LGB 前 10",
    "仅部署前 10": "仅 SARA 前 10",
    "B": "基础排序器",
    "③ 冷恢复 $R^c$@10，对 V7.1（描述性）": "冷恢复 $R^c$@10，对多模态直接融合",
    "① 对仅检索通道 LGB（其自身 V6 500 名单）": "对检索通道 LGB（检索前 500）",
    "② 对 V7.1 直接融合（同一严格 500 重训）": "对多模态直接融合（同一候选名单）",
    "事先登记的判定层级（10,000 次家族 bootstrap）": "主要比较的差值分布（10,000 次家族 bootstrap）",
    "逐疾病胜负（层级①的两种方法）": "逐疾病胜负（SARA 对检索通道 LGB）",
    "V6 500 上通道排序为仅通道 LGB，严格 500 与 950 上为 B": "检索前 500 上为检索通道 LGB，另两种名单上为基础排序器",
    "V6 500（仅通道）": "检索前 500",
    "缺失感知严格 500": "可用性感知 RRF（500）",
    "普通 RRF 严格 500": "普通 RRF（500）",
    "固定 V6 500": "检索前 500",
    "共享严格 500": "SARA 候选（500）",
    "约 950 桥接": "扩展候选（约 950）",
    "候选 A（部署规则）": "SARA 排序器",
    "仅检索通道排序器": "检索通道排序器",
    "外部 618 个疾病的再分析（V7.1 冻结集合，已用于设计）": "外部 618 个疾病的补充评估",
    "历史任务的逐对名次（X2 与 X1，种子 42）": "时间外推任务的逐对名次（种子 42）",
    "X1：截止时无关联\n监督的疾病（39 病）": "新疾病任务：截止时\n无关联的疾病（39 病）",
    "X1：截止时无监督的疾病（65 对）": "新疾病任务：截止时无关联的疾病（65 对）",
    "X2：已有关系疾病\n的新关系（136 病）": "新关联任务：已知疾病\n的新关系（136 病）",
    "X2：已有关系疾病的新关系（348 对）": "新关联任务：已知疾病的新关系（348 对）",
    "* 使用了评价关系的 PMID：自动标记 evaluation_conditioned_audit，只作审计、不回流选择。":
        "* 使用了评价关系的 PMID：仅作敏感性分析，不参与任何模型选择。",
    "开发集上的 A/B 部署规则：纠偏器没有进入“最佳 A − 0.005”区间": "开发集上的模型选择：纠偏器未进入“最佳排序器 − 0.005”区间",
    "测试集：三个 β 的纠偏器相对部署模型（三种子，登记 λ）": "测试集：三个 β 的纠偏器相对 SARA（三种子，开发集选定 λ）",
    "候选 A 的六个配置：开发 Recall@10（黑框为部署）": "SARA 排序器的六个候选配置：开发 Recall@10（黑框为选定配置）",
    "每个外层：B、最佳纠偏器与最佳 A 的开发 Recall@10": "每个外层：基础排序器、最佳纠偏器与最佳 SARA 排序器的开发 Recall@10",
    "E5：测试屏蔽基因的 GO 被删除后，各补位方式的效果（种子 42）": "测试屏蔽基因的 GO 被删除后，各序列补位方式的效果（种子 42）",
    "学生在开发屏蔽臂上的验收（需 > 0）": "序列补位模型在开发屏蔽臂上的效果（需 > 0）",
    "屏蔽基因阳性的名次分布（E5/E6 各臂，种子 42，全部外层）": "屏蔽基因阳性的名次分布（种子 42，全部外层）",
    "E4 各臂在各疾病大类上的“新旧方法差”变化（删减 − 完整，Recall@10 × 100，种子 42；列按相似性聚类）":
        "各证据删除条件在各疾病大类上的 SARA 优势变化（删减 − 完整，Recall@10 × 100，种子 42；列按相似性聚类）",
    "“新旧方法差”的变化：删减数据 − 完整数据（种子 42）": "SARA 优势的变化：删减数据 − 完整数据（种子 42）",
    "删减数据中的新旧方法差": "删减数据中的 SARA 优势",
    "候选覆盖随预算 K 的变化（种子 42，完整背景重算）": "候选覆盖随预算 K 的变化（种子 42）",
    "测试疾病的 V6 来源可用性组合（UpSet）与各组合的端到端差值": "测试疾病的检索来源可用性组合（UpSet）与各组合的性能差值",
    # bare comparison labels of the Figure 3a source data
    "①": "对检索通道 LightGBM（检索前 500）", "②": "对多模态直接融合（同一候选名单）", "③": "冷恢复，对多模态直接融合",
}

# ------------------------------------------------------------------ substring replacements (single pass, longest first)
SUB = {
    # the proposed method
    "V8.1 部署模型": "SARA", "V8.1 部署": "SARA", "部署模型": "SARA", "部署的候选 A": "SARA 排序器",
    "部署的 A": "选定的 SARA 排序器", "：部署 A": "：选定 SARA 排序器", "候选 A 配置": "SARA 排序器配置",
    "候选 A": "SARA 排序器", "命中：部署": "命中：SARA", "部署/仅通道": "SARA/检索 LGB",
    "部署 − 仅通道 LGB": "SARA − 检索通道 LGB", "部署 − 仅通道": "SARA − 检索 LGB",
    "部署最佳名次": "SARA 最佳名次", "部署更靠前": "SARA 更靠前", "部署更好": "SARA 更好",
    "部署更低": "SARA 更低", "部署更高": "SARA 更高", "部署中的名次": "SARA 中的名次",
    "（部署模型行）": "（SARA 行）", "部署名次（严格 500）": "名次（SARA 候选）",
    "V8.1 语义": "SARA 语义", "V8.1 T": "SARA T", "（V8.1 名单）": "（SARA 候选）", "V8.1 名单": "SARA 候选",
    # retrieval-channel LightGBM
    "仅检索通道 LGB（V6 500）": "检索通道 LGB（检索前 500）", "仅检索通道 LGB": "检索通道 LGB",
    "仅检索通道 LightGBM（自身 V6 500）": "检索通道 LightGBM（检索前 500）", "仅检索通道 LightGBM": "检索通道 LightGBM",
    "仅通道 LGB 名次（自身 V6 500）": "检索通道 LGB 名次（检索前 500）",
    "仅通道 LGB": "检索通道 LGB", "仅通道最佳名次": "检索 LGB 最佳名次", "仅通道更靠前": "检索 LGB 更靠前",
    "（仅通道行）": "（检索 LGB 行）", "· 仅通道": "· 检索 LGB", "/ 仅通道": "/ 检索 LGB", "仅通道 T": "检索 LGB T",
    # multimodal direct fusion and other comparators
    "V7.1 直接融合（严格 500）": "多模态直接融合", "V7.1 直接融合（V6 500）": "多模态直接融合（检索前 500）",
    "对 V7.1 直接融合": "对多模态直接融合", "V7.1 直接融合": "多模态直接融合", "F7 固定融合（V6 500）": "固定权重融合（检索前 500）",
    "F7 固定融合（严格 500）": "固定权重融合", "基础排序器 B": "基础排序器",
    "V7.1 核心 $r$ 单独（全背景）": "多模态语义核心 $r$（全背景）", "V7.1 核心 r": "多模态语义核心 $r$",
    # semantic model
    "教师 $s_F$ 单独（全背景）": "语义检索 $s_F$（全背景）", "教师 $s_F$ 单独": "语义检索 $s_F$",
    "教师 $s_F$（V7.1 短文本）": "语义检索 $s_F$（截断短文本）", "相对教师 $s_F$": "相对语义检索 $s_F$",
    "教师 $s_F$": "语义检索 $s_F$", "教师序列分数（全背景）": "语义模型序列分数（全背景）",
    "教师 s_F（V7.1 短文本）": "语义检索 s_F（截断短文本）", "教师 s_F": "语义检索 s_F",
    "无学习 SapBERT 功能相似单独": "SapBERT 零训练相似度", "无学习 SapBERT 功能相似": "SapBERT 零训练相似度",
    "无学习 SapBERT": "SapBERT 零训练相似度",
    "历史任务": "时间外推任务",
    "教师自身序列分数": "语义模型序列分数", "教师序列投影": "语义模型序列投影", "语义教师": "语义模型",
    "与教师的最佳轮次": "与语义模型的最佳轮次",
    "V6 语义": "基线语义",
    # retrieval channels and candidate lists
    "V6 检索通道": "检索通道", "V6 检索名次": "检索名次", "V6 检索顺序": "检索顺序", "V6 融合分": "检索融合分",
    "V6 500 上，A 减仅通道 LGB": "检索前 500 上，SARA 排序器减检索通道 LGB",
    "A：V6 500 → 严格 500": "SARA 排序器：检索前 500 → SARA 候选", "A：严格 500 → 约 950": "SARA 排序器：SARA 候选 → 扩展候选",
    "V6 500 → 严格 500": "检索前 500 → SARA 候选", "由 V6 500 到严格 500": "由检索前 500 到 SARA 候选",
    "进入严格 500": "进入 SARA 候选", "在严格 500 中": "在 SARA 候选中", "严格 500": "SARA 候选", "V6 500": "检索前 500",
    "约 950 桥接": "扩展候选（约 950）", "缺失感知": "可用性感知",
    # evidence / masking experiments
    "“新旧方法差”": "SARA 优势", "新旧方法差": "SARA 优势", "完整数据中新旧差": "完整数据中 SARA 优势",
    "E6 + 删 GO": "撤出监督 + 删 GO", "E6 撤出监督": "撤出关联监督", "E6：关联监督冷": "关联监督冷",
    "GO 缺失（E5）": "GO 缺失（受控）", "监督冷（E6）": "关联监督冷（受控）",
    # sequence imputation (supplement only)
    "ProtST 输入学生": "ProtST 输入补位", "两层学生（配对打乱）": "两层补位（配对打乱）", "两层学生（打乱）": "两层补位（打乱）",
    "删 GO·两层学生": "删 GO·两层补位", "两层学生": "两层补位", "线性学生": "线性补位", "学生（5 个外层）": "补位模型（5 个外层）",
    "屏蔽基因的学生输出质量": "屏蔽基因的补位输出质量", "序列学生": "序列补位模型",
    # figure-3 verdict lines (the interval is shown; the wording states what it contains)
    "下界 > 0：成立": "区间 > 0", "下界 < −0.005：非劣未确立": "区间含 0", "②未成立：只作描述": "区间含 0",
    # SHAP / figure-6 titles
    "部署模型的 TreeSHAP": "SARA 排序器的 TreeSHAP",
    "：特征增益占比（5 外层 × 3 种子）": "的特征增益占比（5 个外层 × 3 个种子）",
}

_SUB_RX = re.compile("|".join(re.escape(k) for k in sorted(SUB, key=len, reverse=True)))
_SPACE_AFTER = re.compile(r"(SARA|LGB|LightGBM)(?=[一-鿿])")
_SPACE_BEFORE = re.compile(r"(?<=[一-鿿])(SARA)")


def relabel(plain: str) -> str:
    """Map one plain-text figure/text string to its public wording."""
    if plain in FULL:
        return FULL[plain]
    s = _SUB_RX.sub(lambda m: SUB[m.group(0)], plain)
    s = s.replace("部署", "SARA")            # last resort (explicit rules above cover the known strings)
    s = _SPACE_AFTER.sub(r"\1 ", s)
    s = _SPACE_BEFORE.sub(r" \1", s)
    return s.replace("  ", " ")


# ------------------------------------------------------------------ terms that must not survive
FORBIDDEN = [
    r"(?<![A-Za-z])V[5-8](?:\.\d)?(?![0-9])", r"V71", r"本轮", r"登记", r"[①②③]", r"部署", r"候选 ?A(?![A-Za-z])",
    r"仅(?:检索)?通道", r"学生", r"教师", r"(?<![A-Za-z])E[1-9](?![0-9])", r"(?<![A-Za-z0-9])F7(?![0-9])",
    r"严格 ?500", r"桥接", r"验收", r"盲审", r"档案", r"会话中断", r"独立(?:虚拟)?环境", r"Windows", r"venv",
    r"run_id", r"非劣", r"(?<![A-Za-z])X[12](?![0-9])", r"新旧", r"evaluation_conditioned", r"V8T__",
    r"[A-Za-z]:\\", r"gpu8", r"results_v8", r"\.parquet", r"\.json", r"未成立", r"未确立", r"阴性结果",
]
FORBIDDEN_RX = re.compile("|".join(FORBIDDEN))
SUPP_ALLOW = {"门控纠偏", "纠偏", "补位"}          # (these words are not in FORBIDDEN; kept for documentation)


def forbidden_hits(s: str) -> list[str]:
    return [m.group(0) for m in FORBIDDEN_RX.finditer(s)]

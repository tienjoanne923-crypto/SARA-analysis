"""Appendices of the V7.1 manuscript: plan-section execution table, ledger summary,
unfinished / limited items, and the archived V6 appendices."""
from __future__ import annotations

import pandas as pd

from . import paths71

PLAN_ROWS = [
    ("1 执行摘要", "先独立运行、再接入 V6；四级“能够运行”验收", "gpu8v71 独立包；数据合同→真实资源→真实训练与评价→外层增益逐级验收", "完成"),
    ("2 V6 事实基础", "重新登记检索通道 LightGBM 强对照；不重启 PMID 降权、元岭、神经残差", "E1 中重新登记并三种子运行；撤下的部件未重启", "完成"),
    ("3 多模态定义", "文本/HPO 为同一疾病信息两种表示；GO 为功能视图；序列、表达、图像为不同原始模态", "特征包字段与掩码；GO 只作功能语义与锚", "完成"),
    ("4 文本与 HPO", "SapBERT CLS（池化前）、128 token、模板、频率加权 HPO、禁止 LLM 生成文本", "encoders71 / featurepack71；截断计数入报告", "完成"),
    ("5 蛋白序列", "代表蛋白固定规则、异常表、ESM-2 35M 分窗覆盖归一", "sequences71 / encoders71；窗口单元测试", "完成"),
    ("6 组织表达", "HPA 共识 51 组织、表头核对、零与缺失分存、中心化、疾病组织条件、c≥0.8", "expression71（别名表登记）", "完成"),
    ("7 显微图像", "DenseNet 1,024 维（不与 SubCell 混称）、细胞系≠组织、层级等权聚合、定位标签/配对置换/细胞系对照", "image71 + E8（定位标签、细胞系内置换、细胞系中心化）", "完成（图像—定位锚预训练未做）"),
    ("8 IMPC 跨物种", "实验调用、同源映射、可核验 HP–MP 桥、映射未通过则关闭", "mouse71：MGI 1:1（HCOP 404）、MGI 人工整理 SSSOM、程序核验、抽查表导出", "探索性完成（人工抽查未做）"),
    ("9 数据合同", "主键对齐、缓存键、来源隔离、两种基因冷", "featurepack71 哈希；strict 变体", "完成（provenance_reviewed=false）"),
    ("10 模型公式", "P_T/P_S/P_I、行内归一、掩码混合、逐模态分数、门控、valid、参数量", "core71；参数量 80,069/145,670 单元测试", "完成"),
    ("11 对齐与训练", "锚损失、多正例排序、默认配置、开发对照", "train71；8 配置开发网格", "完成"),
    ("12 两阶段接回 V6", "独立检索→工作池→二级模型；OOF 覆盖整条链；三种归因", "chain71（60 次整链）/ pools71 / fusion71", "完成"),
    ("13 验收五层", "数据/计算/行为/配对/预测；必做负对照", "单元测试、探针、E5（关闭/置换/只保留）、仅掩码模型、原始拼接", "完成"),
    ("14 评价矩阵", "E1–E8；指标口径不变；新确认集", "evaluate71；新确认集建模前冻结", "完成（ProtST/BioTranslator/DeepPheno 以可控对照替代）"),
    ("15 资源预算", "本机探针、GPU 分配/保留、RSS、CUDA 不可用即报错", "probe71 与逐运行资源记录", "完成"),
    ("16 工程顺序", "固定现状→最小核心→OOF 接回→条件扩展→冻结评价", "按顺序执行并登记", "完成"),
    ("17 附包脚本", "run_core 等脚本", "附包不在本机；按数据合同在 gpu8v71 中重新实现", "受限（重新实现）"),
    ("18 失败与停止规则", "取不到资源不顶替、模态无效则关闭等", "登记为停止与降级规则并执行", "完成"),
    ("19 论文定位", "只有满足判定规则才在题目中写多模态", "manuscript71 按登记规则自动决定题目", "完成"),
    ("20 参考资料", "官方资源与方法文献", "参考文献 41–60", "完成"),
]

UNFINISHED = [
    "人工双人盲审：未完成（沿用 V6 的抽样表）。",
    "小鼠 MP–HP 映射的人工抽查：已导出 100 行抽查表（audit/hp_mp_bridge_spotcheck_sample_v71.csv），未完成人工核对。",
    "图像—定位锚预训练、预测表型头、154 种细胞类型表达、更大的 ESM：方案列为后续增强，本轮未实现。",
    "ProtST、BioTranslator、DeepPheno：未完整复现，以同资源固定融合、线性双塔、锚翻转与原始拼接替代。",
    "多模态核心的严格历史版本（编码器与 GO 的时间边界）：未构建；外部与时间任务沿用 V6。",
    "HCOP 批量文件（404）：以 MGI HOM_MouseHumanSequence 替代。",
    "PDF 所述附包脚本（run_core.py 等）不在本机：按数据合同重新实现，未逐字对照。",
]


def ledger_table() -> str:
    p = paths71.MANIFESTS / "run_ledger_v71.csv"
    if not p.exists():
        return "（账本未生成）"
    L = pd.read_csv(p, encoding="utf-8-sig")
    L["group"] = L["run_id"].str.split("__").str[0]
    L["method"] = L["run_id"].str.split("__").str[1]
    t = L.groupby(["group", "method"]).agg(runs=("run_id", "size"), done=("status", lambda s: int((s == "done").sum())),
                                           hash_ok=("hash_ok", "sum"), seconds=("seconds", "mean")).reset_index()
    rows = ["| 阶段 | 方法 / 变体 | 运行数 | 完成 | 哈希通过 | 平均秒数 |", "|---|---|---|---|---|---|"]
    for r in t.itertuples():
        rows.append(f"| {r.group} | {r.method} | {r.runs} | {r.done} | {int(r.hash_ok)} | {r.seconds:.0f} |")
    return "\n".join(rows)


def appendices(secs) -> str:
    parts = ["## 附录 A　V7.1 方案逐节执行对照",
             "| 方案节 | 要求 | 落实 | 状态 |", "|---|---|---|---|"]
    parts += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in PLAN_ROWS]
    parts += ["", "## 附录 B　V7.1 运行账本摘要", ledger_table(), "",
              "## 附录 C　V7.1 未完成与受限项"]
    parts += [f"{i + 1}. {x}" for i, x in enumerate(UNFINISHED)]
    import re
    on = False
    v6 = []
    remap = {"A": "D", "B": "E", "C": "F", "D": "G"}
    for h, b in secs:
        if h.startswith("## 附录 A"):
            on = True
        if on:
            h2 = re.sub(r"^## 附录 ([A-D])", lambda m: f"## 附录 {remap[m.group(1)]}（V6 存档）", h, count=1)
            v6.append((h2, b))
    for h, b in v6:
        parts.append(f"{h}\n\n{b}")
    return "\n".join(parts)

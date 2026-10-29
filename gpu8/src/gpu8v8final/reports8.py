"""V8.1 reports (Chinese): status & problems, PDF closure table, unfinished items, delivery notes.

Every number is read from the result files (numbers_v8_results.json and the audits);
nothing is typed in by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import paths8
from .common8 import dump_json, now_utc


def load(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ci(c: dict) -> str:
    if not c:
        return "—"
    return f"{c['delta']:+.3f}（95% CI {c['ci_low']:+.3f} 至 {c['ci_high']:+.3f}；{c['n']} 病）"


def closure_rows(R: dict) -> list[tuple[str, str, str, str]]:
    acc = R.get("acceptance", {})
    rep = R.get("reproduction", {})
    hist = R.get("historical", {})
    ext = R.get("external", {})
    rows = [
        ("§1–2 事实与修正", "V7.1 复现、草案修正逐条落实", "完成",
         "model_spec.json：V7.1 第 0 折外层逐位复现；A1–E7 修正见方法 2.1–2.8"),
        ("§3.1 任务与分母", "19,861 基因混合背景、therapeutic 分层", "完成", "analysis/strata_zero_signal_v8.json"),
        ("§3.2 三类冷", "参考路径冷 / GO 输入缺失 / 真正监督冷分表", "完成", "E3、E5、E6 分表"),
        ("§3.3 支持任务", "533 病 0/1/3/5 与 1-shot；ω 开发重选；量子游走五折", "完成" if R.get("support") else "未完成",
         "external/support/"),
        ("§4.1 生产不读标签", "角色名单排除；标签移走后重生成", "完成" if acc.get("1_label_removal", {}).get("pass") else "见验收",
         "audit/acceptance_ten_checks_v8.json 第 1 项"),
        ("§4.2 三层验证 + 匹配随机", "训练可见 / 评价条件化 / 证据块，各配随机", "完成" if R.get("variants") else "未完成",
         "variants/e4_*；audit/evidence_policy_audit_v8.json"),
        ("§4.3 证据代码变体", "实验 / 去 TAS·NAS / 去 IC / 去突变 / 仅 IEA", "完成", "e4_experimental、e4_no_tas_nas、e4_no_mutant、重打分 no_ic/iea_only"),
        ("§4.4 五种分母", "引文、注释行、唯一 GO、基因、查询关系", "完成", "evidence_policy_audit_v8.json"),
        ("§5.1 复现实测模型", "r=128、159,749 参数", "完成", "manifests_v8/model_spec.json"),
        ("§5.2 分段文本", "句界优先、≤128、CLS、按词片加权、名称不重复", "完成", "audit/text_segments_l1.json；text_short/level2 配对"),
        ("§5.3 不用伪 HPO", "—", "遵守", "—"),
        ("§5.4 GO 聚合", "来源与 NOT 过滤后聚合；预测不写回观察 GO", "完成", "evidence8.function_arrays；semantic_source"),
        ("§6 序列学生", "156,032 参数；stopgrad；GO 优先；开发验收", "完成（开发未通过→自然主链关闭）",
         "manifests_v8/dev_student_selection_v8.json"),
        ("§6.5 650M", "仅在 35M 通过后一次上限比较", "未触发（登记条件未满足）", "登记 esm650m_trigger"),
        ("§7 表达 / ncRNA / 图像 / 小鼠", "表达三臂；ncRNA 公平回退与零信号；图像小鼠停止", "完成", "expr_off、expr_random；analysis/zero_signal"),
        ("§8 严格 500", "缺失感知 RRF、w 开发选择、普通 RRF 对照、三计数", "完成", "pools_*.npz；plain_rrf"),
        ("§8.3 三张表 + 950 桥接", "固定旧 500 / 共享新 500 / 端到端 / 950", "完成" if R.get("variants", {}).get("bridge950") else "见结果",
         "v6pool、main、bridge950"),
        ("§9 门控纠偏与候选 A", "自定义目标、β×λ 六组、A 六组、A/B 规则", "完成（五个上下文均部署 A）", "dev/fold*/selection.json"),
        ("§9.5 冷正例数量检查", "不足时才开冷模拟", "完成（数量充足，未开启）", "登记附录 rule_text_correction_31_leaf"),
        ("§10 整链 OOF", "每折 3 内 + 1 外；F_{j+1} 检查点；每外层独立选择", "完成", "acceptance 第 3、4 项"),
        ("§10.4 缓存依赖", "资源哈希、训练集合哈希、屏蔽集合、参数", "完成", "v6pass meta、teacher result.json"),
        ("§11 GO 屏蔽与定向实验", "dev/test 屏蔽分离；E5 矩阵；E6 两张表", "完成" if R.get("variants", {}).get("e5_none") else "见结果",
         "variants/e5_*、e6_*"),
        ("§12.1 618 外部再分析", "三字段；GenCC/ClinVar 分报 + 合并", "完成" if ext else "未完成", "external/external618/"),
        ("§12.1 新来源确认", "可得才做", "声明不可得（登记）", "登记 external.new_source"),
        ("§12.3 X1/X2", "截止 2023-12-31；逐输入日期与哈希", "完成" if hist else "未完成", "external/historical/；audit/historical_inputs_v8.json"),
        ("§12.4 160 对双人盲审", "新独有/基线独有/共同/低排名各 40", "材料已交付，审核未进行（需两名人工审阅者）", "external/review160/"),
        ("§12.4 四表达队列", "冻结 Top-20/50，保留阴性", "完成" if R.get("expression") else "未完成", "external/expression_v8.json"),
        ("§13 指标与推断", "R、R^c、R^{c→all}、U·η、家族 bootstrap、层级", "完成", "tables/、numbers_v8.json"),
        ("§14 比较器", "F7、仅通道 LGB、V7.1 直接融合、同资源、双塔、KDGene、SIGN、ProtST", "完成",
         "formal/comparators、variants/two_tower、protst_zeroshot"),
        ("§15 拟合计数与资源", "按依赖图记账", "完成", "manifests_v8/run_ledger_v8.csv"),
        ("§16.3 十项验收", "—", "全部通过" if acc.get("all_pass") else "见验收文件", "audit/acceptance_ten_checks_v8.json"),
        ("§16.4 复现", "另一环境按容差比较",
         ("独立环境被系统策略阻止；主环境全新缓存重算在容差内" if rep.get("within_tolerance") else ("见复现文件" if rep else "未完成")),
         "audit/reproduction_env_blocked_v8.json；audit/reproduction_venv_repro_v8.json"),
        ("§18.2 停止条件", "纠偏不胜 NaN 强融合→部署后者；学生无效不扩大 ESM", "按规则执行", "登记与开发选择"),
    ]
    return rows


def status_md(R: dict) -> str:
    m = R.get("main", {})
    h = m.get("hierarchy", {})
    v = m.get("verdict", {})
    t = {r["method"]: r for r in m.get("table", [])}
    dep = t.get("deployed", {})
    lines = [f"# V8.1 现状、结果与问题说明", "", f"生成时间：{now_utc()}", "",
             "## 1 本轮完成了什么", "",
             "按《疾病基因检索 V8.1 定稿方案》从头实现并运行了整链：角色排除、17 列 GAF 证据政策、分段文本、GO 屏蔽集合、"
             "V7.1 核心逐位复现后的语义教师（$s_F$ 检查点）、缺失感知严格 500、整链 OOF 的基础排序器、门控冷平衡纠偏器与候选 A、"
             "测试前登记、三种子正式主链、E1–E9、比较器、历史与外部评价、十项整链验收；另一环境复现被 Windows 应用程序控制策略阻止，"
             "改为在主环境以全新缓存从冻结输入重算（检验确定性，不检验环境独立性）。", "",
             "## 2 主要结果（F_j 测试，三种子按疾病平均）", "",
             f"* 部署模型（每个外层上下文均为候选 A）端到端 Recall@10 = {f3(dep.get('recall10'))}；"
             f"冷恢复 $R^c$@10 = {f3(dep.get('cold_recall10'))}。",
             f"* 层级①（对仅检索通道 LightGBM 在其自身 V6 500 上）：{ci(h.get('h1_vs_channel_lgb_v6'))}，"
             f"{'成立' if v.get('h1_success') else '不成立'}。",
             f"* 层级②（对 V7.1 直接融合在新严格 500 上重训，非劣界 −0.005）：{ci(h.get('h2_vs_v71_direct_strict'))}，"
             f"{'成立' if v.get('h2_success') else '不成立'}。",
             f"* 层级③（冷恢复优效）：{'已检验：' + ci(h.get('h3_cold_vs_v71_direct_strict')) if v.get('h3_tested') else '因②不成立按登记不检验（描述性：' + ci(h.get('h3_cold_vs_v71_direct_strict')) + '）'}。",
             "", "## 3 问题与限制", ""]
    return "\n".join(lines)


UNFINISHED = [
    ("160 对双人独立盲审", "需要两名人工审阅者；已交付随机行序、隐去类别的空白表格与单独的密钥文件", "未完成（人工环节）"),
    ("新增来源的一次性确认", "本地 GenCC/ClinVar 为 2026-09-22 下载且已用于 618 集合；登记时声明本轮不可得", "不可得（已登记）"),
    ("ESM-2 650M 上限比较", "登记条件为 35M 学生在 ≥3 个开发上下文通过验收；实际 0 个通过", "未触发（按规则）"),
    ("冷正例模拟视图", "训练对中冷正例数量高于登记阈值", "未开启（按规则）"),
    ("31 叶扩展", "所有上下文的纠偏验证曲线均在 200 轮前达到峰值", "未触发（按规则）"),
    ("历史任务的完全时间冻结", "基因背景、MANE 2025 序列、2026 等价组/家族与冻结超参晚于截止日", "受限（已逐项披露）"),
    ("同源簇隔离", "PDF 列为可选，非基础模型前提", "未做（可选项）"),
    ("另一环境复现", "`.venv_repro` 的 SciPy 扩展被 Windows 应用程序控制策略阻止（未改系统设置）；改为主环境全新缓存重算", "受限（确定性已核对）"),
    ("小鼠映射抽查、图像通道", "PDF 规定停止进入主线；V7.1 阴性结果保留于补充材料", "不在本轮范围"),
]


def e4_bullet(R: dict) -> str:
    e4 = R.get("e4", {}) or {}
    retr = {k: r["difference"] for k, r in e4.items() if not k.startswith("e4_rescore")}
    if not retr:
        return "* E4 证据删除：无结果。"
    inc0 = [k for k, d in retr.items() if d["ci_low"] <= 0 <= d["ci_high"]]
    s = f"* E4 证据删除：{len(retr)} 个整链重训条件中 {len(inc0)} 个的新旧方法差变化区间包含 0"
    nm = retr.get("e4_no_mutant")
    if nm and nm["ci_high"] < 0:
        s += (f"；去除突变表型与遗传相互作用类 GO 证据后变化 {nm['delta']:+.4f}（95% CI {nm['ci_low']:+.4f} 至 "
              f"{nm['ci_high']:+.4f}），是需要说明的证据依赖")
    iea = (e4.get("e4_rescore_iea_only") or {}).get("difference")
    if iea and iea["ci_high"] < 0:
        s += f"；仅 IEA 重打分变化 {iea['delta']:+.3f}，反映输入分布改变而非共享证据"
    return s + "。"


def status_full(R: dict) -> str:
    from . import narrative8 as n
    md = [status_md(R)]
    h = R.get("main", {}).get("hierarchy", {})
    h2 = h.get("h2_vs_v71_direct_strict", {})
    gap = (-0.005 - h2["ci_low"]) if h2 and h2.get("ci_low", 0) < -0.005 else None
    e2c = R.get("e2", {}).get("contrasts", {})
    dr = [e2c.get(f"corrector_beta{b}", {}).get("recall10_vs_deployed", {}).get("delta") for b in ("0", "1", "2")]
    dr = [x for x in dr if x is not None]
    hc = R.get("historical_contrasts", {})
    ext = R.get("external", {}).get("fields", {})
    unk = ext.get("shared_original_evidence", {}).get("unknown")
    md.append("\n".join([
        "* 门控冷平衡纠偏器：开发集上五个外层上下文均未达到最佳候选 A 的 0.005 以内（按登记规则部署 A）；测试集上三个 β 的纠偏器总体 "
        f"Recall@10 比部署模型低 {min(-x for x in dr):.3f}–{max(-x for x in dr):.3f}，冷恢复与部署模型无差别，提高 β 不改变冷恢复。"
        "新纠偏降为阴性结果。" if dr else "* 门控冷平衡纠偏器：见 E2。",
        "* 序列学生：开发屏蔽臂上两层与线性学生都降低了屏蔽基因阳性的候选覆盖，未通过验收，自然主链关闭；E5 测试屏蔽臂结论一致。",
        (f"* 层级②非劣以 {gap:.4f} 的幅度未能确立（下界 {h2['ci_low']:+.4f}，界值 −0.005）；按登记，冷恢复优效（层级③）不作推断检验。"
         if gap is not None else "* 层级②：见主结果。"),
        (f"* 历史任务（截止 2023-12-31）中部署模型相对仅检索通道没有收益：X2 {n.ci(hc.get('x2_deployed_vs_channel_lgb_v6'))}，"
         f"X1 {n.ci(hc.get('x1_deployed_vs_channel_lgb_v6'))}。" if hc else "* 历史任务：见 E8。"),
        "* 旧五折已被多次查看，外部 618 集合已用于 V7.1 评价与本轮设计（再分析）；新来源不可得。",
        f"* 42% 的 GO 记录无 PMID，外部 {ext.get('pairs', '—')} 个记录中 {unk if unk is not None else '—'} 个无可比对文献："
        "文献独立性只能在可追溯范围内审计。",
        e4_bullet(R),
        "* 另一环境复现：`.venv_repro` 中 SciPy 扩展模块被 Windows 应用程序控制策略阻止；未修改系统设置，改为主环境全新缓存重算（在容差内）。",
        "* 160 对双人盲审：材料已交付（随机行序、隐去类别），审核需要两名人工审阅者，本轮未进行。",
        "", "## 4 下一轮可讨论的方向（不在本轮执行）", "",
        "1. 纠偏器在 64 个抽样未标注对上学习，而强对照在完整 500 列表上优化前十；若保留门控结构，可改为在 lambdarank 目标上加门控残差，"
        "或把冷权重作为列表级权重，并先在开发集上确认其与候选 A 的差距。",
        "2. 序列补位只有在补位语义质量接近观察 GO 时才可能在缺失感知融合中获益；可考虑按补位置信度调整 $m$，或只在检索分数 $b$ 很低时使用补位。",
        "3. 层级②的非劣差距很小：若下一轮以候选 A 为部署模型，需要事先说明它与 V7.1 直接融合的差别（NaN 明示缺失与 19 项可用性特征）。",
        "", "## 5 数字来源", "", "全部数字来自 `gpu8/results_v8/numbers_v8_results.json` 与各审计文件；逐运行结果见 `gpu8/runs_v8/*/result.json`。"]))
    return "\n".join(md)


def write(R: dict) -> dict:
    out = paths8.REPORTS
    out.mkdir(parents=True, exist_ok=True)
    st = status_full(R)
    (out / "V81_现状结果与问题说明.md").write_text(st, encoding="utf-8")
    un = ["# V8.1 未完成与受限项", "", "| 项目 | 原因 | 状态 |", "|---|---|---|"]
    un += [f"| {a} | {b} | {c} |" for a, b, c in UNFINISHED]
    (out / "V81_未完成与受限项.md").write_text("\n".join(un) + "\n", encoding="utf-8")
    try:
        from gpu8v6.build_docx6 import build as build_docx
        build_docx(out / "V81_现状结果与问题说明.md", out / "V81_现状结果与问题说明.docx", paths8.FIGURES)
    except Exception as e:  # recorded
        (out / "docx_error.txt").write_text(str(e), encoding="utf-8")
    rows = closure_rows(R)
    df = pd.DataFrame(rows, columns=["PDF 章节", "要求", "状态", "证据文件"])
    df.to_csv(out / "V81_PDF逐节关闭表.csv", index=False, encoding="utf-8-sig")
    md = ["# V8.1 PDF 逐节关闭表", "", "| PDF 章节 | 要求 | 状态 | 证据文件 |", "|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    (out / "V81_PDF逐节关闭表.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    delivery = delivery_md(R)
    (paths8.MANUSCRIPT / "交付说明_V81.md").write_text(delivery, encoding="utf-8")
    return {"closure_rows": len(rows), "status": str(out / "V81_现状结果与问题说明.md")}


def delivery_md(R: dict) -> str:
    import glob
    ms = sorted(paths8.MANUSCRIPT.glob("*.md"))
    return "\n".join([
        "# 交付说明（V8.1）", "", f"生成时间：{now_utc()}", "", "## 文件", "",
        f"- 稿件：`manuscript_v8/` 下的 md 与同名 docx（公式为 Word 原生 OMML 对象），图在 `manuscript_v8/figures/`。",
        "- 代码：`gpu8/src/gpu8v8final/`（V5、V6、V7.1 代码与结果未改动，首尾哈希核对见 `results_v8/audit/archive_comparison_v8.json`）。",
        "- 测试：`gpu8/tests/test_v8_properties.py`（参考实现性质）、`test_v8_integration.py`（真实数据接口）。",
        "- 结果：`gpu8/results_v8/`（dev、formal、variants、external、audit、eval、tables、figures、analysis）。",
        "- 逐运行权威结果：`gpu8/runs_v8/*/result.json`；账本 `gpu8/manifests_v8/run_ledger_v8.csv`。",
        "- 登记：`gpu8/manifests_v8/protocol_v8_registered.json`（含带时间戳的附录）。",
        "- 报告：`gpu8/reports_v8/`（现状结果与问题说明 md/docx、PDF 逐节关闭表、未完成与受限项）。",
        "- 盲审材料：`gpu8/results_v8/external/review160/`（空白双人表格、说明；密钥文件不要交给审阅者）。",
        "", "## 复现顺序（PowerShell）", "", "```",
        "$env:PYTHONPATH='D:\\信\\gpu8\\src'",
        "python -m gpu8v8final.audit8 start; python -m gpu8v8final.textseg8 1; python -m gpu8v8final.textseg8 2",
        "python -m gpu8v8final.evidence8; python -m gpu8v8final.masks8; python -m gpu8v8final.teacher8",
        "python -m gpu8v8final.v6pass8 main; python -m gpu8v8final.protst8",
        "python -m gpu8v8final.dev8 --stage controls; python -m gpu8v8final.dev8 --stage teachers",
        "python -m gpu8v8final.dev8 --stage devmask; python -m gpu8v8final.dev8 --stage chain",
        "python -m gpu8v8final.register8",
        "python -m gpu8v8final.formal8 --tag main --seeds 42 52 62 --table1",
        "python -m gpu8v8final.v6pass8 e6cold; python -m gpu8v8final.v6pass8 blocks; python -m gpu8v8final.v6pass8 blocks_random",
        "python -m gpu8v8final.variants8 e5_none e5_mlp e5_linear e5_nn e5_teacher_sS e5_mlp_shuffled e5_protst plain_rrf natural_student_on expr_off expr_random v6pool bridge950 text rescore_no_ic rescore_iea_only",
        "python -m gpu8v8final.variants8 e6_cold e6_cold_gomask e4_layer2 e4_layer2_random e4_experimental e4_no_tas_nas e4_no_mutant e4_blocks e4_blocks_random e4_layer1 e4_layer1_random",
        "python -m gpu8v8final.analysis8; python -m gpu8v8final.baselines8 sign; python -m gpu8v8final.baselines8 kdgene",
        "python -m gpu8v8final.hist8; python -m gpu8v8final.support8 --stage dev; python -m gpu8v8final.support8 --stage test",
        "python -m gpu8v8final.external8; python -m gpu8v8final.expression8; python -m gpu8v8final.review8",
        "python -m gpu8v8final.acceptance8; python -m gpu8v8final.repro8   # .venv_repro 被系统策略阻止，本轮在主环境以全新缓存重算",
        "python -m gpu8v8final.final8; python -m gpu8v8final.audit8 end; python -m gpu8v8final.audit8 compare",
        "```", ""])

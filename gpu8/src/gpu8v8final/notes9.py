"""投稿说明 of the SARA package: file list, changes relative to the 2026-09-26 revision, items the authors must
fill in, the evidence boundaries kept in the text, and the results of the checks (read from the reports)."""
from __future__ import annotations

import json

from .manuscript9 import DATE, STEM, TITLE, table_md
from .numbers9 import ROOT
from .supp9 import SUPP_STEM

AUDIT = ROOT.parent / "gpu8" / "submission9" / "audit"


def J(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def junit(p) -> dict:
    import xml.etree.ElementTree as ET
    if not p.exists():
        return {}
    r = ET.parse(p).getroot()
    s = r if r.tag == "testsuite" else r.find("testsuite")
    n = {k: int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    n["passed"] = n["tests"] - n["failures"] - n["errors"] - n["skipped"]
    n["failed_names"] = [c.get("name") for c in s.iter("testcase") if c.find("failure") is not None or c.find("error") is not None]
    n["reasons"] = sorted({(c.find("failure") if c.find("failure") is not None else c.find("error")).get("message", "")[:60]
                           for c in s.iter("testcase") if c.find("failure") is not None or c.find("error") is not None})
    return n


def compose() -> str:
    ver = J(ROOT / "build" / "verify9_report.json")
    bld = J(ROOT / "build" / "docx_build_report9.json")
    cmp_ = J(AUDIT / "audit9_comparison.json")
    note = J(AUDIT / "audit9_note.json")
    info = ver.get("_info", {})
    sub, rev, sup = bld.get("submission", {}), bld.get("review", {}), bld.get("supplement", {})
    terms = [["部署模型 / 候选 A（部署）", "SARA / SARA 排序器"], ["仅检索通道 LightGBM", "检索通道 LightGBM"],
             ["V7.1 直接融合（严格 500 重训）", "多模态直接融合"], ["基础排序器 B", "基础排序器"], ["F7 固定融合", "固定权重融合"],
             ["教师 $s_F$ 单独", "语义检索 $s_F$"], ["语义教师 / 序列学生", "语义模型 / 序列补位模型"],
             ["严格 500 / V6 500 / 约 950 桥接", "SARA 候选 / 检索前 500 / 扩展候选"], ["缺失感知 RRF", "可用性感知 RRF"],
             ["X2 / X1（历史任务）", "新关联任务 / 新疾病任务（时间外推）"], ["外部 618 个疾病", "618 个测试疾病的 GenCC/ClinVar 外部标签"],
             ["E1–E9、层级①②③", "按问题命名的分析（不再使用编号）"]]
    checks = [
        ["禁用词扫描（正文、补充材料、3 个 Word 文件、18 张图的文字、公开作图数据 CSV）",
         "0 命中" if not any(ver.get(k) for k in ver if k.startswith("1_")) else "见报告"],
        ["数字溯源（小数、百分比与千分位整数均由结果文件格式化生成，设计常数除外）",
         "全部可溯源" if not (ver.get("2_untraced_main") or ver.get("2_untraced_supp")) else "见报告"],
        ["格式（无带符号的零、无 nan、负号为 U+2212、无嵌套括号）",
         "通过" if not (ver.get("3_format_main") or ver.get("3_format_supp")) else "见报告"],
        [f"参考文献（正文 {info.get('refs_main')} 条、补充材料 {info.get('refs_supp')} 条；Crossref 核对、按首次引用编号、无重复 DOI）",
         "通过" if not (ver.get("5_references_main") or ver.get("5_references_supp")) else "见报告"],
        ["交叉引用（图 1–10 与表 1–3 按顺序首次引用；图面板、补充表、补充说明均存在；图注面板与图一致）",
         "通过" if not (ver.get("6_xrefs_main") or ver.get("6_xrefs_supp") or ver.get("9_legend_panels")) else "见报告"],
        [f"Word 文件（A4、连续行号、页码；正文公式 {sub.get('equations', 0)} 个行间 + {sub.get('inline_equations', 0)} 个行内，"
         f"OMML 对象 {sub.get('omml_objects', 0)} 个；审阅版嵌图 {rev.get('verified_images', 0)} 张；补充材料嵌图 "
         f"{sup.get('verified_images', 0)} 张）", "通过" if bld.get("all_ok") else "见报告"],
        ["配色的色觉缺陷模拟检查", "通过" if not ver.get("8_palette") else "见报告"],
    ]
    arch = ("V5/V6/V7.1 档案、2026-09-24 与 2026-09-26 的稿件、figures_v2 与 59 个既有代码模块均未改动"
            if not cmp_.get("changed_modules") else "见 audit9_comparison.json")
    led = note.get("content_check", {})
    tn, to = junit(ROOT / "build" / "pytest_new.xml"), junit(ROOT / "build" / "pytest_old.xml")
    lgbm = all("lightgbm" in r for r in to.get("reasons", [])) and to.get("reasons")
    test_line = (f"- 测试：新增 {tn.get('tests', 0)} 项中 {tn.get('passed', 0)} 项通过；既有 {to.get('tests', 0)} 项中 {to.get('passed', 0)} 项"
                 f"通过" + (f"，另 {to['tests'] - to['passed']} 项需要 LightGBM——当前可用的 Python 环境未安装 LightGBM，而项目自带"
                            "虚拟环境中的 pandas 动态库被 Windows 应用程序控制策略阻止，因此这些测试本次未能运行（它们覆盖的代码本次未改动）。"
                            if lgbm else "。"))
    parts = [
        f"# SARA 投稿包说明（{DATE[:4]}-{DATE[4:6]}-{DATE[6:]}）",
        f"**论文题目**　{TITLE}",
        "## 一、文件清单",
        "\n".join([
            f"- `{STEM}.docx`：投稿版。A4，连续行号，页码；要点、正文、方法、声明、参考文献之后依次为表 1–3 与图注；不含图（图件按期刊要求另行上传）。",
            f"- `{STEM.replace('投稿版', '审阅版')}.docx`：审阅版。正文相同，每个图注前嵌入 300 dpi 的图，便于通读。",
            f"- `{SUPP_STEM}.docx`：补充材料（补充说明 S1–S6、补充表 S1–S10、补充图 S1–S8）。",
            "- `figures/`：主图 10 张、补充图 8 张，均提供矢量 PDF、SVG 与 600 dpi PNG；`figures/source_data/` 为每个面板的作图数据（CSV）。",
            f"- `SARA_图集_{DATE}.pdf`：全部图件的合集（矢量）。",
            "- `markdown/`：正文与补充材料的 Markdown 源文件（与 Word 文件内容相同）。",
            "- `checks/`：校验报告（verify9_report.json、docx_build_report9.json）。",
        ]),
        "## 二、相对 2026-09-26 修订稿的主要改动",
        "\n".join([
            "1. **体例**：按 Briefings in Bioinformatics 的结构重写——摘要、关键词、要点 4 条；引言补充相关工作并以编号引用；结果 10 节，"
            "小标题为结论句；讨论含一段局限；方法 13 节；数据与代码可用性、资助、利益冲突、作者贡献、致谢与作者简介的占位；参考文献、表与图注置于文末。",
            "2. **方法命名与术语统一**：方法命名为 SARA（Semantic retrieval with Availability-aware RAnking），正文、图中文字与作图数据 CSV 的"
            "名称同步替换（对照见表 1），替换前后的数值自动核对一致。",
            "3. **新写入正文的结果**：U@10；与基础排序器、固定权重融合、单独语义检索、KDGene 与 SIGN 式图模型的比较；单/多关联、OMIM、"
            "阳性负担与治疗类关联分层；预算归因与扩展候选；逐关系名次流转与零信号查询；TreeSHAP 解释（新图 6）；表达特征消融；受控 GO "
            "缺失与关联监督冷（新图 7）；外部标签上对基础排序器的比较与文献独立子集；760 个疾病的单样本支持任务；表达队列的三种子重复；案例网络。",
            "4. **删去过程性内容**：版本沿革、编号化的实验与判定层级、运行编号、文件路径以及计算环境与会话的过程记录。",
            "5. **未进入最终模型的两种设计**（门控冷平衡纠偏器、序列补位模型）移入补充说明 S4、图 S8 与表 S8，并完整报告其结果。",
            "6. **据实修正**：复现改写为“同一计算环境中以全新缓存重算，候选名单与前十完全一致，排序分数最大差 1.2 × 10⁻⁷”；预设非劣界"
            "（−0.005）未达成的说明放在补充说明 S6，正文对多模态直接融合只写“性能相当”；外部评价集合更正为“618 个测试疾病的 GenCC/ClinVar "
            "外部标签（过滤排名、三种子平均）”；六类信息层齐全的基因占 13.8%。",
        ]),
        "表 1　术语对照（修订稿 → 投稿版）",
        table_md(["修订稿中的名称", "投稿版中的名称"], terms),
        "## 三、需要作者补充或确认的内容",
        "\n".join([
            "- 作者、单位、通讯作者与邮箱；短标题可按期刊字数要求调整。",
            "- 资助、利益冲突、作者贡献（建议按 CRediT）、致谢与作者简介（BIB 要求每位作者一段简介）。",
            "- 代码仓库与数据存档的链接或 DOI。",
            "- 若在研究或写作中使用了人工智能工具，按期刊政策在致谢或方法中说明。",
            "- Briefings in Bioinformatics 只接收英文稿件：本稿为按该刊体例组织的中文稿，正式投稿前需译为英文（术语对照见表 1）。",
        ]),
        "## 四、正文中保持的证据边界",
        "\n".join([
            "- 交叉验证折与外部标签集合在方法迭代中使用过，结果属于回顾性的内部验证证据（局限段第二点）。",
            "- SARA 相对多模态直接融合写作“性能相当”，不宣称更优；时间外推中 SARA 与检索通道 LightGBM 表现相当。",
            "- 表达队列只提供相关性证据；未标注候选不等于阴性。",
        ]),
        "## 五、校验结果",
        table_md(["检查项", "结果"], checks),
        "\n".join([
            f"- 档案核对：{arch}。`manifests_v8/run_ledger_v8.csv` 在运行既有测试时被其账本重建函数按原逻辑重写，内容与 2026-09-24 的"
            f"存档逐字节一致（SHA-256 {'相同' if led.get('identical') else '不同'}），仅文件时间戳变化。",
            test_line,
            "- 图 2e 的 UpSet 面板由已保存的作图数据重绘（中间缓存目录 caches_v8 已于 2026-09-29 13:08 被外部清空），数值与原图一致。",
            "- 通过 Word 自动化把 docx 导出为 PDF 做逐页目检的尝试两次均停滞，已终止；Word 文件的版式由程序逐项核对（见上表）。",
        ]),
    ]
    return "\n\n".join(parts) + "\n"


def write() -> str:
    p = ROOT / f"SARA_投稿说明_{DATE}.md"
    p.write_text(compose(), encoding="utf-8")
    return str(p)


if __name__ == "__main__":
    print(write())

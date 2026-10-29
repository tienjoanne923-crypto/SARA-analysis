"""投稿说明 of the second-revision SARA package: file list, changes relative to the 2026-09-29 version, how to work
with the MathType equations, items the authors must fill in, the evidence boundaries kept in the text and the results
of the checks (read from the reports)."""
from __future__ import annotations

import json
import re

from .build_docx10 import NOTES_STEM, REVIEW_STEM
from .manuscript10 import DATE, STEM, TITLE
from .manuscript9 import table_md
from .notes9 import J, junit
from .numbers9 import ROOT
from .supp10 import SUPP_STEM


def compose() -> str:
    ver = J(ROOT / "build" / "verify10_report.json")
    bld = J(ROOT / "build" / "docx_build_report10.json")
    wpdf = J(ROOT / "build" / "word_pdf10_report.json")
    info = ver.get("_info", {})
    sub, rev, sup = bld.get("submission", {}), bld.get("review", {}), bld.get("supplement", {})
    em, es = info.get("equations_main", {}), info.get("equations_supp", {})

    def ok(*keys):
        return "通过" if ver and not any(ver.get(k) for k in keys) else "见报告"
    checks = [
        ["禁用词与第一人称扫描（正文、补充材料、3 个 Word 文件、18 张图的文字、公开作图数据 CSV）",
         ok(*[k for k in ver if k.startswith("1_")])],
        ["数字溯源（小数、百分比与千分位整数均由结果文件格式化生成，设计常数除外）", ok("2_untraced_main", "2_untraced_supp")],
        ["数值格式（无带符号的零、无 nan、负号为 U+2212、无嵌套括号）", ok("3_format_main", "3_format_supp")],
        [f"参考文献（GB/T 7714—2015；正文 {info.get('refs_main', '—')} 条、补充材料 {info.get('refs_supp', '—')} 条；按首次引用"
         "顺序编号、上标引用、文献类型标识与期刊卷期页齐全、题名与 Crossref 一致、无重复 DOI）",
         ok("5_citations_main", "5_citations_supp", "5_gbt_main", "5_gbt_supp")],
        ["交叉引用（图 1–10 与表 1–3 按顺序首次引用；图面板、补充表、补充说明与“式（n）”均存在）",
         ok("6_xrefs_main", "6_xrefs_supp", "9_legend_panels")],
        [f"MathType 公式（正文 {em.get('display', '—')} 个行间、{em.get('inline', '—')} 个行内；补充材料 {es.get('display', '—')} "
         f"个行间、{es.get('inline', '—')} 个行内；全部为 Equation.DSMT4 对象，Word 原生公式 0 个）", ok("7_docx")],
        [f"Word 文件（A4、连续行号、页码；审阅版嵌图 {rev.get('verified_images', '—')} 张，补充材料嵌图 "
         f"{sup.get('verified_images', '—')} 张；编号域、交叉引用域与文献链接全部可解析）",
         "通过" if bld.get("all_ok") else "见报告"],
        ["缩写在正文首次出现处给出全称", ok("10_abbreviations")],
        ["标题为中性名词短语（章节标题不超过 24 字，不含评价性措辞）", ok("11_titles")],
        ["配色的色觉缺陷模拟检查", ok("8_palette")],
    ]
    pages = "；".join(f"{k} {v.get('pages')} 页" for k, v in wpdf.items() if isinstance(v, dict) and v.get("pages"))
    rt = J(ROOT / "build" / "mathtype" / "roundtrip10_documents.json")
    rs = J(ROOT / "build" / "mathtype" / "resize10.json")
    au = J(ROOT.parent / "gpu8" / "submission9" / "audit" / "audit10_comparison.json")
    n_rt = sum(v.get("objects", 0) for v in rt.values() if isinstance(v, dict))
    n_same = sum(v.get("identical_size", 0) for v in rt.values() if isinstance(v, dict))
    mt_line = (f"- MathType 可编辑性：由 MathType 对投稿版与补充材料中的全部 {n_rt} 个公式对象重新排版，全部读取成功，其中 {n_same} 个"
               f"排版后的尺寸与文件中完全一致（{'通过' if rt.get('ok') else '见报告'}）；表格与图注中的公式对象（{rs.get('harvested', '—')} "
               "种）由 MathType 按所在字号直接排版生成。")
    audit_line = ("- 档案核对：V5/V6/V7.1 档案、2026-09-24 与 09-26 的稿件、figures_v2、results_v8 与既有代码模块均与 2026-09-29 结束时"
                  f"的快照一致；2026-09-29 投稿包中的 {au.get('round9_deliverables_checked', '—')} 个文件与文件夹中的同名文件逐字节一致"
                  f"（{'通过' if au.get('untouched') else '见 audit10_comparison.json'}）；本轮只新增了第二轮修订的生成模块。")
    tn, to = junit(ROOT / "build" / "pytest_new10.xml"), junit(ROOT / "build" / "pytest_old10.xml")
    test_line = (f"- 测试：本轮新增 {tn.get('tests', 0)} 项中 {tn.get('passed', 0)} 项通过；既有 {to.get('tests', 0)} 项中 "
                 f"{to.get('passed', 0)} 项通过（LightGBM 已安装，上一轮因缺少 LightGBM 未能运行的测试本轮均已运行；会重写运行账本的"
                 "账本哈希测试不在本轮运行范围内）。")
    parts = [
        f"# SARA 投稿包说明（{DATE[:4]}-{DATE[4:6]}-{DATE[6:]}）",
        f"**论文题目**　{TITLE}",
        "## 一、文件清单",
        "\n".join([
            f"- `{STEM}.docx`：投稿版。A4，连续行号，页码；正文之后依次为声明、参考文献、表 1–3 与图注；不含图（图件按期刊要求"
            "另行上传）。",
            f"- `{REVIEW_STEM}.docx` 与同名 PDF：审阅版。正文相同，每个图注前嵌入 300 dpi 的图；PDF 由 Word 导出。",
            f"- `{SUPP_STEM}.docx` 与同名 PDF：补充材料（补充说明 S1–S6、补充表 S1–S10、补充图 S1–S8）。",
            f"- `SARA_参考文献_{DATE}.ris`：正文参考文献的 RIS 文件，可导入 EndNote、NoteExpress 或 Zotero。",
            "- `figures/`：主图 10 张、补充图 8 张（矢量 PDF、SVG 与 600 dpi PNG）；`figures/source_data/` 为每个面板的作图数据（CSV）。",
            f"- `SARA_图集_{DATE}.pdf`：全部图件的合集，图题与正文一致。",
            "- `markdown/`：正文、补充材料与本说明的 Markdown 源文件（公式为 LaTeX 源码，便于检索与比对）。",
            "- `checks/`：校验报告（verify10_report.json、docx_build_report10.json）。",
        ]),
        "## 二、相对 2026-09-29 投稿版的修改",
        "\n".join([
            "1. **题目与结构**：题目改为“" + TITLE + "”。全文按引言—材料与方法—结果—讨论—结论的顺序重组；引言按研究背景、研究空白、"
            "本文贡献与全文安排展开；讨论分为可用性感知设计的作用、与现有方法的比较、局限性与展望三节。",
            "2. **标题**：各级小标题改为客观名词短语，结论性表述移至各节首句；图 1–10、表 1–3 与全部补充图表的标题同步改为描述"
            "研究对象与分析内容的名词短语。",
            "3. **公式**：方法部分新增 10 个编号公式——信息量加权的表型相似度、表型迁移、重启随机游走、多来源倒数名次融合、多正例"
            "排序损失与功能锚损失、语义分数、可用性感知的倒数名次融合、召回率与候选上界——并逐一与实现代码核对；补充材料中门控"
            "纠偏器的公式编号为 (S1)。正文以“式（n）”引用公式，式中符号在公式后逐一说明。",
            "4. **MathType 公式**：全部行间公式与行内符号（含表格与图注中的符号）均为可编辑的 MathType 公式对象；行间公式采用 "
            "MathType 的标准版式（MTDisplayEquation 样式，公式居中、编号右对齐）。向量与矩阵按 ISO 80000-2 排为粗斜体，函数名"
            "与说明性下标为正体。",
            "5. **参考文献**：按 GB/T 7714—2015 顺序编码制著录，正文为上标引用（如 [1-4]、[2,5]），文献表标注文献类型（[J]、[C]、"
            "[M]、[R]、[EB/OL] 等），期刊文献著录年、卷（期）与页码，电子文献注明引用日期；每个上标编号链接到对应文献条目，"
            "DOI 与网址可直接点击。",
            "6. **行文**：缩写在正文首次出现时给出中文名与英文全称；全文统一使用无人称表述；各节首句交代目的与主要发现、末句承接"
            "下一节；术语统一（如“冷启动阳性”“可用性感知 RRF”“检索通道 LightGBM”）；数值区间统一用“～”，统计量 $P$ 为斜体。",
            "7. **数值与结论不变**：全部数值仍由结果文件格式化生成（见数字溯源检查）；与上一版相比，没有任何数值或结论被修改。",
        ]),
        "## 三、MathType 公式的使用",
        "\n".join([
            "- 双击任一公式即可在 MathType 中编辑；表格（9 磅）与图注（10.5 磅）中的公式已按相应字号保存尺寸设置，编辑后大小不变。",
            "- 行间公式的编号为 MathType 编号域，正文中的“式（n）”为引用该编号的交叉引用域。增删行间公式后，可在 MathType 选项卡"
            "中执行“更新公式编号”，或在 Word 中全选（Ctrl+A）后按 F9 更新域，编号与正文引用将同步更新。",
            "- 如期刊要求使用 Word 自带公式，可在 MathType 选项卡中用“转换公式”一次性转换。",
        ]),
        "## 四、需要作者补充或确认的内容",
        "\n".join([
            "- 作者、单位、通讯作者与邮箱；短标题可按期刊字数要求调整。",
            "- 资助、利益冲突、作者贡献（建议按 CRediT）、致谢与作者简介（BIB 要求每位作者一段简介）。",
            "- 代码仓库与数据存档的链接或 DOI。",
            "- 若在研究或写作中使用了人工智能工具，按期刊政策在致谢或方法中说明。",
            "- Briefings in Bioinformatics 只接收英文稿件：本稿为按该刊体例组织的中文稿，正式投稿前需译为英文；若改投中文期刊，"
            "参考文献已符合 GB/T 7714—2015。",
        ]),
        "## 五、正文中保持的证据边界",
        "\n".join([
            "- 交叉验证折与外部标签集合在方法迭代中使用过，结果属于回顾性的内部验证证据（4.3 节第二点）。",
            "- SARA 相对多模态直接融合写作“性能相当”，不宣称更优；预设非劣界未达成的说明见补充说明 S6。",
            "- 时间外推中 SARA 与检索通道 LightGBM 表现相当；复现为同一计算环境中以全新缓存重算。",
            "- 表达队列只提供相关性证据；未标注候选不等同于阴性。",
        ]),
        "## 六、校验结果",
        table_md(["检查项", "结果"], checks),
        "\n".join([
            f"- 版式目检：Word 导出的 PDF（{pages or '见 checks'}）逐页检查了公式位置与编号对齐、表格、图注与参考文献的版式。",
            mt_line,
            test_line,
            audit_line,
        ]),
    ]
    return "\n\n".join(parts) + "\n"


def write() -> str:
    p = ROOT / f"{NOTES_STEM}.md"
    p.write_text(compose(), encoding="utf-8")
    return str(p)


if __name__ == "__main__":
    print(write())

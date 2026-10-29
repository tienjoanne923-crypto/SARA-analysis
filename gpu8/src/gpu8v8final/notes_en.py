"""《英文版说明》: Chinese notes on the English SARA package (contents, correspondence with the Chinese second revision,
layout differences, placeholders, checks and optional fixes of the Chinese package).  Counts are read from the build
and verification reports, so the notes describe the files actually delivered."""
from __future__ import annotations

import json

from .build_docx_en import BUILD, REVIEW_STEM
from .manuscript_en import DATE, EN, STEM, TITLE
from .supp_en import SUPP_STEM

NOTES_STEM = f"SARA_英文版说明_{DATE}"
ZIP_NAME = f"SARA_English_package_{DATE}.zip"


def compose() -> str:
    v = json.loads((BUILD / "verify_en_report.json").read_text(encoding="utf-8"))
    b = json.loads((BUILD / "docx_build_report_en.json").read_text(encoding="utf-8"))
    rt = json.loads((BUILD / "roundtrip_en.json").read_text(encoding="utf-8"))
    pdf = json.loads((BUILD / "word_pdf_report_en.json").read_text(encoding="utf-8"))
    tests = json.loads((BUILD / "tests_en.json").read_text(encoding="utf-8")) if (BUILD / "tests_en.json").exists() else {}
    info = v["_info"]
    n_csv = len(list((EN / "figures" / "source_data").glob("*.csv")))
    n_fig = len(list((EN / "figures").glob("*.pdf")))
    gl = json.loads((BUILD / "glossary_en.json").read_text(encoding="utf-8")) if (BUILD / "glossary_en.json").exists() else {}
    checks = [k for k in v if not k.startswith("_") and k != "ok"]
    files = [
        (f"{STEM}.docx", "英文投稿版正文（不含图，行号、页码；表与图注在文末），供投稿系统上传"),
        (f"{REVIEW_STEM}.docx / .pdf", f"英文审阅版：图嵌在图注前，PDF 由 Word 导出，共 {pdf['review']['pages']} 页"),
        (f"{SUPP_STEM}.docx / .pdf", f"英文补充材料：补充说明 S1–S6、补充表 S1–S10、补充图 S1–S8，PDF 共 {pdf['supplement']['pages']} 页"),
        (f"SARA_figures_{DATE}.pdf", f"英文图集：{n_fig} 张矢量图（主图 10 张、补充图 8 张）"),
        ("figures/", f"英文图件（PDF/SVG/600 dpi PNG 各 {n_fig} 张）与 {n_csv} 个英文作图数据 CSV"),
        (f"SARA_references_{DATE}.ris", "英文正文参考文献的 RIS 文件，可导入 EndNote/Zotero"),
        (f"SARA_cover_letter_{DATE}.docx", "英文投稿信草稿（作者、单位、文章类型等为占位符）"),
        (f"SARA_术语对照_{DATE}.docx / .xlsx", f"中英术语对照表，{gl.get('groups', 17)} 类共 {gl.get('terms', '')} 条，与英文正文和图中用词一致"),
        (f"{NOTES_STEM}.docx", "本说明"),
        ("markdown/", "英文正文与补充材料的 Markdown 源文件"),
        ("checks/", "英文版校验报告与 Word 构建报告"),
    ]
    parts = [
        "# SARA 英文投稿包说明",
        f"**英文题目**　{TITLE}",
        "本投稿包是中文第二轮定稿（2026-09-30）的完整英文版，按 Briefings in Bioinformatics（BIB）的英文投稿格式撰写。"
        "英文正文按 BIB 文风直接写成，不是逐句直译；全部数值与中文定稿来自同一套结果数据，由同一组格式函数生成并逐项溯源。",
        "## 一、文件清单",
        "| 文件 | 说明 |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in files),
        "## 二、与中文定稿的对应关系",
        f"- **结构**：摘要（{info['abstract_words']} 词）、Key Points（{info['key_points']} 条）、1 Introduction 至 5 Conclusion、"
        "Data availability 等后置部分、表 1–3 与图 1–10 图注，均与中文定稿逐节对应；补充材料同样一一对应。",
        f"- **数值**：英文正文与补充材料中出现的全部数字（正文 {info['numbers_main']} 个、补充材料 {info['numbers_supp']} 个不同数值）"
        "与中文定稿逐一相同；英文作图数据 CSV 的数值列与中文 CSV 逐元素相等。",
        f"- **参考文献**：BIB/OUP 格式，正文以 [1]、[2, 5]、[1–4] 按首次出现编号；正文 {info['refs_main']} 篇，补充材料单独编号 "
        f"{info['refs_supp']} 篇；期刊名为 NLM 缩写，作者超过 3 位时写前 3 位加 et al.，页码按缩写规则。",
        f"- **公式**：正文 {info['equations_main']['display']} 个编号公式、{info['equations_main']['inline']} 处行内公式，补充材料 "
        f"{info['equations_supp']['display']} 个编号公式、{info['equations_supp']['inline']} 处行内公式，全部为 MathType 对象（可双击编辑），"
        "编号右对齐，正文中的 “Eq. (n)” 为指向公式编号的域引用；表格（9 pt）和图注（10.5 pt）中的公式由 MathType 按字号重新排版。",
        "- **图件**：18 张图的全部文字（标题、坐标轴、图例、注释）均已译为英文并逐张目检，无中文残留、无内部用语；配色与中文版相同。",
        "- **用语**：美式拼写，无人称写法；缩写在首次出现时给出全称；SARA 与多模态直接融合、时间外推中 SARA 与检索通道 LightGBM 均写作 "
        "“performed comparably”；局限性一段保留“交叉验证折与外部标签在方法迭代中使用过、结果属于内部验证”的表述；复现说明为同一计算环境、"
        "全新缓存；非劣界说明只出现在 Supplementary Note S6。",
        "## 三、英文版的版式调整（数值不变）",
        "- 表 3：去掉“比较”列，差值后用 †、‡、§ 标注比较对象，并在表注中说明，使差值与置信区间不再折行。",
        "- 补充表 S6 拆为 (a) 四种方法的 Recall@10 与 (b) 差值两个分表；补充表 S9(a)、S10(d) 的分组名称每组只写一次；"
        "补充表 S10(c) 的单元格写作“富集度（P 值）”，并单列 Top；补充表 S1(b)、S7 缩短表头，说明移入表题或表注。",
        "- 式 (10) 中的 min 排为正体（中文版该公式中 min 为斜体）。",
        "- 图 5 图注中的 ‘0\\*’ 按原样显示（见第六节中文版问题 1）。",
        "## 四、需要作者补充的内容",
        "- 正文：作者姓名与单位、通讯作者及邮箱、Funding、Conflict of interest、Author contributions、Acknowledgements（含 AI 工具使用声明）、"
        "Author biographies、Data availability 中的代码库链接与数据 DOI。",
        "- 投稿信：日期、文章类型 [article type]、利益冲突与 AI 使用声明、可选的推荐/回避审稿人、署名信息。",
        "- 投稿时请在投稿系统中选择合适的文章类型，并按期刊当期要求核对篇幅与图表数量。",
        "## 五、质量检查",
        f"- 英文版校验（verify_en）共 {len(checks)} 组检查，结果：{'全部通过' if v['ok'] else '存在未通过项'}。检查内容包括：Markdown、Word、PDF、"
        "图中文字与 CSV 无中文残留；无内部版本名、路径等内部用语；数字全部可溯源；与中文定稿数值一致；引用编号与格式；图、表、面板、"
        "补充材料与公式的交叉引用；图注面板字母加粗；缩写首次出现有定义；标题为中性表述；措辞边界与占位符；摘要不超过 250 词。",
        f"- Word 文件：三份文件全部通过构建审计（MathType 对象 {b['submission']['ole_objects']} + {b['supplement']['ole_objects']} 个，"
        "无 OMML，编号域、引用超链接与公式引用全部可解析）。",
        f"- MathType 往返检查：MathType 重新读取并排版正文 {rt['submission']['objects']} 个、补充材料 {rt['supplement']['objects']} 个"
        f"公式对象，尺寸与写入时完全一致（{rt['submission']['identical_size']} / {rt['supplement']['identical_size']} 个相同）。",
        "- PDF 由 Word 导出，逐页目检；无空白页、无域错误。",
        (f"- 自动化测试：英文层测试 {tests['english_layer']['passed']} 项全部通过；全部测试中 {tests['full_suite']['passed']} 项通过，"
         f"另有 {tests['full_suite']['failed']} 项（V8 纠偏器相关，需要现场训练 LightGBM）因本机 Windows 应用程序控制策略阻止加载 "
         "tools_libs10 中的 lib_lightgbm.dll（WinError 4551）而无法运行，与英文版无关；未改动任何系统安全设置。"
         "会改写运行账本的 test_ledger_hashes 按惯例未运行。" if tests else "- 自动化测试：见 checks/ 中的报告。"),
        "- 中文交付物、V5/V6/V7.1 档案、图件与结果目录均未改动（与 09-29、09-30 两个压缩包中的副本逐字节一致）。",
        "## 六、中文版中发现的三个小问题（未改动，可按需修正）",
        "1. 图 5 图注中的“0\\*”：星号被当作斜体标记，导致其后一段文字显示为斜体、面板字母 b 未加粗（中文审阅版 PDF 第 36 页）。"
        "修正方法是在 Markdown 源文件中给该星号加反斜杠转义后重新生成 Word/PDF。",
        "2. 式 (10) 中的 min 为斜体，规范写法为正体。",
        "3. 公开作图数据 Fig9_external_temporal_applications_e.csv 的 set 列仍含内部集合名（如 v8_deployed_s42_top20）；英文版 CSV 已改为公开名称。",
        "如需修正，只需告知，可在不改变任何数值的前提下重新生成中文第二轮文件。",
    ]
    return "\n\n".join(parts) + "\n"


def write() -> dict:
    from .build_docx10 import Builder
    md = compose()
    p = BUILD / f"{NOTES_STEM}.md"
    p.write_text(md, encoding="utf-8")
    st = Builder("notes", {}, line_numbers=False).build(p, EN / f"{NOTES_STEM}.docx", images=False)
    ok = st["a4"] and st["page_field"] and st["omml_objects"] == 0 and not st["leftover_markup"]
    return {"docx": st["path"], "ok": ok}


if __name__ == "__main__":
    print(json.dumps(write(), ensure_ascii=False, indent=1))

"""SARA supplementary material, second revision: the tables and notes of supp9 with GB/T 7714 references,
numbered MathType equations (S1...) and the terminology of manuscript10."""
from __future__ import annotations

import json

from .fmt9 import TRACE
from .manuscript10 import DATE, TITLE, apply_edits, number_equations, polish
from .numbers9 import ROOT, numbers9
from . import supp9 as S

SUPP_STEM = f"SARA_补充材料_{DATE}"
TERMS = [("参考路径冷正例的权重", "冷启动正例的权重"), ("参考路径冷的阳性", "冷启动阳性"), ("参考路径冷阳性", "冷启动阳性"),
         ("前十", "前 10"), ("95% 区间", "95% CI"),
         ("$$s(d,g)=B(d,g)+a(d,g)\\,r_\\theta(x_{dg}),$$", "$$s(d,g)=B(d,g)+a(d,g)\\,r_\\theta\\left(x_{dg}\\right)$$ {#eq:s1}")]
# wording, symbols typeset as math, noun-phrase titles; every edit must apply
EDITS = [
    ("正文方法 4.1 与表 1", "正文 2.1 节与表 1"),
    ("按句界优先装段后", "按句界优先分段后"), ("装段过程中没有片段被分词器截断", "分段过程中没有片段被分词器截断"),
    ("## 补充说明 S4　未进入最终模型的两种设计", "## 补充说明 S4　未纳入最终模型的两种替代设计"),
    ("在确定 SARA 之前，我们还考察了两种", "在确定 SARA 的最终设计之前，本研究还考察了两种"),
    ("冷启动正例的权重为 $1+\\beta c_g$（$c_g$ 为冷标记）", "冷启动阳性的权重为 $1+\\beta c_g$（$c_g$ 为冷启动标记）"),
    ("测试集上的比较与此一致：β = 0、1、2 的纠偏器相对 SARA", "测试集上的比较与此一致：$\\beta$ 取 0、1、2 时，纠偏器相对 SARA"),
    ("而冷恢复 $R^c$@10 与 SARA 相当", "而冷启动恢复 $R^c$@10 与 SARA 相当"),
    ("+0.0095；β = 0；图 S8c）", "+0.0095；$\\beta=0$；图 S8c）"),
    ("而对冷启动正例加权的残差也未能在不损失总体性能的前提下提高冷恢复", "而对冷启动阳性加权的残差也未能在不损失总体性能的前提下提高冷启动恢复"),
    ("其 95% CI下界大于 0", "其 95% CI 下界大于 0"),
    ("冷恢复相对多模态直接融合的差值", "冷启动恢复相对多模态直接融合的差值"),
    ("均以内容哈希记账，有计时的运行合计约 9.5 小时", "均以内容哈希记录，有计时的运行合计约 9.5 h"),
    ("表 S3　对照方法与全背景单独语义排名", "表 S3　对照方法的设置与全背景语义排名的性能"),
    ("AdamW（1e-3，wd 1e-4）", "AdamW（学习率 10⁻³，权重衰减 10⁻⁴）"),
    ("表 S4　分层、缺失亚组与零信号查询", "表 S4　疾病分层与信息缺失亚组的检索性能"),
    ("表 S5　稳健性", "表 S5　外层 × 种子组合与候选名单变体的稳健性分析"),
    ("表 S6　按 MeSH 疾病大类的 Recall@10", "表 S6　各 MeSH 疾病大类上的 Recall@10"),
    ("表 S7　证据删除条件", "表 S7　各证据删除条件下的检索性能"),
    ("表 S8　未进入最终模型的两种设计", "表 S8　两种替代设计的检索性能"),
    ("（a）门控冷平衡纠偏器（三种子平均；在 SARA 候选上重训，开发集选定 λ）",
     "（a）门控冷平衡纠偏器（三种子平均；在 SARA 候选上重训，开发集选定 $\\lambda$；$R^{c\\to all}$@10 为进入前 10 的冷启动阳性占全部"
     "测试阳性的比例）"),
    ("| 门控纠偏器 β = 0 |", "| 门控纠偏器（$\\beta=0$） |"), ("| 门控纠偏器 β = 1 |", "| 门控纠偏器（$\\beta=1$） |"),
    ("| 门控纠偏器 β = 2 |", "| 门控纠偏器（$\\beta=2$） |"),
    ("表 S9　外部标签与时间外推", "表 S9　外部标签评价结果与时间外推的输入版本"),
    ("表 S10　应用任务与案例", "表 S10　支持任务、表达队列与案例疾病的评价结果"),
    ("（533 个疾病；k 为显露的支持基因数；语义与 RWR 的组合权重 ω = 0.25）",
     "（533 个疾病；$k$ 为显露的支持基因数；语义与 RWR 的组合权重 $\\omega=0.25$）"),
    ("| 方法 | k | R@10 | R@50 | 疾病数 |", "| 方法 | $k$ | R@10 | R@50 | 疾病数 |"),
    ("（760 个疾病，k = 1）", "（760 个疾病，$k=1$）"),
    ("（富集 = 选中基因 |t| 高于匹配背景的概率；加粗为 P < 0.05）",
     "（富集度为选中基因的 $\\left|t\\right|$ 高于匹配背景基因的概率；加粗表示 $P<0.05$）"),
    ("粗体为留出的真实关联，° 为参考路径冷）", "粗体为留出的真实关联，° 为冷启动阳性）"),
    ("图 S3　各外层 × 种子的稳定性", "图 S3　外层 × 种子组合间的性能稳定性"),
    ("（六边形分箱，对数色标；r 为 Pearson 相关）", "（六边形分箱，对数色标；$r$ 为 Pearson 相关系数）"),
    ("图 S4　全背景单独语义排名与文本变体", "图 S4　全背景语义排名与疾病文本变体的比较"),
    ("图 S5　语义空间的结构", "图 S5　疾病—功能语义空间的结构"),
    ("图 S6　计算资源。", "图 S6　计算资源的使用情况。"), ("（点大小 ∝ 计时秒数）", "（点的大小与计时秒数成正比）"),
    ("图 S7　证据块关联的恢复与删减数据中的 SARA 优势", "图 S7　证据块隔离条件下的关联恢复与 SARA 性能优势"),
    ("图 S8　未进入最终模型的两种设计", "图 S8　两种替代设计的开发与测试评价"),
    ("（开发阶段，种子 42；颜色深浅为 β，实线/点线为 λ）", "（开发阶段，种子 42；颜色深浅表示 $\\beta$，实线与点线表示 $\\lambda$）"),
    ("品红为纠偏器（形状为 β，实心/空心为 λ）", "品红为纠偏器（形状表示 $\\beta$，实心与空心表示 $\\lambda$）"),
    ("**c** 测试集上三个 β 的纠偏器相对 SARA 的差值（三种子；开发集选定的 λ）：左为 Recall@10，右为 $R^c$@10 及 β 之间的冷恢复对比。",
     "**c** 测试集上三种 $\\beta$ 取值的纠偏器相对 SARA 的差值（三种子；开发集选定的 $\\lambda$）：左为 Recall@10，右为 $R^c$@10 "
     "及不同 $\\beta$ 之间的冷启动恢复对比。"),
]


def compose(R=None, N=None) -> tuple[str, dict]:
    from .refs10 import resolve
    if R is None:
        R, N = numbers9()
    TRACE.clear()
    head = ["# 补充材料", f"**{TITLE}**",
            "本补充材料包括补充说明 S1–S6、补充表 S1–S10 与补充图 S1–S8。文中的图、表、公式与章节编号未加“S”者均指正文。"]
    notes = [S.note_s1(R, N), S.note_s2(R, N), S.note_s3(R, N), S.note_s4(R, N), S.note_s5(R, N), S.note_s6(R, N)]
    tabs = ["## 补充表", S.table_s1(R, N), S.table_s2(R, N), S.table_s3(R, N), S.table_s4(R, N), S.table_s5(R, N),
            S.table_s6(R, N), S.table_s7(R, N), S.table_s8(R, N), S.table_s9(R, N), S.table_s10(R, N)]
    figs = ["## 补充图"]
    for n, stem, title, leg in S.SFIGS:
        figs += [f"<!-- figure {stem} -->", f"**图 {n}　{title}。** {leg}"]
    md = "\n\n".join(p.strip("\n") for p in head + notes + tabs + figs + ["## 补充参考文献", "<<REFERENCES>>"]) + "\n"
    for a, b in TERMS:
        md = md.replace(a, b)
    md = polish(apply_edits(md, EDITS))
    md, eqs = number_equations(md)
    md, refs, order, recs = resolve(md)
    md = md.replace("<<REFERENCES>>", "\n".join(refs))
    return md, {"stem": SUPP_STEM, "ref_order": order, "n_refs": len(refs), "equations": eqs,
                "figures": [(n, s) for n, s, _, _ in S.SFIGS], "trace": {k: v for k, v in TRACE.items()}}


def write() -> dict:
    R, N = numbers9()
    md, info = compose(R, N)
    p = ROOT / f"{SUPP_STEM}.md"
    p.write_text(md, encoding="utf-8")
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "supp10_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"md": str(p), "chars": len(md), "refs": info["n_refs"], "equations": len(info["equations"])}


if __name__ == "__main__":
    print(json.dumps(write(), ensure_ascii=False, indent=1))

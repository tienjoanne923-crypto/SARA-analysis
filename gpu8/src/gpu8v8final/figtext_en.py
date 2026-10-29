"""Chinese -> English wording of the SARA figures and their public source-data tables.

``translate(plain)`` maps one plain figure string (mathtext CJK wrappers removed, public Chinese wording of
publabel9) to English: exact entries (``EXACT``, loaded from figtext_en.json) first, then pattern rules for the
repetitive families (folds, seeds, ranks, MeSH categories with counts or codes, method names with counts, ...),
then structural fallbacks (line by line, "label（n）", "a · b").  Strings that still contain CJK are reported by
``missing``; the English figures are not written while any remain.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CJK = re.compile(r"[⺀-鿿　-〿＀-￯]")
DATA = Path(__file__).with_name("figtext_en.json")

# ------------------------------------------------------------------ fixed terminology
METHOD = {
    "SARA": "SARA", "SARA 排序器": "SARA ranker", "SARA 候选": "SARA candidates",
    "检索通道 LightGBM": "Retrieval-channel LightGBM", "检索通道 LGB": "Retrieval LGB", "检索 LGB": "Retrieval LGB",
    "多模态直接融合": "Multimodal direct fusion", "直接融合": "Direct fusion",
    "多模态直接融合（检索前 500）": "Multimodal direct fusion (retrieval top 500)",
    "基础排序器": "Base ranker", "固定权重融合": "Fixed-weight fusion",
    "固定权重融合（检索前 500）": "Fixed-weight fusion (retrieval top 500)",
    "语义检索 $s_F$": "Semantic retrieval $s_F$", "多模态语义核心 $r$": "Multimodal semantic core $r$",
    "SapBERT 零训练相似度": "Untrained SapBERT similarity", "语义模型序列分数": "Semantic-model sequence score",
    "SIGN 式": "SIGN-style", "SIGN 式图模型": "SIGN-style GNN", "KDGene": "KDGene",
    "门控纠偏器": "Gated corrector", "序列补位模型": "Sequence imputation model", "语义模型": "Semantic model",
    "检索通道排序器": "Retrieval-channel ranker", "检索前 500": "Retrieval top 500", "扩展候选（约 950）": "Extended candidates (~950)",
    "可用性感知 RRF": "Availability-aware RRF", "普通 RRF": "Standard RRF",
}
# MeSH top-level categories: Chinese full name -> (figure label, text name)
MESH = {
    "感染": ("Infections", "infections"), "肿瘤": ("Neoplasms", "neoplasms"),
    "肌肉骨骼疾病": ("Musculoskeletal", "musculoskeletal diseases"),
    "消化系统疾病": ("Digestive", "digestive system diseases"), "口腔与牙齿疾病": ("Stomatognathic", "stomatognathic diseases"),
    "呼吸系统疾病": ("Respiratory", "respiratory tract diseases"), "耳鼻喉疾病": ("Otorhinolaryngologic", "otorhinolaryngologic diseases"),
    "神经系统疾病": ("Nervous system", "nervous system diseases"), "眼部疾病": ("Eye", "eye diseases"),
    "泌尿生殖系统疾病": ("Urogenital", "urogenital diseases"), "心血管疾病": ("Cardiovascular", "cardiovascular diseases"),
    "血液与淋巴系统疾病": ("Hemic & lymphatic", "hemic and lymphatic diseases"),
    "先天性、遗传性与新生儿疾病": ("Congenital & hereditary", "congenital, hereditary and neonatal diseases"),
    "皮肤与结缔组织疾病": ("Skin & connective tissue", "skin and connective tissue diseases"),
    "营养与代谢性疾病": ("Nutritional & metabolic", "nutritional and metabolic diseases"),
    "内分泌系统疾病": ("Endocrine", "endocrine system diseases"), "免疫系统疾病": ("Immune", "immune system diseases"),
    "病理状态、体征与症状": ("Pathological conditions", "pathological conditions, signs and symptoms"),
    "化学物质诱发疾病": ("Chemically induced", "chemically induced disorders"), "创伤与损伤": ("Wounds & injuries", "wounds and injuries"),
    "精神障碍": ("Mental disorders", "mental disorders"),
}
MESH_SHORT = {
    "感染": "Infections", "肿瘤": "Neoplasms", "肌骨": "Musculoskeletal", "消化": "Digestive", "口腔": "Stomatognathic",
    "呼吸": "Respiratory", "耳鼻喉": "ENT", "神经": "Nervous", "眼": "Eye", "泌尿生殖": "Urogenital", "心血管": "Cardiovascular",
    "血液/淋巴": "Hemic/lymphatic", "先天/遗传": "Congenital", "皮肤/结缔": "Skin/connective", "营养/代谢": "Nutr./metabolic",
    "内分泌": "Endocrine", "免疫": "Immune", "病理状态": "Pathological", "化学诱发": "Chemically induced", "创伤": "Wounds",
    "精神": "Mental",
}
EVIDENCE = {"实验": "experimental", "高通量": "high-thr.", "系统发育": "phylogenetic", "管理员": "curator",
            "电子": "electronic", "计算": "computational", "作者声明": "author"}
CONTRAST = {"病灶/对照": "lesional vs control", "病灶/非病灶配对": "lesional vs non-lesional (paired)"}


def mesh_label(zh: str) -> str:
    return MESH[zh][0]


# ------------------------------------------------------------------ pattern rules (full match on the plain string)
def _m(x: str) -> str:
    return METHOD.get(x, x)


def lc(s: str) -> str:
    """lower-case the first letter unless the first word is an acronym or a name (SARA, KDGene, ...)."""
    w = s.split(" ", 1)[0]
    return s if (len(w) > 1 and (w[1].isupper() or w.startswith("SARA"))) else s[:1].lower() + s[1:]


PATTERNS = [
    (r"外层 (\d)", lambda g: f"Fold {g[1]}"),
    (r"外层 (\d) · 对(.+)", lambda g: f"Fold {g[1]} · vs {lc(_m(g[2]))}"),
    (r"外层 (\d)（最佳轮次 (\d+)–(\d+)）", lambda g: f"Fold {g[1]} (best rounds {g[2]}–{g[3]})"),
    (r"全部 · 对(.+)", lambda g: f"All · vs {lc(_m(g[1]))}"),
    (r"种子 (\d+)", lambda g: f"Seed {g[1]}"),
    (r"全背景对照 · 种子 (\d+)", lambda g: f"Background controls · seed {g[1]}"),
    (r"第 (\d+) 名", lambda g: f"Rank {g[1]}"),
    (r"(\d+) 次", lambda g: f"{g[1]} runs"),
    (r"([\d,]+) \((\d+)% 无 PMID\)", lambda g: f"{g[1]} ({g[2]}% without PMID)"),
    (r"([A-Z]{2,4})（(.+)）", lambda g: f"{g[1]} ({EVIDENCE[g[2]]})" if g[2] in EVIDENCE else None),
    (r"(.+?) ([\d,]+)", lambda g: f"{MESH_SHORT[g[1]]} {g[2]}" if g[1] in MESH_SHORT else None),
    (r"(.+?)（((?:C|F)\d\d)）", lambda g: (f"{MESH[g[1]][0]} ({g[2]})" if g[1] in MESH else
                                          f"{MESH_SHORT[g[1]]} ({g[2]})" if g[1] in MESH_SHORT else None)),
    (r"(.+?)（([\d,]+)）", lambda g: f"{MESH[g[1]][0]} ({g[2]})" if g[1] in MESH else None),
    (r"(GSE\d+)\n(.+)", lambda g: f"{g[1]}\n{CONTRAST[g[2]]}" if g[2] in CONTRAST else None),
    (r"前 10 命中：SARA (\d+) · 检索 LGB (\d+)", lambda g: f"Top-10 hits: SARA {g[1]} · retrieval LGB {g[2]}"),
    (r"MESH:(D\d+) · (.+?)：(增益最大|损失最大) · 留出阳性 (\d+) 个 · 前 10 命中：SARA (\d+) / 检索 LGB (\d+)",
     lambda g: (f"MESH:{g[1]} · {'all diseases' if g[2] == '全体' else MESH[g[2]][1]}: "
                f"{'largest gain' if g[3] == '增益最大' else 'largest loss'} · {g[4]} held-out positives · "
                f"top-10 hits: SARA {g[5]} / retrieval LGB {g[6]}")),
    (r"对(.+?) · (.+?)（([\d,]+)）", lambda g: f"vs {lc(_m(g[1]))} · {SUBSET[g[2]]} ({g[3]})"),
]
SUBSET = {"合并": "union", "GenCC": "GenCC", "ClinVar": "ClinVar", "独立证据子集": "literature-independent"}
_PAT = [(re.compile(p, re.S), f) for p, f in PATTERNS]

EXACT: dict[str, str] = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else {}


def _structural(plain: str) -> str | None:
    if "\n" in plain:
        parts = [translate(x) if x.strip() else x for x in plain.split("\n")]
        if all(not CJK.search(p) for p in parts):
            return "\n".join(parts)
    m = re.fullmatch(r"(.+?)（([\d,]+(?:\.\d+)?%?)）", plain, re.S)
    if m:
        head = translate(m.group(1))
        if not CJK.search(head):
            return f"{head} ({m.group(2)})"
    m = re.fullmatch(r"(.+?)\n([\d,]+%?)", plain, re.S)
    if m:
        head = translate(m.group(1))
        if not CJK.search(head):
            return f"{head}\n{m.group(2)}"
    return None


def translate(plain: str) -> str:
    if not plain or not CJK.search(plain):
        return plain
    if plain in EXACT:
        return EXACT[plain]
    if plain in METHOD:
        return METHOD[plain]
    if plain in MESH:
        return MESH[plain][0]
    if plain in MESH_SHORT:
        return MESH_SHORT[plain]
    for rx, f in _PAT:
        m = rx.fullmatch(plain)
        if m:
            try:
                out = f(m)
            except KeyError:
                out = None
            if out is not None:
                return out
    out = _structural(plain)
    return out if out is not None else plain


def missing(strings) -> list[str]:
    return sorted({s for s in strings if CJK.search(translate(s))})

"""Save hook of the English SARA figures.

``save_en`` replaces the save function of every figure module: the figure is first brought to the public Chinese
wording exactly as for the Chinese package (pubstyle9: Fig 3 surgery, relabelling, layout fix-ups), then every drawn
string -- titles, labels, legends, annotations and the output of every tick formatter -- is translated by
``figtext_en.translate``.  The rendered text is scanned (no CJK left, no internal terms, small text, overlaps of the
translated strings) and the figure is written as 600-dpi PNG, PDF and SVG to manuscript_sara/english/figures, with
the public source-data tables translated (headers and string cells; every numeric column identical).
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FuncFormatter

from . import figtext_en as T
from . import pubstyle9 as P9
from .pubstyle8 import MM

OUT = P9.ROOT / "english" / "figures"
SRC = OUT / "source_data"
LOG: dict = {}
FORBIDDEN_EN = [r"(?<![A-Za-z0-9])V[5-8](?:\.\d)?(?![0-9])", r"V71", r"\bdeployed\b", r"\bteacher\b", r"\bstudent\b",
                r"channel[- _]only", r"strict ?500", r"\bbridge\b", r"[Cc]andidate A\b", r"(?<![A-Za-z0-9])E[1-9](?![0-9A-Za-z])",
                r"(?<![A-Za-z0-9])F7(?![0-9])", r"(?<![A-Za-z0-9])X[12](?![0-9A-Za-z])", r"run_id", r"results_v8", r"gpu8",
                r"\.parquet", r"\.json", r"[A-Za-z]:\\", r"[Nn]on-?inferior", r"Windows", r"venv", r"V8T__",
                r"evaluation_conditioned", r"channel_lgb", r"v71_", r"F7_", r"\bv6\b", r"\bV6\b", r"deployed", r"v8_",
                r"_v6\b", r"_v8\b"]
FORBIDDEN_EN_RX = re.compile("|".join(FORBIDDEN_EN))
# internal set / method keys that survive as string values in a few source-data tables
CSV_KEYS = [(re.compile(r"v8_deployed_s(\d+)"), r"SARA_seed\1"), (re.compile(r"v71_direct_strict_s(\d+)"), r"multimodal_fusion_seed\1"),
            (re.compile(r"channel_lgb_v6_s(\d+)"), r"retrieval_LGB_seed\1"), (re.compile(r"\bF7_v6\b"), "fixed_weight_fusion_top500"),
            (re.compile(r"\bchannel_lgb_v6\b"), "retrieval_LGB"),
            (re.compile(r"(?<![A-Za-z])deployed(?![A-Za-z])"), "SARA"),
            (re.compile(r"(?<=_)channel(?![A-Za-z])|(?<![A-Za-z])channel(?=_)"), "retrieval_LGB"),
            (re.compile(r"_v8\b"), "_SARA")]


def _csv_value(v):
    if not isinstance(v, str):
        return v
    v = T.translate(v)
    for rx, rep in CSV_KEYS:
        v = rx.sub(rep, v)
    return v


def en_fonts() -> None:
    """Latin faces for mathtext too (the Chinese figures used DengXian for \\mathrm/\\mathbf); DejaVu Sans supplies
    symbols that Arial lacks (e.g. the proportionality sign)."""
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial"})


def translate_text(s: str) -> str:
    if not s or not s.strip() or not T.CJK.search(s):
        return s
    plain, _ = P9.unwrap_mt(s)
    return T.translate(plain)


def _wrap(fmt: FuncFormatter) -> FuncFormatter:
    f0 = fmt.func

    def f(x, pos=None):
        return translate_text(f0(x, pos))
    new = FuncFormatter(f)
    new._sara_translated = True
    return new


def translate_figure(fig) -> list[tuple[str, str]]:
    changes = []
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for which in ("major", "minor"):
                fmt = axis.get_major_formatter() if which == "major" else axis.get_minor_formatter()
                if isinstance(fmt, FixedFormatter):
                    new = [translate_text(x) for x in fmt.seq]
                    changes += [(a, b) for a, b in zip(fmt.seq, new) if a != b]
                    fmt.seq = new
                elif isinstance(fmt, FuncFormatter) and not getattr(fmt, "_sara_translated", False):
                    (axis.set_major_formatter if which == "major" else axis.set_minor_formatter)(_wrap(fmt))
    fig.canvas.draw()
    for t in fig.findobj(Text):
        s = t.get_text()
        new = translate_text(s)
        if new != s:
            t.set_text(new)
            changes.append((s, new))
    return changes


def scan(fig, translated: set[str]) -> dict:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    cjk, forb, small, boxes, allb = [], [], [], [], []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s or not s.strip() or not t.get_visible():
            continue
        if T.CJK.search(s):
            cjk.append(s[:60])
        h = [m.group(0) for m in FORBIDDEN_EN_RX.finditer(s)]
        if h:
            forb.append((s[:60], h))
        if t.get_fontsize() < 4.8:
            small.append((s[:40], round(t.get_fontsize(), 2)))
        try:
            bb = t.get_window_extent(rend)
        except Exception:
            continue
        allb.append((s, bb))
        if s in translated:
            boxes.append((s, bb))
    overlaps = []
    for a, ba in boxes:
        for b, bb in allb:
            if a == b:
                continue
            ix = max(0.0, min(ba.x1, bb.x1) - max(ba.x0, bb.x0))
            iy = max(0.0, min(ba.y1, bb.y1) - max(ba.y0, bb.y0))
            if ix * iy > 0.25 * min(ba.width * ba.height, bb.width * bb.height) and ix * iy > 4:
                overlaps.append((a[:40], b[:40]))
    return {"cjk": cjk, "forbidden": forb, "small_text": small, "overlaps": overlaps[:60]}


def public_table_en(df: pd.DataFrame) -> pd.DataFrame:
    d = P9.public_table(df)
    d = d.rename(columns={c: _csv_value(str(c)) for c in d.columns})
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].map(_csv_value)
    return d


# ------------------------------------------------------------------ figure-specific layout fix-ups (after translation)
def _texts(fig):
    """(text artist, relative height of its bottom edge) for every non-empty string."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H = fig.bbox.height
    for t in fig.findobj(Text):
        if t.get_text():
            try:
                yield t, t.get_window_extent(r).y0 / H
            except Exception:
                continue


def _fix_fig1(fig):
    """panel c: the bin labels of the GO axis ran into each other (also in the Chinese figure); rotate them."""
    for ax in fig.axes:
        if ax.get_xlabel() == "GO annotations (gene–term pairs)":
            for t in ax.get_xticklabels():
                t.set_rotation(35)
                t.set_ha("right")
                t.set_rotation_mode("anchor")


def _fix_fig2(fig):
    short = {"Phenotype RWR": "Phen. RWR", "Compatibility": "Compat."}           # the small source boxes of panel a
    rows = {"Reference-path cold": "Reference-path\ncold", "GO removal (controlled)": "GO removal\n(controlled)",
            "Supervision cold (controlled)": "Supervision cold\n(controlled)",
            "Withdrawn + GO removed": "Withdrawn +\nGO removed"}                   # row (tick) labels of panel d
    for ax in fig.axes:
        labels = [t.get_text() for t in ax.get_yticklabels()]
        if "Reference-path cold" in labels:
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([rows.get(s, s) for s in labels], fontsize=5.8, linespacing=0.95)
    for t, y in list(_texts(fig)):
        s = t.get_text()
        if s in short and y > 0.7:
            t.set_text(short[s])
        elif s in rows:
            t.set_text(rows[s])
            t.set_fontsize(5.8)


def _fix_fig4(fig):
    two = {"Retrieval top 500": "Retrieval\ntop 500", "SARA candidates (500)": "SARA\ncandidates (500)",
           "Extended candidates (~950)": "Extended\ncandidates (~950)"}
    for ax in fig.axes:
        if ax.get_ylabel() == "Recall@10 (macro-average)":
            labels = [t.get_text() for t in ax.get_xticklabels()]
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels([two.get(s, s) for s in labels])


def _fix_fig7(fig):
    """the English mask-set legend of panel a is too wide for either subplot: one row above the two subplots."""
    left = next((a for a in fig.axes if a.get_xlabel() == "GO terms"), None)
    for ax in fig.axes:
        leg = ax.get_legend()
        if left is not None and leg is not None and any(t.get_text().startswith("Unmasked") for t in leg.get_texts()):
            handles = getattr(leg, "legend_handles", None) or leg.legendHandles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            left.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=5.4,
                        frameon=False, handlelength=1.2, columnspacing=0.9, borderaxespad=0.0)
            break


def _retick(fig, mapping: dict, **kw):
    """replace tick labels (which are regenerated from the formatter at every draw) on any axis that shows them."""
    for ax in fig.axes:
        for axis, get, setl, sett in ((ax.xaxis, ax.get_xticklabels, ax.set_xticklabels, ax.set_xticks),
                                      (ax.yaxis, ax.get_yticklabels, ax.set_yticklabels, ax.set_yticks)):
            labels = [t.get_text() for t in get()]
            if any(s in mapping for s in labels):
                sett(axis.get_ticklocs())
                setl([mapping.get(s, s) for s in labels], **kw)


def _fix_fig8(fig):
    _retick(fig, {"Annotation rows": "Annotation\nrows"})
    for t, _ in list(_texts(fig)):
        if t.get_text() == "Annotation rows":
            t.set_text("Annotation\nrows")
    for ax in fig.axes:          # panel b, right: its right-hand tick labels fell outside the figure (also in Chinese)
        if ax.get_xlabel() == "Blocks, cumulative by decreasing size":
            p = ax.get_position()
            ax.set_position([p.x0, p.y0, p.width * 0.86, p.height])


def _fix_fig10(fig):
    names = {"Cardiomyopathy, Hypertrophic, Familial": "Familial hypertrophic cardiomyopathy",
             "Alzheimer Disease": "Alzheimer's disease", "Cell Transformation, Neoplastic": "Neoplastic cell transformation",
             "Retinitis Pigmentosa": "Retinitis pigmentosa", "Liver Neoplasms": "Liver neoplasms"}
    for t, _ in list(_texts(fig)):
        if t.get_text() in names:
            t.set_text(names[t.get_text()])


FIXUPS_EN: dict = {"Fig1_knowledge_landscape": _fix_fig1, "Fig2_method_overview": _fix_fig2,
                   "Fig4_budget_fusion_flow": _fix_fig4,
                   "Fig7_annotation_supervision": _fix_fig7, "Fig8_evidence_sources": _fix_fig8,
                   "Fig10_case_studies": _fix_fig10}


def save_en(fig, stem: str, sources: dict | None = None, meta: dict | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    n_surg = P9.surgery(fig, stem)
    P9.relabel_figure(fig)
    if stem in P9.FIXUPS:
        P9.FIXUPS[stem](fig)
    changes = translate_figure(fig)
    if stem in FIXUPS_EN:
        FIXUPS_EN[stem](fig)
    rep = scan(fig, {b for _, b in changes})
    LOG[stem] = {"changes": changes, **rep}
    out = {"stem": stem, "files": [], "translated": len(changes), "surgery_removed": n_surg,
           "cjk_left": len(rep["cjk"]), "forbidden_hits": len(rep["forbidden"]), "overlaps": len(rep["overlaps"])}
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, **kw)
        out["files"].append(p.name)
    w, h = fig.get_size_inches() / MM
    out["size_mm"] = [round(w, 1), round(h, 1)]
    for k, df in (sources or {}).items():
        pub = public_table_en(df)
        assert P9._numeric_equal(df.drop(columns=[c for c in df.columns if str(c) in P9.DROP_COLS], errors="ignore"),
                                 pub), f"public table changed numbers: {stem}_{k}"
        p = SRC / f"{stem}_{k}.csv"
        pub.to_csv(p, index=False, encoding="utf-8-sig")
        out["files"].append(f"source_data/{p.name}")
    if meta:
        out.update(meta)
    plt.close(fig)
    return out

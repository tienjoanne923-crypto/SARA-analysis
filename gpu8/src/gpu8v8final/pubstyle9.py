"""Save hook of the SARA submission figures.

``save9`` replaces ``pubstyle8.save`` in every figure module: it removes the non-inferiority
annotations of Fig 3, rewrites every drawn string to the public wording (publabel9), scans the
rendered text for internal terms, and writes PNG (600 dpi), PDF and SVG plus two sets of
source-data tables -- the internal ones (identical in format to figures_v2, read by the text
generator) and public ones (method/arm keys and headers in public wording, numbers identical).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FuncFormatter

from .publabel9 import PUBLIC, forbidden_hits, relabel
from .pubstyle8 import MM, mt

ROOT = Path(r"D:\信\manuscript_sara")
OUT = ROOT / "figures"
SRC_PUB = OUT / "source_data"
SRC_INT = OUT / "source_data_internal"
MT_RX = re.compile(r"\$\\math(rm|bf)\{([^{}$]*)\}\$")
LOG: dict = {}                 # stem -> relabel / scan report (read by pubrun9)

ARM_PUBLIC = {
    "main": "完整模型", "e5_none": "删 GO·不补位", "e5_mlp": "两层补位", "e5_linear": "线性补位",
    "e5_nn": "序列近邻转移", "e5_teacher_sS": "语义模型序列投影", "e5_mlp_shuffled": "两层补位（配对打乱）",
    "e5_protst": "ProtST 输入补位", "e6_cold": "撤出关联监督", "e6_cold_gomask": "撤出监督 + 删 GO",
    "e4_layer1": "训练可见来源排除", "e4_layer1_random": "训练可见来源排除·匹配随机",
    "e4_layer2": "评价条件化删除", "e4_layer2_random": "评价条件化删除·匹配随机",
    "e4_blocks": "证据块隔离", "e4_blocks_random": "证据块隔离·匹配随机", "e4_experimental": "仅实验类证据",
    "e4_no_tas_nas": "去 TAS/NAS", "e4_no_mutant": "去突变/遗传相互作用", "e4_rescore_no_ic": "去 IC（重打分）",
    "e4_rescore_iea_only": "仅 IEA（重打分）", "x1": "新疾病任务", "x2": "新关联任务",
    "joint_v8_w0.25": "SARA 语义 + RWR", "joint_v6": "基线语义 + RWR", "rwr_v8list": "RWR（SARA 候选）",
    "rwr_v6list": "RWR（检索前 500）", "qwalk_v8list": "量子游走（SARA 候选）", "qwalk_v6list": "量子游走（检索前 500）",
    "sem_v8": "SARA 语义", "sem_v6": "基线语义", "A": "SARA 排序器", "corrector": "门控纠偏器",
    "teacher": "语义模型", "student": "序列补位模型", "base_ranker": "基础排序器",
}
DROP_COLS = {"run_id", "ctx", "path", "file"}


# ------------------------------------------------------------------ text helpers
def unwrap_mt(s: str) -> tuple[str, bool]:
    return MT_RX.sub(lambda m: m.group(2), s), "\\mathbf{" in s


SIGNED_ZERO = re.compile(r"(?<![\d.])[-−+](0\.0+)(?!\d)")


def fix_zero(s: str) -> str:
    """A value that rounds to zero is printed without a sign (e.g. heat-map cells "−0.0" -> "0.0")."""
    return SIGNED_ZERO.sub(r"\1", s)


ASCII_MINUS = re.compile(r"(?<![\w.)\]}])-(?=\d)")


def fix_minus(s: str) -> str:
    """ASCII hyphen used as a minus sign -> U+2212 (outside mathtext; ranges such as 2023-12-31 are untouched)."""
    parts = re.split(r"(\$[^$]*\$)", s)
    return "".join(p if (len(p) > 1 and p.startswith("$") and p.endswith("$")) else ASCII_MINUS.sub("−", p) for p in parts)


def relabel_text(s: str) -> str:
    if not s or not s.strip():
        return s
    plain, bold = unwrap_mt(s)
    new = fix_zero(fix_minus(relabel(plain)))
    if new == plain:
        return s
    return mt(new, bold) if "$" in new else new


def _wrap_func(fmt: FuncFormatter) -> FuncFormatter:
    f0 = fmt.func

    def f(x, pos=None):
        return relabel_text(f0(x, pos))
    new = FuncFormatter(f)
    new._sara_relabelled = True
    return new


def relabel_figure(fig) -> list[tuple[str, str]]:
    changes = []
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            before = [t.get_text() for t in axis.get_ticklabels()]
            for which in ("major", "minor"):
                fmt = axis.get_major_formatter() if which == "major" else axis.get_minor_formatter()
                if isinstance(fmt, FixedFormatter):
                    fmt.seq = [relabel_text(x) for x in fmt.seq]
                elif isinstance(fmt, FuncFormatter) and not getattr(fmt, "_sara_relabelled", False):
                    (axis.set_major_formatter if which == "major" else axis.set_minor_formatter)(_wrap_func(fmt))
            changes += [(a, relabel_text(a)) for a in before if a and relabel_text(a) != a]
    fig.canvas.draw()
    for t in fig.findobj(Text):
        s = t.get_text()
        new = relabel_text(s)
        if new != s:
            t.set_text(new)
            changes.append((s, new))
    return changes


def surgery(fig, stem: str) -> int:
    """Fig 3: remove the non-inferiority band and margin lines (the verdict words are relabelled)."""
    if not stem.startswith("Fig3"):
        return 0
    n = 0
    for ax in fig.axes:
        for p in list(ax.patches):
            try:
                if to_hex(p.get_facecolor()) == "#f5eeeb":
                    p.remove()
                    n += 1
            except (ValueError, AttributeError):
                pass
        for ln in list(ax.lines):
            try:
                if to_hex(ln.get_color()) == "#c9b5ae":
                    ln.remove()
                    n += 1
            except (ValueError, AttributeError):
                pass
    return n


def _visible_texts(fig):
    for t in fig.findobj(Text):
        s = t.get_text()
        if s and s.strip() and t.get_visible():
            yield t, s


def scan_figure(fig, relabelled: set[str]) -> dict:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    hits, small, boxes = [], [], []
    for t, s in _visible_texts(fig):
        plain = unwrap_mt(s)[0]
        h = forbidden_hits(plain)
        if h:
            hits.append((plain, h))
        if SIGNED_ZERO.search(plain):
            hits.append((plain, ["signed zero"]))
        if t.get_fontsize() < 4.8:
            small.append((plain[:40], round(t.get_fontsize(), 2)))
        if s in relabelled:
            try:
                boxes.append((plain, t.get_window_extent(rend)))
            except Exception:
                pass
    overlaps = []
    allb = []
    for t, s in _visible_texts(fig):
        try:
            allb.append((unwrap_mt(s)[0], t.get_window_extent(rend)))
        except Exception:
            pass
    for a, ba in boxes:
        for b, bb in allb:
            if a == b:
                continue
            ix = max(0.0, min(ba.x1, bb.x1) - max(ba.x0, bb.x0))
            iy = max(0.0, min(ba.y1, bb.y1) - max(ba.y0, bb.y0))
            if ix * iy > 0.25 * min(ba.width * ba.height, bb.width * bb.height) and ix * iy > 4:
                overlaps.append((a[:30], b[:30]))
    return {"forbidden": hits, "small_text": small, "overlaps": overlaps[:40]}


# ------------------------------------------------------------------ public source data
def _pub_value(v):
    if isinstance(v, str):
        if v in PUBLIC:
            return PUBLIC[v][0]
        if v in ARM_PUBLIC:
            return ARM_PUBLIC[v]
        return relabel(v)
    return v


def public_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.drop(columns=[c for c in df.columns if str(c) in DROP_COLS], errors="ignore").copy()
    ren = {}
    for c in d.columns:
        cs = str(c)
        ren[c] = PUBLIC[cs][0] if cs in PUBLIC else (ARM_PUBLIC.get(cs) or relabel(cs))
    d = d.rename(columns=ren)
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].map(_pub_value)
    return d


def _numeric_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    na = a.select_dtypes(include=[np.number]).to_numpy()
    nb = b.select_dtypes(include=[np.number]).to_numpy()
    return na.shape == nb.shape and bool(np.array_equal(na, nb, equal_nan=True))


# ------------------------------------------------------------------ figure-specific layout fix-ups (after relabelling)
def _fix_fig5(fig):
    W, H = fig.get_size_inches() / MM
    for t in fig.findobj(Text):
        if t.get_text().startswith("冷：无参考边"):
            t.set_transform(fig.transFigure)
            t.set_position((4 / W, 1 - 68.0 / H))
            t.set_ha("left")


FIXUPS = {"Fig5_knowledge_strata": _fix_fig5}


# ------------------------------------------------------------------ the hook
def save9(fig, stem: str, sources: dict | None = None, meta: dict | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    SRC_PUB.mkdir(parents=True, exist_ok=True)
    SRC_INT.mkdir(parents=True, exist_ok=True)
    n_surg = surgery(fig, stem)
    changes = relabel_figure(fig)
    if stem in FIXUPS:
        FIXUPS[stem](fig)
    scan = scan_figure(fig, {b for _, b in changes})
    out = {"stem": stem, "files": [], "relabelled": len(changes), "surgery_removed": n_surg,
           "forbidden_hits": len(scan["forbidden"])}
    LOG[stem] = {"changes": changes, **scan}
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, **kw)
        out["files"].append(p.name)
    w, h = fig.get_size_inches() / MM
    out["size_mm"] = [round(w, 1), round(h, 1)]
    for k, df in (sources or {}).items():
        pi = SRC_INT / f"{stem}_{k}.csv"
        df.to_csv(pi, index=False, encoding="utf-8-sig")
        pdf_ = public_table(df)
        assert _numeric_equal(df.drop(columns=[c for c in df.columns if str(c) in DROP_COLS], errors="ignore"),
                              pdf_), f"public table changed numbers: {stem}_{k}"
        pp = SRC_PUB / f"{stem}_{k}.csv"
        pdf_.to_csv(pp, index=False, encoding="utf-8-sig")
        out["files"].append(f"source_data/{pp.name}")
    if meta:
        out.update(meta)
    plt.close(fig)
    return out

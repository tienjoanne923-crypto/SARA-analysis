"""Render the round-five manuscript Markdown into a Word document.

Typesetting contract inherited from the previous rounds of this project:
SimSun throughout, black text only, three-line tables, figures inserted at
6.75 inch width with their captions, and an explicit validation pass that
counts paragraphs, tables and images so a silently truncated build fails.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from . import paths
from .hashing import atomic_write_json

BODY_FONT = "SimSun"
LATIN_FONT = "Times New Roman"
BODY_PT = 10.5
BLACK = RGBColor(0, 0, 0)


def set_run(run, size=BODY_PT, bold=False, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = BLACK
    run.font.name = LATIN_FONT
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:ascii"), LATIN_FONT)
    rf.set(qn("w:hAnsi"), LATIN_FONT)
    rf.set(qn("w:eastAsia"), BODY_FONT)
    return run


def add_paragraph(doc, text="", size=BODY_PT, bold=False, align=None,
                  space_after=6, first_line_indent=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.35
    if align is not None:
        p.alignment = align
    if first_line_indent:
        p.paragraph_format.first_line_indent = Pt(first_line_indent)
    for seg, is_bold, is_italic in parse_inline(text):
        set_run(p.add_run(seg), size=size, bold=bold or is_bold, italic=is_italic)
    return p


def parse_inline(text: str):
    """Split **bold**, *italic* and `code` into styled segments."""
    out = []
    pattern = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)")
    for part in pattern.split(text or ""):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            out.append((part[2:-2], True, False))
        elif part.startswith("*") and part.endswith("*"):
            out.append((part[1:-1], False, True))
        elif part.startswith("`") and part.endswith("`"):
            out.append((part[1:-1], False, False))
        else:
            out.append((part, False, False))
    return out


def set_cell_border(cell, top=False, bottom=False, sz=8):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for tag, on in (("top", top), ("bottom", bottom)):
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:val"), "single" if on else "nil")
        if on:
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), "000000")
        borders.append(el)


def add_table(doc, rows: list[list[str]]):
    """Three-line table: rule above the header, below the header, and at the end."""
    if not rows:
        return None
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
            val = row[j] if j < len(row) else ""
            for seg, b, it in parse_inline(val):
                set_run(p.add_run(seg), size=9, bold=(i == 0) or b, italic=it)
            set_cell_border(cell, top=(i == 0), bottom=(i in (0, len(rows) - 1)))
    return t


def parse_markdown_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for ln in lines:
        if re.match(r"^\s*\|[\s:|-]+\|\s*$", ln):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def build(md_path: Path, out_path: Path, figure_dir: Path | None = None) -> dict:
    text = Path(md_path).read_text(encoding="utf-8")
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.1)
        section.right_margin = Inches(1.1)
    style = doc.styles["Normal"]
    style.font.name = LATIN_FONT
    style.font.size = Pt(BODY_PT)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)

    lines = text.split("\n")
    i = 0
    stats = {"paragraphs": 0, "tables": 0, "images": 0, "headings": 0,
             "equations": 0, "missing_figures": []}
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, parse_markdown_table(block))
            stats["tables"] += 1
            add_paragraph(doc, "", space_after=4)
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level, title = len(m.group(1)), m.group(2)
            size = {1: 15, 2: 13, 3: 11.5, 4: 10.5}[level]
            p = add_paragraph(doc, title, size=size, bold=True, space_after=8)
            p.paragraph_format.space_before = Pt(10 if level <= 2 else 6)
            stats["headings"] += 1
            i += 1
            continue

        if stripped.startswith("$$"):
            block = [stripped]
            i += 1
            while i < len(lines) and not lines[i].strip().endswith("$$"):
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
                i += 1
            eq = " ".join(block).replace("$$", "").strip()
            p = add_paragraph(doc, eq, size=10, align=WD_ALIGN_PARAGRAPH.CENTER,
                              space_after=8)
            stats["equations"] += 1
            continue

        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if m:
            caption, rel = m.group(1), m.group(2)
            src = (figure_dir / Path(rel).name) if figure_dir else Path(rel)
            if src.exists():
                doc.add_picture(str(src), width=Inches(6.5))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                stats["images"] += 1
            else:
                stats["missing_figures"].append(str(src))
                add_paragraph(doc, f"[图缺失：{src.name}]", size=9,
                              align=WD_ALIGN_PARAGRAPH.CENTER)
            i += 1
            continue

        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            body = re.sub(r"^([-*]|\d+\.)\s+", "", stripped)
            p = add_paragraph(doc, body, space_after=3)
            p.paragraph_format.left_indent = Pt(18)
            stats["paragraphs"] += 1
            i += 1
            continue

        add_paragraph(doc, stripped, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                      first_line_indent=21)
        stats["paragraphs"] += 1
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))

    check = Document(str(out_path))
    stats["verified_paragraphs"] = len(check.paragraphs)
    stats["verified_tables"] = len(check.tables)
    stats["path"] = str(out_path)
    stats["bytes"] = out_path.stat().st_size
    stats["source_markdown"] = str(md_path)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the manuscript to DOCX.")
    ap.add_argument("--markdown", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--figure_dir", default="")
    args = ap.parse_args(argv)

    md_dir = paths.GPU8_ROOT.parent / "manuscript_round5_gpu8"
    md = Path(args.markdown) if args.markdown else \
        sorted(md_dir.glob("*.md"))[-1]
    out = Path(args.out) if args.out else md.with_suffix(".docx")
    figdir = Path(args.figure_dir) if args.figure_dir else paths.FIGURE_DIR
    stats = build(md, out, figdir)
    atomic_write_json(paths.REPORT_DIR / "docx_build_report.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

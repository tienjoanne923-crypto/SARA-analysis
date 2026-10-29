"""PDF export of the Word files by Word itself (tools/mathtype/export_pdf_word.ps1: separate hidden instance on a copy of
the file).  The review and supplement PDFs are deliverables; the submission PDF is only rendered for inspection.
Page counts go to ``build/word_pdf10_report.json``; pages are rendered to PNG with PyMuPDF for the visual check."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from .build_docx10 import REVIEW_STEM
from .manuscript10 import STEM
from .numbers9 import ROOT
from .supp10 import SUPP_STEM

SCRIPT = Path(r"D:\信\gpu8\submission9\tools\mathtype\export_pdf_word.ps1")
QA = ROOT / "build" / "pdf10_word"


def export(docx: Path, pdf: Path, timeout: int = 900) -> dict:
    r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                        "-InputPath", str(docx), "-PdfPath", str(pdf)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0 or not pdf.exists():
        raise RuntimeError(f"Word export failed for {docx.name}:\n{(r.stdout + r.stderr)[-2000:]}")
    m = re.search(r"Pages=(\d+) InlineShapes=(\d+)", r.stdout)
    return {"pdf": str(pdf), "pages": int(m.group(1)) if m else None, "inline_shapes": int(m.group(2)) if m else None}


def run(render: bool = True) -> dict:
    jobs = {"投稿版": (ROOT / f"{STEM}.docx", QA / "submission_word.pdf"),
            "审阅版": (ROOT / f"{REVIEW_STEM}.docx", ROOT / f"{REVIEW_STEM}.pdf"),
            "补充材料": (ROOT / f"{SUPP_STEM}.docx", ROOT / f"{SUPP_STEM}.pdf")}
    rep = {}
    for name, (docx, pdf) in jobs.items():
        pdf.parent.mkdir(parents=True, exist_ok=True)
        rep[name] = export(docx, pdf)
        if render:
            sys.path.append(r"D:\信\gpu8\tools_libs10")
            from .qa10 import render as png
            out = png(pdf, dpi=90) if pdf.parent == QA else png(_qa_copy(pdf), dpi=90)
            rep[name]["png_dir"], rep[name]["blank_pages"] = out["png_dir"], out["blank_pages"]
    (ROOT / "build" / "word_pdf10_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


def _qa_copy(pdf: Path) -> Path:
    """render deliverable PDFs from a copy in the QA folder (the PNG folders stay out of the delivery folder)."""
    import shutil
    QA.mkdir(parents=True, exist_ok=True)
    dst = QA / pdf.name
    shutil.copy2(pdf, dst)
    return dst


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=1))

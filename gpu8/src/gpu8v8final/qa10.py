"""Visual QA of the Word files: LibreOffice (headless, private profile) converts each .docx to PDF and PyMuPDF renders
the pages to PNG for inspection; page counts and blank pages are reported to ``build/qa10_report.json``."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from .numbers9 import ROOT

sys.path.append(r"D:\信\gpu8\tools_libs10")                 # PyMuPDF
SOFFICE = Path(r"D:\tools\LibreOffice\program\soffice.exe")
PROFILE = "file:///D:/tools/lo_profile"
OUT = ROOT / "build" / "pdf10"


def to_pdf(docx: Path, outdir: Path = OUT, timeout: int = 600) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(SOFFICE), "--headless", "--norestore", f"-env:UserInstallation={PROFILE}", "--convert-to", "pdf",
                    "--outdir", str(outdir), str(docx)], capture_output=True, timeout=timeout, check=False)
    pdf = outdir / (docx.stem + ".pdf")
    if not pdf.exists() or pdf.stat().st_mtime < docx.stat().st_mtime:
        raise RuntimeError(f"LibreOffice did not write {pdf}")
    return pdf


def render(pdf: Path, dpi: int = 80, pages=None) -> dict:
    import fitz
    doc = fitz.open(str(pdf))
    pngdir = pdf.parent / (pdf.stem + "_png")
    pngdir.mkdir(exist_ok=True)
    blank, words = [], []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        words.append(len(text))
        if not re.sub(r"[\d\s]", "", text) and not page.get_images():     # line and page numbers only
            blank.append(i + 1)
        if pages is None or (i + 1) in pages:
            page.get_pixmap(dpi=dpi).save(str(pngdir / f"p{i + 1:03d}.png"))
    return {"pdf": str(pdf), "pages": len(doc), "blank_pages": blank, "png_dir": str(pngdir)}


def run(names: list[str]) -> dict:
    rep = {}
    for n in names:
        pdf = to_pdf(ROOT / n)
        rep[n] = render(pdf)
    (ROOT / "build" / "qa10_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1:]), ensure_ascii=False, indent=1))

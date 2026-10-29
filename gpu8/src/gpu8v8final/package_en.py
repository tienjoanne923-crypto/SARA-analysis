"""Zip package of the English SARA submission (``english/SARA_English_package_<date>.zip``).

Documents, PDFs, figure book, RIS, cover letter, terminology table and notes at the top level; the English figures
(PDF/SVG/PNG) with their source-data CSVs under figures/; the Markdown sources under markdown/; the verification and
Word build reports under checks/ (local file paths removed).  Internal figure reports are not included.
"""
from __future__ import annotations

import json
import re
import zipfile

from .build_docx_en import BUILD, REVIEW_STEM
from .manuscript_en import DATE, EN, STEM
from .notes_en import NOTES_STEM, ZIP_NAME
from .supp_en import SUPP_STEM

TOP = [f"{STEM}.docx", f"{REVIEW_STEM}.docx", f"{REVIEW_STEM}.pdf", f"{SUPP_STEM}.docx", f"{SUPP_STEM}.pdf",
       f"SARA_figures_{DATE}.pdf", f"SARA_references_{DATE}.ris", f"SARA_cover_letter_{DATE}.docx",
       f"SARA_术语对照_{DATE}.docx", f"SARA_术语对照_{DATE}.xlsx", f"{NOTES_STEM}.docx"]


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k not in ("path", "pdf", "png_dir", "stdout")}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, str) and re.search(r"[A-Za-z]:\\", obj):
        return "<local path>"
    return obj


def build() -> dict:
    root = f"SARA_English_package_{DATE}"
    out = EN / ZIP_NAME
    names = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in TOP:
            p = EN / n
            assert p.exists(), n
            z.write(p, f"{root}/{n}")
            names.append(n)
        for p in sorted((EN / "figures").glob("*")):
            if p.suffix in (".pdf", ".svg", ".png"):
                z.write(p, f"{root}/figures/{p.name}")
                names.append(f"figures/{p.name}")
        for p in sorted((EN / "figures" / "source_data").glob("*.csv")):
            z.write(p, f"{root}/figures/source_data/{p.name}")
            names.append(f"figures/source_data/{p.name}")
        for n in (f"{STEM}.md", f"{SUPP_STEM}.md"):
            z.write(EN / n, f"{root}/markdown/{n}")
            names.append(f"markdown/{n}")
        for n in ("verify_en_report.json", "docx_build_report_en.json", "roundtrip_en.json"):
            data = _clean(json.loads((BUILD / n).read_text(encoding="utf-8")))
            z.writestr(f"{root}/checks/{n}", json.dumps(data, ensure_ascii=False, indent=1))
            names.append(f"checks/{n}")
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
    return {"zip": str(out), "files": len(names), "bytes": out.stat().st_size, "testzip": bad,
            "figures": sum(1 for n in names if n.startswith("figures/") and not n.startswith("figures/source_data")),
            "csv": sum(1 for n in names if n.startswith("figures/source_data/"))}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=1))

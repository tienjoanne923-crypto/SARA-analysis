"""Zip of the second-revision SARA submission package (public files only)."""
from __future__ import annotations

import json
import zipfile

from .build_docx10 import NOTES_STEM, REVIEW_STEM
from .manuscript10 import DATE, STEM
from .numbers9 import FIG, ROOT
from .supp10 import SUPP_STEM

PDF = ROOT / "build" / "pdf10_word"


def build() -> dict:
    name = f"SARA_投稿包_{DATE}"
    out = ROOT / f"{name}.zip"
    files = [(ROOT / f"{STEM}.docx", f"{STEM}.docx"),
             (ROOT / f"{REVIEW_STEM}.docx", f"{REVIEW_STEM}.docx"), (ROOT / f"{REVIEW_STEM}.pdf", f"{REVIEW_STEM}.pdf"),
             (ROOT / f"{SUPP_STEM}.docx", f"{SUPP_STEM}.docx"), (ROOT / f"{SUPP_STEM}.pdf", f"{SUPP_STEM}.pdf"),
             (ROOT / f"{NOTES_STEM}.docx", f"{NOTES_STEM}.docx"),
             (ROOT / f"SARA_图集_{DATE}.pdf", f"SARA_图集_{DATE}.pdf"),
             (ROOT / f"SARA_参考文献_{DATE}.ris", f"SARA_参考文献_{DATE}.ris"),
             (ROOT / f"{STEM}.md", f"markdown/{STEM}.md"), (ROOT / f"{SUPP_STEM}.md", f"markdown/{SUPP_STEM}.md"),
             (ROOT / f"{NOTES_STEM}.md", f"markdown/{NOTES_STEM}.md"),
             (ROOT / "build" / "verify10_report.json", "checks/verify10_report.json"),
             (ROOT / "build" / "docx_build_report10.json", "checks/docx_build_report10.json")]
    for ext in ("pdf", "svg", "png"):
        files += [(p, f"figures/{p.name}") for p in sorted(FIG.glob(f"Fig*.{ext}"))]
    files += [(p, f"figures/source_data/{p.name}") for p in sorted((FIG / "source_data").glob("*.csv"))]
    missing = [str(s) for s, _ in files if not s.exists()]
    assert not missing, missing
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in files:
            z.write(src, f"{name}/{arc}")
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        n = len(z.namelist())
    rep = {"zip": str(out), "files": n, "bytes": out.stat().st_size, "testzip": bad,
           "figures": sum(1 for _, a in files if a.startswith("figures/Fig")),
           "csv": sum(1 for _, a in files if a.startswith("figures/source_data/"))}
    (ROOT / "build" / "package10_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=1))

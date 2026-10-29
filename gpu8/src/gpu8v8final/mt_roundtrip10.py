"""Round-trip test of the MathType objects written by build_docx10: MathType itself re-typesets every object of a test
document (body 12 pt, legend 10.5 pt, table 9 pt, display) from the stored MTEF.  The test passes when MathType
reads every object without an error and the re-typeset sizes agree with the sizes written by build_docx10, i.e.
MathType honours the size-scaled preferences.  Report: build/mathtype/roundtrip10.json."""
from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

from lxml import etree

from . import mt_factory10 as MT
from .build_docx10 import Builder, load_assets
from .manuscript10 import STEM
from .numbers9 import ROOT
from .supp10 import SUPP_STEM

SCRIPT = Path(r"D:\信\gpu8\submission9\tools\mathtype\convert_mt10.ps1")
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "v": "urn:schemas-microsoft-com:vml",
      "o": "urn:schemas-microsoft-com:office:office"}


def objects(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        root = etree.fromstring(z.read("word/document.xml"))
    out = []
    for ob in root.iter(f"{{{NS['w']}}}object"):
        style = ob.find("v:shape", NS).get("style")
        out.append({"dxa": int(ob.get(f"{{{NS['w']}}}dxaOrig")), "dya": int(ob.get(f"{{{NS['w']}}}dyaOrig")),
                    "style": style})
    return out


def run() -> dict:
    mds = [ROOT / f"{STEM}.md", ROOT / f"{SUPP_STEM}.md"]
    assets = load_assets(mds)
    inline = [t for t, m in assets if m == "inline" and ("frac" in t or "left" in t or "sum" in t)][:3] + ["s_F"]
    disp = next(t for t, m in assets if m == "display")
    body = " ".join(f"${t}$" for t in inline)
    md = "\n\n".join(["# 测试", f"正文 {body}。", "| 列 | 值 |", "", f"**图 1　测试。** 图注 {body}。", f"$${disp}$$"])
    md = md.replace("| 列 | 值 |\n\n", "| 列 | 值 |\n|---|---|\n" + "".join(f"| 表 | ${t}$ |\n" for t in inline) + "\n")
    work = MT.CACHE / "work"
    src, out = work / "roundtrip10.md", work / "roundtrip10.docx"
    src.write_text(md, encoding="utf-8")
    Builder("main", assets).build(src, out, images=False)
    conv = work / "roundtrip10_mt.docx"
    conv.unlink(missing_ok=True)
    r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT), "-InputPath", str(out),
                        "-OutputPath", str(conv), "-TimeoutSeconds", "600", "-SourceMathType"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    rep = {"returncode": r.returncode, "stdout": r.stdout[-800:], "stderr": r.stderr[-800:]}
    if r.returncode == 0:
        a, b = objects(out), objects(conv)
        sizes = [12.0] * len(inline) + [9.0] * len(inline) + [10.5] * len(inline) + [12.0]
        rows = [{"size": s, "written_dya": x["dya"], "retypeset_dya": y["dya"], "ratio": round(y["dya"] / x["dya"], 3)}
                for s, x, y in zip(sizes, a, b)]
        rep.update({"objects_written": len(a), "objects_after": len(b), "rows": rows,
                    "ok": len(a) == len(b) and all(0.85 <= row["ratio"] <= 1.15 for row in rows)})
    else:
        rep["ok"] = False
    (MT.CACHE / "roundtrip10.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


def run_documents() -> dict:
    """MathType re-typesets every object of copies of the final submission and supplement files; each object must come
    back with the size written by build_docx10."""
    rep = {}
    work = MT.CACHE / "work"
    for name, stem in (("投稿版", STEM), ("补充材料", SUPP_STEM)):
        src = ROOT / f"{stem}.docx"
        conv = work / f"roundtrip10_{'main' if name == '投稿版' else 'supp'}_mt.docx"
        conv.unlink(missing_ok=True)
        r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT), "-InputPath",
                            str(src), "-OutputPath", str(conv), "-TimeoutSeconds", "1200", "-SourceMathType"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500)
        d = {"returncode": r.returncode, "stdout": r.stdout[-600:]}
        if r.returncode == 0:
            a, b = objects(src), objects(conv)
            ratios = [y["dya"] / x["dya"] for x, y in zip(a, b)] + [y["dxa"] / x["dxa"] for x, y in zip(a, b)]
            d.update({"objects": len(a), "objects_after": len(b), "identical_size": sum(
                1 for x, y in zip(a, b) if (x["dxa"], x["dya"]) == (y["dxa"], y["dya"])),
                "ratio_min": round(min(ratios), 3), "ratio_max": round(max(ratios), 3)})
            d["ok"] = len(a) == len(b) and 0.97 <= d["ratio_min"] and d["ratio_max"] <= 1.03
        else:
            d["ok"] = False
        rep[name] = d
    rep["ok"] = all(v["ok"] for v in rep.values() if isinstance(v, dict))
    (MT.CACHE / "roundtrip10_documents.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


if __name__ == "__main__":
    import sys
    print(json.dumps(run_documents() if "--documents" in sys.argv else run(), ensure_ascii=False, indent=1))

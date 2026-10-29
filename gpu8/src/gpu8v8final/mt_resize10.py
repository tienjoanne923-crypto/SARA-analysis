"""Size-specific MathType objects typeset by MathType itself.

Equations in 9 pt tables and 10.5 pt captions/legends first get size-scaled MathType preferences
(``mt_factory10.sized``).  Here every such object is placed once in a scratch document and MathType re-typesets all of
them from their MTEF (``convert_mt10.ps1 -SourceMathType``); the objects it writes -- OLE storage, WMF preview, size
and baseline offset -- are harvested as ``<key>_<per mille>_rt`` assets, which ``sized`` then prefers.  Every object of
the final documents is therefore exactly what MathType produces at that text size."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from . import mt_factory10 as MT
from .build_docx10 import NOTES_STEM, Builder, load_assets
from .manuscript10 import STEM
from .numbers9 import ROOT
from .supp10 import SUPP_STEM

SCRIPT = Path(r"D:\信\gpu8\submission9\tools\mathtype\convert_mt10.ps1")


def requests() -> tuple[dict, set]:
    """(assets, {(tex, mode, size)}) of every object placed in the four documents (built into a temporary folder)."""
    mds = [p for p in (ROOT / f"{STEM}.md", ROOT / f"{SUPP_STEM}.md", ROOT / f"{NOTES_STEM}.md") if p.exists()]
    assets = load_assets(mds)
    req = set()
    with tempfile.TemporaryDirectory(dir=MT.CACHE / "work") as tmp:
        for md, kind in zip(mds, ("main", "supplement", "notes")):
            b = Builder(kind, assets, line_numbers=kind != "notes")
            b.build(md, Path(tmp) / f"{kind}.docx", images=False)
            req |= set(b.requests)
    return assets, req


def run() -> dict:
    assets, req = requests()
    todo = []
    for tex, mode, size in sorted(req):
        f = size / 12.0
        if abs(f - 1) < 1e-9:
            continue
        base = assets[(tex, mode)]
        tag = f"{base['key']}_{round(f * 1000)}"
        if not (MT.CACHE / f"{tag}_rt.json").exists():
            todo.append((tag, tex, mode, MT.sized(base, f)))
    rep = {"placed": len(req), "scaled": sum(1 for *_, s in req if s != 12.0), "retypeset_now": len(todo)}
    if todo:
        work = MT.CACHE / "work"
        src, out = work / "resize10.docx", work / "resize10_mt.docx"
        out.unlink(missing_ok=True)
        b = Builder("notes", {}, line_numbers=False)
        for _, _, _, a in todo:
            p = b.doc.add_paragraph()
            p._p.append(b.object_from_asset(a))
        b.doc.save(str(src))
        r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT), "-InputPath",
                            str(src), "-OutputPath", str(out), "-TimeoutSeconds", "1200", "-SourceMathType"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500)
        (work / "resize10.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"MathType re-typesetting failed:\n{(r.stdout + r.stderr)[-2000:]}")
        harvested = MT.harvest(out, [(f"{tag}_rt", tex, mode) for tag, tex, mode, _ in todo])
        rows = []
        for tag, tex, mode, a in todo:
            h = harvested[f"{tag}_rt"]
            h["base_key"], h["factor"] = a.get("base_key", a["key"]), a["factor"]
            (MT.CACHE / f"{tag}_rt.json").write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
            rows.append(h["dya"] / a["dya"])
        bad = [k for k, v in harvested.items() if not v["mtef_ok"] or v["replacement_chars"]]
        rep.update({"harvested": len(harvested), "bad": bad, "dya_ratio_min": round(min(rows), 3),
                    "dya_ratio_max": round(max(rows), 3)})
    (MT.CACHE / "resize10.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=1))

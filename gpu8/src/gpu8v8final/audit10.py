"""Integrity check of the second revision: the archives, earlier manuscripts, results and code that must stay untouched.

Compares a fresh snapshot with the round-9 start snapshot (``audit9_start.json``: V5/V6/V7.1 archives, manuscript_v8
with figures_v2 and the 2026-09-24/09-26 manuscripts, results/reports/manifests_v8, the second environment and the
pre-round-9 modules), checks that no round-9 module changed, and that the 2026-09-29 deliverables in the manuscript
folder are byte-identical to the copies in the 2026-09-29 zip.  Reports: submission9/audit/audit10_*.json (the
round-9 reports are not rewritten)."""
from __future__ import annotations

import hashlib
import json
import zipfile

from .audit9 import AUD, EXTRA_DIRS, SRC, _sha
from .audit8 import archive_hashes
from .common8 import dir_hash, dump_json, now_utc
from .numbers9 import ROOT

ROUND9_MODULES = ["audit9.py", "build_docx9.py", "fmt9.py", "manuscript9.py", "notes9.py", "numbers9.py", "package9.py",
                  "pubbook9.py", "pubfigs9.py", "publabel9.py", "pubrun9.py", "pubstyle9.py", "refs9.py", "supp9.py",
                  "verify9.py"]


def run() -> dict:
    start = json.loads((AUD / "audit9_start.json").read_text(encoding="utf-8"))
    end9 = json.loads((AUD / "audit9_end.json").read_text(encoding="utf-8"))
    h = archive_hashes()
    for k, d in EXTRA_DIRS.items():
        h[k] = dir_hash(d)
    mods = {n: (_sha(SRC / n) if (SRC / n).exists() else None) for n in start["modules"]}
    changed_dirs = [k for k in start["hashes"] if start["hashes"][k] != h.get(k)]
    changed_since_round9 = [k for k in end9["hashes"] if end9["hashes"][k] != h.get(k)]
    changed_mods = [k for k in start["modules"] if start["modules"][k] != mods.get(k)]
    r9 = {n: (SRC / n).stat().st_mtime for n in ROUND9_MODULES if (SRC / n).exists()}
    r9_changed = [n for n, t in r9.items() if t > end9_time(end9)]
    z9 = ROOT / "SARA_投稿包_20260929.zip"
    diff9, checked9 = [], 0
    with zipfile.ZipFile(z9) as z:
        for name in z.namelist():
            base = name.split("/", 1)[1]
            if "/" in base or not base.endswith((".docx", ".pdf", ".md")):
                continue
            p = ROOT / base
            checked9 += 1
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != hashlib.sha256(z.read(name)).hexdigest():
                diff9.append(base)
    rep = {"generated_utc": now_utc(), "baseline": start["generated_utc"], "round9_end": end9["generated_utc"],
           "changed_dirs_vs_round9_start": changed_dirs, "changed_dirs_vs_round9_end": changed_since_round9,
           "changed_pre_round9_modules": changed_mods, "changed_round9_modules": r9_changed,
           "round9_deliverables_checked": checked9, "round9_deliverables_changed": diff9,
           "n_dirs": len(start["hashes"]), "n_modules": len(start["modules"])}
    rep["untouched"] = not (changed_since_round9 or changed_mods or r9_changed or diff9)
    dump_json(AUD / "audit10_comparison.json", rep)
    return rep


def end9_time(end9: dict) -> float:
    from datetime import datetime
    return datetime.fromisoformat(end9["generated_utc"].replace("Z", "+00:00")).timestamp() + 600


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=1))

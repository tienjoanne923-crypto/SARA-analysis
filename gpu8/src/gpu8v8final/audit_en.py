"""Integrity check after the English translation: everything outside the English folder is untouched.

  * the round-9 baseline checks of audit10 (V5/V6/V7.1 archives, manuscript_v8 with figures_v2, the 09-24/09-26
    manuscripts, results/reports/manifests_v8, the second environment, pre-round-9 and round-9 modules);
  * the 2026-09-29 and 2026-09-30 Chinese deliverables (documents, PDFs, RIS, Markdown and the Chinese figures with
    their source data) are byte-identical to their copies in the corresponding zip packages.
Report: submission9/audit/audit_en_comparison.json (the audit10 report is not rewritten).
"""
from __future__ import annotations

import hashlib
import json
import zipfile

from .audit8 import archive_hashes
from .audit9 import AUD, EXTRA_DIRS, SRC, _sha
from .audit10 import ROUND9_MODULES, end9_time
from .common8 import dir_hash, dump_json, now_utc
from .numbers9 import ROOT


def _zip_vs_folder(zpath) -> tuple[int, list[str]]:
    checked, diff = 0, []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            rel = name.split("/", 1)[1]
            top = rel.split("/", 1)[0] if "/" in rel else ""
            if top == "checks":
                continue
            p = ROOT / (rel.split("/", 1)[1] if top == "markdown" else rel)
            checked += 1
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != hashlib.sha256(z.read(name)).hexdigest():
                diff.append(rel)
    return checked, diff


def run() -> dict:
    start = json.loads((AUD / "audit9_start.json").read_text(encoding="utf-8"))
    end9 = json.loads((AUD / "audit9_end.json").read_text(encoding="utf-8"))
    h = archive_hashes()
    for k, d in EXTRA_DIRS.items():
        h[k] = dir_hash(d)
    mods = {n: (_sha(SRC / n) if (SRC / n).exists() else None) for n in start["modules"]}
    changed_since_round9 = [k for k in end9["hashes"] if end9["hashes"][k] != h.get(k)]
    changed_mods = [k for k in start["modules"] if start["modules"][k] != mods.get(k)]
    r9_changed = [n for n in ROUND9_MODULES if (SRC / n).exists() and (SRC / n).stat().st_mtime > end9_time(end9)]
    n29, d29 = _zip_vs_folder(ROOT / "SARA_投稿包_20260929.zip")
    n30, d30 = _zip_vs_folder(ROOT / "SARA_投稿包_20260930.zip")
    rep = {"generated_utc": now_utc(), "baseline": start["generated_utc"], "round9_end": end9["generated_utc"],
           "changed_dirs_vs_round9_end": changed_since_round9, "changed_pre_round9_modules": changed_mods,
           "changed_round9_modules": r9_changed, "deliverables_20260929_checked": n29,
           "deliverables_20260929_changed": d29, "deliverables_20260930_checked": n30,
           "deliverables_20260930_changed": d30}
    rep["untouched"] = not (changed_since_round9 or changed_mods or r9_changed or d29 or d30)
    dump_json(AUD / "audit_en_comparison.json", rep)
    return rep


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=1))

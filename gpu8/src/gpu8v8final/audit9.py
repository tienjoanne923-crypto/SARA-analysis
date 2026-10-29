"""Integrity snapshot for the SARA submission build: everything that must stay untouched.

Covers the V5/V6/V7.1 archives and legacy folders (audit8 list), the V8.1 manuscripts and
figures (manuscript_v8, incl. figures_v2 and the 2026-09-24 / 09-26 manuscripts), the V8.1
results, the second environment and every gpu8v8final module that existed before this build.
Reports go to gpu8/submission9/audit only (the V8.1 audit files are not rewritten).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from gpu8 import paths as p5

from . import paths8
from .audit8 import archive_hashes
from .common8 import dir_hash, dump_json, now_utc

SUB = paths8.GPU8_ROOT / "submission9"
AUD = SUB / "audit"
EXTRA_DIRS = {"ROOT/manuscript_v8": p5.PROJECT_ROOT / "manuscript_v8",
              "ROOT/manuscript_round5_gpu8": p5.PROJECT_ROOT / "manuscript_round5_gpu8",
              "gpu8/results_v8": paths8.RESULTS,
              "gpu8/reports_v8": paths8.REPORTS,
              "gpu8/manifests_v8": paths8.MANIFESTS,
              "gpu8/.venv_repro": paths8.GPU8_ROOT / ".venv_repro"}
SRC = paths8.GPU8_ROOT / "src" / "gpu8v8final"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot(tag: str) -> dict:
    h = archive_hashes()
    for k, d in EXTRA_DIRS.items():
        h[k] = dir_hash(d)
    if tag == "start":
        mods = {p.name: _sha(p) for p in sorted(SRC.glob("*.py"))}
    else:
        start = json.loads((AUD / "audit9_start.json").read_text(encoding="utf-8"))
        mods = {n: (_sha(SRC / n) if (SRC / n).exists() else None) for n in start["modules"]}
    rep = {"generated_utc": now_utc(), "tag": tag, "hashes": h, "modules": mods}
    dump_json(AUD / f"audit9_{tag}.json", rep)
    return rep


def compare() -> dict:
    a = json.loads((AUD / "audit9_start.json").read_text(encoding="utf-8"))
    b = json.loads((AUD / "audit9_end.json").read_text(encoding="utf-8"))
    changed = [k for k in a["hashes"] if a["hashes"][k] != b["hashes"].get(k)]
    mod_changed = [k for k in a["modules"] if a["modules"][k] != b["modules"].get(k)]
    rep = {"generated_utc": now_utc(), "start": a["generated_utc"], "end": b["generated_utc"],
           "changed_dirs": changed, "changed_modules": mod_changed,
           "untouched": not changed and not mod_changed, "n_dirs": len(a["hashes"]), "n_modules": len(a["modules"])}
    dump_json(AUD / "audit9_comparison.json", rep)
    return rep


if __name__ == "__main__":
    AUD.mkdir(parents=True, exist_ok=True)
    tag = sys.argv[1] if len(sys.argv) > 1 else "start"
    if tag == "compare":
        print(json.dumps(compare(), ensure_ascii=False, indent=1))
    else:
        r = snapshot(tag)
        print(json.dumps({k: v.get("n_files") for k, v in r["hashes"].items()}, indent=1), len(r["modules"]), "modules")

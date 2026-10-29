"""Verify that the V7.1 round did not modify the V5 / V6 archives or the legacy tree.

Lists every file under the archived directories whose modification time is
later than the start of the V7.1 round (first V7.1 external download), and
records the frozen virtual environment's site-packages listing hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from gpu8 import paths as p5

from . import paths71
from .common71 import dump_json, now_utc

ARCHIVES = ["results", "runs", "manifests", "caches", "tables", "reports", "results_v6", "runs_v6",
            "manifests_v6", "caches_v6", "tables_v6", "reports_v6", "configs", "src/gpu8", "src/gpu8v6"]
LEGACY = [p5.PROJECT_ROOT / "code", p5.PROJECT_ROOT / "results", p5.PROJECT_ROOT / "manuscript_v6"]


def v71_start() -> float:
    log = pd.read_csv(paths71.ext("download_log_v71.csv"), encoding="utf-8-sig")
    t = pd.to_datetime(log["timestamp_utc"], utc=True).min()
    return t.timestamp() - 600          # ten minutes of margin before the first download


def scan(root: Path, since: float) -> list[dict]:
    out = []
    if not root.exists():
        return out
    for dp, _, files in os.walk(root):
        if "__pycache__" in dp:
            continue
        for f in files:
            p = Path(dp) / f
            try:
                mt = p.stat().st_mtime
            except OSError:
                continue
            if mt > since:
                out.append({"path": str(p), "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                         time.gmtime(mt))})
    return out


def venv_hash() -> str:
    sp = p5.GPU8_ROOT / ".venv" / "Lib" / "site-packages"
    names = sorted(x.name for x in sp.iterdir()) if sp.exists() else []
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def run() -> dict:
    since = v71_start()
    changed = {}
    for a in ARCHIVES:
        changed[a] = scan(p5.GPU8_ROOT / a, since)
    for L in LEGACY:
        changed[str(L)] = scan(L, since)
    rep = {"generated_utc": now_utc(), "v71_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                      time.gmtime(since)),
           "changed_counts": {k: len(v) for k, v in changed.items()},
           "changed_examples": {k: v[:10] for k, v in changed.items() if v},
           "archives_untouched": all(len(v) == 0 for v in changed.values()),
           "venv_site_packages_listing_sha256": venv_hash(),
           "v71_libs_separate": str(paths71.LIBS)}
    dump_json(paths71.AUDIT / "isolation_v71.json", rep)
    return rep


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False)[:3000])

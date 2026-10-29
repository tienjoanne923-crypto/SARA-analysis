"""Audits: archive hashes (start / end) and the ten whole-chain acceptance checks."""
from __future__ import annotations

import json
import sys

from gpu8 import paths as p5

from . import paths8
from .common8 import dir_hash, dump_json, now_utc

ARCHIVES = ["results", "runs", "manifests", "caches", "tables", "reports", "configs",
            "results_v6", "runs_v6", "manifests_v6", "caches_v6", "tables_v6", "reports_v6",
            "results_v71", "runs_v71", "manifests_v71", "caches_v71", "tables_v71", "reports_v71",
            "src/gpu8", "src/gpu8v6", "src/gpu8v71", ".venv/Lib/site-packages"]
LEGACY = ["code", "results", "manuscript_v6", "manuscript_v71"]


def archive_hashes() -> dict:
    out = {}
    for a in ARCHIVES:
        out[a] = dir_hash(p5.GPU8_ROOT / a)
    for a in LEGACY:
        out["ROOT/" + a] = dir_hash(p5.PROJECT_ROOT / a)
    return out


def snapshot(tag: str) -> dict:
    rep = {"generated_utc": now_utc(), "tag": tag, "hashes": archive_hashes()}
    dump_json(paths8.MANIFESTS / f"archive_hashes_{tag}.json", rep)
    return rep


def compare(start: str = "start", end: str = "end") -> dict:
    a = json.loads((paths8.MANIFESTS / f"archive_hashes_{start}.json").read_text(encoding="utf-8"))
    b = json.loads((paths8.MANIFESTS / f"archive_hashes_{end}.json").read_text(encoding="utf-8"))
    changed = [k for k in a["hashes"] if a["hashes"][k] != b["hashes"].get(k)]
    rep = {"generated_utc": now_utc(), "start": a["generated_utc"], "end": b["generated_utc"],
           "changed": changed, "archives_untouched": not changed}
    dump_json(paths8.AUDIT / "archive_comparison_v8.json", rep)
    return rep


if __name__ == "__main__":
    paths8.ensure()
    tag = sys.argv[1] if len(sys.argv) > 1 else "start"
    if tag == "compare":
        print(json.dumps(compare(), indent=2))
    else:
        r = snapshot(tag)
        print(json.dumps({k: v.get("n_files") for k, v in r["hashes"].items()}, indent=1))

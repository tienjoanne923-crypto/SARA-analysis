"""Render the SARA submission figures: ``python -m gpu8v8final.pubrun9 [fig1 fig2 ...]``.

Unchanged V8.1 figure functions are rebound to ``pubstyle9.save9`` (public wording, text scan, new output
folder); Fig 2, 6, 7, S2 and the new S8 come from ``pubfigs9``.  Writes the manifest (with the palette
checks), the relabel inventory and the text-scan report next to the figures.
"""
from __future__ import annotations

import csv
import json
import sys
import time

from .common8 import dump_json, now_utc
from .pubstyle8 import setup
from . import pubstyle9


def registry() -> dict:
    from . import pubfigs8a as A
    from . import pubfigs8a_part2 as A2
    from . import pubfigs8a_part3 as A3
    from . import pubfigs8b as B
    from . import pubfigs8c as C
    from . import pubfigs8s as S
    from . import pubfigs9 as N
    for mod in (A, A2, A3, B, C, S):
        mod.save = pubstyle9.save9
    return {"fig1": A.fig1, "fig2": N.fig2, "fig3": A2.fig3, "fig4": A3.fig4, "fig5": A3.fig5, "fig6": N.fig6,
            "fig7": N.fig7, "fig8": B.fig8, "fig9": C.fig9, "fig10": C.fig10, "figS1": S.figS1, "figS2": N.figS2,
            "figS3": S.figS3, "figS4": S.figS4, "figS5": S.figS5, "figS6": S.figS6, "figS7": S.figS7,
            "figS8": N.figS8}


def main(which: list[str]) -> dict:
    setup()
    reg = registry()
    todo = which or list(reg)
    out = pubstyle9.OUT
    out.mkdir(parents=True, exist_ok=True)
    man_p = out / "figures_manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    for k in todo:
        t0 = time.time()
        info = reg[k]()
        info["seconds"] = round(time.time() - t0, 1)
        info["generated_utc"] = now_utc()
        man[k] = info
        print(f"  {k}: {info['stem']} {info['size_mm']} relabelled={info['relabelled']} "
              f"forbidden={info['forbidden_hits']} ({info['seconds']}s)", flush=True)
    from .pubcolor8 import run_checks
    man["_palette_checks"] = run_checks()
    dump_json(man_p, man)
    scan_p = out / "text_scan.json"
    scan = json.loads(scan_p.read_text(encoding="utf-8")) if scan_p.exists() else {}
    for stem, rep in pubstyle9.LOG.items():
        scan[stem] = {"forbidden": rep["forbidden"], "small_text": rep["small_text"], "overlaps": rep["overlaps"],
                      "n_relabelled": len(rep["changes"])}
    dump_json(scan_p, scan)
    inv_p = out / "relabel_inventory.csv"
    rows = []
    if inv_p.exists():
        with open(inv_p, encoding="utf-8-sig") as fh:
            rows = [r for r in csv.DictReader(fh) if r["stem"] not in pubstyle9.LOG]
    for stem, rep in pubstyle9.LOG.items():
        rows += [{"stem": stem, "old": a, "new": b} for a, b in sorted(set(rep["changes"]))]
    with open(inv_p, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "old", "new"])
        w.writeheader()
        w.writerows(rows)
    return man


if __name__ == "__main__":
    main(sys.argv[1:])

"""Render the English SARA figures: ``python -m gpu8v8final.pubrun_en [fig1 fig2 ...]``.

The figure functions of the Chinese package (pubrun9 registry) are run unchanged with their save hook rebound to
``pubstyle_en.save_en``; writes the manifest (with the palette checks) and the text-scan report next to the figures.
"""
from __future__ import annotations

import json
import sys
import time

from .common8 import dump_json, now_utc
from .pubstyle8 import setup
from . import pubstyle_en as E


def main(which: list[str]) -> dict:
    setup()
    E.en_fonts()
    from .pubrun9 import registry
    from . import pubfigs8a, pubfigs8a_part2, pubfigs8a_part3, pubfigs8b, pubfigs8c, pubfigs8s, pubfigs9
    reg = registry()
    for mod in (pubfigs8a, pubfigs8a_part2, pubfigs8a_part3, pubfigs8b, pubfigs8c, pubfigs8s, pubfigs9):
        mod.save = E.save_en
    E.OUT.mkdir(parents=True, exist_ok=True)
    man_p = E.OUT / "figures_manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    for k in which or list(reg):
        t0 = time.time()
        info = reg[k]()
        info["seconds"] = round(time.time() - t0, 1)
        info["generated_utc"] = now_utc()
        man[k] = info
        print(f"  {k}: {info['stem']} translated={info['translated']} cjk={info['cjk_left']} "
              f"forbidden={info['forbidden_hits']} overlaps={info['overlaps']} ({info['seconds']}s)", flush=True)
    from .pubcolor8 import run_checks
    man["_palette_checks"] = run_checks()
    dump_json(man_p, man)
    scan_p = E.OUT / "text_scan.json"
    sc = json.loads(scan_p.read_text(encoding="utf-8")) if scan_p.exists() else {}
    for stem, rep in E.LOG.items():
        sc[stem] = {k: rep[k] for k in ("cjk", "forbidden", "small_text", "overlaps")}
    dump_json(scan_p, sc)
    return man


if __name__ == "__main__":
    main(sys.argv[1:])

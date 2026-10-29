"""Inventory of the Chinese strings that the English figures must translate.

Every SARA figure is drawn in memory exactly as for the Chinese package (pubrun9 registry, pubstyle9 surgery and
public relabelling), nothing is saved; all visible strings (titles, labels, tick labels, legends, annotations) and
the strings of the public source-data tables that contain CJK characters are collected per figure.
Output: build/en/figtext_inventory.json  ({"strings": {text: [stems]}, "csv": {text: [files]}})."""
from __future__ import annotations

import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

from . import pubstyle9
from .numbers9 import ROOT

CJK = re.compile(r"[⺀-鿿　-〿＀-￯]")
OUT = ROOT / "build" / "en" / "figtext_inventory.json"
STRINGS: dict = {}
CSV: dict = {}


def fake_save(fig, stem, sources=None, meta=None):
    pubstyle9.surgery(fig, stem)
    pubstyle9.relabel_figure(fig)
    if stem in pubstyle9.FIXUPS:
        pubstyle9.FIXUPS[stem](fig)
    fig.canvas.draw()
    for t in fig.findobj(Text):
        s = t.get_text()
        if s and s.strip() and CJK.search(s):
            STRINGS.setdefault(s, set()).add(stem)
    for k, df in (sources or {}).items():
        pub = pubstyle9.public_table(df)
        for c in pub.columns:
            if CJK.search(str(c)):
                CSV.setdefault(str(c), set()).add(f"{stem}_{k}")
            if pub[c].dtype == object:
                for v in pub[c].dropna().unique():
                    if isinstance(v, str) and CJK.search(v):
                        CSV.setdefault(v, set()).add(f"{stem}_{k}")
    w, h = fig.get_size_inches() / pubstyle9.MM
    plt.close(fig)
    return {"stem": stem, "size_mm": [round(w, 1), round(h, 1)], "files": [], "relabelled": 0, "forbidden_hits": 0}


def main(which: list[str]) -> dict:
    from .pubrun9 import registry
    from .pubstyle8 import setup
    from . import pubfigs8a, pubfigs8a_part2, pubfigs8a_part3, pubfigs8b, pubfigs8c, pubfigs8s, pubfigs9
    setup()
    reg = registry()                                  # rebinds the V8.1 modules' save to pubstyle9.save9
    for mod in (pubfigs8a, pubfigs8a_part2, pubfigs8a_part3, pubfigs8b, pubfigs8c, pubfigs8s, pubfigs9):
        mod.save = fake_save
    for k in which or list(reg):
        reg[k]()
        print(k, "ok", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rep = {"strings": {s: sorted(v) for s, v in sorted(STRINGS.items())}, "csv": {s: sorted(v) for s, v in sorted(CSV.items())}}
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"figure_strings": len(STRINGS), "csv_strings": len(CSV)}


if __name__ == "__main__":
    print(main(sys.argv[1:]))

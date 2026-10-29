"""Render the publication figures: ``python -m gpu8v8final.pubrun8 [fig1 fig2 ...]``."""
from __future__ import annotations

import json
import sys
import time

from . import paths8
from .common8 import dump_json, now_utc
from .pubstyle8 import FIG_DIR, setup


def registry() -> dict:
    from . import pubfigs8a as A
    from . import pubfigs8a_part2 as A2
    reg = {"fig1": A.fig1, "fig2": A2.fig2, "fig3": A2.fig3}
    for mod in (A, A2):
        for name in ("fig4", "fig5"):
            if hasattr(mod, name):
                reg[name] = getattr(mod, name)
    try:
        from . import pubfigs8a_part3 as A3
        for name in ("fig4", "fig5"):
            if hasattr(A3, name):
                reg[name] = getattr(A3, name)
    except ImportError:
        pass
    for modname in ("pubfigs8b", "pubfigs8c", "pubfigs8s"):
        try:
            mod = __import__(f"gpu8v8final.{modname}", fromlist=["x"])
        except ImportError:
            continue
        for name in ("fig6", "fig7", "fig8", "fig9", "fig10", "figS1", "figS2", "figS3", "figS4", "figS5",
                     "figS6", "figS7"):
            if hasattr(mod, name):
                reg[name] = getattr(mod, name)
    return reg


def main(which: list[str]) -> dict:
    setup()
    reg = registry()
    todo = which or list(reg)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    man_p = FIG_DIR / "figures_manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    for k in todo:
        t0 = time.time()
        info = reg[k]()
        info["seconds"] = round(time.time() - t0, 1)
        info["generated_utc"] = now_utc()
        man[k] = info
        print(f"  {k}: {info['stem']} {info['size_mm']} ({info['seconds']}s)", flush=True)
    from .pubcolor8 import run_checks
    man["_palette_checks"] = run_checks()
    dump_json(man_p, man)
    return man


if __name__ == "__main__":
    paths8.ensure()
    main(sys.argv[1:])

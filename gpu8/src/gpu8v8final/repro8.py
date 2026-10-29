"""E9 — re-run the fold-0 seed-42 chain in the separate environment ``.venv_repro``.

Recomputed from the frozen inputs (not copied): the four V6 passes (world
"repro"), the four teachers, the strict pools and the deployed candidate A of
fold 0; compared with the original run under registered tolerances
(weights <= 1e-3, deployed scores <= 1e-4, identical pools and top-10 lists).
Run with:  .venv_repro\\Scripts\\python.exe -m gpu8v8final.repro8
In this round .venv_repro could not load SciPy (blocked by Windows Application
Control, see audit/reproduction_env_blocked_v8.json), so it was run with the main
interpreter on fresh caches: a determinism check, not an environment check.
"""
from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np
import torch

from . import paths8
from .common8 import dump_json, now_utc

TOL = {"weights": 1e-3, "scores": 1e-4}


def run(j: int = 0, seed: int = 42) -> dict:
    from gpu8v6.common import load_manifest as lm6
    from .chain8 import Chain, ctx_list
    from .formal8 import sel_cfg, selection
    from .models8 import CandidateA
    from .pack8 import ArmSpec
    from .v6pass8 import V6Pass, V6World, make_engine, run_context
    t0 = time.time()
    import lightgbm
    env = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
           "torch": torch.__version__, "cuda": torch.cuda.is_available(), "lightgbm": lightgbm.__version__,
           "numpy": np.__version__}
    rw = V6World(name="repro")
    from gpu8v6.common import World6
    from .roles8 import Roles
    w = World6(load_heavy=True)
    roles = Roles(w)
    eng = make_engine(w, roles, rw)
    frozen = lm6("retriever_v6_frozen.json")
    v6cmp = {}
    for c in ctx_list(j):
        run_context(eng, w, roles, rw, c, frozen)
        a, b = V6Pass(rw, c), V6Pass(V6World(), c)
        v6cmp[c] = {"rB_identical": bool(np.array_equal(np.asarray(a.rB), np.asarray(b.rB))),
                    "V_maxabs": float(np.abs(np.asarray(a.V[:200]) - np.asarray(b.V[:200])).max())}
    del eng
    sel = selection(j)
    cfg = sel_cfg(sel)
    ch = Chain(ArmSpec(name="repro"), rw, role="audit", w=w)
    wcmp, sems = {}, {}
    for c in ctx_list(j):
        rid = ch.teacher(c, seed, sel["controls"], role="audit")
        new = torch.load(paths8.RUNS / rid / "model.pt", map_location="cpu", weights_only=True)
        old = torch.load(paths8.RUNS / f"V8T__main__{c}__{sel['controls']}__s{seed}" / "model.pt",
                         map_location="cpu", weights_only=True)
        wcmp[c] = max(float((new[k].float() - old[k].float()).abs().max()) for k in old)
        sems[c] = ch.semantic(c, rid, V6Pass(rw, c), None)
    blocks = {c: ch.blocks_for(c, V6Pass(rw, c), sems[c], sel["w"]["w"]) for c in ctx_list(j)}
    del sems
    outer = f"f{j}_outer"
    z = np.load(paths8.DEV / f"fold{j}" / f"pools_{outer}_test.npz")
    saved = {str(d): g for d, g in zip(z["ids"], z["genes"])}
    tb = blocks[outer]["test"]
    pools_same = float(np.mean([np.array_equal(tb.genes[i].astype(np.int16), saved[d]) for i, d in enumerate(tb.ids)]))
    A = CandidateA(cfg["A"]["num_leaves"], cfg["A"]["min_child_samples"], seed, 8)
    A.fit([blocks[c]["oof"] for c in ctx_list(j)[:3]], [blocks[outer]["val"]])
    sc = A.score(tb)
    p = np.load(paths8.DEV / f"fold{j}" / f"pred_{cfg['A_name']}__test.npz")
    pos = {str(d): i for i, d in enumerate(p["ids"])}
    diffs, top_same = [], []
    for i, d in enumerate(tb.ids):
        k = pos[d]
        diffs.append(float(np.abs(sc[i] - p["score"][k]).max()))
        o1 = tb.genes[i][np.lexsort((tb.genes[i], -sc[i]))][:10]
        g2 = p["genes"][k].astype(np.int64)
        o2 = g2[np.lexsort((g2, -p["score"][k].astype(np.float64)))][:10]
        top_same.append(bool(np.array_equal(o1, o2)))
    out = {"generated_utc": now_utc(), "environment": env, "fold": j, "seed": seed, "v6_pass": v6cmp,
           "teacher_weights_maxabs": wcmp, "pools_identical_share": pools_same,
           "A_best_iteration": A.best_iteration, "deployed_scores_maxabs": float(np.max(diffs)),
           "top10_identical_share": float(np.mean(top_same)), "tolerances": TOL, "seconds": time.time() - t0}
    out["within_tolerance"] = bool(all(v["rB_identical"] for v in v6cmp.values()) and
                                   max(wcmp.values()) <= TOL["weights"] and pools_same == 1.0 and
                                   out["deployed_scores_maxabs"] <= TOL["scores"])
    dump_json(paths8.AUDIT / "reproduction_venv_repro_v8.json", out)
    return out


if __name__ == "__main__":
    paths8.ensure()
    print(json.dumps(run(), indent=1, default=str))

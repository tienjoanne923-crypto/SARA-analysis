"""Choose the V6 main zero-support scorer on development data (1-SE rule, section 11.2).

Candidates, from simplest to most complex by number of fitted quantities:
F (7 fusion weights) < LR < LightGBM < bilinear residual < attention residual.
Each learned model is refitted once on fold-0 OOF training lists with its
development-selected preset and seed 42 and scored on the two inner validation
sets (adjacent fold F1 and the development families, 1,712 diseases together).
The best model by mean per-disease Recall@10 is found; the chosen model is the
simplest one whose paired shortfall to the best is within one standard error
(disease-level) — a small difference never buys a more complex model.
No test fold is read.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths6
from .common import World6, dump_json, load_manifest, write_manifest
from .scoring import fold_blocks, fusion_scores, recall_at
from .zero_support import Kit

ORDER = ["F7", "lr", "lightgbm", "bilinear", "attention"]


def run() -> dict:
    w = World6(load_heavy=False)
    kit = Kit(w)
    sel = load_manifest("devfit_v6_selection.json")["selection"]
    train, valA, devB, _ = fold_blocks(0, w)
    per = {}
    per["F7"] = np.concatenate([recall_at(b, fusion_scores(b, kit.F)) for b in (valA, devB)])
    for kind in ("lr", "lightgbm", "bilinear", "attention"):
        scorer, info, _ = kit.fit_learned(kind, sel[kind]["config"], train, valA, 42)
        per[kind] = np.concatenate([recall_at(b, scorer(b)) for b in (valA, devB)])
    means = {k: float(v.mean()) for k, v in per.items()}
    best = max(means, key=means.get)
    rows = []
    for m in ORDER:
        d = per[best] - per[m]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if m != best else 0.0
        rows.append({"model": m, "mean_recall10": means[m], "shortfall_to_best": float(d.mean()),
                     "se_paired": se, "within_1se": bool(d.mean() <= se + 1e-15)})
    tab = pd.DataFrame(rows)
    chosen = next(r["model"] for r in rows if r["within_1se"])
    tab.to_csv(paths6.SCORING / "main_scorer_selection_dev.csv", index=False, encoding="utf-8-sig")
    main_name = "v6_" + chosen
    out = {"main": main_name, "chosen_model": chosen, "best_model": best,
           "n_validation_diseases": int(len(per["F7"])), "table": rows,
           "rule": "simplest model whose paired disease-level shortfall to the best is <= 1 SE",
           "strong_controls": [m for m in ("F7", "lightgbm") if m != chosen][:2]}
    write_manifest("z_main_scorer.json", out)
    return out


if __name__ == "__main__":
    paths6.ensure()
    print(json.dumps(run(), indent=2))

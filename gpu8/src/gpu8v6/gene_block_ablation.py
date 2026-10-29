"""V6 section 4.2 — add the gene-feature blocks one at a time (exploratory).

Gene function features never enter retrieval, so they cannot change coverage;
this ablation measures what each block adds to top-10 recovery inside the same
V6 candidate lists, with the same LightGBM preset, seed and validation fold:

    channels only  ->  + GO  ->  + GO + Reactome  ->  + GO + Reactome + GTEx (= main)

Direction of every reported difference: (larger feature set) - (smaller set).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8.stats import family_bootstrap

from . import paths6
from .common import World6, dump_json, load_manifest, now_utc
from .gene_features import load as load_gene_feats
from .ledger6 import run_dir, tracked
from .metrics6 import macro
from .scoring import LightGBMScorer, evaluate, fold_blocks
from .zero_support import is_done

LADDER = {
    "channels_only": [],
    "plus_go": ["go_svd", "go_present"],
    "plus_go_reactome": ["go_svd", "go_present", "rx_svd", "reactome_present"],
    "plus_go_reactome_gtex": ["go_svd", "go_present", "rx_svd", "reactome_present", "ex_",
                              "gtex_present"],
}


def run(folds=(0, 1, 2, 3, 4), seed: int = 42) -> dict:
    w = World6(load_heavy=False)
    gf, names = load_gene_feats("current")
    preset = load_manifest("devfit_v6_selection.json")["selection"]["lightgbm"]["config"]
    per_all = []
    for fold in folds:
        train, val, _, test = fold_blocks(fold, w)
        for step, prefixes in LADDER.items():
            cols = [i for i, n in enumerate(names) if any(n.startswith(p) for p in prefixes)]
            g = gf[:, cols] if cols else None
            run_id = f"V6ABL__lightgbm_{step}__f{fold}__s{seed}"
            if not is_done(run_id):
                with tracked(run_id, {"stage": "ablation", "method": f"lightgbm_{step}",
                                      "fold": fold, "seed": seed, "config": preset,
                                      "n_gene_feature_columns": len(cols)}) as box:
                    m = LightGBMScorer(seed=seed, **preset).fit(train, [val], gene_feats=g)
                    per = evaluate(test, m.score(test))
                    per.to_parquet(run_dir(run_id) / "per_disease.parquet", index=False)
                    box["extra"] = {"metrics": macro(per), "n_test_diseases": int(test.D),
                                    "fit_info": {"best_iteration": m.best_iteration}}
            per = pd.read_parquet(run_dir(run_id) / "per_disease.parquet")
            per["step"], per["fold"] = step, fold
            per_all.append(per[["disease_id", "step", "fold", "recall10_end"]])
    df = pd.concat(per_all, ignore_index=True)
    piv = df.pivot_table(index="disease_id", columns="step", values="recall10_end")
    steps = list(LADDER)
    rows = []
    for a, b in zip(steps[1:], steps[:-1]):
        d = (piv[a] - piv[b]).dropna()
        fams = np.array([w.family_of.get(x, x) for x in d.index])
        lo, hi, _ = family_bootstrap(d.to_numpy(), fams, draws=10000)
        rows.append({"added_block": a, "vs": b, "n": int(len(d)), "mean_with": float(piv[a].mean()),
                     "mean_without": float(piv[b].mean()), "delta": float(d.mean()),
                     "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    out.to_csv(paths6.SCORING / "gene_block_ablation.csv", index=False, encoding="utf-8-sig")
    res = {"generated_utc": now_utc(), "means": {s: float(piv[s].mean()) for s in steps},
           "increments": rows, "status": "exploratory, after protocol registration, seed 42"}
    dump_json(paths6.SCORING / "gene_block_ablation.json", res)
    return res


if __name__ == "__main__":
    paths6.ensure()
    print(json.dumps(run(), indent=2, default=str))

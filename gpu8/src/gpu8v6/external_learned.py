"""Apply the frozen formal LightGBM fold models (seed 42) to the external layers.

The external diseases were never a reference or a training disease of any fold,
so each of the five fold models may score them.  Metrics are computed per model
and averaged inside the disease, then summarised with family-bootstrap
intervals; nothing is refitted.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8.stats import family_bootstrap

from . import paths6
from .common import World6, dump_json, now_utc
from .external6 import CTX, label_layers
from .gene_features import load as load_gene_feats
from .metrics6 import evaluate_matrix, per_disease_frame
from .scoring import design, load_block


def run() -> dict:
    import lightgbm as lgb
    w = World6(load_heavy=False)
    gf, _ = load_gene_feats("current")
    layers = label_layers(w)
    targets = sorted({d for df in layers.values() for d in df["disease_id"]} & set(w.idx.d_ix))
    block = load_block(CTX, ids=targets)
    models = []
    for f in range(5):
        txt = (paths6.RUNS / f"V6Z__lightgbm__f{f}__s42" / "model.txt").read_text(encoding="utf-8")
        models.append(lgb.Booster(model_str=txt))
    X = design(block, gf)
    scores = [m.predict(X).reshape(block.D, block.N) for m in models]
    g_ix = w.idx.g_ix
    rows = []
    for lname, lab in layers.items():
        q = lab.groupby("disease_id")["gene_id"].apply(lambda s: {g_ix[str(x)] for x in s
                                                                  if str(x) in g_ix}).to_dict()
        ids = [d for d in block.ids if q.get(d)]
        if not ids:
            continue
        pos = {d: i for i, d in enumerate(block.ids)}
        idx = [pos[d] for d in ids]
        sub = block.subset(ids)
        y = np.stack([np.isin(sub.genes[i], list(q[d])).astype(np.float32) for i, d in enumerate(ids)])
        nq = np.array([len(q[d]) for d in ids], dtype=float)
        per_model = [per_disease_frame(ids, evaluate_matrix(s[idx], y, sub.genes, nq, ks=(10,)))
                     ["recall10_end"].to_numpy() for s in scores]
        vals = np.mean(per_model, axis=0)
        fams = np.array([w.family_of.get(d, d) for d in ids])
        lo, hi, _ = family_bootstrap(vals, fams, draws=10000)
        rows.append({"layer": lname, "scorer": "v6_lightgbm_5fold_models_mean", "n_diseases": len(ids),
                     "recall10_end": float(vals.mean()), "ci_low": lo, "ci_high": hi,
                     "per_model_mean": [float(p.mean()) for p in per_model]})
    out = pd.DataFrame(rows)
    out.to_csv(paths6.EXTERNAL_RES / "external_v6_lightgbm.csv", index=False, encoding="utf-8-sig")
    res = {"generated_utc": now_utc(), "rows": rows}
    dump_json(paths6.EXTERNAL_RES / "external_v6_lightgbm.json", res)
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))

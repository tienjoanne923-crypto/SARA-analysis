"""V6 section 12.4 — degree-preserving rewiring of the graph the method actually uses.

The physical STRING network behind the support propagation is rewired by double
edge swaps (every gene keeps its exact degree; each edge keeps its STRING weight
when it moves), the column-stochastic operator is rebuilt from the rewired graph,
and the frozen support configuration (alpha, equal seeds, k = 5 and k = 1..3) is
re-run on the same test cohort.  The report gives the degree check, the share of
edges that survived, the change in connected components and the metric change.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8.contexts import build_P
from gpu8.stats import family_bootstrap

from . import paths6
from .common import dump_json, load_manifest, now_utc
from .propagate import PPIOperator, column_stochastic, degree_preserving_rewire
from .support_line import SupportExperiment


def run(n_swaps_per_edge: float = 10.0) -> dict:
    exp = SupportExperiment()
    w = exp.w
    sel = load_manifest("support_v6_frozen.json")
    _, _, raw = build_P(w.ppi, w.idx, return_adjacency=True)
    R, info = degree_preserving_rewire(raw, n_swaps_per_edge=n_swaps_per_edge)
    exp.ppi = PPIOperator(column_stochastic(R), device=exp.device)
    frames = []
    for f in range(5):
        fa = w.fold(f)
        cases = exp.cases(w.outer_context(f), w.labelled(fa["test_diseases"]), f)
        df, _ = exp.run_cases(cases, sel["alpha"], 0.0, omegas=(sel["omega"],), draws=1,
                              with_mechanism=False, adaptive=False, tag="rewired")
        frames.append(df)
    rew = pd.concat(frames, ignore_index=True)
    rew.to_parquet(paths6.SUPPORT / "rewired_episodes.parquet", index=False)
    real = pd.read_parquet(paths6.SUPPORT / "support_test_episodes.parquet")
    real = real[(real["tag"] == "test_nu0") & (real["draw"] == 0)]
    out = {"generated_utc": now_utc(), "rewiring": info, "alpha": sel["alpha"], "comparisons": {}}
    fam = w.family_of
    for k in (1, 3, 5):
        a = rew[(rew["method"] == "rwr_equal") & (rew["regime"] == "S_fixed") & (rew["k"] == k)] \
            .groupby("disease_id")["recall10"].mean()
        b = real[(real["method"] == "rwr_equal") & (real["regime"] == "S_fixed") & (real["k"] == k)] \
            .groupby("disease_id")["recall10"].mean()
        common = sorted(set(a.index) & set(b.index))
        d = a.loc[common].to_numpy() - b.loc[common].to_numpy()
        fams = np.array([fam.get(x, x) for x in common])
        lo, hi, _ = family_bootstrap(d, fams, draws=10000)
        out["comparisons"][f"k{k}"] = {"rewired_minus_real": float(d.mean()), "ci_low": lo,
                                       "ci_high": hi, "n_diseases": len(common),
                                       "real": float(b.loc[common].mean()),
                                       "rewired": float(a.loc[common].mean())}
    dump_json(paths6.SUPPORT / "rewiring_control.json", out)
    return out


if __name__ == "__main__":
    paths6.ensure()
    print(json.dumps(run(), indent=2, default=str))

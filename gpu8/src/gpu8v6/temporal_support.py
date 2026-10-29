"""Temporal task X2 with the V6 support path and cut-off-era inputs only.

Task (section 3.1): diseases that already had relations at the 2023Q4 cut-off;
their 2023Q4 relations are the exposed support S_d, the relations that appeared
after the cut-off are the query Q_d.  Every gene except S_d is ranked (background
regime; no candidate list is needed for this task).

Inputs, all no later than the cut-off (see ``temporal_input_history_audit``):
2023Q4 CTD edges and PMIDs, STRING v12.0 (July 2023), the functional bridge from
GO 2023-10-09 and Reactome 86 mapped with HGNC 2023-10-01.  alpha, nu and the
evidence-group rule are the ones frozen on the development families; nothing
is re-tuned on this task.

Compared on the same diseases: RWR from equal seeds on PPI and on P_mix, RWR
from evidence-group seeds (traceable diseases), and a disease-independent
degree prior computed from the 2023Q4 relations.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8 import paths as p5
from gpu8.contexts import build_P
from gpu8.stats import family_bootstrap, paired_permutation

from . import paths6
from .common import World6, dump_json, load_manifest, now_utc
from .gene_features import load_bridge
from .propagate import BridgeOperator, MixOperator, PPIOperator, SeedColumnCache
from .support_line import metrics_on
from .support_seed import build_support_seed


def run(device: str = "cuda") -> dict:
    w = World6()
    sel = load_manifest("support_v6_frozen.json")
    hdir = p5.GPU8_ROOT / "tables_historical"
    h_rel = pd.read_parquet(hdir / "relations.parquet")
    h_ev = pd.read_parquet(hdir / "evidence.parquet")
    prim = set(w.cfg.data.primary_evidence)
    h_rel = h_rel[h_rel["relation_type"].isin(prim)]
    fut = pd.read_csv(p5.RESULTS_DIR / "temporal" / "temporal_future_positives.csv", dtype=str)
    g_ix = w.idx.g_ix
    rel_of = {(r.disease_id, r.gene_id): r.relation_id for r in h_rel.itertuples(index=False)}
    ev = h_ev[h_ev["pmid"].astype(str).str.len() > 0]
    rel_pm = ev.groupby("relation_id")["pmid"].apply(set).to_dict()
    support = h_rel.groupby("disease_id")["gene_id"].apply(lambda s: sorted(set(s))).to_dict()
    query = fut.groupby("disease_id")["gene_id"].apply(lambda s: sorted(set(s))).to_dict()
    ids = sorted(d for d in query if d in support and d in w.idx.d_ix)
    P, deg = build_P(w.ppi, w.idx)
    ppi = PPIOperator(P, device=device)
    bridge = BridgeOperator(load_bridge("hist2023"), device=device)
    ops = {"ppi": ppi, f"mix{sel['nu']:g}": MixOperator(ppi, bridge, sel["nu"])}
    ctd_deg = np.zeros(w.n_g)
    for g, c in h_rel["gene_id"].value_counts().items():
        if g in g_ix:
            ctd_deg[g_ix[g]] = c
    prior = np.log1p(ctd_deg) + 0.5 * np.log1p(deg)
    rows = []
    caches = {name: SeedColumnCache(op, sel["alpha"], device=device) for name, op in ops.items()}
    for d in ids:
        s_genes = [g for g in support[d] if g in g_ix]
        q = {g_ix[g] for g in query[d] if g in g_ix} - {g_ix[g] for g in s_genes}
        if not s_genes or not q:
            continue
        srows = np.array([g_ix[g] for g in s_genes])
        bg = np.setdiff1d(np.arange(w.n_g), srows)
        base = {"disease_id": d, "k_support": len(s_genes), "n_query": len(q)}
        rows.append({**base, "method": "degree_prior_2023",
                     **metrics_on(prior[bg], bg, q, len(q), ks=(10, 50, 100))})
        seed = build_support_seed(d, s_genes, rel_of, rel_pm)
        for name, cache in caches.items():
            p = cache.propagate(dict(zip(srows.tolist(), seed.u)))
            rows.append({**base, "method": f"rwr_equal_{name}",
                         **metrics_on(p[bg], bg, q, len(q), ks=(10, 50, 100))})
            if seed.all_traceable and name.startswith("mix"):
                pg = cache.propagate(dict(zip(srows.tolist(), seed.mixed(1.0))))
                rows.append({**base, "method": f"rwr_group_{name}", "l1_v_u": seed.l1(),
                             **metrics_on(pg[bg], bg, q, len(q), ks=(10, 50, 100))})
                rows.append({**base, "method": f"rwr_equal_{name}_traceable_subset",
                             **metrics_on(p[bg], bg, q, len(q), ks=(10, 50, 100))})
    df = pd.DataFrame(rows)
    out = paths6.EXTERNAL_RES
    df.to_csv(out / "temporal_x2_support_per_disease.csv", index=False, encoding="utf-8-sig")
    fam = w.family_of
    summ = {"generated_utc": now_utc(), "task": "X2 new relations of existing diseases, 2023Q4 cut-off",
            "alpha": sel["alpha"], "nu": sel["nu"], "n_diseases": int(df["disease_id"].nunique()),
            "methods": df.groupby("method")[["recall10", "recall50", "recall100", "mrr"]].mean()
            .reset_index().to_dict(orient="records"),
            "propagation_failures": {k: c.failures() for k, c in caches.items()}}
    comps = {}
    piv = df.pivot_table(index="disease_id", columns="method", values="recall50")
    for a, b in [(f"rwr_equal_mix{sel['nu']:g}", "degree_prior_2023"),
                 ("rwr_equal_ppi", "degree_prior_2023"),
                 (f"rwr_equal_mix{sel['nu']:g}", "rwr_equal_ppi"),
                 (f"rwr_group_mix{sel['nu']:g}", f"rwr_equal_mix{sel['nu']:g}_traceable_subset")]:
        if a in piv and b in piv:
            sub = piv[[a, b]].dropna()
            dlt = (sub[a] - sub[b]).to_numpy()
            f = np.array([fam.get(x, x) for x in sub.index])
            lo, hi, _ = family_bootstrap(dlt, f, draws=10000)
            comps[f"{a}__minus__{b}"] = {"metric": "recall50", "n": int(len(sub)),
                                         "delta": float(dlt.mean()), "ci_low": lo, "ci_high": hi,
                                         "permutation_p": paired_permutation(dlt, f, draws=10000)}
    summ["comparisons_recall50"] = comps
    summ["note"] = ("exploratory; the task differs from the cold-start main task and is never "
                    "pooled with it")
    dump_json(out / "temporal_x2_support_summary.json", summ)
    return summ


if __name__ == "__main__":
    paths6.ensure()
    print(json.dumps(run(), indent=2, default=str))

"""V6 section 6.2 — per-source report on the test folds, same budget for every source.

For each test disease (five folds, outer contexts):

* single-source coverage uses only the genes the source actually scores (its top
  500 with signal); no prior fill is credited to a source;
* the contribution of source s to the common list is the leave-one-source-out
  difference: the frozen fusion with and without s (weights of the remaining
  sources rescaled), at the same 500 budget;
* "new hits per 100 added candidates" = 100 x (positives among the genes s adds
  to the list) / (number of genes s adds);
* overlap = mean Jaccard of two sources' signal top-500 lists;
* rescued diseases = V5 zero-coverage diseases that the V6 list covers, with the
  sources carrying the rescued positives.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8 import paths as p5

from . import paths6
from .common import World6, dump_json, load_manifest, now_utc
from .retrieval import build_lists, max_normalise, top_n
from .sources import SOURCES, SourceEngine

DIRECT = ["pheno", "hier", "onto", "text", "gene_hpo"]


def run(device: str = "cuda", folds=(0, 1, 2, 3, 4)) -> dict:
    w = World6()
    eng = SourceEngine(w, device=device, with_v5=False)
    fr = load_manifest("retriever_v6_frozen.json")
    weights, mode = fr["weights"], fr["mode"]
    used = [s for s, x in weights.items() if x > 0]
    rows, pair_rows = [], []
    for fold in folds:
        fa = w.fold(fold)
        ids = w.labelled(fa["test_diseases"])
        cs = eng.context(w.outer_context(fold))
        prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(eng.ppi_deg)
        prior = prior / max(prior.max(), 1e-12)
        for b0 in range(0, len(ids), 200):
            chunk = ids[b0:b0 + 200]
            raw, info = eng.scores(cs, chunk, sources=SOURCES)
            norm = {s: max_normalise(v) for s, v in raw.items()}
            full, _, reason, _ = build_lists(norm, weights, prior, n=500, mode=mode)
            loo = {}
            for s in used:
                wt = {k: v for k, v in weights.items() if k != s}
                loo[s], _, _, _ = build_lists(norm, wt, prior, n=500, mode=mode)
            singles = {s: top_n(norm[s], 500) for s in SOURCES}
            for i, d in enumerate(chunk):
                q = set(w.query_rows(d).tolist())
                nq = len(q)
                fl = set(full[i].tolist())
                rec = {"disease_id": d, "fold": fold, "n_query": nq,
                       "coverage_full": len(q & fl) / max(nq, 1),
                       "n_prior_fill": int((reason[i] == 2).sum())}
                sig_sets = {}
                for s in SOURCES:
                    lst = [g for g in singles[s][i] if norm[s][i, g] > 0]
                    sig_sets[s] = set(lst)
                    rec[f"single_cov_{s}"] = len(q & sig_sets[s]) / max(nq, 1)
                    rec[f"single_n_{s}"] = len(lst)
                for s in used:
                    ll = set(loo[s][i].tolist())
                    added = fl - ll
                    rec[f"loo_cov_{s}"] = len(q & ll) / max(nq, 1)
                    rec[f"added_by_{s}"] = len(added)
                    rec[f"added_hits_{s}"] = len(added & q)
                for s in DIRECT:
                    carried = {g for g in fl if norm[s][i, g] > 0}
                    rec[f"carried_hits_{s}"] = len(carried & q)
                rows.append(rec)
                for a in range(len(SOURCES)):
                    for b in range(a + 1, len(SOURCES)):
                        A, B = sig_sets[SOURCES[a]], sig_sets[SOURCES[b]]
                        if A or B:
                            pair_rows.append({"a": SOURCES[a], "b": SOURCES[b],
                                              "jaccard": len(A & B) / max(len(A | B), 1)})
        print(f"  source report fold {fold}", flush=True)
    df = pd.DataFrame(rows)
    out = paths6.RETRIEVAL
    df.to_parquet(out / "source_report_test_per_disease.parquet", index=False)
    summary = {"generated_utc": now_utc(), "n_diseases": int(len(df)),
               "coverage_full": float(df["coverage_full"].mean()), "sources": {}}
    for s in SOURCES:
        rec = {"single_source_coverage": float(df[f"single_cov_{s}"].mean()),
               "mean_signal_genes_top500": float(df[f"single_n_{s}"].mean())}
        if s in used:
            rec["coverage_without_s"] = float(df[f"loo_cov_{s}"].mean())
            rec["coverage_contribution"] = float(df["coverage_full"].mean() - df[f"loo_cov_{s}"].mean())
            add = df[f"added_by_{s}"].sum()
            rec["genes_added_per_disease"] = float(df[f"added_by_{s}"].mean())
            rec["new_hits_per_100_added"] = float(100 * df[f"added_hits_{s}"].sum() / max(add, 1))
        if s in DIRECT:
            rec["positives_carried_in_final_list"] = int(df[f"carried_hits_{s}"].sum())
        summary["sources"][s] = rec
    ov = pd.DataFrame(pair_rows).groupby(["a", "b"])["jaccard"].mean().reset_index()
    ov.to_csv(out / "source_overlap_test.csv", index=False, encoding="utf-8-sig")
    summary["overlap_mean_jaccard"] = ov.to_dict(orient="records")
    # rescued V5 zero-coverage diseases
    dec = pd.read_parquet(paths6.AUDIT / "oracle_decomposition_v5_per_disease.parquet")
    v5 = dec[dec["method"] == "b3_cpu_fusion"].set_index("disease_id")
    zero5 = set(v5.index[v5["n_query_in_pool"] == 0])
    sub = df[df["disease_id"].isin(zero5)]
    summary["v5_zero_coverage_diseases"] = int(len(sub))
    summary["rescued_by_v6"] = int((sub["coverage_full"] > 0).sum())
    summary["rescued_share"] = float((sub["coverage_full"] > 0).mean()) if len(sub) else 0.0
    dump_json(out / "source_report_test.json", summary)
    return summary


if __name__ == "__main__":
    paths6.ensure()
    print(json.dumps(run(), indent=2, default=str)[:5000])

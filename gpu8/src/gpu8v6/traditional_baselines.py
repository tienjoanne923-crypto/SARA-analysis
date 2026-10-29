"""Stand-alone traditional methods on the V6 test diseases (full gene background).

Each classical single-evidence approach ranks all ~19,861 genes on its own, for
the same 4,238 five-fold test diseases and in the same outer contexts (reference
relations only), so its top-10 can be compared with the V6 pipeline:

  popularity        genes ranked by how often they are studied (visible CTD
                    degree + PPI degree); disease independent
  pheno_v5          V5 phenotype-similarity transfer (SVD HPO view)
  pheno             HPO similarity transfer, IC-weighted, repaired mapping
                    (Phenomizer-style disease similarity)
  gene_hpo          disease phenotype vs gene phenotype profile
                    (Exomiser/Phenomizer-style gene matching)
  hier              ontology-neighbour transfer (CTD parents, distance <= 2)
  onto              MeSH information-content neighbour transfer
  text              masked disease-text similarity transfer
  rwr_pheno         network propagation seeded by phenotype-transfer genes

Genes a method gives no score keep the popularity order as a tie-break, exactly
as the pipeline's fallback does.  Metrics: Recall@10/@50, hit@10/@50.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths6
from .common import World6, dump_json, now_utc
from .retrieval import max_normalise
from .sources import SOURCES, SourceEngine

METHODS = ["pheno_v5", "pheno", "gene_hpo", "hier", "onto", "text", "rwr_pheno"]


def topk_metrics(scores: np.ndarray, q: np.ndarray, ks=(10, 50)) -> dict:
    order = np.argsort(-scores, kind="stable")
    qs = set(q.tolist())
    out = {}
    for k in ks:
        h = sum(1 for g in order[:k] if int(g) in qs)
        out[f"recall{k}"] = h / max(len(qs), 1)
        out[f"hit{k}"] = float(h > 0)
    return out


def run(device: str = "cuda") -> dict:
    w = World6()
    eng = SourceEngine(w, device=device, with_v5=True)
    rows = []
    for fold in range(5):
        ids = w.labelled(w.fold(fold)["test_diseases"])
        cs = eng.context(w.outer_context(fold))
        prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(eng.ppi_deg)
        prior = prior / max(prior.max(), 1e-12)
        tie = 1e-9 * prior
        for b0 in range(0, len(ids), 200):
            chunk = ids[b0:b0 + 200]
            raw, _ = eng.scores(cs, chunk, sources=list(SOURCES) + ["pheno_v5"])
            norm = {s: max_normalise(v) for s, v in raw.items()}
            for i, d in enumerate(chunk):
                q = w.query_rows(d)
                rows.append({"disease_id": d, "fold": fold, "method": "popularity",
                             **topk_metrics(prior, q)})
                for m in METHODS:
                    if m in norm:
                        rows.append({"disease_id": d, "fold": fold, "method": m,
                                     **topk_metrics(norm[m][i].astype(np.float64) + tie, q)})
        print(f"  traditional fold {fold}: {len(ids)} diseases", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(paths6.RESULTS / "tables" / "traditional_per_disease.parquet", index=False)
    summ = df.groupby("method")[["recall10", "hit10", "recall50", "hit50"]].mean()
    z = pd.read_csv(paths6.TABLES_OUT / "z_main_endpoints.csv").set_index("method")
    for m, lab in (("v6_lightgbm", "V6 pipeline (LightGBM)"), ("v6_F7", "V6 pipeline (tuned fusion F7)"),
                   ("v5_b3_cpu_fusion", "V5 CPU fusion"), ("v5_b4_lightgbm", "V5 LightGBM"),
                   ("v5_b8_meta_ridge", "V5 meta-ridge")):
        summ.loc[lab] = [z.loc[m, "recall10_end"], z.loc[m, "hit10"], z.loc[m, "recall50_end"],
                         z.loc[m, "hit50"]]
    summ = summ.sort_values("recall10", ascending=False)
    summ.to_csv(paths6.TABLES_OUT / "traditional_comparison.csv", encoding="utf-8-sig")
    # support setting: classical RWR from the five known genes over the whole network
    ep = pd.read_parquet(paths6.SUPPORT / "support_test_episodes.parquet")
    ep = ep[(ep["k"].isin([0, 5])) & (ep["regime"] != "meta")]
    sup = (ep.groupby(["tag", "method", "regime", "k", "disease_id"])[["recall10", "recall50"]]
           .mean().groupby(["tag", "method", "regime", "k"]).mean().reset_index())
    sup.to_csv(paths6.TABLES_OUT / "traditional_support_comparison.csv", index=False,
               encoding="utf-8-sig")
    out = {"generated_utc": now_utc(), "zero_support": summ.reset_index().to_dict(orient="records"),
           "support_k5": sup.to_dict(orient="records"), "n_diseases": int(df["disease_id"].nunique())}
    dump_json(paths6.TABLES_OUT / "traditional_comparison.json", out)
    return out


if __name__ == "__main__":
    paths6.ensure()
    r = run()
    print(pd.DataFrame(r["zero_support"]).round(4).to_string())
    print(pd.DataFrame(r["support_k5"]).round(4).to_string())

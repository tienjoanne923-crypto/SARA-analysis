"""Post-hoc diagnostics for the V6 status document (exploratory, test folds).

1. Where the remaining zero-coverage diseases come from: routing class, known-
   gene burden, broad category; how full their 500 list was with real signal;
   whether a positive sits in ranks 501-1000; and, for covered diseases, where
   the first positive lands inside the list.
2. Text-mask sensitivity: the text source alone with the main symbol mask (level
   1) versus the stricter mask that also removes full gene names (level 2).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths6
from .common import World6, dump_json, now_utc
from .retrieval import max_normalise
from .sources import SourceEngine

CAND = paths6.CACHES / "candidates"


def zero_coverage(w: World6) -> dict:
    zc = pd.read_parquet(paths6.TABLES_OUT / "z_per_disease_collapsed.parquet")
    lg = zc[zc["method"] == "v6_lightgbm"].set_index("disease_id")
    av = pd.read_csv(paths6.AUDIT / "availability_per_disease.csv").set_index("disease_id")
    out = {"n": int(len(lg))}
    lg["zero"] = (lg["n_query_in_pool"] == 0).astype(float)
    lg["covered"] = 1.0 - lg["zero"]
    for col in ("route", "support_stratum"):
        g = lg.groupby(col).agg(n=("zero", "size"), zero_share=("zero", "mean"),
                                hit10=("hit10", "mean"), recall10=("recall10_end", "mean"))
        out[f"by_{col}"] = g.round(4).reset_index().to_dict(orient="records")
    lg["category"] = av["category_name"].reindex(lg.index)
    g = lg.groupby("category").agg(n=("zero", "size"), zero_share=("zero", "mean"),
                                   hit10=("hit10", "mean"), recall10=("recall10_end", "mean"))
    g = g[g["n"] >= 40].sort_values("zero_share", ascending=False)
    out["by_category_n_ge_40"] = g.round(4).reset_index().to_dict(orient="records")
    # list composition and ranks 501-1000 for the test diseases
    rows = []
    for f in range(5):
        cand = pd.read_parquet(CAND / f"f{f}_outer_candidates.parquet",
                               columns=["disease_id", "gene_row", "rank", "reason"])
        dis = pd.read_parquet(CAND / f"f{f}_outer_diseases.parquet")
        lab = pd.read_parquet(CAND / f"f{f}_outer_labels.parquet")
        test = set(w.labelled(w.fold(f)["test_diseases"]))
        cand = cand[cand["disease_id"].isin(test)]
        q = lab[lab["disease_id"].isin(test)].groupby("disease_id")["gene_row"].apply(set).to_dict()
        for d, sub in cand.groupby("disease_id"):
            s500 = sub[sub["rank"] <= 500]
            qs = q.get(d, set())
            in500 = set(s500["gene_row"]) & qs
            in1000 = set(sub["gene_row"]) & qs
            ranks = sub[sub["gene_row"].isin(qs)]["rank"]
            rows.append({"disease_id": d, "n_query": len(qs), "m500": len(in500),
                         "m1000": len(in1000),
                         "signal_genes_500": int((s500["reason"] == 0).sum()),
                         "first_rank_500": float(ranks[ranks <= 500].min()) if len(in500) else np.nan})
    comp = pd.DataFrame(rows).set_index("disease_id")
    zero = comp[comp["m500"] == 0]
    out["zero_n"] = int(len(zero))
    out["zero_share_list_mostly_prior_fill"] = float((zero["signal_genes_500"] < 250).mean())
    out["zero_share_list_full_signal"] = float((zero["signal_genes_500"] >= 500).mean())
    out["zero_share_hit_in_501_1000"] = float((zero["m1000"] > 0).mean())
    cov = comp[comp["m500"] > 0]
    fr = cov["first_rank_500"]
    out["covered_n"] = int(len(cov))
    out["covered_first_positive_rank_median"] = float(fr.median())
    out["covered_first_positive_in_top10"] = float((fr <= 10).mean())
    out["covered_first_positive_in_top50"] = float((fr <= 50).mean())
    out["covered_first_positive_beyond_100"] = float((fr > 100).mean())
    out["note"] = ("retrieval-order ranks (list position), not the scorer's ranks; the scorer's "
                   "first-hit rank is reported in z_main_endpoints")
    return out


def text_mask_sensitivity(w: World6) -> dict:
    res = {}
    for level in (1, 2):
        eng = SourceEngine(w, with_v5=False, text_level=level)
        rec, hit = [], []
        for f in range(5):
            ids = w.labelled(w.fold(f)["test_diseases"])
            cs = eng.context(w.outer_context(f))
            prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(eng.ppi_deg)
            tie = 1e-9 * prior / max(prior.max(), 1e-12)
            for b0 in range(0, len(ids), 250):
                chunk = ids[b0:b0 + 250]
                raw, _ = eng.scores(cs, chunk, sources=["text"])
                x = max_normalise(raw["text"]).astype(np.float64)
                for i, d in enumerate(chunk):
                    qs = set(w.query_rows(d).tolist())
                    top = np.argsort(-(x[i] + tie), kind="stable")[:10]
                    h = sum(1 for g in top if int(g) in qs)
                    rec.append(h / max(len(qs), 1))
                    hit.append(float(h > 0))
        res[f"level{level}"] = {"recall10": float(np.mean(rec)), "hit10": float(np.mean(hit))}
    res["difference_level2_minus_level1_recall10"] = res["level2"]["recall10"] - res["level1"]["recall10"]
    res["rule"] = "level 1 = symbol mask (main); level 2 = also full HGNC gene names"
    return res


def answer_visibility(w: World6) -> dict:
    """Is each test positive visible anywhere in the legal reference knowledge?

    visible_ctd   the gene has >= 1 reference relation in the outer context
    visible_hpo   the gene has a source-excluded gene-HPO profile
    ppi_only      neither, but the gene has PPI neighbours
    unseen        none of the above (transfer-type retrieval cannot reach it)
    """
    eng = SourceEngine(w, with_v5=False)
    rows = []
    for f in range(5):
        cs = eng.context(w.outer_context(f))
        deg = np.asarray(cs.A.sum(axis=0)).ravel()
        hpo = np.diff(cs.Gh.indptr) > 0
        cand = pd.read_parquet(CAND / f"f{f}_outer_candidates.parquet",
                               columns=["disease_id", "gene_row", "rank"])
        c500 = cand[cand["rank"] <= 500].groupby("disease_id")["gene_row"].apply(set).to_dict()
        for d in w.labelled(w.fold(f)["test_diseases"]):
            for g in w.query_rows(d):
                cls = ("visible_ctd" if deg[g] > 0 else "visible_hpo_only" if hpo[g]
                       else "ppi_only" if eng.ppi_deg[g] > 0 else "unseen")
                rows.append({"disease_id": d, "gene_row": int(g), "class": cls,
                             "in500": int(g) in c500.get(d, set()),
                             "single": len(w.positives[d]) == 1})
    df = pd.DataFrame(rows)
    g = df.groupby("class").agg(n=("in500", "size"), share_in_list=("in500", "mean"))
    g["share_of_positives"] = g["n"] / g["n"].sum()
    s = df[df["single"]].groupby("class").agg(n=("in500", "size"), share_in_list=("in500", "mean"))
    s["share_of_single_gene_positives"] = s["n"] / s["n"].sum()
    return {"all_positives": g.round(4).reset_index().to_dict(orient="records"),
            "single_gene_diseases": s.round(4).reset_index().to_dict(orient="records")}


def main() -> int:
    paths6.ensure()
    w = World6()
    out = {"generated_utc": now_utc(), "status": "exploratory diagnostics after registration",
           "zero_coverage": zero_coverage(w), "text_mask_sensitivity": text_mask_sensitivity(w),
           "answer_visibility": answer_visibility(w)}
    dump_json(paths6.AUDIT / "v6_diagnostics.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

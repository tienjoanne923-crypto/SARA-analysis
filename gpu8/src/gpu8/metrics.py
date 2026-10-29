"""Disease-level ranking metrics (plan section 7.3 and 16.1).

Three recall quantities are reported side by side and never conflated:

    Coverage_d        = |Q_d ∩ C_d| / |Q_d|
    Recall_end@K(d)   = |Q_d ∩ Top_K| / |Q_d|          (primary endpoint)
    Recall_cond@K(d)  = |Q_d ∩ Top_K| / |Q_d ∩ C_d|    (defined only when > 0)

A disease whose retriever found nothing still contributes Recall_end = 0; it is
never silently dropped.  When the first positive does not appear at all, the
rank is recorded as missing with a censoring flag rather than given a made-up
"successful" rank.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EPS = 1e-12


def dcg(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    gains = (2.0 ** labels) - 1.0
    discounts = np.log2(np.arange(2, len(labels) + 2))
    return float((gains / discounts).sum())


def rank_one_disease(scores: np.ndarray, labels: np.ndarray, gene_rows: np.ndarray,
                     ks=(10, 20, 50), n_query_total: int | None = None) -> dict:
    """Metrics for a single disease.  Ties broken by stable gene id order."""
    order = np.lexsort((gene_rows, -scores))
    lab = labels[order]
    n_in_pool = int(lab.sum())
    n_query = int(n_query_total if n_query_total is not None else n_in_pool)

    out: dict[str, float] = {}
    for k in ks:
        top = lab[:k]
        hits = float(top.sum())
        out[f"hit{k}"] = 1.0 if hits > 0 else 0.0
        out[f"recall{k}_end"] = hits / max(n_query, 1)
        out[f"recall{k}_cond"] = (hits / n_in_pool) if n_in_pool > 0 else math.nan
        ideal = np.array([1.0] * min(n_in_pool, k) + [0.0] * max(0, k - n_in_pool))
        out[f"ndcg{k}_end"] = dcg(top) / max(dcg(ideal), EPS) if n_in_pool > 0 else 0.0

    pos_idx = np.flatnonzero(lab > 0)
    if len(pos_idx):
        first = int(pos_idx[0]) + 1
        out["first_hit_rank"] = float(first)
        out["first_hit_censored"] = 0.0
        out["mrr"] = 1.0 / first
        prec = np.cumsum(lab)[pos_idx] / (pos_idx + 1)
        out["ap_end"] = float(prec.sum()) / max(n_query, 1)
        out["ap_cond"] = float(prec.mean())
        out["median_positive_rank"] = float(np.median(pos_idx + 1))
    else:
        out["first_hit_rank"] = math.nan
        out["first_hit_censored"] = 1.0
        out["mrr"] = 0.0
        out["ap_end"] = 0.0
        out["ap_cond"] = math.nan
        out["median_positive_rank"] = math.nan

    out["n_query"] = float(n_query)
    out["n_query_in_pool"] = float(n_in_pool)
    out["n_candidates"] = float(len(scores))
    out["coverage"] = (n_in_pool / n_query) if n_query else math.nan
    out["zero_coverage"] = 1.0 if (n_query > 0 and n_in_pool == 0) else 0.0
    out["recall10_upper_bound"] = (min(10, n_query) / n_query) if n_query else math.nan
    ub = out["recall10_upper_bound"]
    out["recall10_normalised"] = (out["recall10_end"] / ub) if ub and ub > 0 else math.nan
    return out


def macro_average(per_disease: pd.DataFrame, ks=(10, 20, 50)) -> dict:
    """Disease macro-average; conditional metrics report their own denominator."""
    if per_disease.empty:
        return {}
    out: dict[str, float] = {"n_diseases": float(len(per_disease))}
    end_cols = [f"recall{k}_end" for k in ks] + [f"ndcg{k}_end" for k in ks] \
        + [f"hit{k}" for k in ks] + ["mrr", "ap_end", "coverage", "zero_coverage",
                                     "recall10_normalised"]
    for c in end_cols:
        if c in per_disease.columns:
            vals = per_disease[c].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            out[c] = float(vals.mean()) if len(vals) else math.nan
    for k in ks:
        c = f"recall{k}_cond"
        if c in per_disease.columns:
            v = per_disease[c].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            out[c] = float(v.mean()) if len(v) else math.nan
            out[f"n_diseases_{c}"] = float(len(v))
    if "ap_cond" in per_disease.columns:
        v = per_disease["ap_cond"].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        out["ap_cond"] = float(v.mean()) if len(v) else math.nan
        out["n_diseases_ap_cond"] = float(len(v))
    for c in ("first_hit_rank", "median_positive_rank"):
        if c in per_disease.columns:
            v = per_disease[c].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            out[f"median_{c}"] = float(np.median(v)) if len(v) else math.nan
            out[f"n_uncensored_{c}"] = float(len(v))
    if "first_hit_censored" in per_disease.columns:
        out["censored_fraction"] = float(per_disease["first_hit_censored"].mean())
    if "n_query" in per_disease.columns:
        out["mean_query_positives"] = float(per_disease["n_query"].mean())
        out["frac_query_gt_10"] = float((per_disease["n_query"] > 10).mean())
    if "n_candidates" in per_disease.columns:
        out["mean_candidates"] = float(per_disease["n_candidates"].mean())
    # primary endpoint alias
    out["primary_recall10_end"] = out.get("recall10_end", math.nan)
    return out


def evaluate_predictions(pred: pd.DataFrame, ks=(10, 20, 50),
                         query_totals: dict[str, int] | None = None) -> tuple[pd.DataFrame, dict]:
    """pred needs columns disease_id, gene_row, score, label."""
    rows = []
    for did, sub in pred.groupby("disease_id", sort=True):
        m = rank_one_disease(sub["score"].to_numpy(dtype=float),
                             sub["label"].to_numpy(dtype=float),
                             sub["gene_row"].to_numpy(dtype=np.int64),
                             ks=ks,
                             n_query_total=(query_totals or {}).get(did))
        m["disease_id"] = did
        rows.append(m)
    per = pd.DataFrame(rows)
    return per, macro_average(per, ks=ks)

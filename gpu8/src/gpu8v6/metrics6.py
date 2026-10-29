"""Disease-level metrics with the oracle decomposition of V6 section 3.2.

For disease d with query positives Q_d (n_d = |Q_d|), candidate list C_d and the
top-K list of a ranker:

    m_d      = |Q_d ∩ C_d|                   positives that entered the pool
    h_d(K)   = |Q_d ∩ TopK(C_d)|             positives recovered in the top K
    c_d      = m_d / n_d                     coverage
    U_d(K)   = min(K, m_d) / n_d             oracle upper bound of any ranker on C_d
    R_d(K)   = h_d(K) / n_d                  end-to-end recall (primary endpoint)
    eta_d(K) = h_d(K) / min(K, m_d)          conditional efficiency, undefined if m_d = 0

so that R_d(K) = U_d(K) * eta_d(K) whenever m_d > 0 and R_d(K) = 0 otherwise.
The ranking loss is U_d - R_d and the retrieval capacity loss is
min(K, n_d)/n_d - U_d.  The multi-positive normalised recall
R^norm_d(K) = h_d(K) / min(K, n_d) is a secondary reading aid only.

Macro recall is the mean of R_d; it is *not* the product of macro coverage and
macro efficiency, and the evaluator never forms that product.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

KS = (10, 20, 50)


def _dcg(labels: np.ndarray) -> np.ndarray:
    """DCG along the last axis for binary labels."""
    k = labels.shape[-1]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    return (labels * disc).sum(axis=-1)


def evaluate_matrix(scores: np.ndarray, labels: np.ndarray, gene_rows: np.ndarray,
                    n_query: np.ndarray, valid: np.ndarray | None = None,
                    ks=KS) -> dict[str, np.ndarray]:
    """Metrics for D diseases laid out as padded [D, C] arrays.

    Ties are broken by ascending gene row, exactly as the V5 evaluator does.
    Padded (invalid) slots are pushed to the end and never counted.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    gene_rows = np.asarray(gene_rows, dtype=np.int64)
    n_query = np.asarray(n_query, dtype=np.float64)
    if valid is None:
        valid = np.ones_like(labels, dtype=bool)
    s = np.where(valid, scores, -np.inf)
    lab = np.where(valid, labels, 0.0)
    # lexsort: last key is primary -> sort by -score, then gene row
    order = np.lexsort((gene_rows, -s), axis=-1)
    ranked = np.take_along_axis(lab, order, axis=-1)
    m = lab.sum(axis=1)
    out: dict[str, np.ndarray] = {"n_query": n_query, "n_query_in_pool": m,
                                  "n_candidates": valid.sum(axis=1).astype(float)}
    nq = np.maximum(n_query, 1.0)
    out["coverage"] = np.where(n_query > 0, m / nq, np.nan)
    out["zero_coverage"] = ((n_query > 0) & (m == 0)).astype(float)
    for k in ks:
        h = ranked[:, :k].sum(axis=1)
        out[f"hit{k}"] = (h > 0).astype(float)
        out[f"h{k}"] = h
        out[f"recall{k}_end"] = h / nq
        cap = np.minimum(k, m)
        out[f"oracle_recall{k}"] = cap / nq
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"eta{k}"] = np.where(m > 0, h / np.maximum(cap, 1.0), np.nan)
            out[f"recall{k}_cond"] = np.where(m > 0, h / np.maximum(m, 1.0), np.nan)
        out[f"ranking_loss{k}"] = out[f"oracle_recall{k}"] - out[f"recall{k}_end"]
        out[f"capacity_loss{k}"] = np.minimum(k, n_query) / nq - out[f"oracle_recall{k}"]
        out[f"recall{k}_norm"] = h / np.maximum(np.minimum(k, n_query), 1.0)
        ideal = (np.arange(k)[None, :] < np.minimum(m, k)[:, None]).astype(float)
        idcg = _dcg(ideal)
        out[f"ndcg{k}_end"] = np.where(m > 0, _dcg(ranked[:, :k]) / np.maximum(idcg, 1e-12), 0.0)
    pos_any = ranked > 0
    first = np.where(pos_any.any(axis=1), pos_any.argmax(axis=1) + 1, 0)
    out["first_hit_rank"] = np.where(first > 0, first, np.nan).astype(float)
    out["first_hit_censored"] = (first == 0).astype(float)
    out["mrr"] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
    cum = np.cumsum(ranked, axis=1)
    ranks = np.arange(1, ranked.shape[1] + 1)[None, :]
    prec = np.where(ranked > 0, cum / ranks, 0.0)
    out["ap_end"] = prec.sum(axis=1) / nq
    return out


def per_disease_frame(disease_ids, metrics: dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame({k: v for k, v in metrics.items()})
    df.insert(0, "disease_id", list(disease_ids))
    return df


MACRO_COLS = ["recall10_end", "recall20_end", "recall50_end", "ndcg10_end", "mrr",
              "ap_end", "hit10", "coverage", "zero_coverage", "oracle_recall10",
              "ranking_loss10", "capacity_loss10", "recall10_norm"]


def macro(per: pd.DataFrame) -> dict:
    """Disease macro averages; conditional quantities carry their own n."""
    if per.empty:
        return {"n_diseases": 0}
    out = {"n_diseases": int(len(per))}
    for c in MACRO_COLS:
        if c in per.columns:
            v = per[c].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            out[c] = float(v.mean()) if len(v) else math.nan
    for c in ("eta10", "recall10_cond"):
        if c in per.columns:
            v = per[c].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            out[c] = float(v.mean()) if len(v) else math.nan
            out[f"n_{c}"] = int(len(v))
    if "first_hit_rank" in per.columns:
        v = per["first_hit_rank"].dropna()
        out["median_first_hit_rank"] = float(v.median()) if len(v) else math.nan
    if "zero_coverage" in per.columns:
        out["n_zero_coverage"] = int(per["zero_coverage"].sum())
    return out


def check_decomposition(per: pd.DataFrame, k: int = 10, tol: float = 1e-12) -> dict:
    """Verify R = U * eta disease by disease and the macro identity."""
    R = per[f"recall{k}_end"].to_numpy(dtype=float)
    U = per[f"oracle_recall{k}"].to_numpy(dtype=float)
    eta = per[f"eta{k}"].to_numpy(dtype=float)
    covered = ~np.isnan(eta)
    prod = np.where(covered, U * np.nan_to_num(eta), 0.0)
    max_err = float(np.max(np.abs(prod - R))) if len(R) else 0.0
    macro_r = float(R.mean()) if len(R) else math.nan
    macro_prod_wrong = float(np.nanmean(per["coverage"]) * np.nanmean(eta)) if covered.any() else math.nan
    return {"k": k, "n": int(len(R)), "n_covered": int(covered.sum()),
            "max_abs_error_R_equals_U_eta": max_err, "identity_holds": max_err <= tol,
            "macro_recall": macro_r, "macro_mean_of_products": float(prod.mean()) if len(prod) else math.nan,
            "macro_coverage_times_macro_eta_NOT_recall": macro_prod_wrong}


def outcome_classes(per: pd.DataFrame, k: int = 10) -> pd.Series:
    """Disease outcome: not retrieved / in pool but missed / recovered."""
    m = per["n_query_in_pool"].to_numpy(dtype=float)
    h = per[f"h{k}"].to_numpy(dtype=float) if f"h{k}" in per.columns else \
        per[f"recall{k}_end"].to_numpy(dtype=float) * per["n_query"].to_numpy(dtype=float)
    cls = np.where(m == 0, "not_retrieved", np.where(h > 0, "recovered", "in_pool_missed"))
    return pd.Series(cls, index=per.index)

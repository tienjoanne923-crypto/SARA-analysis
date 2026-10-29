"""V8.1 §13 — end-to-end metrics, cold recovery, U x eta decomposition, family bootstrap.

    R_d@K   = |Q_d n Top_K(C_d)| / |Q_d|
    R^c_d@K = |Q^c_d n Top_K(C_d)| / |Q^c_d|          (only |Q^c_d| > 0; counts reported)
    R^{c->all}_d@K = |Q^c_d n Top_K(C_d)| / |Q_d|
    U_d(K) = min(K, m_d) / n_d,  eta_d(K) = h_d / min(K, m_d),  R_d = U_d eta_d

Q^c_d = query genes that are reference-path cold in the evaluation context.
Seeds are averaged per disease (never unioned); every disease with queries stays
in the denominator (zero coverage counts as a failure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gpu8v71.evaluate71 import fast_family_bootstrap  # pure function (no writes)

from .feats8 import Labels


def load_pred(path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def per_disease(pred: dict, labels: Labels, role: str, cold_g: np.ndarray, src_g: np.ndarray | None = None,
                gtype: dict | None = None, ks=(10, 20, 50)) -> pd.DataFrame:
    ids = [str(x) for x in pred["ids"]]
    genes = pred["genes"].astype(np.int64)
    s = pred["score"].astype(np.float64)
    valid = pred["valid"].astype(bool) if "valid" in pred else np.isfinite(s)
    rows = []
    for i, d in enumerate(ids):
        q = labels.rows(d, role)
        n = len(q)
        if n == 0:
            continue
        v = valid[i]
        gi, si = genes[i, v], s[i, v]
        o = np.lexsort((gi, -si))
        ranked = gi[o]
        qs = set(q.tolist())
        qc = {g for g in qs if cold_g[g]}
        inpool = qs & set(gi.tolist())
        rec = {"disease_id": d, "n_query": n, "m_pool": len(inpool), "coverage500": len(inpool) / n,
               "n_query_cold": len(qc), "m_pool_cold": len(qc & inpool), "pool_size": int(v.sum())}
        for k in ks:
            top = set(ranked[:k].tolist())
            h = len(qs & top)
            rec[f"hits{k}"] = h
            rec[f"recall{k}"] = h / n
            hc = len(qc & top)
            rec[f"cold_hits{k}"] = hc
            rec[f"cold_recall{k}"] = hc / len(qc) if qc else np.nan
            rec[f"cold_to_all{k}"] = hc / n
        mk = min(10, len(inpool))
        rec["U10"] = mk / n
        rec["eta10"] = rec["hits10"] / mk if mk else np.nan
        rec["hit10"] = float(rec["hits10"] > 0)
        rec["burden_norm_recall10"] = rec["hits10"] / min(10, n)
        if src_g is not None:
            qu = {g for g in qs if src_g[g] == 0}
            rec["n_query_sem_unavail"] = len(qu)
            rec["hits10_sem_unavail"] = len(qu & set(ranked[:10].tolist()))
            rec["pool_sem_unavail"] = len(qu & inpool)
        if gtype is not None:
            for t, mask in gtype.items():
                qt = {g for g in qs if mask[g]}
                rec[f"n_query_{t}"] = len(qt)
                rec[f"hits10_{t}"] = len(qt & set(ranked[:10].tolist()))
                rec[f"pool_{t}"] = len(qt & inpool)
        rows.append(rec)
    return pd.DataFrame(rows)


def macro(per: pd.DataFrame) -> dict:
    out = {"n_diseases": int(len(per)), "n_queries": int(per["n_query"].sum()),
           "recall10": float(per["recall10"].mean()), "recall20": float(per["recall20"].mean()),
           "recall50": float(per["recall50"].mean()), "hit10": float(per["hit10"].mean()),
           "coverage500": float(per["coverage500"].mean()), "U10": float(per["U10"].mean()),
           "eta10_defined_mean": float(per["eta10"].mean()),
           "zero_coverage_share": float((per["m_pool"] == 0).mean()),
           "cold_recall10": float(per["cold_recall10"].mean()),
           "n_diseases_cold": int((per["n_query_cold"] > 0).sum()),
           "n_queries_cold": int(per["n_query_cold"].sum()),
           "cold_to_all10": float(per["cold_to_all10"].mean())}
    if "n_query_sem_unavail" in per:
        n = per["n_query_sem_unavail"].sum()
        out["sem_unavail_queries"] = int(n)
        out["sem_unavail_hits10_micro"] = float(per["hits10_sem_unavail"].sum() / n) if n else None
        out["sem_unavail_pool_micro"] = float(per["pool_sem_unavail"].sum() / n) if n else None
    return out


def seed_mean(pers: list[pd.DataFrame], cols=("recall10", "cold_recall10", "cold_to_all10", "coverage500",
                                              "hit10", "recall50")) -> pd.DataFrame:
    """Average per disease over seeds (no union of hits)."""
    df = pd.concat(pers, ignore_index=True)
    agg = {c: "mean" for c in cols if c in df}
    agg.update({"n_query": "first", "n_query_cold": "first"})
    return df.groupby("disease_id").agg(agg).reset_index()


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, family_of: dict, col: str = "recall10",
                 draws: int = 10000) -> dict:
    m = a[["disease_id", col]].merge(b[["disease_id", col]], on="disease_id", suffixes=("_a", "_b"))
    m = m.dropna()
    d = (m[f"{col}_a"] - m[f"{col}_b"]).to_numpy()
    fam = np.array([family_of.get(x, x) for x in m["disease_id"]])
    lo, hi = fast_family_bootstrap(d, fam, draws=draws)
    return {"metric": col, "n": int(len(d)), "a": float(m[f"{col}_a"].mean()), "b": float(m[f"{col}_b"].mean()),
            "delta": float(d.mean()), "ci_low": lo, "ci_high": hi,
            "n_families": int(len(np.unique(fam))), "share_a_better": float((d > 0).mean()),
            "share_a_worse": float((d < 0).mean())}

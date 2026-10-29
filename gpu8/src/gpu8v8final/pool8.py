"""V8.1 §8 — the availability-aware strict 500 candidate list.

    b = 1 / (60 + r_B),   e = 1 / (60 + r_S) if m = 1 else 0,
    R = (w b + (1 - w) m e) / (w + (1 - w) m),   0 < w <= 1,
    C_d = Top500_{g in G} R(d, g),   ties by gene row (= ascending GeneID).

r_B is the 1-based position in the frozen V6 full ordering of the context,
r_S the semantic rank among semantically available genes.  If m = 0 then R = b
exactly (no extra down-weighting for a missing source).  The plain RRF control
uses the same w:  R_plain = w b + (1 - w) m e.  No query positive is ever added;
the list is exactly 500 distinct genes; three counts are recorded.
"""
from __future__ import annotations

import numpy as np

from .common8 import list_hash

K0 = 60.0
W_GRID = (0.3, 0.5, 0.7, 0.9)
BUDGET = 500


def rrf_score(rB: np.ndarray, rS: np.ndarray, m: np.ndarray, w: float, plain: bool = False) -> np.ndarray:
    b = 1.0 / (K0 + rB.astype(np.float64))
    e = np.where(m, 1.0 / (K0 + np.maximum(rS, 1).astype(np.float64)), 0.0)
    num = w * b + (1.0 - w) * m * e
    if plain:
        return num
    return num / (w + (1.0 - w) * m)


def top_budget(R: np.ndarray, n: int = BUDGET) -> np.ndarray:
    """[D, n] genes by (-R, gene row); exact (no floating offsets)."""
    D, G = R.shape
    out = np.empty((D, n), dtype=np.int64)
    rows = np.arange(G)
    kk = min(G, n + 64)
    for i in range(D):
        part = np.argpartition(-R[i], kk - 1)[:kk]
        thr = R[i, part].min()
        cand = np.flatnonzero(R[i] >= thr)          # every gene tied at the threshold
        o = cand[np.lexsort((cand, -R[i, cand]))]
        out[i] = o[:n]
    return out


def strict_pool(rB: np.ndarray, rS: np.ndarray, m: np.ndarray, w: float, plain: bool = False,
                n: int = BUDGET) -> dict:
    R = rrf_score(rB, rS, m, w, plain)
    genes = top_budget(R, n)
    pos = np.take_along_axis(R, genes, axis=1)
    return {"genes": genes, "R": pos.astype(np.float64), "w": w, "plain": plain,
            "counts": {"n_retrieval_scored": int(rB.shape[1]), "n_second_stage_scored": n, "n_output": n}}


def pool_hash(ids: list, genes: np.ndarray) -> str:
    return list_hash(f"{d}:{','.join(map(str, sorted(g.tolist())))}" for d, g in zip(ids, genes))


def v6_list(rB: np.ndarray, n: int = BUDGET) -> np.ndarray:
    """The frozen V6 500-list in V6 order."""
    return np.argsort(rB, axis=1, kind="stable")[:, :n].astype(np.int64)

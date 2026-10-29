"""V6 sections 6.1-6.2 — ``retrieve_candidates``.

Every source is ranked over the full gene background of the context.  Fusion is
either a weighted sum of per-disease max-normalised scores (``score``) or a
weighted reciprocal-rank fusion (``rrf``); neither ever re-standardises inside a
500-gene list.  A source that returned nothing for a disease is masked and its
weight is redistributed to the remaining sources, so a zero is never read as
"proven irrelevant".

The budget is fixed at 500.  An optional quota reserves the top ``q`` genes of
every available source before the fused ranking fills the rest.  When fewer than
500 genes carry any signal the list is completed from a disease-independent
degree prior, and every such gene carries ``fallback_reason``; those genes are
never described as evidence-based retrieval.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

EPS = 1e-12
FALLBACK_SIGNAL = "fused_signal"
FALLBACK_PRIOR = "no_signal_degree_prior_fill"
FALLBACK_QUOTA = "source_quota"


def max_normalise(X: np.ndarray) -> np.ndarray:
    m = X.max(axis=1, keepdims=True)
    return np.where(m > 0, X / np.maximum(m, EPS), 0.0).astype(np.float32)


def rank_transform(X: np.ndarray, k0: float = 60.0) -> np.ndarray:
    """1 / (k0 + rank) for genes with a positive score, 0 otherwise (rank 1 = best)."""
    out = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        x = X[i]
        pos = np.flatnonzero(x > 0)
        if len(pos) == 0:
            continue
        order = pos[np.lexsort((pos, -x[pos]))]
        out[i, order] = 1.0 / (k0 + np.arange(1, len(order) + 1))
    return out


def simplex_grid(n_sources: int, step: float = 0.25) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    out = []
    for c in itertools.combinations(range(units + n_sources - 1), n_sources - 1):
        prev, parts = -1, []
        for x in c:
            parts.append(x - prev - 1)
            prev = x
        parts.append(units + n_sources - 1 - prev - 1)
        out.append(tuple(p * step for p in parts))
    return out


def fuse(norm: dict[str, np.ndarray], weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Weighted fusion with per-disease redistribution over available sources.

    Returns (fused [D x n_g], available weight mass [D]).
    """
    names = [s for s, wt in weights.items() if wt > 0 and s in norm]
    if not names:
        D, n = next(iter(norm.values())).shape
        return np.zeros((D, n), dtype=np.float32), np.zeros(D)
    avail = np.stack([(norm[s].max(axis=1) > 0).astype(np.float64) for s in names], axis=1)
    w = np.array([weights[s] for s in names], dtype=np.float64)
    mass = avail @ w
    scale = np.where(mass > 0, 1.0 / np.maximum(mass, EPS), 0.0)
    fused = np.zeros_like(norm[names[0]], dtype=np.float32)
    for j, s in enumerate(names):
        coef = (w[j] * avail[:, j] * scale).astype(np.float32)
        fused += coef[:, None] * norm[s]
    return fused, mass


def top_n(scores: np.ndarray, n: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of the n best genes per row, ties broken by ascending gene row."""
    D, G = scores.shape
    s = scores.astype(np.float64).copy()
    if exclude is not None:
        s[exclude] = -np.inf
    # a tiny gene-row-ordered offset makes argpartition respect the tie-break
    s = s - np.arange(G)[None, :] * 1e-15
    kk = min(n, G)
    part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
    ps = np.take_along_axis(s, part, axis=1)
    order = np.lexsort((part, -ps), axis=1)
    return np.take_along_axis(part, order, axis=1)


def build_lists(norm: dict[str, np.ndarray], weights: dict[str, float], prior: np.ndarray,
                n: int = 500, quota: int = 0, exclude: np.ndarray | None = None,
                mode: str = "score") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Candidate lists for a batch.

    Returns (genes [D x n], fused score of each gene [D x n],
             reason code [D x n] (0 signal, 1 quota, 2 prior fill), fused [D x G]).
    """
    if mode == "rrf":
        base = {s: rank_transform(v) for s, v in norm.items()}
    else:
        base = norm
    fused, _ = fuse(base, weights)
    D, G = fused.shape
    blocked = np.zeros((D, G), dtype=bool) if exclude is None else exclude.copy()
    genes = np.full((D, n), -1, dtype=np.int64)
    reason = np.full((D, n), 2, dtype=np.int8)
    filled = np.zeros(D, dtype=int)
    if quota > 0:
        for s in [s for s, wt in weights.items() if wt > 0 and s in norm]:
            x = norm[s]
            tq = top_n(x, quota, exclude=blocked)
            for i in range(D):
                picks = [g for g in tq[i] if x[i, g] > 0 and not blocked[i, g]]
                room = n - filled[i]
                picks = picks[:max(0, room)]
                if picks:
                    genes[i, filled[i]:filled[i] + len(picks)] = picks
                    reason[i, filled[i]:filled[i] + len(picks)] = 1
                    blocked[i, picks] = True
                    filled[i] += len(picks)
    rest = top_n(fused, n, exclude=blocked)
    pri = top_n(np.repeat(prior[None, :], D, axis=0), n + int(filled.max()) + n,
                exclude=None)
    for i in range(D):
        room = n - filled[i]
        if room <= 0:
            continue
        cand = [g for g in rest[i] if fused[i, g] > 0][:room]
        genes[i, filled[i]:filled[i] + len(cand)] = cand
        reason[i, filled[i]:filled[i] + len(cand)] = 0
        blocked[i, cand] = True
        filled[i] += len(cand)
        room = n - filled[i]
        if room > 0:
            fill = [g for g in pri[i] if not blocked[i, g]][:room]
            genes[i, filled[i]:filled[i] + len(fill)] = fill
            reason[i, filled[i]:filled[i] + len(fill)] = 2
            filled[i] += len(fill)
    fs = np.take_along_axis(fused, np.maximum(genes, 0), axis=1)
    return genes, fs, reason, fused


def coverage_stats(genes: np.ndarray, query: list[np.ndarray], k: int = 10) -> dict:
    """Per-disease m_d, coverage and oracle U_d(k) of a candidate block."""
    m = np.array([len(np.intersect1d(genes[i], q)) for i, q in enumerate(query)])
    nq = np.array([len(q) for q in query])
    ok = nq > 0
    cov = np.where(ok, m / np.maximum(nq, 1), np.nan)
    U = np.where(ok, np.minimum(k, m) / np.maximum(nq, 1), np.nan)
    return {"m": m, "n": nq, "coverage": cov, "U": U, "zero": (ok & (m == 0)).astype(float)}


REASON_NAMES = {0: FALLBACK_SIGNAL, 1: FALLBACK_QUOTA, 2: FALLBACK_PRIOR}

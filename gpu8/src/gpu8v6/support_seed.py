"""V6 section 10.2-10.3 — ``build_support_seed``.

Only the *currently exposed* support S_d and the traceable original evidence of
those support relations are read.  The bipartite incidence B^S (support gene x
source id) marks gene g as supported by source c when c is one of the PMIDs of
relation (d, g).  Duplicate records of the same source id are merged (B is
binary); sources are **not** merged transitively across papers.

    u_d(g) = 1 / |S_d|
    v_d(g) = (1 / C) * sum_c B_gc / sum_{g'} B_g'c          (each source has equal
                                                             mass, split evenly)
    p0_d   = (1 - eta) u_d + eta v_d,   eta in {0, 0.25, 0.5, 1}

If any exposed support gene has no traceable source, the main mechanism
analysis does not use the episode (the application path falls back to eta = 0
and the fallback is counted).  The "unknown as singleton" variant is a separate
sensitivity analysis.  Group sizes are computed inside S_d only; hidden
positives and query genes never enter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class SeedResult:
    genes: list[str]
    u: np.ndarray
    v: np.ndarray | None
    B: np.ndarray                 # k x C binary
    sources: list[str]
    all_traceable: bool
    source_coverage: float
    n_sources: int

    @property
    def k(self) -> int:
        return len(self.genes)

    def l1(self) -> float:
        return float(np.abs(self.v - self.u).sum()) if self.v is not None else float("nan")

    def entropy(self, which: str = "v") -> float:
        x = self.v if which == "v" else self.u
        if x is None:
            return float("nan")
        x = x[x > 0]
        return float(-(x * np.log(x)).sum())

    def mixed(self, eta: float) -> np.ndarray:
        if self.v is None or eta == 0:
            return self.u.copy()
        return (1.0 - eta) * self.u + eta * self.v


def group_weights(B: np.ndarray) -> np.ndarray:
    """v from a binary k x C incidence (every column has at least one gene)."""
    B = (np.asarray(B) > 0).astype(np.float64)
    col = B.sum(axis=0)
    keep = col > 0
    if not keep.any():
        return np.full(B.shape[0], 1.0 / max(B.shape[0], 1))
    Bk = B[:, keep] / col[keep][None, :]
    return Bk.sum(axis=1) / keep.sum()


def build_support_seed(disease_id: str, support_genes: list[str], relation_of: dict,
                       relation_pmids: dict, unknown_as_singleton: bool = False) -> SeedResult:
    genes = sorted(set(support_genes))
    k = len(genes)
    u = np.full(k, 1.0 / k) if k else np.zeros(0)
    per_gene = []
    for g in genes:
        rid = relation_of.get((disease_id, g), "")
        pm = sorted(set(relation_pmids.get(rid, set())))
        per_gene.append(pm)
    traceable = [len(p) > 0 for p in per_gene]
    all_tr = bool(k) and all(traceable)
    if unknown_as_singleton:
        per_gene = [p if p else [f"UNKNOWN::{g}"] for p, g in zip(per_gene, genes)]
    sources = sorted({s for p in per_gene for s in p})
    s_ix = {s: i for i, s in enumerate(sources)}
    B = np.zeros((k, len(sources)))
    for i, p in enumerate(per_gene):
        for s in p:
            B[i, s_ix[s]] = 1.0
    v = None
    if k and (all_tr or unknown_as_singleton) and len(sources):
        v = group_weights(B)
    return SeedResult(genes=genes, u=u, v=v, B=B, sources=sources, all_traceable=all_tr,
                      source_coverage=float(np.mean(traceable)) if k else 0.0,
                      n_sources=len(sources))


def shuffled_weights(res: SeedResult, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Source-membership permutation within degree strata.

    Rows of B (the source pattern of each support gene) are permuted among
    genes of the same stratum: every source keeps its size, every gene keeps
    its number of sources, the support count is unchanged; only which gene
    carries which pattern changes.
    """
    if res.v is None:
        return res.u.copy()
    perm = np.arange(res.k)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if len(idx) > 1:
            perm[idx] = idx[rng.permutation(len(idx))]
    return group_weights(res.B[perm])


def degree_weights(deg: np.ndarray) -> np.ndarray:
    w = 1.0 / np.sqrt(1.0 + np.asarray(deg, dtype=np.float64))
    return w / w.sum()

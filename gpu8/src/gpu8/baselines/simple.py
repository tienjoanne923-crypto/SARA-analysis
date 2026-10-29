"""Training-free and lightly-trained comparators: B0, B1, B2, B3, B10, B11."""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.stats import hypergeom

from ..config import Gpu8Config
from ..episodes import EpisodeBuilder

EPS = 1e-12


def stable_tie_noise(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Deterministic ~1e-12 jitter keyed by gene id, so ties break reproducibly."""
    out = np.zeros(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        h = hashlib.md5(str(int(r)).encode()).digest()
        out[i] = (int.from_bytes(h[:4], "little") / 2 ** 32) * 1e-12
    return values.astype(np.float64) + out


# --------------------------------------------------------------------------
# B0 gene degree / topology prior
# --------------------------------------------------------------------------
def b0_degree_prior(cfg: Gpu8Config, builder: EpisodeBuilder, disease_id: str,
                    support_genes: list[str], candidate_rows: np.ndarray) -> np.ndarray:
    """A disease-independent reference: the query disease never enters the score.

    Named a degree prior rather than Adamic-Adar, because the target disease
    contributes nothing to the sum.
    """
    b = builder.b
    deg_ctd = b.gene_ctd_degree[candidate_rows].astype(np.float64)
    deg_ppi = b.gene_ppi_degree[candidate_rows].astype(np.float64)
    score = np.log1p(deg_ctd) + 0.5 * np.log1p(deg_ppi)
    return stable_tie_noise(score, candidate_rows)


# --------------------------------------------------------------------------
# B1 phenotype neighbour voting
# --------------------------------------------------------------------------
def b1_phenotype_vote(cfg: Gpu8Config, builder: EpisodeBuilder, disease_id: str,
                      support_genes: list[str], candidate_rows: np.ndarray) -> np.ndarray:
    r = builder.retriever_scores(disease_id, builder.b.idx.d_ix[disease_id])
    return stable_tie_noise(r["phenotype"][candidate_rows], candidate_rows)


# --------------------------------------------------------------------------
# B10 PPI random walk with restart, seeded by the exposed support
# --------------------------------------------------------------------------
def rwr_from_seeds(seed_rows: np.ndarray, P: sparse.csr_matrix, restart: float = 0.30,
                   max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    n = P.shape[0]
    if len(seed_rows) == 0:
        return np.zeros(n, dtype=np.float64)
    s = np.zeros(n, dtype=np.float64)
    s[seed_rows] = 1.0 / len(seed_rows)
    p = s.copy()
    for _ in range(max_iter):
        new = restart * s + (1.0 - restart) * (P @ p)
        if np.abs(new - p).sum() < tol:
            p = new
            break
        p = new
    return p


def b10_ppi_rwr(cfg: Gpu8Config, builder: EpisodeBuilder, disease_id: str,
                support_genes: list[str], candidate_rows: np.ndarray) -> np.ndarray | None:
    g_ix = builder.b.idx.g_ix
    seeds = np.array([g_ix[g] for g in support_genes if g in g_ix], dtype=np.int64)
    seeds = seeds[builder.b.gene_ppi_degree[seeds] > 0] if len(seeds) else seeds
    if len(seeds) == 0:
        return None          # not applicable without disease-specific seeds
    p = rwr_from_seeds(seeds, builder.b.P, restart=0.30)
    return stable_tie_noise(p[candidate_rows], candidate_rows)


# --------------------------------------------------------------------------
# B11 continuous-time quantum walk (classical numerical implementation)
# --------------------------------------------------------------------------
def b11_quantum_walk(cfg: Gpu8Config, builder: EpisodeBuilder, disease_id: str,
                     support_genes: list[str], candidate_rows: np.ndarray,
                     t: float = 0.45, diag: float = 5.0) -> np.ndarray | None:
    """s(g) = sum over seeds of |[exp(-i t A_S) e_s]_g|^2.

    Amplitudes are squared *per seed* and then summed; summing amplitudes first
    would introduce cross terms and is a different quantity.  The seed-only
    diagonal term follows the reference implementation.
    """
    g_ix = builder.b.idx.g_ix
    seeds = np.array([g_ix[g] for g in support_genes if g in g_ix], dtype=np.int64)
    if len(seeds) == 0:
        return None
    A = builder.b.ppi_adj
    if A is None:
        A = (builder.b.P > 0).astype(np.float64)
    n = A.shape[0]
    D = sparse.coo_matrix((np.full(len(seeds), diag, dtype=np.float64), (seeds, seeds)),
                          shape=(n, n)).tocsr()
    H = (A.astype(np.float64) + D).tocsc()
    Z = np.zeros((n, len(seeds)), dtype=np.complex128)
    Z[seeds, np.arange(len(seeds))] = 1.0
    try:
        res = expm_multiply(-1j * t * H, Z)
    except Exception:
        return None
    scores = (np.abs(res) ** 2).sum(axis=1)
    return stable_tie_noise(scores[candidate_rows], candidate_rows)


# --------------------------------------------------------------------------
# B10b DIAMOnD-like seed-neighbour enrichment (reported alongside RWR)
# --------------------------------------------------------------------------
def b10b_diamond_like(cfg: Gpu8Config, builder: EpisodeBuilder, disease_id: str,
                      support_genes: list[str], candidate_rows: np.ndarray) -> np.ndarray | None:
    g_ix = builder.b.idx.g_ix
    seeds = set(g_ix[g] for g in support_genes if g in g_ix)
    if not seeds:
        return None
    A = builder.b.ppi_adj if builder.b.ppi_adj is not None else (builder.b.P > 0)
    A = A.tocsr()
    n = A.shape[0]
    deg = np.diff(A.indptr)
    n_seed = len(seeds)
    out = np.zeros(len(candidate_rows), dtype=np.float64)
    for i, g in enumerate(candidate_rows):
        nb = set(A.indices[A.indptr[g]:A.indptr[g + 1]].tolist())
        links = len(nb & seeds)
        if links == 0 or deg[g] == 0:
            continue
        sf = hypergeom.sf(links - 1, n - 1, min(n_seed, n - 1), min(int(deg[g]), n - 1))
        out[i] = -math.log10(max(float(sf), 1e-300))
    return stable_tie_noise(out, candidate_rows)


# --------------------------------------------------------------------------
# B3 CPU-style fusion with a fixed (non-learned) ridge adaptation
# --------------------------------------------------------------------------
def b3_fusion_fixed_adapt(cfg: Gpu8Config, builder: EpisodeBuilder, disease_id: str,
                          support_genes: list[str], candidate_rows: np.ndarray,
                          weights: np.ndarray | None = None,
                          lam: float = 1.0, rho: float = 1.0) -> np.ndarray:
    """Fixed channel fusion plus a fixed 32-dim projection ridge update.

    Represents the previous CPU design: the projection is frozen (a fixed random
    orthogonal map of the cached views) rather than learned through the solver.
    """
    b = builder.b
    d_row = b.idx.d_ix[disease_id]
    r = builder.retriever_scores(disease_id, d_row)
    w = weights if weights is not None else np.array([0.4, 0.2, 0.3, 0.1])
    chans = np.stack([r["phenotype"], r["hierarchy"], r["rwr"], r["compat"]], axis=1)
    chans = np.log1p(np.maximum(chans, 0.0))
    base = chans @ w

    g_ix = b.idx.g_ix
    sup = np.array([g_ix[g] for g in support_genes if g in g_ix], dtype=np.int64)
    if len(sup) == 0:
        return stable_tie_noise(base[candidate_rows], candidate_rows)

    rng = np.random.default_rng(cfg.protocol.split_seed)
    dim = b.gene_views.shape[1]
    proj = rng.standard_normal((dim, 32)).astype(np.float32) / math.sqrt(32)
    phi_all = b.gene_views @ proj
    norms = np.linalg.norm(phi_all, axis=1, keepdims=True)
    phi_all = phi_all / np.maximum(norms, 1.0)

    from ..candidates import sample_anchors
    known = {g_ix[g] for g in builder.positives.get(disease_id, []) if g in g_ix}
    anchors = sample_anchors(cfg, disease_id, b.background_rows, set(sup.tolist()),
                             known, fused_scores=r["fused"])
    if len(anchors) == 0:
        return stable_tie_noise(base[candidate_rows], candidate_rows)

    D = (phi_all[sup][:, None, :] - phi_all[anchors][None, :, :]).reshape(-1, 32)
    margin = base[sup][:, None] - base[anchors][None, :]
    bvec = np.clip(1.0 - margin, -5, 5).reshape(-1)
    wv = np.full(len(bvec), 1.0 / max(len(anchors), 1), dtype=np.float64)
    G = D.T @ (D * wv[:, None]) + lam * np.eye(32)
    rhs = (D * (wv * bvec)[:, None]).sum(0)
    try:
        a = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        a = np.zeros(32)
    nrm = np.linalg.norm(a)
    if nrm > rho:
        a = a * (rho / max(nrm, EPS))
    score = base[candidate_rows] + phi_all[candidate_rows] @ a
    return stable_tie_noise(score, candidate_rows)

"""Candidate generation and the per-pair scalar channels / visibility fields.

Plan section 7: coverage and ranking are separated.  Two frozen retrievers —
phenotype transfer and ontology-hierarchy transfer — run over the *full* gene
background using only reference diseases and their visible relations, and a
fixed fusion rule keeps the top N unique genes.  The query positives are never
read, so a disease whose retriever finds nothing still enters the denominator.

Plan section 8.2 defines the four scalar evidence channels and appendix A.1 the
twelve visibility fields; both are produced here because they are per (disease,
gene) pair rather than per node.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .config import Gpu8Config
from .contexts import EntityIndex

EPS = 1e-8


# --------------------------------------------------------------------------
# deterministic per-disease RNG
# --------------------------------------------------------------------------
def disease_rng(disease_id: str, protocol_id: str, seed: int, tag: str = "") -> np.random.Generator:
    """Random state determined by disease id + protocol + fixed seed only.

    Never a function of the current candidate table, so adding irrelevant
    candidates cannot change anchor sampling.
    """
    h = hashlib.sha256(f"{disease_id}|{protocol_id}|{seed}|{tag}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


# --------------------------------------------------------------------------
# retrievers
# --------------------------------------------------------------------------
def phenotype_transfer_scores(d_row: int, D_V0: np.ndarray, A: sparse.csr_matrix,
                              ref_rows: np.ndarray, n_neighbours: int,
                              gene_deg: np.ndarray) -> tuple[np.ndarray, dict]:
    """v_d(g) = sum over top-k reference diseases of alpha_dd' A_d'g / max(1, deg(d')).

    alpha is the phenotype-similarity weight normalised to sum to one.
    """
    n_g = A.shape[1]
    if len(ref_rows) == 0:
        return np.zeros(n_g, dtype=np.float32), {"n_ref_used": 0, "max_sim": 0.0}
    sims = D_V0[ref_rows] @ D_V0[d_row]
    self_pos = np.searchsorted(ref_rows, d_row)
    if self_pos < len(ref_rows) and ref_rows[self_pos] == d_row:
        sims[self_pos] = -np.inf
    k = int(min(n_neighbours, len(ref_rows)))
    top = np.argpartition(-sims, k - 1)[:k] if k > 1 else np.array([int(np.argmax(sims))])
    w = np.where(np.isfinite(sims[top]), np.maximum(sims[top], 0.0), 0.0)
    if w.sum() <= 0:
        return np.zeros(n_g, dtype=np.float32), {"n_ref_used": 0, "max_sim": 0.0}
    alpha = w / w.sum()
    rows = ref_rows[top]
    deg_d = np.maximum(np.asarray(A[rows].sum(axis=1)).ravel(), 1.0).astype(np.float32)
    weights = (alpha / deg_d).astype(np.float32)
    scores = (A[rows].T @ weights).ravel().astype(np.float32)
    info = {"n_ref_used": int((w > 0).sum()), "max_sim": float(np.max(w))}
    return scores, info


def hierarchy_transfer_scores(d_row: int, hier_nb: dict, A: sparse.csr_matrix,
                              ref_set_rows: set) -> tuple[np.ndarray, dict]:
    """o_d(g): distance-decayed aggregation over ontology neighbours."""
    n_g = A.shape[1]
    nbs = hier_nb.get(d_row, {})
    rows, weights = [], []
    for nb, w in nbs.items():
        if nb in ref_set_rows:
            rows.append(nb)
            weights.append(w)
    if not rows:
        return np.zeros(n_g, dtype=np.float32), {"n_hier_ref": 0, "hier_coverage": 0.0}
    rows_a = np.array(rows, dtype=np.int64)
    w = np.array(weights, dtype=np.float32)
    w = w / max(w.sum(), EPS)
    deg_d = np.maximum(np.asarray(A[rows_a].sum(axis=1)).ravel(), 1.0).astype(np.float32)
    scores = (A[rows_a].T @ (w / deg_d).astype(np.float32)).ravel().astype(np.float32)
    coverage = float(len(rows) / max(1, len(nbs)))
    return scores, {"n_hier_ref": len(rows), "hier_coverage": coverage}


def rwr_channel(seed_vec: np.ndarray, P: sparse.csr_matrix, restart: float,
                iters: int) -> np.ndarray:
    """p_d = restart * v + (1 - restart) * P p, iterated `iters` times."""
    s = seed_vec.astype(np.float32)
    total = s.sum()
    if total <= 0:
        return np.zeros_like(s)
    s = s / total
    p = s.copy()
    for _ in range(iters):
        p = restart * s + (1.0 - restart) * np.asarray(P @ p, dtype=np.float32)
    return p


def compatibility_channel(d_row: int, D_pheno: np.ndarray, gene_seed: np.ndarray,
                          gene_ids: np.ndarray | None = None) -> np.ndarray:
    """Normalised cosine compatibility between the disease profile and Z0."""
    dv = D_pheno[d_row]
    dn = np.linalg.norm(dv)
    if dn <= EPS:
        return np.zeros(gene_seed.shape[0], dtype=np.float32)
    gn = np.linalg.norm(gene_seed, axis=1)
    cos = (gene_seed @ dv) / np.maximum(gn * dn, EPS)
    if gene_ids is not None:
        cos = cos[gene_ids]
    return cos.astype(np.float32)


# --------------------------------------------------------------------------
# candidate list
# --------------------------------------------------------------------------
@dataclass
class CandidateResult:
    disease_id: str
    gene_rows: np.ndarray            # indices into the gene background
    source: str                      # fused | phenotype | hierarchy
    n_phenotype_hits: int
    n_hierarchy_hits: int
    tie_break_fallback: float
    retriever_empty: bool


def generate_candidates(cfg: Gpu8Config, disease_id: str, d_row: int,
                        A: sparse.csr_matrix, D_V0: np.ndarray, ref_rows: np.ndarray,
                        hier_nb: dict, ref_set_rows: set, gene_deg: np.ndarray,
                        background_rows: np.ndarray, n_candidates: int | None = None,
                        source: str = "fused") -> CandidateResult:
    """Top-N unique genes from the frozen retrievers; query labels never read."""
    cc = cfg.candidates
    n = int(n_candidates or cc.n_candidates)
    ph, ph_info = phenotype_transfer_scores(d_row, D_V0, A, ref_rows,
                                            cc.phenotype_neighbours, gene_deg)
    hi, hi_info = hierarchy_transfer_scores(d_row, hier_nb, A, ref_set_rows)

    def norm(x):
        m = x.max()
        return x / m if m > 0 else x

    if source == "phenotype":
        fused = norm(ph)
    elif source == "hierarchy":
        fused = norm(hi)
    else:
        fused = cc.fusion_weight_phenotype * norm(ph) + cc.fusion_weight_hierarchy * norm(hi)

    scores = fused[background_rows]
    positive = int((scores > 0).sum())
    # stable tie-break by gene id order, applied as a tiny deterministic epsilon
    order = np.lexsort((background_rows, -scores))
    picked = background_rows[order[:n]]
    fallback = float(max(0, n - positive) / max(1, n))
    return CandidateResult(
        disease_id=disease_id, gene_rows=picked, source=source,
        n_phenotype_hits=int((ph[background_rows] > 0).sum()),
        n_hierarchy_hits=int((hi[background_rows] > 0).sum()),
        tie_break_fallback=fallback,
        retriever_empty=(positive == 0),
    )


def sample_anchors(cfg: Gpu8Config, disease_id: str, background_rows: np.ndarray,
                   exposed_support_rows: set, known_rows: set,
                   fused_scores: np.ndarray | None = None) -> np.ndarray:
    """U_d: half uniform, half from the frozen retriever's high-scoring set.

    Depends on the disease id, protocol id and a fixed seed only — never on the
    current candidate table, so batching cannot change the adaptation system.
    """
    cc = cfg.candidates
    rng = disease_rng(disease_id, cfg.protocol_id, cc.anchor_seed, "anchors")
    excluded = exposed_support_rows | known_rows
    if excluded:
        ex = np.fromiter(excluded, dtype=np.int64, count=len(excluded))
        pool = background_rows[~np.isin(background_rows, ex, assume_unique=False)]
    else:
        pool = background_rows
    if len(pool) == 0:
        return np.array([], dtype=np.int64)
    n_total = min(cc.n_anchors, len(pool))
    n_uniform = int(round(cc.anchor_uniform_fraction * n_total))
    n_top = n_total - n_uniform
    chosen = np.array([], dtype=np.int64)
    if n_top > 0 and fused_scores is not None:
        s = fused_scores[pool]
        order = np.lexsort((pool, -s))
        chosen = pool[order[:n_top]]
    if n_uniform > 0:
        remaining = pool[~np.isin(pool, chosen, assume_unique=False)] \
            if len(chosen) else pool
        if len(remaining):
            take = min(n_uniform, len(remaining))
            chosen = np.concatenate([chosen, rng.choice(remaining, size=take, replace=False)])
    return np.unique(chosen.astype(np.int64))


# --------------------------------------------------------------------------
# per-pair channels and visibility fields
# --------------------------------------------------------------------------
def scalar_channels(cfg: Gpu8Config, d_row: int, gene_rows: np.ndarray,
                    ph_scores: np.ndarray, hi_scores: np.ndarray,
                    rwr_scores: np.ndarray, compat: np.ndarray,
                    stats: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (channels [n x 4], mask [n x 4]).

    Skewed channels use a fixed log(1 + c x) transform, then the training-fixed
    mean/sd carried in ``stats``.  A missing channel is masked, never imputed as
    "confirmed irrelevant".
    """
    fc = cfg.features
    raw = np.stack([ph_scores[gene_rows], hi_scores[gene_rows],
                    rwr_scores[gene_rows], compat[gene_rows]], axis=1).astype(np.float32)
    mask = np.ones_like(raw)
    # a channel is absent when its retriever produced nothing at all
    for j in range(3):
        if not np.any(raw[:, j] > 0):
            mask[:, j] = 0.0
    out = np.log1p(fc.channel_log_c * np.maximum(raw, 0.0)).astype(np.float32)
    if stats:
        mu = np.asarray(stats.get("channel_mean", np.zeros(4)), dtype=np.float32)
        sd = np.asarray(stats.get("channel_std", np.ones(4)), dtype=np.float32)
        out = (out - mu) / np.maximum(sd, EPS)
        out = np.clip(out, -fc.clip_sigma, fc.clip_sigma)
    return out * mask, mask


QUALITY_FIELDS = [
    "disease_hpo_count",        # 0  log1p
    "hpo_mean_ic",              # 1  train scale
    "n_effective_reference",    # 2  /20
    "max_phenotype_similarity", # 3  [0,1]
    "hierarchy_reference_cov",  # 4  [0,1]
    "gene_train_degree",        # 5  log1p
    "gene_ppi_degree",          # 6  log1p
    "reference_source_complete",# 7  [0,1]
    "phenotype_transfer_missing",   # 8  binary
    "hierarchy_transfer_missing",   # 9  binary
    "propagation_missing",          # 10 binary
    "channel_disagreement",         # 11 clipped sd
]


def visibility_fields(cfg: Gpu8Config, gene_rows: np.ndarray, channels: np.ndarray,
                      mask: np.ndarray, disease_info: dict, gene_ctd_deg: np.ndarray,
                      gene_ppi_deg: np.ndarray, stats: dict | None = None) -> np.ndarray:
    """The twelve q_dg fields of appendix A.1 (no candidate-rank information)."""
    n = len(gene_rows)
    q = np.zeros((n, cfg.features.n_quality_fields), dtype=np.float32)
    q[:, 0] = np.log1p(float(disease_info.get("hpo_count", 0)))
    q[:, 1] = float(disease_info.get("hpo_mean_ic", 0.0))
    q[:, 2] = float(disease_info.get("n_ref_used", 0)) / float(cfg.candidates.phenotype_neighbours)
    q[:, 3] = float(np.clip(disease_info.get("max_sim", 0.0), 0.0, 1.0))
    q[:, 4] = float(np.clip(disease_info.get("hier_coverage", 0.0), 0.0, 1.0))
    q[:, 5] = np.log1p(gene_ctd_deg[gene_rows])
    q[:, 6] = np.log1p(gene_ppi_deg[gene_rows])
    q[:, 7] = float(np.clip(disease_info.get("reference_source_complete", 0.0), 0.0, 1.0))
    q[:, 8] = 1.0 - mask[:, 0]
    q[:, 9] = 1.0 - mask[:, 1]
    q[:, 10] = 1.0 - mask[:, 2]
    present = mask > 0
    with np.errstate(invalid="ignore"):
        counts = present.sum(axis=1)
        mean = np.where(counts > 0, (channels * present).sum(axis=1) / np.maximum(counts, 1), 0.0)
        var = np.where(counts > 1,
                       ((channels - mean[:, None]) ** 2 * present).sum(axis=1)
                       / np.maximum(counts - 1, 1), 0.0)
    q[:, 11] = np.clip(np.sqrt(np.maximum(var, 0.0)), 0.0, cfg.features.clip_sigma)
    if stats:
        mu = np.asarray(stats.get("quality_mean", np.zeros(q.shape[1])), dtype=np.float32)
        sd = np.asarray(stats.get("quality_std", np.ones(q.shape[1])), dtype=np.float32)
        cont = np.array([0, 1, 5, 6], dtype=int)   # only continuous fields rescaled
        q[:, cont] = np.clip((q[:, cont] - mu[cont]) / np.maximum(sd[cont], EPS),
                             -cfg.features.clip_sigma, cfg.features.clip_sigma)
    return q


# --------------------------------------------------------------------------
# coverage metrics (plan 7.3)
# --------------------------------------------------------------------------
def coverage_metrics(query_rows: set, candidate_rows: np.ndarray) -> dict:
    cand = set(candidate_rows.tolist())
    inter = query_rows & cand
    n_q = len(query_rows)
    return {
        "n_query": n_q,
        "n_candidates": len(cand),
        "n_query_in_candidates": len(inter),
        "coverage": (len(inter) / n_q) if n_q else float("nan"),
        "zero_coverage": int(n_q > 0 and len(inter) == 0),
    }

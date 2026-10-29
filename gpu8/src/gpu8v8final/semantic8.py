"""Semantic scores of one context: s(d, g) = z_d^T z^sem_g and the semantic ranks.

z^sem_g follows the fixed priority (observed GO -> teacher z^F; no GO + sequence
+ accepted student -> student output; otherwise unavailable).  The availability
m(d, g) = [z_d valid] * [z^sem_g exists]; missing is never written as a 0 score
that could be read as "low".  The parameter-free expression score s_E of the
V7.1 core is exported as a switchable secondary feature (rule / off / random
tissue condition).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from gpu8v71.core71 import row_norm

from .student8 import SRC_GO, SRC_SEQ, SRC_UNAVAILABLE
from .teacher8 import disease_z, gene_zF, gene_zS

N_G = 19861


@dataclass
class Semantic:
    ids: list
    s: np.ndarray            # [D, G] float32 (0 where m = 0; never read without m)
    m: np.ndarray            # [D, G] bool
    rS: np.ndarray           # [D, G] int32 semantic rank among available genes (0 = unavailable)
    src: np.ndarray          # [G] int8 semantic_source
    sE: np.ndarray           # [D, G] float32 expression score (0 where aE = 0)
    aE: np.ndarray           # [D, G] bool
    info: dict = field(default_factory=dict)


def full_rank(s: np.ndarray, m: np.ndarray) -> np.ndarray:
    """1-based rank among available genes by (-s, gene row); 0 where unavailable."""
    D, G = s.shape
    out = np.zeros((D, G), dtype=np.int32)
    rows = np.arange(G)
    for i in range(D):
        v = np.flatnonzero(m[i])
        if len(v) == 0:
            continue
        o = v[np.lexsort((v, -s[i, v].astype(np.float64)))]
        out[i, o] = np.arange(1, len(o) + 1, dtype=np.int32)
    return out


def gene_semantic(model, T: dict, student_z: np.ndarray | None = None,
                  student_genes: np.ndarray | None = None) -> tuple[torch.Tensor, np.ndarray]:
    """[G, 128] z^sem and the per-gene semantic_source."""
    zF, mF = gene_zF(model, T)
    src = np.where(mF.cpu().numpy(), SRC_GO, SRC_UNAVAILABLE).astype(np.int8)
    z = zF.clone()
    if student_z is not None and student_genes is not None and len(student_genes):
        sg = np.asarray(student_genes, dtype=np.int64)
        if (src[sg] == SRC_GO).any():
            raise RuntimeError("student output must never replace observed GO")
        z[torch.as_tensor(sg, device=z.device)] = torch.as_tensor(student_z, device=z.device, dtype=z.dtype)
        src[sg] = SRC_SEQ
    return z, src


@torch.no_grad()
def expression_scores(T: dict, rows: np.ndarray, q_override: np.ndarray | None = None,
                      scale: float = 3.0, min_cov: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    q = T["q"][torch.as_tensor(rows, device=T["q"].device)]
    mq = T["mq"][torch.as_tensor(rows, device=T["q"].device)]
    if q_override is not None:
        q = torch.as_tensor(q_override, device=q.device, dtype=q.dtype)
    c = q @ T["o"].T
    num = q @ T["e"].T
    sE = torch.tanh(num / (scale * c.clamp_min(1e-8)))
    aE = mq[:, None] & (c >= min_cov)
    sE = torch.where(aE, sE, torch.zeros_like(sE))
    return sE.float().cpu().numpy(), aE.cpu().numpy()


@torch.no_grad()
def semantic_for(model, T: dict, ids: list, d_ix: dict, zsem: torch.Tensor, src: np.ndarray,
                 expression: str = "rule", q_random: np.ndarray | None = None) -> Semantic:
    rows = np.array([d_ix[d] for d in ids], dtype=np.int64)
    z, ok = disease_z(model, T, rows)
    has = torch.as_tensor(src != SRC_UNAVAILABLE, device=z.device)
    D = len(rows)
    s = np.zeros((D, N_G), dtype=np.float32)
    m = np.zeros((D, N_G), dtype=bool)
    for b0 in range(0, D, 512):
        sc = z[b0:b0 + 512] @ zsem.T
        mv = ok[b0:b0 + 512, None] & has[None, :]
        s[b0:b0 + 512] = torch.where(mv, sc, torch.zeros_like(sc)).float().cpu().numpy()
        m[b0:b0 + 512] = mv.cpu().numpy()
    rS = full_rank(s, m)
    if expression == "off":
        sE = np.zeros((D, N_G), dtype=np.float32)
        aE = np.zeros((D, N_G), dtype=bool)
    else:
        qo = q_random[rows] if expression == "random" else None
        sE, aE = expression_scores(T, rows, qo)
    return Semantic(ids=list(ids), s=s, m=m, rS=rS, src=src, sE=sE, aE=aE,
                    info={"diseases_semantic_valid": int(ok.sum()), "genes_available": int(has.sum()),
                          "genes_observed_go": int((src == SRC_GO).sum()),
                          "genes_sequence_imputed": int((src == SRC_SEQ).sum())})


def random_tissue(T: dict, tag: str = "expr_random") -> np.ndarray:
    """Tissue vectors permuted among diseases that have one (fixed seed; label-free)."""
    from .common8 import rng_for
    q = T["q"].cpu().numpy()
    mq = T["mq"].cpu().numpy()
    out = q.copy()
    idx = np.flatnonzero(mq)
    out[idx] = q[rng_for(tag).permutation(idx)]
    return out

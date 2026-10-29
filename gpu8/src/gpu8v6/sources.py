"""V6 section 6 — the zero-support retrieval sources, computed per context.

For one graph context (a reference disease set) and a batch of target diseases,
every source produces a score over the **full gene background**:

  pheno      R1   HPO transfer: IC-weighted cosine on ancestor-closed valid terms
                  (repaired mapping), top-20 reference diseases, degree-normalised
  pheno_v5   R1'  the V5 SVD phenotype view (kept for the old-vs-new ladder)
  hier       R1b  V5 hierarchy transfer over CTD ParentIDs (distance <= 2)
  gene_hpo   R2   cosine between the disease profile and source-excluded gene HPO
  text       R3a  masked TF-IDF text, top-20 reference diseases
  onto       R3b  IC-weighted MeSH-tree ancestor similarity, top-20 references
  rwr_pheno       RWR seeded by the pheno scores (the V5 propagation channel,
                  now with residual stopping)
  compat          cosine between the disease profile and the reference-relation
                  gene phenotype profile A^T Q (V5 compatibility channel)

Reference sets are the V5 context references minus every strict equivalent of a
labelled disease held out of the context.  No function here receives a query
label; the retriever reads references, their visible relations and the target's
own legal information only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from gpu8.contexts import build_hierarchy_neighbours, build_P
from gpu8.candidates import hierarchy_transfer_scores

from . import paths6
from .common import ContextSpec, World6
from .disease_text import tfidf_matrix
from .gene_hpo import GeneHPOSource, HPOSpace, disease_profiles, gene_hpo_scores, ic_from_reference
from .ontology_v2 import OntologyIndex, ontology_transfer
from .propagate import PPIOperator, rwr

EPS = 1e-12
SOURCES = ["pheno", "hier", "onto", "text", "gene_hpo", "rwr_pheno", "compat"]
LADDER_EXTRA = ["pheno_v5", "gene_hpo_lenient"]


def _transfer(sims: np.ndarray, ref_rows: np.ndarray, A: sparse.csr_matrix, k: int
              ) -> tuple[np.ndarray, int, float]:
    n_g = A.shape[1]
    if len(ref_rows) == 0:
        return np.zeros(n_g, dtype=np.float32), 0, 0.0
    s = np.where(np.isfinite(sims), np.maximum(sims, 0.0), 0.0)
    kk = int(min(k, len(s)))
    top = np.argpartition(-s, kk - 1)[:kk] if kk > 1 else np.array([int(np.argmax(s))])
    w = s[top]
    keep = w > 0
    if not keep.any():
        return np.zeros(n_g, dtype=np.float32), 0, 0.0
    top, w = top[keep], w[keep]
    rows = ref_rows[top]
    wn = w / w.sum()
    deg = np.maximum(np.asarray(A[rows].sum(axis=1)).ravel(), 1.0)
    out = np.asarray(A[rows].T @ (wn / deg)).ravel().astype(np.float32)
    return out, int(len(rows)), float(w.max())


@dataclass
class ContextSources:
    ctx: ContextSpec
    reference: list[str]
    ref_rows: np.ndarray
    removed_equivalents: int
    A: sparse.csr_matrix
    ic: np.ndarray
    Qw_ref: sparse.csr_matrix
    Gh: sparse.csr_matrix
    Gh_info: dict
    Gh_len: sparse.csr_matrix | None
    Gh_len_info: dict
    Z0: np.ndarray
    ref_set_rows: set
    text_ref: sparse.csr_matrix
    info: dict = field(default_factory=dict)


class SourceEngine:
    def __init__(self, world: World6, device: str = "cuda", text_level: int = 1,
                 k_ref: int = 20, with_v5: bool = True, lenient: bool = False):
        t0 = time.time()
        self.w = world
        self.device = device
        self.k_ref = k_ref
        self.space = HPOSpace(world.hpo_terms)
        self.Q = disease_profiles(self.space, world.n_d, world.idx.d_ix)     # n_d x n_t
        self.has_hpo = np.diff(self.Q.indptr) > 0
        self.ghpo = GeneHPOSource(world, self.space)
        self.onto = OntologyIndex(world.diseases)
        ids = world.idx.diseases
        X, _ = tfidf_matrix(level=text_level, ids=ids)
        self.X = X
        self.has_text = np.diff(X.indptr) > 0
        P, deg = build_P(world.ppi, world.idx)
        self.ppi_deg = deg
        self.op = PPIOperator(P, device=device)
        self.hier_nb = build_hierarchy_neighbours(world.hierarchy, world.idx,
                                                  world.cfg.candidates.hierarchy_max_distance)
        self.eq = dict(zip(world.diseases["disease_id"], world.diseases["equivalence_group"]))
        self.with_v5 = with_v5
        self.lenient = lenient
        self._v5_cache: dict[str, np.ndarray] = {}
        self.init_seconds = time.time() - t0

    # -- context ---------------------------------------------------------
    def v6_reference(self, ctx: ContextSpec) -> tuple[list[str], int]:
        ref = set(ctx.reference)
        held_groups = {self.eq.get(d, d) for d in self.w.positives if d not in ref}
        keep = [d for d in ctx.reference if self.eq.get(d, d) not in held_groups]
        return keep, len(ctx.reference) - len(keep)

    def context(self, ctx: ContextSpec) -> ContextSources:
        ref, removed = self.v6_reference(ctx)
        ref_set = set(ref)
        cs = ContextSpec(ctx.context_id, ctx.fold, ctx.kind, ctx.block_index, ref, ctx.held_out)
        A = self.w.adjacency(cs)
        d_ix = self.w.idx.d_ix
        ref_rows = np.array(sorted(d_ix[d] for d in ref if d in d_ix), dtype=np.int64)
        ic = ic_from_reference(self.Q, ref_rows)
        W = sparse.diags(ic)
        Qw = (self.Q @ W).tocsr()
        norms = np.sqrt(np.asarray(Qw.multiply(Qw).sum(axis=1)).ravel())
        Qn = (sparse.diags(1.0 / np.maximum(norms, EPS)) @ Qw).tocsr()
        Gh, gi = self.ghpo.matrix(ref_set, lenient=False)
        Gl, gli = (self.ghpo.matrix(ref_set, lenient=True) if self.lenient else (None, {}))
        # compatibility: gene phenotype profile through visible reference relations
        gdeg = np.asarray(A.sum(axis=0)).ravel()
        Z0 = (A.T @ Qn).tocsr()
        Z0 = sparse.diags(1.0 / np.maximum(gdeg, 1.0)) @ Z0
        zn = np.sqrt(np.asarray(Z0.multiply(Z0).sum(axis=1)).ravel())
        Z0 = (sparse.diags(1.0 / np.maximum(zn, EPS)) @ Z0).tocsr()
        return ContextSources(ctx=cs, reference=ref, ref_rows=ref_rows, removed_equivalents=removed,
                              A=A, ic=ic, Qw_ref=Qn[ref_rows], Gh=Gh, Gh_info=gi, Gh_len=Gl,
                              Gh_len_info=gli, Z0=Z0, ref_set_rows=set(ref_rows.tolist()),
                              text_ref=self.X[ref_rows],
                              info={"n_reference_v5": len(ctx.reference), "n_reference_v6": len(ref),
                                    "removed_strict_equivalents": removed,
                                    "n_visible_relations": int(A.nnz), "gene_hpo": gi})

    def _pheno_v5(self, ctx_id: str) -> np.ndarray | None:
        if not self.with_v5:
            return None
        if ctx_id not in self._v5_cache:
            from gpu8.contexts import load_context
            try:
                _, arrays = load_context(ctx_id)
            except KeyError:
                return None
            dv = arrays["disease_views"]
            self._v5_cache = {ctx_id: dv[:, :dv.shape[1] // 4]}
        return self._v5_cache[ctx_id]

    # -- per-disease sources ---------------------------------------------
    def scores(self, cs: ContextSources, disease_ids: list[str],
               sources: list[str] | None = None) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
        """Return {source: [len(ids) x n_g] float32} and a per-disease info frame."""
        sources = sources or SOURCES
        w, d_ix, n_g = self.w, self.w.idx.d_ix, self.w.n_g
        rows = np.array([d_ix[d] for d in disease_ids], dtype=np.int64)
        D = len(rows)
        out = {s: np.zeros((D, n_g), dtype=np.float32) for s in sources}
        info = {"disease_id": list(disease_ids)}
        Qq = self.Q[rows]
        W = sparse.diags(cs.ic)
        Qqw = (Qq @ W).tocsr()
        qn = np.sqrt(np.asarray(Qqw.multiply(Qqw).sum(axis=1)).ravel())
        Qqn = (sparse.diags(1.0 / np.maximum(qn, EPS)) @ Qqw).tocsr()
        info["hpo_terms"] = np.diff(Qq.indptr)
        # pheno: IC cosine against references
        if "pheno" in sources or "rwr_pheno" in sources:
            S = (Qqn @ cs.Qw_ref.T).toarray()
            n_ref, mx = np.zeros(D, int), np.zeros(D)
            for i in range(D):
                sc, n_ref[i], mx[i] = _transfer(S[i], cs.ref_rows, cs.A, self.k_ref)
                out.setdefault("pheno", np.zeros((D, n_g), dtype=np.float32))[i] = sc
            info["pheno_n_ref"], info["pheno_max_sim"] = n_ref, mx
        if "pheno_v5" in sources:
            V0 = self._pheno_v5(cs.ctx.context_id)
            if V0 is not None:
                R = V0[cs.ref_rows]
                for i, r in enumerate(rows):
                    out["pheno_v5"][i], _, _ = _transfer(R @ V0[r], cs.ref_rows, cs.A, self.k_ref)
        if "hier" in sources:
            n_h = np.zeros(D, int)
            for i, r in enumerate(rows):
                sc, hi = hierarchy_transfer_scores(int(r), self.hier_nb, cs.A, cs.ref_set_rows)
                out["hier"][i] = sc
                n_h[i] = hi["n_hier_ref"]
            info["hier_n_ref"] = n_h
        if "onto" in sources:
            n_o, ic_o = np.zeros(D, int), np.zeros(D)
            ref_set = set(cs.reference)
            for i, did in enumerate(disease_ids):
                sc, oi = ontology_transfer(did, self.onto, ref_set, cs.A, d_ix, k=self.k_ref)
                out["onto"][i] = sc
                n_o[i], ic_o[i] = oi["onto_n_ref"], oi["onto_max_ic"]
            info["onto_n_ref"], info["onto_max_ic"] = n_o, ic_o
        if "text" in sources:
            St = (self.X[rows] @ cs.text_ref.T).toarray()
            n_t, mx_t = np.zeros(D, int), np.zeros(D)
            for i in range(D):
                out["text"][i], n_t[i], mx_t[i] = _transfer(St[i], cs.ref_rows, cs.A, self.k_ref)
            info["text_n_ref"], info["text_max_sim"] = n_t, mx_t
        if "gene_hpo" in sources:
            out["gene_hpo"] = gene_hpo_scores(Qq, cs.Gh, cs.ic)
        if "gene_hpo_lenient" in sources and cs.Gh_len is not None:
            out["gene_hpo_lenient"] = gene_hpo_scores(Qq, cs.Gh_len, cs.ic)
        if "compat" in sources:
            C = (Qqn @ cs.Z0.T).toarray().astype(np.float32)
            out["compat"] = np.maximum(C, 0.0)
        if "rwr_pheno" in sources:
            seeds = out["pheno"].T.astype(np.float64)
            if seeds.any():
                res = rwr(self.op, seeds, alpha=0.15, eps=1e-6, device=self.device)
                out["rwr_pheno"] = res.scores.T.astype(np.float32)
                info["rwr_pheno_iterations"] = np.full(D, res.iterations)
                info["rwr_pheno_converged"] = res.converged | (seeds.sum(0) == 0)
        info["has_hpo"] = self.has_hpo[rows].astype(int)
        info["has_text"] = self.has_text[rows].astype(int)
        info["onto_level"] = [self.onto.level(d) for d in disease_ids]
        info_df = pd.DataFrame({k: (list(v) if not isinstance(v, list) else v)
                                for k, v in info.items()})
        return out, info_df

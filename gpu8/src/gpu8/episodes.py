"""Episode assembly: turn a context + a disease into tensors the model can eat.

This is the single place where the visibility contract is enforced, so the
audit only has to check one code path:

* node views, channels and quality fields all come from the *context* the
  episode belongs to, and that context was built from reference relations only;
* the hidden query never reaches the graph, the degree, the retriever or the
  anchor sampler;
* anchors exclude the currently exposed support and the legally known relations,
  but are *not* cleaned using hidden labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from . import candidates as cand
from .config import Gpu8Config
from .contexts import EntityIndex, build_hierarchy_neighbours
from .support import (SupportGroups, build_relation_pmids, group_exposed_support,
                      support_summary, support_weights)


@dataclass
class ContextBundle:
    """Everything a context provides, loaded once and shared across episodes."""
    context_id: str
    idx: EntityIndex
    A: sparse.csr_matrix
    disease_views: np.ndarray
    gene_views: np.ndarray
    disease_mask: np.ndarray
    gene_mask: np.ndarray
    disease_pheno: np.ndarray
    gene_seed: np.ndarray
    gene_ctd_degree: np.ndarray
    gene_ppi_degree: np.ndarray
    ic: np.ndarray
    reference_rows: np.ndarray
    P: sparse.csr_matrix
    hier_nb: dict
    H: sparse.csr_matrix
    background_rows: np.ndarray
    stats: dict = field(default_factory=dict)
    ref_set_rows: set = field(default_factory=set)
    disease_V0: np.ndarray | None = None
    ppi_adj: sparse.csr_matrix | None = None

    def __post_init__(self):
        if self.disease_V0 is None:
            dim = self.disease_views.shape[1] // 4
            self.disease_V0 = self.disease_views[:, :dim]
        if not self.ref_set_rows:
            self.ref_set_rows = set(self.reference_rows.tolist())


def load_bundle(cfg: Gpu8Config, context_id: str, idx: EntityIndex,
                phenotypes: pd.DataFrame, hierarchy: pd.DataFrame,
                ppi: pd.DataFrame, background_rows: np.ndarray,
                stats: dict | None = None,
                shared: dict | None = None) -> ContextBundle:
    from .contexts import build_H, build_P, load_context
    meta, arrays = load_context(context_id)
    shared = shared or {}
    H = shared.get("H")
    if H is None:
        H = build_H(phenotypes, idx)
    P = shared.get("P")
    ppi_adj = shared.get("ppi_adj")
    if P is None:
        P, _, ppi_adj = build_P(ppi, idx, return_adjacency=True)
    hier_nb = shared.get("hier_nb")
    if hier_nb is None:
        hier_nb = build_hierarchy_neighbours(hierarchy, idx, cfg.candidates.hierarchy_max_distance)
    return ContextBundle(
        context_id=context_id, idx=idx, A=arrays["A"],
        disease_views=arrays["disease_views"], gene_views=arrays["gene_views"],
        disease_mask=arrays["disease_mask"], gene_mask=arrays["gene_mask"],
        disease_pheno=arrays["disease_pheno"], gene_seed=arrays["gene_seed"],
        gene_ctd_degree=arrays["gene_ctd_degree"], gene_ppi_degree=arrays["gene_ppi_degree"],
        ic=arrays["ic"], reference_rows=arrays["reference_rows"], P=P,
        hier_nb=hier_nb, H=H, background_rows=background_rows, stats=stats or {},
        ppi_adj=ppi_adj)


@dataclass
class Episode:
    disease_id: str
    d_row: int
    support_rows: np.ndarray
    anchor_rows: np.ndarray
    query_pos_rows: np.ndarray
    query_anchor_rows: np.ndarray
    candidate_rows: np.ndarray
    support_weights: np.ndarray
    support_summary: np.ndarray
    k: int
    info: dict = field(default_factory=dict)


class EpisodeBuilder:
    """Builds episodes and the tensors they need, with per-disease caching."""

    def __init__(self, cfg: Gpu8Config, bundle: ContextBundle,
                 relations: pd.DataFrame, evidence: pd.DataFrame):
        self.cfg = cfg
        self.b = bundle
        primary = set(cfg.data.primary_evidence)
        rel = relations[relations["relation_type"].isin(primary)]
        self.relation_of = {(r.disease_id, r.gene_id): r.relation_id
                            for r in rel.itertuples(index=False)}
        self.positives: dict[str, list[str]] = (
            rel.groupby("disease_id")["gene_id"].apply(list).to_dict())
        self.relation_pmids = build_relation_pmids(evidence)
        self._retriever_cache: dict[str, dict] = {}
        self._hard_pool_cache: dict[str, np.ndarray] = {}

    def hard_pool(self, disease_id: str, d_row: int, n: int | None = None) -> np.ndarray:
        """Top retrieved genes minus the disease's legally known positives.

        Training query anchors are drawn from here so that the negatives a task
        learns against come from the same distribution as the negatives the
        evaluation candidate list is made of.  Without this the model is trained
        against trivially easy random genes and the signal does not transfer to
        the retrieved pool.  The retriever never reads a label, so this remains
        inside the visibility contract.
        """
        if disease_id in self._hard_pool_cache:
            return self._hard_pool_cache[disease_id]
        rows, _ = self.candidate_rows(disease_id, d_row, n=n)
        g_ix = self.b.idx.g_ix
        known = {g_ix[g] for g in self.positives.get(disease_id, []) if g in g_ix}
        if known:
            ex = np.fromiter(known, dtype=np.int64, count=len(known))
            rows = rows[~np.isin(rows, ex)]
        self._hard_pool_cache[disease_id] = rows
        return rows

    # -- retrieval --------------------------------------------------------
    def retriever_scores(self, disease_id: str, d_row: int) -> dict:
        if disease_id in self._retriever_cache:
            return self._retriever_cache[disease_id]
        b, cfg = self.b, self.cfg
        ph, ph_info = cand.phenotype_transfer_scores(
            d_row, b.disease_V0, b.A, b.reference_rows,
            cfg.candidates.phenotype_neighbours, b.gene_ctd_degree)
        hi, hi_info = cand.hierarchy_transfer_scores(d_row, b.hier_nb, b.A, b.ref_set_rows)

        def norm(x):
            m = x.max()
            return x / m if m > 0 else x

        rwr = cand.rwr_channel(ph, b.P, cfg.features.rwr_restart, cfg.features.rwr_iters)
        fused = (cfg.candidates.fusion_weight_phenotype * norm(ph)
                 + cfg.candidates.fusion_weight_hierarchy * norm(hi)
                 + getattr(cfg.candidates, "fusion_weight_rwr", 0.0) * norm(rwr))
        compat = cand.compatibility_channel(d_row, b.disease_pheno, b.gene_seed)
        info = {**ph_info, **hi_info}
        hcount = int(b.H[d_row].nnz) if b.H.shape[0] > d_row else 0
        if hcount:
            cols = b.H[d_row].indices
            info["hpo_mean_ic"] = float(np.mean(b.ic[cols])) if len(cols) else 0.0
        else:
            info["hpo_mean_ic"] = 0.0
        info["hpo_count"] = hcount
        info["reference_source_complete"] = float(info.get("n_ref_used", 0) > 0)
        out = {"phenotype": ph, "hierarchy": hi, "fused": fused, "rwr": rwr,
               "compat": compat, "info": info}
        self._retriever_cache[disease_id] = out
        return out

    def candidate_rows(self, disease_id: str, d_row: int, n: int | None = None,
                       source: str = "fused",
                       exclude_rows: set | None = None) -> tuple[np.ndarray, dict]:
        b, cfg = self.b, self.cfg
        r = self.retriever_scores(disease_id, d_row)
        scores = r[source] if source in ("phenotype", "hierarchy") else r["fused"]
        pool = b.background_rows
        if exclude_rows:
            ex = np.fromiter(exclude_rows, dtype=np.int64, count=len(exclude_rows))
            pool = pool[~np.isin(pool, ex, assume_unique=False)]
        s = scores[pool]
        order = np.lexsort((pool, -s))
        n = int(n or cfg.candidates.n_candidates)
        picked = pool[order[:n]]
        positive = int((s > 0).sum())
        info = {"n_positive_scored": positive,
                "tie_break_fallback": float(max(0, n - positive) / max(1, n)),
                "retriever_empty": positive == 0,
                "source": source}
        return np.sort(picked), info

    # -- features ---------------------------------------------------------
    def pair_features(self, disease_id: str, d_row: int, gene_rows: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.retriever_scores(disease_id, d_row)
        ch, mask = cand.scalar_channels(self.cfg, d_row, gene_rows, r["phenotype"],
                                        r["hierarchy"], r["rwr"], r["compat"],
                                        stats=self.b.stats)
        q = cand.visibility_fields(self.cfg, gene_rows, ch, mask, r["info"],
                                   self.b.gene_ctd_degree, self.b.gene_ppi_degree,
                                   stats=self.b.stats)
        return ch, mask, q

    def node_features(self, d_row: int, gene_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dv = np.repeat(self.b.disease_views[d_row][None, :], len(gene_rows), axis=0)
        gv = self.b.gene_views[gene_rows]
        return dv, gv

    # -- support ----------------------------------------------------------
    def support_bundle(self, disease_id: str, support_genes: list[str],
                       equal_weight: bool = False) -> tuple[np.ndarray, np.ndarray, SupportGroups]:
        rel_of_gene = {g: self.relation_of.get((disease_id, g), "") for g in support_genes}
        groups = group_exposed_support(support_genes, rel_of_gene, self.relation_pmids)
        w = support_weights(groups, support_genes, equal_weight=equal_weight)
        summary = support_summary(groups, w)
        return w, summary, groups

    # -- episode ----------------------------------------------------------
    def build_episode(self, disease_id: str, support_genes: list[str],
                      query_genes: list[str], n_candidates: int | None = None,
                      max_query_pos: int | None = None,
                      max_query_anchor: int | None = None,
                      rng: np.random.Generator | None = None,
                      equal_weight: bool = False,
                      candidate_source: str = "fused",
                      exclude_pool: set | None = None,
                      with_candidates: bool = True) -> Episode | None:
        b, cfg, idx = self.b, self.cfg, self.b.idx
        d_row = idx.d_ix.get(disease_id)
        if d_row is None:
            return None
        g_ix = idx.g_ix
        support_rows = np.array([g_ix[g] for g in support_genes if g in g_ix], dtype=np.int64)
        query_rows = np.array([g_ix[g] for g in query_genes if g in g_ix], dtype=np.int64)

        # the maximal support pool never enters the candidate list (plan 6.5)
        if with_candidates:
            exclude = set(exclude_pool or set()) | set(support_rows.tolist())
            cand_rows, cinfo = self.candidate_rows(disease_id, d_row, n=n_candidates,
                                                   source=candidate_source,
                                                   exclude_rows=exclude)
        else:
            # training episodes never consult the candidate list
            cand_rows, cinfo = np.array([], dtype=np.int64), {}

        known = {g_ix[g] for g in self.positives.get(disease_id, []) if g in g_ix}
        r = self.retriever_scores(disease_id, d_row)
        anchors = cand.sample_anchors(cfg, disease_id, b.background_rows,
                                      set(support_rows.tolist()), known,
                                      fused_scores=r["fused"])

        if rng is not None and max_query_pos and len(query_rows) > max_query_pos:
            query_rows = np.sort(rng.choice(query_rows, size=max_query_pos, replace=False))
        qa = np.array([], dtype=np.int64)
        if max_query_anchor:
            rr = rng or cand.disease_rng(disease_id, cfg.protocol_id,
                                         cfg.candidates.anchor_seed, "query_anchor")
            block = known | set(anchors.tolist())
            ex = np.fromiter(block, dtype=np.int64, count=len(block)) if block \
                else np.array([], dtype=np.int64)
            # three quarters from the retrieved pool (hard), one quarter from the
            # wider background, so the model neither over-fits the retriever nor
            # trains only against trivially easy genes
            hard = self.hard_pool(disease_id, d_row)
            if len(ex):
                hard = hard[~np.isin(hard, ex)]
            n_hard = min(int(round(0.75 * max_query_anchor)), len(hard))
            picks = [rr.choice(hard, size=n_hard, replace=False)] if n_hard else []
            wide = b.background_rows[~np.isin(b.background_rows, ex)] if len(ex) \
                else b.background_rows
            n_wide = min(max_query_anchor - n_hard, len(wide))
            if n_wide > 0:
                picks.append(rr.choice(wide, size=n_wide, replace=False))
            if picks:
                qa = np.unique(np.concatenate(picks).astype(np.int64))

        w, summary, groups = self.support_bundle(disease_id, support_genes,
                                                 equal_weight=equal_weight)
        info = dict(cinfo)
        info.update({"n_groups": groups.n_groups, "n_effective": groups.n_effective,
                     "unknown_fraction": groups.unknown_fraction,
                     "provenance_complete": groups.provenance_complete})
        return Episode(disease_id=disease_id, d_row=d_row, support_rows=support_rows,
                       anchor_rows=anchors, query_pos_rows=query_rows,
                       query_anchor_rows=qa, candidate_rows=cand_rows,
                       support_weights=w, support_summary=summary,
                       k=len(support_rows), info=info)


# --------------------------------------------------------------------------
# training statistics (channel / quality scaling), fitted on training rows only
# --------------------------------------------------------------------------
def fit_feature_stats(builder: "EpisodeBuilder", disease_ids: list[str],
                      max_diseases: int = 400, per_disease: int = 64,
                      seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ids = list(disease_ids)
    rng.shuffle(ids)
    ids = ids[:max_diseases]
    ch_acc, q_acc = [], []
    for did in ids:
        d_row = builder.b.idx.d_ix.get(did)
        if d_row is None:
            continue
        rows, _ = builder.candidate_rows(did, d_row, n=per_disease)
        if len(rows) == 0:
            continue
        r = builder.retriever_scores(did, d_row)
        ch, mask = cand.scalar_channels(builder.cfg, d_row, rows, r["phenotype"],
                                        r["hierarchy"], r["rwr"], r["compat"], stats=None)
        q = cand.visibility_fields(builder.cfg, rows, ch, mask, r["info"],
                                   builder.b.gene_ctd_degree, builder.b.gene_ppi_degree,
                                   stats=None)
        ch_acc.append(ch)
        q_acc.append(q)
    if not ch_acc:
        return {}
    C = np.concatenate(ch_acc, axis=0)
    Q = np.concatenate(q_acc, axis=0)
    return {
        "channel_mean": C.mean(axis=0).tolist(),
        "channel_std": np.maximum(C.std(axis=0), 1e-6).tolist(),
        "quality_mean": Q.mean(axis=0).tolist(),
        "quality_std": np.maximum(Q.std(axis=0), 1e-6).tolist(),
        "n_rows": int(C.shape[0]), "n_diseases": len(ch_acc),
    }

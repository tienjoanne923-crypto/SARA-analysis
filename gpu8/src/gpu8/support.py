"""Support sets, evidence-dependence groups and the support weights.

Plan section 9.4 and appendix A.2.  Two ideas are kept strictly apart:

* *duplicate curation* — the same original report recorded by several database
  rows.  Removing it is an engineering correctness fix.
* *independent evidence* — genuinely separate studies.  We cannot observe this,
  so the code only ever produces an auditable **dependence proxy**: relations
  that share a PMID land in one group, groups share their mass, and records
  without provenance collapse into a single conservative ``unknown`` group.

Plan section 6.5 fixes the dose design: one maximal support pool per disease,
nested S1 subset S3 subset S5, the same disease / query / candidates for every
k, and ten nested order draws that only change which genes are exposed at k=1
and k=3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .candidates import disease_rng
from .config import Gpu8Config

EPS = 1e-8
UNKNOWN_GROUP = "__unknown__"


# --------------------------------------------------------------------------
# evidence groups
# --------------------------------------------------------------------------
def build_relation_pmids(evidence: pd.DataFrame) -> dict[str, set[str]]:
    ev = evidence[evidence["pmid"].astype(str).str.len() > 0]
    if ev.empty:
        return {}
    return ev.groupby("relation_id")["pmid"].apply(set).to_dict()


@dataclass
class SupportGroups:
    gene_to_group: dict[str, str]
    group_quality: dict[str, float]
    unknown_fraction: float
    n_groups: int
    n_effective: float
    provenance_complete: float


def group_exposed_support(gene_ids: list[str], relation_of_gene: dict[str, str],
                          relation_pmids: dict[str, set[str]],
                          quality_of_group: dict[str, float] | None = None) -> SupportGroups:
    """Union-find over shared PMIDs among the exposed support genes only."""
    parent: dict[str, str] = {g: g for g in gene_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, str] = {}
    unknown: list[str] = []
    for g in gene_ids:
        rid = relation_of_gene.get(g, "")
        pmids = relation_pmids.get(rid, set())
        if not pmids:
            unknown.append(g)
            continue
        for p in pmids:
            if p in owner:
                a, b = find(owner[p]), find(g)
                if a != b:
                    parent[b] = a
            else:
                owner[p] = g

    gene_to_group: dict[str, str] = {}
    for g in gene_ids:
        gene_to_group[g] = UNKNOWN_GROUP if g in unknown else find(g)

    groups = sorted(set(gene_to_group.values()))
    quality = {c: float((quality_of_group or {}).get(c, 1.0)) for c in groups}
    v = np.array([quality[c] for c in groups], dtype=np.float64)
    n_eff = float(min(len(gene_ids), (v.sum() ** 2) / (np.square(v).sum() + EPS))) if len(v) else 0.0
    unknown_frac = len(unknown) / max(1, len(gene_ids))
    return SupportGroups(gene_to_group=gene_to_group, group_quality=quality,
                         unknown_fraction=unknown_frac, n_groups=len(groups),
                         n_effective=n_eff, provenance_complete=1.0 - unknown_frac)


def support_weights(groups: SupportGroups, gene_ids: list[str],
                    equal_weight: bool = False) -> np.ndarray:
    """omega_g: group mass split evenly inside a group, then normalised."""
    if not gene_ids:
        return np.zeros(0, dtype=np.float32)
    if equal_weight:
        return np.full(len(gene_ids), 1.0 / len(gene_ids), dtype=np.float32)
    members: dict[str, list[str]] = {}
    for g in gene_ids:
        members.setdefault(groups.gene_to_group[g], []).append(g)
    raw = np.zeros(len(gene_ids), dtype=np.float64)
    for i, g in enumerate(gene_ids):
        c = groups.gene_to_group[g]
        raw[i] = groups.group_quality.get(c, 1.0) / max(1, len(members[c]))
    total = raw.sum()
    if total <= 0:
        return np.full(len(gene_ids), 1.0 / len(gene_ids), dtype=np.float32)
    return (raw / total).astype(np.float32)


def support_summary(groups: SupportGroups, weights: np.ndarray,
                    max_groups: int = 8) -> np.ndarray:
    """The four-dimensional r_d fed to the control head.

    Never reads a hidden query, a candidate ranking, or the number of citations
    of an unexposed relation.
    """
    n_eff_norm = float(groups.n_effective) / float(max_groups)
    dispersion = float(np.std(weights)) if len(weights) > 1 else 0.0
    comparable = 1.0 if len(set(groups.group_quality.values())) > 1 else 0.0
    return np.array([np.clip(n_eff_norm, 0.0, 1.0),
                     np.clip(groups.provenance_complete, 0.0, 1.0),
                     comparable,
                     np.clip(dispersion * 4.0, 0.0, 1.0)], dtype=np.float32)


# --------------------------------------------------------------------------
# dose design
# --------------------------------------------------------------------------
@dataclass
class DoseDesign:
    disease_id: str
    support_pool: list[str]           # S_max, always excluded from the candidates
    query_genes: list[str]            # Q_d, identical for every k
    nested_order: list[str]           # the order S1 subset S3 subset S5 follows
    eligible_k: list[int] = field(default_factory=list)

    def support_at(self, k: int) -> list[str]:
        return list(self.nested_order[:k])


def build_dose_design(cfg: Gpu8Config, disease_id: str, positives: list[str],
                      draw: int = 0) -> DoseDesign | None:
    """Fix S_max and Q_d first, then choose the nested exposure order."""
    p = cfg.protocol
    pos = sorted(positives)
    k_max = max(p.support_doses)
    if len(pos) < k_max + 1:
        return None
    rng = disease_rng(disease_id, cfg.protocol_id, p.split_seed, f"dose_pool")
    perm = rng.permutation(len(pos))
    pool = [pos[i] for i in perm[:k_max]]
    query = sorted(pos[i] for i in perm[k_max:])
    # the order draw only permutes the *pool*, so the query never changes
    order_rng = disease_rng(disease_id, cfg.protocol_id, p.split_seed, f"dose_order_{draw}")
    order = [pool[i] for i in order_rng.permutation(len(pool))]
    eligible = [k for k in p.support_doses if k <= len(pool)]
    return DoseDesign(disease_id=disease_id, support_pool=sorted(pool),
                      query_genes=query, nested_order=order, eligible_k=eligible)


def single_shot_only_set(cfg: Gpu8Config, positives_by_disease: dict[str, list[str]]
                         ) -> list[str]:
    """Diseases that can only ever offer one support gene: a separate cohort."""
    out = []
    for did, pos in positives_by_disease.items():
        if len(pos) == 2:          # exactly one support + one query
            out.append(did)
    return sorted(out)


# --------------------------------------------------------------------------
# training-time support sampling
# --------------------------------------------------------------------------
def sample_training_support(cfg: Gpu8Config, disease_id: str, positives: list[str],
                            k: int, rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Support / query split for one training episode.

    The support genes are *visible* to the model by construction; the query
    positives are the learning signal.  They never overlap.
    """
    pos = sorted(positives)
    if len(pos) <= k:
        return [], pos
    perm = rng.permutation(len(pos))
    support = [pos[i] for i in perm[:k]]
    query = [pos[i] for i in perm[k:]]
    return sorted(support), sorted(query)


def cross_disease_replacement(cfg: Gpu8Config, disease_id: str, support_genes: list[str],
                              n_replace: int, candidate_pool: np.ndarray,
                              gene_ctd_deg: np.ndarray, gene_ppi_deg: np.ndarray,
                              gene_index: dict[str, int], draw: int = 0) -> list[str]:
    """Replace n support genes with degree-matched genes from other diseases.

    Deliberately named 'replacement support', not 'wrong genes': a replaced gene
    may still be biologically related to the target disease.
    """
    if n_replace <= 0 or not support_genes:
        return list(support_genes)
    rng = disease_rng(disease_id, cfg.protocol_id, cfg.protocol.split_seed,
                      f"xdisease_{draw}")
    rev = {v: k for k, v in gene_index.items()}
    out = list(support_genes)
    idx_choice = rng.permutation(len(out))[:n_replace]
    banned = {gene_index[g] for g in support_genes if g in gene_index}
    for j in idx_choice:
        g = out[j]
        gi = gene_index.get(g)
        if gi is None:
            continue
        target = np.array([gene_ctd_deg[gi], gene_ppi_deg[gi]], dtype=np.float64)
        pool = np.array([c for c in candidate_pool if c not in banned], dtype=np.int64)
        if len(pool) == 0:
            continue
        feats = np.stack([gene_ctd_deg[pool], gene_ppi_deg[pool]], axis=1).astype(np.float64)
        dist = np.abs(np.log1p(feats) - np.log1p(target)).sum(axis=1)
        best = np.flatnonzero(dist <= dist.min() + 1e-9)
        pick = int(pool[best[rng.integers(len(best))]])
        banned.add(pick)
        out[j] = rev.get(pick, g)
    return out

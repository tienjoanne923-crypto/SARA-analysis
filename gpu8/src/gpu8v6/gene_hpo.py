"""V6 sections 2.1 and 4.3 — gene-side HPO rebuilt with source exclusion.

``genes_to_phenotype.txt`` links a gene to HPO terms *through a source disease*.
Aggregating it to genes after dropping ``disease_id`` would let a held-out
disease's own answer re-enter through "target disease -> gene HPO annotation ->
phenotype match".  The builder therefore:

1. reads the raw rows with their source disease id;
2. maps the source to a project disease (V5 path + strict V6 path) and to its
   strict-equivalence group;
3. **excludes every row whose source is in the equivalence group of a labelled
   disease that is not a reference disease of the current context** (validation,
   test, development and external diseases, plus the training block held out in
   an inner context);
4. only then de-duplicates, expands to allowed ancestor terms and aggregates to
   genes.

Rows whose source disease cannot be mapped are *not* verifiable against the
isolation rule: the strict matrix leaves them out, the lenient matrix (a
separate sensitivity analysis) keeps them.  Every context records how many rows
were excluded and how many genes lost their profile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from gpu8 import paths as p5

from . import paths6
from .phenotypes_v6 import hpo_ancestors, hpo_depth_and_roots, hpo_parents, term_class

EPS = 1e-12


class HPOSpace:
    """Valid-term index and the ancestor-closure operator T (n_t x n_t)."""

    def __init__(self, hpo_terms: pd.DataFrame | None = None):
        hpo_terms = hpo_terms if hpo_terms is not None else \
            pd.read_parquet(p5.TABLES_DIR / "hpo_terms.parquet")
        parents = hpo_parents(hpo_terms)
        depth, _ = hpo_depth_and_roots(parents)
        anc = hpo_ancestors(parents)
        valid = sorted(t for t in parents if term_class(t, depth, anc) == "valid")
        self.terms = valid
        self.t_ix = {t: i for i, t in enumerate(valid)}
        rows, cols = [], []
        for t, i in self.t_ix.items():
            rows.append(i)
            cols.append(i)
            for a in anc.get(t, ()):
                j = self.t_ix.get(a)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        n = len(valid)
        self.T = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                   shape=(n, n))
        self.n_t = n

    def closure(self, M: sparse.csr_matrix) -> sparse.csr_matrix:
        """Binary ancestor closure of an incidence matrix over valid terms."""
        C = (M @ self.T).tocsr()
        C.data[:] = 1.0
        return C


def disease_profiles(space: HPOSpace, n_d: int, d_ix: dict,
                     phenotypes_v6: pd.DataFrame | None = None) -> sparse.csr_matrix:
    ph = phenotypes_v6 if phenotypes_v6 is not None else \
        pd.read_parquet(paths6.TABLES / "phenotypes_v6.parquet")
    r = ph["disease_id"].map(d_ix)
    c = ph["hpo_id"].map(space.t_ix)
    keep = r.notna() & c.notna()
    M = sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32),
                           (r[keep].astype(int), c[keep].astype(int))),
                          shape=(n_d, space.n_t))
    M.sum_duplicates()
    M.data[:] = 1.0
    return space.closure(M)


def ic_from_reference(Q: sparse.csr_matrix, ref_rows: np.ndarray) -> np.ndarray:
    n = max(1, len(ref_rows))
    cnt = np.asarray(Q[ref_rows].sum(axis=0)).ravel()
    return (-np.log((cnt + 1.0) / (n + 1.0))).astype(np.float32)


class GeneHPOSource:
    def __init__(self, world, space: HPOSpace, g2p_path=None):
        self.w = world
        self.space = space
        g2p_path = g2p_path or (p5.PHENOTYPE_DIR / "genes_to_phenotype.txt")
        g2p = pd.read_csv(g2p_path, sep="\t", dtype=str)
        g2p = g2p.rename(columns={"disease_id": "hpoa_id"})
        idmap = pd.read_parquet(paths6.TABLES / "hpoa_id_map.parquet")
        # a strict mapping never overrides a V5 mapping of the same id
        idmap = idmap.sort_values("path").drop_duplicates("hpoa_id", keep="first")
        dmap = dict(zip(idmap["hpoa_id"], idmap["disease_id"]))
        eq = dict(zip(world.diseases["disease_id"], world.diseases["equivalence_group"]))
        g2p["gene_row"] = g2p["ncbi_gene_id"].map(world.idx.g_ix)
        g2p["t"] = g2p["hpo_id"].map(space.t_ix)
        g2p["source_disease"] = g2p["hpoa_id"].map(dmap)
        g2p["source_group"] = g2p["source_disease"].map(eq)
        self.n_raw = int(len(g2p))
        self.n_gene_unmapped = int(g2p["gene_row"].isna().sum())
        self.n_invalid_term = int(g2p["t"].isna().sum())
        rows = g2p[g2p["gene_row"].notna() & g2p["t"].notna()].copy()
        rows["gene_row"] = rows["gene_row"].astype(int)
        rows["t"] = rows["t"].astype(int)
        rows["mapped"] = rows["source_disease"].notna()
        rows = rows.drop_duplicates(subset=["gene_row", "t", "hpoa_id"])
        self.rows = rows[["gene_row", "t", "hpoa_id", "source_disease", "source_group",
                          "mapped"]].reset_index(drop=True)
        labelled = set(world.positives)
        self.labelled_group = {eq.get(d, d) for d in labelled}
        self.eq = eq

    def forbidden_groups(self, reference: set) -> set:
        """Groups of labelled diseases that are not references of this context."""
        ref_groups = {self.eq.get(d, d) for d in reference}
        held = {self.eq.get(d, d) for d in self.w.positives if d not in reference}
        # a group containing any held-out labelled disease is forbidden even if
        # another member of it is a reference (that member is removed from the
        # V6 reference set as a strict equivalent)
        return held

    def matrix(self, reference: set, lenient: bool = False) -> tuple[sparse.csr_matrix, dict]:
        forb = self.forbidden_groups(reference)
        r = self.rows
        allowed = r["mapped"] & ~r["source_group"].isin(forb)
        if lenient:
            allowed = allowed | ~r["mapped"]
        sub = r[allowed]
        M = sparse.csr_matrix((np.ones(len(sub), dtype=np.float32),
                               (sub["gene_row"].to_numpy(), sub["t"].to_numpy())),
                              shape=(self.w.n_g, self.space.n_t))
        M.sum_duplicates()
        M.data[:] = 1.0
        G = self.space.closure(M)
        genes_all = r.loc[r["mapped"] | lenient, "gene_row"].nunique()
        info = {"rows_total": int(len(r)), "rows_mapped": int(r["mapped"].sum()),
                "rows_excluded_forbidden_source": int((r["mapped"]
                                                       & r["source_group"].isin(forb)).sum()),
                "rows_unmapped_source": int((~r["mapped"]).sum()),
                "rows_used": int(len(sub)), "lenient": bool(lenient),
                "genes_with_profile": int((np.diff(G.indptr) > 0).sum()),
                "genes_before_exclusion": int(genes_all)}
        return G, info


def gene_hpo_scores(q_rows: sparse.csr_matrix, G: sparse.csr_matrix,
                    ic: np.ndarray) -> np.ndarray:
    """Cosine between IC-weighted disease profiles (rows of q_rows) and genes."""
    W = sparse.diags(ic.astype(np.float32))
    Gw = (G @ W).tocsr()
    gn = np.sqrt(np.asarray(Gw.multiply(Gw).sum(axis=1)).ravel())
    Qw = (q_rows @ W).tocsr()
    qn = np.sqrt(np.asarray(Qw.multiply(Qw).sum(axis=1)).ravel())
    S = (Qw @ Gw.T).toarray().astype(np.float32)
    S /= np.maximum(qn[:, None] * gn[None, :], EPS)
    return S

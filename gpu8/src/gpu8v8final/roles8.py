"""Role-based exclusion (PDF §4.1, §16.3): nothing here reads a query label.

V6 derived the held-out equivalence groups from ``world.positives`` (the label
table).  V8.1 derives them from the frozen split roles instead:

    held_out(outer j)    = F_j  U  F_{j+1}  U  development  U  external
    held_out(inner j,b)  = held_out(outer j)  U  I_{jb}

so that inputs and rankings can still be generated with the label files removed.
``assert_matches_v6`` checks, on the current data, that the role-based sets equal
the label-based V6 sets (they must; any difference is logged, not silently used).

``RoleSourceEngine`` is the V6 source engine with (a) the role-based reference
set and (b) gene-HPO source exclusion from role-based forbidden groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from gpu8v6.common import ContextSpec, World6
from gpu8v6.gene_hpo import GeneHPOSource, ic_from_reference
from gpu8v6.sources import EPS, ContextSources, SourceEngine

N_FOLDS = 5


def adjacency(w: World6, reference, drop_relations: set | None = None,
              drop_genes: np.ndarray | None = None) -> sparse.csr_matrix:
    """Disease x gene adjacency of the reference diseases' visible primary relations."""
    rel = w.primary_relations
    rel = rel[rel["disease_id"].isin(set(reference))]
    if drop_relations:
        key = list(zip(rel["disease_id"], rel["gene_id"].astype(str)))
        rel = rel[[k not in drop_relations for k in key]]
    rows = rel["disease_id"].map(w.idx.d_ix).to_numpy()
    cols = rel["gene_id"].map(w.idx.g_ix).to_numpy()
    keep = ~(pd.isna(rows) | pd.isna(cols))
    rows, cols = rows[keep].astype(np.int64), cols[keep].astype(np.int64)
    if drop_genes is not None and len(drop_genes):
        m = ~np.isin(cols, np.asarray(drop_genes, dtype=np.int64))
        rows, cols = rows[m], cols[m]
    A = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(w.n_d, w.n_g)).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0
    return A


class Roles:
    def __init__(self, w: World6, extra_held: dict[str, set] | None = None):
        self.w = w
        sp = w.spec
        self.dev = set(sp.dev_diseases)
        self.ext = set(sp.external_diseases)
        self.folds = {i: set(sp.folds[f"F{i}"]) for i in range(N_FOLDS)}
        self.eq = dict(zip(w.diseases["disease_id"], w.diseases["equivalence_group"]))
        self.extra_held = extra_held or {}

    def held_out(self, ctx: ContextSpec) -> set:
        if ctx.context_id in self.extra_held:
            return set(self.extra_held[ctx.context_id])
        j = ctx.fold
        held = self.folds[j] | self.folds[(j + 1) % N_FOLDS] | self.dev | self.ext
        if ctx.kind == "inner":
            held |= set(ctx.held_out)
        return held

    def forbidden_groups(self, ctx: ContextSpec) -> set:
        return {self.eq.get(d, d) for d in self.held_out(ctx)}

    def reference(self, ctx: ContextSpec) -> list[str]:
        forb = self.forbidden_groups(ctx)
        return [d for d in ctx.reference if self.eq.get(d, d) not in forb]

    def targets(self, ctx: ContextSpec) -> dict[str, list[str]]:
        """Diseases a context scores (role lists, sorted; no label filter)."""
        j = ctx.fold
        val = sorted(self.folds[(j + 1) % N_FOLDS])
        if ctx.kind == "inner":
            return {"oof": sorted(ctx.held_out), "val": val}
        return {"test": sorted(self.folds[j]), "val": val, "dev": sorted(self.dev)}


class RoleGeneHPOSource(GeneHPOSource):
    """Gene-HPO profiles with an explicit, role-based forbidden-group set."""

    def matrix_forbidden(self, forb: set, lenient: bool = False,
                         drop_genes: np.ndarray | None = None) -> tuple[sparse.csr_matrix, dict]:
        r = self.rows
        allowed = r["mapped"] & ~r["source_group"].isin(forb)
        if lenient:
            allowed = allowed | ~r["mapped"]
        if drop_genes is not None and len(drop_genes):
            allowed = allowed & ~r["gene_row"].isin(set(int(x) for x in drop_genes))
        sub = r[allowed]
        M = sparse.csr_matrix((np.ones(len(sub), dtype=np.float32),
                               (sub["gene_row"].to_numpy(), sub["t"].to_numpy())),
                              shape=(self.w.n_g, self.space.n_t))
        M.sum_duplicates()
        M.data[:] = 1.0
        G = self.space.closure(M)
        info = {"rows_total": int(len(r)), "rows_used": int(len(sub)),
                "rows_excluded_forbidden_source": int((r["mapped"] & r["source_group"].isin(forb)).sum()),
                "genes_with_profile": int((np.diff(G.indptr) > 0).sum()), "lenient": bool(lenient),
                "dropped_genes": int(0 if drop_genes is None else len(drop_genes))}
        return G, info


class RoleSourceEngine(SourceEngine):
    """V6 SourceEngine with role-based references and gene-HPO exclusion.

    ``drop_relations`` (a set of (disease_id, gene_id)) and ``drop_genes`` (gene
    rows) support the evidence-block and true-supervision-cold arms: those
    relations / genes are removed from the reference graph and the gene-HPO table.
    """

    def __init__(self, world: World6, roles: Roles, device: str = "cuda", text_level: int = 1,
                 k_ref: int = 20, drop_relations: set | None = None,
                 drop_genes: np.ndarray | None = None):
        super().__init__(world, device=device, text_level=text_level, k_ref=k_ref, with_v5=False,
                         lenient=False)
        self.roles = roles
        self.ghpo.__class__ = RoleGeneHPOSource
        self.drop_relations = drop_relations or set()
        self.drop_genes = np.asarray(drop_genes if drop_genes is not None else [], dtype=np.int64)

    def v6_reference(self, ctx: ContextSpec) -> tuple[list[str], int]:
        ref = self.roles.reference(ctx)
        return ref, len(ctx.reference) - len(ref)

    def adjacency(self, cs: ContextSpec) -> sparse.csr_matrix:
        return adjacency(self.w, cs.reference, self.drop_relations, self.drop_genes)

    def context(self, ctx: ContextSpec) -> ContextSources:
        ref, removed = self.v6_reference(ctx)
        cs = ContextSpec(ctx.context_id, ctx.fold, ctx.kind, ctx.block_index, ref, ctx.held_out)
        A = self.adjacency(cs)
        d_ix = self.w.idx.d_ix
        ref_rows = np.array(sorted(d_ix[d] for d in ref if d in d_ix), dtype=np.int64)
        ic = ic_from_reference(self.Q, ref_rows)
        W = sparse.diags(ic)
        Qw = (self.Q @ W).tocsr()
        norms = np.sqrt(np.asarray(Qw.multiply(Qw).sum(axis=1)).ravel())
        Qn = (sparse.diags(1.0 / np.maximum(norms, EPS)) @ Qw).tocsr()
        Gh, gi = self.ghpo.matrix_forbidden(self.roles.forbidden_groups(ctx), drop_genes=self.drop_genes)
        gdeg = np.asarray(A.sum(axis=0)).ravel()
        Z0 = (A.T @ Qn).tocsr()
        Z0 = sparse.diags(1.0 / np.maximum(gdeg, 1.0)) @ Z0
        zn = np.sqrt(np.asarray(Z0.multiply(Z0).sum(axis=1)).ravel())
        Z0 = (sparse.diags(1.0 / np.maximum(zn, EPS)) @ Z0).tocsr()
        return ContextSources(ctx=cs, reference=ref, ref_rows=ref_rows, removed_equivalents=removed,
                              A=A, ic=ic, Qw_ref=Qn[ref_rows], Gh=Gh, Gh_info=gi, Gh_len=None,
                              Gh_len_info={}, Z0=Z0, ref_set_rows=set(ref_rows.tolist()),
                              text_ref=self.X[ref_rows],
                              info={"n_reference_v5": len(ctx.reference), "n_reference_v8": len(ref),
                                    "removed_strict_equivalents": removed,
                                    "n_visible_relations": int(A.nnz), "gene_hpo": gi,
                                    "exclusion": "role-based (no label read)"})


def assert_matches_v6(w: World6, roles: Roles) -> dict:
    """Role-based held-out / reference sets equal the V6 label-based sets on current data."""
    eq = roles.eq
    out = {"contexts": 0, "mismatches": []}
    for cid, ctx in w.contexts.items():
        if not (cid.startswith("f") and ("_outer" in cid or "_inner" in cid)):
            continue
        ref = set(ctx.reference)
        v6_held = {eq.get(d, d) for d in w.positives if d not in ref}
        role_held = roles.forbidden_groups(ctx)
        v6_ref = [d for d in ctx.reference if eq.get(d, d) not in v6_held]
        role_ref = roles.reference(ctx)
        out["contexts"] += 1
        if v6_held != role_held or v6_ref != role_ref:
            out["mismatches"].append({"context": cid, "v6_only": len(v6_held - role_held),
                                      "role_only": len(role_held - v6_held),
                                      "ref_equal": v6_ref == role_ref})
    out["all_equal"] = not out["mismatches"]
    return out

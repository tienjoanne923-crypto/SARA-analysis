"""Sparse precomputation: the four 128-dim cached views and the scalar channels.

Plan section 8.  Everything expensive and *reusable* is computed once per graph
context on the CPU and cached; the GPU only ever sees small dense batches.

Hard rules enforced here:

* the HPO information content and the 128-dim basis are fitted on the
  *reference* diseases of the context only;
* the disease-gene adjacency A contains reference relations only, so a held-out
  query can never reach the network through degree, propagation or retrieval;
* no dense all-gene x all-disease matrix is ever materialised.

Cache key = resource manifest hash + protocol + context id + config hash + code
version.  Changing a training seed does not invalidate a cache; changing the
visible relations, the PPI or the historical cut-off does.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from . import paths
from .config import Gpu8Config, load_config
from .hashing import atomic_write_json, code_version_hash, read_json, sha256_obj

EPS = 1e-8


# --------------------------------------------------------------------------
# entity index
# --------------------------------------------------------------------------
@dataclass
class EntityIndex:
    diseases: list[str]
    genes: list[str]
    hpo: list[str]
    d_ix: dict[str, int]
    g_ix: dict[str, int]
    h_ix: dict[str, int]

    @classmethod
    def build(cls, diseases: list[str], genes: list[str], hpo: list[str]) -> "EntityIndex":
        return cls(diseases, genes, hpo,
                   {d: i for i, d in enumerate(diseases)},
                   {g: i for i, g in enumerate(genes)},
                   {h: i for i, h in enumerate(hpo)})

    @property
    def n_d(self) -> int:
        return len(self.diseases)

    @property
    def n_g(self) -> int:
        return len(self.genes)

    @property
    def n_h(self) -> int:
        return len(self.hpo)


def load_entity_index(tables_dir: Path | None = None) -> EntityIndex:
    tables_dir = Path(tables_dir) if tables_dir else paths.TABLES_DIR
    diseases = pd.read_parquet(tables_dir / "diseases.parquet")
    genes = pd.read_parquet(tables_dir / "genes.parquet")
    phen = pd.read_parquet(tables_dir / "phenotypes.parquet")
    hpo_ids = sorted(set(phen["hpo_id"])) if not phen.empty else []
    return EntityIndex.build(sorted(diseases["disease_id"]),
                             sorted(genes["gene_id"], key=lambda x: int(x) if str(x).isdigit() else 0),
                             hpo_ids)


# --------------------------------------------------------------------------
# sparse builders
# --------------------------------------------------------------------------
def build_A(relations: pd.DataFrame, idx: EntityIndex, reference: set[str],
            relation_types: set[str]) -> sparse.csr_matrix:
    """Disease x gene adjacency restricted to reference diseases."""
    rel = relations[relations["relation_type"].isin(relation_types)]
    rel = rel[rel["disease_id"].isin(reference)]
    rows = rel["disease_id"].map(idx.d_ix).to_numpy()
    cols = rel["gene_id"].map(idx.g_ix).to_numpy()
    keep = ~(pd.isna(rows) | pd.isna(cols))
    rows, cols = rows[keep].astype(np.int64), cols[keep].astype(np.int64)
    data = np.ones(len(rows), dtype=np.float32)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(idx.n_d, idx.n_g), dtype=np.float32)
    A.sum_duplicates()
    A.data[:] = 1.0
    return A.tocsr()


def build_H(phenotypes: pd.DataFrame, idx: EntityIndex) -> sparse.csr_matrix:
    """Disease x HPO matrix; NOT-qualified annotations are dropped, never positive."""
    if phenotypes.empty:
        return sparse.csr_matrix((idx.n_d, max(1, idx.n_h)), dtype=np.float32)
    ph = phenotypes[phenotypes["negated"] == 0]
    rows = ph["disease_id"].map(idx.d_ix).to_numpy()
    cols = ph["hpo_id"].map(idx.h_ix).to_numpy()
    keep = ~(pd.isna(rows) | pd.isna(cols))
    rows, cols = rows[keep].astype(np.int64), cols[keep].astype(np.int64)
    H = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(idx.n_d, max(1, idx.n_h)), dtype=np.float32)
    H.sum_duplicates()
    H.data[:] = 1.0
    return H.tocsr()


def build_P(ppi: pd.DataFrame, idx: EntityIndex, network_type: str = "physical",
            return_adjacency: bool = False):
    """Column-stochastic PPI propagation matrix plus the unweighted degree.

    With ``return_adjacency`` the raw symmetric adjacency is also returned; the
    quantum-walk and DIAMOnD comparators need the unnormalised matrix.
    """
    sub = ppi[ppi["network_type"] == network_type] if "network_type" in ppi.columns else ppi
    if sub.empty:
        deg = np.zeros(idx.n_g, dtype=np.float32)
        eye = sparse.identity(idx.n_g, format="csr", dtype=np.float32)
        return (eye, deg, eye) if return_adjacency else (eye, deg)
    a = sub["gene_a"].map(idx.g_ix).to_numpy()
    b = sub["gene_b"].map(idx.g_ix).to_numpy()
    keep = ~(pd.isna(a) | pd.isna(b))
    a, b = a[keep].astype(np.int64), b[keep].astype(np.int64)
    w = (sub["score"].to_numpy()[keep] / 1000.0).astype(np.float32)
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    data = np.concatenate([w, w])
    Adj = sparse.coo_matrix((data, (rows, cols)), shape=(idx.n_g, idx.n_g), dtype=np.float32).tocsr()
    Adj.sum_duplicates()
    raw_adj = Adj.copy()
    deg = np.asarray((Adj > 0).sum(axis=1)).ravel().astype(np.float32)
    # isolated genes get a self-loop so propagation is well defined
    isolated = np.where(np.asarray(Adj.sum(axis=0)).ravel() <= 0)[0]
    if len(isolated):
        Adj = Adj + sparse.coo_matrix(
            (np.ones(len(isolated), dtype=np.float32), (isolated, isolated)),
            shape=Adj.shape, dtype=np.float32).tocsr()
    colsum = np.asarray(Adj.sum(axis=0)).ravel()
    inv = np.where(colsum > 0, 1.0 / np.maximum(colsum, EPS), 0.0).astype(np.float32)
    P = Adj @ sparse.diags(inv)
    if return_adjacency:
        return P.tocsr(), deg, raw_adj
    return P.tocsr(), deg


def build_hierarchy_neighbours(hierarchy: pd.DataFrame, idx: EntityIndex,
                               max_distance: int = 2) -> dict[int, dict[int, float]]:
    """Ontology neighbourhood with a distance decay, as index -> {index: weight}."""
    parents: dict[int, list[int]] = {}
    children: dict[int, list[int]] = {}
    if hierarchy is not None and not hierarchy.empty:
        for r in hierarchy.itertuples(index=False):
            ci, pi = idx.d_ix.get(r.child_disease), idx.d_ix.get(r.parent_disease)
            if ci is None or pi is None:
                continue
            parents.setdefault(ci, []).append(pi)
            children.setdefault(pi, []).append(ci)
    out: dict[int, dict[int, float]] = {}
    for d in range(idx.n_d):
        acc: dict[int, float] = {}
        frontier = [(d, 0)]
        seen = {d}
        while frontier:
            node, dist = frontier.pop()
            if dist >= max_distance:
                continue
            for nb in parents.get(node, []) + children.get(node, []):
                if nb in seen:
                    continue
                seen.add(nb)
                acc[nb] = max(acc.get(nb, 0.0), 1.0 / (dist + 2.0))
                frontier.append((nb, dist + 1))
        if acc:
            out[d] = acc
    return out


# --------------------------------------------------------------------------
# HPO information content and the shared 128-dim basis
# --------------------------------------------------------------------------
def hpo_information_content(H: sparse.csr_matrix, reference_rows: np.ndarray) -> np.ndarray:
    """IC estimated on reference diseases only (plan 8.1)."""
    n = max(1, len(reference_rows))
    counts = np.asarray(H[reference_rows].sum(axis=0)).ravel()
    freq = np.clip(counts / n, EPS, 1.0)
    return (-np.log(freq)).astype(np.float32)


def fit_hpo_basis(H: sparse.csr_matrix, reference_rows: np.ndarray, ic: np.ndarray,
                  dim: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sparse truncated SVD of the IC-weighted reference block.

    Returns (components [k x n_hpo], singular values).
    """
    W = H @ sparse.diags(ic.astype(np.float32))
    ref = W[reference_rows]
    k = int(min(dim, max(1, min(ref.shape) - 1)))
    if k < 1 or ref.nnz == 0:
        return np.zeros((dim, H.shape[1]), dtype=np.float32), np.zeros(dim, dtype=np.float32)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(ref.shape)).astype(np.float64)
    u, s, vt = svds(ref.astype(np.float64), k=k, v0=v0)
    order = np.argsort(-s)
    vt, s = vt[order], s[order]
    if k < dim:  # pad so every context yields the same width
        vt = np.vstack([vt, np.zeros((dim - k, vt.shape[1]))])
        s = np.concatenate([s, np.zeros(dim - k)])
    return vt.astype(np.float32), s.astype(np.float32)


def project_disease_phenotype(H: sparse.csr_matrix, ic: np.ndarray,
                              basis: np.ndarray) -> np.ndarray:
    W = H @ sparse.diags(ic.astype(np.float32))
    return np.asarray(W @ basis.T, dtype=np.float32)


# --------------------------------------------------------------------------
# context build
# --------------------------------------------------------------------------
def context_hash(cfg: Gpu8Config, context_id: str, resource_hash: str,
                 code_hash: str, extra: dict | None = None) -> str:
    return sha256_obj({"context": context_id, "resource": resource_hash,
                       "config": cfg.config_hash, "code": code_hash,
                       "protocol": cfg.protocol_id, "extra": extra or {}})


def l2_normalise(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.maximum(norm, EPS)).astype(np.float32)


def build_context(cfg: Gpu8Config, idx: EntityIndex, relations: pd.DataFrame,
                  phenotypes: pd.DataFrame, hierarchy: pd.DataFrame, ppi: pd.DataFrame,
                  reference: list[str], context_id: str, out_dir: Path,
                  hier_nb: dict | None = None,
                  H: sparse.csr_matrix | None = None,
                  P: sparse.csr_matrix | None = None,
                  ppi_deg: np.ndarray | None = None) -> dict:
    """Materialise the four 128-dim views plus the channel inputs for one context."""
    fc = cfg.features
    ref_set = set(reference)
    ref_rows = np.array(sorted(idx.d_ix[d] for d in reference if d in idx.d_ix), dtype=np.int64)

    A = build_A(relations, idx, ref_set, set(cfg.data.primary_evidence))
    if H is None:
        H = build_H(phenotypes, idx)
    if P is None or ppi_deg is None:
        P, ppi_deg = build_P(ppi, idx)
    if hier_nb is None:
        hier_nb = build_hierarchy_neighbours(hierarchy, idx, cfg.candidates.hierarchy_max_distance)

    ic = hpo_information_content(H, ref_rows)
    basis, sing = fit_hpo_basis(H, ref_rows, ic, fc.hpo_basis_dim,
                                seed=cfg.protocol.split_seed)

    # ---- V0 disease: target HPO profile in the reference basis ----------
    D_pheno = project_disease_phenotype(H, ic, basis)              # n_d x 128
    D_V0 = l2_normalise(D_pheno)

    # ---- gene phenotype seed Z0 = normalised A^T x disease phenotype ----
    gene_deg = np.asarray(A.sum(axis=0)).ravel().astype(np.float32)   # visible CTD degree
    Z0 = np.asarray(A.T @ D_pheno, dtype=np.float32)
    Z0 = Z0 / np.maximum(gene_deg[:, None], 1.0)
    G_V0 = l2_normalise(Z0)

    # ---- V1 / V2 gene: one and two PPI propagation steps ----------------
    PZ = np.asarray(P @ Z0, dtype=np.float32)
    P2Z = np.asarray(P @ PZ, dtype=np.float32)
    G_V1, G_V2 = l2_normalise(PZ), l2_normalise(P2Z)

    # ---- V1 disease: ontology-neighbour reference aggregation ----------
    ref_set_rows = set(ref_rows.tolist())
    D_V1 = np.zeros_like(D_pheno)
    for d, nbs in hier_nb.items():
        acc = np.zeros(D_pheno.shape[1], dtype=np.float32)
        wsum = 0.0
        for nb, w in nbs.items():
            if nb in ref_set_rows:
                acc += w * D_pheno[nb]
                wsum += w
        if wsum > 0:
            D_V1[d] = acc / wsum
    D_V1 = l2_normalise(D_V1)

    # ---- V2 disease: phenotype-neighbour reference aggregation ---------
    D_V2 = phenotype_neighbour_aggregate(D_V0, ref_rows, cfg.candidates.phenotype_neighbours)
    D_V2 = l2_normalise(D_V2)

    # ---- V3: hierarchy / coverage descriptors --------------------------
    D_V3 = hierarchy_descriptor(idx, hierarchy, basis.shape[0])
    G_V3 = gene_hierarchy_descriptor(A, D_V3)
    D_V3, G_V3 = l2_normalise(D_V3), l2_normalise(G_V3)

    disease_views = np.concatenate([D_V0, D_V1, D_V2, D_V3], axis=1).astype(np.float32)
    gene_views = np.concatenate([G_V0, G_V1, G_V2, G_V3], axis=1).astype(np.float32)

    # ---- missing-view masks ---------------------------------------------
    d_mask = np.stack([
        (np.abs(D_V0).sum(1) > 0), (np.abs(D_V1).sum(1) > 0),
        (np.abs(D_V2).sum(1) > 0), (np.abs(D_V3).sum(1) > 0)], axis=1).astype(np.float32)
    g_mask = np.stack([
        (np.abs(G_V0).sum(1) > 0), (np.abs(G_V1).sum(1) > 0),
        (np.abs(G_V2).sum(1) > 0), (np.abs(G_V3).sum(1) > 0)], axis=1).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "views.npz",
        disease_views=disease_views, gene_views=gene_views,
        disease_mask=d_mask, gene_mask=g_mask,
        disease_pheno=D_pheno.astype(np.float32),
        gene_seed=Z0.astype(np.float32),
        gene_pz=PZ.astype(np.float32), gene_p2z=P2Z.astype(np.float32),
        ic=ic, singular_values=sing,
        gene_ctd_degree=gene_deg, gene_ppi_degree=ppi_deg.astype(np.float32),
        reference_rows=ref_rows,
    )
    sparse.save_npz(out_dir / "A.npz", A)
    np.save(out_dir / "hpo_basis.npy", basis)

    meta = {
        "context_id": context_id,
        "n_reference": len(reference),
        "n_visible_relations": int(A.nnz),
        "view_dim": fc.view_dim, "n_views": fc.n_views,
        "disease_view_shape": list(disease_views.shape),
        "gene_view_shape": list(gene_views.shape),
        "mean_disease_views_present": float(d_mask.mean()),
        "mean_gene_views_present": float(g_mask.mean()),
        "singular_value_top": float(sing[0]) if len(sing) else 0.0,
        "hpo_terms_with_ic": int((ic > 0).sum()),
    }
    atomic_write_json(out_dir / "meta.json", meta)
    return meta


def phenotype_neighbour_aggregate(D_V0: np.ndarray, ref_rows: np.ndarray,
                                  k: int, block: int = 512) -> np.ndarray:
    """For each disease, mean of its k most phenotype-similar reference diseases.

    Cosine similarity on unit vectors, computed in blocks so the full
    disease x disease matrix is never held in memory.
    """
    out = np.zeros_like(D_V0)
    if len(ref_rows) == 0:
        return out
    R = D_V0[ref_rows]                                  # n_ref x 128 (unit rows)
    for start in range(0, D_V0.shape[0], block):
        stop = min(start + block, D_V0.shape[0])
        sims = D_V0[start:stop] @ R.T                   # block x n_ref
        # never let a disease retrieve itself
        for i, row in enumerate(range(start, stop)):
            pos = np.searchsorted(ref_rows, row)
            if pos < len(ref_rows) and ref_rows[pos] == row:
                sims[i, pos] = -np.inf
        kk = int(min(k, sims.shape[1]))
        if kk <= 0:
            continue
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for i in range(sims.shape[0]):
            cols = top[i]
            w = sims[i, cols]
            w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
            if w.sum() <= 0:
                continue
            out[start + i] = (w[:, None] * R[cols]).sum(0) / w.sum()
    return out


def hierarchy_descriptor(idx: EntityIndex, hierarchy: pd.DataFrame, dim: int) -> np.ndarray:
    """Sparse ancestor indicator projected to `dim` via a fixed random basis.

    The projection matrix is derived from the protocol seed, so it is identical
    across contexts and carries no information from any specific split.
    """
    out = np.zeros((idx.n_d, dim), dtype=np.float32)
    if hierarchy is None or hierarchy.empty:
        return out
    parents: dict[int, list[int]] = {}
    for r in hierarchy.itertuples(index=False):
        ci, pi = idx.d_ix.get(r.child_disease), idx.d_ix.get(r.parent_disease)
        if ci is not None and pi is not None:
            parents.setdefault(ci, []).append(pi)
    rng = np.random.default_rng(20260922)
    proj = rng.standard_normal((idx.n_d, dim)).astype(np.float32) / math.sqrt(dim)
    for d in range(idx.n_d):
        acc = np.zeros(dim, dtype=np.float32)
        frontier, seen, depth = [d], {d}, {d: 0}
        while frontier:
            node = frontier.pop()
            for p in parents.get(node, []):
                if p in seen:
                    continue
                seen.add(p)
                depth[p] = depth[node] + 1
                if depth[p] <= 3:
                    acc += proj[p] / float(depth[p] + 1)
                    frontier.append(p)
        out[d] = acc
    return out


def gene_hierarchy_descriptor(A: sparse.csr_matrix, D_V3: np.ndarray) -> np.ndarray:
    deg = np.asarray(A.sum(axis=0)).ravel().astype(np.float32)
    out = np.asarray(A.T @ D_V3, dtype=np.float32)
    return out / np.maximum(deg[:, None], 1.0)


# --------------------------------------------------------------------------
# build all contexts
# --------------------------------------------------------------------------
def build_all(cfg: Gpu8Config | None = None, tables_dir: Path | None = None,
              only: list[str] | None = None, force: bool = False) -> dict:
    cfg = cfg or load_config()
    tables_dir = Path(tables_dir) if tables_dir else paths.TABLES_DIR
    idx = load_entity_index(tables_dir)
    relations = pd.read_parquet(tables_dir / "relations.parquet")
    phenotypes = pd.read_parquet(tables_dir / "phenotypes.parquet")
    hierarchy = pd.read_parquet(tables_dir / "hierarchy.parquet")
    ppi = pd.read_parquet(tables_dir / "ppi.parquet")

    ctx_blob = read_json(paths.MANIFEST_DIR / "contexts.json", {})
    specs = ctx_blob.get("contexts", [])
    if only:
        specs = [s for s in specs if s["context_id"] in set(only)]

    resource = read_json(paths.MANIFEST_DIR / "resource_manifest.json", {})
    resource_hash = resource.get("manifest_hash", "")
    code_hash = code_version_hash(Path(__file__).resolve().parent)

    # shared structures (identical across contexts)
    H = build_H(phenotypes, idx)
    P, ppi_deg = build_P(ppi, idx)
    hier_nb = build_hierarchy_neighbours(hierarchy, idx, cfg.candidates.hierarchy_max_distance)

    built, skipped = [], []
    index: dict[str, dict] = {}
    for spec in specs:
        cid = spec["context_id"]
        chash = context_hash(cfg, cid, resource_hash, code_hash)
        out_dir = paths.CACHE_DIR / chash
        index[cid] = {"context_hash": chash, "path": str(out_dir), **{
            k: spec[k] for k in ("fold", "kind", "block_index", "n_reference")}}
        if (out_dir / "meta.json").exists() and not force:
            skipped.append(cid)
            continue
        print(f"  context {cid} ({spec['n_reference']} reference diseases) ...", flush=True)
        meta = build_context(cfg, idx, relations, phenotypes, hierarchy, ppi,
                             spec["reference_diseases"], cid, out_dir,
                             hier_nb=hier_nb, H=H, P=P, ppi_deg=ppi_deg)
        index[cid].update({k: meta[k] for k in ("n_visible_relations",
                                                "mean_disease_views_present",
                                                "mean_gene_views_present")})
        built.append(cid)

    payload = {"resource_hash": resource_hash, "code_hash": code_hash,
               "config_hash": cfg.config_hash, "n_contexts": len(index),
               "contexts": index, "built": built, "skipped": skipped,
               "entity_counts": {"diseases": idx.n_d, "genes": idx.n_g, "hpo": idx.n_h}}
    atomic_write_json(paths.MANIFEST_DIR / "context_index.json", payload)
    return payload


def load_context(context_id: str) -> tuple[dict, dict]:
    """Return (meta, arrays) for a built context."""
    index = read_json(paths.MANIFEST_DIR / "context_index.json", {})
    entry = index.get("contexts", {}).get(context_id)
    if entry is None:
        raise KeyError(f"unknown context {context_id}")
    d = Path(entry["path"])
    arrays = dict(np.load(d / "views.npz"))
    arrays["A"] = sparse.load_npz(d / "A.npz")
    arrays["hpo_basis"] = np.load(d / "hpo_basis.npy")
    meta = read_json(d / "meta.json", {})
    meta["path"] = str(d)
    meta["context_hash"] = entry["context_hash"]
    return meta, arrays


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build sparse feature contexts.")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tables_dir", type=Path, default=None)
    args = ap.parse_args(argv)
    payload = build_all(tables_dir=args.tables_dir, only=args.only, force=args.force)
    print(json.dumps({k: payload[k] for k in
                      ("n_contexts", "built", "skipped", "entity_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V6 section 4.2 — gene features that do not come from the CTD target edges.

Three low-cost blocks, each with an explicit missing flag:

* GO: gene x GO-term incidence (GOA human GAF, NOT qualifiers removed, true-path
  propagation over is_a / part_of), TF-IDF over genes, truncated SVD to 64 dims;
* Reactome: gene x lowest-level pathway incidence (NCBI2Reactome, Homo sapiens),
  TF-IDF, truncated SVD to 32 dims;
* expression: GTEx v8 median TPM per tissue, log1p, centred, SVD to 16 dims plus
  the mean log expression.

The SVDs are fitted on unlabelled public gene attributes of the whole background
(no disease label is involved); the fitting scope and the resource versions are
recorded with hashes.  A gene without a block gets zeros *and* the missing flag,
never a random vector.  The same module also returns the gene x function
incidence B used by the functional propagation bridge P_F (section 9.2) and the
GO-reference PMIDs used by the shared-literature audit (section 2.2).
"""
from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from gpu8 import paths as p5
from gpu8.hashing import sha256_file

from . import paths6
from .common import dump_json, now_utc

VERSIONS = {
    "current": {"gaf": "go/goa_human_current.gaf.gz", "obo": "go/go-basic_current.obo",
                "reactome": "reactome/NCBI2Reactome_current.txt",
                "hgnc": None},
    "hist2023": {"gaf": "go/goa_human_2023-10-09.gaf.gz", "obo": "go/go-basic_2023-10-09.obo",
                 "reactome": "reactome/NCBI2Reactome_v86.txt",
                 "hgnc": "hgnc/hgnc_complete_set_2023-10-01.txt"},
}
GTEX = "gtex/GTEx_v8_gene_median_tpm.gct.gz"


def parse_go_obo(path: Path) -> tuple[dict, dict]:
    """(term -> parents over is_a/part_of, term -> namespace); obsolete dropped."""
    parents, ns = {}, {}
    cur, obsolete = None, False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                if cur and not obsolete:
                    parents[cur["id"]] = cur["p"]
                    ns[cur["id"]] = cur["ns"]
                cur, obsolete = {"id": "", "p": [], "ns": ""}, False
                continue
            if line.startswith("[") and line.endswith("]"):
                if cur and not obsolete:
                    parents[cur["id"]] = cur["p"]
                    ns[cur["id"]] = cur["ns"]
                cur = None
                continue
            if cur is None:
                continue
            if line.startswith("id: "):
                cur["id"] = line[4:].strip()
            elif line.startswith("namespace: "):
                cur["ns"] = line[11:].strip()
            elif line.startswith("is_a: "):
                cur["p"].append(line[6:].split("!")[0].strip())
            elif line.startswith("relationship: part_of "):
                cur["p"].append(line[len("relationship: part_of "):].split("!")[0].strip())
            elif line.startswith("is_obsolete: true"):
                obsolete = True
    if cur and not obsolete and cur["id"]:
        parents[cur["id"]] = cur["p"]
        ns[cur["id"]] = cur["ns"]
    return parents, ns


def go_closure(parents: dict) -> dict:
    anc: dict[str, set] = {}
    import sys
    sys.setrecursionlimit(100000)

    def get(t):
        if t in anc:
            return anc[t]
        acc = {t}
        for p in parents.get(t, []):
            if p in parents:
                acc |= get(p)
        anc[t] = acc
        return acc

    for t in list(parents):
        get(t)
    return anc


def symbol_to_gene(world, hgnc_path: Path | None) -> dict[str, str]:
    """Approved/previous/alias symbol and UniProt -> project gene id (Entrez)."""
    genes = world.genes
    out: dict[str, str] = {}
    if hgnc_path is None:
        hgnc_path = p5.external("hgnc", "hgnc_complete_set.txt")
    hg = pd.read_csv(hgnc_path, sep="\t", dtype=str, low_memory=False)
    ent_ok = set(genes["gene_id"])
    for r in hg.itertuples(index=False):
        ent = str(getattr(r, "entrez_id", "") or "")
        if ent not in ent_ok:
            continue
        out.setdefault("SYM:" + str(r.symbol), ent)
        for u in str(getattr(r, "uniprot_ids", "") or "").split("|"):
            if u and u != "nan":
                out.setdefault("UNI:" + u, ent)
    for r in hg.itertuples(index=False):
        ent = str(getattr(r, "entrez_id", "") or "")
        if ent not in ent_ok:
            continue
        for col in ("prev_symbol", "alias_symbol"):
            for s in str(getattr(r, col, "") or "").split("|"):
                if s and s != "nan":
                    out.setdefault("SYM:" + s, ent)
    return out


def read_gaf(path: Path, sym2gene: dict) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 15:
                continue
            qual = f[3]
            if "NOT" in qual.split("|"):
                continue
            if f[6] == "ND":
                continue
            ent = sym2gene.get("UNI:" + f[1]) or sym2gene.get("SYM:" + f[2])
            if not ent:
                continue
            refs = [x.split(":", 1)[1] for x in f[5].split("|") if x.startswith("PMID:")]
            rows.append((ent, f[4], f[6], f[8], "|".join(refs)))
    return pd.DataFrame(rows, columns=["gene_id", "go_id", "evidence", "aspect", "pmids"])


def tfidf_svd(M: sparse.csr_matrix, dim: int, seed: int = 20260923) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer
    X = TfidfTransformer(sublinear_tf=True).fit_transform(M)
    k = int(min(dim, max(1, min(X.shape) - 1)))
    Z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)
    if k < dim:
        Z = np.hstack([Z, np.zeros((Z.shape[0], dim - k))])
    return Z.astype(np.float32)


def build(world, version: str = "current", write: bool = True) -> dict:
    v = VERSIONS[version]
    n_g = world.n_g
    g_ix = world.idx.g_ix
    hgnc = paths6.ext(v["hgnc"]) if v["hgnc"] else None
    sym2gene = symbol_to_gene(world, hgnc)

    # ---- GO ---------------------------------------------------------------
    parents, ns = parse_go_obo(paths6.ext(v["obo"]))
    anc = go_closure(parents)
    gaf = read_gaf(paths6.ext(v["gaf"]), sym2gene)
    gaf = gaf[gaf["go_id"].isin(parents)]
    gaf["gene_row"] = gaf["gene_id"].map(g_ix)
    gaf = gaf[gaf["gene_row"].notna()]
    pairs = set()
    for gr, t in zip(gaf["gene_row"].astype(int), gaf["go_id"]):
        for a in anc.get(t, {t}):
            pairs.add((gr, a))
    roots = {"GO:0008150", "GO:0003674", "GO:0005575"}
    pairs = {(g, t) for g, t in pairs if t not in roots}
    terms = sorted({t for _, t in pairs})
    t_ix = {t: i for i, t in enumerate(terms)}
    gi = np.fromiter((g for g, _ in pairs), dtype=np.int64, count=len(pairs))
    ti = np.fromiter((t_ix[t] for _, t in pairs), dtype=np.int64, count=len(pairs))
    GO = sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), (gi, ti)),
                           shape=(n_g, len(terms)))
    go_emb = tfidf_svd(GO, 64)
    go_has = (np.diff(GO.indptr) > 0).astype(np.float32)
    go_emb[go_has == 0] = 0.0

    # ---- Reactome -----------------------------------------------------------
    rx = pd.read_csv(paths6.ext(v["reactome"]), sep="\t", header=None, dtype=str,
                     names=["gene_id", "pathway", "url", "name", "evidence", "species"])
    rx = rx[rx["species"] == "Homo sapiens"]
    rx["gene_row"] = rx["gene_id"].map(g_ix)
    rx = rx[rx["gene_row"].notna()].drop_duplicates(["gene_row", "pathway"])
    paths_ = sorted(rx["pathway"].unique())
    p_ix = {p: i for i, p in enumerate(paths_)}
    RX = sparse.csr_matrix((np.ones(len(rx), dtype=np.float32),
                            (rx["gene_row"].astype(int), rx["pathway"].map(p_ix))),
                           shape=(n_g, len(paths_)))
    rx_emb = tfidf_svd(RX, 32)
    rx_has = (np.diff(RX.indptr) > 0).astype(np.float32)
    rx_emb[rx_has == 0] = 0.0

    # ---- GTEx expression -----------------------------------------------------
    gt = pd.read_csv(paths6.ext(GTEX), sep="\t", skiprows=2, compression="gzip")
    gt["ens"] = gt["Name"].str.split(".").str[0]
    ens2gene = {}
    for r in world.genes.itertuples(index=False):
        if isinstance(r.ensembl_gene_id, str) and r.ensembl_gene_id:
            ens2gene[r.ensembl_gene_id] = r.gene_id
    gt["gene_row"] = gt["ens"].map(ens2gene).map(g_ix)
    gt = gt[gt["gene_row"].notna()].drop_duplicates("gene_row")
    tissues = [c for c in gt.columns if c not in ("Name", "Description", "ens", "gene_row")]
    E = np.log1p(gt[tissues].to_numpy(dtype=np.float64))
    mean_expr = E.mean(axis=1, keepdims=True)
    Ec = E - mean_expr
    from sklearn.decomposition import TruncatedSVD
    ex16 = TruncatedSVD(n_components=16, random_state=20260923).fit_transform(Ec)
    ex_emb = np.zeros((n_g, 17), dtype=np.float32)
    rows_ = gt["gene_row"].astype(int).to_numpy()
    ex_emb[rows_, :16] = ex16
    ex_emb[rows_, 16] = mean_expr.ravel()
    ex_has = np.zeros(n_g, dtype=np.float32)
    ex_has[rows_] = 1.0

    feats = np.concatenate([go_emb, rx_emb, ex_emb, go_has[:, None], rx_has[:, None],
                            ex_has[:, None]], axis=1).astype(np.float32)
    names = ([f"go_svd{i}" for i in range(64)] + [f"rx_svd{i}" for i in range(32)]
             + [f"ex_svd{i}" for i in range(16)] + ["ex_mean_log_tpm", "go_present",
                                                   "reactome_present", "gtex_present"])

    # ---- functional bridge incidence B (informative terms only) -------------
    bp = [t for t in terms if ns.get(t) == "biological_process"]
    size = np.asarray(GO.sum(axis=0)).ravel()
    keep_go = [t_ix[t] for t in bp if 5 <= size[t_ix[t]] <= 200]
    rsize = np.asarray(RX.sum(axis=0)).ravel()
    keep_rx = [i for i in range(len(paths_)) if 5 <= rsize[i] <= 200]
    B = sparse.hstack([GO[:, keep_go], RX[:, keep_rx]]).tocsr()
    B.data[:] = 1.0

    # ---- GO reference PMIDs per gene (shared-literature audit) --------------
    go_pmids = defaultdict(set)
    for gid, pm in zip(gaf["gene_id"], gaf["pmids"]):
        if pm:
            go_pmids[gid].update(pm.split("|"))

    files = {k: paths6.ext(val) for k, val in v.items() if val}
    files["gtex"] = paths6.ext(GTEX)
    report = {
        "generated_utc": now_utc(), "version": version,
        "files": {k: {"path": p5.rel(f), "sha256": sha256_file(f), "bytes": f.stat().st_size}
                  for k, f in files.items()},
        "n_genes_background": int(n_g),
        "go": {"annotations_after_filters": int(len(gaf)), "terms_propagated": len(terms),
               "genes_with_go": int(go_has.sum()), "evidence_codes": gaf["evidence"]
               .value_counts().head(15).to_dict()},
        "reactome": {"pathways": len(paths_), "genes_with_pathway": int(rx_has.sum())},
        "gtex": {"tissues": len(tissues), "genes_with_expression": int(ex_has.sum())},
        "feature_dim": int(feats.shape[1]),
        "bridge": {"go_bp_terms_5_200": len(keep_go), "reactome_5_200": len(keep_rx),
                   "genes_in_bridge": int((np.diff(B.indptr) > 0).sum())},
        "svd_fit_scope": "all background genes, unlabelled public attributes",
        "nonzero_rate_is_not_effectiveness": True,
    }
    if write:
        out = paths6.TABLES
        np.save(out / f"gene_features_{version}.npy", feats)
        (out / f"gene_features_{version}_names.json").write_text(json.dumps(names), encoding="utf-8")
        sparse.save_npz(out / f"gene_function_bridge_{version}.npz", B)
        pd.DataFrame({"gene_id": list(go_pmids), "pmids": ["|".join(sorted(s)) for s in
                                                           go_pmids.values()]}) \
            .to_parquet(out / f"go_reference_pmids_{version}.parquet", index=False)
        dump_json(paths6.AUDIT / f"gene_features_{version}_report.json", report)
    return report


def load(version: str = "current") -> tuple[np.ndarray, list[str]]:
    f = np.load(paths6.TABLES / f"gene_features_{version}.npy")
    names = json.loads((paths6.TABLES / f"gene_features_{version}_names.json").read_text())
    return f, names


def load_bridge(version: str = "current") -> sparse.csr_matrix:
    return sparse.load_npz(paths6.TABLES / f"gene_function_bridge_{version}.npz").tocsr()


def main(argv=None) -> int:
    import argparse
    from .common import World6
    ap = argparse.ArgumentParser(description="Build V6 gene features.")
    ap.add_argument("--version", default="current", choices=list(VERSIONS))
    args = ap.parse_args(argv)
    paths6.ensure()
    w = World6(load_heavy=False)
    rep = build(w, args.version)
    print(json.dumps({k: v for k, v in rep.items() if k != "files"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

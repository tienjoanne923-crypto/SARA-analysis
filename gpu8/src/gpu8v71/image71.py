"""V7.1 section 7 — HPA subcellular image features (conditional extension, E8).

Resource: ``subcell_image_umap_features.tsv.zip`` (1,024-d DenseNet image
features + UMAP coordinates; the UMAP columns are only for display).  The
``cell line`` column is a cell line, never mixed with the tissue axis of the
expression modality.

Mapping and aggregation (fixed before modelling):

1. drop images whose ``genes`` field names more than one gene (no unique
   attribution) and images whose symbol does not map uniquely to a project gene
   (HGNC approved symbol; ambiguous -> exception count);
2. de-duplicate by image id (``file_prefix``);
3. equal-weight hierarchy: images -> mean per (antibody, cell line) -> mean over
   antibodies per (gene, cell line) -> mean over cell lines per gene, so a gene
   with many images does not dominate.

Outputs ``g_image [G, 1024]``, ``g_image_mask [G]``; the localisation-only
control ``g_loc [G, n_loc]`` (multi-hot main + additional HPA locations from
``subcellular_location.tsv``) and each gene's dominant cell line (for the
within-cell-line permutation).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8 import paths as p5

from . import paths71
from .common71 import array_hash, dump_json, now_utc

IMG_FILE = ("hpa", "subcell_image_umap_features.tsv.zip")
LOC_FILE = ("hpa", "subcellular_location.tsv.zip")


def symbol_map(world) -> tuple[dict, set]:
    hg = pd.read_csv(p5.external("hgnc", "hgnc_complete_set.txt"), sep="\t", dtype=str,
                     low_memory=False)
    ids = set(world.genes["gene_id"].astype(str))
    hg = hg[hg["entrez_id"].isin(ids)]
    cnt = hg["symbol"].value_counts()
    amb = set(cnt[cnt > 1].index)
    m = {s: world.idx.g_ix[e] for s, e in zip(hg["symbol"], hg["entrez_id"]) if s not in amb}
    return m, amb


def build_features(world=None, write: bool = True, chunksize: int = 4000) -> dict:
    from gpu8v6.common import World6
    w = world or World6(load_heavy=False)
    sym, amb = symbol_map(w)
    feat_cols = [f"image feature {i}" for i in range(1, 1025)]
    usecols = ["file_prefix", "cell line", "antibody", "genes", "locations"] + feat_cols
    seen = set()
    sums: dict[tuple, np.ndarray] = {}
    cnts: dict[tuple, int] = {}
    n_rows = n_multi = n_unmapped = n_dup = 0
    for ch in pd.read_csv(paths71.ext(*IMG_FILE), sep="\t", usecols=usecols, chunksize=chunksize,
                          dtype={c: np.float32 for c in feat_cols}):
        n_rows += len(ch)
        genes = ch["genes"].fillna("").astype(str)
        multi = genes.str.contains(r"[,;|]", regex=True) | (genes == "")
        n_multi += int(multi.sum())
        ch = ch[~multi]
        rows = ch["genes"].map(sym)
        n_unmapped += int(rows.isna().sum())
        ch = ch[rows.notna()].assign(gene_row=rows[rows.notna()].astype(int))
        dup = ch["file_prefix"].isin(seen) | ch["file_prefix"].duplicated()
        n_dup += int(dup.sum())
        ch = ch[~dup]
        seen.update(ch["file_prefix"])
        X = ch[feat_cols].to_numpy(dtype=np.float64)
        keys = list(zip(ch["gene_row"], ch["cell line"].astype(str), ch["antibody"].astype(str)))
        for k, x in zip(keys, X):
            if k in sums:
                sums[k] += x
                cnts[k] += 1
            else:
                sums[k] = x.copy()
                cnts[k] = 1
    # (gene, cell line, antibody) -> (gene, cell line) -> gene
    gcl: dict[tuple, list] = {}
    for (g, cl, ab), s in sums.items():
        gcl.setdefault((g, cl), []).append(s / cnts[(g, cl, ab)])
    per_gene: dict[int, list] = {}
    cl_count: dict[int, dict] = {}
    for (g, cl), lst in gcl.items():
        per_gene.setdefault(g, []).append(np.mean(lst, axis=0))
        cl_count.setdefault(g, {})[cl] = len(lst)
    G = w.n_g
    X = np.zeros((G, 1024), dtype=np.float32)
    m = np.zeros(G, dtype=bool)
    dom = np.full(G, "", dtype=object)
    for g, lst in per_gene.items():
        X[g] = np.mean(lst, axis=0).astype(np.float32)
        m[g] = True
        dom[g] = sorted(cl_count[g].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    # localisation-only control
    loc = pd.read_csv(paths71.ext(*LOC_FILE), sep="\t", dtype=str)
    ens2row = {}
    for r in w.genes.itertuples(index=False):
        e = str(r.ensembl_gene_id or "")
        if e and e != "nan":
            ens2row.setdefault(e, w.idx.g_ix[r.gene_id])
    loc["gene_row"] = loc["Gene"].map(ens2row)
    loc = loc[loc["gene_row"].notna()]
    labels = sorted({x.strip() for c in ("Main location", "Additional location")
                     for v in loc[c].fillna("") for x in v.split(";") if x.strip()})
    l_ix = {x: i for i, x in enumerate(labels)}
    L = np.zeros((G, len(labels)), dtype=np.float32)
    for r in loc.itertuples(index=False):
        for c in (r[3], r[4]):                       # Main location, Additional location
            for x in str(c if isinstance(c, str) else "").split(";"):
                if x.strip() in l_ix:
                    L[int(r.gene_row), l_ix[x.strip()]] = 1.0
    lm = L.sum(axis=1) > 0
    rep = {"generated_utc": now_utc(), "rows_read": n_rows, "multi_or_empty_gene_images": n_multi,
           "unmapped_symbol_images": n_unmapped, "duplicate_image_ids": n_dup,
           "ambiguous_symbols": len(amb), "genes_with_image": int(m.sum()),
           "cell_lines": int(len({cl for (_, cl) in gcl})),
           "median_cell_lines_per_gene": float(np.median([len(v) for v in cl_count.values()])),
           "localisation_labels": labels, "genes_with_location": int(lm.sum()),
           "rule": "unique-gene images; image -> antibody x cell line -> cell line -> gene; "
                   "UMAP columns not used as features"}
    if write:
        np.save(paths71.TABLES / "g_image.npy", X)
        np.save(paths71.TABLES / "g_image_mask.npy", m)
        np.save(paths71.TABLES / "g_loc.npy", L)
        np.save(paths71.TABLES / "g_loc_mask.npy", lm)
        pd.DataFrame({"gene_row": np.arange(G), "dominant_cell_line": dom}).to_parquet(
            paths71.TABLES / "g_image_cell_line.parquet", index=False)
        rep["hashes"] = {"g_image": array_hash(X, m), "g_loc": array_hash(L)}
        dump_json(paths71.AUDIT / "image_features_v71_report.json", rep)
    return rep


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="V7.1 image extension")
    ap.add_argument("--stage", choices=["features"], default="features")
    args = ap.parse_args(argv)
    paths71.ensure()
    rep = build_features()
    print(json.dumps({k: v for k, v in rep.items() if k != "localisation_labels"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

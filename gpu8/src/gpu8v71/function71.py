"""V7.1 sections 9-11 — legal function semantics and functional anchors.

* ``g_function [G, 768]``: mean SapBERT vector of the gene's *directly* annotated
  GO terms (GOA human, NOT qualifiers and ND removed, three namespaces, root
  terms excluded).  No disease column, clinical variant interpretation or
  literature after the GAF release is read.  Genes without a legal term keep
  ``g_function_mask = False``.
* functional anchors: GO terms (all namespaces, is_a/part_of propagated) that
  annotate 5-500 background genes, root terms excluded.  ``A_plus`` is the
  sparse gene x anchor incidence; anchor definitions are encoded with the same
  frozen SapBERT, so the anchors live in the semantic space of the disease text.

GO and function text can share literature with disease-gene relations; the
anchors are therefore described as auxiliary function supervision, never as an
independent confirmation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse

from gpu8v6 import paths6
from gpu8v6.gene_features import go_closure, parse_go_obo, read_gaf, symbol_to_gene

from . import paths71
from .common71 import array_hash, dump_json, now_utc
from .encoders71 import load_encoded

ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}
ANCHOR_MIN, ANCHOR_MAX = 5, 500


def build(world=None, write: bool = True) -> dict:
    from gpu8v6.common import World6
    w = world or World6(load_heavy=False)
    n_g, g_ix = w.n_g, w.idx.g_ix
    parents, ns = parse_go_obo(paths6.ext("go", "go-basic_current.obo"))
    anc = go_closure(parents)
    sym2gene = symbol_to_gene(w, None)
    gaf = read_gaf(paths6.ext("go", "goa_human_current.gaf.gz"), sym2gene)
    gaf = gaf[gaf["go_id"].isin(parents) & ~gaf["go_id"].isin(ROOTS)]
    gaf["gene_row"] = gaf["gene_id"].map(g_ix)
    gaf = gaf[gaf["gene_row"].notna()]
    gaf["gene_row"] = gaf["gene_row"].astype(int)
    direct = gaf.drop_duplicates(["gene_row", "go_id"])[["gene_row", "go_id", "aspect"]]

    go_x, go_key = load_encoded("sapbert_go")
    go_ids = pd.read_parquet(paths71.TABLES / "go_term_texts_v71.parquet")["term_id"].tolist()
    go_pos = {t: i for i, t in enumerate(go_ids)}
    direct = direct[direct["go_id"].isin(go_pos)]

    # g_function: mean of the direct term vectors
    G = np.zeros((n_g, go_x.shape[1]), dtype=np.float64)
    cnt = np.zeros(n_g, dtype=np.int64)
    for gr, t in zip(direct["gene_row"].to_numpy(), direct["go_id"]):
        G[gr] += go_x[go_pos[t]]
        cnt[gr] += 1
    has_f = cnt > 0
    G[has_f] /= cnt[has_f, None]

    # anchors: propagated incidence, 5-500 genes
    pairs = set()
    for gr, t in zip(direct["gene_row"].to_numpy(), direct["go_id"]):
        for a in anc.get(t, {t}):
            if a not in ROOTS:
                pairs.add((int(gr), a))
    terms = sorted({t for _, t in pairs})
    t_ix = {t: i for i, t in enumerate(terms)}
    gi = np.fromiter((g for g, _ in pairs), dtype=np.int64, count=len(pairs))
    ti = np.fromiter((t_ix[t] for _, t in pairs), dtype=np.int64, count=len(pairs))
    M = sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), (gi, ti)), shape=(n_g, len(terms)))
    size = np.asarray(M.sum(axis=0)).ravel()
    keep = [i for i, t in enumerate(terms) if ANCHOR_MIN <= size[i] <= ANCHOR_MAX and t in go_pos]
    anchors = [terms[i] for i in keep]
    A_plus = M[:, keep].tocsr()
    A_plus.data[:] = 1.0
    anchor_x = np.stack([go_x[go_pos[t]] for t in anchors]).astype(np.float32)
    rep = {"generated_utc": now_utc(),
           "gaf": {"annotations_after_filters": int(len(gaf)), "direct_pairs": int(len(direct)),
                   "aspects": direct["aspect"].value_counts().to_dict(),
                   "evidence_top": gaf["evidence"].value_counts().head(10).to_dict()},
           "g_function": {"genes_with_function": int(has_f.sum()),
                          "median_terms_per_gene": float(np.median(cnt[has_f])) if has_f.any() else 0.0},
           "anchors": {"n_terms": len(anchors), "size_rule": f"{ANCHOR_MIN}-{ANCHOR_MAX} genes",
                       "namespaces": pd.Series([ns.get(t, "") for t in anchors]).value_counts().to_dict(),
                       "genes_with_anchor": int((np.diff(A_plus.indptr) > 0).sum()),
                       "median_anchors_per_gene": float(np.median(np.diff(A_plus.indptr)[
                           np.diff(A_plus.indptr) > 0]))},
           "encoder_key": {k: go_key.get(k) for k in ("model", "revision", "pooling", "input_sha256")},
           "rule": "direct GO terms -> mean SapBERT vector; anchors = propagated GO terms of "
                   "5-500 genes; roots excluded; no disease column read"}
    if write:
        np.save(paths71.TABLES / "g_function.npy", G.astype(np.float32))
        np.save(paths71.TABLES / "g_function_mask.npy", has_f)
        sparse.save_npz(paths71.TABLES / "anchor_incidence.npz", A_plus)
        np.save(paths71.TABLES / "anchor_vectors.npy", anchor_x)
        pd.DataFrame({"term_id": anchors, "namespace": [ns.get(t, "") for t in anchors],
                      "n_genes": size[keep].astype(int)}).to_parquet(
            paths71.TABLES / "anchor_terms.parquet", index=False)
        rep["hashes"] = {"g_function": array_hash(G.astype(np.float32)),
                         "anchor_incidence": array_hash(A_plus.indptr, A_plus.indices),
                         "anchor_vectors": array_hash(anchor_x)}
        dump_json(paths71.AUDIT / "function_v71_report.json", rep)
    return rep


def main() -> int:
    paths71.ensure()
    print(json.dumps(build(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Legal disease-side representations for the residual rankers (section 7.2).

No free disease-ID embedding is trained.  A disease is described only by what a
deployed system could read about it:

* hpo64  — truncated SVD of its IC-weighted, ancestor-closed valid HPO profile;
* text64 — truncated SVD of its masked TF-IDF text;
* flags  — HPO present, definition present;
* the (at most 32) most informative directly annotated valid HPO terms, chosen
  by global information content, for the cross-attention residual, with fixed
  64-dim term embeddings taken from the same SVD.

All transforms are fitted on unlabelled annotations of every project disease;
no disease-gene label enters, and the fitting scope is recorded.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse

from . import paths6
from .common import World6, dump_json, now_utc
from .disease_text import tfidf_matrix
from .gene_hpo import HPOSpace, disease_profiles

MAX_TERMS = 32


def build(w: World6 | None = None) -> dict:
    from sklearn.decomposition import TruncatedSVD
    w = w or World6(load_heavy=False)
    space = HPOSpace(w.hpo_terms)
    Q = disease_profiles(space, w.n_d, w.idx.d_ix)
    has_hpo = np.diff(Q.indptr) > 0
    cnt = np.asarray(Q[has_hpo].sum(axis=0)).ravel()
    ic = -np.log((cnt + 1.0) / (has_hpo.sum() + 1.0))
    Qw = (Q @ sparse.diags(ic)).tocsr()
    nrm = np.sqrt(np.asarray(Qw.multiply(Qw).sum(axis=1)).ravel())
    Qn = sparse.diags(1.0 / np.maximum(nrm, 1e-12)) @ Qw
    svd = TruncatedSVD(n_components=64, random_state=20260923)
    hpo64 = svd.fit_transform(Qn).astype(np.float32)
    hpo64[~has_hpo] = 0.0
    term_emb = (svd.components_.T * svd.singular_values_[None, :]).astype(np.float32)
    term_emb /= np.maximum(np.linalg.norm(term_emb, axis=1, keepdims=True), 1e-6)
    X, _ = tfidf_matrix(level=1, ids=w.idx.diseases)
    has_text = np.diff(X.indptr) > 0
    text64 = TruncatedSVD(n_components=64, random_state=20260923).fit_transform(X).astype(np.float32)
    # direct annotations only, ranked by information content
    ph = pd.read_parquet(paths6.TABLES / "phenotypes_v6.parquet")
    terms = np.full((w.n_d, MAX_TERMS), -1, dtype=np.int32)
    for d, sub in ph.groupby("disease_id"):
        r = w.idx.d_ix.get(d)
        if r is None:
            continue
        t = [space.t_ix[h] for h in sub["hpo_id"] if h in space.t_ix]
        t = sorted(set(t), key=lambda x: -ic[x])[:MAX_TERMS]
        terms[r, :len(t)] = t
    defs = pd.read_parquet(paths6.TABLES / "disease_text.parquet").set_index("disease_id")
    has_def = np.array([int(defs.loc[d, "has_definition"]) if d in defs.index else 0
                        for d in w.idx.diseases], dtype=np.float32)
    np.savez_compressed(paths6.TABLES / "disease_semantics.npz", hpo64=hpo64, text64=text64,
                        has_hpo=has_hpo.astype(np.float32), has_def=has_def,
                        terms=terms, term_emb=term_emb, ic=ic.astype(np.float32))
    rep = {"generated_utc": now_utc(), "n_diseases": int(w.n_d),
           "diseases_with_hpo": int(has_hpo.sum()), "diseases_with_text": int(has_text.sum()),
           "hpo_svd_explained": float(svd.explained_variance_ratio_.sum()),
           "max_terms_for_attention": MAX_TERMS,
           "fit_scope": "all project diseases; annotations and ontology text only; no labels"}
    dump_json(paths6.AUDIT / "disease_semantics_report.json", rep)
    return rep


def load() -> dict:
    z = np.load(paths6.TABLES / "disease_semantics.npz")
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    paths6.ensure()
    print(json.dumps(build(), indent=2))

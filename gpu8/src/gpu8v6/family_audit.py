"""V6 section 11.2 / problem A6 — audit the frozen family definition, two levels.

The primary family (MeSH branch at depth 3, frozen before V5 modelling) is kept
as the resampling unit; it is never re-chosen by looking at results.  A depth-4
definition is reported next to it as a sensitivity level.  For each level: the
size distribution, the largest groups, families per fold, whether any family
crosses folds, and — for every test disease — the similarity of its nearest
reference disease (text, HPO and ontology), because a family split that leaves
near-duplicates on both sides would not isolate anything.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8.mapping import primary_tree_prefix

from . import paths6
from .common import World6, dump_json, now_utc
from .disease_text import tfidf_matrix
from .gene_hpo import HPOSpace, disease_profiles
from .ontology_v2 import OntologyIndex


def run() -> dict:
    w = World6(load_heavy=False)
    dis = w.diseases.set_index("disease_id")
    labelled = [d for d in w.positives if d in w.idx.d_ix]
    fam3 = {d: w.family_of.get(d, d) for d in labelled}
    fam4 = {}
    for d in labelled:
        pref = primary_tree_prefix(dis.loc[d, "tree_numbers"], depth=4) if d in dis.index else ""
        fam4[d] = f"MESH:{pref}" if pref else fam3[d]
    role = {}
    for f, ids in w.spec.folds.items():
        for d in ids:
            role[d] = f
    out = {"generated_utc": now_utc(), "levels": {}}
    for name, fam in (("depth3_primary", fam3), ("depth4_sensitivity", fam4)):
        s = pd.Series(fam)
        sizes = s.value_counts()
        cross = 0
        for g, members in s.groupby(s):
            folds = {role.get(d) for d in members.index if role.get(d, "").startswith("F")}
            if len(folds) > 1:
                cross += 1
        out["levels"][name] = {
            "n_families": int(sizes.size), "max_size": int(sizes.max()),
            "median_size": float(sizes.median()), "p95_size": float(sizes.quantile(0.95)),
            "singletons": int((sizes == 1).sum()),
            "largest": {k: int(v) for k, v in sizes.head(5).items()},
            "families_crossing_folds": int(cross),
            "families_per_fold": {f: int(s[[d for d in ids if d in s.index]].nunique())
                                  for f, ids in w.spec.folds.items()},
        }
    # nearest-reference similarity for test diseases (outer context of their fold)
    X, _ = tfidf_matrix(level=1, ids=w.idx.diseases)
    space = HPOSpace(w.hpo_terms)
    Q = disease_profiles(space, w.n_d, w.idx.d_ix)
    Qn = Q.multiply(1.0 / np.maximum(np.sqrt(np.asarray(Q.multiply(Q).sum(axis=1))), 1e-12)).tocsr()
    onto = OntologyIndex(w.diseases)
    rows = []
    for f in range(5):
        ctx = w.outer_context(f)
        ref_rows = np.array([w.idx.d_ix[d] for d in ctx.reference if d in w.idx.d_ix])
        test = w.labelled(w.fold(f)["test_diseases"])
        t_rows = np.array([w.idx.d_ix[d] for d in test])
        St = (X[t_rows] @ X[ref_rows].T).toarray()
        Sh = (Qn[t_rows] @ Qn[ref_rows].T).toarray()
        refset = set(ctx.reference)
        for i, d in enumerate(test):
            nb = onto.neighbours(d, refset, k=1)
            rows.append({"disease_id": d, "fold": f, "max_text_cosine": float(St[i].max()),
                         "max_hpo_cosine": float(Sh[i].max()) if Sh[i].size else 0.0,
                         "max_onto_ic": float(nb[0][1]) if nb else 0.0})
    nn = pd.DataFrame(rows)
    nn.to_csv(paths6.AUDIT / "family_nearest_reference_similarity.csv", index=False,
              encoding="utf-8-sig")
    out["nearest_reference"] = {c: {"median": float(nn[c].median()),
                                    "p90": float(nn[c].quantile(0.9)),
                                    "share_ge_0.9": float((nn[c] >= 0.9).mean())}
                                for c in ("max_text_cosine", "max_hpo_cosine")}
    out["nearest_reference"]["max_onto_ic"] = {"median": float(nn["max_onto_ic"].median()),
                                               "p90": float(nn["max_onto_ic"].quantile(0.9))}
    out["rule"] = ("depth 3 stays the primary resampling unit (fixed before modelling); depth 4 "
                   "is reported, never selected by results")
    dump_json(paths6.AUDIT / "family_audit.json", out)
    return out


if __name__ == "__main__":
    paths6.ensure()
    print(json.dumps(run(), indent=2, default=str))

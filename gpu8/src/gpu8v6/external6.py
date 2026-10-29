"""V6 section 12.3 — external labels, kept as exploratory re-analysis.

The external cold diseases were fixed before V5 modelling, but their V5 results
have been seen, so they are *not* an untouched confirmation.  V6 therefore:

* scores them with the frozen V6 retriever and scorers in a context whose
  references are all non-development, non-external diseases (minus strict
  equivalents of held-out diseases);
* evaluates each label layer separately and never merges denominators:
  GenCC Definitive/Strong (primary external layer), GenCC Moderate (separate
  sensitivity layer), ClinVar >= 2 stars, and the ClinVar subset whose citations
  are condition-verified and independent of CTD (section 12.2);
* reports the V5 high-confidence external set with disease-family bootstrap
  intervals, without adding seeds or draws to inflate its effective size.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gpu8 import paths as p5
from gpu8.stats import family_bootstrap

from . import paths6
from .build_candidates import build_context
from .common import ContextSpec, World6, dump_json, load_manifest, now_utc
from .metrics6 import evaluate_matrix, macro, per_disease_frame
from .scoring import HAND_WEIGHTS, fusion_scores, load_block
from .sources import SourceEngine

CTX = "external_v6"


def label_layers(w: World6) -> dict[str, pd.DataFrame]:
    ext = pd.read_parquet(p5.TABLES_DIR / "external_labels.parquet")
    ext = ext[ext["disease_id"].isin(set(w.spec.external_diseases))]
    ext = ext[ext["layer1_relation_overlap"] == 0]
    chain = pd.read_csv(paths6.EXTERNAL_RES / "clinvar_citation_chain_pairs.csv", dtype=str)
    verified = set(zip(chain.loc[chain["layer3_v6"] == "independent_verified", "disease_id"],
                       chain.loc[chain["layer3_v6"] == "independent_verified", "gene_id"]))
    g = ext[ext["source"] == "GenCC"]
    layers = {
        "gencc_definitive_strong": g[g["classification"].isin(["Definitive", "Strong"])],
        "gencc_moderate_sensitivity": g[g["classification"] == "Moderate"],
        "clinvar_2star": ext[(ext["source"] == "ClinVar") & (ext["review_stars"].fillna(0) >= 2)],
    }
    cv = ext[ext["source"] == "ClinVar"]
    layers["clinvar_citation_verified_independent"] = cv[
        [(d, str(x)) in verified for d, x in zip(cv["disease_id"], cv["gene_id"])]]
    return {k: v.drop_duplicates(["disease_id", "gene_id"]) for k, v in layers.items()}


def run(device: str = "cuda") -> dict:
    w = World6()
    eng = SourceEngine(w, device=device, with_v5=False)
    frozen_ret = load_manifest("retriever_v6_frozen.json")
    scorer = load_manifest("scorer_v6_frozen.json")
    ref = sorted(set(w.spec.family_of) - set(w.spec.dev_diseases) - set(w.spec.external_diseases))
    ref = [d for d in ref if d in w.positives]
    layers = label_layers(w)
    targets = sorted({d for df in layers.values() for d in df["disease_id"]} & set(w.idx.d_ix))
    ctx = ContextSpec(CTX, -1, "external", -1, ref, targets)
    out_dir = paths6.CACHES / "candidates"
    meta = build_context(eng, w, ctx, {d: 500 for d in targets}, frozen_ret, out_dir)
    cand = pd.read_parquet(out_dir / f"{CTX}_candidates.parquet")
    genes_by_d = {d: sub.sort_values("rank")["gene_row"].to_numpy() for d, sub in
                  cand.groupby("disease_id")}
    block = load_block(CTX, ids=targets)
    scores = {"F_main": fusion_scores(block, scorer["weights"]),
              "F_hand": fusion_scores(block, HAND_WEIGHTS),
              "retrieval_order": block.feat("fused"), "degree_prior": block.feat("prior")}
    fam = w.family_of
    rows, per_all = [], []
    g_ix = w.idx.g_ix
    for lname, lab in layers.items():
        q = lab.groupby("disease_id")["gene_id"].apply(lambda s: {g_ix[str(x)] for x in s
                                                                  if str(x) in g_ix}).to_dict()
        ids = [d for d in block.ids if q.get(d)]
        if not ids:
            continue
        sub = block.subset(ids)
        y = np.stack([np.isin(sub.genes[i], list(q[d])).astype(np.float32)
                      for i, d in enumerate(ids)])
        nq = np.array([len(q[d]) for d in ids], dtype=float)
        pos = {d: i for i, d in enumerate(block.ids)}
        for sname, sc in scores.items():
            s = sc[[pos[d] for d in ids]]
            per = per_disease_frame(ids, evaluate_matrix(s, y, sub.genes, nq, ks=(10, 50)))
            per["layer"], per["scorer"] = lname, sname
            per_all.append(per)
            vals = per["recall10_end"].to_numpy()
            fams = np.array([fam.get(d, d) for d in ids])
            lo, hi, _ = family_bootstrap(vals, fams, draws=10000)
            m = macro(per)
            rows.append({"layer": lname, "scorer": sname, "n_diseases": len(ids),
                         "n_pairs": int(nq.sum()), "n_families": int(len(set(fams))),
                         "recall10_end": m["recall10_end"], "ci_low": lo, "ci_high": hi,
                         "coverage": m["coverage"], "oracle_recall10": m["oracle_recall10"],
                         "zero_coverage": m["zero_coverage"]})
    res = pd.DataFrame(rows)
    out = paths6.EXTERNAL_RES
    res.to_csv(out / "external_v6_layers.csv", index=False, encoding="utf-8-sig")
    pd.concat(per_all).to_csv(out / "external_v6_per_disease.csv", index=False, encoding="utf-8-sig")
    summ = {"generated_utc": now_utc(), "context": meta,
            "status": "exploratory re-analysis of an already-viewed external set; no new untouched source",
            "layers": res.to_dict(orient="records")}
    dump_json(out / "external_v6_summary.json", summ)
    return summ


if __name__ == "__main__":
    paths6.ensure()
    print(json.dumps(run(), indent=2, default=str)[:6000])

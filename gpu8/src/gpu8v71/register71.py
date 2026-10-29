"""Register the V7.1 analysis plan before any V7.1 test-fold evaluation.

The five outer folds have been used in V5 and V6 designs, so V7.1 five-fold
numbers are internal estimates.  What is fixed here, before the formal chain and
before any test disease is scored:

* the evaluation matrix E1-E8, the strong controls (F7 and the re-registered
  retrieval-channel-only LightGBM), the same-resource fixed fusion and the late
  fusion variants;
* the development selection of the core (fold 0 dev/F1 only), the modality set
  and the admission decision;
* the **new confirmation set**: five-fold diseases' GenCC Definitive/Strong and
  ClinVar >= 2-star pairs that are *not* CTD relations of the disease.  It was
  never used by V5 or V6 for design or evaluation; its hash is frozen now and it
  is evaluated once at the end (filtered ranking: the disease's CTD positives are
  removed from each ranked list before scoring);
* the fit budget and the stopping / downgrade rules.
"""
from __future__ import annotations

import json

import pandas as pd

from gpu8 import paths as p5
from gpu8.hashing import sha256_obj

from . import paths71
from .common71 import load_manifest, now_utc, write_manifest

PLAN = {
    "status_of_folds": ("the five outer folds were used by V5/V6; V7.1 five-fold numbers are "
                        "internal estimates; the new confirmation set below is the only untouched "
                        "evaluation"),
    "primary_endpoint": "disease macro end-to-end Recall@10 over the fixed evaluation background; "
                        "multi-seed: per run per disease first, then averaged (never a union of hits)",
    "secondary": ["Hit@10", "Recall@50", "coverage (500 / work pool)", "oracle U@10", "NDCG@10",
                  "new and squeezed-out hits"],
    "E1": {"task": "outer cold start, 4,238 five-fold diseases, 5 folds x 3 seeds",
           "methods": ["F7 (V6 frozen fusion, deterministic)",
                       "V6 LightGBM (archived V6 main scorer, read from runs_v6)",
                       "retrieval-channel-only LightGBM on the V6 500 (re-registered strong control, 3 seeds)",
                       "same-resource fixed fusion (zero-training SapBERT/HPA/ESM similarities, weights on dev)",
                       "multimodal core alone (own 500 list, degree-prior fill marked)",
                       "late fusion, non-negative fixed (F7 + core, weight on dev)",
                       "late fusion, LightGBM on the OOF work pool (V6 500 U core top-500)"]},
    "E2": {"task": "same data, different algorithms (seed 42)",
           "methods": ["raw concatenation of modality vectors into LightGBM", "linear two-tower",
                       "core without / with anchor alignment (lambda_A flipped)", "full core"]},
    "E3": {"task": "association-cold genes",
           "strata": ["visible_ctd", "visible_hpo_only", "ppi_only", "unseen"],
           "readouts": ["work-pool coverage", "final-500 coverage", "top-10 recovery"],
           "strict_variant": "anchor training restricted to association-warm genes of the context "
                             "(stricter gene-supervision cold); cold genes keep frozen features only"},
    "E4": {"task": "resource x method 2x2",
           "cells": {"old resources, old method": "V6 LightGBM",
                     "new resources, old method": "LightGBM on the V6 500 + raw V7.1 modality vectors",
                     "old resources, new method": "late fusion with a core on GO function + GTEx expression (no ESM, no HPA)",
                     "new resources, new method": "late fusion with the full core"}},
    "E5": {"task": "modality behaviour on fixed candidates",
           "checks": ["single-modality shutdown", "paired permutations (sequence: length+degree; "
                      "function: annotation depth+degree; tissue weights: tissue count+broad flag)",
                      "missing-mask-only second-level model", "full-background chunk invariance"]},
    "E6": {"task": "candidate attribution",
           "comparisons": ["old V6 500 fixed, only the score changes", "shared new work pool",
                           "each system end to end", "new hits and squeezed-out old hits"]},
    "E7": {"task": "support and confirmation",
           "parts": ["V6 support task kept as archived", "small support sets (k = 1, diseases "
                     "with 2-5 known genes) with the V6 support line and an optional core term "
                     "whose weight is set on dev (0 allowed)",
                     "new confirmation set (frozen below), every method evaluated once"]},
    "E8": {"task": "optional channels, exploratory",
           "image": "HPA DenseNet image features (obtained); controls: localisation-only, "
                    "within-cell-line image-gene permutation, cell-line-centred features",
           "mouse": "IMPC calls + MGI 1:1 homology + MGI curated MP-HP SSSOM; directional match "
                    "S(d,m); enabled only if the programmatic mapping checks pass; manual spot check "
                    "exported, not performed"},
    "admission_rules": {
        "core_config": "one-SE rule on fold-0 dev, simplest first (rank, gate, lambda_A)",
        "modality_kept": "paired permutation lowers dev Recall@10 by > 1 paired SE",
        "late_fusion": "core enters late fusion iff it adds dev positives outside the V6 500",
        "second_level": "fixed fusion and LightGBM both reported; LightGBM early-stopped on F_{j+1}",
        "multimodal_claim": ("'multimodal gain' may be claimed only if (i) late fusion beats the "
                             "channel-only LightGBM and F7 on E1, (ii) paired permutations lower "
                             "performance (E5), and (iii) the gain is not reproduced by the "
                             "mask-only model or the raw concatenation"),
    },
    "stopping_and_downgrade_rules": [
        "if the same-resource fixed fusion beats the learned gate, keep the fixed fusion",
        "if coverage rises but top-10 falls, report pool change and squeezed-out hits, do not claim",
        "if permutation leaves a modality's effect unchanged, report it as not used",
        "if the anchor stage gives no dev gain, report lambda_A = 0 as the selected configuration",
        "the title contains 'multimodal' only if the multimodal_claim rule is met",
    ],
    "fit_budget": {"main_chain": "5 folds x 3 seeds x (3 inner + 1 outer) = 60 core fits",
                   "variant_chains": "seed 42: core_altanchor, two_tower, core_oldres, core_strict "
                                     "= 4 x 20 = 80 fits",
                   "outer_only": "seed 42: core_bgctrl, core_image, core_mouse (if admitted) = 15 fits",
                   "second_level": "LightGBM late fusion 15, channel-only LightGBM 15, E2/E4/E5 "
                                   "variants seed 42 x 5 folds"},
}


def confirmation_set(w) -> pd.DataFrame:
    ext = pd.read_parquet(p5.TABLES_DIR / "external_labels.parquet")
    fold_d = {d for f in w.spec.folds.values() for d in f}
    e = ext[ext["disease_id"].isin(fold_d) & (ext["layer1_relation_overlap"] == 0)]
    g = e[(e["source"] == "GenCC") & e["classification"].isin(["Definitive", "Strong"])]
    c = e[(e["source"] == "ClinVar") & (e["review_stars"].fillna(0) >= 2)]
    u = pd.concat([g, c]).drop_duplicates(["disease_id", "gene_id"])
    u = u[u["disease_id"].isin(set(w.positives)) & u["gene_id"].astype(str).isin(w.idx.g_ix)]
    pos = {(d, g) for d, gs in w.positives.items() for g in gs}
    u = u[[(d, str(x)) not in pos for d, x in zip(u["disease_id"], u["gene_id"])]]
    u = u.assign(gene_id=u["gene_id"].astype(str),
                 gene_row=u["gene_id"].astype(str).map(w.idx.g_ix).astype(int))
    return u[["disease_id", "gene_id", "gene_row", "source", "classification", "review_stars"]] \
        .sort_values(["disease_id", "gene_row"]).reset_index(drop=True)


def main() -> int:
    from gpu8v6.common import World6
    paths71.ensure()
    prev = load_manifest("protocol_v71_registered.json")
    if prev:
        print("already registered at", prev.get("written_utc"))
        return 0
    tested = [p for p in paths71.RUNS.glob("V71Z__*") if (p / "result.json").exists()]
    if tested:
        raise RuntimeError("V7.1 test evaluations exist before registration: refuse to register")
    w = World6(load_heavy=False)
    conf = confirmation_set(w)
    conf.to_parquet(paths71.TABLES / "confirmation_set_v71.parquet", index=False)
    conf_hash = sha256_obj(conf[["disease_id", "gene_id"]].values.tolist())
    frozen = {k: (load_manifest(k) or {}).get("payload_hash") for k in
              ("featurepack_v71.json", "models_v71.json", "resources_v71.json",
               "core_dev_selection_v71.json")}
    if not frozen["core_dev_selection_v71.json"]:
        raise RuntimeError("development selection missing")
    sel = load_manifest("core_dev_selection_v71.json")
    write_manifest("protocol_v71_registered.json", {
        "plan": PLAN, "frozen_inputs": frozen,
        "core_selection": {"selected": sel["selected"], "modalities": sel["modalities"],
                           "admission": sel["admission"]},
        "confirmation_set": {"n_pairs": int(len(conf)), "n_diseases": int(conf["disease_id"].nunique()),
                             "by_source": conf["source"].value_counts().to_dict(),
                             "sha256_pairs": conf_hash,
                             "file": "tables_v71/confirmation_set_v71.parquet",
                             "evaluation": "filtered: each method's ranked list minus the disease's CTD "
                                           "positives; pool misses count as misses"},
        "registered_utc": now_utc(),
        "registered_before_any_v71_test_evaluation": True})
    print(json.dumps({"confirmation_pairs": len(conf), "diseases": conf["disease_id"].nunique(),
                      "hash": conf_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

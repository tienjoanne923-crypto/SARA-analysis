"""Register the V6 analysis plan before any V6 test-fold evaluation (section 11).

The five outer folds were used by V5, so V6 results on them are internal
estimates, not an untouched confirmatory test.  What this registration fixes in
advance is the V6 decision logic, so that no choice below is taken from a V6
test score.
"""
from __future__ import annotations

import json

from . import paths6
from .common import load_manifest, now_utc, write_manifest

PLAN = {
    "status_of_folds": ("the five outer folds are V5 diagnostic material; V6 five-fold "
                        "numbers are internal estimates, not a new confirmatory test"),
    "primary_endpoint": "disease macro end-to-end Recall@10 over a fixed 500-gene budget",
    "secondary": ["oracle U@10", "coverage", "zero-coverage share", "NDCG@10", "MRR",
                  "Recall@50", "ranking loss U-R", "capacity loss", "conditional efficiency eta"],
    "families": {
        "Z_zero_support": {
            "Z-A": "V6 pipeline (V6 candidates + selected scorer) vs V5 b3 fusion on V5 candidates (paired by disease)",
            "Z-B": "best admitted learned scorer vs tuned fusion F on identical V6 candidates",
            "Z-C": "V6 retrieval vs V5 retrieval: oracle U@10 at budget 500 (paired by disease)",
        },
        "S_support": {
            "S-A": "rwr_equal at k = 5 vs semantic score at k = 0, S-fixed candidates",
            "S-B": "joint (omega frozen on dev) vs rwr_equal at k = 5, S-fixed",
            "S-C": "evidence-group seeds (eta = 1) vs equal seeds at k = 5 (Delta_seed), traceable episodes",
        },
        "multiplicity": "Holm within each family; everything else exploratory",
        "resampling": "disease-family bootstrap (10,000); draws / seeds averaged inside a disease first",
    },
    "admission_rules": {
        "learned_scorer": ("main zero-support scorer = the simplest of F7 < LR < LightGBM < bilinear < "
                           "attention whose paired disease-level shortfall to the best model on the "
                           "two inner validation sets (1,712 diseases, fold 0) is within one SE "
                           "(manifests_v6/z_main_scorer.json); result: LightGBM, with F7 as the "
                           "strong deterministic control"),
        "formal_fits": ("LightGBM x 5 folds x 3 seeds (15), attention residual x 5 x 3 (15), bilinear "
                        "residual x 5 x seed 42 (5), LR x 5 x seed 42 (5) = 40 core fits; retrieval "
                        "order, degree prior, F_hand, F4 and F7 are deterministic, one run per fold"),
        "residual_kappa": "kappa = 0 is always admissible; the residual is kept per fit only if it beats F on the inner validation fold",
        "neural_branch": "a disease branch counts as useful only if P1 lowers the metric and normal-input performance beats F (section 8.4)",
        "support_config": "alpha, nu and omega frozen on development families before the test cohort is scored",
    },
    "stopping_and_downgrade_rules": [
        "new data raise coverage but tuned F is best: keep retrieval and data contribution, drop the neural residual",
        "residual does not improve normal metrics and only its permutation is significant: not part of the main method",
        "support helps in S-adaptive but not in S-fixed: contribution is support-driven candidate recovery, not in-list ranking",
        "group weights change seeds and ranks but not the metric: report a negative mechanism and withdraw the evidence-dependence accuracy claim",
        "historical inputs not recoverable: keep the edge-level temporal wording",
    ],
    "fits_budget": "at most 45 core formal fits (main method + two strong controls, 5 folds x 3 seeds); deterministic scorers and propagation are not repeated over fake seeds",
}


def main() -> int:
    paths6.ensure()
    prev = load_manifest("protocol_v6_registered.json")
    if prev:
        print("already registered at", prev.get("written_utc"))
        return 0
    frozen = {k: (load_manifest(k) or {}).get("payload_hash") for k in
              ("retriever_v6_frozen.json", "scorer_v6_frozen.json", "devfit_v6_selection.json",
               "support_v6_frozen.json", "z_main_scorer.json")}
    write_manifest("protocol_v6_registered.json", {"plan": PLAN, "frozen_inputs": frozen,
                                                   "registered_before_any_v6_test_evaluation": True})
    print(json.dumps(PLAN, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

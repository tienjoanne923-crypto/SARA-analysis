"""V8.1 registration — written before any V8.1 test evaluation (§10, §13, §16).

Refuses to run when a V8.1 test evaluation already exists.  Freezes: mask sets,
evidence blocks, evidence variants, the decision hierarchy, every development
choice per outer context, the rules for tasks without a native context, the
bootstrap, the external / historical eligibility definitions, the comparator
protocols, the 650M trigger, reused runs and the label-free F_j candidate hashes.
Later development choices (support omega/gamma, cohort Top-K runs) are added as
time-stamped appendices *before* their own test use.
"""
from __future__ import annotations

import json
from collections import Counter

from . import paths8
from .common8 import load_manifest, now_utc, write_manifest

HIERARCHY = [
    {"id": 1, "claim": "end-to-end", "a": "deployed V8.1 (strict 500 full pipeline)",
     "b": "channel-only LightGBM on its own V6 500-list (table 3)",
     "test": "paired family-bootstrap 95% CI of the per-disease Recall@10 difference (seeds averaged per disease)",
     "success": "CI lower bound > 0"},
    {"id": 2, "claim": "non-inferiority to the strongest same-budget fusion", "condition": "tested after 1",
     "a": "deployed V8.1", "b": "V7.1 direct late fusion retrained on the new strict 500",
     "success": "CI lower bound of the difference > -0.005"},
    {"id": 3, "claim": "cold recovery", "condition": "only if 2 holds",
     "a": "deployed V8.1", "b": "V7.1 direct late fusion on the new strict 500",
     "metric": "R^c@10 over diseases with |Q^c_d| > 0", "success": "CI lower bound > 0"},
    {"id": 4, "claim": "all other comparisons", "success": "descriptive, effect with 95% family-bootstrap CI"},
]


def test_results_exist() -> list[str]:
    hits = []
    for pat in ("**/eval_test*.parquet", "**/per_disease__test*.parquet", "**/metrics_test*.json"):
        hits += [str(p) for p in paths8.RESULTS.glob(pat)]
    return hits


def majority(vals: list, simpler_key=None):
    c = Counter(json.dumps(v, sort_keys=True) for v in vals)
    top = max(c.values())
    cands = [json.loads(k) for k, n in c.items() if n == top]
    if simpler_key and len(cands) > 1:
        cands.sort(key=simpler_key)
    return cands[0]


def register() -> dict:
    prev = load_manifest("protocol_v8_registered.json")
    if prev:
        print("already registered at", prev.get("written_utc"))
        return prev
    ex = test_results_exist()
    if ex:
        raise RuntimeError(f"V8.1 test evaluations exist before registration: {ex[:3]}")
    from .formal8 import sel_cfg, selection
    per = {}
    for j in range(5):
        s = selection(j)
        cfg = sel_cfg(s)
        per[f"f{j}_outer"] = {"controls": s["controls"], "student": s["student"], "w": s["w"]["w"],
                              "w_dev_coverage": s["w"]["coverage"], "corrector": cfg["corrector"],
                              "A": cfg["A"], "deploy": cfg["deploy"], "ab": s["ab"],
                              "underfit_31_leaf_trigger": s.get("underfit_31_leaf_trigger", False),
                              "pool_hashes": s["summary"].get("pool_hashes", {}),
                              "cold_positive_diagnostic": [
                                  {k: r["diag"][k] for k in ("cold_positives_in_pairs", "diseases_with_cold_positive",
                                                             "D_star", "skipped_no_positive_in_pool")}
                                  for r in s["summary"]["corrector"]][:1]}
    no_native = {
        "controls": majority([p["controls"] for p in per.values()]),
        "student": majority([p["student"] for p in per.values()], simpler_key=lambda x: x is not None),
        "w": majority([p["w"] for p in per.values()], simpler_key=lambda x: -x),
        "corrector": majority([p["corrector"] for p in per.values()], simpler_key=lambda x: (x["beta"], -x["lam"])),
        "A": majority([p["A"] for p in per.values()], simpler_key=lambda x: (x["num_leaves"], -x["min_child_samples"])),
        "deploy": majority([p["deploy"] for p in per.values()], simpler_key=lambda x: 0 if x == "A" else 1),
        "rule": "most common setting over the five outer contexts; ties -> the simpler setting"}
    masks = load_manifest("go_masks_v8.json")
    ev = json.loads((paths8.AUDIT / "evidence_policy_audit_v8.json").read_text(encoding="utf-8"))
    reg = {
        "registered_before_any_v8_test_evaluation": True,
        "hierarchy": HIERARCHY,
        "primary_endpoint": "disease-macro end-to-end Recall@10 (500 budget), seeds averaged per disease",
        "key_secondary": "R^c@10 (reference-path cold queries) with R^{c->all}@10 reported alongside",
        "bootstrap": {"unit": "disease family", "draws": 10000, "seed": 20260923, "ci": 0.95},
        "non_inferiority_margin": -0.005,
        "development_is_not_independent": "5 x 864 development evaluations are correlated repeats",
        "per_context_choices": per,
        "tasks_without_native_context": no_native,
        "teacher": {"architecture": "V7.1 core (rank 128, global gate, lambda_A 0.2, 159,749 params)",
                    "checkpoint": "s_F full-background Recall@10 on F_{j+1} (same context)",
                    "controls_choice": "per outer context on development s_F Recall@10"},
        "student": {"structure": "480-256(GELU)-128-L2, 156,032 params; linear control 61,568",
                    "acceptance": load_manifest("dev_student_selection_v8.json", {}).get("rule"),
                    "provisional_w": 0.5},
        "corrector": {"objective": "gated positive-unlabelled pairwise softplus, v = 1 + beta c, L2 on a r, x|D*|",
                      "grid": "beta {0,1,2} x lambda {0.03, 0.1}; 15 leaves; 200 rounds; best F_{j+1} round",
                      "unlabelled": "64 per disease: 32 highest B + 32 uniform (hashed seed)",
                      "cold_positive_threshold": "cold-view simulation only if cold positives in training pairs "
                                                 "< 300 or diseases with one < 100 (training pool only)",
                      "underfit_31_leaves": "only if the selected corrector's training loss still fell by > 2% "
                                            "between rounds 150 and 200 (training pool only)"},
        "ab_rule": "keep correctors within 0.005 of the best A on dev R@10; pick max dev R^c (ties R@10, "
                   "smaller beta, larger lambda); deploy B if it beats A on dev R@10, or is within 0.005 "
                   "and has higher dev R^c; otherwise A",
        "go_masks": {k: masks.get(k) for k in ("G_mask_dev", "G_mask_test", "hash_dev", "hash_test", "rule")},
        "evidence": {"variants": list(ev["variants"].keys()),
                     "blocks": {k: ev["layer3_evidence_blocks"][k] for k in
                                ("n_components", "giant_component_relations", "blocks_chosen",
                                 "relations_held", "pmids_held", "rule")},
                     "block_evaluation_relations": "held relations of F_j test diseases, in their own outer "
                                                   "context; new vs channel-only in the full and the isolated world",
                     "layer2_flag": "evaluation_conditioned_audit = true; never a production input"},
        "external": {"618_set": "tables_v71/confirmation_set_v71.parquet re-analysed (labelled external "
                                "re-analysis, not a first confirmation)",
                     "fields": ["seen_in_training_disease", "same_relation_seen", "shared_original_evidence"],
                     "unknown_provenance": "never passes the literature-independence field",
                     "new_source": "declared unobtainable in this round: the local GenCC and ClinVar files are "
                                   "dated 2026-09-22 (already used to build the 618 set); a re-download would add "
                                   "almost nothing and other common curated sources are contained in GenCC"},
        "historical": {"cutoff": "2023-12-31", "X2": "new relations of diseases known at the cut-off; old "
                       "genes removed from the ranking; not disease-cold", "X1": "diseases without association "
                       "supervision at the cut-off; record whether text / HPO existed then",
                       "sequence_version": "MANE 2025 sequences (static attribute, stated as a limitation)"},
        "comparators": {"KDGene": "gpu8/baselines/kdgene.py on the strict 500 of every fold, epoch chosen on "
                                  "F_{j+1} cold validation", "SIGN": "gpu8/baselines/learned.py SIGN-style on the "
                                  "strict 500 of every fold", "ProtST": "weights only; zero-shot text-protein "
                                  "retrieval (task adaptation) and student input (open-knowledge control)",
                        "quantum_walk": "gpu8/baselines/simple.py b11 on the support task, all folds"},
        "esm650m_trigger": "run once only if the 35M student is accepted in >= 3 of 5 development contexts",
        "reused": {"seed42_main": "development fits reused for the formal seed-42 main chain (identical contract)"},
        "appendices": [],
    }
    return write_manifest("protocol_v8_registered.json", {**reg, "registered_utc": now_utc()})


def appendix(name: str, payload: dict) -> dict:
    reg = load_manifest("protocol_v8_registered.json")
    if not reg:
        raise RuntimeError("no registration")
    reg = dict(reg)
    reg.setdefault("appendices", []).append({"name": name, "added_utc": now_utc(), **payload})
    reg.pop("payload_hash", None)
    return write_manifest("protocol_v8_registered.json", reg)


if __name__ == "__main__":
    paths8.ensure()
    register()
    r = load_manifest("protocol_v8_registered.json")
    print(json.dumps(r["tasks_without_native_context"], indent=1, default=str))

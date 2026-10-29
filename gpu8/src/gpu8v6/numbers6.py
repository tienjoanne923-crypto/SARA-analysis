"""Collect every number the V6 manuscript quotes, with its source file.

``collect()`` returns a flat dict ``key -> value`` and writes
``reports_v6/manuscript_numbers_v6.csv`` (key, value, source, note) so that each
number in the text can be traced to the artefact that produced it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gpu8 import paths as p5

from . import paths6
from .common import dump_json, load_manifest
from .ledger6 import collect as collect_ledger


class Book:
    def __init__(self):
        self.v: dict = {}
        self.rows: list = []

    def put(self, key, value, source, note=""):
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, (np.integer,)):
            value = int(value)
        self.v[key] = value
        self.rows.append({"key": key, "value": value, "source": str(source), "note": note})
        return value


def J(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def collect() -> dict:
    b = Book()
    A = paths6.AUDIT
    # ---- availability ---------------------------------------------------
    av = J(A / "availability_summary.json")
    src = A / "availability_summary.json"
    b.put("n_labelled", av["n_labelled_diseases"], src)
    b.put("hpo_v5", av["hpo_v5"], src)
    b.put("hpo_v6", av["hpo_v6"], src)
    b.put("five_n", av["five_shot"]["n"], src)
    b.put("five_nohpo_v5", av["five_shot"]["frac_without_hpo_v5"], src)
    b.put("five_nohpo_v6", av["five_shot"]["frac_without_hpo_v6"], src)
    b.put("five_no_record", av["five_shot"]["hpo_class_counts"].get("no_resource_record", 0), src)
    b.put("five_repaired", av["five_shot"]["hpo_class_counts"].get("mapping_failure_repaired", 0), src)
    b.put("single_n", av["single_positive"]["n"], src)
    b.put("single_frac", av["single_positive"]["frac"], src)
    b.put("single_hpo", av["single_positive"]["frac_with_hpo_v6"], src)
    for k, v in av["route_zero_support_counts"].items():
        b.put(f"route_{k}", v, src)
    for k, v in av["hpo_class_counts"].items():
        b.put(f"hpocls_{k}", v, src)
    b.put("hier_nohpo_neighbour", av["hierarchy_retriever_check"]["frac_no_hpo_with_ontology_neighbour_in_reference"], src)
    b.put("eq_leak_test", av["strict_equivalence"]["test_diseases_with_equivalent_in_v5_outer_reference"], src)
    b.put("eq_groups", av["strict_equivalence"]["labelled_groups_with_several_members"], src)
    b.put("go_shared_frac", av["shared_literature_go_vs_ctd"]["fraction"], src)
    b.put("go_shared_n", av["shared_literature_go_vs_ctd"]["relations_whose_pmid_is_also_a_go_reference_of_the_gene"], src)
    b.put("frac_definition", av["frac_with_definition"], src)
    ph = J(A / "phenotypes_v6_report.json")
    src = A / "phenotypes_v6_report.json"
    b.put("hpoa_ids", ph["n_hpoa_database_ids"], src)
    b.put("hpoa_mapped_v5", ph["n_hpoa_ids_mapped_v5"], src)
    b.put("hpoa_mapped_v6", ph["n_hpoa_ids_mapped_strict_v6"], src)
    b.put("hpoa_new_v6", ph["n_hpoa_ids_new_in_v6"], src)
    b.put("strict_conflicts", ph["n_strict_conflicts_dropped"], src)
    b.put("dis_gain_hpo_all", ph["n_diseases_gaining_valid_hpo"], src)
    tx = J(A / "disease_text_report.json")
    b.put("text_masked_diseases", tx["diseases_with_any_mask_l1"], A / "disease_text_report.json")
    b.put("text_masked_tokens", tx["tokens_masked_l1"], A / "disease_text_report.json")
    gfr = J(A / "gene_features_current_report.json")
    src = A / "gene_features_current_report.json"
    b.put("genes_go", gfr["go"]["genes_with_go"], src)
    b.put("genes_rx", gfr["reactome"]["genes_with_pathway"], src)
    b.put("genes_gtex", gfr["gtex"]["genes_with_expression"], src)
    b.put("gene_feat_dim", gfr["feature_dim"], src)
    b.put("bridge_genes", gfr["bridge"]["genes_in_bridge"], src)
    b.put("n_genes", gfr["n_genes_background"], src)
    fam = J(A / "family_audit.json")
    src = A / "family_audit.json"
    b.put("fam3_n", fam["levels"]["depth3_primary"]["n_families"], src)
    b.put("fam3_max", fam["levels"]["depth3_primary"]["max_size"], src)
    b.put("fam4_n", fam["levels"]["depth4_sensitivity"]["n_families"], src)
    b.put("fam4_max", fam["levels"]["depth4_sensitivity"]["max_size"], src)
    b.put("fam_cross", fam["levels"]["depth3_primary"]["families_crossing_folds"], src)
    b.put("nn_text_ge09", fam["nearest_reference"]["max_text_cosine"]["share_ge_0.9"], src)
    b.put("nn_hpo_ge09", fam["nearest_reference"]["max_hpo_cosine"]["share_ge_0.9"], src)
    # ---- V5 audit ---------------------------------------------------------
    dec = J(A / "oracle_decomposition_v5.json")
    src = A / "oracle_decomposition_v5.json"
    per = {r["method"]: r for r in dec["per_method"]}
    b.put("v5_b3_R", per["b3_cpu_fusion"]["macro_recall10_end"], src)
    b.put("v5_b3_U", per["b3_cpu_fusion"]["macro_oracle_recall10"], src)
    b.put("v5_b3_eta", per["b3_cpu_fusion"]["macro_eta10"], src)
    b.put("v5_b9_R", per["b9_ours"]["macro_recall10_end"], src)
    b.put("v5_b9_eta", per["b9_ours"]["macro_eta10"], src)
    b.put("v5_b3_share_recovered", per["b3_cpu_fusion"]["share_recovered"], src)
    b.put("v5_identity", all(c["identity_holds"] for c in dec["identity_checks"].values()), src)
    ab = pd.read_csv(A / "ablation_unified.csv").set_index("variant")
    src = A / "ablation_unified.csv"
    for v in ab.index:
        short = v.split("_")[0]
        for c in ("M_full_mean", "M_abl_mean", "delta_abl", "ci_low", "ci_high",
                  "run_level_delta_sd", "runs_abl_worse", "b9_seed_to_seed_mean_abs_diff",
                  "v5_table4_published_full_minus_abl_unmatched_seeds", "mean_coverage_abl",
                  "mean_coverage_full"):
            if c in ab.columns:
                b.put(f"abl_{short}_{c}", ab.loc[v, c], src)
        b.put(f"abl_{short}_same_cand", bool(ab.loc[v, "candidates_identical_in_all_runs"]), src)
    hf = J(A / "hitflow_v5_b9_f0_s42.json")
    src = A / "hitflow_v5_b9_f0_s42.json"
    for k in ("coverage_small", "coverage_big", "recall_small", "recall_big", "total_pushed_out",
              "total_new_positive_entered_top", "total_new_nonhit_in_top",
              "total_top_slots_taken_by_new", "max_abs_score_diff", "all_nested"):
        b.put(f"hf_{k}", hf[k], src)
    pv5 = J(paths6.PERMUTATION / "v5_b9_permutations_f0.json")
    src = paths6.PERMUTATION / "v5_b9_permutations_f0.json"
    for r in pv5["summary"]:
        b.put(f"pv5_{r['test']}_{r['scheme']}_delta", r["delta"], src)
        b.put(f"pv5_{r['test']}_{r['scheme']}_perm", r["perm"], src)
        b.put(f"pv5_orig_{r['scheme']}", r["orig"], src)
    # ---- retrieval -----------------------------------------------------------
    fr = load_manifest("retriever_v6_frozen.json")
    src = paths6.MANIFESTS / "retriever_v6_frozen.json"
    b.put("ret_mode", fr["mode"], src)
    b.put("ret_weights", json.dumps({k: round(v, 2) for k, v in fr["weights"].items()},
                                    ensure_ascii=False), src)
    b.put("ret_dev_U", fr["dev_metrics"]["U10"], src)
    b.put("ret_dev_cov", fr["dev_metrics"]["coverage"], src)
    b.put("ret_dev_zero", fr["dev_metrics"]["zero_coverage"], src)
    b.put("ret_dev_direct", fr["dev_metrics"]["coverage_direct_evidence"], src)
    b.put("ret_dev_fill", fr["dev_metrics"]["coverage_prior_fill"], src)
    b.put("ret_n_grid", fr["n_grid_configs"], src)
    b.put("ret_n_refine", fr["n_refine_configs"], src)
    for k, v in fr["ladder"].items():
        b.put(f"lad_{k}_U", v["U10"], src)
        b.put(f"lad_{k}_direct", v["coverage_direct_evidence"], src)
        b.put(f"lad_{k}_fill", v["coverage_prior_fill"], src)
        b.put(f"lad_{k}_zero", v["zero_coverage"], src)
    for k, v in fr["singles"].items():
        b.put(f"single_{k}_cov", v["coverage"], src)
    tdev = pd.read_csv(paths6.RETRIEVAL / "retriever_tuning_dev_summary.csv", index_col=0)
    b.put("prior_only_cov", tdev.loc["prior_only", "coverage"], paths6.RETRIEVAL / "retriever_tuning_dev_summary.csv")
    sr = J(paths6.RETRIEVAL / "source_report_test.json")
    src = paths6.RETRIEVAL / "source_report_test.json"
    b.put("test_cov", sr["coverage_full"], src)
    b.put("v5_zero_n", sr["v5_zero_coverage_diseases"], src)
    b.put("rescued_n", sr["rescued_by_v6"], src)
    b.put("rescued_share", sr["rescued_share"], src)
    for s_, r in sr["sources"].items():
        for k_, v_ in r.items():
            b.put(f"src_{s_}_{k_}", v_, src)
    # ---- scoring ------------------------------------------------------------------
    sc = load_manifest("scorer_v6_frozen.json")
    src = paths6.MANIFESTS / "scorer_v6_frozen.json"
    b.put("dev_hand", sc["hand_v5_weights_on_v6_channels"]["dev_recall10"], src)
    b.put("dev_ret", sc["retrieval_order"]["dev_recall10"], src)
    b.put("dev_F4", sc["F4"]["dev_recall10"], src)
    b.put("dev_F7", sc["F7"]["dev_recall10"], src)
    b.put("F7_weights", json.dumps({k: round(v, 2) for k, v in sc["F7"]["weights"].items() if v},
                                   ensure_ascii=False), src)
    b.put("F4_weights", json.dumps({k: round(v, 2) for k, v in sc["F4"]["weights"].items() if v},
                                   ensure_ascii=False), src)
    zm = load_manifest("z_main_scorer.json")
    src = paths6.MANIFESTS / "z_main_scorer.json"
    for r in zm["table"]:
        b.put(f"sel_{r['model']}_mean", r["mean_recall10"], src)
        b.put(f"sel_{r['model']}_short", r["shortfall_to_best"], src)
        b.put(f"sel_{r['model']}_se", r["se_paired"], src)
    b.put("sel_main", zm["chosen_model"], src)
    dv = pd.read_csv(paths6.MANIFESTS / "development_registry_v6.csv")
    b.put("devfit_n", int(len(dv)), paths6.MANIFESTS / "development_registry_v6.csv")
    b.put("devfit_kappa0_share", float((dv["selected"] == "kappa0").sum()
                                       / max(dv["selected"].notna().sum(), 1)),
          paths6.MANIFESTS / "development_registry_v6.csv")
    zs = J(paths6.TABLES_OUT / "z_summary.json")
    src = paths6.TABLES_OUT / "z_summary.json"
    for r in zs["endpoints"]:
        m = r["method"]
        for k in ("recall10_end", "oracle_recall10", "eta10", "coverage", "zero_coverage",
                  "ndcg10_end", "mrr", "recall50_end", "ranking_loss10", "capacity_loss10",
                  "share_not_retrieved", "share_in_pool_missed", "share_recovered",
                  "recall10_norm", "n_diseases", "hit10", "hit20", "hit50", "recall20_end"):
            if k in r:
                b.put(f"z_{m}_{k}", r[k], src)
    for r in zs["comparisons"]:
        c = r["comparison"].replace("-", "")
        for k in ("delta", "ci_low", "ci_high", "holm_p", "permutation_p", "n_diseases",
                  "frac_benefit", "frac_harm", "mean_a", "mean_b"):
            b.put(f"cmp_{c}_{k}", r[k], src)
    st = pd.read_csv(paths6.TABLES_OUT / "z_strata.csv")
    src = paths6.TABLES_OUT / "z_strata.csv"
    for r in st.itertuples(index=False):
        if r.stratum_type in ("route", "support_stratum"):
            b.put(f"strat_{r.method}_{r.stratum}_R", r.recall10_end, src)
            b.put(f"strat_{r.method}_{r.stratum}_U", r.oracle_recall10, src)
            b.put(f"strat_n_{r.stratum}", r.n, src)
    ga = J(paths6.SCORING / "gene_block_ablation.json")
    src = paths6.SCORING / "gene_block_ablation.json"
    for k, v in ga["means"].items():
        b.put(f"gab_{k}", v, src)
    for r in ga["increments"]:
        b.put(f"gab_inc_{r['added_block']}", r["delta"], src)
        b.put(f"gab_inc_{r['added_block']}_lo", r["ci_low"], src)
        b.put(f"gab_inc_{r['added_block']}_hi", r["ci_high"], src)
    pm = pd.read_csv(paths6.PERMUTATION / "v6_permutations_summary.csv")
    src = paths6.PERMUTATION / "v6_permutations_summary.csv"
    for r in pm.itertuples(index=False):
        b.put(f"pv6_{r.scorer}_{r.test}_{r.scheme}_{r.subset}_delta", r.delta_perm_minus_orig, src)
        b.put(f"pv6_{r.scorer}_{r.test}_{r.scheme}_{r.subset}_orig", r.recall10_orig, src)
        b.put(f"pv6_{r.scorer}_{r.test}_{r.scheme}_{r.subset}_perm", r.recall10_perm, src)
        if r.mean_spearman == r.mean_spearman:
            b.put(f"pv6_{r.scorer}_{r.test}_{r.scheme}_{r.subset}_rho", r.mean_spearman, src)
            b.put(f"pv6_{r.scorer}_{r.test}_{r.scheme}_{r.subset}_top10", r.mean_top10_overlap, src)
        b.put(f"pv6_{r.scorer}_{r.test}_{r.scheme}_{r.subset}_n", r.n_diseases, src)
    kd = J(paths6.SCORING / "kdgene_coldval_f0.json")
    src = paths6.SCORING / "kdgene_coldval_f0.json"
    b.put("kd_epochs", kd["epochs_run"], src)
    b.put("kd_best_epoch", kd["best_epoch"], src)
    b.put("kd_best_val", kd["best_cold_val_recall10"], src)
    b.put("kd_test_R", kd["test_metrics"]["recall10_end"], src)
    b.put("kd_seconds", kd["seconds"], src)
    b.put("kd_loss_first", kd["history"][0]["loss"], src)
    b.put("kd_loss_best", kd["history"][kd["best_epoch"]]["loss"], src)
    b.put("kd_loss_last", kd["history"][-1]["loss"], src)
    b.put("kd_val_last", kd["history"][-1]["val_recall10"], src)
    b.put("kd_conn_pheno", kd["connectivity"]["val_with_phenotype_edge"], src)
    b.put("kd_conn_onto", kd["connectivity"]["val_with_ontology_edge"], src)
    # ---- support -------------------------------------------------------------------
    ss = J(paths6.TABLES_OUT / "s_summary.json")
    src = paths6.TABLES_OUT / "s_summary.json"
    b.put("s_alpha", ss["selected"]["alpha"], src)
    b.put("s_nu", ss["selected"]["nu"], src)
    b.put("s_omega", ss["selected"]["omega"], src)
    b.put("s_dev_cases", ss["selected"]["dev_cases"], src)
    for k, v in ss["selected"]["dev_recall10_k5"].items():
        b.put(f"s_dev_{k}", v, src)
    for key, r in ss["registered"].items():
        c = key.replace("-", "")
        for k in ("delta", "ci_low", "ci_high", "holm_p", "n_diseases", "mean_a", "mean_b",
                  "frac_benefit", "frac_harm", "frac_tie"):
            b.put(f"s_{c}_{k}", r[k], src)
    for key, r in {**ss["secondary"], **ss["regimes"]}.items():
        for k in ("delta", "ci_low", "ci_high", "n_diseases", "mean_a", "mean_b", "permutation_p"):
            if k in r:
                b.put(f"s2_{key}_{k}", r[k], src)
    for kk, r in ss["four_layer"].items():
        for k, v in r.items():
            b.put(f"fl_{kk}_{k}", v, src)
    b.put("s_k1_exact", ss["k1_invariance_exact"], src)
    b.put("s_n", ss["cohort"]["n_diseases"], src)
    for k, v in ss["cohort"]["five_shot_by_role"].items():
        b.put(f"s_role_{k}", v, src)
    for k, v in ss["cohort"]["traceable_episode_share_by_k"].items():
        b.put(f"s_trace_k{k}", v, src)
    cur = pd.read_csv(paths6.TABLES_OUT / "s_dose_curves.csv")
    src = paths6.TABLES_OUT / "s_dose_curves.csv"
    for r in cur.itertuples(index=False):
        b.put(f"dose_{r.method}_{r.regime}_k{r.k}", r.recall10, src)
        b.put(f"dosecov_{r.method}_{r.regime}_k{r.k}", r.coverage, src)
    rw = J(paths6.SUPPORT / "rewiring_control.json")
    src = paths6.SUPPORT / "rewiring_control.json"
    b.put("rw_unchanged", rw["rewiring"]["fraction_edges_unchanged"], src)
    b.put("rw_degree_ok", rw["rewiring"]["degree_sequence_identical"], src)
    b.put("rw_comp_before", rw["rewiring"]["components_excluding_isolated_before"], src)
    b.put("rw_comp_after", rw["rewiring"]["components_excluding_isolated_after"], src)
    for k, v in rw["comparisons"].items():
        for kk in ("real", "rewired", "rewired_minus_real", "ci_low", "ci_high"):
            b.put(f"rw_{k}_{kk}", v[kk], src)
    pl = J(paths6.SUPPORT / "propagation_log.json")
    b.put("rwr_failures", pl.get("total_failures", 0), paths6.SUPPORT / "propagation_log.json")
    # ---- external / temporal / expression -------------------------------------------------
    ext = pd.read_csv(paths6.EXTERNAL_RES / "external_v6_layers.csv")
    src = paths6.EXTERNAL_RES / "external_v6_layers.csv"
    for r in ext.itertuples(index=False):
        b.put(f"ext_{r.layer}_{r.scorer}_R", r.recall10_end, src)
        b.put(f"ext_{r.layer}_{r.scorer}_lo", r.ci_low, src)
        b.put(f"ext_{r.layer}_{r.scorer}_hi", r.ci_high, src)
        b.put(f"ext_{r.layer}_n", r.n_diseases, src)
        b.put(f"ext_{r.layer}_pairs", r.n_pairs, src)
        b.put(f"ext_{r.layer}_cov", r.coverage, src)
    el = J(paths6.EXTERNAL_RES / "external_v6_lightgbm.json")
    src = paths6.EXTERNAL_RES / "external_v6_lightgbm.json"
    for r in el["rows"]:
        b.put(f"extl_{r['layer']}_R", r["recall10_end"], src)
        b.put(f"extl_{r['layer']}_lo", r["ci_low"], src)
        b.put(f"extl_{r['layer']}_hi", r["ci_high"], src)
    cc = J(paths6.EXTERNAL_RES / "clinvar_citation_chain_summary.json")
    src = paths6.EXTERNAL_RES / "clinvar_citation_chain_summary.json"
    b.put("cv_pairs", cc["clinvar_external_pairs"], src)
    b.put("cv_unknown_rate", cc["unknown_rate"], src)
    for k, v in cc["layer3_v6_counts"].items():
        b.put(f"cv_{k}", v, src)
    b.put("cv_variants", cc["variants_kept"], src)
    b.put("cv_variants_cited", cc["variants_with_pubmed_citation"], src)
    b.put("cv_variants_single", cc["variants_single_mapped_condition"], src)
    b.put("cv_variants_multi", cc["variants_multi_mapped_condition"], src)
    ta = J(paths6.EXTERNAL_RES / "temporal_input_history_audit.json")
    src = paths6.EXTERNAL_RES / "temporal_input_history_audit.json"
    b.put("t_report_created", ta["wayback"].get("report_created_line", ""), src)
    b.put("t_all_hist", ta["all_model_inputs_no_later_than_cutoff_if_mondo_dropped"], src)
    ts = J(paths6.EXTERNAL_RES / "temporal_x2_support_summary.json")
    src = paths6.EXTERNAL_RES / "temporal_x2_support_summary.json"
    b.put("t_n", ts["n_diseases"], src)
    for r in ts["methods"]:
        for k in ("recall10", "recall50", "recall100"):
            b.put(f"t_{r['method']}_{k}", r[k], src)
    for k, v in ts["comparisons_recall50"].items():
        for kk in ("delta", "ci_low", "ci_high", "permutation_p", "n"):
            b.put(f"tc_{k}_{kk}", v[kk], src)
    ex = J(paths6.EXTERNAL_RES / "expression_v6.json")
    src = paths6.EXTERNAL_RES / "expression_v6.json"
    for s in ex["series"]:
        acc = s.get("accession")
        for cname, entry in (s.get("contrasts") or {}).items():
            for setn, v in entry.items():
                if isinstance(v, dict) and "enrichment" in v:
                    b.put(f"ex_{acc}_{cname}_{setn}_auc", v["enrichment"], src)
                    b.put(f"ex_{acc}_{cname}_{setn}_p", v["p_value"], src)
        b.put(f"ex_{acc}_pairs", s.get("n_pairs", 0), src)
    for s in ex["sets"]:
        b.put(f"exset_{s['disease']}_{s['set']}_known", s.get("n_known_positive"), src)
    # ---- resources ------------------------------------------------------------------------
    led = collect_ledger()
    src = paths6.MANIFESTS / "run_ledger_v6.csv"
    b.put("n_runs_v6", int(len(led)), src)
    b.put("n_runs_done", int((led["status"] == "done").sum()), src)
    b.put("n_runs_failed", int((led["status"] == "failed").sum()), src)
    b.put("n_hash_ok", int(led["hash_ok"].sum()), src)
    fz = led[(led["stage"] == "formal") & (led["status"] == "done")]
    b.put("n_formal", int(len(fz)), src)
    b.put("n_formal_learned", int(fz["seed"].astype(str).str.isdigit().sum()), src)
    for m, sub in fz.groupby("method"):
        b.put(f"res_{m}_median_s", float(sub["seconds"].median()), src)
        b.put(f"res_{m}_rss_gib", float(sub["peak_rss_mib"].max() / 1024), src)
        b.put(f"res_{m}_gpu_mib", float(sub["peak_gpu_reserved_mib"].max()), src)
    b.put("n_ablation_fits", int(((led["stage"] == "ablation") & (led["status"] == "done")).sum()), src)
    for kind in ("attention", "bilinear"):
        fits = [J(p) for p in paths6.RUNS.glob(f"V6Z__{kind}__f*__s*/result.json")]
        b.put(f"formal_{kind}_fits", len(fits), paths6.RUNS)
        b.put(f"formal_{kind}_residual_kept",
              sum(1 for d in fits if d.get("fit_info", {}).get("selected") == "residual"), paths6.RUNS)
        ks = [d.get("fit_info", {}).get("kappa") for d in fits if d.get("fit_info", {}).get("selected") == "residual"]
        if ks:
            b.put(f"formal_{kind}_kappa_min", min(ks), paths6.RUNS)
            b.put(f"formal_{kind}_kappa_max", max(ks), paths6.RUNS)
    fa = load_manifest("failed_attempts_v6.json", {}) or {}
    b.put("n_failed_attempts", len(fa.get("attempts", [])), paths6.MANIFESTS / "failed_attempts_v6.json")
    pd.DataFrame(b.rows).to_csv(paths6.REPORTS / "manuscript_numbers_v6.csv", index=False,
                                encoding="utf-8-sig")
    dump_json(paths6.REPORTS / "manuscript_numbers_v6.json", b.v)
    return b.v


if __name__ == "__main__":
    paths6.ensure()
    v = collect()
    print(len(v), "numbers collected")

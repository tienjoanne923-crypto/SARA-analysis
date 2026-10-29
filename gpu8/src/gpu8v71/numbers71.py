"""Collect every number the V7.1 manuscript cites into one keyed table.

``reports_v71/manuscript_numbers_v71.json`` / ``.csv`` hold key -> value -> source
file; the manuscript generator reads numbers only through this table and records
which keys it used, so every cited number can be traced to a result file.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths71
from .common71 import dump_json, load_manifest

SLUG = {"F7 (V6 frozen fusion)": "f7", "V6 LightGBM (archived V6 main)": "v6lgb",
        "Channel-only LightGBM (strong control)": "chlgb", "Same-resource fixed fusion": "srff",
        "Multimodal core alone": "core", "Late fusion, fixed": "lffix", "Late fusion, LightGBM": "lflgb",
        "core (full alignment) alone": "e2_core", "core, lambda_A flipped, alone": "e2_alt",
        "linear two-tower alone": "e2_tt", "same-resource fixed fusion": "e2_srff",
        "late fusion LGB, full core": "e2_lf", "late fusion LGB, lambda_A flipped": "e2_lfalt",
        "late fusion LGB, two-tower": "e2_lftt", "raw concatenation LGB (same pool)": "e2_raw",
        "V6 500 fixed: channel-only LGB": "e6_v6ch", "V6 500 fixed: LGB + core features": "e6_v6core",
        "shared pool: F7": "e6_poolf7", "shared pool: channel-only LGB": "e6_poolch",
        "shared pool: late fusion LGB": "e6_poollf", "shared pool: mask-only LGB": "e6_mask",
        "core (s42) alone": "e8_core", "core + image": "e8_img",
        "core + localisation labels (image control)": "e8_loc",
        "core + cell-line-centred image": "e8_cent", "core + mouse (IMPC)": "e8_mouse",
        "core, background controls (sensitivity)": "e8_bg",
        "old resources / old method": "e4_oo", "new resources / old method": "e4_no",
        "old resources / new method": "e4_on", "new resources / new method": "e4_nn",
        "V6 LightGBM": "v6lgb", "Channel-only LightGBM": "chlgb",
        "Late fusion LightGBM": "lflgb", "Strict gene-cold core alone": "strictcore",
        "Strict gene-cold late fusion": "strictlf"}


class Collector:
    def __init__(self):
        self.v: dict = {}
        self.src: dict = {}

    def put(self, k, x, src):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return
        self.v[k] = float(x) if isinstance(x, (int, float, np.floating, np.integer)) else x
        self.src[k] = src


def jl(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def collect() -> dict:
    c = Collector()
    A = paths71.AUDIT
    T = paths71.TABLES_OUT
    # ---- resources ----------------------------------------------------------------
    sq = jl(A / "sequences_v71_report.json")
    for k in ("n_genes_background", "n_with_sequence", "n_protein_coding_background",
              "n_protein_coding_with_sequence", "n_longer_than_1022", "n_exceptions"):
        c.put(f"seq_{k}", sq.get(k), "audit/sequences_v71_report.json")
    for k, v in sq.get("by_reason", {}).items():
        c.put(f"seq_reason_{k}", v, "audit/sequences_v71_report.json")
    enc = jl(A / "encoders_v71_report.json")
    for part in ("disease", "hpo", "go"):
        c.put(f"enc_{part}_n", enc.get(part, {}).get("n"), "audit/encoders_v71_report.json")
        c.put(f"enc_{part}_trunc", enc.get(part, {}).get("n_truncated_at_128"), "audit/encoders_v71_report.json")
    c.put("enc_esm_seconds", enc.get("esm", {}).get("seconds"), "audit/encoders_v71_report.json")
    c.put("enc_esm_multiwindow", enc.get("esm", {}).get("n_multi_window"), "audit/encoders_v71_report.json")
    fn = jl(A / "function_v71_report.json")
    c.put("fun_genes", fn.get("g_function", {}).get("genes_with_function"), "audit/function_v71_report.json")
    c.put("fun_anchor_terms", fn.get("anchors", {}).get("n_terms"), "audit/function_v71_report.json")
    c.put("fun_anchor_genes", fn.get("anchors", {}).get("genes_with_anchor"), "audit/function_v71_report.json")
    c.put("fun_anchor_median", fn.get("anchors", {}).get("median_anchors_per_gene"), "audit/function_v71_report.json")
    c.put("fun_direct_pairs", fn.get("gaf", {}).get("direct_pairs"), "audit/function_v71_report.json")
    ex = jl(A / "expression_v71_report.json")
    c.put("expr_genes", ex.get("hpa", {}).get("genes_measured"), "audit/expression_v71_report.json")
    c.put("expr_zero_cells", ex.get("hpa", {}).get("cells_zero_but_observed"), "audit/expression_v71_report.json")
    c.put("expr_gtex_hpa_tissues", ex.get("gtex_arm", {}).get("hpa_tissues_with_gtex"), "audit/expression_v71_report.json")
    dt = ex.get("disease_tissue", {})
    c.put("tis_diseases", dt.get("with_tissue"), "audit/expression_v71_report.json")
    c.put("tis_labelled", dt.get("labelled_with_tissue"), "audit/expression_v71_report.json")
    c.put("tis_labelled_n", dt.get("labelled_n"), "audit/expression_v71_report.json")
    c.put("tis_only_broad", dt.get("only_broad_rules"), "audit/expression_v71_report.json")
    from .expression71 import ALIAS_RULES
    c.put("tis_n_rules", len(ALIAS_RULES), "expression71.ALIAS_RULES")
    cov = jl(A / "featurepack_coverage_v71.json")
    for k, v in cov.get("genes", {}).items():
        c.put(f"cov_gene_{k}", v.get("share_all"), "audit/featurepack_coverage_v71.json")
        c.put(f"cov_gene_{k}_pos", v.get("share_positive_genes"), "audit/featurepack_coverage_v71.json")
        c.put(f"cov_gene_{k}_n", v.get("all"), "audit/featurepack_coverage_v71.json")
    for k, v in cov.get("diseases", {}).items():
        c.put(f"cov_dis_{k}", v.get("share_labelled"), "audit/featurepack_coverage_v71.json")
    c.put("cov_gene_none", cov.get("gene_modality_patterns", {}).get("none"), "audit/featurepack_coverage_v71.json")
    c.put("cov_gene_all3", cov.get("gene_modality_patterns", {}).get("SFE"), "audit/featurepack_coverage_v71.json")
    im = jl(A / "image_features_v71_report.json")
    for k in ("rows_read", "genes_with_image", "cell_lines", "multi_or_empty_gene_images",
              "genes_with_location"):
        c.put(f"img_{k}", im.get(k), "audit/image_features_v71_report.json")
    mo = load_manifest("mouse_channel_v71.json") or {}
    for k, v in mo.get("checks", {}).items():
        if isinstance(v, (int, float)):
            c.put(f"mouse_{k}", v, "manifests_v71/mouse_channel_v71.json")
    pr = jl(A / "cuda_probe_v71.json")
    if pr:
        c.put("probe_update_s", float(np.mean(pr["update_seconds"][1:])), "audit/cuda_probe_v71.json")
        c.put("probe_rss", pr["peak_rss_mib"], "audit/cuda_probe_v71.json")
        c.put("probe_gpu", pr["gpu_peak_allocated_mib"], "audit/cuda_probe_v71.json")
        c.put("probe_params_img", pr["trainable_parameters"], "audit/cuda_probe_v71.json")
    # ---- development selection ------------------------------------------------------
    sel = load_manifest("core_dev_selection_v71.json") or {}
    for g in sel.get("grid", []):
        c.put(f"dev_{g['name']}", g["dev_recall10"], "manifests_v71/core_dev_selection_v71.json")
    if sel:
        c.put("dev_sel_rank", sel["selected"]["rank"], "manifests_v71/core_dev_selection_v71.json")
        c.put("dev_sel_lambda", sel["selected"]["lambda_A"], "manifests_v71/core_dev_selection_v71.json")
        c.put("dev_sel_recall", sel["modality_check"]["normal_dev_recall10"], "manifests_v71/core_dev_selection_v71.json")
        for m in ("S", "F", "E"):
            mc = sel["modality_check"][m]
            c.put(f"dev_perm_{m}", mc["perm_mean_drop"], "manifests_v71/core_dev_selection_v71.json")
            c.put(f"dev_perm_{m}_se", mc["perm_se"], "manifests_v71/core_dev_selection_v71.json")
            c.put(f"dev_off_{m}", mc["shutdown_drop"], "manifests_v71/core_dev_selection_v71.json")
        ad = sel["admission"]
        for k in ("dev_positives", "in_v6_500", "added_by_core_top500", "union_coverage_gain"):
            c.put(f"adm_{k}", ad[k], "manifests_v71/core_dev_selection_v71.json")
    reg = load_manifest("protocol_v71_registered.json") or {}
    if reg:
        c.put("conf_pairs", reg["confirmation_set"]["n_pairs"], "manifests_v71/protocol_v71_registered.json")
        c.put("conf_diseases", reg["confirmation_set"]["n_diseases"], "manifests_v71/protocol_v71_registered.json")
        for k, v in reg["confirmation_set"]["by_source"].items():
            c.put(f"conf_{k}", v, "manifests_v71/protocol_v71_registered.json")
    bt = load_manifest("fixed_late_beta_core.json") or {}
    c.put("lf_beta", bt.get("beta"), "manifests_v71/fixed_late_beta_core.json")
    sr = load_manifest("srff_v71.json") or {}
    for k, v in (sr.get("weights") or {}).items():
        c.put(f"srff_w_{k}", v, "manifests_v71/srff_v71.json")
    gm = load_manifest("small_support_gamma_v71.json") or {}
    c.put("ss_gamma", gm.get("gamma"), "manifests_v71/small_support_gamma_v71.json")
    # ---- tables ---------------------------------------------------------------------
    for fname, pref in (("e1_main.csv", "e1"), ("e2_same_data.csv", "e2t"), ("e4_resource_method.csv", "e4t"),
                        ("e6_attribution.csv", "e6t"), ("e8_optional.csv", "e8t")):
        p = T / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for r in df.to_dict(orient="records"):
            s = SLUG.get(r["method"], r["method"])
            for m in ("recall10_end", "hit10", "recall50_end", "ndcg10_end", "coverage",
                      "coverage_final500", "oracle_recall10", "zero_coverage", "n_diseases"):
                if m in r and r[m] == r[m]:
                    c.put(f"{pref}_{s}_{m}", r[m], f"results_v71/tables/{fname}")
    v = c.v
    if all(k in v for k in ("e4t_e4_oo_recall10_end", "e4t_e4_no_recall10_end", "e4t_e4_on_recall10_end",
                            "e4t_e4_nn_recall10_end")):
        c.put("e4_method_gain_oldres", v["e4t_e4_on_recall10_end"] - v["e4t_e4_oo_recall10_end"], "derived from e4 table")
        c.put("e4_method_gain_newres", v["e4t_e4_nn_recall10_end"] - v["e4t_e4_no_recall10_end"], "derived from e4 table")
        c.put("e4_resource_gain_oldmethod", v["e4t_e4_no_recall10_end"] - v["e4t_e4_oo_recall10_end"], "derived from e4 table")
        c.put("e4_resource_gain_newmethod", v["e4t_e4_nn_recall10_end"] - v["e4t_e4_on_recall10_end"], "derived from e4 table")
    if "e6t_e6_v6core_recall10_end" in v and "e6t_e6_v6ch_recall10_end" in v:
        c.put("e6_inlist_gain", v["e6t_e6_v6core_recall10_end"] - v["e6t_e6_v6ch_recall10_end"], "derived from e6 table")
    if "e6t_e6_poollf_recall10_end" in v and "e6t_e6_poolch_recall10_end" in v:
        c.put("e6_pool_gain", v["e6t_e6_poollf_recall10_end"] - v["e6t_e6_poolch_recall10_end"], "derived from e6 table")
    p = T / "e1_contrasts.csv"
    if p.exists():
        for r in pd.read_csv(p).to_dict(orient="records"):
            if r.get("status") == "missing":
                continue
            k = f"c1_{SLUG.get(r['a'], r['a'])}_vs_{SLUG.get(r['b'], r['b'])}_{r['metric']}"
            for f in ("delta", "ci_low", "ci_high", "share_a_better", "share_b_better"):
                c.put(f"{k}_{f}", r.get(f), "results_v71/tables/e1_contrasts.csv")
    res = jl(T / "v71_results.json")
    for sec in ("E2", "E8"):
        for r in (res.get(sec, {}) or {}).get("contrasts", []):
            if r.get("status") == "missing":
                continue
            k = f"c{sec[1]}_{SLUG.get(r['a'], r['a'])}_vs_{SLUG.get(r['b'], r['b'])}"
            for f in ("delta", "ci_low", "ci_high"):
                c.put(f"{k}_{f}", r.get(f), "results_v71/tables/v71_results.json")
    p = T / "e3_visibility.csv"
    if p.exists():
        for r in pd.read_csv(p).to_dict(orient="records"):
            s = SLUG.get(r["method"], r["method"])
            for m in ("in_list", "in500", "top10", "share_of_positives"):
                c.put(f"e3_{s}_{r['class']}_{m}", r[m], "results_v71/tables/e3_visibility.csv")
    p = T / "e5_behaviour.csv"
    if p.exists():
        for r in pd.read_csv(p).to_dict(orient="records"):
            for m in ("lf_recall10", "core_recall10", "lf_drop", "core_drop", "lf_drop_ci_low",
                      "lf_drop_ci_high", "core_drop_ci_low", "core_drop_ci_high"):
                if m in r:
                    c.put(f"e5_{r['perturbation']}_{r['modality']}_{m}", r[m], "results_v71/tables/e5_behaviour.csv")
    fl = jl(T / "e6_hit_flow.json")
    for k, v in fl.items():
        for kk, vv in (v or {}).items():
            if isinstance(vv, (int, float)):
                c.put(f"flow_{k}_{kk}", vv, "results_v71/tables/e6_hit_flow.json")
    p = T / "e7_confirmation.csv"
    if p.exists():
        for r in pd.read_csv(p).to_dict(orient="records"):
            s = SLUG.get(r["method"], r["method"])
            for m in ("recall10", "hit10", "recall50", "coverage_in_list", "n_diseases"):
                c.put(f"e7_{s}_{m}", r[m], "results_v71/tables/e7_confirmation.csv")
    cc = paths71.EXT_RES / "confirmation_contrasts_v71.json"
    for r in (jl(cc) if cc.exists() else []):
        k = f"c7_{SLUG.get(r['a'], r['a'])}_vs_{SLUG.get(r['b'], r['b'])}"
        for f in ("delta_recall10", "ci_low", "ci_high"):
            c.put(f"{k}_{f}", r.get(f), "results_v71/external/confirmation_contrasts_v71.json")
    p = T / "e7_small_support.csv"
    if p.exists():
        for r in pd.read_csv(p).to_dict(orient="records"):
            for m in ("recall10", "recall50", "n_diseases"):
                c.put(f"ss_{r['method']}_{m}", r[m], "results_v71/tables/e7_small_support.csv")
    p = paths71.BEHAVIOR / "e8_permutation_test.csv"
    if p.exists():
        df = pd.read_csv(p).groupby("variant")[["normal", "permuted", "drop"]].mean()
        for v, r in df.iterrows():
            for m in ("normal", "permuted", "drop"):
                c.put(f"e8p_{v}_{m}", r[m], "results_v71/behavior/e8_permutation_test.csv")
    sens = jl(paths71.BEHAVIOR / "sensitivity_text_go_v71.json")
    if sens:
        for k, v in sens.get("mean", {}).items():
            c.put(f"sens_{k}", v, "results_v71/behavior/sensitivity_text_go_v71.json")
        c.put("sens_go2023_genes", sens.get("genes_with_function_2023"), "results_v71/behavior/sensitivity_text_go_v71.json")
    chk = jl(paths71.BEHAVIOR / "chunk_invariance_v71.json")
    if chk:
        c.put("chunk_invariance_diff", f"{chk['max_abs_difference']:.1e}", "results_v71/behavior/chunk_invariance_v71.json")
    # ---- ledger ---------------------------------------------------------------------------
    from .ledger71 import collect as ledger
    L = ledger()
    if not L.empty:
        c.put("ledger_runs", len(L), "manifests_v71/run_ledger_v71.csv")
        c.put("ledger_done", int((L["status"] == "done").sum()), "manifests_v71/run_ledger_v71.csv")
        c.put("ledger_failed", int((L["status"] == "failed").sum()), "manifests_v71/run_ledger_v71.csv")
        c.put("ledger_hash_ok", int(L["hash_ok"].sum()), "manifests_v71/run_ledger_v71.csv")
        c.put("ledger_chain_fits", int(L["run_id"].str.startswith("V71CHAIN__").sum()), "manifests_v71/run_ledger_v71.csv")
        c.put("ledger_main_chain", int(L["run_id"].str.startswith("V71CHAIN__core__").sum()), "manifests_v71/run_ledger_v71.csv")
        z = L[L["run_id"].str.startswith("V71Z__")]
        if len(z):
            c.put("ledger_first_test_utc", str(z["started_utc"].min()), "manifests_v71/run_ledger_v71.csv")
        ch = L[L["run_id"].str.startswith("V71CHAIN__core__")]
        if len(ch):
            c.put("chain_mean_seconds", float(ch["seconds"].mean()), "manifests_v71/run_ledger_v71.csv")
            c.put("chain_peak_gpu", float(ch["peak_gpu_allocated_mib"].max()), "manifests_v71/run_ledger_v71.csv")
            c.put("chain_peak_rss", float(ch["peak_rss_mib"].max()), "manifests_v71/run_ledger_v71.csv")
    if reg:
        c.put("registered_utc", reg.get("registered_utc"), "manifests_v71/protocol_v71_registered.json")
    # parameters of the selected core
    import glob
    fits = sorted(glob.glob(str(paths71.RUNS / "V71CHAIN__core__f0_outer__s42" / "result.json")))
    if fits:
        r = json.loads(open(fits[0], encoding="utf-8").read())
        c.put("core_params_selected", r["fit"]["n_params"], "runs_v71/V71CHAIN__core__f0_outer__s42/result.json")
    out = {"values": c.v, "sources": c.src}
    dump_json(paths71.REPORTS / "manuscript_numbers_v71.json", out)
    pd.DataFrame({"key": list(c.v), "value": list(c.v.values()),
                  "source": [c.src[k] for k in c.v]}).to_csv(
        paths71.REPORTS / "manuscript_numbers_v71.csv", index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    paths71.ensure()
    o = collect()
    print(len(o["values"]), "numbers")

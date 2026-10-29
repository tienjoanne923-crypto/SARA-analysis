"""Aggregate the V6 support line (task S) and run the registered S-family tests.

Draws are averaged inside a disease before anything else.  Differences are
always (named first) - (named second).  The k = 1 invariance of group seeds is a
hard check: any non-zero difference at k = 1 aborts the report.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8.stats import family_bootstrap, holm, paired_permutation

from . import paths6
from .common import World6, dump_json, load_manifest, now_utc


def paired(a: pd.Series, b: pd.Series, fam: dict, draws: int = 10000) -> dict:
    common = sorted(set(a.index) & set(b.index))
    av, bv = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    d = av - bv
    f = np.array([fam.get(x, x) for x in common])
    lo, hi, _ = family_bootstrap(d, f, draws=draws)
    p = paired_permutation(d, f, draws=draws)
    return {"n_diseases": len(common), "n_families": int(len(set(f))), "mean_a": float(av.mean()),
            "mean_b": float(bv.mean()), "delta": float(d.mean()), "ci_low": lo, "ci_high": hi,
            "permutation_p": p, "frac_benefit": float((d > 1e-12).mean()),
            "frac_tie": float((np.abs(d) <= 1e-12).mean()), "frac_harm": float((d < -1e-12).mean())}


def run() -> dict:
    w = World6(load_heavy=False)
    fam = w.family_of
    sel = load_manifest("support_v6_frozen.json")
    ep = pd.read_parquet(paths6.SUPPORT / "support_test_episodes.parquet")
    dg = pd.read_parquet(paths6.SUPPORT / "support_test_seed_diagnostics.parquet")
    nu_sel = float(sel["nu"])
    main_tag = f"test_nu{nu_sel:g}"
    ep_main = ep[ep["tag"] == main_tag]
    meta = ep_main[ep_main["regime"] == "meta"]
    ep_m = ep_main[ep_main["regime"] != "meta"]
    agg = (ep_m.groupby(["disease_id", "method", "regime", "k"], as_index=False)
           [["recall10", "recall50", "mrr", "coverage"]].mean())
    curve = (agg.groupby(["method", "regime", "k"])
             .agg(n=("recall10", "size"), recall10=("recall10", "mean"),
                  recall50=("recall50", "mean"), coverage=("coverage", "mean")).reset_index())

    def series(method, regime, k):
        s = agg[(agg["method"] == method) & (agg["regime"] == regime) & (agg["k"] == k)]
        return s.set_index("disease_id")["recall10"]

    omega = f"joint_w{sel['omega']:g}"
    sem0 = series("sem", "S_fixed", 0)
    res = {}
    res["S-A"] = {"a": "rwr_equal k=5 (S-fixed)", "b": "sem k=0 (S-fixed)",
                  **paired(series("rwr_equal", "S_fixed", 5), sem0, fam)}
    res["S-B"] = {"a": f"{omega} k=5 (S-fixed)", "b": "rwr_equal k=5 (S-fixed)",
                  **paired(series(omega, "S_fixed", 5), series("rwr_equal", "S_fixed", 5), fam)}
    # mechanism on traceable episodes only
    tr = meta[(meta["method"] == "traceable") & (meta["recall10"] == 1.0)]
    trace_keys = set(zip(tr["disease_id"], tr["k"], tr["draw"]))
    ep_tr = ep_m[[(d, k, r) in trace_keys for d, k, r in zip(ep_m["disease_id"], ep_m["k"],
                                                               ep_m["draw"])]]
    agg_tr = (ep_tr.groupby(["disease_id", "method", "regime", "k"], as_index=False)
              [["recall10"]].mean())

    def s_tr(method, k):
        s = agg_tr[(agg_tr["method"] == method) & (agg_tr["regime"] == "S_fixed") & (agg_tr["k"] == k)]
        return s.set_index("disease_id")["recall10"]

    res["S-C"] = {"a": "rwr_group_eta1 k=5", "b": "rwr_equal k=5 (traceable episodes)",
                  **paired(s_tr("rwr_group_eta1", 5), s_tr("rwr_equal", 5), fam)}
    adj = holm({k: v["permutation_p"] for k, v in res.items()})
    for k in res:
        res[k]["holm_p"] = adj[k]
    # Gamma_grp and secondary k
    extra = {}
    for k in (1, 3, 5):
        g = paired(s_tr("rwr_group_eta1", k), s_tr("rwr_shuffle_eta1", k), fam)
        extra[f"Gamma_grp_k{k}"] = {"a": "true groups", "b": "shuffled groups (mean of 20)", **g}
        extra[f"Delta_seed_k{k}"] = {"a": "rwr_group_eta1", "b": "rwr_equal",
                                     **paired(s_tr("rwr_group_eta1", k), s_tr("rwr_equal", k), fam)}
        extra[f"degree_vs_equal_k{k}"] = paired(series("rwr_degree", "S_fixed", k),
                                                series("rwr_equal", "S_fixed", k), fam)
        for eta in (0.25, 0.5):
            extra[f"eta{eta}_vs_equal_k{k}"] = paired(s_tr(f"rwr_group_eta{eta:g}", k),
                                                      s_tr("rwr_equal", k), fam)
    k1 = extra["Delta_seed_k1"]
    invariance_ok = abs(k1["delta"]) < 1e-12 and k1["frac_benefit"] == 0 and k1["frac_harm"] == 0
    if not invariance_ok:
        raise RuntimeError(f"k = 1 invariance violated: {k1}")
    # regimes and the functional bridge
    reg = {}
    for k in (1, 3, 5):
        reg[f"adaptive_vs_fixed_{omega}_k{k}"] = paired(series(omega, "S_adaptive", k),
                                                        series(omega, "S_fixed", k), fam)
        reg[f"background_{omega}_vs_rwr_k{k}"] = paired(series(omega, "background", k),
                                                        series("rwr_equal", "background", k), fam)
    ppi = ep[(ep["tag"] == "test_nu0") & (ep["regime"] != "meta")]
    if nu_sel > 0 and len(ppi):
        pa = (ppi.groupby(["disease_id", "method", "regime", "k"], as_index=False)[["recall10"]]
              .mean())
        s_ppi = pa[(pa["method"] == "rwr_equal") & (pa["regime"] == "S_fixed") & (pa["k"] == 5)]
        reg["mix_vs_ppi_rwr_equal_k5"] = paired(series("rwr_equal", "S_fixed", 5),
                                                s_ppi.set_index("disease_id")["recall10"], fam)
    # four-layer diagnostics
    d5 = dg[dg["eta"] == 1.0]
    layers = {}
    for k, sub in d5.groupby("k"):
        layers[f"k{k}"] = {
            "episodes": int(len(sub)),
            "layer1_mean_l1_v_minus_u": float(sub["l1_v_minus_u"].mean()),
            "layer1_share_seed_changed": float((sub["l1_v_minus_u"] > 1e-12).mean()),
            "layer2_mean_l1_propagated": float(sub["l1_propagated"].mean()),
            "layer2_bound_always_holds": bool(sub["bound_holds"].all()),
            "layer3_mean_spearman": float(sub["spearman_candidates"].mean()),
            "layer3_share_top10_changed": float((sub["top10_changed"] > 0).mean()),
            "layer4_mean_delta_recall10": float(sub["delta_recall10"].mean()),
            "layer4_share_metric_changed": float((sub["delta_recall10"].abs() > 1e-12).mean()),
            "mean_n_sources": float(sub["n_sources"].mean()),
            "mean_entropy_v": float(sub["entropy_v"].mean()),
            "mean_entropy_u": float(sub["entropy_u"].mean())}
    tr_rate = meta[meta["method"] == "traceable"].groupby("k")["recall10"].mean().to_dict()
    cohort = {"n_diseases": int(ep_main["disease_id"].nunique()),
              "by_fold": ep_main.groupby("fold")["disease_id"].nunique().to_dict(),
              "traceable_episode_share_by_k": {int(k): float(v) for k, v in tr_rate.items()},
              "fallback_share_by_k": {int(k): float(1 - v) for k, v in tr_rate.items()}}
    av = pd.read_csv(paths6.AUDIT / "availability_per_disease.csv")
    five = av[av["eligible_5shot"] == 1]
    cohort["five_shot_eligible_all"] = int(len(five))
    cohort["five_shot_by_role"] = five["role"].value_counts().to_dict()
    out = paths6.TABLES_OUT
    curve.to_csv(out / "s_dose_curves.csv", index=False, encoding="utf-8-sig")
    summ = {"generated_utc": now_utc(), "selected": sel, "registered": res, "secondary": extra,
            "regimes": reg, "four_layer": layers, "k1_invariance_exact": invariance_ok,
            "cohort": cohort}
    dump_json(out / "s_summary.json", summ)
    return summ


if __name__ == "__main__":
    paths6.ensure()
    s = run()
    print(json.dumps({k: s[k] for k in ("registered", "four_layer", "cohort")}, indent=2,
                     default=str)[:8000])

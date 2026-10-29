"""Aggregate the V6 zero-support (task Z) results and the registered Z family tests.

Reads only ``runs_v6/V6Z__*`` result files (authoritative) and the V5 archive
per-disease files for the V5 references.  Seeds are averaged inside a disease
before any macro average or paired difference.  Every difference is written as
(method A) - (method B) with A named first.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8 import paths as p5
from gpu8.stats import compare_methods, holm

from . import paths6
from .audit_metrics import decompose_per_disease
from .common import World6, dump_json, load_manifest, now_utc
from .ledger6 import collect
from .metrics6 import macro, outcome_classes

V5_REF = {"v5_b3_cpu_fusion": "b3_cpu_fusion", "v5_b9_meta_adapt": "b9_ours",
          "v5_b8_meta_ridge": "b8_meta_ridge", "v5_b4_lightgbm": "b4_lightgbm"}


def v6_per_disease() -> pd.DataFrame:
    led = collect()
    led = led[(led["stage"] == "formal") & (led["status"] == "done")
              & led["run_id"].str.startswith("V6Z__")]
    frames = []
    for r in led.itertuples(index=False):
        p = paths6.RUNS / r.run_id / "per_disease.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["method"], df["fold"], df["seed"] = "v6_" + r.method, r.fold, str(r.seed)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def v5_per_disease() -> pd.DataFrame:
    reg = pd.read_csv(p5.MANIFEST_DIR / "run_registry_166.csv")
    e01 = reg[(reg["experiment"] == "E01") & (reg["status"] == "done")]
    frames = []
    for name, m in V5_REF.items():
        for rid in e01.loc[e01["method"] == m, "run_id"]:
            p = p5.RUNS_DIR / rid / "per_disease.csv"
            if p.exists():
                df = decompose_per_disease(pd.read_csv(p))
                df["method"] = name
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


COLS = ["recall10_end", "oracle_recall10", "eta10", "ranking_loss10", "capacity_loss10",
        "coverage", "zero_coverage", "ndcg10_end", "mrr", "recall20_end", "recall50_end",
        "recall10_norm", "hit10", "hit20", "hit50", "n_query", "n_query_in_pool", "h10"]


def run() -> dict:
    w = World6(load_heavy=False)
    v6 = v6_per_disease()
    v5 = v5_per_disease()
    cols = [c for c in COLS if c in v6.columns]
    both = pd.concat([v6[["method", "disease_id"] + cols],
                      v5[["method", "disease_id"] + [c for c in cols if c in v5.columns]]],
                     ignore_index=True)
    # outcome indicators are formed per run and then averaged over seeds, so a
    # disease hit by one of three seeds counts as 1/3 recovered, not as recovered
    both["_recovered"] = ((both["n_query_in_pool"] > 0) & (both["h10"] > 0)).astype(float)
    both["_not_retrieved"] = (both["n_query_in_pool"] == 0).astype(float)
    cols = cols + ["_recovered", "_not_retrieved"]
    by_d = both.groupby(["method", "disease_id"], as_index=False)[cols].mean()
    av = pd.read_csv(paths6.AUDIT / "availability_per_disease.csv").set_index("disease_id")
    by_d["route"] = by_d["disease_id"].map(av["route_zero_support"])
    by_d["support_stratum"] = by_d["disease_id"].map(av["support_stratum"])
    by_d["category"] = by_d["disease_id"].map(av["category_name"])
    rows = []
    for m, sub in by_d.groupby("method"):
        rec = {"method": m, "n_diseases": int(len(sub))}
        rec.update({c: float(np.nanmean(sub[c])) for c in cols
                    if c not in ("n_query", "h10", "_recovered", "_not_retrieved")})
        rec["n_eta_defined"] = int(sub["eta10"].notna().sum())
        rec["share_recovered"] = float(sub["_recovered"].mean())
        rec["share_not_retrieved"] = float(sub["_not_retrieved"].mean())
        rec["share_in_pool_missed"] = 1.0 - rec["share_recovered"] - rec["share_not_retrieved"]
        rows.append(rec)
    main = pd.DataFrame(rows).sort_values("recall10_end", ascending=False)
    strata = []
    for (m, key), sub in pd.concat([
            by_d.assign(stratum_type="route", stratum=by_d["route"]),
            by_d.assign(stratum_type="support_stratum", stratum=by_d["support_stratum"]),
            by_d.assign(stratum_type="category", stratum=by_d["category"])]).groupby(
            ["method", "stratum_type"]):
        for s, s2 in sub.groupby("stratum"):
            strata.append({"method": m, "stratum_type": key, "stratum": s, "n": int(len(s2)),
                           "recall10_end": float(s2["recall10_end"].mean()),
                           "oracle_recall10": float(s2["oracle_recall10"].mean()),
                           "coverage": float(s2["coverage"].mean())})
    strata = pd.DataFrame(strata)
    # ---- registered Z family ------------------------------------------------
    fam = w.family_of
    sel = load_manifest("devfit_v6_selection.json", {}).get("selection", {})
    main_scorer = load_manifest("z_main_scorer.json", {}).get("main", "v6_F7")
    learned = [k for k, v in sel.items() if v.get("beats_F_on_dev") and k != "lr"]
    best_learned = None
    if learned:
        cand = main[main["method"].isin([f"v6_{k}" for k in learned])]
        best_learned = cand.iloc[0]["method"] if len(cand) else None
    F = "v6_" + load_manifest("scorer_v6_frozen.json", {}).get("main", "F7")
    comps = {"Z-A": (main_scorer, "v5_b3_cpu_fusion", "recall10_end"),
             "Z-B": (best_learned or "v6_lightgbm", F, "recall10_end"),
             "Z-C": (F, "v5_b3_cpu_fusion", "oracle_recall10")}
    long = by_d.rename(columns={})
    res, praw = [], {}
    for key, (a, b, metric) in comps.items():
        r = compare_methods(long, metric, a, b, fam, draws=10000)
        if r is None:
            continue
        d = r.as_dict()
        d["comparison"] = key
        res.append(d)
        praw[key] = r.permutation_p
    adj = holm(praw)
    comp = pd.DataFrame(res)
    if not comp.empty:
        comp["holm_p"] = comp["comparison"].map(adj)
    out = paths6.TABLES_OUT
    main.to_csv(out / "z_main_endpoints.csv", index=False, encoding="utf-8-sig")
    strata.to_csv(out / "z_strata.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(out / "z_registered_comparisons.csv", index=False, encoding="utf-8-sig")
    by_d.to_parquet(out / "z_per_disease_collapsed.parquet", index=False)
    summ = {"generated_utc": now_utc(), "main_scorer": main_scorer, "best_learned": best_learned,
            "endpoints": main.to_dict(orient="records"),
            "comparisons": comp.to_dict(orient="records")}
    dump_json(out / "z_summary.json", summ)
    return summ


if __name__ == "__main__":
    paths6.ensure()
    s = run()
    print(json.dumps({"main_scorer": s["main_scorer"],
                      "endpoints": [{k: r[k] for k in ("method", "n_diseases", "recall10_end",
                                                        "oracle_recall10", "coverage")}
                                    for r in s["endpoints"]],
                      "comparisons": [{k: r.get(k) for k in ("comparison", "method_a", "method_b",
                                                              "metric", "delta", "ci_low", "ci_high",
                                                              "holm_p")} for r in s["comparisons"]]},
                     indent=2, default=str))

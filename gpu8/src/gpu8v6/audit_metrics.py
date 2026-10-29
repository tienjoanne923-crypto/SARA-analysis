"""V6 section 3.2, 3.3 and 2.5 — zero-training audit of the V5 outputs.

Three products, all computed from the files the V5 runs already wrote:

1. the oracle decomposition R = U * eta per disease and per method, with the
   ranking loss U - R and the retrieval capacity loss min(K, n)/n - U, and the
   explicit statement that the 12.0% "recovered" outcome share is a disease
   proportion, not a macro recall;
2. a unified ablation table with Delta_abl = M_abl - M_full, the raw full and
   ablated values, the number of diseases, the candidate hash of each run and
   the seeds, so the sign convention can be read off the table itself;
3. the candidate-hash comparison that shows which ablations also changed the
   candidate lists (and therefore confound scoring with retrieval).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gpu8 import paths as p5
from gpu8.stats import family_bootstrap, paired_permutation

from . import paths6
from .common import candidate_hash, dump_json, now_utc
from .metrics6 import check_decomposition, macro, outcome_classes

K = 10


def _read_registry() -> pd.DataFrame:
    return pd.read_csv(p5.MANIFEST_DIR / "run_registry_166.csv")


def decompose_per_disease(per: pd.DataFrame, k: int = K) -> pd.DataFrame:
    """Add h, U, eta, losses to a V5 per-disease frame (no rescoring needed)."""
    out = per.copy()
    n = out["n_query"].to_numpy(dtype=float)
    m = out["n_query_in_pool"].to_numpy(dtype=float)
    R = out[f"recall{k}_end"].to_numpy(dtype=float)
    h = np.rint(R * n)
    nq = np.maximum(n, 1.0)
    cap = np.minimum(k, m)
    out[f"h{k}"] = h
    out[f"oracle_recall{k}"] = cap / nq
    with np.errstate(invalid="ignore", divide="ignore"):
        out[f"eta{k}"] = np.where(m > 0, h / np.maximum(cap, 1.0), np.nan)
    out[f"ranking_loss{k}"] = out[f"oracle_recall{k}"] - R
    out[f"capacity_loss{k}"] = np.minimum(k, n) / nq - out[f"oracle_recall{k}"]
    out[f"recall{k}_norm"] = h / np.maximum(np.minimum(k, n), 1.0)
    return out


def oracle_decomposition_v5() -> dict:
    reg = _read_registry()
    e01 = reg[(reg["experiment"] == "E01") & (reg["status"] == "done")]
    frames = []
    for rid in e01["run_id"]:
        p = p5.RUNS_DIR / rid / "per_disease.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["run_id"] = rid
            frames.append(df)
    per = pd.concat(frames, ignore_index=True)
    per = decompose_per_disease(per)
    cols = ["recall10_end", "oracle_recall10", "eta10", "ranking_loss10", "capacity_loss10",
            "recall10_norm", "coverage", "zero_coverage", "n_query", "n_query_in_pool", "h10"]
    # seeds are averaged inside a disease before any macro average
    by_d = per.groupby(["method", "disease_id"], as_index=False)[cols].mean()
    rows, checks = [], {}
    for method, sub in by_d.groupby("method"):
        chk = check_decomposition(decompose_per_disease(
            per[per["method"] == method].assign()), k=K)
        checks[method] = chk
        cls = outcome_classes(sub.assign(h10=sub["h10"]), k=K).value_counts(normalize=True)
        rec = {"method": method, "n_diseases": int(len(sub))}
        rec.update({f"macro_{c}": float(np.nanmean(sub[c])) for c in cols
                    if c not in ("h10",)})
        rec["n_covered_for_eta"] = int(sub["eta10"].notna().sum())
        rec["share_not_retrieved"] = float(cls.get("not_retrieved", 0.0))
        rec["share_in_pool_missed"] = float(cls.get("in_pool_missed", 0.0))
        rec["share_recovered"] = float(cls.get("recovered", 0.0))
        rec["macro_recall_minus_mean_U_eta"] = float(
            sub["recall10_end"].mean()
            - np.where(sub["eta10"].notna(), sub["oracle_recall10"] * sub["eta10"].fillna(0), 0).mean())
        rows.append(rec)
    table = pd.DataFrame(rows).sort_values("macro_recall10_end", ascending=False)
    out_dir = paths6.AUDIT
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "oracle_decomposition_v5_by_method.csv", index=False,
                 encoding="utf-8-sig")
    by_d.to_parquet(out_dir / "oracle_decomposition_v5_per_disease.parquet", index=False)
    # the shared-candidate bound is method independent: every V5 method ranks the
    # same 500 candidates, so U is identical across methods for a disease
    u_by_method = by_d.pivot_table(index="disease_id", columns="method",
                                   values="oracle_recall10")
    u_spread = float((u_by_method.max(axis=1) - u_by_method.min(axis=1)).max())
    summary = {
        "generated_utc": now_utc(),
        "source": "runs/E01__*/per_disease.csv (V5 archive, read-only)",
        "k": K,
        "per_method": table.to_dict(orient="records"),
        "identity_checks": checks,
        "max_oracle_bound_spread_across_methods": u_spread,
        "note": ("R_d = U_d * eta_d holds disease by disease; macro recall is the "
                 "mean of R_d and is not macro coverage times macro eta. The outcome "
                 "shares (not retrieved / in pool but missed / recovered) are "
                 "proportions of diseases and are not interchangeable with macro "
                 "recall or macro coverage."),
    }
    dump_json(out_dir / "oracle_decomposition_v5.json", summary)
    return summary


# --------------------------------------------------------------------------
# unified ablation table
# --------------------------------------------------------------------------
def _pred_hash(run_dir: Path) -> tuple[str, int]:
    p = run_dir / "predictions.parquet"
    if not p.exists():
        return "", 0
    pred = pd.read_parquet(p, columns=["disease_id", "gene_row"])
    return candidate_hash(pred["disease_id"], pred["gene_row"]), int(pred["disease_id"].nunique())


def unified_ablation(draws: int = 10000) -> pd.DataFrame:
    diseases = pd.read_parquet(p5.TABLES_DIR / "diseases.parquet")
    family_of = dict(zip(diseases["disease_id"], diseases["family_group"]))
    reg = _read_registry()
    e06 = reg[(reg["experiment"] == "E06") & (reg["status"] == "done")]
    run_rows, per_rows = [], []
    for r in e06.itertuples(index=False):
        fold, seed, variant = int(r.fold), int(r.seed), str(r.variant)
        full_id = f"E01__b9_ours__main__f{fold}__s{seed}"
        abl_dir, full_dir = p5.RUNS_DIR / r.run_id, p5.RUNS_DIR / full_id
        if not (abl_dir / "result.json").exists() or not (full_dir / "result.json").exists():
            continue
        ra = json.loads((abl_dir / "result.json").read_text(encoding="utf-8"))
        rf = json.loads((full_dir / "result.json").read_text(encoding="utf-8"))
        ha, na = _pred_hash(abl_dir)
        hf, nf = _pred_hash(full_dir)
        m_abl = float(ra["metrics"]["recall10_end"])
        m_full = float(rf["metrics"]["recall10_end"])
        run_rows.append({"variant": variant, "fold": fold, "seed": seed,
                         "run_full": full_id, "run_ablated": r.run_id,
                         "M_full": m_full, "M_abl": m_abl,
                         "delta_abl_minus_full": m_abl - m_full,
                         "v5_printed_full_minus_abl": m_full - m_abl,
                         "coverage_full": float(rf["metrics"].get("coverage", np.nan)),
                         "coverage_abl": float(ra["metrics"].get("coverage", np.nan)),
                         "n_diseases_full": nf, "n_diseases_abl": na,
                         "candidate_hash_full": hf, "candidate_hash_abl": ha,
                         "same_candidates": hf == ha and hf != ""})
        pa = pd.read_csv(abl_dir / "per_disease.csv")[["disease_id", "recall10_end"]]
        pf = pd.read_csv(full_dir / "per_disease.csv")[["disease_id", "recall10_end"]]
        mrg = pf.merge(pa, on="disease_id", suffixes=("_full", "_abl"))
        mrg["variant"], mrg["fold"], mrg["seed"] = variant, fold, seed
        per_rows.append(mrg)
    runs = pd.DataFrame(run_rows).sort_values(["variant", "fold", "seed"])
    per = pd.concat(per_rows, ignore_index=True)
    # seed-to-seed yardstick: how far two fits of the *same* full model differ
    # on the same fold.  Disease-level intervals condition on the fitted models
    # and do not contain this fit-to-fit variance.
    noise = []
    for fold in sorted(runs["fold"].unique()):
        vals = {}
        for s in (42, 52, 62):
            p = p5.RUNS_DIR / f"E01__b9_ours__main__f{fold}__s{s}" / "result.json"
            if p.exists():
                vals[s] = float(json.loads(p.read_text(encoding="utf-8"))["metrics"]["recall10_end"])
        ss = sorted(vals)
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                noise.append({"fold": fold, "seed_a": ss[i], "seed_b": ss[j],
                              "abs_diff": abs(vals[ss[i]] - vals[ss[j]])})
    noise_df = pd.DataFrame(noise)
    seed_noise = float(noise_df["abs_diff"].mean()) if len(noise_df) else float("nan")
    noise_df.to_csv(paths6.AUDIT / "b9_seed_to_seed_differences.csv", index=False,
                    encoding="utf-8-sig")
    # seeds averaged inside a disease, then disease-level paired differences
    summary_rows = []
    for variant, sub in per.groupby("variant"):
        d = sub.groupby("disease_id")[["recall10_end_full", "recall10_end_abl"]].mean()
        delta = (d["recall10_end_abl"] - d["recall10_end_full"]).to_numpy()
        fams = np.array([family_of.get(x, x) for x in d.index])
        lo, hi, _ = family_bootstrap(delta, fams, draws=draws)
        p = paired_permutation(delta, fams, draws=draws)
        rr = runs[runs["variant"] == variant]
        summary_rows.append({
            "variant": variant, "direction": "Delta_abl = M_abl - M_full",
            "folds": ",".join(str(x) for x in sorted(rr["fold"].unique())),
            "seeds": ",".join(str(x) for x in sorted(rr["seed"].unique())),
            "n_runs": int(len(rr)), "n_diseases": int(len(d)),
            "M_full_mean": float(d["recall10_end_full"].mean()),
            "M_abl_mean": float(d["recall10_end_abl"].mean()),
            "delta_abl": float(delta.mean()), "ci_low": lo, "ci_high": hi,
            "permutation_p": p,
            "frac_diseases_abl_better": float((delta > 1e-12).mean()),
            "frac_diseases_abl_worse": float((delta < -1e-12).mean()),
            "run_level_delta_mean": float(rr["delta_abl_minus_full"].mean()),
            "run_level_delta_sd": float(rr["delta_abl_minus_full"].std(ddof=1)),
            "run_level_delta_min": float(rr["delta_abl_minus_full"].min()),
            "run_level_delta_max": float(rr["delta_abl_minus_full"].max()),
            "runs_abl_worse": int((rr["delta_abl_minus_full"] < 0).sum()),
            "b9_seed_to_seed_mean_abs_diff": seed_noise,
            "abs_delta_exceeds_seed_noise": bool(abs(float(delta.mean())) > seed_noise),
            "matched_seed_full_minus_abl_v5_sign": float(-rr["delta_abl_minus_full"].mean()),
            "candidates_identical_in_all_runs": bool(rr["same_candidates"].all()),
            "mean_coverage_full": float(rr["coverage_full"].mean()),
            "mean_coverage_abl": float(rr["coverage_abl"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    t4p = p5.RESULT_TABLE_DIR / "table4_ablation.csv"
    if t4p.exists():
        t4 = pd.read_csv(t4p)
        pub = dict(zip(t4["variant"], t4["delta"]))
        summary["v5_table4_published_full_minus_abl_unmatched_seeds"] = summary["variant"].map(pub)
        summary["v5_table4_note"] = ("V5 table 4 averaged the full model over seeds "
                                     "42/52/62 but the ablations over 42/52 only")
    out = paths6.AUDIT
    runs.to_csv(out / "ablation_unified_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "ablation_unified.csv", index=False, encoding="utf-8-sig")
    per.to_parquet(out / "ablation_unified_per_disease.parquet", index=False)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V6 zero-training metric audit.")
    ap.add_argument("--draws", type=int, default=10000)
    args = ap.parse_args(argv)
    paths6.ensure()
    dec = oracle_decomposition_v5()
    abl = unified_ablation(draws=args.draws)
    print(json.dumps({"oracle": [{k: r[k] for k in ("method", "macro_recall10_end",
                                                     "macro_oracle_recall10", "macro_eta10",
                                                     "share_recovered")}
                                 for r in dec["per_method"]],
                      "identity": {m: c["identity_holds"] for m, c in dec["identity_checks"].items()},
                      "ablation": abl[["variant", "M_full_mean", "M_abl_mean", "delta_abl",
                                       "ci_low", "ci_high",
                                       "candidates_identical_in_all_runs"]].to_dict(orient="records")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

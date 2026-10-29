"""Collect run outputs into the six manuscript tables.

Table 1 data versions, de-duplication, splits and external exclusions
Table 2 method inputs, training and tuning budget
Table 3 primary and secondary endpoints with effect sizes and intervals
Table 4 ablations, plain meta-ridge and adaptation-stage sensitivity
Table 5 external / temporal / expression / manual-review denominators
Table 6 real time, memory, GPU memory, completion and failure status

Every cell traces back to a run_id in the ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .baselines.registry import METHODS
from .config import Gpu8Config, load_config
from .hashing import atomic_write_df, atomic_write_json, read_json
from .registry import RunRegistry
from .stats import compare_methods, gamma_comparison, holm

PRIMARY = "recall10_end"
SECONDARY = ["ndcg10_end", "recall50_end", "ap_end", "mrr", "recall10_cond",
             "coverage", "zero_coverage"]


def collect_per_disease(run_ids: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for d in sorted(paths.RUNS_DIR.glob("*")):
        if not d.is_dir():
            continue
        if run_ids and d.name not in set(run_ids):
            continue
        p = d / "per_disease.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["run_id"] = d.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_results(run_ids: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for d in sorted(paths.RUNS_DIR.glob("*")):
        p = d / "result.json"
        if not p.exists():
            continue
        if run_ids and d.name not in set(run_ids):
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        row = {"run_id": blob.get("run_id"), "method": blob.get("method"),
               "mode": blob.get("mode", ""),
               "fold": blob.get("fold"), "seed": blob.get("seed"),
               "variant": blob.get("variant", ""), "device": blob.get("device"),
               "n_test_diseases": blob.get("n_test_diseases"),
               "seconds": blob.get("resources", {}).get("seconds"),
               "peak_rss_mib": blob.get("resources", {}).get("peak_rss_mib"),
               "peak_gpu_allocated_mib": blob.get("resources", {}).get("peak_gpu_allocated_mib"),
               "peak_gpu_reserved_mib": blob.get("resources", {}).get("peak_gpu_reserved_mib"),
               "n_parameters": blob.get("fit_info", {}).get("n_parameters"),
               "epochs_run": blob.get("fit_info", {}).get("epochs_run"),
               "best_epoch": blob.get("fit_info", {}).get("best_epoch"),
               "stop_reason": blob.get("fit_info", {}).get("stop_reason"),
               "n_not_applicable": blob.get("metrics", {}).get("n_not_applicable", 0)}
        for k, v in (blob.get("metrics") or {}).items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Table 3: main endpoints
# --------------------------------------------------------------------------
def main_fold_runs(results: pd.DataFrame) -> pd.DataFrame:
    """Only the five numbered main folds, with no variant and no alternative split."""
    if results.empty:
        return results
    out = results[results["variant"].fillna("") == ""]
    if "mode" in out.columns:
        out = out[out["mode"].fillna("") == ""]
    fold_num = pd.to_numeric(out["fold"], errors="coerce")
    return out[fold_num.notna()]


def table3_main(results: pd.DataFrame, cfg: Gpu8Config) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    main = main_fold_runs(results)
    keep = [PRIMARY] + [c for c in SECONDARY if c in main.columns]
    rows = []
    for method, sub in main.groupby("method"):
        # average seeds inside a fold first, then across folds
        by_fold = sub.groupby("fold")[keep].mean()
        rec = {"method": method,
               "label": METHODS[method].label if method in METHODS else method,
               "family": METHODS[method].family if method in METHODS else "",
               "n_folds": int(by_fold.shape[0]),
               "n_runs": int(len(sub)),
               "n_seeds": int(sub["seed"].nunique())}
        for c in keep:
            rec[c] = float(by_fold[c].mean())
            rec[c + "_sd_fold"] = float(by_fold[c].std(ddof=1)) if len(by_fold) > 1 else 0.0
        rec["not_applicable_runs"] = int((sub["n_not_applicable"].fillna(0) > 0).sum())
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(PRIMARY, ascending=False)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Table 4: ablations and adaptation-stage sensitivity
# --------------------------------------------------------------------------
def table4_ablation(results: pd.DataFrame, per_disease: pd.DataFrame,
                    cfg: Gpu8Config, family_of: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()
    main = main_fold_runs(results)
    base = main[main["method"] == "b9_ours"]
    rows = []
    for variant, sub in results[results["variant"].fillna("") != ""].groupby("variant"):
        folds = sorted(set(sub["fold"]) & set(base["fold"]))
        if not folds:
            continue
        b = base[base["fold"].isin(folds)].groupby("fold")[PRIMARY].mean()
        a = sub[sub["fold"].isin(folds)].groupby("fold")[PRIMARY].mean()
        delta = (b - a).dropna()
        rows.append({"variant": variant, "n_folds": len(delta),
                     "full": float(b.mean()), "ablated": float(a.mean()),
                     "delta": float(delta.mean()),
                     "delta_sd": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                     "kind": "retraining"})
    ablation = pd.DataFrame(rows)

    # adaptation-stage sensitivity (inference-time only, never a substitute
    # for a fully retrained configuration)
    sens_path = paths.RESULTS_DIR / "experiments" / "adaptation_sensitivity.csv"
    sens = pd.read_csv(sens_path) if sens_path.exists() else pd.DataFrame()
    return ablation, sens


# --------------------------------------------------------------------------
# Table 6: resources
# --------------------------------------------------------------------------
def table6_resources(results: pd.DataFrame, registry: RunRegistry) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    for method, sub in results.groupby("method"):
        rows.append({
            "method": method,
            "label": METHODS[method].label if method in METHODS else method,
            "n_runs": int(len(sub)),
            "median_minutes": float(sub["seconds"].median() / 60) if sub["seconds"].notna().any() else np.nan,
            "total_hours": float(sub["seconds"].sum() / 3600) if sub["seconds"].notna().any() else np.nan,
            "median_peak_rss_gib": float(sub["peak_rss_mib"].median() / 1024)
            if sub["peak_rss_mib"].notna().any() else np.nan,
            "max_gpu_allocated_mib": float(sub["peak_gpu_allocated_mib"].max())
            if sub["peak_gpu_allocated_mib"].notna().any() else np.nan,
            "max_gpu_reserved_mib": float(sub["peak_gpu_reserved_mib"].max())
            if sub["peak_gpu_reserved_mib"].notna().any() else np.nan,
            "n_parameters": int(sub["n_parameters"].dropna().iloc[0])
            if sub["n_parameters"].notna().any() else None,
            "median_epochs": float(sub["epochs_run"].median())
            if sub["epochs_run"].notna().any() else np.nan,
        })
    df = pd.DataFrame(rows)
    counts = registry.status_counts()
    df.attrs["ledger_status"] = counts
    return df


# --------------------------------------------------------------------------
# primary statistical comparisons
# --------------------------------------------------------------------------
def primary_comparisons(per_disease: pd.DataFrame, cfg: Gpu8Config,
                        family_of: dict) -> pd.DataFrame:
    if per_disease.empty:
        return pd.DataFrame()
    pd_main = per_disease
    if "variant" in pd_main.columns:
        pd_main = pd_main[pd_main["variant"].fillna("") == ""]
    if "fold" in pd_main.columns:
        # the alternative splits write their mode name into the fold column
        pd_main = pd_main[pd.to_numeric(pd_main["fold"], errors="coerce").notna()]
    rows, praw = [], {}
    for comp in cfg.evaluation.primary_comparisons:
        a, b = comp.split("__vs__")
        res = compare_methods(pd_main, PRIMARY, a, b, family_of,
                              draws=cfg.evaluation.bootstrap_draws,
                              seed=cfg.evaluation.stat_seed,
                              note="primary endpoint, zero-shot family")
        if res is None:
            continue
        rows.append(res.as_dict())
        praw[comp] = res.permutation_p
    if not rows:
        return pd.DataFrame()
    adj = holm(praw)
    out = pd.DataFrame(rows)
    out["comparison"] = [f"{r['method_a']}__vs__{r['method_b']}" for r in rows]
    out["holm_p"] = out["comparison"].map(adj)
    out["hypothesis_family"] = "zero_shot_primary"
    out["multiplicity"] = cfg.evaluation.multiplicity
    return out


def run(cfg: Gpu8Config | None = None, registry_path: Path | None = None,
        formal_only: bool = True) -> dict:
    cfg = cfg or load_config()
    reg = RunRegistry(registry_path or (paths.MANIFEST_DIR / "run_registry_166.csv"))
    reg.reconcile_from_runs(paths.RUNS_DIR)
    # Only ledger runs enter the manuscript tables; development, calibration and
    # ad-hoc runs live in the same directory but must never be averaged in.
    ledger_ids = {r["run_id"] for r in reg.rows} if formal_only else None
    results = collect_results(sorted(ledger_ids) if ledger_ids else None)
    per = collect_per_disease(sorted(ledger_ids) if ledger_ids else None)
    diseases = pd.read_parquet(paths.TABLES_DIR / "diseases.parquet")
    family_of = dict(zip(diseases["disease_id"], diseases["family_group"]))

    out_dir = paths.RESULT_TABLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    if not results.empty:
        atomic_write_df(results, out_dir / "all_runs.csv")
        t3 = table3_main(results, cfg)
        atomic_write_df(t3, out_dir / "table3_main_endpoints.csv")
        written["table3_rows"] = int(len(t3))
        t4, sens = table4_ablation(results, per, cfg, family_of)
        atomic_write_df(t4, out_dir / "table4_ablation.csv")
        written["table4_rows"] = int(len(t4))
        t6 = table6_resources(results, reg)
        atomic_write_df(t6, out_dir / "table6_resources.csv")
        written["table6_rows"] = int(len(t6))
    if not per.empty:
        atomic_write_df(per, out_dir / "per_disease_all.csv")
        comps = primary_comparisons(per, cfg, family_of)
        if not comps.empty:
            atomic_write_df(comps, out_dir / "primary_comparisons.csv")
            written["comparisons"] = int(len(comps))

    payload = {"n_runs_collected": int(len(results)),
               "n_per_disease_rows": int(len(per)),
               "ledger_status": reg.status_counts(),
               "written": written}
    atomic_write_json(out_dir / "aggregate_report.json", payload)
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate runs into manuscript tables.")
    ap.add_argument("--registry", default="")
    args = ap.parse_args(argv)
    payload = run(registry_path=Path(args.registry) if args.registry else None)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

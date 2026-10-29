"""Turn the raw per-disease experiment outputs into the summary tables.

Keeps the disease as the unit throughout: seeds and support-order draws are
averaged inside a disease before any difference is taken, and every summary
carries the number of diseases and families it rests on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .config import Gpu8Config, load_config
from .hashing import atomic_write_df, atomic_write_json
from .stats import family_bootstrap, gamma_comparison, paired_permutation

EXP = paths.RESULTS_DIR / "experiments"
PRIMARY = "recall10_end"


def _read(name: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(EXP.glob(name)) if p.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise_dose(cfg: Gpu8Config) -> pd.DataFrame:
    df = _read("dose_f*.csv")
    if df.empty:
        return df
    df = df[df["k"].notna()]
    if df.empty:
        return df
    per = (df.groupby(["method", "k", "disease_id"], as_index=False)
           [[PRIMARY, "ndcg10_end", "recall10_cond", "mrr"]].mean())
    rows = []
    for (m, k), sub in per.groupby(["method", "k"]):
        base = per[(per["method"] == m) & (per["k"] == 0)].set_index("disease_id")[PRIMARY]
        cur = sub.set_index("disease_id")[PRIMARY]
        common = sorted(set(base.index) & set(cur.index))
        delta = float((cur.loc[common] - base.loc[common]).mean()) if common else np.nan
        rows.append({"method": m, "k": int(k), "n_diseases": int(len(sub)),
                     "recall10_end": float(sub[PRIMARY].mean()),
                     "ndcg10_end": float(sub["ndcg10_end"].mean()),
                     "recall10_cond": float(sub["recall10_cond"].mean()),
                     "mrr": float(sub["mrr"].mean()),
                     "delta_vs_k0": delta,
                     "frac_benefit": float((cur.loc[common] > base.loc[common]).mean())
                     if common else np.nan,
                     "frac_harm": float((cur.loc[common] < base.loc[common]).mean())
                     if common else np.nan})
    out = pd.DataFrame(rows).sort_values(["method", "k"])
    atomic_write_df(out, EXP / "dose_summary.csv")
    return out


def summarise_gamma(cfg: Gpu8Config, family_of: dict) -> pd.DataFrame:
    df = _read("dose_f*.csv")
    if df.empty or df["method"].nunique() < 2:
        return pd.DataFrame()
    out = gamma_comparison(df, PRIMARY, "b9_ours", "b8_meta_ridge", family_of,
                           k_values=(1, 3, 5), draws=cfg.evaluation.bootstrap_draws,
                           seed=cfg.evaluation.stat_seed)
    if not out.empty:
        atomic_write_df(out, EXP / "gamma_summary.csv")
    return out


def summarise_evidence(cfg: Gpu8Config) -> pd.DataFrame:
    df = _read("evidence_f*.csv")
    if df.empty:
        return df
    per = (df.groupby(["condition", "disease_id"], as_index=False)
           [[PRIMARY, "a_norm", "lam", "rho", "n_groups", "n_effective"]].mean())
    base = per[per["condition"] == "baseline"].set_index("disease_id")[PRIMARY]
    rows = []
    for cond, sub in per.groupby("condition"):
        cur = sub.set_index("disease_id")[PRIMARY]
        common = sorted(set(base.index) & set(cur.index))
        d = (cur.loc[common] - base.loc[common]) if common else pd.Series(dtype=float)
        rows.append({"condition": cond, "n_diseases": int(len(sub)),
                     "recall10_end": float(sub[PRIMARY].mean()),
                     "delta_vs_baseline": float(d.mean()) if len(d) else np.nan,
                     "mean_a_norm": float(sub["a_norm"].mean()),
                     "mean_lambda": float(sub["lam"].mean()),
                     "mean_rho": float(sub["rho"].mean()),
                     "mean_evidence_groups": float(sub["n_groups"].mean()),
                     "frac_harmed": float((d < 0).mean()) if len(d) else np.nan})
    out = pd.DataFrame(rows)
    atomic_write_df(out, EXP / "evidence_summary.csv")
    return out


def summarise_coverage(cfg: Gpu8Config) -> pd.DataFrame:
    df = _read("coverage_f*.csv")
    if df.empty:
        return df
    cols = [PRIMARY, "recall10_cond", "coverage", "zero_coverage",
            "tie_break_fallback", "ndcg10_end"]
    have = [c for c in cols if c in df.columns]
    out = (df.groupby(["source", "n_candidates_requested"], as_index=False)
           .agg(n_diseases=("disease_id", "nunique"),
                **{c: (c, "mean") for c in have}))
    atomic_write_df(out, EXP / "coverage_summary.csv")
    return out


def summarise_negative(cfg: Gpu8Config) -> pd.DataFrame:
    df = _read("negative_f*.csv")
    if df.empty:
        return df
    per = df.groupby(["condition", "disease_id"], as_index=False)[[PRIMARY]].mean()
    base = per[per["condition"] == "intact"].set_index("disease_id")[PRIMARY]
    rows = []
    for cond, sub in per.groupby("condition"):
        cur = sub.set_index("disease_id")[PRIMARY]
        common = sorted(set(base.index) & set(cur.index))
        d = (cur.loc[common] - base.loc[common]) if common else pd.Series(dtype=float)
        rows.append({"condition": cond, "n_diseases": int(len(sub)),
                     "recall10_end": float(sub[PRIMARY].mean()),
                     "delta_vs_intact": float(d.mean()) if len(d) else np.nan})
    out = pd.DataFrame(rows)
    atomic_write_df(out, EXP / "negative_summary.csv")
    return out


def summarise_network(cfg: Gpu8Config) -> pd.DataFrame:
    df = _read("network_f*.csv")
    if df.empty:
        return df
    out = (df.groupby(["method", "k"], as_index=False)
           .agg(n_diseases=("disease_id", "nunique"),
                recall10_end=(PRIMARY, "mean"),
                ndcg10_end=("ndcg10_end", "mean"),
                recall10_cond=("recall10_cond", "mean"),
                mrr=("mrr", "mean"),
                n_not_applicable=("n_not_applicable", "max")))
    atomic_write_df(out, EXP / "network_summary.csv")
    return out


def summarise_alt_runs(cfg: Gpu8Config) -> dict:
    """External, temporal, family and second-network runs into their tables."""
    rows = []
    for d in sorted(paths.RUNS_DIR.glob("*")):
        p = d / "result.json"
        if not p.exists():
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        mode = blob.get("mode")
        if not mode:
            continue
        m = blob.get("metrics", {})
        rows.append({"mode": mode, "method": blob.get("method"),
                     "seed": blob.get("seed"),
                     "label_source": blob.get("label_source"),
                     "n_diseases": blob.get("n_test_diseases"),
                     "recall10_end": m.get("recall10_end"),
                     "ndcg10_end": m.get("ndcg10_end"),
                     "recall10_cond": m.get("recall10_cond"),
                     "mrr": m.get("mrr"), "coverage": m.get("coverage"),
                     "zero_coverage": m.get("zero_coverage"),
                     "run_id": blob.get("run_id")})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out: dict = {}
    ext = df[df["mode"] == "external"]
    if not ext.empty:
        g = (ext.groupby("method", as_index=False)
             .agg(n_diseases=("n_diseases", "max"), n_pairs=("n_diseases", "max"),
                  recall10_end=("recall10_end", "mean"),
                  recall10_cond=("recall10_cond", "mean"),
                  ndcg10_end=("ndcg10_end", "mean")))
        g["source"] = "GenCC/ClinVar high tier"
        g["layer"] = "layer1+ evaluable"
        atomic_write_df(g, paths.RESULTS_DIR / "external" / "external_eval_summary.csv")
        out["external"] = int(len(g))
    tmp = df[df["mode"] == "temporal"]
    if not tmp.empty:
        g = (tmp.groupby("method", as_index=False)
             .agg(n_pairs=("n_diseases", "max"), n_diseases=("n_diseases", "max"),
                  recall10_end=("recall10_end", "mean"),
                  ndcg10_end=("ndcg10_end", "mean"),
                  recall10_cond=("recall10_cond", "mean")))
        atomic_write_df(g, paths.RESULTS_DIR / "temporal" / "temporal_summary.csv")
        out["temporal"] = int(len(g))
    for mode, name in (("family", "family_summary.csv"), ("ppi2", "ppi2_summary.csv")):
        sub = df[df["mode"] == mode]
        if not sub.empty:
            g = (sub.groupby("method", as_index=False)
                 .agg(n_diseases=("n_diseases", "max"),
                      recall10_end=("recall10_end", "mean"),
                      recall10_end_sd=("recall10_end", "std"),
                      ndcg10_end=("ndcg10_end", "mean"),
                      recall10_cond=("recall10_cond", "mean")))
            atomic_write_df(g, paths.RESULT_TABLE_DIR / name)
            out[mode] = int(len(g))
    atomic_write_df(df, paths.RESULT_TABLE_DIR / "alt_runs_all.csv")
    return out


def run(cfg: Gpu8Config | None = None) -> dict:
    cfg = cfg or load_config()
    diseases = pd.read_parquet(paths.TABLES_DIR / "diseases.parquet")
    family_of = dict(zip(diseases["disease_id"], diseases["family_group"]))
    out = {
        "dose": int(len(summarise_dose(cfg))),
        "gamma": int(len(summarise_gamma(cfg, family_of))),
        "evidence": int(len(summarise_evidence(cfg))),
        "coverage": int(len(summarise_coverage(cfg))),
        "negative": int(len(summarise_negative(cfg))),
        "network": int(len(summarise_network(cfg))),
        "alt": summarise_alt_runs(cfg),
    }
    atomic_write_json(EXP / "summary_report.json", out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarise the non-refit experiments.")
    args = ap.parse_args(argv)
    print(json.dumps(run(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

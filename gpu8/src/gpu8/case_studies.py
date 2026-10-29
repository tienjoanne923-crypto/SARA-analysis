"""Export per-disease top-K candidates and recovery / failure patterns.

These are reading material for the discussion and the blind review sheet, not
claims of discovery.  A gene ranked highly but not currently recorded is a
candidate for inspection; the export says so in its own column names and in the
accompanying note.
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


def load_runs(method: str, fold: int | None = None, seed: int | None = None
              ) -> pd.DataFrame:
    frames = []
    for d in sorted(paths.RUNS_DIR.glob("E01__*")):
        res = d / "result.json"
        pred = d / "predictions.parquet"
        if not (res.exists() and pred.exists()):
            continue
        blob = json.loads(res.read_text(encoding="utf-8"))
        if blob.get("method") != method:
            continue
        if fold is not None and int(blob.get("fold", -1)) != fold:
            continue
        if seed is not None and str(blob.get("seed")) != str(seed):
            continue
        df = pd.read_parquet(pred)
        df["fold"] = blob.get("fold")
        df["seed"] = blob.get("seed")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def export(cfg: Gpu8Config | None = None, method: str = "b9_ours",
           comparator: str = "b8_meta_ridge", top_k: int = 20) -> dict:
    cfg = cfg or load_config()
    pred = load_runs(method)
    if pred.empty:
        return {"status": "no_predictions", "method": method}

    genes = pd.read_parquet(paths.TABLES_DIR / "genes.parquet")
    diseases = pd.read_parquet(paths.TABLES_DIR / "diseases.parquet")
    order = sorted(genes["gene_id"], key=lambda x: int(x) if str(x).isdigit() else 0)
    row_to_gene = {i: g for i, g in enumerate(order)}
    symbol = dict(zip(genes["gene_id"].astype(str), genes["symbol"]))
    dname = dict(zip(diseases["disease_id"], diseases["disease_name"]))
    fam = dict(zip(diseases["disease_id"], diseases["family_group"]))

    # average the score of a pair across seeds before ranking
    avg = (pred.groupby(["disease_id", "gene_row"], as_index=False)
           .agg(score=("score", "mean"), label=("label", "max"),
                n_seeds=("seed", "nunique")))

    rows = []
    for did, sub in avg.groupby("disease_id"):
        sub = sub.sort_values("score", ascending=False).head(top_k)
        for rank, r in enumerate(sub.itertuples(index=False), start=1):
            gid = row_to_gene.get(int(r.gene_row), "")
            rows.append({
                "disease_id": did, "disease_name": dname.get(did, ""),
                "family_group": fam.get(did, ""), "rank": rank,
                "gene_id": gid, "gene_symbol": symbol.get(gid, ""),
                "score": float(r.score), "n_seeds_averaged": int(r.n_seeds),
                "is_recorded_positive": int(r.label > 0),
                "status": "recorded_association" if r.label > 0
                          else "not_currently_recorded",
            })
    topk = pd.DataFrame(rows)
    atomic_write_df(topk, paths.RESULTS_DIR / "case_studies" / "topk_per_disease.csv")

    # disease-level recovery summary
    summ = []
    for did, sub in avg.groupby("disease_id"):
        s = sub.sort_values("score", ascending=False)
        pos = int(s["label"].sum())
        hits = int(s.head(10)["label"].sum())
        first = s.reset_index(drop=True)
        idx = first.index[first["label"] > 0]
        summ.append({"disease_id": did, "disease_name": dname.get(did, ""),
                     "family_group": fam.get(did, ""),
                     "n_candidates": int(len(s)), "n_positives_in_pool": pos,
                     "hits_at_10": hits,
                     "first_hit_rank": int(idx[0] + 1) if len(idx) else None,
                     "outcome": "recovered" if hits > 0 else
                                ("in_pool_missed" if pos > 0 else "not_retrieved")})
    summary = pd.DataFrame(summ)
    atomic_write_df(summary, paths.RESULTS_DIR / "case_studies" / "disease_summary.csv")

    # contrast against the nearest comparator
    contrast = pd.DataFrame()
    cpred = load_runs(comparator)
    if not cpred.empty:
        cavg = (cpred.groupby(["disease_id", "gene_row"], as_index=False)
                .agg(score=("score", "mean"), label=("label", "max")))
        rows = []
        for did in sorted(set(avg["disease_id"]) & set(cavg["disease_id"])):
            a = avg[avg["disease_id"] == did].sort_values("score", ascending=False)
            b = cavg[cavg["disease_id"] == did].sort_values("score", ascending=False)
            at = set(a.head(10)["gene_row"].astype(int))
            bt = set(b.head(10)["gene_row"].astype(int))
            posr = set(a[a["label"] > 0]["gene_row"].astype(int))
            rows.append({"disease_id": did, "disease_name": dname.get(did, ""),
                         "family_group": fam.get(did, ""),
                         "recovered_by_both": len(at & bt & posr),
                         "recovered_only_by_method": len((at - bt) & posr),
                         "recovered_only_by_comparator": len((bt - at) & posr),
                         "missed_by_both": len(posr - at - bt),
                         "n_positives_in_pool": len(posr)})
        contrast = pd.DataFrame(rows)
        atomic_write_df(contrast,
                        paths.RESULTS_DIR / "case_studies" / "recovery_contrast.csv")

    payload = {
        "status": "ok", "method": method, "comparator": comparator,
        "n_diseases": int(summary["disease_id"].nunique()),
        "top_k": top_k,
        "outcome_counts": summary["outcome"].value_counts().to_dict(),
        "n_not_currently_recorded_in_top10": int(
            topk[(topk["rank"] <= 10) & (topk["is_recorded_positive"] == 0)].shape[0]),
        "contrast": (contrast[["recovered_by_both", "recovered_only_by_method",
                               "recovered_only_by_comparator", "missed_by_both"]]
                     .sum().to_dict() if not contrast.empty else {}),
        "note": ("A gene ranked highly but not currently recorded is a candidate "
                 "for inspection, not a discovery. These exports feed the blind "
                 "review sheet and the failure analysis; they are never presented "
                 "as confirmed new biology."),
    }
    atomic_write_json(paths.RESULTS_DIR / "case_studies" / "case_study_report.json",
                      payload)
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export case studies.")
    ap.add_argument("--method", default="b9_ours")
    ap.add_argument("--comparator", default="b8_meta_ridge")
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args(argv)
    print(json.dumps(export(method=args.method, comparator=args.comparator,
                            top_k=args.top_k), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

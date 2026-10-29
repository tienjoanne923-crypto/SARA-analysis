"""Prepare the blind literature-review sheet (plan section 15.4).

Samples about forty diseases and four disease-gene pairs each, covering four
pre-defined outcome classes: recovered only by the proposed method, recovered
only by the strongest comparator, recovered by both, and a low-ranked control.
Reviewers receive the pairs with the algorithm names and ranks removed.

An automatic PubMed co-occurrence pre-screen is attached as an aid.  It is not
a substitute for human review: a co-occurrence count says nothing about the
direction or quality of the evidence, and the sheet records that explicitly.
The human review itself is *not* performed here.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .config import Gpu8Config, load_config
from .hashing import atomic_write_df, atomic_write_json

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def load_predictions(method: str, fold: int, seed: int = 42) -> pd.DataFrame:
    frames = []
    for d in paths.RUNS_DIR.glob(f"*{method}*f{fold}*s{seed}*"):
        p = d / "predictions.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def topk_sets(pred: pd.DataFrame, k: int = 10) -> dict[str, set]:
    out = {}
    for did, sub in pred.groupby("disease_id"):
        top = sub.sort_values("score", ascending=False).head(k)
        out[did] = set(top["gene_row"].astype(int))
    return out


def build_sheet(cfg: Gpu8Config, fold: int = 0, n_diseases: int = 40,
                per_disease: int = 4, seed: int = 20260922) -> pd.DataFrame:
    ours = load_predictions("b9_ours", fold)
    comp = load_predictions("b8_meta_ridge", fold)
    if ours.empty or comp.empty:
        return pd.DataFrame()
    genes = pd.read_parquet(paths.TABLES_DIR / "genes.parquet")
    diseases = pd.read_parquet(paths.TABLES_DIR / "diseases.parquet")
    row_to_gene = {i: g for i, g in enumerate(
        sorted(genes["gene_id"], key=lambda x: int(x) if str(x).isdigit() else 0))}
    gene_symbol = dict(zip(genes["gene_id"].astype(str), genes["symbol"]))
    disease_name = dict(zip(diseases["disease_id"], diseases["disease_name"]))

    a_top, b_top = topk_sets(ours), topk_sets(comp)
    a_pos = {d: set(s[s["label"] > 0]["gene_row"].astype(int))
             for d, s in ours.groupby("disease_id")}
    rng = np.random.default_rng(seed)

    # disease selection stratified by coverage and positive burden, not by score
    cand_ids = sorted(set(a_top) & set(b_top))
    info = []
    for d in cand_ids:
        pos = a_pos.get(d, set())
        info.append({"disease_id": d, "n_pos": len(pos),
                     "hit_a": len(a_top[d] & pos), "hit_b": len(b_top[d] & pos)})
    idf = pd.DataFrame(info)
    if idf.empty:
        return pd.DataFrame()
    idf["stratum"] = pd.cut(idf["n_pos"], [0, 1, 3, 10, 10 ** 6],
                            labels=["1", "2-3", "4-10", ">10"])
    picked: list[str] = []
    per_stratum = max(1, n_diseases // max(1, idf["stratum"].nunique()))
    for _, sub in idf.groupby("stratum", observed=True):
        take = min(per_stratum, len(sub))
        picked += list(rng.choice(sub["disease_id"].to_numpy(), size=take, replace=False))
    picked = picked[:n_diseases]

    rows = []
    for d in picked:
        pos = a_pos.get(d, set())
        classes = {
            "method_a_only": sorted((a_top[d] - b_top[d])),
            "method_b_only": sorted((b_top[d] - a_top[d])),
            "shared": sorted((a_top[d] & b_top[d])),
            "low_ranked_control": [],
        }
        low = ours[(ours["disease_id"] == d)].sort_values("score").head(30)
        classes["low_ranked_control"] = list(low["gene_row"].astype(int))
        for cls in ("method_a_only", "method_b_only", "shared", "low_ranked_control"):
            pool = classes[cls]
            if not pool:
                continue
            take = min(max(1, per_disease // 4), len(pool))
            chosen = rng.choice(np.array(pool), size=take, replace=False)
            for gr in chosen:
                gid = row_to_gene.get(int(gr), "")
                rows.append({
                    "review_id": f"R{len(rows):04d}",
                    "disease_id": d, "disease_name": disease_name.get(d, ""),
                    "gene_id": gid, "gene_symbol": gene_symbol.get(gid, ""),
                    "outcome_class_blinded": "", "is_known_positive": int(int(gr) in pos),
                    "reviewer_1_verdict": "", "reviewer_2_verdict": "",
                    "evidence_type": "", "pmid_cited": "", "notes": "",
                    "_hidden_class": cls, "_hidden_gene_row": int(gr),
                })
    df = pd.DataFrame(rows)
    return df


def pubmed_cooccurrence(disease: str, gene: str, sleep: float = 0.4,
                        timeout: int = 20) -> dict:
    """Count PubMed records mentioning both terms. An aid, never a verdict."""
    if not disease or not gene:
        return {"count": None, "query": "", "error": "missing term"}
    term = f'("{disease}"[Title/Abstract]) AND ("{gene}"[Title/Abstract])'
    url = EUTILS + "?" + urllib.parse.urlencode(
        {"db": "pubmed", "term": term, "retmode": "json", "retmax": 0,
         "tool": "gpu8-pipeline", "email": "noreply@example.org"})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            blob = json.loads(fh.read().decode("utf-8"))
        n = int(blob.get("esearchresult", {}).get("count", 0))
        time.sleep(sleep)
        return {"count": n, "query": term, "error": ""}
    except Exception as exc:
        time.sleep(sleep)
        return {"count": None, "query": term, "error": f"{type(exc).__name__}: {exc}"}


def run(cfg: Gpu8Config | None = None, fold: int = 0, prescreen: bool = True,
        max_prescreen: int = 200) -> dict:
    cfg = cfg or load_config()
    df = build_sheet(cfg, fold=fold)
    out_dir = paths.RESULTS_DIR / "manual_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        payload = {"status": "no_predictions",
                   "note": "prediction files for the two compared methods were not found"}
        atomic_write_json(out_dir / "manual_review_status.json", payload)
        return payload

    if prescreen:
        counts, errors = [], 0
        for i, r in enumerate(df.itertuples(index=False)):
            if i >= max_prescreen:
                counts.append(None)
                continue
            res = pubmed_cooccurrence(getattr(r, "disease_name", ""),
                                      getattr(r, "gene_symbol", ""))
            counts.append(res["count"])
            errors += 1 if res["error"] else 0
            if (i + 1) % 20 == 0:
                print(f"  pre-screened {i + 1}/{min(len(df), max_prescreen)}", flush=True)
        df["pubmed_cooccurrence"] = counts
    else:
        df["pubmed_cooccurrence"] = None
        errors = 0

    blind_cols = [c for c in df.columns if not c.startswith("_")]
    atomic_write_df(df[blind_cols], out_dir / "blind_review_sheet.csv")
    atomic_write_df(df, out_dir / "blind_review_key.csv")

    payload = {
        "status": "prepared_not_reviewed",
        "n_pairs": int(len(df)),
        "n_diseases": int(df["disease_id"].nunique()),
        "class_counts": df["_hidden_class"].value_counts().to_dict(),
        "prescreen_errors": errors,
        "sheet": paths.rel(out_dir / "blind_review_sheet.csv"),
        "key": paths.rel(out_dir / "blind_review_key.csv"),
        "caveat": ("The human blind review has NOT been performed. The sheet is "
                   "de-identified and ready for two independent reviewers. PubMed "
                   "co-occurrence counts are an aid only: they do not indicate the "
                   "direction, quality or independence of the evidence, and a "
                   "co-occurrence is not a confirmation."),
    }
    atomic_write_json(out_dir / "manual_review_status.json", payload)
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prepare the blind review sheet.")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--no_prescreen", action="store_true")
    ap.add_argument("--max_prescreen", type=int, default=200)
    args = ap.parse_args(argv)
    payload = run(fold=args.fold, prescreen=not args.no_prescreen,
                  max_prescreen=args.max_prescreen)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

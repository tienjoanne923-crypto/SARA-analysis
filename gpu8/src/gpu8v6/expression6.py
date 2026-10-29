"""V6 section 12.4 — pass the frozen V6 top genes into the existing expression analysis.

For psoriasis (test fold F3) and ulcerative colitis (test fold F1) the frozen
formal rankings are read from the run files (never re-fitted): the V6 main
scorer (LightGBM, seed 42), the deterministic fusion F7, the V5 CPU fusion from
the V5 archive, and a random set matched to the V6 top genes on mean
expression, PPI degree and CTD degree.  Every input gene set is hashed; the
measurable background, the matched controls and every failure ("too few
measurable genes") are saved.

Interpretation limits: lesional-vs-uninvolved differential expression supports
relevance only, never causality; tissue expression (GTEx) is a V6 gene feature,
so it is a correlated source, while the GEO disease contrasts are not used for
training anywhere.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from gpu8 import paths as p5
from gpu8.expression import matched_background, run_series

from . import paths6
from .common import World6, dump_json, now_utc

TARGETS = {"MESH:D011565": ("psoriasis", ["GSE13355", "GSE30999"]),
           "MESH:D003093": ("ulcerative_colitis", ["GSE87466", "GSE75214"])}


def top_from_predictions(path, disease_id: str, k: int, id_of_row: dict) -> list[str]:
    pred = pd.read_parquet(path)
    sub = pred[pred["disease_id"] == disease_id]
    sub = sub.sort_values(["score", "gene_row"], ascending=[False, True]).head(k)
    return [id_of_row[int(r)] for r in sub["gene_row"]]


def set_hash(genes: list[str]) -> str:
    return hashlib.sha256("|".join(sorted(genes)).encode()).hexdigest()


def run() -> dict:
    w = World6()
    role = {}
    for f, ids in w.spec.folds.items():
        for d in ids:
            role[d] = int(f[1])
    genes = w.genes
    ppi = w.ppi
    deg = pd.concat([ppi["gene_a"], ppi["gene_b"]]).value_counts()
    cdeg = w.relations["gene_id"].value_counts()
    gene_meta = pd.DataFrame({"gene_id": genes["gene_id"].astype(str)})
    gene_meta["ppi_degree"] = gene_meta["gene_id"].map(deg).fillna(0.0)
    gene_meta["ctd_degree"] = gene_meta["gene_id"].map(cdeg).fillna(0.0)
    results, sets_log = [], []
    for did, (ctx_name, series) in TARGETS.items():
        fold = role[did]
        sources = {
            "v6_main_lightgbm": paths6.RUNS / f"V6Z__lightgbm__f{fold}__s42" / "predictions.parquet",
            "v6_F7": paths6.RUNS / f"V6Z__F7__f{fold}" / "predictions.parquet",
            "v5_b3_fusion": p5.RUNS_DIR / f"E01__b3_cpu_fusion__main__f{fold}__sfixed" / "predictions.parquet",
        }
        top_sets = {}
        for name, path in sources.items():
            if not path.exists():
                sets_log.append({"disease": did, "set": name, "status": "missing predictions"})
                continue
            for k in (20, 50):
                g = top_from_predictions(path, did, k, w.gene_id_of_row)
                top_sets[f"{name}_top{k}"] = g
        # matched random control: for every V6 top gene the nearest unused gene by
        # (log PPI degree, log CTD degree), excluding every gene of every top set
        taken = set().union(*[set(v) for v in top_sets.values()]) if top_sets else set()
        feat = np.log1p(gene_meta[["ppi_degree", "ctd_degree"]].to_numpy(float))
        ids = gene_meta["gene_id"].tolist()
        pos = {g: i for i, g in enumerate(ids)}
        rng = np.random.default_rng(20260923)
        for k in (20, 50):
            base = top_sets.get(f"v6_main_lightgbm_top{k}", [])
            used, pick = set(taken), []
            for g in base:
                if g not in pos:
                    continue
                dist = np.abs(feat - feat[pos[g]]).sum(axis=1) + rng.random(len(ids)) * 1e-9
                for j in np.argsort(dist):
                    if ids[j] not in used:
                        used.add(ids[j])
                        pick.append(ids[j])
                        break
            top_sets[f"matched_random_top{k}"] = pick
        known = set(w.positives.get(did, []))
        for name, g in top_sets.items():
            sets_log.append({"disease": did, "set": name, "n": len(g), "sha256": set_hash(g),
                             "n_known_positive": len(set(g) & known)})
        for acc in series:
            res = run_series(acc, top_sets, gene_meta)
            res["disease_id"], res["fold"] = did, fold
            results.append(res)
    out = {"generated_utc": now_utc(), "sets": sets_log, "series": results,
           "claims_boundary": ("relevance support only; lesional expression does not show "
                               "causality; GTEx tissue expression is a V6 feature (correlated), "
                               "GEO contrasts are never used for training")}
    dump_json(paths6.EXTERNAL_RES / "expression_v6.json", out)
    return out


if __name__ == "__main__":
    paths6.ensure()
    r = run()
    print(json.dumps(r["sets"], indent=2))
    for s in r["series"]:
        print(s.get("accession"), s.get("status"), json.dumps(s.get("contrasts", {}), default=str)[:1500])

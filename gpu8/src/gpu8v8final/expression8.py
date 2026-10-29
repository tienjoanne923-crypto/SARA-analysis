"""E9 — the four GEO expression cohorts with frozen V8.1 Top-20 / Top-50 lists.

Same analysis rules as V6 (gpu8.expression.run_series: one probe per gene by mean
expression, moderated t, rank-based enrichment against a background matched on mean
expression, PPI degree and CTD degree).  The runs that define the Top-K sets are
registered (time-stamped appendix) before the cohorts are read; new output files
only.  Lesional expression supports relevance, never causality; GO / GTEx used by
the model are correlated knowledge, the GEO contrasts are never used for training.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from gpu8.expression import run_series

from . import paths8
from .common8 import dump_json, load_manifest, now_utc

TARGETS = {"MESH:D011565": ("psoriasis", ["GSE13355", "GSE30999"]),
           "MESH:D003093": ("ulcerative_colitis", ["GSE87466", "GSE75214"])}
RUNS = {"v8_deployed_s42": ("main", 42, "deployed"), "v8_deployed_s52": ("main", 52, "deployed"),
        "v8_deployed_s62": ("main", 62, "deployed"), "channel_lgb_v6_s42": ("main", 42, "channel_lgb_v6"),
        "v71_direct_strict_s42": ("main", 42, "v71_direct_strict"), "F7_v6": ("main", 42, "F7_v6")}


def top_from(fold: int, tag: str, seed: int, method: str, d: str, k: int, id_of_row: dict) -> list[str]:
    z = np.load(paths8.FORMAL / tag / f"s{seed}" / f"fold{fold}" / f"pred_{method}__test.npz")
    ids = [str(x) for x in z["ids"]]
    if d not in ids:
        return []
    i = ids.index(d)
    v = np.isfinite(z["score"][i])
    g, s = z["genes"][i][v].astype(np.int64), z["score"][i][v].astype(np.float64)
    o = g[np.lexsort((g, -s))][:k]
    return [id_of_row[int(x)] for x in o]


def register_runs() -> dict:
    from .register8 import appendix
    reg = load_manifest("protocol_v8_registered.json")
    if any(a["name"] == "expression_topk_runs" for a in reg.get("appendices", [])):
        return reg
    return appendix("expression_topk_runs", {
        "primary": "v8_deployed_s42 (Top-20 and Top-50)", "descriptive": sorted(set(RUNS) - {"v8_deployed_s42"}),
        "matched_random": "nearest unused gene on (log PPI degree, log CTD degree) per primary Top gene",
        "diseases": {k: v[0] for k, v in TARGETS.items()}, "series": {k: v[1] for k, v in TARGETS.items()}})


def run() -> dict:
    from gpu8v6.common import World6
    register_runs()
    w = World6()
    fold_of = {d: int(f[1]) for f, ids in w.spec.folds.items() for d in ids}
    deg = pd.concat([w.ppi["gene_a"], w.ppi["gene_b"]]).value_counts()
    cdeg = w.relations["gene_id"].value_counts()
    meta = pd.DataFrame({"gene_id": w.genes["gene_id"].astype(str)})
    meta["ppi_degree"] = meta["gene_id"].map(deg).fillna(0.0)
    meta["ctd_degree"] = meta["gene_id"].map(cdeg).fillna(0.0)
    results, log = [], []
    for did, (name, series) in TARGETS.items():
        f = fold_of[did]
        sets = {}
        for rn, (tag, seed, meth) in RUNS.items():
            for k in (20, 50):
                g = top_from(f, tag, seed, meth, did, k, w.gene_id_of_row)
                if g:
                    sets[f"{rn}_top{k}"] = g
        taken = set().union(*[set(v) for v in sets.values()]) if sets else set()
        feat = np.log1p(meta[["ppi_degree", "ctd_degree"]].to_numpy(float))
        ids = meta["gene_id"].tolist()
        pos = {g: i for i, g in enumerate(ids)}
        rng = np.random.default_rng(20260924)
        for k in (20, 50):
            used, pick = set(taken), []
            for g in sets.get(f"v8_deployed_s42_top{k}", []):
                dist = np.abs(feat - feat[pos[g]]).sum(axis=1) + rng.random(len(ids)) * 1e-9
                for jx in np.argsort(dist):
                    if ids[jx] not in used:
                        used.add(ids[jx])
                        pick.append(ids[jx])
                        break
            sets[f"matched_random_top{k}"] = pick
        known = set(w.positives.get(did, []))
        for sn, g in sets.items():
            log.append({"disease": did, "set": sn, "n": len(g), "n_known_positive": len(set(g) & known),
                        "sha256": hashlib.sha256("|".join(sorted(g)).encode()).hexdigest()})
        for acc in series:
            r = run_series(acc, sets, meta)
            r["disease_id"], r["fold"] = did, f
            results.append(r)
    out = {"generated_utc": now_utc(), "sets": log, "series": results,
           "claims_boundary": "relevance support only; no causal claim; negative results kept"}
    dump_json(paths8.EXT_RES / "expression_v8.json", out)
    rows = []
    for r in results:
        for cn, entry in (r.get("contrasts") or {}).items():
            for sn, v in entry.items():
                if isinstance(v, dict):
                    rows.append({"accession": r["accession"], "disease": r["disease_id"], "contrast": cn, "set": sn,
                                 **{k: v.get(k) for k in ("n_selected", "enrichment", "p_value", "note")}})
    pd.DataFrame(rows).to_csv(paths8.EXT_RES / "expression_v8_table.csv", index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    paths8.ensure()
    r = run()
    print(json.dumps(r["sets"], indent=1)[:2000])

"""E9 — materials for the 160-pair two-reviewer blinded assessment (§12.4).

Four categories of 40 predicted (disease, gene) pairs from the frozen F_j test lists
(seed 42): V8.1-only top-10, strong-baseline-only top-10 (channel-only LightGBM on
its own V6 500-list), top-10 of both, and low-rank matched pairs (ranked beyond 200
by both, same disease, matched on the PPI-degree tertile of a V8.1-only pair).  The
reviewer sheet has a random row order and hides the category, the method, the rank
and whether the pair is a known CTD relation; the key is written separately.  The
V5 defects (glob over several runs, category order visible, known positives shown)
are avoided by construction.  The review itself needs two independent human
reviewers: it is delivered as blank forms and reported as not done.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import paths8
from .common8 import dump_json, now_utc, rng_for

OUT = paths8.EXT_RES / "review160"
N_PER = 40


def ranked(z, i):
    v = np.isfinite(z["score"][i])
    g, s = z["genes"][i][v].astype(np.int64), z["score"][i][v].astype(np.float64)
    return g[np.lexsort((g, -s))]


def build() -> dict:
    from gpu8v6.common import World6
    w = World6()
    OUT.mkdir(parents=True, exist_ok=True)
    deg = pd.concat([w.ppi["gene_a"], w.ppi["gene_b"]]).value_counts()
    gdeg = np.array([deg.get(w.gene_id_of_row[i], 0) for i in range(w.n_g)])
    tert = np.digitize(gdeg, np.quantile(gdeg[gdeg > 0], [1 / 3, 2 / 3]))
    cats = {"v8_only": [], "baseline_only": [], "both": [], "low_rank_matched": []}
    run_ids = {}
    for j in range(5):
        d = paths8.FORMAL / "main" / "s42" / f"fold{j}"
        a, b = np.load(d / "pred_deployed__test.npz"), np.load(d / "pred_channel_lgb_v6__test.npz")
        run_ids[j] = {"v8": str(d / "pred_deployed__test.npz"), "baseline": str(d / "pred_channel_lgb_v6__test.npz")}
        pb = {str(x): i for i, x in enumerate(b["ids"])}
        for i, dd in enumerate(a["ids"]):
            dd = str(dd)
            if dd not in pb:
                continue
            ra, rb = ranked(a, i), ranked(b, pb[dd])
            ta, tb = set(ra[:10].tolist()), set(rb[:10].tolist())
            for g in ta - tb:
                cats["v8_only"].append((dd, int(g), j))
            for g in tb - ta:
                cats["baseline_only"].append((dd, int(g), j))
            for g in ta & tb:
                cats["both"].append((dd, int(g), j))
    rng = rng_for("review160")
    sample = {}
    for c in ("v8_only", "baseline_only", "both"):
        idx = rng.choice(len(cats[c]), size=min(N_PER, len(cats[c])), replace=False)
        sample[c] = [cats[c][k] for k in sorted(idx)]
    low = []
    for dd, g, j in sample["v8_only"]:
        d = paths8.FORMAL / "main" / "s42" / f"fold{j}"
        a, b = np.load(d / "pred_deployed__test.npz"), np.load(d / "pred_channel_lgb_v6__test.npz")
        ia = [str(x) for x in a["ids"]].index(dd)
        ib = [str(x) for x in b["ids"]].index(dd)
        top_a, top_b = set(ranked(a, ia)[:200].tolist()), set(ranked(b, ib)[:200].tolist())
        cand = [x for x in range(w.n_g) if x not in top_a and x not in top_b and tert[x] == tert[g]]
        low.append((dd, int(rng.choice(cand)), j))
    sample["low_rank_matched"] = low
    names = dict(zip(w.diseases["disease_id"], w.diseases["disease_name"]))
    sym = dict(zip(w.genes["gene_id"].astype(str), w.genes["symbol"]))
    rows = []
    for c, items in sample.items():
        for dd, g, j in items:
            gid = w.gene_id_of_row[g]
            rows.append({"category": c, "disease_id": dd, "disease_name": names.get(dd, ""), "gene_id": gid,
                         "gene_symbol": sym.get(gid, ""), "fold": j,
                         "known_ctd_relation": gid in set(w.positives.get(dd, []))})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    df.insert(0, "item_id", [f"R{k:03d}" for k in range(len(df))])
    sheet = df[["item_id", "disease_id", "disease_name", "gene_id", "gene_symbol"]].copy()
    for r in ("reviewer1", "reviewer2"):
        sheet[f"{r}_plausibility_0to3"] = ""
        sheet[f"{r}_evidence_pmids"] = ""
        sheet[f"{r}_notes"] = ""
    sheet.to_csv(OUT / "review160_blinded_sheet.csv", index=False, encoding="utf-8-sig")
    key = df[["item_id", "category", "fold", "known_ctd_relation"]]
    key.to_csv(OUT / "review160_key_DO_NOT_SHARE_WITH_REVIEWERS.csv", index=False, encoding="utf-8-sig")
    (OUT / "review160_instructions.md").write_text(
        "# Two-reviewer blinded assessment (160 pairs)\n\n"
        "Each reviewer works independently on `review160_blinded_sheet.csv`.\n\n"
        "Score the disease-gene pair from the literature and public databases:\n\n"
        "* 3: direct human genetic or functional evidence for this disease\n"
        "* 2: strong indirect evidence (same pathway / model organism / closely related disease)\n"
        "* 1: weak or speculative link\n* 0: no support found\n\n"
        "Record PMIDs used.  Do not look up which method proposed the pair; categories, ranks and known "
        "relations are hidden on purpose.  Disagreements are resolved after both sheets are complete.\n",
        encoding="utf-8")
    info = {"generated_utc": now_utc(), "counts": df["category"].value_counts().to_dict(),
            "pool_sizes": {c: len(v) for c, v in cats.items() if c != "low_rank_matched"},
            "runs": run_ids, "sheet_sha256": hashlib.sha256(sheet.to_csv(index=False).encode()).hexdigest(),
            "status": "materials delivered; review NOT performed (two independent human reviewers required)",
            "known_ctd_relation_share": df.groupby("category")["known_ctd_relation"].mean().to_dict()}
    dump_json(OUT / "review160_info.json", info)
    return info


if __name__ == "__main__":
    paths8.ensure()
    print(json.dumps(build(), indent=1))

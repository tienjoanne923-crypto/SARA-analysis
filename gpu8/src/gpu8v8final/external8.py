"""E8 — external re-analysis of the V7.1 618-disease confirmation set (§12.1-12.2).

The set was frozen by V7.1 and already evaluated there; V8.1 re-uses it as an
*external re-analysis* (not a first confirmation).  Each disease is scored in its
own outer test fold with the saved V8.1 lists; the disease's CTD positives are
removed from every ranked list before scoring (filtered ranking).

Three independent fields per external pair:
  seen_in_training_disease   the disease (or a strict equivalent) is a reference disease
                             of the scoring context (expected: never);
  same_relation_seen         the gene is a visible reference positive of a hierarchy
                             neighbour (CTD parent / child) or of a family member of the
                             disease in the scoring context;
  shared_original_evidence   the external record's PMIDs intersect the PMIDs of the
                             context's training relations or of the gene's GO records;
                             records without PMIDs are "unknown" and never count as independent.
GenCC and ClinVar are reported separately and as the de-duplicated union.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8 import paths as p5

from . import paths8
from .common8 import dump_json, load_manifest, now_utc
from .evaluate8 import fast_family_bootstrap

OUT = paths8.EXT_RES / "external618"


def pairs_with_fields(w, roles) -> pd.DataFrame:
    conf = pd.read_parquet(paths8.V71_TABLES / "confirmation_set_v71.parquet")
    ext = pd.read_parquet(p5.TABLES_DIR / "external_labels.parquet")
    ext = ext.assign(gene_id=ext["gene_id"].astype(str))
    fold_of = {d: j for j in range(5) for d in w.fold(j)["test_diseases"]}
    eq = roles.eq
    fam = w.family_of
    par = w.hierarchy.groupby("child_disease")["parent_disease"].apply(set).to_dict()
    chi = w.hierarchy.groupby("parent_disease")["child_disease"].apply(set).to_dict()
    go = pd.read_parquet(paths8.TABLES / "go_records_current_v8.parquet", columns=["gene_row", "pmids"])
    go_pm = go[go["pmids"].str.len() > 0].groupby("gene_row")["pmids"].apply(
        lambda s: {p for x in s for p in x.split("|") if p}).to_dict()
    rows = []
    cache = {}
    for r in conf.itertuples(index=False):
        j = fold_of.get(r.disease_id)
        if j is None:
            continue
        if j not in cache:
            ctx = w.contexts[f"f{j}_outer"]
            ref = set(roles.reference(ctx))
            rel = w.primary_relations[w.primary_relations["disease_id"].isin(ref)]
            pos_by_d = rel.groupby("disease_id")["gene_id"].apply(lambda s: set(s.astype(str))).to_dict()
            pm = set()
            for rid in rel["relation_id"]:
                pm |= w.relation_pmids.get(rid, set())
            cache[j] = (ref, {eq.get(d, d) for d in ref}, pos_by_d, pm)
        ref, ref_groups, pos_by_d, pm = cache[j]
        d, g = r.disease_id, str(r.gene_id)
        nb = (par.get(d, set()) | chi.get(d, set()) | {x for x in ref if fam.get(x) == fam.get(d)}) & ref
        same_rel = any(g in pos_by_d.get(x, set()) for x in nb)
        recs = ext[(ext["disease_id"] == d) & (ext["gene_id"] == g)]
        for s in ("GenCC", "ClinVar"):
            sub = recs[recs["source"] == s]
            if s == "GenCC":
                sub = sub[sub["classification"].isin(["Definitive", "Strong"])]
            else:
                sub = sub[sub["review_stars"].fillna(0) >= 2]
            if sub.empty:
                continue
            pmids = {p for x in sub["pmids"].fillna("") for p in str(x).split("|") if p.strip()}
            if not pmids:
                shared = "unknown"
            else:
                shared = "yes" if (pmids & pm) or (pmids & go_pm.get(int(r.gene_row), set())) else "no"
            rows.append({"disease_id": d, "gene_id": g, "gene_row": int(r.gene_row), "fold": j, "source": s,
                         "seen_in_training_disease": eq.get(d, d) in ref_groups, "same_relation_seen": same_rel,
                         "shared_original_evidence": shared, "n_pmids": len(pmids)})
    return pd.DataFrame(rows)


def score(w, pairs: pd.DataFrame, method: str, seeds=(42, 52, 62)) -> pd.DataFrame:
    """Filtered-ranking Recall@10 / hit@10 / Recall@50 per disease and subset, seeds averaged."""
    ctd = {d: set(w.query_rows(d).tolist()) for d in pairs["disease_id"].unique()}
    recs = []
    for seed in seeds:
        for j in range(5):
            p = paths8.FORMAL / "main" / f"s{seed}" / f"fold{j}" / f"pred_{method}__test.npz"
            if not p.exists():
                continue
            z = np.load(p)
            pos = {str(x): i for i, x in enumerate(z["ids"])}
            sub = pairs[pairs["fold"] == j]
            for d, grp in sub.groupby("disease_id"):
                if d not in pos:
                    continue
                i = pos[d]
                v = z["valid"][i] if "valid" in z.files else np.isfinite(z["score"][i])
                g, s = z["genes"][i][v].astype(np.int64), z["score"][i][v].astype(np.float64)
                o = g[np.lexsort((g, -s))]
                o = o[~np.isin(o, list(ctd.get(d, set())))]
                t10, t50, allp = set(o[:10].tolist()), set(o[:50].tolist()), set(o.tolist())
                for name, gg in (("union", grp.drop_duplicates("gene_row")),
                                 ("GenCC", grp[grp["source"] == "GenCC"]), ("ClinVar", grp[grp["source"] == "ClinVar"]),
                                 ("independent_evidence", grp[grp["shared_original_evidence"] == "no"].drop_duplicates("gene_row")),
                                 ("no_neighbour_relation", grp[~grp["same_relation_seen"]].drop_duplicates("gene_row"))):
                    q = set(gg["gene_row"].tolist())
                    if not q:
                        continue
                    recs.append({"method": method, "seed": seed, "subset": name, "disease_id": d, "n": len(q),
                                 "recall10": len(q & t10) / len(q), "hit10": float(bool(q & t10)),
                                 "recall50": len(q & t50) / len(q), "in_list": len(q & allp) / len(q)})
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    return df.groupby(["method", "subset", "disease_id"]).agg(
        {"n": "first", "recall10": "mean", "hit10": "mean", "recall50": "mean", "in_list": "mean"}).reset_index()


def run(methods=("deployed", "B", "channel_lgb_v6", "v71_direct_strict", "F7_v6")) -> dict:
    from gpu8v6.common import World6
    from .roles8 import Roles
    if not load_manifest("protocol_v8_registered.json"):
        raise PermissionError("register first")
    OUT.mkdir(parents=True, exist_ok=True)
    w = World6(load_heavy=False)
    roles = Roles(w)
    pairs = pairs_with_fields(w, roles)
    pairs.to_parquet(OUT / "external_pairs_fields.parquet", index=False)
    per = pd.concat([score(w, pairs, m) for m in methods], ignore_index=True)
    per.to_parquet(OUT / "external_per_disease.parquet", index=False)
    tab = per.groupby(["method", "subset"]).agg(n_diseases=("disease_id", "nunique"), n_pairs=("n", "sum"),
                                                recall10=("recall10", "mean"), hit10=("hit10", "mean"),
                                                recall50=("recall50", "mean"), in_list=("in_list", "mean")).reset_index()
    tab.to_csv(OUT / "external_table.csv", index=False, encoding="utf-8-sig")
    fam = w.family_of
    ctr = []
    for a, b in (("deployed", "channel_lgb_v6"), ("deployed", "v71_direct_strict"), ("deployed", "B")):
        for sname in ("union", "GenCC", "ClinVar", "independent_evidence"):
            x = per[(per["method"] == a) & (per["subset"] == sname)].set_index("disease_id")["recall10"]
            y = per[(per["method"] == b) & (per["subset"] == sname)].set_index("disease_id")["recall10"]
            k = x.index.intersection(y.index)
            if len(k) < 5:
                continue
            dlt = (x[k] - y[k]).to_numpy()
            lo, hi = fast_family_bootstrap(dlt, np.array([fam.get(d, d) for d in k]))
            ctr.append({"a": a, "b": b, "subset": sname, "n": int(len(k)), "delta": float(dlt.mean()),
                        "ci_low": lo, "ci_high": hi})
    fields = {"pairs": int(len(pairs)), "diseases": int(pairs["disease_id"].nunique()),
              "by_source": pairs["source"].value_counts().to_dict(),
              "seen_in_training_disease": int(pairs["seen_in_training_disease"].sum()),
              "same_relation_seen": int(pairs["same_relation_seen"].sum()),
              "shared_original_evidence": pairs["shared_original_evidence"].value_counts().to_dict()}
    out = {"generated_utc": now_utc(), "status": "external re-analysis of the V7.1 confirmation set",
           "fields": fields, "table": tab.to_dict(orient="records"), "contrasts": ctr,
           "new_source": "unobtainable this round (declared at registration)"}
    dump_json(OUT / "external618_v8.json", out)
    return out


if __name__ == "__main__":
    paths8.ensure()
    print(json.dumps(run(), indent=1, default=str)[:3000])

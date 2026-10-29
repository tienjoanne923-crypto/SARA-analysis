"""V7.1 evaluation — E1-E8 tables from the authoritative run records.

Every number is read from ``runs_v71/<run>/per_disease.parquet`` (or the archived
``runs_v6`` files for the V6 LightGBM main scorer), computed per run and per
disease first; seeds are averaged inside a disease; macro values are disease
means.  Paired contrasts use disease-family bootstrap intervals (10,000 draws,
vectorised: each draw re-weights whole families with multinomial counts; the
interval is the percentile interval of the family-reweighted mean difference).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8v6 import paths6
from gpu8v6.common import World6

from . import paths71
from .common71 import dump_json, load_manifest, now_utc

SEEDS = (42, 52, 62)
METRICS = ["recall10_end", "hit10", "recall50_end", "ndcg10_end", "coverage", "coverage_final500",
           "oracle_recall10", "zero_coverage", "mrr"]


def fast_family_bootstrap(delta: np.ndarray, fam: np.ndarray, draws: int = 10000, seed: int = 20260923,
                          alpha: float = 0.05) -> tuple[float, float]:
    uniq, inv = np.unique(fam, return_inverse=True)
    sums = np.bincount(inv, weights=delta)
    cnts = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    n = len(uniq)
    out = np.empty(draws)
    for b0 in range(0, draws, 1000):
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=min(1000, draws - b0)).astype(np.float64)
        out[b0:b0 + len(W)] = (W @ sums) / np.maximum(W @ cnts, 1.0)
    return float(np.percentile(out, 100 * alpha / 2)), float(np.percentile(out, 100 * (1 - alpha / 2)))


def run_path(run_id: str):
    p = paths71.RUNS / run_id
    return p if p.exists() else paths6.RUNS / run_id


def load_per(run_id: str) -> pd.DataFrame | None:
    p = run_path(run_id) / "per_disease.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "coverage_final500" not in df.columns and "coverage" in df.columns:
        df["coverage_final500"] = df["coverage"]
    return df


def method_frame(pattern: str, seeds=SEEDS, folds=range(5)) -> pd.DataFrame | None:
    """Per-disease metrics averaged over seeds (pattern uses {j} and optionally {s})."""
    frames = []
    for j in folds:
        for s in (seeds if "{s}" in pattern else [None]):
            rid = pattern.format(j=j, s=s)
            df = load_per(rid)
            if df is None:
                continue
            df = df.assign(fold=j, seed=s if s is not None else "det")
            frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in METRICS if c in df.columns]
    per = df.groupby("disease_id")[cols].mean()
    per["n_runs"] = df.groupby("disease_id").size()
    per["fold"] = df.groupby("disease_id")["fold"].first()
    return per


E1_METHODS = {
    "F7 (V6 frozen fusion)": "V71Z__F7__f{j}",
    "V6 LightGBM (archived V6 main)": "V6Z__lightgbm__f{j}__s{s}",
    "Channel-only LightGBM (strong control)": "V71Z__v6_lgb_channels__f{j}__s{s}",
    "Same-resource fixed fusion": "V71Z__srff__f{j}",
    "Multimodal core alone": "V71Z__core__f{j}__s{s}",
    "Late fusion, fixed": "V71Z__lf_fixed__f{j}__s{s}",
    "Late fusion, LightGBM": "V71Z__lf_lgb__f{j}__s{s}",
}
E2_METHODS = {
    "core (full alignment) alone": ("V71Z__core__f{j}__s{s}", (42,)),
    "core, lambda_A flipped, alone": ("V71Z__core_altanchor__f{j}__s{s}", (42,)),
    "linear two-tower alone": ("V71Z__two_tower__f{j}__s{s}", (42,)),
    "same-resource fixed fusion": ("V71Z__srff__f{j}", (42,)),
    "late fusion LGB, full core": ("V71Z__lf_lgb__f{j}__s{s}", (42,)),
    "late fusion LGB, lambda_A flipped": ("V71Z__e2_altanchor_lf__f{j}__s{s}", (42,)),
    "late fusion LGB, two-tower": ("V71Z__e2_twotower_lf__f{j}__s{s}", (42,)),
    "raw concatenation LGB (same pool)": ("V71Z__e2_rawconcat__f{j}__s{s}", (42,)),
}
E4_CELLS = {
    ("old resources", "old method"): ("V6Z__lightgbm__f{j}__s{s}", (42,)),
    ("new resources", "old method"): ("V71Z__e4_newres_oldmethod__f{j}__s{s}", (42,)),
    ("old resources", "new method"): ("V71Z__e4_oldres_newmethod__f{j}__s{s}", (42,)),
    ("new resources", "new method"): ("V71Z__lf_lgb__f{j}__s{s}", (42,)),
}
E6_METHODS = {
    "V6 500 fixed: channel-only LGB": ("V71Z__v6_lgb_channels__f{j}__s{s}", (42,)),
    "V6 500 fixed: LGB + core features": ("V71Z__e6_v6pool_corefeat__f{j}__s{s}", (42,)),
    "shared pool: F7": ("V71Z__F7_on_pool__f{j}__s{s}", (42,)),
    "shared pool: channel-only LGB": ("V71Z__e6_channels_on_pool__f{j}__s{s}", (42,)),
    "shared pool: late fusion LGB": ("V71Z__lf_lgb__f{j}__s{s}", (42,)),
    "shared pool: mask-only LGB": ("V71Z__e5_maskonly__f{j}__s{s}", (42,)),
}


def summarise(methods: dict, fam: dict, ref: str | None = None) -> tuple[pd.DataFrame, dict]:
    rows, pers = [], {}
    for name, spec in methods.items():
        pat, seeds = (spec, SEEDS) if isinstance(spec, str) else spec
        per = method_frame(pat, seeds)
        if per is None:
            rows.append({"method": name, "status": "not run"})
            continue
        pers[name] = per
        r = {"method": name, "n_diseases": int(len(per)),
             "runs_per_disease": float(per["n_runs"].mean())}
        for c in METRICS:
            if c in per.columns:
                r[c] = float(per[c].mean())
        rows.append(r)
    df = pd.DataFrame(rows)
    return df, pers


def contrast(pers: dict, a: str, b: str, fam: dict, metric: str = "recall10_end") -> dict:
    if a not in pers or b not in pers:
        return {"a": a, "b": b, "status": "missing"}
    j = pers[a][[metric]].join(pers[b][[metric]], lsuffix="_a", rsuffix="_b", how="inner")
    d = (j[f"{metric}_a"] - j[f"{metric}_b"]).to_numpy()
    f = np.array([fam.get(x, x) for x in j.index])
    lo, hi = fast_family_bootstrap(d, f)
    return {"a": a, "b": b, "metric": metric, "n": int(len(d)), "delta": float(d.mean()),
            "ci_low": lo, "ci_high": hi, "share_a_better": float((d > 0).mean()),
            "share_b_better": float((d < 0).mean()), "share_tied": float((d == 0).mean())}


# --------------------------------------------------------------------------
# E3 answer visibility
# --------------------------------------------------------------------------
def visibility_classes(w: World6) -> pd.DataFrame:
    p = paths71.EXT_RES / "answer_visibility_v71.parquet"
    if p.exists():
        return pd.read_parquet(p)
    from gpu8.contexts import build_P
    from gpu8v6.gene_hpo import GeneHPOSource, HPOSpace
    from .chain71 import Contexts
    cx = Contexts(w)
    space = HPOSpace(w.hpo_terms)
    gh = GeneHPOSource(w, space)
    # the V6 definition: degree in the physical PPI network used by the V6 source engine
    ppi = pd.read_parquet(__import__("gpu8").paths.TABLES_DIR / "ppi.parquet")
    _, deg_ppi = build_P(ppi, w.idx)
    rows = []
    for j in range(5):
        ctx = w.outer_context(j)
        ref = cx.v6_reference(ctx)
        from gpu8v6.common import ContextSpec
        cs = ContextSpec(ctx.context_id, ctx.fold, ctx.kind, ctx.block_index, ref, ctx.held_out)
        A = w.adjacency(cs)
        deg = np.asarray(A.sum(axis=0)).ravel()
        G, _ = gh.matrix(set(ref), lenient=False)
        hpo = np.diff(G.indptr) > 0
        for d in w.labelled(w.fold(j)["test_diseases"]):
            for g in w.query_rows(d):
                cls = ("visible_ctd" if deg[g] > 0 else "visible_hpo_only" if hpo[g]
                       else "ppi_only" if deg_ppi[g] > 0 else "unseen")
                rows.append({"disease_id": d, "gene_row": int(g), "fold": j, "class": cls})
    df = pd.DataFrame(rows)
    df.to_parquet(p, index=False)
    return df


def positive_status(run_id: str) -> pd.DataFrame | None:
    """(disease, gene) -> in list/pool, rank in the final ranking (ties by gene row)."""
    p = run_path(run_id) / "predictions.parquet"
    if not p.exists():
        return None
    pr = pd.read_parquet(p)
    if "score" not in pr.columns:
        pr["score"] = -pr["rank"].astype(np.float64)
    pr = pr.sort_values(["disease_id", "score", "gene_row"], ascending=[True, False, True])
    pr["pos"] = pr.groupby("disease_id").cumcount() + 1
    return pr[["disease_id", "gene_row", "pos"]]


def e3_table(w: World6) -> pd.DataFrame:
    vis = visibility_classes(w)
    methods = {"V6 LightGBM": "V6Z__lightgbm__f{j}__s{s}",
               "Channel-only LightGBM": "V71Z__v6_lgb_channels__f{j}__s{s}",
               "Multimodal core alone": "V71Z__core__f{j}__s{s}",
               "Late fusion LightGBM": "V71Z__lf_lgb__f{j}__s{s}",
               "Strict gene-cold core alone": "V71Z__core_strict__f{j}__s42",
               "Strict gene-cold late fusion": "V71Z__e3_strict_lf__f{j}__s42"}
    out = []
    for name, pat in methods.items():
        seeds = SEEDS if "{s}" in pat else [None]
        accs = []
        for s in seeds:
            fr = []
            for j in range(5):
                st = positive_status(pat.format(j=j, s=s))
                if st is not None:
                    fr.append(st)
            if not fr:
                continue
            st = pd.concat(fr)
            m = vis.merge(st, on=["disease_id", "gene_row"], how="left")
            m["in_list"] = m["pos"].notna()
            m["in500"] = m["pos"].fillna(1e9) <= 500
            m["top10"] = m["pos"].fillna(1e9) <= 10
            accs.append(m.groupby("class")[["in_list", "in500", "top10"]].mean())
        if not accs:
            continue
        a = sum(accs) / len(accs)
        a["method"] = name
        out.append(a.reset_index())
    tab = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    share = vis["class"].value_counts(normalize=True).rename("share_of_positives")
    if not tab.empty:
        tab = tab.merge(share, left_on="class", right_index=True)
    return tab


# --------------------------------------------------------------------------
# E6 hit flow
# --------------------------------------------------------------------------
def top10_sets(run_id: str) -> dict:
    st = positive_status(run_id)
    if st is None:
        return {}
    t = st[st["pos"] <= 10]
    return {d: set(g) for d, g in t.groupby("disease_id")["gene_row"]}


def hit_flow(w: World6, new_pat: str, old_pat: str, seeds=SEEDS) -> dict:
    rows = []
    for s in seeds:
        for j in range(5):
            a, b = top10_sets(new_pat.format(j=j, s=s)), top10_sets(old_pat.format(j=j, s=s))
            if not a or not b:
                continue
            for d in w.labelled(w.fold(j)["test_diseases"]):
                q = set(w.query_rows(d).tolist())
                ha, hb = a.get(d, set()) & q, b.get(d, set()) & q
                rows.append({"seed": s, "disease_id": d, "new_hits": len(ha - hb),
                             "lost_hits": len(hb - ha), "kept_hits": len(ha & hb),
                             "n_query": len(q)})
    if not rows:
        return {"status": "missing"}
    df = pd.DataFrame(rows)
    per = df.groupby("disease_id")[["new_hits", "lost_hits", "kept_hits", "n_query"]].mean()
    return {"new_hits_total_per_seed": float(df.groupby("seed")["new_hits"].sum().mean()),
            "lost_hits_total_per_seed": float(df.groupby("seed")["lost_hits"].sum().mean()),
            "kept_hits_total_per_seed": float(df.groupby("seed")["kept_hits"].sum().mean()),
            "diseases_gaining": float((per["new_hits"] > per["lost_hits"]).mean()),
            "diseases_losing": float((per["new_hits"] < per["lost_hits"]).mean()),
            "n_diseases": int(len(per))}


# --------------------------------------------------------------------------
# E7 confirmation set (filtered)
# --------------------------------------------------------------------------
def confirmation(w: World6) -> pd.DataFrame:
    conf = pd.read_parquet(paths71.TABLES / "confirmation_set_v71.parquet")
    lab = conf.groupby("disease_id")["gene_row"].apply(set).to_dict()
    fold_of = {d: j for j in range(5) for d in w.labelled(w.fold(j)["test_diseases"])}
    methods = {k: v for k, v in E1_METHODS.items()}
    rows = []
    fam = w.family_of
    per_all = {}
    for name, pat in methods.items():
        seeds = SEEDS if "{s}" in pat else [None]
        recs = []
        for s in seeds:
            for j in range(5):
                st = positive_status(pat.format(j=j, s=s)) if True else None
                if st is None:
                    continue
                st = st[st["disease_id"].isin(lab)]
                for d, sub in st.groupby("disease_id"):
                    if fold_of.get(d) != j:
                        continue
                    ctd = set(w.query_rows(d).tolist())
                    sub = sub[~sub["gene_row"].isin(ctd)].sort_values("pos")
                    top = sub["gene_row"].to_numpy()
                    q = lab[d]
                    h10 = len(set(top[:10].tolist()) & q)
                    h50 = len(set(top[:50].tolist()) & q)
                    recs.append({"disease_id": d, "seed": s, "recall10": h10 / len(q),
                                 "hit10": float(h10 > 0), "recall50": h50 / len(q),
                                 "in_list": len(set(top.tolist()) & q) / len(q)})
        if not recs:
            continue
        df = pd.DataFrame(recs).groupby("disease_id").mean(numeric_only=True)
        df = df.reindex(sorted(lab)).fillna(0.0)          # a disease without a list scores 0
        per_all[name] = df
        rows.append({"method": name, "n_diseases": int(len(df)), "n_pairs": int(sum(len(v) for v in lab.values())),
                     "recall10": float(df["recall10"].mean()), "hit10": float(df["hit10"].mean()),
                     "recall50": float(df["recall50"].mean()), "coverage_in_list": float(df["in_list"].mean())})
    tab = pd.DataFrame(rows)
    ctr = []
    for a, b in (("Late fusion, LightGBM", "Channel-only LightGBM (strong control)"),
                 ("Late fusion, LightGBM", "V6 LightGBM (archived V6 main)"),
                 ("Late fusion, LightGBM", "F7 (V6 frozen fusion)"),
                 ("Multimodal core alone", "Same-resource fixed fusion")):
        if a in per_all and b in per_all:
            dlt = (per_all[a]["recall10"] - per_all[b]["recall10"]).to_numpy()
            f = np.array([fam.get(x, x) for x in per_all[a].index])
            lo, hi = fast_family_bootstrap(dlt, f)
            ctr.append({"a": a, "b": b, "delta_recall10": float(dlt.mean()), "ci_low": lo, "ci_high": hi})
    dump_json(paths71.EXT_RES / "confirmation_contrasts_v71.json", ctr)
    return tab


# --------------------------------------------------------------------------
def run_all() -> dict:
    w = World6(load_heavy=False)
    fam = w.family_of
    out = {"generated_utc": now_utc()}
    t1, p1 = summarise(E1_METHODS, fam)
    t1.to_csv(paths71.TABLES_OUT / "e1_main.csv", index=False, encoding="utf-8-sig")
    c1 = [contrast(p1, a, b, fam) for a, b in (
        ("Late fusion, LightGBM", "Channel-only LightGBM (strong control)"),
        ("Late fusion, LightGBM", "F7 (V6 frozen fusion)"),
        ("Late fusion, LightGBM", "V6 LightGBM (archived V6 main)"),
        ("Late fusion, fixed", "F7 (V6 frozen fusion)"),
        ("Multimodal core alone", "Same-resource fixed fusion"),
        ("Channel-only LightGBM (strong control)", "V6 LightGBM (archived V6 main)"),
        ("Late fusion, LightGBM", "Late fusion, fixed"))]
    c1 += [contrast(p1, "Late fusion, LightGBM", "Channel-only LightGBM (strong control)", fam, m)
           for m in ("hit10", "recall50_end", "coverage_final500")]
    pd.DataFrame(c1).to_csv(paths71.TABLES_OUT / "e1_contrasts.csv", index=False, encoding="utf-8-sig")
    out["E1"] = {"table": t1.to_dict(orient="records"), "contrasts": c1}
    t2, p2 = summarise(E2_METHODS, fam)
    t2.to_csv(paths71.TABLES_OUT / "e2_same_data.csv", index=False, encoding="utf-8-sig")
    c2 = [contrast(p2, "late fusion LGB, full core", b, fam) for b in
          ("late fusion LGB, lambda_A flipped", "late fusion LGB, two-tower", "raw concatenation LGB (same pool)")]
    c2 += [contrast(p2, "core (full alignment) alone", b, fam) for b in
           ("core, lambda_A flipped, alone", "linear two-tower alone", "same-resource fixed fusion")]
    out["E2"] = {"table": t2.to_dict(orient="records"), "contrasts": c2}
    t3 = e3_table(w)
    t3.to_csv(paths71.TABLES_OUT / "e3_visibility.csv", index=False, encoding="utf-8-sig")
    out["E3"] = t3.to_dict(orient="records")
    t4, p4 = summarise({f"{a} / {b}": v for (a, b), v in E4_CELLS.items()}, fam)
    t4.to_csv(paths71.TABLES_OUT / "e4_resource_method.csv", index=False, encoding="utf-8-sig")
    out["E4"] = t4.to_dict(orient="records")
    beh = paths71.BEHAVIOR / "e5_behaviour_test.csv"
    if beh.exists():
        b = pd.read_csv(beh)
        e5 = b.groupby(["perturbation", "modality"])[["lf_recall10", "core_recall10", "lf_drop",
                                                      "core_drop"]].mean().reset_index()
        # family-bootstrap intervals of the per-disease drops (draw-averaged permutations)
        per = pd.concat([pd.read_parquet(p) for p in sorted(paths71.BEHAVIOR.glob("e5_per_disease_f*.parquet"))],
                        ignore_index=True)
        f = np.array([fam.get(d, d) for d in per["disease_id"]])
        cis = []
        for r in e5.itertuples():
            if r.perturbation == "none":
                cis.append((np.nan, np.nan, np.nan, np.nan))
                continue
            out_ci = []
            for which in ("lf", "core"):
                d = (per[f"none|-|{which}"] - per[f"{r.perturbation}|{r.modality}|{which}"]).to_numpy()
                out_ci.extend(fast_family_bootstrap(d, f))
            cis.append(tuple(out_ci))
        e5[["lf_drop_ci_low", "lf_drop_ci_high", "core_drop_ci_low", "core_drop_ci_high"]] = pd.DataFrame(
            cis, index=e5.index)
        e5.to_csv(paths71.TABLES_OUT / "e5_behaviour.csv", index=False, encoding="utf-8-sig")
        out["E5"] = e5.to_dict(orient="records")
    t6, p6 = summarise(E6_METHODS, fam)
    t6.to_csv(paths71.TABLES_OUT / "e6_attribution.csv", index=False, encoding="utf-8-sig")
    flow = {"lf_vs_v6_lightgbm": hit_flow(w, "V71Z__lf_lgb__f{j}__s{s}", "V6Z__lightgbm__f{j}__s{s}"),
            "lf_vs_channel_lgb": hit_flow(w, "V71Z__lf_lgb__f{j}__s{s}",
                                          "V71Z__v6_lgb_channels__f{j}__s{s}")}
    dump_json(paths71.TABLES_OUT / "e6_hit_flow.json", flow)
    out["E6"] = {"table": t6.to_dict(orient="records"), "hit_flow": flow}
    t7 = confirmation(w)
    t7.to_csv(paths71.TABLES_OUT / "e7_confirmation.csv", index=False, encoding="utf-8-sig")
    out["E7"] = {"confirmation": t7.to_dict(orient="records")}
    ss = paths71.TABLES_OUT / "e7_small_support.csv"
    if ss.exists():
        out["E7"]["small_support"] = pd.read_csv(ss).to_dict(orient="records")
    e8 = {"core (s42) alone": ("V71Z__core__f{j}__s{s}", (42,)),
          "core + image": ("V71Z__core_image__f{j}__s{s}", (42,)),
          "core + localisation labels (image control)": ("V71Z__core_imgloc__f{j}__s{s}", (42,)),
          "core + cell-line-centred image": ("V71Z__core_imgcentred__f{j}__s{s}", (42,)),
          "core + mouse (IMPC)": ("V71Z__core_mouse__f{j}__s{s}", (42,)),
          "core, background controls (sensitivity)": ("V71Z__core_bgctrl__f{j}__s{s}", (42,))}
    t8, p8 = summarise(e8, fam)
    t8.to_csv(paths71.TABLES_OUT / "e8_optional.csv", index=False, encoding="utf-8-sig")
    c8 = [contrast(p8, a, "core (s42) alone", fam) for a in list(e8)[1:]]
    out["E8"] = {"table": t8.to_dict(orient="records"), "contrasts": c8}
    dump_json(paths71.TABLES_OUT / "v71_results.json", out)
    return out


if __name__ == "__main__":
    paths71.ensure()
    res = run_all()
    print(json.dumps({k: v for k, v in res.items() if k in ("E1",)}, indent=2, default=str)[:4000])

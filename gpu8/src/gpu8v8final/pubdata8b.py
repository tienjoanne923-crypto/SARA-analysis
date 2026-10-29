"""Second-round source data for the top-tier figure set (descriptive, from frozen predictions and models).

* SHAP (TreeSHAP via LightGBM ``pred_contrib``) of the deployed candidate A on the reconstructed test
  feature matrix of outer context 0 (seed 42); the reconstruction is verified against the saved scores.
* disease-category knowledge matrices (gene sharing between MeSH branches; per-branch indicators);
* per-disease best-positive ranks of every method; masked-gene positive ranks under the E5/E6 arms;
* E4 difference-of-differences by disease category; disease-gene networks of the case studies.
Outputs: ``results_v8/pub/*.parquet|json``.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import paths8
from .common8 import dump_json, now_utc
from .pubdata8 import MESH_CAT, OUT, disease_table, gene_table

SEEDS = (42, 52, 62)


# ------------------------------------------------------------------ SHAP of the deployed model
def shap_fold(j: int = 0, seed: int = 42, n_neg: int = 40000, device: str = "cuda", force: bool = False) -> dict:
    p = OUT / f"shap_f{j}_s{seed}.parquet"
    if p.exists() and not force:
        return json.loads((OUT / f"shap_f{j}_s{seed}_meta.json").read_text(encoding="utf-8"))
    import lightgbm as lgb
    from gpu8v6.scoring import FEATS

    from .chain8 import Chain
    from .evaluate8 import load_pred
    from .feats8 import EXTRA, Labels
    from .formal8 import sel_cfg, selection
    from .pack8 import ArmSpec
    from .teacher8 import teacher_run_id
    from .v6pass8 import V6Pass, V6World
    t0 = time.time()
    ch = Chain(ArmSpec(), V6World(), device=device, role="analysis")
    ctx = f"f{j}_outer"
    sel = selection(j)
    cfg = sel_cfg(sel)
    v6 = V6Pass(V6World(), ctx)
    sem = ch.semantic(ctx, teacher_run_id("main", ctx, sel["controls"], seed), v6, None)
    blk = ch.blocks_for(ctx, v6, sem, sel["w"]["w"])["test"]
    del sem
    X = np.concatenate([blk.XB, blk.XE_nan()], axis=2)
    names = list(FEATS) + list(EXTRA)
    mp = (paths8.DEV / f"fold{j}" / f"A_{cfg['A_name']}_model.txt") if seed == 42 else \
        (paths8.FORMAL / "main" / f"s{seed}" / f"fold{j}" / f"A_{cfg['A_name']}_model.txt")
    b = lgb.Booster(model_str=mp.read_text(encoding="utf-8"))
    D, N, F = X.shape
    s = b.predict(X.reshape(-1, F)).reshape(D, N)
    saved = load_pred(paths8.FORMAL / "main" / f"s{seed}" / f"fold{j}" / "pred_deployed__test.npz")
    assert [str(x) for x in saved["ids"]] == list(blk.ids)
    assert np.array_equal(saved["genes"].astype(np.int64), blk.genes.astype(np.int64))
    ok = blk.valid & np.isfinite(saved["score"])
    maxdiff = float(np.max(np.abs(s[ok] - saved["score"][ok])))
    lab = Labels(ch.w)
    Y = np.zeros((D, N), dtype=bool)
    for i, d in enumerate(blk.ids):
        q = set(lab.rows(d, "test").tolist())
        Y[i] = np.isin(blk.genes[i], list(q))
    rng = np.random.default_rng(20260926)
    pos_idx = np.flatnonzero((Y & ok).ravel())
    neg_all = np.flatnonzero((~Y & ok).ravel())
    neg_idx = rng.choice(neg_all, size=min(n_neg, len(neg_all)), replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    Xs = X.reshape(-1, F)[idx]
    contrib = b.predict(Xs, pred_contrib=True)
    rank = np.argsort(np.argsort(-np.where(ok, s, -np.inf), axis=1, kind="stable"), axis=1) + 1
    df = pd.DataFrame(Xs, columns=[f"x_{n}" for n in names])
    for k, n in enumerate(names):
        df[f"shap_{n}"] = contrib[:, k]
    df["shap_base"] = contrib[:, -1]
    di, gi = np.unravel_index(idx, (D, N))
    df["disease_id"] = [blk.ids[i] for i in di]
    df["gene_row"] = blk.genes[di, gi]
    df["positive"] = Y.ravel()[idx]
    df["cold"] = blk.cold[di, gi]
    df["src"] = blk.src[di, gi]
    df["score"] = s[di, gi]
    df["rank"] = rank[di, gi]
    df.to_parquet(p, index=False)
    meta = {"generated_utc": now_utc(), "fold": j, "seed": seed, "model": cfg["A_name"], "rows_total": int(ok.sum()),
            "positives": int(len(pos_idx)), "negatives_sampled": int(len(neg_idx)),
            "reconstruction_maxabs_vs_saved_scores": maxdiff, "seconds": time.time() - t0,
            "method": "LightGBM TreeSHAP (pred_contrib); features = 30 V6 channels + 19 availability/semantic"}
    dump_json(OUT / f"shap_f{j}_s{seed}_meta.json", meta)
    return meta


# ------------------------------------------------------------------ disease-category knowledge matrices
def category_data() -> dict:
    from gpu8v6.common import World6
    w = World6(load_heavy=False)
    dis = disease_table().set_index("disease_id")
    genes = gene_table().set_index("gene_id")
    rel = w.primary_relations[["disease_id", "gene_id"]].copy()
    rel["gene_id"] = rel["gene_id"].astype(str)
    rel["cats"] = rel["disease_id"].map(dis["cats"]).fillna("")
    rows, members = [], {}
    for code, name in MESH_CAT.items():
        sub = rel[rel["cats"].str.contains(code)]
        if sub["disease_id"].nunique() < 40:
            continue
        gs = set(sub["gene_id"])
        members[code] = gs
        dd = dis.loc[dis.index.intersection(sub["disease_id"].unique())]
        gg = genes.loc[genes.index.intersection(list(gs))]
        rows.append({"code": code, "name": name, "diseases": int(dd.shape[0]), "genes": len(gs),
                     "relations": int(len(sub)), "rel_per_disease": float(sub.groupby("disease_id").size().median()),
                     "def_share": float(dd["has_def"].mean()), "hpo_share": float((dd["n_hpo"] > 0).mean()),
                     "tokens_median": float(dd["n_tokens"].median()),
                     "gene_go_median": float(gg["n_go_pairs"].median()), "gene_rel_median": float(gg["n_rel"].median()),
                     "gene_ppi_share": float(gg["has_ppi"].mean()), "gene_nogo_share": float((~gg["has_go"]).mean()),
                     "gene_ncrna_share": float((gg["gtype"] == "ncrna").mean())})
    ind = pd.DataFrame(rows)
    codes = ind.sort_values("genes", ascending=False)["code"].tolist()
    M = np.array([[len(members[a] & members[b]) for b in codes] for a in codes])
    return {"indicators": ind, "codes": codes, "shared": M}


# ------------------------------------------------------------------ per-disease best-positive ranks
def best_ranks() -> pd.DataFrame:
    qr = pd.read_parquet(OUT / "query_ranks.parquet")
    r = qr["rank"].astype(float).where(qr["rank"] > 0, np.nan)
    qr = qr.assign(r=r, rr=(1.0 / r).fillna(0.0))
    g = qr.groupby(["seed", "method", "disease_id"]).agg(best=("r", "min"), mrr=("rr", "mean"), n=("r", "size"),
                                                          listed=("r", "count")).reset_index()
    g["best"] = g["best"].fillna(501.0)
    return g


# ------------------------------------------------------------------ E5/E6: masked-gene positive ranks per arm
E5_ARMS = ["main", "e5_none", "e5_mlp", "e5_linear", "e5_nn", "e5_teacher_sS", "e5_mlp_shuffled", "e5_protst",
           "e6_cold", "e6_cold_gomask"]


def e5_ranks() -> pd.DataFrame:
    from gpu8v6.common import World6

    from .evaluate8 import load_pred
    from .feats8 import Labels
    from .masks8 import load as load_masks
    w = World6(load_heavy=False)
    lab = Labels(w)
    mt = set(load_masks()["test"].tolist())
    rows = []
    for arm in E5_ARMS:
        base = paths8.FORMAL / "main" if arm == "main" else paths8.VARIANTS / arm
        for j in range(5):
            fp = base / "s42" / f"fold{j}" / "pred_deployed__test.npz"
            if not fp.exists():
                continue
            pred = load_pred(fp)
            for i, d in enumerate(pred["ids"]):
                d = str(d)
                q = [g for g in lab.rows(d, "test").tolist() if g in mt]
                if not q:
                    continue
                v = np.isfinite(pred["score"][i])
                gg = pred["genes"][i][v].astype(np.int64)
                ss = pred["score"][i][v].astype(np.float64)
                o = gg[np.lexsort((gg, -ss))]
                pos = {int(x): k + 1 for k, x in enumerate(o.tolist())}
                for g in q:
                    rows.append((arm, j, d, int(g), pos.get(int(g), 0)))
    return pd.DataFrame(rows, columns=["arm", "fold", "disease_id", "gene_row", "rank"])


# ------------------------------------------------------------------ E4 difference-of-differences by category
E4_TAGS = ["e4_layer1", "e4_layer1_random", "e4_layer2", "e4_layer2_random", "e4_blocks", "e4_blocks_random",
           "e4_experimental", "e4_no_tas_nas", "e4_no_mutant", "e4_rescore_no_ic", "e4_rescore_iea_only"]


def e4_category() -> pd.DataFrame:
    ev = paths8.RESULTS / "eval"
    main = pd.read_parquet(ev / "eval_test_main.parquet")
    m42 = main[main["seed"] == 42]
    nf = m42[m42["method"] == "deployed"].set_index("disease_id")["recall10"]
    of = m42[m42["method"] == "channel_lgb_v6"].set_index("disease_id")["recall10"]
    dis = disease_table().set_index("disease_id")
    rows = []
    for tag in E4_TAGS:
        df = pd.read_parquet(ev / f"eval_test_{tag}.parquet")
        na = df[df["method"] == "deployed"].groupby("disease_id")["recall10"].mean()
        oa = (df[df["method"] == "channel_lgb_v6"].groupby("disease_id")["recall10"].mean()
              if "channel_lgb_v6" in set(df["method"]) else of)
        d = ((na - oa.reindex(na.index)) - (nf.reindex(na.index) - of.reindex(na.index))).dropna()
        cats = dis.reindex(d.index)["cats"].fillna("")
        for code, name in MESH_CAT.items():
            m = cats.str.contains(code)
            if m.sum() < 60:
                continue
            rows.append({"arm": tag, "code": code, "name": name, "n": int(m.sum()), "dod": float(d[m].mean())})
        rows.append({"arm": tag, "code": "ALL", "name": "全部", "n": int(len(d)), "dod": float(d.mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ case networks
def case_networks() -> dict:
    from gpu8v6.common import World6
    man = json.loads((paths8.MANUSCRIPT / "figures_v2" / "figures_manifest.json").read_text(encoding="utf-8"))
    cases = [c["disease_id"] for c in man["fig10"]["cases"]]
    rows = pd.read_csv(paths8.MANUSCRIPT / "figures_v2" / "source_data" / "Fig10_case_studies_a.csv", encoding="utf-8-sig")
    qr = pd.read_parquet(OUT / "query_ranks.parquet")
    q42 = qr[(qr["seed"] == 42) & qr["method"].isin(["deployed", "channel_lgb_v6"])]
    w = World6(load_heavy=True)
    genes = gene_table().set_index("gene_row")
    ppi = w.ppi.copy()
    ppi["a"] = ppi["gene_a"].astype(str).map(w.idx.g_ix)
    ppi["b"] = ppi["gene_b"].astype(str).map(w.idx.g_ix)
    ppi = ppi.dropna(subset=["a", "b"])
    rec = pd.read_parquet(paths8.TABLES / "go_records_current_v8.parquet", columns=["gene_row", "go_id"])
    out = {}
    for d in cases:
        r = rows[rows["disease_id"] == d]
        pos = set(q42[(q42["disease_id"] == d) & (q42["method"] == "deployed")]["gene_row"].astype(int))
        ranks = q42[q42["disease_id"] == d].pivot_table(index="gene_row", columns="method", values="rank", aggfunc="first")
        nodes = sorted(set(r["gene_row"].astype(int)) | pos)
        go_sets = rec[rec["gene_row"].isin(nodes)].groupby("gene_row")["go_id"].apply(set).to_dict()
        node_rows = []
        for g in nodes:
            in_dep = r[(r["method"] == "deployed") & (r["gene_row"] == g)]
            in_ch = r[(r["method"] == "channel_lgb_v6") & (r["gene_row"] == g)]
            node_rows.append({"gene_row": int(g), "symbol": str(genes.loc[g, "symbol"]), "positive": g in pos,
                              "top10_deployed": int(in_dep["rank"].iloc[0]) if len(in_dep) else None,
                              "top10_channel": int(in_ch["rank"].iloc[0]) if len(in_ch) else None,
                              "rank_deployed": int(ranks.loc[g, "deployed"]) if g in ranks.index and pd.notna(
                                  ranks.loc[g].get("deployed")) else None,
                              "rank_channel": int(ranks.loc[g, "channel_lgb_v6"]) if g in ranks.index and pd.notna(
                                  ranks.loc[g].get("channel_lgb_v6")) else None,
                              "cold": bool(r[r["gene_row"] == g]["reference_path_cold"].any()),
                              "n_go": int(genes.loc[g, "n_go_pairs"]), "n_rel": int(genes.loc[g, "n_rel"])})
        ns = set(nodes)
        e = ppi[ppi["a"].isin(ns) & ppi["b"].isin(ns)]
        edges = [{"a": int(x.a), "b": int(x.b), "kind": x.network_type, "score": int(x.score)} for x in e.itertuples()]
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                sa, sb = go_sets.get(a, set()), go_sets.get(b, set())
                if sa and sb:
                    jac = len(sa & sb) / len(sa | sb)
                    if jac >= 0.15:
                        edges.append({"a": int(a), "b": int(b), "kind": "go", "score": float(jac)})
        out[d] = {"nodes": node_rows, "edges": edges}
    return out


# ------------------------------------------------------------------ historical (X2 / X1) per-pair ranks
def historical_ranks(seed: int = 42) -> pd.DataFrame:
    """Rank of every X2 / X1 test pair in each saved historical list, ordered exactly as hist8.evaluate
    (score desc, GeneID-row tie-break; X2 lists drop the disease's cut-off genes).  0 = not listed."""
    from .temporal8 import HistWorld, tasks
    d = paths8.EXT_RES / "historical" / f"s{seed}"
    hw = HistWorld()
    tk = tasks(hw)
    g_ix = hw.idx.g_ix
    cur = tk["current_positives"]
    old = {k: {g_ix[g] for g in v if g in g_ix} for k, v in hw.positives.items()}
    rows = []
    for task in ("x2", "x1"):
        for m in ("deployed", "channel_lgb_v6", "standalone_sF", "B"):
            z = np.load(d / f"pred_{m}__{task}.npz")
            ids = [str(x) for x in z["ids"]]
            for i, dd in enumerate(ids):
                allq = {g_ix[g] for g in cur.get(dd, []) if g in g_ix}
                q = allq - old.get(dd, set()) if task == "x2" else allq
                if not q:
                    continue
                v = z["valid"][i] if "valid" in z.files else np.isfinite(z["score"][i])
                g, s = z["genes"][i][v].astype(np.int64), z["score"][i][v].astype(np.float64)
                o = g[np.lexsort((g, -s))]
                if task == "x2":
                    o = o[~np.isin(o, list(old.get(dd, set())))]
                pos = {int(x): k + 1 for k, x in enumerate(o.tolist())}
                for gq in sorted(q):
                    rows.append({"task": task, "method": m, "disease_id": dd, "gene_row": int(gq),
                                 "rank": int(pos.get(int(gq), 0))})
    df = pd.DataFrame(rows)
    # consistency with the saved per-disease evaluation
    hp = pd.read_parquet(d / "historical_per_disease.parquet")
    chk = (df.assign(hit=df["rank"].between(1, 10)).groupby(["task", "method", "disease_id"])["hit"].mean()
           .rename("r10").reset_index().merge(hp, on=["task", "method", "disease_id"]))
    assert np.allclose(chk["r10"], chk["recall10"]), "historical ranks disagree with the saved evaluation"
    return df


def build(shap: bool = True) -> dict:
    info = {"generated_utc": now_utc()}
    cd = category_data()
    cd["indicators"].to_parquet(OUT / "category_indicators.parquet", index=False)
    dump_json(OUT / "category_shared_genes.json", {"codes": cd["codes"], "shared": cd["shared"].tolist()})
    best_ranks().to_parquet(OUT / "best_ranks.parquet", index=False)
    e5_ranks().to_parquet(OUT / "e5_ranks.parquet", index=False)
    e4_category().to_parquet(OUT / "e4_category.parquet", index=False)
    dump_json(OUT / "case_networks.json", case_networks())
    if shap:
        info["shap"] = shap_fold(0, 42)
    dump_json(OUT / "build_info_b.json", info)
    return info


if __name__ == "__main__":
    import sys
    paths8.ensure()
    if "hist" in sys.argv:
        h = historical_ranks()
        h.to_parquet(OUT / "historical_ranks.parquet", index=False)
        print(h.groupby(["task", "method"])["rank"].agg(["size", lambda r: (r > 0).mean()]))
    else:
        print(json.dumps(build("noshap" not in sys.argv), ensure_ascii=False, indent=1, default=str)[:3000])

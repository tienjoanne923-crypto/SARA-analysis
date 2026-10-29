"""Extra registered analyses (after registration; evaluation-side only).

* two_tower     the V7.1 linear two-tower (same inputs, no types / gate / anchors), outer
                contexts, seed 42; standalone full-background lists (semantic-method comparator)
* protst_zs     ProtST zero-shot text-protein retrieval (task adaptation / open knowledge)
* zero_signal   test queries whose gene has no V6 signal and no semantics in the scoring
                context (cannot be recovered by any channel) - the remaining zero-signal set
* strata        therapeutic-only relations (not labels), single-relation diseases, OMIM-mapped
                diseases, positive burden (evaluation strata only; never routing inputs)
* hitflow       top-10 hits kept / lost / gained between the strong control and V8.1
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from . import paths8
from .common8 import dump_json, now_utc

OUT = paths8.RESULTS / "analysis"


def two_tower(seed: int = 42) -> dict:
    from gpu8v71.core71 import CoreConfig
    from gpu8v71.train71 import CoreTrainer, TrainConfig
    from .chain8 import Chain
    from .comparators8 import standalone_top
    from .formal8 import out_dir, selection
    from .pack8 import ArmSpec
    from .teacher8 import teacher_data
    from .v6pass8 import V6World
    ch = Chain(ArmSpec(), V6World(), role="comparator")
    res = {}
    for j in range(5):
        d = out_dir("two_tower", seed, j)
        if (d / "done.json").exists():
            continue
        sel = selection(j)
        data = teacher_data(ch.w, ch.roles, f"f{j}_outer")
        tr = CoreTrainer(ch.T, ch.device)
        fit = tr.fit(data.train, data.controls, data.val, seed=seed, kind="two_tower",
                     core_cfg=CoreConfig(rank=128), cfg=TrainConfig(lambda_A=0.0, controls=sel["controls"]))
        m = tr.load("two_tower", CoreConfig(rank=128), fit.state)
        for role in ("dev", "test"):
            ids = [x for x in ch.roles.targets(ch.w.contexts[f"f{j}_outer"])[role] if x in ch.w.idx.d_ix]
            rows = np.array([ch.w.idx.d_ix[x] for x in ids])
            R, V = tr.score(m, rows)
            top = standalone_top(R, V, 500)
            sc = np.where(top >= 0, np.take_along_axis(R, np.maximum(top, 0), axis=1), -np.inf)
            np.savez_compressed(d / f"pred_standalone_twotower__{role}.npz", ids=np.array(ids),
                                genes=np.maximum(top, 0).astype(np.int16), score=sc.astype(np.float32), valid=top >= 0)
        dump_json(d / "done.json", {"fold": j, "seed": seed, "best_epoch": fit.best_epoch,
                                    "val_r_recall10": fit.best_val_recall10, "n_params": fit.n_params,
                                    "generated_utc": now_utc()})
        res[j] = fit.best_val_recall10
        print(f"  two-tower fold {j}: ep {fit.best_epoch} val {fit.best_val_recall10:.4f}", flush=True)
    return res


def protst_zero_shot() -> dict:
    from gpu8v6.common import World6
    from .comparators8 import standalone_top
    from .formal8 import out_dir
    from .roles8 import Roles
    w = World6(load_heavy=False)
    roles = Roles(w)
    enc = paths8.CACHES / "encoders"
    P = np.load(enc / "protst_protein512.npy")
    T = np.load(enc / "protst_disease_text512.npy")
    import pandas as pd
    from gpu8 import paths as p5
    ids_all = pd.read_parquet(paths8.V71_TABLES / "disease_templates_v71.parquet")["disease_id"].tolist()
    tpos = {d: i for i, d in enumerate(ids_all)}
    hasP = np.linalg.norm(P, axis=1) > 0
    Pn = torch.as_tensor(P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12), device="cuda")
    out = {}
    for j in range(5):
        d = out_dir("protst_zeroshot", 42, j)
        if (d / "done.json").exists():
            continue
        for role in ("dev", "test"):
            ids = [x for x in roles.targets(w.contexts[f"f{j}_outer"])[role] if x in tpos]
            Tx = T[[tpos[x] for x in ids]]
            Tn = torch.as_tensor(Tx / np.maximum(np.linalg.norm(Tx, axis=1, keepdims=True), 1e-12), device="cuda")
            R = (Tn @ Pn.T).float().cpu().numpy()
            V = np.repeat(hasP[None, :], len(ids), axis=0) & (np.linalg.norm(Tx, axis=1) > 0)[:, None]
            top = standalone_top(R, V, 500)
            sc = np.where(top >= 0, np.take_along_axis(R, np.maximum(top, 0), axis=1), -np.inf)
            np.savez_compressed(d / f"pred_protst_zeroshot__{role}.npz", ids=np.array(ids),
                                genes=np.maximum(top, 0).astype(np.int16), score=sc.astype(np.float32), valid=top >= 0)
        dump_json(d / "done.json", {"fold": j, "rule": "cos(ProtST text feature of the disease template, ProtST "
                                                      "protein feature); proteins only; open-knowledge encoder",
                                    "generated_utc": now_utc()})
        out[j] = "done"
    return out


def zero_signal_and_strata() -> dict:
    """Evaluation-side strata of the F_j queries (after registration)."""
    from gpu8v6.common import World6
    from .feats8 import Labels
    from .v6pass8 import V6Pass, V6World
    w = World6(load_heavy=False)
    lab = Labels(w)
    OUT.mkdir(parents=True, exist_ok=True)
    rel = w.relations
    ther = rel[rel["relation_type"] == "therapeutic"]
    prim = {(d, g) for d, g in zip(w.primary_relations["disease_id"], w.primary_relations["gene_id"].astype(str))}
    ther_only = ther[[(d, str(g)) not in prim for d, g in zip(ther["disease_id"], ther["gene_id"])]]
    ther_by_d = ther_only.groupby("disease_id")["gene_id"].apply(lambda s: {w.idx.g_ix[str(g)] for g in s
                                                                            if str(g) in w.idx.g_ix}).to_dict()
    omim = dict(zip(w.diseases["disease_id"], w.diseases["omim_ids"].fillna("").astype(str).str.len() > 0))
    rows, zs = [], []
    for seed in (42, 52, 62):
        for j in range(5):
            d = paths8.FORMAL / "main" / f"s{seed}" / f"fold{j}"
            if not (d / "pred_deployed__test.npz").exists():
                continue
            v6p = V6Pass(V6World(), f"f{j}_outer")
            src = np.load(d / f"gene_src_f{j}_outer.npy")
            for meth in ("deployed", "channel_lgb_v6", "v71_direct_strict"):
                p = d / f"pred_{meth}__test.npz"
                if not p.exists():
                    continue
                z = np.load(p)
                for i, dd in enumerate(z["ids"]):
                    dd = str(dd)
                    g, s = z["genes"][i].astype(np.int64), z["score"][i].astype(np.float64)
                    v = np.isfinite(s)
                    o = g[v][np.lexsort((g[v], -s[v]))]
                    top = set(o[:10].tolist())
                    q = lab.rows(dd, "test")
                    qs = set(q.tolist())
                    rec = {"seed": seed, "fold": j, "method": meth, "disease_id": dd, "n_query": len(qs),
                           "recall10": len(qs & top) / max(len(qs), 1), "single_relation": len(qs) == 1,
                           "omim_mapped": bool(omim.get(dd, False))}
                    tq = ther_by_d.get(dd, set())
                    rec["n_therapeutic_only"] = len(tq)
                    rec["therapeutic_hits10"] = len(tq & top)
                    rows.append(rec)
                    if meth == "deployed" and seed == 42 and len(q):
                        fu = np.asarray(v6p.fused[v6p.pos[dd]])
                        for gq in q:
                            zs.append({"fold": j, "disease_id": dd, "gene_row": int(gq),
                                       "v6_signal": bool(fu[gq] > 0), "semantic": bool(src[gq] != 0),
                                       "in_pool": gq in set(g.tolist()), "in_top10": gq in top})
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "strata_per_disease.parquet", index=False)
    zsd = pd.DataFrame(zs)
    zsd.to_parquet(OUT / "zero_signal_queries_s42.parquet", index=False)
    per = df.groupby(["method", "seed", "disease_id"]).first().reset_index()
    summ = {}
    for meth, sub in per.groupby("method"):
        a = sub.groupby("disease_id").agg(recall10=("recall10", "mean"), single=("single_relation", "first"),
                                          omim=("omim_mapped", "first"), n=("n_query", "first"),
                                          ther_n=("n_therapeutic_only", "first"), ther_h=("therapeutic_hits10", "mean"))
        summ[meth] = {"single_relation_recall10": float(a.loc[a["single"], "recall10"].mean()),
                      "multi_relation_recall10": float(a.loc[~a["single"], "recall10"].mean()),
                      "omim_mapped_recall10": float(a.loc[a["omim"], "recall10"].mean()),
                      "not_omim_recall10": float(a.loc[~a["omim"], "recall10"].mean()),
                      "burden_1": float(a.loc[a["n"] == 1, "recall10"].mean()),
                      "burden_2_5": float(a.loc[(a["n"] >= 2) & (a["n"] <= 5), "recall10"].mean()),
                      "burden_gt5": float(a.loc[a["n"] > 5, "recall10"].mean()),
                      "therapeutic_only_pairs": int(a["ther_n"].sum()),
                      "therapeutic_only_hit10_micro": float(a["ther_h"].sum() / max(a["ther_n"].sum(), 1))}
    zsum = {"queries": int(len(zsd)), "no_v6_signal": int((~zsd["v6_signal"]).sum()),
            "no_semantic": int((~zsd["semantic"]).sum()),
            "zero_signal_both": int((~zsd["v6_signal"] & ~zsd["semantic"]).sum()),
            "zero_signal_share": float((~zsd["v6_signal"] & ~zsd["semantic"]).mean()),
            "zero_signal_in_pool": int((~zsd["v6_signal"] & ~zsd["semantic"] & zsd["in_pool"]).sum()),
            "zero_signal_in_top10": int((~zsd["v6_signal"] & ~zsd["semantic"] & zsd["in_top10"]).sum())} if len(zsd) else {}
    out = {"generated_utc": now_utc(), "strata": summ, "zero_signal_s42": zsum,
           "note": "strata are evaluation-only; positive burden never routes a query"}
    dump_json(OUT / "strata_zero_signal_v8.json", out)
    return out


def hitflows() -> dict:
    from .eval8 import Ctx, hitflow
    cx = Ctx()
    tot = {}
    for new in ("deployed", "C_b2_l0.1", "C_b0_l0.1"):
        agg = {}
        for j in range(5):
            d = paths8.FORMAL / "main" / "s42" / f"fold{j}"
            if not (d / f"pred_{new}__test.npz").exists():
                continue
            for old in ("channel_lgb_v6", "B"):
                r = hitflow(cx, d, j, old, new)
                key = f"{old}->{new}"
                agg.setdefault(key, {k: 0 for k in r})
                for k, v in r.items():
                    agg[key][k] += v
        tot.update(agg)
    OUT.mkdir(parents=True, exist_ok=True)
    dump_json(OUT / "hitflow_s42_v8.json", {"generated_utc": now_utc(), "flows": tot})
    return tot


if __name__ == "__main__":
    import sys
    paths8.ensure()
    which = sys.argv[1:] or ["two_tower", "protst", "strata", "hitflow"]
    for w_ in which:
        fn = {"two_tower": two_tower, "protst": protst_zero_shot, "strata": zero_signal_and_strata,
              "hitflow": hitflows}[w_]
        print(w_, json.dumps(fn(), indent=1, default=str)[:2500], flush=True)

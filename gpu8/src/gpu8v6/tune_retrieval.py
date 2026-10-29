"""Freeze the V6 retriever on the development families only (section 6.2, 15.1).

The 864 development diseases are scored in each of the five outer contexts (their
labels are never references anywhere).  The script reports, on development data
only:

* every source alone (single-source coverage, zero coverage, oracle U@10);
* the pre-declared ladder L0 (V5 components) -> L5 (all V6 sources, equal
  weights), adding one source at a time;
* the 0.25-step simplex over the seven V6 sources in two fusion modes
  (210 x 2 = 420 configurations), a 0.05-step local refinement around the best,
  and quota variants q in {0, 25, 50};

and writes ``manifests_v6/retriever_v6_frozen.json``.  The objective is the macro
oracle bound U@10 at the fixed budget of 500, coverage breaks ties.  Test folds
are never read.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from gpu8.candidates import rwr_channel

from . import paths6
from .common import World6, dump_json, now_utc, write_manifest
from .retrieval import max_normalise, rank_transform, simplex_grid
from .sources import SourceEngine

V6_SOURCES = ["pheno", "hier", "onto", "text", "gene_hpo", "rwr_pheno", "compat"]
ALL_SOURCES = V6_SOURCES + ["pheno_v5", "rwr_v5"]
LADDER = [
    ("L0_v5_components", "score", {"pheno_v5": 1.0, "hier": 1.0, "rwr_v5": 0.25}),
    ("L1_repaired_hpo", "score", {"pheno": 1.0, "hier": 1.0, "rwr_pheno": 0.25}),
    ("L2_plus_onto", "score", {"pheno": 1.0, "hier": 1.0, "rwr_pheno": 0.25, "onto": 1.0}),
    ("L3_plus_text", "score", {"pheno": 1.0, "hier": 1.0, "rwr_pheno": 0.25, "onto": 1.0,
                               "text": 1.0}),
    ("L4_plus_gene_hpo", "score", {"pheno": 1.0, "hier": 1.0, "rwr_pheno": 0.25, "onto": 1.0,
                                   "text": 1.0, "gene_hpo": 1.0}),
    ("L5_plus_compat", "score", {"pheno": 1.0, "hier": 1.0, "rwr_pheno": 0.25, "onto": 1.0,
                                 "text": 1.0, "gene_hpo": 1.0, "compat": 1.0}),
]
N = 500
PRIOR_EPS = 1e-9


def batch_sources(engine: SourceEngine, cs, ids: list[str]):
    out, info = engine.scores(cs, ids, sources=V6_SOURCES + ["pheno_v5"])
    # the exact V5 propagation channel: seeded by the V5 phenotype view, 32 steps
    v5 = np.zeros_like(out["pheno_v5"])
    P = engine.op.P.astype(np.float32).tocsr()
    for i in range(len(ids)):
        if out["pheno_v5"][i].max() > 0:
            v5[i] = rwr_channel(out["pheno_v5"][i], P, 0.15, 32)
    out["rwr_v5"] = v5
    return {s: max_normalise(v) for s, v in out.items()}, info


DIRECT = ["pheno", "pheno_v5", "hier", "onto", "text", "gene_hpo"]


def evaluate_config(norm, rr, weights: dict, mode: str, L: np.ndarray, prior_n: np.ndarray,
                    n: int = N) -> dict:
    """Fast top-n evaluation; zero-signal slots are filled by the degree prior.

    Besides m (positives in the list) it counts positives carried by genes with
    direct transfer evidence (m_direct), by any fused signal including
    propagation (m_signal), and by the disease-independent prior fill (m_fill).
    """
    base = rr if mode == "rrf" else norm
    names = [s for s, w in weights.items() if w > 0]
    D, G = L.shape
    if names:
        avail = np.stack([(base[s].max(axis=1) > 0) for s in names], axis=1).astype(np.float64)
        w = np.array([weights[s] for s in names])
        mass = avail @ w
        scale = np.where(mass > 0, 1.0 / np.maximum(mass, 1e-12), 0.0)
        fused = np.zeros((D, G), dtype=np.float64)
        for j, s in enumerate(names):
            fused += (w[j] * avail[:, j] * scale)[:, None] * base[s]
    else:
        fused = np.zeros((D, G), dtype=np.float64)
    signal_mask = fused > 0
    direct_mask = np.zeros((D, G), dtype=bool)
    for s in names:
        if s in DIRECT:
            direct_mask |= norm[s] > 0
    fused = fused + PRIOR_EPS * prior_n[None, :]
    top = np.argpartition(-fused, n - 1, axis=1)[:, :n]
    lab = np.take_along_axis(L, top, axis=1)
    sig = np.take_along_axis(signal_mask, top, axis=1)
    dirm = np.take_along_axis(direct_mask, top, axis=1)
    return {"m": lab.sum(axis=1), "n": L.sum(axis=1),
            "m_signal": (lab * sig).sum(axis=1), "m_direct": (lab * dirm).sum(axis=1),
            "signal": sig.sum(axis=1), "direct": dirm.sum(axis=1)}


def config_list(best=None) -> list[tuple[str, str, dict]]:
    cfgs = list(LADDER) + [("prior_only", "score", {})]
    for s in ALL_SOURCES:
        cfgs.append((f"single_{s}", "score", {s: 1.0}))
    for mode in ("score", "rrf"):
        for wt in simplex_grid(len(V6_SOURCES), 0.25):
            if sum(wt) <= 0:
                continue
            d = {s: x for s, x in zip(V6_SOURCES, wt) if x > 0}
            name = f"grid_{mode}_" + "_".join(f"{x:.2f}" for x in wt)
            cfgs.append((name, mode, d))
    return cfgs


def refine_list(mode: str, wt: dict, step: float = 0.05, radius: int = 2) -> list[tuple[str, str, dict]]:
    base = np.array([wt.get(s, 0.0) for s in V6_SOURCES])
    out, seen = [], set()
    frontier = [base]
    for _ in range(radius):
        nxt = []
        for b in frontier:
            for i in range(len(b)):
                for j in range(len(b)):
                    if i == j or b[i] < step - 1e-9:
                        continue
                    c = b.copy()
                    c[i] -= step
                    c[j] += step
                    key = tuple(np.round(c, 4))
                    if key in seen:
                        continue
                    seen.add(key)
                    nxt.append(c)
        frontier = nxt
        for c in nxt:
            d = {s: float(x) for s, x in zip(V6_SOURCES, c) if x > 1e-9}
            out.append((f"refine_{mode}_" + "_".join(f"{x:.2f}" for x in c), mode, d))
    return out


def run(max_dev: int | None = None, folds=(0, 1, 2, 3, 4), device: str = "cuda") -> dict:
    t0 = time.time()
    w = World6()
    eng = SourceEngine(w, device=device)
    dev = w.labelled(w.spec.dev_diseases)
    if max_dev:
        dev = dev[:max_dev]
    cache_dir = paths6.CACHES / "tuning_dev"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfgs = config_list()
    rows = []
    ctx_info = []
    stash = {}
    for fold in folds:
        ctx = w.outer_context(fold)
        cs = eng.context(ctx)
        ctx_info.append({"fold": fold, **cs.info})
        prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(eng.ppi_deg)
        prior_n = prior / max(prior.max(), 1e-12)
        for b0 in range(0, len(dev), 216):
            ids = dev[b0:b0 + 216]
            norm, info = batch_sources(eng, cs, ids)
            rr = {s: rank_transform(v) for s, v in norm.items()}
            L = np.zeros((len(ids), w.n_g), dtype=np.float32)
            for i, d in enumerate(ids):
                L[i, w.query_rows(d)] = 1.0
            np.savez_compressed(cache_dir / f"f{fold}_b{b0}.npz",
                                **{s: v.astype(np.float32) for s, v in norm.items()},
                                L=L.astype(np.int8), prior_n=prior_n.astype(np.float32))
            stash[(fold, b0)] = ids
            for name, mode, wt in cfgs:
                r = evaluate_config(norm, rr, wt, mode, L, prior_n)
                for i, d in enumerate(ids):
                    rows.append((name, fold, d, int(r["m"][i]), int(r["n"][i]), int(r["signal"][i]),
                                 int(r["m_signal"][i]), int(r["m_direct"][i]), int(r["direct"][i])))
            info["fold"] = fold
            info.to_csv(cache_dir / f"info_f{fold}_b{b0}.csv", index=False)
            print(f"  fold {fold} batch {b0}: {len(ids)} diseases, {len(cfgs)} configs, "
                  f"{time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows, columns=["config", "fold", "disease_id", "m", "n", "signal",
                                     "m_signal", "m_direct", "direct"])

    def summarise(df):
        df = df.copy()
        df["coverage"] = df["m"] / df["n"].clip(lower=1)
        df["U10"] = np.minimum(10, df["m"]) / df["n"].clip(lower=1)
        df["zero"] = (df["m"] == 0).astype(float)
        nn = df["n"].clip(lower=1)
        df["cov_signal"] = df["m_signal"] / nn
        df["cov_direct"] = df["m_direct"] / nn
        df["cov_fill"] = (df["m"] - df["m_signal"]) / nn
        g = df.groupby("config").agg(U10=("U10", "mean"), coverage=("coverage", "mean"),
                                     zero_coverage=("zero", "mean"), n=("U10", "size"),
                                     coverage_direct_evidence=("cov_direct", "mean"),
                                     coverage_any_signal=("cov_signal", "mean"),
                                     coverage_prior_fill=("cov_fill", "mean"),
                                     mean_signal_genes=("signal", "mean"),
                                     mean_direct_genes=("direct", "mean"))
        return g.sort_values(["U10", "coverage"], ascending=False)

    summ = summarise(res)
    grid = summ[summ.index.str.startswith("grid_")]
    best_name = grid.index[0]
    best_mode = best_name.split("_")[1]
    best_w = dict(next((c[2] for c in cfgs if c[0] == best_name)))
    # pass 2: local refinement around the best grid point, from the cached sources
    ref_cfgs = refine_list(best_mode, best_w)
    rows2 = []
    for (fold, b0), ids in stash.items():
        z = np.load(cache_dir / f"f{fold}_b{b0}.npz")
        norm = {s: z[s].astype(np.float32) for s in ALL_SOURCES if s in z.files}
        rr = {s: rank_transform(v) for s, v in norm.items()} if best_mode == "rrf" else {}
        L = z["L"].astype(np.float32)
        prior_n = z["prior_n"]
        for name, mode, wt in ref_cfgs:
            r = evaluate_config(norm, rr, wt, mode, L, prior_n)
            for i, d in enumerate(ids):
                rows2.append((name, fold, d, int(r["m"][i]), int(r["n"][i]), int(r["signal"][i]),
                              int(r["m_signal"][i]), int(r["m_direct"][i]), int(r["direct"][i])))
    res2 = pd.DataFrame(rows2, columns=res.columns)
    summ2 = summarise(pd.concat([res[res["config"] == best_name], res2]))
    final_name = summ2.index[0]
    final_cfg = next((c for c in ref_cfgs if c[0] == final_name), (best_name, best_mode, best_w))
    summ_all = pd.concat([summ, summ2[~summ2.index.isin(summ.index)]])
    out = paths6.RETRIEVAL
    out.mkdir(parents=True, exist_ok=True)
    summ_all.to_csv(out / "retriever_tuning_dev_summary.csv", encoding="utf-8-sig")
    res.to_parquet(out / "retriever_tuning_dev_per_disease.parquet", index=False)
    res2.to_parquet(out / "retriever_refine_dev_per_disease.parquet", index=False)
    frozen = {
        "decided_on": "development families only (864 labelled dev diseases x 5 outer contexts)",
        "objective": "macro oracle U@10 at budget 500; coverage tie-break",
        "mode": final_cfg[1], "weights": final_cfg[2], "config_name": final_cfg[0],
        "budget": N, "quota": 0,
        "dev_metrics": summ_all.loc[final_cfg[0]].to_dict(),
        "best_grid_point": {"name": best_name, **summ.loc[best_name].to_dict()},
        "ladder": {name: summ.loc[name].to_dict() for name, _, _ in LADDER},
        "singles": {s: summ.loc[f"single_{s}"].to_dict() for s in ALL_SOURCES},
        "n_grid_configs": int(sum(1 for c in cfgs if c[0].startswith("grid_"))),
        "n_refine_configs": len(ref_cfgs),
        "contexts": ctx_info,
        "seconds": time.time() - t0,
    }
    write_manifest("retriever_v6_frozen.json", frozen)
    return frozen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tune and freeze the V6 retriever on dev.")
    ap.add_argument("--max_dev", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)
    paths6.ensure()
    fr = run(max_dev=args.max_dev or None, folds=tuple(args.folds))
    print(json.dumps({k: fr[k] for k in ("mode", "weights", "dev_metrics", "best_grid_point",
                                         "ladder", "singles")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

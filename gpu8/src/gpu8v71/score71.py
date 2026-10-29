"""Re-score the saved core models: work-pool lists and the standalone core (E1).

For every chain fit (variant, seed, context) the saved model scores its target
diseases over the full gene background:

* inner context  -> out-of-fold core top-500 (valid genes only) for the held-out
                    training block (second-level training data);
* outer context  -> core top-500 for test F_j, validation F_{j+1} and the
                    development families, and the **standalone** evaluation of
                    the core on the test fold: its own 500 list (valid genes by r,
                    ties by gene row; a degree-prior fill is marked and only used
                    if fewer than 500 genes are valid).

Only valid matches produce core candidates; an invalid pair's 0 is never read
as a score.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from gpu8v6.common import ContextSpec
from gpu8v6.metrics6 import evaluate_matrix, macro, per_disease_frame

from . import paths71
from .behavior71 import ppi_degree
from .chain71 import SEEDS, Runner, core_list, variant_spec
from .common71 import dump_json, load_manifest
from .ledger71 import is_done, run_dir, tracked

N_G = 19861


def context_prior(run: Runner, ctx_id: str) -> np.ndarray:
    ctx = run.cx.spec(ctx_id)
    ref = run.cx.v6_reference(ctx)
    cs = ContextSpec(ctx.context_id, ctx.fold, ctx.kind, ctx.block_index, ref, ctx.held_out)
    A = run.w.adjacency(cs)
    prior = np.log1p(np.asarray(A.sum(axis=0)).ravel()) + 0.5 * np.log1p(ppi_degree(run.w))
    return prior / max(prior.max(), 1e-12)


def pool_path(variant: str, seed: int, ctx: str):
    return paths71.POOLS / f"core__{variant}__s{seed}__{ctx}.parquet"


def score_variant(variant: str, seeds, folds, device: str = "cuda", run: Runner | None = None) -> None:
    if not load_manifest("protocol_v71_registered.json"):
        raise RuntimeError("protocol not registered")
    run = run or Runner(device)
    spec = variant_spec(variant)
    for seed in seeds:
        for j in folds:
            ctxs = [f"f{j}_outer"] if spec["outer_only"] else \
                [c.context_id for c in run.w.inner_contexts(j)] + [f"f{j}_outer"]
            for ctx in ctxs:
                fit_id = f"V71CHAIN__{variant}__{ctx}__s{seed}"
                if not is_done(fit_id):
                    print(f"  SKIP {fit_id}: fit not finished (rerun score71 later)", flush=True)
                    continue
                pp = pool_path(variant, seed, ctx)
                eval_id = f"V71Z__{variant}__f{j}__s{seed}"
                need_eval = ctx.endswith("outer") and not is_done(eval_id)
                if pp.exists() and not need_eval:
                    continue
                t0 = time.time()
                tr, m = run.model(fit_id, spec)
                tg = run.cx.targets(ctx)
                ids = list(dict.fromkeys(d for v in tg.values() for d in v))
                rows = np.array([run.w.idx.d_ix[d] for d in ids])
                R, V = tr.score(m, rows)
                g500, _ = core_list(R, V, np.zeros(N_G), 500, fill=False)
                recs = []
                for i, d in enumerate(ids):
                    g = g500[i][g500[i] >= 0]
                    recs.append(pd.DataFrame({"disease_id": d, "gene_row": g.astype(np.int32),
                                              "core_rank": np.arange(1, len(g) + 1, dtype=np.int32),
                                              "r": R[i, g].astype(np.float32)}))
                pd.concat(recs, ignore_index=True).to_parquet(pp, index=False)
                if need_eval:
                    test = tg["test"]
                    pos = {d: i for i, d in enumerate(ids)}
                    ti = np.array([pos[d] for d in test])
                    prior = context_prior(run, ctx)
                    gl, fill = core_list(R[ti], V[ti], prior, 500, fill=True)
                    y = np.zeros(gl.shape, dtype=np.float32)
                    nq = np.zeros(len(test))
                    for k, d in enumerate(test):
                        q = run.w.query_rows(d)
                        nq[k] = len(q)
                        y[k] = np.isin(gl[k], q)
                    sc = -np.tile(np.arange(gl.shape[1], dtype=np.float64), (len(test), 1))
                    meta = {"stage": "formal_standalone", "method": "core_standalone",
                            "variant": variant, "fold": j, "seed": seed, "task": "Z",
                            "context_id": ctx, "fit_run": fit_id}
                    with tracked(eval_id, meta, device=device) as box:
                        per = per_disease_frame(test, evaluate_matrix(sc, y, gl, nq, ks=(10, 20, 50)))
                        per["n_valid_genes"] = V[ti].sum(axis=1)
                        per["n_prior_fill"] = fill.sum(axis=1)
                        d = run_dir(eval_id)
                        per.to_parquet(d / "per_disease.parquet", index=False)
                        pd.DataFrame({"disease_id": np.repeat(test, gl.shape[1]),
                                      "gene_row": gl.reshape(-1).astype(np.int32),
                                      "rank": np.tile(np.arange(1, gl.shape[1] + 1), len(test)),
                                      "r": np.take_along_axis(R[ti], gl, axis=1).reshape(-1),
                                      "is_fill": fill.reshape(-1),
                                      "label": y.reshape(-1).astype(np.int8)}).to_parquet(
                            d / "predictions.parquet", index=False)
                        box["extra"] = {"metrics": macro(per), "n_test_diseases": len(test),
                                        "diseases_with_fill": int((fill.sum(axis=1) > 0).sum())}
                    print(f"  {eval_id}: R@10 {box['extra']['metrics']['recall10_end']:.4f}", flush=True)
                print(f"  pools {variant} s{seed} {ctx}: {len(ids)} diseases, {time.time() - t0:.0f}s",
                      flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V7.1 core scoring")
    ap.add_argument("--variants", nargs="*", default=["core"])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)
    paths71.ensure()
    run = Runner()
    for v in args.variants:
        seeds = args.seeds if v == "core" else [42]
        score_variant(v, seeds, args.folds, run=run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

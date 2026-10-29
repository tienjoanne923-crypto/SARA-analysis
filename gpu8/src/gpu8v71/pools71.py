"""V7.1 section 12 — work pools and their features.

Work pool of a disease in a context = frozen V6 500-list  U  core top-500 of a
given (variant, seed) core fitted in that same context.  Nothing is added from
a query answer.  For each context one pass of the frozen V6 source engine
computes the V6 channel features of every gene that any registered core list
brings in (the union over variants and seeds), so that V6 genes and new genes
are described by the same quantities:

    v_s   max-normalised source score,   r_s   rank over the full background,
    fused the frozen V6 reciprocal-rank fusion score,
    v6_rank  position in the frozen V6 ordering (list, then fused, then prior),
    reason   0 signal / 1 quota / 2 prior fill for V6-listed genes, 3 = outside the V6 500.

For V6-listed genes the archived V6 values are used (the pass recomputes them
only to record the agreement).  Core features are recomputed from the saved
core model at assembly time.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from gpu8v6 import paths6
from gpu8v6.common import World6
from gpu8v6.common import load_manifest as load_manifest6
from gpu8v6.retrieval import build_lists, max_normalise
from gpu8v6.sources import SOURCES

from . import paths71
from .common71 import dump_json, load_manifest, now_utc
from .score71 import pool_path

N_G = 19861
CAND6 = paths6.CACHES / "candidates"
POOL_VARIANTS = [("core", 42), ("core", 52), ("core", 62), ("core_altanchor", 42),
                 ("two_tower", 42), ("core_oldres", 42), ("core_strict", 42)]


def v6_path(ctx: str):
    return paths71.POOLS / f"v6feats__{ctx}.parquet"


def v6_dis_path(ctx: str):
    return paths71.POOLS / f"v6dis__{ctx}.parquet"


def load_v6_list(ctx: str, ids: list[str] | None = None) -> pd.DataFrame:
    c = pd.read_parquet(CAND6 / f"{ctx}_candidates.parquet")
    c = c[c["rank"] <= 500]
    if ids is not None:
        c = c[c["disease_id"].isin(set(ids))]
    return c


def union_genes(ctx: str, variants=POOL_VARIANTS) -> dict[str, set]:
    u: dict[str, set] = {}
    for v, s in variants:
        p = pool_path(v, s, ctx)
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["disease_id", "gene_row"])
        for d, g in df.groupby("disease_id")["gene_row"]:
            u.setdefault(d, set()).update(int(x) for x in g)
    return u


def full_ranks(x: np.ndarray) -> np.ndarray:
    """Rank (1 = best, ties by gene row) over the background; 0 where x <= 0."""
    order = np.lexsort((np.arange(len(x)), -x))
    rank = np.empty(len(x), dtype=np.int32)
    rank[order] = np.arange(1, len(x) + 1, dtype=np.int32)
    return np.where(x > 0, rank, 0)


def context_pass(eng, w: World6, ctx_id: str, frozen: dict, batch: int = 200) -> dict:
    t0 = time.time()
    v6 = load_v6_list(ctx_id)
    targets = list(dict.fromkeys(v6["disease_id"]))
    extra = union_genes(ctx_id)
    listed = {d: set(g.astype(int)) for d, g in v6.groupby("disease_id")["gene_row"]}
    cs = eng.context(w.contexts[ctx_id])
    prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(eng.ppi_deg)
    prior = prior / max(prior.max(), 1e-12)
    rows, drows, dev_check = [], [], []
    for b0 in range(0, len(targets), batch):
        ids = targets[b0:b0 + batch]
        raw, info = eng.scores(cs, ids, sources=SOURCES)
        norm = {s: max_normalise(v) for s, v in raw.items()}
        genes, fs, reason, fused = build_lists(norm, frozen["weights"], prior, n=500,
                                               quota=int(frozen.get("quota", 0)), mode=frozen["mode"])
        for i, d in enumerate(ids):
            new = sorted(extra.get(d, set()) - listed.get(d, set()))
            lst = genes[i]
            # agreement with the archived V6 list
            dev_check.append(len(set(lst.tolist()) ^ listed.get(d, set())))
            if not new:
                continue
            g = np.array(new, dtype=np.int64)
            inlist = np.zeros(N_G, dtype=bool)
            inlist[lst] = True
            rest = np.flatnonzero(~inlist)
            order = rest[np.lexsort((rest, -prior[rest], -fused[i, rest].astype(np.float64)))]
            pos = np.empty(N_G, dtype=np.int32)
            pos[lst] = np.arange(1, 501, dtype=np.int32)
            pos[order] = np.arange(501, 501 + len(order), dtype=np.int32)
            rec = {"disease_id": d, "gene_row": g.astype(np.int32), "v6_rank": pos[g],
                   "fused": fused[i, g].astype(np.float32), "reason": np.full(len(g), 3, np.int8),
                   "prior": prior[g].astype(np.float32)}
            for s in SOURCES:
                rec[f"v_{s}"] = norm[s][i, g].astype(np.float32)
                rec[f"r_{s}"] = full_ranks(norm[s][i].astype(np.float64))[g]
            rows.append(pd.DataFrame(rec))
        print(f"    {ctx_id}: {min(b0 + batch, len(targets))}/{len(targets)} ({time.time() - t0:.0f}s)",
              flush=True)
    new_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    old = v6.rename(columns={"rank": "v6_rank"})
    old = old[["disease_id", "gene_row", "v6_rank", "fused", "reason", "prior"]
              + [f"{p}_{s}" for s in SOURCES for p in ("v", "r")]]
    allp = pd.concat([old, new_df], ignore_index=True)
    allp["in_v6_500"] = allp["reason"] != 3
    allp.to_parquet(v6_path(ctx_id), index=False)
    dis = pd.read_parquet(CAND6 / f"{ctx_id}_diseases.parquet")
    dis.to_parquet(v6_dis_path(ctx_id), index=False)
    rep = {"context_id": ctx_id, "n_targets": len(targets), "new_pairs": int(len(new_df)),
           "mean_union_per_disease": float(len(allp) / max(len(targets), 1)),
           "v6_list_symmetric_difference_mean": float(np.mean(dev_check)) if dev_check else 0.0,
           "v6_list_exact_share": float(np.mean([x == 0 for x in dev_check])) if dev_check else 1.0,
           "seconds": time.time() - t0}
    return rep


def run(contexts: list[str] | None = None, device: str = "cuda") -> dict:
    from gpu8v6.sources import SourceEngine
    w = World6()
    eng = SourceEngine(w, device=device, with_v5=False)
    frozen = load_manifest6("retriever_v6_frozen.json")
    if contexts is None:
        contexts = [c for j in range(5) for c in
                    [f"f{j}_inner0", f"f{j}_inner1", f"f{j}_inner2", f"f{j}_outer"]]
    reps = []
    for ctx in contexts:
        if v6_path(ctx).exists():
            continue
        rep = context_pass(eng, w, ctx, frozen)
        reps.append(rep)
        dump_json(paths71.AUDIT / f"pool_pass_{ctx}.json", rep)
        print(json.dumps(rep), flush=True)
    return {"generated_utc": now_utc(), "contexts": reps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V7.1 pool V6 features")
    ap.add_argument("--contexts", nargs="*", default=None)
    args = ap.parse_args(argv)
    paths71.ensure()
    run(args.contexts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

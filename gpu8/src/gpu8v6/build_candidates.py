"""Materialise the frozen V6 retriever's candidate tables for every context.

Per outer fold j:

* ``f{j}_outer``   targets = test F_j (budget 1000, so the 500 -> 1000 audit can be
                   repeated), validation F_{j+1} and the development families
                   (budget 500);
* ``f{j}_inner{b}`` targets = the training block held out of that context; these
                   are the out-of-fold (OOF) lists that training uses.

Each row is one (disease, gene) pair with the retrieval rank, the fused score,
the fallback reason, every source's max-normalised score and its rank over the
full background.  Query labels are written to a separate file that only the
evaluator and the training loss read.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import paths6
from .common import World6, candidate_hash, dump_json, load_manifest, now_utc
from .retrieval import REASON_NAMES, build_lists, max_normalise
from .sources import SOURCES, SourceEngine

BATCH = 200


def source_ranks(x: np.ndarray, genes: np.ndarray) -> np.ndarray:
    """Rank (1 = best) of each listed gene within the full background, 0 if no signal."""
    out = np.zeros(genes.shape, dtype=np.int32)
    for i in range(x.shape[0]):
        row = x[i]
        pos = row > 0
        if not pos.any():
            continue
        order = np.lexsort((np.arange(len(row)), -row))
        rank = np.empty(len(row), dtype=np.int32)
        rank[order] = np.arange(1, len(row) + 1)
        r = rank[genes[i]]
        out[i] = np.where(pos[genes[i]], r, 0)
    return out


def build_context(eng: SourceEngine, w: World6, ctx, targets: dict[str, int], frozen: dict,
                  out_dir) -> dict:
    cs = eng.context(ctx)
    prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(eng.ppi_deg)
    prior = prior / max(prior.max(), 1e-12)
    weights, mode, quota = frozen["weights"], frozen["mode"], int(frozen.get("quota", 0))
    rows, drows, lrows = [], [], []
    ids_all = list(targets)
    for b0 in range(0, len(ids_all), BATCH):
        ids = ids_all[b0:b0 + BATCH]
        raw, info = eng.scores(cs, ids, sources=SOURCES)
        norm = {s: max_normalise(v) for s, v in raw.items()}
        budget = max(targets[d] for d in ids)
        genes, fs, reason, _ = build_lists(norm, weights, prior, n=budget, quota=quota, mode=mode)
        ranks = {s: source_ranks(norm[s], genes) for s in SOURCES}
        for i, d in enumerate(ids):
            n_i = targets[d]
            g = genes[i, :n_i]
            rec = {"disease_id": np.repeat(d, n_i), "gene_row": g.astype(np.int32),
                   "rank": np.arange(1, n_i + 1, dtype=np.int32),
                   "fused": fs[i, :n_i].astype(np.float32),
                   "reason": reason[i, :n_i].astype(np.int8),
                   "prior": prior[g].astype(np.float32)}
            for s in SOURCES:
                rec[f"v_{s}"] = norm[s][i, g].astype(np.float32)
                rec[f"r_{s}"] = ranks[s][i, :n_i]
            rows.append(pd.DataFrame(rec))
            q = w.query_rows(d)
            lrows.append(pd.DataFrame({"disease_id": d, "gene_row": q.astype(np.int32)}))
            dr = {"disease_id": d, "budget": n_i,
                  "n_signal": int((reason[i, :n_i] == 0).sum()),
                  "n_prior_fill": int((reason[i, :n_i] == 2).sum()),
                  "n_query": int(len(q))}
            for s in SOURCES:
                dr[f"a_{s}"] = int(norm[s][i].max() > 0)
            drows.append(dr)
        dinfo = info.copy()
        drows_df = pd.DataFrame(drows[-len(ids):])
        for c in dinfo.columns:
            if c != "disease_id":
                drows_df[c] = dinfo[c].to_numpy()
        drows[-len(ids):] = drows_df.to_dict(orient="records")
        print(f"    {ctx.context_id}: {min(b0 + BATCH, len(ids_all))}/{len(ids_all)}", flush=True)
    cand = pd.concat(rows, ignore_index=True)
    dis = pd.DataFrame(drows)
    lab = pd.concat(lrows, ignore_index=True)
    cand.to_parquet(out_dir / f"{ctx.context_id}_candidates.parquet", index=False)
    dis.to_parquet(out_dir / f"{ctx.context_id}_diseases.parquet", index=False)
    lab.to_parquet(out_dir / f"{ctx.context_id}_labels.parquet", index=False)
    c500 = cand[cand["rank"] <= 500]
    meta = {"context_id": ctx.context_id, **cs.info,
            "n_targets": len(ids_all), "rows": int(len(cand)),
            "candidate_hash_500": candidate_hash(c500["disease_id"], c500["gene_row"]),
            "reason_counts_500": {REASON_NAMES[int(k)]: int(v) for k, v in
                                  c500["reason"].value_counts().items()}}
    return meta


def run(folds=(0, 1, 2, 3, 4), device: str = "cuda") -> dict:
    t0 = time.time()
    frozen = load_manifest("retriever_v6_frozen.json")
    if not frozen:
        raise RuntimeError("retriever_v6_frozen.json missing: run gpu8v6.tune_retrieval first")
    w = World6()
    eng = SourceEngine(w, device=device, with_v5=False)
    out_dir = paths6.CACHES / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    dev = w.labelled(w.spec.dev_diseases)
    for fold in folds:
        fa = w.fold(fold)
        targets = {d: 1000 for d in w.labelled(fa["test_diseases"])}
        for d in w.labelled(fa["val_diseases"]) + dev:
            targets.setdefault(d, 500)
        metas.append(build_context(eng, w, w.outer_context(fold), targets, frozen, out_dir))
        for ctx in w.inner_contexts(fold):
            tg = {d: 500 for d in w.labelled(ctx.held_out)}
            metas.append(build_context(eng, w, ctx, tg, frozen, out_dir))
        print(f"  fold {fold} done, {time.time() - t0:.0f}s", flush=True)
    rep = {"generated_utc": now_utc(), "retriever": {k: frozen[k] for k in ("mode", "weights")},
           "retriever_payload_hash": frozen.get("payload_hash"), "contexts": metas,
           "seconds": time.time() - t0}
    dump_json(paths6.MANIFESTS / "candidates_v6_index.json", rep)
    return rep


def rebuild_index() -> dict:
    """Index of every candidate table on disk (parallel workers each wrote a partial one)."""
    out_dir = paths6.CACHES / "candidates"
    frozen = load_manifest("retriever_v6_frozen.json")
    metas = []
    for p in sorted(out_dir.glob("*_candidates.parquet")):
        cid = p.name.replace("_candidates.parquet", "")
        cand = pd.read_parquet(p, columns=["disease_id", "gene_row", "rank", "reason"])
        c500 = cand[cand["rank"] <= 500]
        metas.append({"context_id": cid, "n_targets": int(cand["disease_id"].nunique()),
                      "rows": int(len(cand)),
                      "candidate_hash_500": candidate_hash(c500["disease_id"], c500["gene_row"]),
                      "reason_counts_500": {REASON_NAMES[int(k)]: int(v) for k, v in
                                            c500["reason"].value_counts().items()}})
    rep = {"generated_utc": now_utc(), "retriever": {k: frozen[k] for k in ("mode", "weights")},
           "retriever_payload_hash": frozen.get("payload_hash"), "contexts": metas}
    dump_json(paths6.MANIFESTS / "candidates_v6_index.json", rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build V6 candidate tables.")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--index_only", action="store_true")
    args = ap.parse_args(argv)
    paths6.ensure()
    if args.index_only:
        rep = rebuild_index()
        print(json.dumps({"n_contexts": len(rep["contexts"])}, indent=2))
        return 0
    rep = run(tuple(args.folds))
    print(json.dumps({"n_contexts": len(rep["contexts"]), "seconds": rep["seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

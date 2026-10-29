"""V6 section 3.3 — hit-flow audit when the candidate budget grows 500 -> 1000.

First the two conditions the plan requires before any counting:

* the lists must be nested (the top 500 of the retriever is inside its top 1000);
* the same disease-gene pair must receive the same raw score in both runs
  (otherwise per-list standardisation, cross-candidate attention, batch
  randomness or a configuration difference has to be found first).

Only then are the flows counted per disease: positives that were in the top 10
with 500 candidates and are pushed out with 1000, new positives (retriever ranks
501-1000) that enter the top 10, and new non-positives that occupy top-10 slots.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import paths6
from .common import dump_json, now_utc


def flow_one(c_small: np.ndarray, s_small: np.ndarray, c_big: np.ndarray,
             s_big: np.ndarray, qrows: set, k: int = 10) -> dict:
    """Counts for one disease; candidate arrays are gene rows, scores aligned."""
    small_set = set(int(x) for x in c_small)
    big_set = set(int(x) for x in c_big)
    nested = small_set <= big_set
    pos_big = {int(g): i for i, g in enumerate(c_big)}
    common = [i for i, g in enumerate(c_small) if int(g) in pos_big]
    diffs = np.abs(s_small[common] - s_big[[pos_big[int(c_small[i])] for i in common]]) \
        if common else np.array([0.0])

    def topk(c, s):
        order = np.lexsort((c, -s))
        return [int(c[i]) for i in order[:k]]

    t_small, t_big = topk(c_small, s_small), topk(c_big, s_big)
    hits_small = {g for g in t_small if g in qrows}
    hits_big = {g for g in t_big if g in qrows}
    new_genes = big_set - small_set
    return {
        "nested": bool(nested), "max_abs_score_diff": float(diffs.max()),
        "n_query": len(qrows), "m_small": len(qrows & small_set), "m_big": len(qrows & big_set),
        "h_small": len(hits_small), "h_big": len(hits_big),
        "pushed_out": len(hits_small - set(t_big)),
        "new_positive_entered_top": len({g for g in hits_big if g in new_genes}),
        "new_nonhit_in_top": len({g for g in t_big if g in new_genes and g not in qrows}),
        "top_slots_taken_by_new": len({g for g in t_big if g in new_genes}),
    }


def summarise(df: pd.DataFrame, k: int = 10) -> dict:
    n = df["n_query"].clip(lower=1)
    return {
        "n_diseases": int(len(df)),
        "all_nested": bool(df["nested"].all()),
        "max_abs_score_diff": float(df["max_abs_score_diff"].max()),
        "scores_identical": bool(df["max_abs_score_diff"].max() <= 1e-5),
        "coverage_small": float((df["m_small"] / n).mean()),
        "coverage_big": float((df["m_big"] / n).mean()),
        "recall_small": float((df["h_small"] / n).mean()),
        "recall_big": float((df["h_big"] / n).mean()),
        "total_pushed_out": int(df["pushed_out"].sum()),
        "total_new_positive_entered_top": int(df["new_positive_entered_top"].sum()),
        "total_new_nonhit_in_top": int(df["new_nonhit_in_top"].sum()),
        "total_top_slots_taken_by_new": int(df["top_slots_taken_by_new"].sum()),
        "diseases_losing_hits": int((df["h_big"] < df["h_small"]).sum()),
        "diseases_gaining_hits": int((df["h_big"] > df["h_small"]).sum()),
        "macro_recall_change": float(((df["h_big"] - df["h_small"]) / n).mean()),
        "macro_change_from_pushed_out": float((-df["pushed_out"] / n).mean()),
        "macro_change_from_new_entries": float((df["new_positive_entered_top"] / n).mean()),
    }


def run_v5(fold: int = 0, seed: int = 42, device: str | None = None) -> dict:
    """Audit the V5 b9 checkpoint on its own frozen candidates."""
    import torch
    from gpu8 import paths as p5
    from gpu8.config import load_config
    from gpu8.episodes import fit_feature_stats
    from gpu8.model import build_model
    from gpu8.protocol import fold_assignment, load_spec
    from gpu8.runner import World, primary_positive_map
    from gpu8.train import score_disease

    cfg = load_config()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    world = World(cfg)
    spec = load_spec()
    fa = fold_assignment(spec, fold, cfg.protocol.n_outer_folds)
    builder = world.builder(f"f{fold}_outer")
    builder.b.stats = fit_feature_stats(builder, fa["train_diseases"],
                                        seed=cfg.protocol.split_seed)
    model = build_model(cfg.model, seed=seed)
    ckpt = p5.GPU8_ROOT / "checkpoints" / f"b9_ours__f{fold}__s{seed}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    positives = primary_positive_map(world)
    g_ix = world.idx.g_ix
    rows = []
    for did in fa["test_diseases"]:
        pos = positives.get(did, [])
        d_row = world.idx.d_ix.get(did)
        if not pos or d_row is None:
            continue
        qrows = {g_ix[g] for g in pos if g in g_ix}
        c_small, _ = builder.candidate_rows(did, d_row, n=500)
        c_big, _ = builder.candidate_rows(did, d_row, n=1000)
        with torch.no_grad():
            s_small, _ = score_disease(cfg, model, builder, did, [], c_small, device)
            s_big, _ = score_disease(cfg, model, builder, did, [], c_big, device)
        rec = flow_one(c_small, np.asarray(s_small, dtype=np.float64), c_big,
                       np.asarray(s_big, dtype=np.float64), qrows)
        rec["disease_id"] = did
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = paths6.AUDIT
    df.to_csv(out / f"hitflow_v5_b9_f{fold}_s{seed}.csv", index=False, encoding="utf-8-sig")
    summ = summarise(df)
    summ.update({"generated_utc": now_utc(), "model": f"V5 b9_ours checkpoint f{fold} s{seed}",
                 "candidates": "V5 fused retriever, budgets 500 and 1000"})
    dump_json(out / f"hitflow_v5_b9_f{fold}_s{seed}.json", summ)
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="500 -> 1000 hit-flow audit.")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    paths6.ensure()
    print(json.dumps(run_v5(args.fold, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

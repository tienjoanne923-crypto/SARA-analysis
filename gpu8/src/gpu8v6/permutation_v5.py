"""Classify and repeat the V5 disease permutation with the V6 definitions.

The V5 negative control (``experiments.run_negative_controls``) scored the target's
fixed candidates with ``score_disease(donor, support_of_target, ...)`` at k = 5:
disease views, the four channels, the quality fields and the anchors all came
from the donor, while the exposed support stayed the target's.  In V6 terms that
is P2 on the semantic inputs with the support path held fixed — neither a clean
P1 nor a clean P2.

This module re-runs the V5 b9 checkpoint at k = 0 (no support) with:

P1  donor disease views only (the neural disease branch), target channels;
P2  every disease-dependent input from the donor, target candidates;
P3  donor candidates and inputs, target labels;

using the same donor schemes as the V6 suite.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from . import paths6
from .common import dump_json, now_utc
from .permutation import R_DRAWS, donor_table, draw_donors


def run(fold: int = 0, seed: int = 42, draws: int = R_DRAWS) -> dict:
    from gpu8 import paths as p5
    from gpu8.adapt import apply_adaptation
    from gpu8.config import load_config
    from gpu8.episodes import fit_feature_stats
    from gpu8.model import build_model
    from gpu8.protocol import fold_assignment, load_spec
    from gpu8.runner import World, primary_positive_map
    from gpu8.train import to_tensor

    from .common import World6

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world = World(cfg)
    w6 = World6(load_heavy=False)
    spec = load_spec()
    fa = fold_assignment(spec, fold, cfg.protocol.n_outer_folds)
    b = world.builder(f"f{fold}_outer")
    b.b.stats = fit_feature_stats(b, fa["train_diseases"], seed=cfg.protocol.split_seed)
    model = build_model(cfg.model, seed=seed)
    model.load_state_dict(torch.load(p5.GPU8_ROOT / "checkpoints" / f"b9_ours__f{fold}__s{seed}.pt",
                                     map_location=device))
    model.to(device).eval()
    positives = primary_positive_map(world)
    g_ix, d_ix = world.idx.g_ix, world.idx.d_ix
    ids = [d for d in fa["test_diseases"] if positives.get(d) and d in d_ix]
    cand = {d: b.candidate_rows(d, d_ix[d])[0] for d in ids}
    qsets = {d: {g_ix[g] for g in positives[d] if g in g_ix} for d in ids}

    @torch.no_grad()
    def score(chan_id, view_id, rows):
        ch, mask, q = b.pair_features(chan_id, d_ix[chan_id], rows)
        dv, gv = b.node_features(d_ix[view_id], rows)
        enc = model.encode(to_tensor(dv, device), to_tensor(gv, device), to_tensor(ch, device),
                           to_tensor(mask, device), to_tensor(q, device))
        s = apply_adaptation(enc.base_score, enc.phi,
                             torch.zeros(cfg.model.adapt_dim, device=device))
        return s.double().cpu().numpy()

    def recall(s, rows, qs, nq):
        order = np.lexsort((rows, -s))
        return sum(1 for g in rows[order[:10]] if int(g) in qs) / max(nq, 1)

    base = {d: score(d, d, cand[d]) for d in ids}
    t = donor_table(w6, ids)
    rows_out = []
    for scheme in ("distant", "close"):
        for draw in range(draws):
            donors = draw_donors(t, scheme, draw)
            for d, dn in donors.items():
                nq = len(qsets[d])
                r0 = recall(base[d], cand[d], qsets[d], nq)
                s1 = score(d, dn, cand[d])             # P1: donor views only
                s2 = score(dn, dn, cand[d])            # P2: all disease inputs from donor
                s3 = score(dn, dn, cand[dn])           # P3: donor list too
                for kind, s, rws in (("P1", s1, cand[d]), ("P2", s2, cand[d]),
                                     ("P3", s3, cand[dn])):
                    rows_out.append({"disease_id": d, "donor": dn, "scheme": scheme,
                                     "draw": draw, "test": kind, "recall10_orig": r0,
                                     "recall10_perm": recall(s, rws, qsets[d], nq),
                                     "covered": float(len(qsets[d] & set(cand[d].tolist())) > 0)})
    df = pd.DataFrame(rows_out)
    df["delta"] = df["recall10_perm"] - df["recall10_orig"]
    df.to_csv(paths6.PERMUTATION / f"v5_b9_permutations_f{fold}.csv", index=False,
              encoding="utf-8-sig")
    g = df.groupby(["test", "scheme", "disease_id"])[["delta", "recall10_orig", "recall10_perm",
                                                      "covered"]].mean().reset_index()
    summ = g.groupby(["test", "scheme"]).agg(n=("delta", "size"), delta=("delta", "mean"),
                                             orig=("recall10_orig", "mean"),
                                             perm=("recall10_perm", "mean")).reset_index()
    out = {"generated_utc": now_utc(), "model": f"V5 b9_ours f{fold} s{seed}, k = 0",
           "classification_of_v5_control": (
               "V5 'disease_permuted' = P2 on all semantic inputs (views, channels, quality, "
               "anchors from the donor) with the target's true k = 5 support retained; "
               "random unmatched donors; fold-0 dose cohort (107 diseases)"),
           "summary": summ.to_dict(orient="records")}
    dump_json(paths6.PERMUTATION / f"v5_b9_permutations_f{fold}.json", out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V5 b9 permutations with V6 definitions.")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args(argv)
    paths6.ensure()
    print(json.dumps(run(args.fold), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

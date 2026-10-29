"""E1 control — same-resource fixed fusion (no trained adapter).

Uses exactly the core's inputs, with zero trainable parameters:

* s_F0(d, g) = cos(z0_d, x^F_g): SapBERT disease vector (mean of the available
  text / HPO vectors) against the SapBERT GO-function mean of the gene; both live
  in the same frozen space, so no projection is needed;
* s_E(d, g)  = the core's tissue-expression match (same formula, c_dg >= 0.8);
* s_S0(d, g) = sequence-neighbour transfer: the 5 reference diseases most similar
  to d (SapBERT cosine) give their visible genes as seeds; a gene scores its
  maximum centred-ESM cosine to any seed gene (a seed gene itself scores 1).

Scores are combined with non-negative simplex weights (step 0.1) chosen on the
development families of the five outer contexts; an unavailable component is
dropped and the remaining weights are renormalised per pair.  Output: a
standalone full-background list, evaluated exactly like the core.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch

from gpu8v6.common import ContextSpec
from gpu8v6.metrics6 import evaluate_matrix, macro, per_disease_frame

from . import paths71
from .chain71 import Runner, core_list
from .common71 import load_manifest, write_manifest
from .ledger71 import is_done, run_dir, tracked
from .score71 import context_prior

K_REF = 5
N_G = 19861


class SRFF:
    def __init__(self, run: Runner):
        self.run = run
        self.dev = run.device
        a = run.pack.a
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.dev)
        xT, xH = a["d_text"], a["d_hpo"]
        nT = xT / np.maximum(np.linalg.norm(xT, axis=1, keepdims=True), 1e-12)
        nH = xH / np.maximum(np.linalg.norm(xH, axis=1, keepdims=True), 1e-12)
        z0 = nT * a["d_text_mask"][:, None] + nH * a["d_hpo_mask"][:, None]
        z0 = z0 / np.maximum(np.linalg.norm(z0, axis=1, keepdims=True), 1e-12)
        self.z0 = t(z0)
        self.sem = torch.as_tensor(a["d_text_mask"] | a["d_hpo_mask"], device=self.dev)
        xF = a["g_function"]
        self.xF = t(xF / np.maximum(np.linalg.norm(xF, axis=1, keepdims=True), 1e-12))
        self.mF = torch.as_tensor(a["g_function_mask"], device=self.dev)
        S = a["g_sequence"].copy()
        mS = a["g_sequence_mask"]
        S[mS] -= S[mS].mean(axis=0)
        S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        S[~mS] = 0.0
        self.xS = t(S)
        self.mS = torch.as_tensor(mS, device=self.dev)
        self.q = t(a["d_tissue"])
        self.mq = torch.as_tensor(a["d_tissue_mask"], device=self.dev)
        self.e = t(a["g_expression"])
        self.o = t(a["g_expression_mask"])

    def components(self, ctx_id: str, ids: list[str]):
        cx = self.run.cx
        ctx = cx.spec(ctx_id)
        ref = cx.v6_reference(ctx)
        cs = ContextSpec(ctx.context_id, ctx.fold, ctx.kind, ctx.block_index, ref, ctx.held_out)
        A = self.run.w.adjacency(cs).tocsr()
        d_ix = self.run.w.idx.d_ix
        ref_rows = np.array([d_ix[d] for d in self.run.w.labelled(ref)])
        ref_rows = ref_rows[np.diff(A.indptr)[ref_rows] > 0]
        rows = np.array([d_ix[d] for d in ids])
        out = {k: np.zeros((len(ids), N_G), dtype=np.float32) for k in ("F", "E", "S")}
        av = {k: np.zeros((len(ids), N_G), dtype=bool) for k in ("F", "E", "S")}
        with torch.no_grad():
            zr = self.z0[torch.as_tensor(ref_rows, device=self.dev)]
            for b0 in range(0, len(rows), 64):
                r = torch.as_tensor(rows[b0:b0 + 64], device=self.dev)
                z = self.z0[r]
                sl = slice(b0, b0 + len(r))
                out["F"][sl] = (z @ self.xF.T).cpu().numpy()
                av["F"][sl] = (self.sem[r][:, None] & self.mF[None, :]).cpu().numpy()
                c = self.q[r] @ self.o.T
                out["E"][sl] = torch.tanh((self.q[r] @ self.e.T) / (3.0 * c.clamp_min(1e-8))).cpu().numpy()
                av["E"][sl] = (self.mq[r][:, None] & (c >= 0.8)).cpu().numpy()
                sim = z @ zr.T                                         # [b, n_ref]
                top = torch.topk(sim, K_REF, dim=1).indices.cpu().numpy()
                for i in range(len(r)):
                    if not bool(self.sem[r[i]]):
                        continue
                    seeds = np.unique(np.concatenate([A[ref_rows[k]].indices for k in top[i]]))
                    seeds = seeds[self.run.pack.a["g_sequence_mask"][seeds]]
                    if len(seeds) == 0:
                        continue
                    cs_ = self.xS @ self.xS[torch.as_tensor(seeds, device=self.dev)].T
                    out["S"][b0 + i] = cs_.max(dim=1).values.cpu().numpy()
                    av["S"][b0 + i] = self.run.pack.a["g_sequence_mask"]
        return out, av


def combine(out, av, w: dict) -> tuple[np.ndarray, np.ndarray]:
    num = np.zeros_like(out["F"], dtype=np.float64)
    den = np.zeros_like(out["F"], dtype=np.float64)
    for k, wt in w.items():
        if wt > 0:
            num += wt * np.where(av[k], out[k], 0.0)
            den += wt * av[k]
    valid = den > 0
    return np.where(valid, num / np.maximum(den, 1e-12), 0.0).astype(np.float32), valid


def recall10(R, V, ids, w) -> np.ndarray:
    rec = []
    for i, d in enumerate(ids):
        v = np.flatnonzero(V[i])
        top = v[np.lexsort((v, -R[i, v].astype(np.float64)))][:10]
        q = set(w.query_rows(d).tolist())
        rec.append(sum(1 for g in top if int(g) in q) / max(len(q), 1))
    return np.array(rec)


def tune(run: Runner) -> dict:
    srff = SRFF(run)
    comps = []
    for j in range(5):
        ids = run.w.labelled(run.w.spec.dev_diseases)
        comps.append((ids, *srff.components(f"f{j}_outer", ids)))
    grid = [c for c in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=3)
            if abs(sum(c) - 1.0) < 1e-6]
    rows = []
    for wF, wE, wS in grid:
        wt = {"F": wF, "E": wE, "S": wS}
        rec = np.concatenate([recall10(*combine(o, a, wt), ids, run.w) for ids, o, a in comps])
        rows.append({"F": wF, "E": wE, "S": wS, "dev_recall10": float(rec.mean())})
    df = pd.DataFrame(rows).sort_values("dev_recall10", ascending=False)
    df.to_csv(paths71.CORE / "srff_dev_grid.csv", index=False)
    best = df.iloc[0]
    out = {"weights": {"F": float(best["F"]), "E": float(best["E"]), "S": float(best["S"])},
           "dev_recall10": float(best["dev_recall10"]), "n_grid": len(df), "k_ref": K_REF,
           "decided_on": "development families x 5 outer contexts (full background)"}
    write_manifest("srff_v71.json", out)
    return out


def formal(run: Runner) -> None:
    if not load_manifest("protocol_v71_registered.json"):
        raise RuntimeError("protocol not registered")
    wt = load_manifest("srff_v71.json")["weights"]
    srff = SRFF(run)
    for j in range(5):
        run_id = f"V71Z__srff__f{j}"
        if is_done(run_id):
            continue
        ctx = f"f{j}_outer"
        test = run.w.labelled(run.w.fold(j)["test_diseases"])
        o, a = srff.components(ctx, test)
        R, V = combine(o, a, wt)
        gl, fill = core_list(R, V, context_prior(run, ctx), 500, fill=True)
        y = np.zeros(gl.shape, dtype=np.float32)
        nq = np.zeros(len(test))
        for k, d in enumerate(test):
            q = run.w.query_rows(d)
            nq[k] = len(q)
            y[k] = np.isin(gl[k], q)
        sc = -np.tile(np.arange(gl.shape[1], dtype=np.float64), (len(test), 1))
        with tracked(run_id, {"stage": "formal_standalone", "method": "srff", "fold": j,
                              "seed": "deterministic", "task": "Z", "config": wt}) as box:
            per = per_disease_frame(test, evaluate_matrix(sc, y, gl, nq, ks=(10, 20, 50)))
            per.to_parquet(run_dir(run_id) / "per_disease.parquet", index=False)
            pd.DataFrame({"disease_id": np.repeat(test, gl.shape[1]), "gene_row": gl.reshape(-1),
                          "rank": np.tile(np.arange(1, gl.shape[1] + 1), len(test)),
                          "label": y.reshape(-1).astype(np.int8)}).to_parquet(
                run_dir(run_id) / "predictions.parquet", index=False)
            box["extra"] = {"metrics": macro(per), "n_test_diseases": len(test)}
        print(f"  {run_id}: R@10 {box['extra']['metrics']['recall10_end']:.4f}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "formal"], required=True)
    args = ap.parse_args(argv)
    paths71.ensure()
    run = Runner()
    if args.stage == "tune":
        print(json.dumps(tune(run), indent=2))
    else:
        formal(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

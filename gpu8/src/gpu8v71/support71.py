"""E7 — small support sets (k = 1) with the frozen V6 support line.

The archived V6 support task (k = 0/1/3/5 on diseases with >= 6 known genes) is
kept unchanged and cited from V6.  Here the diseases the V6 design cannot use
(2-5 known genes) get a one-gene support episode:

* support = one known gene (5 deterministic draws per disease), query = the rest;
* ``sem``        frozen V6 zero-support fusion over the background (RRF, the
                 support gene excluded);
* ``rwr``        RWR from the support gene on the frozen V6 operator
                 (alpha, nu from support_v6_frozen.json);
* ``joint``      (1 - omega) sem + omega rwr, omega frozen in V6;
* ``core``       the V7.1 core of the outer fit (seed 42), RRF-normalised ranks;
* ``joint_core`` (1 - gamma) joint + gamma core, gamma in {0, .1, ..., .5} chosen
                 on the development families (gamma = 0 allowed).

Ranking is over the full background minus the support gene (no candidate list);
Recall@10 / @50 per episode, averaged over draws inside a disease first.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gpu8v6.common import load_manifest as lm6
from gpu8v6.retrieval import build_lists, max_normalise
from gpu8v6.sources import SOURCES
from gpu8v6.support_line import SupportExperiment, metrics_on

from . import paths71
from .chain71 import Runner, variant_spec
from .common71 import dump_json, load_manifest, rng_for, write_manifest

DRAWS = 5
GAMMAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def rrf_dist(R: np.ndarray, V: np.ndarray, k0: float = 60.0) -> np.ndarray:
    out = np.zeros(R.shape, dtype=np.float64)
    v = np.flatnonzero(V)
    if len(v) == 0:
        return out
    order = v[np.lexsort((v, -R[v].astype(np.float64)))]
    out[order] = 1.0 / (k0 + np.arange(1, len(order) + 1))
    return out / out.sum()


def episodes(exp: SupportExperiment, run: Runner, fold: int, ids: list[str], seed: int = 42) -> pd.DataFrame:
    w = exp.w
    sel = lm6("support_v6_frozen.json")
    alpha, nu, omega = sel["alpha"], sel["nu"], sel["omega"]
    cache = exp.cache(nu, alpha)
    ctx = w.outer_context(fold)
    cs = exp.eng.context(ctx)
    prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(exp.eng.ppi_deg)
    prior = prior / max(prior.max(), 1e-12)
    spec = variant_spec("core")
    tr, m = run.model(f"V71CHAIN__core__f{fold}_outer__s{seed}", spec)
    rows = []
    g_ix = w.idx.g_ix
    for b0 in range(0, len(ids), 200):
        chunk = ids[b0:b0 + 200]
        raw, _ = exp.eng.scores(cs, chunk, sources=SOURCES)
        norm = {s: max_normalise(v) for s, v in raw.items()}
        _, _, _, fused = build_lists(norm, exp.frozen_ret["weights"], prior, n=500,
                                     mode=exp.frozen_ret["mode"])
        R, V = tr.score(m, np.array([w.idx.d_ix[d] for d in chunk]))
        for i, d in enumerate(chunk):
            known = w.query_rows(d)
            p_core = rrf_dist(R[i], V[i])
            rng = rng_for("small_support", d)
            picks = rng.permutation(len(known))[:DRAWS] if len(known) >= DRAWS else \
                np.array([rng.integers(len(known)) for _ in range(DRAWS)])
            for draw, pi in enumerate(picks):
                sup = int(known[pi])
                q = np.array([g for g in known if g != sup])
                if len(q) == 0:
                    continue
                qset = set(q.tolist())
                bg = np.delete(np.arange(w.n_g, dtype=np.int64), sup)
                ps = fused[i].astype(np.float64).copy()
                ps[sup] = 0.0
                ps = ps / ps.sum() if ps.sum() > 0 else ps
                p_eq = cache.propagate({sup: 1.0})
                joint = (1 - omega) * ps + omega * p_eq if ps.sum() > 0 else p_eq
                base = {"disease_id": d, "fold": fold, "draw": draw, "n_known": len(known),
                        "n_query": len(q)}
                for meth, sc in (("sem", ps), ("rwr", p_eq), ("joint", joint), ("core", p_core)):
                    rows.append({**base, "method": meth, **metrics_on(sc[bg], bg, qset, len(q), ks=(10, 50))})
                for gm in GAMMAS[1:]:
                    sc = (1 - gm) * joint + gm * p_core
                    rows.append({**base, "method": f"joint_core_g{gm:g}",
                                 **metrics_on(sc[bg], bg, qset, len(q), ks=(10, 50))})
    return pd.DataFrame(rows)


def cohort(w, ids):
    return [d for d in w.labelled(ids) if 2 <= len(w.query_rows(d)) <= 5]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "test"], required=True)
    args = ap.parse_args(argv)
    paths71.ensure()
    if not load_manifest("protocol_v71_registered.json"):
        raise RuntimeError("protocol not registered")
    exp = SupportExperiment()
    run = Runner()
    w = exp.w
    if args.stage == "dev":
        frames = []
        dev = cohort(w, w.spec.dev_diseases)
        for f in range(5):
            frames.append(episodes(exp, run, f, dev))
        df = pd.concat(frames)
        df.to_parquet(paths71.EXT_RES / "small_support_dev.parquet", index=False)
        per = df.groupby(["method", "disease_id"])["recall10"].mean().groupby("method").mean()
        cands = {g: float(per.get("joint" if g == 0 else f"joint_core_g{g:g}", np.nan)) for g in GAMMAS}
        best = max(cands.items(), key=lambda kv: (kv[1], -kv[0]))
        out = {"gamma": best[0], "dev_recall10": cands, "n_dev_diseases": len(dev),
               "rule": "gamma maximising dev Recall@10 (k=1), ties -> smaller gamma; gamma=0 allowed"}
        write_manifest("small_support_gamma_v71.json", out)
        print(json.dumps(out, indent=2))
    else:
        gamma = load_manifest("small_support_gamma_v71.json")["gamma"]
        frames = []
        for f in range(5):
            frames.append(episodes(exp, run, f, cohort(w, w.fold(f)["test_diseases"])))
        df = pd.concat(frames)
        df.to_parquet(paths71.EXT_RES / "small_support_test.parquet", index=False)
        per = df.groupby(["method", "disease_id"])[["recall10", "recall50"]].mean()
        summ = per.groupby("method").mean()
        summ["n_diseases"] = per.groupby("method").size()
        summ.to_csv(paths71.TABLES_OUT / "e7_small_support.csv", encoding="utf-8-sig")
        dump_json(paths71.EXT_RES / "small_support_summary.json",
                  {"gamma_frozen": gamma, "summary": summ.reset_index().to_dict(orient="records")})
        print(summ.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

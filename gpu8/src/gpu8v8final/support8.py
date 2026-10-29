"""E9 — the support task S (a separate task; never added to Z into a "total accuracy").

Cohort 533: the V5/V6 dose design (fixed S_max pool of 5, fixed queries, nested
exposure orders), every 5-shot eligible test disease of the five folds; k in
{0, 1, 3, 5}.  Cohort 1-shot: diseases with 2-5 known genes, five one-gene draws,
the rest are queries (V7.1 design).  Support genes are removed from every list.

Methods (same queries, same exposures):
  sem_v6       the frozen V6 zero-support score (V6 list)             [archived design]
  rwr          RWR from the exposed genes, frozen alpha / nu of V6
  joint_v6     (1 - omega_6) sem_v6 + omega_6 rwr (omega_6 frozen in V6)
  sem_v8       the deployed V8.1 zero-support ranking (strict 500 list; background
               regime: the availability-aware RRF over all genes)
  joint_v8     (1 - omega_8) sem_v8 + omega_8 rwr, omega_8 re-chosen on the development
               families (registered as a time-stamped appendix before test use)
  qwalk        continuous-time quantum walk from the exposed genes (PPI), all folds
Every list is scored per episode; draws are averaged inside a disease first.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from gpu8v6.common import World6
from gpu8v6.common import load_manifest as lm6
from gpu8v6.gene_features import load_bridge
from gpu8v6.propagate import BridgeOperator
from gpu8v6.retrieval import build_lists, max_normalise
from gpu8v6.sources import SOURCES
from gpu8v6.support_line import SupportExperiment, metrics_on
from gpu8.support import build_dose_design

from . import paths8
from .common8 import dump_json, load_manifest, now_utc, rng_for
from .roles8 import Roles, RoleSourceEngine

OUT = paths8.EXT_RES / "support"
OMEGAS = (0.25, 0.5, 0.75)
K_SUPPORT = (1, 3, 5)


class Support8(SupportExperiment):
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.w = World6()
        self.roles = Roles(self.w)
        self.eng = RoleSourceEngine(self.w, self.roles, device=device)
        self.frozen_ret = lm6("retriever_v6_frozen.json")
        self.frozen_score = lm6("scorer_v6_frozen.json")
        self.cfg5 = self.w.cfg
        self.ppi = self.eng.op
        self.bridge = BridgeOperator(load_bridge("current"), device=device)
        deg = self.eng.ppi_deg
        q = np.quantile(deg[deg > 0], [1 / 3, 2 / 3])
        self.strata = np.digitize(deg, q)
        self.deg = deg
        self.caches, self.released_logs = {}, {}
        ppi = self.w.ppi[self.w.ppi["network_type"] == "physical"]
        a = ppi["gene_a"].map(self.w.idx.g_ix)
        b = ppi["gene_b"].map(self.w.idx.g_ix)
        k = a.notna() & b.notna()
        a, b = a[k].astype(int).to_numpy(), b[k].astype(int).to_numpy()
        A = sparse.coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(self.w.n_g,) * 2).tocsr()
        A.data[:] = 1.0
        self.adj = A


def qwalk(adj: sparse.csr_matrix, seeds: np.ndarray, t: float = 0.45, diag: float = 5.0) -> np.ndarray:
    n = adj.shape[0]
    D = sparse.coo_matrix((np.full(len(seeds), diag), (seeds, seeds)), shape=(n, n)).tocsr()
    H = (adj + D).tocsc()
    Z = np.zeros((n, len(seeds)), dtype=np.complex128)
    Z[seeds, np.arange(len(seeds))] = 1.0
    return (np.abs(expm_multiply(-1j * t * H, Z)) ** 2).sum(axis=1)


def rrf_over(genes: np.ndarray, scores: np.ndarray, G: int, k0: float = 60.0) -> np.ndarray:
    p = np.zeros(G)
    o = genes[np.lexsort((genes, -scores))]
    p[o] = 1.0 / (k0 + np.arange(1, len(o) + 1))
    return p / p.sum() if p.sum() > 0 else p


def v8_list(fold: int, role: str, seed: int = 42) -> dict:
    z = np.load(paths8.FORMAL / "main" / f"s{seed}" / f"fold{fold}" / f"pred_deployed__{role}.npz")
    return {str(d): (z["genes"][i].astype(np.int64), z["score"][i].astype(np.float64)) for i, d in enumerate(z["ids"])}


def aware_background(fold: int, ids: list, seed: int = 42) -> dict:
    """Availability-aware RRF over the full background (the V8.1 candidate score) per disease."""
    from .chain8 import Chain
    from .formal8 import selection
    from .pack8 import ArmSpec
    from .pool8 import rrf_score
    from .v6pass8 import V6Pass, V6World
    ch = Chain(ArmSpec(), V6World(), role="variant")
    sel = selection(fold)
    rid = ch.teacher(f"f{fold}_outer", seed, sel["controls"])
    v6p = V6Pass(V6World(), f"f{fold}_outer")
    sem = ch.semantic(f"f{fold}_outer", rid, v6p, None)
    sp = {d: i for i, d in enumerate(sem.ids)}
    out = {}
    for d in ids:
        if d not in v6p.pos:
            continue
        rB = np.asarray(v6p.rB[v6p.pos[d]]).astype(np.int32)[None]
        i = sp[d]
        out[d] = rrf_score(rB, sem.rS[i][None], sem.m[i][None], sel["w"]["w"])[0]
    del ch
    return out


def cohort533(exp: Support8, fold: int, ids: list, role: str, omegas, seed: int = 42, draws: int = 10,
              with_qwalk: bool = True) -> pd.DataFrame:
    w = exp.w
    frozen6 = lm6("support_v6_frozen.json")
    alpha, nu, om6 = frozen6["alpha"], frozen6["nu"], frozen6["omega"]
    cache = exp.cache(nu, alpha)
    cases = exp.cases(w.outer_context(fold), ids, fold)
    lists = v8_list(fold, role, seed)
    rows = []
    for c in cases:
        if c.disease_id not in lists:
            continue
        g8, s8 = lists[c.disease_id]
        keep = ~np.isin(g8, c.pool_rows)
        g8, s8 = g8[keep], s8[keep]
        qset = set(c.query_rows.tolist())
        nq = len(c.query_rows)
        p8 = rrf_over(g8, s8, w.n_g)
        Fs = c.F / c.F.sum() if c.F.sum() > 0 else np.full(len(c.F), 1.0 / len(c.F))
        base = {"disease_id": c.disease_id, "fold": fold, "n_query": nq}
        rows.append({**base, "k": 0, "draw": 0, "method": "sem_v6", **metrics_on(c.F, c.cand, qset, nq)})
        rows.append({**base, "k": 0, "draw": 0, "method": "sem_v8", **metrics_on(s8, g8, qset, nq)})
        cache.ensure(c.pool_rows)
        for draw in range(draws):
            des = c.design if draw == 0 else build_dose_design(exp.cfg5, c.disease_id, w.positives[c.disease_id],
                                                               draw=draw)
            for k in K_SUPPORT:
                if k == max(K_SUPPORT) and draw > 0:
                    continue
                sup = [w.idx.g_ix[g] for g in des.support_at(k) if g in w.idx.g_ix]
                p_eq = cache.propagate({g: 1.0 / len(sup) for g in sup})
                b2 = {**base, "k": k, "draw": draw}
                rows.append({**b2, "method": "rwr_v6list", **metrics_on(p_eq[c.cand], c.cand, qset, nq)})
                rows.append({**b2, "method": "rwr_v8list", **metrics_on(p_eq[g8], g8, qset, nq)})
                pe6 = p_eq[c.cand] / max(p_eq[c.cand].sum(), 1e-300)
                rows.append({**b2, "method": "joint_v6", **metrics_on((1 - om6) * Fs + om6 * pe6, c.cand, qset, nq)})
                pe8 = p_eq[g8] / max(p_eq[g8].sum(), 1e-300)
                for om in omegas:
                    rows.append({**b2, "method": f"joint_v8_w{om:g}",
                                 **metrics_on((1 - om) * p8[g8] + om * pe8, g8, qset, nq)})
                if with_qwalk and draw == 0:
                    qs = qwalk(exp.adj, np.array(sup, dtype=np.int64))
                    rows.append({**b2, "method": "qwalk_v8list", **metrics_on(qs[g8], g8, qset, nq)})
                    rows.append({**b2, "method": "qwalk_v6list", **metrics_on(qs[c.cand], c.cand, qset, nq)})
    return pd.DataFrame(rows)


def cohort1shot(exp: Support8, fold: int, ids: list, omegas, seed: int = 42) -> pd.DataFrame:
    w = exp.w
    frozen6 = lm6("support_v6_frozen.json")
    alpha, nu, om6 = frozen6["alpha"], frozen6["nu"], frozen6["omega"]
    cache = exp.cache(nu, alpha)
    ctx = w.outer_context(fold)
    cs = exp.eng.context(ctx)
    prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(exp.eng.ppi_deg)
    prior = prior / max(prior.max(), 1e-12)
    R8 = aware_background(fold, ids, seed)
    rows = []
    for b0 in range(0, len(ids), 200):
        chunk = ids[b0:b0 + 200]
        raw, _ = exp.eng.scores(cs, chunk, sources=SOURCES)
        norm = {s: max_normalise(v) for s, v in raw.items()}
        _, _, _, fused = build_lists(norm, exp.frozen_ret["weights"], prior, n=500, mode=exp.frozen_ret["mode"])
        for i, d in enumerate(chunk):
            known = w.query_rows(d)
            rng = rng_for("small_support", d)
            picks = rng.permutation(len(known))[:5] if len(known) >= 5 else \
                np.array([rng.integers(len(known)) for _ in range(5)])
            r8 = R8.get(d)
            for draw, pi in enumerate(picks):
                sup = int(known[pi])
                q = [g for g in known if g != sup]
                if not q:
                    continue
                qset = set(q)
                bg = np.delete(np.arange(w.n_g, dtype=np.int64), sup)
                ps = fused[i].astype(np.float64).copy()
                ps[sup] = 0.0
                ps = ps / ps.sum() if ps.sum() > 0 else ps
                p_eq = cache.propagate({sup: 1.0})
                base = {"disease_id": d, "fold": fold, "draw": draw, "k": 1, "n_query": len(q)}
                j6 = (1 - om6) * ps + om6 * p_eq if ps.sum() > 0 else p_eq
                for meth, sc in (("sem_v6", ps), ("rwr", p_eq), ("joint_v6", j6)):
                    rows.append({**base, "method": meth, **metrics_on(sc[bg], bg, qset, len(q))})
                if r8 is not None:
                    p8 = r8.copy()
                    p8[sup] = 0.0
                    p8 = p8 / p8.sum()
                    rows.append({**base, "method": "sem_v8", **metrics_on(p8[bg], bg, qset, len(q))})
                    for om in omegas:
                        rows.append({**base, "method": f"joint_v8_w{om:g}",
                                     **metrics_on(((1 - om) * p8 + om * p_eq)[bg], bg, qset, len(q))})
    return pd.DataFrame(rows)


def per_disease_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["method", "k", "disease_id"])[["recall10", "recall50"]].mean().reset_index()


def run(stage: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    exp = Support8()
    w = exp.w
    small = lambda ids: [d for d in w.labelled(ids) if 2 <= len(w.query_rows(d)) <= 5]
    if stage == "dev":
        f533 = [cohort533(exp, f, w.labelled(w.spec.dev_diseases), "dev", OMEGAS, draws=1, with_qwalk=False)
                for f in range(5)]
        d533 = per_disease_mean(pd.concat(f533))
        c533 = d533[d533["k"] == 5].groupby("method")["recall10"].mean()
        best533 = max(OMEGAS, key=lambda o: (c533.get(f"joint_v8_w{o:g}", -1), -o))
        f1 = [cohort1shot(exp, f, small(w.spec.dev_diseases), OMEGAS) for f in range(5)]
        d1 = per_disease_mean(pd.concat(f1))
        c1 = d1.groupby("method")["recall10"].mean()
        best1 = max(OMEGAS, key=lambda o: (c1.get(f"joint_v8_w{o:g}", -1), -o))
        out = {"omega_533": best533, "omega_1shot": best1, "dev_k5_recall10": c533.to_dict(),
               "dev_1shot_recall10": c1.to_dict(), "rule": "max development Recall@10 (k = 5 for the 533 design; "
                                                           "k = 1 for the one-shot cohort); ties -> smaller omega"}
        from .register8 import appendix
        appendix("support_omega_v8", out)
        dump_json(OUT / "support_dev_v8.json", out)
        return out
    reg = load_manifest("protocol_v8_registered.json")
    app = [a for a in reg.get("appendices", []) if a["name"] == "support_omega_v8"]
    if not app:
        raise RuntimeError("support omega not registered")
    om533, om1 = app[-1]["omega_533"], app[-1]["omega_1shot"]
    t0 = time.time()
    p533 = OUT / "support533_test_episodes.parquet"
    if p533.exists():                       # completed before an interruption: identical deterministic run
        e533 = pd.read_parquet(p533)
    else:
        f533 = []
        for f in range(5):
            f533.append(cohort533(exp, f, w.labelled(w.fold(f)["test_diseases"]), "test", (om533,)))
            print(f"  support 533 fold {f} ({time.time() - t0:.0f}s)", flush=True)
        e533 = pd.concat(f533)
        e533.to_parquet(p533, index=False)
    f1 = []
    for f in range(5):
        f1.append(cohort1shot(exp, f, small(w.fold(f)["test_diseases"]), (om1,)))
        print(f"  support 1-shot fold {f} ({time.time() - t0:.0f}s)", flush=True)
    e1 = pd.concat(f1)
    e1.to_parquet(OUT / "support1shot_test_episodes.parquet", index=False)
    s533 = per_disease_mean(e533).groupby(["method", "k"]).agg(recall10=("recall10", "mean"),
                                                               recall50=("recall50", "mean"),
                                                               n_diseases=("disease_id", "nunique")).reset_index()
    s1 = per_disease_mean(e1).groupby(["method"]).agg(recall10=("recall10", "mean"), recall50=("recall50", "mean"),
                                                      n_diseases=("disease_id", "nunique")).reset_index()
    s533.to_csv(OUT / "support533_summary.csv", index=False, encoding="utf-8-sig")
    s1.to_csv(OUT / "support1shot_summary.csv", index=False, encoding="utf-8-sig")
    out = {"generated_utc": now_utc(), "omega_533": om533, "omega_1shot": om1,
           "cohort533": s533.to_dict(orient="records"), "cohort1shot": s1.to_dict(orient="records"),
           "seconds": time.time() - t0, "note": "support metrics are reported separately from task Z"}
    dump_json(OUT / "support_test_v8.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "test"], required=True)
    a = ap.parse_args()
    paths8.ensure()
    print(json.dumps(run(a.stage), indent=1, default=str)[:3000])

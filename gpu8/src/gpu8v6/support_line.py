"""V6 sections 9-10 — the support line: graph propagation as the main path.

Dose design: the V5 fixed-support design is reused exactly (same S_max pool of 5,
same query, same ten nested exposure orders), for *every* 5-shot eligible test
disease of all five folds (out-of-fold), not only fold 0.

For each disease, draw and k in {0, 1, 3, 5} the module evaluates:

* ``sem``         zero-support semantic score (dev-tuned fusion F) — k independent
* ``rwr_equal``   RWR from uniform seeds u_d on operator P (PPI or P_mix)
* ``rwr_group``   RWR from p0 = (1 - eta) u + eta v, eta in {0.25, 0.5, 1}
* ``rwr_shuffle`` same with source memberships permuted within degree strata
* ``rwr_degree``  RWR from connectivity-only weights
* ``joint``       (1 - omega) p_sem + omega p_support

under three candidate regimes:

* S-fixed     the zero-support 500 list (S_max excluded) — support only rescores;
* S-adaptive  the list is regenerated from the joint score over the background;
* background  every gene except the S_max pool is ranked (no candidate list).

The four-layer seed diagnostics (seed L1, propagated L1 with the eta*||v-u||_1
bound, ranking change, metric change) are recorded per episode.  alpha, nu and
omega are chosen on the development families only and frozen before the test
cohort is scored.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gpu8.support import build_dose_design

from . import paths6
from .common import World6, dump_json, load_manifest, now_utc, rng_for, write_manifest
from .gene_features import load_bridge
from .propagate import BridgeOperator, MixOperator, PPIOperator, SeedColumnCache
from .retrieval import build_lists, fuse, max_normalise
from .sources import SOURCES, SourceEngine
from .support_seed import build_support_seed, degree_weights, shuffled_weights

K_SUPPORT = (1, 3, 5)
ETAS = (0.25, 0.5, 1.0)
N_SHUFFLE = 20
BUDGET = 500


@dataclass
class DiseaseCase:
    disease_id: str
    fold: int
    design: object
    pool_rows: np.ndarray
    query_rows: np.ndarray
    cand: np.ndarray              # S-fixed candidate genes (500)
    F: np.ndarray                 # tuned fusion on cand
    p_sem: np.ndarray             # semantic distribution over the background
    has_sem: bool


def metrics_on(scores: np.ndarray, genes: np.ndarray, qset: set, n_query: int,
               ks=(10, 50)) -> dict:
    order = np.lexsort((genes, -scores))
    top = genes[order]
    out = {}
    for k in ks:
        h = sum(1 for g in top[:k] if int(g) in qset)
        out[f"recall{k}"] = h / max(n_query, 1)
    lab = np.array([int(g) in qset for g in top])
    pos = np.flatnonzero(lab)
    out["mrr"] = 1.0 / (pos[0] + 1) if len(pos) else 0.0
    out["coverage"] = lab.sum() / max(n_query, 1)
    return out


def top10_set(scores, genes):
    order = np.lexsort((genes, -scores))
    return set(int(g) for g in genes[order[:10]])


class SupportExperiment:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.w = World6()
        self.eng = SourceEngine(self.w, device=device, with_v5=False)
        self.frozen_ret = load_manifest("retriever_v6_frozen.json")
        self.frozen_score = load_manifest("scorer_v6_frozen.json")
        self.cfg5 = self.w.cfg          # V5 frozen config: identical dose designs
        self.ppi = self.eng.op
        self.bridge = BridgeOperator(load_bridge("current"), device=device)
        deg = self.eng.ppi_deg
        q = np.quantile(deg[deg > 0], [1 / 3, 2 / 3])
        self.strata = np.digitize(deg, q)
        self.deg = deg
        self.caches: dict = {}
        self.released_logs: dict = {}

    def op(self, nu: float):
        return self.ppi if nu == 0 else MixOperator(self.ppi, self.bridge, nu)

    def cache(self, nu: float, alpha: float) -> SeedColumnCache:
        key = (nu, alpha)
        if key not in self.caches:
            self.caches[key] = SeedColumnCache(self.op(nu), alpha, eps=1e-6, device=self.device)
        return self.caches[key]

    def release(self, nu: float, alpha: float) -> None:
        """Drop a column cache (its log is kept) to bound memory."""
        c = self.caches.pop((nu, alpha), None)
        if c is not None:
            self.released_logs[f"{nu}_{alpha}"] = {"log": c.log, "failures": c.failures()}

    # -- cohort ------------------------------------------------------------
    def cases(self, ctx, disease_ids: list[str], fold: int) -> list[DiseaseCase]:
        w = self.w
        g_ix = w.idx.g_ix
        designs = {}
        for d in disease_ids:
            des = build_dose_design(self.cfg5, d, w.positives.get(d, []), draw=0)
            if des is not None:
                designs[d] = des
        ids = sorted(designs)
        if not ids:
            return []
        cs = self.eng.context(ctx)
        prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(self.eng.ppi_deg)
        prior = prior / max(prior.max(), 1e-12)
        out = []
        for b0 in range(0, len(ids), 200):
            chunk = ids[b0:b0 + 200]
            raw, _ = self.eng.scores(cs, chunk, sources=SOURCES)
            norm = {s: max_normalise(v) for s, v in raw.items()}
            excl = np.zeros((len(chunk), w.n_g), dtype=bool)
            for i, d in enumerate(chunk):
                excl[i, [g_ix[g] for g in designs[d].support_pool if g in g_ix]] = True
            genes, _, _, fused = build_lists(norm, self.frozen_ret["weights"], prior, n=BUDGET,
                                             exclude=excl, mode=self.frozen_ret["mode"])
            wts = self.frozen_score["weights"]
            for i, d in enumerate(chunk):
                des = designs[d]
                cand = genes[i]
                Fv = np.zeros(BUDGET)
                for s, wt in wts.items():
                    Fv += wt * norm[s][i, cand]
                ps = fused[i].astype(np.float64).copy()
                ps[excl[i]] = 0.0
                tot = ps.sum()
                ps = ps / tot if tot > 0 else ps
                out.append(DiseaseCase(
                    disease_id=d, fold=fold, design=des,
                    pool_rows=np.array([g_ix[g] for g in des.support_pool if g in g_ix]),
                    query_rows=np.array([g_ix[g] for g in des.query_genes if g in g_ix]),
                    cand=cand, F=Fv, p_sem=ps, has_sem=tot > 0))
        return out

    # -- one configuration over a cohort -----------------------------------
    def run_cases(self, cases: list[DiseaseCase], alpha: float, nu: float, omegas=(0.5,),
                  draws: int = 10, with_mechanism: bool = True, adaptive: bool = True,
                  tag: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
        cache = self.cache(nu, alpha)
        cache.ensure(np.concatenate([c.pool_rows for c in cases]))
        w = self.w
        rows, diag = [], []
        for c in cases:
            qset = set(c.query_rows.tolist())
            nq = len(c.query_rows)
            excl = set(c.pool_rows.tolist())
            bg = np.array([g for g in range(w.n_g) if g not in excl], dtype=np.int64)
            Fs = c.F / c.F.sum() if c.F.sum() > 0 else np.full(len(c.F), 1.0 / len(c.F))
            base = {"disease_id": c.disease_id, "fold": c.fold, "alpha": alpha, "nu": nu,
                    "n_query": nq, "tag": tag}
            m = metrics_on(c.F, c.cand, qset, nq)
            rows.append({**base, "k": 0, "draw": 0, "method": "sem", "regime": "S_fixed", **m})
            if adaptive:
                m = metrics_on(c.p_sem[bg], bg, qset, nq, ks=(10, 50))
                rows.append({**base, "k": 0, "draw": 0, "method": "sem", "regime": "background", **m})
            for draw in range(draws):
                des = c.design if draw == 0 else build_dose_design(
                    self.cfg5, c.disease_id, w.positives[c.disease_id], draw=draw)
                for k in K_SUPPORT:
                    if k == max(K_SUPPORT) and draw > 0:
                        continue
                    sup = des.support_at(k)
                    seed = build_support_seed(c.disease_id, sup, w.relation_of, w.relation_pmids)
                    srows = np.array([w.idx.g_ix[g] for g in seed.genes], dtype=np.int64)
                    p_eq = cache.propagate(dict(zip(srows.tolist(), seed.u)))
                    b2 = {**base, "k": k, "draw": draw}
                    s_eq = p_eq[c.cand]
                    m_eq = metrics_on(s_eq, c.cand, qset, nq)
                    rows.append({**b2, "method": "rwr_equal", "regime": "S_fixed", **m_eq})
                    peq_c = s_eq / s_eq.sum() if s_eq.sum() > 0 else s_eq
                    for om in omegas:
                        sj = (1 - om) * Fs + om * peq_c
                        rows.append({**b2, "method": f"joint_w{om:g}", "regime": "S_fixed",
                                     **metrics_on(sj, c.cand, qset, nq)})
                        if adaptive:
                            pj = (1 - om) * c.p_sem + om * p_eq if c.has_sem else p_eq
                            sc = pj[bg]
                            rows.append({**b2, "method": f"joint_w{om:g}", "regime": "background",
                                         **metrics_on(sc, bg, qset, nq, ks=(10, 50))})
                            top = bg[np.lexsort((bg, -sc))[:BUDGET]]
                            m_ad = metrics_on(pj[top], top, qset, nq)
                            rows.append({**b2, "method": f"joint_w{om:g}", "regime": "S_adaptive",
                                         **m_ad})
                    if adaptive:
                        sc = p_eq[bg]
                        rows.append({**b2, "method": "rwr_equal", "regime": "background",
                                     **metrics_on(sc, bg, qset, nq, ks=(10, 50))})
                    if not with_mechanism:
                        continue
                    # ---- evidence-group seeds (section 10) ----------------
                    dw = degree_weights(self.deg[srows])
                    p_deg = cache.propagate(dict(zip(srows.tolist(), dw)))
                    rows.append({**b2, "method": "rwr_degree", "regime": "S_fixed",
                                 **metrics_on(p_deg[c.cand], c.cand, qset, nq)})
                    trace = seed.all_traceable
                    for eta in ETAS:
                        if not trace:
                            continue
                        p0 = seed.mixed(eta)
                        p_gr = cache.propagate(dict(zip(srows.tolist(), p0)))
                        s_gr = p_gr[c.cand]
                        m_gr = metrics_on(s_gr, c.cand, qset, nq)
                        rows.append({**b2, "method": f"rwr_group_eta{eta:g}", "regime": "S_fixed",
                                     **m_gr})
                        l1_seed = float(np.abs(p0 - seed.u).sum())
                        l1_prop = float(np.abs(p_gr - p_eq).sum())
                        rho = spearmanr(s_gr, s_eq).correlation if l1_seed > 0 else 1.0
                        t_eq, t_gr = top10_set(s_eq, c.cand), top10_set(s_gr, c.cand)
                        diag.append({**b2, "eta": eta, "l1_v_minus_u": seed.l1(),
                                     "l1_seed": l1_seed, "l1_propagated": l1_prop,
                                     "bound_eta_l1": eta * seed.l1(),
                                     "bound_holds": l1_prop <= eta * seed.l1() + 1e-6,
                                     "entropy_u": seed.entropy("u"), "entropy_v": seed.entropy("v"),
                                     "n_sources": seed.n_sources,
                                     "source_coverage": seed.source_coverage,
                                     "spearman_candidates": float(rho) if rho == rho else 1.0,
                                     "top10_changed": len(t_eq - t_gr),
                                     "delta_recall10": m_gr["recall10"] - m_eq["recall10"]})
                        if eta == 1.0:
                            rng = rng_for("shuffle", c.disease_id, draw, k)
                            accs = []
                            for _ in range(N_SHUFFLE):
                                vs = shuffled_weights(seed, self.strata[srows], rng)
                                ps = cache.propagate(dict(zip(srows.tolist(), vs)))
                                accs.append(metrics_on(ps[c.cand], c.cand, qset, nq))
                            mean_sh = {kk: float(np.mean([a[kk] for a in accs])) for kk in accs[0]}
                            rows.append({**b2, "method": "rwr_shuffle_eta1", "regime": "S_fixed",
                                         **mean_sh})
                    rows.append({**b2, "method": "traceable", "regime": "meta",
                                 "recall10": float(trace), "recall50": seed.source_coverage,
                                 "mrr": seed.n_sources, "coverage": seed.l1() if trace else np.nan})
        return pd.DataFrame(rows), pd.DataFrame(diag)


def select_on_dev(exp: SupportExperiment, folds=(0, 1, 2, 3, 4)) -> dict:
    """alpha (PPI), nu (bridge) and omega (joint) from the development families."""
    w = exp.w
    dev = w.labelled(w.spec.dev_diseases)
    cases = []
    for f in folds:
        cases += exp.cases(w.outer_context(f), dev, f)
    grid = []
    for alpha in (0.1, 0.2, 0.4):
        for nu in (0.0, 0.1, 0.3):
            df, _ = exp.run_cases(cases, alpha, nu, omegas=(0.25, 0.5, 0.75), draws=1,
                                  with_mechanism=False, adaptive=False, tag="dev")
            exp.release(nu, alpha)
            sub = df[(df["regime"] == "S_fixed") & (df["k"] == 5)]
            for meth, s2 in sub.groupby("method"):
                grid.append({"alpha": alpha, "nu": nu, "method": meth,
                             "recall10_k5": float(s2["recall10"].mean()),
                             "n_cases": int(s2["disease_id"].nunique())})
            print(f"  dev alpha={alpha} nu={nu}", flush=True)
    g = pd.DataFrame(grid)
    g.to_csv(paths6.SUPPORT / "support_selection_dev.csv", index=False, encoding="utf-8-sig")
    eq = g[g["method"] == "rwr_equal"]
    ppi = eq[eq["nu"] == 0].sort_values("recall10_k5", ascending=False).iloc[0]
    best_nu = eq[eq["alpha"] == ppi["alpha"]].sort_values("recall10_k5", ascending=False).iloc[0]
    jt = g[g["method"].str.startswith("joint") & (g["alpha"] == ppi["alpha"])
           & (g["nu"] == best_nu["nu"])].sort_values("recall10_k5", ascending=False).iloc[0]
    sel = {"alpha": float(ppi["alpha"]), "nu": float(best_nu["nu"]),
           "omega": float(jt["method"].split("_w")[1]),
           "dev_cases": int(len(cases)),
           "dev_recall10_k5": {"rwr_equal_ppi": float(ppi["recall10_k5"]),
                               "rwr_equal_best_nu": float(best_nu["recall10_k5"]),
                               "joint": float(jt["recall10_k5"])},
           "rule": "alpha by rwr_equal on PPI at k=5 (S-fixed); nu at that alpha; omega for joint at (alpha, nu)"}
    write_manifest("support_v6_frozen.json", sel)
    return sel


def run_test(exp: SupportExperiment, sel: dict, folds=(0, 1, 2, 3, 4)) -> dict:
    w = exp.w
    frames, diags = [], []
    for f in folds:
        fa = w.fold(f)
        cases = exp.cases(w.outer_context(f), w.labelled(fa["test_diseases"]), f)
        for nu in sorted({0.0, sel["nu"]}):
            df, dg = exp.run_cases(cases, sel["alpha"], nu, omegas=(sel["omega"],), draws=10,
                                   with_mechanism=(nu == sel["nu"]), adaptive=True,
                                   tag=f"test_nu{nu:g}")
            frames.append(df)
            diags.append(dg)
        print(f"  test fold {f}: {len(cases)} cases", flush=True)
    res = pd.concat(frames, ignore_index=True)
    dg = pd.concat(diags, ignore_index=True)
    res.to_parquet(paths6.SUPPORT / "support_test_episodes.parquet", index=False)
    dg.to_parquet(paths6.SUPPORT / "support_test_seed_diagnostics.parquet", index=False)
    return {"n_rows": int(len(res)), "n_cases": int(res["disease_id"].nunique())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V6 support line.")
    ap.add_argument("--stage", choices=["select", "test", "all"], default="all")
    args = ap.parse_args(argv)
    paths6.ensure()
    t0 = time.time()
    exp = SupportExperiment()
    sel = load_manifest("support_v6_frozen.json")
    if args.stage in ("select", "all") or not sel:
        sel = select_on_dev(exp)
        print(json.dumps(sel, indent=2), flush=True)
    if args.stage in ("test", "all"):
        print(json.dumps(run_test(exp, sel), indent=2), flush=True)
    logs = {f"{k[0]}_{k[1]}": {"log": c.log, "failures": c.failures()}
            for k, c in exp.caches.items()}
    logs.update({f"released_{k}": v for k, v in exp.released_logs.items()})
    dump_json(paths6.SUPPORT / "propagation_log.json",
              {"generated_utc": now_utc(), "caches": logs,
               "total_failures": int(sum(v["failures"] for v in logs.values())),
               "seconds": time.time() - t0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

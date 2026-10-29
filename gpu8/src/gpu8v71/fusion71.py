"""V7.1 section 12 — second-level models on out-of-fold work pools.

A ``PoolBlock`` holds, for one context and a list of target diseases, the work
pool of every disease (padded to the longest pool, with a validity mask), the
V6 channel features (the 30 V6 ``FEATS``), the core features of one core fit and
the labels (evaluator / training loss only).

Second-level models (each ranks the pool once; no double truncation):

* ``lgb``    LightGBM lambdarank trained on the OOF pools of the outer training
             diseases (three inner contexts), early-stopped on F_{j+1};
* ``fixed``  non-negative fixed fusion (1 - beta) F7 + beta r+, beta chosen on the
             development families (seed-42 pools of the five outer contexts);
* feature sets: ``channels`` (V6 FEATS only), ``late`` (FEATS + core features),
  ``maskonly`` (FEATS + core availability only), ``rawconcat`` (FEATS + raw
  modality vectors, PCA-reduced and fitted without labels).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from gpu8v6.common import candidate_hash
from gpu8v6.metrics6 import evaluate_matrix, macro, per_disease_frame
from gpu8v6.scoring import FEATS
from gpu8v6.sources import SOURCES

from . import paths71
from .chain71 import SEEDS, Runner, variant_spec
from .common71 import dump_json, load_manifest, now_utc, write_manifest
from .ledger71 import is_done, run_dir, tracked
from .pools71 import load_v6_list, v6_dis_path, v6_path
from .score71 import pool_path

N_G = 19861
CORE_FEATS = ["c_r", "c_valid", "c_rank", "c_in500", "c_sS", "c_sF", "c_sE", "c_aS", "c_aF", "c_aE",
              "c_wS", "c_wF", "c_wE", "c_cov"]
MASK_FEATS = ["c_valid", "c_aS", "c_aF", "c_aE", "c_cov"]
F7_W = None


def f7_weights() -> dict:
    global F7_W
    if F7_W is None:
        from gpu8v6.common import load_manifest as lm6
        F7_W = lm6("scorer_v6_frozen.json")["F7"]["weights"]
    return F7_W


@dataclass
class PoolBlock:
    context_id: str
    ids: list[str]
    genes: np.ndarray            # [D, N] (-1 padded)
    valid: np.ndarray            # [D, N]
    X: np.ndarray                # [D, N, F]
    y: np.ndarray                # [D, N]
    n_query: np.ndarray          # [D]
    feat_names: list[str]
    extra: dict = field(default_factory=dict)

    @property
    def D(self):
        return len(self.ids)

    def feat(self, name):
        return self.X[:, :, self.feat_names.index(name)]

    def subset_rows(self, idx):
        return PoolBlock(self.context_id, [self.ids[i] for i in idx], self.genes[idx], self.valid[idx],
                         self.X[idx], self.y[idx], self.n_query[idx], self.feat_names,
                         {k: v[idx] for k, v in self.extra.items()})


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
class Assembler:
    def __init__(self, run: Runner | None = None, device: str = "cuda"):
        self.run = run or Runner(device)
        self.w = self.run.w
        self._v6 = {}
        self._raw = None

    def v6_tables(self, ctx):
        if ctx not in self._v6:
            self._v6.clear()
            f = pd.read_parquet(v6_path(ctx))
            d = pd.read_parquet(v6_dis_path(ctx)).set_index("disease_id")
            self._v6[ctx] = (f, d)
        return self._v6[ctx]

    def core_outputs(self, variant, seed, ctx, ids, T_override: dict | None = None):
        """Full-background core outputs (r, valid, per-modality s/a/w, coverage, rank).

        ``T_override`` replaces the feature tensors (behaviour checks: shutdown or
        permuted modalities) while the fitted weights stay the same."""
        spec = variant_spec(variant)
        tr, m = self.run.model(f"V71CHAIN__{variant}__{ctx}__s{seed}", spec)
        if T_override is not None:
            from .train71 import CoreTrainer
            tr = CoreTrainer(T_override, self.run.device)
        rows = np.array([self.w.idx.d_ix[d] for d in ids])
        T = tr.T
        from .train71 import D_KEYS, take
        out = {k: np.zeros((len(ids), N_G), dtype=np.float32) for k in
               ["r", "sS", "sF", "sE", "wS", "wF", "wE", "cov"]}
        for k in ("valid", "aS", "aF", "aE"):
            out[k] = np.zeros((len(ids), N_G), dtype=bool)
        with torch.no_grad():
            gr = m.gene_repr(tr.g_all)
            for b0 in range(0, len(rows), 128):
                idx = torch.as_tensor(rows[b0:b0 + 128], device=tr.device)
                d = take(T, D_KEYS, idx)
                o = m(d, tr.g_all, gene_reprs=gr)
                sl = slice(b0, b0 + len(idx))
                out["r"][sl] = o["r"].float().cpu().numpy()
                out["valid"][sl] = o["valid"].cpu().numpy()
                if "s" in o:
                    for i, mod in enumerate(m.mods):
                        if mod in ("S", "F", "E"):
                            out["s" + mod][sl] = o["s"][i].float().cpu().numpy()
                            out["a" + mod][sl] = o["avail"][i].cpu().numpy()
                            out["w" + mod][sl] = o["w"][i].float().cpu().numpy()
                out["cov"][sl] = (d["q"] @ T["o"].T).float().cpu().numpy()
        rank = np.zeros((len(ids), N_G), dtype=np.float32)
        for i in range(len(ids)):
            v = np.flatnonzero(out["valid"][i])
            order = v[np.lexsort((v, -out["r"][i, v].astype(np.float64)))]
            rank[i, order] = np.log(N_G / np.arange(1, len(order) + 1))
        out["rank"] = rank
        return out

    def block(self, ctx: str, ids: list[str], variant: str | None, seed: int | None,
              pool: str = "pool", featset: str = "late", core_for_feats: tuple | None = None
              ) -> PoolBlock:
        """pool: 'pool' (V6 500 U core top-500 of (variant, seed)) or 'v6' (V6 500 only).
        core_for_feats: (variant, seed) whose core features are attached (default = pool's)."""
        f, dinfo = self.v6_tables(ctx)
        f = f[f["disease_id"].isin(set(ids))]
        v6 = f[f["in_v6_500"]]
        v6lists = {d: set(int(x) for x in g) for d, g in v6.groupby("disease_id")["gene_row"]}
        lists = {d: set(s) for d, s in v6lists.items()}
        if pool == "pool":
            cp = pd.read_parquet(pool_path(variant, seed, ctx))
            cp = cp[cp["disease_id"].isin(set(ids))]
            for d, g in cp.groupby("disease_id")["gene_row"]:
                lists.setdefault(d, set()).update(int(x) for x in g)
        ids = [d for d in ids if d in lists]
        N = max(len(v) for v in lists.values())
        D = len(ids)
        genes = np.full((D, N), -1, dtype=np.int64)
        for i, d in enumerate(ids):
            g = np.array(sorted(lists[d]), dtype=np.int64)
            genes[i, :len(g)] = g
        valid = genes >= 0
        fi = f.set_index(["disease_id", "gene_row"])
        key = pd.MultiIndex.from_arrays([np.repeat(ids, N), genes.reshape(-1)])
        sel = fi.reindex(key)
        if sel.loc[valid.reshape(-1)].isna().any().any():
            raise RuntimeError(f"{ctx}: pool genes without V6 features (run pools71 first)")
        feats = {}
        for s in SOURCES:
            feats[f"v_{s}"] = sel[f"v_{s}"].to_numpy(np.float32)
            r = sel[f"r_{s}"].to_numpy(np.float64)
            feats[f"lr_{s}"] = np.where(r > 0, np.log(N_G / np.maximum(r, 1)), 0.0).astype(np.float32)
            feats[f"a_s_{s}"] = np.repeat(dinfo[f"a_{s}"].reindex(ids).to_numpy(np.float32), N)
        feats["fused"] = sel["fused"].to_numpy(np.float32)
        feats["ret_rank"] = np.log(500.0 / sel["v6_rank"].to_numpy(np.float64)).astype(np.float32)
        feats["is_fill"] = (sel["reason"].to_numpy() == 2).astype(np.float32)
        feats["prior"] = sel["prior"].to_numpy(np.float32)
        rep = lambda col, fn=lambda x: x: np.repeat(fn(dinfo[col].reindex(ids).to_numpy(np.float64)), N).astype(np.float32)
        feats["d_hpo"] = rep("hpo_terms", np.log1p)
        feats["d_pheno_sim"] = rep("pheno_max_sim")
        feats["d_text_sim"] = rep("text_max_sim")
        feats["d_onto_ic"] = rep("onto_max_ic", lambda x: x / 10.0)
        feats["d_onto_inf"] = np.repeat((dinfo["onto_level"].reindex(ids) == "informative")
                                        .to_numpy(np.float32), N)
        names = [n if not n.startswith("a_") else "a_s_" + n[2:] for n in FEATS]
        cols = [feats[n].reshape(D, N) for n in names]
        fnames = list(FEATS)
        extra = {"in_v6_500": np.zeros((D, N), dtype=bool)}
        for i, d in enumerate(ids):
            extra["in_v6_500"][i] = np.isin(genes[i], list(v6lists.get(d, ())))
        cf = core_for_feats or ((variant, seed) if variant else None)
        if featset in ("late", "maskonly") and cf is not None:
            cfe = self.core_feature_arrays(cf, ctx, ids, genes)
            use = CORE_FEATS if featset == "late" else MASK_FEATS
            cols += [cfe[k] for k in use]
            fnames += list(use)
            extra["c_r"] = cfe["c_r"]
            extra["c_valid"] = cfe["c_valid"]
        if featset == "rawconcat":
            raw_g, raw_d = self.raw_vectors()
            gi = np.maximum(genes, 0)
            rows = np.array([self.w.idx.d_ix[d] for d in ids])
            for k in range(raw_g.shape[1]):
                cols.append(raw_g[gi, k].astype(np.float32))
                fnames.append(f"raw_g{k}")
            for k in range(raw_d.shape[1]):
                cols.append(np.repeat(raw_d[rows, k][:, None], N, axis=1).astype(np.float32))
                fnames.append(f"raw_d{k}")
        X = np.stack(cols, axis=2).astype(np.float32)
        X[~valid] = 0.0
        y = np.zeros((D, N), dtype=np.float32)
        nq = np.zeros(D)
        for i, d in enumerate(ids):
            q = self.w.query_rows(d)
            nq[i] = len(q)
            y[i] = np.isin(genes[i], q) & valid[i]
        return PoolBlock(ctx, ids, genes, valid, X, y, nq, fnames, extra)

    def core_feature_arrays(self, cf: tuple, ctx: str, ids: list[str], genes: np.ndarray,
                            T_override: dict | None = None) -> dict:
        """Core features ([D, N] each) of fit ``cf`` = (variant, seed) at the given pool genes."""
        D, N = genes.shape
        co = self.core_outputs(cf[0], cf[1], ctx, ids, T_override=T_override)
        gi = np.maximum(genes, 0)
        take = lambda a: np.take_along_axis(a, gi, axis=1).astype(np.float32)
        cpool = pd.read_parquet(pool_path(cf[0], cf[1], ctx), columns=["disease_id", "gene_row"])
        in500 = np.zeros((D, N), dtype=np.float32)
        cg = {d: set(g) for d, g in cpool.groupby("disease_id")["gene_row"]}
        for i, d in enumerate(ids):
            in500[i] = np.isin(genes[i], list(cg.get(d, set())))
        out = {"c_r": take(co["r"]), "c_valid": take(co["valid"]), "c_rank": take(co["rank"]),
               "c_in500": in500, "c_sS": take(co["sS"]), "c_sF": take(co["sF"]),
               "c_sE": take(co["sE"]), "c_aS": take(co["aS"]), "c_aF": take(co["aF"]),
               "c_aE": take(co["aE"]), "c_wS": take(co["wS"]), "c_wF": take(co["wF"]),
               "c_wE": take(co["wE"]), "c_cov": take(co["cov"])}
        out["_standalone"] = (co["r"], co["valid"])
        return out

    def raw_vectors(self):
        """Label-free PCA summaries of the raw modality vectors (E2 / E4 raw concatenation)."""
        if self._raw is None:
            from sklearn.decomposition import PCA
            a = self.run.pack.a
            def pca(X, m, k, seed=20260923):
                Z = np.zeros((X.shape[0], k), dtype=np.float32)
                if m.sum() > k:
                    Z[m] = PCA(n_components=k, random_state=seed).fit_transform(X[m]).astype(np.float32)
                return Z
            G = np.concatenate([pca(a["g_sequence"], a["g_sequence_mask"], 16),
                                pca(a["g_function"], a["g_function_mask"], 16),
                                a["g_expression"].astype(np.float32),
                                a["g_sequence_mask"][:, None].astype(np.float32),
                                a["g_function_mask"][:, None].astype(np.float32),
                                a["g_expression_mask"].any(axis=1)[:, None].astype(np.float32)], axis=1)
            Dm = np.concatenate([pca(a["d_text"], a["d_text_mask"], 16),
                                 pca(a["d_hpo"], a["d_hpo_mask"], 16),
                                 a["d_tissue"].astype(np.float32),
                                 a["d_hpo_mask"][:, None].astype(np.float32),
                                 a["d_tissue_mask"][:, None].astype(np.float32)], axis=1)
            self._raw = (G, Dm)
        return self._raw


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class LGBPool:
    def __init__(self, seed=42, threads=10, num_leaves=31, min_child_samples=100,
                 learning_rate=0.05, n_estimators=1500, early_stopping=100):
        self.params = dict(num_leaves=num_leaves, min_child_samples=min_child_samples,
                           learning_rate=learning_rate, n_estimators=n_estimators, random_state=seed,
                           n_jobs=threads, verbose=-1, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8)
        self.es = early_stopping

    @staticmethod
    def stack(blocks: list[PoolBlock]):
        Xs, ys, gs = [], [], []
        for b in blocks:
            for i in range(b.D):
                if b.y[i].sum() <= 0:
                    continue
                v = b.valid[i]
                Xs.append(b.X[i, v])
                ys.append(b.y[i, v])
                gs.append(int(v.sum()))
        return np.concatenate(Xs), np.concatenate(ys), np.array(gs)

    def fit(self, train: list[PoolBlock], val: list[PoolBlock]):
        import lightgbm as lgb
        X, y, g = self.stack(train)
        Xv, yv, gv = self.stack(val)
        self.m = lgb.LGBMRanker(objective="lambdarank", **self.params)
        self.m.fit(X, y, group=g, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[10],
                   callbacks=[lgb.early_stopping(self.es, verbose=False)])
        self.best_iteration = int(self.m.best_iteration_ or self.params["n_estimators"])
        self.n_train_lists = int(len(g))
        self.n_train_rows = int(len(y))
        return self

    def score(self, b: PoolBlock) -> np.ndarray:
        out = np.full(b.valid.shape, -np.inf)
        v = b.valid
        out[v] = self.m.predict(b.X[v], num_iteration=self.best_iteration)
        return out

    def importance(self, names) -> dict:
        imp = self.m.booster_.feature_importance(importance_type="gain")
        return {n: float(x) for n, x in zip(names, imp)}


def f7_pool(b: PoolBlock) -> np.ndarray:
    w = f7_weights()
    s = np.zeros(b.valid.shape)
    for src, wt in w.items():
        if wt:
            s += wt * b.feat(f"v_{src}")
    return np.where(b.valid, s, -np.inf)


def fixed_late(b: PoolBlock, beta: float) -> np.ndarray:
    f7 = f7_pool(b)
    r = np.where(b.extra["c_valid"] > 0, np.maximum(b.extra["c_r"], 0.0), 0.0)
    mx = np.maximum(r.max(axis=1, keepdims=True), 1e-12)
    s = (1 - beta) * np.where(b.valid, f7, 0.0) + beta * r / mx
    return np.where(b.valid, s, -np.inf)


def evaluate_pool(b: PoolBlock, scores: np.ndarray) -> pd.DataFrame:
    s = np.where(b.valid, scores, -np.inf)
    m = evaluate_matrix(s, b.y, np.where(b.valid, b.genes, N_G + 1), b.n_query, valid=b.valid,
                        ks=(10, 20, 50))
    per = per_disease_frame(b.ids, m)
    order = np.lexsort((np.where(b.valid, b.genes, N_G + 1), -s), axis=1)
    top500 = np.take_along_axis(b.y, order[:, :500], axis=1).sum(axis=1)
    per["coverage_final500"] = np.where(b.n_query > 0, top500 / np.maximum(b.n_query, 1), np.nan)
    per["pool_size"] = b.valid.sum(axis=1)
    per["context_id"] = b.context_id
    return per


def recall10_fast(b: PoolBlock, s: np.ndarray) -> np.ndarray:
    s = np.where(b.valid, s, -np.inf) - np.where(b.valid, b.genes, 0) * 1e-15
    top = np.argpartition(-s, 9, axis=1)[:, :10]
    h = np.take_along_axis(b.y, top, axis=1).sum(axis=1)
    return h / np.maximum(b.n_query, 1)


def save_run(run_id: str, b: PoolBlock, scores: np.ndarray, meta: dict, extra: dict, device="cpu"):
    with tracked(run_id, meta, device=device) as box:
        per = evaluate_pool(b, scores)
        d = run_dir(run_id)
        per.to_parquet(d / "per_disease.parquet", index=False)
        v = b.valid
        pd.DataFrame({"disease_id": np.repeat(b.ids, b.genes.shape[1])[v.reshape(-1)],
                      "gene_row": b.genes[v].astype(np.int32),
                      "score": scores[v].astype(np.float64),
                      "label": b.y[v].astype(np.int8),
                      "in_v6_500": b.extra["in_v6_500"][v]}).to_parquet(d / "predictions.parquet",
                                                                          index=False)
        m = macro(per)
        m["coverage_final500"] = float(per["coverage_final500"].mean())
        m["mean_pool_size"] = float(per["pool_size"].mean())
        box["extra"] = {"metrics": m, "n_test_diseases": int(b.D),
                        "candidate_hash": candidate_hash(np.repeat(b.ids, b.genes.shape[1])[v.reshape(-1)],
                                                         b.genes[v]), **extra}
    return box["extra"]["metrics"]


# --------------------------------------------------------------------------
# formal second-level runs
# --------------------------------------------------------------------------
def fold_sets(asm: Assembler, j: int):
    w = asm.w
    fa = w.fold(j)
    return {"inner": [c.context_id for c in w.inner_contexts(j)],
            "outer": f"f{j}_outer",
            "test": w.labelled(fa["test_diseases"]), "val": w.labelled(fa["val_diseases"]),
            "dev": w.labelled(w.spec.dev_diseases)}


def train_blocks(asm, j, variant, seed, pool, featset, core_for_feats=None):
    fs = fold_sets(asm, j)
    tr = []
    for c in fs["inner"]:
        ids = asm.w.labelled(asm.w.contexts[c].held_out)
        tr.append(asm.block(c, ids, variant, seed, pool=pool, featset=featset,
                            core_for_feats=core_for_feats))
    val = asm.block(fs["outer"], fs["val"], variant, seed, pool=pool, featset=featset,
                    core_for_feats=core_for_feats)
    test = asm.block(fs["outer"], fs["test"], variant, seed, pool=pool, featset=featset,
                     core_for_feats=core_for_feats)
    return tr, val, test


def run_lgb(asm, method: str, j: int, seed: int, variant, core_seed, pool: str, featset: str,
            core_for_feats=None, threads: int = 10):
    run_id = f"V71Z__{method}__f{j}__s{seed}"
    if is_done(run_id):
        return
    t0 = time.time()
    tr, val, test = train_blocks(asm, j, variant, core_seed, pool, featset, core_for_feats)
    m = LGBPool(seed=seed, threads=threads).fit(tr, [val])
    sc = m.score(test)
    meta = {"stage": "formal_second_level", "method": method, "variant": variant, "fold": j,
            "seed": seed, "task": "Z", "config": {"pool": pool, "featset": featset,
                                                   "core_seed": core_seed,
                                                   "core_for_feats": core_for_feats,
                                                   "lightgbm": m.params},
            "validation": "F_{j+1} pools in the outer context"}
    met = save_run(run_id, test, sc, meta, {"fit_info": {"best_iteration": m.best_iteration,
                                                         "n_train_lists": m.n_train_lists,
                                                         "n_train_rows": m.n_train_rows,
                                                         "feature_names": test.feat_names,
                                                         "gain_importance": m.importance(test.feat_names),
                                                         "seconds_fit": time.time() - t0}})
    (run_dir(run_id) / "model.txt").write_text(m.m.booster_.model_to_string(
        num_iteration=m.best_iteration), encoding="utf-8")
    print(f"  {run_id}: R@10 {met['recall10_end']:.4f} cov500 {met['coverage_final500']:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)


def tune_beta(asm, variant="core", seed=42) -> dict:
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    rows = []
    blocks = []
    for j in range(5):
        fs = fold_sets(asm, j)
        blocks.append(asm.block(fs["outer"], fs["dev"], variant, seed, pool="pool", featset="late"))
    for beta in grid:
        rec = np.concatenate([recall10_fast(b, fixed_late(b, beta)) for b in blocks])
        rows.append({"beta": float(beta), "dev_recall10": float(rec.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(paths71.FUSION / f"fixed_late_beta_dev_{variant}.csv", index=False)
    best = df.sort_values(["dev_recall10", "beta"], ascending=[False, True]).iloc[0]
    out = {"variant": variant, "beta": float(best["beta"]), "dev_recall10": float(best["dev_recall10"]),
           "dev_recall10_beta0_F7": float(df.loc[df["beta"] == 0, "dev_recall10"].iloc[0]),
           "grid": rows, "decided_on": "development families x 5 outer contexts, seed-42 pools"}
    write_manifest(f"fixed_late_beta_{variant}.json", out)
    return out


def run_fixed(asm, j, seed, variant="core"):
    run_id = f"V71Z__lf_fixed__f{j}__s{seed}" if variant == "core" else f"V71Z__lf_fixed_{variant}__f{j}__s{seed}"
    if is_done(run_id):
        return
    beta = load_manifest(f"fixed_late_beta_{variant}.json")["beta"]
    fs = fold_sets(asm, j)
    test = asm.block(fs["outer"], fs["test"], variant, seed, pool="pool", featset="late")
    sc = fixed_late(test, beta)
    met = save_run(run_id, test, sc, {"stage": "formal_second_level", "method": "lf_fixed",
                                      "variant": variant, "fold": j, "seed": seed, "task": "Z",
                                      "config": {"beta": beta}}, {})
    # F7 on the same shared pool (E6)
    rid = f"V71Z__F7_on_pool__f{j}__s{seed}"
    if variant == "core" and not is_done(rid):
        save_run(rid, test, f7_pool(test), {"stage": "formal_second_level", "method": "F7_on_pool",
                                            "variant": variant, "fold": j, "seed": seed, "task": "Z"}, {})
    print(f"  {run_id}: R@10 {met['recall10_end']:.4f}", flush=True)


def run_f7_v6(asm, j):
    run_id = f"V71Z__F7__f{j}"
    if is_done(run_id):
        return
    fs = fold_sets(asm, j)
    test = asm.block(fs["outer"], fs["test"], None, None, pool="v6", featset="channels")
    met = save_run(run_id, test, f7_pool(test), {"stage": "formal_second_level", "method": "F7",
                                                 "fold": j, "seed": "deterministic", "task": "Z"}, {})
    print(f"  {run_id}: R@10 {met['recall10_end']:.4f}", flush=True)


PLANS = {
    # method: (variant for pool, pool kind, featset, core_for_feats rule)
    "lf_lgb": ("core", "pool", "late"),
    "v6_lgb_channels": (None, "v6", "channels"),
    "e2_rawconcat": ("core", "pool", "rawconcat"),
    "e2_twotower_lf": ("two_tower", "pool", "late"),
    "e2_altanchor_lf": ("core_altanchor", "pool", "late"),
    "e3_strict_lf": ("core_strict", "pool", "late"),
    "e4_newres_oldmethod": (None, "v6", "rawconcat"),
    "e4_oldres_newmethod": ("core_oldres", "pool", "late"),
    "e5_maskonly": ("core", "pool", "maskonly"),
    "e6_v6pool_corefeat": ("core", "v6", "late"),
    "e6_channels_on_pool": ("core", "pool", "channels"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V7.1 second-level runs")
    ap.add_argument("--stage", choices=["tune_beta", "lgb", "fixed", "f7"], required=True)
    ap.add_argument("--methods", nargs="*", default=["lf_lgb", "v6_lgb_channels"])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args(argv)
    paths71.ensure()
    if not load_manifest("protocol_v71_registered.json"):
        raise RuntimeError("protocol not registered")
    asm = Assembler()
    if args.stage == "tune_beta":
        print(json.dumps(tune_beta(asm), indent=2)[:2000])
        return 0
    for j in args.folds:
        if args.stage == "f7":
            run_f7_v6(asm, j)
            continue
        for seed in args.seeds:
            if args.stage == "fixed":
                run_fixed(asm, j, seed)
                continue
            for meth in args.methods:
                variant, pool, featset = PLANS[meth]
                core_seed = seed if variant == "core" else 42
                if variant not in (None, "core") and seed != 42:
                    continue
                if meth in ("e6_v6pool_corefeat",):
                    cff = ("core", core_seed)
                    run_lgb(asm, meth, j, seed, "core", core_seed, "v6", "late", core_for_feats=cff,
                            threads=args.threads)
                    continue
                run_lgb(asm, meth, j, seed, variant, core_seed, pool, featset, threads=args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

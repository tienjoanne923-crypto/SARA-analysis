"""V8.1 §9 — the base ranker B, the availability-gated cold-balanced corrector and
candidate A (direct LightGBM with explicit missing values).

Base ranker B: LightGBM lambdarank on the 30 V6 channel features (V6 preset),
trained on OOF pools, early-stopped on F_{j+1}; a fixed shift / scale fitted on
its own training-pool predictions is saved with the model and never refitted on
the scored table.

Corrector (custom objective, no init_score; B, a and the pair index live in the
closure):
    s(d, g) = B(d, g) + a(d, g) r_theta(x_dg),    a = semantic availability,
    L = sum_{d,g,u} w_dgu softplus(s(d,u) - s(d,g)) + lambda/2 sum_i eta_i (a_i r_i)^2,
    w_dgu = 1/|D*| * v_dg / sum_h v_dh * 1/|U_d|,   v = 1 + beta * cold,
pairs only positive x unlabelled (no positive-positive order), eta_i uniform in a
disease and normalised over diseases; the whole objective is multiplied by |D*|
(gradient *and* curvature).  With m = s_i - s_j, p = sigmoid(-m):
    g_i = -w p,  g_j = w p,  h = w p (1 - p);
    dL/dr_i = a_i g_i + lambda eta_i a_i^2 r_i,  d2L/dr_i^2 = a_i^2 h_i + lambda eta_i a_i^2.
Only a = 1 rows enter the LightGBM data set (a = 0 rows keep s = B exactly and
still act as the other side of a pair).  15 leaves, 200 rounds, the round with the
best F_{j+1} Recall@10 (earliest on ties), metric "None", deterministic.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import PROTOCOL_ID
from .feats8 import EXTRA, RowBlock, assert_feature_whitelist

BASE_PARAMS = dict(num_leaves=31, min_child_samples=100, learning_rate=0.05, n_estimators=1500,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
A_GRID = [dict(num_leaves=nl, min_child_samples=mc) for nl in (15, 31, 63) for mc in (50, 200)]
C_GRID = [dict(beta=b, lam=l) for b in (0.0, 1.0, 2.0) for l in (0.03, 0.1)]
N_UNL = 64


# ---------------------------------------------------------------------------
# metrics on pools
# ---------------------------------------------------------------------------
def topk_idx(s: np.ndarray, genes: np.ndarray, k: int = 10, valid: np.ndarray | None = None) -> np.ndarray:
    """[D, k] positions of the top-k scores (ties: ascending gene row; padding last)."""
    out = np.empty((s.shape[0], k), dtype=np.int64)
    for i in range(s.shape[0]):
        si = s[i].astype(np.float64)
        if valid is not None:
            si = np.where(valid[i], si, -np.inf)
        o = np.lexsort((genes[i], -si))
        out[i] = o[:k]
    return out


def recall_at(blk: RowBlock, s: np.ndarray, k: int = 10) -> np.ndarray:
    ti = topk_idx(s, blk.genes, k, getattr(blk, "valid", None))
    h = np.take_along_axis(blk.y, ti, axis=1).sum(1)
    return h / np.maximum(blk.n_query, 1)


def cold_recall_at(blk: RowBlock, s: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    ti = topk_idx(s, blk.genes, k, blk.valid)
    hc = np.take_along_axis(blk.y * blk.cold, ti, axis=1).sum(1)
    rc = np.where(blk.n_query_cold > 0, hc / np.maximum(blk.n_query_cold, 1), np.nan)
    return rc, hc / np.maximum(blk.n_query, 1)


# ---------------------------------------------------------------------------
# base ranker B
# ---------------------------------------------------------------------------
def stack_lambdarank(blocks: list[RowBlock], X_of) -> tuple:
    """Lists of diseases with >= 1 positive among the rows allowed in a training table."""
    Xs, ys, gs = [], [], []
    for b in blocks:
        X = X_of(b)
        ok = b.valid & b.train_ok
        for i in range(b.D):
            m = ok[i]
            if b.y[i, m].sum() <= 0:
                continue
            Xs.append(X[i, m])
            ys.append(b.y[i, m])
            gs.append(int(m.sum()))
    return np.concatenate(Xs), np.concatenate(ys), np.array(gs)


class BaseRanker:
    feature_names = None

    def __init__(self, seed: int = 42, threads: int = 8):
        self.seed, self.threads = seed, threads

    @staticmethod
    def X(b: RowBlock) -> np.ndarray:
        return b.XB

    def fit(self, train: list[RowBlock], val: list[RowBlock]):
        import lightgbm as lgb
        from gpu8v6.scoring import FEATS
        assert_feature_whitelist(list(FEATS))
        X, y, g = stack_lambdarank(train, self.X)
        Xv, yv, gv = stack_lambdarank(val, self.X)
        self.params = dict(objective="lambdarank", random_state=self.seed, n_jobs=self.threads,
                           deterministic=True, force_row_wise=True, **BASE_PARAMS)
        self.m = lgb.LGBMRanker(**self.params)
        self.m.fit(X, y, group=g, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[10],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
        self.best_iteration = int(self.m.best_iteration_ or BASE_PARAMS["n_estimators"])
        self.booster = self.m.booster_
        # fixed shift / scale from the training-pool predictions (all allowed rows of the training pools)
        allX = np.concatenate([self.X(b)[b.valid & b.train_ok] for b in train])
        p = self.booster.predict(allX, num_iteration=self.best_iteration)
        self.mu, self.sd = float(p.mean()), float(max(p.std(), 1e-9))
        self.n_train_lists, self.n_train_rows = int(len(g)), int(len(y))
        return self

    def raw(self, b: RowBlock) -> np.ndarray:
        X = self.X(b).reshape(-1, self.X(b).shape[2])
        return self.booster.predict(X, num_iteration=self.best_iteration).reshape(b.D, b.N)

    def score(self, b: RowBlock) -> np.ndarray:
        return np.where(b.valid, (self.raw(b) - self.mu) / self.sd, -np.inf)

    def save(self, d) -> dict:
        import lightgbm as lgb
        s = self.booster.model_to_string(num_iteration=self.best_iteration)
        (d / "base_model.txt").write_text(s, encoding="utf-8")
        meta = {"mu": self.mu, "sd": self.sd, "best_iteration": self.best_iteration, "params": self.params,
                "n_train_lists": self.n_train_lists, "n_train_rows": self.n_train_rows}
        (d / "base_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        return meta

    @classmethod
    def load(cls, d) -> "BaseRanker":
        import lightgbm as lgb
        obj = cls()
        obj.booster = lgb.Booster(model_str=(d / "base_model.txt").read_text(encoding="utf-8"))
        meta = json.loads((d / "base_meta.json").read_text(encoding="utf-8"))
        obj.mu, obj.sd = meta["mu"], meta["sd"]
        obj.best_iteration = meta["best_iteration"]
        obj.params = meta["params"]
        return obj


def reload_check(model, d, blk: RowBlock) -> float:
    """Max |difference| between the in-memory model and its reloaded string."""
    again = type(model).load(d)
    return float(np.abs(model.score(blk) - again.score(blk)).max())


# ---------------------------------------------------------------------------
# corrector
# ---------------------------------------------------------------------------
def corrector_X(b: RowBlock) -> np.ndarray:
    return np.concatenate([b.XB, b.XE], axis=2)


def sample_unlabelled(B: np.ndarray, y: np.ndarray, key: str, n: int = N_UNL) -> np.ndarray:
    """Half the highest-B unlabelled genes of the pool, half uniform among the rest (hashed seed)."""
    unl = np.flatnonzero(y <= 0)
    if len(unl) <= n:
        return unl
    o = unl[np.lexsort((unl, -B[unl]))]
    top = o[: n // 2]
    rest = o[n // 2:]
    seed = int(hashlib.sha256(f"{PROTOCOL_ID}|unl|{key}".encode()).hexdigest()[:15], 16)
    uni = np.random.default_rng(seed).choice(rest, size=n - len(top), replace=False)
    return np.sort(np.concatenate([top, uni]))


@dataclass
class CorrectorConfig:
    beta: float = 1.0
    lam: float = 0.1
    num_leaves: int = 15
    rounds: int = 200
    learning_rate: float = 0.05
    min_data_in_leaf: int = 20
    min_sum_hessian_in_leaf: float = 1e-3


class Corrector:
    def __init__(self, cfg: CorrectorConfig, seed: int = 42, threads: int = 8):
        self.cfg, self.seed, self.threads = cfg, seed, threads

    # -- training set -------------------------------------------------------
    def training_rows(self, blocks: list[RowBlock], Bs: list[np.ndarray]) -> dict:
        Xs, Bv, av, cv, yv, dv = [], [], [], [], [], []
        pairs_i, pairs_j, pairs_w = [], [], []
        eta = []
        n_dstar, skipped_no_pos, skipped_no_unl = 0, 0, 0
        cold_pos, cold_dis = 0, 0
        samp_hash = hashlib.sha256()
        off = 0
        D_star = []
        for blk, B in zip(blocks, Bs):
            X = corrector_X(blk)
            ok = blk.valid & blk.train_ok
            for i in range(blk.D):
                P = np.flatnonzero((blk.y[i] > 0) & ok[i])
                if len(P) == 0:
                    skipped_no_pos += 1
                    continue
                yu = np.where(ok[i], blk.y[i], 1.0)             # rows outside the table count as known
                U = sample_unlabelled(np.where(ok[i], B[i], -np.inf), yu, f"{blk.ctx}|{blk.ids[i]}|{self.seed}")
                if len(U) == 0:
                    skipped_no_unl += 1
                    continue
                D_star.append((blk, i, P, U, X, B))
        nD = len(D_star)
        for blk, i, P, U, X, B in D_star:
            idx = np.concatenate([P, U])
            samp_hash.update(f"{blk.ctx}|{blk.ids[i]}|".encode() + idx.astype(np.int64).tobytes())
            k = len(idx)
            Xs.append(X[i, idx])
            Bv.append(B[i, idx])
            a = blk.a[i, idx]
            av.append(a)
            c = blk.cold[i, idx]
            cv.append(c)
            yv.append(blk.y[i, idx])
            dv.append(np.full(k, len(dv)))
            v = 1.0 + self.cfg.beta * blk.cold[i, P].astype(np.float64)
            wp = v / v.sum() / len(U)                        # x |D*| already applied (1/|D*| cancels)
            pi = np.repeat(np.arange(len(P)), len(U)) + off
            pj = np.tile(np.arange(len(U)), len(P)) + off + len(P)
            pairs_i.append(pi)
            pairs_j.append(pj)
            pairs_w.append(np.repeat(wp, len(U)))
            na = max(int(a.sum()), 1)
            eta.append(np.where(a, 1.0 / na, 0.0))           # x |D*| applied
            cp = int(blk.cold[i, P].sum())
            cold_pos += cp
            cold_dis += int(cp > 0)
            off += k
        out = {"X": np.concatenate(Xs).astype(np.float32), "B": np.concatenate(Bv).astype(np.float64),
               "a": np.concatenate(av).astype(bool), "cold": np.concatenate(cv).astype(bool),
               "y": np.concatenate(yv), "d": np.concatenate(dv),
               "pi": np.concatenate(pairs_i), "pj": np.concatenate(pairs_j), "pw": np.concatenate(pairs_w),
               "eta": np.concatenate(eta)}
        out["diag"] = {"D_star": nD, "skipped_no_positive_in_pool": skipped_no_pos,
                       "skipped_no_unlabelled": skipped_no_unl, "rows": int(len(out["y"])),
                       "pairs": int(len(out["pi"])), "cold_positives_in_pairs": cold_pos,
                       "diseases_with_cold_positive": cold_dis,
                       "rows_gated_on": int(out["a"].sum()), "sample_hash": samp_hash.hexdigest()}
        return out

    # -- objective ----------------------------------------------------------
    def make_objective(self, tr: dict):
        a_idx = np.flatnonzero(tr["a"])
        n = len(tr["B"])
        pi, pj, pw = tr["pi"], tr["pj"], tr["pw"]
        B = tr["B"]
        eta_a = tr["eta"][a_idx]
        lam = self.cfg.lam
        self.loss_trace = []

        def obj(preds, data):
            r = np.zeros(n)
            r[a_idx] = preds
            s = B + r
            m = s[pi] - s[pj]
            p = 0.5 * (1.0 - np.tanh(0.5 * m))              # sigmoid(-m), overflow-free
            wp = pw * p
            g = np.bincount(pj, weights=wp, minlength=n) - np.bincount(pi, weights=wp, minlength=n)
            hh = wp * (1.0 - p)
            h = np.bincount(pi, weights=hh, minlength=n) + np.bincount(pj, weights=hh, minlength=n)
            grad = g[a_idx] + lam * eta_a * preds
            hess = h[a_idx] + lam * eta_a
            self.loss_trace.append(float((pw * np.logaddexp(0.0, -m)).sum()
                                         + 0.5 * lam * (eta_a * preds ** 2).sum()))
            return grad, hess
        return obj, a_idx

    def fit(self, blocks: list[RowBlock], Bs: list[np.ndarray], val: RowBlock, Bval: np.ndarray):
        import lightgbm as lgb
        assert_feature_whitelist(EXTRA)
        t0 = time.time()
        tr = self.training_rows(blocks, Bs)
        obj, a_idx = self.make_objective(tr)
        params = {"objective": obj, "metric": "None", "num_leaves": self.cfg.num_leaves,
                  "learning_rate": self.cfg.learning_rate, "min_data_in_leaf": self.cfg.min_data_in_leaf,
                  "min_sum_hessian_in_leaf": self.cfg.min_sum_hessian_in_leaf, "lambda_l2": 0.0,
                  "bagging_fraction": 1.0, "bagging_freq": 0, "feature_fraction": 1.0,
                  "deterministic": True, "force_row_wise": True, "num_threads": self.threads,
                  "seed": self.seed, "verbosity": -1, "boost_from_average": False}
        dtr = lgb.Dataset(tr["X"][a_idx], label=np.zeros(len(a_idx)), free_raw_data=False)
        Xv = corrector_X(val).reshape(-1, len(EXTRA) + val.XB.shape[2])
        dval = lgb.Dataset(Xv, label=np.zeros(len(Xv)), reference=dtr, free_raw_data=False)
        av = (val.a & val.valid).reshape(-1)
        curve = []

        def feval(preds, data):
            s = Bval.reshape(-1) + np.where(av, preds, 0.0)
            rec = recall_at(val, s.reshape(val.D, val.N)).mean()
            curve.append(float(rec))
            return "recall10", float(rec), True

        self.booster = lgb.train({k: v for k, v in params.items()}, dtr, num_boost_round=self.cfg.rounds,
                                 valid_sets=[dval], valid_names=["val"], feval=feval)
        curve = np.array(curve)
        self.best_iteration = int(np.argmax(curve) + 1)       # earliest maximum
        self.curve = curve.tolist()
        self.diag = {**tr["diag"], "seconds": time.time() - t0, "best_iteration": self.best_iteration,
                     "val_recall10_best": float(curve.max()), "val_recall10_round1": float(curve[0]),
                     "train_loss_first": self.loss_trace[0] if self.loss_trace else None,
                     "train_loss_r150": self.loss_trace[149] if len(self.loss_trace) >= 150 else None,
                     "train_loss_last": self.loss_trace[-1] if self.loss_trace else None,
                     "params": {k: v for k, v in params.items() if k != "objective"},
                     "objective": "gated pairwise PU softplus (custom), scaled by |D*|"}
        # registered 31-leaf rule (PDF §15.2 "validation shows under-fitting"): the best F_{j+1}
        # round is in the last 5 % of the 200-round budget, i.e. still improving at the end
        self.diag["underfit_rule"] = bool(self.best_iteration >= int(0.95 * self.cfg.rounds))
        return self

    def r(self, b: RowBlock) -> np.ndarray:
        X = corrector_X(b).reshape(-1, b.XB.shape[2] + len(EXTRA))
        return self.booster.predict(X, num_iteration=self.best_iteration, raw_score=True).reshape(b.D, b.N)

    def score(self, b: RowBlock, B: np.ndarray) -> np.ndarray:
        return np.where(b.valid, np.where(b.a, B + self.r(b), B), -np.inf)

    def save(self, d) -> dict:
        (d / "corrector_model.txt").write_text(
            self.booster.model_to_string(num_iteration=self.best_iteration), encoding="utf-8")
        meta = {"config": asdict(self.cfg), "best_iteration": self.best_iteration, "curve": self.curve,
                "diag": self.diag}
        (d / "corrector_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        return meta

    @classmethod
    def load(cls, d) -> "Corrector":
        import lightgbm as lgb
        meta = json.loads((d / "corrector_meta.json").read_text(encoding="utf-8"))
        obj = cls(CorrectorConfig(**meta["config"]))
        obj.booster = lgb.Booster(model_str=(d / "corrector_model.txt").read_text(encoding="utf-8"))
        obj.best_iteration = meta["best_iteration"]
        return obj


# ---------------------------------------------------------------------------
# candidate A
# ---------------------------------------------------------------------------
class CandidateA:
    def __init__(self, num_leaves: int, min_child_samples: int, seed: int = 42, threads: int = 8):
        self.cfg = dict(num_leaves=num_leaves, min_child_samples=min_child_samples)
        self.seed, self.threads = seed, threads

    @staticmethod
    def X(b: RowBlock) -> np.ndarray:
        return np.concatenate([b.XB, b.XE_nan()], axis=2)

    def fit(self, train: list[RowBlock], val: list[RowBlock]):
        import lightgbm as lgb
        X, y, g = stack_lambdarank(train, self.X)
        Xv, yv, gv = stack_lambdarank(val, self.X)
        self.params = dict(objective="lambdarank", random_state=self.seed, n_jobs=self.threads,
                           deterministic=True, force_row_wise=True,
                           **{**BASE_PARAMS, **self.cfg})
        self.m = lgb.LGBMRanker(**self.params)
        self.m.fit(X, y, group=g, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[10],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
        self.best_iteration = int(self.m.best_iteration_ or BASE_PARAMS["n_estimators"])
        self.booster = self.m.booster_
        return self

    def score(self, b: RowBlock) -> np.ndarray:
        X = self.X(b).reshape(-1, b.XB.shape[2] + len(EXTRA))
        s = self.booster.predict(X, num_iteration=self.best_iteration).reshape(b.D, b.N)
        return np.where(b.valid, s, -np.inf)

    def save(self, d, tag: str) -> None:
        (d / f"A_{tag}_model.txt").write_text(self.booster.model_to_string(num_iteration=self.best_iteration),
                                              encoding="utf-8")

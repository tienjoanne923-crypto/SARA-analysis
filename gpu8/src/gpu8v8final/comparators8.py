"""V8.1 §14 — comparators on the same folds, resources and budgets.

* ``F7``            the frozen V6 fixed fusion sum_s w_s v_s (on any list);
* ``channel_lgb``   LightGBM lambdarank on the 30 V6 channels trained on the OOF
                    pools of *its own* list (V6 500 = the strong end-to-end control
                    of comparison 1; on the strict pool it is B itself);
* ``v71_direct``    the V7.1 direct late fusion (30 channels + 14 core features,
                    missing written as 0) retrained on a given list (comparison 2);
* ``sF0``           learning-free same-resource semantic score cos(z0_d, x^GO_g)
                    (no projection; tests whether representation learning matters);
* teacher standalone scores (r of the V7.1 core, s_F, the teacher's own s_S).
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from gpu8v71.core71 import row_norm
from gpu8v71.train71 import D_KEYS, take

from . import paths8
from .feats8 import RowBlock
from .models8 import BASE_PARAMS, stack_lambdarank

N_G = 19861
CORE_FEATS = ["c_r", "c_valid", "c_rank", "c_in500", "c_sS", "c_sF", "c_sE", "c_aS", "c_aF", "c_aE",
              "c_wS", "c_wF", "c_wE", "c_cov"]
_F7 = None


def f7_weights() -> dict:
    global _F7
    if _F7 is None:
        from gpu8v6.common import load_manifest as lm6
        _F7 = lm6("scorer_v6_frozen.json")["F7"]["weights"]
    return _F7


def f7_score(b: RowBlock) -> np.ndarray:
    from gpu8v6.scoring import FEATS
    s = np.zeros((b.D, b.N))
    for src, wt in f7_weights().items():
        if wt:
            s += wt * b.XB[:, :, FEATS.index(f"v_{src}")]
    return s


@torch.no_grad()
def core_features(model, T: dict, ids: list, d_ix: dict, pools: dict[str, np.ndarray],
                  chunk: int = 128) -> dict[str, np.ndarray]:
    """CORE_FEATS at the genes of every pool in ``pools`` (name -> [D, N]) for the V7.1 direct fusion."""
    rows = np.array([d_ix[d] for d in ids], dtype=np.int64)
    g_all = {k: T[k] for k in ("xS", "mS", "xF", "mF", "e", "o") if k in T}
    gr = model.gene_repr(g_all)
    out = {name: np.zeros(P.shape + (len(CORE_FEATS),), dtype=np.float32) for name, P in pools.items()}
    for b0 in range(0, len(rows), chunk):
        idx = torch.as_tensor(rows[b0:b0 + chunk], device=T["xT"].device)
        d = take(T, D_KEYS, idx)
        o = model(d, g_all, gene_reprs=gr)
        r = o["r"].float().cpu().numpy()
        valid = o["valid"].cpu().numpy()
        s = o["s"].float().cpu().numpy()
        av = o["avail"].cpu().numpy()
        wv = o["w"].float().cpu().numpy()
        cov = (d["q"] @ T["o"].T).float().cpu().numpy()
        for i in range(len(idx)):
            v = np.flatnonzero(valid[i])
            order = v[np.lexsort((v, -r[i, v].astype(np.float64)))]
            rank = np.zeros(N_G, dtype=np.float32)
            rank[order] = np.log(N_G / np.arange(1, len(order) + 1))
            in500 = np.zeros(N_G, dtype=np.float32)
            in500[order[:500]] = 1.0
            k = b0 + i
            cols = [r[i], valid[i].astype(np.float32), rank, in500, s[0, i], s[1, i], s[2, i],
                    av[0, i].astype(np.float32), av[1, i].astype(np.float32), av[2, i].astype(np.float32),
                    wv[0, i], wv[1, i], wv[2, i], cov[i]]
            for name, P in pools.items():
                g = P[k]
                out[name][k] = np.stack([c[g] for c in cols], axis=1)
    return out


class DirectLGB:
    """LightGBM lambdarank on an arbitrary feature function (V6 preset, early stop on F_{j+1})."""

    def __init__(self, X_of, seed: int = 42, threads: int = 8, **over):
        self.X_of, self.seed, self.threads = X_of, seed, threads
        self.params = dict(objective="lambdarank", random_state=seed, n_jobs=threads, deterministic=True,
                           force_row_wise=True, **{**BASE_PARAMS, **over})

    def fit(self, train: list[RowBlock], val: list[RowBlock]):
        import lightgbm as lgb
        X, y, g = stack_lambdarank(train, self.X_of)
        Xv, yv, gv = stack_lambdarank(val, self.X_of)
        self.m = lgb.LGBMRanker(**self.params)
        self.m.fit(X, y, group=g, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[10],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
        self.best_iteration = int(self.m.best_iteration_ or self.params["n_estimators"])
        return self

    def score(self, b: RowBlock) -> np.ndarray:
        X = self.X_of(b)
        return self.m.booster_.predict(X.reshape(-1, X.shape[2]),
                                       num_iteration=self.best_iteration).reshape(b.D, b.N)


def attach_core(b: RowBlock, cf: np.ndarray) -> RowBlock:
    b.info["core_feats"] = cf
    return b


def v71_X(b: RowBlock) -> np.ndarray:
    return np.concatenate([b.XB, b.info["core_feats"]], axis=2)


def channel_X(b: RowBlock) -> np.ndarray:
    return b.XB


@torch.no_grad()
def standalone_scores(model, T: dict, ids: list, d_ix: dict, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Full-background standalone scores: 'r' (V7.1 core), 'sF', 'sS' (teacher sequence), 'sF0'."""
    rows = np.array([d_ix[d] for d in ids], dtype=np.int64)
    R = np.zeros((len(rows), N_G), dtype=np.float32)
    V = np.zeros((len(rows), N_G), dtype=bool)
    if kind == "sF0":
        xT, xH = T["xT"], T["xH"]
        nT, nH = row_norm(xT), row_norm(xH)
        z0 = row_norm(nT * T["mT"][:, None] + nH * T["mH"][:, None])
        ok = T["mT"] | T["mH"]
        xF = row_norm(T["xF"])
    for b0 in range(0, len(rows), 256):
        idx = torch.as_tensor(rows[b0:b0 + 256], device=T["xT"].device)
        if kind == "r":
            g_all = {k: T[k] for k in ("xS", "mS", "xF", "mF", "e", "o")}
            o = model(take(T, D_KEYS, idx), g_all)
            s, v = o["r"], o["valid"]
        elif kind in ("sF", "sS"):
            z, ok_d = model.disease_repr(T["xT"][idx], T["mT"][idx], T["xH"][idx], T["mH"][idx])
            if kind == "sF":
                zg = row_norm(model.P_T(T["xF"]))
                mg = T["mF"]
            else:
                zg = row_norm(model.P_S(T["xS"]))
                mg = T["mS"]
            s = z @ zg.T
            v = ok_d[:, None] & mg[None, :]
        else:
            s = z0[idx] @ xF.T
            v = ok[idx][:, None] & T["mF"][None, :]
        R[b0:b0 + len(idx)] = torch.where(v, s, torch.zeros_like(s)).float().cpu().numpy()
        V[b0:b0 + len(idx)] = v.cpu().numpy()
    return R, V


def standalone_top(R: np.ndarray, V: np.ndarray, k: int = 500) -> np.ndarray:
    out = np.full((R.shape[0], k), -1, dtype=np.int64)
    for i in range(R.shape[0]):
        v = np.flatnonzero(V[i])
        o = v[np.lexsort((v, -R[i, v].astype(np.float64)))][:k]
        out[i, :len(o)] = o
    return out

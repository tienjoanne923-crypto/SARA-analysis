"""Trained comparators that share the project's cached features.

B2  same-feature logistic regression
B4  LightGBM ranker, grouped by disease
B5  same-capacity MLP (no adaptation learning) with an optional fixed ridge
B6  SIGN-style graph learning on precomputed propagation scales
B8  plain meta-ridge: identical encoder, support, anchors and task budget, with
    equal support weights and a fixed regularisation — the nearest neighbour of
    the proposed method, and the comparison that actually tests the mechanism.

All of them read the same 512-dim node views, 4 channels and 12 quality fields,
so a difference in performance cannot be attributed to a difference in inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Gpu8Config
from ..episodes import EpisodeBuilder
from .simple import stable_tie_noise


# --------------------------------------------------------------------------
# shared feature assembly
# --------------------------------------------------------------------------
def pair_matrix(builder: EpisodeBuilder, disease_id: str, gene_rows: np.ndarray
                ) -> np.ndarray:
    """[disease views | gene views | channels | quality] for a set of pairs."""
    d_row = builder.b.idx.d_ix[disease_id]
    ch, mask, q = builder.pair_features(disease_id, d_row, gene_rows)
    dv, gv = builder.node_features(d_row, gene_rows)
    return np.concatenate([dv, gv, np.abs(dv - gv), dv * gv, ch, mask, q],
                          axis=1).astype(np.float32)


def compact_pair_matrix(builder: EpisodeBuilder, disease_id: str,
                        gene_rows: np.ndarray) -> np.ndarray:
    """Reduced-width variant for the tree model: interactions plus channels."""
    d_row = builder.b.idx.d_ix[disease_id]
    ch, mask, q = builder.pair_features(disease_id, d_row, gene_rows)
    dv, gv = builder.node_features(d_row, gene_rows)
    dot = (dv * gv).sum(axis=1, keepdims=True)
    cos = dot / np.maximum(np.linalg.norm(dv, axis=1, keepdims=True)
                           * np.linalg.norm(gv, axis=1, keepdims=True), 1e-8)
    l2 = np.linalg.norm(dv - gv, axis=1, keepdims=True)
    dim = builder.b.disease_views.shape[1] // 4
    per_view = []
    for v in range(4):
        sl = slice(v * dim, (v + 1) * dim)
        pv = (dv[:, sl] * gv[:, sl]).sum(axis=1, keepdims=True)
        per_view.append(pv)
    return np.concatenate([ch, mask, q, dot, cos, l2] + per_view, axis=1).astype(np.float32)


def build_training_rows(cfg: Gpu8Config, builder: EpisodeBuilder, diseases: list[str],
                        rng: np.random.Generator, n_neg_per_pos: int = 5,
                        max_pos_per_disease: int = 16, compact: bool = False
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labelled rows from the *training* diseases of this context only."""
    X, y, groups = [], [], []
    fn = compact_pair_matrix if compact else pair_matrix
    g_ix = builder.b.idx.g_ix
    for did in diseases:
        pos = builder.positives.get(did, [])
        if not pos:
            continue
        d_row = builder.b.idx.d_ix.get(did)
        if d_row is None:
            continue
        prow = np.array([g_ix[g] for g in pos if g in g_ix], dtype=np.int64)
        if len(prow) == 0:
            continue
        if len(prow) > max_pos_per_disease:
            prow = rng.choice(prow, size=max_pos_per_disease, replace=False)
        known = set(np.array([g_ix[g] for g in pos if g in g_ix]).tolist())
        cands, _ = builder.candidate_rows(did, d_row, n=cfg.candidates.n_candidates)
        pool = np.array([c for c in cands if c not in known], dtype=np.int64)
        if len(pool) == 0:
            continue
        n_neg = min(len(pool), n_neg_per_pos * len(prow))
        nrow = rng.choice(pool, size=n_neg, replace=False)
        rows = np.concatenate([prow, nrow])
        X.append(fn(builder, did, rows))
        y.append(np.concatenate([np.ones(len(prow)), np.zeros(len(nrow))]))
        groups.append(len(rows))
    if not X:
        return np.zeros((0, 1), dtype=np.float32), np.zeros(0), np.zeros(0, dtype=int)
    return (np.concatenate(X, axis=0), np.concatenate(y),
            np.array(groups, dtype=int))


# --------------------------------------------------------------------------
# B2 logistic regression
# --------------------------------------------------------------------------
class B2LogisticRegression:
    name = "b2_lr"
    uses_support = False

    def __init__(self, cfg: Gpu8Config, C: float = 1.0):
        self.cfg, self.C = cfg, C
        self.model = None
        self.mu = None
        self.sd = None

    def fit(self, builder: EpisodeBuilder, diseases: list[str], seed: int = 42):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(seed)
        X, y, _ = build_training_rows(self.cfg, builder, diseases, rng)
        if len(X) == 0:
            return self
        self.mu, self.sd = X.mean(0), np.maximum(X.std(0), 1e-6)
        Xs = (X - self.mu) / self.sd
        self.model = LogisticRegression(max_iter=2000, C=self.C, class_weight="balanced")
        self.model.fit(Xs, y)
        return self

    def score(self, builder: EpisodeBuilder, disease_id: str,
              support_genes: list[str], candidate_rows: np.ndarray) -> np.ndarray:
        if self.model is None:
            return stable_tie_noise(np.zeros(len(candidate_rows)), candidate_rows)
        X = (pair_matrix(builder, disease_id, candidate_rows) - self.mu) / self.sd
        return stable_tie_noise(self.model.decision_function(X), candidate_rows)


# --------------------------------------------------------------------------
# B4 LightGBM ranker
# --------------------------------------------------------------------------
class B4LightGBMRanker:
    name = "b4_lightgbm"
    uses_support = False

    def __init__(self, cfg: Gpu8Config, num_leaves: int = 63, min_child_samples: int = 20,
                 learning_rate: float = 0.05, n_estimators: int = 400):
        self.cfg = cfg
        self.params = dict(num_leaves=num_leaves, min_child_samples=min_child_samples,
                           learning_rate=learning_rate, n_estimators=n_estimators)
        self.model = None

    def fit(self, builder: EpisodeBuilder, diseases: list[str], seed: int = 42,
            val_builder: EpisodeBuilder | None = None, val_diseases: list[str] | None = None):
        import lightgbm as lgb
        rng = np.random.default_rng(seed)
        X, y, groups = build_training_rows(self.cfg, builder, diseases, rng, compact=True)
        if len(X) == 0:
            return self
        self.model = lgb.LGBMRanker(objective="lambdarank", random_state=seed,
                                    n_jobs=self.cfg.hardware.cpu_threads, verbose=-1,
                                    **self.params)
        eval_set, eval_group = None, None
        if val_builder is not None and val_diseases:
            Xv, yv, gv = build_training_rows(self.cfg, val_builder, val_diseases,
                                             np.random.default_rng(seed + 1), compact=True)
            if len(Xv):
                eval_set, eval_group = [(Xv, yv)], [gv]
        callbacks = [__import__("lightgbm").early_stopping(50, verbose=False)] if eval_set else None
        self.model.fit(X, y, group=groups, eval_set=eval_set, eval_group=eval_group,
                       eval_at=[10], callbacks=callbacks)
        return self

    def score(self, builder: EpisodeBuilder, disease_id: str,
              support_genes: list[str], candidate_rows: np.ndarray) -> np.ndarray:
        if self.model is None:
            return stable_tie_noise(np.zeros(len(candidate_rows)), candidate_rows)
        X = compact_pair_matrix(builder, disease_id, candidate_rows)
        return stable_tie_noise(self.model.predict(X), candidate_rows)


# --------------------------------------------------------------------------
# B6 SIGN-style: precomputed propagation scales + a trained decoder
# --------------------------------------------------------------------------
def sign_features(builder: EpisodeBuilder, disease_id: str, gene_rows: np.ndarray
                  ) -> np.ndarray:
    """Concatenate propagation scales 0, 1 and 2 on the gene side.

    The scales come from the same cached P^k Z arrays, so this is an adaptation
    of the SIGN idea to the task, not a re-run of the original model.
    """
    b = builder.b
    d_row = b.idx.d_ix[disease_id]
    dim = b.gene_views.shape[1] // 4
    g0 = b.gene_views[gene_rows, 0 * dim:1 * dim]
    g1 = b.gene_views[gene_rows, 1 * dim:2 * dim]
    g2 = b.gene_views[gene_rows, 2 * dim:3 * dim]
    dv = b.disease_views[d_row]
    d0 = np.repeat(dv[None, 0 * dim:1 * dim], len(gene_rows), axis=0)
    inter = np.concatenate([g0 * d0, g1 * d0, g2 * d0], axis=1)
    return np.concatenate([g0, g1, g2, inter], axis=1).astype(np.float32)


class B6SignStyle:
    name = "b6_sign"
    uses_support = False

    def __init__(self, cfg: Gpu8Config, hidden: int = 128, lr: float = 1e-3,
                 epochs: int = 30, weight_decay: float = 1e-4):
        self.cfg = cfg
        self.hidden, self.lr, self.epochs, self.wd = hidden, lr, epochs, weight_decay
        self.model = None
        self.mu = self.sd = None

    def fit(self, builder: EpisodeBuilder, diseases: list[str], seed: int = 42, device="cpu"):
        import torch
        import torch.nn as nn
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        X, y = build_training_rows_generic(self.cfg, builder, diseases, rng, sign_features)
        if len(X) == 0:
            return self
        self.mu, self.sd = X.mean(0), np.maximum(X.std(0), 1e-6)
        Xs = torch.as_tensor((X - self.mu) / self.sd, dtype=torch.float32, device=device)
        ys = torch.as_tensor(y, dtype=torch.float32, device=device)
        self.model = nn.Sequential(nn.Linear(Xs.shape[1], self.hidden), nn.GELU(),
                                   nn.Linear(self.hidden, self.hidden), nn.GELU(),
                                   nn.Linear(self.hidden, 1)).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = nn.BCEWithLogitsLoss()
        n = Xs.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=device)
            for s in range(0, n, 4096):
                idx = perm[s:s + 4096]
                opt.zero_grad(set_to_none=True)
                out = self.model(Xs[idx]).squeeze(-1)
                loss = lossf(out, ys[idx])
                loss.backward()
                opt.step()
        self.device = device
        return self

    def score(self, builder: EpisodeBuilder, disease_id: str,
              support_genes: list[str], candidate_rows: np.ndarray) -> np.ndarray:
        import torch
        if self.model is None:
            return stable_tie_noise(np.zeros(len(candidate_rows)), candidate_rows)
        X = (sign_features(builder, disease_id, candidate_rows) - self.mu) / self.sd
        with torch.no_grad():
            t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            out = self.model(t).squeeze(-1).cpu().numpy()
        return stable_tie_noise(out, candidate_rows)


def build_training_rows_generic(cfg: Gpu8Config, builder: EpisodeBuilder,
                                diseases: list[str], rng: np.random.Generator,
                                featfn, n_neg_per_pos: int = 5,
                                max_pos_per_disease: int = 16):
    X, y = [], []
    g_ix = builder.b.idx.g_ix
    for did in diseases:
        pos = builder.positives.get(did, [])
        d_row = builder.b.idx.d_ix.get(did)
        if not pos or d_row is None:
            continue
        prow = np.array([g_ix[g] for g in pos if g in g_ix], dtype=np.int64)
        if len(prow) == 0:
            continue
        if len(prow) > max_pos_per_disease:
            prow = rng.choice(prow, size=max_pos_per_disease, replace=False)
        known = set(np.array([g_ix[g] for g in pos if g in g_ix]).tolist())
        cands, _ = builder.candidate_rows(did, d_row, n=cfg.candidates.n_candidates)
        pool = np.array([c for c in cands if c not in known], dtype=np.int64)
        if len(pool) == 0:
            continue
        nrow = rng.choice(pool, size=min(len(pool), n_neg_per_pos * len(prow)), replace=False)
        rows = np.concatenate([prow, nrow])
        X.append(featfn(builder, did, rows))
        y.append(np.concatenate([np.ones(len(prow)), np.zeros(len(nrow))]))
    if not X:
        return np.zeros((0, 1), dtype=np.float32), np.zeros(0)
    return np.concatenate(X, axis=0), np.concatenate(y)

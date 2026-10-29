"""V6 section 7 — the scoring stage inside a fixed candidate list.

Everything here ranks a *frozen* 500-gene list; nothing re-generates candidates.

* ``Block`` holds one context's candidate lists as padded [D, N] arrays plus the
  per-pair channel features and, separately, the labels the evaluator and the
  training loss read.
* ``fusion_scores`` is the fixed linear fusion F(d, g) = sum_s w_s v_s over the
  max-normalised source scores, with non-negative weights on the simplex.
* ``tune_simplex`` runs the 0.25-step grid (35 points for the four V5 channels,
  210 for the seven V6 sources) and a small 0.05-step refinement, on the
  development families only; every configuration evaluated is logged.
* ``LogisticScorer`` and ``LightGBMScorer`` are trained on out-of-fold (OOF)
  candidate pools of the outer training diseases, with early stopping on an
  inner validation set, never on a test fold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import paths6
from .metrics6 import evaluate_matrix, macro, per_disease_frame
from .retrieval import simplex_grid
from .sources import SOURCES

CAND_DIR = paths6.CACHES / "candidates"
N_G = 19861
V5_CHANNELS = ["pheno", "hier", "rwr_pheno", "compat"]
HAND_WEIGHTS = {"pheno": 0.4, "hier": 0.2, "rwr_pheno": 0.3, "compat": 0.1}
FEATS = ([f"v_{s}" for s in SOURCES] + [f"lr_{s}" for s in SOURCES]
         + [f"a_{s}" for s in SOURCES]
         + ["fused", "ret_rank", "is_fill", "prior", "d_hpo", "d_pheno_sim", "d_text_sim",
            "d_onto_ic", "d_onto_inf"])


@dataclass
class Block:
    context_id: str
    ids: list[str]
    genes: np.ndarray          # [D, N]
    X: np.ndarray              # [D, N, F]
    y: np.ndarray              # [D, N]   evaluator / loss only
    n_query: np.ndarray        # [D]      evaluator only
    dinfo: pd.DataFrame
    feat_names: list[str] = field(default_factory=lambda: list(FEATS))

    @property
    def D(self) -> int:
        return len(self.ids)

    @property
    def N(self) -> int:
        return self.genes.shape[1]

    def feat(self, name: str) -> np.ndarray:
        return self.X[:, :, self.feat_names.index(name)]

    def subset(self, ids: list[str]) -> "Block":
        pos = {d: i for i, d in enumerate(self.ids)}
        idx = np.array([pos[d] for d in ids if d in pos], dtype=np.int64)
        return Block(self.context_id, [self.ids[i] for i in idx], self.genes[idx], self.X[idx],
                     self.y[idx], self.n_query[idx],
                     self.dinfo.iloc[idx].reset_index(drop=True), self.feat_names)


def load_block(context_id: str, ids: list[str] | None = None, budget: int = 500) -> Block:
    cand = pd.read_parquet(CAND_DIR / f"{context_id}_candidates.parquet")
    dis = pd.read_parquet(CAND_DIR / f"{context_id}_diseases.parquet")
    lab = pd.read_parquet(CAND_DIR / f"{context_id}_labels.parquet")
    cand = cand[cand["rank"] <= budget]
    if ids is not None:
        keep = set(ids)
        cand = cand[cand["disease_id"].isin(keep)]
        dis = dis[dis["disease_id"].isin(keep)]
    present = set(cand["disease_id"].unique())
    order = [d for d in (ids if ids is not None else dis["disease_id"]) if d in present]
    pos = {d: i for i, d in enumerate(order)}
    # positional sort instead of .loc on a non-unique index (which is very slow)
    dis = dis[dis["disease_id"].isin(present)].copy()
    dis["_o"] = dis["disease_id"].map(pos)
    dis = dis.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    cand = cand.copy()
    cand["_o"] = cand["disease_id"].map(pos)
    cand = cand.sort_values(["_o", "rank"], kind="stable").drop(columns="_o").reset_index(drop=True)
    D = len(order)
    N = budget
    counts = cand.groupby("disease_id", sort=False).size().reindex(order).to_numpy()
    if not (counts == N).all():
        raise ValueError(f"{context_id}: ragged candidate lists")
    genes = cand["gene_row"].to_numpy().reshape(D, N).astype(np.int64)
    feats = {}
    for s in SOURCES:
        feats[f"v_{s}"] = cand[f"v_{s}"].to_numpy(np.float32)
        r = cand[f"r_{s}"].to_numpy(np.float64)
        feats[f"lr_{s}"] = np.where(r > 0, np.log(N_G / np.maximum(r, 1)), 0.0).astype(np.float32)
        feats[f"a_{s}"] = np.repeat(dis[f"a_{s}"].to_numpy(np.float32), N)
    feats["fused"] = cand["fused"].to_numpy(np.float32)
    feats["ret_rank"] = np.log(N / cand["rank"].to_numpy(np.float64)).astype(np.float32)
    feats["is_fill"] = (cand["reason"].to_numpy() == 2).astype(np.float32)
    feats["prior"] = cand["prior"].to_numpy(np.float32)
    rep = lambda col, f=lambda x: x: np.repeat(f(dis[col].to_numpy(np.float64)), N).astype(np.float32)
    feats["d_hpo"] = rep("hpo_terms", np.log1p)
    feats["d_pheno_sim"] = rep("pheno_max_sim")
    feats["d_text_sim"] = rep("text_max_sim")
    feats["d_onto_ic"] = rep("onto_max_ic", lambda x: x / 10.0)
    feats["d_onto_inf"] = np.repeat((dis["onto_level"] == "informative").to_numpy(np.float32), N)
    X = np.stack([feats[f].reshape(D, N) for f in FEATS], axis=2).astype(np.float32)
    lab = lab[lab["disease_id"].isin(set(order))]
    pos = {d: i for i, d in enumerate(order)}
    y = np.zeros((D, N), dtype=np.float32)
    qsets: dict[int, set] = {}
    for d, g in zip(lab["disease_id"], lab["gene_row"]):
        qsets.setdefault(pos[d], set()).add(int(g))
    for i in range(D):
        q = qsets.get(i, set())
        if q:
            y[i] = np.isin(genes[i], list(q)).astype(np.float32)
    n_query = dis["n_query"].to_numpy(np.float64)
    return Block(context_id, order, genes, X, y, n_query, dis)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(block: Block, scores: np.ndarray, ks=(10, 20, 50)) -> pd.DataFrame:
    m = evaluate_matrix(scores, block.y, block.genes, block.n_query, ks=ks)
    per = per_disease_frame(block.ids, m)
    per["context_id"] = block.context_id
    return per


def recall_at(block: Block, scores: np.ndarray, k: int = 10) -> np.ndarray:
    """Fast per-disease end-to-end recall@k with the gene-row tie-break."""
    s = scores.astype(np.float64) - block.genes * 1e-15
    top = np.argpartition(-s, k - 1, axis=1)[:, :k]
    h = np.take_along_axis(block.y, top, axis=1).sum(axis=1)
    return h / np.maximum(block.n_query, 1.0)


# --------------------------------------------------------------------------
# fixed fusion
# --------------------------------------------------------------------------
def fusion_scores(block: Block, weights: dict[str, float]) -> np.ndarray:
    s = np.zeros((block.D, block.N), dtype=np.float64)
    for src, wt in weights.items():
        if wt:
            s += wt * block.feat(f"v_{src}")
    return s


def tune_simplex(blocks: list[Block], channels: list[str], step: float = 0.25,
                 refine_step: float = 0.05, refine_radius: int = 2) -> tuple[dict, pd.DataFrame]:
    """Grid + local refinement of non-negative simplex weights on dev blocks."""
    V = [np.stack([b.feat(f"v_{c}") for c in channels], axis=2).astype(np.float64) for b in blocks]

    def score(w):
        vals = []
        for b, v in zip(blocks, V):
            vals.append(recall_at(b, v @ np.asarray(w)))
        return float(np.concatenate(vals).mean())

    log = []
    for wt in simplex_grid(len(channels), step):
        if sum(wt) <= 0:
            continue
        log.append({"stage": "grid", **{c: x for c, x in zip(channels, wt)}, "recall10": score(wt)})
    grid = pd.DataFrame(log).sort_values("recall10", ascending=False)
    best = grid.iloc[0][channels].to_numpy(dtype=float)
    frontier, seen = [best], {tuple(np.round(best, 4))}
    for _ in range(refine_radius):
        nxt = []
        for b0 in frontier:
            for i in range(len(b0)):
                for j in range(len(b0)):
                    if i == j or b0[i] < refine_step - 1e-9:
                        continue
                    c = b0.copy()
                    c[i] -= refine_step
                    c[j] += refine_step
                    key = tuple(np.round(c, 4))
                    if key in seen:
                        continue
                    seen.add(key)
                    nxt.append(c)
                    log.append({"stage": "refine", **{ch: x for ch, x in zip(channels, c)},
                                "recall10": score(c)})
        frontier = nxt
    table = pd.DataFrame(log).sort_values("recall10", ascending=False).reset_index(drop=True)
    top = table.iloc[0]
    return {c: float(top[c]) for c in channels}, table


# --------------------------------------------------------------------------
# learned scorers on OOF pools
# --------------------------------------------------------------------------
def design(block: Block, gene_feats: np.ndarray | None, rows: np.ndarray | None = None
           ) -> np.ndarray:
    X = block.X.reshape(-1, block.X.shape[2])
    if gene_feats is not None:
        G = gene_feats[block.genes.reshape(-1)]
        X = np.concatenate([X, G], axis=1)
    return X if rows is None else X[rows]


class LogisticScorer:
    name = "lr"

    def __init__(self, C: float = 1.0, neg_per_disease: int = 60, seed: int = 42):
        self.C, self.neg, self.seed = C, neg_per_disease, seed

    def fit(self, train: list[Block], gene_feats=None):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(self.seed)
        Xs, ys = [], []
        for b in train:
            for i in range(b.D):
                pos = np.flatnonzero(b.y[i] > 0)
                if len(pos) == 0:
                    continue
                neg = np.flatnonzero(b.y[i] == 0)
                neg = rng.choice(neg, size=min(self.neg, len(neg)), replace=False)
                idx = np.concatenate([pos, neg])
                Xi = b.X[i, idx]
                if gene_feats is not None:
                    Xi = np.concatenate([Xi, gene_feats[b.genes[i, idx]]], axis=1)
                Xs.append(Xi)
                ys.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
        X, y = np.concatenate(Xs), np.concatenate(ys)
        self.mu, self.sd = X.mean(0), np.maximum(X.std(0), 1e-6)
        self.m = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        self.m.fit((X - self.mu) / self.sd, y)
        self.gene_feats = gene_feats
        return self

    def score(self, b: Block) -> np.ndarray:
        X = design(b, self.gene_feats)
        return self.m.decision_function((X - self.mu) / self.sd).reshape(b.D, b.N)


class LightGBMScorer:
    name = "lightgbm"

    def __init__(self, num_leaves: int = 31, min_child_samples: int = 100,
                 learning_rate: float = 0.05, n_estimators: int = 1500, seed: int = 42,
                 threads: int = 4, early_stopping: int = 100):
        self.params = dict(num_leaves=num_leaves, min_child_samples=min_child_samples,
                           learning_rate=learning_rate, n_estimators=n_estimators,
                           random_state=seed, n_jobs=threads, verbose=-1,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
        self.es = early_stopping

    @staticmethod
    def _stack(blocks: list[Block], gene_feats):
        Xs, ys, gs = [], [], []
        for b in blocks:
            has = b.y.sum(axis=1) > 0
            if not has.any():
                continue
            sub = b.subset([d for d, h in zip(b.ids, has) if h])
            Xs.append(design(sub, gene_feats))
            ys.append(sub.y.reshape(-1))
            gs.extend([sub.N] * sub.D)
        return np.concatenate(Xs), np.concatenate(ys), np.array(gs)

    def fit(self, train: list[Block], val: list[Block], gene_feats=None):
        import lightgbm as lgb
        X, y, g = self._stack(train, gene_feats)
        Xv, yv, gv = self._stack(val, gene_feats)
        self.m = lgb.LGBMRanker(objective="lambdarank", **self.params)
        self.m.fit(X, y, group=g, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[10],
                   callbacks=[lgb.early_stopping(self.es, verbose=False)])
        self.best_iteration = int(self.m.best_iteration_ or self.params["n_estimators"])
        self.gene_feats = gene_feats
        self.n_train_lists = int(len(g))
        return self

    def score(self, b: Block) -> np.ndarray:
        return self.m.predict(design(b, self.gene_feats),
                              num_iteration=self.best_iteration).reshape(b.D, b.N)


def fold_blocks(fold: int, w, budget_test: int = 500):
    """(train OOF blocks, val block F_{j+1}, dev block, test block) for one fold."""
    fa = w.fold(fold)
    train = [load_block(c.context_id) for c in w.inner_contexts(fold)]
    outer = f"f{fold}_outer"
    val = load_block(outer, ids=w.labelled(fa["val_diseases"]))
    dev = load_block(outer, ids=w.labelled(w.spec.dev_diseases))
    test = load_block(outer, ids=w.labelled(fa["test_diseases"]), budget=budget_test)
    return train, val, dev, test

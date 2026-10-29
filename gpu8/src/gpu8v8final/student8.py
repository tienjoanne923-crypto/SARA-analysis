"""V8.1 §6 — the sequence student (imputation of the function-semantic space only).

    z_hat^F_g = norm(P_S(x^S_g)),  P_S: 480 -> 256 (GELU) -> 128  (156,032 parameters),
    L_S = mean_g [1 - z_hat^F_g . stopgrad(z^F_g)]

over genes with a legal sequence *and* legal GO that are in no GO-mask set.  The
teacher's z^F is computed once without gradient, so training the student never
changes the teacher.  Linear control: 480 -> 128 (61,568 parameters).  AdamW
(lr 1e-3), two weight-decay levels; epochs chosen on a fixed hashed 10 % of the
training genes (gene-level validation, disjoint from both mask sets).

Inference priority (§6.3):  observed GO -> z^F;  no GO, sequence, student accepted
-> z_hat^F;  otherwise unavailable.  ``semantic_source`` records which.
"""
from __future__ import annotations

import copy
import hashlib
import time

import numpy as np
import torch
import torch.nn as nn

from . import PROTOCOL_ID
from gpu8v71.core71 import row_norm

N_PARAMS_MLP = 156_032
N_PARAMS_LIN = 61_568
WD_LEVELS = (1e-4, 1e-2)
KINDS = ("mlp", "linear")
SRC_UNAVAILABLE, SRC_GO, SRC_SEQ = 0, 1, 2


class Student(nn.Module):
    def __init__(self, kind: str = "mlp", in_dim: int = 480):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, 128))
        elif kind == "linear":
            self.net = nn.Linear(in_dim, 128)
        else:
            raise KeyError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return row_norm(self.net(x))

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def hashed_split(genes: np.ndarray, share: float = 0.10, tag: str = "student_val") -> tuple[np.ndarray, np.ndarray]:
    h = np.array([int(hashlib.sha256(f"{PROTOCOL_ID}|{tag}|{int(g)}".encode()).hexdigest()[:12], 16)
                  for g in genes])
    val = (h % 1000) < int(share * 1000)
    return genes[~val], genes[val]


def train_student(X: torch.Tensor, Z: torch.Tensor, train_g: np.ndarray, val_g: np.ndarray,
                  kind: str, wd: float, seed: int, device: str = "cuda", max_epochs: int = 300,
                  patience: int = 20, batch: int = 256, lr: float = 1e-3) -> dict:
    """Z: teacher targets [G, 128] (already detached, unit norm)."""
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Student(kind, X.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Zd = Z.detach()
    tg = torch.as_tensor(train_g, device=device)
    vg = torch.as_tensor(val_g, device=device)
    best, best_state, best_ep, bad, hist = np.inf, None, 0, 0, []
    for ep in range(1, max_epochs + 1):
        m.train()
        perm = rng.permutation(len(train_g))
        tot, nb = 0.0, 0
        for b0 in range(0, len(perm), batch):
            idx = tg[torch.as_tensor(perm[b0:b0 + batch], device=device)]
            zh = m(X[idx])
            loss = (1.0 - (zh * Zd[idx]).sum(1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        m.eval()
        with torch.no_grad():
            vl = float((1.0 - (m(X[vg]) * Zd[vg]).sum(1)).mean())
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": vl})
        if vl < best - 1e-6:
            best, best_ep, bad = vl, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in m.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return {"model": m, "state": best_state, "best_epoch": best_ep, "best_val_loss": best,
            "history": hist, "n_params": m.n_trainable(), "seconds": time.time() - t0}


@torch.no_grad()
def gene_level_check(m: Student, X: torch.Tensor, Z: torch.Tensor, val_g: np.ndarray,
                     pool_g: np.ndarray) -> dict:
    """Held-out gene retrieval (true teacher vector among the pool) and collapse check."""
    zh = m(X[torch.as_tensor(val_g, device=X.device)])
    P = Z[torch.as_tensor(pool_g, device=X.device)]
    S = zh @ P.T
    pos = {int(g): i for i, g in enumerate(pool_g)}
    true_ix = torch.as_tensor([pos[int(g)] for g in val_g], device=X.device)
    ts = S.gather(1, true_ix[:, None])
    rank = (S > ts).sum(1) + 1
    rank = rank.cpu().numpy()
    k = min(2000, len(val_g))
    sub = zh[:k]
    C = sub @ sub.T
    off = C[~torch.eye(k, dtype=torch.bool, device=X.device)]
    sv = torch.linalg.svdvals(sub - sub.mean(0))
    p = (sv / sv.sum()).clamp_min(1e-12)
    eff_rank = float(torch.exp(-(p * p.log()).sum()))
    cos_true = float((zh * Z[torch.as_tensor(val_g, device=X.device)]).sum(1).mean())
    return {"n_val_genes": int(len(val_g)), "pool": int(len(pool_g)),
            "top1": float((rank == 1).mean()), "top10": float((rank <= 10).mean()),
            "median_rank": float(np.median(rank)), "mean_cos_true": cos_true,
            "mean_pairwise_cos": float(off.mean()), "effective_rank": eff_rank,
            "collapsed": bool(float(off.mean()) > 0.95 or eff_rank < 2.0)}


def length_strata(lengths: np.ndarray, n: int = 10) -> np.ndarray:
    import pandas as pd
    r = pd.Series(lengths).rank(method="first").to_numpy()
    return np.minimum((n * (r - 1) / max(len(r), 1)).astype(int), n - 1)


def shuffled_inputs(X: np.ndarray, genes: np.ndarray, lengths: np.ndarray, seed_tag: str) -> np.ndarray:
    """Sequence embeddings permuted among ``genes`` inside length-decile strata."""
    from .common8 import rng_for
    rng = rng_for("student_pair_shuffle", seed_tag)
    Xs = X.copy()
    st = length_strata(lengths[genes])
    for s in np.unique(st):
        g = genes[st == s]
        Xs[g] = X[rng.permutation(g)]
    return Xs


def nearest_neighbour_transfer(X: np.ndarray, zF: np.ndarray, targets: np.ndarray,
                               donors: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Each target takes the teacher vector of its most similar donor sequence (cosine)."""
    Xt = torch.as_tensor(X, device=device)
    Xt = row_norm(Xt)
    D = Xt[torch.as_tensor(donors, device=device)]
    out = np.zeros((len(targets), zF.shape[1]), dtype=np.float32)
    for b0 in range(0, len(targets), 1024):
        t = Xt[torch.as_tensor(targets[b0:b0 + 1024], device=device)]
        nn_ = (t @ D.T).argmax(1).cpu().numpy()
        out[b0:b0 + len(nn_)] = zF[donors[nn_]]
    return out

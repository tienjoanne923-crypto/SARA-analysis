"""V6 sections 9.1-9.2 and 13.2 — ``propagate_support``.

Random walk with restart on a column-stochastic operator P:

    p(t+1) = alpha * u + (1 - alpha) * P p(t),        p* = K_alpha u,
    K_alpha = alpha [I - (1 - alpha) P]^{-1}          (analysis only, never formed)

Stopping is by residual, not by a fixed number of steps: because P is column
stochastic, ||p(t) - p*||_1 <= ||p(t+1) - p(t)||_1 / alpha, so stopping when the
step is at most alpha * eps guarantees an L1 error of at most eps.  A column
that reaches ``max_iter`` without meeting the tolerance is reported as a
failure; it is never silently treated as converged.

Operators:

* ``PPIOperator``   the V5 physical STRING network (score >= 700), isolated genes
                    given a self-loop, columns normalised;
* ``BridgeOperator`` P_F = B D_F^{-1} B^T D_G^{-1} applied with two sparse
                    products (the gene x gene matrix is never built); genes with
                    no function term keep their mass (self-loop fallback);
* ``MixOperator``   (1 - nu) P_PPI + nu P_F, again column stochastic.

RWR is linear in the seed, so single-gene columns K_alpha e_g can be cached and
combined; ``SeedColumnCache`` does that for one (operator, alpha) pair and is
invalidated by construction when either changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse

EPS = 1e-12


def _to_torch_csr(M: sparse.csr_matrix, device, dtype=torch.float64):
    M = M.tocsr()
    return torch.sparse_csr_tensor(torch.as_tensor(M.indptr, dtype=torch.int64),
                                   torch.as_tensor(M.indices, dtype=torch.int64),
                                   torch.as_tensor(M.data, dtype=dtype),
                                   size=M.shape, device=device)


def column_stochastic(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    A = adj.tocsr().astype(np.float64)
    colsum = np.asarray(A.sum(axis=0)).ravel()
    iso = np.where(colsum <= 0)[0]
    if len(iso):
        A = A + sparse.csr_matrix((np.ones(len(iso)), (iso, iso)), shape=A.shape)
        colsum = np.asarray(A.sum(axis=0)).ravel()
    return (A @ sparse.diags(1.0 / np.maximum(colsum, EPS))).tocsr()


class PPIOperator:
    name = "ppi"

    def __init__(self, P: sparse.csr_matrix, device="cuda"):
        self.P = P.tocsr().astype(np.float64)
        self.n = P.shape[0]
        self.device = device
        self.Pt = _to_torch_csr(self.P, device)

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        return self.Pt @ X

    def column_sums(self) -> np.ndarray:
        return np.asarray(self.P.sum(axis=0)).ravel()


class BridgeOperator:
    name = "bridge"

    def __init__(self, B: sparse.csr_matrix, device="cuda"):
        B = B.tocsr().astype(np.float64)
        B.data[:] = 1.0
        self.n = B.shape[0]
        dG = np.asarray(B.sum(axis=1)).ravel()
        dF = np.asarray(B.sum(axis=0)).ravel()
        self.has = dG > 0
        self.device = device
        self.B = _to_torch_csr(B, device)
        self.Bt = _to_torch_csr(B.T.tocsr(), device)
        self.invG = torch.as_tensor(np.where(dG > 0, 1.0 / np.maximum(dG, EPS), 0.0),
                                    dtype=torch.float64, device=device)[:, None]
        self.invF = torch.as_tensor(np.where(dF > 0, 1.0 / np.maximum(dF, EPS), 0.0),
                                    dtype=torch.float64, device=device)[:, None]
        self.self_loop = torch.as_tensor((~self.has).astype(np.float64),
                                         dtype=torch.float64, device=device)[:, None]
        self._Bs = B

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        term = self.Bt @ (X * self.invG)
        return self.B @ (term * self.invF) + X * self.self_loop

    def column_sums(self) -> np.ndarray:
        eye = torch.eye(min(self.n, 64), dtype=torch.float64, device=self.device)
        X = torch.zeros((self.n, eye.shape[0]), dtype=torch.float64, device=self.device)
        X[:eye.shape[0], :] = eye
        return self.apply(X).sum(0).cpu().numpy()


class MixOperator:
    def __init__(self, a, b, nu: float):
        self.a, self.b, self.nu = a, b, float(nu)
        self.n = a.n
        self.name = f"mix{nu:g}"

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        if self.nu == 0.0:
            return self.a.apply(X)
        return (1.0 - self.nu) * self.a.apply(X) + self.nu * self.b.apply(X)


@dataclass
class RWRResult:
    scores: np.ndarray          # n x B
    iterations: int
    residual: np.ndarray        # final ||p(t+1) - p(t)||_1 per column
    converged: np.ndarray       # bool per column
    seconds: float
    alpha: float
    eps: float


def rwr(op, seeds: np.ndarray, alpha: float, eps: float = 1e-6, max_iter: int = 2000,
        device="cuda", batch: int = 512) -> RWRResult:
    """Batched RWR. ``seeds`` is n x B; each non-zero column is normalised to sum 1."""
    t0 = time.time()
    S = np.asarray(seeds, dtype=np.float64)
    if S.ndim == 1:
        S = S[:, None]
    tot = S.sum(axis=0)
    out = np.zeros_like(S)
    res_all = np.zeros(S.shape[1])
    conv_all = np.ones(S.shape[1], dtype=bool)
    it_max = 0
    for start in range(0, S.shape[1], batch):
        sl = slice(start, min(start + batch, S.shape[1]))
        cols = S[:, sl]
        t = tot[sl]
        live = t > 0
        if not live.any():
            continue
        U = torch.as_tensor(cols[:, live] / t[live], dtype=torch.float64, device=device)
        X = U.clone()
        res = None
        for it in range(1, max_iter + 1):
            Xn = alpha * U + (1.0 - alpha) * op.apply(X)
            res = (Xn - X).abs().sum(dim=0)
            X = Xn
            if bool((res <= alpha * eps).all()):
                break
        it_max = max(it_max, it)
        r = res.cpu().numpy()
        idx = np.arange(sl.start, sl.stop)[live]
        out[:, idx] = X.cpu().numpy()
        res_all[idx] = r
        conv_all[idx] = r <= alpha * eps
    return RWRResult(scores=out, iterations=it_max, residual=res_all, converged=conv_all,
                     seconds=time.time() - t0, alpha=alpha, eps=eps)


class SeedColumnCache:
    """K_alpha e_g for a set of genes, for one (operator, alpha, eps)."""

    def __init__(self, op, alpha: float, eps: float = 1e-6, device="cuda"):
        self.op, self.alpha, self.eps, self.device = op, alpha, eps, device
        self.cols: dict[int, int] = {}
        self.K = np.zeros((op.n, 0), dtype=np.float32)
        self.log: list[dict] = []

    def ensure(self, genes) -> None:
        need = sorted({int(g) for g in genes} - set(self.cols))
        if not need:
            return
        S = np.zeros((self.op.n, len(need)))
        S[need, np.arange(len(need))] = 1.0
        res = rwr(self.op, S, self.alpha, self.eps, device=self.device)
        base = self.K.shape[1]
        self.K = np.concatenate([self.K, res.scores.astype(np.float32)], axis=1)
        for i, g in enumerate(need):
            self.cols[g] = base + i
        self.log.append({"n_new_columns": len(need), "iterations": res.iterations,
                         "max_residual": float(res.residual.max()),
                         "n_not_converged": int((~res.converged).sum()),
                         "seconds": res.seconds})

    def propagate(self, seed_weights: dict[int, float]) -> np.ndarray:
        """p = sum_g w_g K e_g with weights normalised to one."""
        if not seed_weights:
            return np.zeros(self.op.n, dtype=np.float64)
        self.ensure(seed_weights.keys())
        g = np.array(list(seed_weights.keys()), dtype=np.int64)
        w = np.array([seed_weights[x] for x in g], dtype=np.float64)
        w = w / max(w.sum(), EPS)
        cols = np.array([self.cols[int(x)] for x in g], dtype=np.int64)
        return self.K[:, cols].astype(np.float64) @ w

    def failures(self) -> int:
        return int(sum(x["n_not_converged"] for x in self.log))


def degree_preserving_rewire(adj: sparse.csr_matrix, n_swaps_per_edge: float = 10.0,
                             seed: int = 20260923) -> tuple[sparse.csr_matrix, dict]:
    """Double-edge swaps on an undirected weighted graph; weights travel with edges.

    Every node keeps its exact degree.  Returns the rewired symmetric adjacency
    and a report of the swaps and the change in connected components.
    """
    from scipy.sparse.csgraph import connected_components
    A = sparse.triu(adj.tocsr(), k=1).tocoo()
    u, v, w = A.row.copy(), A.col.copy(), A.data.copy()
    m = len(u)
    rng = np.random.default_rng(seed)
    edges = set(zip(np.minimum(u, v).tolist(), np.maximum(u, v).tolist()))
    target = int(n_swaps_per_edge * m)
    done, tries = 0, 0
    while done < target and tries < target * 20:
        tries += 1
        i, j = rng.integers(m), rng.integers(m)
        if i == j:
            continue
        a, b, c, d = u[i], v[i], u[j], v[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        e1, e2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if e1 in edges or e2 in edges:
            continue
        edges.discard((min(a, b), max(a, b)))
        edges.discard((min(c, d), max(c, d)))
        edges.add(e1)
        edges.add(e2)
        u[i], v[i] = e1
        u[j], v[j] = e2
        done += 1
    n = adj.shape[0]
    R = sparse.coo_matrix((np.concatenate([w, w]), (np.concatenate([u, v]),
                                                    np.concatenate([v, u]))), shape=(n, n)).tocsr()
    deg0 = np.diff((adj > 0).astype(np.int8).tocsr().indptr)
    deg1 = np.diff((R > 0).astype(np.int8).tocsr().indptr)
    orig = set(zip(np.minimum(A.row, A.col).tolist(), np.maximum(A.row, A.col).tolist()))
    kept = len(orig & edges)
    nc0 = connected_components(adj > 0, directed=False)[0]
    nc1 = connected_components(R > 0, directed=False)[0]
    iso0 = int((deg0 == 0).sum())
    return R, {"edges": int(m), "swaps_done": int(done), "swap_attempts": int(tries),
               "degree_sequence_identical": bool(np.array_equal(deg0, deg1)),
               "fraction_edges_unchanged": float(kept / max(m, 1)),
               "connected_components_before": int(nc0), "connected_components_after": int(nc1),
               "components_excluding_isolated_before": int(nc0 - iso0),
               "components_excluding_isolated_after": int(nc1 - iso0),
               "weights": "each edge keeps its STRING weight when it is swapped"}

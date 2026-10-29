"""V6 section 19 guardrails, run on small synthetic inputs.

Mirrors the checks the plan lists for its algebra script: single-support
invariance, equal-group invariance, duplicate-source invariance, hidden support
excluded from grouping, RWR linearity, the L1 bound, residual-stopping error
control and the Recall decomposition.  Also: column stochasticity of the
functional bridge and of the mixture, degree preservation under rewiring, and
that the V5 CPU fusion reduces to pure base fusion with zero support.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpu8v6.metrics6 import check_decomposition, evaluate_matrix, per_disease_frame  # noqa: E402
from gpu8v6.propagate import (BridgeOperator, MixOperator, PPIOperator,  # noqa: E402
                              column_stochastic, degree_preserving_rewire, rwr)
from gpu8v6.support_seed import build_support_seed, group_weights  # noqa: E402

DEV = "cpu"


def _graph(n=40, p=0.12, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < p).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    A[5, :] = 0
    A[:, 5] = 0          # one isolated node exercises the self-loop fallback
    return sparse.csr_matrix(A)


def _ev(groups):
    """relation_of / relation_pmids for support genes -> list of source ids."""
    rel_of = {("D", g): f"R_{g}" for g in groups}
    pm = {f"R_{g}": set(s) for g, s in groups.items()}
    return rel_of, pm


# ---------------------------------------------------------------- seeds
def test_single_support_invariance():
    rel_of, pm = _ev({"g1": ["p1", "p2"]})
    r = build_support_seed("D", ["g1"], rel_of, pm)
    assert r.k == 1 and np.allclose(r.u, [1.0]) and np.allclose(r.v, [1.0])
    assert r.l1() == 0.0


def test_equal_group_invariance():
    rel_of, pm = _ev({f"g{i}": ["same"] for i in range(5)})
    r = build_support_seed("D", [f"g{i}" for i in range(5)], rel_of, pm)
    assert np.allclose(r.v, r.u)
    rel_of, pm = _ev({f"g{i}": [f"p{i}"] for i in range(5)})
    r = build_support_seed("D", [f"g{i}" for i in range(5)], rel_of, pm)
    assert np.allclose(r.v, r.u)


def test_plan_example():
    rel_of, pm = _ev({"g1": ["a"], "g2": ["a"], "g3": ["a"], "g4": ["b"], "g5": ["c"]})
    r = build_support_seed("D", ["g1", "g2", "g3", "g4", "g5"], rel_of, pm)
    assert np.allclose(r.v, [1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 3])
    assert np.isclose(r.l1(), 3 * (1 / 5 - 1 / 9) + 2 * (1 / 3 - 1 / 5))


def test_duplicate_source_invariance():
    rel_of, pm = _ev({"g1": ["a"], "g2": ["a", "b"], "g3": ["c"]})
    r1 = build_support_seed("D", ["g1", "g2", "g3"], rel_of, pm)
    pm2 = {k: set(list(v) + list(v)) for k, v in pm.items()}      # duplicated records
    r2 = build_support_seed("D", ["g1", "g2", "g3", "g2"], rel_of, pm2)
    assert np.allclose(r1.v, r2.v)


def test_hidden_support_not_used():
    groups = {"g1": ["a"], "g2": ["a"], "g3": ["b"], "hidden": ["a", "b", "z"]}
    rel_of, pm = _ev(groups)
    r = build_support_seed("D", ["g1", "g2", "g3"], rel_of, pm)
    rel_of2, pm2 = _ev({k: v for k, v in groups.items() if k != "hidden"})
    r2 = build_support_seed("D", ["g1", "g2", "g3"], rel_of2, pm2)
    assert np.allclose(r.v, r2.v)


def test_untraceable_falls_back():
    rel_of, pm = _ev({"g1": ["a"], "g2": []})
    r = build_support_seed("D", ["g1", "g2"], rel_of, pm)
    assert not r.all_traceable and r.v is None
    assert np.allclose(r.mixed(1.0), r.u)


# ---------------------------------------------------------------- propagation
def test_rwr_linearity_and_error():
    A = _graph()
    P = column_stochastic(A)
    op = PPIOperator(P, device=DEV)
    n = P.shape[0]
    rng = np.random.default_rng(1)
    u1, u2 = rng.random(n), rng.random(n)
    u1 /= u1.sum()
    u2 /= u2.sum()
    alpha, eps = 0.2, 1e-9
    p1 = rwr(op, u1[:, None], alpha, eps, device=DEV).scores[:, 0]
    p2 = rwr(op, u2[:, None], alpha, eps, device=DEV).scores[:, 0]
    p12 = rwr(op, (0.3 * u1 + 0.7 * u2)[:, None], alpha, eps, device=DEV).scores[:, 0]
    assert np.abs(p12 - (0.3 * p1 + 0.7 * p2)).sum() < 1e-7
    # direct solve of the fixed point
    K = alpha * np.linalg.inv(np.eye(n) - (1 - alpha) * P.toarray())
    exact = K @ u1
    assert np.abs(p1 - exact).sum() <= 1e-8
    assert np.allclose(K.sum(axis=0), 1.0) and (K >= -1e-12).all()


def test_l1_bound():
    A = _graph(seed=3)
    op = PPIOperator(column_stochastic(A), device=DEV)
    rel_of, pm = _ev({"g0": ["a"], "g1": ["a"], "g2": ["a"], "g3": ["b"], "g4": ["c"]})
    r = build_support_seed("D", [f"g{i}" for i in range(5)], rel_of, pm)
    rows = [1, 2, 3, 4, 6]
    n = op.n
    for eta in (0.25, 0.5, 1.0):
        pe = np.zeros(n)
        pg = np.zeros(n)
        pe[rows] = r.u
        pg[rows] = r.mixed(eta)
        a = rwr(op, pe[:, None], 0.2, 1e-10, device=DEV).scores[:, 0]
        b = rwr(op, pg[:, None], 0.2, 1e-10, device=DEV).scores[:, 0]
        assert np.abs(b - a).sum() <= eta * r.l1() + 1e-8


def test_bridge_and_mix_column_stochastic():
    rng = np.random.default_rng(2)
    B = sparse.csr_matrix((rng.random((30, 12)) < 0.2).astype(float))
    B[7, :] = 0          # a gene with no function term keeps its mass
    B.eliminate_zeros()
    br = BridgeOperator(B, device=DEV)
    import torch
    X = torch.eye(30, dtype=torch.float64)
    Y = br.apply(X).numpy()
    assert np.allclose(Y.sum(axis=0), 1.0)
    assert np.isclose(Y[7, 7], 1.0)
    P = column_stochastic(_graph(n=30))
    mix = MixOperator(PPIOperator(P, device=DEV), br, 0.3)
    Z = mix.apply(X).numpy()
    assert np.allclose(Z.sum(axis=0), 1.0)


def test_rewire_preserves_degree():
    A = _graph(n=80, p=0.1, seed=5)
    R, info = degree_preserving_rewire(A, n_swaps_per_edge=5, seed=1)
    assert info["degree_sequence_identical"]
    assert info["fraction_edges_unchanged"] < 0.9


# ---------------------------------------------------------------- metrics
def test_recall_decomposition():
    rng = np.random.default_rng(4)
    D, C = 50, 30
    scores = rng.random((D, C))
    labels = (rng.random((D, C)) < 0.05).astype(float)
    genes = np.tile(np.arange(C), (D, 1))
    extra = rng.integers(0, 4, size=D)          # positives outside the list
    nq = labels.sum(axis=1) + extra
    nq[nq == 0] = 1
    per = per_disease_frame([f"d{i}" for i in range(D)],
                            evaluate_matrix(scores, labels, genes, nq, ks=(10,)))
    chk = check_decomposition(per, k=10)
    assert chk["identity_holds"]
    assert abs(chk["macro_recall"] - chk["macro_mean_of_products"]) < 1e-12


# ---------------------------------------------------------------- V5 comparator
def test_b3_zero_support_is_pure_fusion():
    from gpu8.baselines.simple import b3_fusion_fixed_adapt
    from gpu8.config import Gpu8Config

    class _B:
        pass

    n = 20
    rng = np.random.default_rng(7)
    r = {k: rng.random(n) for k in ("phenotype", "hierarchy", "rwr", "compat", "fused")}
    b = _B()
    b.idx = _B()
    b.idx.d_ix = {"D": 0}
    b.idx.g_ix = {}
    builder = _B()
    builder.b = b
    builder.retriever_scores = lambda did, row: r
    cand = np.arange(n)
    s = b3_fusion_fixed_adapt(Gpu8Config(), builder, "D", [], cand)
    chans = np.log1p(np.stack([r["phenotype"], r["hierarchy"], r["rwr"], r["compat"]], 1))
    base = chans @ np.array([0.4, 0.2, 0.3, 0.1])
    assert np.max(np.abs(s - base)) < 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

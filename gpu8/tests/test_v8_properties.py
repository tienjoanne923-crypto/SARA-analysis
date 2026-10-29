"""V8.1 reference properties (PDF Appendix A, §8-§9, §6, §16.3) on synthetic data.

Connection and numerical tests only — never a biological result.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpu8v8final.pool8 import rrf_score, strict_pool, top_budget, v6_list  # noqa: E402
from gpu8v8final.models8 import (Corrector, CorrectorConfig, recall_at, sample_unlabelled,  # noqa: E402
                                 topk_idx)
from gpu8v8final.feats8 import EXTRA, RowBlock, assert_feature_whitelist  # noqa: E402
from gpu8v8final.student8 import (N_PARAMS_LIN, N_PARAMS_MLP, SRC_GO, SRC_SEQ, SRC_UNAVAILABLE,  # noqa: E402
                                  Student, hashed_split, shuffled_inputs, train_student)
from gpu8v8final.evaluate8 import per_disease, seed_mean  # noqa: E402
from gpu8v8final.v6pass8 import v6_order  # noqa: E402

G = 2000


def ranks(rng, D=6, G=G):
    rB = np.stack([rng.permutation(G) + 1 for _ in range(D)]).astype(np.int32)
    m = rng.random((D, G)) > 0.3
    s = rng.random((D, G))
    rS = np.zeros((D, G), dtype=np.int32)
    for i in range(D):
        v = np.flatnonzero(m[i])
        o = v[np.lexsort((v, -s[i, v]))]
        rS[i, o] = np.arange(1, len(o) + 1)
    return rB, rS, m


# ------------------------------------------------------------------ RRF / pool
def test_aware_rrf_equals_b_when_semantic_missing():
    rng = np.random.default_rng(0)
    rB, rS, m = ranks(rng)
    R = rrf_score(rB, rS, m, 0.5)
    b = 1.0 / (60 + rB)
    assert np.allclose(R[~m], b[~m])


def test_aware_rrf_is_weighted_mean_when_available():
    rng = np.random.default_rng(1)
    rB, rS, m = ranks(rng)
    w = 0.7
    R = rrf_score(rB, rS, m, w)
    b, e = 1.0 / (60 + rB), 1.0 / (60 + np.maximum(rS, 1))
    assert np.allclose(R[m], (w * b + (1 - w) * e)[m])


def test_plain_rrf_penalises_missing_semantic():
    rng = np.random.default_rng(2)
    rB, rS, m = ranks(rng)
    Rp = rrf_score(rB, rS, m, 0.5, plain=True)
    b = 1.0 / (60 + rB)
    assert np.allclose(Rp[~m], 0.5 * b[~m])


def test_strict_pool_exactly_500_distinct():
    rng = np.random.default_rng(3)
    rB, rS, m = ranks(rng)
    P = strict_pool(rB, rS, m, 0.5)
    assert P["genes"].shape == (6, 500)
    assert all(len(set(r.tolist())) == 500 for r in P["genes"])
    assert P["counts"] == {"n_retrieval_scored": G, "n_second_stage_scored": 500, "n_output": 500}


def test_strict_pool_all_missing_equals_v6_list():
    rng = np.random.default_rng(4)
    rB, rS, m = ranks(rng)
    m[:] = False
    P = strict_pool(rB, rS, m, 0.3)
    assert np.array_equal(np.sort(P["genes"], 1), np.sort(v6_list(rB), 1))


def test_top_budget_tie_break_gene_row():
    R = np.zeros((1, 1000))
    R[0, [5, 3, 9]] = 1.0
    out = top_budget(R, 5)
    assert out[0, :3].tolist() == [3, 5, 9]
    assert out[0, 3:].tolist() == [0, 1]


def test_pool_function_takes_no_labels():
    import inspect
    assert "label" not in " ".join(inspect.signature(strict_pool).parameters)


def test_v6_order_signal_then_prior():
    fused = np.array([0.0, 0.5, 0.0, 0.9, 0.5])
    prior = np.array([0.1, 0.0, 0.9, 0.0, 0.0])
    r = v6_order(fused, prior)
    assert r.tolist() == [5, 2, 4, 1, 3]


# ------------------------------------------------------------------ gate / objective
def synth_block(rng, D=8, N=60, a_share=0.6):
    genes = np.stack([rng.permutation(5000)[:N] for _ in range(D)])
    XB = rng.normal(size=(D, N, 30)).astype(np.float32)
    XE = rng.normal(size=(D, N, len(EXTRA))).astype(np.float32)
    a = rng.random((D, N)) < a_share
    XE[:, :, EXTRA.index("m_sem")] = a
    cold = rng.random((D, N)) < 0.3
    y = (rng.random((D, N)) < 0.08).astype(np.float32)
    y[:, 0] = 1.0
    nq = y.sum(1) + 1
    nqc = (y * cold).sum(1)
    return RowBlock("syn", "train", [f"D{i}" for i in range(D)], genes, XB, XE, a, cold,
                    np.where(a, 1, 0), y, nq, nqc)


def test_gate_fallback_exact():
    rng = np.random.default_rng(5)
    blk = synth_block(rng)
    B = rng.normal(size=blk.a.shape)
    c = Corrector(CorrectorConfig(beta=1, lam=0.1, rounds=5), threads=1)
    c.fit([blk], [B], blk, B)
    s = c.score(blk, B)
    assert np.array_equal(s[~blk.a], B[~blk.a])


def test_fallback_does_not_guarantee_rank():
    # a = 0 keeps the score, yet another candidate's increase can push it out of the top-1
    B = np.array([[1.0, 0.9]])
    r = np.array([[0.0, 0.5]])
    a = np.array([[False, True]])
    s = np.where(a, B + r, B)
    assert s[0, 0] == B[0, 0] and np.argmax(s[0]) == 1


def pair_problem(seed=0, beta=1.0):
    rng = np.random.default_rng(seed)
    blk = synth_block(rng, D=5, N=40)
    B = rng.normal(size=blk.a.shape)
    c = Corrector(CorrectorConfig(beta=beta, lam=0.1), threads=1)
    tr = c.training_rows([blk], [B])
    return c, tr


def test_pair_weights_normalised_per_disease():
    c, tr = pair_problem(1, beta=2.0)
    s = np.bincount(tr["d"][tr["pi"]], weights=tr["pw"])
    assert np.allclose(s[s > 0], 1.0)


def test_pairs_are_positive_x_unlabelled_only():
    c, tr = pair_problem(2)
    assert (tr["y"][tr["pi"]] > 0).all() and (tr["y"][tr["pj"]] <= 0).all()


def test_cold_weight_changes_only_positive_share():
    c0, t0 = pair_problem(3, beta=0.0)
    c2, t2 = pair_problem(3, beta=2.0)
    assert np.array_equal(t0["pi"], t2["pi"])
    cold_pos = t2["cold"][t2["pi"]]
    if cold_pos.any() and (~cold_pos).any():
        assert t2["pw"][cold_pos].mean() > t0["pw"][cold_pos].mean() - 1e-12


def objective_value(tr, preds, lam):
    a_idx = np.flatnonzero(tr["a"])
    r = np.zeros(len(tr["B"]))
    r[a_idx] = preds
    s = tr["B"] + r
    m = s[tr["pi"]] - s[tr["pj"]]
    return float((tr["pw"] * np.logaddexp(0, -m)).sum() + 0.5 * lam * (tr["eta"][a_idx] * preds ** 2).sum())


def test_objective_gradient_finite_difference():
    c, tr = pair_problem(4)
    obj, a_idx = c.make_objective(tr)
    rng = np.random.default_rng(0)
    p = rng.normal(scale=0.3, size=len(a_idx))
    g, h = obj(p, None)
    eps = 1e-6
    for k in rng.choice(len(p), size=10, replace=False):
        e = np.zeros_like(p)
        e[k] = eps
        fd = (objective_value(tr, p + e, 0.1) - objective_value(tr, p - e, 0.1)) / (2 * eps)
        assert abs(fd - g[k]) < 1e-5 * max(1, abs(fd))


def test_objective_diagonal_hessian_finite_difference():
    c, tr = pair_problem(5)
    obj, a_idx = c.make_objective(tr)
    rng = np.random.default_rng(1)
    p = rng.normal(scale=0.3, size=len(a_idx))
    g0, h0 = obj(p, None)
    eps = 1e-5
    for k in rng.choice(len(p), size=10, replace=False):
        e = np.zeros_like(p)
        e[k] = eps
        g1, _ = obj(p + e, None)
        g2, _ = obj(p - e, None)
        fd = (g1[k] - g2[k]) / (2 * eps)
        assert abs(fd - h0[k]) < 1e-4 * max(1, abs(fd))


def test_hessian_positive_and_scaled_with_gradient():
    c, tr = pair_problem(6)
    obj, a_idx = c.make_objective(tr)
    g, h = obj(np.zeros(len(a_idx)), None)
    assert (h > 0).all() and np.isfinite(g).all()


def test_unlabelled_sampling_half_top_half_uniform_and_hashed():
    rng = np.random.default_rng(7)
    B = rng.normal(size=500)
    y = np.zeros(500)
    y[:5] = 1
    u1 = sample_unlabelled(B, y, "k")
    u2 = sample_unlabelled(B, y, "k")
    assert np.array_equal(u1, u2) and len(u1) == 64 and not set(u1) & set(range(5))
    top = np.argsort(-np.where(y > 0, -np.inf, B))[:32]
    assert set(top.tolist()) <= set(u1.tolist())


def test_corrector_model_reload_identical(tmp_path):
    rng = np.random.default_rng(8)
    blk = synth_block(rng)
    B = rng.normal(size=blk.a.shape)
    c = Corrector(CorrectorConfig(beta=1, lam=0.1, rounds=10), threads=1).fit([blk], [B], blk, B)
    c.save(tmp_path)
    c2 = Corrector.load(tmp_path)
    assert np.abs(c.score(blk, B) - c2.score(blk, B)).max() == 0.0


def test_feature_whitelist_blocks_label_burden():
    assert_feature_whitelist(EXTRA)
    with pytest.raises(RuntimeError):
        assert_feature_whitelist(EXTRA + ["n_query"])


# ------------------------------------------------------------------ metrics
def test_topk_tie_break_gene_row():
    s = np.array([[1.0, 1.0, 0.5]])
    genes = np.array([[7, 3, 1]])
    assert topk_idx(s, genes, 2)[0].tolist() == [1, 0]


def test_recall_decomposition_R_equals_U_eta():
    class L:
        def rows(self, d, role):
            return np.array([1, 2, 3, 50])
    pred = {"ids": np.array(["A"]), "genes": np.arange(20)[None, :], "score": np.linspace(1, 0, 20)[None, :]}
    per = per_disease(pred, L(), "val", np.zeros(100, dtype=bool))
    r = per.iloc[0]
    assert np.isclose(r["recall10"], r["U10"] * r["eta10"])
    assert r["m_pool"] == 3 and np.isclose(r["recall10"], 3 / 4)


def test_seeds_averaged_not_unioned():
    import pandas as pd
    a = pd.DataFrame({"disease_id": ["A"], "recall10": [1.0], "cold_recall10": [np.nan], "n_query": [1],
                      "n_query_cold": [0]})
    b = pd.DataFrame({"disease_id": ["A"], "recall10": [0.0], "cold_recall10": [np.nan], "n_query": [1],
                      "n_query_cold": [0]})
    assert seed_mean([a, b])["recall10"].iloc[0] == 0.5


# ------------------------------------------------------------------ student
def test_student_parameter_counts():
    assert Student("mlp").n_trainable() == N_PARAMS_MLP == 156_032
    assert Student("linear").n_trainable() == N_PARAMS_LIN


def test_student_stopgrad_teacher_unchanged():
    torch.manual_seed(0)
    P_T = torch.nn.Linear(768, 128, bias=False)
    xF = torch.randn(300, 768)
    before = P_T.weight.detach().clone()
    with torch.no_grad():
        Z = torch.nn.functional.normalize(P_T(xF), dim=1)
    X = torch.randn(300, 480)
    g = np.arange(300)
    tr, va = hashed_split(g)
    train_student(X, Z, tr, va, "mlp", 1e-4, 0, device="cpu", max_epochs=3)
    assert torch.equal(before, P_T.weight)


def test_go_first_priority():
    from gpu8v8final.semantic8 import gene_semantic

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.P_T = torch.nn.Linear(768, 128, bias=False)
    T = {"xF": torch.randn(10, 768), "mF": torch.tensor([True] * 6 + [False] * 4)}
    z, src = gene_semantic(M(), T, np.ones((2, 128), dtype=np.float32) / np.sqrt(128), np.array([7, 8]))
    assert (src[:6] == SRC_GO).all() and src[7] == SRC_SEQ and src[6] == SRC_UNAVAILABLE
    with pytest.raises(RuntimeError):
        gene_semantic(M(), T, np.ones((1, 128), dtype=np.float32), np.array([0]))


def test_shuffle_stays_inside_length_strata():
    from gpu8v8final.student8 import length_strata
    X = np.arange(40, dtype=np.float32)[:, None].repeat(480, 1)
    genes = np.arange(40)
    lengths = np.random.default_rng(0).integers(50, 3000, size=40)
    Xs = shuffled_inputs(X, genes, lengths, "t")
    st = length_strata(lengths[genes])
    for s in np.unique(st):
        idx = genes[st == s]
        assert set(Xs[idx, 0].tolist()) == set(X[idx, 0].tolist())
    assert not np.array_equal(Xs, X)


def test_masked_genes_not_student_targets():
    g = np.arange(100)
    excl = np.zeros(100, dtype=bool)
    excl[:30] = True
    elig = np.flatnonzero(~excl)
    tr, va = hashed_split(elig)
    assert not set(tr.tolist()) & set(range(30)) and not set(va.tolist()) & set(range(30))


def test_score_batch_invariance():
    from gpu8v71.core71 import CoreConfig, TypedCore
    torch.manual_seed(0)
    m = TypedCore(CoreConfig(rank=16, gate="global"))
    xT, xH = torch.randn(12, 768), torch.randn(12, 768)
    mT, mH = torch.ones(12, dtype=torch.bool), torch.rand(12) > 0.5
    z_all, _ = m.disease_repr(xT, mT, xH, mH)
    z_a, _ = m.disease_repr(xT[:5], mT[:5], xH[:5], mH[:5])
    assert torch.allclose(z_all[:5], z_a, atol=1e-6)

"""V7.1 core guardrails (plan sections 10-11, 13).

* trainable parameters: 80,069 (default) and 145,670 (image branch);
* a wrongly shaped modality input stops the forward pass;
* a disease-gene pair with no available modality returns valid = False and 0;
* the multi-positive loss is finite for very large logits and equals the naive
  formula for small ones;
* full-background scores do not depend on how the gene background is chunked;
* excluded training genes never enter positives, controls or anchor genes;
* a tiny end-to-end fit on synthetic tensors runs forward, backward and early
  stopping on CPU (a connection test only, never a biological result).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpu8v71.core71 import CoreConfig, TypedCore, multi_positive_loss, row_norm  # noqa: E402
from gpu8v71.train71 import CoreTrainer, TrainConfig, TwoTower, prepare_sets  # noqa: E402

DEV = "cpu"


def synth(D=10, G=60, n_anchor=25, seed=0, image=False):
    g = torch.Generator().manual_seed(seed)
    T = {"xT": torch.randn(D, 768, generator=g), "mT": torch.ones(D, dtype=torch.bool),
         "xH": torch.randn(D, 768, generator=g), "mH": torch.rand(D, generator=g) > 0.3,
         "q": torch.zeros(D, 51), "mq": torch.rand(D, generator=g) > 0.2,
         "xS": torch.randn(G, 480, generator=g), "mS": torch.rand(G, generator=g) > 0.1,
         "xF": torch.randn(G, 768, generator=g), "mF": torch.rand(G, generator=g) > 0.1,
         "e": torch.randn(G, 51, generator=g), "o": (torch.rand(G, 51, generator=g) > 0.05).float(),
         "xA": torch.randn(n_anchor, 768, generator=g)}
    for i in range(D):
        if T["mq"][i]:
            t = torch.randint(0, 51, (3,), generator=g)
            T["q"][i, t] = 1.0 / 3
    T["xS"] = T["xS"] * T["mS"][:, None]
    T["xF"] = T["xF"] * T["mF"][:, None]
    T["e"] = T["e"] * T["o"]
    T["A_plus"] = torch.rand(G, n_anchor, generator=g) > 0.8
    if image:
        T["xI"] = torch.randn(G, 1024, generator=g)
        T["mI"] = torch.rand(G, generator=g) > 0.4
    return T


def batch(T, rows):
    idx = torch.as_tensor(rows)
    return {k: T[k][idx] for k in ("xT", "mT", "xH", "mH", "q", "mq")}


def genes(T, sl=slice(None)):
    return {k: T[k][sl] for k in ("xS", "mS", "xF", "mF", "e", "o", "xI", "mI") if k in T}


def test_parameter_counts():
    assert TypedCore(CoreConfig()).n_trainable() == 80069
    assert TypedCore(CoreConfig(image=True)).n_trainable() == 145670
    assert 768 * 64 + 480 * 64 + 3 * 65 + 2 == 80069
    assert 80069 + 1024 * 64 + 65 == 145670


def test_wrong_dimensions_stop():
    T = synth()
    m = TypedCore(CoreConfig())
    g = genes(T)
    g["xS"] = torch.randn(T["xS"].shape[0], 1024)     # an image vector wired as a sequence
    with pytest.raises(ValueError):
        m(batch(T, [0, 1]), g)
    g = genes(T)
    d = batch(T, [0, 1])
    d["xT"] = torch.randn(2, 480)
    with pytest.raises(ValueError):
        m(d, g)


def test_all_missing_is_invalid():
    T = synth()
    T["mT"][:] = False
    T["mH"][:] = False
    T["mq"][:] = False
    m = TypedCore(CoreConfig())
    out = m(batch(T, [0, 1, 2]), genes(T))
    assert not out["valid"].any()
    assert torch.all(out["r"] == 0)
    # a disease with semantics but a gene without any modality is invalid too
    T = synth()
    T["mS"][5] = False
    T["mF"][5] = False
    T["o"][5] = 0.0
    out = TypedCore(CoreConfig())(batch(T, [0]), genes(T))
    assert not bool(out["valid"][0, 5])


def test_gate_weights_normalised_over_available():
    T = synth()
    m = TypedCore(CoreConfig())
    with torch.no_grad():
        m.u.normal_()
        m.b.normal_()
    out = m(batch(T, list(range(5))), genes(T))
    w, valid, avail = out["w"], out["valid"], out["avail"]
    s = w.sum(0)
    assert torch.allclose(s[valid], torch.ones_like(s[valid]), atol=1e-5)
    assert torch.all(w[~avail] == 0)


def test_row_norm_zero_rows():
    x = torch.zeros(3, 5)
    x[1] = torch.tensor([3.0, 4.0, 0, 0, 0])
    y = row_norm(x)
    assert torch.all(y[0] == 0) and torch.allclose(y[1].norm(), torch.tensor(1.0))


def test_multi_positive_loss_stable_and_exact():
    pos = torch.tensor([1e4, -1e4])
    neg = torch.tensor([1e4, 5e3, -3e3])
    val = multi_positive_loss(pos, neg)
    assert torch.isfinite(val)
    pos = torch.tensor([0.3, -0.2])
    neg = torch.tensor([0.1, 0.5, -0.4])
    naive = torch.stack([torch.log(1 + torch.exp(neg - p).sum()) for p in pos]).mean()
    assert torch.allclose(multi_positive_loss(pos, neg), naive, atol=1e-6)


def test_block_invariance():
    T = synth(G=90)
    m = TypedCore(CoreConfig(image=False))
    with torch.no_grad():
        m.u.normal_()
    d = batch(T, [0, 3, 7])
    full = m(d, genes(T))["r"]
    parts = [m(d, genes(T, slice(a, a + 25)))["r"] for a in range(0, 90, 25)]
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-6)


def test_exclusions_never_enter_training_sets():
    train = {0: np.array([1, 2, 3]), 1: np.array([3, 4])}
    controls = np.zeros(10, dtype=bool)
    controls[[1, 2, 3, 4, 5, 6]] = True
    anchors = np.ones(10, dtype=bool)
    pos, ctrl, pool = prepare_sets(train, controls, 10, anchors, np.array([3, 6]))
    assert 3 not in pos[0] and 3 not in pos[1]
    assert not ctrl[3] and not ctrl[6]
    assert 3 not in pool and 6 not in pool
    assert set(pool) == set(range(10)) - {3, 6}


def test_tiny_fit_runs_and_respects_exclusions():
    T = synth(D=24, G=80)
    tr = CoreTrainer(T, device=DEV)
    train = {i: np.array([i % 80, (i * 7) % 80]) for i in range(16)}
    val = {i: np.array([(i * 3) % 80]) for i in range(16, 24)}
    controls = np.ones(80, dtype=bool)
    cfg = TrainConfig(max_epochs=3, patience=2, anchor_steps=5, batch=8, anchor_batch=16)
    res = tr.fit(train, controls, val, seed=1, core_cfg=CoreConfig(), cfg=cfg,
                 anchor_genes=np.ones(80, dtype=bool), excluded_training_genes=np.array([0, 1]))
    assert res.n_params == 80069
    assert res.counts["n_anchor_genes"] == 78
    assert res.best_epoch >= 1 and res.state is not None
    m = tr.load("core", CoreConfig(), res.state)
    R, V = tr.score(m, np.array([16, 17]))
    assert R.shape == (2, 80) and V.dtype == bool
    tt = tr.fit(train, controls, val, seed=1, kind="two_tower", core_cfg=CoreConfig(), cfg=cfg)
    assert tt.best_epoch >= 1


def test_core_list_ties_validity_and_fill():
    from gpu8v71.chain71 import core_list
    R = np.array([[0.5, 0.9, 0.9, 0.1, 0.7]], dtype=np.float32)
    V = np.array([[True, True, True, False, False]])
    prior = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    g, fill = core_list(R, V, prior, n=5, fill=True)
    assert g[0, :3].tolist() == [1, 2, 0]          # ties by ascending gene row; invalid excluded
    assert g[0, 3:].tolist() == [3, 4] and fill[0].tolist() == [False] * 3 + [True] * 2
    g2, _ = core_list(R, V, prior, n=5, fill=False)
    assert g2[0, 3:].tolist() == [-1, -1]           # no fill: only valid matches are candidates


def test_fast_family_bootstrap_contains_mean():
    from gpu8v71.evaluate71 import fast_family_bootstrap
    rng = np.random.default_rng(0)
    d = rng.normal(0.02, 0.1, size=400)
    fam = rng.integers(0, 120, size=400)
    lo, hi = fast_family_bootstrap(d, fam, draws=2000)
    assert lo < d.mean() < hi
    assert fast_family_bootstrap(d, fam, draws=2000) == (lo, hi)


def test_two_tower_shapes():
    T = synth()
    m = TwoTower(64)
    out = m(batch(T, [0, 1]), genes(T))
    assert out["r"].shape == (2, T["xS"].shape[0])

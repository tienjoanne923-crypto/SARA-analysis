"""B8 plain meta-ridge and B5 same-capacity MLP.

B8 is the comparison that decides whether the *evidence-dependence* machinery
earns its place.  It is deliberately given everything the proposed method has:

* the same encoder capacity and the same 64-dim adaptation representation;
* the same support genes, the same anchors, the same task sampling and the same
  optimisation budget;
* the same differentiable ridge solver, trained through the solver.

The only differences are the ones under test: support weights are equal per
unique gene (no evidence grouping), lambda and rho are fixed rather than
predicted from a support summary, and the structured missing-information views
are replaced by ordinary cold-task training.

If the full method cannot beat this, the plan says the claim must shrink — the
code therefore makes the comparison as easy as possible to run honestly.
"""
from __future__ import annotations

import numpy as np
import torch

from .. import adapt as adapt_mod
from ..config import Gpu8Config
from ..episodes import EpisodeBuilder
from ..model import Gpu8Model, PlainMLP
from ..train import EpisodeTensors, to_tensor
from .simple import stable_tie_noise


class FixedControlWrapper(torch.nn.Module):
    """Wraps the shared encoder but replaces the learned control head."""

    def __init__(self, base: Gpu8Model, lam: float, rho: float):
        super().__init__()
        self.base = base
        self.lam_value, self.rho_value = float(lam), float(rho)

    def encode(self, *args, **kwargs):
        return self.base.encode(*args, **kwargs)

    def controls(self, summary: torch.Tensor):
        device = summary.device if torch.is_tensor(summary) else "cpu"
        return (torch.tensor(self.lam_value, device=device),
                torch.tensor(self.rho_value, device=device))

    def n_trainable(self) -> int:
        return self.base.n_trainable()

    def module_totals(self) -> dict:
        return self.base.module_totals()


def build_meta_ridge(cfg: Gpu8Config, seed: int = 42, lam: float = 1.0,
                     rho: float | None = None) -> FixedControlWrapper:
    """Identical capacity to the proposed model, fixed regularisation."""
    torch.manual_seed(seed)
    base = Gpu8Model(cfg.model)
    return FixedControlWrapper(base, lam=lam,
                               rho=float(rho if rho is not None else cfg.model.rho_max))


def build_plain_mlp(cfg: Gpu8Config, seed: int = 42) -> PlainMLP:
    torch.manual_seed(seed)
    return PlainMLP(cfg.model)


@torch.no_grad()
def score_mlp_with_fixed_ridge(cfg: Gpu8Config, model: PlainMLP, builder: EpisodeBuilder,
                               disease_id: str, support_genes: list[str],
                               candidate_rows: np.ndarray, device,
                               lam: float = 1.0, rho: float | None = None,
                               use_ridge: bool = True) -> np.ndarray:
    """B5 scoring: plain MLP, optionally followed by a *fixed* ridge post-hoc step.

    The fixed-ridge variant is the fair support comparator for B5: the support
    is not withheld from it just because its encoder was not trained through a
    solver.
    """
    from ..candidates import sample_anchors
    rho = float(rho if rho is not None else cfg.model.rho_max)
    idx = builder.b.idx
    d_row = idx.d_ix[disease_id]
    g_ix = idx.g_ix
    support_rows = np.array([g_ix[g] for g in support_genes if g in g_ix], dtype=np.int64)

    a = torch.zeros(cfg.model.adapt_dim, device=device)
    if use_ridge and len(support_rows):
        r = builder.retriever_scores(disease_id, d_row)
        known = {g_ix[g] for g in builder.positives.get(disease_id, []) if g in g_ix}
        anchors = sample_anchors(cfg, disease_id, builder.b.background_rows,
                                 set(support_rows.tolist()), known, fused_scores=r["fused"])
        if len(anchors):
            rows = np.unique(np.concatenate([support_rows, anchors]))
            ch, mask, q = builder.pair_features(disease_id, d_row, rows)
            dv, gv = builder.node_features(d_row, rows)
            enc = model.encode(to_tensor(dv, device), to_tensor(gv, device),
                               to_tensor(ch, device), to_tensor(mask, device),
                               to_tensor(q, device))
            pos = {int(g): i for i, g in enumerate(rows)}
            i_s = torch.as_tensor([pos[int(x)] for x in support_rows], dtype=torch.long,
                                  device=device)
            i_a = torch.as_tensor([pos[int(x)] for x in anchors], dtype=torch.long,
                                  device=device)
            w = torch.full((len(i_s),), 1.0 / len(i_s), device=device)
            res = adapt_mod.adapt(enc.phi[i_s], enc.phi[i_a], enc.base_score[i_s],
                                  enc.base_score[i_a], w,
                                  torch.tensor(lam, device=device),
                                  torch.tensor(rho, device=device),
                                  b_clip=cfg.model.ridge_b_clip)
            a = res.a

    scores = np.zeros(len(candidate_rows), dtype=np.float32)
    bs = cfg.train.score_batch
    for start in range(0, len(candidate_rows), bs):
        chunk = candidate_rows[start:start + bs]
        ch, mask, q = builder.pair_features(disease_id, d_row, chunk)
        dv, gv = builder.node_features(d_row, chunk)
        enc = model.encode(to_tensor(dv, device), to_tensor(gv, device),
                           to_tensor(ch, device), to_tensor(mask, device),
                           to_tensor(q, device))
        s = adapt_mod.apply_adaptation(enc.base_score, enc.phi, a)
        scores[start:start + len(chunk)] = s.detach().cpu().numpy()
    return stable_tie_noise(scores, candidate_rows)

"""V6 sections 7.2-7.3 — ``fit_residual_ranker``.

    s_theta(d, g) = F(d, g) + kappa * tanh(delta_theta(d, g)),   kappa >= 0

F is the frozen, dev-tuned linear fusion; it is never re-learned here.
kappa = softplus(k_raw), and kappa = 0 (output identical to F) is always kept as
a legal model: after training, the residual is retained only if it beats F on
the inner validation set, otherwise the selected model *is* F.

Two residual families, both without any free disease-ID embedding:

* ``bilinear``  delta = <W_d x_d, W_g x_g> / sqrt(r) + u^T c_dg
  (x_d legal disease semantics, x_g gene function features, c_dg pair channels);
* ``attention`` the candidate gene attends over the disease's (<= 32) most
  informative HPO terms (2 heads); diseases without HPO receive delta = 0.
  Attention runs between one disease's terms and one candidate gene only, never
  between candidates, so a pair's score does not depend on the rest of the list.

Training is listwise over out-of-fold candidate pools C^OOF_d:

    L_rank = - 1/|D_hit| sum_d 1/|P_d| sum_{g in P_d} log softmax_{C^OOF_d}(s/tau)_g

with P_d = Q^tr_d ∩ C^OOF_d.  Diseases whose pool holds no positive are not given
an invented positive; they simply do not enter this loss.  Inference returns the
raw s, never a within-list normalised probability.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .scoring import Block, fusion_scores, recall_at


class BilinearResidual(nn.Module):
    kind = "bilinear"

    def __init__(self, d_dim: int, g_dim: int, c_dim: int, rank: int = 32, **_):
        super().__init__()
        self.Wd = nn.Linear(d_dim, rank, bias=False)
        self.Wg = nn.Linear(g_dim, rank, bias=False)
        self.lin = nn.Linear(c_dim, 1)
        self.k_raw = nn.Parameter(torch.tensor(-2.0))
        self.rank = rank

    def kappa(self):
        return Fn.softplus(self.k_raw)

    def delta(self, xd, xg, c, terms=None, term_emb=None):
        zd = self.Wd(xd)
        zg = self.Wg(xg)
        return (zg * zd[:, None, :]).sum(-1) / math.sqrt(self.rank) + self.lin(c).squeeze(-1)


class AttentionResidual(nn.Module):
    kind = "attention"

    def __init__(self, d_dim: int, g_dim: int, c_dim: int, hidden: int = 64, heads: int = 2,
                 e_dim: int = 64, **_):
        super().__init__()
        self.h, self.heads = hidden, heads
        self.q = nn.Linear(g_dim, hidden)
        self.k = nn.Linear(e_dim, hidden)
        self.v = nn.Linear(e_dim, hidden)
        self.gp = nn.Linear(g_dim, hidden)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden + c_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        self.k_raw = nn.Parameter(torch.tensor(-2.0))

    def kappa(self):
        return Fn.softplus(self.k_raw)

    def delta(self, xd, xg, c, terms=None, term_emb=None):
        B, N, _ = xg.shape
        mask = terms >= 0                                    # [B, T]
        E = term_emb[terms.clamp(min=0)]                     # [B, T, e]
        dh = self.h // self.heads
        Q = self.q(xg).view(B, N, self.heads, dh).transpose(1, 2)        # [B, H, N, dh]
        K = self.k(E).view(B, -1, self.heads, dh).transpose(1, 2)        # [B, H, T, dh]
        V = self.v(E).view(B, -1, self.heads, dh).transpose(1, 2)
        att = (Q @ K.transpose(-1, -2)) / math.sqrt(dh)                  # [B, H, N, T]
        att = att.masked_fill(~mask[:, None, None, :], -1e9)
        a = torch.softmax(att, dim=-1)
        ctx = (a @ V).transpose(1, 2).reshape(B, N, self.h)
        out = self.mlp(torch.cat([ctx, self.gp(xg), c], dim=-1)).squeeze(-1)
        has = mask.any(dim=1).float()[:, None]
        return out * has


MODELS = {"bilinear": BilinearResidual, "attention": AttentionResidual}


@dataclass
class ResidualFit:
    model: nn.Module
    kind: str
    config: dict
    history: list
    best_epoch: int
    val_recall_residual: float
    val_recall_base: float
    selected: str                 # "residual" or "kappa0"
    kappa: float
    seconds: float
    n_train_lists: int
    n_params: int


class Standardiser:
    def __init__(self, arr: np.ndarray):
        flat = arr.reshape(-1, arr.shape[-1])
        self.mu = flat.mean(0)
        self.sd = np.maximum(flat.std(0), 1e-6)

    def __call__(self, a):
        return ((a - self.mu) / self.sd).astype(np.float32)


class ResidualTrainer:
    def __init__(self, weights: dict, gene_feats: np.ndarray, sem: dict, device: str = "cuda",
                 tau: float = 1.0):
        self.w = weights
        self.dev = device
        self.tau = tau
        g = gene_feats.astype(np.float32)
        self.g_std = Standardiser(g)
        self.G = torch.as_tensor(self.g_std(g), device=device)
        xd = np.concatenate([sem["hpo64"], sem["text64"], sem["has_hpo"][:, None],
                             sem["has_def"][:, None]], axis=1).astype(np.float32)
        self.d_std = Standardiser(xd)
        self.XD = torch.as_tensor(self.d_std(xd), device=device)
        self.terms = torch.as_tensor(sem["terms"].astype(np.int64), device=device)
        self.term_emb = torch.as_tensor(sem["term_emb"], device=device)
        self.c_std = None

    def tensors(self, b: Block, d_ix: dict):
        rows = torch.as_tensor([d_ix[d] for d in b.ids], device=self.dev)
        Fb = torch.as_tensor(fusion_scores(b, self.w).astype(np.float32), device=self.dev)
        C = torch.as_tensor(self.c_std(b.X), device=self.dev)
        genes = torch.as_tensor(b.genes, device=self.dev)
        y = torch.as_tensor(b.y, device=self.dev)
        return rows, Fb, C, genes, y

    def forward(self, model, rows, Fb, C, genes, sl):
        xd = self.XD[rows[sl]]
        xg = self.G[genes[sl]]
        d = model.delta(xd, xg, C[sl], terms=self.terms[rows[sl]], term_emb=self.term_emb)
        return Fb[sl] + model.kappa() * torch.tanh(d)

    @torch.no_grad()
    def score(self, model, b: Block, d_ix: dict, batch: int = 128) -> np.ndarray:
        model.eval()
        rows, Fb, C, genes, _ = self.tensors(b, d_ix)
        out = []
        for s in range(0, b.D, batch):
            sl = slice(s, s + batch)
            out.append(self.forward(model, rows, Fb, C, genes, sl).double().cpu().numpy())
        return np.concatenate(out, axis=0)

    def fit(self, kind: str, train: list[Block], val: Block, d_ix: dict, seed: int = 42,
            config: dict | None = None, max_epochs: int = 40, patience: int = 6,
            lr: float = 1e-3, wd: float = 1e-4, batch: int = 32) -> ResidualFit:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        config = dict(config or {})
        allX = np.concatenate([b.X.reshape(-1, b.X.shape[2]) for b in train], axis=0)
        self.c_std = Standardiser(allX)
        del allX
        model = MODELS[kind](d_dim=self.XD.shape[1], g_dim=self.G.shape[1],
                             c_dim=train[0].X.shape[2], **config).to(self.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        packs = []
        for b in train:
            has = b.y.sum(axis=1) > 0
            sub = b.subset([d for d, h in zip(b.ids, has) if h])
            packs.append(self.tensors(sub, d_ix))
        n_lists = int(sum(p[0].shape[0] for p in packs))
        base_val = float(recall_at(val, fusion_scores(val, self.w)).mean())
        best, best_state, best_ep, bad, hist = -1.0, None, -1, 0, []
        rng = np.random.default_rng(seed)
        for ep in range(max_epochs):
            model.train()
            tot, nb = 0.0, 0
            order = [(pi, s) for pi, p in enumerate(packs) for s in range(0, p[0].shape[0], batch)]
            rng.shuffle(order)
            for pi, s in order:
                rows, Fb, C, genes, y = packs[pi]
                sl = slice(s, s + batch)
                sc = self.forward(model, rows, Fb, C, genes, sl) / self.tau
                logp = torch.log_softmax(sc, dim=1)
                yy = y[sl]
                loss = -((logp * yy).sum(1) / yy.sum(1).clamp(min=1)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss)
                nb += 1
            vr = float(recall_at(val, self.score(model, val, d_ix)).mean())
            hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_recall10": vr,
                         "kappa": float(model.kappa())})
            if vr > best + 1e-6:
                best, best_ep, bad = vr, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        model.load_state_dict(best_state)
        selected = "residual" if best > base_val + 1e-9 else "kappa0"
        return ResidualFit(model=model, kind=kind, config=config, history=hist,
                           best_epoch=best_ep, val_recall_residual=best,
                           val_recall_base=base_val, selected=selected,
                           kappa=float(model.kappa()), seconds=time.time() - t0,
                           n_train_lists=n_lists,
                           n_params=int(sum(p.numel() for p in model.parameters())))

    def final_scores(self, fit: ResidualFit, b: Block, d_ix: dict) -> np.ndarray:
        if fit.selected == "kappa0":
            return fusion_scores(b, self.w)
        return self.score(fit.model, b, d_ix)

"""The small neural representation (plan section 9.2, 9.3, 9.5).

Shapes follow the plan exactly:

    f_D, f_G : 512 -> 128           node encoders (Linear + LayerNorm + GELU)
    f_P      : 528 -> 128           pair encoder on [z_d; z_g; |z_d-z_g|; z_d*z_g; t; q]
    f_Q      : 12 -> 32 -> 4        channel gate
    f_R      : 4 -> 16 -> 2         evidence-dependence control head
    W_phi    : 128 -> 64            adaptation representation, unit-ball clipped

LayerNorm rather than BatchNorm and no cross-candidate softmax: a pair's raw
score must not depend on which other candidates happen to be in the batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class NodeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.lin(x)))


class PairEncoder(nn.Module):
    def __init__(self, hidden: int, n_channels: int, n_quality: int, out_dim: int):
        super().__init__()
        in_dim = 4 * hidden + n_channels + n_quality
        self.in_dim = in_dim
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, zd: torch.Tensor, zg: torch.Tensor,
                t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([zd, zg, (zd - zg).abs(), zd * zg, t, q], dim=-1)
        return F.gelu(self.norm(self.lin(feats)))


class ChannelGate(nn.Module):
    """f_Q: 12 -> 32 -> 4, softmax over the *available* channels only."""

    def __init__(self, n_quality: int, hidden: int, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_quality, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_channels))

    def forward(self, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.net(q)
        neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(mask > 0, logits, torch.full_like(logits, neg))
        all_missing = (mask.sum(dim=-1, keepdim=True) <= 0)
        pi = torch.softmax(masked, dim=-1)
        # every channel missing: contribute exactly zero rather than softmax(-inf)
        return torch.where(all_missing, torch.zeros_like(pi), pi)


class ControlHead(nn.Module):
    """f_R: 4 -> 16 -> 2 producing (lambda_d, rho_d) in their frozen ranges."""

    def __init__(self, hidden: int, lambda_min: float, lambda_max: float, rho_max: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.lambda_min, self.lambda_max, self.rho_max = lambda_min, lambda_max, rho_max

    def forward(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(r)
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(out[..., 0])
        rho = self.rho_max * torch.sigmoid(out[..., 1])
        return lam, rho


@dataclass
class EncodeResult:
    base_score: torch.Tensor      # s0(d, g)
    phi: torch.Tensor             # unit-ball adaptation representation
    pair_hidden: torch.Tensor
    channel_pi: torch.Tensor


class Gpu8Model(nn.Module):
    """Small neural representation + evidence-dependent support adaptation."""

    def __init__(self, cfg: ModelConfig, n_channels: int = 4, n_quality: int = 12):
        super().__init__()
        self.cfg = cfg
        self.n_channels, self.n_quality = n_channels, n_quality
        self.f_D = NodeEncoder(cfg.node_in_dim, cfg.hidden_dim)
        self.f_G = NodeEncoder(cfg.node_in_dim, cfg.hidden_dim)
        self.f_P = PairEncoder(cfg.hidden_dim, n_channels, n_quality, cfg.pair_hidden_dim)
        self.f_Q = ChannelGate(n_quality, cfg.channel_net_hidden, n_channels)
        self.f_R = ControlHead(cfg.control_net_hidden, cfg.lambda_min,
                               cfg.lambda_max, cfg.rho_max)
        self.W_phi = nn.Linear(cfg.pair_hidden_dim, cfg.adapt_dim)
        self.score_head = nn.Linear(cfg.pair_hidden_dim, 1)

    # -- encoding ---------------------------------------------------------
    def encode(self, disease_x: torch.Tensor, gene_x: torch.Tensor,
               channels: torch.Tensor, channel_mask: torch.Tensor,
               quality: torch.Tensor) -> EncodeResult:
        """disease_x: [B, 512] broadcast per pair; gene_x: [B, 512]."""
        zd = self.f_D(disease_x)
        zg = self.f_G(gene_x)
        h = self.f_P(zd, zg, channels, quality)
        pi = self.f_Q(quality, channel_mask)
        channel_term = (pi * channels * channel_mask).sum(dim=-1)
        base = self.score_head(h).squeeze(-1) + channel_term
        raw_phi = self.W_phi(h)
        norm = raw_phi.norm(dim=-1, keepdim=True)
        phi = raw_phi / torch.clamp(norm, min=1.0)      # ||phi|| <= 1
        return EncodeResult(base_score=base, phi=phi, pair_hidden=h, channel_pi=pi)

    def controls(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.f_R(summary)

    # -- bookkeeping ------------------------------------------------------
    def n_trainable(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def parameter_table(self) -> list[dict]:
        rows = []
        for name, p in self.named_parameters():
            rows.append({"name": name, "shape": list(p.shape),
                         "numel": int(p.numel()), "trainable": bool(p.requires_grad)})
        return rows

    def module_totals(self) -> dict:
        out: dict[str, int] = {}
        for name, p in self.named_parameters():
            key = name.split(".")[0]
            out[key] = out.get(key, 0) + int(p.numel())
        out["TOTAL"] = self.n_trainable()
        return out


def build_model(cfg: ModelConfig, n_channels: int = 4, n_quality: int = 12,
                seed: int | None = None) -> Gpu8Model:
    if seed is not None:
        torch.manual_seed(seed)
    return Gpu8Model(cfg, n_channels=n_channels, n_quality=n_quality)


# --------------------------------------------------------------------------
# Same-capacity MLP comparator (B5) — no meta adaptation learning
# --------------------------------------------------------------------------
class PlainMLP(nn.Module):
    """Matched-capacity comparator: identical encoders, no adaptation learning."""

    def __init__(self, cfg: ModelConfig, n_channels: int = 4, n_quality: int = 12):
        super().__init__()
        self.f_D = NodeEncoder(cfg.node_in_dim, cfg.hidden_dim)
        self.f_G = NodeEncoder(cfg.node_in_dim, cfg.hidden_dim)
        self.f_P = PairEncoder(cfg.hidden_dim, n_channels, n_quality, cfg.pair_hidden_dim)
        self.head = nn.Sequential(nn.Linear(cfg.pair_hidden_dim, cfg.pair_hidden_dim),
                                  nn.GELU(), nn.Linear(cfg.pair_hidden_dim, 1))
        self.W_phi = nn.Linear(cfg.pair_hidden_dim, cfg.adapt_dim)

    def encode(self, disease_x, gene_x, channels, channel_mask, quality) -> EncodeResult:
        zd, zg = self.f_D(disease_x), self.f_G(gene_x)
        h = self.f_P(zd, zg, channels, quality)
        base = self.head(h).squeeze(-1)
        raw_phi = self.W_phi(h)
        phi = raw_phi / torch.clamp(raw_phi.norm(dim=-1, keepdim=True), min=1.0)
        return EncodeResult(base_score=base, phi=phi, pair_hidden=h,
                            channel_pi=torch.zeros_like(channels))

    def n_trainable(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

"""V7.1 section 10 — the typed multimodal retrieval core.

Semantic space (shared rank r = 64 by default):

    z^T_d = norm(P_T x^T_d),  z^H_d = norm(P_T x^H_d),
    z^S_g = norm(P_S x^S_g),  z^F_g = norm(P_T x^F_g),  [z^I_g = norm(P_I x^I_g)]

Text, HPO, function terms and anchor definitions all come from the same frozen
SapBERT, so they share P_T; sequences (ESM-2, 480-d) have their own P_S, images
(1,024-d) their own P_I.  Linear maps carry no bias, every ``norm`` is row-wise
(never batch dependent), and a row with no input maps to the zero vector.

Text and HPO are mixed with two logits restricted to the available inputs; if
both are missing the disease representation is zero and *semantically invalid*
(no softmax over an all -inf row).

Per-modality scores  s_S = <z_d, z^S_g>,  s_F = <z_d, z^F_g>,  [s_I = <z_d, z^I_g>],
    s_E = tanh( sum_t q_d(t) o_gt E~_gt / (3 c_dg) )  valid iff d has tissues and
          c_dg = sum_t q_d(t) o_gt >= 0.8,
    [s_M = 2 S(d, g) - 1 for a checked mouse match S in [0, 1]].

Gate  w_m(d,g) = a_m(d,g) exp(b_m + u_m^T z_d) / sum_j a_j(d,g) exp(b_j + u_j^T z_d),
r_MM(d,g) = sum_m w_m s_m;  if no modality is available the score is 0 **and**
``valid = False``: callers must read ``valid`` and never treat that 0 as a
neutral score.  The gate reads the disease representation and availability
only; never a label, a hit, the number of positives or the cold status.

Trainable parameters: 768*64 + 480*64 + 3*(1 + 64) + 2 = 80,069;
with the image branch + 1024*64 + 65 = 145,670 (frozen encoders excluded).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

D_SEM, D_SEQ, D_IMG, N_TISSUE = 768, 480, 1024, 51
EPS = 1e-8


def row_norm(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalisation; an all-zero row stays zero."""
    n = x.norm(dim=-1, keepdim=True)
    return torch.where(n > EPS, x / n.clamp_min(EPS), torch.zeros_like(x))


@dataclass
class CoreConfig:
    rank: int = 64
    gate: str = "small"            # small | global
    image: bool = False
    mouse: bool = False
    expr_scale: float = 3.0
    expr_min_cov: float = 0.8
    use_sequence: bool = True
    use_function: bool = True
    use_expression: bool = True

    def modalities(self) -> list[str]:
        m = []
        if self.use_sequence:
            m.append("S")
        if self.use_function:
            m.append("F")
        if self.use_expression:
            m.append("E")
        if self.image:
            m.append("I")
        if self.mouse:
            m.append("M")
        return m


class TypedCore(nn.Module):
    def __init__(self, cfg: CoreConfig | None = None):
        super().__init__()
        self.cfg = cfg or CoreConfig()
        r = self.cfg.rank
        self.mods = self.cfg.modalities()
        self.P_T = nn.Linear(D_SEM, r, bias=False)
        self.P_S = nn.Linear(D_SEQ, r, bias=False) if self.cfg.use_sequence else None
        self.P_I = nn.Linear(D_IMG, r, bias=False) if self.cfg.image else None
        self.mix = nn.Parameter(torch.zeros(2))
        M = len(self.mods)
        self.b = nn.Parameter(torch.zeros(M))
        if self.cfg.gate == "small":
            self.u = nn.Parameter(torch.zeros(M, r))
        else:
            self.register_buffer("u", torch.zeros(M, r))
        for lin in (self.P_T, self.P_S, self.P_I):
            if lin is not None:
                nn.init.normal_(lin.weight, std=1.0 / lin.in_features ** 0.5)

    # ---------------------------------------------------------------- inputs
    @staticmethod
    def _check(x: torch.Tensor, dim: int, name: str) -> None:
        if x.dim() != 2 or x.shape[1] != dim:
            raise ValueError(f"{name}: expected [*, {dim}], got {tuple(x.shape)}")

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------- disease
    def disease_repr(self, xT, mT, xH, mH) -> tuple[torch.Tensor, torch.Tensor]:
        self._check(xT, D_SEM, "d_text")
        self._check(xH, D_SEM, "d_hpo")
        zT = row_norm(self.P_T(xT))
        zH = row_norm(self.P_T(xH))
        avail = torch.stack([mT, mH], dim=1).to(zT.dtype)                  # [B, 2]
        logits = self.mix[None, :].expand_as(avail)
        m = torch.where(avail > 0, logits, torch.full_like(logits, -1e4)).max(dim=1, keepdim=True).values
        e = torch.exp(logits - m) * avail
        den = e.sum(dim=1, keepdim=True)
        w = torch.where(den > 0, e / den.clamp_min(EPS), torch.zeros_like(e))
        z = row_norm(w[:, :1] * zT + w[:, 1:] * zH)
        valid = (avail.sum(dim=1) > 0)
        z = z * valid[:, None].to(z.dtype)
        return z, valid

    # ---------------------------------------------------------------- genes
    def gene_repr(self, g: dict) -> dict:
        out = {}
        if "S" in self.mods:
            self._check(g["xS"], D_SEQ, "g_sequence")
            out["S"] = row_norm(self.P_S(g["xS"])) * g["mS"][:, None].to(g["xS"].dtype)
        if "F" in self.mods:
            self._check(g["xF"], D_SEM, "g_function")
            out["F"] = row_norm(self.P_T(g["xF"])) * g["mF"][:, None].to(g["xF"].dtype)
        if "I" in self.mods:
            self._check(g["xI"], D_IMG, "g_image")
            out["I"] = row_norm(self.P_I(g["xI"])) * g["mI"][:, None].to(g["xI"].dtype)
        return out

    def anchor_repr(self, xA: torch.Tensor) -> torch.Tensor:
        self._check(xA, D_SEM, "anchors")
        return row_norm(self.P_T(xA))

    def sequence_repr(self, xS: torch.Tensor) -> torch.Tensor:
        self._check(xS, D_SEQ, "g_sequence")
        return row_norm(self.P_S(xS))

    # --------------------------------------------------------------- scores
    def forward(self, d: dict, g: dict, gene_reprs: dict | None = None) -> dict:
        """Scores of a disease batch against a gene block.

        d: xT, mT, xH, mH [B, ...], q [B, 51], mq [B], (mouse: sM [B, Gb], mM [B, Gb])
        g: xS, mS, xF, mF, e [Gb, 51] (centred), o [Gb, 51] (float), (xI, mI)
        """
        z, sem_ok = self.disease_repr(d["xT"], d["mT"], d["xH"], d["mH"])
        gr = gene_reprs if gene_reprs is not None else self.gene_repr(g)
        B = z.shape[0]
        scores, avail = {}, {}
        for m in self.mods:
            if m in ("S", "F", "I"):
                zg = gr[m]
                mg = g["m" + m].to(torch.bool)
                scores[m] = z @ zg.T
                avail[m] = sem_ok[:, None] & mg[None, :]
            elif m == "E":
                q = d["q"]
                if q.shape[1] != g["e"].shape[1]:
                    raise ValueError("tissue axes differ between d_tissue and g_expression")
                c = q @ g["o"].T                                          # [B, Gb]
                num = q @ g["e"].T
                scores[m] = torch.tanh(num / (self.cfg.expr_scale * c.clamp_min(EPS)))
                avail[m] = d["mq"].to(torch.bool)[:, None] & (c >= self.cfg.expr_min_cov)
            elif m == "M":
                if d["sM"].shape[0] != z.shape[0] or d["sM"].shape != d["mM"].shape:
                    raise ValueError("mouse score block does not match the disease batch")
                scores[m] = 2.0 * d["sM"].float() - 1.0
                avail[m] = d["mM"].to(torch.bool)
        g_log = self.b[None, :] + z @ self.u.T                            # [B, M]
        g_log = g_log - g_log.max(dim=1, keepdim=True).values
        ex = torch.exp(g_log)
        num = torch.stack([avail[m].to(z.dtype) * ex[:, i:i + 1] for i, m in enumerate(self.mods)], 0)
        den = num.sum(0)
        valid = den > 0
        w = torch.where(valid[None], num / den.clamp_min(EPS)[None], torch.zeros_like(num))
        S = torch.stack([torch.where(avail[m], scores[m], torch.zeros_like(scores[m]))
                         for m in self.mods], 0)
        r = (w * S).sum(0)
        return {"r": torch.where(valid, r, torch.zeros_like(r)), "valid": valid, "w": w,
                "s": S, "avail": torch.stack([avail[m] for m in self.mods], 0), "z": z,
                "sem_ok": sem_ok, "gate_logits": g_log, "B": B}


def multi_positive_loss(pos: torch.Tensor, neg: torch.Tensor, neg_mask: torch.Tensor | None = None
                        ) -> torch.Tensor:
    """mean_p log(1 + sum_u exp(neg_u - pos_p)) = softplus(logsumexp(neg) - pos_p).

    pos [P], neg [U] (already divided by the temperature).  Stable for large
    logits (log-sum-exp / softplus); returns 0-dim tensor.
    """
    if neg_mask is not None:
        neg = neg.masked_fill(~neg_mask, float("-inf"))
    lse = torch.logsumexp(neg, dim=-1)
    return F.softplus(lse - pos).mean()

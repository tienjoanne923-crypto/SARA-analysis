"""Low-dimensional weighted ridge adaptation with a radial projection.

Plan section 9.6 and 11.  For exposed support genes g in S_d and anchors u in U_d:

    D_gu = phi_dg - phi_du
    b_gu = clip(1 - [s0(d,g) - s0(d,u)], -5, 5)
    W_gu = omega_g / |U_d|

    (D^T W D + lambda_d I) a_hat = D^T W b          (Cholesky, FP32)
    a_d = a_hat * min(1, rho_d / ||a_hat||)
    s(d, g | S_d) = s0(d, g) + a_d . phi_dg

Property 1: with W >= 0 and lambda > 0 the coefficient matrix is positive
definite, so the unprojected solution is unique.
Property 2: ||phi|| <= 1 and ||a|| <= rho give |s - s0| <= rho, i.e. the score
perturbation is bounded and two candidates whose base scores differ by more
than 2*rho cannot swap order because of the adaptation.

Everything here is differentiable: gradients flow through the solve into the
encoder, the adaptation representation and the control head.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

EPS = 1e-8


@dataclass
class AdaptResult:
    a: torch.Tensor                   # projected coefficients [r]
    a_raw: torch.Tensor               # ridge solution before projection
    radius_used: torch.Tensor
    lam: torch.Tensor
    n_support: int
    n_anchor: int
    condition_number: float = float("nan")
    solver: str = "cholesky"


def build_design(phi_support: torch.Tensor, phi_anchor: torch.Tensor,
                 base_support: torch.Tensor, base_anchor: torch.Tensor,
                 weights: torch.Tensor, b_clip: float = 5.0
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (D [k*u, r], b [k*u], w [k*u]) for the support-anchor pairs."""
    k, r = phi_support.shape
    u = phi_anchor.shape[0]
    D = (phi_support.unsqueeze(1) - phi_anchor.unsqueeze(0)).reshape(k * u, r)
    margin = base_support.unsqueeze(1) - base_anchor.unsqueeze(0)
    b = torch.clamp(1.0 - margin, min=-b_clip, max=b_clip).reshape(k * u)
    w = (weights.unsqueeze(1).expand(k, u) / float(max(u, 1))).reshape(k * u)
    return D, b, w


def solve_ridge(D: torch.Tensor, b: torch.Tensor, w: torch.Tensor,
                lam: torch.Tensor, compute_cond: bool = False
                ) -> tuple[torch.Tensor, float, str]:
    """(D^T W D + lambda I) a = D^T W b via Cholesky, never an explicit inverse."""
    r = D.shape[1]
    Dw = D * w.unsqueeze(1)
    G = D.transpose(0, 1) @ Dw
    rhs = (D * (w * b).unsqueeze(1)).sum(0)
    lam_t = lam if torch.is_tensor(lam) else torch.tensor(float(lam), device=D.device)
    A = G + lam_t * torch.eye(r, device=D.device, dtype=D.dtype)
    cond = float("nan")
    if compute_cond:
        with torch.no_grad():
            try:
                s = torch.linalg.svdvals(A.double())
                cond = float(s[0] / torch.clamp(s[-1], min=1e-300))
            except Exception:
                cond = float("nan")
    try:
        L = torch.linalg.cholesky(A)
        a = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)
        return a, cond, "cholesky"
    except Exception:
        a = torch.linalg.solve(A, rhs)
        return a, cond, "lu_fallback"


def radial_project(a: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Post-hoc projection onto the ball of radius rho (not a constrained optimum)."""
    norm = a.norm()
    scale = torch.clamp(rho / torch.clamp(norm, min=EPS), max=1.0)
    return a * scale


def adapt(phi_support: torch.Tensor, phi_anchor: torch.Tensor,
          base_support: torch.Tensor, base_anchor: torch.Tensor,
          weights: torch.Tensor, lam: torch.Tensor, rho: torch.Tensor,
          b_clip: float = 5.0, compute_cond: bool = False) -> AdaptResult:
    """Full adaptation step; k = 0 returns exactly zero without a solve."""
    r = phi_anchor.shape[1] if phi_anchor.dim() == 2 else phi_support.shape[1]
    device = phi_anchor.device if phi_anchor.numel() else phi_support.device
    dtype = phi_anchor.dtype if phi_anchor.numel() else phi_support.dtype
    if phi_support.numel() == 0 or phi_anchor.numel() == 0:
        zero = torch.zeros(r, device=device, dtype=dtype)
        return AdaptResult(a=zero, a_raw=zero, radius_used=torch.zeros((), device=device),
                           lam=lam if torch.is_tensor(lam) else torch.tensor(0.0),
                           n_support=int(phi_support.shape[0]),
                           n_anchor=int(phi_anchor.shape[0]), solver="skipped")
    D, b, w = build_design(phi_support, phi_anchor, base_support, base_anchor,
                           weights, b_clip=b_clip)
    a_raw, cond, solver = solve_ridge(D, b, w, lam, compute_cond=compute_cond)
    a = radial_project(a_raw, rho)
    return AdaptResult(a=a, a_raw=a_raw, radius_used=rho, lam=lam,
                       n_support=int(phi_support.shape[0]),
                       n_anchor=int(phi_anchor.shape[0]),
                       condition_number=cond, solver=solver)


def apply_adaptation(base_scores: torch.Tensor, phi: torch.Tensor,
                     a: torch.Tensor) -> torch.Tensor:
    """s(d, g | S_d) = s0(d, g) + a . phi_dg."""
    if a.numel() == 0:
        return base_scores
    if not a.requires_grad and float(a.detach().abs().sum()) == 0.0:
        return base_scores
    return base_scores + phi @ a


# --------------------------------------------------------------------------
# numerical properties used by the test suite (plan 18.2)
# --------------------------------------------------------------------------
def score_perturbation_bound(a: torch.Tensor, phi: torch.Tensor) -> tuple[float, float]:
    """Return (observed max |a.phi|, ||a||) so the bound can be asserted."""
    with torch.no_grad():
        obs = float((phi @ a).abs().max()) if phi.numel() else 0.0
        return obs, float(a.norm())


def sensitivity_bound(a1: torch.Tensor, a2: torch.Tensor, A1: torch.Tensor,
                      A2: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor,
                      lam_min: float) -> dict:
    """Equation (21): ||a1 - a2|| <= (||c1-c2|| + ||A1-A2|| ||a2||) / lambda_min.

    The regularisation change must be inside ``A1 - A2``; comparing only the
    Gram matrices after changing lambda would understate the bound.
    """
    with torch.no_grad():
        lhs = float((a1 - a2).norm())
        rhs = float((c1 - c2).norm() + torch.linalg.matrix_norm(A1 - A2, ord=2) * a2.norm()) / lam_min
        return {"lhs": lhs, "rhs": rhs, "holds": lhs <= rhs * (1 + 1e-4) + 1e-6}

"""V7.1 section 11 — anchor alignment, then multi-positive disease-gene ranking.

Stage 1 (``anchor_steps`` updates, skipped when lambda_A = 0):

    l_ga = <z^S_g, z_a> / tau_a,
    L_A  = 1/|G_A| sum_g 1/|A+_g| sum_{a in A+_g} log(1 + sum_{b in U_g} exp(l_gb - l_ga)),

U_g = anchors not observed as positive for g (unlabelled, not proven negative);
only P_S and the shared semantic projection P_T receive this gradient.

Stage 2 (joint, L = L_DG + lambda_A L_A):

    L_DG = 1/|D_tr| sum_d 1/|P_d| sum_{g in P_d} log(1 + sum_{u in U_d} exp((r(d,u) - r(d,g)) / tau)),

P_d = the training disease's visible positives; U_d = genes that appear in the
context's reference relations (default) minus d's own known positives, or the
whole background (sensitivity).  A positive without any valid modality cannot
produce a gradient: it is skipped and *counted*; diseases stay in every
evaluation denominator.  AdamW (lr 1e-3, weight decay 1e-4), 16 diseases per
batch, gradient-norm clipping 1.0, at most 60 epochs with early stopping on
the cold validation full-background Recall@10 (patience 8).  Encoders frozen.

``excluded_training_genes`` never enter positives, controls or anchor genes.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core71 import CoreConfig, TypedCore, row_norm


@dataclass
class TrainConfig:
    lambda_A: float = 0.2
    tau: float = 0.1
    tau_a: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 16
    anchor_batch: int = 128
    max_epochs: int = 60
    patience: int = 8
    anchor_steps: int = 100
    grad_clip: float = 1.0
    controls: str = "reference"          # reference | background


# ---------------------------------------------------------------------------
# linear two-tower control (E2): no types, no gate, no anchors
# ---------------------------------------------------------------------------
class TwoTower(nn.Module):
    def __init__(self, rank: int = 64):
        super().__init__()
        self.Wd = nn.Linear(768 + 768 + 51, rank, bias=False)
        self.Wg = nn.Linear(480 + 768 + 51, rank, bias=False)
        self.mods = ["TT"]

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, d: dict, g: dict, gene_reprs=None) -> dict:
        xd = torch.cat([d["xT"] * d["mT"][:, None], d["xH"] * d["mH"][:, None], d["q"]], dim=1)
        zd = row_norm(self.Wd(xd))
        dv = d["mT"] | d["mH"] | d["mq"]
        if gene_reprs is None:
            xg = torch.cat([g["xS"] * g["mS"][:, None], g["xF"] * g["mF"][:, None], g["e"]], dim=1)
            zg = row_norm(self.Wg(xg))
            gv = g["mS"] | g["mF"] | (g["o"].sum(1) > 0)
        else:
            zg, gv = gene_reprs["TT"], gene_reprs["TT_valid"]
        r = zd @ zg.T
        valid = dv[:, None] & gv[None, :]
        return {"r": torch.where(valid, r, torch.zeros_like(r)), "valid": valid}

    def gene_repr(self, g: dict) -> dict:
        xg = torch.cat([g["xS"] * g["mS"][:, None], g["xF"] * g["mF"][:, None], g["e"]], dim=1)
        return {"TT": row_norm(self.Wg(xg)), "TT_valid": g["mS"] | g["mF"] | (g["o"].sum(1) > 0)}


# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    state: dict
    history: list
    best_epoch: int
    best_val_recall10: float
    n_params: int
    seconds: float
    skipped_positive_rows: int
    counts: dict = field(default_factory=dict)


def take(t: dict, keys: list[str], idx: torch.Tensor) -> dict:
    return {k: t[k][idx] for k in keys if k in t}


D_KEYS = ["xT", "mT", "xH", "mH", "q", "mq", "sM", "mM"]     # sM/mM only when the mouse channel is on
G_KEYS = ["xS", "mS", "xF", "mF", "e", "o", "xI", "mI"]


def prepare_sets(train: dict[int, np.ndarray], controls: np.ndarray, n_genes: int,
                 anchor_genes: np.ndarray | None, excluded: np.ndarray | None,
                 control_rule: str = "reference") -> tuple[dict, np.ndarray, np.ndarray]:
    """(positive lists, control mask, anchor gene pool) with the exclusions applied.

    ``excluded`` genes never appear as a training positive, an unlabelled control
    or an anchor-training gene.
    """
    excl = np.zeros(n_genes, dtype=bool)
    if excluded is not None and len(excluded):
        excl[np.asarray(excluded, dtype=np.int64)] = True
    ctrl = (np.asarray(controls, dtype=bool).copy() if control_rule == "reference"
            else np.ones(n_genes, dtype=bool))
    ctrl &= ~excl
    pos = {int(r): np.setdiff1d(np.asarray(v, dtype=np.int64), np.flatnonzero(excl))
           for r, v in train.items()}
    pool = np.array([], dtype=np.int64)
    if anchor_genes is not None:
        pool = np.flatnonzero(np.asarray(anchor_genes, dtype=bool) & ~excl)
    return pos, ctrl, pool


class CoreTrainer:
    def __init__(self, T: dict, device: str = "cuda"):
        """T: feature-pack tensors on ``device`` (FeaturePack.tensors)."""
        self.T = T
        self.device = device
        self.G = T["xS"].shape[0]
        self.g_all = {k: T[k] for k in G_KEYS if k in T}

    def build_model(self, kind: str, core_cfg: CoreConfig | None) -> nn.Module:
        if kind == "two_tower":
            return TwoTower(core_cfg.rank if core_cfg else 64).to(self.device)
        return TypedCore(core_cfg or CoreConfig()).to(self.device)

    # ------------------------------------------------------------- scoring
    @torch.no_grad()
    def score(self, model: nn.Module, d_rows: np.ndarray, chunk: int = 128, extra_d: dict | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """Full-background r and valid for the given disease rows (CPU float32 / bool)."""
        model.eval()
        gr = model.gene_repr(self.g_all)
        R = np.zeros((len(d_rows), self.G), dtype=np.float32)
        V = np.zeros((len(d_rows), self.G), dtype=bool)
        for b0 in range(0, len(d_rows), chunk):
            idx = torch.as_tensor(d_rows[b0:b0 + chunk], device=self.device)
            d = take(self.T, D_KEYS, idx)
            if extra_d:
                d.update({k: v[b0:b0 + chunk] for k, v in extra_d.items()})
            out = model(d, self.g_all, gene_reprs=gr)
            R[b0:b0 + chunk] = out["r"].float().cpu().numpy()
            V[b0:b0 + chunk] = out["valid"].cpu().numpy()
        return R, V

    @torch.no_grad()
    def recall10(self, model, val: dict[int, np.ndarray]) -> float:
        rows = np.array(sorted(val))
        if len(rows) == 0:
            return 0.0
        model.eval()
        gr = model.gene_repr(self.g_all)
        rec = []
        for b0 in range(0, len(rows), 256):
            idx = torch.as_tensor(rows[b0:b0 + 256], device=self.device)
            out = model(take(self.T, D_KEYS, idx), self.g_all, gene_reprs=gr)
            s = torch.where(out["valid"], out["r"], torch.full_like(out["r"], -1e9))
            top = torch.topk(s, 10, dim=1).indices.cpu().numpy()
            okv = torch.gather(out["valid"], 1, torch.topk(s, 10, dim=1).indices).cpu().numpy()
            for i, r in enumerate(rows[b0:b0 + 256]):
                q = val[int(r)]
                hits = sum(1 for g, ok in zip(top[i], okv[i]) if ok and g in q)
                rec.append(hits / max(len(q), 1))
        return float(np.mean(rec))

    # ------------------------------------------------------------- fitting
    def fit(self, train: dict[int, np.ndarray], controls: np.ndarray, val: dict[int, np.ndarray],
            seed: int, kind: str = "core", core_cfg: CoreConfig | None = None,
            cfg: TrainConfig | None = None, anchor_genes: np.ndarray | None = None,
            excluded_training_genes: np.ndarray | None = None, log_every: int = 0) -> FitResult:
        cfg = cfg or TrainConfig()
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        rng = np.random.default_rng(seed)
        model = self.build_model(kind, core_cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        G = self.G
        use_anchor = kind == "core" and cfg.lambda_A > 0 and "S" in getattr(model, "mods", [])
        if use_anchor and anchor_genes is None:
            raise ValueError("anchor_genes eligibility mask required")
        pos_lists, ctrl, anchor_pool = prepare_sets(train, controls, G, anchor_genes,
                                                    excluded_training_genes, cfg.controls)
        ctrl_t = torch.as_tensor(ctrl, device=self.device)
        d_rows = np.array(sorted(train), dtype=np.int64)
        A = self.T["A_plus"] if use_anchor else None             # dense bool [G, n_anchor]
        n_excl = 0 if excluded_training_genes is None else int(len(np.unique(excluded_training_genes)))
        counts = {"n_train_diseases": int(len(d_rows)), "n_controls": int(ctrl.sum()),
                  "n_excluded_training_genes": n_excl,
                  "n_anchor_genes": int(len(anchor_pool)) if use_anchor else 0}

        def anchor_loss():
            gi = rng.choice(anchor_pool, size=min(cfg.anchor_batch, len(anchor_pool)), replace=False)
            gi_t = torch.as_tensor(gi, device=self.device)
            zS = model.sequence_repr(self.T["xS"][gi_t])
            zA = model.anchor_repr(self.T["xA"])
            logits = zS @ zA.T / cfg.tau_a                              # [b, n_anchor]
            pos = A[gi_t].to(torch.bool)                                # [b, n_anchor]
            neg = logits.masked_fill(pos, float("-inf"))
            lse = torch.logsumexp(neg, dim=1, keepdim=True)
            per = F.softplus(lse - logits)                              # [b, n_anchor]
            per = torch.where(pos, per, torch.zeros_like(per))
            npos = pos.sum(1).clamp_min(1)
            return (per.sum(1) / npos).mean()

        history = []
        if use_anchor:
            model.train()
            for step in range(cfg.anchor_steps):
                loss = anchor_loss()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                if step in (0, cfg.anchor_steps - 1):
                    history.append({"stage": "anchor", "step": step, "L_A": float(loss.detach())})
        best, best_state, best_ep, bad = -1.0, None, 0, 0
        skipped_total = 0
        for ep in range(1, cfg.max_epochs + 1):
            model.train()
            perm = rng.permutation(d_rows)
            tot, tot_dg, tot_a, nb, skipped = 0.0, 0.0, 0.0, 0, 0
            for b0 in range(0, len(perm), cfg.batch):
                rows = perm[b0:b0 + cfg.batch]
                idx = torch.as_tensor(rows, device=self.device)
                out = model(take(self.T, D_KEYS, idx), self.g_all)
                r, valid = out["r"], out["valid"]
                P = torch.zeros_like(valid)
                for i, dr in enumerate(rows):
                    pl = pos_lists[int(dr)]
                    if len(pl):
                        P[i, torch.as_tensor(pl, device=self.device)] = True
                known = P.clone()
                Pv = P & valid
                skipped += int((P & ~valid).sum())
                U = ctrl_t[None, :] & ~known & valid
                logits = r / cfg.tau
                lse = torch.logsumexp(logits.masked_fill(~U, float("-inf")), dim=1, keepdim=True)
                per = F.softplus(lse - logits)
                per = torch.where(Pv, per, torch.zeros_like(per))
                npos = Pv.sum(1)
                ok = (npos > 0) & (U.sum(1) > 0)
                if not ok.any():
                    continue
                L_dg = (per.sum(1)[ok] / npos[ok]).mean()
                loss = L_dg
                la = torch.zeros((), device=self.device)
                if use_anchor:
                    la = anchor_loss()
                    loss = loss + cfg.lambda_A * la
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                tot += float(loss)
                tot_dg += float(L_dg)
                tot_a += float(la)
                nb += 1
            skipped_total += skipped
            vr = self.recall10(model, val)
            rec = {"stage": "joint", "epoch": ep, "loss": tot / max(nb, 1), "L_DG": tot_dg / max(nb, 1),
                   "L_A": tot_a / max(nb, 1), "val_recall10": vr, "skipped_positive_rows": skipped}
            if kind == "core":
                rec["gate_b"] = [float(x) for x in model.b.detach().cpu()]
                rec["mix"] = [float(x) for x in model.mix.detach().cpu()]
            history.append(rec)
            if log_every and ep % log_every == 0:
                print(f"      ep {ep}: loss {rec['loss']:.4f} val R@10 {vr:.4f}", flush=True)
            if vr > best + 1e-9:
                best, best_ep, bad = vr, ep, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        return FitResult(state=best_state, history=history, best_epoch=best_ep,
                         best_val_recall10=best, n_params=model.n_trainable(),
                         seconds=time.time() - t0, skipped_positive_rows=skipped_total,
                         counts=counts)

    def load(self, kind: str, core_cfg: CoreConfig | None, state: dict) -> nn.Module:
        m = self.build_model(kind, core_cfg)
        m.load_state_dict(state)
        m.eval()
        return m


def config_dict(core_cfg: CoreConfig | None, cfg: TrainConfig) -> dict:
    return {"core": asdict(core_cfg) if core_cfg else None, "train": asdict(cfg)}

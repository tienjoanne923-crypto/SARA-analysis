"""Disease-balanced episodic training with backpropagation through the solver.

Plan section 10.  One *epoch* equals as many tasks as there are training
diseases, not an arbitrary step count.  Each task draws a support size
k in {0,1,3,5} with the frozen mixture, exposes k support genes, and optimises
the query ranking loss after adaptation plus an auxiliary zero-shot term so the
base ranking is never sacrificed for few-shot behaviour.

    L_adapt = mean over (g in Q, u in V) log(1 + exp(-s(d,g|S) + s(d,u|S)))
    L       = E_d [ L_adapt + eta * L_0 ] + gamma ||theta||^2

Structured missing-information views (plan 10.3) are applied at the level of
the episode features: the target phenotype channel can be reduced, or the
reference-derived transfer can be reduced, but a "reduced" view always rebuilds
everything downstream of it rather than zeroing one column.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import adapt as adapt_mod
from .config import Gpu8Config
from .episodes import Episode, EpisodeBuilder
from .metrics import evaluate_predictions
from .model import Gpu8Model, PlainMLP
from .support import sample_training_support

TaskMix = (0, 1, 3, 5)


@dataclass
class TrainState:
    epoch: int = 0
    best_metric: float = -math.inf
    best_epoch: int = -1
    epochs_since_improve: int = 0
    history: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    n_tasks_seen: int = 0


def to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)


class EpisodeTensors:
    """Dense tensors for one episode; all shapes are small by construction."""

    def __init__(self, builder: EpisodeBuilder, ep: Episode, device,
                 include_candidates: bool = False):
        b = builder.b
        rows = [ep.support_rows, ep.anchor_rows, ep.query_pos_rows, ep.query_anchor_rows]
        if include_candidates:
            rows.append(ep.candidate_rows)
        all_rows = np.concatenate([r for r in rows if len(r)]) if any(len(r) for r in rows) \
            else np.array([], dtype=np.int64)
        uniq, inverse = np.unique(all_rows, return_inverse=True)
        self.uniq = uniq
        ch, mask, q = builder.pair_features(ep.disease_id, ep.d_row, uniq)
        dv, gv = builder.node_features(ep.d_row, uniq)
        self.disease_x = to_tensor(dv, device)
        self.gene_x = to_tensor(gv, device)
        self.channels = to_tensor(ch, device)
        self.mask = to_tensor(mask, device)
        self.quality = to_tensor(q, device)
        self.pos = {int(g): i for i, g in enumerate(uniq)}

    def slice_of(self, rows: np.ndarray) -> torch.Tensor:
        if len(rows) == 0:
            return torch.zeros(0, dtype=torch.long, device=self.disease_x.device)
        return torch.as_tensor([self.pos[int(r)] for r in rows], dtype=torch.long,
                               device=self.disease_x.device)


def encode_episode(model, tensors: EpisodeTensors):
    return model.encode(tensors.disease_x, tensors.gene_x, tensors.channels,
                        tensors.mask, tensors.quality)


def episode_loss(cfg: Gpu8Config, model, builder: EpisodeBuilder, ep: Episode,
                 device, equal_weight: bool = False,
                 use_controls: bool = True) -> tuple[torch.Tensor, dict]:
    t = EpisodeTensors(builder, ep, device)
    enc = encode_episode(model, t)
    base, phi = enc.base_score, enc.phi

    i_sup = t.slice_of(ep.support_rows)
    i_anc = t.slice_of(ep.anchor_rows)
    i_qp = t.slice_of(ep.query_pos_rows)
    i_qa = t.slice_of(ep.query_anchor_rows)

    if len(i_qp) == 0 or len(i_qa) == 0:
        return torch.zeros((), device=device, requires_grad=True), {"skipped": 1}

    summary = to_tensor(ep.support_summary, device)
    if use_controls and hasattr(model, "controls"):
        lam, rho = model.controls(summary)
    else:
        lam = torch.tensor(1.0, device=device)
        rho = torch.tensor(cfg.model.rho_max, device=device)

    if len(i_sup) and len(i_anc):
        w = to_tensor(ep.support_weights, device)
        res = adapt_mod.adapt(phi[i_sup], phi[i_anc], base[i_sup], base[i_anc],
                              w, lam, rho, b_clip=cfg.model.ridge_b_clip)
        a = res.a
    else:
        a = torch.zeros(cfg.model.adapt_dim, device=device)
        res = None

    s_all = adapt_mod.apply_adaptation(base, phi, a)
    diff = s_all[i_qp].unsqueeze(1) - s_all[i_qa].unsqueeze(0)
    l_adapt = F.softplus(-diff).mean()
    diff0 = base[i_qp].unsqueeze(1) - base[i_qa].unsqueeze(0)
    l_zero = F.softplus(-diff0).mean()
    loss = l_adapt + cfg.train.eta_zero_shot * l_zero

    info = {"l_adapt": float(l_adapt.detach()), "l_zero": float(l_zero.detach()),
            "k": ep.k, "n_query": int(len(i_qp)), "n_anchor": int(len(i_anc)),
            "a_norm": float(a.detach().norm()) if a.numel() else 0.0,
            "lam": float(lam.detach()) if torch.is_tensor(lam) else float(lam),
            "rho": float(rho.detach()) if torch.is_tensor(rho) else float(rho),
            "solver": res.solver if res else "skipped"}
    return loss, info


# --------------------------------------------------------------------------
# structured missing views (plan 10.3)
# --------------------------------------------------------------------------
def apply_missing_view(cfg: Gpu8Config, builder: EpisodeBuilder, disease_id: str,
                       view: int, rng: np.random.Generator) -> dict | None:
    """Return a patch to the cached retriever result implementing one view.

    view 0: allowed knowledge as-is.
    view 1: the target disease's own phenotype evidence is reduced, so the
            phenotype channel, its transfer and its propagation seed all shrink.
    view 2: reference associations are reduced by source/coverage, which
            rebuilds the transfer, hierarchy and propagation channels together.
    """
    if view == 0:
        return None
    r = builder.retriever_scores(disease_id, builder.b.idx.d_ix[disease_id])
    patch = {k: v.copy() if isinstance(v, np.ndarray) else dict(v) for k, v in r.items()}
    if view == 1:
        frac = float(rng.uniform(0.3, 0.8))
        patch["phenotype"] = patch["phenotype"] * frac
        patch["rwr"] = patch["rwr"] * frac
        patch["compat"] = patch["compat"] * frac
        patch["info"] = dict(r["info"])
        patch["info"]["hpo_count"] = int(r["info"].get("hpo_count", 0) * frac)
        patch["info"]["max_sim"] = float(r["info"].get("max_sim", 0.0) * frac)
    else:
        frac = float(rng.uniform(0.3, 0.8))
        keep = rng.random(patch["phenotype"].shape[0]) < frac
        patch["phenotype"] = patch["phenotype"] * keep
        patch["hierarchy"] = patch["hierarchy"] * keep
        patch["rwr"] = patch["rwr"] * keep
        patch["info"] = dict(r["info"])
        patch["info"]["n_ref_used"] = int(r["info"].get("n_ref_used", 0) * frac)
        patch["info"]["reference_source_complete"] = float(
            r["info"].get("reference_source_complete", 0.0) * frac)
    patch["fused"] = (cfg.candidates.fusion_weight_phenotype * _norm(patch["phenotype"])
                      + cfg.candidates.fusion_weight_hierarchy * _norm(patch["hierarchy"])
                      + getattr(cfg.candidates, "fusion_weight_rwr", 0.0)
                      * _norm(patch["rwr"]))
    return patch


def _norm(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x / m if m > 0 else x


# --------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------
def sample_task_k(cfg: Gpu8Config, rng: np.random.Generator, n_pos: int) -> int:
    mix = np.array(cfg.train.task_mix, dtype=float)
    ks = np.array(TaskMix, dtype=int)
    feasible = ks < n_pos            # need at least one query positive left
    if not feasible.any():
        return 0
    p = mix * feasible
    if p.sum() <= 0:
        return 0
    return int(rng.choice(ks, p=p / p.sum()))


def train_model(cfg: Gpu8Config, model, builders: dict[str, EpisodeBuilder],
                train_diseases: list[str], val_fn, device, seed: int,
                max_epochs: int | None = None, log_path: Path | None = None,
                equal_weight: bool = False, use_controls: bool = True,
                missing_views: bool = True, progress: bool = True) -> TrainState:
    """builders maps inner context id -> EpisodeBuilder (three per outer fold)."""
    tc = cfg.train
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    state = TrainState()
    ctx_ids = sorted(builders)
    block_of: dict[str, str] = {}
    for cid, b in builders.items():
        for did in getattr(b, "block_diseases", []):
            block_of[did] = cid

    max_epochs = int(max_epochs or tc.max_epochs)
    accum_target = max(1, tc.accumulate_to_tasks // max(1, tc.tasks_per_microbatch))

    for epoch in range(max_epochs):
        model.train()
        t0 = time.perf_counter()
        order = rng.permutation(len(train_diseases))
        losses, infos = [], []
        opt.zero_grad(set_to_none=True)
        micro = 0
        for count, di in enumerate(order):
            did = train_diseases[di]
            cid = block_of.get(did, ctx_ids[count % len(ctx_ids)])
            builder = builders[cid]
            pos = builder.positives.get(did, [])
            if len(pos) < 1:
                continue
            # a single-positive disease is a legitimate zero-shot task: k = 0,
            # empty support, that one positive as the query.  Seven in ten
            # diseases look like this, so skipping them would throw away most
            # of the low-resource training signal.
            k = sample_task_k(cfg, rng, len(pos))
            support, query = sample_training_support(cfg, did, pos, k, rng)
            view = int(rng.choice([0, 1, 2], p=np.array(tc.missing_view_probs))) \
                if missing_views else 0
            patch = apply_missing_view(cfg, builder, did, view, rng) if view else None
            if patch is not None:
                builder._retriever_cache[did] = patch
            ep = builder.build_episode(did, support, query,
                                       max_query_pos=tc.max_query_pos_per_task,
                                       max_query_anchor=tc.max_query_anchor_per_task,
                                       rng=rng, equal_weight=equal_weight,
                                       with_candidates=False)
            if patch is not None:
                builder._retriever_cache.pop(did, None)
            if ep is None:
                continue
            loss, info = episode_loss(cfg, model, builder, ep, device,
                                      equal_weight=equal_weight, use_controls=use_controls)
            if info.get("skipped"):
                continue
            (loss / float(tc.tasks_per_microbatch)).backward()
            losses.append(float(loss.detach()))
            infos.append(info)
            state.n_tasks_seen += 1
            micro += 1
            if micro % tc.tasks_per_microbatch == 0:
                if micro % (tc.tasks_per_microbatch * accum_target) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
        if micro % (tc.tasks_per_microbatch * accum_target) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        val = val_fn(model)
        metric = float(val.get(tc.select_metric, float("nan")))
        tie = float(val.get(tc.tie_break_metric, float("nan")))
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else math.nan,
               "n_tasks": micro, "seconds": time.perf_counter() - t0,
               "val_" + tc.select_metric: metric,
               "val_" + tc.tie_break_metric: tie,
               "mean_a_norm": float(np.mean([i["a_norm"] for i in infos])) if infos else 0.0,
               "mean_lambda": float(np.mean([i["lam"] for i in infos])) if infos else 0.0,
               "mean_rho": float(np.mean([i["rho"] for i in infos])) if infos else 0.0}
        state.history.append(rec)
        if progress:
            print(f"    epoch {epoch:3d} loss={rec['train_loss']:.4f} "
                  f"val_{tc.select_metric}={metric:.4f} ({rec['seconds']:.1f}s)", flush=True)
        improved = (not math.isnan(metric)) and metric > state.best_metric + 1e-6
        if improved:
            state.best_metric, state.best_epoch = metric, epoch
            state.epochs_since_improve = 0
            state.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state.epochs_since_improve += 1
        state.epoch = epoch
        if log_path:
            Path(log_path).write_text(json.dumps(state.history, indent=2), encoding="utf-8")
        if state.epochs_since_improve >= tc.patience:
            state.stop_reason = f"patience_{tc.patience}"
            break
    if not state.stop_reason:
        state.stop_reason = "max_epochs"
    if getattr(state, "best_state", None) is not None:
        model.load_state_dict(state.best_state)
    return state


# --------------------------------------------------------------------------
# scoring (no grad, chunked)
# --------------------------------------------------------------------------
@torch.no_grad()
def score_disease(cfg: Gpu8Config, model, builder: EpisodeBuilder, disease_id: str,
                  support_genes: list[str], candidate_rows: np.ndarray, device,
                  equal_weight: bool = False, use_controls: bool = True,
                  disable_adaptation: bool = False) -> tuple[np.ndarray, dict]:
    """Score a frozen candidate list; batching never changes a pair's score."""
    idx = builder.b.idx
    d_row = idx.d_ix[disease_id]
    g_ix = idx.g_ix
    support_rows = np.array([g_ix[g] for g in support_genes if g in g_ix], dtype=np.int64)
    r = builder.retriever_scores(disease_id, d_row)
    known = {g_ix[g] for g in builder.positives.get(disease_id, []) if g in g_ix}
    from .candidates import sample_anchors
    anchors = sample_anchors(cfg, disease_id, builder.b.background_rows,
                             set(support_rows.tolist()), known, fused_scores=r["fused"])

    a = torch.zeros(cfg.model.adapt_dim, device=device)
    info: dict = {"k": len(support_rows), "n_anchor": int(len(anchors))}
    if len(support_rows) and len(anchors) and not disable_adaptation:
        rows = np.unique(np.concatenate([support_rows, anchors]))
        ch, mask, q = builder.pair_features(disease_id, d_row, rows)
        dv, gv = builder.node_features(d_row, rows)
        enc = model.encode(to_tensor(dv, device), to_tensor(gv, device),
                           to_tensor(ch, device), to_tensor(mask, device),
                           to_tensor(q, device))
        pos = {int(g): i for i, g in enumerate(rows)}
        i_s = torch.as_tensor([pos[int(x)] for x in support_rows], dtype=torch.long, device=device)
        i_a = torch.as_tensor([pos[int(x)] for x in anchors], dtype=torch.long, device=device)
        w, summary, groups = builder.support_bundle(disease_id, list(support_genes),
                                                    equal_weight=equal_weight)
        st = to_tensor(summary, device)
        if use_controls and hasattr(model, "controls"):
            lam, rho = model.controls(st)
        else:
            lam = torch.tensor(1.0, device=device)
            rho = torch.tensor(cfg.model.rho_max, device=device)
        res = adapt_mod.adapt(enc.phi[i_s], enc.phi[i_a], enc.base_score[i_s],
                              enc.base_score[i_a], to_tensor(w, device), lam, rho,
                              b_clip=cfg.model.ridge_b_clip)
        a = res.a
        info.update({"a_norm": float(a.norm()), "lam": float(lam), "rho": float(rho),
                     "solver": res.solver, "n_groups": groups.n_groups,
                     "n_effective": groups.n_effective})

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
    return scores, info


@torch.no_grad()
def evaluate_diseases(cfg: Gpu8Config, model, builder: EpisodeBuilder,
                      tasks: list[dict], device, ks=(10, 20, 50),
                      equal_weight: bool = False,
                      disable_adaptation: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """tasks: [{disease_id, support_genes, query_genes, candidate_rows}]."""
    model.eval()
    preds = []
    for t in tasks:
        did = t["disease_id"]
        cand_rows = t["candidate_rows"]
        if len(cand_rows) == 0:
            continue
        scores, info = score_disease(cfg, model, builder, did, t["support_genes"],
                                     cand_rows, device, equal_weight=equal_weight,
                                     disable_adaptation=disable_adaptation)
        g_ix = builder.b.idx.g_ix
        qrows = {g_ix[g] for g in t["query_genes"] if g in g_ix}
        labels = np.array([1.0 if int(g) in qrows else 0.0 for g in cand_rows], dtype=np.float32)
        preds.append(pd.DataFrame({"disease_id": did, "gene_row": cand_rows,
                                   "score": scores, "label": labels}))
    if not preds:
        return pd.DataFrame(), {}, pd.DataFrame()
    pred = pd.concat(preds, ignore_index=True)
    totals = {t["disease_id"]: len(t["query_genes"]) for t in tasks}
    per, macro = evaluate_predictions(pred, ks=ks, query_totals=totals)
    return per, macro, pred

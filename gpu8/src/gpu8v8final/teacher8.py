"""V8.1 §5 — the semantic teacher (= the reproduced V7.1 core).

Architecture and training loss are exactly the V7.1 core (rank 128, shared
semantic projection P_T, global gate, lambda_A = 0.2, 159,749 trainable
parameters; L_DG + lambda_A L_A).  What changes in V8.1:

* the main semantic score is s_F(d, g) = z_d^T z^F_g with
  z_d = norm(P_T mix(x^text_d, x^HPO_d)) and z^F_g = norm(P_T x^GO_g);
* the checkpoint is chosen by the *s_F* full-background Recall@10 on F_{j+1}
  (inner and outer contexts alike; the loss is unchanged);
* the unlabelled-control set {reference genes, whole background} is chosen per
  outer context on the development families (s_F Recall@10);
* disease text is the segmented encoding;
* every set is role-based (no label table is used to build an exclusion).

The teacher's own sequence score s_S is kept only as an extra imputation control.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from gpu8v6.common import ContextSpec, World6
from gpu8v71.core71 import CoreConfig, row_norm
from gpu8v71.train71 import D_KEYS, CoreTrainer, TrainConfig, take

from . import paths8
from .common8 import dump_json, list_hash, now_utc
from .ledger8 import is_done, run_dir, tracked
from .roles8 import Roles, adjacency

CORE = CoreConfig(rank=128, gate="global")
TRAIN = TrainConfig(lambda_A=0.2)
N_PARAMS = 159_749


@dataclass
class TeacherData:
    ctx_id: str
    fold: int
    kind: str
    reference: list
    train: dict           # disease row -> positive gene rows (training labels)
    controls: np.ndarray  # genes in the visible reference relations
    val: dict             # F_{j+1} disease row -> query rows (checkpoint only)
    fit_disease_hash: str
    dropped_no_semantic: int = 0


def teacher_data(w: World6, roles: Roles, ctx_id: str, drop_relations: set | None = None,
                 drop_genes: np.ndarray | None = None, valid_rows: np.ndarray | None = None) -> TeacherData:
    """``valid_rows`` (disease mask of legal text or HPO input): a training disease without any
    semantic input carries no gradient signal and only produces an empty control set (whose
    log-sum-exp gradient is undefined), so it is left out of the teacher's training set and counted."""
    ctx = w.contexts[ctx_id]
    ref = roles.reference(ctx)
    A = adjacency(w, ref, drop_relations, drop_genes)
    controls = np.asarray(A.sum(axis=0)).ravel() > 0
    d_ix = w.idx.d_ix
    train = {}
    dropped = 0
    for d in ref:
        if d not in d_ix:
            continue
        r = d_ix[d]
        g = A[r].indices
        if len(g):
            if valid_rows is not None and not valid_rows[r]:
                dropped += 1
                continue
            train[r] = np.sort(g.astype(np.int64))
    j = ctx.fold
    val = {}
    for d in sorted(roles.folds[(j + 1) % 5]):
        q = w.query_rows(d)
        if len(q) and d in d_ix:
            val[d_ix[d]] = q
    return TeacherData(ctx_id, j, ctx.kind, ref, train, controls, val,
                       list_hash(sorted(w.idx.diseases[r] for r in train)), dropped)


# ---------------------------------------------------------------------------
@torch.no_grad()
def disease_z(model, T: dict, rows: np.ndarray, chunk: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    zs, oks = [], []
    for b0 in range(0, len(rows), chunk):
        idx = torch.as_tensor(rows[b0:b0 + chunk], device=T["xT"].device)
        z, ok = model.disease_repr(T["xT"][idx], T["mT"][idx], T["xH"][idx], T["mH"][idx])
        zs.append(z)
        oks.append(ok)
    return torch.cat(zs), torch.cat(oks)


@torch.no_grad()
def gene_zF(model, T: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """z^F_g = norm(P_T x^GO_g) for genes with legal GO (zero elsewhere)."""
    z = row_norm(model.P_T(T["xF"])) * T["mF"][:, None].to(T["xF"].dtype)
    return z, T["mF"].clone()


@torch.no_grad()
def gene_zS(model, T: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The teacher's own sequence representation (imputation control only)."""
    z = row_norm(model.P_S(T["xS"])) * T["mS"][:, None].to(T["xS"].dtype)
    return z, T["mS"].clone()


@torch.no_grad()
def topk_recall(z: torch.Tensor, ok: torch.Tensor, zg: torch.Tensor, mg: torch.Tensor,
                rows: np.ndarray, val: dict, k: int = 10) -> np.ndarray:
    """Per-disease full-background Recall@k of z^T zg over the valid genes."""
    out = np.zeros(len(rows))
    for b0 in range(0, len(rows), 256):
        s = z[b0:b0 + 256] @ zg.T
        valid = ok[b0:b0 + 256, None] & mg[None, :]
        s = torch.where(valid, s, torch.full_like(s, -1e9))
        top = torch.topk(s, k, dim=1)
        ti = top.indices.cpu().numpy()
        tv = torch.gather(valid, 1, top.indices).cpu().numpy()
        for i, r in enumerate(rows[b0:b0 + 256]):
            q = val[int(r)]
            out[b0 + i] = sum(1 for g, v in zip(ti[i], tv[i]) if v and g in q) / max(len(q), 1)
    return out


class TeacherTrainer(CoreTrainer):
    """CoreTrainer whose checkpoint metric is the s_F full-background Recall@10."""

    def __init__(self, T: dict, device: str = "cuda", metric: str = "sF"):
        super().__init__(T, device)
        self.metric = metric
        self.aux: list[dict] = []

    @torch.no_grad()
    def recall10(self, model, val: dict) -> float:
        rows = np.array(sorted(val))
        if len(rows) == 0:
            return 0.0
        model.eval()
        z, ok = disease_z(model, self.T, rows)
        zF, mF = gene_zF(model, self.T)
        rF = float(topk_recall(z, ok, zF, mF, rows, val).mean())
        rR = float(CoreTrainer.recall10(self, model, val))
        self.aux.append({"sF_recall10": rF, "r_recall10": rR})
        return rF if self.metric == "sF" else rR


def fit_teacher(T: dict, data: TeacherData, seed: int, controls: str = "reference",
                excluded_genes: np.ndarray | None = None, metric: str = "sF",
                pack=None, device: str = "cuda"):
    tr = TeacherTrainer(T, device, metric=metric)
    cfg = TrainConfig(**{**asdict(TRAIN), "controls": controls})
    anchors = T["mS"].cpu().numpy() & (T["A_plus"].any(dim=1).cpu().numpy())
    res = tr.fit(data.train, data.controls, data.val, seed=seed, kind="core", core_cfg=CORE, cfg=cfg,
                 anchor_genes=anchors, excluded_training_genes=excluded_genes)
    ep = [h for h in res.history if h["stage"] == "joint"]
    for h, a in zip(ep, tr.aux):
        h.update(a)
    return res


def teacher_run_id(arm: str, ctx: str, controls: str, seed: int) -> str:
    return f"V8T__{arm}__{ctx}__{controls}__s{seed}"


def run_teacher(run_id: str, T: dict, data: TeacherData, seed: int, controls: str, role: str,
                arm_key: str, pack_hash: str, excluded_genes=None, meta_extra: dict | None = None):
    if is_done(run_id):
        return json.loads((run_dir(run_id) / "result.json").read_text(encoding="utf-8"))
    meta = {"role": role, "stage": "teacher_fit", "component": "teacher", "arm": arm_key,
            "context_id": data.ctx_id, "fold": data.fold, "seed": seed,
            "config": {"core": asdict(CORE), "train": {**asdict(TRAIN), "controls": controls},
                       "checkpoint": "s_F full-background Recall@10 on F_{j+1}"},
            "pack_hash": pack_hash, "fit_disease_hash": data.fit_disease_hash, **(meta_extra or {})}
    with tracked(run_id, meta) as box:
        res = fit_teacher(T, data, seed, controls, excluded_genes)
        torch.save(res.state, run_dir(run_id) / "model.pt")
        dump_json(run_dir(run_id) / "history.json", res.history)
        box["extra"] = {"fit": {"best_epoch": res.best_epoch, "best_val_sF_recall10": res.best_val_recall10,
                                "n_params": res.n_params, "seconds": res.seconds,
                                "skipped_positive_rows": res.skipped_positive_rows,
                                "n_reference": len(data.reference), **res.counts,
                                "dropped_training_diseases_without_text_or_hpo": data.dropped_no_semantic,
                                "epochs_run": sum(1 for h in res.history if h["stage"] == "joint")}}
        if res.n_params != N_PARAMS:
            raise RuntimeError(f"teacher has {res.n_params} parameters, expected {N_PARAMS}")
    return json.loads((run_dir(run_id) / "result.json").read_text(encoding="utf-8"))


def load_teacher(run_id: str, T: dict, device: str = "cuda"):
    tr = CoreTrainer(T, device)
    state = torch.load(run_dir(run_id) / "model.pt", map_location=device, weights_only=True)
    return tr.load("core", CORE, state)


# ---------------------------------------------------------------------------
# V7.1 reproduction and model_spec.json (§5.1)
# ---------------------------------------------------------------------------
def reproduce_v71(device: str = "cuda") -> dict:
    """Re-fit V71CHAIN__core__f0_outer__s42 with the archived V7.1 inputs and compare."""
    from gpu8v71.chain71 import Contexts
    from gpu8v71.featurepack71 import FeaturePack
    w = World6(load_heavy=False)
    pack = FeaturePack()
    T = pack.tensors(device, expression="hpa")
    T["A_plus"] = torch.as_tensor(pack.A.toarray().astype(bool), device=device)
    cx = Contexts(w)
    train, controls, ref = cx.training("f0_outer")
    val = cx.early_stop_set("f0_outer")
    anchors = pack.a["g_sequence_mask"] & (np.diff(pack.A.indptr) > 0)
    t0 = time.time()
    res = CoreTrainer(T, device).fit(train, controls, val, seed=42, kind="core", core_cfg=CORE,
                                     cfg=TRAIN, anchor_genes=anchors)
    arch = paths8.V71_RUNS / "V71CHAIN__core__f0_outer__s42"
    old = torch.load(arch / "model.pt", map_location="cpu", weights_only=True)
    hist_old = json.loads((arch / "history.json").read_text(encoding="utf-8"))
    diffs = {k: float((res.state[k].float() - old[k].float()).abs().max()) for k in old}
    ho = [h["val_recall10"] for h in hist_old if h["stage"] == "joint"]
    hn = [h["val_recall10"] for h in res.history if h["stage"] == "joint"]
    n = min(len(ho), len(hn))
    rep = {"generated_utc": now_utc(), "run": "V71CHAIN__core__f0_outer__s42",
           "best_epoch_archived": int(json.loads((arch / "result.json").read_text(encoding="utf-8"))["fit"]["best_epoch"]),
           "best_epoch_refit": res.best_epoch, "best_val_archived": max(ho), "best_val_refit": res.best_val_recall10,
           "history_maxabs_diff": float(np.max(np.abs(np.array(ho[:n]) - np.array(hn[:n])))) if n else None,
           "epochs_archived": len(ho), "epochs_refit": len(hn),
           "weights_maxabs_diff": diffs, "n_params": res.n_params, "seconds": time.time() - t0,
           "tolerance": {"weights_maxabs": 1e-3, "history_maxabs": 0.005},
           "note": "GPU kernels are not bit-deterministic; differences within tolerance count as reproduced"}
    rep["reproduced"] = bool(res.n_params == N_PARAMS and rep["best_epoch_refit"] == rep["best_epoch_archived"]
                             and max(diffs.values()) <= 1e-3 and (rep["history_maxabs_diff"] or 0) <= 0.005)
    spec = {"generated_utc": now_utc(), "source": "gpu8v71.core71.TypedCore / gpu8v71.train71.CoreTrainer",
            "archived_run": "runs_v71/V71CHAIN__core__f0_outer__s42",
            "core": asdict(CORE), "train": asdict(TRAIN), "n_trainable_parameters": res.n_params,
            "parameter_breakdown": {"P_T": 768 * 128, "P_S": 480 * 128, "mix": 2, "gate_b": 3},
            "inputs": {"d_text": 768, "d_hpo": 768, "g_sequence": 480, "g_function": 768,
                       "g_expression": 51, "d_tissue": 51, "anchors": "SapBERT GO definitions"},
            "modalities": CORE.modalities(), "gate": "global (u fixed at 0)",
            "featurepack_v71_hash": json.loads((paths8.V71_MANIFESTS / "featurepack_v71.json")
                                               .read_text(encoding="utf-8"))["payload_hash"],
            "selection": json.loads((paths8.V71_MANIFESTS / "core_dev_selection_v71.json")
                                    .read_text(encoding="utf-8"))["selected"],
            "reproduction": rep}
    dump_json(paths8.MANIFESTS / "model_spec.json", spec)
    return rep


if __name__ == "__main__":
    paths8.ensure()
    print(json.dumps(reproduce_v71(), indent=2))

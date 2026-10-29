"""V7.1 sections 11.3 and 12.2 — development selection and the OOF retrieval chain.

Contexts are the V5/V6 ones.  For a context the core trains on the labelled
V6 reference diseases (V5 reference minus strict equivalents of held-out
labelled diseases) with their visible relations as positives; unlabelled
controls are genes that appear in those reference relations.

* ``dev``    fold 0 only: 8 configurations (rank 64/128 x lambda_A 0/0.2 x
             global/small gate), trained in ``f0_outer``, early-stopped on F1,
             scored on the development families (full background).  The one-SE
             rule picks the simplest configuration; a paired permutation check
             on dev decides which modalities are kept; the admission check asks
             whether the core adds development positives that the frozen V6
             500-list misses.
* ``chain``  per fold j and seed: three inner fits (inner block held out,
             early stopping on the development families) and one outer fit
             (early stopping on F_{j+1}); 5 folds x 3 seeds = 60 chain fits for
             the main core, plus seed-42 chains of the E2/E3/E4 variants and
             outer-only fits of the E8 / sensitivity variants.

Models are tiny, so each fit saves its weights and every later stage re-scores
deterministically instead of caching 20,000-gene score matrices.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch

from gpu8v6.common import ContextSpec, World6

from . import paths71
from .common71 import dump_json, load_manifest, now_utc, write_manifest
from .core71 import CoreConfig
from .featurepack71 import FeaturePack
from .ledger71 import is_done, run_dir, tracked
from .train71 import CoreTrainer, TrainConfig

SEEDS = (42, 52, 62)
DEV_GRID = [{"rank": r, "lambda_A": la, "gate": g} for r in (64, 128) for la in (0.0, 0.2)
            for g in ("global", "small")]


def complexity(c: dict) -> tuple:
    """Simpler first: rank, then gate (global < small), then lambda_A (0 < 0.2)."""
    return (c["rank"], 0 if c["gate"] == "global" else 1, c["lambda_A"])


# --------------------------------------------------------------------------
# context data
# --------------------------------------------------------------------------
class Contexts:
    def __init__(self, w: World6):
        self.w = w
        self.eq = dict(zip(w.diseases["disease_id"], w.diseases["equivalence_group"]))
        self.d_ix = w.idx.d_ix

    def v6_reference(self, ctx: ContextSpec) -> list[str]:
        ref = set(ctx.reference)
        held = {self.eq.get(d, d) for d in self.w.positives if d not in ref}
        return [d for d in ctx.reference if self.eq.get(d, d) not in held]

    def spec(self, ctx_id: str) -> ContextSpec:
        return self.w.contexts[ctx_id]

    def training(self, ctx_id: str) -> tuple[dict, np.ndarray, list[str]]:
        ctx = self.spec(ctx_id)
        ref = self.v6_reference(ctx)
        cs = ContextSpec(ctx.context_id, ctx.fold, ctx.kind, ctx.block_index, ref, ctx.held_out)
        A = self.w.adjacency(cs)
        controls = np.asarray(A.sum(axis=0)).ravel() > 0
        lab = self.w.labelled(ref)
        train = {self.d_ix[d]: self.w.query_rows(d) for d in lab}
        train = {k: v for k, v in train.items() if len(v)}
        return train, controls, ref

    def labels(self, ids: list[str]) -> dict[int, np.ndarray]:
        out = {}
        for d in self.w.labelled(ids):
            q = self.w.query_rows(d)
            if len(q):
                out[self.d_ix[d]] = q
        return out

    def targets(self, ctx_id: str) -> dict[str, list[str]]:
        ctx = self.spec(ctx_id)
        if ctx.kind == "inner":
            return {"oof": self.w.labelled(ctx.held_out)}
        fa = self.w.fold(ctx.fold)
        return {"test": self.w.labelled(fa["test_diseases"]),
                "val": self.w.labelled(fa["val_diseases"]),
                "dev": self.w.labelled(self.w.spec.dev_diseases)}

    def early_stop_set(self, ctx_id: str) -> dict[int, np.ndarray]:
        ctx = self.spec(ctx_id)
        if ctx.kind == "inner":
            return self.labels(self.w.spec.dev_diseases)
        return self.labels(self.w.fold(ctx.fold)["val_diseases"])


# --------------------------------------------------------------------------
# variants
# --------------------------------------------------------------------------
def selected() -> dict:
    sel = load_manifest("core_dev_selection_v71.json")
    if not sel:
        raise RuntimeError("run the development selection first")
    return sel


def variant_spec(name: str) -> dict:
    sel = selected()
    c = sel["selected"]
    mods = sel.get("modalities", {"S": True, "F": True, "E": True})
    base_core = CoreConfig(rank=c["rank"], gate=c["gate"], use_sequence=mods.get("S", True),
                           use_function=mods.get("F", True), use_expression=mods.get("E", True))
    base_train = TrainConfig(lambda_A=c["lambda_A"] if base_core.use_sequence else 0.0)
    spec = {"kind": "core", "core": base_core, "train": base_train, "expression": "hpa",
            "anchor_rule": "all", "outer_only": False, "image": False, "mouse": False}
    if name == "core":
        return spec
    if name == "core_altanchor":
        return {**spec, "train": replace(base_train, lambda_A=0.2 if base_train.lambda_A == 0 else 0.0)}
    if name == "two_tower":
        return {**spec, "kind": "two_tower", "train": replace(base_train, lambda_A=0.0)}
    if name == "core_oldres":
        return {**spec, "core": replace(base_core, use_sequence=False),
                "train": replace(base_train, lambda_A=0.0), "expression": "gtex"}
    if name == "core_strict":
        return {**spec, "anchor_rule": "association_warm_only"}
    if name == "core_bgctrl":
        return {**spec, "train": replace(base_train, controls="background"), "outer_only": True}
    if name == "core_image":
        return {**spec, "core": replace(base_core, image=True), "image": True, "outer_only": True}
    if name == "core_imgloc":        # localisation-label-only control through the image adapter
        return {**spec, "core": replace(base_core, image=True), "image": "loc", "outer_only": True}
    if name == "core_imgcentred":    # cell-line-centred image features
        return {**spec, "core": replace(base_core, image=True), "image": "centred", "outer_only": True}
    if name == "core_mouse":
        return {**spec, "core": replace(base_core, mouse=True), "mouse": True, "outer_only": True}
    raise KeyError(name)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
class Runner:
    def __init__(self, device: str = "cuda"):
        self.w = World6(load_heavy=False)
        self.cx = Contexts(self.w)
        self.pack = FeaturePack()
        self.device = device
        self._T = {}

    def tensors(self, expression: str = "hpa", image=False) -> dict:
        key = (expression, image)
        if key not in self._T:
            self._T.clear()
            torch.cuda.empty_cache()
            T = self.pack.tensors(self.device, expression=expression, image=bool(image))
            T["A_plus"] = torch.as_tensor(self.pack.A.toarray().astype(bool), device=self.device)
            if image == "loc":
                L = np.load(paths71.TABLES / "g_loc.npy")
                X = np.zeros((L.shape[0], 1024), dtype=np.float32)
                X[:, :L.shape[1]] = L
                T["xI"] = torch.as_tensor(X, device=self.device)
                T["mI"] = torch.as_tensor(np.load(paths71.TABLES / "g_loc_mask.npy"), device=self.device)
            elif image == "centred":
                X = self.pack.a["g_image"].copy()
                m = self.pack.a["g_image_mask"]
                cl = pd.read_parquet(paths71.TABLES / "g_image_cell_line.parquet")["dominant_cell_line"]
                for c in cl[m].unique():
                    sel = m & (cl == c).to_numpy()
                    X[sel] -= X[sel].mean(axis=0)
                T["xI"] = torch.as_tensor(X, device=self.device)
            self._T[key] = T
        return self._T[key]

    def anchor_genes(self, rule: str, controls: np.ndarray) -> np.ndarray:
        elig = self.pack.a["g_sequence_mask"] & (np.diff(self.pack.A.indptr) > 0)
        if rule == "association_warm_only":
            elig = elig & controls
        return elig

    def fit(self, run_id: str, variant: str, ctx_id: str, seed: int, spec: dict,
            stop_set: dict | None = None, extra_meta: dict | None = None,
            extra_d_full: dict | None = None):
        T = self.tensors(spec["expression"], spec["image"])
        if spec.get("mouse"):
            from .mouse71 import attach_mouse
            T = attach_mouse(T, self.device)
        tr = CoreTrainer(T, self.device)
        train, controls, ref = self.cx.training(ctx_id)
        val = stop_set if stop_set is not None else self.cx.early_stop_set(ctx_id)
        anchors = self.anchor_genes(spec["anchor_rule"], controls)
        meta = {"stage": "core_fit", "method": spec["kind"], "variant": variant,
                "context_id": ctx_id, "fold": self.cx.spec(ctx_id).fold, "seed": seed,
                "config": {"core": asdict(spec["core"]), "train": asdict(spec["train"]),
                           "expression": spec["expression"], "anchor_rule": spec["anchor_rule"]},
                **(extra_meta or {})}
        with tracked(run_id, meta, device=self.device) as box:
            res = tr.fit(train, controls, val, seed=seed, kind=spec["kind"], core_cfg=spec["core"],
                         cfg=spec["train"], anchor_genes=anchors)
            torch.save(res.state, run_dir(run_id) / "model.pt")
            dump_json(run_dir(run_id) / "history.json", res.history)
            box["extra"] = {"fit": {"best_epoch": res.best_epoch,
                                    "best_val_recall10": res.best_val_recall10,
                                    "n_params": res.n_params, "seconds": res.seconds,
                                    "skipped_positive_rows": res.skipped_positive_rows,
                                    "n_reference_v6": len(ref), **res.counts,
                                    "epochs_run": sum(1 for h in res.history if h["stage"] == "joint")}}
        return res

    def model(self, run_id: str, spec: dict):
        T = self.tensors(spec["expression"], spec["image"])
        if spec.get("mouse"):
            from .mouse71 import attach_mouse
            T = attach_mouse(T, self.device)
        tr = CoreTrainer(T, self.device)
        state = torch.load(run_dir(run_id) / "model.pt", map_location=self.device)
        return tr, tr.load(spec["kind"], spec["core"], state)


def core_list(R: np.ndarray, V: np.ndarray, prior: np.ndarray, n: int = 500,
              fill: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Top-n valid genes by r (ties: gene row); optional prior fill. Returns (genes, is_fill)."""
    D, G = R.shape
    genes = np.full((D, n), -1, dtype=np.int64)
    isfill = np.zeros((D, n), dtype=bool)
    pri_order = np.lexsort((np.arange(G), -prior))
    for i in range(D):
        v = np.flatnonzero(V[i])
        order = v[np.lexsort((v, -R[i, v].astype(np.float64)))][:n]
        k = len(order)
        genes[i, :k] = order
        if fill and k < n:
            used = np.zeros(G, dtype=bool)
            used[order] = True
            rest = [g for g in pri_order if not used[g]][:n - k]
            genes[i, k:k + len(rest)] = rest
            isfill[i, k:] = True
    return genes, isfill


def dev_recall_per_disease(R, V, rows, labels) -> np.ndarray:
    out = []
    for i, r in enumerate(rows):
        v = np.flatnonzero(V[i])
        top = v[np.lexsort((v, -R[i, v].astype(np.float64)))][:10]
        q = set(labels[int(r)].tolist())
        out.append(sum(1 for g in top if int(g) in q) / max(len(q), 1))
    return np.array(out)


# --------------------------------------------------------------------------
# development selection (fold 0)
# --------------------------------------------------------------------------
def dev_select(device: str = "cuda") -> dict:
    run = Runner(device)
    cx = run.cx
    dev_lab = cx.labels(run.w.spec.dev_diseases)
    dev_rows = np.array(sorted(dev_lab))
    per, rows = {}, []
    for c in DEV_GRID:
        name = f"r{c['rank']}_lam{c['lambda_A']}_{c['gate']}"
        run_id = f"V71DEV__{name}__f0__s42"
        spec = {"kind": "core", "core": CoreConfig(rank=c["rank"], gate=c["gate"]),
                "train": TrainConfig(lambda_A=c["lambda_A"]), "expression": "hpa",
                "anchor_rule": "all", "image": False, "mouse": False}
        if not is_done(run_id):
            run.fit(run_id, "dev_" + name, "f0_outer", 42, spec,
                    extra_meta={"stage": "dev_fit", "evaluated_on": "development families"})
        tr, m = run.model(run_id, spec)
        R, V = tr.score(m, dev_rows)
        rec = dev_recall_per_disease(R, V, dev_rows, dev_lab)
        per[name] = rec
        rows.append({**c, "name": name, "dev_recall10": float(rec.mean()),
                     "dev_valid_share": float(V.any(axis=1).mean())})
        print(f"  dev {name}: {rec.mean():.4f}", flush=True)
    df = pd.DataFrame(rows)
    best = df.sort_values("dev_recall10", ascending=False).iloc[0]
    within = []
    for r in df.itertuples(index=False):
        diff = per[best["name"]] - per[r.name]
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if r.name != best["name"] else 0.0
        ok = diff.mean() <= se + 1e-12
        within.append(ok)
    df["within_one_se_of_best"] = within
    cand = [dict(rank=int(r.rank), lambda_A=float(r.lambda_A), gate=r.gate)
            for r in df.itertuples(index=False) if r.within_one_se_of_best]
    pick = sorted(cand, key=complexity)[0]
    df.to_csv(paths71.MANIFESTS / "core_dev_registry_v71.csv", index=False, encoding="utf-8-sig")
    name = f"r{pick['rank']}_lam{pick['lambda_A']}_{pick['gate']}"
    sel_run = f"V71DEV__{name}__f0__s42"

    # ---- modality retention on dev: paired permutation of each modality ----
    from .behavior71 import dev_modality_check
    mod = dev_modality_check(run, sel_run, pick, dev_rows, dev_lab)

    # ---- admission: does the core add dev positives outside the V6 500? ----
    from gpu8v6.scoring import load_block
    blk = load_block("f0_outer", ids=[run.w.idx.diseases[r] for r in dev_rows])
    spec = {"kind": "core", "core": CoreConfig(rank=pick["rank"], gate=pick["gate"]),
            "train": TrainConfig(lambda_A=pick["lambda_A"]), "expression": "hpa",
            "anchor_rule": "all", "image": False, "mouse": False}
    tr, m = run.model(sel_run, spec)
    rows_b = np.array([run.w.idx.d_ix[d] for d in blk.ids])
    R, V = tr.score(m, rows_b)
    g500, _ = core_list(R, V, np.zeros(run.w.n_g), 500, fill=False)
    new_pos, v6_pos, n_q = 0, 0, 0
    for i, d in enumerate(blk.ids):
        q = set(dev_lab[rows_b[i]].tolist())
        v6 = set(blk.genes[i].tolist())
        core = set(int(x) for x in g500[i] if x >= 0)
        new_pos += len((core - v6) & q)
        v6_pos += len(v6 & q)
        n_q += len(q)
    adm = {"dev_positives": n_q, "in_v6_500": v6_pos, "added_by_core_top500": new_pos,
           "union_coverage_gain": new_pos / max(n_q, 1), "admitted": bool(new_pos > 0)}
    out = {"grid": df.to_dict(orient="records"), "best": best["name"],
           "selected": pick, "selected_run": sel_run,
           "rule": "one-SE (paired per-disease SE vs the best) then simplest: rank, gate, lambda_A",
           "modalities": mod["keep"], "modality_check": mod, "admission": adm,
           "evaluated_on": "development families (864), full background, fit f0_outer, early stop F1"}
    write_manifest("core_dev_selection_v71.json", out)
    return out


# --------------------------------------------------------------------------
# formal chain
# --------------------------------------------------------------------------
def chain(variants: list[str], seeds=SEEDS, folds=(0, 1, 2, 3, 4), device: str = "cuda") -> None:
    if not load_manifest("protocol_v71_registered.json"):
        raise RuntimeError("register the V7.1 protocol before the formal chain")
    run = Runner(device)
    for v in variants:
        spec = variant_spec(v)
        for seed in seeds:
            for j in folds:
                ctxs = [f"f{j}_outer"] if spec["outer_only"] else \
                    [c.context_id for c in run.w.inner_contexts(j)] + [f"f{j}_outer"]
                for ctx in ctxs:
                    run_id = f"V71CHAIN__{v}__{ctx}__s{seed}"
                    if is_done(run_id):
                        continue
                    t0 = time.time()
                    res = run.fit(run_id, v, ctx, seed, spec)
                    print(f"  {run_id}: best ep {res.best_epoch} val {res.best_val_recall10:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V7.1 core chain")
    ap.add_argument("--stage", choices=["dev", "chain"], required=True)
    ap.add_argument("--variants", nargs="*", default=["core"])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)
    paths71.ensure()
    if args.stage == "dev":
        print(json.dumps(dev_select(), indent=2, default=str)[:6000])
    else:
        chain(args.variants, tuple(args.seeds), tuple(args.folds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

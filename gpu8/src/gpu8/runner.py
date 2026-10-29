"""Run one registered experiment: build contexts, fit a method, score, record.

The runner is the only place that touches the ledger, so every number in the
paper can be traced back to a ``run_id`` with its configuration hash, data hash,
context hash, wall-clock time and peak memory.

Fold wiring (plan 6.2): a training disease is always encoded in the context
whose inner block *holds it out*, so its own relations are absent from the
reference graph it learns from.  Validation and test diseases are scored on the
outer context, whose reference set is the outer training diseases only, so
neither ever appears in the visible adjacency.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import paths
from .baselines import learned, meta_ridge, simple
from .config import Gpu8Config, load_config
from .contexts import EntityIndex, build_H, build_P, build_hierarchy_neighbours, load_entity_index
from .episodes import EpisodeBuilder, fit_feature_stats, load_bundle
from .hashing import atomic_write_df, atomic_write_json, code_version_hash, read_json, sha256_obj
from .metrics import evaluate_predictions
from .model import build_model
from .protocol import SplitSpec, fold_assignment, inner_blocks, load_spec
from .registry import RunRegistry
from .resources import ResourceMonitor, host_report
from .train import evaluate_diseases, score_disease, train_model


# --------------------------------------------------------------------------
# shared world
# --------------------------------------------------------------------------
class World:
    """Tables, entity index and the structures shared by every context."""

    def __init__(self, cfg: Gpu8Config, tables_dir: Path | None = None):
        self.cfg = cfg
        td = Path(tables_dir) if tables_dir else paths.TABLES_DIR
        self.tables_dir = td
        self.idx = load_entity_index(td)
        self.relations = pd.read_parquet(td / "relations.parquet")
        self.evidence = pd.read_parquet(td / "evidence.parquet")
        self.phenotypes = pd.read_parquet(td / "phenotypes.parquet")
        self.hierarchy = pd.read_parquet(td / "hierarchy.parquet")
        self.ppi = pd.read_parquet(td / "ppi.parquet")
        self.genes = pd.read_parquet(td / "genes.parquet")
        self.diseases = pd.read_parquet(td / "diseases.parquet")
        print("  building shared H / P / hierarchy neighbourhood ...", flush=True)
        H = build_H(self.phenotypes, self.idx)
        P, deg, adj = build_P(self.ppi, self.idx, return_adjacency=True)
        hn = build_hierarchy_neighbours(self.hierarchy, self.idx,
                                        cfg.candidates.hierarchy_max_distance)
        self.shared = {"H": H, "P": P, "hier_nb": hn, "ppi_adj": adj}
        self.background_rows = np.arange(self.idx.n_g, dtype=np.int64)
        self.gene_row_to_id = {i: g for g, i in self.idx.g_ix.items()}
        self.family_of = dict(zip(self.diseases["disease_id"], self.diseases["family_group"]))
        self._bundles: dict[str, object] = {}
        self._builders: dict[str, EpisodeBuilder] = {}

    def builder(self, context_id: str, stats: dict | None = None) -> EpisodeBuilder:
        if context_id not in self._builders:
            bundle = load_bundle(self.cfg, context_id, self.idx, self.phenotypes,
                                 self.hierarchy, self.ppi, self.background_rows,
                                 stats=stats, shared=self.shared)
            self._builders[context_id] = EpisodeBuilder(self.cfg, bundle,
                                                        self.relations, self.evidence)
        elif stats:
            self._builders[context_id].b.stats = stats
        return self._builders[context_id]

    def positives(self, disease_id: str) -> list[str]:
        key = set(self.cfg.data.primary_evidence)
        sub = self.relations[(self.relations["disease_id"] == disease_id)
                             & self.relations["relation_type"].isin(key)]
        return sorted(sub["gene_id"])


def primary_positive_map(world: World) -> dict[str, list[str]]:
    key = set(world.cfg.data.primary_evidence)
    rel = world.relations[world.relations["relation_type"].isin(key)]
    return rel.groupby("disease_id")["gene_id"].apply(list).to_dict()


# --------------------------------------------------------------------------
# task construction
# --------------------------------------------------------------------------
def build_eval_tasks(cfg: Gpu8Config, world: World, builder: EpisodeBuilder,
                     disease_ids: list[str], positives: dict[str, list[str]],
                     k: int = 0, n_candidates: int | None = None,
                     candidate_source: str = "fused",
                     dose_designs: dict | None = None) -> list[dict]:
    """Zero-shot: all positives are the query.  Dose task: fixed pool and query."""
    tasks = []
    g_ix = world.idx.g_ix
    for did in disease_ids:
        pos = positives.get(did, [])
        if not pos:
            continue
        d_row = world.idx.d_ix.get(did)
        if d_row is None:
            continue
        if dose_designs is not None:
            design = dose_designs.get(did)
            if design is None:
                continue
            support = design.support_at(k)
            query = design.query_genes
            exclude = {g_ix[g] for g in design.support_pool if g in g_ix}
        else:
            support, query = [], pos
            exclude = set()
        if not query:
            continue
        rows, info = builder.candidate_rows(did, d_row, n=n_candidates,
                                            source=candidate_source,
                                            exclude_rows=exclude)
        tasks.append({"disease_id": did, "support_genes": support,
                      "query_genes": query, "candidate_rows": rows,
                      "candidate_info": info, "k": k})
    return tasks


# --------------------------------------------------------------------------
# method fitting
# --------------------------------------------------------------------------
def fit_and_score(cfg: Gpu8Config, world: World, method: str, fold: int, seed: int,
                  variant: str, device: str, train_builders: dict,
                  outer_builder: EpisodeBuilder, train_diseases: list[str],
                  val_tasks: list[dict], test_tasks: list[dict],
                  positives: dict, max_epochs: int | None = None,
                  progress: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    """Returns (predictions, metrics, fit_info)."""
    ks = cfg.evaluation.ks
    info: dict = {"method": method, "variant": variant}

    def val_metric(model, equal_weight=False, disable_adapt=False):
        _, macro, _ = evaluate_diseases(cfg, model, outer_builder, val_tasks, device,
                                        ks=ks, equal_weight=equal_weight,
                                        disable_adaptation=disable_adapt)
        return macro

    # ---- neural family -------------------------------------------------
    if method in ("b9_ours", "b8_meta_ridge", "b5_mlp"):
        ablate = {
            "a1_no_missing_training": dict(missing_views=False),
            "a2_equal_weight_fixed": dict(equal_weight=True, use_controls=False),
        }.get(variant, {})
        equal_weight = bool(ablate.get("equal_weight", method == "b8_meta_ridge"))
        use_controls = bool(ablate.get("use_controls", method == "b9_ours"))
        missing_views = bool(ablate.get("missing_views", method == "b9_ours"))

        if method == "b5_mlp":
            model = meta_ridge.build_plain_mlp(cfg, seed=seed)
            equal_weight, use_controls, missing_views = True, False, False
        elif method == "b8_meta_ridge":
            model = meta_ridge.build_meta_ridge(cfg, seed=seed)
        else:
            model = build_model(cfg.model, seed=seed)
        info["n_parameters"] = int(model.n_trainable())
        if hasattr(model, "module_totals"):
            info["parameter_breakdown"] = model.module_totals()

        state = train_model(cfg, model, train_builders, train_diseases,
                            lambda m: val_metric(m, equal_weight=equal_weight),
                            device, seed, max_epochs=max_epochs,
                            equal_weight=equal_weight, use_controls=use_controls,
                            missing_views=missing_views, progress=progress)
        info["checkpoint"] = str(_save_checkpoint(model, method, fold, seed, variant))
        info.update({"epochs_run": state.epoch + 1, "best_epoch": state.best_epoch,
                     "best_val_metric": state.best_metric,
                     "stop_reason": state.stop_reason,
                     "n_tasks_seen": state.n_tasks_seen,
                     "history": state.history})
        per, macro, pred = evaluate_diseases(cfg, model, outer_builder, test_tasks,
                                             device, ks=ks, equal_weight=equal_weight)
        return pred, macro, info

    # ---- engineered / tree / graph comparators -------------------------
    if method == "b2_lr":
        m = learned.B2LogisticRegression(cfg).fit(train_builders[sorted(train_builders)[0]],
                                                  train_diseases, seed=42)
        scorer = lambda b, d, s, c: m.score(b, d, s, c)
    elif method == "b4_lightgbm":
        first = sorted(train_builders)[0]
        m = learned.B4LightGBMRanker(cfg).fit(train_builders[first], train_diseases,
                                              seed=seed)
        info["best_iteration"] = getattr(getattr(m.model, "best_iteration_", None), "real", None) \
            or getattr(m.model, "best_iteration_", None)
        scorer = lambda b, d, s, c: m.score(b, d, s, c)
    elif method == "b6_sign":
        first = sorted(train_builders)[0]
        m = learned.B6SignStyle(cfg).fit(train_builders[first], train_diseases,
                                         seed=seed, device=device)
        scorer = lambda b, d, s, c: m.score(b, d, s, c)
    elif method == "b0_degree_prior":
        scorer = lambda b, d, s, c: simple.b0_degree_prior(cfg, b, d, s, c)
    elif method == "b1_phenotype_vote":
        scorer = lambda b, d, s, c: simple.b1_phenotype_vote(cfg, b, d, s, c)
    elif method == "b3_cpu_fusion":
        scorer = lambda b, d, s, c: simple.b3_fusion_fixed_adapt(cfg, b, d, s, c)
    elif method == "b10_ppi_rwr":
        scorer = lambda b, d, s, c: simple.b10_ppi_rwr(cfg, b, d, s, c)
    elif method == "b10b_diamond":
        scorer = lambda b, d, s, c: simple.b10b_diamond_like(cfg, b, d, s, c)
    elif method == "b11_quantum_walk":
        scorer = lambda b, d, s, c: simple.b11_quantum_walk(cfg, b, d, s, c)
    elif method == "b7_kdgene":
        return fit_and_score_kdgene(cfg, world, fold, seed, device, train_diseases,
                                    test_tasks, val_tasks, info, progress=progress)
    else:
        raise ValueError(f"unknown method {method}")

    preds, skipped = [], 0
    for t in test_tasks:
        s = scorer(outer_builder, t["disease_id"], t["support_genes"], t["candidate_rows"])
        if s is None:
            skipped += 1
            continue
        g_ix = world.idx.g_ix
        qrows = {g_ix[g] for g in t["query_genes"] if g in g_ix}
        labels = np.array([1.0 if int(g) in qrows else 0.0 for g in t["candidate_rows"]],
                          dtype=np.float32)
        preds.append(pd.DataFrame({"disease_id": t["disease_id"],
                                   "gene_row": t["candidate_rows"],
                                   "score": s, "label": labels}))
    info["not_applicable_diseases"] = skipped
    if not preds:
        return pd.DataFrame(), {"not_applicable": True, "n_skipped": skipped}, info
    pred = pd.concat(preds, ignore_index=True)
    totals = {t["disease_id"]: len(t["query_genes"]) for t in test_tasks}
    per, macro = evaluate_predictions(pred, ks=ks, query_totals=totals)
    macro["n_not_applicable"] = skipped
    return pred, macro, info


def _save_checkpoint(model, method: str, fold: int, seed: int, variant: str) -> Path:
    """Persist the selected weights so the non-refit experiments can reuse them."""
    d = paths.GPU8_ROOT / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{method}__f{fold}__s{seed}" + (f"__{variant}" if variant else "") + ".pt"
    p = d / name
    state = model.state_dict()
    torch.save({k: v.detach().cpu() for k, v in state.items()}, p)
    return p


def fit_and_score_kdgene(cfg: Gpu8Config, world: World, fold: int, seed: int,
                         device: str, train_diseases: list[str],
                         test_tasks: list[dict], val_tasks: list[dict],
                         info: dict, progress: bool = True):
    from .baselines import kdgene as kd
    kg = kd.build_kg(cfg, world.relations, world.phenotypes, world.hierarchy,
                     world.ppi, set(train_diseases),
                     [world.gene_row_to_id[i] for i in world.background_rows])
    info.update({"kg_entities": kg.n_entities, "kg_relations": kg.n_relations,
                 "kg_triples": int(kg.triples.shape[0])})
    edim = int(read_json(paths.CONFIG_DIR / "kdgene_config.json", {}).get("edim", 1000))
    model, hist = kd.train_kdgene(cfg, kg, device, seed=seed, edim=edim, rdim=edim,
                                  max_epochs=30, patience=4, progress=progress)
    info.update(hist)
    preds, skipped = [], 0
    for t in test_tasks:
        s = kd.score_kdgene(model, kg, t["disease_id"],
                            [world.gene_row_to_id[i] for i in world.background_rows],
                            t["candidate_rows"], world.gene_row_to_id, device)
        if s is None:
            skipped += 1
            continue
        g_ix = world.idx.g_ix
        qrows = {g_ix[g] for g in t["query_genes"] if g in g_ix}
        labels = np.array([1.0 if int(g) in qrows else 0.0 for g in t["candidate_rows"]],
                          dtype=np.float32)
        preds.append(pd.DataFrame({"disease_id": t["disease_id"],
                                   "gene_row": t["candidate_rows"],
                                   "score": s, "label": labels}))
    info["not_applicable_diseases"] = skipped
    if not preds:
        return pd.DataFrame(), {"not_applicable": True}, info
    pred = pd.concat(preds, ignore_index=True)
    totals = {t["disease_id"]: len(t["query_genes"]) for t in test_tasks}
    per, macro = evaluate_predictions(pred, ks=cfg.evaluation.ks, query_totals=totals)
    macro["n_not_applicable"] = skipped
    return pred, macro, info


# --------------------------------------------------------------------------
# one registered run
# --------------------------------------------------------------------------
def run_one(cfg: Gpu8Config, world: World, run_id: str, method: str, fold: int,
            seed: int, variant: str = "", registry: RunRegistry | None = None,
            device: str | None = None, max_epochs: int | None = None,
            n_val: int = 200, progress: bool = True,
            val_on: str = "fold", skip_test: bool = False,
            overrides: dict | None = None) -> dict:
    """val_on='dev' validates on the held-out development families instead of
    F_{fold+1}.  Development runs use it so that the five formal folds are never
    consulted while architecture and hyper-parameters are being chosen."""
    device = device or (cfg.hardware.device if torch.cuda.is_available() else "cpu")
    spec = load_spec()
    fa = fold_assignment(spec, fold, cfg.protocol.n_outer_folds)
    positives = primary_positive_map(world)

    # A3 and A4 change the inputs, so they run on contexts rebuilt without the
    # phenotype path or without the interaction propagation.  A1 and A2 change
    # only the training procedure and reuse the ordinary fold contexts.
    prefix = {"a3_no_phenotype": f"a3nophen_f{fold}",
              "a4_no_ppi": f"a4noppi_f{fold}"}.get(variant, f"f{fold}")
    ctx_available = read_json(paths.MANIFEST_DIR / "context_index.json", {}) \
        .get("contexts", {})
    if f"{prefix}_outer" not in ctx_available:
        if prefix != f"f{fold}":
            raise RuntimeError(
                f"variant {variant} needs context {prefix}_outer; build it with "
                f"gpu8.alt_contexts --which ablation before running this variant")
    blocks = inner_blocks(fa["train_diseases"], spec.family_of, cfg.protocol.inner_blocks,
                          cfg.protocol.split_seed + 100 + fold)
    outer = world.builder(f"{prefix}_outer")
    stats = fit_feature_stats(outer, fa["train_diseases"], seed=cfg.protocol.split_seed)
    outer.b.stats = stats
    train_builders = {}
    for bi, block in enumerate(blocks):
        cid = f"{prefix}_inner{bi}"
        if cid not in ctx_available:
            continue
        b = world.builder(cid, stats=stats)
        b.b.stats = stats
        held = ctx_available[cid].get("held_out_block") or block
        setattr(b, "block_diseases", held)
        train_builders[cid] = b
    if not train_builders:
        train_builders = {f"{prefix}_outer": outer}

    val_source = spec.dev_diseases if val_on == "dev" else fa["val_diseases"]
    val_ids = [d for d in val_source if positives.get(d)][:n_val]
    val_tasks = build_eval_tasks(cfg, world, outer, val_ids, positives)
    if skip_test:
        test_tasks = val_tasks
    else:
        test_ids = [d for d in fa["test_diseases"] if positives.get(d)]
        test_tasks = build_eval_tasks(cfg, world, outer, test_ids, positives)

    out_dir = paths.run_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_index = read_json(paths.MANIFEST_DIR / "context_index.json", {})
    code_hash = code_version_hash(Path(__file__).resolve().parent)
    if registry:
        registry.mark_running(run_id, method=method, fold=str(fold), seed=str(seed),
                              variant=variant, device=device,
                              config_hash=cfg.config_hash, code_hash=code_hash,
                              context_hash=ctx_index.get("contexts", {})
                              .get(f"f{fold}_outer", {}).get("context_hash", ""),
                              data_hash=ctx_index.get("resource_hash", ""),
                              output_dir=str(out_dir))

    with ResourceMonitor(interval=0.25, sample_nvidia=(device == "cuda")) as mon:
        pred, macro, info = fit_and_score(cfg, world, method, fold, seed, variant,
                                          device, train_builders, outer,
                                          fa["train_diseases"], val_tasks, test_tasks,
                                          positives, max_epochs=max_epochs,
                                          progress=progress)
    res = mon.result

    if not pred.empty:
        totals = {t["disease_id"]: len(t["query_genes"]) for t in test_tasks}
        per, _ = evaluate_predictions(pred, ks=cfg.evaluation.ks, query_totals=totals)
        per["method"], per["fold"], per["seed"], per["variant"] = method, fold, seed, variant
        per["k"] = 0
        per["run_id"] = run_id
        fam = world.family_of
        per["family_group"] = per["disease_id"].map(fam)
        atomic_write_df(per, out_dir / "per_disease.csv")
        atomic_write_df(pred, out_dir / "predictions.parquet")
    payload = {"run_id": run_id, "method": method, "fold": fold, "seed": seed,
               "variant": variant, "device": device, "metrics": macro,
               "fit_info": {k: v for k, v in info.items() if k != "history"},
               "resources": res.as_dict(),
               "n_test_diseases": len(test_tasks), "n_val_diseases": len(val_tasks),
               "config_hash": cfg.config_hash, "code_hash": code_hash}
    atomic_write_json(out_dir / "result.json", payload)
    if "history" in info:
        atomic_write_json(out_dir / "training_history.json", info["history"])

    if registry:
        registry.mark_done(
            run_id, seconds=round(res.seconds, 2),
            peak_rss_mib=round(res.peak_rss_mib, 1),
            peak_gpu_allocated_mib=round(res.peak_gpu_allocated_mib, 1),
            peak_gpu_reserved_mib=round(res.peak_gpu_reserved_mib, 1),
            epochs_run=str(info.get("epochs_run", "")),
            best_epoch=str(info.get("best_epoch", "")),
            best_val_metric=str(round(info["best_val_metric"], 6))
            if isinstance(info.get("best_val_metric"), float) else "",
            n_test_diseases=str(len(test_tasks)),
            output_hash=sha256_obj(macro), output_dir=str(out_dir))
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one registered GPU8 experiment.")
    ap.add_argument("--method", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", default="")
    ap.add_argument("--run_id", default="")
    ap.add_argument("--registry", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--max_epochs", type=int, default=0)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--val_on", default="fold", choices=["fold", "dev"])
    ap.add_argument("--skip_test", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    world = World(cfg)
    reg = RunRegistry(Path(args.registry)) if args.registry else None
    run_id = args.run_id or f"AD_HOC__{args.method}__f{args.fold}__s{args.seed}"
    payload = run_one(cfg, world, run_id, args.method, args.fold, args.seed,
                      args.variant, registry=reg, device=args.device or None,
                      max_epochs=args.max_epochs or None, n_val=args.n_val,
                      val_on=args.val_on, skip_test=args.skip_test)
    print(json.dumps({k: payload[k] for k in ("run_id", "method", "metrics")},
                     indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

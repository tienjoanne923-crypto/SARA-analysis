"""Run the family, external, temporal and second-network experiments.

Each reuses the same training code and the same visibility contract as the main
folds; only the disease partition, the reference graph and the evaluation label
source differ.  The external run is the one confirmation set that never feeds
back into development: its diseases were removed from every reference graph
before any modelling began.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import paths
from .config import Gpu8Config, load_config
from .episodes import fit_feature_stats
from .hashing import atomic_write_df, atomic_write_json, read_json, sha256_obj
from .metrics import evaluate_predictions
from .protocol import fold_assignment, load_spec
from .registry import RunRegistry
from .resources import ResourceMonitor
from .runner import World, build_eval_tasks, fit_and_score, primary_positive_map

MODES = ("family", "external", "temporal", "ppi2")


def partition(cfg: Gpu8Config, world: World, mode: str) -> dict:
    """Return train / val / test disease lists and the label source."""
    if mode == "family":
        blob = read_json(paths.MANIFEST_DIR / "splits_family.json", {})
        return {"train": blob["train_diseases"], "val": blob["val_diseases"],
                "test": blob["test_diseases"], "labels": "primary_relations"}
    if mode == "external":
        spec = load_spec()
        train = sorted(set(spec.family_of) - set(spec.dev_diseases)
                       - set(spec.external_diseases))
        return {"train": train, "val": spec.dev_diseases[:200],
                "test": spec.external_diseases, "labels": "external"}
    if mode == "temporal":
        hist = pd.read_parquet(paths.GPU8_ROOT / "tables_historical" / "relations.parquet")
        prim = hist[hist["relation_type"].isin(set(cfg.data.primary_evidence))]
        train = sorted(set(prim["disease_id"]))
        fut = pd.read_csv(paths.RESULTS_DIR / "temporal" / "temporal_future_positives.csv",
                          dtype=str)
        test = sorted(set(fut["disease_id"]))
        return {"train": [d for d in train if d not in set(test)],
                "val": [d for d in train if d not in set(test)][:200],
                "test": test, "labels": "temporal"}
    if mode == "ppi2":
        spec = load_spec()
        fa = fold_assignment(spec, 0, cfg.protocol.n_outer_folds)
        return {"train": fa["train_diseases"], "val": fa["val_diseases"][:200],
                "test": fa["test_diseases"], "labels": "primary_relations"}
    raise ValueError(mode)


def label_map(cfg: Gpu8Config, world: World, mode: str, test_ids: list[str]) -> dict:
    """Query positives per test disease, by label source."""
    if mode == "external":
        p = paths.TABLES_DIR / "external_labels.parquet"
        if not p.exists():
            return {}
        ext = pd.read_parquet(p)
        ext = ext[(ext["evaluable"] == 1) & (ext["tier"] == "high")]
        ext = ext[ext["disease_id"].isin(set(test_ids))]
        return ext.groupby("disease_id")["gene_id"].apply(
            lambda s: sorted(set(s.astype(str)))).to_dict()
    if mode == "temporal":
        fut = pd.read_csv(paths.RESULTS_DIR / "temporal" / "temporal_future_positives.csv",
                          dtype=str)
        return fut.groupby("disease_id")["gene_id"].apply(
            lambda s: sorted(set(s))).to_dict()
    pos = primary_positive_map(world)
    return {d: pos[d] for d in test_ids if d in pos}


def run_alt(cfg: Gpu8Config, world: World, mode: str, method: str, seed: int,
            run_id: str, registry: RunRegistry | None = None,
            device: str | None = None, n_val: int = 150,
            progress: bool = True) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    part = partition(cfg, world, mode)
    positives = primary_positive_map(world)

    outer = world.builder(f"{mode}_outer")
    stats = fit_feature_stats(outer, part["train"], seed=cfg.protocol.split_seed)
    outer.b.stats = stats
    index = read_json(paths.MANIFEST_DIR / "context_index.json", {})
    train_builders = {}
    for i in range(cfg.protocol.inner_blocks):
        cid = f"{mode}_inner{i}"
        if cid not in index.get("contexts", {}):
            continue
        b = world.builder(cid, stats=stats)
        b.b.stats = stats
        setattr(b, "block_diseases", index["contexts"][cid].get("held_out_block", []))
        train_builders[cid] = b
    if not train_builders:
        train_builders = {f"{mode}_outer": outer}

    val_ids = [d for d in part["val"] if positives.get(d)][:n_val]
    val_tasks = build_eval_tasks(cfg, world, outer, val_ids, positives)

    labels = label_map(cfg, world, mode, part["test"])
    test_ids = [d for d in part["test"] if labels.get(d)]
    test_tasks = build_eval_tasks(cfg, world, outer, test_ids, labels)
    if mode in ("external", "temporal"):
        # a query gene already visible in training is not new knowledge
        train_pairs = set(zip(
            world.relations[world.relations["relation_type"].isin(
                set(cfg.data.primary_evidence))]["disease_id"],
            world.relations[world.relations["relation_type"].isin(
                set(cfg.data.primary_evidence))]["gene_id"])) if mode == "external" else set()
        if train_pairs:
            for t in test_tasks:
                t["query_genes"] = [g for g in t["query_genes"]
                                    if (t["disease_id"], g) not in train_pairs]
            test_tasks = [t for t in test_tasks if t["query_genes"]]

    print(f"  {mode}: train {len(part['train'])}, val {len(val_tasks)}, "
          f"test {len(test_tasks)} diseases", flush=True)

    out_dir = paths.run_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    if registry:
        registry.mark_running(run_id, method=method, seed=str(seed), device=device,
                              config_hash=cfg.config_hash, output_dir=str(out_dir))

    with ResourceMonitor(interval=0.25, sample_nvidia=(device == "cuda")) as mon:
        pred, macro, info = fit_and_score(cfg, world, method, 0, seed, "", device,
                                          train_builders, outer, part["train"],
                                          val_tasks, test_tasks, positives,
                                          progress=progress)
    res = mon.result

    if not pred.empty:
        totals = {t["disease_id"]: len(t["query_genes"]) for t in test_tasks}
        per, _ = evaluate_predictions(pred, ks=cfg.evaluation.ks, query_totals=totals)
        per["method"], per["fold"], per["seed"] = method, mode, seed
        per["variant"], per["k"], per["run_id"] = "", 0, run_id
        per["family_group"] = per["disease_id"].map(world.family_of)
        atomic_write_df(per, out_dir / "per_disease.csv")
        atomic_write_df(pred, out_dir / "predictions.parquet")

    payload = {"run_id": run_id, "method": method, "fold": mode, "seed": seed,
               "variant": "", "device": device, "mode": mode,
               "label_source": part["labels"], "metrics": macro,
               "fit_info": {k: v for k, v in info.items() if k != "history"},
               "resources": res.as_dict(),
               "n_test_diseases": len(test_tasks), "n_val_diseases": len(val_tasks),
               "config_hash": cfg.config_hash}
    atomic_write_json(out_dir / "result.json", payload)
    if registry:
        registry.mark_done(run_id, seconds=round(res.seconds, 2),
                           peak_rss_mib=round(res.peak_rss_mib, 1),
                           peak_gpu_allocated_mib=round(res.peak_gpu_allocated_mib, 1),
                           n_test_diseases=str(len(test_tasks)),
                           output_hash=sha256_obj(macro), output_dir=str(out_dir))
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run an alternative-split experiment.")
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_id", default="")
    ap.add_argument("--registry", default="")
    ap.add_argument("--n_val", type=int, default=150)
    args = ap.parse_args(argv)

    cfg = load_config()
    world = World(cfg)
    reg = RunRegistry(Path(args.registry)) if args.registry else None
    run_id = args.run_id or f"ALT__{args.mode}__{args.method}__s{args.seed}"
    payload = run_alt(cfg, world, args.mode, args.method, args.seed, run_id,
                      registry=reg, n_val=args.n_val)
    print(json.dumps({k: payload[k] for k in ("run_id", "mode", "metrics")},
                     indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

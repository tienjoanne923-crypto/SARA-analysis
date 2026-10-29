"""The non-refit experiments: dose, evidence dependence, coverage, negatives.

Plan section 13.1 marks E03, E04, E05 and E10 as *not* global refits: they
reuse a frozen model and change only what is exposed or what is scored, so
they never enter the 166-fit ledger.  Keeping them here, separate from
``runner.py``, makes that boundary explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import paths
from .baselines import simple
from .config import Gpu8Config, load_config
from .hashing import atomic_write_df, atomic_write_json
from .metrics import evaluate_predictions
from .protocol import fold_assignment, load_spec
from .runner import World, build_eval_tasks, primary_positive_map
from .support import DoseDesign, build_dose_design, cross_disease_replacement
from .train import score_disease


# --------------------------------------------------------------------------
# E03 fixed support dose
# --------------------------------------------------------------------------
def dose_cohort(cfg: Gpu8Config, world: World, disease_ids: list[str],
                positives: dict, draw: int = 0) -> dict[str, DoseDesign]:
    out = {}
    for did in disease_ids:
        d = build_dose_design(cfg, did, positives.get(did, []), draw=draw)
        if d is not None:
            out[did] = d
    return out


def run_dose_experiment(cfg: Gpu8Config, world: World, model, builder, fold: int,
                        device: str, draws: int | None = None,
                        equal_weight: bool = False, tag: str = "b9_ours") -> pd.DataFrame:
    """Same disease, same query, same candidates for every k; only exposure moves."""
    spec = load_spec()
    fa = fold_assignment(spec, fold, cfg.protocol.n_outer_folds)
    positives = primary_positive_map(world)
    draws = int(draws or cfg.protocol.support_order_draws)
    rows = []
    base_designs = dose_cohort(cfg, world, fa["test_diseases"], positives, draw=0)
    if not base_designs:
        return pd.DataFrame()
    g_ix = world.idx.g_ix
    # candidates are fixed once: the maximal pool is excluded for every k
    tasks: dict[str, dict] = {}
    for did, design in base_designs.items():
        d_row = world.idx.d_ix[did]
        exclude = {g_ix[g] for g in design.support_pool if g in g_ix}
        cand, info = builder.candidate_rows(did, d_row, exclude_rows=exclude)
        tasks[did] = {"candidate_rows": cand, "info": info,
                      "query": design.query_genes}

    for draw in range(draws):
        designs = dose_cohort(cfg, world, list(base_designs), positives, draw=draw) \
            if draw else base_designs
        for k in cfg.protocol.support_doses:
            preds = []
            for did, design in designs.items():
                t = tasks[did]
                support = design.support_at(k)
                s, info = score_disease(cfg, model, builder, did, support,
                                        t["candidate_rows"], device,
                                        equal_weight=equal_weight)
                qrows = {g_ix[g] for g in t["query"] if g in g_ix}
                labels = np.array([1.0 if int(g) in qrows else 0.0
                                   for g in t["candidate_rows"]], dtype=np.float32)
                preds.append(pd.DataFrame({"disease_id": did,
                                           "gene_row": t["candidate_rows"],
                                           "score": s, "label": labels}))
            if not preds:
                continue
            pred = pd.concat(preds, ignore_index=True)
            totals = {d: len(tasks[d]["query"]) for d in designs}
            per, macro = evaluate_predictions(pred, ks=cfg.evaluation.ks,
                                              query_totals=totals)
            per["k"], per["draw"], per["method"], per["fold"] = k, draw, tag, fold
            per["family_group"] = per["disease_id"].map(world.family_of)
            rows.append(per)
            # k = 5 is order-invariant, so one draw is enough
            if k == max(cfg.protocol.support_doses) and draw > 0:
                rows[-1] = rows[-1].iloc[0:0]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# E04 evidence dependence and cross-disease replacement support
# --------------------------------------------------------------------------
def run_evidence_experiment(cfg: Gpu8Config, world: World, model, builder, fold: int,
                            device: str, tag: str = "b9_ours",
                            **_) -> pd.DataFrame:
    """Exact-duplicate invariance, natural source overlap strata, replacement support."""
    spec = load_spec()
    fa = fold_assignment(spec, fold, cfg.protocol.n_outer_folds)
    positives = primary_positive_map(world)
    designs = dose_cohort(cfg, world, fa["test_diseases"], positives, draw=0)
    if not designs:
        return pd.DataFrame()
    g_ix = world.idx.g_ix
    feas = pd.read_csv(paths.MANIFEST_DIR / "support_feasibility.csv")
    dep = dict(zip(feas["disease_id"], feas["duplicate_record_ratio"]))
    k = 5
    rows = []

    for condition in ("baseline", "equal_weight", "replace_1", "replace_2"):
        preds, meta = [], []
        for did, design in designs.items():
            d_row = world.idx.d_ix[did]
            exclude = {g_ix[g] for g in design.support_pool if g in g_ix}
            cand, _ = builder.candidate_rows(did, d_row, exclude_rows=exclude)
            support = design.support_at(k)
            equal = condition == "equal_weight"
            if condition.startswith("replace_"):
                n_rep = int(condition.split("_")[1])
                support = cross_disease_replacement(
                    cfg, did, support, n_rep, world.background_rows,
                    builder.b.gene_ctd_degree, builder.b.gene_ppi_degree, g_ix)
            s, info = score_disease(cfg, model, builder, did, support, cand, device,
                                    equal_weight=equal)
            qrows = {g_ix[g] for g in design.query_genes if g in g_ix}
            labels = np.array([1.0 if int(g) in qrows else 0.0 for g in cand],
                              dtype=np.float32)
            preds.append(pd.DataFrame({"disease_id": did, "gene_row": cand,
                                       "score": s, "label": labels}))
            meta.append({"disease_id": did, "a_norm": info.get("a_norm", 0.0),
                         "n_groups": info.get("n_groups", 0),
                         "n_effective": info.get("n_effective", 0.0),
                         "lam": info.get("lam", float("nan")),
                         "rho": info.get("rho", float("nan"))})
        if not preds:
            continue
        pred = pd.concat(preds, ignore_index=True)
        totals = {d: len(designs[d].query_genes) for d in designs}
        per, _ = evaluate_predictions(pred, ks=cfg.evaluation.ks, query_totals=totals)
        m = pd.DataFrame(meta).set_index("disease_id")
        per = per.join(m, on="disease_id")
        per["condition"], per["method"], per["fold"], per["k"] = condition, tag, fold, k
        per["duplicate_record_ratio"] = per["disease_id"].map(dep)
        per["family_group"] = per["disease_id"].map(world.family_of)
        rows.append(per)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# E05 candidate coverage and retriever source sensitivity
# --------------------------------------------------------------------------
def run_coverage_experiment(cfg: Gpu8Config, world: World, model, builder, fold: int,
                            device: str, tag: str = "b9_ours") -> pd.DataFrame:
    spec = load_spec()
    fa = fold_assignment(spec, fold, cfg.protocol.n_outer_folds)
    positives = primary_positive_map(world)
    g_ix = world.idx.g_ix
    rows = []
    for source in ("fused", "phenotype", "hierarchy"):
        for n in cfg.candidates.stress_sizes:
            preds, cov = [], []
            for did in fa["test_diseases"]:
                pos = positives.get(did, [])
                d_row = world.idx.d_ix.get(did)
                if not pos or d_row is None:
                    continue
                cand, info = builder.candidate_rows(did, d_row, n=n, source=source)
                qrows = {g_ix[g] for g in pos if g in g_ix}
                inter = qrows & set(cand.tolist())
                cov.append({"disease_id": did, "n_query": len(qrows),
                            "n_in_pool": len(inter),
                            "coverage": len(inter) / max(1, len(qrows)),
                            "zero_coverage": int(len(inter) == 0),
                            "tie_break_fallback": info["tie_break_fallback"],
                            "retriever_empty": int(info["retriever_empty"])})
                s, _ = score_disease(cfg, model, builder, did, [], cand, device)
                labels = np.array([1.0 if int(g) in qrows else 0.0 for g in cand],
                                  dtype=np.float32)
                preds.append(pd.DataFrame({"disease_id": did, "gene_row": cand,
                                           "score": s, "label": labels}))
            if not preds:
                continue
            pred = pd.concat(preds, ignore_index=True)
            totals = {d: len(positives.get(d, [])) for d in fa["test_diseases"]}
            per, _ = evaluate_predictions(pred, ks=cfg.evaluation.ks, query_totals=totals)
            per = per.merge(pd.DataFrame(cov), on="disease_id", how="left",
                            suffixes=("", "_cov"))
            per["source"], per["n_candidates_requested"] = source, n
            per["method"], per["fold"] = tag, fold
            per["family_group"] = per["disease_id"].map(world.family_of)
            rows.append(per)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# E10 disease-specific negative controls
# --------------------------------------------------------------------------
def run_negative_controls(cfg: Gpu8Config, world: World, model, builder, fold: int,
                          device: str, tag: str = "b9_ours", seed: int = 20260922
                          ) -> pd.DataFrame:
    """Disease permutation and support permutation, rescoring a frozen model.

    Degree-preserving rewiring is *not* run here: it changes the graph, so it
    needs its own context rebuild and is reported as a separate experiment.
    """
    spec = load_spec()
    fa = fold_assignment(spec, fold, cfg.protocol.n_outer_folds)
    positives = primary_positive_map(world)
    designs = dose_cohort(cfg, world, fa["test_diseases"], positives, draw=0)
    g_ix = world.idx.g_ix
    rng = np.random.default_rng(seed)
    rows = []
    ids = sorted(designs)
    if len(ids) < 4:
        return pd.DataFrame()
    permuted = list(rng.permutation(ids))

    for condition in ("intact", "disease_permuted", "support_permuted"):
        preds = []
        for i, did in enumerate(ids):
            design = designs[did]
            d_row = world.idx.d_ix[did]
            exclude = {g_ix[g] for g in design.support_pool if g in g_ix}
            cand, _ = builder.candidate_rows(did, d_row, exclude_rows=exclude)
            if condition == "support_permuted":
                other = permuted[i] if permuted[i] != did else permuted[(i + 1) % len(ids)]
                support = designs[other].support_at(5)
            else:
                support = design.support_at(5)
            score_disease_id = permuted[i] if condition == "disease_permuted" else did
            if condition == "disease_permuted" and score_disease_id == did:
                score_disease_id = permuted[(i + 1) % len(ids)]
            s, _ = score_disease(cfg, model, builder, score_disease_id, support,
                                 cand, device)
            qrows = {g_ix[g] for g in design.query_genes if g in g_ix}
            labels = np.array([1.0 if int(g) in qrows else 0.0 for g in cand],
                              dtype=np.float32)
            preds.append(pd.DataFrame({"disease_id": did, "gene_row": cand,
                                       "score": s, "label": labels}))
        pred = pd.concat(preds, ignore_index=True)
        totals = {d: len(designs[d].query_genes) for d in ids}
        per, _ = evaluate_predictions(pred, ks=cfg.evaluation.ks, query_totals=totals)
        per["condition"], per["method"], per["fold"] = condition, tag, fold
        per["family_group"] = per["disease_id"].map(world.family_of)
        rows.append(per)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# seeded network comparators on the dose cohort (B10 / B10b / B11)
# --------------------------------------------------------------------------
def run_network_comparators(cfg: Gpu8Config, world: World, builder, fold: int
                            ) -> pd.DataFrame:
    spec = load_spec()
    fa = fold_assignment(spec, fold, cfg.protocol.n_outer_folds)
    positives = primary_positive_map(world)
    designs = dose_cohort(cfg, world, fa["test_diseases"], positives, draw=0)
    g_ix = world.idx.g_ix
    fns = {"b10_ppi_rwr": simple.b10_ppi_rwr,
           "b10b_diamond": simple.b10b_diamond_like,
           "b11_quantum_walk": simple.b11_quantum_walk}
    rows = []
    for name, fn in fns.items():
        for k in (1, 3, 5):
            preds, skipped = [], 0
            for did, design in designs.items():
                d_row = world.idx.d_ix[did]
                exclude = {g_ix[g] for g in design.support_pool if g in g_ix}
                cand, _ = builder.candidate_rows(did, d_row, exclude_rows=exclude)
                s = fn(cfg, builder, did, design.support_at(k), cand)
                if s is None:
                    skipped += 1
                    continue
                qrows = {g_ix[g] for g in design.query_genes if g in g_ix}
                labels = np.array([1.0 if int(g) in qrows else 0.0 for g in cand],
                                  dtype=np.float32)
                preds.append(pd.DataFrame({"disease_id": did, "gene_row": cand,
                                           "score": s, "label": labels}))
            if not preds:
                continue
            pred = pd.concat(preds, ignore_index=True)
            totals = {d: len(designs[d].query_genes) for d in designs}
            per, _ = evaluate_predictions(pred, ks=cfg.evaluation.ks, query_totals=totals)
            per["method"], per["k"], per["fold"] = name, k, fold
            per["n_not_applicable"] = skipped
            per["family_group"] = per["disease_id"].map(world.family_of)
            rows.append(per)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the non-refit experiments.")
    ap.add_argument("--experiment", required=True,
                    choices=["dose", "evidence", "coverage", "negative", "network"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--method", default="b9_ours",
                    choices=["b9_ours", "b8_meta_ridge"])
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--draws", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config()
    world = World(cfg)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from .episodes import fit_feature_stats
    from .model import build_model
    from .protocol import load_spec as _ls
    spec = _ls()
    fa = fold_assignment(spec, args.fold, cfg.protocol.n_outer_folds)
    builder = world.builder(f"f{args.fold}_outer")
    builder.b.stats = fit_feature_stats(builder, fa["train_diseases"],
                                        seed=cfg.protocol.split_seed)

    # The meta-ridge comparator wraps the same encoder but replaces the control
    # head, so its checkpoint has a "base." prefix and must be rebuilt as the
    # wrapper rather than loaded into the plain model.
    if args.method == "b8_meta_ridge":
        from .baselines.meta_ridge import build_meta_ridge
        model = build_meta_ridge(cfg, seed=42)
        equal_weight = True
    else:
        model = build_model(cfg.model, seed=42)
        equal_weight = False
    ckpt = args.checkpoint or str(paths.GPU8_ROOT / "checkpoints"
                                  / f"{args.method}__f{args.fold}__s42.pt")
    if Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  loaded checkpoint {Path(ckpt).name}", flush=True)
    else:
        print(f"  WARNING: no checkpoint at {ckpt}; using an untrained model",
              flush=True)
    model.to(device).eval()

    fn = {"evidence": run_evidence_experiment, "coverage": run_coverage_experiment,
          "negative": run_negative_controls}
    if args.experiment == "network":
        df = run_network_comparators(cfg, world, builder, args.fold)
    elif args.experiment == "dose":
        df = run_dose_experiment(cfg, world, model, builder, args.fold, device,
                                 draws=args.draws or None, tag=args.method,
                                 equal_weight=equal_weight)
    else:
        df = fn[args.experiment](cfg, world, model, builder, args.fold, device,
                                 tag=args.method)

    suffix = "" if args.method == "b9_ours" else f"_{args.method}"
    out = Path(args.out) if args.out else (paths.RESULTS_DIR / "experiments"
                                           / f"{args.experiment}_f{args.fold}{suffix}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_df(df, out)
    print(json.dumps({"experiment": args.experiment, "fold": args.fold,
                      "rows": int(len(df)), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

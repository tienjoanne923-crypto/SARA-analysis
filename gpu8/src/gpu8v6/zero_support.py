"""V6 zero-support line (task Z): scorer tuning, development fits, formal runs.

Stages (each writes its own artefacts; nothing reads a test fold before the
protocol manifest ``protocol_v6_registered.json`` exists):

``tune``    dev-only tuning of the fixed fusion F: the hand-set V5 weights, the
            0.25-step simplex over the four V5 channels (35 points + refinement)
            and over the seven V6 sources (210 points + refinement).
``devfit``  fold 0, 3 learned classes x 2 presets x 2 inner validation splits
            (A = adjacent fold F1 in the outer context, B = development
            families) = 12 fits, plus logistic regression; one-standard-error
            rule picks each class's preset.
``formal``  five folds; deterministic scorers once per fold, learned scorers with
            seeds 42/52/62 (at most 45 core fits in total).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from . import paths6
from .common import World6, candidate_hash, dump_json, load_manifest, now_utc, write_manifest
from .disease_semantics import load as load_sem
from .gene_features import load as load_gene_feats
from .ledger6 import run_dir, tracked
from .metrics6 import macro
from .residual import ResidualTrainer
from .scoring import (HAND_WEIGHTS, V5_CHANNELS, LightGBMScorer, LogisticScorer, evaluate,
                      fold_blocks, fusion_scores, load_block, recall_at, tune_simplex)
from .sources import SOURCES

SEEDS = (42, 52, 62)
PRESETS = {
    "lightgbm": [{"num_leaves": 31, "min_child_samples": 100, "learning_rate": 0.05},
                 {"num_leaves": 63, "min_child_samples": 20, "learning_rate": 0.1}],
    "bilinear": [{"rank": 32}, {"rank": 64}],
    "attention": [{"hidden": 32, "heads": 2}, {"hidden": 64, "heads": 2}],
    "lr": [{"C": 0.1}, {"C": 1.0}],
}
SIMPLER_FIRST = {"lightgbm": 0, "bilinear": 0, "attention": 0, "lr": 0}


# --------------------------------------------------------------------------
# stage 1: fixed fusion on dev
# --------------------------------------------------------------------------
def tune(w: World6) -> dict:
    dev = w.labelled(w.spec.dev_diseases)
    blocks = [load_block(f"f{f}_outer", ids=dev) for f in range(5)]
    out = paths6.SCORING
    res = {}
    hand = float(np.concatenate([recall_at(b, fusion_scores(b, HAND_WEIGHTS)) for b in blocks]).mean())
    ret = float(np.concatenate([recall_at(b, b.feat("fused")) for b in blocks]).mean())
    w4, t4 = tune_simplex(blocks, V5_CHANNELS)
    t4.to_csv(out / "fusion4_tuning_dev.csv", index=False, encoding="utf-8-sig")
    w7, t7 = tune_simplex(blocks, SOURCES)
    t7.to_csv(out / "fusion7_tuning_dev.csv", index=False, encoding="utf-8-sig")
    r4, r7 = float(t4["recall10"].iloc[0]), float(t7["recall10"].iloc[0])
    main = ("F7", w7, r7) if r7 >= r4 else ("F4", w4, r4)
    res = {"decided_on": "development families x 5 outer contexts (budget 500)",
           "objective": "macro end-to-end Recall@10 within the frozen candidates",
           "hand_v5_weights_on_v6_channels": {"weights": HAND_WEIGHTS, "dev_recall10": hand},
           "retrieval_order": {"dev_recall10": ret},
           "F4": {"weights": w4, "dev_recall10": r4, "n_grid": int((t4["stage"] == "grid").sum()),
                  "n_refine": int((t4["stage"] == "refine").sum())},
           "F7": {"weights": w7, "dev_recall10": r7, "n_grid": int((t7["stage"] == "grid").sum()),
                  "n_refine": int((t7["stage"] == "refine").sum())},
           "main": main[0], "weights": main[1], "dev_recall10": main[2],
           "n_dev_disease_contexts": int(sum(b.D for b in blocks))}
    write_manifest("scorer_v6_frozen.json", res)
    return res


# --------------------------------------------------------------------------
# shared fitting helpers
# --------------------------------------------------------------------------
class Kit:
    def __init__(self, w: World6, device: str = "cuda"):
        self.w = w
        self.device = device
        self.gene_feats, self.gene_names = load_gene_feats("current")
        self.sem = load_sem()
        self.frozen = load_manifest("scorer_v6_frozen.json")
        self.F = self.frozen["weights"]
        self.d_ix = w.idx.d_ix

    def fit_learned(self, kind: str, preset: dict, train, val, seed: int):
        if kind == "lightgbm":
            m = LightGBMScorer(seed=seed, **preset).fit(train, [val], gene_feats=self.gene_feats)
            info = {"best_iteration": m.best_iteration, "n_train_lists": m.n_train_lists}
            return (lambda b: m.score(b)), info, m
        if kind == "lr":
            m = LogisticScorer(seed=seed, **preset).fit(train, gene_feats=self.gene_feats)
            return (lambda b: m.score(b)), {}, m
        tr = ResidualTrainer(self.F, self.gene_feats, self.sem, device=self.device)
        fit = tr.fit(kind, train, val, self.d_ix, seed=seed, config=preset)
        info = {"selected": fit.selected, "kappa": fit.kappa, "best_epoch": fit.best_epoch,
                "val_recall_residual": fit.val_recall_residual,
                "val_recall_base": fit.val_recall_base, "n_params": fit.n_params,
                "n_train_lists": fit.n_train_lists, "history": fit.history}
        return (lambda b: tr.final_scores(fit, b, self.d_ix)), info, (tr, fit)


def devfit(w: World6, device: str = "cuda") -> pd.DataFrame:
    kit = Kit(w, device)
    train, valA, devB, _ = fold_blocks(0, w)
    splits = {"A_adjacent_fold": valA, "B_dev_families": devB}
    rows = []
    for kind, presets in PRESETS.items():
        for pi, preset in enumerate(presets):
            for sname, vblock in splits.items():
                run_id = f"V6DEV__{kind}__p{pi}__{sname.split('_')[0]}"
                meta = {"stage": "devfit", "method": kind, "fold": 0, "seed": 42,
                        "config": preset, "validation_split": sname}
                with tracked(run_id, meta, device=device) as box:
                    scorer, info, _ = kit.fit_learned(kind, preset, train, vblock, 42)
                    va = float(recall_at(vblock, scorer(vblock)).mean())
                    other = devB if sname.startswith("A") else valA
                    vo = float(recall_at(other, scorer(other)).mean())
                    base = float(recall_at(vblock, fusion_scores(vblock, kit.F)).mean())
                    box["extra"] = {"metrics": {"val_recall10": va, "other_split_recall10": vo,
                                                "val_recall10_fusion_F": base},
                                    "fit_info": {k: v for k, v in info.items() if k != "history"}}
                    if "history" in info:
                        dump_json(run_dir(run_id) / "history.json", info["history"])
                rows.append({"run_id": run_id, "kind": kind, "preset": pi,
                             "config": json.dumps(preset), "split": sname,
                             "val_recall10": va, "other_split_recall10": vo,
                             "val_recall10_F": base, **{k: v for k, v in info.items()
                                                        if k in ("selected", "kappa",
                                                                 "best_iteration", "best_epoch")}})
                print(f"  devfit {run_id}: val {va:.4f} (F {base:.4f}) other {vo:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(paths6.MANIFESTS / "development_registry_v6.csv", index=False, encoding="utf-8-sig")
    # one-standard-error rule per class, on the mean over the two splits
    sel = {}
    for kind, sub in df.groupby("kind"):
        agg = sub.groupby("preset")["val_recall10"].agg(["mean", "std"]).reset_index()
        best = agg.sort_values("mean", ascending=False).iloc[0]
        se = float(best["std"]) / np.sqrt(2) if best["std"] == best["std"] else 0.0
        simple = agg[agg["preset"] == SIMPLER_FIRST[kind]].iloc[0]
        pick = int(simple["preset"]) if simple["mean"] >= best["mean"] - se else int(best["preset"])
        mean_F = float(sub["val_recall10_F"].mean())
        sel[kind] = {"preset": pick, "config": PRESETS[kind][pick],
                     "mean_val_recall10": float(agg.loc[agg["preset"] == pick, "mean"].iloc[0]),
                     "mean_val_recall10_F": mean_F,
                     "beats_F_on_dev": bool(float(agg.loc[agg["preset"] == pick, "mean"].iloc[0])
                                            > mean_F + 1e-9)}
    write_manifest("devfit_v6_selection.json", {"selection": sel, "n_fits": int(len(df)),
                                                "rule": "one-standard-error, simpler preset first"})
    return df


# --------------------------------------------------------------------------
# formal five-fold runs
# --------------------------------------------------------------------------
def is_done(run_id: str) -> bool:
    """Resume by content, not by file existence: only a completed run is skipped."""
    p = run_dir(run_id) / "result.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status") == "done"
    except json.JSONDecodeError:
        return False



def score_record(run_id, per, pred_scores, block, extra):
    d = run_dir(run_id)
    per.to_parquet(d / "per_disease.parquet", index=False)
    pd.DataFrame({"disease_id": np.repeat(block.ids, block.N),
                  "gene_row": block.genes.reshape(-1).astype(np.int32),
                  "score": pred_scores.reshape(-1).astype(np.float64),
                  "label": block.y.reshape(-1).astype(np.int8)}).to_parquet(
        d / "predictions.parquet", index=False)
    return {"metrics": macro(per), "n_test_diseases": int(block.D),
            "candidate_hash": candidate_hash(np.repeat(block.ids, block.N), block.genes.reshape(-1)),
            **extra}


def formal(w: World6, learned: list[str], folds=(0, 1, 2, 3, 4), device: str = "cuda",
           seeds=SEEDS) -> None:
    kit = Kit(w, device)
    sel = load_manifest("devfit_v6_selection.json", {}).get("selection", {})
    reg = load_manifest("protocol_v6_registered.json")
    if not reg:
        raise RuntimeError("register the V6 protocol before touching test folds")
    for fold in folds:
        train, val, dev, test = fold_blocks(fold, w)
        prior_scores = test.feat("prior")
        det = {"retrieval_order": test.feat("fused"),
               "degree_prior": prior_scores,
               "F_hand": fusion_scores(test, HAND_WEIGHTS),
               "F4": fusion_scores(test, kit.frozen["F4"]["weights"]),
               "F7": fusion_scores(test, kit.frozen["F7"]["weights"])}
        for name, sc in det.items():
            run_id = f"V6Z__{name}__f{fold}"
            if is_done(run_id):
                continue
            with tracked(run_id, {"stage": "formal", "method": name, "fold": fold,
                                  "seed": "deterministic", "task": "Z"}, device=device) as box:
                per = evaluate(test, sc)
                box["extra"] = score_record(run_id, per, sc, test, {})
        for kind in learned:
            preset = sel.get(kind, {}).get("config", PRESETS[kind][0])
            for seed in seeds:
                run_id = f"V6Z__{kind}__f{fold}__s{seed}"
                if is_done(run_id):
                    continue
                meta = {"stage": "formal", "method": kind, "fold": fold, "seed": seed,
                        "config": preset, "task": "Z", "validation": "adjacent fold F_{j+1}"}
                with tracked(run_id, meta, device=device) as box:
                    scorer, info, obj = kit.fit_learned(kind, preset, train, val, seed)
                    sc = scorer(test)
                    per = evaluate(test, sc)
                    box["extra"] = score_record(run_id, per, sc, test,
                                                {"fit_info": {k: v for k, v in info.items()
                                                              if k != "history"}})
                    if "history" in info:
                        dump_json(run_dir(run_id) / "history.json", info["history"])
                    if kind in ("bilinear", "attention"):
                        tr, fit = obj
                        torch.save({k: v.detach().cpu() for k, v in fit.model.state_dict().items()},
                                   run_dir(run_id) / "model.pt")
                        dump_json(run_dir(run_id) / "standardisers.json",
                                  {"c_mu": tr.c_std.mu.tolist(), "c_sd": tr.c_std.sd.tolist()})
                    if kind == "lightgbm":
                        # the LightGBM C writer cannot open non-ASCII paths; serialise in Python
                        (run_dir(run_id) / "model.txt").write_text(
                            obj.m.booster_.model_to_string(num_iteration=obj.best_iteration),
                            encoding="utf-8")
                print(f"  formal {run_id}: {box['extra']['metrics'].get('recall10_end'):.4f}",
                      flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V6 zero-support line.")
    ap.add_argument("--stage", choices=["tune", "devfit", "formal"], required=True)
    ap.add_argument("--learned", nargs="*", default=["lightgbm", "bilinear", "attention", "lr"])
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = ap.parse_args(argv)
    paths6.ensure()
    w = World6(load_heavy=False)
    t0 = time.time()
    if args.stage == "tune":
        print(json.dumps(tune(w), indent=2, default=str))
    elif args.stage == "devfit":
        df = devfit(w)
        print(df.to_string())
    else:
        formal(w, args.learned, folds=tuple(args.folds), seeds=tuple(args.seeds))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

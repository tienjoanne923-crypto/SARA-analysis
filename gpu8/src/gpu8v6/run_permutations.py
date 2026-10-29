"""Run the P1/P2/P3 suite on the frozen V6 scorers (seed-42 models of each fold)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from . import paths6
from .common import World6, dump_json, load_manifest, now_utc
from .disease_semantics import load as load_sem
from .gene_features import load as load_gene_feats
from .permutation import run_fold, summarise
from .residual import MODELS, ResidualTrainer, Standardiser
from .scoring import design, fusion_scores
from .sources import SourceEngine


def lightgbm_scorer(fold: int, gene_feats):
    import lightgbm as lgb
    txt = (paths6.RUNS / f"V6Z__lightgbm__f{fold}__s42" / "model.txt").read_text(encoding="utf-8")
    bst = lgb.Booster(model_str=txt)
    return lambda b: bst.predict(design(b, gene_feats)).reshape(b.D, b.N)


def residual_scorer(kind: str, fold: int, F: dict, gene_feats, sem, d_ix):
    rd = paths6.RUNS / f"V6Z__{kind}__f{fold}__s42"
    res = json.loads((rd / "result.json").read_text(encoding="utf-8"))
    info = res.get("fit_info", {})
    if info.get("selected") != "residual":
        return (lambda b: fusion_scores(b, F)), False
    tr = ResidualTrainer(F, gene_feats, sem)
    st = json.loads((rd / "standardisers.json").read_text())
    tr.c_std = Standardiser(np.zeros((2, len(st["c_mu"]))))
    tr.c_std.mu, tr.c_std.sd = np.array(st["c_mu"]), np.array(st["c_sd"])
    cfg = res.get("config", {})
    model = MODELS[kind](d_dim=tr.XD.shape[1], g_dim=tr.G.shape[1], c_dim=len(st["c_mu"]), **cfg)
    model.load_state_dict(torch.load(rd / "model.pt", map_location="cuda"))
    model.to("cuda").eval()
    return (lambda b: tr.score(model, b, d_ix)), True


def main() -> int:
    paths6.ensure()
    w = World6()
    eng = SourceEngine(w, with_v5=False)
    gene_feats, _ = load_gene_feats("current")
    sem = load_sem()
    F = load_manifest("scorer_v6_frozen.json")["weights"]
    frames = []
    for fold in range(5):
        scorers = {"F7": ((lambda b: fusion_scores(b, F)), False),
                   "lightgbm": (lightgbm_scorer(fold, gene_feats), False)}
        att, has = residual_scorer("attention", fold, F, gene_feats, sem, w.idx.d_ix)
        scorers["attention_residual"] = (att, has)
        frames.append(run_fold(w, fold, scorers, eng))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(paths6.PERMUTATION / "v6_permutations_test.parquet", index=False)
    av = pd.read_csv(paths6.AUDIT / "availability_per_disease.csv")
    route = dict(zip(av["disease_id"], av["route_zero_support"]))
    summ = summarise(df, route)
    summ.to_csv(paths6.PERMUTATION / "v6_permutations_summary.csv", index=False, encoding="utf-8-sig")
    dump_json(paths6.PERMUTATION / "v6_permutations_summary.json",
              {"generated_utc": now_utc(), "rows": summ.to_dict(orient="records")})
    print(summ.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

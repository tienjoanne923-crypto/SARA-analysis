"""V7.1 section 13 — does the model actually use its modalities?

Behaviour layer (fixed model, fixed candidates, no refit):

* single-modality shutdown: availability a_m set to 0 for one modality, the gate
  renormalises over the rest;
* paired permutations that keep marginal distributions: sequence vectors are
  exchanged between genes of similar length and PPI degree, function vectors
  between genes of similar GO-term count and PPI degree, disease tissue weights
  between diseases with a similar number of tissues (and the same broad-only
  flag), image vectors between genes of the same dominant cell line; gene ids,
  labels and every other channel stay fixed;
* full-background chunk invariance is checked in the unit tests and recorded at
  scoring time.

``dev_modality_check`` applies the permutations on the development families
before registration; a modality is kept only if permuting it lowers the dev
Recall@10 by more than one paired standard error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from gpu8 import paths as p5

from . import paths71
from .common71 import rng_for

N_DRAWS_DEV = 5


def ppi_degree(w) -> np.ndarray:
    p = paths71.CACHES / "ppi_degree.npy"
    if p.exists():
        return np.load(p)
    ppi = pd.read_parquet(p5.TABLES_DIR / "ppi.parquet")
    a, b = ppi.columns[:2]
    cnt = pd.concat([ppi[a], ppi[b]]).astype(str).map(w.idx.g_ix).dropna().astype(int)
    deg = np.bincount(cnt.to_numpy(), minlength=w.n_g).astype(np.float64)
    np.save(p, deg)
    return deg


def _bins(x: np.ndarray, n: int) -> np.ndarray:
    q = np.quantile(x, np.linspace(0, 1, n + 1)[1:-1]) if len(x) else []
    return np.searchsorted(q, x, side="right")


def strata_permutation(keys: np.ndarray, eligible: np.ndarray, rng) -> np.ndarray:
    """Identity outside ``eligible``; a random permutation inside every stratum."""
    perm = np.arange(len(keys))
    for k in np.unique(keys[eligible]):
        idx = np.flatnonzero(eligible & (keys == k))
        if len(idx) > 1:
            perm[idx] = rng.permutation(idx)
    return perm


def gene_strata(pack, w, kind: str) -> tuple[np.ndarray, np.ndarray]:
    deg = np.log1p(ppi_degree(w))
    if kind == "S":
        sq = pd.read_parquet(paths71.TABLES / "sequences_v71.parquet")
        L = np.log1p(sq["length"].to_numpy(dtype=np.float64))
        elig = pack.a["g_sequence_mask"]
        key = _bins(L, 10) * 100 + _bins(deg, 10)
    elif kind == "F":
        n_go = np.log1p(np.diff(pack.A.indptr).astype(np.float64))   # anchor count ~ annotation depth
        elig = pack.a["g_function_mask"]
        key = _bins(n_go, 10) * 100 + _bins(deg, 10)
    elif kind == "I":
        cl = pd.read_parquet(paths71.TABLES / "g_image_cell_line.parquet")["dominant_cell_line"]
        codes = pd.factorize(cl.fillna(""))[0]
        elig = pack.a["g_image_mask"]
        key = codes * 100 + _bins(deg, 5)
    else:
        raise KeyError(kind)
    return key.astype(np.int64), elig.astype(bool)


def disease_strata(pack, w) -> tuple[np.ndarray, np.ndarray]:
    q = pack.a["d_tissue"]
    audit = pd.read_parquet(paths71.TABLES / "d_tissue_audit.parquet").set_index("disease_id")
    ids = [None] * w.n_d
    for d, r in w.idx.d_ix.items():
        ids[r] = d
    broad = audit["only_broad"].reindex(ids).fillna(False).to_numpy().astype(int)
    n_t = (q > 0).sum(axis=1)
    key = np.minimum(n_t, 12) * 10 + broad
    return key.astype(np.int64), pack.a["d_tissue_mask"].astype(bool)


def permuted_tensors(T: dict, kind: str, perm: np.ndarray, device: str) -> dict:
    P = dict(T)
    idx = torch.as_tensor(perm, device=device)
    if kind == "S":
        P["xS"], P["mS"] = T["xS"][idx], T["mS"][idx]
    elif kind == "F":
        P["xF"], P["mF"] = T["xF"][idx], T["mF"][idx]
    elif kind == "I":
        P["xI"], P["mI"] = T["xI"][idx], T["mI"][idx]
    elif kind == "E":                       # disease-side tissue weights
        P["q"], P["mq"] = T["q"][idx], T["mq"][idx]
    return P


def shutdown_tensors(T: dict, kind: str) -> dict:
    P = dict(T)
    if kind == "S":
        P["mS"] = torch.zeros_like(T["mS"])
    elif kind == "F":
        P["mF"] = torch.zeros_like(T["mF"])
    elif kind == "E":
        P["mq"] = torch.zeros_like(T["mq"])
    elif kind == "I":
        P["mI"] = torch.zeros_like(T["mI"])
    return P


def standalone_recall10(R: np.ndarray, V: np.ndarray, ids: list[str], w) -> np.ndarray:
    out = []
    for i, d in enumerate(ids):
        v = np.flatnonzero(V[i])
        top = v[np.lexsort((v, -R[i, v].astype(np.float64)))][:10]
        q = set(w.query_rows(d).tolist())
        out.append(sum(1 for g in top if int(g) in q) / max(len(q), 1))
    return np.array(out)


def test_behaviour(folds=(0, 1, 2, 3, 4), seed: int = 42, draws: int = 3) -> pd.DataFrame:
    """E5 on the test folds: fixed pools, fixed fitted models, perturbed inputs only."""
    import lightgbm as lgb
    from .common71 import dump_json
    from .fusion71 import CORE_FEATS, Assembler, recall10_fast
    from .ledger71 import run_dir
    asm = Assembler()
    run = asm.run
    rows = []
    for j in folds:
        ctx = f"f{j}_outer"
        test = run.w.labelled(run.w.fold(j)["test_diseases"])
        b = asm.block(ctx, test, "core", seed, pool="pool", featset="late")
        bst = lgb.Booster(model_str=(run_dir(f"V71Z__lf_lgb__f{j}__s{seed}") / "model.txt")
                          .read_text(encoding="utf-8"))
        idx = [b.feat_names.index(k) for k in CORE_FEATS]

        def lf_recall(cfe):
            X = b.X.copy()
            for k, c in zip(CORE_FEATS, idx):
                X[:, :, c] = cfe[k]
            s = np.full(b.valid.shape, -np.inf)
            s[b.valid] = bst.predict(X[b.valid])
            return recall10_fast(b, s)

        T = run.tensors("hpa", False)
        base = asm.core_feature_arrays(("core", seed), ctx, b.ids, b.genes)
        lf0 = lf_recall(base)
        sa0 = standalone_recall10(*base["_standalone"], b.ids, run.w)
        rows.append({"fold": j, "perturbation": "none", "modality": "-", "draw": 0,
                     "lf_recall10": lf0.mean(), "core_recall10": sa0.mean(),
                     "lf_drop": 0.0, "core_drop": 0.0})
        per_rows = {("none", "-"): (lf0, sa0)}
        for kind in ("S", "F", "E"):
            # single modality only: the other two switched off
            P = T
            for other in ("S", "F", "E"):
                if other != kind:
                    P = shutdown_tensors(P, other)
            cfe = asm.core_feature_arrays(("core", seed), ctx, b.ids, b.genes, T_override=P)
            lf, sa = lf_recall(cfe), standalone_recall10(*cfe["_standalone"], b.ids, run.w)
            rows.append({"fold": j, "perturbation": "only", "modality": kind, "draw": 0,
                         "lf_recall10": lf.mean(), "core_recall10": sa.mean(),
                         "lf_drop": (lf0 - lf).mean(), "core_drop": (sa0 - sa).mean()})
            per_rows[("only", kind)] = (lf, sa)
        for kind in ("S", "F", "E"):
            cfe = asm.core_feature_arrays(("core", seed), ctx, b.ids, b.genes,
                                          T_override=shutdown_tensors(T, kind))
            lf, sa = lf_recall(cfe), standalone_recall10(*cfe["_standalone"], b.ids, run.w)
            rows.append({"fold": j, "perturbation": "shutdown", "modality": kind, "draw": 0,
                         "lf_recall10": lf.mean(), "core_recall10": sa.mean(),
                         "lf_drop": (lf0 - lf).mean(), "core_drop": (sa0 - sa).mean()})
            per_rows[("shutdown", kind)] = (lf, sa)
            acc_lf, acc_sa = [], []
            for k in range(draws):
                rng = rng_for("test_perm", kind, j, k)
                key, elig = (disease_strata(run.pack, run.w) if kind == "E"
                             else gene_strata(run.pack, run.w, kind))
                P = permuted_tensors(T, kind, strata_permutation(key, elig, rng), run.device)
                cfe = asm.core_feature_arrays(("core", seed), ctx, b.ids, b.genes, T_override=P)
                lf, sa = lf_recall(cfe), standalone_recall10(*cfe["_standalone"], b.ids, run.w)
                acc_lf.append(lf)
                acc_sa.append(sa)
                rows.append({"fold": j, "perturbation": "paired_permutation", "modality": kind,
                             "draw": k, "lf_recall10": lf.mean(), "core_recall10": sa.mean(),
                             "lf_drop": (lf0 - lf).mean(), "core_drop": (sa0 - sa).mean()})
            per_rows[("paired_permutation", kind)] = (np.mean(acc_lf, 0), np.mean(acc_sa, 0))
        pd.DataFrame({f"{p}|{m}|{which}": v[i] for (p, m), v in per_rows.items()
                      for i, which in enumerate(("lf", "core"))} | {"disease_id": b.ids}).to_parquet(
            paths71.BEHAVIOR / f"e5_per_disease_f{j}.parquet", index=False)
        print(f"  E5 fold {j} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(paths71.BEHAVIOR / "e5_behaviour_test.csv", index=False, encoding="utf-8-sig")
    return df


def e8_behaviour(folds=(0, 1, 2, 3, 4), draws: int = 3) -> pd.DataFrame:
    """Exploratory E8 checks on the outer test folds (seed 42, standalone core variants).

    image: g_image rows exchanged between genes of the same dominant cell line;
    mouse: mouse-score columns exchanged between genes with a similar number of IMPC calls.
    """
    from .chain71 import Runner, variant_spec
    from .train71 import CoreTrainer
    run = Runner()
    rows = []
    for variant, kind in (("core_image", "I"), ("core_mouse", "M")):
        spec = variant_spec(variant)
        for j in folds:
            rid = f"V71CHAIN__{variant}__f{j}_outer__s42"
            if not (paths71.RUNS / rid / "model.pt").exists():
                continue
            tr, m = run.model(rid, spec)
            test = run.w.labelled(run.w.fold(j)["test_diseases"])
            drows = np.array([run.w.idx.d_ix[d] for d in test])
            R, V = tr.score(m, drows)
            base = standalone_recall10(R, V, test, run.w)
            for k in range(draws):
                rng = rng_for("e8_perm", kind, j, k)
                T = tr.T
                if kind == "I":
                    key, elig = gene_strata(run.pack, run.w, "I")
                    P = permuted_tensors(T, "I", strata_permutation(key, elig, rng), run.device)
                else:
                    has = T["mM"].any(dim=0).cpu().numpy()
                    # stratify by the gene's mean match over diseases (a proxy of its
                    # phenotype-profile breadth, i.e. how many MP calls it carries)
                    z = np.load(paths71.CACHES / "mouse_score_v71.npz")
                    depth = np.zeros(run.w.n_g)
                    depth[z["genes"]] = np.asarray(z["S"], dtype=np.float32).mean(axis=0)
                    key = _bins(depth, 10)
                    perm = strata_permutation(key.astype(np.int64), has.astype(bool), rng)
                    idx = torch.as_tensor(perm, device=run.device)
                    P = dict(T)
                    P["sM"], P["mM"] = T["sM"][:, idx], T["mM"][:, idx]
                Rp, Vp = CoreTrainer(P, run.device).score(m, drows)
                pr = standalone_recall10(Rp, Vp, test, run.w)
                rows.append({"variant": variant, "fold": j, "draw": k, "normal": base.mean(),
                             "permuted": pr.mean(), "drop": (base - pr).mean()})
            print(f"  E8 {variant} fold {j}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(paths71.BEHAVIOR / "e8_permutation_test.csv", index=False, encoding="utf-8-sig")
    return df


def dev_modality_check(run, sel_run: str, pick: dict, dev_rows: np.ndarray, dev_lab: dict) -> dict:
    from .chain71 import dev_recall_per_disease
    from .core71 import CoreConfig
    from .train71 import CoreTrainer, TrainConfig
    spec = {"kind": "core", "core": CoreConfig(rank=pick["rank"], gate=pick["gate"]),
            "train": TrainConfig(lambda_A=pick["lambda_A"]), "expression": "hpa",
            "anchor_rule": "all", "image": False, "mouse": False}
    tr, m = run.model(sel_run, spec)
    T = run.tensors("hpa", False)
    R, V = tr.score(m, dev_rows)
    base = dev_recall_per_disease(R, V, dev_rows, dev_lab)
    res = {"normal_dev_recall10": float(base.mean()), "draws": N_DRAWS_DEV}
    keep = {}
    for kind in ("S", "F", "E"):
        drops = []
        for k in range(N_DRAWS_DEV):
            rng = rng_for("dev_perm", kind, k)
            if kind == "E":
                key, elig = disease_strata(run.pack, run.w)
            else:
                key, elig = gene_strata(run.pack, run.w, kind)
            perm = strata_permutation(key, elig, rng)
            P = permuted_tensors(T, kind, perm, run.device)
            trp = CoreTrainer(P, run.device)
            Rp, Vp = trp.score(m, dev_rows)
            drops.append(base - dev_recall_per_disease(Rp, Vp, dev_rows, dev_lab))
        d = np.mean(drops, axis=0)
        se = d.std(ddof=1) / np.sqrt(len(d))
        S = shutdown_tensors(T, kind)
        Rs, Vs = CoreTrainer(S, run.device).score(m, dev_rows)
        off = base - dev_recall_per_disease(Rs, Vs, dev_rows, dev_lab)
        res[kind] = {"perm_mean_drop": float(d.mean()), "perm_se": float(se),
                     "perm_share_of_draws_lower": float(np.mean([x.mean() > 0 for x in drops])),
                     "shutdown_drop": float(off.mean())}
        keep[kind] = bool(d.mean() > se)
    res["keep"] = keep
    res["rule"] = "keep a modality iff its paired permutation lowers dev Recall@10 by > 1 paired SE"
    return res

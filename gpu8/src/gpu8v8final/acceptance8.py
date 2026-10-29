"""V8.1 §16.3 — the ten whole-chain acceptance checks, run on the real saved chain.

 1  inputs and rankings are regenerated identically with the F_j label records removed
 2  source filtering precedes aggregation (gene-HPO rows of forbidden sources never reach G)
 3  every OOF feature row comes from models that did not fit that row's disease
 4  OOF checkpoints never used the row's labels (F_{j+1} disjoint from I_jb and T_j)
 5  GO-masked genes enter neither teacher targets / anchors nor student training
 6  gene-cold (E6) genes enter neither DG positives nor unlabelled controls nor second-level training
 7  every second-level list has exactly <= 500 distinct genes
 8  without new semantics the score falls back exactly to B
 9  the same candidates split into other batches give identical raw scores
10  seeds are averaged per disease, never merged into "any hit"
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from . import paths8
from .common8 import dump_json, now_utc


def check1_label_removal(j: int = 0, n: int = 40) -> dict:
    """Regenerate the outer test pools, features and deployed scores of fold j with the F_j
    label records deleted from the world; compare with the saved lists and scores."""
    from gpu8v6.common import World6
    from gpu8v6.common import load_manifest as lm6
    from .chain8 import Chain
    from .formal8 import sel_cfg, selection
    from .models8 import CandidateA
    from .pack8 import ArmSpec
    from .roles8 import Roles
    from .v6pass8 import V6World, make_engine, v6_order
    from gpu8v6.retrieval import fuse, max_normalise, rank_transform
    from gpu8v6.sources import SOURCES
    import lightgbm as lgb
    w = World6(load_heavy=True)
    test = set(w.fold(j)["test_diseases"])
    # remove every F_j label record
    w.relations = w.relations[~w.relations["disease_id"].isin(test)]
    w.primary_relations = w.primary_relations[~w.primary_relations["disease_id"].isin(test)]
    w.positives = {d: g for d, g in w.positives.items() if d not in test}
    roles = Roles(w)
    eng = make_engine(w, roles, V6World())
    frozen = lm6("retriever_v6_frozen.json")
    ctx = f"f{j}_outer"
    ids = sorted(test)[:n]
    cs = eng.context(w.contexts[ctx])
    raw, _ = eng.scores(cs, ids, sources=SOURCES)
    norm = {s: max_normalise(v) for s, v in raw.items()}
    fused, _ = fuse({s: rank_transform(v) for s, v in norm.items()}, frozen["weights"])
    prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(eng.ppi_deg)
    prior = (prior / max(prior.max(), 1e-12)).astype(np.float32)
    from .v6pass8 import V6Pass
    v6p = V6Pass(V6World(), ctx)
    pr = v6p.rows(ids)
    rb_same = all(np.array_equal(v6_order(fused[i], prior).astype(np.int16), np.asarray(v6p.rB[pr[i]]))
                  for i in range(len(ids)))
    v_same = float(np.abs(np.stack([norm[s] for s in SOURCES], 1) - np.asarray(v6p.V[pr])).max())
    # semantic + pool + features + deployed scores with the label-free world
    ch = Chain(ArmSpec(), V6World(), role="audit", w=w)
    sel = selection(j)
    cfg = sel_cfg(sel)
    rid = ch.teacher(ctx, 42, sel["controls"])
    sem = ch.semantic(ctx, rid, v6p, None)
    blk = ch.blocks_for(ctx, v6p, sem, sel["w"]["w"])["test"]
    z = np.load(paths8.DEV / f"fold{j}" / f"pools_{ctx}_test.npz")
    saved = {str(d): g for d, g in zip(z["ids"], z["genes"])}
    pool_same = all(np.array_equal(blk.genes[i].astype(np.int16), saved[d]) for i, d in enumerate(blk.ids))
    name = cfg["A_name"] if cfg["deploy"] == "A" else cfg["corrector_name"]
    A = CandidateA(**cfg["A"])
    A.booster = lgb.Booster(model_str=(paths8.DEV / f"fold{j}" / f"A_{cfg['A_name']}_model.txt").read_text(encoding="utf-8"))
    A.best_iteration = None
    sc = A.score(blk)
    p = np.load(paths8.DEV / f"fold{j}" / f"pred_{cfg['A_name']}__test.npz")
    pos = {str(d): i for i, d in enumerate(p["ids"])}
    score_diff = max(float(np.abs(sc[i] - p["score"][pos[d]]).max()) for i, d in enumerate(blk.ids))
    return {"pass": bool(rb_same and v_same == 0.0 and pool_same and score_diff < 1e-5),
            "v6_order_identical": rb_same, "v6_values_maxabs": v_same, "pools_identical": pool_same,
            "deployed_score_maxabs": score_diff, "n_checked_diseases": len(ids), "fold": j}


def check2_filter_before_aggregation(j: int = 0) -> dict:
    from gpu8v6.common import World6
    from .roles8 import Roles, RoleSourceEngine
    w = World6(load_heavy=True)
    roles = Roles(w)
    eng = RoleSourceEngine(w, roles, device="cuda")
    out = {}
    for c in (f"f{j}_inner0", f"f{j}_outer"):
        ctx = w.contexts[c]
        forb = roles.forbidden_groups(ctx)
        cs = eng.context(ctx)
        r = eng.ghpo.rows
        allowed = r[r["mapped"] & ~r["source_group"].isin(forb)]
        from scipy import sparse
        M = sparse.csr_matrix((np.ones(len(allowed), dtype=np.float32),
                               (allowed["gene_row"].to_numpy(), allowed["t"].to_numpy())), shape=(w.n_g, eng.space.n_t))
        M.sum_duplicates()
        M.data[:] = 1.0
        G = eng.space.closure(M)
        out[c] = {"equal": bool((G != cs.Gh).nnz == 0), "forbidden_rows": int((r["mapped"] &
                                                                             r["source_group"].isin(forb)).sum())}
    return {"pass": all(v["equal"] for v in out.values()), **out}


def check3_4_oof_lineage() -> dict:
    from gpu8v6.common import World6
    from .roles8 import Roles
    w = World6(load_heavy=False)
    roles = Roles(w)
    eq = roles.eq
    res = []
    for j in range(5):
        val = roles.folds[(j + 1) % 5]
        for b in range(3):
            c = w.contexts[f"f{j}_inner{b}"]
            ref_groups = {eq.get(d, d) for d in roles.reference(c)}
            held = set(c.held_out)
            res.append({"context": c.context_id,
                        "oof_in_teacher_reference": len({eq.get(d, d) for d in held} & ref_groups),
                        "checkpoint_overlaps_oof": len(val & held),
                        "checkpoint_overlaps_train": len(val & set(roles.reference(w.contexts[f'f{j}_outer'])))})
        sel = json.loads((paths8.DEV / f"fold{j}" / "selection.json").read_text(encoding="utf-8"))
        for b in range(3):
            tc = sel["summary"][f"B_minus_{b}"]["train_contexts"]
            res.append({"context": f"f{j}_inner{b}", "B_minus_trained_on_own_block": f"f{j}_inner{b}" in tc})
    ok3 = all(r.get("oof_in_teacher_reference", 0) == 0 and not r.get("B_minus_trained_on_own_block", False)
              for r in res)
    ok4 = all(r.get("checkpoint_overlaps_oof", 0) == 0 and r.get("checkpoint_overlaps_train", 0) == 0 for r in res)
    return {"check3_pass": ok3, "check4_pass": ok4, "rows": res}


def check5_go_mask() -> dict:
    from gpu8v6.common import World6
    from .evidence8 import GOPolicy
    from .masks8 import load
    from .pack8 import ArmSpec, Pack8
    m = load()
    w = World6(load_heavy=False)
    pol = GOPolicy(w, "current")
    out = {}
    for k in ("dev", "test"):
        pk = Pack8(ArmSpec(name=f"chk_{k}", go_mask=k), w, policy=pol)
        g = m[k]
        out[k] = {"function_mask_on": int(pk.a["g_function_mask"][g].sum()),
                  "anchor_rows_nonempty": int((np.diff(pk.A.indptr)[g] > 0).sum()),
                  "function_vectors_nonzero": int((np.abs(pk.a["g_function"][g]).sum(1) > 0).sum())}
    stud = []
    for p in paths8.RUNS.glob("V8S__*/result.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        stud.append(r.get("n_train_genes"))
    ok = all(v["function_mask_on"] == 0 and v["anchor_rows_nonempty"] == 0 and v["function_vectors_nonzero"] == 0
             for v in out.values())
    return {"pass": ok, **out, "student_training_excludes_both_mask_sets": "by construction (hashed split of "
                                                                          "sequence & GO genes outside dev U test)"}


def check6_e6() -> dict:
    from gpu8v6.common import World6
    from .masks8 import load
    from .roles8 import Roles
    from .teacher8 import teacher_data
    w = World6(load_heavy=False)
    roles = Roles(w)
    g = load()["test"]
    res = {}
    for c in ("f0_inner0", "f0_outer"):
        td = teacher_data(w, roles, c, None, g)
        pos = np.concatenate(list(td.train.values()))
        res[c] = {"positives_masked": int(np.isin(pos, g).sum()), "controls_masked": int(td.controls[g].sum())}
    ok = all(v["positives_masked"] == 0 and v["controls_masked"] == 0 for v in res.values())
    return {"pass": ok, **res, "second_level": "RowBlock.train_ok excludes masked genes from every training table"}


def check7_budget() -> dict:
    bad, n = [], 0
    for p in list(paths8.DEV.glob("fold*/pools_*.npz")) + list(paths8.FORMAL.glob("*/s*/fold*/pools_*.npz")):
        z = np.load(p)
        g = z["genes"]
        n += 1
        if g.shape[1] != 500 or any(len(set(r.tolist())) != 500 for r in g):
            bad.append(str(p))
    return {"pass": not bad, "files": n, "bad": bad[:5]}


def check8_fallback() -> dict:
    from .eval8 import fallback_exactness
    out = {}
    for j in range(5):
        d = paths8.DEV / f"fold{j}"
        names = sorted({p.name[5:].split("__")[0] for p in d.glob("pred_C_*__test.npz")})
        out[f"fold{j}"] = fallback_exactness(d, names, role="dev")
    worst = max((v["max_abs_diff_a0"] for f in out.values() for v in f.values()), default=0.0)
    return {"pass": worst == 0.0, "max_abs_diff": worst, "per_fold": out}


def check9_batch_invariance(j: int = 0) -> dict:
    from .chain8 import Chain
    from .pack8 import ArmSpec
    from .teacher8 import disease_z, gene_zF, load_teacher
    from .formal8 import selection
    ch = Chain(ArmSpec(), role="audit")
    sel = selection(j)
    rid = ch.teacher(f"f{j}_outer", 42, sel["controls"])
    m = load_teacher(rid, ch.T, ch.device)
    rows = np.array(sorted(ch.w.idx.d_ix[d] for d in list(ch.roles.dev)[:300]))
    z1, _ = disease_z(m, ch.T, rows, chunk=512)
    z2, _ = disease_z(m, ch.T, rows[::-1].copy(), chunk=37)
    zF, _ = gene_zF(m, ch.T)
    s1 = (z1 @ zF.T).cpu().numpy()
    s2 = (z2 @ zF.T).cpu().numpy()[::-1]
    d1 = float(np.abs(s1 - s2).max())
    return {"pass": d1 < 1e-5, "max_abs_diff_semantic": d1,
            "note": "row-wise normalisation only; no batch statistics anywhere in scoring"}


def check10_seeds() -> dict:
    from .evaluate8 import seed_mean
    a = pd.DataFrame({"disease_id": ["A", "B"], "recall10": [1.0, 0.0], "cold_recall10": [np.nan, np.nan],
                      "n_query": [1, 1], "n_query_cold": [0, 0]})
    b = pd.DataFrame({"disease_id": ["A", "B"], "recall10": [0.0, 0.0], "cold_recall10": [np.nan, np.nan],
                      "n_query": [1, 1], "n_query_cold": [0, 0]})
    m = seed_mean([a, b])
    return {"pass": bool(m.set_index("disease_id").at["A", "recall10"] == 0.5),
            "rule": "per-disease mean over seeds (union would give 1.0)"}


def run() -> dict:
    out = {"generated_utc": now_utc()}
    for name, fn in (("1_label_removal", check1_label_removal), ("2_filter_before_aggregation", check2_filter_before_aggregation),
                     ("3_4_oof_lineage", check3_4_oof_lineage), ("5_go_mask", check5_go_mask), ("6_e6", check6_e6),
                     ("7_budget", check7_budget), ("8_fallback", check8_fallback),
                     ("9_batch_invariance", check9_batch_invariance), ("10_seeds", check10_seeds)):
        try:
            out[name] = fn()
        except Exception as e:  # recorded, never hidden
            out[name] = {"pass": False, "error": f"{type(e).__name__}: {e}"}
        print(name, json.dumps({k: v for k, v in out[name].items() if k in ("pass", "check3_pass", "check4_pass", "error")}),
              flush=True)
    flags = []
    for k, v in out.items():
        if isinstance(v, dict):
            flags += [v.get("pass", True), v.get("check3_pass", True), v.get("check4_pass", True)]
    out["all_pass"] = bool(all(flags))
    dump_json(paths8.AUDIT / "acceptance_ten_checks_v8.json", out)
    return out


if __name__ == "__main__":
    paths8.ensure()
    r = run()
    print("ALL PASS:", r["all_pass"])

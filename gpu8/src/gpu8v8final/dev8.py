"""V8.1 development selection (seed 42; development families and F_{j+1} only).

Order (dependency graph):
  1. controls   per outer context: teacher with reference vs background controls,
                chosen by the development s_F full-background Recall@10;
  2. teachers   the three inner teachers of every fold with c_j;
  3. devmask    development-masked arm (G_mask,dev GO removed upstream): outer
                teacher + four student configurations; acceptance per context;
  4. chain      per fold: students (if accepted), w_j by development coverage,
                strict pools, B (4 fits), six correctors, six A, A/B rule.
No F_j label is read.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from . import paths8
from .chain8 import Chain, ab_rule, ctx_list, fold_models
from .common8 import dump_json, load_manifest, now_utc, write_manifest
from .pack8 import ArmSpec
from .pool8 import W_GRID, strict_pool
from .student8 import KINDS, SRC_GO, WD_LEVELS, gene_level_check, shuffled_inputs
from .teacher8 import disease_z, gene_zF, load_teacher, topk_recall
from .v6pass8 import V6Pass, V6World

SEED = 42
PROVISIONAL_W = 0.5


def dev_labels(ch: Chain) -> dict:
    d_ix = ch.w.idx.d_ix
    return {d_ix[d]: ch.w.query_rows(d) for d in sorted(ch.roles.dev) if d in d_ix and len(ch.w.query_rows(d))}


# ---------------------------------------------------------------------------
def stage_controls(ch: Chain) -> dict:
    lab = dev_labels(ch)
    rows = np.array(sorted(lab))
    out = {}
    for j in range(5):
        res = {}
        for c in ("reference", "background"):
            rid = ch.teacher(f"f{j}_outer", SEED, c, role="dev")
            m = load_teacher(rid, ch.T, ch.device)
            z, ok = disease_z(m, ch.T, rows)
            zF, mF = gene_zF(m, ch.T)
            res[c] = float(topk_recall(z, ok, zF, mF, rows, lab).mean())
        pick = "reference" if res["reference"] >= res["background"] else "background"
        out[f"f{j}_outer"] = {"dev_sF_recall10": res, "controls": pick}
        print(f"  f{j}_outer controls: {res} -> {pick}", flush=True)
    rep = {"generated_utc": now_utc(), "per_context": out, "seed": SEED,
           "rule": "higher development s_F full-background Recall@10; tie -> reference (V7.1 default)"}
    write_manifest("dev_teacher_controls_v8.json", rep)
    return rep


def controls_of(j: int) -> str:
    sel = load_manifest("dev_teacher_controls_v8.json")
    return sel["per_context"][f"f{j}_outer"]["controls"]


def stage_teachers(ch: Chain, seed: int = SEED, folds=range(5), role: str = "dev") -> None:
    for j in folds:
        for c in ctx_list(j):
            ch.teacher(c, seed, controls_of(j), role=role)


# ---------------------------------------------------------------------------
def coverage(genes: np.ndarray, qsets: list[set], sub: list[set] | None = None) -> tuple[float, float, int]:
    """(macro coverage of all positives, micro coverage of the ``sub`` positives, n sub positives)."""
    cov, hit_s, n_s = [], 0, 0
    for i, q in enumerate(qsets):
        g = set(genes[i].tolist())
        cov.append(len(q & g) / max(len(q), 1))
        if sub is not None:
            hit_s += len(sub[i] & g)
            n_s += len(sub[i])
    return float(np.mean(cov)), (hit_s / n_s if n_s else float("nan")), n_s


def stage_devmask() -> dict:
    """Student acceptance on the development-masked arm, per outer context."""
    arm = ArmSpec(name="devmask", go_mask="dev")
    ch = Chain(arm, role="dev")
    main = Chain(ArmSpec(), role="dev", w=ch.w)            # true GO for the gene-level check only
    masks = ch.masks
    dev_mask = set(masks["dev"].tolist())
    lengths = __import__("pandas").read_parquet(paths8.V71_TABLES / "sequences_v71.parquet")["length"].to_numpy()
    out = {}
    for j in range(5):
        ctx = f"f{j}_outer"
        rid = ch.teacher(ctx, SEED, controls_of(j), role="dev")
        tm = load_teacher(rid, ch.T, ch.device)
        v6p = V6Pass(V6World(), ctx)
        ids = [d for d in sorted(ch.roles.dev) if d in v6p.pos]
        q = [set(ch.w.query_rows(d).tolist()) for d in ids]
        qm = [x & dev_mask for x in q]
        pr = v6p.rows(ids)
        rB = np.asarray(v6p.rB[pr]).astype(np.int32)

        def pool_cov(student):
            sem = ch.semantic(ctx, rid, v6p, student)
            sp = {d: i for i, d in enumerate(sem.ids)}
            sr = np.array([sp[d] for d in ids])
            P = strict_pool(rB, sem.rS[sr], sem.m[sr], PROVISIONAL_W)
            return coverage(P["genes"], q, qm)

        base = pool_cov(None)
        rows = [{"config": "none", "coverage": base[0], "masked_coverage": base[1], "n_masked_pos": base[2]}]
        # true teacher vectors of the masked genes (evaluation only): arm teacher on the unmasked GO
        with torch.no_grad():
            from gpu8v71.core71 import row_norm
            zF_true = row_norm(tm.P_T(main.T["xF"])) * main.T["mF"][:, None].float()
        for kind in KINDS:
            for wd in WD_LEVELS:
                zh, imp = ch.student(ctx, SEED, rid, kind, wd, role="dev")
                c = pool_cov((zh, imp))
                # paired destruction: the same student on length-stratum-shuffled sequences
                from .student8 import Student
                sd = paths8.RUNS / f"V8S__devmask__{ctx}__{kind}_wd{wd:g}__s{SEED}" / "student.pt"
                sm = Student(kind).to(ch.device)
                sm.load_state_dict(torch.load(sd, map_location=ch.device, weights_only=True))
                sm.eval()
                Xs = shuffled_inputs(ch.pack.a["g_sequence"], imp, lengths, f"{ctx}|{kind}|{wd}")
                with torch.no_grad():
                    zs = sm(torch.as_tensor(Xs[imp], device=ch.device)).cpu().numpy()
                cs = pool_cov((zs, imp))
                mk = np.array(sorted(dev_mask & set(imp.tolist())))
                chk = gene_level_check(sm, ch.T["xS"], zF_true, mk,
                                       np.flatnonzero(main.T["mF"].cpu().numpy() & ch.T["mS"].cpu().numpy()))
                rows.append({"config": f"{kind}_wd{wd:g}", "kind": kind, "wd": wd, "coverage": c[0],
                             "masked_coverage": c[1], "shuffled_masked_coverage": cs[1],
                             "shuffled_coverage": cs[0], "gain": c[1] - base[1], "shuffled_gain": cs[1] - base[1],
                             "coverage_delta": c[0] - base[0], "gene_level_masked": chk})
                print(f"  {ctx} {kind} wd{wd:g}: masked cov {c[1]:.4f} (none {base[1]:.4f}, shuffled {cs[1]:.4f}) "
                      f"all {c[0]:.4f} (none {base[0]:.4f}) top10 {chk['top10']:.3f}", flush=True)
        ok = [r for r in rows[1:] if r["gain"] > 0 and r["shuffled_gain"] < r["gain"]
              and r["coverage_delta"] >= -0.005]
        pick = None
        if ok:
            pick = sorted(ok, key=lambda r: (-round(r["gain"], 6), 0 if r["kind"] == "linear" else 1, -r["wd"]))[0]
        out[ctx] = {"rows": rows, "accepted": pick is not None,
                    "config": None if pick is None else {"kind": pick["kind"], "wd": pick["wd"]}}
    rep = {"generated_utc": now_utc(), "provisional_w": PROVISIONAL_W, "per_context": out,
           "rule": "accept iff masked-positive coverage gain > 0, paired (length-stratum) shuffle lowers it, "
                   "and overall development coverage does not drop by more than 0.005; among passing "
                   "configurations the largest gain, ties -> linear, then the larger weight decay"}
    write_manifest("dev_student_selection_v8.json", rep)
    return rep


def student_cfg(j: int):
    sel = load_manifest("dev_student_selection_v8.json")
    r = sel["per_context"][f"f{j}_outer"]
    return r["config"] if r["accepted"] else None


# ---------------------------------------------------------------------------
def select_w(ch: Chain, j: int, sem_outer, v6p) -> dict:
    ids = [d for d in sorted(ch.roles.dev) if d in v6p.pos]
    q = [set(ch.w.query_rows(d).tolist()) for d in ids]
    pr = v6p.rows(ids)
    rB = np.asarray(v6p.rB[pr]).astype(np.int32)
    sp = {d: i for i, d in enumerate(sem_outer.ids)}
    sr = np.array([sp[d] for d in ids])
    res = {}
    for w in W_GRID:
        for plain in (False, True):
            P = strict_pool(rB, sem_outer.rS[sr], sem_outer.m[sr], w, plain=plain)
            res[f"{'plain' if plain else 'aware'}_w{w}"] = coverage(P["genes"], q)[0]
    from .pool8 import v6_list
    res["v6_500"] = coverage(v6_list(rB), q)[0]
    aware = {w: res[f"aware_w{w}"] for w in W_GRID}
    best = max(aware.values())
    w_j = max(w for w, v in aware.items() if v >= best - 1e-12)
    return {"coverage": res, "w": w_j, "rule": "max development coverage@500 of the aware RRF; tie -> larger w"}


def stage_chain(folds=range(5), threads: int = 8) -> None:
    ch = Chain(ArmSpec(), role="dev", threads=threads)
    for j in folds:
        fdir = paths8.DEV / f"fold{j}"
        fdir.mkdir(parents=True, exist_ok=True)
        if (fdir / "selection.json").exists():
            continue
        t0 = time.time()
        c_j = controls_of(j)
        st = student_cfg(j)
        sems, v6ps = {}, {}
        for c in ctx_list(j):
            rid = ch.teacher(c, SEED, c_j, role="dev")
            stu = ch.student(c, SEED, rid, st["kind"], st["wd"], role="dev") if st else None
            v6ps[c] = V6Pass(V6World(), c)
            sems[c] = ch.semantic(c, rid, v6ps[c], stu)
        wsel = select_w(ch, j, sems[f"f{j}_outer"], v6ps[f"f{j}_outer"])
        print(f"  fold {j}: w_j = {wsel['w']}  {wsel['coverage']}", flush=True)
        blocks = {c: ch.blocks_for(c, v6ps[c], sems[c], wsel["w"]) for c in ctx_list(j)}
        for c in ctx_list(j):
            np.save(fdir / f"gene_src_{c}.npy", sems[c].src)
        from .chain8 import save_pools
        save_pools(fdir, blocks)
        del sems
        summ = fold_models(ch, j, SEED, blocks, "main", True, None, fdir)
        dec = ab_rule(summ)
        underfit = summ.get("underfit_trigger_config", [])
        sel = {"fold": j, "controls": c_j, "student": st, "w": wsel, "summary": summ, "ab": dec,
               "underfit_31_leaf_trigger": bool(underfit), "seconds": time.time() - t0,
               "generated_utc": now_utc()}
        dump_json(fdir / "selection.json", sel)
        print(f"  fold {j} decision: {json.dumps(dec)} ({time.time() - t0:.0f}s)", flush=True)
        del blocks
        import gc
        gc.collect()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["controls", "teachers", "devmask", "chain"])
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    paths8.ensure()
    if a.stage == "controls":
        stage_controls(Chain(ArmSpec(), role="dev"))
    elif a.stage == "teachers":
        stage_teachers(Chain(ArmSpec(), role="dev"), folds=a.folds)
    elif a.stage == "devmask":
        stage_devmask()
    else:
        stage_chain(a.folds, a.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

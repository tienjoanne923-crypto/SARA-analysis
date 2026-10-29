"""Formal chains (after registration) and variant arms, with the registered choices.

Every fold uses the configuration chosen for *its own* outer context on the
development families (controls c_j, student decision, w_j, corrector lambda, A
configuration, A/B deployment).  Formal correctors are fitted for beta = 0, 1, 2
at the selected lambda (E2); candidate A at its selected configuration.  The
seed-42 main chain reuses the development fits (identical contract) and records
``reused_from``; comparators are added for every seed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np

from . import paths8
from .chain8 import Chain, ctx_list, fold_models, save_pred
from .common8 import dump_json, load_manifest, now_utc
from .comparators8 import (DirectLGB, attach_core, channel_X, core_features, f7_score, standalone_scores,
                           standalone_top, v71_X)
from .ledger8 import write_result
from .models8 import cold_recall_at, recall_at
from .pack8 import ArmSpec
from .teacher8 import load_teacher
from .v6pass8 import V6Pass, V6World

SEEDS = (42, 52, 62)


def selection(j: int) -> dict:
    return json.loads((paths8.DEV / f"fold{j}" / "selection.json").read_text(encoding="utf-8"))


def sel_cfg(sel: dict) -> dict:
    ab = sel["ab"]
    cor = sel["summary"]["corrector"]
    if ab.get("chosen_B"):
        lam, beta = ab["chosen_B_lam"], ab["chosen_B_beta"]
    else:
        best = sorted(cor, key=lambda r: (-r["dev_cold_recall10"], -r["dev_recall10"], r["beta"], -r["lam"]))[0]
        lam, beta = best["lam"], best["beta"]
    a = next(r for r in sel["summary"]["A"] if r["name"] == ab["best_A"])
    return {"corrector": {"lam": lam, "beta": beta}, "A": {"num_leaves": a["num_leaves"],
                                                          "min_child_samples": a["min_child_samples"]},
            "deploy": ab["deploy"], "corrector_name": f"C_b{beta:g}_l{lam:g}", "A_name": ab["best_A"]}


def out_dir(tag: str, seed: int, j: int):
    base = paths8.FORMAL if tag == "main" else paths8.VARIANTS
    d = base / tag / f"s{seed}" / f"fold{j}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def deployed_copy(d, cfg: dict) -> None:
    src = cfg["corrector_name"] if cfg["deploy"] == "B" else cfg["A_name"]
    for r in ("val", "dev", "test"):
        shutil.copyfile(d / f"pred_{src}__{r}.npz", d / f"pred_deployed__{r}.npz")


def run(tag: str, arm: ArmSpec, v6world: V6World, seeds=SEEDS, folds=range(5), role: str = "formal",
        comparators: bool = True, student_mode: str = "dev", threads: int = 8, table1: bool = False) -> None:
    ch = Chain(arm, v6world, role=role, threads=threads)
    from .dev8 import student_cfg
    for seed in seeds:
        for j in folds:
            d = out_dir(tag, seed, j)
            if (d / "done.json").exists():
                continue
            t0 = time.time()
            sel = selection(j)
            cfg = sel_cfg(sel)
            c_j, w_j = sel["controls"], sel["w"]["w"]
            st = student_cfg(j) if student_mode == "dev" else student_mode
            if isinstance(st, str):
                st = None if st == "off" else {"kind": st.split("_")[0], "wd": float(st.split("wd")[1])}
            sems, v6ps, trids = {}, {}, {}
            for c in ctx_list(j):
                trids[c] = ch.teacher(c, seed, c_j)
                stu = ch.student(c, seed, trids[c], st["kind"], st["wd"]) if st else None
                v6ps[c] = V6Pass(v6world, c)
                sems[c] = ch.semantic(c, trids[c], v6ps[c], stu)
                np.save(d / f"gene_src_{c}.npy", sems[c].src)
            blocks = {c: ch.blocks_for(c, v6ps[c], sems[c], w_j) for c in ctx_list(j)}
            v6b = ({c: ch.blocks_for(c, v6ps[c], sems[c], 0.5, pool="v6") for c in ctx_list(j)}
                   if comparators else None)
            del sems
            import gc
            gc.collect()
            from .chain8 import save_pools
            save_pools(d, blocks)
            reused = None
            if tag == "main" and seed == 42:
                src = paths8.DEV / f"fold{j}"
                for p in src.glob("pred_*.npz"):
                    shutil.copyfile(p, d / p.name)
                summ = sel["summary"]
                reused = paths8.rel(src)
            else:
                summ = fold_models(ch, j, seed, blocks, tag, False, cfg, d)
            deployed_copy(d, cfg)
            if comparators:
                comp = run_comparators(ch, j, seed, blocks, v6ps, v6b, trids, d, table1=table1)
            else:
                comp = {}
            info = {"tag": tag, "arm": arm.__dict__, "v6world": v6world.__dict__, "seed": seed, "fold": j,
                    "config": cfg, "controls": c_j, "w": w_j, "student": st, "reused_from": reused,
                    "summary": summ, "comparators": comp, "seconds": time.time() - t0,
                    "generated_utc": now_utc()}
            dump_json(d / "done.json", info)
            write_result(f"V8F__{tag}__f{j}__s{seed}", {"role": role, "stage": "fold_chain",
                                                        "component": "fold", "arm": arm.key(), "tag": tag,
                                                        "fold": j, "seed": seed, "reused_from": reused,
                                                        "config": cfg, "status": "done",
                                                        "outputs": paths8.rel(d)})
            print(f"  {tag} s{seed} fold {j} done ({time.time() - t0:.0f}s)", flush=True)
            del blocks, v6b
            gc.collect()


def run_comparators(ch: Chain, j: int, seed: int, blocks: dict, v6ps: dict, v6b: dict, trids: dict, d,
                    table1: bool = False) -> dict:
    inner = ctx_list(j)[:3]
    outer = f"f{j}_outer"
    info = {}
    # --- the V6 500 lists of every context (channel-only LightGBM on its own list = comparison 1)
    cl = DirectLGB(channel_X, seed, ch.threads).fit([v6b[c]["oof"] for c in inner], [v6b[outer]["val"]])
    for r in ("val", "dev", "test"):
        save_pred(d, f"channel_lgb_v6__{r}", v6b[outer][r], cl.score(v6b[outer][r]))
        save_pred(d, f"F7_v6__{r}", v6b[outer][r], f7_score(v6b[outer][r]))
        save_pred(d, f"F7_strict__{r}", blocks[outer][r], f7_score(blocks[outer][r]))
    info["channel_lgb_v6"] = {"best_iteration": cl.best_iteration}
    # --- V7.1 direct late fusion (30 channels + 14 core features, zeros for missing) on the strict pool
    for c in ctx_list(j):
        m = load_teacher(trids[c], ch.T, ch.device)
        for role, blk in blocks[c].items():
            pools = {"strict": blk.genes}
            if table1:
                pools["v6"] = v6b[c][role].genes
            cf = core_features(m, ch.T, blk.ids, ch.w.idx.d_ix, pools)
            attach_core(blk, cf["strict"])
            if table1:
                attach_core(v6b[c][role], cf["v6"])
    vd = DirectLGB(v71_X, seed, ch.threads).fit([blocks[c]["oof"] for c in inner], [blocks[outer]["val"]])
    for r in ("val", "dev", "test"):
        save_pred(d, f"v71_direct_strict__{r}", blocks[outer][r], vd.score(blocks[outer][r]))
    info["v71_direct_strict"] = {"best_iteration": vd.best_iteration}
    if table1:
        vd6 = DirectLGB(v71_X, seed, ch.threads).fit([v6b[c]["oof"] for c in inner], [v6b[outer]["val"]])
        for r in ("val", "dev", "test"):
            save_pred(d, f"v71_direct_v6__{r}", v6b[outer][r], vd6.score(v6b[outer][r]))
        info["v71_direct_v6"] = {"best_iteration": vd6.best_iteration}
    # --- standalone semantic lists (full background, top 500) of the outer teacher
    m = load_teacher(trids[outer], ch.T, ch.device)
    for kind in ("r", "sF", "sS", "sF0"):
        for r in ("dev", "test"):
            ids = blocks[outer][r].ids
            R, V = standalone_scores(m, ch.T, ids, ch.w.idx.d_ix, kind)
            top = standalone_top(R, V, 500)
            sc = np.take_along_axis(R, np.maximum(top, 0), axis=1)
            sc = np.where(top >= 0, sc, -np.inf)
            np.savez_compressed(d / f"pred_standalone_{kind}__{r}.npz", ids=np.array(ids),
                                genes=np.maximum(top, 0).astype(np.int16), score=sc.astype(np.float32),
                                valid=top >= 0)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--table1", action="store_true")
    a = ap.parse_args(argv)
    paths8.ensure()
    if a.tag == "main":
        if not load_manifest("protocol_v8_registered.json"):
            raise RuntimeError("register V8.1 before the formal chain")
        run("main", ArmSpec(), V6World(), a.seeds, a.folds, "formal", True, threads=a.threads, table1=a.table1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

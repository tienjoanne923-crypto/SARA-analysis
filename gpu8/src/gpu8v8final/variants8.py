"""V8.1 directed experiments and controls (seed 42 unless stated; registered choices only).

E4  evidence:   layer1 (training-visible source exclusion, per context) + matched random;
                layer2 (evaluation-conditioned, audit flag) + matched random; experimental
                GO; no TAS/NAS; no IMP/IGI/HMP/HGI; evidence-block isolation (relations,
                GO, traceable HPO) + matched random.  No-IC and IEA-only are re-scoring
                sensitivities (fitted models kept, GO input swapped at inference).
E5  GO input missing on G_mask,test: no imputation / two-layer student / linear student /
                sequence nearest-neighbour transfer / teacher's own sequence projection /
                two-layer student on shuffled pairs / ProtST-input student (open-knowledge).
E6  true association-supervision cold on G_mask,test (GO kept); and with GO also masked.
Controls: plain RRF with the same w_j; natural chain with the student on; expression off
and random tissue condition; short (V7.1) text and level-2 masked text (teacher level).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import paths8
from .chain8 import Chain, ctx_list, fold_models, save_pools, save_pred
from .common8 import dump_json, load_manifest, now_utc
from .formal8 import deployed_copy, out_dir, sel_cfg, selection
from .ledger8 import write_result
from .pack8 import ArmSpec
from .v6pass8 import WORLDS, V6Pass, V6World


def dev_best_student(j: int, kind: str) -> str:
    """Registered rule for the directed arms: the weight decay with the highest development
    masked-positive coverage for that student kind (even if the student was not accepted)."""
    sel = load_manifest("dev_student_selection_v8.json")
    rows = [r for r in sel["per_context"][f"f{j}_outer"]["rows"] if r.get("kind") == kind]
    best = sorted(rows, key=lambda r: (-r["masked_coverage"], -r["wd"]))[0]
    return f"{kind}_wd{best['wd']:g}"


ARMS = {
    # E4 -----------------------------------------------------------------
    "e4_layer1": dict(arm=ArmSpec(name="e4_layer1", go_layer="layer1")),
    "e4_layer1_random": dict(arm=ArmSpec(name="e4_layer1_random", go_layer="layer1_random")),
    "e4_layer2": dict(arm=ArmSpec(name="e4_layer2", go_layer="layer2", evaluation_conditioned=True)),
    "e4_layer2_random": dict(arm=ArmSpec(name="e4_layer2_random", go_layer="layer2_random",
                                         evaluation_conditioned=True)),
    "e4_experimental": dict(arm=ArmSpec(name="e4_experimental", go_variant="experimental")),
    "e4_no_tas_nas": dict(arm=ArmSpec(name="e4_no_tas_nas", go_variant="no_tas_nas")),
    "e4_no_mutant": dict(arm=ArmSpec(name="e4_no_mutant", go_variant="no_mutant_genetic")),
    "e4_blocks": dict(arm=ArmSpec(name="e4_blocks", go_layer="blocks", evidence_blocks=True), world="blocks"),
    "e4_blocks_random": dict(arm=ArmSpec(name="e4_blocks_random", go_layer="blocks_random",
                                         evidence_blocks=True), world="blocks_random"),
    # E5 (one teacher set, several imputations) -----------------------------
    "e5_none": dict(arm=ArmSpec(name="e5_testmask", go_mask="test"), student="off", tag="e5_none"),
    "e5_mlp": dict(arm=ArmSpec(name="e5_testmask", go_mask="test"), student="best_mlp", tag="e5_mlp"),
    "e5_linear": dict(arm=ArmSpec(name="e5_testmask", go_mask="test"), student="best_linear", tag="e5_linear"),
    "e5_nn": dict(arm=ArmSpec(name="e5_testmask", go_mask="test"), student="nn", tag="e5_nn"),
    "e5_teacher_sS": dict(arm=ArmSpec(name="e5_testmask", go_mask="test"), student="teacher_sS",
                          tag="e5_teacher_sS"),
    "e5_mlp_shuffled": dict(arm=ArmSpec(name="e5_testmask", go_mask="test"), student="best_mlp_shuffled",
                            tag="e5_mlp_shuffled"),
    "e5_protst": dict(arm=ArmSpec(name="e5_testmask", go_mask="test"), student="protst_best_mlp",
                      tag="e5_protst"),
    # E6 -------------------------------------------------------------------
    "e6_cold": dict(arm=ArmSpec(name="e6_cold", e6_cold=True), world="e6cold"),
    "e6_cold_gomask": dict(arm=ArmSpec(name="e6_cold_gomask", e6_cold=True, go_mask="test"), world="e6cold"),
    # controls ---------------------------------------------------------------
    "plain_rrf": dict(arm=ArmSpec(), pool="plain", tag="plain_rrf"),
    "natural_student_on": dict(arm=ArmSpec(), student="best_mlp", tag="natural_student_on"),
    "expr_off": dict(arm=ArmSpec(name="expr_off", expression="none"), tag="expr_off"),
    "expr_random": dict(arm=ArmSpec(name="expr_random", expression="random"), tag="expr_random"),
    "bridge950": dict(arm=ArmSpec(), pool="bridge", tag="bridge950"),
    "v6pool": dict(arm=ArmSpec(), pool="v6", tag="v6pool"),
}


def resolve_student(mode: str, j: int) -> str | None:
    if mode in ("off", None):
        return None
    if mode == "dev":
        from .dev8 import student_cfg
        st = student_cfg(j)
        return None if st is None else f"{st['kind']}_wd{st['wd']:g}"
    out = mode
    for kind in ("mlp", "linear"):
        out = out.replace(f"best_{kind}", dev_best_student(j, kind))
    return out


def run_arm(name: str, seeds=(42,), folds=range(5), threads: int = 8) -> None:
    spec = ARMS[name]
    arm = spec["arm"]
    world = WORLDS[spec.get("world", "main")]
    tag = spec.get("tag", name)
    pool = spec.get("pool", "strict")
    ch = Chain(arm, world, role="variant", threads=threads)
    for seed in seeds:
        for j in folds:
            d = out_dir(tag, seed, j)
            if (d / "done.json").exists():
                continue
            t0 = time.time()
            sel = selection(j)
            cfg = {**sel_cfg(sel), "only_selected": True}
            c_j, w_j = sel["controls"], sel["w"]["w"]
            mode = resolve_student(spec.get("student", "dev"), j)
            sems, v6ps, trids = {}, {}, {}
            for c in ctx_list(j):
                trids[c] = ch.teacher(c, seed, c_j)
                imp = ch.imputation(c, seed, trids[c], mode) if mode else None
                v6ps[c] = V6Pass(world, c)
                sems[c] = ch.semantic(c, trids[c], v6ps[c], imp)
                np.save(d / f"gene_src_{c}.npy", sems[c].src)
            blocks = {c: ch.blocks_for(c, v6ps[c], sems[c], w_j, plain=(pool == "plain"),
                                       pool="strict" if pool == "plain" else pool) for c in ctx_list(j)}
            del sems
            import gc
            gc.collect()
            if pool == "strict" or pool == "plain":
                save_pools(d, blocks)
            summ = fold_models(ch, j, seed, blocks, tag, False, cfg, d)
            deployed_copy(d, cfg)
            info = {"tag": tag, "arm": arm.__dict__, "world": world.__dict__, "seed": seed, "fold": j,
                    "config": cfg, "imputation": mode, "pool": pool, "summary": summ,
                    "evaluation_conditioned_audit": bool(arm.evaluation_conditioned),
                    "seconds": time.time() - t0, "generated_utc": now_utc(),
                    "pack": ch.pack.info}
            dump_json(d / "done.json", info)
            write_result(f"V8V__{tag}__f{j}__s{seed}", {"role": "variant", "stage": "fold_chain", "component": "fold",
                                                        "arm": arm.key(), "tag": tag, "fold": j, "seed": seed,
                                                        "status": "done", "config": cfg,
                                                        "evaluation_conditioned_audit": bool(arm.evaluation_conditioned),
                                                        "outputs": paths8.rel(d)})
            print(f"  {tag} fold {j} done ({time.time() - t0:.0f}s)", flush=True)
            del blocks
            gc.collect()


def arm_channel_lgb(name: str, seed: int = 42, folds=range(5), threads: int = 6) -> None:
    """The old method (channel-only LightGBM on its own V6 500-list) inside an arm whose reference
    graph differs from the main world (evidence-block isolation and its random control)."""
    from .comparators8 import DirectLGB, channel_X
    spec = ARMS[name]
    arm = spec["arm"]
    world = WORLDS[spec.get("world", "main")]
    tag = spec.get("tag", name)
    ch = Chain(arm, world, role="variant", threads=threads)
    for j in folds:
        d = out_dir(tag, seed, j)
        if not (d / "done.json").exists() or (d / "pred_channel_lgb_v6__test.npz").exists():
            continue
        sel = selection(j)
        v6b = {}
        for c in ctx_list(j):
            rid = ch.teacher(c, seed, sel["controls"])
            v6p = V6Pass(world, c)
            sem = ch.semantic(c, rid, v6p, None)
            v6b[c] = ch.blocks_for(c, v6p, sem, 0.5, pool="v6")
            del sem
        outer = f"f{j}_outer"
        cl = DirectLGB(channel_X, seed, threads).fit([v6b[c]["oof"] for c in ctx_list(j)[:3]], [v6b[outer]["val"]])
        for r in ("val", "dev", "test"):
            save_pred(d, f"channel_lgb_v6__{r}", v6b[outer][r], cl.score(v6b[outer][r]))
        print(f"  {tag} fold {j}: channel-only LightGBM on its own V6 500 (it {cl.best_iteration})", flush=True)


def rescore(name: str, variant: str, seed: int = 42, folds=range(5)) -> None:
    """No-IC / IEA-only: main-chain fitted models, GO input swapped at inference (no refit)."""
    from .models8 import BaseRanker, CandidateA, Corrector
    tag = f"e4_rescore_{variant}"
    arm_main = ArmSpec()
    ch_main = Chain(arm_main, V6World(), role="variant")
    ch_var = Chain(ArmSpec(name=f"rescore_{variant}", go_variant=variant), V6World(), role="variant", w=ch_main.w)
    for j in folds:
        d = out_dir(tag, seed, j)
        if (d / "done.json").exists():
            continue
        sel = selection(j)
        cfg = sel_cfg(sel)
        outer = f"f{j}_outer"
        rid = ch_main.teacher(outer, seed, sel["controls"])
        v6p = V6Pass(V6World(), outer)
        # the main teacher's weights, the variant's GO input
        sem = ch_var.semantic(outer, rid, v6p, None)
        blocks = ch_var.blocks_for(outer, v6p, sem, sel["w"]["w"])
        src = paths8.DEV / f"fold{j}"
        B = BaseRanker.load(src / "B_outer")
        C = Corrector.load(src / cfg["corrector_name"])
        import lightgbm as lgb
        A = CandidateA(**cfg["A"])
        A.booster = lgb.Booster(model_str=(src / f"A_{cfg['A_name']}_model.txt").read_text(encoding="utf-8"))
        A.best_iteration = None
        for r in ("dev", "test"):
            b = blocks[r]
            Bs = B.score(b)
            save_pred(d, f"B__{r}", b, Bs)
            save_pred(d, f"{cfg['corrector_name']}__{r}", b, C.score(b, Bs))
            save_pred(d, f"{cfg['A_name']}__{r}", b, A.score(b))
        for r in ("dev", "test"):
            srcn = cfg["corrector_name"] if cfg["deploy"] == "B" else cfg["A_name"]
            import shutil
            shutil.copyfile(d / f"pred_{srcn}__{r}.npz", d / f"pred_deployed__{r}.npz")
        np.save(d / f"gene_src_{outer}.npy", sem.src)
        dump_json(d / "done.json", {"tag": tag, "fold": j, "seed": seed, "variant": variant, "config": cfg,
                                    "rule": "re-scoring sensitivity: fitted main models, GO swapped at inference",
                                    "generated_utc": now_utc()})
        print(f"  {tag} fold {j} done", flush=True)


def teacher_text_arms(seed: int = 42) -> dict:
    """Short (V7.1) text and level-2 masked text: outer teachers, standalone s_F lists (paired, descriptive)."""
    from .comparators8 import standalone_scores, standalone_top
    from .teacher8 import load_teacher
    out = {}
    for name, arm in (("text_short", ArmSpec(name="text_short", text="short")),
                      ("text_level2", ArmSpec(name="text_level2", text="seg_l2"))):
        ch = Chain(arm, V6World(), role="variant")
        for j in range(5):
            d = out_dir(name, seed, j)
            if (d / "done.json").exists():
                continue
            sel = selection(j)
            rid = ch.teacher(f"f{j}_outer", seed, sel["controls"])
            m = load_teacher(rid, ch.T, ch.device)
            ids_all = ch.roles.targets(ch.w.contexts[f"f{j}_outer"])
            for r in ("dev", "test"):
                ids = [x for x in ids_all[r] if x in ch.w.idx.d_ix]
                R, V = standalone_scores(m, ch.T, ids, ch.w.idx.d_ix, "sF")
                top = standalone_top(R, V, 500)
                sc = np.where(top >= 0, np.take_along_axis(R, np.maximum(top, 0), axis=1), -np.inf)
                np.savez_compressed(d / f"pred_standalone_sF__{r}.npz", ids=np.array(ids),
                                    genes=np.maximum(top, 0).astype(np.int16), score=sc.astype(np.float32),
                                    valid=top >= 0)
            dump_json(d / "done.json", {"tag": name, "fold": j, "seed": seed, "teacher": rid,
                                        "generated_utc": now_utc()})
            print(f"  {name} fold {j} done", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="*", default=[42])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    paths8.ensure()
    if not load_manifest("protocol_v8_registered.json"):
        raise RuntimeError("register V8.1 before running the directed experiments")
    for name in a.arms:
        if name.startswith("oldmethod:"):
            arm_channel_lgb(name.split(":", 1)[1], folds=a.folds, threads=a.threads)
        elif name == "text":
            teacher_text_arms()
        elif name in ("rescore_no_ic", "rescore_iea_only"):
            rescore(name, name.replace("rescore_", ""), folds=a.folds)
        else:
            run_arm(name, tuple(a.seeds), a.folds, a.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

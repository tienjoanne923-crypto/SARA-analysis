"""V8.1 §10 — the whole-chain OOF runner for one arm and one seed.

Per outer fold j (contexts f{j}_inner0..2 and f{j}_outer):

  teacher   4 fits (controls c_j chosen on the development families),
            checkpoint = s_F Recall@10 on F_{j+1} scored in the same context;
  student   4 fits when accepted for context j (development-masked arm);
  semantic  z_d^T z^sem_g for every target of the context;
  pool      availability-aware RRF with w_j (development choice), exactly 500;
  blocks    row features from the row's own context;
  B         B^{-b} on the other two inner blocks (checkpoint F_{j+1} in inner b)
            scores I_jb; B_outer on all three blocks (checkpoint F_{j+1} outer)
            scores F_j / F_{j+1} / development;
  corrector trained on T_j OOF rows with B^{-b}; checkpoint F_{j+1} (outer, B_outer);
  A         direct LightGBM on the same pools with explicit NaN;
  output    scores of every method on F_j (sealed until registration), F_{j+1}, dev.

Nothing in this module reads an F_j label: test blocks are built without labels.
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from gpu8v6.common import World6

from . import paths8
from .common8 import dump_json, load_manifest, now_utc, write_manifest
from .feats8 import GeneStatic, Labels, RowBlock, build_block
from .ledger8 import is_done, run_dir, tracked
from .masks8 import load as load_masks
from .models8 import (A_GRID, C_GRID, BaseRanker, CandidateA, Corrector, CorrectorConfig, cold_recall_at,
                      recall_at, reload_check)
from .pack8 import ArmSpec, Pack8
from .pool8 import pool_hash, strict_pool, v6_list
from .roles8 import Roles
from .semantic8 import gene_semantic, random_tissue, semantic_for
from .student8 import SRC_GO, KINDS, WD_LEVELS, hashed_split, train_student
from .teacher8 import load_teacher, run_teacher, teacher_data, teacher_run_id
from .v6pass8 import V6Pass, V6World

SEEDS = (42, 52, 62)
N_G = 19861


def ctx_list(j: int) -> list[str]:
    return [f"f{j}_inner0", f"f{j}_inner1", f"f{j}_inner2", f"f{j}_outer"]


def save_pools(d, blocks: dict) -> None:
    """Label-free pool lists of every context and role (ids, genes, semantic source)."""
    for c, roles in blocks.items():
        for r, b in roles.items():
            np.savez_compressed(d / f"pools_{c}_{r}.npz", ids=np.array(b.ids), genes=b.genes.astype(np.int16),
                                a=b.a, cold=b.cold, src=b.src.astype(np.int8), pool_hash=b.info["pool_hash"])


def save_pred(d, name: str, blk: RowBlock, s: np.ndarray, extra: dict | None = None) -> None:
    np.savez_compressed(d / f"pred_{name}.npz", ids=np.array(blk.ids), genes=blk.genes.astype(np.int16),
                        score=s.astype(np.float32), a=blk.a, cold=blk.cold, src=blk.src.astype(np.int8),
                        **(extra or {}))


class Chain:
    def __init__(self, arm: ArmSpec, world6: V6World | None = None, device: str = "cuda",
                 role: str = "formal", w: World6 | None = None, policy=None, threads: int = 8):
        self.arm = arm
        self.v6w = world6 or V6World()
        self.device = device
        self.role = role
        self.threads = threads
        self.w = w or World6(load_heavy=False)
        self.roles = Roles(self.w)
        self.drop_relations = None
        self.hpo_drop_rows = None
        if arm.go_layer in ("blocks", "blocks_random"):
            from .evidence8 import block_sets
            bs = block_sets(arm.go_layer)
            self.drop_relations = bs["relations"]
            self.hpo_drop_rows = bs["hpo_drop_rows"]
        self.labels = Labels(self.w, drop_pairs=self.drop_relations)
        self.policy = policy
        if policy is None and (arm.go_layer.startswith("layer1") or arm.go_variant != "full" or
                               arm.go_layer != "none" or arm.go_mask != "none"):
            from .evidence8 import GOPolicy
            self.policy = GOPolicy(self.w, arm.go_release)
        self._ctx_pack = None
        self.pack = Pack8(arm, self.w, policy=self.policy, hpo_drop_rows=self.hpo_drop_rows,
                          ctx_relation_ids=[] if arm.go_layer.startswith("layer1") else None)
        if not arm.go_layer.startswith("layer1"):
            self.policy = None                      # the parsed GAF is only needed to build the pack
            import gc
            gc.collect()
        self.T = self.pack.tensors(device)
        self.gs = GeneStatic(self.w, self.pack.a)
        self.masks = load_masks()
        self.e6_genes = self.masks["test"] if arm.e6_cold else None
        self.q_random = random_tissue(self.T) if arm.expression == "random" else None

    def use(self, ctx: str) -> None:
        """Layer-1 arms: the GO input depends on the context's own training relations."""
        if not self.arm.go_layer.startswith("layer1") or self._ctx_pack == ctx:
            return
        ref = set(self.roles.reference(self.w.contexts[ctx]))
        rel = self.w.primary_relations
        rids = rel.loc[rel["disease_id"].isin(ref), "relation_id"].tolist()
        self.pack = Pack8(self.arm, self.w, policy=self.policy, ctx_relation_ids=rids)
        del self.T
        torch.cuda.empty_cache()
        self.T = self.pack.tensors(self.device)
        self.gs = GeneStatic(self.w, self.pack.a)
        self._ctx_pack = ctx

    # ------------------------------------------------------------------ teacher
    def teacher(self, ctx: str, seed: int, controls: str, role: str | None = None) -> str:
        rid = teacher_run_id(self.arm.name, ctx, controls, seed)
        if not is_done(rid):
            self.use(ctx)
            ok = (self.T["mT"] | self.T["mH"]).cpu().numpy()
            data = teacher_data(self.w, self.roles, ctx, self.drop_relations, self.e6_genes, valid_rows=ok)
            t0 = time.time()
            r = run_teacher(rid, self.T, data, seed, controls, role or self.role, self.arm.key(),
                            self.pack.hash, excluded_genes=self.e6_genes,
                            meta_extra={"arm_spec": asdict(self.arm)})
            print(f"    teacher {rid}: ep {r['fit']['best_epoch']} sF@10(F_j+1) "
                  f"{r['fit']['best_val_sF_recall10']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        return rid

    # ------------------------------------------------------------------ student
    def student(self, ctx: str, seed: int, t_rid: str, kind: str, wd: float, role: str | None = None):
        """Train (or load) the student for one context; returns (z_hat, impute genes)."""
        self.use(ctx)
        rid = f"V8S__{self.arm.name}__{ctx}__{kind}_wd{wd:g}__s{seed}"
        model = load_teacher(t_rid, self.T, self.device)
        with torch.no_grad():
            from .teacher8 import gene_zF
            zF, mF = gene_zF(model, self.T)
        mS = self.T["mS"].cpu().numpy()
        mFn = mF.cpu().numpy()
        excl = np.zeros(N_G, dtype=bool)
        excl[self.masks["dev"]] = True
        excl[self.masks["test"]] = True
        elig = np.flatnonzero(mS & mFn & ~excl)
        tr_g, va_g = hashed_split(elig)
        impute = np.flatnonzero(mS & ~mFn)
        d = run_dir(rid)
        if not is_done(rid):
            meta = {"role": role or self.role, "stage": "student_fit", "component": "student",
                    "arm": self.arm.key(), "context_id": ctx, "fold": int(ctx[1]), "seed": seed,
                    "config": {"kind": kind, "weight_decay": wd, "lr": 1e-3, "batch": 256,
                               "max_epochs": 300, "patience": 20}, "teacher_run": t_rid,
                    "n_train_genes": int(len(tr_g)), "n_val_genes": int(len(va_g)),
                    "n_impute_genes": int(len(impute))}
            with tracked(rid, meta) as box:
                res = train_student(self.T["xS"], zF, tr_g, va_g, kind, wd, seed, self.device)
                torch.save(res["state"], d / "student.pt")
                from .student8 import gene_level_check
                chk = gene_level_check(res["model"], self.T["xS"], zF, va_g, elig)
                box["extra"] = {"fit": {k: res[k] for k in ("best_epoch", "best_val_loss", "n_params",
                                                            "seconds")}, "gene_level": chk}
        from .student8 import Student
        m = Student(kind).to(self.device)
        m.load_state_dict(torch.load(d / "student.pt", map_location=self.device, weights_only=True))
        m.eval()
        with torch.no_grad():
            zh = m(self.T["xS"][torch.as_tensor(impute, device=self.device)]).cpu().numpy()
        return zh, impute

    def imputation(self, ctx: str, seed: int, t_rid: str, mode: str, role: str | None = None):
        """Imputation for genes without GO but with a sequence, by mode:
        mlp_wd*/linear_wd* (student), nn (sequence nearest-neighbour transfer), teacher_sS
        (the teacher's own sequence projection), protst_mlp_wd* (student on ProtST input,
        open-knowledge control), *_shuffled (the student on length-stratum-shuffled inputs)."""
        from .student8 import Student, nearest_neighbour_transfer, shuffled_inputs
        from .teacher8 import gene_zF, gene_zS
        self.use(ctx)
        model = load_teacher(t_rid, self.T, self.device)
        with torch.no_grad():
            zF, mF = gene_zF(model, self.T)
        mS = self.T["mS"].cpu().numpy()
        mFn = mF.cpu().numpy()
        impute = np.flatnonzero(mS & ~mFn)
        shuffled = mode.endswith("_shuffled")
        base = mode[:-len("_shuffled")] if shuffled else mode
        if base == "nn":
            donors = np.flatnonzero(mS & mFn)
            zh = nearest_neighbour_transfer(self.pack.a["g_sequence"], zF.cpu().numpy(), impute, donors, self.device)
            return zh, impute
        if base == "teacher_sS":
            with torch.no_grad():
                zS, _ = gene_zS(model, self.T)
            return zS.cpu().numpy()[impute], impute
        protst = base.startswith("protst_")
        kind, wd = base.replace("protst_", "").split("_wd")
        wd = float(wd)
        if protst:
            X = np.load(paths8.CACHES / "encoders" / "protst_protein512.npy").astype(np.float32)
            Xt = torch.as_tensor(X, device=self.device)
            rid = f"V8S__{self.arm.name}__{ctx}__protst_{kind}_wd{wd:g}__s{seed}"
        else:
            Xt = self.T["xS"]
            rid = f"V8S__{self.arm.name}__{ctx}__{kind}_wd{wd:g}__s{seed}"
        excl = np.zeros(N_G, dtype=bool)
        excl[self.masks["dev"]] = True
        excl[self.masks["test"]] = True
        elig = np.flatnonzero(mS & mFn & ~excl)
        tr_g, va_g = hashed_split(elig)
        d = run_dir(rid)
        if not is_done(rid):
            meta = {"role": role or self.role, "stage": "student_fit", "component": "student",
                    "arm": self.arm.key(), "context_id": ctx, "fold": int(ctx[1]), "seed": seed,
                    "config": {"kind": kind, "weight_decay": wd, "input": "protst512" if protst else "esm2_35m_480"},
                    "teacher_run": t_rid, "n_train_genes": int(len(tr_g)), "n_impute_genes": int(len(impute))}
            with tracked(rid, meta) as box:
                res = train_student(Xt, zF, tr_g, va_g, kind, wd, seed, self.device)
                torch.save(res["state"], d / "student.pt")
                from .student8 import gene_level_check
                chk = gene_level_check(res["model"], Xt, zF, va_g, elig)
                box["extra"] = {"fit": {k: res[k] for k in ("best_epoch", "best_val_loss", "n_params", "seconds")},
                                "gene_level": chk}
        m = Student(kind, Xt.shape[1]).to(self.device)
        m.load_state_dict(torch.load(d / "student.pt", map_location=self.device, weights_only=True))
        m.eval()
        Xin = Xt
        if shuffled:
            import pandas as pd
            lengths = pd.read_parquet(paths8.V71_TABLES / "sequences_v71.parquet")["length"].to_numpy()
            Xs = shuffled_inputs(Xt.cpu().numpy(), impute, lengths, f"{self.arm.name}|{ctx}|{mode}")
            Xin = torch.as_tensor(Xs, device=self.device)
        with torch.no_grad():
            zh = m(Xin[torch.as_tensor(impute, device=self.device)]).cpu().numpy()
        return zh, impute

    # ------------------------------------------------------------------ semantic
    def semantic(self, ctx: str, t_rid: str, v6p: V6Pass, student=None):
        self.use(ctx)
        model = load_teacher(t_rid, self.T, self.device)
        sz, sg = (student if student is not None else (None, None))
        zsem, src = gene_semantic(model, self.T, sz, sg)
        expr = {"hpa": "rule", "none": "off", "random": "random"}[self.arm.expression]
        return semantic_for(model, self.T, v6p.ids, self.w.idx.d_ix, zsem, src, expression=expr,
                            q_random=self.q_random)

    # ------------------------------------------------------------------ blocks
    def blocks_for(self, ctx: str, v6p: V6Pass, sem, w_j: float, plain: bool = False,
                   pool: str = "strict") -> dict[str, RowBlock]:
        self.use(ctx)
        t = self.roles.targets(self.w.contexts[ctx])
        pos = v6p.pos
        out = {}
        for role, ids in t.items():
            ids = [d for d in ids if d in pos]
            if not ids:
                continue
            pr = v6p.rows(ids)
            spos = {d: i for i, d in enumerate(sem.ids)}
            sr = np.array([spos[d] for d in ids])
            rB = np.asarray(v6p.rB[pr]).astype(np.int32)
            if pool == "v6":
                genes = v6_list(rB)
                R = None
            elif pool == "bridge":
                # V7.1-style work pool: V6 500 U semantic top-500 (about 950), padded with -1
                v6l = v6_list(rB)
                rs = sem.rS[sr]
                genes = np.full((len(ids), 1000), -1, dtype=np.int64)
                for i in range(len(ids)):
                    top = np.flatnonzero((rs[i] > 0) & (rs[i] <= 500))
                    top = top[np.argsort(rs[i, top], kind="stable")]
                    u = list(dict.fromkeys(v6l[i].tolist() + top.tolist()))
                    genes[i, :len(u)] = u
                R = None
            else:
                P = strict_pool(rB, sem.rS[sr], sem.m[sr], w_j, plain=plain)
                genes, R = P["genes"], P["R"]
            lab = None if role in ("test", "x1", "x2") else self.labels
            erole = "train" if role == "oof" else role
            blk = build_block(ctx, erole, ids, genes, R, v6p, sem, self.gs, self.w.idx.d_ix, lab)
            if self.e6_genes is not None and role == "oof":
                blk.train_ok = blk.valid & ~np.isin(blk.genes, self.e6_genes)
            out[role] = blk
        return out


# ---------------------------------------------------------------------------
# fold runner
# ---------------------------------------------------------------------------
def dev_choice(name: str, default=None):
    sel = load_manifest(name)
    return sel if sel else default


def _ledger(rid: str, meta: dict, extra: dict) -> None:
    from .ledger8 import write_result
    write_result(rid, {**meta, **extra, "status": "done"})


def fold_models(ch: Chain, j: int, seed: int, blocks: dict, tag: str, selection: bool,
                sel_cfg: dict | None, out_dir, ctxs: list | None = None) -> dict:
    """B (4 fits), corrector configs, A configs, predictions; returns a summary."""
    inner = list(ctxs[:3]) if ctxs else ctx_list(j)[:3]
    outer = ctxs[3] if ctxs else f"f{j}_outer"
    summ = {"fold": j, "seed": seed, "tag": tag}
    base_meta = {"role": ch.role, "arm": ch.arm.key(), "arm_name": ch.arm.name, "tag": tag, "fold": j,
                 "seed": seed, "outputs": paths8.rel(out_dir)}
    # ---- B^{-b}
    B_oof = {}
    for b, c in enumerate(inner):
        t0 = time.time()
        others = [blocks[x]["oof"] for x in inner if x != c]
        bm = BaseRanker(seed, ch.threads).fit(others, [blocks[c]["val"]])
        dd = out_dir / f"B_minus_{b}"
        dd.mkdir(exist_ok=True)
        bm.save(dd)
        B_oof[c] = bm.score(blocks[c]["oof"])
        summ[f"B_minus_{b}"] = {"best_iteration": bm.best_iteration, "mu": bm.mu, "sd": bm.sd,
                                "reload_maxabs": reload_check(bm, dd, blocks[c]["oof"]),
                                "train_contexts": [x for x in inner if x != c], "checkpoint": f"F_(j+1) in {c}",
                                "seconds": time.time() - t0}
        _ledger(f"V8B__{tag}__{c}__s{seed}", {**base_meta, "component": "base_ranker", "context_id": c,
                                              "stage": "base_fit"}, {"fit": summ[f"B_minus_{b}"]})
    t0 = time.time()
    bo = BaseRanker(seed, ch.threads).fit([blocks[c]["oof"] for c in inner], [blocks[outer]["val"]])
    dd = out_dir / "B_outer"
    dd.mkdir(exist_ok=True)
    bo.save(dd)
    OUT = [r for r in blocks[outer] if r != "oof"]
    EV = "dev" if "dev" in blocks[outer] else "val"
    summ["eval_role"] = EV
    Bout = {r: bo.score(blocks[outer][r]) for r in OUT}
    summ["B_outer"] = {"best_iteration": bo.best_iteration, "mu": bo.mu, "sd": bo.sd,
                       "reload_maxabs": reload_check(bo, dd, blocks[outer]["val"]), "seconds": time.time() - t0}
    _ledger(f"V8B__{tag}__{outer}__s{seed}", {**base_meta, "component": "base_ranker", "context_id": outer,
                                              "stage": "base_fit"}, {"fit": summ["B_outer"]})
    summ["pool_hashes"] = {c: {r: blocks[c][r].info["pool_hash"] for r in blocks[c]} for c in blocks}
    for r in OUT:
        save_pred(out_dir, f"B__{r}", blocks[outer][r], Bout[r])
    tr_blocks = [blocks[c]["oof"] for c in inner]
    tr_B = [B_oof[c] for c in inner]
    # ---- corrector configurations
    if selection:
        cfgs = C_GRID
    elif sel_cfg.get("only_selected"):
        cfgs = [c for c in C_GRID if c["lam"] == sel_cfg["corrector"]["lam"]
                and c["beta"] == sel_cfg["corrector"]["beta"]]
    else:
        cfgs = [c for c in C_GRID if c["lam"] == sel_cfg["corrector"]["lam"]]
    dev_rows = []
    for cfg in cfgs:
        name = f"C_b{cfg['beta']:g}_l{cfg['lam']:g}"
        cc = Corrector(CorrectorConfig(beta=cfg["beta"], lam=cfg["lam"]), seed, ch.threads)
        cc.fit(tr_blocks, tr_B, blocks[outer]["val"], Bout["val"])
        dd = out_dir / name
        dd.mkdir(exist_ok=True)
        meta = cc.save(dd)
        for r in OUT:
            save_pred(out_dir, f"{name}__{r}", blocks[outer][r], cc.score(blocks[outer][r], Bout[r]))
        dv = blocks[outer][EV]
        sd = cc.score(dv, Bout[EV])
        rc, _ = cold_recall_at(dv, sd)
        dev_rows.append({"name": name, **cfg, "dev_recall10": float(recall_at(dv, sd).mean()),
                         "dev_cold_recall10": float(np.nanmean(rc)), "best_iteration": cc.best_iteration,
                         "underfit_rule": meta["diag"]["underfit_rule"], "diag": meta["diag"]})
        _ledger(f"V8C__{tag}__{name}__f{j}__s{seed}", {**base_meta, "component": "corrector",
                                                       "context_id": outer, "stage": "corrector_fit",
                                                       "config": meta["config"]},
                {"fit": {k: v for k, v in dev_rows[-1].items() if k != "diag"}, "diag": meta["diag"]})
        print(f"      {name}: dev R@10 {dev_rows[-1]['dev_recall10']:.4f} R^c {dev_rows[-1]['dev_cold_recall10']:.4f}"
              f" it {cc.best_iteration} ({meta['diag']['seconds']:.0f}s)", flush=True)
    # ---- registered 31-leaf extension (development only): the corrector the A/B rule would prefer,
    # only if its F_{j+1} curve was still improving at the end of the 200-round budget
    if selection:
        pref = sorted(dev_rows, key=lambda r: (-r["dev_cold_recall10"], -r["dev_recall10"], r["beta"], -r["lam"]))[0]
        summ["underfit_trigger_config"] = [pref["name"]] if pref["underfit_rule"] else []
        if pref["underfit_rule"]:
            name = pref["name"] + "_nl31"
            cc = Corrector(CorrectorConfig(beta=pref["beta"], lam=pref["lam"], num_leaves=31), seed, ch.threads)
            cc.fit(tr_blocks, tr_B, blocks[outer]["val"], Bout["val"])
            dd = out_dir / name
            dd.mkdir(exist_ok=True)
            meta = cc.save(dd)
            for r in OUT:
                save_pred(out_dir, f"{name}__{r}", blocks[outer][r], cc.score(blocks[outer][r], Bout[r]))
            dv = blocks[outer][EV]
            sd = cc.score(dv, Bout[EV])
            rc, _ = cold_recall_at(dv, sd)
            dev_rows.append({"name": name, "beta": pref["beta"], "lam": pref["lam"], "num_leaves": 31,
                             "dev_recall10": float(recall_at(dv, sd).mean()), "dev_cold_recall10": float(np.nanmean(rc)),
                             "best_iteration": cc.best_iteration, "underfit_rule": meta["diag"]["underfit_rule"],
                             "diag": meta["diag"]})
    summ["corrector"] = dev_rows
    # ---- candidate A
    a_cfgs = A_GRID if selection else [sel_cfg["A"]]
    a_rows = []
    for cfg in a_cfgs:
        name = f"A_nl{cfg['num_leaves']}_mc{cfg['min_child_samples']}"
        am = CandidateA(cfg["num_leaves"], cfg["min_child_samples"], seed, ch.threads)
        am.fit(tr_blocks, [blocks[outer]["val"]])
        am.save(out_dir, name)
        for r in OUT:
            save_pred(out_dir, f"{name}__{r}", blocks[outer][r], am.score(blocks[outer][r]))
        dv = blocks[outer][EV]
        sd = am.score(dv)
        rc, _ = cold_recall_at(dv, sd)
        a_rows.append({"name": name, **cfg, "dev_recall10": float(recall_at(dv, sd).mean()),
                       "dev_cold_recall10": float(np.nanmean(rc)), "best_iteration": am.best_iteration})
        _ledger(f"V8A__{tag}__{name}__f{j}__s{seed}", {**base_meta, "component": "candidate_A",
                                                       "context_id": outer, "stage": "A_fit",
                                                       "config": am.params}, {"fit": a_rows[-1]})
        print(f"      {name}: dev R@10 {a_rows[-1]['dev_recall10']:.4f} R^c {a_rows[-1]['dev_cold_recall10']:.4f}",
              flush=True)
    summ["A"] = a_rows
    dv = blocks[outer][EV]
    rcB, _ = cold_recall_at(dv, Bout[EV])
    summ["B_dev"] = {"dev_recall10": float(recall_at(dv, Bout[EV]).mean()),
                     "dev_cold_recall10": float(np.nanmean(rcB))}
    return summ


def ab_rule(summ: dict, margin: float = 0.005) -> dict:
    """Per-context A/B deployment rule (registered)."""
    bestA = max(summ["A"], key=lambda r: (r["dev_recall10"], -r["num_leaves"]))
    keep = [r for r in summ["corrector"] if r["dev_recall10"] >= bestA["dev_recall10"] - margin]
    out = {"best_A": bestA["name"], "best_A_dev_recall10": bestA["dev_recall10"],
           "best_A_dev_cold_recall10": bestA["dev_cold_recall10"], "kept_B_configs": [r["name"] for r in keep]}
    if not keep:
        out.update({"chosen_B": None, "deploy": "A", "reason": "no corrector within the margin of the best A"})
        return out
    pick = sorted(keep, key=lambda r: (-r["dev_cold_recall10"], -r["dev_recall10"], r["beta"], -r["lam"]))[0]
    beats = pick["dev_recall10"] > bestA["dev_recall10"]
    ni_cold = (pick["dev_recall10"] >= bestA["dev_recall10"] - margin and
               pick["dev_cold_recall10"] > bestA["dev_cold_recall10"])
    out.update({"chosen_B": pick["name"], "chosen_B_beta": pick["beta"], "chosen_B_lam": pick["lam"],
                "chosen_B_dev_recall10": pick["dev_recall10"], "chosen_B_dev_cold_recall10": pick["dev_cold_recall10"],
                "deploy": "B" if (beats or ni_cold) else "A",
                "reason": "B beats A on dev R@10" if beats else
                ("B within 0.005 of A and higher dev cold recall" if ni_cold else "A kept")})
    return out

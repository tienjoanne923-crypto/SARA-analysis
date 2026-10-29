"""V8.1 test evaluation (only after registration): per-disease metrics of every saved list.

Output: ``results_v8/eval/eval_test_<tag>.parquet`` (long format: tag, seed, fold, method,
disease_id, metrics), plus E2 hit-flow and the E3 fall-back exactness check.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths8
from .common8 import dump_json, load_manifest, now_utc
from .evaluate8 import load_pred, per_disease
from .feats8 import GeneStatic, Labels
from .masks8 import load as load_masks
from .v6pass8 import WORLDS, V6Pass

EVAL = paths8.RESULTS / "eval"
PRED_RE = re.compile(r"pred_(.+)__(test|dev|val)\.npz$")


class Ctx:
    """Shared evaluation context: labels, gene groups, cold vectors per (world, fold)."""

    def __init__(self):
        from gpu8v6.common import World6
        from .pack8 import ArmSpec, Pack8
        if not load_manifest("protocol_v8_registered.json"):
            raise PermissionError("evaluation of F_j requires the V8.1 registration")
        self.w = World6(load_heavy=False)
        self.labels = Labels(self.w)
        pack = Pack8(ArmSpec(), self.w)
        self.gs = GeneStatic(self.w, pack.a)
        m = load_masks()
        G = self.w.n_g
        mk = lambda idx: np.isin(np.arange(G), idx)
        self.gtype = {"ncrna": self.gs.ncrna, "nogo_protein": self.gs.protein & ~self.gs.has_go,
                      "nogo": ~self.gs.has_go, "noseq": ~self.gs.has_seq,
                      "mask_test": mk(m["test"]), "mask_dev": mk(m["dev"])}
        self._cold = {}
        held = pd.read_parquet(paths8.TABLES / "evidence_block_relations_v8.parquet")
        held = held.assign(gene_row=held["gene_id"].astype(str).map(self.w.idx.g_ix))
        self.block_pairs = {(d, int(g)) for d, g in zip(held["disease_id"], held["gene_row"]) if pd.notna(g)}
        self.d_hpo = pack.a["d_hpo_mask"]
        self.d_text = pack.a["d_text_mask"]

    def cold(self, world: str, j: int) -> np.ndarray:
        k = (world, j)
        if k not in self._cold:
            self._cold[k] = V6Pass(WORLDS[world], f"f{j}_outer").cold()
        return self._cold[k]


def eval_dir(cx: Ctx, d: Path, tag: str, seed, j: int, world: str = "main", role: str = "test",
             methods: list[str] | None = None) -> pd.DataFrame:
    src_p = d / f"gene_src_f{j}_outer.npy"
    src = np.load(src_p) if src_p.exists() else None
    out = []
    for p in sorted(d.glob(f"pred_*__{role}.npz")):
        m = PRED_RE.search(p.name)
        meth = m.group(1)
        if methods and meth not in methods:
            continue
        pred = load_pred(p)
        per = per_disease(pred, cx.labels, role, cx.cold(world, j), src, cx.gtype)
        # pair-level subgroup: evidence-block relations
        blk = []
        ids = [str(x) for x in pred["ids"]]
        pos = {x: i for i, x in enumerate(ids)}
        for r in per.itertuples(index=False):
            i = pos[r.disease_id]
            v = pred["valid"][i] if "valid" in pred else np.isfinite(pred["score"][i])
            gi, si = pred["genes"][i][v].astype(np.int64), pred["score"][i][v].astype(np.float64)
            top = set(gi[np.lexsort((gi, -si))][:10].tolist())
            q = [g for g in cx.labels.rows(r.disease_id, role) if (r.disease_id, int(g)) in cx.block_pairs]
            blk.append((len(q), sum(1 for g in q if g in top)))
        per["n_query_block"] = [a for a, _ in blk]
        per["hits10_block"] = [b for _, b in blk]
        per.insert(0, "method", meth)
        per.insert(0, "fold", j)
        per.insert(0, "seed", seed)
        per.insert(0, "tag", tag)
        out.append(per)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fallback_exactness(d: Path, corr_names: list[str], role: str = "test") -> dict:
    """E3: rows with a = 0 carry exactly the base score B under every corrector."""
    B = load_pred(d / f"pred_B__{role}.npz")
    out = {}
    for n in corr_names:
        p = d / f"pred_{n}__{role}.npz"
        if not p.exists():
            continue
        C = load_pred(p)
        a = C["a"].astype(bool)
        fin = np.isfinite(B["score"])
        diff = np.abs(C["score"][~a & fin] - B["score"][~a & fin])
        out[n] = {"rows_a0": int((~a & fin).sum()), "max_abs_diff_a0": float(diff.max()) if diff.size else 0.0,
                  "rows_a1": int((a & fin).sum())}
    return out


def hitflow(cx: Ctx, d: Path, j: int, old: str, new: str, role: str = "test", world: str = "main") -> dict:
    """Hits of the old list kept / lost by the new list in the top 10, and new hits gained (E2)."""
    a, b = load_pred(d / f"pred_{old}__{role}.npz"), load_pred(d / f"pred_{new}__{role}.npz")
    cold = cx.cold(world, j)
    ids_b = {str(x): i for i, x in enumerate(b["ids"])}
    tot = {"kept": 0, "lost": 0, "gained": 0, "kept_cold": 0, "lost_cold": 0, "gained_cold": 0, "diseases": 0}

    def top(pred, i):
        v = pred["valid"][i] if "valid" in pred else np.isfinite(pred["score"][i])
        g, s = pred["genes"][i][v].astype(np.int64), pred["score"][i][v].astype(np.float64)
        return set(g[np.lexsort((g, -s))][:10].tolist())

    for i, dd in enumerate(a["ids"]):
        dd = str(dd)
        if dd not in ids_b:
            continue
        q = set(cx.labels.rows(dd, role).tolist())
        if not q:
            continue
        ha, hb = top(a, i) & q, top(b, ids_b[dd]) & q
        tot["diseases"] += 1
        for k, s in (("kept", ha & hb), ("lost", ha - hb), ("gained", hb - ha)):
            tot[k] += len(s)
            tot[k + "_cold"] += sum(1 for g in s if cold[g])
    return tot


def run(which: list[str] | None = None) -> dict:
    cx = Ctx()
    EVAL.mkdir(parents=True, exist_ok=True)
    done = {}
    jobs = []
    for tagdir in sorted(paths8.FORMAL.glob("*")):
        if tagdir.name in ("comparators",):
            continue
        for sd in sorted(tagdir.glob("s*")):
            for fd in sorted(sd.glob("fold*")):
                jobs.append((tagdir.name, int(sd.name[1:]), int(fd.name[4:]), fd, "main"))
    for tagdir in sorted(paths8.VARIANTS.glob("*")):
        info_world = "main"
        for sd in sorted(tagdir.glob("s*")):
            for fd in sorted(sd.glob("fold*")):
                dj = fd / "done.json"
                if dj.exists():
                    info = json.loads(dj.read_text(encoding="utf-8"))
                    info_world = (info.get("world") or {}).get("name", "main")
                jobs.append((tagdir.name, int(sd.name[1:]), int(fd.name[4:]), fd, info_world))
    for comp in ("kdgene", "sign"):
        for fd in sorted((paths8.FORMAL / "comparators" / comp).glob("fold*")):
            jobs.append((f"comparator_{comp}", 42, int(fd.name[4:]), fd, "main"))
    by_tag: dict[str, list] = {}
    for tag, seed, j, fd, world in jobs:
        if which and tag not in which:
            continue
        df = eval_dir(cx, fd, tag, seed, j, world if world in WORLDS else "main")
        if len(df):
            by_tag.setdefault(tag, []).append(df)
    for tag, dfs in by_tag.items():
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(EVAL / f"eval_test_{tag}.parquet", index=False)
        done[tag] = {"rows": int(len(df)), "methods": sorted(df["method"].unique().tolist())}
        print(f"  eval {tag}: {len(df)} rows, methods {done[tag]['methods'][:12]}", flush=True)
    dump_json(EVAL / "eval_index.json", {"generated_utc": now_utc(), "tags": done})
    return done


if __name__ == "__main__":
    import sys
    paths8.ensure()
    run(sys.argv[1:] or None)

"""One full-background pass of the frozen V6 retriever per context (§8, §10.2).

For every target disease of a context (role lists only: inner = I_jb + F_{j+1};
outer = F_j + F_{j+1} + development) the frozen V6 source engine (role-based
references and gene-HPO exclusion) scores all 19,861 genes; stored per context:

  V   float32 [D, 7, G]   max-normalised source scores (the V6 channel values)
  fused float32 [D, G]    the frozen V6 reciprocal-rank fusion
  rB  int16 [D, G]        1-based position in the frozen V6 ordering over the full
                          background: signal genes (fused > 0) by fused, then the
                          remaining genes by the degree prior; gene-row (= GeneID)
                          tie-break.  Its first 500 are the V6 500-list.
  prior float32 [G], ref_degree int32 [G], hpo_visible bool [G]
  dinfo.parquet           per-disease source availability and V6 disease features

GO is not an input of any V6 source, so GO-only arms share this cache; worlds
that change the reference graph or gene HPO (evidence blocks, E6, historical)
have their own ``world`` key.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gpu8v6.retrieval import fuse, max_normalise, rank_transform
from gpu8v6.sources import SOURCES

from . import paths8
from .common8 import dep_key, dump_json, list_hash, now_utc

N_G = 19861
N_SRC = len(SOURCES)


@dataclass(frozen=True)
class V6World:
    name: str = "main"
    drop_genes: str = "none"          # none | mask_test
    drop_relations: str = "none"      # none | evidence_blocks
    hpo_filter: str = "none"          # none | evidence_blocks

    def key(self) -> str:
        return dep_key(**asdict(self))


WORLDS = {"main": V6World(),
          "e6cold": V6World(name="e6cold", drop_genes="mask_test"),
          "blocks": V6World(name="blocks", drop_relations="evidence_blocks", hpo_filter="evidence_blocks"),
          "blocks_random": V6World(name="blocks_random", drop_relations="blocks_random",
                                   hpo_filter="blocks_random")}


def ctx_dir(world: V6World, ctx: str):
    return paths8.V6PASS / world.name / ctx


def targets(w, roles, ctx_id: str) -> dict[str, list[str]]:
    ctx = w.contexts[ctx_id]
    t = roles.targets(ctx)
    d_ix = w.idx.d_ix
    return {k: [d for d in v if d in d_ix] for k, v in t.items()}


def target_list(w, roles, ctx_id) -> list[str]:
    t = targets(w, roles, ctx_id)
    out = []
    for k in ("oof", "test", "val", "dev", "x2", "x1"):
        out += t.get(k, [])
    return list(dict.fromkeys(out))


def v6_order(fused: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """1-based full ordering: signal genes by (-fused, row), then the rest by (-prior, row)."""
    G = len(prior)
    rows = np.arange(G)
    sig = fused > 0
    s_idx = np.flatnonzero(sig)
    s_ord = s_idx[np.lexsort((s_idx, -fused[s_idx].astype(np.float64)))]
    r_idx = np.flatnonzero(~sig)
    r_ord = r_idx[np.lexsort((r_idx, -prior[r_idx].astype(np.float64)))]
    order = np.concatenate([s_ord, r_ord])
    rank = np.empty(G, dtype=np.int32)
    rank[order] = np.arange(1, G + 1, dtype=np.int32)
    return rank


def make_engine(w, roles, world: V6World, device: str = "cuda"):
    from .roles8 import RoleSourceEngine
    drop_rel, drop_g = None, None
    if world.drop_genes == "mask_test":
        from .masks8 import load
        drop_g = load()["test"]
    bs = None
    if world.drop_relations in ("evidence_blocks", "blocks_random"):
        from .evidence8 import block_sets
        bs = block_sets("blocks" if world.drop_relations == "evidence_blocks" else "blocks_random")
        drop_rel = bs["relations"]
    eng = RoleSourceEngine(w, roles, device=device, drop_relations=drop_rel, drop_genes=drop_g)
    if world.hpo_filter in ("evidence_blocks", "blocks_random"):
        from gpu8v6 import paths6
        from gpu8v6.gene_hpo import disease_profiles
        ph = pd.read_parquet(paths6.TABLES / "phenotypes_v6.parquet")
        keep = np.ones(len(ph), dtype=bool)
        keep[bs["hpo_drop_rows"]] = False
        eng.Q = disease_profiles(eng.space, w.n_d, w.idx.d_ix, phenotypes_v6=ph[keep])
        eng.has_hpo = np.diff(eng.Q.indptr) > 0
        eng.hpo_rows_removed = int((~keep).sum())
    return eng


def run_context(eng, w, roles, world: V6World, ctx_id: str, frozen: dict, batch: int = 200) -> dict:
    out_dir = ctx_dir(world, ctx_id)
    meta_p = out_dir / "meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ids = target_list(w, roles, ctx_id)
    D = len(ids)
    cs = eng.context(w.contexts[ctx_id])
    prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(eng.ppi_deg)
    prior = (prior / max(prior.max(), 1e-12)).astype(np.float32)
    ref_degree = np.asarray(cs.A.sum(axis=0)).ravel().astype(np.int32)
    hpo_vis = np.diff(cs.Gh.indptr) > 0
    V = np.lib.format.open_memmap(out_dir / "V.npy", mode="w+", dtype=np.float32, shape=(D, N_SRC, N_G))
    FU = np.lib.format.open_memmap(out_dir / "fused.npy", mode="w+", dtype=np.float32, shape=(D, N_G))
    RB = np.lib.format.open_memmap(out_dir / "rB.npy", mode="w+", dtype=np.int16, shape=(D, N_G))
    infos = []
    for b0 in range(0, D, batch):
        bid = ids[b0:b0 + batch]
        raw, info = eng.scores(cs, bid, sources=SOURCES)
        norm = {s: max_normalise(v) for s, v in raw.items()}
        if frozen["mode"] == "rrf":
            base = {s: rank_transform(v) for s, v in norm.items()}
        else:
            base = norm
        fused, _ = fuse(base, frozen["weights"])
        for i in range(len(bid)):
            RB[b0 + i] = v6_order(fused[i], prior).astype(np.int16)
        FU[b0:b0 + len(bid)] = fused
        V[b0:b0 + len(bid)] = np.stack([norm[s] for s in SOURCES], axis=1)
        for s in SOURCES:
            info[f"a_{s}"] = (norm[s].max(axis=1) > 0).astype(int)
        info["n_signal"] = (fused > 0).sum(axis=1)
        infos.append(info)
        print(f"    {world.name}/{ctx_id}: {min(b0 + batch, D)}/{D} ({time.time() - t0:.0f}s)", flush=True)
    V.flush(), FU.flush(), RB.flush()
    del V, FU, RB
    dinfo = pd.concat(infos, ignore_index=True)
    dinfo.to_parquet(out_dir / "dinfo.parquet", index=False)
    np.save(out_dir / "prior.npy", prior)
    np.save(out_dir / "ref_degree.npy", ref_degree)
    np.save(out_dir / "hpo_visible.npy", hpo_vis)
    # agreement of the first 500 with the archived V6 lists (main world only)
    agree = None
    if world.name == "main":
        from gpu8v6 import paths6
        p = paths6.CACHES / "candidates" / f"{ctx_id}_candidates.parquet"
        if p.exists():
            c = pd.read_parquet(p, columns=["disease_id", "gene_row", "rank"])
            c = c[c["rank"] <= 500]
            arch = {d: set(g.astype(int)) for d, g in c.groupby("disease_id")["gene_row"]}
            RB = np.load(out_dir / "rB.npy", mmap_mode="r")
            ex, n = 0, 0
            for i, d in enumerate(ids):
                if d in arch:
                    n += 1
                    ex += int(set(np.flatnonzero(RB[i] <= 500).tolist()) == arch[d])
            agree = {"compared": n, "exact_500_equal": ex, "share": ex / max(n, 1)}
    meta = {"context_id": ctx_id, "world": asdict(world), "world_key": world.key(), "n_targets": D,
            "ids_hash": list_hash(ids), "reference_hash": list_hash(sorted(cs.reference)),
            "n_reference": len(cs.reference), "visible_relations": int(cs.A.nnz),
            "gene_hpo": cs.info.get("gene_hpo"), "v6_500_agreement": agree,
            "hpo_rows_removed": getattr(eng, "hpo_rows_removed", 0),
            "seconds": time.time() - t0, "written_utc": now_utc(), "exclusion": "role-based"}
    dump_json(out_dir / "ids.json", ids)
    dump_json(meta_p, meta)
    return meta


class V6Pass:
    """Read access to one context's pass (memory-mapped)."""

    def __init__(self, world: V6World, ctx_id: str):
        d = ctx_dir(world, ctx_id)
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.ids = json.loads((d / "ids.json").read_text(encoding="utf-8"))
        self.pos = {x: i for i, x in enumerate(self.ids)}
        self.V = np.load(d / "V.npy", mmap_mode="r")
        self.fused = np.load(d / "fused.npy", mmap_mode="r")
        self.rB = np.load(d / "rB.npy", mmap_mode="r")
        self.prior = np.load(d / "prior.npy")
        self.ref_degree = np.load(d / "ref_degree.npy")
        self.hpo_visible = np.load(d / "hpo_visible.npy")
        self.dinfo = pd.read_parquet(d / "dinfo.parquet").set_index("disease_id")

    def rows(self, ids: list[str]) -> np.ndarray:
        return np.array([self.pos[d] for d in ids], dtype=np.int64)

    def cold(self) -> np.ndarray:
        """Reference-path cold genes of this context: no reference edge and no direct gene HPO."""
        return (self.ref_degree == 0) & ~self.hpo_visible


def run(world: V6World, contexts: list[str] | None = None, device: str = "cuda") -> list[dict]:
    from gpu8v6.common import World6
    from gpu8v6.common import load_manifest as lm6
    from .roles8 import Roles
    w = World6(load_heavy=True)
    roles = Roles(w)
    frozen = lm6("retriever_v6_frozen.json")
    if contexts is None:
        contexts = [c for j in range(5) for c in (f"f{j}_outer", f"f{j}_inner0", f"f{j}_inner1", f"f{j}_inner2")]
    todo = [c for c in contexts if not (ctx_dir(world, c) / "meta.json").exists()]
    if not todo:
        return []
    eng = make_engine(w, roles, world, device)
    reps = []
    for c in todo:
        rep = run_context(eng, w, roles, world, c, frozen)
        print(json.dumps({k: rep[k] for k in ("context_id", "n_targets", "v6_500_agreement", "seconds")}),
              flush=True)
        reps.append(rep)
    return reps


if __name__ == "__main__":
    import sys
    paths8.ensure()
    name = sys.argv[1] if len(sys.argv) > 1 else "main"
    worlds = WORLDS
    ctxs = sys.argv[2:] or None
    run(worlds[name], ctxs)

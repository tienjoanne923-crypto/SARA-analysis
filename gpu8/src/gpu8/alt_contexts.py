"""Extra graph contexts for the family, external, temporal and second-network runs.

These experiments do not reuse the five main folds, so each gets its own
reference graph plus three inner blocks, built under exactly the same rules as
the main folds: a training disease is encoded in the context that holds its own
block out, and the outer context contains the training diseases only.

    family_*     reference = the frozen family-split training diseases
    external_*   reference = every non-development, non-external disease
    temporal_*   reference = the 2023Q4 historical relations
    ppi2_*       the fold-0 split rebuilt on the functional interaction network
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .config import Gpu8Config, load_config
from .contexts import (build_H, build_P, build_context, build_hierarchy_neighbours,
                       context_hash, load_entity_index)
from .hashing import atomic_write_json, code_version_hash, read_json
from .protocol import fold_assignment, inner_blocks, load_spec


def _make_group(cfg, idx, prefix, reference, relations, phenotypes, hierarchy, ppi,
                family_of, index, resource_hash, code_hash, H, P, deg, hier_nb,
                force: bool, seed_offset: int) -> list[str]:
    """Outer context plus three inner-block contexts for one alternative split."""
    built = []
    blocks = inner_blocks(list(reference), family_of, cfg.protocol.inner_blocks,
                          cfg.protocol.split_seed + seed_offset)
    specs = [(f"{prefix}_inner{i}", sorted(set(reference) - set(b)), b)
             for i, b in enumerate(blocks)]
    specs.append((f"{prefix}_outer", sorted(reference), []))
    for cid, refs, held in specs:
        chash = context_hash(cfg, cid, resource_hash, code_hash, {"group": prefix})
        out_dir = paths.CACHE_DIR / chash
        if force or not (out_dir / "meta.json").exists():
            print(f"  {cid}: {len(refs)} reference diseases", flush=True)
            meta = build_context(cfg, idx, relations, phenotypes, hierarchy, ppi,
                                 refs, cid, out_dir, hier_nb=hier_nb, H=H, P=P,
                                 ppi_deg=deg)
        else:
            meta = read_json(out_dir / "meta.json", {})
        index["contexts"][cid] = {
            "context_hash": chash, "path": str(out_dir), "fold": prefix,
            "kind": prefix, "block_index": -1 if cid.endswith("outer") else
            int(cid[-1]), "n_reference": len(refs),
            "held_out_block": held,
            "n_visible_relations": meta.get("n_visible_relations", 0)}
        built.append(cid)
    return built


def build_alt(cfg: Gpu8Config | None = None, force: bool = False,
              which: list[str] | None = None) -> dict:
    cfg = cfg or load_config()
    idx = load_entity_index()
    relations = pd.read_parquet(paths.TABLES_DIR / "relations.parquet")
    phenotypes = pd.read_parquet(paths.TABLES_DIR / "phenotypes.parquet")
    hierarchy = pd.read_parquet(paths.TABLES_DIR / "hierarchy.parquet")
    ppi = pd.read_parquet(paths.TABLES_DIR / "ppi.parquet")
    diseases = pd.read_parquet(paths.TABLES_DIR / "diseases.parquet")
    family_of = dict(zip(diseases["disease_id"], diseases["family_group"]))

    index = read_json(paths.MANIFEST_DIR / "context_index.json", {})
    index.setdefault("contexts", {})
    resource_hash = index.get("resource_hash", "")
    code_hash = code_version_hash(Path(__file__).resolve().parent)

    H = build_H(phenotypes, idx)
    P, deg = build_P(ppi, idx)
    hier_nb = build_hierarchy_neighbours(hierarchy, idx,
                                         cfg.candidates.hierarchy_max_distance)

    spec_blob = read_json(paths.MANIFEST_DIR / "splits_main_5fold.json", {})
    fam_blob = read_json(paths.MANIFEST_DIR / "splits_family.json", {})
    which = set(which or ["family", "external", "temporal", "ppi2", "ablation"])
    built: list[str] = []

    if "family" in which and fam_blob:
        built += _make_group(cfg, idx, "family", fam_blob["train_diseases"],
                             relations, phenotypes, hierarchy, ppi, family_of,
                             index, resource_hash, code_hash, H, P, deg, hier_nb,
                             force, 900)

    if "external" in which and spec_blob:
        reference = sorted(set(spec_blob["family_of"])
                           - set(spec_blob["dev_diseases"])
                           - set(spec_blob["external_diseases"]))
        built += _make_group(cfg, idx, "external", reference, relations, phenotypes,
                             hierarchy, ppi, family_of, index, resource_hash,
                             code_hash, H, P, deg, hier_nb, force, 910)

    hist_dir = paths.GPU8_ROOT / "tables_historical"
    if "temporal" in which and (hist_dir / "relations.parquet").exists():
        h_rel = pd.read_parquet(hist_dir / "relations.parquet")
        h_phen = pd.read_parquet(hist_dir / "phenotypes.parquet")
        h_hier = pd.read_parquet(hist_dir / "hierarchy.parquet")
        H_h = build_H(h_phen, idx)
        hn_h = build_hierarchy_neighbours(h_hier, idx,
                                          cfg.candidates.hierarchy_max_distance)
        reference = sorted(set(h_rel["disease_id"]))
        built += _make_group(cfg, idx, "temporal", reference, h_rel, h_phen, h_hier,
                             ppi, family_of, index, resource_hash, code_hash,
                             H_h, P, deg, hn_h, force, 920)

    if "ppi2" in which and spec_blob:
        sspec = load_spec()
        P2, deg2 = build_P(ppi, idx, network_type="functional")
        fa = fold_assignment(sspec, 0, cfg.protocol.n_outer_folds)
        built += _make_group(cfg, idx, "ppi2", fa["train_diseases"], relations,
                             phenotypes, hierarchy, ppi, family_of, index,
                             resource_hash, code_hash, H, P2, deg2, hier_nb,
                             force, 930)

    # ---- retraining ablations that change the inputs themselves -----------
    # A3 removes the phenotype path and A4 removes the interaction propagation.
    # Both rebuild every downstream quantity (retrieval, gene summaries,
    # propagation seeds) rather than zeroing one column of a cached view.
    if "ablation" in which and spec_blob:
        sspec = load_spec()
        empty_phen = phenotypes.iloc[0:0]
        empty_ppi = ppi.iloc[0:0]
        H_empty = build_H(empty_phen, idx)
        P_empty, deg_empty = build_P(empty_ppi, idx)
        for fold in cfg.protocol.ablation_folds:
            fa = fold_assignment(sspec, fold, cfg.protocol.n_outer_folds)
            built += _make_group(cfg, idx, f"a3nophen_f{fold}", fa["train_diseases"],
                                 relations, empty_phen, hierarchy, ppi, family_of,
                                 index, resource_hash, code_hash, H_empty, P, deg,
                                 hier_nb, force, 940 + fold)
            built += _make_group(cfg, idx, f"a4noppi_f{fold}", fa["train_diseases"],
                                 relations, phenotypes, hierarchy, empty_ppi,
                                 family_of, index, resource_hash, code_hash, H,
                                 P_empty, deg_empty, hier_nb, force, 960 + fold)

    index["n_contexts"] = len(index["contexts"])
    atomic_write_json(paths.MANIFEST_DIR / "context_index.json", index)
    return {"built": built, "n_contexts": index["n_contexts"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the alternative-split contexts.")
    ap.add_argument("--which", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    print(json.dumps(build_alt(which=args.which, force=args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

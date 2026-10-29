"""Disease partitioning, query isolation and the inner cold contexts.

Plan section 6:

* a development set of ~15% of families, reserved before modelling;
* an external hold-out of cold diseases kept out of supervised training;
* five frozen outer folds F0..F4 over the remaining diseases, fold j testing on
  F_j and validating on F_{j+1 mod 5};
* each outer training set is split into three inner blocks B1..B3, giving four
  graph contexts per fold (three inner reference contexts + one full outer
  context), 20 contexts in total for the main analysis;
* a separate frozen family-level 60/20/20 split.

Every split is written once, hashed, and never regenerated from model output.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .config import Gpu8Config, load_config
from .hashing import atomic_write_df, atomic_write_json, sha256_obj


@dataclass
class SplitSpec:
    name: str
    dev_diseases: list[str]
    external_diseases: list[str]
    folds: dict[str, list[str]]           # "F0".."F4" -> disease ids
    family_of: dict[str, str]
    seed: int
    spec_hash: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_dev"] = len(self.dev_diseases)
        d["n_external"] = len(self.external_diseases)
        d["fold_sizes"] = {k: len(v) for k, v in self.folds.items()}
        return d


def _family_blocks(diseases: pd.DataFrame, rng: np.random.Generator) -> list[tuple[str, list[str]]]:
    """Families shuffled deterministically, largest-first inside the shuffle."""
    fam = diseases.groupby("family_group")["disease_id"].apply(list).to_dict()
    keys = sorted(fam)
    rng.shuffle(keys)
    return [(k, sorted(fam[k])) for k in keys]


def build_splits(cfg: Gpu8Config, tables_dir: Path | None = None) -> SplitSpec:
    tables_dir = Path(tables_dir) if tables_dir else paths.TABLES_DIR
    diseases = pd.read_parquet(tables_dir / "diseases.parquet")
    relations = pd.read_parquet(tables_dir / "relations.parquet")

    primary = set(cfg.data.primary_evidence)
    rel_primary = relations[relations["relation_type"].isin(primary)]
    labelled = set(rel_primary["disease_id"])
    diseases = diseases[diseases["disease_id"].isin(labelled)].copy()

    rng = np.random.default_rng(cfg.protocol.split_seed)
    blocks = _family_blocks(diseases, rng)
    total = sum(len(v) for _, v in blocks)

    # 1) development families (~15%)
    dev_target = int(round(cfg.protocol.dev_family_fraction * total))
    dev, rest = [], []
    acc = 0
    for fam, ids in blocks:
        if acc < dev_target:
            dev.extend(ids)
            acc += len(ids)
        else:
            rest.append((fam, ids))

    # 2) external hold-out families (~10% of the remainder), isolated from
    #    supervised training so external confirmation is not a re-analysis
    rest_total = sum(len(v) for _, v in rest)
    ext_target = int(round(0.10 * rest_total)) if cfg.protocol.external_holdout else 0
    external, remaining = [], []
    acc = 0
    for fam, ids in rest:
        if acc < ext_target:
            external.extend(ids)
            acc += len(ids)
        else:
            remaining.append((fam, ids))

    # 3) five outer folds, families assigned greedily to the smallest fold
    n = cfg.protocol.n_outer_folds
    folds: dict[str, list[str]] = {f"F{i}": [] for i in range(n)}
    for fam, ids in sorted(remaining, key=lambda x: -len(x[1])):
        target = min(folds, key=lambda k: len(folds[k]))
        folds[target].extend(ids)
    folds = {k: sorted(v) for k, v in folds.items()}

    family_of = dict(zip(diseases["disease_id"], diseases["family_group"]))
    spec = SplitSpec(name="main_5fold", dev_diseases=sorted(dev),
                     external_diseases=sorted(external), folds=folds,
                     family_of=family_of, seed=cfg.protocol.split_seed)
    spec.spec_hash = sha256_obj({"dev": spec.dev_diseases, "ext": spec.external_diseases,
                                 "folds": spec.folds})
    return spec


def build_family_split(cfg: Gpu8Config, spec: SplitSpec) -> dict:
    """Frozen family-level 60/20/20 split over the non-dev, non-external pool."""
    pool_fams: dict[str, list[str]] = {}
    for ids in spec.folds.values():
        for d in ids:
            pool_fams.setdefault(spec.family_of.get(d, d), []).append(d)
    keys = sorted(pool_fams)
    rng = np.random.default_rng(cfg.protocol.split_seed + 7)
    rng.shuffle(keys)
    total = sum(len(pool_fams[k]) for k in keys)
    want_train, want_val = 0.6 * total, 0.2 * total
    train, val, test = [], [], []
    acc = 0
    for k in keys:
        ids = sorted(pool_fams[k])
        if acc < want_train:
            train.extend(ids)
        elif acc < want_train + want_val:
            val.extend(ids)
        else:
            test.extend(ids)
        acc += len(ids)
    out = {"split_type": "family_isolation", "seed": cfg.protocol.split_seed + 7,
           "train_diseases": sorted(train), "val_diseases": sorted(val),
           "test_diseases": sorted(test),
           "n_train": len(train), "n_val": len(val), "n_test": len(test)}
    out["spec_hash"] = sha256_obj({k: out[k] for k in
                                   ("train_diseases", "val_diseases", "test_diseases")})
    return out


def fold_assignment(spec: SplitSpec, fold: int, n_folds: int = 5) -> dict:
    """Test = F_fold, validation = F_{fold+1 mod n}, training = the rest."""
    test = spec.folds[f"F{fold}"]
    val = spec.folds[f"F{(fold + 1) % n_folds}"]
    train: list[str] = []
    for i in range(n_folds):
        if i in (fold, (fold + 1) % n_folds):
            continue
        train.extend(spec.folds[f"F{i}"])
    return {"fold": fold, "test_diseases": sorted(test),
            "val_diseases": sorted(val), "train_diseases": sorted(train)}


def inner_blocks(train_diseases: list[str], family_of: dict[str, str],
                 n_blocks: int, seed: int) -> list[list[str]]:
    """Split the outer training diseases into n_blocks by family."""
    fam: dict[str, list[str]] = {}
    for d in train_diseases:
        fam.setdefault(family_of.get(d, d), []).append(d)
    keys = sorted(fam)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    blocks: list[list[str]] = [[] for _ in range(n_blocks)]
    for k in sorted(keys, key=lambda x: -len(fam[x])):
        target = min(range(n_blocks), key=lambda i: len(blocks[i]))
        blocks[target].extend(sorted(fam[k]))
    return [sorted(b) for b in blocks]


def context_specs(cfg: Gpu8Config, spec: SplitSpec) -> list[dict]:
    """All graph contexts: 4 per outer fold (3 inner + 1 outer) = 20."""
    out: list[dict] = []
    for fold in range(cfg.protocol.n_outer_folds):
        fa = fold_assignment(spec, fold, cfg.protocol.n_outer_folds)
        blocks = inner_blocks(fa["train_diseases"], spec.family_of,
                              cfg.protocol.inner_blocks, cfg.protocol.split_seed + 100 + fold)
        for bi, block in enumerate(blocks):
            reference = sorted(set(fa["train_diseases"]) - set(block))
            out.append({
                "context_id": f"f{fold}_inner{bi}",
                "fold": fold, "kind": "inner", "block_index": bi,
                "held_out_block": block,
                "reference_diseases": reference,
                "n_reference": len(reference), "n_held_out": len(block),
            })
        out.append({
            "context_id": f"f{fold}_outer",
            "fold": fold, "kind": "outer", "block_index": -1,
            "held_out_block": [],
            "reference_diseases": sorted(fa["train_diseases"]),
            "n_reference": len(fa["train_diseases"]), "n_held_out": 0,
        })
    return out


def support_feasibility(cfg: Gpu8Config, tables_dir: Path | None = None) -> pd.DataFrame:
    """Per-disease support/query eligibility inventory (plan appendix A.3)."""
    tables_dir = Path(tables_dir) if tables_dir else paths.TABLES_DIR
    relations = pd.read_parquet(tables_dir / "relations.parquet")
    evidence = pd.read_parquet(tables_dir / "evidence.parquet")
    diseases = pd.read_parquet(tables_dir / "diseases.parquet")

    primary = set(cfg.data.primary_evidence)
    rel = relations[relations["relation_type"].isin(primary)].copy()

    ev = evidence.copy()
    ev["has_pmid"] = (ev["pmid"].astype(str).str.len() > 0).astype(int)
    pmid_per_rel = ev.groupby("relation_id")["has_pmid"].sum()
    npmids = ev[ev["has_pmid"] == 1].groupby("relation_id")["pmid"].nunique()

    rel["n_pmid"] = rel["relation_id"].map(npmids).fillna(0).astype(int)
    rel["with_source"] = (rel["n_pmid"] > 0).astype(int)

    # evidence groups: relations sharing at least one PMID form one group
    pmid_map = ev[ev["has_pmid"] == 1].groupby("relation_id")["pmid"].apply(set).to_dict()
    rows = []
    hpo_cov = dict(zip(diseases["disease_id"], diseases["hpo_coverage"]))
    fam = dict(zip(diseases["disease_id"], diseases["family_group"]))
    for did, sub in rel.groupby("disease_id"):
        rel_ids = list(sub["relation_id"])
        # union-find over shared PMIDs
        parent: dict[str, str] = {r: r for r in rel_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        pmid_owner: dict[str, str] = {}
        for r in rel_ids:
            for p in pmid_map.get(r, ()):  # type: ignore[arg-type]
                if p in pmid_owner:
                    a, b = find(pmid_owner[p]), find(r)
                    if a != b:
                        parent[b] = a
                else:
                    pmid_owner[p] = r
        n_groups = len({find(r) for r in rel_ids})
        n_pos = len(sub)
        rows.append({
            "disease_id": did,
            "family_group": fam.get(did, ""),
            "n_unique_positives": int(n_pos),
            "n_evidence_groups": int(n_groups),
            "frac_with_source": float(sub["with_source"].mean()),
            "n_pmid_total": int(sub["n_pmid"].sum()),
            "duplicate_record_ratio": float(1.0 - (n_groups / max(1, n_pos))),
            "hpo_coverage": int(hpo_cov.get(did, 0)),
            "eligible_zero": 1,
            "eligible_1shot": int(n_pos >= 2),
            "eligible_3shot": int(n_pos >= 4),
            "eligible_5shot": int(n_pos >= cfg.protocol.min_positives_for_dose),
            "n_query_if_5shot": int(max(0, n_pos - 5)),
            "single_positive_only": int(n_pos == 1),
        })
    return pd.DataFrame(rows).sort_values("disease_id").reset_index(drop=True)


def run(cfg: Gpu8Config | None = None, tables_dir: Path | None = None) -> dict:
    cfg = cfg or load_config()
    spec = build_splits(cfg, tables_dir)
    fam_split = build_family_split(cfg, spec)
    ctxs = context_specs(cfg, spec)
    feas = support_feasibility(cfg, tables_dir)

    atomic_write_json(paths.MANIFEST_DIR / "splits_main_5fold.json", spec.to_dict())
    atomic_write_json(paths.MANIFEST_DIR / "splits_family.json", fam_split)
    atomic_write_json(paths.MANIFEST_DIR / "contexts.json",
                      {"n_contexts": len(ctxs), "contexts": ctxs,
                       "spec_hash": spec.spec_hash})
    atomic_write_df(feas, paths.MANIFEST_DIR / "support_feasibility.csv")

    report = {
        "n_labelled_diseases": len(spec.family_of),
        "n_dev": len(spec.dev_diseases),
        "n_external": len(spec.external_diseases),
        "fold_sizes": {k: len(v) for k, v in spec.folds.items()},
        "n_contexts": len(ctxs),
        "family_split": {k: fam_split[k] for k in ("n_train", "n_val", "n_test")},
        "spec_hash": spec.spec_hash,
        "support_eligibility": {
            "diseases": int(len(feas)),
            "eligible_1shot": int(feas["eligible_1shot"].sum()),
            "eligible_3shot": int(feas["eligible_3shot"].sum()),
            "eligible_5shot": int(feas["eligible_5shot"].sum()),
            "single_positive_only": int(feas["single_positive_only"].sum()),
            "median_positives": float(feas["n_unique_positives"].median()),
            "median_evidence_groups": float(feas["n_evidence_groups"].median()),
            "mean_frac_with_source": float(feas["frac_with_source"].mean()),
        },
    }
    atomic_write_json(paths.MANIFEST_DIR / "protocol_report.json", report)
    return report


def load_spec() -> SplitSpec:
    blob = json.loads((paths.MANIFEST_DIR / "splits_main_5fold.json").read_text(encoding="utf-8"))
    return SplitSpec(name=blob["name"], dev_diseases=blob["dev_diseases"],
                     external_diseases=blob["external_diseases"], folds=blob["folds"],
                     family_of=blob["family_of"], seed=blob["seed"],
                     spec_hash=blob.get("spec_hash", ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Freeze disease splits and contexts.")
    ap.add_argument("--tables_dir", type=Path, default=None)
    args = ap.parse_args(argv)
    print(json.dumps(run(tables_dir=args.tables_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

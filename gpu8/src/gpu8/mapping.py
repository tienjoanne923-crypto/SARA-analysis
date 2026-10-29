"""Disease identifier harmonisation and family grouping (plan section 5.3, 6.4).

Two jobs:

1. Attach Mondo identities to the CTD disease space.  Only *strict* equivalence
   (``source="MONDO:equivalentTo"`` in the OBO, or ``skos:exactMatch`` in the
   SSSOM file) merges two diseases; a broad parent term never absorbs a subtype
   just because they share an ancestor.
2. Derive frozen disease *families* for the family-isolation split and for the
   grouped bootstrap.  Families come from ontology structure and equivalence,
   never from model behaviour.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from . import paths
from .config import load_config
from .hashing import atomic_write_df, atomic_write_json, sha256_obj

MONDO_XREF_RE = re.compile(r"^xref:\s*(\S+)(?:\s*\{(.*)\})?\s*$")
EQUIV_TAG = 'source="MONDO:equivalentTo"'


# --------------------------------------------------------------------------
# Mondo parsing
# --------------------------------------------------------------------------
def parse_mondo_equivalences(obo_path: Path) -> dict[str, list[str]]:
    """external CURIE -> [MONDO ids] for strict equivalences only."""
    out: dict[str, list[str]] = defaultdict(list)
    if not Path(obo_path).exists():
        return {}
    current = ""
    in_term = False
    with open(obo_path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line == "[Term]":
                in_term, current = True, ""
                continue
            if line.startswith("[") and line.endswith("]"):
                in_term, current = False, ""
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                current = line[4:].strip()
            elif line.startswith("is_obsolete: true"):
                current = ""
            elif line.startswith("xref: ") and current:
                m = MONDO_XREF_RE.match(line)
                if not m:
                    continue
                curie, annots = m.group(1), (m.group(2) or "")
                if EQUIV_TAG in annots:
                    out[normalize_curie(curie)].append(current)
    return {k: sorted(set(v)) for k, v in out.items()}


def parse_mondo_sssom(path: Path) -> dict[str, list[str]]:
    """external CURIE -> [MONDO ids] from skos:exactMatch rows."""
    if not Path(path).exists():
        return {}
    rows: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header: list[str] | None = None
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = [c.strip() for c in fields]
                continue
            rec = dict(zip(header, fields))
            if rec.get("predicate_id", "").strip() != "skos:exactMatch":
                continue
            subj = rec.get("subject_id", "").strip()
            obj = rec.get("object_id", "").strip()
            if subj.startswith("MONDO:") and obj:
                rows.append((normalize_curie(obj), subj))
            elif obj.startswith("MONDO:") and subj:
                rows.append((normalize_curie(subj), obj))
    out: dict[str, list[str]] = defaultdict(list)
    for ext, mondo in rows:
        out[ext].append(mondo)
    return {k: sorted(set(v)) for k, v in out.items()}


def normalize_curie(curie: str) -> str:
    """Uppercase the prefix and strip duplicated namespaces."""
    c = (curie or "").strip()
    if not c:
        return ""
    while c.count(":") > 1 and c.split(":")[0].upper() == c.split(":")[1].upper():
        c = c.split(":", 1)[1]
    if ":" not in c:
        return c
    prefix, _, rest = c.partition(":")
    prefix = prefix.upper()
    aliases = {"MESH": "MESH", "OMIM": "OMIM", "ORPHANET": "Orphanet",
               "ORPHA": "Orphanet", "MONDO": "MONDO", "DOID": "DOID",
               "UMLS": "UMLS", "MEDGEN": "MedGen", "NCIT": "NCIT",
               "EFO": "EFO", "MEDDRA": "MedDRA", "ICD10CM": "ICD10CM"}
    return f"{aliases.get(prefix, prefix)}:{rest}"


# --------------------------------------------------------------------------
# attaching Mondo to the project disease space
# --------------------------------------------------------------------------
def disease_external_curies(row: pd.Series) -> list[str]:
    """All CURIEs a project disease can be looked up by."""
    out = [normalize_curie(str(row["disease_id"]))]
    for field in ("source_ids", "omim_ids"):
        raw = str(row.get(field) or "")
        for part in re.split(r"[|,;]", raw):
            part = part.strip()
            if part:
                out.append(normalize_curie(part))
    return sorted({c for c in out if c})


def attach_mondo(diseases: pd.DataFrame, equiv: dict[str, list[str]]) -> pd.DataFrame:
    mondo_ids, mapping_type, all_curies = [], [], []
    for _, row in diseases.iterrows():
        curies = disease_external_curies(row)
        hits: list[str] = []
        for c in curies:
            hits.extend(equiv.get(c, []))
        hits = sorted(set(hits))
        all_curies.append("|".join(curies))
        if len(hits) == 1:
            mondo_ids.append(hits[0])
            mapping_type.append("mondo_exact_unique")
        elif len(hits) > 1:
            mondo_ids.append("|".join(hits))
            mapping_type.append("mondo_exact_multiple")
        else:
            mondo_ids.append("")
            mapping_type.append("unmapped")
    out = diseases.copy()
    out["mondo_ids"] = mondo_ids
    out["mapping_type"] = mapping_type
    out["external_curies"] = all_curies
    return out


def build_equivalence_groups(diseases: pd.DataFrame) -> pd.DataFrame:
    """Union-find over unique Mondo equivalence; unmapped diseases stay singletons."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, row in diseases.iterrows():
        did = row["disease_id"]
        find(did)
        if row["mapping_type"] == "mondo_exact_unique" and row["mondo_ids"]:
            union(f"MONDO_KEY::{row['mondo_ids']}", did)

    groups = {}
    for _, row in diseases.iterrows():
        did = row["disease_id"]
        root = find(did)
        groups[did] = root if not root.startswith("MONDO_KEY::") else root.split("::", 1)[1]
    out = diseases.copy()
    out["equivalence_group"] = out["disease_id"].map(groups)
    return out


# --------------------------------------------------------------------------
# family grouping
# --------------------------------------------------------------------------
def primary_tree_prefix(tree_numbers: str, depth: int = 3) -> str:
    """The single MeSH tree branch that owns a disease.

    A disease carrying several tree numbers is assigned to the lexicographically
    smallest branch rather than linking all of its branches together.  Transitive
    linking was tried first and produced one component of over nine thousand
    diseases, which would make family isolation meaningless.
    """
    best = ""
    for t in re.split(r"[|,;]", str(tree_numbers or "")):
        t = t.strip()
        if not t:
            continue
        parts = t.split(".")
        key = ".".join(parts[:depth])
        if not best or key < best:
            best = key
    return best


def build_family_groups(diseases: pd.DataFrame, hierarchy: pd.DataFrame,
                        max_depth: int = 3, tree_depth: int = 3) -> pd.DataFrame:
    """Assign each disease to exactly one frozen family.

    Priority: own MeSH branch -> nearest ancestor's branch -> equivalence group.
    Deterministic, bounded, and fixed before any modelling.
    """
    own: dict[str, str] = {}
    for row in diseases.itertuples(index=False):
        pref = primary_tree_prefix(getattr(row, "tree_numbers", ""), tree_depth)
        if pref:
            own[row.disease_id] = pref

    parents: dict[str, list[str]] = defaultdict(list)
    if hierarchy is not None and not hierarchy.empty:
        for r in hierarchy.itertuples(index=False):
            parents[r.child_disease].append(r.parent_disease)

    def nearest_ancestor_prefix(did: str) -> str:
        seen, frontier = {did}, deque([(did, 0)])
        while frontier:
            node, d = frontier.popleft()
            if d >= max_depth:
                continue
            for p in sorted(parents.get(node, [])):
                if p in seen:
                    continue
                seen.add(p)
                if p in own:
                    return own[p]
                frontier.append((p, d + 1))
        return ""

    equiv = dict(zip(diseases["disease_id"], diseases["equivalence_group"]))
    fam: dict[str, str] = {}
    source: dict[str, str] = {}
    for did in diseases["disease_id"]:
        if did in own:
            fam[did], source[did] = f"MESH:{own[did]}", "own_tree"
        else:
            pref = nearest_ancestor_prefix(did)
            if pref:
                fam[did], source[did] = f"MESH:{pref}", "ancestor_tree"
            else:
                fam[did], source[did] = f"EQ:{equiv.get(did, did)}", "equivalence_singleton"

    out = diseases.copy()
    out["family_group"] = out["disease_id"].map(fam)
    out["family_source"] = out["disease_id"].map(source)
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run(tables_dir: Path | None = None) -> dict:
    tables_dir = Path(tables_dir) if tables_dir else paths.TABLES_DIR
    cfg = load_config()
    diseases = pd.read_parquet(tables_dir / "diseases.parquet")
    hierarchy = pd.read_parquet(tables_dir / "hierarchy.parquet")

    equiv = parse_mondo_equivalences(paths.external("mondo", "mondo.obo"))
    sssom = parse_mondo_sssom(paths.external("mondo", "mondo.sssom.tsv"))
    merged: dict[str, list[str]] = defaultdict(list)
    for src in (equiv, sssom):
        for k, v in src.items():
            merged[k].extend(v)
    merged = {k: sorted(set(v)) for k, v in merged.items()}

    diseases = attach_mondo(diseases, merged)
    diseases = build_equivalence_groups(diseases)
    diseases = build_family_groups(diseases, hierarchy)

    atomic_write_df(diseases, tables_dir / "diseases.parquet")

    fam_sizes = diseases.groupby("family_group").size()
    report = {
        "n_diseases": int(len(diseases)),
        "mondo_source_terms": len(merged),
        "mapping_type_counts": diseases["mapping_type"].value_counts().to_dict(),
        "n_equivalence_groups": int(diseases["equivalence_group"].nunique()),
        "n_family_groups": int(diseases["family_group"].nunique()),
        "family_source_counts": diseases["family_source"].value_counts().to_dict(),
        "family_size_max": int(fam_sizes.max()) if len(fam_sizes) else 0,
        "family_size_p95": float(fam_sizes.quantile(0.95)) if len(fam_sizes) else 0.0,
        "family_size_median": float(fam_sizes.median()) if len(fam_sizes) else 0.0,
        "family_singletons": int((fam_sizes == 1).sum()),
        "mapping_hash": sha256_obj(
            diseases[["disease_id", "equivalence_group", "family_group"]]
            .sort_values("disease_id").values.tolist()),
    }
    atomic_write_json(paths.MANIFEST_DIR / "disease_mapping_report.json", report)

    # exclusion / ambiguity log required by T03
    amb = diseases[diseases["mapping_type"] == "mondo_exact_multiple"][
        ["disease_id", "disease_name", "mondo_ids", "external_curies"]]
    atomic_write_df(amb, paths.MANIFEST_DIR / "disease_mapping_conflicts.csv")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Harmonise disease identifiers and families.")
    ap.add_argument("--tables_dir", type=Path, default=None)
    args = ap.parse_args(argv)
    print(json.dumps(run(args.tables_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

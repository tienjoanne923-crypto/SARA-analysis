"""V6 section 4.1 — repaired disease phenotype table.

Two fixes, both decided before any model is fitted and both given to every
comparator:

1. **strict mapping repair.**  V5 attached HPO annotations only through CTD's
   own OMIM alternative identifiers.  V6 additionally follows *strict*
   equivalence: project disease -> unique Mondo class (exact / equivalentTo) ->
   OMIM or Orphanet identifiers that are themselves exact matches of that same
   Mondo class and of no other.  Broad, narrow, close or related matches and
   one-to-many mappings are logged but never used.
2. **term filtering.**  Only the phenotypic-abnormality aspect (P) is kept;
   inheritance mode, clinical course, modifiers and frequency terms are removed,
   and so are over-general terms (depth <= 2 below the HPO root, i.e. organ-
   system headings).  NOT-qualified annotations are dropped, never positive.

Every row keeps its provenance (``mapping_path`` and the HPOA database id), and
the number of newly mappable diseases and annotations is reported separately.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from gpu8 import paths as p5
from gpu8.ingest import build_disease_vocabulary
from gpu8.mapping import normalize_curie, parse_mondo_equivalences

from . import paths6
from .common import dump_json, now_utc

NON_SYMPTOM_ROOTS = {
    "HP:0000005": "mode_of_inheritance",
    "HP:0012823": "clinical_modifier",
    "HP:0031797": "clinical_course",
    "HP:0040279": "frequency",
    "HP:0032443": "past_medical_history",
    "HP:0032223": "blood_group",
}
PHENOTYPIC_ABNORMALITY = "HP:0000118"


# --------------------------------------------------------------------------
# HPO structure
# --------------------------------------------------------------------------
def hpo_parents(hpo_terms: pd.DataFrame) -> dict[str, list[str]]:
    out = {}
    for r in hpo_terms.itertuples(index=False):
        ps = [p for p in str(r.is_a or "").split("|") if p]
        out[r.hpo_id] = ps
    return out


def hpo_depth_and_roots(parents: dict[str, list[str]]) -> tuple[dict, dict]:
    """Minimum depth below HP:0000001 and the set of top-level branches of each term."""
    depth: dict[str, int] = {}
    branch: dict[str, set] = {}

    def resolve(t, stack=()):
        if t in depth:
            return depth[t], branch[t]
        ps = parents.get(t, [])
        if not ps or t == "HP:0000001":
            depth[t], branch[t] = 0, {t}
            return 0, branch[t]
        best, br = 10 ** 6, set()
        for p in ps:
            if p in stack:
                continue
            d, b = resolve(p, stack + (t,))
            best = min(best, d + 1)
            br |= b if d > 0 else {t}
        depth[t] = best if best < 10 ** 6 else 0
        branch[t] = br or {t}
        return depth[t], branch[t]

    import sys
    sys.setrecursionlimit(100000)
    for t in list(parents):
        resolve(t)
    return depth, branch


def hpo_ancestors(parents: dict[str, list[str]]) -> dict[str, set]:
    anc: dict[str, set] = {}

    def get(t):
        if t in anc:
            return anc[t]
        acc = set()
        for p in parents.get(t, []):
            acc.add(p)
            acc |= get(p)
        anc[t] = acc
        return acc

    import sys
    sys.setrecursionlimit(100000)
    for t in list(parents):
        get(t)
    return anc


def term_class(term: str, depth: dict, anc: dict) -> str:
    """valid | too_general | <non-symptom branch name> | unknown_term"""
    if term not in depth:
        return "unknown_term"
    a = anc.get(term, set()) | {term}
    for root, name in NON_SYMPTOM_ROOTS.items():
        if root in a:
            return name
    if PHENOTYPIC_ABNORMALITY not in a:
        return "outside_phenotypic_abnormality"
    # depth 1 = Phenotypic abnormality, depth 2 = organ-system heading
    if depth.get(term, 0) <= 2:
        return "too_general"
    return "valid"


# --------------------------------------------------------------------------
# HPOA and Mondo
# --------------------------------------------------------------------------
def read_hpoa(path: Path) -> pd.DataFrame:
    rows = []
    header = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if header is None:
                header = [c.strip() for c in f]
                continue
            rows.append(f)
    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={"database_id": "hpoa_id", "reference": "source"})
    df["negated"] = (df["qualifier"].str.strip().str.upper() == "NOT").astype(int)
    return df[["hpoa_id", "disease_name", "hpo_id", "negated", "frequency", "source",
               "aspect", "biocuration"]]


def hpoa_curie(ext: str) -> str:
    """Mondo-normalised CURIE -> the identifier style used in phenotype.hpoa."""
    if ext.startswith("Orphanet:"):
        return "ORPHA:" + ext.split(":", 1)[1]
    return ext


def parse_sssom_all(path: Path) -> pd.DataFrame:
    rows = []
    header = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if header is None:
                header = [c.strip() for c in f]
                continue
            rec = dict(zip(header, f))
            s, o = rec.get("subject_id", "").strip(), rec.get("object_id", "").strip()
            pred = rec.get("predicate_id", "").strip()
            if s.startswith("MONDO:"):
                rows.append((s, normalize_curie(o), pred))
            elif o.startswith("MONDO:"):
                rows.append((o, normalize_curie(s), pred))
    return pd.DataFrame(rows, columns=["mondo", "ext", "predicate"])


def mondo_maps() -> tuple[dict, dict, dict]:
    """(ext -> strict mondo set, mondo -> strict ext set, mondo -> non-strict ext set)."""
    equiv = parse_mondo_equivalences(p5.external("mondo", "mondo.obo"))
    sssom = parse_sssom_all(p5.external("mondo", "mondo.sssom.tsv"))
    ext_to_m: dict[str, set] = defaultdict(set)
    m_to_ext: dict[str, set] = defaultdict(set)
    m_to_loose: dict[str, set] = defaultdict(set)
    for ext, ms in equiv.items():
        for m in ms:
            ext_to_m[ext].add(m)
            m_to_ext[m].add(ext)
    for r in sssom.itertuples(index=False):
        if r.predicate == "skos:exactMatch":
            ext_to_m[r.ext].add(r.mondo)
            m_to_ext[r.mondo].add(r.ext)
        else:
            m_to_loose[r.mondo].add(r.ext)
    return ext_to_m, m_to_ext, m_to_loose


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def build(write: bool = True) -> dict:
    diseases = pd.read_parquet(p5.TABLES_DIR / "diseases.parquet")
    hpo_terms = pd.read_parquet(p5.TABLES_DIR / "hpo_terms.parquet")
    parents = hpo_parents(hpo_terms)
    depth, _ = hpo_depth_and_roots(parents)
    anc = hpo_ancestors(parents)

    hpoa = read_hpoa(p5.PHENOTYPE_DIR / "phenotype.hpoa")
    hpoa_ids = set(hpoa["hpoa_id"])
    _, alias_to_disease, _ = build_disease_vocabulary(p5.ASSOCIATION_DIR / "CTD_diseases.tsv.gz")
    project = set(diseases["disease_id"])

    # V5 path: CTD DiseaseID / AltDiseaseIDs, first claim wins (as in ingest.py)
    v5_map = {h: alias_to_disease[h] for h in hpoa_ids
              if h in alias_to_disease and alias_to_disease[h] in project}

    ext_to_m, m_to_ext, m_to_loose = mondo_maps()
    strict_map: dict[str, str] = {}
    strict_conflicts: list[dict] = []
    loose_reach: dict[str, set] = defaultdict(set)
    for r in diseases.itertuples(index=False):
        if r.mapping_type != "mondo_exact_unique" or not r.mondo_ids:
            continue
        m = r.mondo_ids
        for ext in m_to_ext.get(m, ()):
            if not (ext.startswith("OMIM:") or ext.startswith("Orphanet:")):
                continue
            h = hpoa_curie(ext)
            if h not in hpoa_ids:
                continue
            if len(ext_to_m.get(ext, ())) != 1:
                strict_conflicts.append({"disease_id": r.disease_id, "hpoa_id": h,
                                         "reason": "external id maps to several Mondo classes"})
                continue
            prev = strict_map.get(h)
            if prev is not None and prev != r.disease_id:
                strict_conflicts.append({"disease_id": r.disease_id, "hpoa_id": h,
                                         "reason": f"also strict for {prev}"})
                strict_map[h] = "__AMBIGUOUS__"
                continue
            strict_map[h] = r.disease_id
        for ext in m_to_loose.get(m, ()):
            h = hpoa_curie(ext)
            if h in hpoa_ids:
                loose_reach[r.disease_id].add(h)
    strict_map = {h: d for h, d in strict_map.items() if d != "__AMBIGUOUS__"}

    # rows by path; a V5 row keeps its V5 path even if the strict path agrees
    hp = hpoa.copy()
    hp["disease_v5"] = hp["hpoa_id"].map(v5_map)
    hp["disease_strict"] = hp["hpoa_id"].map(strict_map)
    v5_rows = hp[hp["disease_v5"].notna()].assign(disease_id=lambda x: x["disease_v5"],
                                                   mapping_path="v5_ctd_alt_id")
    add = hp[hp["disease_strict"].notna()
             & (hp["disease_v5"].isna() | (hp["disease_v5"] != hp["disease_strict"]))]
    add = add.assign(disease_id=lambda x: x["disease_strict"], mapping_path="v6_mondo_strict")
    allrows = pd.concat([v5_rows, add], ignore_index=True)
    allrows["term_class"] = [term_class(t, depth, anc) for t in allrows["hpo_id"]]
    allrows["keep_v6"] = ((allrows["negated"] == 0) & (allrows["aspect"] == "P")
                          & (allrows["term_class"] == "valid")).astype(int)
    cols = ["disease_id", "hpo_id", "negated", "frequency", "source", "aspect",
            "hpoa_id", "mapping_path", "term_class", "keep_v6"]
    allrows = allrows[cols].drop_duplicates(subset=["disease_id", "hpo_id", "negated",
                                                    "hpoa_id"])
    kept = allrows[allrows["keep_v6"] == 1].drop_duplicates(subset=["disease_id", "hpo_id"])

    # disease-level classification (section 4.1: three kinds of "no HPO")
    v5_any = set(allrows.loc[allrows["mapping_path"] == "v5_ctd_alt_id", "disease_id"])
    v5_valid = set(allrows.loc[(allrows["mapping_path"] == "v5_ctd_alt_id")
                               & (allrows["keep_v6"] == 1), "disease_id"])
    strict_any = set(allrows.loc[allrows["mapping_path"] == "v6_mondo_strict", "disease_id"])
    v6_valid = set(kept["disease_id"])
    v5_h = pd.read_parquet(p5.TABLES_DIR / "phenotypes.parquet")
    v5_h_any = set(v5_h.loc[v5_h["negated"] == 0, "disease_id"])
    recs = []
    for did in diseases["disease_id"]:
        if did in v5_valid:
            cls = "valid_hpo_v5"
        elif did in v5_any or did in v5_h_any:
            cls = ("filtered_empty_repaired" if did in v6_valid
                   else "record_present_no_valid_symptom")
        elif did in strict_any:
            cls = ("mapping_failure_repaired" if did in v6_valid
                   else "mapping_failure_repaired_but_filtered_empty")
        elif loose_reach.get(did):
            cls = "mapping_nonstrict_only"
        else:
            cls = "no_resource_record"
        recs.append({"disease_id": did, "hpo_class": cls,
                     "n_hpo_terms_v5": int(len(set(v5_h.loc[(v5_h["disease_id"] == did)
                                                            & (v5_h["negated"] == 0), "hpo_id"])))
                     if did in v5_h_any else 0,
                     "n_hpo_valid_v6": 0, "n_loose_hpoa_ids": len(loose_reach.get(did, ()))})
    cls_df = pd.DataFrame(recs)
    nv6 = kept.groupby("disease_id")["hpo_id"].nunique()
    cls_df["n_hpo_valid_v6"] = cls_df["disease_id"].map(nv6).fillna(0).astype(int)

    term_stats = allrows.groupby(["mapping_path", "term_class"]).size().unstack(fill_value=0)
    report = {
        "generated_utc": now_utc(),
        "hpoa_file": p5.rel(p5.PHENOTYPE_DIR / "phenotype.hpoa"),
        "n_hpoa_database_ids": len(hpoa_ids),
        "n_hpoa_ids_mapped_v5": len(v5_map),
        "n_hpoa_ids_mapped_strict_v6": len(strict_map),
        "n_hpoa_ids_new_in_v6": int(len(set(strict_map) - set(v5_map))),
        "n_strict_conflicts_dropped": len(strict_conflicts),
        "n_diseases_with_any_v5_row": len(v5_any),
        "n_diseases_valid_v5": len(v5_valid),
        "n_diseases_valid_v6": len(v6_valid),
        "n_diseases_gaining_valid_hpo": len(v6_valid - v5_valid),
        "n_rows_all": int(len(allrows)), "n_rows_kept_v6": int(len(kept)),
        "n_rows_added_by_strict_path": int((allrows["mapping_path"] == "v6_mondo_strict").sum()),
        "hpo_class_counts_all_diseases": cls_df["hpo_class"].value_counts().to_dict(),
        "term_class_by_path": {k: {kk: int(vv) for kk, vv in v.items()}
                               for k, v in term_stats.to_dict(orient="index").items()},
        "rules": {"strict": "unique Mondo exact/equivalentTo on both sides; one project disease per HPOA id",
                  "kept_terms": "aspect P, not NOT, depth > 2 below HP:0000001, outside inheritance/course/modifier/frequency branches",
                  "decided_before_modelling": True},
    }
    if write:
        paths6.TABLES.mkdir(parents=True, exist_ok=True)
        idmap = pd.concat([
            pd.DataFrame({"hpoa_id": list(v5_map), "disease_id": list(v5_map.values()),
                          "path": "v5_ctd_alt_id"}),
            pd.DataFrame({"hpoa_id": list(strict_map), "disease_id": list(strict_map.values()),
                          "path": "v6_mondo_strict"})], ignore_index=True)
        idmap.to_parquet(paths6.TABLES / "hpoa_id_map.parquet", index=False)
        allrows.to_parquet(paths6.TABLES / "phenotypes_v6_all_rows.parquet", index=False)
        kept.to_parquet(paths6.TABLES / "phenotypes_v6.parquet", index=False)
        cls_df.to_parquet(paths6.TABLES / "disease_hpo_class.parquet", index=False)
        pd.DataFrame(strict_conflicts).to_csv(paths6.AUDIT / "hpo_strict_mapping_conflicts.csv",
                                              index=False, encoding="utf-8-sig")
        dump_json(paths6.AUDIT / "phenotypes_v6_report.json", report)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the repaired V6 phenotype table.")
    ap.parse_args(argv)
    paths6.ensure()
    rep = build()
    print(json.dumps({k: v for k, v in rep.items() if k != "term_class_by_path"},
                     indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

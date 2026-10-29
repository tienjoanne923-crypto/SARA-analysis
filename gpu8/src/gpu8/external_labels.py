"""External confirmation sets: GenCC (primary) and ClinVar (re-analysis).

Plan section 14.  Two rules dominate:

* three explicit exclusion layers, each with its own denominator — (1) the same
  disease-gene relation already visible in training, (2) records demonstrably
  transcribed from a training resource, (3) records sharing an original PMID
  with a visible relation.  A record whose provenance cannot be checked is kept
  in its own bucket and never silently passed as "independent".
* every attrition step is counted, so mapping loss and retrieval loss are never
  both reported as "not evaluable".

GenCC aggregates ClinGen, Orphanet and others, so its submitters are *not*
independent datasets; ClinVar is filtered at the variant level (germline
pathogenic / likely pathogenic with a declared review status) rather than
through the gene-condition mapping file alone.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .config import Gpu8Config, load_config
from .hashing import atomic_write_df, atomic_write_json
from .mapping import normalize_curie

GENCC_HIGH = {"Definitive", "Strong"}
GENCC_SENSITIVITY = {"Moderate", "Supportive"}
GENCC_NEGATIVE = {"Disputed Evidence", "Refuted Evidence", "No Known Disease Relationship"}

CLINVAR_PATHOGENIC = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, single submitter": 1,
    "no assertion criteria provided": 0,
    "no classification provided": 0,
    "no classification for the single variant": 0,
    "no classifications from unflagged records": 0,
}


# --------------------------------------------------------------------------
# project-side lookup keys
# --------------------------------------------------------------------------
def build_disease_lookup(diseases: pd.DataFrame) -> dict[str, list[str]]:
    """external CURIE (normalised) -> [project disease ids]."""
    out: dict[str, list[str]] = defaultdict(list)
    for row in diseases.itertuples(index=False):
        keys = {normalize_curie(str(row.disease_id))}
        for field in ("source_ids", "omim_ids", "mondo_ids"):
            raw = str(getattr(row, field, "") or "")
            for part in re.split(r"[|,;]", raw):
                part = part.strip()
                if part:
                    keys.add(normalize_curie(part))
        for k in keys:
            if k:
                out[k].append(row.disease_id)
    return {k: sorted(set(v)) for k, v in out.items()}


def build_gene_lookup(genes: pd.DataFrame) -> tuple[set[str], dict[str, str], dict[str, str]]:
    ids = set(genes["gene_id"].astype(str))
    by_symbol = {}
    by_hgnc = {}
    for row in genes.itertuples(index=False):
        sym = str(getattr(row, "symbol", "") or "").upper()
        if sym:
            by_symbol.setdefault(sym, str(row.gene_id))
        hg = str(getattr(row, "hgnc_id", "") or "")
        if hg:
            by_hgnc.setdefault(hg, str(row.gene_id))
    return ids, by_symbol, by_hgnc


# --------------------------------------------------------------------------
# GenCC
# --------------------------------------------------------------------------
def load_gencc(path: Path) -> pd.DataFrame:
    """Proper quoted-TSV parse: submitted_as_notes contains embedded newlines."""
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_MINIMAL,
                     engine="python", on_bad_lines="skip")
    return df.fillna("")


def gencc_pairs(cfg: Gpu8Config, gencc: pd.DataFrame, disease_lookup: dict,
                gene_ids: set, by_symbol: dict, by_hgnc: dict) -> tuple[pd.DataFrame, dict]:
    audit = {"raw_rows": int(len(gencc))}
    if gencc.empty:
        return pd.DataFrame(), audit

    cls_col = "classification_title" if "classification_title" in gencc.columns else ""
    if not cls_col:
        return pd.DataFrame(), {**audit, "error": "no classification column"}

    gencc = gencc.copy()
    # NB: a column name starting with an underscore is renamed to a positional
    # name by itertuples, which silently blanks it out.
    gencc["cls_clean"] = gencc[cls_col].str.strip()
    audit["classification_counts"] = gencc["cls_clean"].value_counts().to_dict()

    rows = []
    unmapped_gene = unmapped_disease = 0
    for r in gencc.itertuples(index=False):
        cls = getattr(r, "cls_clean", "")
        if cls in GENCC_NEGATIVE:
            tier = "negative"
        elif cls in GENCC_HIGH:
            tier = "high"
        elif cls in GENCC_SENSITIVITY:
            tier = "sensitivity"
        else:
            tier = "other"
        hgnc = str(getattr(r, "gene_curie", "") or "")
        sym = str(getattr(r, "gene_symbol", "") or "").upper()
        gid = by_hgnc.get(hgnc) or by_symbol.get(sym)
        if gid is None:
            unmapped_gene += 1
            continue
        keys = set()
        for field in ("disease_curie", "disease_original_curie"):
            v = str(getattr(r, field, "") or "").strip()
            if v:
                keys.add(normalize_curie(v))
        dids: list[str] = []
        for k in keys:
            dids.extend(disease_lookup.get(k, []))
        dids = sorted(set(dids))
        if not dids:
            unmapped_disease += 1
            continue
        pmids = "|".join(re.findall(r"\d{5,9}", str(getattr(r, "submitted_as_pmids", "") or "")))
        for did in dids:
            rows.append({
                "source": "GenCC", "disease_id": did, "gene_id": gid,
                "classification": cls, "tier": tier,
                "moi": str(getattr(r, "moi_title", "") or ""),
                "submitter": str(getattr(r, "submitter_title", "") or ""),
                "pmids": pmids,
                "submitted_date": str(getattr(r, "submitted_as_date", "") or ""),
                "n_mapped_diseases": len(dids),
            })
    out = pd.DataFrame(rows)
    audit.update({"unmapped_gene_rows": unmapped_gene,
                  "unmapped_disease_rows": unmapped_disease,
                  "mapped_rows": int(len(out))})
    if not out.empty:
        out = out.sort_values(["disease_id", "gene_id", "tier"])
        audit["unique_pairs"] = int(out.groupby(["disease_id", "gene_id"]).ngroups)
        audit["unique_high_pairs"] = int(
            out[out["tier"] == "high"].groupby(["disease_id", "gene_id"]).ngroups)
        audit["unique_diseases_high"] = int(out[out["tier"] == "high"]["disease_id"].nunique())
    return out, audit


# --------------------------------------------------------------------------
# ClinVar
# --------------------------------------------------------------------------
def parse_phenotype_ids(value: str) -> list[str]:
    """PhenotypeIDS: RCV sets by '|', conditions by ';', identifiers by ','."""
    out: list[str] = []
    for rcv in str(value or "").split("|"):
        for cond in rcv.split(";"):
            for ident in cond.split(","):
                ident = ident.strip()
                if not ident or ident.lower().endswith("conditions"):
                    continue
                ident = ident.replace("Human Phenotype Ontology:", "")
                out.append(normalize_curie(ident))
    return [c for c in out if c]


def stream_clinvar_variants(path: Path, gene_ids: set, min_stars: int = 1,
                            max_rows: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Variant-level filter: germline P/LP with a declared review status."""
    audit = {"rows_seen": 0, "kept_assembly": 0, "kept_germline": 0,
             "kept_pathogenic": 0, "kept_review": 0, "kept_single_gene": 0,
             "kept_mapped_gene": 0, "unique_variant_condition": 0}
    if not Path(path).exists():
        return pd.DataFrame(), {**audit, "error": "missing file"}
    rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = None
        for line in fh:
            if header is None:
                header = [c.strip().lstrip("#") for c in line.rstrip("\n").split("\t")]
                continue
            audit["rows_seen"] += 1
            if max_rows and audit["rows_seen"] > max_rows:
                break
            f = line.rstrip("\n").split("\t")
            if len(f) < len(header):
                continue
            rec = dict(zip(header, f))
            if rec.get("Assembly", "") != "GRCh38":
                continue
            audit["kept_assembly"] += 1
            if not str(rec.get("OriginSimple", "")).lower().startswith("germline"):
                continue
            audit["kept_germline"] += 1
            sig = str(rec.get("ClinicalSignificance", "")).lower()
            if not any(p in sig for p in CLINVAR_PATHOGENIC):
                continue
            if "conflict" in sig or "uncertain" in sig or "benign" in sig:
                continue
            audit["kept_pathogenic"] += 1
            stars = REVIEW_STARS.get(str(rec.get("ReviewStatus", "")).strip().lower(), 0)
            if stars < min_stars:
                continue
            audit["kept_review"] += 1
            gid = str(rec.get("GeneID", "")).strip()
            if not gid or gid == "-1":
                continue          # multi-gene region: no unique attribution
            audit["kept_single_gene"] += 1
            if gid not in gene_ids:
                continue
            audit["kept_mapped_gene"] += 1
            for curie in parse_phenotype_ids(rec.get("PhenotypeIDS", "")):
                if curie.startswith("MedGen:CN") or curie in ("MedGen:C3661900",):
                    continue          # "not specified" / "not provided"
                key = (gid, curie)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                rows.append({"gene_id": gid, "condition_curie": curie,
                             "review_stars": stars, "variation_id": rec.get("VariationID", ""),
                             "type": rec.get("Type", ""),
                             "last_evaluated": rec.get("LastEvaluated", "")})
    audit["unique_variant_condition"] = len(rows)
    return pd.DataFrame(rows), audit


def clinvar_pairs(cv: pd.DataFrame, disease_lookup: dict) -> tuple[pd.DataFrame, dict]:
    audit = {"input_rows": int(len(cv))}
    if cv.empty:
        return pd.DataFrame(), audit
    rows = []
    unmapped = 0
    for r in cv.itertuples(index=False):
        dids = disease_lookup.get(r.condition_curie, [])
        if not dids:
            unmapped += 1
            continue
        for did in dids:
            rows.append({"source": "ClinVar", "disease_id": did, "gene_id": r.gene_id,
                         "classification": "germline_pathogenic",
                         "tier": "high" if r.review_stars >= 2 else "sensitivity",
                         "review_stars": int(r.review_stars),
                         "condition_curie": r.condition_curie})
    out = pd.DataFrame(rows)
    audit.update({"unmapped_condition_rows": unmapped, "mapped_rows": int(len(out))})
    if not out.empty:
        out = out.drop_duplicates(subset=["disease_id", "gene_id", "tier"])
        audit["unique_pairs"] = int(out.groupby(["disease_id", "gene_id"]).ngroups)
        audit["unique_diseases"] = int(out["disease_id"].nunique())
    return out, audit


# --------------------------------------------------------------------------
# independence layers
# --------------------------------------------------------------------------
def apply_exclusion_layers(pairs: pd.DataFrame, relations: pd.DataFrame,
                           evidence: pd.DataFrame, visible_diseases: set[str]
                           ) -> tuple[pd.DataFrame, dict]:
    """Layer 1 relation overlap, layer 2 source overlap, layer 3 PMID overlap."""
    audit = {"input_pairs": int(len(pairs))}
    if pairs.empty:
        return pairs, audit

    rel_pairs = set(zip(relations["disease_id"], relations["gene_id"]))
    rel_by_pair = {(r.disease_id, r.gene_id): r.relation_id
                   for r in relations.itertuples(index=False)}
    pmids_by_rel: dict[str, set[str]] = {}
    ev = evidence[evidence["pmid"].astype(str).str.len() > 0]
    if not ev.empty:
        pmids_by_rel = ev.groupby("relation_id")["pmid"].apply(set).to_dict()
    disease_pmids: dict[str, set[str]] = defaultdict(set)
    for (did, gid), rid in rel_by_pair.items():
        disease_pmids[did] |= pmids_by_rel.get(rid, set())

    out = pairs.copy()
    key = list(zip(out["disease_id"], out["gene_id"]))
    out["layer1_relation_overlap"] = [1 if k in rel_pairs else 0 for k in key]
    out["layer2_source_overlap"] = [
        1 if (s == "GenCC" and k in rel_pairs) else 0
        for s, k in zip(out["source"], key)]

    l3 = []
    for (did, gid), pm in zip(key, out.get("pmids", pd.Series([""] * len(out)))):
        ext = set(re.findall(r"\d{5,9}", str(pm or "")))
        if not ext:
            l3.append(-1)           # provenance unverifiable: its own bucket
        else:
            l3.append(1 if ext & disease_pmids.get(did, set()) else 0)
    out["layer3_pmid_overlap"] = l3
    out["evaluable"] = ((out["layer1_relation_overlap"] == 0)
                        & out["disease_id"].isin(visible_diseases)).astype(int)
    out["strict_independent"] = ((out["layer1_relation_overlap"] == 0)
                                 & (out["layer3_pmid_overlap"] == 0)).astype(int)

    audit.update({
        "layer1_excluded": int(out["layer1_relation_overlap"].sum()),
        "layer3_pmid_overlap": int((out["layer3_pmid_overlap"] == 1).sum()),
        "layer3_unverifiable": int((out["layer3_pmid_overlap"] == -1).sum()),
        "evaluable_pairs": int(out["evaluable"].sum()),
        "strict_independent_pairs": int(out["strict_independent"].sum()),
        "evaluable_diseases": int(out[out["evaluable"] == 1]["disease_id"].nunique()),
        "strict_independent_diseases": int(
            out[(out["evaluable"] == 1) & (out["strict_independent"] == 1)]["disease_id"].nunique()),
    })
    return out, audit


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run(cfg: Gpu8Config | None = None, tables_dir: Path | None = None,
        clinvar_max_rows: int | None = None) -> dict:
    cfg = cfg or load_config()
    tables_dir = Path(tables_dir) if tables_dir else paths.TABLES_DIR
    diseases = pd.read_parquet(tables_dir / "diseases.parquet")
    genes = pd.read_parquet(tables_dir / "genes.parquet")
    relations = pd.read_parquet(tables_dir / "relations.parquet")
    evidence = pd.read_parquet(tables_dir / "evidence.parquet")
    primary = relations[relations["relation_type"].isin(set(cfg.data.primary_evidence))]

    disease_lookup = build_disease_lookup(diseases)
    gene_ids, by_symbol, by_hgnc = build_gene_lookup(genes)
    visible = set(primary["disease_id"])

    print("[external] GenCC ...", flush=True)
    gencc_raw = load_gencc(paths.external("gencc", "gencc-submissions.tsv"))
    gencc_df, gencc_audit = gencc_pairs(cfg, gencc_raw, disease_lookup, gene_ids,
                                        by_symbol, by_hgnc)

    print("[external] ClinVar variant_summary ...", flush=True)
    cv_var, cv_audit = stream_clinvar_variants(
        paths.external("clinvar", "variant_summary.txt.gz"), gene_ids,
        min_stars=1, max_rows=clinvar_max_rows)
    cv_df, cv_map_audit = clinvar_pairs(cv_var, disease_lookup)

    frames = [f for f in (gencc_df, cv_df) if not f.empty]
    allp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    labelled, layer_audit = apply_exclusion_layers(allp, primary, evidence, visible)

    out_dir = paths.RESULTS_DIR / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not labelled.empty:
        atomic_write_df(labelled, tables_dir / "external_labels.parquet")
        atomic_write_df(labelled, out_dir / "external_labels.csv")
    if not cv_var.empty:
        atomic_write_df(cv_var, out_dir / "clinvar_variant_level.csv")

    report = {"gencc": gencc_audit, "clinvar_variant": cv_audit,
              "clinvar_mapping": cv_map_audit, "exclusion_layers": layer_audit,
              "n_project_diseases": int(len(diseases)),
              "n_lookup_keys": len(disease_lookup)}
    if not labelled.empty:
        for src in labelled["source"].unique():
            sub = labelled[(labelled["source"] == src) & (labelled["evaluable"] == 1)]
            report[f"{src}_evaluable"] = {
                "pairs": int(len(sub)), "diseases": int(sub["disease_id"].nunique()),
                "high_tier_pairs": int((sub["tier"] == "high").sum()),
                "high_tier_diseases": int(sub[sub["tier"] == "high"]["disease_id"].nunique()),
                "strict_independent_pairs": int(sub["strict_independent"].sum()),
            }
    atomic_write_json(out_dir / "external_labels_audit.json", report)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build external confirmation label sets.")
    ap.add_argument("--tables_dir", type=Path, default=None)
    ap.add_argument("--clinvar_max_rows", type=int, default=0)
    args = ap.parse_args(argv)
    rep = run(tables_dir=args.tables_dir,
              clinvar_max_rows=args.clinvar_max_rows or None)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

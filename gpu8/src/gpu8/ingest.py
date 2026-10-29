"""Build the nine canonical tables from raw upstream resources.

Plan section 5.4 fixes the schema:

    genes        gene_id, ncbi_gene_id, hgnc_id, symbol, locus_type, map_path,
                 ambiguity_status, version
    diseases     disease_id, source_ids, equivalence_group, family_group,
                 hpo_coverage, mapping_type
    relations    relation_id, disease_id, gene_id, relation_type  (one row per
                 unique relation)
    evidence     relation_id, source, source_record, pmid, level, original_date,
                 first_curated_date, version
    phenotypes   disease_id, hpo_id, negated, frequency, source, date
    hierarchy    child_disease, parent_disease, relation_type, resource_version
    ppi          gene_a, gene_b, score, network_type, source
    external_labels  source, disease_id, gene_id, relation_qualification,
                 include/exclude reason, overlap level
    manifest.json    sha256, bytes, download date, resource version, code version

Streaming is mandatory for CTD (~125 M rows): the parser reads line by line and
never materialises the raw file.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from . import paths
from .config import Gpu8Config, load_config
from .hashing import atomic_write_df, atomic_write_json, sha256_file, sha256_obj

CTD_CHUNK_ROWS = 50_000

DEFAULT_CTD_GENE_DISEASE_HEADER = [
    "GeneSymbol", "GeneID", "DiseaseName", "DiseaseID", "DirectEvidence",
    "InferenceChemicalName", "InferenceScore", "OmimIDs", "PubMedIDs",
]
DEFAULT_CTD_DISEASE_HEADER = [
    "DiseaseName", "DiseaseID", "AltDiseaseIDs", "Definition", "ParentIDs",
    "TreeNumbers", "ParentTreeNumbers", "Synonyms", "SlimMappings",
]


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------
def open_text(path: Path) -> io.TextIOBase:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _split_multi(value: str) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in re.split(r"[|;,]", str(value)) if p.strip()]


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def iter_delimited(path: Path, default_header: list[str]) -> Iterator[dict]:
    """Stream a CTD-style '#'-commented TSV, one dict per data row."""
    header: list[str] | None = None
    with open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith("#"):
                candidate = [c.strip() for c in line.lstrip("#").strip().split("\t")]
                low = [c.lower() for c in candidate]
                if "diseaseid" in low and len(candidate) >= 4:
                    header = candidate
                continue
            fields = line.split("\t")
            if header is None:
                header = list(default_header)
            if len(fields) < len(header):
                fields = fields + [""] * (len(header) - len(fields))
            elif len(fields) > len(header):
                fields = fields[: len(header)]
            yield dict(zip(header, fields))


def parse_obo(path: Path, prefix: str) -> dict[str, dict]:
    """Minimal OBO term parser; keeps id, name, def, synonyms, is_a, xrefs."""
    terms: dict[str, dict] = {}
    cur: dict | None = None
    in_term = False
    if not Path(path).exists():
        return terms
    with open_text(Path(path)) as fh:
        for raw in fh:
            line = raw.strip()
            if line == "[Term]":
                if cur and cur.get("term_id", "").startswith(prefix) and not cur.get("obsolete"):
                    terms[cur["term_id"]] = cur
                cur = {"term_id": "", "name": "", "definition": "", "synonyms": [],
                       "is_a": [], "xref": [], "obsolete": False}
                in_term = True
                continue
            if line.startswith("[") and line.endswith("]"):
                if cur and cur.get("term_id", "").startswith(prefix) and not cur.get("obsolete"):
                    terms[cur["term_id"]] = cur
                cur, in_term = None, False
                continue
            if not in_term or cur is None or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "id":
                cur["term_id"] = normalize_obo_id(value)
            elif key == "name":
                cur["name"] = value
            elif key == "def":
                m = re.match(r'"(.*?)"', value)
                cur["definition"] = m.group(1) if m else value
            elif key == "synonym":
                m = re.match(r'"(.*?)"', value)
                if m:
                    cur["synonyms"].append(m.group(1))
            elif key == "is_a":
                cur["is_a"].append(normalize_obo_id(value.split("!")[0].strip()))
            elif key == "xref":
                cur["xref"].append(value)
            elif key == "is_obsolete" and value.lower() == "true":
                cur["obsolete"] = True
    if cur and cur.get("term_id", "").startswith(prefix) and not cur.get("obsolete"):
        terms[cur["term_id"]] = cur
    return terms


def normalize_obo_id(value: str) -> str:
    value = (value or "").strip()
    if value.startswith("http"):
        value = value.rsplit("/", 1)[-1]
    return value.replace("_", ":") if re.match(r"^[A-Za-z]+_\d+$", value) else value


# --------------------------------------------------------------------------
# gene identifier layer
# --------------------------------------------------------------------------
def load_ncbi_gene_info(path: Path) -> tuple[dict, dict, dict]:
    """Return (gene_id -> record, upper symbol -> gene_id, ambiguous aliases)."""
    by_id: dict[str, dict] = {}
    by_symbol: dict[str, str] = {}
    alias_hits: dict[str, set] = defaultdict(set)
    if not Path(path).exists():
        return by_id, by_symbol, {}
    with open_text(Path(path)) as fh:
        header: list[str] | None = None
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                header = [c.strip() for c in line.lstrip("#").strip().split("\t")]
                continue
            fields = line.split("\t")
            if header is None or len(fields) < 3:
                continue
            row = dict(zip(header, fields))
            gid = (row.get("GeneID") or "").strip()
            sym = (row.get("Symbol") or "").strip()
            if not gid or gid == "-":
                continue
            by_id[gid] = {
                "ncbi_gene_id": gid,
                "symbol": sym,
                "type_of_gene": (row.get("type_of_gene") or "").strip(),
                "synonyms": (row.get("Synonyms") or "").strip(),
                "dbxrefs": (row.get("dbXrefs") or "").strip(),
                "chromosome": (row.get("chromosome") or "").strip(),
            }
            if sym and sym != "-":
                by_symbol.setdefault(sym.upper(), gid)
                alias_hits[sym.upper()].add(gid)
            for alias in _split_multi(row.get("Synonyms", "")):
                if alias and alias != "-":
                    by_symbol.setdefault(alias.upper(), gid)
                    alias_hits[alias.upper()].add(gid)
    ambiguous = {k: sorted(v) for k, v in alias_hits.items() if len(v) > 1}
    return by_id, by_symbol, ambiguous


def load_hgnc(path: Path) -> tuple[dict, dict]:
    """HGNC complete set -> (entrez_id -> record, symbol_upper -> record)."""
    by_entrez: dict[str, dict] = {}
    by_symbol: dict[str, dict] = {}
    if not Path(path).exists():
        return by_entrez, by_symbol
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, quoting=csv.QUOTE_NONE,
                     on_bad_lines="skip")
    df = df.fillna("")
    keep = ["hgnc_id", "symbol", "name", "locus_group", "locus_type", "status",
            "entrez_id", "ensembl_gene_id", "alias_symbol", "prev_symbol", "omim_id"]
    for col in keep:
        if col not in df.columns:
            df[col] = ""
    for row in df[keep].itertuples(index=False):
        rec = {k: getattr(row, k) for k in keep}
        eid = str(rec["entrez_id"]).strip()
        if eid and eid != "":
            eid = eid.split(".")[0]
            by_entrez.setdefault(eid, rec)
        sym = str(rec["symbol"]).strip().upper()
        if sym:
            by_symbol.setdefault(sym, rec)
    return by_entrez, by_symbol


# --------------------------------------------------------------------------
# CTD disease vocabulary
# --------------------------------------------------------------------------
def build_disease_vocabulary(ctd_diseases: Path) -> tuple[dict, dict, list]:
    nodes: dict[str, dict] = {}
    alias_to_disease: dict[str, str] = {}
    hierarchy: list[dict] = []
    for row in iter_delimited(ctd_diseases, DEFAULT_CTD_DISEASE_HEADER):
        did = (row.get("DiseaseID") or "").strip()
        if not did:
            continue
        alts = _split_multi(row.get("AltDiseaseIDs", ""))
        omim = sorted({a for a in alts + [did] if a.upper().startswith("OMIM:")})
        nodes[did] = {
            "disease_id": did,
            "disease_name": (row.get("DiseaseName") or "").strip(),
            "definition": (row.get("Definition") or "").strip(),
            "synonyms": (row.get("Synonyms") or "").strip(),
            "alt_ids": "|".join(alts),
            "omim_ids": "|".join(omim),
            "tree_numbers": (row.get("TreeNumbers") or "").strip(),
            "parent_ids": (row.get("ParentIDs") or "").strip(),
            "slim_mappings": (row.get("SlimMappings") or "").strip(),
            "source": "CTD",
        }
        alias_to_disease.setdefault(did, did)
        for a in alts:
            alias_to_disease.setdefault(a, did)
        for parent in _split_multi(row.get("ParentIDs", "")):
            hierarchy.append({"child_disease": did, "parent_disease": parent,
                              "relation_type": "ctd_parent", "resource_version": "CTD"})
    return nodes, alias_to_disease, hierarchy


# --------------------------------------------------------------------------
# CTD relations and evidence (streaming)
# --------------------------------------------------------------------------
def stream_ctd_relations(
    path: Path,
    disease_nodes: dict,
    alias_to_disease: dict,
    gene_by_id: dict,
    primary_evidence: Iterable[str],
    secondary_evidence: Iterable[str],
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (relations_df, evidence_df, stats).

    One relations row per unique (disease, gene, relation_type).  One evidence
    row per (relation, source_record, pmid) so that duplicate curation of the
    same original report can be identified later.
    """
    primary = {e.lower() for e in primary_evidence}
    secondary = {e.lower() for e in secondary_evidence}
    wanted = primary | secondary

    rel_index: dict[tuple[str, str, str], int] = {}
    rel_rows: list[dict] = []
    ev_rows: list[dict] = []
    stats = {"rows_seen": 0, "skipped_no_direct": 0, "skipped_nonhuman": 0,
             "skipped_unmapped_disease": 0, "skipped_other_evidence": 0,
             "kept_rows": 0}

    for i, row in enumerate(iter_delimited(path, DEFAULT_CTD_GENE_DISEASE_HEADER)):
        stats["rows_seen"] += 1
        if max_rows and i >= max_rows:
            break
        direct = (row.get("DirectEvidence") or "").strip()
        if not direct:
            stats["skipped_no_direct"] += 1
            continue
        gene_id = (row.get("GeneID") or "").strip()
        if not gene_id or gene_id not in gene_by_id:
            stats["skipped_nonhuman"] += 1
            continue
        raw_did = (row.get("DiseaseID") or "").strip()
        did = alias_to_disease.get(raw_did, raw_did if raw_did in disease_nodes else "")
        if not did:
            stats["skipped_unmapped_disease"] += 1
            continue
        parts = [p.strip().lower() for p in direct.split("|") if p.strip()]
        applicable = [p for p in parts if p in wanted]
        if not applicable:
            stats["skipped_other_evidence"] += 1
            continue
        stats["kept_rows"] += 1
        pmids = _split_multi(row.get("PubMedIDs", ""))
        omim_ids = "|".join(_split_multi(row.get("OmimIDs", "")))
        for rtype in applicable:
            key = (did, gene_id, rtype)
            if key not in rel_index:
                rel_index[key] = len(rel_rows)
                rel_rows.append({
                    "relation_id": f"R{len(rel_rows):07d}",
                    "disease_id": did,
                    "gene_id": gene_id,
                    "relation_type": rtype,
                    "n_pmids": 0,
                    "omim_ids": omim_ids,
                })
            rid = rel_rows[rel_index[key]]["relation_id"]
            if pmids:
                for pmid in pmids:
                    ev_rows.append({
                        "relation_id": rid, "source": "CTD",
                        "source_record": f"CTD:{did}:{gene_id}:{rtype}",
                        "pmid": pmid, "level": "direct_evidence",
                        "original_date": "", "first_curated_date": "",
                        "version": "CTD_2026-06",
                    })
            else:
                ev_rows.append({
                    "relation_id": rid, "source": "CTD",
                    "source_record": f"CTD:{did}:{gene_id}:{rtype}",
                    "pmid": "", "level": "direct_evidence_no_pmid",
                    "original_date": "", "first_curated_date": "",
                    "version": "CTD_2026-06",
                })

    relations = pd.DataFrame(rel_rows)
    evidence = pd.DataFrame(ev_rows)
    if not evidence.empty:
        # deduplicate identical (relation, source_record, pmid) curations
        before = len(evidence)
        evidence = evidence.drop_duplicates(subset=["relation_id", "source", "source_record", "pmid"])
        stats["evidence_rows_before_dedup"] = int(before)
        stats["evidence_rows_after_dedup"] = int(len(evidence))
        counts = evidence.groupby("relation_id").size()
        relations["n_pmids"] = relations["relation_id"].map(counts).fillna(0).astype(int)
    return relations, evidence, stats


# --------------------------------------------------------------------------
# phenotypes
# --------------------------------------------------------------------------
def build_phenotypes(hpoa_path: Path, alias_to_disease: dict, hpo_terms: dict) -> pd.DataFrame:
    rows: list[dict] = []
    if not Path(hpoa_path).exists():
        return pd.DataFrame(columns=["disease_id", "hpo_id", "negated", "frequency",
                                     "source", "date", "aspect"])
    with open_text(Path(hpoa_path)) as fh:
        header: list[str] | None = None
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split("\t")
            if header is None:
                header = [c.strip() for c in fields]
                continue
            row = dict(zip(header, fields))
            raw_id = (row.get("database_id") or row.get("DatabaseID") or "").strip()
            hpo_id = (row.get("hpo_id") or row.get("HPO_ID") or "").strip()
            if not raw_id or not hpo_id:
                continue
            did = alias_to_disease.get(raw_id)
            if not did:
                continue
            qualifier = (row.get("qualifier") or "").strip().upper()
            rows.append({
                "disease_id": did,
                "hpo_id": hpo_id,
                "negated": 1 if qualifier == "NOT" else 0,
                "frequency": (row.get("frequency") or "").strip(),
                "source": (row.get("reference") or "").strip(),
                "date": (row.get("biocuration") or "").strip(),
                "aspect": (row.get("aspect") or "").strip(),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["disease_id", "hpo_id", "negated"])
        df = df[df["hpo_id"].isin(set(hpo_terms)) | (len(hpo_terms) == 0)]
    return df


# --------------------------------------------------------------------------
# PPI
# --------------------------------------------------------------------------
def load_string_protein_map(info_path: Path, aliases_path: Path,
                            gene_by_id: dict, symbol_to_gene: dict) -> dict:
    protein_to_gene: dict[str, str] = {}
    if Path(info_path).exists():
        with open_text(Path(info_path)) as fh:
            header = None
            for line in fh:
                fields = line.rstrip("\n").split("\t")
                if header is None:
                    header = [c.strip().lstrip("#").strip() for c in fields]
                    continue
                row = dict(zip(header, fields))
                pid = (row.get("#string_protein_id") or row.get("string_protein_id")
                       or fields[0]).strip()
                name = (row.get("preferred_name") or "").strip().upper()
                gid = symbol_to_gene.get(name)
                if pid and gid:
                    protein_to_gene.setdefault(pid, gid)
    if Path(aliases_path).exists():
        with open_text(Path(aliases_path)) as fh:
            header = None
            for line in fh:
                fields = line.rstrip("\n").split("\t")
                if header is None:
                    header = [c.strip().lstrip("#").strip() for c in fields]
                    continue
                if len(fields) < 3:
                    continue
                pid, alias, source = fields[0].strip(), fields[1].strip(), fields[2].strip()
                if pid in protein_to_gene:
                    continue
                if alias.isdigit() and alias in gene_by_id:
                    protein_to_gene[pid] = alias
                elif "entrez" in source.lower() and alias.isdigit():
                    if alias in gene_by_id:
                        protein_to_gene[pid] = alias
                else:
                    gid = symbol_to_gene.get(alias.upper())
                    if gid:
                        protein_to_gene.setdefault(pid, gid)
    return protein_to_gene


def build_ppi(links_path: Path, protein_to_gene: dict, threshold: int,
              network_type: str) -> pd.DataFrame:
    best: dict[tuple[str, str], int] = {}
    if not Path(links_path).exists():
        return pd.DataFrame(columns=["gene_a", "gene_b", "score", "network_type", "source"])
    with open_text(Path(links_path)) as fh:
        header_seen = False
        for line in fh:
            parts = line.split()
            if not header_seen:
                header_seen = True
                if parts and not parts[-1].isdigit():
                    continue
            if len(parts) < 3:
                continue
            p1, p2, score_s = parts[0], parts[1], parts[-1]
            try:
                score = int(score_s)
            except ValueError:
                continue
            if score < threshold:
                continue
            g1 = protein_to_gene.get(p1)
            g2 = protein_to_gene.get(p2)
            if not g1 or not g2 or g1 == g2:
                continue
            key = (g1, g2) if g1 < g2 else (g2, g1)
            if score > best.get(key, -1):
                best[key] = score
    rows = [{"gene_a": a, "gene_b": b, "score": s, "network_type": network_type,
             "source": "STRING_v12.0"} for (a, b), s in best.items()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# main build
# --------------------------------------------------------------------------
def build_tables(cfg: Gpu8Config, out_dir: Path | None = None,
                 max_ctd_rows: int | None = None,
                 secondary_ppi: bool = True) -> dict:
    out_dir = Path(out_dir) if out_dir else paths.TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = datetime.now(timezone.utc)
    report: dict = {"created_utc": t_start.isoformat(), "out_dir": str(out_dir)}

    ctd_diseases = paths.ASSOCIATION_DIR / "CTD_diseases.tsv.gz"
    ctd_gene_disease = paths.ASSOCIATION_DIR / "CTD_genes_diseases.tsv.gz"
    gene_info = paths.FUNCTION_DIR / "Homo_sapiens.gene_info.gz"
    hp_obo = paths.PHENOTYPE_DIR / "hp.obo"
    hpoa = paths.PHENOTYPE_DIR / "phenotype.hpoa"
    doid = paths.ONTOLOGY_DIR / "doid.obo"
    hgnc_path = paths.external("hgnc", "hgnc_complete_set.txt")

    print("[1/8] NCBI gene info ...", flush=True)
    gene_by_id, symbol_to_gene, ambiguous_alias = load_ncbi_gene_info(gene_info)
    print(f"      human genes: {len(gene_by_id)}  ambiguous aliases: {len(ambiguous_alias)}", flush=True)

    print("[2/8] HGNC ...", flush=True)
    hgnc_by_entrez, hgnc_by_symbol = load_hgnc(hgnc_path)
    print(f"      HGNC records with entrez id: {len(hgnc_by_entrez)}", flush=True)

    print("[3/8] CTD disease vocabulary ...", flush=True)
    disease_nodes, alias_to_disease, hierarchy_rows = build_disease_vocabulary(ctd_diseases)
    print(f"      CTD diseases: {len(disease_nodes)}", flush=True)

    print("[4/8] CTD relations + evidence (streaming) ...", flush=True)
    relations, evidence, ctd_stats = stream_ctd_relations(
        ctd_gene_disease, disease_nodes, alias_to_disease, gene_by_id,
        cfg.data.primary_evidence, cfg.data.secondary_evidence, max_rows=max_ctd_rows)
    print(f"      unique relations: {len(relations)}  evidence rows: {len(evidence)}", flush=True)

    print("[5/8] ontologies + phenotypes ...", flush=True)
    hpo_terms = parse_obo(hp_obo, "HP:")
    do_terms = parse_obo(doid, "DOID:")
    phenotypes = build_phenotypes(hpoa, alias_to_disease, hpo_terms)
    print(f"      HPO terms: {len(hpo_terms)}  DO terms: {len(do_terms)}  "
          f"phenotype rows: {len(phenotypes)}", flush=True)

    print("[6/8] STRING PPI ...", flush=True)
    info_path = paths.PPI_DIR / "9606.protein.info.v12.0.txt.gz"
    alias_path = paths.PPI_DIR / "9606.protein.aliases.v12.0.txt.gz"
    protein_to_gene = load_string_protein_map(info_path, alias_path, gene_by_id, symbol_to_gene)
    ppi_primary = build_ppi(paths.PPI_DIR / "9606.protein.physical.links.v12.0.txt.gz",
                            protein_to_gene, cfg.data.ppi_score_threshold, "physical")
    ppi_frames = [ppi_primary]
    second_path = paths.external("string", "9606.protein.links.v12.0.txt.gz")
    if secondary_ppi and second_path.exists():
        ppi_second = build_ppi(second_path, protein_to_gene,
                               cfg.data.ppi_secondary_threshold, "functional")
        ppi_frames.append(ppi_second)
        print(f"      functional PPI edges: {len(ppi_second)}", flush=True)
    ppi = pd.concat([f for f in ppi_frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in ppi_frames) else ppi_primary
    print(f"      mapped STRING proteins: {len(protein_to_gene)}  "
          f"physical edges: {len(ppi_primary)}", flush=True)

    print("[7/8] assembling canonical tables ...", flush=True)
    # ---- genes table -----------------------------------------------------
    used_genes = sorted(set(relations["gene_id"])) if not relations.empty else []
    legacy_genes: set[str] = set()
    legacy_nodes = paths.LEGACY_PROCESSED / "nodes_gene.csv"
    if cfg.data.keep_legacy_gene_space and legacy_nodes.exists():
        legacy_genes = set(pd.read_csv(legacy_nodes, dtype=str)["gene_id"].astype(str))
    background: set[str] = set(used_genes) | legacy_genes
    for gid, rec in hgnc_by_entrez.items():
        if rec.get("locus_group", "") == "protein-coding gene" and rec.get("status") == "Approved":
            background.add(gid)
    background &= set(gene_by_id)

    gene_rows = []
    for gid in sorted(background, key=lambda x: int(x) if x.isdigit() else 0):
        ncbi = gene_by_id.get(gid, {})
        hg = hgnc_by_entrez.get(gid, {})
        sym = hg.get("symbol") or ncbi.get("symbol", "")
        ambiguous = any(gid in v for k, v in ambiguous_alias.items()
                        if k == (ncbi.get("symbol") or "").upper())
        gene_rows.append({
            "gene_id": gid,
            "ncbi_gene_id": gid,
            "hgnc_id": hg.get("hgnc_id", ""),
            "symbol": sym,
            "ensembl_gene_id": hg.get("ensembl_gene_id", ""),
            "locus_type": hg.get("locus_type", "") or ncbi.get("type_of_gene", ""),
            "locus_group": hg.get("locus_group", ""),
            "map_path": "hgnc_entrez" if hg else ("ncbi_only" if ncbi else "unmapped"),
            "ambiguity_status": "ambiguous_alias" if ambiguous else "ok",
            "in_legacy_space": int(gid in legacy_genes),
            "protein_coding": int(hg.get("locus_group", "") == "protein-coding gene"),
            "version": "NCBI_2026-06/HGNC_2026-09",
        })
    genes = pd.DataFrame(gene_rows)

    # ---- diseases table --------------------------------------------------
    hpo_cov = phenotypes[phenotypes["negated"] == 0].groupby("disease_id").size() \
        if not phenotypes.empty else pd.Series(dtype=int)
    rel_disease_counts = relations.groupby("disease_id").size() if not relations.empty else pd.Series(dtype=int)
    dis_rows = []
    for did, rec in disease_nodes.items():
        dis_rows.append({
            "disease_id": did,
            "disease_name": rec["disease_name"],
            "source_ids": rec["alt_ids"],
            "omim_ids": rec["omim_ids"],
            "definition": rec["definition"],
            "synonyms": rec["synonyms"],
            "tree_numbers": rec["tree_numbers"],
            "parent_ids": rec["parent_ids"],
            "equivalence_group": "",     # filled by mapping.py (Mondo)
            "family_group": "",          # filled by mapping.py
            "mapping_type": "",          # filled by mapping.py
            "hpo_coverage": int(hpo_cov.get(did, 0)),
            "n_relations": int(rel_disease_counts.get(did, 0)),
            "source": "CTD",
        })
    diseases = pd.DataFrame(dis_rows)

    # ---- hierarchy -------------------------------------------------------
    hierarchy = pd.DataFrame(hierarchy_rows)
    if not hierarchy.empty:
        valid = set(diseases["disease_id"])
        hierarchy = hierarchy[hierarchy["child_disease"].isin(valid)
                              & hierarchy["parent_disease"].isin(valid)]
        hierarchy = hierarchy.drop_duplicates()

    # ---- filter tables to the resolved entity space ----------------------
    gene_space = set(genes["gene_id"])
    disease_space = set(diseases["disease_id"])
    if not relations.empty:
        relations = relations[relations["gene_id"].isin(gene_space)
                              & relations["disease_id"].isin(disease_space)].reset_index(drop=True)
        evidence = evidence[evidence["relation_id"].isin(set(relations["relation_id"]))].reset_index(drop=True)
    if not phenotypes.empty:
        phenotypes = phenotypes[phenotypes["disease_id"].isin(disease_space)].reset_index(drop=True)
    if not ppi.empty:
        ppi = ppi[ppi["gene_a"].isin(gene_space) & ppi["gene_b"].isin(gene_space)].reset_index(drop=True)

    # ---- HPO / DO term tables (auxiliary) --------------------------------
    hpo_df = pd.DataFrame([
        {"hpo_id": t["term_id"], "name": t["name"], "definition": t["definition"],
         "synonyms": "|".join(t["synonyms"]), "is_a": "|".join(t["is_a"])}
        for t in hpo_terms.values()])
    do_df = pd.DataFrame([
        {"term_id": t["term_id"], "name": t["name"], "definition": t["definition"],
         "synonyms": "|".join(t["synonyms"]), "is_a": "|".join(t["is_a"]),
         "xref": "|".join(t["xref"])}
        for t in do_terms.values()])

    print("[8/8] writing parquet tables ...", flush=True)
    written = {}
    for name, df in [("genes", genes), ("diseases", diseases), ("relations", relations),
                     ("evidence", evidence), ("phenotypes", phenotypes),
                     ("hierarchy", hierarchy), ("ppi", ppi),
                     ("hpo_terms", hpo_df), ("do_terms", do_df)]:
        p = atomic_write_df(df, paths.table(name) if out_dir == paths.TABLES_DIR
                            else out_dir / f"{name}.parquet")
        written[name] = {"rows": int(len(df)), "path": str(p),
                         "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        print(f"      {name:12s} {len(df):>9,} rows", flush=True)

    # ambiguity isolation table (plan 5.3)
    amb_rows = [{"alias": k, "gene_ids": "|".join(v), "n": len(v)}
                for k, v in sorted(ambiguous_alias.items())]
    amb_path = out_dir / "gene_alias_ambiguity.csv"
    atomic_write_df(pd.DataFrame(amb_rows), amb_path)

    report.update({
        "ctd_stats": ctd_stats,
        "counts": {k: v["rows"] for k, v in written.items()},
        "tables": written,
        "relation_type_counts": relations["relation_type"].value_counts().to_dict()
        if not relations.empty else {},
        "ppi_network_counts": ppi["network_type"].value_counts().to_dict() if not ppi.empty else {},
        "ambiguous_alias_count": len(ambiguous_alias),
        "config_hash": cfg.config_hash,
        "input_files": {
            "CTD_diseases": str(ctd_diseases), "CTD_genes_diseases": str(ctd_gene_disease),
            "gene_info": str(gene_info), "hp_obo": str(hp_obo), "hpoa": str(hpoa),
            "doid": str(doid), "hgnc": str(hgnc_path),
        },
        "elapsed_seconds": (datetime.now(timezone.utc) - t_start).total_seconds(),
    })
    atomic_write_json(out_dir / "build_report.json", report)
    return report


def build_resource_manifest(out_path: Path | None = None) -> dict:
    """SHA-256 / bytes / date / licence entry point for every input file."""
    out_path = Path(out_path) if out_path else paths.MANIFEST_DIR / "resource_manifest.json"
    licence = {
        "CTD": "https://ctdbase.org/about/legal.jsp",
        "HPO": "https://hpo.jax.org/app/license",
        "DO": "https://disease-ontology.org (CC0)",
        "STRING": "https://string-db.org/cgi/access (CC BY 4.0)",
        "NCBI Gene": "https://www.ncbi.nlm.nih.gov/home/about/policies/",
        "HGNC": "https://www.genenames.org/about/ (no restrictions)",
        "Mondo": "https://mondo.monarchinitiative.org (CC BY 4.0)",
        "GenCC": "https://thegencc.org/terms (CC0 1.0)",
        "ClinVar": "https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/",
        "GEO": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
    }
    groups = {
        "association": paths.ASSOCIATION_DIR, "phenotype": paths.PHENOTYPE_DIR,
        "function": paths.FUNCTION_DIR, "ontology": paths.ONTOLOGY_DIR,
        "ppi": paths.PPI_DIR, "external": paths.EXTERNAL_DIR,
    }
    entries = []
    for group, d in groups.items():
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix == ".part":
                continue
            st = p.stat()
            entries.append({
                "group": group, "name": p.name, "path": paths.rel(p),
                "bytes": st.st_size, "sha256": sha256_file(p),
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            })
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(entries), "files": entries, "license_entry_points": licence,
        "manifest_hash": sha256_obj([(e["path"], e["sha256"]) for e in entries]),
    }
    atomic_write_json(out_path, manifest)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the nine canonical GPU8 tables.")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--max_ctd_rows", type=int, default=0)
    ap.add_argument("--no_secondary_ppi", action="store_true")
    ap.add_argument("--manifest_only", action="store_true")
    args = ap.parse_args(argv)

    paths.ensure_dirs()
    cfg = load_config()
    cfg.save()
    if args.manifest_only:
        m = build_resource_manifest()
        print(json.dumps({"n_files": m["n_files"], "hash": m["manifest_hash"]}, indent=2))
        return 0
    report = build_tables(cfg, out_dir=args.out_dir,
                          max_ctd_rows=args.max_ctd_rows or None,
                          secondary_ppi=not args.no_secondary_ppi)
    build_resource_manifest()
    print(json.dumps(report["counts"], indent=2))
    print(json.dumps(report["ctd_stats"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

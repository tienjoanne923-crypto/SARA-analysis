from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd


DEFAULT_CTD_GENE_DISEASE_HEADER = [
    "GeneSymbol",
    "GeneID",
    "DiseaseName",
    "DiseaseID",
    "DirectEvidence",
    "InferenceChemicalName",
    "InferenceScore",
    "OmimIDs",
    "PubMedIDs",
]

DEFAULT_CTD_DISEASE_HEADER = [
    "DiseaseName",
    "DiseaseID",
    "Definition",
    "AltDiseaseIDs",
    "ParentIDs",
    "TreeNumbers",
    "ParentTreeNumbers",
    "Synonyms",
    "SlimMappings",
]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def split_multi(value: str) -> list[str]:
    value = str(value or "").strip()
    if not value:
        return []
    return [x.strip() for x in re.split(r"[|;,]", value) if x.strip()]


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def normalize_obo_id(term_id: str) -> str:
    term_id = str(term_id or "").strip()
    if term_id.startswith("http://purl.obolibrary.org/obo/"):
        term_id = term_id.rsplit("/", 1)[-1].replace("_", ":")
    return term_id


def iter_ctd_records(path: Path, default_header: list[str]) -> Iterator[dict[str, str]]:
    header: list[str] | None = None
    with open_text(path) as handle:
        for raw in handle:
            line = raw.rstrip("\n\r")
            if not line:
                continue
            if line.startswith("#"):
                maybe = line.lstrip("#").strip()
                if "\t" in maybe:
                    cols = maybe.split("\t")
                    joined = "".join(c.lower() for c in cols)
                    if "geneid" in joined or "diseaseid" in joined:
                        header = cols
                continue
            parts = line.split("\t")
            if header is None:
                header = default_header if len(parts) == len(default_header) else parts
                if header is parts:
                    continue
            if len(parts) < len(header):
                parts = parts + [""] * (len(header) - len(parts))
            elif len(parts) > len(header):
                parts = parts[: len(header)]
            yield dict(zip(header, parts))


def parse_obo_terms(path: Path, prefix: str | None = None) -> dict[str, dict]:
    if not path.exists():
        return {}
    terms: dict[str, dict] = {}
    current: dict | None = None

    def flush(term: dict | None) -> None:
        if not term or not term.get("id"):
            return
        term_id = normalize_obo_id(term["id"])
        if prefix and not term_id.startswith(prefix):
            return
        if term.get("is_obsolete") == "true":
            return
        terms[term_id] = {
            "term_id": term_id,
            "term_name": term.get("name", ""),
            "definition": term.get("def", ""),
            "synonyms": " | ".join(term.get("synonym", [])),
            "is_a": " | ".join(term.get("is_a", [])),
        }

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n\r")
            if line == "[Term]":
                flush(current)
                current = {}
                continue
            if current is None:
                continue
            if not line:
                continue
            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("def: "):
                current["def"] = line[5:].strip()
            elif line.startswith("synonym: "):
                current.setdefault("synonym", []).append(line[9:].strip())
            elif line.startswith("is_a: "):
                parent = normalize_obo_id(line[6:].split("!")[0].strip())
                current.setdefault("is_a", []).append(parent)
            elif line.startswith("is_obsolete: "):
                current["is_obsolete"] = line[13:].strip()
        flush(current)
    return terms


def load_human_gene_info(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    gene_id_to_symbol: dict[str, str] = {}
    symbol_to_gene_id: dict[str, str] = {}
    if not path.exists():
        return gene_id_to_symbol, symbol_to_gene_id
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene_id = str(row.get("GeneID", "")).strip()
            symbol = str(row.get("Symbol", "")).strip()
            if not gene_id or not symbol or symbol == "-":
                continue
            gene_id_to_symbol[gene_id] = symbol
            symbol_to_gene_id.setdefault(symbol.upper(), gene_id)
            for syn in split_multi(row.get("Synonyms", "")):
                if syn and syn != "-":
                    symbol_to_gene_id.setdefault(syn.upper(), gene_id)
    return gene_id_to_symbol, symbol_to_gene_id


def build_ctd_disease_nodes(ctd_diseases_path: Path) -> tuple[dict[str, dict], dict[str, str], list[dict]]:
    disease_nodes: dict[str, dict] = {}
    alias_to_disease: dict[str, str] = {}
    ontology_edges: list[dict] = []
    if not ctd_diseases_path.exists():
        return disease_nodes, alias_to_disease, ontology_edges

    for row in iter_ctd_records(ctd_diseases_path, DEFAULT_CTD_DISEASE_HEADER):
        disease_id = str(row.get("DiseaseID", "")).strip()
        if not disease_id:
            continue
        omim_ids = [x for x in split_multi(row.get("AltDiseaseIDs", "")) if x.startswith("OMIM:")]
        if disease_id.startswith("OMIM:"):
            omim_ids.append(disease_id)
        disease_nodes[disease_id] = {
            "disease_id": disease_id,
            "disease_name": str(row.get("DiseaseName", "")).strip(),
            "omim_ids": "|".join(sorted(set(omim_ids))),
            "definition": str(row.get("Definition", "")).strip(),
            "synonyms": str(row.get("Synonyms", "")).strip(),
            "source": "CTD",
        }
        alias_to_disease[disease_id] = disease_id
        for alt in split_multi(row.get("AltDiseaseIDs", "")):
            alias_to_disease.setdefault(alt, disease_id)
        for omim in omim_ids:
            alias_to_disease.setdefault(omim, disease_id)
        for parent in split_multi(row.get("ParentIDs", "")):
            if parent:
                ontology_edges.append(
                    {
                        "source_id": disease_id,
                        "target_id": parent,
                        "relation": "ctd_parent",
                        "confidence": 0.85,
                    }
                )
    return disease_nodes, alias_to_disease, ontology_edges


def should_keep_ctd_pair(row: dict[str, str], include_inferred: bool, min_inference_score: float) -> bool:
    direct = str(row.get("DirectEvidence", "")).strip()
    if direct:
        return True
    if not include_inferred:
        return False
    return parse_float(row.get("InferenceScore", ""), default=0.0) >= min_inference_score


def build_ctd_gene_disease_edges(
    ctd_gene_disease_path: Path,
    disease_nodes: dict[str, dict],
    alias_to_disease: dict[str, str],
    human_gene_id_to_symbol: dict[str, str],
    include_inferred: bool,
    min_inference_score: float,
    max_rows: int | None,
    max_pairs: int | None,
) -> tuple[dict[str, dict], dict[tuple[str, str], dict], dict]:
    gene_nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}
    stats = {
        "ctd_rows_seen": 0,
        "ctd_rows_kept": 0,
        "ctd_rows_skipped_nonhuman": 0,
        "ctd_rows_skipped_weak": 0,
    }
    human_filter_enabled = bool(human_gene_id_to_symbol)

    for row in iter_ctd_records(ctd_gene_disease_path, DEFAULT_CTD_GENE_DISEASE_HEADER):
        stats["ctd_rows_seen"] += 1
        if max_rows is not None and stats["ctd_rows_seen"] > max_rows:
            break

        gene_id = str(row.get("GeneID", "")).strip()
        disease_id_raw = str(row.get("DiseaseID", "")).strip()
        if not gene_id or not disease_id_raw:
            continue
        if human_filter_enabled and gene_id not in human_gene_id_to_symbol:
            stats["ctd_rows_skipped_nonhuman"] += 1
            continue
        if not should_keep_ctd_pair(row, include_inferred=include_inferred, min_inference_score=min_inference_score):
            stats["ctd_rows_skipped_weak"] += 1
            continue

        disease_id = alias_to_disease.get(disease_id_raw, disease_id_raw)
        gene_symbol = human_gene_id_to_symbol.get(gene_id, str(row.get("GeneSymbol", "")).strip())
        gene_nodes.setdefault(
            gene_id,
            {
                "gene_id": gene_id,
                "gene_symbol": gene_symbol,
                "source": "CTD/NCBI",
            },
        )

        if disease_id not in disease_nodes:
            disease_nodes[disease_id] = {
                "disease_id": disease_id,
                "disease_name": str(row.get("DiseaseName", "")).strip(),
                "omim_ids": str(row.get("OmimIDs", "")).strip(),
                "definition": "",
                "synonyms": "",
                "source": "CTD_gene_disease",
            }
            alias_to_disease.setdefault(disease_id, disease_id)
        elif row.get("OmimIDs"):
            old = set(split_multi(disease_nodes[disease_id].get("omim_ids", "")))
            old.update(split_multi(row.get("OmimIDs", "")))
            disease_nodes[disease_id]["omim_ids"] = "|".join(sorted(old))

        direct = str(row.get("DirectEvidence", "")).strip()
        inference_score = parse_float(row.get("InferenceScore", ""), default=0.0)
        key = (disease_id, gene_id)
        existing = edges.get(key)
        confidence = 1.0 if direct else min(0.95, max(0.55, inference_score / 100.0))
        evidence = direct if direct else "inferred"
        if existing is None or confidence > existing["confidence"]:
            edges[key] = {
                "source_id": disease_id,
                "target_id": gene_id,
                "evidence": evidence,
                "inference_score": inference_score,
                "confidence": confidence,
                "pubmed_ids": str(row.get("PubMedIDs", "")).strip(),
            }
        stats["ctd_rows_kept"] += 1
        if max_pairs is not None and len(edges) >= max_pairs:
            break

    return gene_nodes, edges, stats


def add_hpo_phenotype_edges(
    phenotype_hpoa_path: Path,
    alias_to_disease: dict[str, str],
    phenotype_terms: dict[str, dict],
) -> list[dict]:
    if not phenotype_hpoa_path.exists():
        return []
    edges: dict[tuple[str, str], dict] = {}
    with phenotype_hpoa_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
        for row in reader:
            disease_key = str(row.get("database_id", row.get("DatabaseID", ""))).strip()
            hpo_id = normalize_obo_id(row.get("hpo_id", row.get("HPO_ID", "")))
            if not disease_key or not hpo_id:
                continue
            disease_id = alias_to_disease.get(disease_key)
            if not disease_id:
                continue
            if hpo_id not in phenotype_terms:
                phenotype_terms[hpo_id] = {
                    "term_id": hpo_id,
                    "term_name": "",
                    "definition": "",
                    "synonyms": "",
                    "is_a": "",
                }
            key = (disease_id, hpo_id)
            edges.setdefault(
                key,
                {
                    "source_id": disease_id,
                    "target_id": hpo_id,
                    "evidence": str(row.get("evidence", row.get("Evidence", ""))).strip(),
                    "reference": str(row.get("reference", row.get("Reference", ""))).strip(),
                    "confidence": 0.8,
                },
            )
    return list(edges.values())


def load_string_alias_map(
    aliases_path: Path,
    info_path: Path,
    gene_nodes: dict[str, dict],
    symbol_to_gene_id: dict[str, str],
) -> dict[str, str]:
    gene_ids = set(gene_nodes)
    protein_to_gene: dict[str, str] = {}

    if info_path.exists():
        with open_text(info_path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                protein_id = str(row.get("#string_protein_id", row.get("string_protein_id", ""))).strip()
                preferred = str(row.get("preferred_name", "")).strip().upper()
                gid = symbol_to_gene_id.get(preferred)
                if protein_id and gid in gene_ids:
                    protein_to_gene.setdefault(protein_id, gid)

    if aliases_path.exists():
        with open_text(aliases_path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                protein_id = str(row.get("#string_protein_id", row.get("string_protein_id", ""))).strip()
                alias = str(row.get("alias", "")).strip()
                source = str(row.get("source", "")).strip().lower()
                if not protein_id or not alias:
                    continue
                gid = None
                if alias in gene_ids and ("entrez" in source or alias.isdigit()):
                    gid = alias
                else:
                    gid = symbol_to_gene_id.get(alias.upper())
                if gid in gene_ids:
                    protein_to_gene.setdefault(protein_id, gid)
    return protein_to_gene


def build_ppi_edges(links_path: Path, protein_to_gene: dict[str, str], threshold: int) -> list[dict]:
    if not links_path.exists():
        return []
    edges: dict[tuple[str, str], dict] = {}
    with open_text(links_path) as handle:
        header = None
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if header is None:
                header = parts
                continue
            if len(parts) < 3:
                continue
            row = dict(zip(header, parts))
            p1 = row.get("protein1") or row.get("#protein1") or parts[0]
            p2 = row.get("protein2") or parts[1]
            score = int(parse_float(row.get("combined_score", parts[-1]), default=0))
            if score < threshold:
                continue
            g1 = protein_to_gene.get(p1)
            g2 = protein_to_gene.get(p2)
            if not g1 or not g2 or g1 == g2:
                continue
            key = tuple(sorted((g1, g2)))
            old = edges.get(key)
            if old is None or score > old["combined_score"]:
                edges[key] = {
                    "source_id": key[0],
                    "target_id": key[1],
                    "combined_score": score,
                    "confidence": score / 1000.0,
                }
    return list(edges.values())


def dataframe_from_records(records: Iterable[dict], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    if df.empty:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns].drop_duplicates().reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare processed CSV files from downloaded biomedical raw data.")
    parser.add_argument("--root", type=Path, default=project_root(), help="Project root containing association/ phenotype/ ppi/ etc.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory. Default: <root>/processed_data.")
    parser.add_argument("--include_inferred", action="store_true", help="Include high-scoring inferred CTD associations.")
    parser.add_argument("--min_inference_score", type=float, default=50.0)
    parser.add_argument("--ppi_score_threshold", type=int, default=700)
    parser.add_argument("--max_ctd_rows", type=int, default=0, help="Debug limit; 0 means all rows.")
    parser.add_argument("--max_pairs", type=int, default=0, help="Debug limit on disease-gene pairs; 0 means no limit.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or (root / "processed_data")).resolve()

    association_dir = root / "association"
    phenotype_dir = root / "phenotype"
    function_dir = root / "function"
    ontology_dir = root / "ontology"
    ppi_dir = root / "ppi"

    ctd_diseases_path = association_dir / "CTD_diseases.tsv.gz"
    ctd_gene_disease_path = association_dir / "CTD_genes_diseases.tsv.gz"
    hp_obo_path = phenotype_dir / "hp.obo"
    phenotype_hpoa_path = phenotype_dir / "phenotype.hpoa"
    human_gene_info_path = function_dir / "Homo_sapiens.gene_info.gz"
    doid_path = ontology_dir / "doid.obo"

    required = [ctd_diseases_path, ctd_gene_disease_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required raw file(s): " + ", ".join(missing))

    print("Loading NCBI human gene metadata...")
    human_gene_id_to_symbol, symbol_to_gene_id = load_human_gene_info(human_gene_info_path)
    print(f"  human genes: {len(human_gene_id_to_symbol)}")

    print("Parsing CTD disease vocabulary...")
    disease_nodes, alias_to_disease, disease_ontology_edges = build_ctd_disease_nodes(ctd_diseases_path)
    print(f"  CTD disease nodes: {len(disease_nodes)}")

    print("Parsing CTD gene-disease associations...")
    gene_nodes, disease_gene_edges, ctd_stats = build_ctd_gene_disease_edges(
        ctd_gene_disease_path,
        disease_nodes=disease_nodes,
        alias_to_disease=alias_to_disease,
        human_gene_id_to_symbol=human_gene_id_to_symbol,
        include_inferred=bool(args.include_inferred),
        min_inference_score=float(args.min_inference_score),
        max_rows=int(args.max_ctd_rows) if args.max_ctd_rows else None,
        max_pairs=int(args.max_pairs) if args.max_pairs else None,
    )
    print(f"  disease-gene pairs: {len(disease_gene_edges)}")

    print("Parsing ontology and phenotype files...")
    phenotype_terms = parse_obo_terms(hp_obo_path, prefix="HP:")
    do_terms = parse_obo_terms(doid_path, prefix="DOID:")
    disease_phenotype_edges = add_hpo_phenotype_edges(phenotype_hpoa_path, alias_to_disease, phenotype_terms)

    print("Parsing STRING PPI files...")
    string_links = sorted(ppi_dir.glob("9606.protein.*links*.v*.txt.gz"))
    string_links = [p for p in string_links if "physical.links" in p.name] or string_links
    links_path = string_links[0] if string_links else ppi_dir / "9606.protein.physical.links.v12.0.txt.gz"
    info_candidates = sorted(ppi_dir.glob("9606.protein.info.v*.txt.gz"))
    alias_candidates = sorted(ppi_dir.glob("9606.protein.aliases.v*.txt.gz"))
    protein_to_gene = load_string_alias_map(
        aliases_path=alias_candidates[0] if alias_candidates else ppi_dir / "9606.protein.aliases.v12.0.txt.gz",
        info_path=info_candidates[0] if info_candidates else ppi_dir / "9606.protein.info.v12.0.txt.gz",
        gene_nodes=gene_nodes,
        symbol_to_gene_id=symbol_to_gene_id,
    )
    gene_gene_ppi_edges = build_ppi_edges(links_path, protein_to_gene, threshold=int(args.ppi_score_threshold))
    print(f"  mapped STRING proteins: {len(protein_to_gene)}")
    print(f"  gene-gene PPI edges: {len(gene_gene_ppi_edges)}")

    valid_diseases = set(disease_nodes)
    valid_genes = set(gene_nodes)
    valid_phenotypes = set(phenotype_terms)
    disease_ontology_edges = [
        e for e in disease_ontology_edges if e["source_id"] in valid_diseases and e["target_id"] in valid_diseases
    ]
    disease_phenotype_edges = [
        e for e in disease_phenotype_edges if e["source_id"] in valid_diseases and e["target_id"] in valid_phenotypes
    ]
    gene_gene_ppi_edges = [
        e for e in gene_gene_ppi_edges if e["source_id"] in valid_genes and e["target_id"] in valid_genes
    ]

    nodes_disease_df = dataframe_from_records(
        disease_nodes.values(),
        ["disease_id", "disease_name", "omim_ids", "definition", "synonyms", "source"],
    )
    nodes_gene_df = dataframe_from_records(gene_nodes.values(), ["gene_id", "gene_symbol", "source"])
    nodes_phenotype_df = dataframe_from_records(
        (
            {
                "phenotype_id": term["term_id"],
                "phenotype_name": term.get("term_name", ""),
                "definition": term.get("definition", ""),
                "synonyms": term.get("synonyms", ""),
                "is_a": term.get("is_a", ""),
            }
            for term in phenotype_terms.values()
        ),
        ["phenotype_id", "phenotype_name", "definition", "synonyms", "is_a"],
    )
    disease_gene_df = dataframe_from_records(
        disease_gene_edges.values(),
        ["source_id", "target_id", "evidence", "inference_score", "confidence", "pubmed_ids"],
    )
    disease_phenotype_df = dataframe_from_records(
        disease_phenotype_edges,
        ["source_id", "target_id", "evidence", "reference", "confidence"],
    )
    disease_ontology_df = dataframe_from_records(
        disease_ontology_edges,
        ["source_id", "target_id", "relation", "confidence"],
    )
    gene_gene_ppi_df = dataframe_from_records(
        gene_gene_ppi_edges,
        ["source_id", "target_id", "combined_score", "confidence"],
    )
    do_terms_df = dataframe_from_records(
        do_terms.values(),
        ["term_id", "term_name", "definition", "synonyms", "is_a"],
    )

    write_csv(nodes_disease_df, out_dir / "nodes_disease.csv")
    write_csv(nodes_gene_df, out_dir / "nodes_gene.csv")
    write_csv(nodes_phenotype_df, out_dir / "nodes_phenotype.csv")
    write_csv(disease_gene_df, out_dir / "disease_gene_edges.csv")
    write_csv(disease_phenotype_df, out_dir / "disease_phenotype_edges.csv")
    write_csv(disease_ontology_df, out_dir / "disease_disease_ontology_edges.csv")
    write_csv(gene_gene_ppi_df, out_dir / "gene_gene_ppi_edges.csv")
    write_csv(do_terms_df, out_dir / "do_terms.csv")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "out_dir": str(out_dir),
        "options": {
            "include_inferred": bool(args.include_inferred),
            "min_inference_score": float(args.min_inference_score),
            "ppi_score_threshold": int(args.ppi_score_threshold),
            "max_ctd_rows": int(args.max_ctd_rows),
            "max_pairs": int(args.max_pairs),
        },
        "ctd_stats": ctd_stats,
        "counts": {
            "nodes_disease": int(len(nodes_disease_df)),
            "nodes_gene": int(len(nodes_gene_df)),
            "nodes_phenotype": int(len(nodes_phenotype_df)),
            "disease_gene_edges": int(len(disease_gene_df)),
            "disease_phenotype_edges": int(len(disease_phenotype_df)),
            "disease_disease_ontology_edges": int(len(disease_ontology_df)),
            "gene_gene_ppi_edges": int(len(gene_gene_ppi_df)),
            "do_terms": int(len(do_terms_df)),
        },
        "input_files": {
            "CTD_diseases": str(ctd_diseases_path),
            "CTD_genes_diseases": str(ctd_gene_disease_path),
            "hp_obo": str(hp_obo_path),
            "phenotype_hpoa": str(phenotype_hpoa_path),
            "Homo_sapiens_gene_info": str(human_gene_info_path),
            "doid_obo": str(doid_path),
            "string_links": str(links_path),
        },
    }
    (out_dir / "processed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(f"Processed data written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

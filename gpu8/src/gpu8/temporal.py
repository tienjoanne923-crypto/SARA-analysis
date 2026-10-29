"""Temporal validation against the frozen 2023Q4 snapshot.

Plan section 14.4 is strict about what may be claimed.  On this machine the
historical CTD association dump is no longer present; what survives is the
processed 2023Q4 disease-gene edge table (with its PubMed identifiers), the
period-matched HPO release, the period-matched Disease Ontology and the CTD
disease vocabulary of that era.  The protein interaction network has no
historical copy.

Therefore this experiment is reported as an **edge-level temporal freeze with
period-matched phenotype and ontology resources and a current interaction
network** — not a complete point-in-time freeze.  Using the current network is
a documented relaxation, applied identically to every method, and the claim is
downgraded accordingly rather than the experiment being presented as stricter
than it is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ingest, paths
from .config import Gpu8Config, load_config
from .hashing import atomic_write_df, atomic_write_json

SNAPSHOT = paths.SNAPSHOT_ROOT


def build_historical_tables(cfg: Gpu8Config, out_dir: Path | None = None) -> dict:
    """Recreate the canonical tables for the historical cut-off."""
    out_dir = Path(out_dir) if out_dir else (paths.GPU8_ROOT / "tables_historical")
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_edges = SNAPSHOT / "processed_data" / "disease_gene_edges.csv"
    if not hist_edges.exists():
        return {"status": "unavailable", "reason": f"missing {hist_edges}"}

    edges = pd.read_csv(hist_edges, dtype=str).fillna("")
    cur_genes = pd.read_parquet(paths.TABLES_DIR / "genes.parquet")
    cur_ppi = pd.read_parquet(paths.TABLES_DIR / "ppi.parquet")
    gene_space = set(cur_genes["gene_id"].astype(str))

    # relations and evidence, from the historical edge table
    prim = set(cfg.data.primary_evidence)
    rel_rows, ev_rows = [], []
    for i, r in enumerate(edges.itertuples(index=False)):
        types = [t.strip().lower() for t in str(r.evidence).split("|") if t.strip()]
        types = [t for t in types if t in prim or t in set(cfg.data.secondary_evidence)]
        if not types or str(r.target_id) not in gene_space:
            continue
        for t in types:
            rid = f"H{len(rel_rows):07d}"
            rel_rows.append({"relation_id": rid, "disease_id": r.source_id,
                             "gene_id": str(r.target_id), "relation_type": t,
                             "n_pmids": 0, "omim_ids": ""})
            pmids = [p for p in str(getattr(r, "pubmed_ids", "")).replace(";", "|").split("|")
                     if p.strip()]
            if pmids:
                for p in pmids:
                    ev_rows.append({"relation_id": rid, "source": "CTD_2023Q4",
                                    "source_record": f"CTD:{r.source_id}:{r.target_id}:{t}",
                                    "pmid": p.strip(), "level": "direct_evidence",
                                    "original_date": "", "first_curated_date": "",
                                    "version": "CTD_2023Q4"})
            else:
                ev_rows.append({"relation_id": rid, "source": "CTD_2023Q4",
                                "source_record": f"CTD:{r.source_id}:{r.target_id}:{t}",
                                "pmid": "", "level": "direct_evidence_no_pmid",
                                "original_date": "", "first_curated_date": "",
                                "version": "CTD_2023Q4"})
    relations = pd.DataFrame(rel_rows)
    evidence = pd.DataFrame(ev_rows)
    if not evidence.empty:
        counts = evidence[evidence["pmid"] != ""].groupby("relation_id")["pmid"].nunique()
        relations["n_pmids"] = relations["relation_id"].map(counts).fillna(0).astype(int)

    # period-matched vocabulary, phenotypes and hierarchy
    hist_ctd = SNAPSHOT / "association" / "CTD_diseases.tsv.gz"
    nodes, alias, hier_rows = ingest.build_disease_vocabulary(hist_ctd)
    hpo_terms = ingest.parse_obo(SNAPSHOT / "phenotype" / "hp.obo", "HP:")
    phen = ingest.build_phenotypes(SNAPSHOT / "phenotype" / "phenotype.hpoa", alias, hpo_terms)

    cur_diseases = pd.read_parquet(paths.TABLES_DIR / "diseases.parquet")
    keep_cols = ["disease_id", "disease_name", "source_ids", "omim_ids", "definition",
                 "synonyms", "tree_numbers", "parent_ids", "equivalence_group",
                 "family_group", "mapping_type", "hpo_coverage", "n_relations", "source"]
    diseases = cur_diseases[[c for c in keep_cols if c in cur_diseases.columns]].copy()
    hist_ids = set(relations["disease_id"]) | set(phen["disease_id"]) if not phen.empty \
        else set(relations["disease_id"])
    diseases = diseases[diseases["disease_id"].isin(hist_ids | set(nodes))]

    hierarchy = pd.DataFrame(hier_rows)
    if not hierarchy.empty:
        ok = set(diseases["disease_id"])
        hierarchy = hierarchy[hierarchy["child_disease"].isin(ok)
                              & hierarchy["parent_disease"].isin(ok)].drop_duplicates()
    if not phen.empty:
        phen = phen[phen["disease_id"].isin(set(diseases["disease_id"]))]

    written = {}
    for name, df in [("genes", cur_genes), ("diseases", diseases),
                     ("relations", relations), ("evidence", evidence),
                     ("phenotypes", phen), ("hierarchy", hierarchy),
                     ("ppi", cur_ppi)]:
        p = atomic_write_df(df, out_dir / f"{name}.parquet")
        written[name] = int(len(df))

    report = {
        "status": "ok", "out_dir": str(out_dir), "counts": written,
        "resources": {
            "disease_gene_edges": str(hist_edges),
            "ctd_disease_vocabulary": str(hist_ctd),
            "hpo": str(SNAPSHOT / "phenotype" / "hp.obo"),
            "hpoa": str(SNAPSHOT / "phenotype" / "phenotype.hpoa"),
            "ppi": "current project network (no historical copy on disk)",
            "gene_space": "current project gene background",
        },
        "claim_boundary": (
            "edge-level temporal freeze with period-matched phenotype and ontology "
            "resources; the interaction network and the gene background are current, "
            "so this is NOT a complete point-in-time freeze"),
    }
    atomic_write_json(out_dir / "historical_build_report.json", report)
    return report


def build_temporal_split(cfg: Gpu8Config, hist_dir: Path | None = None) -> dict:
    """Future positives = current primary relations absent from the historical set."""
    hist_dir = Path(hist_dir) if hist_dir else (paths.GPU8_ROOT / "tables_historical")
    hist_rel = pd.read_parquet(hist_dir / "relations.parquet")
    cur_rel = pd.read_parquet(paths.TABLES_DIR / "relations.parquet")
    prim = set(cfg.data.primary_evidence)
    h = hist_rel[hist_rel["relation_type"].isin(prim)]
    c = cur_rel[cur_rel["relation_type"].isin(prim)]
    hp = set(zip(h["disease_id"], h["gene_id"]))
    cp = set(zip(c["disease_id"], c["gene_id"]))
    future = sorted(cp - hp)

    hist_diseases = set(h["disease_id"])
    hist_genes = set(h["gene_id"]) | set(
        pd.read_parquet(hist_dir / "genes.parquet")["gene_id"].astype(str))
    filtered = [(d, g) for d, g in future if d in hist_diseases and g in hist_genes]

    audit = {
        "historical_positives": int(len(hp)),
        "current_positives": int(len(cp)),
        "future_positives_before_node_filter": int(len(future)),
        "future_positives_after_node_filter": int(len(filtered)),
        "node_mapping_loss_edges": int(len(future) - len(filtered)),
        "future_diseases_after_filter": int(len({d for d, _ in filtered})),
        "train_test_overlap_edges": int(len(set(filtered) & hp)),
        "note": ("only diseases that already existed in the historical graph are "
                 "evaluated; diseases that did not exist then are reported separately "
                 "and never counted as recovered knowledge"),
    }
    df = pd.DataFrame(filtered, columns=["disease_id", "gene_id"])
    atomic_write_df(df, paths.RESULTS_DIR / "temporal" / "temporal_future_positives.csv")
    atomic_write_json(paths.RESULTS_DIR / "temporal" / "temporal_split_audit.json", audit)
    return audit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the historical snapshot tables and split.")
    ap.add_argument("--step", default="all", choices=["tables", "split", "all"])
    args = ap.parse_args(argv)
    cfg = load_config()
    out = {}
    if args.step in ("tables", "all"):
        out["tables"] = build_historical_tables(cfg)
    if args.step in ("split", "all"):
        out["split"] = build_temporal_split(cfg)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V6 section 12.1 — full-input history audit for the temporal task.

The V5 temporal experiment is task X2 of section 3.1 ("new relations of diseases
that already existed at the cut-off"), with an edge-level freeze.  This module
lists every input a V6 temporal run would read, with the version actually
available, its earliest-known date, where it came from, its hash, and whether
it is no later than the cut-off (2023-12-05).  The archived CTD relation dump is
verified by reading the header of the Wayback capture itself.
"""
from __future__ import annotations

import json
import zlib

import pandas as pd

from gpu8 import paths as p5
from gpu8.hashing import sha256_file

from . import paths6
from .common import dump_json, now_utc

CUTOFF = "2023-12-05"


def wayback_header() -> dict:
    p = paths6.ext("wayback", "CTD_genes_diseases_20231205_head.bin")
    cdx = json.loads(paths6.ext("wayback", "cdx_CTD_genes_diseases_2023.json").read_text())
    out = {"cdx_rows": cdx[1:]}
    if p.exists():
        txt = zlib.decompressobj(31).decompress(p.read_bytes()).decode("utf-8", "replace")
        created = [l for l in txt.splitlines() if l.startswith("# Report created:")]
        out["report_created_line"] = created[0] if created else ""
        out["first_bytes_sha256"] = sha256_file(p)
    return out


def audit() -> dict:
    snap = p5.SNAPSHOT_ROOT
    ext = paths6.ext
    rows = [
        ("CTD disease-gene relations (direct evidence)", "processed 2023Q4 edges from the Wayback capture of CTD_genes_diseases.tsv.gz",
         "2023-11-29 (report created); captured 2023-12-05", snap / "processed_data" / "disease_gene_edges.csv", True,
         "edge-level; raw dump re-obtainable from the Wayback capture (2.66 GB), header verified"),
        ("CTD relation PMIDs", "same edge table (pubmed_ids column)", "2023-11-29", snap / "processed_data" / "disease_gene_edges.csv", True, ""),
        ("CTD MEDIC vocabulary (names, synonyms, definitions, tree numbers, parents)", "Wayback capture of CTD_diseases.tsv.gz",
         "2022-01-24", snap / "association" / "CTD_diseases.tsv.gz", True, "older than the cut-off, not contemporaneous"),
        ("HPO ontology", "hp.obo v2023-10-09", "2023-10-09", snap / "phenotype" / "hp.obo", True, ""),
        ("HPO disease annotations", "phenotype.hpoa v2023-10-09", "2023-10-09", snap / "phenotype" / "phenotype.hpoa", True, ""),
        ("Gene-HPO with source disease", "genes_to_phenotype.txt v2023-10-09 (HPO GitHub release)", "2023-10-09",
         ext("hpo", "genes_to_phenotype_v2023-10-09.txt"), True, "source-disease exclusion applies as in the main task"),
        ("Disease Ontology", "doid.obo 2023-11-30", "2023-11-30", snap / "ontology" / "doid.obo", True, ""),
        ("Mondo (strict mapping repair, definitions)", "current release only", "2026", p5.external("mondo", "mondo.obo"), False,
         "historical 2023 releases returned 404; the temporal run does not use Mondo"),
        ("Protein interactions", "STRING v12.0 physical, score >= 700", "2023-07-26 release", p5.PPI_DIR / "9606.protein.physical.links.v12.0.txt.gz", True,
         "released before the cut-off"),
        ("GO annotations", "goa_human.gaf 2023-10-09 + go-basic 2023-10-09", "2023-10-09", ext("go", "goa_human_2023-10-09.gaf.gz"), True, ""),
        ("Reactome pathways", "NCBI2Reactome release 86", "2023-09", ext("reactome", "NCBI2Reactome_v86.txt"), True, ""),
        ("Tissue expression", "GTEx v8 median TPM", "2019", ext("gtex", "GTEx_v8_gene_median_tpm.gct.gz"), True, ""),
        ("Gene background and symbols", "HGNC complete set 2023-10-01", "2023-10-01", ext("hgnc", "hgnc_complete_set_2023-10-01.txt"), True,
         "used for the background and for the text mask"),
        ("Disease families / equivalence groups", "derived with current Mondo", "2026", p5.TABLES_DIR / "diseases.parquet", False,
         "used only to group the bootstrap, never as a model input"),
    ]
    recs = []
    for name, version, date, path, hist, note in rows:
        recs.append({"input": name, "version_used": version, "earliest_known_date": date,
                     "file": p5.rel(path), "exists": path.exists(),
                     "sha256": sha256_file(path) if path.exists() and path.stat().st_size < 3e9 else "",
                     "bytes": path.stat().st_size if path.exists() else 0,
                     "no_later_than_cutoff": bool(hist), "note": note})
    df = pd.DataFrame(recs)
    out = paths6.EXTERNAL_RES
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "temporal_input_history_audit.csv", index=False, encoding="utf-8-sig")
    model_inputs = df[~df["input"].str.startswith("Disease families")]
    full = bool(model_inputs.loc[~model_inputs["input"].str.startswith("Mondo"),
                                 "no_later_than_cutoff"].all())
    summ = {"generated_utc": now_utc(), "cutoff": CUTOFF,
            "task_type": ("X2: new relations of diseases that already existed at the cut-off "
                          "(not new-disease cold start)"),
            "wayback": wayback_header(),
            "all_model_inputs_no_later_than_cutoff_if_mondo_dropped": full,
            "freeze_level": ("edge-level freeze of CTD relations with every other model input taken "
                             "from a release no later than the cut-off (Mondo not used); the raw "
                             "2.66 GB relation dump itself was not re-downloaded, the processed "
                             "2023Q4 edge table is used"),
            "download_vs_release": "download timestamps are in external/v6/download_log_v6.csv; the dates above are release / report dates"}
    dump_json(out / "temporal_input_history_audit.json", summ)
    return summ


if __name__ == "__main__":
    paths6.ensure()
    print(json.dumps(audit(), indent=2, default=str))

"""V7.1 resource log: frozen encoder snapshots and external files.

* ``models``   downloads the two frozen encoders from the Hugging Face hub at the
               commit that ``main`` resolves to *now*, then pins that commit in
               ``manifests_v71/models_v71.json`` (repo, revision, pooling, every file
               with its size and SHA-256).  The pooling rule is part of the record:
               SapBERT-from-PubMedBERT-fulltext uses the [CLS] token *before* the
               pooler (the checkpoint authors' instruction); ESM-2 35M uses the
               residue-coverage-normalised mean over windows.
* ``log``      merges ``external/v71/download_log_v71.csv`` with version, licence,
               mapping rule and pending checks into ``manifests_v71/resources_v71.json``.

Nothing here is fitted on labels.  A resource that could not be obtained is
logged as such and never replaced by a random matrix.
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from gpu8.hashing import sha256_file

from . import paths71
from .common71 import dump_json, now_utc, write_manifest

MODELS = {
    "sapbert": {"repo": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                "files": ["config.json", "pytorch_model.bin", "vocab.txt", "tokenizer_config.json",
                          "special_tokens_map.json"],
                "pooling": "[CLS] hidden state of the last layer, before the pooler",
                "max_tokens": 128, "dim": 768,
                "licence": "Apache-2.0 (model card)"},
    "esm2_35m": {"repo": "facebook/esm2_t12_35M_UR50D",
                 "files": ["config.json", "model.safetensors", "vocab.txt", "tokenizer_config.json",
                           "special_tokens_map.json"],
                 "pooling": ("mean over residues of the last layer; windows of <= 1,022 residues "
                             "with overlap 128; each residue weighted 1/c_i (c_i = number of windows "
                             "covering it); CLS/EOS/padding excluded"),
                 "window": 1022, "overlap": 128, "dim": 480, "layers": 12,
                 "licence": "MIT (facebookresearch/esm)"},
}

RESOURCE_META = {
    "rna_tissue_consensus.tsv.zip": {
        "what": "HPA consensus normal-tissue RNA (nTPM), 51 tissues, HPA + GTEx consensus",
        "version": "HPA release current at download (page shows HPA 25.1, Ensembl 109)",
        "licence": "CC BY-SA 4.0 (Human Protein Atlas)",
        "mapping": "Ensembl gene id -> project gene via genes.ensembl_gene_id (explicit table)",
        "pending": "header checked at parse time; consensus values are not 51 independent donors"},
    "subcellular_location.tsv.zip": {
        "what": "HPA subcellular location labels (localisation-only control for the image channel)",
        "version": "HPA current", "licence": "CC BY-SA 4.0",
        "mapping": "Ensembl gene id", "pending": "used only in the conditional image extension"},
    "subcell_image_umap_features.tsv.zip": {
        "what": "HPA subcellular image features (1,024-d DenseNet features + UMAP coordinates)",
        "version": "HPA current", "licence": "CC BY-SA 4.0",
        "mapping": "image -> antibody x cell line -> gene (hierarchical equal-weight means)",
        "pending": "'tissue' column there is a cell line, never mixed with the tissue axis"},
    "UP000005640_9606.fasta.gz": {
        "what": "UniProt human reference proteome, one canonical sequence per gene",
        "version": "see uniprot/RELEASE.metalink", "licence": "CC BY 4.0 (UniProt)",
        "mapping": "UniProt accession -> Entrez via HGNC uniprot_ids; fallback after MANE",
        "pending": "ambiguous multi-gene accessions go to the exception table"},
    "MANE.GRCh38.v1.5.summary.txt.gz": {
        "what": "MANE Select / Plus Clinical summary (GeneID -> RefSeq protein)",
        "version": "MANE v1.5", "licence": "public domain (NCBI) / EMBL-EBI terms",
        "mapping": "NCBI GeneID (= project gene id) -> MANE Select RefSeq protein",
        "pending": "none"},
    "MANE.GRCh38.v1.5.refseq_protein.faa.gz": {
        "what": "MANE RefSeq protein sequences", "version": "MANE v1.5",
        "licence": "public domain (NCBI)", "mapping": "RefSeq protein accession", "pending": "none"},
    "genotype-phenotype-assertions-ALL.csv.gz": {
        "what": "IMPC genotype-phenotype calls (MP terms after gene perturbation)",
        "version": "IMPC latest release at download", "licence": "CC BY 4.0 (IMPC)",
        "mapping": "MGI marker -> human gene through MGI HOM_MouseHumanSequence (1:1 only)",
        "pending": "manual spot check of the HP-MP bridge not performed (exported sample)"},
    "mp.obo": {"what": "Mammalian Phenotype ontology", "version": "data-version in header",
               "licence": "CC BY 4.0", "mapping": "is_a closure", "pending": "none"},
    "HOM_MouseHumanSequence.rpt": {
        "what": "MGI human-mouse homology classes", "version": "MGI report at download",
        "licence": "MGI terms of use (CC BY 4.0)",
        "mapping": "DB Class Key groups; only one human and one mouse gene per class kept",
        "pending": "replaces HCOP, whose bulk file URL returned 404"},
    "mp_hp_mgi_all.sssom.tsv": {
        "what": "MGI curated MP-HP mappings (mapping-commons mh_mapping_initiative, SSSOM)",
        "version": "master branch at download", "licence": "CC0 / CC BY (mapping commons)",
        "mapping": "exact / close / broad / narrow match kept with their predicate",
        "pending": "programmatic checks only; a random sample is exported for manual review"},
}


def download_models() -> dict:
    paths71.add_libs()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    out = {}
    for key, m in MODELS.items():
        info = api.model_info(m["repo"])
        rev = info.sha
        local = paths71.MODELS / key
        local.mkdir(parents=True, exist_ok=True)
        files = {}
        t0 = time.time()
        for fn in m["files"]:
            try:
                p = hf_hub_download(m["repo"], fn, revision=rev, local_dir=str(local))
            except Exception as exc:  # optional tokenizer files may be absent
                files[fn] = {"status": f"not available: {type(exc).__name__}"}
                continue
            from pathlib import Path
            p = Path(p)
            files[fn] = {"bytes": p.stat().st_size, "sha256": sha256_file(p),
                         "path": paths71.rel(p)}
        out[key] = {**{k: v for k, v in m.items() if k != "files"}, "revision": rev,
                    "files": files, "downloaded_utc": now_utc(),
                    "seconds": round(time.time() - t0, 1)}
        print(f"  {key}: {m['repo']}@{rev[:12]} ({len(files)} files)", flush=True)
    write_manifest("models_v71.json", {"models": out,
                                       "rule": "encoders are frozen; revision pinned; "
                                               "no fine-tuning, no ESMFold"})
    return out


def resource_log() -> dict:
    log = pd.read_csv(paths71.ext("download_log_v71.csv"), encoding="utf-8-sig")
    ok = log[log["status"] == "ok"].sort_values("timestamp_utc").drop_duplicates("name", keep="last")
    failed = log[log["status"] != "ok"]
    rows = []
    for r in ok.itertuples(index=False):
        meta = RESOURCE_META.get(r.name, {})
        p = paths71.ext(r.group, r.name)
        rows.append({"file": r.name, "group": r.group, "url": r.url, "fetched_utc": r.timestamp_utc,
                     "bytes": int(r.bytes), "sha256": r.sha256.lower(),
                     "present": p.exists(), **meta})
    not_obtained = []
    for r in failed.itertuples(index=False):
        if r.name not in set(ok["name"]):
            not_obtained.append({"file": r.name, "url": r.url, "status": r.status})
    extra = [{"file": "human_mouse_hcop_fifteen_column.txt.gz",
              "url": "https://ftp.ebi.ac.uk/pub/databases/genenames/hcop/human_mouse_hcop_fifteen_column.txt.gz",
              "status": "HTTP 404 at feasibility check (2026-09-23); MGI homology used instead"},
             {"file": "subcell_image_umap_features.tsv.zip (tsv/ path)",
              "url": "https://www.proteinatlas.org/download/tsv/subcell_image_umap_features.tsv.zip",
              "status": "HTTP 404; the file is served from /download/ without tsv/"}]
    rep = {"generated_utc": now_utc(), "resources": rows, "not_obtained": not_obtained + extra,
           "reused_from_v6": ["go/go-basic_current.obo", "go/goa_human_current.gaf.gz",
                              "gtex/GTEx_v8_gene_median_tpm.gct.gz (E4 old-resource arm)"],
           "provenance_reviewed": False,
           "note": ("provenance_reviewed=false: sources, licences and versions were recorded "
                    "programmatically; no manual source audit is claimed")}
    write_manifest("resources_v71.json", rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V7.1 resources.")
    ap.add_argument("--stage", choices=["models", "log"], required=True)
    args = ap.parse_args(argv)
    paths71.ensure()
    if args.stage == "models":
        print(json.dumps(download_models(), indent=2)[:3000])
    else:
        rep = resource_log()
        print(json.dumps({"n": len(rep["resources"]), "not_obtained": rep["not_obtained"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

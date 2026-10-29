"""V6 section 12.2 — ClinVar literature links scoped to the disease condition.

``var_citations.txt`` cross-references a variant (VariationID / AlleleID) with
publications.  A variant can be linked to several conditions, so its PMIDs are
*candidate* literature for a disease-gene pair, not evidence of it.  The chain
built here:

1. variant -> gene and the full list of its conditions (variant_summary, GRCh38,
   germline pathogenic / likely pathogenic, declared review status);
2. conditions -> project diseases (the same strict lookup the external labels
   use);
3. variant -> PubMed ids (var_citations, citation_source = PubMed);
4. a PMID is attributed to (disease, gene) **only** when every condition of the
   variant maps to that single project disease; a variant with several mapped
   conditions leaves its PMIDs ``unknown`` for each of them — they are never
   multiplied across all its conditions.

The third independence layer of every ClinVar external pair is then re-derived
against the CTD evidence of the same pair and of the same disease; a pair with
no condition-verified citation stays ``unknown`` rather than passing the layer.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from gpu8 import paths as p5
from gpu8.external_labels import (CLINVAR_PATHOGENIC, REVIEW_STARS, build_disease_lookup,
                                  parse_phenotype_ids)

from . import paths6
from .common import World6, dump_json, now_utc


def stream_variants(gene_ids: set, lookup: dict, min_stars: int = 1) -> pd.DataFrame:
    rows = []
    path = p5.external("clinvar", "variant_summary.txt.gz")
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = None
        for line in fh:
            if header is None:
                header = [c.strip().lstrip("#") for c in line.rstrip("\n").split("\t")]
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < len(header):
                continue
            rec = dict(zip(header, f))
            if rec.get("Assembly") != "GRCh38":
                continue
            if not str(rec.get("OriginSimple", "")).lower().startswith("germline"):
                continue
            sig = str(rec.get("ClinicalSignificance", "")).lower()
            if not any(p in sig for p in CLINVAR_PATHOGENIC) or any(
                    x in sig for x in ("conflict", "uncertain", "benign")):
                continue
            stars = REVIEW_STARS.get(str(rec.get("ReviewStatus", "")).strip().lower(), 0)
            if stars < min_stars:
                continue
            gid = str(rec.get("GeneID", "")).strip()
            if not gid or gid == "-1" or gid not in gene_ids:
                continue
            conds = [c for c in parse_phenotype_ids(rec.get("PhenotypeIDS", ""))
                     if not (c.startswith("MedGen:CN") or c == "MedGen:C3661900")]
            mapped = sorted({d for c in conds for d in lookup.get(c, [])})
            rows.append({"variation_id": rec.get("VariationID", ""), "gene_id": gid,
                         "stars": stars, "n_condition_ids": len(conds),
                         "mapped_diseases": "|".join(mapped), "n_mapped": len(mapped)})
    return pd.DataFrame(rows).drop_duplicates(["variation_id", "gene_id"])


def read_citations(variation_ids: set) -> dict[str, set]:
    out: dict[str, set] = defaultdict(set)
    path = paths6.ext("clinvar", "var_citations.txt")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = None
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if header is None:
                header = [c.lstrip("#") for c in f]
                continue
            rec = dict(zip(header, f))
            vid = rec.get("VariationID", "")
            if vid in variation_ids and rec.get("citation_source") == "PubMed":
                out[vid].add(rec.get("citation_id", "").strip())
    return out


def run() -> dict:
    w = World6(load_heavy=False)
    lookup = build_disease_lookup(w.diseases)
    var = stream_variants(set(w.genes["gene_id"]), lookup)
    cites = read_citations(set(var["variation_id"]))
    var["n_pmids"] = var["variation_id"].map(lambda v: len(cites.get(v, ())))
    ext = pd.read_parquet(p5.TABLES_DIR / "external_labels.parquet")
    cv = ext[ext["source"] == "ClinVar"].drop_duplicates(["disease_id", "gene_id"])
    # pair-level evidence
    by_gene = defaultdict(list)
    for r in var.itertuples(index=False):
        by_gene[r.gene_id].append(r)
    ctd_pair = {}
    ctd_dis = defaultdict(set)
    for (d, g), rid in w.relation_of.items():
        pm = w.relation_pmids.get(rid, set())
        ctd_pair[(d, g)] = pm
        ctd_dis[d] |= pm
    rows = []
    for r in cv.itertuples(index=False):
        d, g = r.disease_id, str(r.gene_id)
        verified, unknown, n_var = set(), set(), 0
        for v in by_gene.get(g, []):
            md = v.mapped_diseases.split("|") if v.mapped_diseases else []
            if d not in md:
                continue
            n_var += 1
            pm = cites.get(v.variation_id, set())
            if v.n_mapped == 1:
                verified |= pm
            else:
                unknown |= pm
        unknown -= verified
        if verified:
            ov_pair = bool(verified & ctd_pair.get((d, g), set()))
            ov_dis = bool(verified & ctd_dis.get(d, set()))
            layer3 = "overlap_with_ctd" if ov_dis else "independent_verified"
        else:
            ov_pair = ov_dis = False
            layer3 = "unknown_no_condition_verified_citation"
        rows.append({"disease_id": d, "gene_id": g, "n_supporting_variants": n_var,
                     "n_verified_pmids": len(verified), "n_unknown_pmids": len(unknown),
                     "overlap_same_pair": int(ov_pair), "overlap_same_disease": int(ov_dis),
                     "layer3_v6": layer3,
                     "layer3_v5": int(getattr(r, "layer3_pmid_overlap", -1))})
    chain = pd.DataFrame(rows)
    out = paths6.EXTERNAL_RES
    out.mkdir(parents=True, exist_ok=True)
    chain.to_csv(out / "clinvar_citation_chain_pairs.csv", index=False, encoding="utf-8-sig")
    var.to_parquet(out / "clinvar_variants_filtered.parquet", index=False)
    summ = {
        "generated_utc": now_utc(),
        "variants_kept": int(len(var)),
        "variants_with_pubmed_citation": int((var["n_pmids"] > 0).sum()),
        "variants_single_mapped_condition": int((var["n_mapped"] == 1).sum()),
        "variants_multi_mapped_condition": int((var["n_mapped"] > 1).sum()),
        "variants_unmapped_condition": int((var["n_mapped"] == 0).sum()),
        "clinvar_external_pairs": int(len(chain)),
        "layer3_v6_counts": chain["layer3_v6"].value_counts().to_dict(),
        "unknown_rate": float((chain["layer3_v6"].str.startswith("unknown")).mean()),
        "v5_unverifiable_pairs": int((chain["layer3_v5"] == -1).sum()),
        "rule": ("PMIDs attributed to (disease, gene) only when all conditions of the "
                 "variant map to that single disease; never multiplied across conditions"),
        "limitation": ("condition verification is variant-level (uniqueness); SCV-level "
                       "condition-to-citation links would need the XML release and are not used"),
    }
    dump_json(out / "clinvar_citation_chain_summary.json", summ)
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ClinVar citation chain (V6 12.2).")
    ap.parse_args(argv)
    paths6.ensure()
    print(json.dumps(run(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V7.1 section 5.1 — one representative protein per background gene.

Rule fixed before any modelling (never looks at a disease label):

1. MANE Select: project gene id (= NCBI GeneID) -> MANE v1.5 summary row with
   status "MANE Select" -> RefSeq protein -> sequence from the MANE RefSeq
   protein FASTA.
2. otherwise UniProt reference proteome (canonical, one entry per gene): the
   accession must map to exactly one project gene through the HGNC
   ``uniprot_ids`` column; a Swiss-Prot entry is preferred to TrEMBL.  If a gene
   still has several candidate entries, or an accession maps to several genes,
   the record goes to the exception table and the gene stays *masked* (no first
   match is taken).
3. genes without a sequence keep ``g_sequence_mask = False``; they are never
   removed from the background.

Each record keeps the project gene id, the protein id, the species, the
selection reason, the length, the resource version and the SHA-256 of the
sequence.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from gpu8 import paths as p5

from . import paths71
from .common71 import dump_json, now_utc

MANE_SUMMARY = ("mane", "MANE.GRCh38.v1.5.summary.txt.gz")
MANE_FASTA = ("mane", "MANE.GRCh38.v1.5.refseq_protein.faa.gz")
UNIPROT_FASTA = ("uniprot", "UP000005640_9606.fasta.gz")
MANE_VERSION = "MANE v1.5"


def read_fasta(path, key=lambda h: h.split()[0]) -> dict[str, tuple[str, str]]:
    """{key(header): (header, sequence)}."""
    out: dict[str, tuple[str, str]] = {}
    head, buf = None, []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if head is not None:
                    out[key(head)] = (head, "".join(buf))
                head, buf = line[1:], []
            else:
                buf.append(line.strip())
    if head is not None:
        out[key(head)] = (head, "".join(buf))
    return out


def uniprot_release() -> str:
    p = paths71.ext("uniprot", "RELEASE.metalink")
    if not p.exists():
        return "UniProt current"
    txt = p.read_text(encoding="utf-8", errors="replace")
    import re
    m = re.search(r"<version>([^<]+)</version>", txt)
    return f"UniProt {m.group(1)}" if m else "UniProt current"


def build(world=None, write: bool = True) -> dict:
    from gpu8v6.common import World6
    w = world or World6(load_heavy=False)
    genes = w.genes[["gene_id", "symbol", "hgnc_id"]].copy()
    genes["gene_row"] = genes["gene_id"].map(w.idx.g_ix)
    ids = set(genes["gene_id"].astype(str))

    # --- MANE ------------------------------------------------------------------
    ms = pd.read_csv(paths71.ext(*MANE_SUMMARY), sep="\t", dtype=str, compression="gzip")
    ms = ms.rename(columns={"#NCBI_GeneID": "ncbi"})
    ms["gene_id"] = ms["ncbi"].str.replace("GeneID:", "", regex=False)
    sel = ms[ms["MANE_status"] == "MANE Select"]
    mane_fa = read_fasta(paths71.ext(*MANE_FASTA))
    mane_of = defaultdict(list)
    for r in sel.itertuples(index=False):
        if r.gene_id in ids:
            mane_of[r.gene_id].append(r.RefSeq_prot)

    # --- UniProt through HGNC uniprot_ids (accession -> set of genes) ----------
    hg = pd.read_csv(p5.external("hgnc", "hgnc_complete_set.txt"), sep="\t", dtype=str,
                     low_memory=False)
    acc_genes: dict[str, set] = defaultdict(set)
    for r in hg.itertuples(index=False):
        ent = str(getattr(r, "entrez_id", "") or "")
        if ent not in ids:
            continue
        for a in str(getattr(r, "uniprot_ids", "") or "").split("|"):
            a = a.strip()
            if a and a != "nan":
                acc_genes[a].add(ent)
    up = read_fasta(paths71.ext(*UNIPROT_FASTA), key=lambda h: h.split("|")[1])
    up_rel = uniprot_release()
    cand = defaultdict(list)
    exceptions = []
    for acc, (head, seq) in up.items():
        gs = acc_genes.get(acc, set())
        if len(gs) > 1:
            exceptions.append({"kind": "accession_maps_to_several_genes", "accession": acc,
                               "genes": "|".join(sorted(gs))})
            continue
        for g in gs:
            cand[g].append((acc, head.startswith("sp|"), seq, head))

    rows = []
    for r in genes.itertuples(index=False):
        g = str(r.gene_id)
        rec = {"gene_row": int(r.gene_row), "gene_id": g, "symbol": r.symbol,
               "protein_id": "", "source": "", "reason": "no_sequence", "length": 0,
               "version": "", "sha256": "", "sequence": ""}
        mp = sorted(set(mane_of.get(g, [])))
        if len(mp) == 1 and mp[0] in mane_fa:
            seq = mane_fa[mp[0]][1]
            rec.update(protein_id=mp[0], source="MANE", reason="mane_select", sequence=seq,
                       version=MANE_VERSION)
        elif len(mp) > 1:
            exceptions.append({"kind": "several_mane_select", "gene_id": g, "accession": "|".join(mp)})
        if not rec["sequence"]:
            c = cand.get(g, [])
            sp = [x for x in c if x[1]]
            pool = sp if sp else c
            if len(pool) == 1:
                acc, is_sp, seq, head = pool[0]
                rec.update(protein_id=acc, source="UniProt",
                           reason="uniprot_swissprot_canonical" if is_sp else "uniprot_trembl_canonical",
                           sequence=seq, version=up_rel)
            elif len(pool) > 1:
                exceptions.append({"kind": "several_uniprot_entries", "gene_id": g,
                                   "accession": "|".join(x[0] for x in pool)})
        if rec["sequence"]:
            rec["length"] = len(rec["sequence"])
            rec["sha256"] = hashlib.sha256(rec["sequence"].encode()).hexdigest()
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("gene_row").reset_index(drop=True)
    exc = pd.DataFrame(exceptions)
    has = df["length"] > 0
    rep = {"generated_utc": now_utc(), "n_genes_background": int(len(df)),
           "n_with_sequence": int(has.sum()),
           "by_reason": df["reason"].value_counts().to_dict(),
           "n_protein_coding_background": int(w.genes["protein_coding"].astype(int).sum()),
           "n_protein_coding_with_sequence": int((has & (w.genes["protein_coding"].astype(int)
                                                         .to_numpy() == 1)).sum()),
           "length_quantiles": {str(q): float(df.loc[has, "length"].quantile(q))
                                for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)},
           "n_longer_than_1022": int((df["length"] > 1022).sum()),
           "n_exceptions": int(len(exc)),
           "exceptions_by_kind": exc["kind"].value_counts().to_dict() if len(exc) else {},
           "species": "Homo sapiens (NCBI taxon 9606)",
           "versions": {"MANE": MANE_VERSION, "UniProt": up_rel},
           "rule": "MANE Select first; then a unique UniProt canonical entry (Swiss-Prot preferred); "
                   "ambiguity -> exception table and mask; never label dependent"}
    if write:
        paths71.TABLES.mkdir(parents=True, exist_ok=True)
        df.to_parquet(paths71.TABLES / "sequences_v71.parquet", index=False)
        exc.to_csv(paths71.AUDIT / "sequence_exceptions_v71.csv", index=False, encoding="utf-8-sig")
        dump_json(paths71.AUDIT / "sequences_v71_report.json", rep)
    return rep


def main() -> int:
    paths71.ensure()
    rep = build()
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Source data and access routes

These are upstream sources actually represented in the selected analysis inputs. They are not newly published project repositories. Exact file hashes and recorded fetch URLs are in `input_manifest.json`; routes below are access entry points when no exact recorded fetch URL exists. Third-party bytes are not included.

## CTD

disease vocabulary and direct marker/mechanism associations; historical processed labels. 2026-05-28 reports; historical source dates in manifest.

Access: https://ctdbase.org/downloads/

Terms: https://ctdbase.org/about/legal.jsp

Custom CTD terms; noncommercial availability and commercial download licensing stated in 2025 source paper. Original source and derived CTD-backed tables omitted; hashes and preprocessing only.

Citation or primary resource: https://pmc.ncbi.nlm.nih.gov/articles/PMC11701581/

## HPO

ontology, disease annotations and source-disease gene–phenotype annotations. 2026-02-16 current ontology/annotations; historical2023-10-09.

Access: https://github.com/obophenotype/human-phenotype-ontology/releases

Terms: https://human-phenotype-ontology.github.io/license.html

Attribution/date/integrity terms on available official licence page; annotation-specific clearance not completed. Raw annotations omitted; source-linked acquisition under applicable terms.

Citation or primary resource: https://doi.org/10.1093/nar/gkad1005

## NCBI Gene

GeneID and alias mapping. Exact hash; no explicit source release version recovered.

Access: https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/

Terms: https://www.ncbi.nlm.nih.gov/home/about/policies/

NCBI policy; incorporated source rights may apply. Raw source omitted.

Citation or primary resource: https://www.ncbi.nlm.nih.gov/gene/

## Disease Ontology

ontology and historical ontology input. 2026-05-30; historical2023-11-30.

Access: https://github.com/DiseaseOntology/HumanDiseaseOntology/releases

Terms: https://disease-ontology.org/

Historical project manifest records CC0; not independently cleared here. Raw source omitted.

Citation or primary resource: https://doi.org/10.1093/nar/gkab1063

## STRING

human physical interactions and protein-ID mapping. v12.0.

Access: https://stringdb-downloads.org/download/

Terms: https://string-db.org/cgi/access

CC BY4.0; attribution; incorporated external material retains original terms. Raw network omitted.

Citation or primary resource: https://doi.org/10.1093/nar/gkac1000

## HGNC

human gene background and mappings; historical identifier dictionary. Current fetched2026-09-22; historical2023-10-01.

Access: https://www.genenames.org/download/statistics-and-files/

Terms: https://www.genenames.org/about/

Original manifest records no restrictions; exact composite clearance not completed. Raw source omitted.

Citation or primary resource: https://doi.org/10.1093/nar/gkac888

## Mondo

strict-equivalence mappings and disease definitions. Ontology2026-09-01; mapping exact hash.

Access: https://mondo.monarchinitiative.org/

Terms: https://mondo.monarchinitiative.org/

Actual SSSOM header licence unspecified; do not blanket relicense mapping export. Raw ontology/mapping omitted.

Citation or primary resource: https://mondo.monarchinitiative.org/

## GO

functional ontology, human GOA annotations and provenance. Ontology2026-07-26; GAF generated2026-05-21; historical2023-10-09.

Access: https://geneontology.org/docs/download-ontology/

Terms: https://geneontology.org/docs/go-citation-policy/

CC BY4.0; release identification and attribution. Raw ontology/annotations omitted; moving current endpoints must match frozen hashes.

Citation or primary resource: https://doi.org/10.1093/genetics/iyad031

## GenCC

external-source gene–disease labels. Fetched2026-09-22; exact hash.

Access: https://thegencc.org/download

Terms: https://thegencc.org/terms

CC0 1.0; attribution requested. Raw source omitted.

Citation or primary resource: https://doi.org/10.1016/j.gim.2022.04.017

## ClinVar

variant-derived external labels and PubMed provenance. Variant summary fetched2026-09-22; citation table2026-09-23.

Access: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/

Terms: https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/

NCBI/ClinVar policy and contributing data terms. Raw variant and citation tables omitted.

Citation or primary resource: https://doi.org/10.1093/nar/gkz972

## GEO

expression cohorts GSE13355,GSE30999,GSE87466,GSE75214 and GPL570,GPL13158,GPL6244 mappings. Accession-specific files fetched2026-09-22; exact hashes.

Access: https://www.ncbi.nlm.nih.gov/geo/

Terms: https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html

NCBI/GEO policy; source study rights retained. Raw expression matrices omitted.

Citation or primary resource: https://doi.org/10.1093/nar/gks1193

## Human Protein Atlas

consensus normal-tissue RNA expression. Saved manifest calls current release HPA25.1/Ensembl109; exact hash controls.

Access: https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip

Terms: https://www.proteinatlas.org/about/licence

Current official general database licence CC BY4.0 with third-party constraints; saved CC BY-SA reading corrected separately. Raw expression data omitted; consensus not independent donors.

Citation or primary resource: https://doi.org/10.1126/science.1260419

## MANE

representative protein selection and RefSeq protein sequences. GRCh38v1.5.

Access: https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/

Terms: https://www.ncbi.nlm.nih.gov/home/about/policies/

Saved source metadata: NCBI public-domain/EMBL-EBI terms; exact composite clearance not completed. Raw sequences and summary omitted.

Citation or primary resource: https://doi.org/10.1038/s41586-022-04558-8

## UniProt

fallback canonical reference proteome and release metadata. RELEASE.metalink2026_03.

Access: https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/

Terms: https://www.uniprot.org/help/license

CC BY4.0 recorded in local RELEASE.metalink. Raw sequence source omitted.

Citation or primary resource: https://doi.org/10.1093/nar/gkac1052

## GTEx

historical tissue expression input. v8;2017-06-05 file label is not a claimed formal release date.

Access: https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz

Terms: https://gtexportal.org/home/license

GTEx applicable terms; exact redistribution clearance not completed. Raw expression source omitted.

Citation or primary resource: https://doi.org/10.1126/science.aaz1776

## SapBERT

frozen disease and ontology text encoder. 090663c3ae57bf35ffe4d0d468a2a88d03051a4d.

Access: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext/tree/090663c3ae57bf35ffe4d0d468a2a88d03051a4d

Terms: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext

Saved model manifest records Apache-2.0; preserve model-card terms. Original weights/tokenizer omitted; per-file hashes and pinned URLs retained.

Citation or primary resource: https://doi.org/10.18653/v1/2021.naacl-main.334

## ESM-2 35M

frozen protein-sequence encoder. 6fbf070e65b0b7291e7bbcd451118c216cff79d8.

Access: https://huggingface.co/facebook/esm2_t12_35M_UR50D/tree/6fbf070e65b0b7291e7bbcd451118c216cff79d8

Terms: https://github.com/facebookresearch/esm/blob/main/LICENSE

Saved model manifest records MIT; preserve original attribution. Original weights/tokenizer omitted; per-file hashes and pinned URLs retained.

Citation or primary resource: https://doi.org/10.1126/science.ade2574


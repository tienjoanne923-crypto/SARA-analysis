# SARA download tools and snapshot coverage

This is a static audit of the original download source code. None of the commands was executed while preparing this document. The scripts log downloaded bytes and SHA-256 values but do not compare them with the frozen September/October input manifests. URLs labelled `current`, `latest`, or a moving branch can return different bytes today.

| Original tool | Actual default target | Catalog | Important boundary |
|---|---|---|---|
| `code/download_raw_data.py` | Project root folders | CTD disease–gene and vocabulary, HPO ontology/HPOA/gene phenotype, NCBI gene2go and human gene info, DO, STRING v12 physical links/info/aliases | Fetches the root raw resources missing from the three external scripts. Physical is default; functional network can be selected. |
| `gpu8/tools/download_external.ps1` | `gpu8/external` | HGNC, Mondo OBO/SSSOM, GenCC, ClinVar condition mapping/variant summary, STRING functional network, four GEO series and three platforms, KDGene and qdgp source ZIPs | Does not fetch root CTD, physical STRING or HPO/NCBI inputs. Comparator ZIPs use moving branches. |
| `gpu8/tools/download_external_v6.ps1` | `gpu8/external/v6` | GTEx v8; Reactome current and v86; GO/GOA current and 2023-10-09; HPO gene phenotypes v2023-10-09; HGNC 2023-10-01; ClinVar citations; CTD Wayback CDX index | The CDX JSON is an archive listing, not the historical CTD data file. Does not rebuild the historical edge CSV. |
| `gpu8/tools/download_external_v71.ps1` | `gpu8/external/v71` | HPA tissue consensus/location, UniProt human reference proteome/release file, MANE v1.5 sequences/summary, IMPC calls, MP ontology, MGI homology and MP–HP mapping | `-Large` replaces this catalog with the large HPA image features file only. HPA image/IMPC mouse features are optional extensions for the main SARA task. |
| `python -m gpu8v71.resources71 --stage models` | `gpu8/external/v71/models` | SapBERT and ESM2-35M model/tokenizer files | Queries current model head and writes a new revision manifest. For frozen replay use the archived exact revisions and hashes through a manifest-aware method instead. |

The three PowerShell tools accept `-Root` and `-Overwrite`; V71 additionally accepts `-Large`. Existing nonempty files are skipped unless overwrite is requested. A skipped file is not thereby authenticated against the frozen expected hash. Scripts can log failed downloads and continue; their final `DONE` message does not certify that every required input exists.

The legacy Python downloader accepts `--root`, repeated `--only`, `--overwrite`, `--dry_run`, `--continue_on_error`, `--retries`, `--timeout`, `--min_free_gb`, `--string_version`, and `--string_network {physical,functional}`. Its `--dry_run` still goes through `write_manifest` in `main`; use source inspection for a strict no-write catalog review.

Required source routes and hashes are split among `manifests/resource_manifest.json`, the original V6 download log, `manifests_v71/resources_v71.json`, `manifests_v71/models_v71.json`, and `results_v8/audit/historical_inputs_v8.json`. Some resources are reused across versions, including current GO/GOA and GTEx. Do not count reused files as additional independent data sources.

The original historical acquisition script `code/build_temporal_historical_snapshot.py` records period-specific URLs and can request acquisition and legacy processing. Its option names use hyphens (`--snapshot-root`, `--download`, `--prepare-processed`, `--make-temporal-split`). It is an optional historical tool, not a source of the currently missing raw CTD dump. The surviving historical edge table and period-matched inputs must be verified separately. If those cannot be legally obtained with matching bytes, mark exact historical raw-data regeneration unavailable.

Redistribution permission is separate from technical accessibility. This code-only release omits raw data, derived biological arrays, prediction rows and pretrained weights pending explicit rights review. Preserve the applicable upstream terms; a code license does not relicense third-party resources.

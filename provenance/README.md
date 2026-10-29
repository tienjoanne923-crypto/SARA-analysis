# SARA provenance metadata

This directory records file identities and upstream access routes for the October 18 manuscript and its existing frozen analyses. It contains no third-party raw data, pretrained weights or new scientific results. October 22/P1 extensions are outside scope. All paths are relative to the original project data root.

`input_manifest.json` and `inputs.sha256` describe 101 relevant files selected from a 103-file read-only inventory. Two conditional v7.1 image/mouse extension tables were excluded. Of the 101 files, 62 match previously recorded individual hashes and 39 have newly measured hashes. The absence count for this selection must not be interpreted as completeness of the full pipeline.

`oct18_saved_analysis_manifest.json` separately records 220 existing verification records representing 213 unique files. Seven duplicate records were collapsed. These describe external-label/grouping audit inputs, saved predictions, code and fitting logs, not new retraining. Seven files overlap with the 101-file input selection; the combined catalogue therefore contains 307 unique paths. Do not add 101 and 220 and call the result a count of independent inputs.

`source_access.tsv` and its readable companion name 17 upstream resources and their access/terms routes. `hpa_licence_correction.json` preserves an additive correction to the old HPA licence reading; the historical source manifest is not rewritten. None of these source identifiers is a DOI for this SARA package.

The package has known missing dependencies. The original `gpu8/caches_v8/encoders` directory, including segmented disease embeddings used by Pack8, is absent. The complete historical CTD raw disease–gene capture is not retained; only a 262144-byte head and CDX metadata remain, alongside hashed processed historical tables. The selected inventory does not exhaust every feature cache or every runtime dependency. See `coverage.json` for the exact boundaries.

To verify supplied external data without modifying it, use `python verify_input_hashes.py --input-root PATH_TO_ORIGINAL_LAYOUT`. The script reads the 101-file manifest, reports missing/mismatched inputs and exits nonzero if they are not all identical. Supplying a different snapshot with the same filenames does not satisfy this check. Running this verifier does not train the models or reproduce performance estimates.

No public GitHub release, archive DOI, rights-holder licence grant or named-creator declaration is established by this metadata. Declarations should be updated only after actual access routes are published and checked.

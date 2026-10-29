# Local package verification

Verification date: October 29, 2026. These are bounded software and identity checks of the assembled source package. They do not establish independent biological validation, a fresh environment installation or complete numerical reproduction.

| Check | Observed outcome |
| --- | --- |
| Byte identity of copied originals | 322 files match the selected original hashes; original source files unchanged |
| Python syntax, without importing analysis modules | 264 Python files parsed successfully |
| Existing synthetic SARA tests | 28 passed in the original research environment |
| Standard-library helper behavior | 10 checks passed; no research stages or downloads executed |
| Combined provenance catalogue checked against local workspace | 307 unique files matched their content hashes |
| Relevant-input subset | 101 selected input/provenance files; 62 historical hashes reverified and 39 newly measured; all included in the 307-file identity check |
| Fixed-revision models, offline | 10 recorded model files match their expected sizes and hashes |
| Environment dependency imports | 11 of 11 selected imports succeeded in the original main environment plus encoder overlay |

Detailed records are in `validation/` and `environment/`. The 307-file catalogue includes source inputs, processed inputs, source code/configuration, fitting logs and saved outputs, explicitly labelled by type. It must not be described as 307 raw datasets. Only their hashes and metadata, not those data/model/output bytes, are included in the code archive.

The helpers were checked for fixed-revision download planning, stage planning, missing-input rejection, hash mismatch rejection, absence of analysis writes during preflight, and seven audit path adapters. AST comparison confirmed that these adapters change workspace path string literals only, retaining scientific expressions and embedded audit markers. The original scripts remain byte-identical in the archive. The network download branch has not been executed for this package.

The synthetic test file includes bounded optimization on toy arrays; it does not train the reported biological models. It ran with the original installed Windows environment, not a new installation of the portable specification. The original numerical results were not recomputed or altered.

## Not established

The preprocessing plan was checked against original source and CLI contracts, but its biological stages have not been rerun for this release. No independent full five-fold reproduction, blanket third-party redistribution clearance or persistent archive DOI has been established. The archive creator and MIT licence for original code have since been confirmed by the author; this candidate is distributed through the public GitHub repository. The missing original segmented V8 encoders, missing full historical CTD raw dump and seed-42 development-prediction dependency remain documented limitations.

`MANIFEST.sha256` covers the final payload and the JSON package manifest. The manifest excludes itself and `MANIFEST.sha256` to avoid self-reference. A separate `SHA256SUMS.txt` beside the ZIP identifies the complete deliverable archive. Run `scripts/verify_package.py` after extraction. It rejects changed, missing, duplicate or unlisted files, except documented runtime artefacts (`.git`, Python/pytest caches and the disposable-workspace marker). In a working copy containing restored research data, use `--allow-extra` to report those extra files without treating them as part of the original archive.

## Public candidate metadata update

The rc2 candidate adds the confirmed archive creator, MIT licence, citation/archive metadata and GitHub access wording. All 322 selected original research files retain their byte hashes. No scientific implementation or experimental result was changed. The earlier bounded software and input-identity checks therefore remain identified as checks of the same research source, not a new numerical reproduction.

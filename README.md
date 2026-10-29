# SARA analysis package

This source package accompanies the October 18 SARA scientific manuscript and the October 28 English revision prepared for **BioData Mining**. It contains the original SARA V8.1 implementation, its earlier preprocessing dependencies, frozen settings and development selections, provenance metadata, and input SHA-256 records. It excludes the October 21/22 extensions and P1 experiments. Experimental results have not been changed.

**Status on October 29, 2026:** public source release candidate `v1.0.0-rc2` at [https://github.com/tienjoanne923-crypto/SARA-analysis](https://github.com/tienjoanne923-crypto/SARA-analysis). The archive creator is **Jialu Xu (徐嘉璐)**. Author-owned original code is licensed under **MIT**; third-party resources retain their own terms. A persistent archive DOI has not yet been assigned. This is a code/configuration/provenance package, not a release of raw databases, processed tables, fitted weights or saved prediction arrays. A complete raw-to-results reproduction has not been performed for this package.

## Start here

Use Python 3.12. The following package-only checks need no research data or internet connection:

```text
python scripts/verify_package.py
python scripts/check_python_sources.py
python scripts/preprocess.py --list
python scripts/fetch_pinned_models.py
```

After installing the numerical environment, run the existing synthetic software checks:

```text
python scripts/run_property_checks.py
```

These check implementation properties, not biological performance or independent validation. Read `docs/VALIDATION.md` for the actual checks executed during packaging. A missing data/model file is reported as missing; a hash manifest does not supply that file.

## Environment

Read `environment/README.md`. The environment files distinguish the observed Windows research environment from portable installation specifications. They replace the local PyTorch wheel URI in the install instructions with an explicit version/index; the original environment record remains intact at `gpu8/requirements.txt`. Keep the repository layout: `gpu8/src/gpu8/paths.py` derives the data root from its own location.

## Inputs and preprocessing

`provenance/` contains relative input paths, SHA-256 values, byte sizes and source/version metadata. Obtain omitted third-party inputs from their original providers under the applicable terms, place them at their recorded relative paths in a separate working copy, and verify identity before running analyses:

```text
python scripts/verify_inputs.py --root PATH_TO_RESEARCH_WORKSPACE
```

Verification fails on missing or changed files. A fresh upstream download may differ from the frozen input. The pinned model helper uses the archived Hugging Face revision and verifies every recorded model file. It does not fetch latest model heads. `--download --destination PATH` is required for network downloads.

Original preprocessing code is included under `code/` and the four `gpu8/src` package trees. First create a separate disposable copy with `python scripts/make_working_copy.py --destination NEW_WORKSPACE_PATH`. Restore the required inputs there. `scripts/preprocess.py` lists or preflights the audited preprocessing stages and runs a selected stage only with `--stage STAGE_ID --workspace NEW_WORKSPACE_PATH --execute`. It checks prerequisite existence and known content hashes; it does not automatically run dependencies. The runbook explains prerequisite input files, overwrites and stage order. It does not reconstruct absent historical data.

## Run and audit the analysis

See `docs/REPRODUCING.md` and `docs/preprocessing_plan.json`. `scripts/run_module.py` sets the source import path for the selected workspace. Development selection, formal training, saved-prediction rescoring and manuscript assembly are distinct operations; only the first two fit models.

Important remaining dependencies are explicit: the original segmented V8 SapBERT arrays are absent and need regeneration; the complete historical CTD raw dump is absent; seed 42 formal training reuses development-stage predictions that this code-only archive does not contain. Frozen selections alone do not make that command self-sufficient. Do not interpret a rerun of development selection as an untouched preregistered replication.

## Contents

| Location | Scientific content |
| --- | --- |
| `gpu8/src/gpu8v8final` | SARA V8.1 retrieval, semantic models, availability handling, ranking and evaluation |
| `gpu8/src/gpu8`, `gpu8v6`, `gpu8v71` | Original ingestion, mapping, retrieval and encoder dependencies |
| `gpu8/configs`, `gpu8/manifests*` | Frozen protocol, split/context and model metadata |
| `gpu8/results_v8/dev/fold*/selection.json` | Existing development choices, not fitted models |
| `sara_revision_20261018` | Original saved-result audit code and aggregate audit metadata |
| `scripts` | Portable verification, fixed-revision model fetch and execution adapters |
| `provenance`, `MANIFEST.sha256` | Input identity records and the package's own file checksums |
| `declarations` | Current-status availability wording and clearly labelled post-publication templates |

The research source includes historical authoring and exploratory modules for provenance. The runbook identifies the analysis entrypoints; source inclusion is not a claim that every module was used in the final main comparison. Original local-path audit scripts remain byte-identical. `scripts/run_oct18_audit.py` supplies an explicit portable path adapter in a disposable working copy.

## Scientific and access limits

Retain the manuscript's assigned-group definition, cross-role equivalence limitation, development-based selection, post hoc sensitivity status and database dependence. This archive adds no experiments or evidence of clinical applicability. Historical directory digests based on paths/size/mtime are not content hashes; use the new SHA-256 lists instead.

The MIT licence covers author-owned original code and associated original documentation. Third-party data, pretrained models and incorporated third-party material retain their original terms. See `LICENSE`, `COPYRIGHT_AND_LICENSING.md` and `provenance/DATA_SOURCES.md`. `CITATION.cff` supplies the confirmed software creator and version; `.zenodo.json` prepares corresponding archive metadata. Neither file asserts a DOI.

## Repository and DOI

The candidate release is [v1.0.0-rc2](https://github.com/tienjoanne923-crypto/SARA-analysis/releases/tag/v1.0.0-rc2). Resolve its exact source revision with `git rev-parse refs/tags/v1.0.0-rc2^{commit}`, or inspect the commit linked from the release. Release assets include the prepared analysis ZIP, availability wording and SHA-256 checksums. The release record identifies the full source commit and payload identity. A persistent archive DOI remains pending. `declarations/CURRENT_STATUS.txt` describes GitHub access; `declarations/AFTER_PUBLICATION_TEMPLATE.txt` remains a template until a **published version DOI** is verified. No DOI badge or example identifier is presented as real.

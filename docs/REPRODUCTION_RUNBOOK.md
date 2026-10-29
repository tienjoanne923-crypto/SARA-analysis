# SARA reproduction runbook

This runbook describes the October 18 scientific baseline used by the BioData Mining revision. It separates inspecting a code archive, rebuilding inputs, and reproducing fitted predictions. The source and CLI contracts below were checked by reading the original files on October 29, 2026. The archive preparation did not execute end-to-end training or an independent-environment reproduction.

The code-only archive does not contain raw or processed research data, prediction arrays, or fitted weights. Obtain the required snapshots through their recorded source routes, subject to their terms, and verify the recorded input hashes before attempting numerical reproduction. Fetching a current database release is a new-input analysis and may change every downstream result.

## Choose the intended task

| Task | Needs third-party inputs | Writes analysis artifacts | What success establishes |
|---|---|---|---|
| Inspect source and verify archive integrity | No | No, apart from an explicitly chosen report location | Files match the released archive and code parses |
| Run synthetic property tests | No biological input | Pytest cache and temporary test files | Specific numerical and software properties on synthetic examples |
| Rebuild base tables and features | Yes, matching snapshots and pretrained encoders | Yes | Locally reconstructed input artifacts; compare hashes and schema before use |
| Rescore saved predictions | Yes, ground truth, entity index and matching prediction arrays | Yes | Metrics conditional on supplied predictions |
| Refit frozen selected models | Yes, all model inputs, original role definitions and selected configurations | Yes, potentially large | A new execution of the frozen fitting procedure; compare predictions with a reference release |
| Repeat method development and registration | Yes | Yes, including new selections and protocol records | A separate research execution, not independent confirmation and not a new preregistration of old results |

Inspect-only verification can be completed without satisfying the later rows. Neither passing synthetic tests nor recalculating input hashes reproduces the reported biological results.

## Work in a disposable copy

The code derives paths from the location of `gpu8/src/gpu8/paths.py`. Changing the working directory alone does not redirect output. Place the complete release tree in a new location and point `PYTHONPATH` at that copy. Do not import the source from a live analysis directory while expecting output to go elsewhere.

Expected layout:

```text
SARA_ROOT/
  code/                         legacy raw-data and gene-background preparation
  association/                  CTD snapshots supplied separately
  phenotype/                    HPO snapshots supplied separately
  function/                     NCBI resources supplied separately
  ontology/                     Disease Ontology supplied separately
  ppi/                          STRING physical network and mappings supplied separately
  processed_data/               legacy nodes_gene.csv regenerated separately
  data_snapshots/historical_2023q4/   optional historical inputs
  gpu8/
    src/gpu8/
    src/gpu8v6/
    src/gpu8v71/
    src/gpu8v8final/
    configs/
    manifests/                  frozen split and context definitions
    manifests_v6/               frozen retriever/scorer choices
    manifests_v71/              model revisions and feature provenance
    manifests_v8/               frozen SARA choices and original protocol record
    results_v8/dev/fold0..4/selection.json
```

In PowerShell, after placing the copy in the desired directory:

```powershell
$saraRoot = (Get-Location).Path
$env:PYTHONPATH = Join-Path $saraRoot 'gpu8/src'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

All `python` commands below refer to the Python interpreter of the environment prepared for this copied tree. The recorded scientific run used Windows 11, Python 3.12.10, Torch 2.9.1+cu128, LightGBM 4.7.0 and NumPy 2.5.3. See the release environment documentation for installation. The historical `gpu8/requirements.txt` contains an absolute local Torch wheel URL and does not include the separately installed transformer stack; it is an environment record, not a portable installation command. The normal scientific pipeline does not require Word or MathType. Some preserved manuscript-authoring source modules do.

## Tier A inspect and test

Run the archive's shipped integrity verifier from the top-level README. It should verify released file hashes without downloading anything or creating analysis outputs. Original scientific sources should retain their archived hashes; any adapters should be identified separately.

Python source can be parsed without importing model code or writing bytecode:

```powershell
python -c "import ast,pathlib; fs=list(pathlib.Path('gpu8/src').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p)) for p in fs]; print('Parsed',len(fs),'Python files')"
```

After installing the scientific dependencies, this bounded test file uses synthetic inputs:

```powershell
python -m pytest -p no:cacheprovider gpu8/tests/test_v8_properties.py -q
```

`test_v8_integration.py` requires real frozen tables, mask arrays, development pools and run artifacts. Its ledger test calls `ledger8.collect()`, which is an artifact-generating operation; do not present the entire integration suite as an archive-only read operation. Execute it only in the disposable analysis copy after the real inputs have been supplied. Failure from missing omitted data is an unmet input requirement, not evidence that all algorithms are broken.

## Tier B prepare exact input snapshots

The authoritative identities are the SHA-256 values in the input manifests, not mutable download URLs or local modification times. Keep both the original manifest and a new materialization report. Stop on a missing or mismatched required input. Do not update the expected hash merely to make a verification pass.

The download tools log what they fetch but do not enforce the original expected hashes. They also write logs and manifests. They belong in a disposable acquisition area, not the inspect-only tier. Their catalog is described in `DOWNLOAD_CATALOG.md`.

Optional catalog/download commands, **for obtaining inputs rather than proving snapshot identity**:

```powershell
python code/download_raw_data.py --root . --only association,phenotype,function,ontology,ppi --string_version 12.0 --string_network physical
powershell -File gpu8/tools/download_external.ps1
powershell -File gpu8/tools/download_external_v6.ps1
powershell -File gpu8/tools/download_external_v71.ps1
```

The base external downloader also fetches GEO and comparator archives, which are unnecessary for the minimal SARA main chain. `download_external_v71.ps1 -Large` downloads only the large HPA image file; it replaces, rather than appends to, the ordinary catalog. Main SARA does not require the optional image or mouse channels.

### Preserve the original gene background

`gpu8.ingest` reads `processed_data/nodes_gene.csv` when `keep_legacy_gene_space=true`. If that file is absent the routine continues with a different background. Rebuild or supply the exact legacy gene list before canonical ingestion. The legacy preparation is self-contained in `code/prepare_processed_data.py`; it is not imported automatically by `gpu8.ingest`.

The original legacy summary records inferred associations off, minimum inference score 50, PPI threshold 700 and no row/pair truncation. Corresponding command:

```powershell
python code/prepare_processed_data.py --root . --out_dir processed_data --min_inference_score 50 --ppi_score_threshold 700 --max_ctd_rows 0 --max_pairs 0
```

There is no `--include_inferred` switch in this command because the recorded option was false. The archived legacy result had 8,676 gene nodes. This is not the final SARA background, which contains 19,861 genes after canonical ingestion. Compare the expected file hash as well as the count.

### Canonical tables and role definitions

```powershell
python -m gpu8.ingest
python -m gpu8.mapping
```

Both commands write files. `ingest` also saves the loaded configuration and regenerates the resource manifest. Use a copied configuration, and compare regenerated table/schema identities with the frozen references. Mapping must precede role construction because assigned CTD-tree keys and strict-equivalence fields are part of the role definitions.

For exact replay, retain the shipped `splits_main_5fold.json`, `contexts.json` and related role records. The original role-generation CLI is:

```powershell
python -m gpu8.protocol
```

This command regenerates and overwrites the split/context files. It belongs to rebuilding the original protocol in a separate copy; do not use it casually in a frozen replay and do not describe a newly written timestamp as the original registration. Compare all role memberships and hashes before proceeding.

Reference checks for this baseline: 13,321 disease records; 19,861 background genes; 5,758 positive-label diseases; 31,577 primary marker/mechanism pairs; 864 development diseases, 656 split-holdout diseases and 4,238 outer-test diseases; five outer folds with three inner blocks each, yielding 20 SARA fitting contexts. These counts are diagnostic checks, not replacements for identity hashes or the disclosed limitations of assigned groups.

### Retrieval-side preprocessing

```powershell
python -m gpu8v6.phenotypes_v6
python -m gpu8v6.disease_text
```

The repaired phenotype table and masked disease-text table depend on the exact base tables, HPO annotations, HGNC dictionary and Mondo mappings. They are required by the role-based retrieval engine.

The following original modules support older baselines, functional-feature comparisons and some secondary analyses. They are not sufficient to build the SARA frozen encoder feature pack:

```powershell
python -m gpu8v6.gene_features --version current
python -m gpu8v6.gene_features --version hist2023
python -m gpu8v6.disease_semantics
python -m gpu8.contexts
```

`gpu8.contexts` writes a context index containing local paths. A context index copied from a different location must not be treated as a portable cache locator; regenerate the context caches in the disposable copy when those legacy baseline views are needed. These base contexts are distinct from the SARA full-background retrieval passes below.

### Frozen encoder inputs and semantic features

Use the archived `models_v71.json` revisions and per-file hashes. The original model acquisition command is `python -m gpu8v71.resources71 --stage models`, but it queries the **current model head** and rewrites the manifest. It is unsuitable for acquiring an exact old revision as-is. Obtain each listed model file at the recorded revision using a manifest-aware acquisition method, then verify all expected hashes:

- SapBERT `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` at `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`.
- ESM2-35M `facebook/esm2_t12_35M_UR50D` at `6fbf070e65b0b7291e7bbcd451118c216cff79d8`.

Place verified files at `gpu8/external/v71/models/sapbert` and `esm2_35m`. Then rebuild in dependency order:

```powershell
python -m gpu8v71.sequences71
python -m gpu8v71.encoders71 --device cuda --only disease hpo go esm
python -m gpu8v71.function71
python -m gpu8v71.expression71
python -m gpu8v71.featurepack71
python -m gpu8v8final.textseg8 1
```

The encoder `--only` values above were checked in `encoders71.run`. Sequence construction requires MANE/UniProt plus the base gene index. GO functions require the original GO/GOA snapshot, encoded GO terms, and mapping resources. Expression requires HPA consensus and GTEx v8. The feature pack is aligned to the base entity order and contains masks as well as numeric arrays. Optional image arrays are included by `featurepack71` only when present; omitting them may change the `.npz` byte hash even though the main SARA arrays are unchanged, so report the artifact scope and compare per-array hashes.

**Known missing cache:** `gpu8/caches_v8/encoders` was absent from the inspected local source tree. This includes the required segmented main disease vectors. `textseg8 1` can reconstruct them from verified inputs; reconstructed arrays must receive new hashes and must not be described as the missing original file. The level-2 sensitivity additionally uses `python -m gpu8v8final.textseg8 2`.

The main chain loads the frozen GO mask sets even when optional completion is disabled. Rebuild mask and evidence artifacts after feature preparation if omitted:

```powershell
python -m gpu8v8final.masks8
python -m gpu8v8final.evidence8
python -m gpu8v8final.v6pass8 main
```

Compare the mask array hashes with the frozen mask manifest. `evidence8` parses GO provenance and builds evidence blocks; it writes audit and table files. `v6pass8 main` creates the 20 full-background retrieval contexts and can be large. It uses the frozen V6 retrieval settings; do not retune them for exact replay. The CLI takes a world name positionally (`main`, `e6cold`, `blocks`, `blocks_random`), then optional context identifiers. It does not expose a `--device` flag; the implementation defaults to CUDA.

## Tier C frozen fitting and saved-prediction evaluation

All fitting commands below require the Tier B inputs. They use the original stored configurations. Their presence in this runbook is not evidence that they have been executed successfully in a clean environment.

### Main-chain replay has a seed 42 dependency

For seeds 52 and 62, the original formal runner fits using the frozen per-fold selections:

```powershell
python -m gpu8v8final.formal8 --tag main --seeds 52 62 --folds 0 1 2 3 4 --threads 8
```

The original three-seed command is:

```powershell
python -m gpu8v8final.formal8 --tag main --seeds 42 52 62 --folds 0 1 2 3 4 --threads 8
```

For **main seed 42**, `formal8.run` copies `pred_*.npz` from `results_v8/dev/fold*/` instead of calling `fold_models` for the final rankers. `selection.json` alone cannot supply these predictions. Exact replay of this CLI therefore needs the original development predictions and fitted artifacts, or a separately documented replay adapter that refits only the stored selected configuration. This code-only release has no such development predictions. Do not claim that the three-seed command is independently sufficient, and do not silently rerun development selection to fill the gap.

The original runner also skips a fold when its output `done.json` exists and can reuse cached model fits. A successful process exit in a populated directory does not establish a fresh fit. Use a new output tree and retain an execution record identifying which inputs were imported, recomputed or reused.

After matching prediction files have been supplied or generated:

```powershell
python -m gpu8v8final.eval8 main
```

The positional tag `main` limits the evaluator to the main analysis. With no positional tags it scans other available arms too. The command writes `eval_test_main.parquet` and `eval_index.json`; it is not a read-only archive verifier. Report endpoint definitions, disease denominators, seed averaging and assigned-group bootstrap units exactly as specified in the manuscript.

### Secondary analyses require their own inputs

Actual source CLI examples:

```powershell
python -m gpu8v8final.variants8 plain_rrf v6pool --seeds 42 --folds 0 1 2 3 4 --threads 8
python -m gpu8v8final.external8
python -m gpu8v8final.hist8
python -m gpu8v8final.analysis8 strata hitflow
python -m gpu8v8final.pubdata8 light
python -m gpu8v8final.pubdata8b noshap
```

Do not run the entire set to validate a code-only archive. `variants8` refits or rescores depending on the arm; `external8` consumes the frozen V7.1 confirmation set and main predictions; `hist8` retrains in the historical world; `pubdata8 light` still requires multiple saved model/result artifacts; `pubdata8b noshap` excludes SHAP fitting but still requires secondary saved results. `final8` additionally assembles older manuscript/figure products and is not a minimal scientific replay command.

External-label construction starts with `python -m gpu8.external_labels` and the frozen V7.1 confirmation-set definition. The frozen ClinVar first-representative rule and later strict-subset sensitivity must remain distinct. The supplementary October 18 audit scripts have original local path assumptions and require portable adapters before execution outside the original workspace. They rescore saved predictions and do not retrain models.

Historical reconstruction needs `data_snapshots/historical_2023q4/processed_data/disease_gene_edges.csv`. The historical CTD raw association dump was no longer present in the inspected project; current download scripts cannot recover it by inference. `python -m gpu8.temporal --step tables` builds canonical historical tables from the surviving processed historical edges plus period-specific vocabulary/phenotypes. `hist8` additionally needs the historical inputs enumerated in `historical_inputs_v8.json`. Keep the manuscript's disclosure that the background index, sequences, grouping definitions and selected configurations are later than the cutoff.

### Earlier determinism check

`python -m gpu8v8final.repro8` reruns a fold-0, seed-42 chain and compares stored pools, teacher weights and scores. It also needs reference development predictions and many real inputs. The existing audit reports identical top-10 lists and a maximum score difference of approximately 1.19e-7. Its attempted separate environment was blocked, so the saved execution used the main interpreter and fresh caches. Cite it as the recorded determinism check; do not call it an independently reproduced clean-environment result.

## Research development is a separate execution

The original development CLI sequence is:

```powershell
python -m gpu8v8final.dev8 --stage controls --folds 0 1 2 3 4 --threads 8
python -m gpu8v8final.dev8 --stage teachers --folds 0 1 2 3 4 --threads 8
python -m gpu8v8final.dev8 --stage devmask --folds 0 1 2 3 4 --threads 8
python -m gpu8v8final.dev8 --stage chain --folds 0 1 2 3 4 --threads 8
python -m gpu8v8final.register8
```

These stages select teacher controls, optional completion, fusion weight and final ranker/corrector settings. In the current code, `controls` and `devmask` internally iterate all five folds even though the shared CLI accepts `--folds`; only the `teachers` and `chain` branches forward the requested fold subset. The chain skips an existing `selection.json`. Registration returns an existing record if found, and otherwise refuses registration when V8 test evaluations already exist.

Do not delete an old protocol record merely to create a new registration timestamp. Keep the original record as historical provenance. A fresh research-development run must use a new tree and state that it repeats development on already-used cohorts; it does not provide new independent validation or retrospectively strengthen the original design.

## Record the outcome precisely

For each execution record the source archive hash, copied source tree location, Python/package/CUDA versions, hardware, exact commands, input-hash results, start/end times, exit codes and paths/hashes of produced artifacts. Identify all cache reuse. Where the source exposes only CUDA by default, do not claim CPU portability without testing or an explicitly reviewed adapter.

Use status labels such as `source_verified`, `synthetic_tests_passed`, `inputs_materialized_and_verified`, `recomputed_from_saved_predictions`, or `refit_compared_with_reference`. The current archive preparation establishes source inventory and packaging checks only. A public DOI must refer to an actually published record; a local archive or draft deposit is not a public persistent archive.

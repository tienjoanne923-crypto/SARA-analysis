# Observed SARA software environment

This directory records the installed software used by the original local SARA pipeline and supplies exact package pins without machine-local wheel paths. It supports inspection and environment reconstruction; it is not evidence that a fresh installation or full training reproduction has passed.

## What was observed

The original main environment uses CPython 3.12.10 on Windows 11 x64. Its 50 distribution metadata records and the separately installed encoder overlay's 28 records were read on 2026-10-29. Nine names overlap, all with identical versions. The merged set contains 69 distributions, including 19 encoder distributions absent from the main environment. Package dependency requirements evaluated for Windows/Python 3.12 reported no missing distributions or version conflicts in this merged metadata set.

`observed_environment.json` retains exact versions, package dependency declarations, metadata SHA-256 values, available wheel tags, the source requirement file's hash, and the overlap/compatibility audit. The check is against installed metadata; it does not validate binary ABI, available upstream wheels, resolver behavior on another system or every conditional/optional import.

`requirements.original.txt` preserves the original file unchanged. It includes a local PyTorch wheel URI and should not be passed directly to pip on another computer. `requirements.observed-main.txt` holds its observed main package versions with build tools and PyTorch separated. `requirements.observed-encoders.txt` holds the 19 additional encoder packages. `requirements.portable.txt` includes both. `requirements.build-tools.txt` records observed pip/setuptools/wheel versions, without automatically upgrading the target environment.

## Environment reconstruction

Use a separate Python 3.12.10 environment. A compatible NVIDIA driver and CUDA-capable GPU are required for the original GPU training path. CUDA availability on the original machine does not establish compatibility on another machine. These installation commands are instructions, not steps executed during archival preparation.

```text
python -m venv .venv
python -m pip install -r environment/requirements.torch-cu128.txt
python -m pip install -r environment/requirements.portable.txt
python -m pip check
```

Activate the new environment first, or replace each `python` with its environment-specific executable. The torch-only file uses the official CUDA 12.8 index and pins the exact observed `torch==2.9.1+cu128` build. PyTorch's official instructions list version 2.9.1 for that index: https://pytorch.org/get-started/previous-versions/#v2-9-1 . Neither torchvision nor torchaudio was present in the observed main environment, so neither was added to these pins. The original wheel checksum remains recorded in the unchanged requirements file and available distribution metadata.

Keep the repository's `gpu8` directory structure and add `gpu8/src` to `PYTHONPATH` before invoking research modules. The original pipeline appends the separate encoder library directory after the main environment's paths. The portable requirements place the same versions in one environment; this normalized arrangement has not been freshly installed or benchmarked. Packages from the original installed library trees are not redistributed.

## Checks performed during archive preparation

`runtime_probe.main.json` records a bounded, offline dependency import check with the original main environment plus its encoder overlay. NumPy, SciPy, pandas, PyArrow, LightGBM, PyTorch, pytest, transformers, tokenizers, huggingface_hub and safetensors all imported successfully. PyTorch reported CUDA build 12.8 and CUDA available. No model weights or research inputs were read, and no training occurred.

`runtime_probe.bundled.json` records the editing runtime separately. It is CPython 3.12.14 with NumPy 2.3.5 and pandas 3.0.1, and lacks SciPy, PyArrow, LightGBM, PyTorch, pytest and encoder dependencies. It is suitable for standard-library syntax/hash checks, but cannot be used to claim the scientific property or real-input integration tests passed.

For this workstation's bounded software checks, use the original main environment (`gpu8/.venv/Scripts/python.exe`) and disable bytecode/cache writes where preservation matters. The pipeline's synthetic property tests are `gpu8/tests/test_v8_properties.py`; they require scientific libraries but no biological input. Real-input integration tests require omitted tables, embeddings, pools and ledger outputs and must be run only after those resources are restored and their hashes verified.

`static_imports.json` inventories imports across the original four research packages by parsing source without executing it. It also includes historical manuscript/figure/MathType helper modules. Their optional publication dependencies such as PyMuPDF, pypdf, latex2mathml, olefile and pywin32 are outside the observed main-plus-encoder set and were not added to the scientific requirements. Do not interpret those helper scripts as portable analysis entrypoints or requirements for the core scientific pipeline.

## Limitations

This is an exact installed-version snapshot, not a hash-locked distribution archive. It does not contain dependency wheels, fresh resolution logs, model weights or third-party scientific datasets. Metadata compatibility and successful imports do not establish numerical reproducibility. The archived fold-0/seed-42 historical determinism check must remain distinguished from a new independent-environment reproduction. No source files, original environments or experimental outputs were modified to create this environment record.

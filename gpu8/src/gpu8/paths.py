"""Self-locating project paths for the GPU8 pipeline.

Never hard-codes an absolute project root: every path is derived from the
location of this file, so the tree can be moved or renamed.
"""
from __future__ import annotations

import os
from pathlib import Path

# .../gpu8/src/gpu8/paths.py -> parents[2] == .../gpu8
GPU8_ROOT = Path(__file__).resolve().parents[2]
# .../gpu8 -> parent == project root (D:\信)
PROJECT_ROOT = GPU8_ROOT.parent

# --- raw upstream resources (shared with the legacy pipeline) ---
ASSOCIATION_DIR = PROJECT_ROOT / "association"
PHENOTYPE_DIR = PROJECT_ROOT / "phenotype"
FUNCTION_DIR = PROJECT_ROOT / "function"
ONTOLOGY_DIR = PROJECT_ROOT / "ontology"
PPI_DIR = PROJECT_ROOT / "ppi"
LEGACY_PROCESSED = PROJECT_ROOT / "processed_data"
LEGACY_CODE = PROJECT_ROOT / "code"
SNAPSHOT_ROOT = PROJECT_ROOT / "data_snapshots" / "historical_2023q4"

# --- gpu8 working tree ---
CONFIG_DIR = GPU8_ROOT / "configs"
MANIFEST_DIR = GPU8_ROOT / "manifests"
EXTERNAL_DIR = GPU8_ROOT / "external"
TABLES_DIR = GPU8_ROOT / "tables"          # nine canonical parquet tables
CACHE_DIR = GPU8_ROOT / "caches"
RUNS_DIR = GPU8_ROOT / "runs"
RESULTS_DIR = GPU8_ROOT / "results"
PER_DISEASE_DIR = RESULTS_DIR / "per_disease"
RESOURCE_LOG_DIR = RESULTS_DIR / "resource_logs"
RESULT_TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
REPORT_DIR = GPU8_ROOT / "reports"
LOG_DIR = GPU8_ROOT / "logs"

ALL_DIRS = [
    CONFIG_DIR, MANIFEST_DIR, EXTERNAL_DIR, TABLES_DIR, CACHE_DIR, RUNS_DIR,
    RESULTS_DIR, PER_DISEASE_DIR, RESOURCE_LOG_DIR, RESULT_TABLE_DIR,
    FIGURE_DIR, REPORT_DIR, LOG_DIR,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def external(*parts: str) -> Path:
    return EXTERNAL_DIR.joinpath(*parts)


def table(name: str) -> Path:
    """Path of one of the nine canonical tables (parquet)."""
    if not name.endswith(".parquet"):
        name = name + ".parquet"
    return TABLES_DIR / name


def cache(context_hash: str, *parts: str) -> Path:
    return CACHE_DIR.joinpath(context_hash, *parts)


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def rel(p: Path | str) -> str:
    """Path relative to the project root, for manifests and reports."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT)).replace(os.sep, "/")
    except ValueError:
        return str(p)

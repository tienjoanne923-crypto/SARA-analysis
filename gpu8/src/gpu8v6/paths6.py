"""V6 output locations.  Nothing here points into a V5 results directory."""
from __future__ import annotations

from pathlib import Path

from gpu8 import paths as p5

GPU8_ROOT = p5.GPU8_ROOT
PROJECT_ROOT = p5.PROJECT_ROOT

RESULTS = GPU8_ROOT / "results_v6"
RUNS = GPU8_ROOT / "runs_v6"
MANIFESTS = GPU8_ROOT / "manifests_v6"
CACHES = GPU8_ROOT / "caches_v6"
TABLES = GPU8_ROOT / "tables_v6"
REPORTS = GPU8_ROOT / "reports_v6"
LOGS = GPU8_ROOT / "logs"
EXTERNAL_V6 = p5.EXTERNAL_DIR / "v6"

AUDIT = RESULTS / "audit"
RETRIEVAL = RESULTS / "retrieval"
SCORING = RESULTS / "scoring"
SUPPORT = RESULTS / "support"
PERMUTATION = RESULTS / "permutation"
EXTERNAL_RES = RESULTS / "external"
FIGURES = RESULTS / "figures"
TABLES_OUT = RESULTS / "tables"

ALL = [RESULTS, RUNS, MANIFESTS, CACHES, TABLES, REPORTS, LOGS, AUDIT, RETRIEVAL,
       SCORING, SUPPORT, PERMUTATION, EXTERNAL_RES, FIGURES, TABLES_OUT]


def ensure() -> None:
    for d in ALL:
        d.mkdir(parents=True, exist_ok=True)


def ext(*parts: str) -> Path:
    return EXTERNAL_V6.joinpath(*parts)


def rel(path) -> str:
    return p5.rel(path)

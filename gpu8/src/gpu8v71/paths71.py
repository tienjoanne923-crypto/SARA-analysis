"""V7.1 output locations.  Nothing here points into a V5 or V6 results directory."""
from __future__ import annotations

import sys
from pathlib import Path

from gpu8 import paths as p5

GPU8_ROOT = p5.GPU8_ROOT
PROJECT_ROOT = p5.PROJECT_ROOT

RESULTS = GPU8_ROOT / "results_v71"
RUNS = GPU8_ROOT / "runs_v71"
MANIFESTS = GPU8_ROOT / "manifests_v71"
CACHES = GPU8_ROOT / "caches_v71"
TABLES = GPU8_ROOT / "tables_v71"
REPORTS = GPU8_ROOT / "reports_v71"
LOGS = GPU8_ROOT / "logs"
EXTERNAL = p5.EXTERNAL_DIR / "v71"
MODELS = EXTERNAL / "models"
LIBS = GPU8_ROOT / "v71_libs"

AUDIT = RESULTS / "audit"
CORE = RESULTS / "core"
FUSION = RESULTS / "fusion"
BEHAVIOR = RESULTS / "behavior"
EXT_RES = RESULTS / "external"
FIGURES = RESULTS / "figures"
TABLES_OUT = RESULTS / "tables"
SCORES = CACHES / "scores"
POOLS = CACHES / "pools"

ALL = [RESULTS, RUNS, MANIFESTS, CACHES, TABLES, REPORTS, LOGS, EXTERNAL, MODELS, AUDIT, CORE,
       FUSION, BEHAVIOR, EXT_RES, FIGURES, TABLES_OUT, SCORES, POOLS]


def ensure() -> None:
    for d in ALL:
        d.mkdir(parents=True, exist_ok=True)


def ext(*parts: str) -> Path:
    return EXTERNAL.joinpath(*parts)


def rel(path) -> str:
    return p5.rel(path)


def add_libs() -> None:
    """Make the separately installed encoder libraries importable.

    They are *appended* to ``sys.path`` so that every package already present in
    the frozen virtual environment keeps precedence; the environment itself is
    never modified.
    """
    s = str(LIBS)
    if LIBS.exists() and s not in sys.path:
        sys.path.append(s)

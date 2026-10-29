"""V8.1 output locations.  Nothing here points into a V5, V6 or V7.1 directory."""
from __future__ import annotations

import sys
from pathlib import Path

from gpu8 import paths as p5

GPU8_ROOT = p5.GPU8_ROOT
PROJECT_ROOT = p5.PROJECT_ROOT

RESULTS = GPU8_ROOT / "results_v8"
RUNS = GPU8_ROOT / "runs_v8"
MANIFESTS = GPU8_ROOT / "manifests_v8"
CACHES = GPU8_ROOT / "caches_v8"
TABLES = GPU8_ROOT / "tables_v8"
REPORTS = GPU8_ROOT / "reports_v8"
LOGS = GPU8_ROOT / "logs"
EXTERNAL = p5.EXTERNAL_DIR / "v8"
MANUSCRIPT = PROJECT_ROOT / "manuscript_v8"
LIBS = GPU8_ROOT / "v71_libs"          # separately installed encoder libraries (reused)

AUDIT = RESULTS / "audit"
DEV = RESULTS / "dev"
FORMAL = RESULTS / "formal"
VARIANTS = RESULTS / "variants"
EXT_RES = RESULTS / "external"
FIGURES = RESULTS / "figures"
TABLES_OUT = RESULTS / "tables"
MODELS = CACHES / "models"
POOLS = CACHES / "pools"
V6PASS = CACHES / "v6pass"
FEATS = CACHES / "features"

ALL = [RESULTS, RUNS, MANIFESTS, CACHES, TABLES, REPORTS, LOGS, EXTERNAL, AUDIT, DEV, FORMAL,
       VARIANTS, EXT_RES, FIGURES, TABLES_OUT, MODELS, POOLS, V6PASS, FEATS]

# V7.1 inputs that V8.1 reads (never writes)
V71_CACHES = GPU8_ROOT / "caches_v71"
V71_TABLES = GPU8_ROOT / "tables_v71"
V71_MANIFESTS = GPU8_ROOT / "manifests_v71"
V71_RUNS = GPU8_ROOT / "runs_v71"
V71_EXTERNAL = p5.EXTERNAL_DIR / "v71"


def ensure() -> None:
    for d in ALL:
        d.mkdir(parents=True, exist_ok=True)


def rel(path) -> str:
    return p5.rel(path)


def add_libs() -> None:
    s = str(LIBS)
    if LIBS.exists() and s not in sys.path:
        sys.path.append(s)

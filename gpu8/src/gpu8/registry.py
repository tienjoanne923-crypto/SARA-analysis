"""Run ledger: the 166 formal fits and the 40 development fits.

Each row starts as ``planned``.  A run transitions to ``running`` then
``done``/``failed``; re-executions bump ``attempt`` instead of inventing a new
seed.  Local support solves, support-order draws, candidate cut-offs and
statistical resampling are deliberately *not* rows here — they do not count as
global fits.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from . import paths

COLUMNS = [
    "run_id", "experiment", "method", "protocol", "split", "fold", "seed",
    "variant", "status", "attempt", "planned_group",
    "started_utc", "finished_utc", "seconds",
    "config_hash", "data_hash", "context_hash", "candidate_hash", "code_hash",
    "epochs_run", "best_epoch", "best_val_metric",
    "device", "peak_rss_mib", "peak_gpu_allocated_mib", "peak_gpu_reserved_mib",
    "n_test_diseases", "output_hash", "output_dir", "note",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunRecord:
    run_id: str
    experiment: str = ""
    method: str = ""
    protocol: str = "gpu8_v1"
    split: str = ""
    fold: str = ""
    seed: str = ""
    variant: str = ""
    status: str = "planned"
    attempt: int = 0
    planned_group: str = ""
    started_utc: str = ""
    finished_utc: str = ""
    seconds: float = 0.0
    config_hash: str = ""
    data_hash: str = ""
    context_hash: str = ""
    candidate_hash: str = ""
    code_hash: str = ""
    epochs_run: str = ""
    best_epoch: str = ""
    best_val_metric: str = ""
    device: str = ""
    peak_rss_mib: float = 0.0
    peak_gpu_allocated_mib: float = 0.0
    peak_gpu_reserved_mib: float = 0.0
    n_test_diseases: str = ""
    output_hash: str = ""
    output_dir: str = ""
    note: str = ""

    def as_row(self) -> dict:
        return {k: asdict(self).get(k, "") for k in COLUMNS}


class RunRegistry:
    """CSV-backed ledger with atomic rewrite on every mutation."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.rows: list[dict] = []
        if self.path.exists():
            self.load()

    # -- io ---------------------------------------------------------------
    def load(self) -> "RunRegistry":
        with open(self.path, "r", encoding="utf-8-sig", newline="") as fh:
            self.rows = [dict(r) for r in csv.DictReader(fh)]
        for r in self.rows:
            for c in COLUMNS:
                r.setdefault(c, "")
        return self

    def save(self) -> Path:
        """Write the ledger.

        Several runs execute concurrently, so the temp file is process-unique
        and a failed write is reported but never fatal: the authoritative record
        of a run is its own ``result.json``, and ``reconcile_from_runs`` folds
        those back in afterwards.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".csv.{os.getpid()}.part")
        try:
            with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=COLUMNS)
                w.writeheader()
                for r in self.rows:
                    w.writerow({c: r.get(c, "") for c in COLUMNS})
            os.replace(tmp, self.path)
        except (PermissionError, OSError) as exc:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            print(f"[registry] deferred ledger write ({type(exc).__name__}); "
                  f"the run result file remains authoritative", flush=True)
        return self.path

    # -- mutation ---------------------------------------------------------
    def add(self, rec: RunRecord) -> None:
        self.rows.append(rec.as_row())

    def extend(self, recs: Iterable[RunRecord]) -> None:
        for r in recs:
            self.add(r)

    def get(self, run_id: str) -> dict | None:
        for r in self.rows:
            if r.get("run_id") == run_id:
                return r
        return None

    def update(self, run_id: str, **fields) -> dict:
        row = self.get(run_id)
        if row is None:
            row = RunRecord(run_id=run_id).as_row()
            self.rows.append(row)
        for k, v in fields.items():
            if k in COLUMNS:
                row[k] = v
        self.save()
        return row

    def mark_running(self, run_id: str, **fields) -> dict:
        row = self.get(run_id)
        attempt = int(row.get("attempt") or 0) + 1 if row else 1
        return self.update(run_id, status="running", attempt=attempt,
                           started_utc=utcnow(), **fields)

    def mark_done(self, run_id: str, **fields) -> dict:
        return self.update(run_id, status="done", finished_utc=utcnow(), **fields)

    def mark_failed(self, run_id: str, note: str = "", **fields) -> dict:
        return self.update(run_id, status="failed", finished_utc=utcnow(),
                           note=note[:500], **fields)

    def mark_cancelled(self, run_id: str, note: str = "") -> dict:
        return self.update(run_id, status="cancelled", finished_utc=utcnow(),
                           note=note[:500])

    # -- reporting --------------------------------------------------------
    def reconcile_from_runs(self, runs_dir=None) -> dict:
        """Rebuild statuses from the per-run result files.

        Several runs execute concurrently and each one rewrites the whole CSV,
        so a row written by one process can be clobbered by another.  The
        per-run ``result.json`` is the authoritative record; this pass folds it
        back into the ledger.
        """
        import json as _json

        runs_dir = Path(runs_dir) if runs_dir else (self.path.parents[1] / "runs")
        updated = 0
        for row in self.rows:
            rid = row.get("run_id", "")
            p = runs_dir / rid / "result.json"
            if not p.exists():
                continue
            try:
                blob = _json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            res = blob.get("resources", {}) or {}
            fit = blob.get("fit_info", {}) or {}
            row.update({
                "status": "done",
                "seconds": round(float(res.get("seconds", 0.0)), 2),
                "peak_rss_mib": round(float(res.get("peak_rss_mib", 0.0)), 1),
                "peak_gpu_allocated_mib": round(float(res.get("peak_gpu_allocated_mib", 0.0)), 1),
                "peak_gpu_reserved_mib": round(float(res.get("peak_gpu_reserved_mib", 0.0)), 1),
                "device": blob.get("device", ""),
                "epochs_run": str(fit.get("epochs_run", "")),
                "best_epoch": str(fit.get("best_epoch", "")),
                "n_test_diseases": str(blob.get("n_test_diseases", "")),
                "config_hash": blob.get("config_hash", row.get("config_hash", "")),
                "code_hash": blob.get("code_hash", row.get("code_hash", "")),
                "output_dir": str(runs_dir / rid),
            })
            if not row.get("finished_utc"):
                row["finished_utc"] = utcnow()
            updated += 1
        self.save()
        return {"updated": updated, "status_counts": self.status_counts()}

    def status_counts(self) -> dict:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.get("status", "")] = out.get(r.get("status", ""), 0) + 1
        return out

    def pending(self) -> list[dict]:
        return [r for r in self.rows if r.get("status") in ("planned", "failed", "running")]

    def done_ids(self) -> set[str]:
        return {r["run_id"] for r in self.rows if r.get("status") == "done"}


# --------------------------------------------------------------------------
# Plan of record: the 166-fit ledger (plan section 13.2)
# --------------------------------------------------------------------------
FORMAL_METHODS_TRAINABLE = [
    "b4_lightgbm", "b5_mlp", "b6_sign", "b7_kdgene", "b8_meta_ridge", "b9_ours",
]
FORMAL_METHODS_DETERMINISTIC = ["b2_lr", "b3_cpu_fusion"]
FAMILY_METHODS = ["b4_lightgbm", "b7_kdgene", "b8_meta_ridge", "b9_ours"]
KEY_METHODS = ["b4_lightgbm", "b7_kdgene", "b8_meta_ridge", "b9_ours"]
ABLATIONS = ["a1_no_missing_training", "a2_equal_weight_fixed", "a3_no_phenotype", "a4_no_ppi"]
SECOND_PPI_METHODS = ["b9_ours", "b7_kdgene"]


def build_plan(cfg, route: str = "166") -> list[RunRecord]:
    """Materialise the planned run list.

    route='166' is the recommended path; route='134' is the reduced path
    (single seed for the MLP / SIGN comparators and for the four ablations).
    """
    p = cfg.protocol
    folds = list(range(p.n_outer_folds))
    seeds = list(p.train_seeds)
    recs: list[RunRecord] = []

    def add(exp, method, fold, seed, variant, group, split="main"):
        rid = f"{exp}__{method}__{split}__f{fold}__s{seed}" + (f"__{variant}" if variant else "")
        recs.append(RunRecord(run_id=rid, experiment=exp, method=method,
                              protocol=cfg.protocol_id, split=split,
                              fold=str(fold), seed=str(seed), variant=variant,
                              planned_group=group, config_hash=cfg.config_hash))

    # E01 main task: 2 deterministic models x 5 folds = 10
    for m in FORMAL_METHODS_DETERMINISTIC:
        for f in folds:
            add("E01", m, f, "fixed", "", "main_deterministic")

    # E01 main task: 6 trainable models x 5 folds x 3 seeds = 90 (or 70 on the 134 route)
    reduced = {"b5_mlp", "b6_sign"} if route == "134" else set()
    for m in FORMAL_METHODS_TRAINABLE:
        use_seeds = [seeds[0]] if m in reduced else seeds
        for f in folds:
            for s in use_seeds:
                add("E01", m, f, s, "", "main_trainable")

    # E02 family isolation: 4 methods x 1 split x 3 seeds = 12
    for m in FAMILY_METHODS:
        for s in seeds:
            add("E02", m, "family", s, "", "family_isolation", split="family")

    # E06 retraining ablations: 4 configs x 3 folds x 2 seeds = 24 (12 on the 134 route)
    ab_seeds = [p.ablation_seeds[0]] if route == "134" else list(p.ablation_seeds)
    for v in ABLATIONS:
        for f in p.ablation_folds:
            for s in ab_seeds:
                add("E06", "b9_ours", f, s, v, "ablation")

    # E08 historical freeze: 4 key methods x 3 seeds = 12
    for m in KEY_METHODS:
        for s in seeds:
            add("E08", m, "temporal", s, "", "temporal", split="temporal")

    # E07 external-dedicated fits: 4 key methods x 3 seeds = 12
    for m in KEY_METHODS:
        for s in seeds:
            add("E07", m, "external", s, "", "external", split="external")

    # E09 second PPI refit: 2 methods x 1 split x 3 seeds = 6
    for m in SECOND_PPI_METHODS:
        for s in seeds:
            add("E09", m, "ppi2", s, "", "second_ppi", split="ppi2")

    return recs


DEV_BUDGET = {
    "b9_ours": 8, "b8_meta_ridge": 6, "b5_mlp": 4, "b6_sign": 6,
    "b7_kdgene": 6, "b4_lightgbm": 6, "b2_lr": 2, "b3_cpu_fusion": 2,
}


def build_dev_plan(cfg) -> list[RunRecord]:
    recs: list[RunRecord] = []
    for method, n in DEV_BUDGET.items():
        for i in range(n):
            rid = f"DEV__{method}__cfg{i:02d}"
            recs.append(RunRecord(run_id=rid, experiment="DEV", method=method,
                                  protocol=cfg.protocol_id, split="dev",
                                  fold="dev", seed=str(cfg.protocol.train_seeds[0]),
                                  planned_group="development",
                                  config_hash=cfg.config_hash))
    return recs


def write_plans(cfg, route: str = "166") -> dict:
    formal = build_plan(cfg, route=route)
    dev = build_dev_plan(cfg)
    reg = RunRegistry(paths.MANIFEST_DIR / f"run_registry_{route}.csv")
    if not reg.rows:
        reg.extend(formal)
        reg.save()
    dreg = RunRegistry(paths.MANIFEST_DIR / "development_registry_40.csv")
    if not dreg.rows:
        dreg.extend(dev)
        dreg.save()
    return {"formal_planned": len(formal), "dev_planned": len(dev),
            "formal_path": str(reg.path), "dev_path": str(dreg.path)}

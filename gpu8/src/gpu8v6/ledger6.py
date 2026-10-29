"""V6 section 13.4 — one authoritative result file per run.

Every V6 run writes ``runs_v6/<run_id>/result.json`` through an atomic rename.
Workers never edit a shared CSV; ``collect`` rebuilds the ledger table from the
result files, checking content hashes rather than file existence.  A record
carries the data/code versions, the outer and inner contexts, the model id, the
candidate generator (frozen-retriever hash) and candidate hash, the support
sampling and evidence-grouping rules when applicable, the raw metrics, the
actual number of diseases, peak resources and, for a failed run, the reason.
"""
from __future__ import annotations

import json
import traceback
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from gpu8.hashing import atomic_write_json, code_version_hash, sha256_obj
from gpu8.resources import ResourceMonitor

from . import PROTOCOL_ID, paths6
from .common import load_manifest, now_utc, to_json_safe


def code_hash() -> str:
    return code_version_hash(Path(__file__).resolve().parent)


def run_dir(run_id: str) -> Path:
    d = paths6.RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def keep_failed_attempts(run_id: str) -> None:
    """Rename an earlier failed result so a rerun never overwrites failure history."""
    d = run_dir(run_id)
    p = d / "result.json"
    if not p.exists():
        return
    try:
        old = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        old = {"status": "unreadable"}
    if old.get("status") != "done":
        i = 1
        while (d / f"result_failed_attempt_{i}.json").exists():
            i += 1
        p.rename(d / f"result_failed_attempt_{i}.json")


def write_result(run_id: str, payload: dict) -> Path:
    keep_failed_attempts(run_id)
    payload = to_json_safe(dict(payload))
    payload.setdefault("run_id", run_id)
    payload.setdefault("protocol_id", PROTOCOL_ID)
    payload.setdefault("code_hash", code_hash())
    payload.setdefault("finished_utc", now_utc())
    ret = load_manifest("retriever_v6_frozen.json", {}) or {}
    payload.setdefault("retriever_hash", ret.get("payload_hash", ""))
    payload["content_hash"] = sha256_obj({k: v for k, v in payload.items()
                                          if k not in ("finished_utc", "content_hash")})
    return atomic_write_json(run_dir(run_id) / "result.json", payload)


@contextmanager
def tracked(run_id: str, meta: dict, device: str = "cuda"):
    """Resource-monitored block; a failure is recorded, never swallowed."""
    rec = {"run_id": run_id, "status": "running", "started_utc": now_utc(), **meta}
    mon = ResourceMonitor(interval=0.25, sample_nvidia=(device == "cuda"))
    mon.__enter__()
    box = {"extra": {}}
    try:
        yield box
        mon.__exit__(None, None, None)
        rec["status"] = "done"
    except Exception as exc:  # recorded, then re-raised
        mon.__exit__(None, None, None)
        rec["status"] = "failed"
        rec["failure_reason"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-4000:]
        rec["resources"] = mon.result.as_dict()
        write_result(run_id, {**rec, **box["extra"]})
        raise
    rec["resources"] = mon.result.as_dict()
    write_result(run_id, {**rec, **box["extra"]})


def collect() -> pd.DataFrame:
    rows = []
    for p in sorted(paths6.RUNS.glob("*/result.json")):
        blob = json.loads(p.read_text(encoding="utf-8"))
        ok = blob.get("content_hash") == sha256_obj(
            {k: v for k, v in blob.items() if k not in ("finished_utc", "content_hash")})
        res = blob.get("resources", {}) or {}
        m = blob.get("metrics", {}) or {}
        rows.append({"run_id": blob.get("run_id"), "status": blob.get("status"),
                     "stage": blob.get("stage"), "method": blob.get("method"),
                     "fold": blob.get("fold"), "seed": blob.get("seed"),
                     "config": json.dumps(blob.get("config", {}), sort_keys=True),
                     "n_test_diseases": blob.get("n_test_diseases"),
                     "recall10_end": m.get("recall10_end"),
                     "oracle_recall10": m.get("oracle_recall10"),
                     "seconds": res.get("seconds"), "peak_rss_mib": res.get("peak_rss_mib"),
                     "peak_gpu_allocated_mib": res.get("peak_gpu_allocated_mib"),
                     "peak_gpu_reserved_mib": res.get("peak_gpu_reserved_mib"),
                     "candidate_hash": blob.get("candidate_hash"),
                     "retriever_hash": blob.get("retriever_hash"),
                     "code_hash": blob.get("code_hash"),
                     "failure_reason": blob.get("failure_reason", ""),
                     "hash_ok": ok})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(paths6.MANIFESTS / "run_ledger_v6.csv", index=False, encoding="utf-8-sig")
    return df

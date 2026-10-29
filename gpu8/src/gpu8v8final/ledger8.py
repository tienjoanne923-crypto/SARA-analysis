"""One authoritative ``result.json`` per V8.1 run (V6/V7.1 contract + role accounting).

Each record carries ``role`` (dev / formal / variant / comparator / audit) and
``reused_from`` so every fit is counted once; failed attempts are renamed, never
overwritten; ``collect`` rebuilds the ledger from the result files and checks each
content hash (single writer, no shared CSV edited by workers).
"""
from __future__ import annotations

import json
import traceback
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from gpu8.hashing import atomic_write_json, code_version_hash, sha256_obj
from gpu8.resources import ResourceMonitor

from . import PROTOCOL_ID, paths8
from .common8 import now_utc, to_json_safe


def code_hash() -> str:
    return code_version_hash(Path(__file__).resolve().parent)


def run_dir(run_id: str) -> Path:
    d = paths8.RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_done(run_id: str) -> bool:
    p = paths8.RUNS / run_id / "result.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status") == "done"
    except json.JSONDecodeError:
        return False


def read_result(run_id: str) -> dict:
    return json.loads((paths8.RUNS / run_id / "result.json").read_text(encoding="utf-8"))


def keep_failed_attempts(run_id: str) -> None:
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
    payload.setdefault("reused_from", None)
    payload["content_hash"] = sha256_obj({k: v for k, v in payload.items()
                                          if k not in ("finished_utc", "content_hash")})
    return atomic_write_json(run_dir(run_id) / "result.json", payload)


@contextmanager
def tracked(run_id: str, meta: dict, device: str = "cuda"):
    rec = {"run_id": run_id, "status": "running", "started_utc": now_utc(), **meta}
    if "role" not in rec:
        raise ValueError("every V8.1 run declares its role (dev/formal/variant/comparator/audit)")
    mon = ResourceMonitor(interval=0.5, sample_nvidia=False)
    mon.__enter__()
    box = {"extra": {}}
    try:
        yield box
        mon.__exit__(None, None, None)
        rec["status"] = "done"
    except Exception as exc:
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
    for p in sorted(paths8.RUNS.glob("*/result.json")):
        blob = json.loads(p.read_text(encoding="utf-8"))
        ok = blob.get("content_hash") == sha256_obj(
            {k: v for k, v in blob.items() if k not in ("finished_utc", "content_hash")})
        res = blob.get("resources", {}) or {}
        m = blob.get("metrics", {}) or {}
        fit = blob.get("fit", {}) or {}
        diag = blob.get("diag", {}) or {}
        secs = res.get("seconds") or fit.get("seconds") or diag.get("seconds") or fit.get("seconds_fit")
        rows.append({"run_id": blob.get("run_id"), "status": blob.get("status"),
                     "role": blob.get("role"), "stage": blob.get("stage"),
                     "component": blob.get("component"), "arm": blob.get("arm"),
                     "context_id": blob.get("context_id"), "fold": blob.get("fold"),
                     "seed": blob.get("seed"), "reused_from": blob.get("reused_from"),
                     "config": json.dumps(blob.get("config", {}), sort_keys=True, default=str),
                     "n_test_diseases": blob.get("n_test_diseases"),
                     "recall10_end": m.get("recall10_end"), "cold_recall10": m.get("cold_recall10"),
                     "started_utc": blob.get("started_utc"), "finished_utc": blob.get("finished_utc"),
                     "seconds": secs, "peak_rss_mib": res.get("peak_rss_mib"),
                     "peak_gpu_allocated_mib": res.get("peak_gpu_allocated_mib"),
                     "peak_gpu_reserved_mib": res.get("peak_gpu_reserved_mib"),
                     "code_hash": blob.get("code_hash"), "failure_reason": blob.get("failure_reason", ""),
                     "hash_ok": ok})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(paths8.MANIFESTS / "run_ledger_v8.csv", index=False, encoding="utf-8-sig")
    return df

"""Content hashing and atomic writes.

Every artefact the pipeline produces is written atomically (temp file plus
rename) and recorded with a SHA-256, so a half-written CSV can never be
mistaken for a completed run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

_CHUNK = 1 << 20


def sha256_file(path: Path | str, chunk: int = _CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    """Stable hash of a JSON-serialisable object (sorted keys)."""
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def short_hash(value: str, n: int = 12) -> str:
    return value[:n]


def hash_many(paths: Iterable[Path | str]) -> dict:
    out = {}
    for p in paths:
        p = Path(p)
        if p.exists() and p.is_file():
            out[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding=encoding, newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)
    return path


def atomic_write_json(path: Path | str, obj: Any, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent, default=str))


def atomic_write_df(df, path: Path | str, **kwargs) -> Path:
    """Atomic DataFrame write; format chosen by suffix (.parquet or .csv)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    if path.suffix == ".parquet":
        df.to_parquet(tmp, index=False, **kwargs)
    else:
        kwargs.setdefault("encoding", "utf-8-sig")
        df.to_csv(tmp, index=False, **kwargs)
    os.replace(tmp, path)
    return path


def read_json(path: Path | str, default: Any = None) -> Any:
    """Read JSON, tolerating a UTF-8 BOM (PowerShell writes one by default)."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return default


def code_version_hash(src_dir: Path | str) -> str:
    """Hash of all .py files under src_dir; part of every cache key."""
    src_dir = Path(src_dir)
    items = []
    for p in sorted(src_dir.rglob("*.py")):
        items.append((str(p.relative_to(src_dir)).replace(os.sep, "/"), sha256_file(p)))
    return sha256_obj(items)

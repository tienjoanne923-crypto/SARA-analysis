"""Small V8.1 utilities: manifests, JSON, hashing, seeded RNG, dependency keys."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from gpu8.hashing import atomic_write_json, read_json, sha256_obj
from gpu8v6.common import now_utc, stable_seed, to_json_safe  # noqa: F401 (re-exported)

from . import PROTOCOL_ID, paths8


def write_manifest(name: str, payload: dict) -> Path:
    payload = dict(to_json_safe(payload))
    payload.setdefault("protocol_id", PROTOCOL_ID)
    payload.setdefault("written_utc", now_utc())
    payload["payload_hash"] = sha256_obj({k: v for k, v in payload.items()
                                          if k not in ("written_utc", "payload_hash")})
    return atomic_write_json(paths8.MANIFESTS / name, payload)


def load_manifest(name: str, default=None):
    return read_json(paths8.MANIFESTS / name, default)


def dump_json(path, obj) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(path, to_json_safe(obj))


def array_hash(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def list_hash(xs) -> str:
    h = hashlib.sha256()
    for x in xs:
        h.update(str(x).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(stable_seed(PROTOCOL_ID, *parts))


def dep_key(**parts) -> str:
    """Stable dependency key of a cache entry (resources, training set, rules, params...)."""
    return sha256_obj({k: parts[k] for k in sorted(parts)})[:24]


def dir_hash(root: Path, skip=("__pycache__",)) -> dict:
    """{relative path: (size, mtime_ns)} digest of a directory tree (no content read)."""
    import os
    items = []
    root = Path(root)
    if not root.exists():
        return {"exists": False}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip]
        for f in fns:
            p = Path(dp) / f
            try:
                st = p.stat()
            except OSError:
                continue
            items.append((str(p.relative_to(root)), st.st_size, st.st_mtime_ns))
    items.sort()
    return {"exists": True, "n_files": len(items),
            "digest": hashlib.sha256(json.dumps(items).encode()).hexdigest()}

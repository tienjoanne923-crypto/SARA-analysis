"""Shared V6 state: tables, splits, contexts, labels and small utilities.

``World6`` loads the canonical V5 tables once.  It never builds anything that
depends on a split; context-specific structures are produced by the source
builders, which receive the context's reference set explicitly.  Labels are held
here only so that the *evaluator* and the *training-loss* code can read them;
the retrieval, feature and propagation functions receive disease ids, reference
sets and exposed support, never query labels.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from gpu8 import paths as p5
from gpu8.config import load_config
from gpu8.contexts import load_entity_index
from gpu8.hashing import atomic_write_json, read_json, sha256_obj
from gpu8.protocol import fold_assignment, load_spec

from . import PROTOCOL_ID
from . import paths6


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def candidate_hash(disease_ids, gene_rows) -> str:
    """SHA-256 of the sorted (disease, gene_row) pairs of a candidate table."""
    df = pd.DataFrame({"d": np.asarray(disease_ids, dtype=str),
                       "g": np.asarray(gene_rows, dtype=np.int64)})
    df = df.sort_values(["d", "g"])
    h = hashlib.sha256()
    for d, sub in df.groupby("d", sort=True):
        h.update(d.encode())
        h.update(sub["g"].to_numpy(dtype=np.int64).tobytes())
    return h.hexdigest()


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(stable_seed(PROTOCOL_ID, *parts))


@dataclass
class ContextSpec:
    context_id: str
    fold: int
    kind: str                         # inner | outer
    block_index: int
    reference: list[str]
    held_out: list[str]

    @property
    def reference_set(self) -> set:
        return set(self.reference)


class World6:
    def __init__(self, load_heavy: bool = True):
        t0 = time.time()
        self.cfg = load_config()
        td = p5.TABLES_DIR
        self.idx = load_entity_index(td)
        self.relations = pd.read_parquet(td / "relations.parquet")
        self.evidence = pd.read_parquet(td / "evidence.parquet")
        self.phenotypes = pd.read_parquet(td / "phenotypes.parquet")
        self.hierarchy = pd.read_parquet(td / "hierarchy.parquet")
        self.genes = pd.read_parquet(td / "genes.parquet")
        self.diseases = pd.read_parquet(td / "diseases.parquet")
        self.hpo_terms = pd.read_parquet(td / "hpo_terms.parquet")
        self.ppi = pd.read_parquet(td / "ppi.parquet") if load_heavy else None
        self.spec = load_spec()
        self.family_of = dict(zip(self.diseases["disease_id"], self.diseases["family_group"]))
        primary = set(self.cfg.data.primary_evidence)
        rel = self.relations[self.relations["relation_type"].isin(primary)]
        self.primary_relations = rel
        self.positives: dict[str, list[str]] = rel.groupby("disease_id")["gene_id"].apply(
            lambda s: sorted(set(s))).to_dict()
        self.relation_of = {(r.disease_id, r.gene_id): r.relation_id
                            for r in rel.itertuples(index=False)}
        ev = self.evidence[self.evidence["pmid"].astype(str).str.len() > 0]
        self.relation_pmids: dict[str, set[str]] = ev.groupby("relation_id")["pmid"].apply(
            set).to_dict()
        self.n_g = self.idx.n_g
        self.n_d = self.idx.n_d
        self.background_rows = np.arange(self.n_g, dtype=np.int64)
        self.gene_row_of = self.idx.g_ix
        self.gene_id_of_row = {i: g for g, i in self.idx.g_ix.items()}
        ctx = read_json(p5.MANIFEST_DIR / "contexts.json", {})
        self.contexts: dict[str, ContextSpec] = {}
        for s in ctx.get("contexts", []):
            self.contexts[s["context_id"]] = ContextSpec(
                context_id=s["context_id"], fold=int(s["fold"]), kind=s["kind"],
                block_index=int(s["block_index"]), reference=list(s["reference_diseases"]),
                held_out=list(s.get("held_out_block") or []))
        self.load_seconds = time.time() - t0

    # -- labels (evaluator / loss only) -----------------------------------
    def query_rows(self, disease_id: str, exclude: set | None = None) -> np.ndarray:
        g_ix = self.idx.g_ix
        genes = [g for g in self.positives.get(disease_id, []) if g in g_ix]
        if exclude:
            genes = [g for g in genes if g not in exclude]
        return np.array(sorted(g_ix[g] for g in genes), dtype=np.int64)

    # -- splits ------------------------------------------------------------
    def fold(self, fold: int) -> dict:
        return fold_assignment(self.spec, fold, self.cfg.protocol.n_outer_folds)

    def labelled(self, ids) -> list[str]:
        return [d for d in ids if self.positives.get(d) and d in self.idx.d_ix]

    def outer_context(self, fold: int) -> ContextSpec:
        return self.contexts[f"f{fold}_outer"]

    def inner_contexts(self, fold: int) -> list[ContextSpec]:
        return [self.contexts[f"f{fold}_inner{b}"] for b in range(3)
                if f"f{fold}_inner{b}" in self.contexts]

    def training_context_of(self, fold: int) -> dict[str, str]:
        """Training disease -> the inner context that holds it out (OOF)."""
        out = {}
        for c in self.inner_contexts(fold):
            for d in c.held_out:
                out[d] = c.context_id
        return out

    def reference_rows(self, ctx: ContextSpec) -> np.ndarray:
        d_ix = self.idx.d_ix
        return np.array(sorted(d_ix[d] for d in ctx.reference if d in d_ix), dtype=np.int64)

    def adjacency(self, ctx: ContextSpec) -> sparse.csr_matrix:
        """Disease x gene adjacency of the context's reference relations only."""
        rel = self.primary_relations
        rel = rel[rel["disease_id"].isin(ctx.reference_set)]
        rows = rel["disease_id"].map(self.idx.d_ix).to_numpy()
        cols = rel["gene_id"].map(self.idx.g_ix).to_numpy()
        keep = ~(pd.isna(rows) | pd.isna(cols))
        rows, cols = rows[keep].astype(np.int64), cols[keep].astype(np.int64)
        A = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(self.n_d, self.n_g)).tocsr()
        A.sum_duplicates()
        A.data[:] = 1.0
        return A


def write_manifest(name: str, payload: dict) -> Path:
    payload = dict(payload)
    payload.setdefault("protocol_id", PROTOCOL_ID)
    payload.setdefault("written_utc", now_utc())
    payload["payload_hash"] = sha256_obj({k: v for k, v in payload.items()
                                          if k not in ("written_utc", "payload_hash")})
    return atomic_write_json(paths6.MANIFESTS / name, payload)


def load_manifest(name: str, default=None):
    return read_json(paths6.MANIFESTS / name, default)


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    return obj


def dump_json(path, obj) -> Path:
    return atomic_write_json(path, to_json_safe(obj))

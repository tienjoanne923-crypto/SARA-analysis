"""Row blocks of one context: pool genes, the 30 V6 channel features, the
availability-aware extra features and (training / evaluation only) the labels.

Every input of a row comes from the row's own context (V6 pass, teacher, student,
reference visibility, direct gene HPO visibility, cold flag).  Labels are read
only through ``Labels``, which refuses F_j test labels before the V8.1
registration exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gpu8v6.scoring import FEATS
from gpu8v6.sources import SOURCES

from . import paths8
from .common8 import load_manifest
from .pool8 import pool_hash

N_G = 19861
EXTRA = ["s_sem", "lr_S", "lr_R", "m_sem", "src_go", "src_seq", "s_E", "a_E", "ref_visible",
         "hpo_visible", "lg_ref_degree", "gt_protein", "gt_ncrna", "has_seq", "has_go", "has_expr",
         "dm_text", "dm_hpo", "dm_tissue"]
NAN_WHEN_MISSING = {"s_sem": "m_sem", "lr_S": "m_sem", "s_E": "a_E"}
LABEL_BURDEN = {"n_query", "n_relations", "n_pmids", "n_positive", "n_query_cold", "label", "y"}


class Labels:
    """The only door to relation labels (training loss / checkpoint / evaluation)."""

    def __init__(self, w, test_ids: set | None = None, drop_pairs: set | None = None):
        self.w = w
        self.test_ids = test_ids or set()
        self.drop = drop_pairs or set()        # (disease, gene_id) withheld from training / checkpoint labels

    def registered(self) -> bool:
        return bool(load_manifest("protocol_v8_registered.json"))

    def rows(self, d: str, role: str) -> np.ndarray:
        if role == "test" and not self.registered():
            raise PermissionError("F_j test labels are sealed until the V8.1 registration")
        if self.drop and role != "test":
            ex = {g for (dd, g) in self.drop if dd == d}
            if ex:
                return self.w.query_rows(d, exclude=ex)
        return self.w.query_rows(d)


@dataclass
class RowBlock:
    ctx: str
    role: str
    ids: list
    genes: np.ndarray                 # [D, N]
    XB: np.ndarray                    # [D, N, 30]
    XE: np.ndarray                    # [D, N, len(EXTRA)] (0 where missing)
    a: np.ndarray                     # [D, N] semantic gate
    cold: np.ndarray                  # [D, N] reference-path cold in this context
    src: np.ndarray                   # [D, N] semantic_source
    y: np.ndarray | None = None       # [D, N]
    n_query: np.ndarray | None = None
    n_query_cold: np.ndarray | None = None
    info: dict = field(default_factory=dict)
    valid: np.ndarray | None = None   # [D, N] real candidate (False = padding)
    train_ok: np.ndarray | None = None  # [D, N] may enter a training table (E6 withdraws masked genes)

    def __post_init__(self):
        if self.valid is None:
            self.valid = np.ones(self.genes.shape, dtype=bool)
        if self.train_ok is None:
            self.train_ok = self.valid.copy()

    @property
    def D(self):
        return len(self.ids)

    @property
    def N(self):
        return self.genes.shape[1]

    def XE_nan(self) -> np.ndarray:
        X = self.XE.copy()
        for f, gate in NAN_WHEN_MISSING.items():
            i, g = EXTRA.index(f), EXTRA.index(gate)
            X[:, :, i] = np.where(X[:, :, g] > 0, X[:, :, i], np.nan)
        return X

    def extra(self, name: str) -> np.ndarray:
        return self.XE[:, :, EXTRA.index(name)]

    def base(self, name: str) -> np.ndarray:
        return self.XB[:, :, FEATS.index(name)]

    def subset(self, idx) -> "RowBlock":
        idx = np.asarray(idx, dtype=np.int64)
        pick = lambda a: None if a is None else a[idx]
        return RowBlock(self.ctx, self.role, [self.ids[i] for i in idx], self.genes[idx], self.XB[idx],
                        self.XE[idx], self.a[idx], self.cold[idx], self.src[idx], pick(self.y),
                        pick(self.n_query), pick(self.n_query_cold), dict(self.info), self.valid[idx],
                        self.train_ok[idx])


class GeneStatic:
    def __init__(self, w, pack_a: dict):
        g = w.genes
        lg = g["locus_group"].astype(str).str.lower()
        self.protein = lg.str.contains("protein").to_numpy()
        self.ncrna = lg.str.contains("non-coding").to_numpy()
        self.has_seq = pack_a["g_sequence_mask"].astype(bool)
        self.has_go = pack_a["g_function_mask"].astype(bool)
        self.has_expr = pack_a["g_expression_mask"].any(axis=1)
        self.d_text = pack_a["d_text_mask"].astype(bool)
        self.d_hpo = pack_a["d_hpo_mask"].astype(bool)
        self.d_tissue = pack_a["d_tissue_mask"].astype(bool)


def source_ranks(V: np.ndarray) -> np.ndarray:
    """[7, G] -> full-background 1-based ranks (stable: gene-row tie-break)."""
    out = np.empty(V.shape, dtype=np.int32)
    for s in range(V.shape[0]):
        o = np.argsort(-V[s], kind="stable")
        out[s, o] = np.arange(1, V.shape[1] + 1, dtype=np.int32)
    return out


def build_block(ctx: str, role: str, ids: list, genes: np.ndarray, R: np.ndarray, v6p, sem, gs: GeneStatic,
                d_ix: dict, labels: Labels | None = None) -> RowBlock:
    """genes [D, N] (pool, in pool order; -1 = padding) for the given ids (subset of v6p.ids and sem.ids)."""
    valid = genes >= 0
    genes = np.where(valid, genes, 0)
    D, N = genes.shape
    prow = v6p.rows(ids)
    spos = {d: i for i, d in enumerate(sem.ids)}
    srow = np.array([spos[d] for d in ids], dtype=np.int64)
    XB = np.zeros((D, N, len(FEATS)), dtype=np.float32)
    XE = np.zeros((D, N, len(EXTRA)), dtype=np.float32)
    di = v6p.dinfo.reindex(ids)
    fi = {f: i for i, f in enumerate(FEATS)}
    ei = {f: i for i, f in enumerate(EXTRA)}
    ref_vis = v6p.ref_degree > 0
    cold_g = v6p.cold()
    for k, (pr, sr) in enumerate(zip(prow, srow)):
        g = genes[k]
        Vd = np.asarray(v6p.V[pr])                     # [7, G]
        rk = source_ranks(Vd)
        for si, s in enumerate(SOURCES):
            v = Vd[si, g]
            XB[k, :, fi[f"v_{s}"]] = v
            XB[k, :, fi[f"lr_{s}"]] = np.where(v > 0, np.log(N_G / np.maximum(rk[si, g], 1)), 0.0)
            XB[k, :, fi[f"a_{s}"]] = float(di[f"a_{s}"].iloc[k])
        fu = np.asarray(v6p.fused[pr])[g]
        rb = np.asarray(v6p.rB[pr]).astype(np.float64)[g]
        XB[k, :, fi["fused"]] = fu
        XB[k, :, fi["ret_rank"]] = np.log(500.0 / rb)
        XB[k, :, fi["is_fill"]] = (fu <= 0).astype(np.float32)
        XB[k, :, fi["prior"]] = v6p.prior[g]
        XB[k, :, fi["d_hpo"]] = np.log1p(float(di["hpo_terms"].iloc[k]))
        XB[k, :, fi["d_pheno_sim"]] = float(di["pheno_max_sim"].iloc[k])
        XB[k, :, fi["d_text_sim"]] = float(di["text_max_sim"].iloc[k])
        XB[k, :, fi["d_onto_ic"]] = float(di["onto_max_ic"].iloc[k]) / 10.0
        XB[k, :, fi["d_onto_inf"]] = float(di["onto_level"].iloc[k] == "informative")
        m = sem.m[sr, g]
        XE[k, :, ei["s_sem"]] = np.where(m, sem.s[sr, g], 0.0)
        XE[k, :, ei["lr_S"]] = np.where(m, np.log(N_G / np.maximum(sem.rS[sr, g], 1)), 0.0)
        XE[k, :, ei["lr_R"]] = np.log(500.0 / np.arange(1, N + 1))
        XE[k, :, ei["m_sem"]] = m
        XE[k, :, ei["src_go"]] = sem.src[g] == 1
        XE[k, :, ei["src_seq"]] = sem.src[g] == 2
        XE[k, :, ei["s_E"]] = sem.sE[sr, g]
        XE[k, :, ei["a_E"]] = sem.aE[sr, g]
        dr = d_ix[ids[k]]
        XE[k, :, ei["dm_text"]] = gs.d_text[dr]
        XE[k, :, ei["dm_hpo"]] = gs.d_hpo[dr]
        XE[k, :, ei["dm_tissue"]] = gs.d_tissue[dr]
    G = genes
    XE[:, :, ei["ref_visible"]] = ref_vis[G]
    XE[:, :, ei["hpo_visible"]] = v6p.hpo_visible[G]
    XE[:, :, ei["lg_ref_degree"]] = np.log1p(v6p.ref_degree[G])
    XE[:, :, ei["gt_protein"]] = gs.protein[G]
    XE[:, :, ei["gt_ncrna"]] = gs.ncrna[G]
    XE[:, :, ei["has_seq"]] = gs.has_seq[G]
    XE[:, :, ei["has_go"]] = gs.has_go[G]
    XE[:, :, ei["has_expr"]] = gs.has_expr[G]
    XB[~valid] = 0.0
    XE[~valid] = 0.0
    a = (XE[:, :, ei["m_sem"]] > 0) & valid
    ph = pool_hash(ids, np.where(valid, genes, -1)) if not valid.all() else pool_hash(ids, genes)
    blk = RowBlock(ctx, role, list(ids), genes, XB, XE, a, cold_g[G] & valid, sem.src[G],
                   info={"pool_hash": ph}, valid=valid)
    if labels is not None:
        y = np.zeros((D, N), dtype=np.float32)
        nq = np.zeros(D)
        nqc = np.zeros(D)
        for k, d in enumerate(ids):
            q = labels.rows(d, role)
            nq[k] = len(q)
            nqc[k] = int(cold_g[q].sum()) if len(q) else 0
            y[k] = np.isin(genes[k], q) & valid[k]
        blk.y, blk.n_query, blk.n_query_cold = y, nq, nqc
    return blk


def assert_feature_whitelist(names: list[str]) -> None:
    bad = [n for n in names if n in LABEL_BURDEN]
    if bad:
        raise RuntimeError(f"label-burden columns in features: {bad}")

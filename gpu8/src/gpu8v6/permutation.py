"""V6 section 8 — ``permutation_suite``: three separate disease-permutation tests.

P1  embedding branch: only the neural disease representation is taken from the
    donor; candidates, retrieval scores and channels stay the target's.
    (Defined only for a scorer with a separate disease branch.)
P2  full scoring condition: every disease-dependent scoring input — all source
    scores, their ranks, the fused retrieval score and the disease descriptors —
    is recomputed from the donor on the *target's* candidate list; labels fixed.
P3  full pipeline: the candidate list is also the donor's; labels stay the
    target's.

Donors are other test diseases of the same fold, matched on routing class,
HPO-count bin and text-length bin.  Two pre-declared schemes: ``distant``
(different broad MeSH category) and ``close`` (same category, different
equivalence group).  Each scheme is drawn ``R`` times and averaged inside the
disease before anything is aggregated.  Besides the metric change the suite
records the within-list centred score change, the Spearman correlation and the
top-10 overlap, because a changed raw score need not change a ranking.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import paths6
from .common import World6, dump_json, load_manifest, now_utc, rng_for
from .retrieval import build_lists, max_normalise
from .scoring import FEATS, N_G, Block, evaluate, fusion_scores, load_block, recall_at
from .sources import SOURCES, SourceEngine

R_DRAWS = 5


def donor_table(w: World6, ids: list[str]) -> pd.DataFrame:
    av = pd.read_csv(paths6.AUDIT / "availability_per_disease.csv").set_index("disease_id")
    t = av.loc[[d for d in ids if d in av.index]].copy()
    t["hpo_bin"] = pd.cut(t["n_hpo_valid_v6"], [-1, 0, 5, 15, 40, 10 ** 6], labels=False)
    t["text_bin"] = pd.qcut(t["n_text_tokens"].rank(method="first"), 3, labels=False)
    eq = dict(zip(w.diseases["disease_id"], w.diseases["equivalence_group"]))
    t["eq"] = [eq.get(d, d) for d in t.index]
    return t


def draw_donors(t: pd.DataFrame, scheme: str, draw: int) -> dict[str, str]:
    out = {}
    keys = ["route_zero_support", "hpo_bin", "text_bin"]
    groups = t.groupby(keys)
    for d in t.index:
        row = t.loc[d]
        pool = groups.get_group(tuple(row[k] for k in keys))
        pool = pool[(pool.index != d) & (pool["eq"] != row["eq"])]
        if scheme == "distant":
            cand = pool[pool["category"] != row["category"]]
        else:
            cand = pool[pool["category"] == row["category"]]
        if cand.empty:                       # relax the text bin, then the HPO bin
            cand = t[(t.index != d) & (t["route_zero_support"] == row["route_zero_support"])
                     & ((t["category"] != row["category"]) if scheme == "distant"
                        else (t["category"] == row["category"]))]
            cand = cand[cand["eq"] != row["eq"]]
        if cand.empty:
            continue
        rng = rng_for("donor", scheme, d, draw)
        out[d] = cand.index[rng.integers(len(cand))]
    return out


class DonorFeatures:
    """Full-background source scores of the test diseases, for P2 and P3."""

    def __init__(self, eng: SourceEngine, ctx, ids: list[str], frozen_ret: dict):
        cs = eng.context(ctx)
        self.cs = cs
        prior = np.log1p(np.asarray(cs.A.sum(axis=0)).ravel()) + 0.5 * np.log1p(eng.ppi_deg)
        self.prior = (prior / max(prior.max(), 1e-12)).astype(np.float32)
        self.ids = ids
        self.pos = {d: i for i, d in enumerate(ids)}
        self.norm = {s: np.zeros((len(ids), N_G), dtype=np.float32) for s in SOURCES}
        self.fused = np.zeros((len(ids), N_G), dtype=np.float32)
        self.info = []
        for b0 in range(0, len(ids), 200):
            chunk = ids[b0:b0 + 200]
            raw, info = eng.scores(cs, chunk, sources=SOURCES)
            nm = {s: max_normalise(v) for s, v in raw.items()}
            _, _, _, fused = build_lists(nm, frozen_ret["weights"], self.prior, n=500,
                                         mode=frozen_ret["mode"])
            for s in SOURCES:
                self.norm[s][b0:b0 + len(chunk)] = nm[s]
            self.fused[b0:b0 + len(chunk)] = fused
            self.info.append(info)
        self.info = pd.concat(self.info, ignore_index=True).set_index("disease_id")

    def ranks(self, x_row: np.ndarray, genes: np.ndarray) -> np.ndarray:
        pos = x_row > 0
        order = np.lexsort((np.arange(len(x_row)), -x_row))
        rank = np.empty(len(x_row), dtype=np.int64)
        rank[order] = np.arange(1, len(x_row) + 1)
        return np.where(pos[genes], rank[genes], 0)

    def features(self, donor: str, genes: np.ndarray, N: int) -> np.ndarray:
        i = self.pos[donor]
        f = {}
        for s in SOURCES:
            x = self.norm[s][i]
            f[f"v_{s}"] = x[genes]
            r = self.ranks(x, genes)
            f[f"lr_{s}"] = np.where(r > 0, np.log(N_G / np.maximum(r, 1)), 0.0)
            f[f"a_{s}"] = np.full(len(genes), float(x.max() > 0))
        fu = self.fused[i]
        f["fused"] = fu[genes]
        rr = self.ranks(fu, genes)
        f["ret_rank"] = np.where(rr > 0, np.log(N / np.maximum(rr, 1)), np.log(N / N_G))
        f["is_fill"] = (fu[genes] <= 0).astype(float)
        f["prior"] = self.prior[genes]
        inf = self.info.loc[donor]
        f["d_hpo"] = np.full(len(genes), np.log1p(float(inf["hpo_terms"])))
        f["d_pheno_sim"] = np.full(len(genes), float(inf.get("pheno_max_sim", 0.0)))
        f["d_text_sim"] = np.full(len(genes), float(inf.get("text_max_sim", 0.0)))
        f["d_onto_ic"] = np.full(len(genes), float(inf.get("onto_max_ic", 0.0)) / 10.0)
        f["d_onto_inf"] = np.full(len(genes), float(inf.get("onto_level") == "informative"))
        return np.stack([np.asarray(f[k], dtype=np.float32) for k in FEATS], axis=1)


def permuted_block(orig: Block, donors: dict[str, str], dfeat: DonorFeatures, kind: str
                   ) -> tuple[Block, list[str]]:
    keep = [d for d in orig.ids if d in donors]
    b = orig.subset(keep)
    don = [donors[d] for d in keep]
    if kind == "P1":
        return Block(b.context_id, don, b.genes, b.X, b.y, b.n_query, b.dinfo), keep
    if kind == "P2":
        X = np.stack([dfeat.features(dn, b.genes[i], b.N) for i, dn in enumerate(don)], axis=0)
        return Block(b.context_id, don, b.genes, X, b.y, b.n_query, b.dinfo), keep
    raise ValueError(kind)


def p3_block(orig: Block, donors: dict[str, str], donor_block: Block) -> tuple[Block, list[str]]:
    """Donor's own list and features, evaluated against the target's labels."""
    keep = [d for d in orig.ids if d in donors and donors[d] in set(donor_block.ids)]
    ob = orig.subset(keep)
    db = donor_block.subset([donors[d] for d in keep])
    lab = pd.read_parquet(paths6.CACHES / "candidates" / f"{orig.context_id}_labels.parquet")
    qs = lab.groupby("disease_id")["gene_row"].apply(set).to_dict()
    y = np.stack([np.isin(db.genes[i], list(qs.get(d, set()))).astype(np.float32)
                  for i, d in enumerate(keep)], axis=0)
    return Block(db.context_id, db.ids, db.genes, db.X, y, ob.n_query, db.dinfo), keep


def compare(orig_scores, perm_scores, b_orig: Block, b_perm: Block, keep_ids) -> pd.DataFrame:
    r0 = recall_at(b_orig, orig_scores)
    r1 = recall_at(b_perm, perm_scores)
    rows = []
    same_list = np.array_equal(b_orig.genes, b_perm.genes)
    for i, d in enumerate(keep_ids):
        rec = {"disease_id": d, "recall10_orig": r0[i], "recall10_perm": r1[i],
               "delta": r1[i] - r0[i],
               "covered": float(b_orig.y[i].sum() > 0)}
        if same_list:
            a, c = orig_scores[i], perm_scores[i]
            rho = spearmanr(a, c).correlation
            rec["spearman"] = float(rho) if rho == rho else 1.0
            ca, cc = a - a.mean(), c - c.mean()
            rec["centred_abs_change"] = float(np.abs(cc - ca).mean())
            ta = set(np.argsort(-a)[:10])
            tc = set(np.argsort(-c)[:10])
            rec["top10_overlap"] = len(ta & tc) / 10.0
        rows.append(rec)
    return pd.DataFrame(rows)


def run_fold(w: World6, fold: int, scorers: dict, eng: SourceEngine, draws: int = R_DRAWS
             ) -> pd.DataFrame:
    """scorers: name -> (callable(Block)->scores, has_disease_branch)."""
    fa = w.fold(fold)
    ids = w.labelled(fa["test_diseases"])
    test = load_block(f"f{fold}_outer", ids=ids)
    frozen_ret = load_manifest("retriever_v6_frozen.json")
    dfeat = DonorFeatures(eng, w.outer_context(fold), test.ids, frozen_ret)
    t = donor_table(w, test.ids)
    out = []
    for name, (fn, has_branch) in scorers.items():
        s0 = fn(test)
        for scheme in ("distant", "close"):
            for draw in range(draws):
                donors = draw_donors(t, scheme, draw)
                kinds = (["P1"] if has_branch else []) + ["P2", "P3"]
                for kind in kinds:
                    if kind == "P3":
                        pb, keep = p3_block(test, donors, test)
                    else:
                        pb, keep = permuted_block(test, donors, dfeat, kind)
                    ob = test.subset(keep)
                    pos = {d: i for i, d in enumerate(test.ids)}
                    so = s0[[pos[d] for d in keep]]
                    cmp_ = compare(so, fn(pb), ob, pb, keep)
                    cmp_["scorer"], cmp_["test"], cmp_["scheme"] = name, kind, scheme
                    cmp_["draw"], cmp_["fold"] = draw, fold
                    cmp_["donor"] = [donors[d] for d in keep]
                    out.append(cmp_)
        print(f"  permutation fold {fold} scorer {name} done", flush=True)
    return pd.concat(out, ignore_index=True)


def summarise(df: pd.DataFrame, route: dict) -> pd.DataFrame:
    df = df.copy()
    df["route"] = df["disease_id"].map(route)
    g = df.groupby(["scorer", "test", "scheme", "disease_id"]).agg(
        delta=("delta", "mean"), orig=("recall10_orig", "mean"), perm=("recall10_perm", "mean"),
        covered=("covered", "max"), spearman=("spearman", "mean"),
        overlap=("top10_overlap", "mean"), route=("route", "first")).reset_index()
    rows = []
    for (sc, te, sch), sub in g.groupby(["scorer", "test", "scheme"]):
        for subset, m in (("all", np.ones(len(sub), bool)),
                          ("Z1_hpo", (sub["route"] == "Z1_hpo").to_numpy()),
                          ("covered", (sub["covered"] > 0).to_numpy())):
            s2 = sub[m]
            if s2.empty:
                continue
            rows.append({"scorer": sc, "test": te, "scheme": sch, "subset": subset,
                         "n_diseases": int(len(s2)), "recall10_orig": float(s2["orig"].mean()),
                         "recall10_perm": float(s2["perm"].mean()),
                         "delta_perm_minus_orig": float(s2["delta"].mean()),
                         "frac_worse": float((s2["delta"] < -1e-12).mean()),
                         "frac_better": float((s2["delta"] > 1e-12).mean()),
                         "mean_spearman": float(s2["spearman"].mean()) if s2["spearman"].notna().any() else np.nan,
                         "mean_top10_overlap": float(s2["overlap"].mean()) if s2["overlap"].notna().any() else np.nan})
    return pd.DataFrame(rows)

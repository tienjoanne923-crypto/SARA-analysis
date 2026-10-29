"""V8.1 §11.1 — two disjoint GO-mask gene sets, drawn without any disease label.

Eligible genes: legal production GO *and* a legal protein sequence.  Strata:
deciles of the number of direct GO terms x deciles of the physical PPI degree
(the V6 ``build_P`` degree).  Inside every stratum genes are ordered by a stable
hash of (protocol, gene id); the first 10 % (rounded) form G_mask,dev and the
next 20 % G_mask,test.  Nothing here reads a relation, a label or a result.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import PROTOCOL_ID, paths8
from .common8 import array_hash, dump_json, now_utc, write_manifest

DEV_SHARE, TEST_SHARE = 0.10, 0.20


def deciles(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="first").to_numpy()
    return np.minimum((10 * (r - 1) / len(x)).astype(int), 9)


def build() -> dict:
    from gpu8.contexts import build_P
    from gpu8v6.common import World6
    from gpu8v71.featurepack71 import FeaturePack
    from .evidence8 import GOPolicy
    w = World6(load_heavy=True)
    pol = GOPolicy(w, "current")
    fa = pol.function_arrays()
    pack = FeaturePack()
    seq = pack.a["g_sequence_mask"]
    go = fa["g_function_mask"]
    elig = np.flatnonzero(seq & go)
    _, deg = build_P(w.ppi, w.idx)
    nterm = fa["n_terms_per_gene"][elig]
    dg = np.asarray(deg)[elig]
    stratum = deciles(nterm) * 10 + deciles(dg)
    gid = [w.gene_id_of_row[int(g)] for g in elig]
    h = np.array([int(hashlib.sha256(f"{PROTOCOL_ID}|gmask|{x}".encode()).hexdigest()[:15], 16)
                  for x in gid])
    df = pd.DataFrame({"gene_row": elig, "gene_id": gid, "n_go_terms": nterm, "ppi_degree": dg,
                       "stratum": stratum, "h": h})
    out = []
    for s, sub in df.groupby("stratum"):
        sub = sub.sort_values("h")
        n = len(sub)
        nd = int(round(DEV_SHARE * n))
        nt = int(round(TEST_SHARE * n))
        lab = np.array([""] * n, dtype=object)
        lab[:nd] = "dev"
        lab[nd:nd + nt] = "test"
        out.append(sub.assign(mask_set=lab))
    df = pd.concat(out)
    df = df[df["mask_set"] != ""].drop(columns="h").sort_values("gene_row").reset_index(drop=True)
    df.to_parquet(paths8.TABLES / "go_masks_v8.parquet", index=False)
    dev = df.loc[df["mask_set"] == "dev", "gene_row"].to_numpy()
    test = df.loc[df["mask_set"] == "test", "gene_row"].to_numpy()
    assert not set(dev) & set(test)
    rep = {"generated_utc": now_utc(), "eligible_genes": int(len(elig)),
           "G_mask_dev": int(len(dev)), "G_mask_test": int(len(test)),
           "hash_dev": array_hash(np.sort(dev)), "hash_test": array_hash(np.sort(test)),
           "rule": "genes with production GO and sequence; strata = GO-term decile x physical PPI "
                   "degree decile; stable hash order inside strata; first 10% dev, next 20% test",
           "labels_read": False}
    write_manifest("go_masks_v8.json", rep)
    return rep


def load() -> dict[str, np.ndarray]:
    df = pd.read_parquet(paths8.TABLES / "go_masks_v8.parquet")
    return {k: np.sort(df.loc[df["mask_set"] == k, "gene_row"].to_numpy()) for k in ("dev", "test")}


def evaluation_counts() -> dict:
    """Evaluation-side description (after the sets are frozen): relations touching them."""
    from gpu8v6.common import World6
    w = World6(load_heavy=False)
    m = load()
    rel = w.primary_relations.assign(gene_row=w.primary_relations["gene_id"].map(w.idx.g_ix))
    out = {}
    for k, g in m.items():
        sel = rel["gene_row"].isin(set(g.tolist()))
        out[k] = {"genes": int(len(g)), "relations": int(sel.sum()),
                  "diseases": int(rel.loc[sel, "disease_id"].nunique())}
    return out


if __name__ == "__main__":
    paths8.ensure()
    print(json.dumps(build(), indent=2))
    print(json.dumps(evaluation_counts(), indent=2))

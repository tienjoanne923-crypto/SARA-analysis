"""V7.1 section 9.1 — the canonical feature pack and its coverage report.

All arrays are aligned on primary keys (never on raw file order): gene rows in
the fixed V5/V6 ``gene_row`` order, disease rows in the fixed ``disease_row``
order.  ``build`` checks shapes, masks, finiteness and non-constant
distributions, hashes every array and writes ``manifests_v71/featurepack_v71.json``;
``FeaturePack`` loads the arrays for the core (optionally onto a device).

Disease HPO aggregation (section 4.2):  x^H_d = sum_h a_dh e_h / (sum_h a_dh + eps)
over the disease's positive, valid, de-duplicated V6 terms; a_dh is the HPO
frequency (obligate 1, very frequent .895, frequent .545, occasional .17, very
rare .025, n/m, x%), missing frequency -> fixed weight 1.  No HPO ->
``d_hpo_mask = False``; no invented phenotype.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy import sparse

from gpu8 import paths as p5
from gpu8v6 import paths6

from . import paths71
from .common71 import array_hash, dump_json, now_utc, write_manifest
from .encoders71 import load_encoded

FREQ = {"HP:0040280": 1.0, "HP:0040281": 0.895, "HP:0040282": 0.545, "HP:0040283": 0.17,
        "HP:0040284": 0.025, "HP:0040285": 0.0}


def freq_weight(f) -> float:
    s = str(f or "").strip()
    if not s or s == "nan":
        return 1.0
    if s in FREQ:
        return FREQ[s]
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", s)
    if m and int(m.group(2)) > 0:
        return max(0.025, int(m.group(1)) / int(m.group(2)))
    m = re.match(r"^([\d.]+)\s*%$", s)
    if m:
        return max(0.025, float(m.group(1)) / 100.0)
    return 1.0


def aggregate_terms(term_vecs: np.ndarray, term_pos: dict, table: pd.DataFrame, n_rows: int,
                    row_of: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Weighted mean of term vectors per entity; the table must be filtered upstream."""
    t = table.copy()
    t["row"] = t["entity"].map(row_of)
    t["tp"] = t["term"].map(term_pos)
    miss_term = int(t["tp"].isna().sum())
    t = t[t["row"].notna() & t["tp"].notna()]
    t = t.groupby(["row", "tp"], as_index=False)["weight"].max()       # de-duplicate
    t = t[t["weight"] > 0]
    X = np.zeros((n_rows, term_vecs.shape[1]), dtype=np.float64)
    W = np.zeros(n_rows, dtype=np.float64)
    r = t["row"].astype(int).to_numpy()
    tp = t["tp"].astype(int).to_numpy()
    wt = t["weight"].to_numpy(dtype=np.float64)
    np.add.at(X, r, term_vecs[tp] * wt[:, None])
    np.add.at(W, r, wt)
    mask = W > 0
    X[mask] /= (W[mask, None] + 1e-12)
    return X.astype(np.float32), mask, {"pairs": int(len(t)), "terms_without_vector": miss_term}


def build(world=None) -> dict:
    from gpu8v6.common import World6
    w = world or World6(load_heavy=False)
    d_ix, g_ix = w.idx.d_ix, w.idx.g_ix
    D, G = w.n_d, w.n_g
    arrays: dict[str, np.ndarray] = {}

    # disease text
    Xd, kd = load_encoded("sapbert_disease")
    tpl = pd.read_parquet(paths71.TABLES / "disease_templates_v71.parquet")
    d_text = np.zeros((D, Xd.shape[1]), dtype=np.float32)
    m_text = np.zeros(D, dtype=bool)
    for i, (did, txt) in enumerate(zip(tpl["disease_id"], tpl["text"])):
        if did in d_ix and str(txt).strip():
            d_text[d_ix[did]] = Xd[i]
            m_text[d_ix[did]] = True
    arrays["d_text"], arrays["d_text_mask"] = d_text, m_text

    # disease HPO
    Xh, kh = load_encoded("sapbert_hpo")
    hids = pd.read_parquet(paths71.TABLES / "hpo_term_texts_v71.parquet")["term_id"].tolist()
    ph = pd.read_parquet(paths6.TABLES / "phenotypes_v6.parquet")
    ph = ph[(ph["keep_v6"] == 1) & (ph["negated"].astype(int) == 0)]
    tab = pd.DataFrame({"entity": ph["disease_id"], "term": ph["hpo_id"],
                        "weight": [freq_weight(f) for f in ph["frequency"]]})
    d_hpo, m_hpo, hinfo = aggregate_terms(Xh, {t: i for i, t in enumerate(hids)}, tab, D, d_ix)
    arrays["d_hpo"], arrays["d_hpo_mask"] = d_hpo, m_hpo

    # gene side
    Xs, ks = load_encoded("esm_gene")
    sq = pd.read_parquet(paths71.TABLES / "sequences_v71.parquet")
    if not (sq["gene_row"].to_numpy() == np.arange(G)).all():
        raise RuntimeError("sequence table is not in gene_row order")
    arrays["g_sequence"] = Xs.astype(np.float32)
    arrays["g_sequence_mask"] = (sq["length"].to_numpy() > 0)
    arrays["g_function"] = np.load(paths71.TABLES / "g_function.npy")
    arrays["g_function_mask"] = np.load(paths71.TABLES / "g_function_mask.npy")
    arrays["g_expression"] = np.load(paths71.TABLES / "g_expression.npy")
    arrays["g_expression_mask"] = np.load(paths71.TABLES / "g_expression_mask.npy")
    arrays["g_expression_gtex"] = np.load(paths71.TABLES / "g_expression_gtex.npy")
    arrays["g_expression_gtex_mask"] = np.load(paths71.TABLES / "g_expression_gtex_mask.npy")
    arrays["d_tissue"] = np.load(paths71.TABLES / "d_tissue.npy")
    arrays["d_tissue_mask"] = np.load(paths71.TABLES / "d_tissue_mask.npy")
    arrays["anchor_vectors"] = np.load(paths71.TABLES / "anchor_vectors.npy")
    A = sparse.load_npz(paths71.TABLES / "anchor_incidence.npz").tocsr()
    img = paths71.TABLES / "g_image.npy"
    if img.exists():
        arrays["g_image"] = np.load(img)
        arrays["g_image_mask"] = np.load(paths71.TABLES / "g_image_mask.npy")

    # ---- checks ---------------------------------------------------------
    expect = {"d_text": (D, 768), "d_text_mask": (D,), "d_hpo": (D, 768), "d_hpo_mask": (D,),
              "g_sequence": (G, 480), "g_sequence_mask": (G,), "g_function": (G, 768),
              "g_function_mask": (G,), "g_expression": (G, 51), "g_expression_mask": (G, 51),
              "g_expression_gtex": (G, 51), "g_expression_gtex_mask": (G, 51),
              "d_tissue": (D, 51), "d_tissue_mask": (D,), "g_image": (G, 1024), "g_image_mask": (G,)}
    checks = {}
    for k, a in arrays.items():
        if k in expect and tuple(a.shape) != expect[k]:
            raise RuntimeError(f"{k}: shape {a.shape} != {expect[k]}")
        if a.dtype != bool and not np.isfinite(a).all():
            raise RuntimeError(f"{k}: non-finite values")
    for base in ("d_text", "d_hpo", "g_sequence", "g_function", "g_image"):
        if base not in arrays:
            continue
        X, m = arrays[base], arrays[base + "_mask"]
        if (np.abs(X[~m]).sum() if (~m).any() else 0.0) != 0.0:
            raise RuntimeError(f"{base}: masked rows must be zero")
        sd = X[m].std(axis=0)
        checks[base] = {"rows_valid": int(m.sum()), "dims_nonconstant_share": float((sd > 1e-6).mean()),
                        "mean_row_norm": float(np.linalg.norm(X[m], axis=1).mean())}
        if checks[base]["dims_nonconstant_share"] < 0.9:
            raise RuntimeError(f"{base}: distribution (nearly) constant")
    o = arrays["g_expression_mask"]
    E = arrays["g_expression"]
    if (np.abs(E[~o]).sum() != 0.0):
        raise RuntimeError("g_expression must be zero where unobserved")
    checks["g_expression"] = {"genes_any_observed": int(o.any(axis=1).sum()),
                              "row_mean_abs_max": float(np.abs(np.where(o, E, 0).sum(1)
                                                                / np.maximum(o.sum(1), 1)).max())}
    q = arrays["d_tissue"]
    qm = arrays["d_tissue_mask"]
    if not np.allclose(q[qm].sum(axis=1), 1.0, atol=1e-5) or np.abs(q[~qm]).sum() != 0:
        raise RuntimeError("d_tissue rows must sum to 1 where valid and be zero elsewhere")

    # ---- coverage report ----------------------------------------------------
    lab_rows = np.array(sorted(d_ix[d] for d in w.positives if d in d_ix))
    gm = {"sequence": arrays["g_sequence_mask"], "function": arrays["g_function_mask"],
          "expression": o.any(axis=1)}
    if "g_image_mask" in arrays:
        gm["image"] = arrays["g_image_mask"]
    pos_rows = np.array(sorted({g_ix[g] for d in w.positives for g in w.positives[d] if g in g_ix}))
    cov = {"genes": {k: {"all": int(v.sum()), "share_all": float(v.mean()),
                         "share_positive_genes": float(v[pos_rows].mean())} for k, v in gm.items()},
           "diseases": {k: {"all": int(arrays[k + "_mask"].sum()),
                            "labelled": int(arrays[k + "_mask"][lab_rows].sum()),
                            "share_labelled": float(arrays[k + "_mask"][lab_rows].mean())}
                        for k in ("d_text", "d_hpo", "d_tissue")},
           "anchors": {"n": int(arrays["anchor_vectors"].shape[0]),
                       "genes_with_anchor": int((np.diff(A.indptr) > 0).sum())}}
    pat = pd.Series(["".join(c for c, v in zip("SFE", (gm["sequence"][i], gm["function"][i],
                                                        gm["expression"][i])) if v) or "none"
                     for i in range(G)]).value_counts()
    cov["gene_modality_patterns"] = pat.to_dict()
    lab_pat = pd.Series(["".join(c for c, v in zip("THQ", (arrays["d_text_mask"][r], arrays["d_hpo_mask"][r],
                                                            arrays["d_tissue_mask"][r])) if v) or "none"
                         for r in lab_rows]).value_counts()
    cov["labelled_disease_patterns"] = lab_pat.to_dict()

    hashes = {k: array_hash(v) for k, v in arrays.items()}
    hashes["anchor_incidence"] = array_hash(A.indptr, A.indices)
    np.savez(paths71.CACHES / "featurepack_v71.npz", **arrays)
    sparse.save_npz(paths71.CACHES / "featurepack_v71_anchors.npz", A)
    rep = {"generated_utc": now_utc(), "shapes": {k: list(v.shape) for k, v in arrays.items()},
           "hashes": hashes, "checks": checks, "coverage": cov, "hpo_aggregation": hinfo,
           "encoder_keys": {"disease": {k: kd.get(k) for k in ("model", "revision", "pooling")},
                            "hpo": {k: kh.get(k) for k in ("model", "revision", "pooling")},
                            "sequence": {k: ks.get(k) for k in ("model", "revision", "pooling",
                                                                "window", "overlap")}},
           "disease_order": "World6.idx.d_ix (disease_row)", "gene_order": "World6.idx.g_ix (gene_row)",
           "provenance_reviewed": False}
    write_manifest("featurepack_v71.json", rep)
    dump_json(paths71.AUDIT / "featurepack_coverage_v71.json", cov)
    return rep


class FeaturePack:
    """In-memory feature pack; ``torch`` tensors on demand."""

    def __init__(self, path=None):
        p = path or (paths71.CACHES / "featurepack_v71.npz")
        z = np.load(p)
        self.a = {k: z[k] for k in z.files}
        self.A = sparse.load_npz(paths71.CACHES / "featurepack_v71_anchors.npz").tocsr()
        self.n_d = self.a["d_text"].shape[0]
        self.n_g = self.a["g_sequence"].shape[0]

    def has(self, k: str) -> bool:
        return k in self.a

    def tensors(self, device: str = "cuda", expression: str = "hpa", image: bool = False) -> dict:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)
        e_key = "g_expression" if expression == "hpa" else "g_expression_gtex"
        out = {"xT": t(self.a["d_text"]), "mT": t(self.a["d_text_mask"], torch.bool),
               "xH": t(self.a["d_hpo"]), "mH": t(self.a["d_hpo_mask"], torch.bool),
               "q": t(self.a["d_tissue"]), "mq": t(self.a["d_tissue_mask"], torch.bool),
               "xS": t(self.a["g_sequence"]), "mS": t(self.a["g_sequence_mask"], torch.bool),
               "xF": t(self.a["g_function"]), "mF": t(self.a["g_function_mask"], torch.bool),
               "e": t(self.a[e_key]), "o": t(self.a[e_key + "_mask"]),
               "xA": t(self.a["anchor_vectors"])}
        if image and "g_image" in self.a:
            out["xI"] = t(self.a["g_image"])
            out["mI"] = t(self.a["g_image_mask"], torch.bool)
        return out


def main() -> int:
    paths71.ensure()
    rep = build()
    print(json.dumps({k: rep[k] for k in ("shapes", "checks", "coverage", "hpo_aggregation")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

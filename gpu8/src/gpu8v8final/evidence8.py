"""V8.1 §4 — GO evidence policy: parsing, production rule, variants, deletion layers.

* ``read_gaf17`` keeps all 17 GAF 2.2 columns (qualifier/NOT, references, evidence
  code, With/From, date, assigned_by, extension, product form) plus the mapped
  project gene; nothing is collapsed to "gene - term" before the audit.
* production rule (= V7.1): NOT and ND removed, entity mapped (UniProt, then
  symbol), term present and non-obsolete in the ontology, root terms removed,
  term text encoded.  No disease column, answer set or PMID blacklist is read.
* evidence variants (§4.3, GO Consortium definitions): full, experimental,
  no TAS/NAS (author statements), no IC (curator inference, separate class),
  no IMP/IGI/HMP/HGI (mutant phenotype / genetic interaction), IEA only
  (electronic sensitivity, never called "no GO").
* deletion layers (§4.2): (1) training-visible source exclusion — GO records
  citing a PMID of the context's *training* relations; (2) evaluation-
  conditioned deletion — records citing any CTD relation PMID (flag
  ``evaluation_conditioned_audit = true``; never a production input generator);
  (3) pre-registered evidence blocks — connected components of the PMID-relation
  graph, the giant component reported and never split.  Every deletion has a
  per-gene count-matched random control (same flag as its layer).  A record with
  several references is deleted whole when any of them hits.
* five denominators for every deletion: citations, annotation rows, unique GO
  terms (gene-term pairs and term ids), affected genes, affected query relations.
* annotations without a PMID are counted separately; unknown provenance is not
  treated as independent.

``function_arrays`` rebuilds g_function and the functional anchors from any
record subset (anchors come from the propagated GO, so every GO change rebuilds
them), identical to ``gpu8v71.function71.build`` on the production records.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
from scipy import sparse

from gpu8v6 import paths6
from gpu8v6.gene_features import go_closure, parse_go_obo, symbol_to_gene

from . import paths8
from .common8 import array_hash, dump_json, list_hash, now_utc, rng_for

GAF_COLS = ["db", "db_object_id", "db_object_symbol", "qualifier", "go_id", "db_reference",
            "evidence_code", "with_from", "aspect", "db_object_name", "db_object_synonym",
            "db_object_type", "taxon", "date", "assigned_by", "annotation_extension",
            "gene_product_form_id"]
ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}
ANCHOR_MIN, ANCHOR_MAX = 5, 500
EXPERIMENTAL = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"}
AUTHOR = {"TAS", "NAS"}
CURATOR = {"IC"}
MUTANT_GENETIC = {"IMP", "IGI", "HMP", "HGI"}
ELECTRONIC = {"IEA"}
VARIANTS = {
    "full": lambda ev: np.ones(len(ev), dtype=bool),
    "experimental": lambda ev: ev.isin(EXPERIMENTAL).to_numpy(),
    "no_tas_nas": lambda ev: ~ev.isin(AUTHOR).to_numpy(),
    "no_ic": lambda ev: ~ev.isin(CURATOR).to_numpy(),
    "no_mutant_genetic": lambda ev: ~ev.isin(MUTANT_GENETIC).to_numpy(),
    "iea_only": lambda ev: ev.isin(ELECTRONIC).to_numpy(),
}


def read_gaf17(path, sym2gene: dict) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh):
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 15:
                continue
            f = (f + [""] * 17)[:17]
            rows.append([ln] + f)
    df = pd.DataFrame(rows, columns=["line"] + GAF_COLS)
    df["gene_id"] = [sym2gene.get("UNI:" + u) or sym2gene.get("SYM:" + s)
                     for u, s in zip(df["db_object_id"], df["db_object_symbol"])]
    df["pmids"] = [tuple(x.split(":", 1)[1] for x in str(r).split("|") if x.startswith("PMID:"))
                   for r in df["db_reference"]]
    df["is_not"] = ["NOT" in str(q).split("|") for q in df["qualifier"]]
    return df


class GOPolicy:
    """Production records + variants + deletion layers for one GO release."""

    def __init__(self, world, release: str = "current"):
        self.w = world
        from gpu8v71 import paths71
        from gpu8v71.encoders71 import load_encoded
        if release == "current":
            obo, gaf, hgnc = (paths6.ext("go", "go-basic_current.obo"),
                              paths6.ext("go", "goa_human_current.gaf.gz"), None)
            self.go_x, _ = load_encoded("sapbert_go")
            ids = pd.read_parquet(paths71.TABLES / "go_term_texts_v71.parquet")["term_id"].tolist()
        else:
            obo, gaf = (paths6.ext("go", "go-basic_2023-10-09.obo"),
                        paths6.ext("go", "goa_human_2023-10-09.gaf.gz"))
            hgnc = paths6.ext("hgnc", "hgnc_complete_set_2023-10-01.txt")
            X = np.load(paths8.CACHES / "encoders" / "sapbert_go_2023.npy")
            self.go_x = X
            ids = pd.read_parquet(paths8.TABLES / "go_term_texts_2023.parquet")["term_id"].tolist()
        self.release = release
        self.go_pos = {t: i for i, t in enumerate(ids)}
        self.parents, self.ns = parse_go_obo(obo)
        self.anc = go_closure(self.parents)
        sym2gene = symbol_to_gene(world, hgnc)
        raw = read_gaf17(gaf, sym2gene)
        self.raw_counts = {"rows_read": int(len(raw)), "not_qualifier": int(raw["is_not"].sum()),
                           "nd": int((raw["evidence_code"] == "ND").sum()),
                           "unmapped_entity": int(raw["gene_id"].isna().sum())}
        keep = (~raw["is_not"]) & (raw["evidence_code"] != "ND") & raw["gene_id"].notna()
        keep &= raw["go_id"].isin(self.parents) & ~raw["go_id"].isin(ROOTS)
        rec = raw[keep].copy()
        rec["gene_row"] = rec["gene_id"].map(world.idx.g_ix)
        rec = rec[rec["gene_row"].notna()]
        rec["gene_row"] = rec["gene_row"].astype(int)
        rec = rec[rec["go_id"].isin(self.go_pos)]
        self.records = rec.reset_index(drop=True)
        self.records["rid"] = np.arange(len(self.records))
        self.raw_counts["production_records"] = int(len(self.records))

    # ------------------------------------------------------------- variants
    def variant_mask(self, name: str) -> np.ndarray:
        return VARIANTS[name](self.records["evidence_code"])

    # ------------------------------------------------------------- layers
    def pmid_hit_mask(self, pmids: set) -> np.ndarray:
        return np.array([any(p in pmids for p in ps) for ps in self.records["pmids"]], dtype=bool)

    def training_visible_exclusion(self, ctx_relation_ids: list[str]) -> np.ndarray:
        """Layer 1: records citing a PMID of the context's training relations (delete mask)."""
        pm = set()
        for r in ctx_relation_ids:
            pm |= self.w.relation_pmids.get(r, set())
        return self.pmid_hit_mask(pm)

    def evaluation_conditioned(self) -> np.ndarray:
        """Layer 2 (audit only): records citing any CTD relation PMID (incl. evaluation)."""
        pm = set()
        for s in self.w.relation_pmids.values():
            pm |= s
        return self.pmid_hit_mask(pm)

    def matched_random(self, delete: np.ndarray, tag: str) -> np.ndarray:
        """Per-gene count-matched random deletion (same number of records per gene)."""
        rng = rng_for("go_matched_random", self.release, tag)
        out = np.zeros(len(self.records), dtype=bool)
        r = self.records
        k = pd.Series(delete).groupby(r["gene_row"].to_numpy()).sum()
        byg = r.groupby("gene_row").indices
        for g, n in k.items():
            n = int(n)
            if n <= 0:
                continue
            idx = byg[g]
            out[rng.choice(idx, size=min(n, len(idx)), replace=False)] = True
        return out

    # ------------------------------------------------------------- arrays
    def function_arrays(self, keep: np.ndarray | None = None, drop_genes=None) -> dict:
        """g_function, mask, anchors from the kept records (drop_genes lose all GO)."""
        r = self.records if keep is None else self.records[keep]
        if drop_genes is not None and len(drop_genes):
            r = r[~r["gene_row"].isin(set(int(x) for x in drop_genes))]
        n_g = self.w.n_g
        direct = r.drop_duplicates(["gene_row", "go_id"])[["gene_row", "go_id"]]
        G = np.zeros((n_g, self.go_x.shape[1]), dtype=np.float64)
        cnt = np.zeros(n_g, dtype=np.int64)
        tp = direct["go_id"].map(self.go_pos).to_numpy()
        gr = direct["gene_row"].to_numpy()
        np.add.at(G, gr, self.go_x[tp].astype(np.float64))
        np.add.at(cnt, gr, 1)
        has = cnt > 0
        G[has] /= cnt[has, None]
        pairs = set()
        for g, t in zip(gr, direct["go_id"]):
            for a in self.anc.get(t, {t}):
                if a not in ROOTS:
                    pairs.add((int(g), a))
        terms = sorted({t for _, t in pairs})
        t_ix = {t: i for i, t in enumerate(terms)}
        gi = np.fromiter((g for g, _ in pairs), dtype=np.int64, count=len(pairs))
        ti = np.fromiter((t_ix[t] for _, t in pairs), dtype=np.int64, count=len(pairs))
        M = sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), (gi, ti)), shape=(n_g, len(terms)))
        size = np.asarray(M.sum(axis=0)).ravel()
        keep_t = [i for i, t in enumerate(terms) if ANCHOR_MIN <= size[i] <= ANCHOR_MAX and t in self.go_pos]
        anchors = [terms[i] for i in keep_t]
        A_plus = M[:, keep_t].tocsr()
        A_plus.data[:] = 1.0
        anchor_x = np.stack([self.go_x[self.go_pos[t]] for t in anchors]).astype(np.float32)
        return {"g_function": G.astype(np.float32), "g_function_mask": has, "A_plus": A_plus,
                "anchor_vectors": anchor_x, "anchor_terms": anchors, "n_terms_per_gene": cnt}

    # ------------------------------------------------------------- audit
    def denominators(self, delete: np.ndarray, base_keep: np.ndarray | None = None,
                     query_pairs: pd.DataFrame | None = None, cold_rows: np.ndarray | None = None) -> dict:
        """Five denominators of a deletion relative to ``base_keep`` (default: all production)."""
        r = self.records
        base = np.ones(len(r), dtype=bool) if base_keep is None else base_keep
        dl = delete & base
        kept = base & ~delete
        pm_all = {p for ps in r.loc[base, "pmids"] for p in ps}
        pm_del = {p for ps in r.loc[dl, "pmids"] for p in ps}
        pm_kept = {p for ps in r.loc[kept, "pmids"] for p in ps}
        pairs_b = set(zip(r.loc[base, "gene_row"], r.loc[base, "go_id"]))
        pairs_k = set(zip(r.loc[kept, "gene_row"], r.loc[kept, "go_id"]))
        terms_b = set(r.loc[base, "go_id"])
        terms_k = set(r.loc[kept, "go_id"])
        genes_b = set(r.loc[base, "gene_row"])
        genes_k = set(r.loc[kept, "gene_row"])
        lost_pairs = pairs_b - pairs_k
        affected = {g for g, _ in lost_pairs}
        lost_all = genes_b - genes_k
        out = {"citations": {"deleted": len(pm_del - pm_kept), "cited_by_deleted_records": len(pm_del),
                             "total": len(pm_all)},
               "annotation_rows": {"deleted": int(dl.sum()), "total": int(base.sum())},
               "unique_go": {"gene_term_pairs_lost": len(lost_pairs), "gene_term_pairs_total": len(pairs_b),
                             "term_ids_lost": len(terms_b - terms_k), "term_ids_total": len(terms_b)},
               "genes": {"affected": len(affected), "lost_all_go": len(lost_all), "total_with_go": len(genes_b)}}
        if query_pairs is not None:
            g = query_pairs["gene_row"].to_numpy()
            aff = np.isin(g, list(affected))
            lost = np.isin(g, list(lost_all))
            q = {"total": int(len(g)), "gene_affected": int(aff.sum()), "gene_lost_all_go": int(lost.sum())}
            if cold_rows is not None:
                c = np.isin(g, cold_rows)
                q.update({"cold_total": int(c.sum()), "cold_gene_affected": int((aff & c).sum()),
                          "cold_gene_lost_all_go": int((lost & c).sum())})
            out["query_relations"] = q
        for k, v in out.items():
            if "deleted" in v and v.get("total"):
                v["share"] = v["deleted"] / v["total"]
        return out

    def no_pmid_report(self) -> dict:
        r = self.records
        nop = np.array([len(p) == 0 for p in r["pmids"]])
        return {"records_without_pmid": int(nop.sum()), "share": float(nop.mean()),
                "by_evidence": r.loc[nop, "evidence_code"].value_counts().head(15).to_dict(),
                "genes_only_without_pmid": int((pd.Series(nop).groupby(r["gene_row"].to_numpy()).all()).sum()),
                "rule": "unknown provenance is kept in production but never counted as independent evidence"}


# ---------------------------------------------------------------------------
# evidence blocks (layer 3)
# ---------------------------------------------------------------------------
def evidence_blocks(w, seed_tag: str = "evidence_blocks_v81", target_share: float = 0.10) -> dict:
    """Connected components of the PMID-relation graph over primary relations.

    Blocks are drawn at random (fixed seed) among the non-giant components until
    ``target_share`` of the primary relations is covered; the giant component is
    reported and never split or drawn.
    """
    rel = w.primary_relations[["relation_id", "disease_id", "gene_id"]].reset_index(drop=True)
    rid = {r: i for i, r in enumerate(rel["relation_id"])}
    pm_nodes: dict[str, int] = {}
    edges_r, edges_p = [], []
    for r, pms in w.relation_pmids.items():
        if r not in rid:
            continue
        for p in pms:
            j = pm_nodes.setdefault(p, len(pm_nodes))
            edges_r.append(rid[r])
            edges_p.append(j)
    nR, nP = len(rel), len(pm_nodes)
    from scipy.sparse.csgraph import connected_components
    rows = np.array(edges_r, dtype=np.int64)
    cols = np.array(edges_p, dtype=np.int64) + nR
    M = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nR + nP, nR + nP))
    n_comp, lab = connected_components(M, directed=False)
    comp_rel = lab[:nR]
    sizes = pd.Series(comp_rel).value_counts()
    giant = int(sizes.index[0])
    no_pmid = np.array([rid_ not in w.relation_pmids for rid_ in rel["relation_id"]])
    comps = [c for c in sizes.index if c != giant]
    rng = rng_for(seed_tag)
    order = rng.permutation(len(comps))
    chosen, covered = [], 0
    target = int(round(target_share * nR))
    for k in order:
        c = comps[k]
        if covered >= target:
            break
        chosen.append(int(c))
        covered += int(sizes[c])
    held = rel[np.isin(comp_rel, chosen)].copy()
    held_pm = {p for r in held["relation_id"] for p in w.relation_pmids.get(r, set())}
    rep = {"generated_utc": now_utc(), "n_relations": nR, "n_pmids": nP, "n_components": int(n_comp),
           "giant_component_relations": int(sizes.iloc[0]),
           "giant_share": float(sizes.iloc[0] / nR), "relations_without_pmid": int(no_pmid.sum()),
           "component_size_quantiles": sizes.quantile([0.5, 0.9, 0.99]).to_dict(),
           "target_share": target_share, "blocks_chosen": len(chosen),
           "relations_held": int(len(held)), "pmids_held": len(held_pm),
           "diseases_touched": int(held["disease_id"].nunique()),
           "rule": "PMID-relation connected components; giant component reported, never split or drawn; "
                   "blocks drawn uniformly at random with a fixed seed until the target share"}
    return {"report": rep, "held_relations": held, "held_pmids": held_pm,
            "component_of_relation": dict(zip(rel["relation_id"], comp_rel.tolist())), "giant": giant}


def block_sets(kind: str = "blocks") -> dict:
    """Relations and disease-HPO rows withdrawn in the evidence-block arm or its matched random control.

    blocks:        the held relations; HPO annotation rows citing a held PMID.
    blocks_random: per disease, the same number of randomly chosen primary relations and
                   the same number of randomly chosen HPO annotation rows (fixed seed).
    """
    import pandas as pd
    from gpu8v6.common import World6
    held = pd.read_parquet(paths8.TABLES / "evidence_block_relations_v8.parquet")
    pm = set(json.loads((paths8.TABLES / "evidence_block_pmids_v8.json").read_text(encoding="utf-8")))
    ph = pd.read_parquet(paths6.TABLES / "phenotypes_v6.parquet")
    src = ph["source"].astype(str)
    hit = (src.str.startswith("PMID:") & src.str[5:].isin(pm)).to_numpy()
    if kind == "blocks":
        rel = set(zip(held["disease_id"], held["gene_id"].astype(str)))
        return {"relations": rel, "hpo_drop_rows": np.flatnonzero(hit), "n_relations": len(rel),
                "n_hpo_rows": int(hit.sum())}
    w = World6(load_heavy=False)
    rng = rng_for("blocks_random_relations")
    pr = w.primary_relations
    rel = set()
    for d, k in held.groupby("disease_id").size().items():
        cand = pr.loc[pr["disease_id"] == d, "gene_id"].astype(str).unique()
        pick = rng.choice(cand, size=min(k, len(cand)), replace=False)
        rel |= {(d, g) for g in pick}
    rng2 = rng_for("blocks_random_hpo")
    drop = []
    k_by_d = pd.Series(hit).groupby(ph["disease_id"].to_numpy()).sum()
    by_d = ph.groupby("disease_id").indices
    for d, k in k_by_d.items():
        k = int(k)
        if k:
            idx = by_d[d]
            drop.extend(rng2.choice(idx, size=min(k, len(idx)), replace=False).tolist())
    return {"relations": rel, "hpo_drop_rows": np.array(sorted(drop), dtype=np.int64), "n_relations": len(rel),
            "n_hpo_rows": len(drop)}


def audit(write: bool = True) -> dict:
    """Stage-1 audit: records, variants, layers, denominators, no-PMID and blocks."""
    from gpu8v6.common import World6
    w = World6(load_heavy=False)
    pol = GOPolicy(w, "current")
    from gpu8v71 import paths71
    fa = pol.function_arrays()
    ref = np.load(paths71.TABLES / "g_function.npy")
    refm = np.load(paths71.TABLES / "g_function_mask.npy")
    A71 = sparse.load_npz(paths71.TABLES / "anchor_incidence.npz")
    same = {"g_function_equal": bool(np.array_equal(fa["g_function"], ref)),
            "g_function_maxabs": float(np.abs(fa["g_function"] - ref).max()),
            "mask_equal": bool(np.array_equal(fa["g_function_mask"], refm)),
            "anchors_equal": bool(A71.shape == fa["A_plus"].shape and (A71 != fa["A_plus"]).nnz == 0)}
    qp = w.primary_relations.assign(gene_row=w.primary_relations["gene_id"].map(w.idx.g_ix))
    qp = qp[qp["gene_row"].notna()].astype({"gene_row": int})
    rep = {"generated_utc": now_utc(), "release": "current", "counts": pol.raw_counts,
           "production_equals_v71": same, "no_pmid": pol.no_pmid_report(),
           "evidence_code_counts": pol.records["evidence_code"].value_counts().to_dict(), "variants": {}}
    for v in VARIANTS:
        keep = pol.variant_mask(v)
        rep["variants"][v] = pol.denominators(~keep, query_pairs=qp)
    ev = pol.evaluation_conditioned()
    rep["layer2_evaluation_conditioned"] = {**pol.denominators(ev, query_pairs=qp),
                                            "evaluation_conditioned_audit": True}
    rnd = pol.matched_random(ev, "layer2")
    rep["layer2_matched_random"] = {**pol.denominators(rnd, query_pairs=qp),
                                    "evaluation_conditioned_audit": True}
    # layer 1 for every outer context (inner contexts are computed where used)
    lay1 = {}
    for j in range(5):
        ctx = w.contexts[f"f{j}_outer"]
        rel = w.primary_relations[w.primary_relations["disease_id"].isin(set(ctx.reference))]
        d1 = pol.training_visible_exclusion(rel["relation_id"].tolist())
        lay1[f"f{j}_outer"] = pol.denominators(d1, query_pairs=qp)
    rep["layer1_training_visible_outer"] = lay1
    blocks = evidence_blocks(w)
    bd = pol.pmid_hit_mask(blocks["held_pmids"])
    rep["layer3_evidence_blocks"] = {**blocks["report"], "go_denominators": pol.denominators(bd, query_pairs=qp)}
    if write:
        dump_json(paths8.AUDIT / "evidence_policy_audit_v8.json", rep)
        blocks["held_relations"].to_parquet(paths8.TABLES / "evidence_block_relations_v8.parquet", index=False)
        dump_json(paths8.TABLES / "evidence_block_pmids_v8.json", sorted(blocks["held_pmids"]))
        pol.records.drop(columns=["pmids"]).assign(
            pmids="|".join([]) if False else pol.records["pmids"].map("|".join)).to_parquet(
            paths8.TABLES / "go_records_current_v8.parquet", index=False)
    return rep


if __name__ == "__main__":
    paths8.ensure()
    r = audit()
    print(json.dumps({k: r[k] for k in ("counts", "production_equals_v71", "no_pmid")}, indent=2, default=str))
    print(json.dumps(r["layer2_evaluation_conditioned"], indent=1, default=str)[:1500])
    print(json.dumps(r["layer3_evidence_blocks"], indent=1, default=str)[:2500])

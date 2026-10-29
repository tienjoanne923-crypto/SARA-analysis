"""V7.1 section 8 — IMPC experimental phenotypes as a cross-species channel (E8).

Inputs (all logged in resources_v71.json):

* IMPC ``genotype-phenotype-assertions-ALL.csv.gz``: significant MP calls after
  gene perturbation (marker, zygosity, sex, parameter, MP term);
* MGI ``HOM_MouseHumanSequence.rpt``: homology classes; only classes with exactly
  one human and one mouse gene are used (1:1);
* MGI curated MP->HP SSSOM (mapping-commons ``mp_hp_mgi_all``): each row keeps
  its predicate (exact / close / narrow / broad), confidence and justification.

``hp_mp_bridge`` (tables_v71/hp_mp_bridge_v71.parquet) records HP, MP, relation,
justification, source version, whether the row is derived (MP ancestor
propagation) and the similarity range.  A mouse model's phenotype set is
translated into HP terms through the bridge (direct mappings of each called MP
term, and of its MP ancestors when the term itself is unmapped, flagged as
derived).  Directional match for disease d and model m:

    S(d, m) = sum_h a_dh max_{h' in M'_m} sim(h, h') / sum_h a_dh,
    sim(h, h') = IC(MICA(h, h')) / IC(h)      in [0, 1],

IC from the (label-free) disease-annotation corpus of the ancestor-closed V6
HPO terms.  Untested genes, genes without a 1:1 orthologue and diseases without
HPO are *missing* (mask False), never 0; IMPC's assertion file has no
"tested normal" calls, so "not significant" cannot be separated from "not
tested" and is also treated as missing.

Admission (programmatic, before any use): ontology ids valid, >= 50 % of IMPC
calls mapped, and on the development families the S of known disease-gene
pairs exceeds that of other scored pairs (AUC > 0.55).  A random sample of the
bridge is exported for the manual spot check, which is *not* performed here.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
from scipy import sparse

from gpu8 import paths as p5
from gpu8v6 import paths6
from gpu8v6.gene_hpo import HPOSpace

from . import paths71
from .common71 import dump_json, now_utc, rng_for, write_manifest

SIM_FILE = paths71.CACHES / "mouse_score_v71.npz"


def parse_obo_parents(path) -> tuple[dict, set]:
    parents, obsolete, cur = {}, set(), None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = {"id": "", "p": [], "obs": False}
                continue
            if line.startswith("[") and line.endswith("]"):
                cur = None
                continue
            if cur is None:
                continue
            if line.startswith("id: "):
                cur["id"] = line[4:].strip()
                parents[cur["id"]] = cur["p"]
            elif line.startswith("is_a: "):
                cur["p"].append(line[6:].split("!")[0].strip())
            elif line.startswith("is_obsolete: true"):
                obsolete.add(cur["id"])
    return parents, obsolete


def closure(parents: dict) -> dict:
    anc: dict[str, set] = {}

    def get(t, depth=0):
        if t in anc:
            return anc[t]
        acc = {t}
        for p in parents.get(t, []):
            if p in parents and depth < 200:
                acc |= get(p, depth + 1)
        anc[t] = acc
        return acc

    import sys
    sys.setrecursionlimit(100000)
    for t in list(parents):
        get(t)
    return anc


def build(world=None) -> dict:
    from gpu8v6.common import World6
    w = world or World6(load_heavy=False)
    # ---- IMPC calls -----------------------------------------------------------
    impc = pd.read_csv(paths71.ext("impc", "genotype-phenotype-assertions-ALL.csv.gz"), dtype=str,
                       usecols=["marker_accession_id", "marker_symbol", "zygosity", "sex",
                                "parameter_stable_id", "mp_term_id"])
    impc = impc[impc["mp_term_id"].fillna("").str.startswith("MP:")]
    # ---- homology (1:1 only) ------------------------------------------------------
    hom = pd.read_csv(paths71.ext("mgi", "HOM_MouseHumanSequence.rpt"), sep="\t", dtype=str)
    hu = hom[hom["NCBI Taxon ID"] == "9606"]
    mo = hom[hom["NCBI Taxon ID"] == "10090"]
    nh = hu.groupby("DB Class Key").size()
    nm = mo.groupby("DB Class Key").size()
    one = set(nh[nh == 1].index) & set(nm[nm == 1].index)
    h1 = hu[hu["DB Class Key"].isin(one)].set_index("DB Class Key")["EntrezGene ID"]
    m1 = mo[mo["DB Class Key"].isin(one)].set_index("DB Class Key")["Mouse MGI ID"]
    mgi2human = {m1[k]: h1[k] for k in one if k in m1.index and k in h1.index}
    g_ix = w.idx.g_ix
    # ---- ontologies and bridge -----------------------------------------------------
    mp_par, mp_obs = parse_obo_parents(paths71.ext("ontology", "mp.obo"))
    mp_anc = closure(mp_par)
    space = HPOSpace(w.hpo_terms)
    hp_ok = set(w.hpo_terms["hpo_id"])
    ss = pd.read_csv(paths71.ext("mapping", "mp_hp_mgi_all.sssom.tsv"), sep="\t", comment="#", dtype=str)
    ss = ss.rename(columns={"object_id": "hp", "subject_id": "mp", "predicate_id": "relation"})
    ss["valid_ids"] = ss["hp"].isin(hp_ok) & ss["mp"].isin(mp_par) & ~ss["mp"].isin(mp_obs)
    bridge = ss[ss["valid_ids"] & ss["hp"].isin(space.t_ix)].copy()
    bridge["derived"] = False
    mp2hp: dict[str, set] = {}
    for m_, h_ in zip(bridge["mp"], bridge["hp"]):
        mp2hp.setdefault(m_, set()).add(h_)
    # ---- mouse model -> HP terms ---------------------------------------------------
    impc["human"] = impc["marker_accession_id"].map(mgi2human)
    impc["gene_row"] = impc["human"].map(g_ix)
    calls = impc[impc["gene_row"].notna()]
    n_calls = len(impc)
    mapped_direct = impc["mp_term_id"].isin(mp2hp)
    derived_rows = []
    model_hp: dict[int, set] = {}
    n_mapped_any = 0
    for g, mp in zip(calls["gene_row"].astype(int), calls["mp_term_id"]):
        hs = mp2hp.get(mp)
        derived = False
        if not hs:
            # nearest mapped MP ancestors (derived, flagged)
            anc = sorted(mp_anc.get(mp, {mp}) - {mp})
            hs = set()
            for a in anc:
                hs |= mp2hp.get(a, set())
            derived = bool(hs)
            if derived:
                derived_rows.append((mp, "|".join(sorted(hs))))
        if hs:
            n_mapped_any += 1
            model_hp.setdefault(g, set()).update(hs)
    # ---- IC over the disease-annotation corpus (label-free) ------------------------
    ph = pd.read_parquet(paths6.TABLES / "phenotypes_v6.parquet")
    r = ph["disease_id"].map(w.idx.d_ix)
    c = ph["hpo_id"].map(space.t_ix)
    k = r.notna() & c.notna()
    Q = sparse.csr_matrix((np.ones(int(k.sum()), np.float32), (r[k].astype(int), c[k].astype(int))),
                          shape=(w.n_d, space.n_t))
    Q.data[:] = 1.0
    Qc = space.closure(Q)
    n_ann = max(1, int((np.diff(Qc.indptr) > 0).sum()))
    cnt = np.asarray(Qc.sum(axis=0)).ravel()
    ic = -np.log((cnt + 1.0) / (n_ann + 1.0))
    T = space.T.tocsr()                                  # term x ancestor-or-self
    # ---- model closure matrix over HP terms ---------------------------------------
    genes_m = np.array(sorted(model_hp), dtype=np.int64)
    Mi, Mj = [], []
    for i, g in enumerate(genes_m):
        for h in model_hp[g]:
            Mi.append(i)
            Mj.append(space.t_ix[h])
    Mdir = sparse.csr_matrix((np.ones(len(Mi), np.float32), (Mi, Mj)), shape=(len(genes_m), space.n_t))
    Mc = (Mdir @ T).tocsr()
    Mc.data[:] = 1.0
    McT = np.ascontiguousarray(Mc.toarray().astype(bool).T)   # [n_t, n_models]
    # ---- disease terms and best-IC per (term, model) ---------------------------------
    used_terms = np.unique(Q.indices)
    best = np.zeros((len(used_terms), len(genes_m)), dtype=np.float32)
    for k_, t in enumerate(used_terms):
        anc = T[t].indices
        order = anc[np.argsort(-ic[anc])]
        todo = np.ones(len(genes_m), dtype=bool)
        for a in order:
            hit = todo & McT[a]
            if hit.any():
                best[k_, hit] = ic[a]
                todo &= ~hit
            if not todo.any():
                break
        best[k_] /= max(ic[t], 1e-6)
    np.clip(best, 0.0, 1.0, out=best)
    # disease weights (frequency weights as for x^H)
    from .featurepack71 import freq_weight
    ph = ph.assign(wt=[freq_weight(f) for f in ph["frequency"]])
    ph = ph[ph["hpo_id"].isin(space.t_ix) & ph["disease_id"].isin(w.idx.d_ix)]
    ph = ph.groupby(["disease_id", "hpo_id"], as_index=False)["wt"].max()
    t_pos = {t: i for i, t in enumerate(used_terms)}
    rows = ph["disease_id"].map(w.idx.d_ix).astype(int).to_numpy()
    cols = ph["hpo_id"].map(space.t_ix).map(t_pos)
    ok = cols.notna().to_numpy()
    W = sparse.csr_matrix((ph["wt"].to_numpy(np.float32)[ok], (rows[ok], cols[ok].astype(int))),
                          shape=(w.n_d, len(used_terms)))
    den = np.asarray(W.sum(axis=1)).ravel()
    Wn = sparse.diags(1.0 / np.maximum(den, 1e-12)) @ W
    S = (Wn @ best).astype(np.float16)                     # [n_d, n_models]
    has_d = den > 0
    # ---- admission checks ---------------------------------------------------------------
    rng = rng_for("mouse_auc")
    dev = [d for d in w.labelled(w.spec.dev_diseases) if has_d[w.idx.d_ix[d]]]
    g_pos = {g: i for i, g in enumerate(genes_m)}
    pos_s, neg_s = [], []
    for d in dev:
        di = w.idx.d_ix[d]
        q = [g_pos[g] for g in w.query_rows(d) if g in g_pos]
        pos_s += [float(S[di, i]) for i in q]
        neg = rng.choice(len(genes_m), size=min(50, len(genes_m)), replace=False)
        neg_s += [float(S[di, i]) for i in neg if i not in set(q)]
    pos_s, neg_s = np.array(pos_s), np.array(neg_s)
    auc = float((pos_s[:, None] > neg_s[None, :]).mean() + 0.5 * (pos_s[:, None] == neg_s[None, :]).mean()) \
        if len(pos_s) and len(neg_s) else float("nan")
    checks = {"ids_valid_share": float(ss["valid_ids"].mean()),
              "bridge_rows": int(len(bridge)), "bridge_predicates": bridge["relation"].value_counts().to_dict(),
              "impc_calls": int(n_calls), "calls_on_1to1_project_genes": int(len(calls)),
              "calls_mapped_share": float(n_mapped_any / max(len(calls), 1)),
              "calls_mapped_direct_share": float(mapped_direct.mean()),
              "genes_with_model": int(len(genes_m)), "one_to_one_classes": int(len(one)),
              "dev_pairs_pos": int(len(pos_s)), "dev_auc_known_vs_other": auc,
              "diseases_with_hpo": int(has_d.sum())}
    admitted = bool(checks["ids_valid_share"] > 0.95 and checks["calls_mapped_share"] >= 0.5
                    and auc == auc and auc > 0.55)
    # ---- outputs ------------------------------------------------------------------------------------
    br = bridge[["hp", "object_label", "mp", "subject_label", "relation", "confidence",
                 "mapping_justification", "mapping_date"]].copy()
    br["derived"] = False
    br["source_version"] = "mh_mapping_initiative mp_hp_mgi_all (master at download)"
    br["similarity_range"] = "[0,1] via IC(MICA)/IC(h)"
    dr = pd.DataFrame(derived_rows, columns=["mp", "hp_via_ancestor"]).drop_duplicates()
    br.to_parquet(paths71.TABLES / "hp_mp_bridge_v71.parquet", index=False)
    dr.to_csv(paths71.AUDIT / "hp_mp_bridge_derived_v71.csv", index=False, encoding="utf-8-sig")
    br.sample(n=min(100, len(br)), random_state=20260923).to_csv(
        paths71.AUDIT / "hp_mp_bridge_spotcheck_sample_v71.csv", index=False, encoding="utf-8-sig")
    np.savez(SIM_FILE, S=S, genes=genes_m, has_d=has_d)
    rep = {"generated_utc": now_utc(), "checks": checks, "admitted": admitted,
           "manual_spot_check": "exported (100 rows), not performed",
           "missing_rule": "untested / no 1:1 orthologue / no HPO -> mask False; not 0",
           "limits": "knockout calls do not represent gain-of-function or dominant-negative mechanisms"}
    write_manifest("mouse_channel_v71.json", rep)
    return rep


def attach_mouse(T: dict, device: str) -> dict:
    import torch
    if "sM" in T:
        return T
    z = np.load(SIM_FILE)
    S, genes, has_d = z["S"], z["genes"], z["has_d"]
    D = T["xT"].shape[0]
    G = T["xS"].shape[0]
    full = torch.zeros((D, G), dtype=torch.float16, device=device)
    full[:, torch.as_tensor(genes, device=device)] = torch.as_tensor(S, device=device)
    mask = torch.zeros((D, G), dtype=torch.bool, device=device)
    mask[:, torch.as_tensor(genes, device=device)] = True
    mask &= torch.as_tensor(has_d, device=device)[:, None]
    T = dict(T)
    T["sM"], T["mM"] = full, mask
    return T


if __name__ == "__main__":
    paths71.ensure()
    print(json.dumps(build(), indent=2, default=str))

"""V8.1 §12.3 — the historical world (cut-off 2023-12-31) and the X1 / X2 tasks.

Every input is checked against the cut-off (date and SHA-256 recorded):
  CTD disease-gene relations  Wayback 2023-12-05 report (tables_historical, built by V5)
  CTD disease vocabulary      Wayback 2022-01-24 (names, definitions, synonyms, tree numbers, alt ids)
  HPO ontology / annotations  v2023-10-09 (hp.obo, phenotype.hpoa, genes_to_phenotype)
  Disease Ontology            v2023-11-30
  GO / GOA human              2023-10-09;  HGNC 2023-10-01 (entity mapping and the text mask)
  GTEx v8 (2017) replaces HPA; STRING v12.0 (2023) physical network
Inputs that are still later than the cut-off are listed, never hidden: the gene
background and MANE-2025 protein sequences (static attribute), the 2026
equivalence groups / families used for the split, the frozen V8.1 hyper-parameters.
MONDO (current) text and mappings are not used.

HistWorld / HistSourceEngine have their own ``__init__`` (nothing from the current
tables is read by accident).  Historical contexts: one outer context (reference =
historical labelled diseases minus a fixed validation block; X2 diseases stay in
the reference) and three inner contexts (the outer reference split into three
family-hashed blocks).  All models are re-trained with the registered settings for
tasks without a native context.

X2 = new relations of diseases that had relations at the cut-off (old genes removed
from the ranking; not disease-cold).  X1 = current relations of diseases without any
association supervision at the cut-off (whether text / HPO existed then is recorded).
The second-level model learns on disease-cold OOF rows while X2 rows are disease-warm.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from gpu8 import paths as p5
from gpu8v6.common import ContextSpec

from . import PROTOCOL_ID, paths8
from .common8 import dump_json, list_hash, now_utc
from .roles8 import RoleGeneHPOSource, RoleSourceEngine

SNAP = p5.PROJECT_ROOT / "data_snapshots" / "historical_2023q4"
HT = p5.GPU8_ROOT / "tables_historical"
EXT6 = p5.EXTERNAL_DIR / "v6"
CUTOFF = "2023-12-31"
HCTX = ["h_inner0", "h_inner1", "h_inner2", "h_outer"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


INPUTS = [
    ("ctd_relations", HT / "relations.parquet", "2023-12-05", "Wayback CTD_genes_diseases 2023-12-05 (via V5 build)"),
    ("ctd_evidence", HT / "evidence.parquet", "2023-12-05", "PMIDs of the same edge table"),
    ("ctd_vocabulary", SNAP / "association" / "CTD_diseases.tsv.gz", "2022-01-24", "Wayback CTD_diseases"),
    ("ctd_hierarchy", HT / "hierarchy.parquet", "2022-01-24", "ParentIDs of the 2022 vocabulary"),
    ("hpo_obo", SNAP / "phenotype" / "hp.obo", "2023-10-09", "data-version hp/releases/2023-10-09"),
    ("hpoa", SNAP / "phenotype" / "phenotype.hpoa", "2023-10-09", "HPO release v2023-10-09"),
    ("hpo_phenotypes_mapped", HT / "phenotypes.parquet", "2023-10-09", "HPOA mapped with the 2022 CTD alt ids"),
    ("genes_to_phenotype", EXT6 / "hpo" / "genes_to_phenotype_v2023-10-09.txt", "2023-10-09", "HPO release"),
    ("doid", SNAP / "ontology" / "doid.obo", "2023-11-30", "data-version releases/2023-11-30"),
    ("go_obo", EXT6 / "go" / "go-basic_2023-10-09.obo", "2023-10-09", "GO release"),
    ("goa_human", EXT6 / "go" / "goa_human_2023-10-09.gaf.gz", "2023-10-09", "GOA release"),
    ("hgnc", EXT6 / "hgnc" / "hgnc_complete_set_2023-10-01.txt", "2023-10-01", "HGNC monthly"),
    ("gtex_v8", EXT6 / "gtex" / "GTEx_v8_gene_median_tpm.gct.gz", "2017-06-05", "GTEx v8 median TPM"),
]
LATER = [{"input": "gene background (19,861) and HGNC-based gene index", "date": "2026", "role": "entity index"},
         {"input": "protein sequences (MANE 2025)", "date": "2025", "role": "static gene attribute"},
         {"input": "STRING v12.0 physical network", "date": "2023-07", "role": "PPI (not later than the cut-off)"},
         {"input": "disease equivalence groups and families", "date": "2026", "role": "split units only"},
         {"input": "V8.1 hyper-parameters and development choices", "date": "2026", "role": "frozen settings"},
         {"input": "SapBERT / ESM-2 checkpoints", "date": "2021 / 2022", "role": "frozen encoders"}]


def audit_inputs() -> dict:
    rows = []
    for name, p, date, note in INPUTS:
        rows.append({"input": name, "path": paths8.rel(p), "exists": p.exists(), "date": date,
                     "not_later_than_cutoff": date <= CUTOFF, "sha256": sha(p) if p.exists() else None,
                     "note": note})
    rep = {"cutoff": CUTOFF, "inputs": rows, "all_present": all(r["exists"] for r in rows),
           "all_not_later": all(r["not_later_than_cutoff"] for r in rows), "later_inputs_disclosed": LATER,
           "mondo_used": False, "generated_utc": now_utc()}
    dump_json(paths8.AUDIT / "historical_inputs_v8.json", rep)
    return rep


# ---------------------------------------------------------------------------
def parse_do(path: Path) -> dict:
    out, cur, obs = {}, None, False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                if cur and not obs:
                    out[cur["id"]] = cur
                cur, obs = {"id": "", "name": "", "def": "", "syn": []}, False
                continue
            if line.startswith("[") and line.endswith("]"):
                if cur and not obs:
                    out[cur["id"]] = cur
                cur = None
                continue
            if cur is None:
                continue
            if line.startswith("id: "):
                cur["id"] = line[4:].strip()
            elif line.startswith("name: "):
                cur["name"] = line[6:].strip()
            elif line.startswith("def: "):
                m = re.match(r'def: "(.*?)"', line)
                cur["def"] = m.group(1) if m else ""
            elif line.startswith("synonym: ") and "EXACT" in line:
                m = re.match(r'synonym: "(.*?)"', line)
                if m:
                    cur["syn"].append(m.group(1))
            elif line.startswith("is_obsolete: true"):
                obs = True
    if cur and not obs:
        out[cur["id"]] = cur
    return out


def symbol_dictionary_2023(genes: pd.DataFrame):
    from gpu8v6.disease_text import SymbolDictionary
    hg = pd.read_csv(EXT6 / "hgnc" / "hgnc_complete_set_2023-10-01.txt", sep="\t", dtype=str, low_memory=False)
    keep = set(genes["hgnc_id"].dropna())
    hg = hg[hg["hgnc_id"].isin(keep)]
    approved = {s.strip() for s in hg["symbol"].dropna() if len(s.strip()) >= 2}
    symbols = set(approved)
    for col in ("alias_symbol", "prev_symbol"):
        for v in hg[col].dropna():
            for s in str(v).split("|"):
                s = s.strip()
                if len(s) >= 2 and not s.islower():
                    symbols.add(s)
    names = {str(v).strip().lower() for v in hg["name"].dropna() if len(str(v).split()) >= 2}
    return SymbolDictionary(approved, symbols, names)


class HistWorld:
    """The cut-off world.  Own __init__: nothing is inherited from the current tables
    except the entity index, the gene table and the (2023) physical PPI."""

    def __init__(self):
        from gpu8.config import load_config
        from gpu8.contexts import load_entity_index
        from gpu8 import ingest
        from gpu8v6.phenotypes_v6 import hpo_ancestors, hpo_depth_and_roots, term_class
        t0 = time.time()
        self.cfg = load_config()
        td = p5.TABLES_DIR
        self.idx = load_entity_index(td)
        self.n_g, self.n_d = self.idx.n_g, self.idx.n_d
        self.genes = pd.read_parquet(td / "genes.parquet")
        self.ppi = pd.read_parquet(td / "ppi.parquet")
        self.gene_id_of_row = {i: g for g, i in self.idx.g_ix.items()}
        cur = pd.read_parquet(td / "diseases.parquet")
        # --- CTD 2022 vocabulary text
        nodes, alias, _ = ingest.build_disease_vocabulary(SNAP / "association" / "CTD_diseases.tsv.gz")
        self.alias_2022 = alias
        rows = []
        n_text = 0
        for r in cur.itertuples(index=False):
            nd = nodes.get(r.disease_id) or nodes.get(alias.get(r.disease_id, ""), None)
            rows.append({"disease_id": r.disease_id,
                         "disease_name": nd["disease_name"] if nd else "",
                         "definition": nd["definition"] if nd else "", "synonyms": nd["synonyms"] if nd else "",
                         "tree_numbers": nd["tree_numbers"] if nd else "", "parent_ids": nd["parent_ids"] if nd else "",
                         "source_ids": r.source_ids, "omim_ids": nd["omim_ids"] if nd else "",
                         "equivalence_group": r.equivalence_group, "family_group": r.family_group,
                         "mapping_type": "", "mondo_ids": "", "in_ctd_2022": bool(nd)})
            n_text += bool(nd)
        self.diseases = pd.DataFrame(rows)
        # --- HPO 2023
        hp = ingest.parse_obo(SNAP / "phenotype" / "hp.obo", "HP:")
        self.hpo_terms = pd.DataFrame([{"hpo_id": k, "name": v.get("name", ""), "definition": v.get("definition", ""),
                                        "synonyms": "|".join(v.get("synonyms", [])),
                                        "is_a": "|".join(v.get("is_a", []))}
                                       for k, v in hp.items() if not v.get("obsolete")])
        parents = {r.hpo_id: [p for p in r.is_a.split("|") if p] for r in self.hpo_terms.itertuples(index=False)}
        depth, _ = hpo_depth_and_roots(parents)
        anc = hpo_ancestors(parents)
        ph = pd.read_parquet(HT / "phenotypes.parquet")
        ph["term_class"] = [term_class(t, depth, anc) for t in ph["hpo_id"]]
        ph["keep_v6"] = ((ph["negated"].astype(int) == 0) & (ph["aspect"] == "P") &
                         (ph["term_class"] == "valid")).astype(int)
        self.phenotypes = ph
        self.phenotypes_v6 = ph[ph["keep_v6"] == 1].drop_duplicates(["disease_id", "hpo_id"]).copy()
        # --- relations / evidence / hierarchy (cut-off)
        self.relations = pd.read_parquet(HT / "relations.parquet")
        self.evidence = pd.read_parquet(HT / "evidence.parquet")
        self.hierarchy = pd.read_parquet(HT / "hierarchy.parquet")
        prim = set(self.cfg.data.primary_evidence)
        rel = self.relations[self.relations["relation_type"].isin(prim)]
        rel = rel[rel["disease_id"].isin(self.idx.d_ix) & rel["gene_id"].astype(str).isin(self.idx.g_ix)]
        self.primary_relations = rel
        self.positives = rel.groupby("disease_id")["gene_id"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
        ev = self.evidence[self.evidence["pmid"].astype(str).str.len() > 0]
        self.relation_pmids = ev.groupby("relation_id")["pmid"].apply(set).to_dict()
        self.contexts: dict[str, ContextSpec] = {}
        self.spec = None
        self.info = {"diseases_in_ctd_2022": n_text, "hpo_terms_2023": int(len(self.hpo_terms)),
                     "phenotype_rows_valid": int(len(self.phenotypes_v6)),
                     "historical_primary_relations": int(len(rel)),
                     "historical_labelled_diseases": int(len(self.positives)), "seconds": time.time() - t0}

    # same label / adjacency helpers as World6
    def query_rows(self, disease_id: str, exclude: set | None = None) -> np.ndarray:
        g_ix = self.idx.g_ix
        genes = [g for g in self.positives.get(disease_id, []) if g in g_ix]
        if exclude:
            genes = [g for g in genes if g not in exclude]
        return np.array(sorted(g_ix[g] for g in genes), dtype=np.int64)

    def labelled(self, ids) -> list[str]:
        return [d for d in ids if self.positives.get(d) and d in self.idx.d_ix]


# ---------------------------------------------------------------------------
def tasks(hw: HistWorld) -> dict:
    """X2 (new relations of diseases known at the cut-off) and X1 (no supervision at the cut-off)."""
    from gpu8v6.common import World6
    w = World6(load_heavy=False)
    cur = {(d, g) for d, gs in w.positives.items() for g in gs}
    old = {(d, g) for d, gs in hw.positives.items() for g in gs}
    x2 = sorted((d, g) for d, g in cur - old if d in hw.positives)
    x1_d = sorted(d for d in w.positives if d not in hw.positives and d in w.idx.d_ix)
    x1 = sorted((d, g) for d in x1_d for g in w.positives[d])
    has_hpo = set(hw.phenotypes_v6["disease_id"])
    x1_info = [{"disease_id": d, "text_at_cutoff": bool(hw.diseases.set_index("disease_id").at[d, "in_ctd_2022"]),
                "hpo_at_cutoff": d in has_hpo, "n_relations": len(w.positives[d])} for d in x1_d]
    return {"x2_pairs": x2, "x2_diseases": sorted({d for d, _ in x2}), "x1_pairs": x1, "x1_diseases": x1_d,
            "x1_info": x1_info, "current_positives": w.positives}


def fam_hash(f: str, tag: str) -> int:
    return int(hashlib.sha256(f"{PROTOCOL_ID}|{tag}|{f}".encode()).hexdigest()[:8], 16)


def build_contexts(hw: HistWorld, tk: dict) -> dict:
    fam = dict(zip(hw.diseases["disease_id"], hw.diseases["family_group"]))
    lab = sorted(hw.positives)
    x2 = set(tk["x2_diseases"])
    x2_fam = {fam.get(d, d) for d in x2}
    val = sorted(d for d in lab if d not in x2 and fam.get(d, d) not in x2_fam
                 and fam_hash(fam.get(d, d), "hist_val") % 100 < 15)
    ref = [d for d in lab if d not in set(val)]
    blocks = {b: sorted(d for d in ref if fam_hash(fam.get(d, d), "hist_inner") % 3 == b) for b in range(3)}
    hw.contexts["h_outer"] = ContextSpec("h_outer", 0, "outer", -1, ref, [])
    for b in range(3):
        hw.contexts[f"h_inner{b}"] = ContextSpec(f"h_inner{b}", 0, "inner", b,
                                                [d for d in ref if d not in set(blocks[b])], blocks[b])
    return {"val": val, "blocks": blocks, "reference": ref}


class HistRoles:
    """Role lists of the historical contexts (no label table is read to exclude)."""

    def __init__(self, hw: HistWorld, parts: dict, tk: dict):
        self.w = hw
        self.eq = dict(zip(hw.diseases["disease_id"], hw.diseases["equivalence_group"]))
        self.val = set(parts["val"])
        self.x1 = set(tk["x1_diseases"])
        self.x2 = set(tk["x2_diseases"])
        self.folds = {i: set() for i in range(5)}
        self.folds[1] = self.val                     # teacher_data reads folds[(j + 1) % 5] as F_{j+1}
        self.dev = set()
        self.ext = set()

    def held_out(self, ctx: ContextSpec) -> set:
        h = self.val | self.x1
        if ctx.kind == "inner":
            h |= set(ctx.held_out)
        return h

    def forbidden_groups(self, ctx: ContextSpec) -> set:
        return {self.eq.get(d, d) for d in self.held_out(ctx)}

    def reference(self, ctx: ContextSpec) -> list[str]:
        forb = self.forbidden_groups(ctx)
        return [d for d in ctx.reference if self.eq.get(d, d) not in forb]

    def targets(self, ctx: ContextSpec) -> dict:
        if ctx.kind == "inner":
            return {"oof": sorted(ctx.held_out), "val": sorted(self.val)}
        return {"val": sorted(self.val), "x2": sorted(self.x2), "x1": sorted(self.x1)}


def hist_text_records(hw: HistWorld) -> pd.DataFrame:
    from gpu8v6.disease_text import mask_segment
    sd = symbol_dictionary_2023(hw.genes)
    do = parse_do(SNAP / "ontology" / "doid.obo")
    rows = []
    for r in hw.diseases.itertuples(index=False):
        name = str(r.disease_name or "").strip()
        defs = [str(r.definition).strip()] if str(r.definition or "").strip() else []
        syns = [s.strip() for s in str(r.synonyms or "").split("|") if s.strip()]
        for sid in str(r.source_ids or "").split("|"):
            sid = sid.strip()
            if sid.startswith("DO:") and ("DOID:" + sid[3:].replace("DOID:", "")) in do:
                t = do["DOID:" + sid[3:].replace("DOID:", "")]
                syns.append(t["name"])
                syns += t["syn"]
                if t["def"]:
                    defs.append(t["def"])
        seen, syn_u = {name.lower()}, []
        for s in syns:
            if s and s.lower() not in seen:
                seen.add(s.lower())
                syn_u.append(s)
        m = lambda x: mask_segment(x, sd, level=1)[0]
        mn = m(name) if name else ""
        md = [m(x) for x in defs]
        ms = [m(x) for x in syn_u[:8]]
        segs_all = [mn] + [m(x) for x in syn_u] + md
        rows.append({"disease_id": r.disease_id, "name": mn, "defs": md, "syns": ms,
                     "text_mask1": " . ".join(x for x in segs_all if x),
                     "has_text": bool(name)})
    return pd.DataFrame(rows)


class HistSourceEngine(RoleSourceEngine):
    """The frozen V6 source engine on the cut-off world (own __init__, role exclusion)."""

    def __init__(self, hw: HistWorld, roles: HistRoles, text: pd.DataFrame, device: str = "cuda"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from gpu8.contexts import build_hierarchy_neighbours, build_P
        from gpu8v6.disease_text import MASK
        from gpu8v6.gene_hpo import HPOSpace, disease_profiles
        from gpu8v6.ontology_v2 import OntologyIndex
        from gpu8v6.propagate import PPIOperator
        self.w = hw
        self.roles = roles
        self.device = device
        self.k_ref = 20
        self.space = HPOSpace(hw.hpo_terms)
        self.Q = disease_profiles(self.space, hw.n_d, hw.idx.d_ix, phenotypes_v6=hw.phenotypes_v6)
        self.has_hpo = np.diff(self.Q.indptr) > 0
        self.ghpo = HistGeneHPO(hw, self.space)
        self.onto = OntologyIndex(hw.diseases)
        rec = text.set_index("disease_id").reindex(hw.idx.diseases)
        texts = rec["text_mask1"].fillna("").str.replace(MASK, " ", regex=False).tolist()
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.2, sublinear_tf=True,
                              stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b")
        self.X = sparse.csr_matrix(vec.fit_transform(texts).astype(np.float32))
        self.has_text = np.diff(self.X.indptr) > 0
        P, deg = build_P(hw.ppi, hw.idx)
        self.ppi_deg = deg
        self.op = PPIOperator(P, device=device)
        self.hier_nb = build_hierarchy_neighbours(hw.hierarchy, hw.idx, hw.cfg.candidates.hierarchy_max_distance)
        self.eq = roles.eq
        self.with_v5, self.lenient, self._v5_cache = False, False, {}
        self.drop_relations, self.drop_genes = set(), np.array([], dtype=np.int64)

    def adjacency(self, cs):
        from .roles8 import adjacency
        return adjacency(self.w, cs.reference)


class HistGeneHPO(RoleGeneHPOSource):
    """2023-10 genes_to_phenotype with the 2022 CTD id map and role-based source exclusion."""

    def __init__(self, hw: HistWorld, space):
        self.w = hw
        self.space = space
        g2p = pd.read_csv(EXT6 / "hpo" / "genes_to_phenotype_v2023-10-09.txt", sep="\t", dtype=str)
        g2p = g2p.rename(columns={"disease_id": "hpoa_id"})
        eq = dict(zip(hw.diseases["disease_id"], hw.diseases["equivalence_group"]))
        g2p["gene_row"] = g2p["ncbi_gene_id"].map(hw.idx.g_ix)
        g2p["t"] = g2p["hpo_id"].map(space.t_ix)
        g2p["source_disease"] = g2p["hpoa_id"].map(lambda h: hw.alias_2022.get(h))
        g2p["source_disease"] = g2p["source_disease"].where(g2p["source_disease"].isin(hw.idx.d_ix))
        g2p["source_group"] = g2p["source_disease"].map(eq)
        rows = g2p[g2p["gene_row"].notna() & g2p["t"].notna()].copy()
        rows["gene_row"] = rows["gene_row"].astype(int)
        rows["t"] = rows["t"].astype(int)
        rows["mapped"] = rows["source_disease"].notna()
        self.rows = rows.drop_duplicates(["gene_row", "t", "hpoa_id"])[
            ["gene_row", "t", "hpoa_id", "source_disease", "source_group", "mapped"]].reset_index(drop=True)
        self.eq = eq


# ---------------------------------------------------------------------------
def encode_hist(hw: HistWorld, text: pd.DataFrame, device: str = "cuda") -> dict:
    """Segmented SapBERT disease text (2023 mask), 2023 HPO term texts, 2023 GO term texts."""
    paths8.add_libs()                        # the encoder libraries live outside the frozen venv
    from gpu8v71.encoders71 import parse_go_text
    from gpu8v71 import paths71
    from transformers import AutoTokenizer
    from .textseg8 import Segmenter, encode_segments
    enc = paths8.CACHES / "encoders"
    enc.mkdir(parents=True, exist_ok=True)
    out = {}
    tok = AutoTokenizer.from_pretrained(str(paths71.MODELS / "sapbert"))
    seg = Segmenter(tok)
    p = enc / "sapbert_disease_seg_hist2023.npy"
    if not p.exists():
        rows = []
        for i, r in enumerate(text.itertuples(index=False)):
            if not r.has_text:
                continue
            for s in seg.segments(r.name, list(r.defs), list(r.syns)):
                rows.append((i, s["text"], max(s["n_tokens"], 1)))
        X, ntr = encode_segments([t for _, t, _ in rows], device)
        V = np.zeros((len(text), X.shape[1]))
        W = np.zeros(len(text))
        for (i, _, wt), x in zip(rows, X):
            V[i] += wt * x
            W[i] += wt
        V = np.where(W[:, None] > 0, V / np.maximum(W, 1e-12)[:, None], 0.0).astype(np.float32)
        np.save(p, V)
        out["disease_segments"] = len(rows)
    p = enc / "sapbert_hpo_2023.npy"
    if not p.exists():
        t = [(n + (" ; " + d if d else "")) for n, d in zip(hw.hpo_terms["name"], hw.hpo_terms["definition"])]
        X, _ = encode_segments(t, device)
        np.save(p, X)
        hw.hpo_terms[["hpo_id"]].to_parquet(paths8.TABLES / "hpo_term_ids_2023.parquet", index=False)
    p = enc / "sapbert_go_2023.npy"
    if not p.exists():
        from gpu8v6 import paths6
        gt = parse_go_text(paths6.ext("go", "go-basic_2023-10-09.obo"))
        X, _ = encode_segments(gt["text"].tolist(), device)
        np.save(p, X)
        gt[["term_id", "text"]].to_parquet(paths8.TABLES / "go_term_texts_2023.parquet", index=False)
    return out


def hist_pack(hw: HistWorld, text: pd.DataFrame) -> dict:
    """Feature arrays of the cut-off world (GTEx expression; MANE-2025 sequences stated)."""
    from gpu8v71.featurepack71 import aggregate_terms, freq_weight
    from gpu8v71.expression71 import ALIAS_RULES
    from .evidence8 import GOPolicy
    from .pack8 import V71_PACK
    z = np.load(V71_PACK)
    a = {k: z[k] for k in z.files}
    d_ix = hw.idx.d_ix
    enc = paths8.CACHES / "encoders"
    V = np.load(enc / "sapbert_disease_seg_hist2023.npy")
    Tx = np.zeros_like(a["d_text"])
    mT = np.zeros(hw.n_d, dtype=bool)
    for i, r in enumerate(text.itertuples(index=False)):
        if r.has_text and r.disease_id in d_ix:
            Tx[d_ix[r.disease_id]] = V[i]
            mT[d_ix[r.disease_id]] = True
    a["d_text"], a["d_text_mask"] = Tx, mT
    Xh = np.load(enc / "sapbert_hpo_2023.npy")
    ph = hw.phenotypes_v6
    tab = pd.DataFrame({"entity": ph["disease_id"], "term": ph["hpo_id"],
                        "weight": [freq_weight(f) for f in ph["frequency"]]})
    a["d_hpo"], a["d_hpo_mask"], _ = aggregate_terms(Xh, {t: i for i, t in enumerate(hw.hpo_terms["hpo_id"])},
                                                    tab, hw.n_d, d_ix)
    # disease tissue rules on the cut-off text and HPO names (V7.1 tissue order and alias rules)
    names = hw.hpo_terms.set_index("hpo_id")["name"]
    hn = ph.assign(n=ph["hpo_id"].map(names).fillna("")).groupby("disease_id")["n"].apply(
        lambda s: " ; ".join(sorted(set(s)))).to_dict()
    to = json.loads((paths8.V71_TABLES / "tissue_order_v71.json").read_text(encoding="utf-8"))
    tlist = to if isinstance(to, list) else (to.get("tissues") or to.get("order"))
    Q = np.zeros((hw.n_d, a["d_tissue"].shape[1]), dtype=np.float64)
    if tlist:
        t_ix = {t: i for i, t in enumerate(tlist)}
        rules = [(rid, re.compile(rx), ts) for rid, rx, ts, _ in ALIAS_RULES]
        tt = dict(zip(text["disease_id"], [(" ; ".join([r.name] + list(r.defs) + list(r.syns))) for r in
                                           text.itertuples(index=False)]))
        for d, di in d_ix.items():
            s = (tt.get(d, "") + " ; " + hn.get(d, "")).lower()
            for _, rx, ts in rules:
                if rx.search(s):
                    for t in ts:
                        if t in t_ix:
                            Q[di, t_ix[t]] += 1.0 / len(ts)
            if Q[di].sum() > 0:
                Q[di] /= Q[di].sum()
        a["d_tissue"], a["d_tissue_mask"] = Q.astype(np.float32), Q.sum(1) > 0
    else:
        a["d_tissue"] = np.zeros_like(a["d_tissue"])
        a["d_tissue_mask"] = np.zeros(hw.n_d, dtype=bool)
    a["g_expression"], a["g_expression_mask"] = a["g_expression_gtex"], a["g_expression_gtex_mask"]
    pol = GOPolicy(hw, "2023")
    fa = pol.function_arrays()
    a["g_function"], a["g_function_mask"], a["anchor_vectors"] = fa["g_function"], fa["g_function_mask"], \
        fa["anchor_vectors"]
    return {"a": a, "A": fa["A_plus"], "go_records": int(len(pol.records)),
            "tissue_rules_applied": bool(tlist)}

"""V7.1 section 6 — normal-tissue expression and disease tissue conditions.

Gene side (HPA ``rna_tissue_consensus.tsv``; header must be exactly
Gene / Gene name / Tissue / nTPM, otherwise the build stops):

    E_gt >= 0 (nTPM), o_gt = 1 if measured (a measured zero is information),
    mu_g = sum_t o_gt log(1 + E_gt) / (sum_t o_gt + eps),
    E~_gt = o_gt [log(1 + E_gt) - mu_g].

Duplicate gene-tissue measurements stop the build (never silently averaged);
an Ensembl id mapped to several project genes goes to the exception table.

Disease side: q_d(t) >= 0, sum 1, from a *registered* organ-word alias table
(below) matched against the masked disease template and the names of the
disease's positive HPO terms.  Each rule that fires spreads one unit of weight
evenly over its tissue set; broad rules (e.g. "brain" -> seven brain regions)
are flagged.  No hit -> ``d_tissue_mask = False``.  Nothing is inferred from a
hidden causal gene.

The GTEx v8 arm (E4, "old resources") is placed on the same 51-tissue axis
through an explicit GTEx -> HPA tissue map; HPA tissues without a GTEx
counterpart are unobserved there (o = 0), so the coverage c_dg < 1 is real.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from gpu8 import paths as p5
from gpu8v6 import paths6

from . import paths71
from .common71 import array_hash, dump_json, now_utc

EPS = 1e-9
HPA_FILE = ("hpa", "rna_tissue_consensus.tsv.zip")
BRAIN = ["cerebral cortex", "cerebellum", "basal ganglia", "hippocampal formation", "amygdala",
         "midbrain", "hypothalamus"]
IMMUNE = ["lymph node", "spleen", "thymus", "tonsil"]

# rule_id, regex (lower-cased text), tissues, broad
ALIAS_RULES = [
    ("brain_broad", r"\bbrain|cerebr|encephal|\bneuro|seizure|epilep|intellectual disabilit|mental retardation|developmental delay|cognitive|dementia|leukodystroph|microcephal|hydroceph|autis|schizophren|psychos|white matter|cortical", BRAIN, True),
    ("cerebellum", r"cerebell|ataxi", ["cerebellum"], False),
    ("basal_ganglia", r"basal ganglia|striat|chorea|dystoni|parkinson", ["basal ganglia", "midbrain"], False),
    ("hippocampus", r"hippocamp|memory", ["hippocampal formation"], False),
    ("hypothalamus", r"hypothalam", ["hypothalamus"], False),
    ("pituitary", r"pituitar|growth hormone|acromegal|prolactin", ["pituitary gland"], False),
    ("choroid_plexus", r"choroid plexus", ["choroid plexus"], False),
    ("spinal_cord", r"spinal cord|myelopath|motor neuron|spinal muscular atrophy|amyotrophic|spastic parapleg", ["spinal cord"], False),
    ("retina", r"retin|macula|photorecept|cone-rod|rod-cone|night blind|optic atroph|usher", ["retina"], False),
    ("eye_broad", r"\beye|ocular|visual|vision|blind", ["retina"], True),
    ("heart", r"heart|cardiac|cardiomyopath|myocard|arrhythm|atrial fibrillation|long qt|brugada|tachycard|bradycard|endocard|pericard", ["heart muscle"], False),
    ("vessel", r"vascul|aort|arter|aneurysm|venous|\bvein|angio|vasculitis|hypertension", ["blood vessel"], False),
    ("kidney", r"kidney|renal|nephr|glomerul", ["kidney"], False),
    ("bladder", r"urinary bladder|\bbladder|cystitis", ["urinary bladder"], False),
    ("liver", r"liver|hepat|cholesta|cirrho|coagul|hemophil|haemophil", ["liver"], False),
    ("gallbladder", r"gallbladder|biliary|cholecyst|gallstone", ["gallbladder", "liver"], False),
    ("pancreas", r"pancrea|diabet|insulin|hyperglyc|hypoglyc", ["pancreas"], False),
    ("lung", r"\blung|pulmon|respirator|bronch|pneum|alveol|asthma|airway|emphysema", ["lung"], False),
    ("skin", r"\bskin|cutane|dermat|ichthyo|epiderm|keratoderma|palmoplantar|\bhair|alopecia|\bnail|blister|bullosa|eczema|psoria|albinism|pigmentation", ["skin"], False),
    ("skeletal_muscle", r"skeletal muscle|myopath|muscular|myosit|myoton|rhabdomyol|myasthen|muscle weakness", ["skeletal muscle"], False),
    ("smooth_muscle", r"smooth muscle|visceral myopathy|megacystis", ["smooth muscle"], False),
    ("bone_marrow", r"anemi|anaemi|leuk(?:a)?emi|myelodysplas|myeloprolif|thrombocyt|neutropen|pancytopen|hemoglobin|haemoglobin|thalass|sickle|hematolog|haematolog|bone marrow|erythro|platelet", ["bone marrow"], False),
    ("immune_broad", r"immunodeficien|autoimmun|lymphoprolif|lymphoma|lymphocyt", IMMUNE, True),
    ("spleen", r"spleen|splen", ["spleen"], False),
    ("thymus", r"\bthym(?:us|ic)", ["thymus"], False),
    ("tonsil", r"tonsil", ["tonsil"], False),
    ("lymph_node", r"lymph node|lymphadenopath", ["lymph node"], False),
    ("adipose", r"adipos|lipodystroph|obesity|lipomat", ["adipose tissue"], False),
    ("thyroid", r"(?<!para)thyroid|goiter|goitre", ["thyroid gland"], False),
    ("parathyroid", r"parathyroid|hypocalcem", ["parathyroid gland"], False),
    ("adrenal", r"adrenal|cortisol|addison|aldosteron", ["adrenal gland"], False),
    ("gonad_broad", r"hypogonad|gonad", ["testis", "ovary"], True),
    ("esophagus", r"esophag|oesophag|achalasia", ["esophagus"], False),
    ("stomach", r"stomach|gastr", ["stomach"], False),
    ("duodenum", r"duoden", ["duodenum"], False),
    ("small_intestine", r"small intestin|\bileum|ileal|jejun|celiac|coeliac|malabsorption", ["small intestine", "duodenum"], False),
    ("colon", r"\bcolon|colitis|colorectal|crohn|inflammatory bowel|\bbowel|hirschsprung", ["colon", "rectum"], False),
    ("rectum", r"\brect(?:al|um)\b", ["rectum"], False),
    ("appendix", r"appendi(?:x|citis)", ["appendix"], False),
    ("intestine_broad", r"intestin", ["small intestine", "colon"], True),
    ("testis", r"\btest(?:is|es|icular)|spermat|azoosperm|male infertil", ["testis"], False),
    ("ovary", r"ovar(?:y|ian)", ["ovary"], False),
    ("endometrium", r"\buterus|(?<!intra)uterine|endometri", ["endometrium"], False),
    ("cervix", r"cervix|cervical cancer|uterine cervi", ["cervix"], False),
    ("placenta", r"placent|preeclampsia|pre-eclampsia", ["placenta"], False),
    ("prostate", r"prostat", ["prostate"], False),
    ("breast", r"breast|mammary", ["breast"], False),
    ("vagina", r"vagin", ["vagina"], False),
    ("fallopian_tube", r"fallopian", ["fallopian tube"], False),
    ("epididymis", r"epididym", ["epididymis"], False),
    ("seminal_vesicle", r"seminal vesic", ["seminal vesicle"], False),
    ("salivary", r"saliva|sjogren|sjögren", ["salivary gland"], False),
    ("tongue", r"tongue|glossal", ["tongue"], False),
]

GTEX_TO_HPA = {
    "Adipose - Subcutaneous": "adipose tissue", "Adipose - Visceral (Omentum)": "adipose tissue",
    "Adrenal Gland": "adrenal gland", "Artery - Aorta": "blood vessel",
    "Artery - Coronary": "blood vessel", "Artery - Tibial": "blood vessel",
    "Bladder": "urinary bladder", "Brain - Amygdala": "amygdala",
    "Brain - Anterior cingulate cortex (BA24)": "cerebral cortex",
    "Brain - Caudate (basal ganglia)": "basal ganglia", "Brain - Cerebellar Hemisphere": "cerebellum",
    "Brain - Cerebellum": "cerebellum", "Brain - Cortex": "cerebral cortex",
    "Brain - Frontal Cortex (BA9)": "cerebral cortex", "Brain - Hippocampus": "hippocampal formation",
    "Brain - Hypothalamus": "hypothalamus",
    "Brain - Nucleus accumbens (basal ganglia)": "basal ganglia",
    "Brain - Putamen (basal ganglia)": "basal ganglia",
    "Brain - Spinal cord (cervical c-1)": "spinal cord", "Brain - Substantia nigra": "midbrain",
    "Breast - Mammary Tissue": "breast", "Cervix - Ectocervix": "cervix",
    "Cervix - Endocervix": "cervix", "Colon - Sigmoid": "colon", "Colon - Transverse": "colon",
    "Esophagus - Gastroesophageal Junction": "esophagus", "Esophagus - Mucosa": "esophagus",
    "Esophagus - Muscularis": "esophagus", "Fallopian Tube": "fallopian tube",
    "Heart - Atrial Appendage": "heart muscle", "Heart - Left Ventricle": "heart muscle",
    "Kidney - Cortex": "kidney", "Kidney - Medulla": "kidney", "Liver": "liver", "Lung": "lung",
    "Minor Salivary Gland": "salivary gland", "Muscle - Skeletal": "skeletal muscle",
    "Ovary": "ovary", "Pancreas": "pancreas", "Pituitary": "pituitary gland",
    "Prostate": "prostate", "Skin - Not Sun Exposed (Suprapubic)": "skin",
    "Skin - Sun Exposed (Lower leg)": "skin", "Small Intestine - Terminal Ileum": "small intestine",
    "Spleen": "spleen", "Stomach": "stomach", "Testis": "testis", "Thyroid": "thyroid gland",
    "Uterus": "endometrium", "Vagina": "vagina", "Whole Blood": "bone marrow",
}


def centre(E: np.ndarray, o: np.ndarray) -> np.ndarray:
    """E~ = o [log1p(E) - mu_g], mu_g over observed tissues only."""
    L = np.where(o, np.log1p(np.maximum(E, 0.0)), 0.0)
    mu = L.sum(axis=1) / np.maximum(o.sum(axis=1), 1)      # the guard only matters when no cell is observed
    return np.where(o, L - mu[:, None], 0.0)


def hpa_matrix(world) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    df = pd.read_csv(paths71.ext(*HPA_FILE), sep="\t", dtype={"Gene": str, "Tissue": str})
    if list(df.columns) != ["Gene", "Gene name", "Tissue", "nTPM"]:
        raise RuntimeError(f"HPA consensus header changed: {list(df.columns)}")
    dup = df.duplicated(["Gene", "Tissue"]).sum()
    if dup:
        raise RuntimeError(f"{dup} duplicate gene-tissue measurements: check the resource type")
    tissues = sorted(df["Tissue"].unique())
    t_ix = {t: i for i, t in enumerate(tissues)}
    ens2rows: dict[str, list[int]] = {}
    for r in world.genes.itertuples(index=False):
        e = str(r.ensembl_gene_id or "")
        if e and e != "nan":
            ens2rows.setdefault(e, []).append(world.idx.g_ix[r.gene_id])
    amb = {e for e, rs in ens2rows.items() if len(rs) > 1}
    df = df[df["Gene"].isin(ens2rows) & ~df["Gene"].isin(amb)]
    E = np.zeros((world.n_g, len(tissues)), dtype=np.float64)
    o = np.zeros((world.n_g, len(tissues)), dtype=bool)
    rows = df["Gene"].map(lambda e: ens2rows[e][0]).to_numpy()
    cols = df["Tissue"].map(t_ix).to_numpy()
    E[rows, cols] = df["nTPM"].to_numpy(dtype=np.float64)
    o[rows, cols] = True
    info = {"n_tissues": len(tissues), "genes_measured": int(o.any(axis=1).sum()),
            "ambiguous_ensembl_ids": sorted(amb), "rows_used": int(len(df)),
            "cells_observed": int(o.sum()), "cells_zero_but_observed": int((o & (E == 0)).sum())}
    return E, o, tissues, info


def gtex_matrix(world, tissues: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    gt = pd.read_csv(paths6.ext("gtex", "GTEx_v8_gene_median_tpm.gct.gz"), sep="\t", skiprows=2,
                     compression="gzip")
    gt["ens"] = gt["Name"].str.split(".").str[0]
    ens2row = {}
    for r in world.genes.itertuples(index=False):
        e = str(r.ensembl_gene_id or "")
        if e and e != "nan":
            ens2row.setdefault(e, world.idx.g_ix[r.gene_id])
    gt["gene_row"] = gt["ens"].map(ens2row)
    gt = gt[gt["gene_row"].notna()].drop_duplicates("gene_row")
    t_ix = {t: i for i, t in enumerate(tissues)}
    E = np.zeros((world.n_g, len(tissues)), dtype=np.float64)
    o = np.zeros((world.n_g, len(tissues)), dtype=bool)
    rows = gt["gene_row"].astype(int).to_numpy()
    groups: dict[str, list[str]] = {}
    for g, h in GTEX_TO_HPA.items():
        if g in gt.columns:
            groups.setdefault(h, []).append(g)
    for h, cols in groups.items():
        # mean of log1p over the GTEx sub-tissues, back-transformed to TPM scale
        val = np.expm1(np.log1p(gt[cols].to_numpy(dtype=np.float64)).mean(axis=1))
        E[rows, t_ix[h]] = val
        o[rows, t_ix[h]] = True
    info = {"gtex_tissues_mapped": sum(len(v) for v in groups.values()),
            "hpa_tissues_with_gtex": len(groups),
            "hpa_tissues_without_gtex": sorted(set(tissues) - set(groups)),
            "genes_measured": int(o.any(axis=1).sum())}
    return E, o, info


def disease_tissue_weights(world, tissues: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    tpl = pd.read_parquet(paths71.TABLES / "disease_templates_v71.parquet").set_index("disease_id")
    ph = pd.read_parquet(paths6.TABLES / "phenotypes_v6.parquet")
    ph = ph[(ph["keep_v6"] == 1) & (ph["negated"].astype(int) == 0)]
    names = pd.read_parquet(p5.TABLES_DIR / "hpo_terms.parquet").set_index("hpo_id")["name"]
    ph = ph.assign(name=ph["hpo_id"].map(names).fillna(""))
    hpo_names = ph.groupby("disease_id")["name"].apply(lambda s: " ; ".join(sorted(set(s)))).to_dict()
    t_ix = {t: i for i, t in enumerate(tissues)}
    rules = [(rid, re.compile(rx), ts, broad) for rid, rx, ts, broad in ALIAS_RULES]
    for rid, _, ts, _ in rules:
        for t in ts:
            if t not in t_ix:
                raise RuntimeError(f"alias rule {rid} names unknown tissue {t}")
    Q = np.zeros((world.n_d, len(tissues)), dtype=np.float64)
    audit = []
    for d, di in world.idx.d_ix.items():
        text = (str(tpl["text"].get(d, "")) + " ; " + hpo_names.get(d, "")).lower()
        fired = []
        for rid, rx, ts, broad in rules:
            if rx.search(text):
                fired.append((rid, broad))
                for t in ts:
                    Q[di, t_ix[t]] += 1.0 / len(ts)
        s = Q[di].sum()
        if s > 0:
            Q[di] /= s
        audit.append({"disease_id": d, "rules": "|".join(r for r, _ in fired),
                      "n_rules": len(fired), "only_broad": bool(fired) and all(b for _, b in fired),
                      "n_tissues": int((Q[di] > 0).sum())})
    mask = Q.sum(axis=1) > 0
    return Q.astype(np.float32), mask, pd.DataFrame(audit)


def build(world=None, write: bool = True) -> dict:
    from gpu8v6.common import World6
    w = world or World6(load_heavy=False)
    E, o, tissues, info = hpa_matrix(w)
    Et = centre(E, o)
    Eg, og, ginfo = gtex_matrix(w, tissues)
    Egt = centre(Eg, og)
    Q, qmask, audit = disease_tissue_weights(w, tissues)
    lab = set(w.positives)
    a_lab = audit[audit["disease_id"].isin(lab)]
    rep = {"generated_utc": now_utc(), "hpa": info, "gtex_arm": ginfo, "tissues": tissues,
           "disease_tissue": {"n_diseases": int(len(audit)), "with_tissue": int(qmask.sum()),
                              "labelled_with_tissue": int((a_lab["n_rules"] > 0).sum()),
                              "labelled_n": int(len(a_lab)),
                              "only_broad_rules": int(audit["only_broad"].sum()),
                              "median_tissues_when_hit": float(audit.loc[audit["n_rules"] > 0,
                                                                         "n_tissues"].median()),
                              "rule_hits": pd.Series("|".join(audit["rules"]).split("|"))
                              .value_counts().drop("", errors="ignore").to_dict()},
           "rule": ("registered alias table (tables_v71/tissue_alias_table_v71.csv), masked "
                    "template + positive HPO names, equal split inside a rule, no negation parsing; "
                    "coverage threshold c_dg >= 0.8 is a development setting")}
    if write:
        np.save(paths71.TABLES / "g_expression.npy", Et.astype(np.float32))
        np.save(paths71.TABLES / "g_expression_mask.npy", o)
        np.save(paths71.TABLES / "g_expression_gtex.npy", Egt.astype(np.float32))
        np.save(paths71.TABLES / "g_expression_gtex_mask.npy", og)
        np.save(paths71.TABLES / "d_tissue.npy", Q)
        np.save(paths71.TABLES / "d_tissue_mask.npy", qmask)
        (paths71.TABLES / "tissue_order_v71.json").write_text(json.dumps(tissues), encoding="utf-8")
        pd.DataFrame([{"rule_id": r, "regex": rx, "tissues": "|".join(ts), "broad": b}
                      for r, rx, ts, b in ALIAS_RULES]).to_csv(
            paths71.TABLES / "tissue_alias_table_v71.csv", index=False, encoding="utf-8-sig")
        audit.to_parquet(paths71.TABLES / "d_tissue_audit.parquet", index=False)
        rep["hashes"] = {"g_expression": array_hash(Et.astype(np.float32), o),
                         "d_tissue": array_hash(Q, qmask)}
        dump_json(paths71.AUDIT / "expression_v71_report.json", rep)
    return rep


def main() -> int:
    paths71.ensure()
    rep = build()
    print(json.dumps({k: v for k, v in rep.items() if k != "tissues"}, indent=2, default=str)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

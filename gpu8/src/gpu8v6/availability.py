"""V6 sections 2.4, 4.1 and 5 — per-disease information-availability audit.

For every labelled disease the audit records what a deployed system could see
(HPO class after the strict repair, text, ontology position, exposed-support
eligibility, relation semantics, broad disease category) and derives the
routing class of section 5 from *observable* quantities only.  The number of
query positives is reported as a stratum, never used as a routing input.

It also checks three things the plan asks for explicitly:

* whether the V5 "hierarchy" retriever reads the disease ontology (CTD
  ParentIDs) rather than HPO neighbours — by construction and empirically for
  diseases without HPO;
* strict-equivalence leakage: labelled diseases whose strict equivalent sits in
  the reference set of the context that evaluates them (V5), and how many V6
  removes;
* shared literature between GO annotations and the CTD relations they would
  help to recover (section 2.2).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from gpu8.contexts import build_hierarchy_neighbours

from . import paths6
from .common import World6, dump_json, now_utc
from .ontology_v2 import OntologyIndex


def broad_category(tree_numbers: str, parent_cat: str = "") -> str:
    cats = sorted({t.strip().split(".")[0].split("/")[0] for t in str(tree_numbers or "").split("|")
                   if t.strip() and t.strip() != "nan"})
    if not cats:
        return parent_cat or "no_tree"
    return cats[0]


CATEGORY_NAMES = {
    "C01": "infections", "C04": "neoplasms", "C05": "musculoskeletal", "C06": "digestive",
    "C07": "stomatognathic", "C08": "respiratory", "C09": "ENT", "C10": "nervous system",
    "C11": "eye", "C12": "urogenital", "C13": "female urogenital/pregnancy",
    "C14": "cardiovascular", "C15": "hemic/lymphatic", "C16": "congenital/hereditary/neonatal",
    "C17": "skin/connective", "C18": "nutritional/metabolic", "C19": "endocrine",
    "C20": "immune", "C21": "environmental", "C22": "animal", "C23": "pathological conditions",
    "C24": "occupational", "C25": "chemically induced", "C26": "wounds", "F01": "behaviour",
    "F03": "mental disorders", "no_tree": "no tree number"}


def run() -> dict:
    w = World6()
    dis = w.diseases.set_index("disease_id")
    hpo_cls = pd.read_parquet(paths6.TABLES / "disease_hpo_class.parquet").set_index("disease_id")
    text = pd.read_parquet(paths6.TABLES / "disease_text.parquet").set_index("disease_id")
    onto = OntologyIndex(w.diseases)
    rel = w.relations
    rtypes = rel.groupby("disease_id")["relation_type"].apply(lambda s: "|".join(sorted(set(s))))
    role = {}
    for d in w.spec.dev_diseases:
        role[d] = "dev"
    for d in w.spec.external_diseases:
        role[d] = "external"
    for f, ids in w.spec.folds.items():
        for d in ids:
            role[d] = f
    # parent category for OMIM concepts without tree numbers
    parent_cat = {}
    for d, row in dis.iterrows():
        ps = [p for p in str(row["parent_ids"] or "").split("|") if p and p != "nan"]
        cats = [broad_category(dis.loc[p, "tree_numbers"]) for p in ps if p in dis.index]
        cats = [c for c in cats if c != "no_tree"]
        parent_cat[d] = sorted(cats)[0] if cats else ""
    hier_nb = build_hierarchy_neighbours(w.hierarchy, w.idx, w.cfg.candidates.hierarchy_max_distance)
    outer_ref = {f: set(w.outer_context(f).reference) for f in range(5)}

    recs = []
    for d in sorted(w.positives):
        if d not in w.idx.d_ix:
            continue
        pos = w.positives[d]
        rids = [w.relation_of.get((d, g)) for g in pos]
        n_pm = [len(w.relation_pmids.get(r, ())) for r in rids]
        row = dis.loc[d]
        r = role.get(d, "unassigned")
        ctx_fold = int(r[1]) if r.startswith("F") else 0
        ref_rows = {w.idx.d_ix[x] for x in outer_ref[ctx_fold] if x in w.idx.d_ix}
        nbs = hier_nb.get(w.idx.d_ix[d], {})
        hier_in_ref = sum(1 for nb in nbs if nb in ref_rows)
        hc = hpo_cls.loc[d] if d in hpo_cls.index else None
        n_valid = int(hc["n_hpo_valid_v6"]) if hc is not None else 0
        tx = text.loc[d] if d in text.index else None
        has_def = int(tx["has_definition"]) if tx is not None else 0
        onto_lvl = onto.level(d)
        cat = broad_category(row["tree_numbers"], parent_cat.get(d, ""))
        n = len(pos)
        recs.append({
            "disease_id": d, "role": r, "namespace": d.split(":")[0] + (
                ":" + d.split(":")[1][0] if d.startswith("MESH:") else ""),
            "category": cat, "category_name": CATEGORY_NAMES.get(cat, cat),
            "n_positives": n,
            "support_stratum": "1" if n == 1 else ("2-5" if n <= 5 else ">=6"),
            "eligible_5shot": int(n >= 6),
            "relation_types": rtypes.get(d, ""),
            "has_therapeutic": int("therapeutic" in rtypes.get(d, "")),
            "frac_relations_with_pmid": float(np.mean([x > 0 for x in n_pm])) if n_pm else 0.0,
            "n_pmid_total": int(sum(n_pm)),
            "hpo_class": hc["hpo_class"] if hc is not None else "no_resource_record",
            "n_hpo_terms_v5": int(hc["n_hpo_terms_v5"]) if hc is not None else 0,
            "n_hpo_valid_v6": n_valid,
            "has_hpo_v5": int(hc is not None and hc["n_hpo_terms_v5"] > 0),
            "has_hpo_v6": int(n_valid > 0),
            "has_definition": has_def,
            "n_synonyms": int(tx["n_synonyms"]) if tx is not None else 0,
            "n_text_tokens": int(tx["n_tokens"]) if tx is not None else 0,
            "n_masked_gene_tokens": int(tx["n_masked_l1"]) if tx is not None else 0,
            "onto_level": onto_lvl,
            "hier_neighbours_total": len(nbs), "hier_neighbours_in_reference": hier_in_ref,
        })
    df = pd.DataFrame(recs)

    def route(r):
        if r["has_hpo_v6"]:
            return "Z1_hpo"
        if r["has_definition"] or r["onto_level"] == "informative":
            return "Z2_no_hpo_text_or_ontology"
        return "Z3_no_legal_disease_information"

    df["route_zero_support"] = df.apply(route, axis=1)
    df["route_support"] = np.where(df["eligible_5shot"] == 1,
                                   np.where(df["has_hpo_v6"] == 1, "S_both_semantic_and_support",
                                            "S_support_without_hpo"), "not_in_support_task")
    out = paths6.AUDIT
    df.to_csv(out / "availability_per_disease.csv", index=False, encoding="utf-8-sig")

    # ---- cross tables -------------------------------------------------------
    df["rel_semantics"] = np.where(df["has_therapeutic"] == 1, "marker+therapeutic", "marker_only")
    ct1 = pd.crosstab([df["rel_semantics"], df["hpo_class"]], df["support_stratum"], margins=True)
    ct1.to_csv(out / "crosstab_relation_hpo_support.csv", encoding="utf-8-sig")
    ct2 = pd.crosstab(df["category_name"], [df["has_hpo_v5"], df["has_hpo_v6"]], margins=True)
    ct2.to_csv(out / "crosstab_category_hpo.csv", encoding="utf-8-sig")
    ct3 = pd.crosstab(df["route_zero_support"], df["role"], margins=True)
    ct3.to_csv(out / "crosstab_route_role.csv", encoding="utf-8-sig")
    ct4 = pd.crosstab(df["hpo_class"], df["support_stratum"], margins=True)
    ct4.to_csv(out / "crosstab_hpo_class_support.csv", encoding="utf-8-sig")
    ct5 = pd.crosstab(df["category_name"], df["support_stratum"], margins=True)
    ct5.to_csv(out / "crosstab_category_support.csv", encoding="utf-8-sig")

    five = df[df["eligible_5shot"] == 1]
    nohpo = df[df["has_hpo_v6"] == 0]
    # hierarchy retriever: CTD ParentIDs by construction; empirical check without HPO
    hier_check = {
        "reads": "hierarchy.parquet (CTD ParentIDs, relation_type=ctd_parent); no HPO input",
        "n_no_hpo_diseases": int(len(nohpo)),
        "frac_no_hpo_with_ontology_neighbour_in_reference":
            float((nohpo["hier_neighbours_in_reference"] > 0).mean()) if len(nohpo) else 0.0,
    }

    # ---- strict-equivalence leakage --------------------------------------------
    eq = dict(zip(w.diseases["disease_id"], w.diseases["equivalence_group"]))
    groups = defaultdict(list)
    for d in w.positives:
        groups[eq.get(d, d)].append(d)
    multi = {g: m for g, m in groups.items() if len(m) > 1}
    leak_rows = []
    for f in range(5):
        ref = outer_ref[f]
        for d in w.spec.folds[f"F{f}"]:
            if d not in w.positives:
                continue
            mates = [x for x in groups.get(eq.get(d, d), []) if x != d and x in ref]
            if mates:
                leak_rows.append({"fold": f, "disease_id": d, "equivalent_in_reference": "|".join(mates)})
    leak = pd.DataFrame(leak_rows)
    leak.to_csv(out / "strict_equivalence_leakage_v5.csv", index=False, encoding="utf-8-sig")

    # ---- shared literature between GO references and CTD relations -----------
    go_pm = pd.read_parquet(paths6.TABLES / "go_reference_pmids_current.parquet")
    gop = {g: set(p.split("|")) for g, p in zip(go_pm["gene_id"], go_pm["pmids"]) if p}
    shared, with_pm = 0, 0
    for r in w.primary_relations.itertuples(index=False):
        pm = w.relation_pmids.get(r.relation_id, set())
        if not pm:
            continue
        with_pm += 1
        if pm & gop.get(r.gene_id, set()):
            shared += 1

    summary = {
        "generated_utc": now_utc(),
        "n_labelled_diseases": int(len(df)),
        "hpo_v5": float(df["has_hpo_v5"].mean()), "hpo_v6": float(df["has_hpo_v6"].mean()),
        "hpo_class_counts": df["hpo_class"].value_counts().to_dict(),
        "five_shot": {"n": int(len(five)),
                      "frac_without_hpo_v5": float(1 - five["has_hpo_v5"].mean()),
                      "frac_without_hpo_v6": float(1 - five["has_hpo_v6"].mean()),
                      "hpo_class_counts": five["hpo_class"].value_counts().to_dict(),
                      "category_counts": five["category_name"].value_counts().head(10).to_dict()},
        "single_positive": {"n": int((df["n_positives"] == 1).sum()),
                            "frac": float((df["n_positives"] == 1).mean()),
                            "frac_with_hpo_v6": float(df.loc[df["n_positives"] == 1, "has_hpo_v6"].mean())},
        "route_zero_support_counts": df["route_zero_support"].value_counts().to_dict(),
        "route_support_counts": df["route_support"].value_counts().to_dict(),
        "onto_level_counts": df["onto_level"].value_counts().to_dict(),
        "frac_with_definition": float(df["has_definition"].mean()),
        "hierarchy_retriever_check": hier_check,
        "strict_equivalence": {
            "labelled_groups_with_several_members": len(multi),
            "labelled_diseases_in_such_groups": int(sum(len(m) for m in multi.values())),
            "test_diseases_with_equivalent_in_v5_outer_reference": int(len(leak)),
            "by_fold": leak.groupby("fold").size().to_dict() if len(leak) else {},
            "v6_action": "V6 removes every strict equivalent of a held-out labelled disease from the reference set of that context",
        },
        "shared_literature_go_vs_ctd": {
            "primary_relations_with_pmid": with_pm,
            "relations_whose_pmid_is_also_a_go_reference_of_the_gene": shared,
            "fraction": shared / max(with_pm, 1),
            "note": "GO is treated as auditable auxiliary knowledge, not as a source independent of the discovery literature",
        },
    }
    dump_json(out / "availability_summary.json", summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V6 availability audit.")
    ap.parse_args(argv)
    paths6.ensure()
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

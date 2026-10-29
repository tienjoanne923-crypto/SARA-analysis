"""V6 sections 2.7 and 11.3 — KDGene with a legal cold-validation curve.

A falling training loss says nothing about ranking.  This run trains the V5
KDGene adapter (official architecture, 1000-dim, shared legal graph) on fold 0's
outer training diseases for up to 100 epochs (development budget), evaluates the
*cold* validation fold F1 after every epoch on the V6 candidate lists
(Recall@10 and NDCG@10), keeps the checkpoint chosen by cold validation only,
and scores the fold-0 test lists once with that checkpoint.

It first checks input connectivity: a cold disease has no disease-gene edge by
construction, so its representation can only come from phenotype and ontology
edges; diseases with neither are counted and reported as an adapter limitation.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from gpu8.baselines import kdgene as kd

from . import paths6
from .common import World6, dump_json, now_utc
from .metrics6 import evaluate_matrix, macro, per_disease_frame
from .scoring import load_block


def score_block(model, kg, b, gene_row_to_id, device):
    out = np.full((b.D, b.N), -1e9)
    for i, d in enumerate(b.ids):
        s = kd.score_kdgene(model, kg, d, [], b.genes[i], gene_row_to_id, device)
        if s is not None:
            out[i] = s
    return out


def run(fold: int = 0, max_epochs: int = 100, patience: int = 10, seed: int = 42,
        device: str = "cuda") -> dict:
    t0 = time.time()
    w = World6()
    fa = w.fold(fold)
    genes = [w.gene_id_of_row[i] for i in range(w.n_g)]
    kg = kd.build_kg(w.cfg, w.relations, w.phenotypes, w.hierarchy, w.ppi,
                     set(fa["train_diseases"]), genes)
    val = load_block(f"f{fold}_outer", ids=w.labelled(fa["val_diseases"]))
    test = load_block(f"f{fold}_outer", ids=w.labelled(fa["test_diseases"]))
    # connectivity of cold diseases
    ph = w.phenotypes[w.phenotypes["negated"] == 0]
    has_ph = set(ph["disease_id"])
    has_hier = set(w.hierarchy["child_disease"]) | set(w.hierarchy["parent_disease"])
    conn = {"n_val": val.D,
            "val_with_entity": int(sum(f"D:{d}" in kg.entity_to_id for d in val.ids)),
            "val_with_phenotype_edge": int(sum(d in has_ph for d in val.ids)),
            "val_with_ontology_edge": int(sum(d in has_hier for d in val.ids)),
            "val_with_neither": int(sum((d not in has_ph) and (d not in has_hier) for d in val.ids)),
            "kg_triples": int(kg.triples.shape[0]), "kg_entities": kg.n_entities}
    curve = []

    def val_fn(model):
        model.eval()
        s = score_block(model, kg, val, w.gene_id_of_row, device)
        m = evaluate_matrix(s, val.y, val.genes, val.n_query, ks=(10,))
        rec = {"recall10": float(np.mean(m["recall10_end"])), "ndcg10": float(np.mean(m["ndcg10_end"])),
               "t": time.time() - t0}
        curve.append(rec)
        model.train()
        return rec["recall10"]

    model, hist = kd.train_kdgene(w.cfg, kg, device, seed=seed, edim=1000, rdim=1000,
                                  max_epochs=max_epochs, patience=patience, val_fn=val_fn,
                                  progress=True)
    for h, c in zip(hist["history"], curve):
        h.update({"val_recall10": c["recall10"], "val_ndcg10": c["ndcg10"],
                  "elapsed_s": c["t"]})
    model.eval()
    s = score_block(model, kg, test, w.gene_id_of_row, device)
    per = per_disease_frame(test.ids, evaluate_matrix(s, test.y, test.genes, test.n_query))
    per.to_csv(paths6.SCORING / f"kdgene_coldval_f{fold}_test_per_disease.csv", index=False)
    out = {"generated_utc": now_utc(), "fold": fold, "seed": seed, "edim": 1000,
           "max_epochs_budget": max_epochs, "patience": patience,
           "epochs_run": len(hist["history"]), "best_epoch": hist["best_epoch"],
           "best_cold_val_recall10": hist["best_val"], "connectivity": conn,
           "history": hist["history"], "test_metrics": macro(per),
           "seconds": time.time() - t0,
           "note": ("development-budget run on fold 0 with V6 candidates; the checkpoint is "
                    "chosen on the cold validation fold only; the loss curve alone is not "
                    "used to judge convergence")}
    dump_json(paths6.SCORING / f"kdgene_coldval_f{fold}.json", out)
    return out


if __name__ == "__main__":
    paths6.ensure()
    r = run()
    print(json.dumps({k: r[k] for k in ("epochs_run", "best_epoch", "best_cold_val_recall10",
                                        "connectivity", "test_metrics", "seconds")}, indent=2,
                     default=str))

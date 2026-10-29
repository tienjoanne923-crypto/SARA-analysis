"""Leakage-sensitivity checks of the core (test folds, fixed fitted models, no refit).

(a) text mask level 2: disease templates re-masked with the V6 level-2 dictionary
    (approved symbols + aliases + multi-word full gene names), re-encoded with the
    same frozen SapBERT; the core is re-scored with the stricter d_text.  A large
    drop would mean the text path matched gene names that level 1 left visible.
(b) historical GO: g_function rebuilt from the 2023-10-09 GO release (ontology +
    GOA human GAF, same filters), GO term texts re-encoded; the core is re-scored
    with the historical function vectors.  A large drop would mean the function
    path relies on recent annotations that may derive from the evaluated relations.

Both are exploratory robustness checks after registration; the fitted weights are
unchanged, so they measure the dependence of the fitted core on the input version.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from gpu8 import paths as p5
from gpu8v6 import paths6
from gpu8v6.gene_features import parse_go_obo, read_gaf, symbol_to_gene

from . import paths71
from .behavior71 import standalone_recall10
from .common71 import dump_json, now_utc
from .encoders71 import _clean, parse_go_text, sapbert_encode


def level2_templates() -> list[str]:
    from gpu8v6.disease_text import load_symbol_dictionary, mask_segment, parse_mondo_text
    diseases = pd.read_parquet(p5.TABLES_DIR / "diseases.parquet")
    genes = pd.read_parquet(p5.TABLES_DIR / "genes.parquet")
    sd = load_symbol_dictionary(genes)
    do = pd.read_parquet(p5.TABLES_DIR / "do_terms.parquet").set_index("term_id")
    mondo = parse_mondo_text(p5.external("mondo", "mondo.obo"))
    out = []
    for r in diseases.itertuples(index=False):
        name = _clean(r.disease_name)
        defs, syns = [], [_clean(s) for s in str(r.synonyms or "").split("|")]
        if _clean(r.definition):
            defs.append(_clean(r.definition))
        for sid in str(r.source_ids or "").split("|"):
            sid = sid.strip()
            if sid.startswith("DO:") and sid[3:] in do.index:
                rec = do.loc[sid[3:]]
                syns.append(_clean(rec["name"]))
                syns += [_clean(s) for s in str(rec["synonyms"] or "").split("|")]
                if _clean(rec["definition"]):
                    defs.append(_clean(rec["definition"]))
        if r.mapping_type == "mondo_exact_unique" and r.mondo_ids in mondo:
            m = mondo[r.mondo_ids]
            if m.get("def"):
                defs.append(m["def"])
            syns += list(m.get("syn", []))
        seen, syn_u = {name.lower()}, []
        for s in syns:
            if s and s.lower() not in seen:
                seen.add(s.lower())
                syn_u.append(s)
        parts = []
        for seg_list, joiner in (([name], " "), (defs, " "), (syn_u[:8], ", ")):
            parts.append(joiner.join(mask_segment(x, sd, level=2)[0] for x in seg_list if x))
        out.append((r.disease_id, " ; ".join(p for p in parts if p)))
    return out


def historical_function(w) -> tuple[np.ndarray, np.ndarray]:
    obo = paths6.ext("go", "go-basic_2023-10-09.obo")
    parents, _ = parse_go_obo(obo)
    gt = parse_go_text(obo)
    X, _ = sapbert_encode(gt["text"].tolist())
    pos = {t: i for i, t in enumerate(gt["term_id"])}
    sym2gene = symbol_to_gene(w, paths6.ext("hgnc", "hgnc_complete_set_2023-10-01.txt"))
    gaf = read_gaf(paths6.ext("go", "goa_human_2023-10-09.gaf.gz"), sym2gene)
    roots = {"GO:0008150", "GO:0003674", "GO:0005575"}
    gaf = gaf[gaf["go_id"].isin(parents) & ~gaf["go_id"].isin(roots) & gaf["go_id"].isin(pos)]
    gaf["gene_row"] = gaf["gene_id"].map(w.idx.g_ix)
    gaf = gaf[gaf["gene_row"].notna()].drop_duplicates(["gene_row", "go_id"])
    G = np.zeros((w.n_g, X.shape[1]), dtype=np.float64)
    cnt = np.zeros(w.n_g)
    for gr, t in zip(gaf["gene_row"].astype(int), gaf["go_id"]):
        G[gr] += X[pos[t]]
        cnt[gr] += 1
    m = cnt > 0
    G[m] /= cnt[m, None]
    return G.astype(np.float32), m


def run() -> dict:
    from .chain71 import Runner, variant_spec
    from .train71 import CoreTrainer
    run = Runner()
    spec = variant_spec("core")
    w = run.w
    tmp = level2_templates()
    X2, ntr = sapbert_encode([t for _, t in tmp])
    d_text2 = np.zeros_like(run.pack.a["d_text"])
    for i, (d, _) in enumerate(tmp):
        d_text2[w.idx.d_ix[d]] = X2[i]
    Gh, mh = historical_function(w)
    rows = []
    for j in range(5):
        rid = f"V71CHAIN__core__f{j}_outer__s42"
        tr, m = run.model(rid, spec)
        test = w.labelled(w.fold(j)["test_diseases"])
        drows = np.array([w.idx.d_ix[d] for d in test])
        R, V = tr.score(m, drows)
        base = standalone_recall10(R, V, test, w)
        T = dict(tr.T)
        T["xT"] = torch.as_tensor(d_text2, device=run.device)
        R2, V2 = CoreTrainer(T, run.device).score(m, drows)
        lv2 = standalone_recall10(R2, V2, test, w)
        T = dict(tr.T)
        T["xF"] = torch.as_tensor(Gh, device=run.device)
        T["mF"] = torch.as_tensor(mh, device=run.device)
        R3, V3 = CoreTrainer(T, run.device).score(m, drows)
        hg = standalone_recall10(R3, V3, test, w)
        rows.append({"fold": j, "normal": base.mean(), "text_mask_level2": lv2.mean(),
                     "historical_go_2023": hg.mean()})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(paths71.BEHAVIOR / "sensitivity_text_go_v71.csv", index=False)
    out = {"generated_utc": now_utc(), "per_fold": rows,
           "mean": df[["normal", "text_mask_level2", "historical_go_2023"]].mean().to_dict(),
           "genes_with_function_current": int(run.pack.a["g_function_mask"].sum()),
           "genes_with_function_2023": int(mh.sum()), "level2_truncated": ntr,
           "note": "fixed fitted core (seed 42), standalone full-background Recall@10; no refit"}
    dump_json(paths71.BEHAVIOR / "sensitivity_text_go_v71.json", out)
    return out


if __name__ == "__main__":
    paths71.ensure()
    print(json.dumps(run()["mean"], indent=2))

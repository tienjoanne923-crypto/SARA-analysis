"""Post-hoc diagnostics for the V7.1 problem statement (exploratory, test folds).

(a) GO-less genes: recovery of test positives whose gene has no legal GO function
    (only sequence / expression can describe them) versus positives with GO.
(b) Where the late-fusion gain comes from: per-disease Recall@10 difference to the
    channel-only LightGBM, split by disease HPO availability and by the number of
    known genes (single vs several).
(c) Who loses: diseases whose top-10 lost more V6 hits than it gained, by group.
All numbers are read from the existing run files; nothing is refitted.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu8v6.common import World6

from . import paths71
from .common71 import dump_json, now_utc
from .evaluate71 import SEEDS, load_per, positive_status, top10_sets


def method_positive_table(w, pattern: str, pos: pd.DataFrame) -> pd.DataFrame:
    accs = []
    for s in (SEEDS if "{s}" in pattern else [None]):
        fr = [positive_status(pattern.format(j=j, s=s)) for j in range(5)]
        fr = [x for x in fr if x is not None]
        if not fr:
            continue
        st = pd.concat(fr)
        m = pos.merge(st, on=["disease_id", "gene_row"], how="left")
        accs.append(pd.DataFrame({"in500": (m["pos"].fillna(1e9) <= 500).to_numpy(float),
                                  "top10": (m["pos"].fillna(1e9) <= 10).to_numpy(float)}))
    return sum(accs) / len(accs)


def run() -> dict:
    w = World6(load_heavy=False)
    z = np.load(paths71.CACHES / "featurepack_v71.npz")
    has_go, has_seq = z["g_function_mask"], z["g_sequence_mask"]
    has_expr = z["g_expression_mask"].any(axis=1)
    d_hpo = z["d_hpo_mask"]
    rows = []
    for j in range(5):
        for d in w.labelled(w.fold(j)["test_diseases"]):
            for g in w.query_rows(d):
                rows.append({"disease_id": d, "gene_row": int(g), "fold": j})
    pos = pd.DataFrame(rows)
    pos["has_go"] = has_go[pos["gene_row"]]
    pos["has_seq"] = has_seq[pos["gene_row"]]
    pos["has_expr"] = has_expr[pos["gene_row"]]
    vis = pd.read_parquet(paths71.EXT_RES / "answer_visibility_v71.parquet")
    pos = pos.merge(vis[["disease_id", "gene_row", "class"]], on=["disease_id", "gene_row"], how="left")
    out = {"generated_utc": now_utc(), "n_positives": int(len(pos)),
           "positives_without_go": int((~pos["has_go"]).sum()),
           "positives_without_go_share": float((~pos["has_go"]).mean()),
           "positives_without_go_with_seq": int(((~pos["has_go"]) & pos["has_seq"]).sum())}
    methods = {"V6 LightGBM": "V6Z__lightgbm__f{j}__s{s}", "core alone": "V71Z__core__f{j}__s{s}",
               "late fusion LGB": "V71Z__lf_lgb__f{j}__s{s}"}
    go_tab = []
    for name, pat in methods.items():
        t = method_positive_table(w, pat, pos[["disease_id", "gene_row"]])
        for grp, mask in (("GO present", pos["has_go"].to_numpy()), ("GO absent", ~pos["has_go"].to_numpy()),
                          ("GO absent, cold (ppi_only/unseen)",
                           (~pos["has_go"].to_numpy()) & pos["class"].isin(["ppi_only", "unseen"]).to_numpy())):
            go_tab.append({"method": name, "group": grp, "n": int(mask.sum()),
                           "in500": float(t["in500"][mask].mean()), "top10": float(t["top10"][mask].mean())})
    out["go_absent_recovery"] = go_tab
    # (b) where the late-fusion gain comes from
    lf = pd.concat([load_per(f"V71Z__lf_lgb__f{j}__s{s}").assign(seed=s) for j in range(5) for s in SEEDS])
    ch = pd.concat([load_per(f"V71Z__v6_lgb_channels__f{j}__s{s}").assign(seed=s) for j in range(5) for s in SEEDS])
    a = lf.groupby("disease_id")["recall10_end"].mean()
    b = ch.groupby("disease_id")["recall10_end"].mean()
    dd = pd.DataFrame({"lf": a, "ch": b}).dropna()
    dd["delta"] = dd["lf"] - dd["ch"]
    dd["has_hpo"] = [bool(d_hpo[w.idx.d_ix[d]]) for d in dd.index]
    dd["n_known"] = [len(w.query_rows(d)) for d in dd.index]
    dd["single_gene"] = dd["n_known"] == 1
    grp = []
    for col, lab in (("has_hpo", "HPO available"), ("single_gene", "single known gene")):
        for val, sub in dd.groupby(col):
            grp.append({"split": lab, "value": bool(val), "n": int(len(sub)), "lf": float(sub["lf"].mean()),
                        "channel_lgb": float(sub["ch"].mean()), "delta": float(sub["delta"].mean())})
    out["gain_by_group"] = grp
    # (c) who loses hits against the archived V6 LightGBM
    lost = []
    for s in SEEDS:
        for j in range(5):
            A = top10_sets(f"V71Z__lf_lgb__f{j}__s{s}")
            B = top10_sets(f"V6Z__lightgbm__f{j}__s{s}")
            for d in w.labelled(w.fold(j)["test_diseases"]):
                q = set(w.query_rows(d).tolist())
                ha, hb = A.get(d, set()) & q, B.get(d, set()) & q
                lost.append({"disease_id": d, "seed": s, "net": len(ha) - len(hb)})
    L = pd.DataFrame(lost).groupby("disease_id")["net"].mean()
    dd["net_vs_v6"] = L.reindex(dd.index)
    loss = []
    for col, lab in (("has_hpo", "HPO available"), ("single_gene", "single known gene")):
        for val, sub in dd.groupby(col):
            loss.append({"split": lab, "value": bool(val), "n": int(len(sub)),
                         "share_net_loss": float((sub["net_vs_v6"] < 0).mean()),
                         "share_net_gain": float((sub["net_vs_v6"] > 0).mean())})
    out["loss_by_group"] = loss
    dump_json(paths71.AUDIT / "problem_diagnostics_v71.json", out)
    return out


if __name__ == "__main__":
    paths71.ensure()
    r = run()
    print(json.dumps({k: v for k, v in r.items()}, indent=1, ensure_ascii=False)[:6000])

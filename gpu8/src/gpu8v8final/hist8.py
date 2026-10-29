"""Historical chain (cut-off 2023-12-31) and the X1 / X2 evaluation (§12.3).

Every model is re-trained on the cut-off world with the registered settings for
tasks without a native context (most common development choice over the five
outer contexts).  The comparison on the same lists: deployed V8.1, the base
ranker B and the teacher s_F; plus channel-only LightGBM on its own V6 500-list.
"""
from __future__ import annotations

import json
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import paths8
from .chain8 import Chain, fold_models, save_pools, save_pred
from .common8 import array_hash, dump_json, load_manifest, now_utc
from .comparators8 import DirectLGB, channel_X, standalone_scores, standalone_top
from .feats8 import GeneStatic, Labels
from .masks8 import load as load_masks
from .pack8 import ArmSpec
from .temporal8 import (HCTX, HistRoles, HistSourceEngine, HistWorld, audit_inputs, build_contexts, encode_hist,
                        hist_pack, hist_text_records, tasks)
from .v6pass8 import V6Pass, V6World, run_context

HW = V6World(name="hist")
OUT = paths8.EXT_RES / "historical"


class HistChain(Chain):
    def __init__(self, hw, roles, hp: dict, device: str = "cuda", threads: int = 8):
        self.arm = ArmSpec(name="hist", go_release="2023", expression="hpa", note="cut-off world; GTEx expression")
        self.v6w = HW
        self.device, self.role, self.threads = device, "variant", threads
        self.w, self.roles = hw, roles
        self.labels = Labels(hw)
        self.policy, self._ctx_pack = None, None
        a, A = hp["a"], hp["A"]
        self.pack = SimpleNamespace(a=a, A=A, info={"world": "historical 2023", "go_records": hp["go_records"]},
                                    hash=array_hash(a["d_text"], a["d_hpo"], a["g_function"], a["g_function_mask"]))
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)
        self.T = {"xT": t(a["d_text"]), "mT": t(a["d_text_mask"], torch.bool), "xH": t(a["d_hpo"]),
                  "mH": t(a["d_hpo_mask"], torch.bool), "q": t(a["d_tissue"]), "mq": t(a["d_tissue_mask"], torch.bool),
                  "xS": t(a["g_sequence"]), "mS": t(a["g_sequence_mask"], torch.bool), "xF": t(a["g_function"]),
                  "mF": t(a["g_function_mask"], torch.bool), "e": t(a["g_expression"]), "o": t(a["g_expression_mask"]),
                  "xA": t(a["anchor_vectors"]),
                  "A_plus": torch.as_tensor(A.toarray().astype(bool), device=device)}
        self.gs = GeneStatic(hw, a)
        self.masks = load_masks()
        self.e6_genes, self.drop_relations, self.hpo_drop_rows, self.q_random = None, None, None, None


def evaluate(hw, tk, d, methods: list[str]) -> dict:
    cur = tk["current_positives"]
    g_ix = hw.idx.g_ix
    old = {k: {g_ix[g] for g in v if g in g_ix} for k, v in hw.positives.items()}
    res, per_rows = {}, []
    for task in ("x2", "x1"):
        for m in methods:
            p = d / f"pred_{m}__{task}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            ids = [str(x) for x in z["ids"]]
            recs = []
            for i, dd in enumerate(ids):
                allq = {g_ix[g] for g in cur.get(dd, []) if g in g_ix}
                q = allq - old.get(dd, set()) if task == "x2" else allq
                if not q:
                    continue
                v = z["valid"][i] if "valid" in z.files else np.isfinite(z["score"][i])
                g, s = z["genes"][i][v].astype(np.int64), z["score"][i][v].astype(np.float64)
                o = g[np.lexsort((g, -s))]
                if task == "x2":
                    o = o[~np.isin(o, list(old.get(dd, set())))]
                top10 = set(o[:10].tolist())
                recs.append({"task": task, "method": m, "disease_id": dd, "n_query": len(q),
                             "recall10": len(q & top10) / len(q), "hit10": float(bool(q & top10)),
                             "coverage": len(q & set(o.tolist())) / len(q)})
            df = pd.DataFrame(recs)
            per_rows.append(df)
            res[f"{task}__{m}"] = {"n_diseases": int(len(df)), "n_pairs": int(df["n_query"].sum()) if len(df) else 0,
                                   "recall10": float(df["recall10"].mean()) if len(df) else None,
                                   "hit10": float(df["hit10"].mean()) if len(df) else None,
                                   "coverage": float(df["coverage"].mean()) if len(df) else None}
    if per_rows:
        pd.concat(per_rows, ignore_index=True).to_parquet(d / "historical_per_disease.parquet", index=False)
    return res


def run(seed: int = 42, device: str = "cuda", threads: int = 8) -> dict:
    reg = load_manifest("protocol_v8_registered.json")
    if not reg:
        raise RuntimeError("register V8.1 first")
    nn = reg["tasks_without_native_context"]
    OUT.mkdir(parents=True, exist_ok=True)
    d = OUT / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    inp = audit_inputs()
    hw = HistWorld()
    tk = tasks(hw)
    parts = build_contexts(hw, tk)
    roles = HistRoles(hw, parts, tk)
    text = hist_text_records(hw)
    encode_hist(hw, text, device)
    hp = hist_pack(hw, text)
    # ---- V6 passes of the cut-off world
    from gpu8v6.common import load_manifest as lm6
    frozen = lm6("retriever_v6_frozen.json")
    eng = None
    for c in HCTX:
        if not (paths8.V6PASS / HW.name / c / "meta.json").exists():
            if eng is None:
                eng = HistSourceEngine(hw, roles, text, device)
            run_context(eng, hw, roles, HW, c, frozen)
    del eng
    torch.cuda.empty_cache()
    ch = HistChain(hw, roles, hp, device, threads)
    controls = nn["controls"]
    w = nn["w"]
    cfg = {"corrector": nn["corrector"], "A": nn["A"], "deploy": nn["deploy"], "only_selected": True,
           "corrector_name": f"C_b{nn['corrector']['beta']:g}_l{nn['corrector']['lam']:g}",
           "A_name": f"A_nl{nn['A']['num_leaves']}_mc{nn['A']['min_child_samples']}"}
    sems, v6ps, trids = {}, {}, {}
    for c in HCTX:
        trids[c] = ch.teacher(c, seed, controls, role="variant")
        v6ps[c] = V6Pass(HW, c)
        sems[c] = ch.semantic(c, trids[c], v6ps[c], None)
    blocks = {c: ch.blocks_for(c, v6ps[c], sems[c], w) for c in HCTX}
    save_pools(d, blocks)
    summ = fold_models(ch, 0, seed, blocks, "hist", False, cfg, d, ctxs=HCTX)
    src = cfg["corrector_name"] if cfg["deploy"] == "B" else cfg["A_name"]
    for r in blocks["h_outer"]:
        if r != "oof":
            shutil.copyfile(d / f"pred_{src}__{r}.npz", d / f"pred_deployed__{r}.npz")
    # channel-only LightGBM on its own V6 500-lists (strong control)
    v6b = {c: ch.blocks_for(c, v6ps[c], sems[c], 0.5, pool="v6") for c in HCTX}
    cl = DirectLGB(channel_X, seed, threads).fit([v6b[c]["oof"] for c in HCTX[:3]], [v6b["h_outer"]["val"]])
    for r in ("val", "x2", "x1"):
        save_pred(d, f"channel_lgb_v6__{r}", v6b["h_outer"][r], cl.score(v6b["h_outer"][r]))
    from .teacher8 import load_teacher
    m = load_teacher(trids["h_outer"], ch.T, device)
    for r in ("x2", "x1"):
        ids = blocks["h_outer"][r].ids
        R, V = standalone_scores(m, ch.T, ids, hw.idx.d_ix, "sF")
        top = standalone_top(R, V, 500)
        sc = np.where(top >= 0, np.take_along_axis(R, np.maximum(top, 0), axis=1), -np.inf)
        np.savez_compressed(d / f"pred_standalone_sF__{r}.npz", ids=np.array(ids), genes=np.maximum(top, 0).astype(np.int16),
                            score=sc.astype(np.float32), valid=top >= 0)
    ev = evaluate(hw, tk, d, ["deployed", "B", cfg["corrector_name"], cfg["A_name"], "channel_lgb_v6",
                              "standalone_sF"])
    x1i = pd.DataFrame(tk["x1_info"])
    out = {"generated_utc": now_utc(), "seed": seed, "inputs_audit": {k: inp[k] for k in ("all_present", "all_not_later")},
           "world": hw.info, "contexts": {k: len(v) for k, v in {"val": parts["val"], "reference": parts["reference"]}.items()},
           "x2": {"pairs": len(tk["x2_pairs"]), "diseases": len(tk["x2_diseases"])},
           "x1": {"pairs": len(tk["x1_pairs"]), "diseases": len(tk["x1_diseases"]),
                  "with_text_at_cutoff": int(x1i["text_at_cutoff"].sum()) if len(x1i) else 0,
                  "with_hpo_at_cutoff": int(x1i["hpo_at_cutoff"].sum()) if len(x1i) else 0},
           "settings": {"controls": controls, "w": w, **{k: cfg[k] for k in ("corrector", "A", "deploy")}},
           "summary": {k: v for k, v in summ.items() if k in ("B_outer", "corrector", "A")},
           "results": ev, "seconds": time.time() - t0,
           "caveats": ["X2 is not disease-cold: the disease's old relations are in the reference",
                       "the second-level model learns on disease-cold OOF rows, X2 rows are disease-warm",
                       "gene background, MANE-2025 sequences, 2026 equivalence groups/families and frozen "
                       "hyper-parameters are later than the cut-off (disclosed)"]}
    dump_json(d / "historical_results.json", out)
    x1i.to_csv(d / "x1_diseases_info.csv", index=False)
    return out


if __name__ == "__main__":
    paths8.ensure()
    print(json.dumps(run(), indent=1, default=str)[:4000])

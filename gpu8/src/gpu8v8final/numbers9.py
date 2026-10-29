"""Numbers of the SARA manuscript.

``numbers9()`` reuses ``manuscript8v2.numbers()`` (which reads the figure source data and the saved results)
with its data readers pointed at the SARA figure folder (internal source-data tables, same format), and adds
``N["z"]``: the results that the V8.1 manuscript did not report (support 1-shot task, strata, U@10, further
contrasts, candidate-budget details, expression ablation, cohort replication across seeds, ...).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import manuscript8v2 as M2
from . import paths8

ROOT = M2.OUT.parent / "manuscript_sara"
FIG = ROOT / "figures"
SDI = FIG / "source_data_internal"
ALIAS = {
    ("Fig6_second_stage_diagnostics", "d"): ("Fig6_ranker_explanation", "a"),
    ("Fig6_second_stage_diagnostics", "e"): ("Fig6_ranker_explanation", "c"),
    ("Fig6_second_stage_diagnostics", "f"): ("Fig6_ranker_explanation", "d"),
    ("Fig6_second_stage_diagnostics", "g"): ("Fig6_ranker_explanation", "e"),
    ("Fig6_second_stage_diagnostics", "h"): ("Fig6_ranker_explanation", "b"),
    ("Fig7_go_mask_sequence", "b"): ("FigS8_alternative_designs", "d"),
    ("Fig7_go_mask_sequence", "d"): ("FigS8_alternative_designs", "f"),
    ("Fig7_go_mask_sequence", "e"): ("Fig7_annotation_supervision", "b"),
    ("Fig7_go_mask_sequence", "f"): ("Fig7_annotation_supervision", "c"),
    ("Fig7_go_mask_sequence", "g"): ("Fig7_annotation_supervision", "d"),
    ("Fig7_go_mask_sequence", "h"): ("FigS8_alternative_designs", "g"),
}


def sd9(stem: str, panel: str) -> pd.DataFrame:
    stem, panel = ALIAS.get((stem, panel), (stem, panel))
    return pd.read_csv(SDI / f"{stem}_{panel}.csv", encoding="utf-8-sig")


def manifest9() -> dict:
    return json.loads((FIG / "figures_manifest.json").read_text(encoding="utf-8"))


def load_R9() -> dict:
    R = json.loads((paths8.RESULTS / "numbers_v8_results.json").read_text(encoding="utf-8"))
    R["strata"] = json.loads((paths8.RESULTS / "analysis" / "strata_zero_signal_v8.json").read_text(encoding="utf-8"))
    R["hitflow"] = json.loads((paths8.RESULTS / "analysis" / "hitflow_s42_v8.json").read_text(encoding="utf-8"))
    R["acceptance"] = json.loads((paths8.AUDIT / "acceptance_ten_checks_v8.json").read_text(encoding="utf-8"))
    sels = [json.loads((paths8.DEV / f"fold{j}" / "selection.json").read_text(encoding="utf-8")) for j in range(5)]
    R["dev"] = {"v6_cov": float(np.mean([s["w"]["coverage"]["v6_500"] for s in sels])),
                "strict_cov": float(np.mean([s["w"]["coverage"][f"aware_w{s['w']['w']}"] for s in sels])),
                "w": [s["w"]["w"] for s in sels]}
    R["audit"] = json.loads((paths8.AUDIT / "evidence_policy_audit_v8.json").read_text(encoding="utf-8"))
    return R


def e4_summary(R: dict) -> dict:
    retrain = [k for k in R["e4"] if not k.startswith("e4_rescore")]
    resc = [k for k in R["e4"] if k.startswith("e4_rescore")]
    inc0 = [k for k in retrain if R["e4"][k]["difference"]["ci_low"] <= 0 <= R["e4"][k]["difference"]["ci_high"]]
    return {"n_retrain": len(retrain), "n_inc0": len(inc0), "n_rescore": len(resc),
            "excl0": [k for k in retrain if k not in inc0],
            "full_gap": R["e4"]["e4_layer1"]["new_minus_old_full"]}


def expression_counts(R: dict) -> dict:
    """number of cohort x contrast x top-K tests with p < 0.05, per gene set family and seed."""
    out: dict = {}
    for s in R["expression"]["series"]:
        if s.get("status") != "ok":
            continue
        for cname, c in s["contrasts"].items():
            for setname, v in c.items():
                if not isinstance(v, dict) or "p_value" not in v:
                    continue
                key = setname.rsplit("_top", 1)[0]
                d = out.setdefault(key, {"tests": 0, "p05": 0})
                d["tests"] += 1
                d["p05"] += int(v["p_value"] < 0.05)
    return out


def numbers9(R: dict | None = None) -> tuple[dict, dict]:
    R = R or load_R9()
    M2.sd = sd9
    M2.manifest = manifest9
    N = M2.numbers(R)
    tab = {r["method"]: r for r in R["main"]["table"]}
    Z = {"table": tab, "h": R["main"]["hierarchy"], "per_seed": R["main"]["deployed_per_seed"],
         "comp": R["comparators"], "strata": R["strata"]["strata"], "zero": R["strata"]["zero_signal_s42"],
         "flows": R["hitflow"]["flows"], "budget": R["budget"], "var": R["variants"], "sem": R["semantic"]["table"],
         "support533": R["support"]["cohort533"], "support1": R["support"]["cohort1shot"],
         "omega": R["support"]["omega_533"], "expr": expression_counts(R), "ext": R["external"],
         "hist": R["historical_contrasts"], "e4": R["e4"], "e4s": e4_summary(R), "repro": R["reproduction"],
         "dev": R["dev"], "audit": R["audit"], "subgroups": {r["method"]: r for r in R["subgroups"]["table"]},
         "e2": R["e2"]}
    N["z"] = Z
    return R, N

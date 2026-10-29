"""Final assembly: evaluation, analyses, numbers, figures, manuscript, reports, ledger, archive check."""
from __future__ import annotations

import json
import sys

import pandas as pd

from . import paths8
from .common8 import dump_json, now_utc


def main(skip_eval: bool = False) -> dict:
    from . import analysis8, audit8, eval8, figures8, ledger8, manuscript8, narrative8, numbers8, reports8
    if not skip_eval:
        eval8.run()
    # complete index of every evaluation file (one per tag), written after the registration
    idx = {}
    for p in sorted(eval8.EVAL.glob("eval_test_*.parquet")):
        df = pd.read_parquet(p, columns=["method"])
        idx[p.stem.replace("eval_test_", "")] = {"rows": int(len(df)), "methods": sorted(df["method"].unique().tolist())}
    dump_json(eval8.EVAL / "eval_index.json", {"generated_utc": now_utc(), "tags": idx})
    st = paths8.RESULTS / "analysis" / "strata_zero_signal_v8.json"
    if not st.exists():
        analysis8.zero_signal_and_strata()
    hf = paths8.RESULTS / "analysis" / "hitflow_s42_v8.json"
    if not hf.exists():
        analysis8.hitflows()
    R = numbers8.build()
    R["strata"] = json.loads(st.read_text(encoding="utf-8"))
    R["hitflow"] = json.loads(hf.read_text(encoding="utf-8"))
    acc = paths8.AUDIT / "acceptance_ten_checks_v8.json"
    if acc.exists():
        a = json.loads(acc.read_text(encoding="utf-8"))
        R["fallback"] = a.get("8_fallback", {})
    sels = [json.loads((paths8.DEV / f"fold{j}" / "selection.json").read_text(encoding="utf-8")) for j in range(5)]
    import numpy as np
    R["dev"] = {"v6_cov": float(np.mean([s["w"]["coverage"]["v6_500"] for s in sels])),
                "strict_cov": float(np.mean([s["w"]["coverage"][f"aware_w{s['w']['w']}"] for s in sels]))}
    R["_txt_inputs"] = narrative8.inputs_text()
    R["_txt_dev"] = narrative8.dev_text()
    R["_txt_seeds"] = narrative8.seeds_text(R)
    R["_txt_cold"] = narrative8.cold_text(R)
    R["_txt_budget"] = narrative8.budget_text(R)
    R["_txt_e4"] = narrative8.e4_text(R)
    R["_txt_e56"] = narrative8.e56_text(R)
    R["_txt_comp"] = narrative8.comp_text(R)
    R["_txt_e8"] = narrative8.e8_text(R)
    R["_txt_e9"] = narrative8.e9_text(R)
    R["_txt_discussion"] = narrative8.discussion_text(R)
    R["_txt_limits"] = narrative8.limits_text(R)
    R["_txt_conclusion"] = narrative8.conclusion_text(R)
    ledger = ledger8.collect()
    flow = R["hitflow"]["flows"].get("channel_lgb_v6->deployed")
    figs = figures8.build(R, flow, ledger)
    ms = manuscript8.write(R)
    rep = reports8.write(R)
    dump_json(paths8.RESULTS / "final_assembly_v8.json", {"generated_utc": now_utc(), "figures": figs,
                                                          "manuscript": ms, "reports": rep,
                                                          "ledger_rows": int(len(ledger)),
                                                          "ledger_hash_ok": bool(ledger["hash_ok"].all())})
    return {"manuscript": ms, "figures": figs, "reports": rep}


if __name__ == "__main__":
    paths8.ensure()
    print(json.dumps(main("--skip-eval" in sys.argv), indent=1, default=str, ensure_ascii=False)[:3000])

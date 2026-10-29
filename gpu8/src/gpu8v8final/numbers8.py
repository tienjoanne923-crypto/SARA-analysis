"""V8.1 tables and the number provenance file (every manuscript number has a source).

Reads results_v8/eval/eval_test_*.parquet (after registration) and the external /
historical / support / cohort outputs; writes results_v8/tables/*.csv and
results_v8/numbers_v8.json (value, source file, filter, method).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths8
from .common8 import dump_json, now_utc
from .evaluate8 import fast_family_bootstrap

EVAL = paths8.RESULTS / "eval"
TAB = paths8.TABLES_OUT
NUM: dict[str, dict] = {}


def num(key: str, value, source: str, note: str = "") -> None:
    NUM[key] = {"value": value, "source": source, "note": note}


def load_tag(tag: str) -> pd.DataFrame:
    p = EVAL / f"eval_test_{tag}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def seed_avg(df: pd.DataFrame, method: str, cols=("recall10", "recall20", "recall50", "hit10", "coverage500",
                                                  "U10", "eta10", "cold_recall10", "cold_to_all10",
                                                  "n_query", "n_query_cold")) -> pd.DataFrame:
    sub = df[df["method"] == method]
    if sub.empty:
        return sub
    agg = {c: "mean" for c in cols if c in sub}
    return sub.groupby("disease_id").agg(agg).reset_index()


def family_of() -> dict:
    from gpu8v6.common import World6
    return World6(load_heavy=False).family_of


def contrast(a: pd.DataFrame, b: pd.DataFrame, fam: dict, col: str = "recall10") -> dict:
    m = a[["disease_id", col]].merge(b[["disease_id", col]], on="disease_id", suffixes=("_a", "_b")).dropna()
    if m.empty:
        return {}
    d = (m[f"{col}_a"] - m[f"{col}_b"]).to_numpy()
    f = np.array([fam.get(x, x) for x in m["disease_id"]])
    lo, hi = fast_family_bootstrap(d, f)
    return {"n": int(len(d)), "a": float(m[f"{col}_a"].mean()), "b": float(m[f"{col}_b"].mean()),
            "delta": float(d.mean()), "ci_low": lo, "ci_high": hi, "families": int(len(np.unique(f))),
            "better": float((d > 0).mean()), "worse": float((d < 0).mean())}


MAIN_METHODS = {
    "deployed": "V8.1 部署（严格 500；各上下文候选 A）",
    "channel_lgb_v6": "仅检索通道 LightGBM（自身 V6 500）",
    "B": "基础排序器 B（30 通道，严格 500）",
    "v71_direct_strict": "V7.1 直接融合（严格 500 重训）",
    "F7_v6": "F7 固定融合（V6 500）",
    "F7_strict": "F7 固定融合（严格 500）",
    "v71_direct_v6": "V7.1 直接融合（V6 500）",
    "standalone_sF": "教师 $s_F$ 单独（全背景）",
    "standalone_r": "V7.1 核心 $r$ 单独（全背景）",
    "standalone_sS": "教师自身序列分数单独",
    "standalone_sF0": "无学习 SapBERT 功能相似单独",
}


def table_main(fam: dict) -> dict:
    df = load_tag("main")
    if df.empty:
        return {}
    rows = []
    # dev-phase alternatives fitted with seed 42 only are reported in E2 / the supplement, not here
    methods = [m for m in sorted(df["method"].unique()) if not (m.startswith("A_nl") or m.startswith("C_b"))]
    for m in methods:
        s = seed_avg(df, m)
        seeds = sorted(df.loc[df["method"] == m, "seed"].unique().tolist())
        rows.append({"method": m, "label": MAIN_METHODS.get(m, m), "seeds": len(seeds), "n_diseases": len(s),
                     **{c: float(s[c].mean()) for c in ("recall10", "recall20", "recall50", "hit10", "coverage500",
                                                        "U10", "cold_recall10", "cold_to_all10") if c in s}})
    t = pd.DataFrame(rows).sort_values("recall10", ascending=False)
    t.to_csv(TAB / "t2_main_comparison.csv", index=False, encoding="utf-8-sig")
    dep = seed_avg(df, "deployed")
    out = {"table": t.to_dict(orient="records")}
    hier = {}
    for key, b in (("h1_vs_channel_lgb_v6", "channel_lgb_v6"), ("h2_vs_v71_direct_strict", "v71_direct_strict"),
                   ("vs_B", "B"), ("vs_F7_v6", "F7_v6"), ("vs_standalone_sF", "standalone_sF")):
        sb = seed_avg(df, b)
        if not sb.empty:
            hier[key] = contrast(dep, sb, fam)
            if key in ("h2_vs_v71_direct_strict",):
                hier["h3_cold_vs_v71_direct_strict"] = contrast(dep, sb, fam, "cold_recall10")
    h1 = hier.get("h1_vs_channel_lgb_v6", {})
    h2 = hier.get("h2_vs_v71_direct_strict", {})
    h3 = hier.get("h3_cold_vs_v71_direct_strict", {})
    verdict = {"h1_success": bool(h1 and h1["ci_low"] > 0),
               "h2_tested": bool(h1 and h1["ci_low"] > 0),
               "h2_success": bool(h1 and h1["ci_low"] > 0 and h2 and h2["ci_low"] > -0.005),
               "h3_tested": bool(h1 and h1["ci_low"] > 0 and h2 and h2["ci_low"] > -0.005)}
    verdict["h3_success"] = bool(verdict["h3_tested"] and h3 and h3["ci_low"] > 0)
    out["hierarchy"] = hier
    out["verdict"] = verdict
    for k, v in hier.items():
        for f in ("delta", "ci_low", "ci_high", "a", "b"):
            if f in v:
                num(f"main.{k}.{f}", v[f], "results_v8/eval/eval_test_main.parquet", "seed-averaged per disease")
    for r in rows:
        num(f"main.{r['method']}.recall10", r["recall10"], "results_v8/tables/t2_main_comparison.csv")
        if "cold_recall10" in r:
            num(f"main.{r['method']}.cold_recall10", r["cold_recall10"], "results_v8/tables/t2_main_comparison.csv")
    # per-seed deployed
    ps = df[df["method"] == "deployed"].groupby("seed")["recall10"].mean().to_dict()
    out["deployed_per_seed"] = {int(k): float(v) for k, v in ps.items()}
    return out


def selected_lambda() -> dict:
    from .formal8 import sel_cfg, selection
    return {j: sel_cfg(selection(j))["corrector"]["lam"] for j in range(5)}


def corrector_view(df: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Rows of the corrector with this beta at each fold's registered lambda (all seeds)."""
    lam = selected_lambda()
    parts = [df[(df["fold"] == j) & (df["method"] == f"C_b{beta:g}_l{lam[j]:g}")] for j in range(5)]
    out = pd.concat(parts, ignore_index=True)
    out["method"] = f"corrector_beta{beta:g}"
    return out


def table_e2(fam: dict) -> dict:
    """E2: gated corrector beta = 0/1/2 (registered lambda) vs direct fusion (A) vs B, shared strict 500."""
    df = load_tag("main")
    if df.empty:
        return {}
    views = {"B": df[df["method"] == "B"], "deployed": df[df["method"] == "deployed"],
             "v71_direct_strict": df[df["method"] == "v71_direct_strict"],
             "channel_lgb_v6": df[df["method"] == "channel_lgb_v6"]}
    for b in (0.0, 1.0, 2.0):
        views[f"corrector_beta{b:g}"] = corrector_view(df, b)
    rows, contr = [], {}
    dep = seed_avg(views["deployed"], "deployed")
    for name, sub in views.items():
        s = seed_avg(sub, sub["method"].iloc[0] if len(sub) else name)
        if s.empty:
            continue
        rows.append({"method": name, "seeds": int(sub["seed"].nunique()), "n": len(s),
                     "recall10": float(s["recall10"].mean()), "cold_recall10": float(s["cold_recall10"].mean()),
                     "cold_to_all10": float(s["cold_to_all10"].mean()),
                     "n_diseases_cold": int((s["n_query_cold"] > 0).sum())})
        if name.startswith("corrector"):
            contr[name] = {"recall10_vs_deployed": contrast(s, dep, fam),
                           "cold_vs_deployed": contrast(s, dep, fam, "cold_recall10"),
                           "cold_to_all_vs_deployed": contrast(s, dep, fam, "cold_to_all10")}
    b0 = seed_avg(views["corrector_beta0"], "corrector_beta0")
    for b in ("corrector_beta1", "corrector_beta2"):
        s = seed_avg(views[b], b)
        contr[f"{b}_vs_beta0_cold"] = contrast(s, b0, fam, "cold_recall10")
        contr[f"{b}_vs_beta0_recall10"] = contrast(s, b0, fam)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "t_e2_corrector.csv", index=False, encoding="utf-8-sig")
    return {"table": rows, "contrasts": contr, "selected_lambda": selected_lambda()}


def table_subgroups() -> dict:
    """E3: subgroups (no GO protein genes, ncRNA, no-HPO diseases, semantic-unavailable queries)."""
    from gpu8v71.featurepack71 import FeaturePack
    df = load_tag("main")
    if df.empty:
        return {}
    rows = []
    for m in ("deployed", "channel_lgb_v6", "B", "v71_direct_strict", "F7_v6"):
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        rec = {"method": m}
        for g in ("ncrna", "nogo_protein", "nogo", "mask_test", "sem_unavail"):
            nq = f"n_query_{g}" if g != "sem_unavail" else "n_query_sem_unavail"
            hk = f"hits10_{g}" if g != "sem_unavail" else "hits10_sem_unavail"
            pk = f"pool_{g}" if g != "sem_unavail" else "pool_sem_unavail"
            if nq in sub:
                n = sub[nq].sum()
                rec[f"{g}_queries"] = int(n / max(sub["seed"].nunique(), 1))
                rec[f"{g}_hit10_micro"] = float(sub[hk].sum() / n) if n else None
                rec[f"{g}_pool_micro"] = float(sub[pk].sum() / n) if n else None
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "t3_subgroups.csv", index=False, encoding="utf-8-sig")
    s42 = {}
    sub = df[(df["method"] == "deployed") & (df["seed"] == 42)]
    for g in ("ncrna", "nogo_protein", "nogo", "mask_test"):
        n = sub[f"n_query_{g}"].sum()
        s42[f"{g}_pool_micro"] = float(sub[f"pool_{g}"].sum() / n) if n else None
        s42[f"{g}_hit10_micro"] = float(sub[f"hits10_{g}"].sum() / n) if n else None
    return {"table": rows, "deployed_seed42": s42}


def table_variants(fam: dict) -> dict:
    """E4/E5/E6/controls: deployed vs B vs channel lists inside every arm (seed 42)."""
    out = {}
    main = load_tag("main")
    main42 = main[main["seed"] == 42] if not main.empty else main
    for p in sorted(EVAL.glob("eval_test_*.parquet")):
        tag = p.stem.replace("eval_test_", "")
        if tag in ("main",) or tag.startswith("comparator_"):
            continue
        df = pd.read_parquet(p)
        if "deployed" not in set(df["method"]):
            continue                                  # standalone-only tags (two-tower, ProtST, text) go to table_semantic
        rec = {}
        for m in ("deployed", "B"):
            s = seed_avg(df, m)
            if s.empty:
                continue
            rec[m] = {"recall10": float(s["recall10"].mean()), "cold_recall10": float(s["cold_recall10"].mean()),
                      "coverage500": float(s["coverage500"].mean()), "n": len(s)}
            for g in ("mask_test", "nogo", "ncrna"):
                nq = f"n_query_{g}"
                if nq in df:
                    sub = df[df["method"] == m]
                    n = sub[nq].sum()
                    if n:
                        rec[m][f"{g}_hit10_micro"] = float(sub[f"hits10_{g}"].sum() / n)
                        rec[m][f"{g}_pool_micro"] = float(sub[f"pool_{g}"].sum() / n)
                        rec[m][f"{g}_queries"] = int(n)
            if "n_query_block" in df:
                sub = df[df["method"] == m]
                n = sub["n_query_block"].sum()
                if n:
                    rec[m]["block_hit10_micro"] = float(sub["hits10_block"].sum() / n)
                    rec[m]["block_queries"] = int(n)
            if not main42.empty:
                ref = seed_avg(main42, m)
                rec[m]["delta_vs_main"] = contrast(s, ref, fam)
        out[tag] = rec
    rows = []
    for tag, rec in out.items():
        for m, v in rec.items():
            rows.append({"arm": tag, "method": m, **{k: (v[k] if not isinstance(v[k], dict) else v[k].get("delta"))
                                                     for k in v}})
    pd.DataFrame(rows).to_csv(TAB / "t_variants.csv", index=False, encoding="utf-8-sig")
    return out


E4_ARMS = ["e4_layer1", "e4_layer1_random", "e4_layer2", "e4_layer2_random", "e4_experimental", "e4_no_tas_nas",
           "e4_no_mutant", "e4_blocks", "e4_blocks_random", "e4_rescore_no_ic", "e4_rescore_iea_only"]


def table_e4(fam: dict) -> dict:
    """PDF §4.4: new-minus-old in the full data, in the reduced data, and their difference (each with a CI).
    Old = channel-only LightGBM on its own V6 500-list; refitted inside an arm only where the reference
    graph changes (evidence blocks); GO-only arms leave it unchanged by construction."""
    main = load_tag("main")
    if main.empty:
        return {}
    m42 = main[main["seed"] == 42]
    new_full = seed_avg(m42, "deployed")
    old_full = seed_avg(m42, "channel_lgb_v6")
    out, rows = {}, []
    full = contrast(new_full, old_full, fam)
    for tag in E4_ARMS:
        df = load_tag(tag)
        if df.empty:
            continue
        new_arm = seed_avg(df, "deployed")
        old_arm = seed_avg(df, "channel_lgb_v6") if "channel_lgb_v6" in set(df["method"]) else old_full
        arm = contrast(new_arm, old_arm, fam)
        m = (new_arm[["disease_id", "recall10"]].merge(old_arm[["disease_id", "recall10"]], on="disease_id",
                                                        suffixes=("_na", "_oa"))
             .merge(new_full[["disease_id", "recall10"]], on="disease_id")
             .merge(old_full[["disease_id", "recall10"]], on="disease_id", suffixes=("_nf", "_of")))
        d = ((m["recall10_na"] - m["recall10_oa"]) - (m["recall10_nf"] - m["recall10_of"])).to_numpy()
        f = np.array([fam.get(x, x) for x in m["disease_id"]])
        lo, hi = fast_family_bootstrap(d, f)
        rec = {"arm": tag, "new_minus_old_full": full, "new_minus_old_reduced": arm,
               "difference": {"delta": float(d.mean()), "ci_low": lo, "ci_high": hi, "n": int(len(d))},
               "old_refitted_in_arm": "channel_lgb_v6" in set(df["method"]),
               "evaluation_conditioned_audit": tag.startswith("e4_layer2")}
        if "n_query_block" in df:
            for meth in ("deployed", "channel_lgb_v6"):
                sub = df[df["method"] == meth]
                n = sub["n_query_block"].sum()
                if n:
                    rec[f"block_hit10_{meth}"] = float(sub["hits10_block"].sum() / n)
                    rec["block_queries"] = int(n)
            for meth in ("deployed", "channel_lgb_v6"):
                sub = m42[m42["method"] == meth]
                n = sub["n_query_block"].sum()
                if n:
                    rec[f"block_hit10_full_{meth}"] = float(sub["hits10_block"].sum() / n)
            if not rec["old_refitted_in_arm"] and "block_hit10_full_channel_lgb_v6" in rec:
                # GO-only deletions leave the V6 channels (and the old method) unchanged
                rec["block_hit10_channel_lgb_v6"] = rec["block_hit10_full_channel_lgb_v6"]
        out[tag] = rec
        rows.append({"arm": tag, "full_delta": full.get("delta"), "reduced_delta": arm.get("delta"),
                     "reduced_ci_low": arm.get("ci_low"), "reduced_ci_high": arm.get("ci_high"),
                     "difference": d.mean(), "diff_ci_low": lo, "diff_ci_high": hi,
                     "old_refitted": rec["old_refitted_in_arm"], "audit_flag": rec["evaluation_conditioned_audit"]})
    pd.DataFrame(rows).to_csv(TAB / "t_e4_difference_of_differences.csv", index=False, encoding="utf-8-sig")
    return out


def table_budget(fam: dict) -> dict:
    """E7: fixed old V6 500 / shared new 500 / strict end-to-end / ~950 bridge (seed 42 for the arms)."""
    main = load_tag("main")
    v6p = load_tag("v6pool")
    br = load_tag("bridge950")
    rows = []

    def add(table, label, df, m, seed=None):
        if df.empty:
            return
        sub = df if seed is None else df[df["seed"] == seed]
        s = seed_avg(sub, m)
        if s.empty:
            return
        rows.append({"table": table, "method": label, "n": len(s), "recall10": float(s["recall10"].mean()),
                     "cold_recall10": float(s["cold_recall10"].mean()), "coverage": float(s["coverage500"].mean()),
                     "pool_size": float(sub[sub["method"] == m]["pool_size"].mean()) if "pool_size" in sub else None})
    add("1_fixed_v6_500", "F7", main, "F7_v6", 42)
    add("1_fixed_v6_500", "仅通道 LightGBM", main, "channel_lgb_v6", 42)
    add("1_fixed_v6_500", "V7.1 直接融合", main, "v71_direct_v6", 42)
    add("1_fixed_v6_500", "候选 A（部署规则）", v6p, "deployed", 42)
    if not v6p.empty:
        from .formal8 import sel_cfg, selection
        parts = []
        for j in range(5):
            nm = sel_cfg(selection(j))["corrector_name"]
            parts.append(v6p[(v6p["fold"] == j) & (v6p["method"] == nm)])
        cv = pd.concat(parts, ignore_index=True).assign(method="corrector_registered")
        add("1_fixed_v6_500", "门控纠偏（各折登记配置）", cv, "corrector_registered", 42)
    add("2_shared_new_500", "F7", main, "F7_strict", 42)
    add("2_shared_new_500", "基础排序器 B", main, "B", 42)
    add("2_shared_new_500", "V7.1 直接融合", main, "v71_direct_strict", 42)
    add("2_shared_new_500", "候选 A（部署）", main, "deployed", 42)
    for m in ("C_b0_l0.1", "C_b1_l0.1", "C_b2_l0.1", "C_b0_l0.03", "C_b1_l0.03", "C_b2_l0.03"):
        add("2_shared_new_500", f"门控纠偏 {m}", main, m, 42)
    add("3_end_to_end", "V8.1 部署（严格 500）", main, "deployed", 42)
    add("3_end_to_end", "仅通道 LightGBM（自身 V6 500）", main, "channel_lgb_v6", 42)
    add("4_bridge_950", "仅通道 LightGBM（同 950 预算）", br, "B", 42)
    add("4_bridge_950", "候选 A（同 950 预算）", br, "deployed", 42)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "t_e7_budgets.csv", index=False, encoding="utf-8-sig")
    out = {"table": rows}
    if not br.empty:
        out["bridge_A_vs_B"] = contrast(seed_avg(br, "deployed"), seed_avg(br, "B"), fam)
        out["bridge_A_vs_strict_A"] = contrast(seed_avg(br, "deployed"), seed_avg(main[main["seed"] == 42], "deployed"), fam)
    if not v6p.empty:
        out["v6pool_A_vs_channel"] = contrast(seed_avg(v6p, "deployed"), seed_avg(main[main["seed"] == 42], "channel_lgb_v6"), fam)
        out["strict_A_vs_v6pool_A"] = contrast(seed_avg(main[main["seed"] == 42], "deployed"), seed_avg(v6p, "deployed"), fam)
    return out


def table_semantic(fam: dict) -> dict:
    """Standalone full-background semantic lists (seed 42): does representation learning matter?"""
    main = load_tag("main")
    m42 = main[main["seed"] == 42] if not main.empty else main
    items = [("main", "standalone_sF", "教师 s_F（分段文本）"), ("main", "standalone_r", "V7.1 核心 r"),
             ("main", "standalone_sS", "教师自身序列分数"), ("main", "standalone_sF0", "无学习 SapBERT 功能相似"),
             ("two_tower", "standalone_twotower", "线性双塔"), ("protst_zeroshot", "protst_zeroshot", "ProtST 零样本"),
             ("text_short", "standalone_sF", "教师 s_F（V7.1 短文本）"),
             ("text_level2", "standalone_sF", "教师 s_F（二级掩蔽文本）")]
    ref = seed_avg(m42, "standalone_sF") if not m42.empty else None
    rows = []
    for tag, m, label in items:
        df = m42 if tag == "main" else load_tag(tag)
        s = seed_avg(df, m) if not df.empty else pd.DataFrame()
        if s.empty:
            continue
        r = {"method": label, "tag": tag, "n": len(s), "recall10": float(s["recall10"].mean()),
             "recall50": float(s["recall50"].mean()), "cold_recall10": float(s["cold_recall10"].mean()),
             "coverage500": float(s["coverage500"].mean())}
        if ref is not None and not (tag == "main" and m == "standalone_sF"):
            r["vs_sF"] = contrast(s, ref, fam)
        rows.append(r)
    pd.DataFrame([{k: (v if not isinstance(v, dict) else v.get("delta")) for k, v in r.items()} for r in rows]).to_csv(
        TAB / "t_semantic_standalone.csv", index=False, encoding="utf-8-sig")
    return {"table": rows}


def table_historical(fam: dict) -> dict:
    p = paths8.EXT_RES / "historical" / "s42" / "historical_per_disease.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    out = {}
    for task in ("x2", "x1"):
        sub = df[df["task"] == task]
        a = sub[sub["method"] == "deployed"][["disease_id", "recall10"]]
        for b in ("channel_lgb_v6", "B", "standalone_sF"):
            bb = sub[sub["method"] == b][["disease_id", "recall10"]]
            if len(a) and len(bb):
                out[f"{task}_deployed_vs_{b}"] = contrast(a, bb, fam)
    return out


def table_comparators(fam: dict) -> dict:
    out = {}
    main = load_tag("main")
    dep = seed_avg(main[main["seed"] == 42], "deployed") if not main.empty else None
    for comp in ("kdgene", "sign"):
        df = load_tag(f"comparator_{comp}")
        if df.empty:
            continue
        s = seed_avg(df, df["method"].iloc[0])
        out[comp] = {"recall10": float(s["recall10"].mean()), "n": len(s)}
        if dep is not None:
            out[comp]["deployed_minus"] = contrast(dep, s, fam)
    return out


def build() -> dict:
    TAB.mkdir(parents=True, exist_ok=True)
    fam = family_of()
    res = {"generated_utc": now_utc()}
    res["main"] = table_main(fam)
    res["e2"] = table_e2(fam)
    res["subgroups"] = table_subgroups()
    res["variants"] = table_variants(fam)
    res["budget"] = table_budget(fam)
    res["semantic"] = table_semantic(fam)
    res["e4"] = table_e4(fam)
    res["historical_contrasts"] = table_historical(fam)
    res["comparators"] = table_comparators(fam)
    for name, p in (("external", paths8.EXT_RES / "external618" / "external618_v8.json"),
                    ("historical", paths8.EXT_RES / "historical" / "s42" / "historical_results.json"),
                    ("support", paths8.EXT_RES / "support" / "support_test_v8.json"),
                    ("expression", paths8.EXT_RES / "expression_v8.json"),
                    ("review", paths8.EXT_RES / "review160" / "review160_info.json"),
                    ("acceptance", paths8.AUDIT / "acceptance_ten_checks_v8.json"),
                    ("reproduction", paths8.AUDIT / "reproduction_venv_repro_v8.json")):
        if p.exists():
            res[name] = json.loads(p.read_text(encoding="utf-8"))
    dump_json(paths8.RESULTS / "numbers_v8_results.json", res)
    dump_json(paths8.RESULTS / "numbers_v8.json", {"generated_utc": now_utc(), "numbers": NUM})
    return res


if __name__ == "__main__":
    paths8.ensure()
    r = build()
    print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) and "verdict" in v else list(v)[:5] if isinstance(v, dict) else v)
                      for k, v in r.items()}, indent=1, default=str)[:3000])
